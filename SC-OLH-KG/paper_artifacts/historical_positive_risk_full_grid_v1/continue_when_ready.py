"""Prepare and screen the frozen full-CDF rule with eight shared worker slots."""
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from performance.launch_historical_component_bank import ROOT,CLI,DEPLOY,PYTHON,QUEUE,launch,collect_remote,refresh_jobs

sys.path.insert(0,str(Path(CLI).parent))
import scheduler

parser=argparse.ArgumentParser()
parser.add_argument('--plan',default='performance/manifests/historical_positive_risk_full_grid_v1.json')
plan=parser.parse_args().plan
spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output']
status=dict(state='preparing',estimated_phase_cpu_seconds=spec['estimated_cpu_seconds_including600reserves'],
    maximum_live_workers=8,full_predictive_integration_admitted=False,independent_confirmation=False)


def save_status():
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')


try:
    saved=out/'preparation_job.json'
    if saved.exists():tid=json.loads(saved.read_text())['id']
    else:
        cmd=shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,
            '-u','-m','performance.prepare_historical_positive_risk_screen','--plan',plan])
        args=[sys.executable,CLI,'submit','--description',f'KG n4 normalized grid {Path(plan).stem}',
            '--cmd',cmd,'--cwd',DEPLOY,'--signature',f'KG_op/{Path(plan).stem}/prepare',
            '--cpu','1','--ram-mb','2000','--vram','0','--priority','low','--project','KG_op',
            '--skip-launch-staging','--env-spec','none']
        for node in range(1,7):args+=['--allowed-node',f'node{node:03d}']
        p=subprocess.run(args,text=True,capture_output=True)
        (out/'preparation_spec.json').write_text(json.dumps(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
        tid=re.search(r'submitted (t\d+)',p.stdout).group(1)
        saved.write_text(json.dumps(dict(id=tid),indent=2)+'\n')
        args=[sys.executable,CLI,'dispatch','--no-bulk-window','--lock-timeout','20','--task-id',tid]
        p=subprocess.run(args,text=True,capture_output=True)
        (out/'preparation_dispatch.json').write_text(json.dumps(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
    status['preparation_id']=tid;save_status()
    while True:
        refresh_jobs({tid})
        task=next(t for t in json.loads(QUEUE.read_text())['tasks'] if t['id']==tid)
        if task['status']=='done':break
        if task['status'] not in ('queued','launching','running'):raise RuntimeError(f'{tid}: '+task['status'])
        time.sleep(20)
    code='import json;from pathlib import Path;lines=Path('+repr(task['log_path'])+').read_text().splitlines();print(json.dumps({"result":json.loads([s[17:] for s in lines if s.startswith("SCHEDULEURM_DONE ")][-1]),"resource":json.loads([s[21:] for s in lines if s.startswith("SCHEDULEURM_RESOURCE ")][-1])}))'
    rc,body,err=scheduler.run_on(task['node'],shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
    if rc:raise RuntimeError(err or body)
    prepared=json.loads(body);prep=prepared['result']
    assert prep['complete'] and prep['estimated_cost_admitted'] and prep['risk_observations']==spec['risk_observations']
    for root in (ROOT,Path(DEPLOY)):
        target=root/spec['output']/'preparation.json';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(prep,indent=2)+'\n')
    (out/'preparation_resource.json').write_text(json.dumps(prepared['resource'],indent=2)+'\n')
    status.update(state='running',cdf_checks=prep['cdf_checks'],worker_count=len(prep['jobs']));save_status()
    launch('positive-risk-grid',plan)
    summary=collect_remote('observed-batch-bound',out,plan)
    status.update(state='screened_coarse_grid' if summary['numerical_containment_passed'] else 'precision_hold',
        weighted_bank_envelope=summary['weighted_bank_envelope'],remaining_totals=summary['totals'])
    if summary['numerical_containment_passed']:
        priced=collect_remote('observed-batch-next-plan',out,plan)
        status.update(next_reference_cost_admitted=priced['cost_admitted'],
            next_reference_worker_count=len(priced['next_joint_jobs']),
            next_estimated_phase_cpu_seconds=priced['expected_cpu_seconds_including600reserves'])
    jobs=json.loads((out/'positive-risk-grid_jobs.json').read_text());ids={j['id'] for j in jobs}
    tasks=[t for t in json.loads(QUEUE.read_text())['tasks'] if t['id'] in ids];resources=[]
    for node in sorted({t['node'] for t in tasks}):
        paths={t['id']:t['log_path'] for t in tasks if t['node']==node}
        code='import json;from pathlib import Path;paths='+repr(paths)+';print(json.dumps({tid:json.loads([s[21:] for s in Path(p).read_text().splitlines() if s.startswith("SCHEDULEURM_RESOURCE ")][-1]) for tid,p in paths.items()}))'
        rc,body,err=scheduler.run_on(node,shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
        if rc:raise RuntimeError(err or body)
        values=json.loads(body)
        resources.extend(dict(id=t['id'],status=t['status'],node=node,resource=values[t['id']]) for t in tasks if t['node']==node)
    (out/'worker_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
    collectors=[]
    for name in ('observed-batch-bound','observed-batch-next-plan'):
        log=out/(name+'_collection.log')
        if log.exists():collectors.extend(json.loads(s[21:]) for s in log.read_text().splitlines() if s.startswith('SCHEDULEURM_RESOURCE '))
    account=dict(estimated_phase_cpu_seconds=status['estimated_phase_cpu_seconds'],shared_reserve_cpu_seconds=600,
        worker_count=len(resources),worker_process_cpu_seconds=sum(r['resource']['process_cpu_seconds'] for r in resources),
        preparation_process_cpu_seconds=prepared['resource']['process_cpu_seconds'],
        collection_process_cpu_seconds=sum(r['process_cpu_seconds'] for r in collectors),
        peak_worker_rss_mb=max(r['resource']['peak_rss_mb'] for r in resources),
        maximum_live_workers=8,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        full_predictive_integration_admitted=False,independent_confirmation=False)
    (out/'execution_account.json').write_text(json.dumps(account,indent=2)+'\n')
except Exception as error:
    status.update(state='failed',error=str(error));raise
finally:save_status()
print('POSITIVE_GRID_DONE '+json.dumps(status),flush=True)
