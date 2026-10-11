"""Run the frozen eight marginal shards after the corrected outer replay."""
from pathlib import Path
import json
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLI = '/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py'
QUEUE = Path('/home/erzhu419/.claude/scheduler/queue.json')
REMOTE = '/home/zhengliang01/scheduleurm_work/KG_op_independent_kg_parallel_deploy/SC-OLH-KG'
PYTHON = '/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python'
PLAN = 'performance/manifests/historical_certificate_bound_body16_v3.json'
sys.path.insert(0,str(Path(CLI).parent))
import scheduler


def wait(ids):
    while True:
        tasks = {t['id']:t for t in json.loads(QUEUE.read_text())['tasks']}
        selected = [tasks[i] for i in ids]
        assert all(t['status'] in ('queued','launching','running','done') for t in selected), selected
        if all(t['status']=='done' for t in selected):return selected
        time.sleep(20)


def fetch(task, directory):
    directory.mkdir(parents=True,exist_ok=True)
    for attempt in range(6):
        rc,body,err = scheduler.run_on(task['node'],shlex.join(['tail','-n','12',task['log_path']]),timeout=35,check=False)
        assert rc==0,(body,err)
        (directory/(task['id']+'.log')).write_text(body+err)
        resources = [json.loads(line.split(' ',1)[1]) for line in body.splitlines() if line.startswith('SCHEDULEURM_RESOURCE ')]
        if resources:return dict(task_id=task['id'],**resources[-1])
        if attempt<5:time.sleep(2)
    raise RuntimeError('Completed task has no resource footer: '+task['id'])


def pull(output, name):
    code = 'from pathlib import Path;print(Path('+repr(REMOTE+'/'+output+'/'+name)+').read_text())'
    rc,body,err = scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=35,check=False)
    assert rc==0,(body,err)
    value = json.loads(body)
    (ROOT/output/name).write_text(json.dumps(value,indent=2)+'\n')
    return value


def submit(args, directory, name):
    p = subprocess.run(args,text=True,capture_output=True)
    assert p.returncode==0,(p.stdout,p.stderr)
    result = dict(args=args,stdout=p.stdout,stderr=p.stderr,returncode=p.returncode,
        task_id=re.search(r'submitted (t\d+)',p.stdout).group(1))
    (directory/name).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    outer = ROOT/'paper_artifacts/historical_outer_bridge_v2'
    job = json.loads((outer/'scheduler_spec.json').read_text())
    task = wait([job['task_id']])[0]
    outer_resource = fetch(task,outer)
    bridge = pull('paper_artifacts/historical_outer_bridge_v2','summary.json')
    assert bridge['finite_outer_diagnostics_passed']
    (outer/'resources.json').write_text(json.dumps(outer_resource,indent=2)+'\n')
    cost = json.loads((OUT/'cost_plan.json').read_text());assert cost['estimated_cost_admitted']
    assert len(cost['jobs'])==8 and cost['cumulative_estimated_cpu_seconds']<=3600
    jobs = json.loads((OUT/'jobs.json').read_text()) if (OUT/'jobs.json').exists() else []
    owned = {j['task_id'] for j in jobs}
    live = [t for t in json.loads(QUEUE.read_text())['tasks'] if t.get('project')=='KG_op'
        and t['id'] not in owned and t['status'] in ('queued','launching','running')]
    assert not live,'Existing KG tasks occupy the eight-worker window'
    template = json.loads((ROOT/'paper_artifacts/historical_certificate_bound_full_v2/jobs.json').read_text())[0]['args']
    for row in cost['jobs']:
        if any(j['identity']==row['identity'] and j['shard']==row['shard'] for j in jobs):continue
        args = template.copy();h,s = row['identity'],row['shard']
        args[args.index('--description')+1] = f'KG body16 marginal bound identity{h} policy shard{s}'
        args[args.index('--signature')+1] = f'KG_op/historical_certificate_bound_body16_v3/{h}/{s}'
        args[args.index('--cmd')+1] = shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m','performance.probe_historical_certificate_bound','run','--identity',str(h),'--shard',str(s),'--plan',PLAN])
        jobs.append(dict(row,**submit(args,OUT,f'job_{h}_{s}.json')))
        (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
        print('SUBMITTED '+jobs[-1]['task_id'],flush=True)
    args = [sys.executable,CLI,'dispatch','--no-bulk-window','--lock-timeout','20']
    for j in jobs:args += ['--task-id',j['task_id']]
    p = subprocess.run(args,text=True,capture_output=True)
    (OUT/'dispatch.json').write_text(json.dumps(dict(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode),indent=2)+'\n');assert p.returncode==0
    tasks = wait([j['task_id'] for j in jobs]);resources = [fetch(t,OUT) for t in tasks]
    (OUT/'worker_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
    args = json.loads((ROOT/'paper_artifacts/historical_certificate_bound_full_v2/collection_scheduler_spec.json').read_text())['args']
    args[args.index('--description')+1] = 'KG body16 cached reference containment and exclusion'
    args[args.index('--signature')+1] = 'KG_op/historical_certificate_bound_body16_v3/collect'
    args[args.index('--cmd')+1] = shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m','performance.probe_historical_certificate_bound','collect','--plan',PLAN])
    collector = submit(args,OUT,'collection_scheduler_spec.json')
    p = subprocess.run([sys.executable,CLI,'dispatch','--task-id',collector['task_id'],'--no-bulk-window','--lock-timeout','20'],text=True,capture_output=True)
    (OUT/'collection_dispatch.json').write_text(json.dumps(dict(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode),indent=2)+'\n');assert p.returncode==0
    task = wait([collector['task_id']])[0];resource = fetch(task,OUT)
    (OUT/'collection_resources.json').write_text(json.dumps(resource,indent=2)+'\n')
    summary = pull('paper_artifacts/historical_certificate_bound_body16_v3','compact_summary.json')
    account = dict(complete=True,estimated_cpu_seconds_with_reserves=cost['cumulative_estimated_cpu_seconds'],
        measured_bound_process_cpu_seconds=sum(r['process_cpu_seconds'] for r in resources)+resource['process_cpu_seconds'],
        measured_outer_replay_process_cpu_seconds=outer_resource['process_cpu_seconds'],maximum_live_workers=8,
        remaining_group_record_checks=summary['unreferenced_group_record_checks_remaining'],
        numerical_gate_passed=summary['raw_marginal_numerical_gate_passed'],containment_passed=summary['cached_reference_containment_passed'],
        new_joint_terminal_integrations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    (OUT/'execution_account.json').write_text(json.dumps(account,indent=2)+'\n')
    (OUT/'scheduler_final_states.json').write_text(json.dumps({t['id']:{k:t.get(k) for k in ('status','node','log_path','exit_code')} for t in tasks+[task]},indent=2)+'\n')
    print('BODY16_DONE '+json.dumps(account),flush=True)


if __name__=='__main__':main()
