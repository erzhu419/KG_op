"""Reuse the historical continuation for the registered full n4 risk grid."""
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from performance.launch_historical_component_bank import ROOT,CLI,DEPLOY,PYTHON,QUEUE,REMOTE,launch,collect_remote,refresh_jobs

sys.path.insert(0,str(Path(CLI).parent))
import scheduler

parser=argparse.ArgumentParser()
parser.add_argument('--source-bound',default='performance/manifests/historical_positive_risk_full_grid_screen_competition_v5.json')
parser.add_argument('--output',default='paper_artifacts/historical_positive_risk_full_grid_continuation_v1')
parser.add_argument('--maximum-new-phases',type=int)
campaign_options=parser.parse_args()
out=ROOT/campaign_options.output;out.mkdir(parents=True,exist_ok=True)
source=campaign_options.source_bound
source_spec=json.loads((ROOT/source).read_text())
closed='performance/manifests/'+Path(source_spec['phase_completion']).parent.name+'.json'
initial=json.loads((ROOT/closed).read_text())
initial_summary=json.loads((ROOT/initial['output']/'compact_summary.json').read_text())
assert initial_summary['local_risk_candidate_passed']
initial_completion=json.loads((ROOT/initial['output']/'completion.json').read_text())
assert initial_completion['state']=='phase_complete' and initial_completion['all_reference_precision_passed']
status=dict(state='running',source_bound_plan=source,completed_phases=[closed],
    phase_estimates={closed:initial_summary['estimated_cpu_seconds_including_reserves']},
    continuation_worker_cpu_seconds=initial_summary['successful_worker_cpu_seconds'],
    recorded_process_cpu_seconds=(initial_completion['total_recorded_process_cpu_seconds']
        if 'total_recorded_process_cpu_seconds' in initial_completion else initial_completion['recorded_process_cpu_seconds']),
    maximum_live_workers=8,full_predictive_integration_admitted=False,independent_confirmation=False,
    historical_controller_source='paper_artifacts/historical_positive_risk_continuation_v3/continue_when_ready.py',
    scope='Continuation from the registered closed grid phase. Earlier costs remain linked in its completion ledger;outer-axis refinement is separate.')
if (out/'status.json').exists():
    status=json.loads((out/'status.json').read_text())
    source=status['source_bound_plan']
    if status['state']=='same_phase_references_ready':status.pop('current_phase',None)
status['maximum_new_phases']=campaign_options.maximum_new_phases


def save_status():
    status['total_phase_estimated_cpu_seconds']=sum(status['phase_estimates'].values())
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')


def publish_completion(phase):
    """Make the closed local account available to the server preparation."""
    names=['completion.json']
    if (ROOT/phase['output']/'status.json').exists():names.append('status.json')
    for name in names:
        relative=phase['output']+'/'+name
        data=(ROOT/relative).read_text()
        target=Path(DEPLOY)/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(data)
        code='from pathlib import Path;p=Path('+repr(REMOTE+'/'+relative)+');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('+repr(data)+')'
        rc,body,err=scheduler.run_on('node005',shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
        if rc:raise RuntimeError(err or body)


try:
    publish_completion(initial)
    status.pop('error',None)
    while True:
        if 'current_phase' in status:
            tag=Path(status['current_phase']).stem.split('_competition_',1)[1]
            result=json.loads((out/f'preparation_{tag}_result.json').read_text())['result']
        else:
            previous=json.loads((ROOT/source).read_text())
            screened=json.loads((ROOT/previous['output']/'compact_summary.json').read_text())
            assert screened['numerical_containment_passed']
            status['remaining_totals']=screened['totals'];save_status()
            if not any(r['record_group_candidates'] for r in screened['totals'].values()):
                status['state']='conditional_actions_closed';break
            if campaign_options.maximum_new_phases is not None and len(status['completed_phases'])-1>=campaign_options.maximum_new_phases:
                status['state']='phase_limit_hold';break
            initial_name=Path(previous['phase_completion']).parent.name
            initial_plan='performance/manifests/'+initial_name+'.json'
            continuing='same_phase_cost_plan' in status
            if continuing:
                suffix=len(previous['joint_reference_plans'])-previous['joint_reference_plans'].index(initial_plan)
                tag=initial_name.split('_competition_',1)[1]+f'_refs_v{suffix}'
            else:tag='v'+str(int(initial_name.rsplit('_v',1)[1])+1)
            mode='prepare-continued' if continuing else 'prepare-next'
            if status.get('state')=='preparing':
                tid=status['preparation_id']
            else:
                cmd=shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,
                    '-u','-m','performance.positive_historical_direct_references',mode,'--plan',source])
                args=[sys.executable,CLI,'submit','--description',f'KG full-grid n4 {tag} matched preparation',
                    '--cmd',cmd,'--cwd',DEPLOY,'--signature',f'KG_op/{out.name}/{tag}/prepare',
                    '--cpu','1','--ram-mb','2000','--vram','0','--priority','low','--project','KG_op',
                    '--skip-launch-staging','--env-spec','none']
                for node in range(1,7):args+=['--allowed-node',f'node{node:03d}']
                p=subprocess.run(args,text=True,capture_output=True)
                (out/f'preparation_{tag}_spec.json').write_text(json.dumps(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
                if p.returncode:raise RuntimeError(p.stderr or p.stdout)
                tid=re.search(r'submitted (t\d+)',p.stdout).group(1)
                status.update(state='preparing',preparation_id=tid);save_status()
                args=[sys.executable,CLI,'dispatch','--no-bulk-window','--lock-timeout','20','--task-id',tid]
                p=subprocess.run(args,text=True,capture_output=True)
                (out/f'preparation_{tag}_dispatch.json').write_text(json.dumps(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
                if p.returncode:
                    if 'SchedulerLockTimeout:' not in p.stderr:raise RuntimeError(p.stderr or p.stdout)
                    print('PREPARATION_DISPATCH_DEFERRED '+tid+' scheduler lock busy; retaining the submitted task for the scheduler watcher',flush=True)
            while True:
                refresh_jobs({tid})
                task=next(t for t in json.loads(QUEUE.read_text())['tasks'] if t['id']==tid)
                if task['status']=='done':break
                if task['status'] not in ('queued','launching','running'):raise RuntimeError(f'{tid}: '+task['status'])
                time.sleep(20)
            log_path=task.get('log_path') or f'/tmp/sched_{tid}.log'
            code='import json;from pathlib import Path;lines=Path('+repr(log_path)+').read_text().splitlines();print(json.dumps({"result":json.loads([s[17:] for s in lines if s.startswith("SCHEDULEURM_DONE ")][-1]),"resource":json.loads([s[21:] for s in lines if s.startswith("SCHEDULEURM_RESOURCE ")][-1])}))'
            rc,body,err=scheduler.run_on(task['node'],shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
            if rc:raise RuntimeError(err or body)
            prepared=json.loads(body);prepared.update(id=tid,node=task['node']);result=prepared['result']
            (out/f'preparation_{tag}_result.json').write_text(json.dumps(prepared,indent=2)+'\n')
            files={result['pricing_plan']:result['pricing_manifest'],result['pricing_manifest']['output']+'/next_cost_plan.json':result['priced']}
            if result['prepared']:
                files.update({result['risk_plan']:result['risk_manifest'],result['bound_plan']:result['bound_manifest'],
                    result['risk_manifest']['output']+'/preparation.json':result['preparation']})
            for name,data in files.items():
                for root in (ROOT,Path(DEPLOY)):
                    target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(data,indent=2)+'\n')
            if not result['prepared']:status['state']=result['state'];break
            if continuing:
                (ROOT/result['risk_manifest']['output']/'preparation_resources.json').write_text(json.dumps(prepared['resource'],indent=2)+'\n')
        plan=result['risk_plan'];phase=result['risk_manifest'];phase_out=ROOT/phase['output']
        phase_status=dict(state='running',registered_workers=len(result['preparation']['jobs']),maximum_live_workers=8,
            cumulative_phase_estimated_cpu_seconds=result['preparation']['estimated_cpu_seconds_including_reserves'],
            shared_phase_reserve_cpu_seconds=600,independent_confirmation=False)
        phase_status_path=phase_out/'status.json'
        phase_status_path.write_text(json.dumps(phase_status,indent=2)+'\n')
        status.update(state='running',current_phase=plan)
        initial_name=Path(result['bound_manifest']['phase_completion']).parent.name
        initial_plan='performance/manifests/'+initial_name+'.json'
        status['phase_estimates'][initial_plan]=result['preparation']['estimated_cpu_seconds_including_reserves'];save_status()
        launch('observed-batch-risk',plan)
        summary=collect_remote('observed-batch-risk',phase_out,plan)
        recorded=(status.get('current_reference_worker_cpu_seconds',0.)
            if status.get('current_reference_worker_cpu_plan')==plan else 0.)
        status['continuation_worker_cpu_seconds']+=summary['successful_worker_cpu_seconds']-recorded
        status.update(current_reference_worker_cpu_plan=plan,
            current_reference_worker_cpu_seconds=summary['successful_worker_cpu_seconds'])
        if not summary['local_risk_candidate_passed']:
            status['state']='precision_hold';phase_status['state']='precision_hold'
            phase_status_path.write_text(json.dumps(phase_status,indent=2)+'\n');break
        source=result['bound_plan'];bound_out=ROOT/result['bound_manifest']['output']
        screened=collect_remote('observed-batch-bound',bound_out,source)
        if not screened['numerical_containment_passed']:
            status['state']='containment_hold';phase_status['state']='containment_hold'
            phase_status_path.write_text(json.dumps(phase_status,indent=2)+'\n');break
        priced=collect_remote('observed-batch-next-plan',bound_out,source)
        phase_status.update(state='joint_references_admitted',remaining_totals=screened['totals'],
            next_cost_admitted=priced['cost_admitted'],next_priced_cases=len(priced['next_joint_cases']))
        phase_status_path.write_text(json.dumps(phase_status,indent=2)+'\n')
        if priced['cost_admitted']:
            status.update(state='same_phase_references_ready',source_bound_plan=source,
                remaining_totals=screened['totals'],same_phase_cost_plan=result['bound_manifest']['output']+'/next_cost_plan.json')
            status.pop('current_phase');save_status();continue
        subprocess.run([sys.executable,str(Path(__file__).with_name('close_phase.py')),'--plan',source,
            '--controller-output',str(out.relative_to(ROOT))],cwd=ROOT,check=True)
        initial_phase=json.loads((ROOT/initial_plan).read_text())
        completed=json.loads((ROOT/initial_phase['output']/'completion.json').read_text())
        assert completed['state']=='phase_complete'
        publish_completion(initial_phase)
        status['recorded_process_cpu_seconds']+=completed['total_recorded_process_cpu_seconds']
        status['completed_phases'].append(initial_plan);status.update(source_bound_plan=source,remaining_totals=screened['totals'])
        status.pop('same_phase_cost_plan',None)
        status.pop('current_phase')
        status.pop('current_reference_worker_cpu_plan',None)
        status.pop('current_reference_worker_cpu_seconds',None);save_status()
        print('POSITIVE_PHASE_DONE '+json.dumps(status),flush=True)
except Exception as error:
    status.update(state='failed',error=str(error));raise
finally:
    save_status()
print('POSITIVE_CONTINUATION_DONE '+json.dumps(status),flush=True)
