"""Complete the frozen raw incumbent references using eight shared slots."""
import json
import shlex
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from performance.launch_historical_component_bank import ROOT,CLI,REMOTE,PYTHON,QUEUE,launch,collect_remote
sys.path.insert(0,str(Path(CLI).parent))
import scheduler

plan='performance/manifests/historical_positive_risk_full_grid_incumbents_v1.json'
spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output'];prep=json.loads((out/'preparation.json').read_text())
assert prep['estimated_cost_admitted']
status=dict(state='running',registered_workers=len(prep['jobs']),maximum_live_workers=8,
    estimated_phase_cpu_seconds=prep['estimated_cpu_seconds_including_reserves'],
    previous_phase_completion=spec['previous_phase_completion'],shared_reserve_cpu_seconds=600,
    full_predictive_integration_admitted=False,independent_confirmation=False)
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
try:
    launch('observed-batch-risk',plan)
    summary=collect_remote('observed-batch-risk',out,plan)
    status['reference_precision_passed']=summary['local_risk_candidate_passed']
    if not summary['local_risk_candidate_passed']:
        status.update(state='precision_hold',failed_records=[r['record'] for r in summary['groups'] if not r['candidate_passed']])
    else:
        following=spec['followup_bound_plan'];follow=json.loads((ROOT/following).read_text());bound_out=ROOT/follow['output']
        screened=collect_remote('observed-batch-bound',bound_out,following)
        status.update(state='containment_hold',remaining_totals=screened['totals'],weighted_bank_envelope=screened['weighted_bank_envelope'])
        if screened['numerical_containment_passed']:
            priced=collect_remote('observed-batch-next-plan',bound_out,following)
            status.update(next_cost_admitted=priced['cost_admitted'],next_estimated_phase_cpu_seconds=priced['expected_cpu_seconds_including600reserves'])
            worker=['performance.replay_historical_positive_risk_subset','--plan',following,'--output',spec['output']+'/raw_incumbent_replay.json']
            command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m']+worker)
            rc,body,err=scheduler.run_on('node005',command,timeout=35,check=False)
            (out/'raw_incumbent_replay.log').write_text(body+err)
            if rc:raise RuntimeError(err or body)
            replay=json.loads([s[17:] for s in body.splitlines() if s.startswith('SCHEDULEURM_DONE ')][-1])
            (out/'raw_incumbent_replay.json').write_text(json.dumps(replay,indent=2)+'\n')
            status.update(state='phase_complete',native_incumbents_all_raw_referenced=replay['native_choices_all_raw_referenced'],
                raw_reward_of_native_incumbent_contrast=replay['weighted_raw_reward_of_native_incumbent_contrast'],
                raw_best_admitted_contrast=replay['weighted_raw_best_admitted_contrast'],
                projected_incumbent_contrast=replay['weighted_projected_incumbent_contrast'])
    jobs=json.loads((out/'observed-batch-risk_jobs.json').read_text());ids={j['id'] for j in jobs}
    tasks=[t for t in json.loads(QUEUE.read_text())['tasks'] if t['id'] in ids];resources=[]
    for node in sorted({t['node'] for t in tasks}):
        paths={t['id']:t['log_path'] for t in tasks if t['node']==node}
        code='import json;from pathlib import Path;paths='+repr(paths)+';print(json.dumps({k:json.loads([s[21:] for s in Path(v).read_text().splitlines() if s.startswith("SCHEDULEURM_RESOURCE ")][-1]) for k,v in paths.items()}))'
        rc,body,err=scheduler.run_on(node,shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
        if rc:raise RuntimeError(err or body)
        rows=json.loads(body);resources.extend(dict(id=t['id'],node=node,status=t['status'],resource=rows[t['id']]) for t in tasks if t['node']==node)
    (out/'worker_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
    collector_logs=[out/'observed-batch-risk_collection.log',out/'raw_incumbent_replay.log']
    if 'following' in locals():collector_logs.extend(bound_out/(name+'_collection.log') for name in ('observed-batch-bound','observed-batch-next-plan'))
    collection=sum(json.loads(s[21:])['process_cpu_seconds'] for path in collector_logs if path.exists() for s in path.read_text().splitlines() if s.startswith('SCHEDULEURM_RESOURCE '))
    preparation=json.loads((out/'preparation_resources.json').read_text())['process_cpu_seconds']
    inspection=json.loads((out/'inspection_resources.json').read_text())['process_cpu_seconds']
    account=dict(complete=True,state=status['state'],estimated_phase_cpu_seconds=status['estimated_phase_cpu_seconds'],
        previous_phase_completion=spec['previous_phase_completion'],shared_reserve_cpu_seconds=600,
        worker_count=len(resources),scientific_work_cpu_seconds=summary['successful_worker_cpu_seconds'],
        worker_process_cpu_seconds=sum(r['resource']['process_cpu_seconds'] for r in resources),
        preparation_process_cpu_seconds=preparation,inspection_process_cpu_seconds=inspection,
        collection_process_cpu_seconds=collection,peak_worker_rss_mb=max(r['resource']['peak_rss_mb'] for r in resources),
        maximum_live_workers=8,reference_groups=len(summary['groups']),reference_precision_passed=summary['local_risk_candidate_passed'],
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,full_predictive_integration_admitted=False,independent_confirmation=False)
    account['total_recorded_process_cpu_seconds']=account['worker_process_cpu_seconds']+preparation+inspection+collection
    (out/'execution_account.json').write_text(json.dumps(account,indent=2)+'\n')
except Exception as error:
    status.update(state='failed',error=str(error));raise
finally:
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print('POSITIVE_GRID_INCUMBENTS_DONE '+json.dumps(status),flush=True)
