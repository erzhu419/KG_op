"""Admit the frozen raw probes within each existing axis phase."""
import json
import shlex
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from performance.launch_historical_component_bank import CLI,PYTHON,REMOTE,QUEUE,launch,collect_remote
sys.path.insert(0,str(Path(CLI).parent))
import scheduler

out=Path(__file__).resolve().parent
frozen=json.loads((out/'incumbent_frozen_plan.json').read_text())
status=dict(state='running',completed_plans=[],maximum_live_workers=8,
    full_predictive_integration_admitted=False,independent_confirmation=False)
try:
    for plan in frozen['plans']:
        spec=json.loads((ROOT/plan).read_text());folder=ROOT/spec['output']
        status['current_plan']=plan;(out/'incumbent_status.json').write_text(json.dumps(status,indent=2)+'\n')
        launch('observed-batch-risk',plan)
        references=collect_remote('observed-batch-risk',folder,plan)
        if not references['local_risk_candidate_passed']:
            status.update(state='precision_hold');break
        bound=spec['followup_bound_plan'];bs=json.loads((ROOT/bound).read_text());target=ROOT/bs['output']
        screened=collect_remote('observed-batch-bound',target,bound)
        if not screened['numerical_containment_passed']:
            status.update(state='containment_hold');break
        collect_remote('observed-batch-next-plan',target,bound)
        destination=spec['output']+'/incumbent_replay_final.json'
        cmd='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,
            '-u','-m','performance.replay_historical_positive_risk_subset','--plan',bound,'--output',destination])
        rc,body,err=scheduler.run_on('node005',cmd,timeout=35,check=False);(folder/'incumbent_replay_final.log').write_text(body+err)
        if rc:raise RuntimeError(err or body)
        replay=json.loads([s[17:] for s in body.splitlines() if s.startswith('SCHEDULEURM_DONE ')][-1])
        (ROOT/destination).write_text(json.dumps(replay,indent=2)+'\n')
        jobs=json.loads((folder/'observed-batch-risk_jobs.json').read_text());ids={j['id'] for j in jobs}
        tasks=[t for t in json.loads(QUEUE.read_text())['tasks'] if t['id'] in ids];resources=[]
        for node in sorted({t['node'] for t in tasks}):
            paths={t['id']:t['log_path'] for t in tasks if t['node']==node}
            code='import json;from pathlib import Path;paths='+repr(paths)+';print(json.dumps({k:json.loads([s[21:] for s in Path(v).read_text().splitlines() if s.startswith("SCHEDULEURM_RESOURCE ")][-1]) for k,v in paths.items()}))'
            rc,body,err=scheduler.run_on(node,shlex.join([PYTHON,'-c',code]),timeout=25,check=False)
            if rc:raise RuntimeError(err or body)
            values=json.loads(body);resources.extend(dict(id=t['id'],node=node,resource=values[t['id']]) for t in tasks if t['node']==node)
        (folder/'worker_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
        logs=[folder/'observed-batch-risk_collection.log',folder/'incumbent_replay_final.log',
            target/'observed-batch-bound_collection.log',target/'observed-batch-next-plan_collection.log']
        support=json.loads((folder/'preparation_resources.json').read_text())['process_cpu_seconds']+sum(
            json.loads(s[21:])['process_cpu_seconds'] for p in logs for s in p.read_text().splitlines() if s.startswith('SCHEDULEURM_RESOURCE '))
        account=dict(state='references_admitted',registered_workers=len(resources),shared_phase_reserve_cpu_seconds=600,
            estimated_phase_cpu_seconds=references['estimated_cpu_seconds_including_reserves'],
            additional_recorded_process_cpu_seconds=support+sum(r['resource']['process_cpu_seconds'] for r in resources),
            additional_support_cpu_seconds=support,native_choices_all_raw_referenced=replay['native_choices_all_raw_referenced'],
            raw_covered_grid_weight=replay['raw_covered_grid_weight'],missing_raw_observations=len(replay['missing_raw_references']),
            weighted_bank_envelope=screened['weighted_bank_envelope'],remaining_totals=screened['totals'],
            full_predictive_integration_admitted=False,independent_confirmation=False)
        (folder/'execution_account.json').write_text(json.dumps(account,indent=2)+'\n');status['completed_plans'].append(plan)
    else:
        status.update(state='incumbent_probes_complete');status.pop('current_plan')
except Exception as error:
    status.update(state='failed',error=str(error));raise
finally:
    (out/'incumbent_status.json').write_text(json.dumps(status,indent=2)+'\n')
print('INCUMBENT_PROBES_DONE '+json.dumps(status),flush=True)
