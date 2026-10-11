import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
"""Finish the frozen workers and refill the existing all-action screen."""
import json
from performance.launch_historical_component_bank import ROOT,launch,collect_remote

plan='performance/manifests/historical_positive_risk_full_grid_competition_v5.json'
spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output']
prep=json.loads((out/'preparation.json').read_text());assert prep['estimated_cost_admitted']
status=dict(state='running',registered_workers=len(prep['jobs']),maximum_live_workers=8,
    cumulative_phase_estimated_cpu_seconds=prep['estimated_cpu_seconds_including_reserves'],
    shared_phase_reserve_cpu_seconds=600,independent_confirmation=False)
path=out/'status.json';path.write_text(json.dumps(status,indent=2)+'\n')
try:
    launch('observed-batch-risk',plan)
    summary=collect_remote('observed-batch-risk',out,plan)
    if summary['local_risk_candidate_passed']:
        following=spec['followup_bound_plan'];follow=json.loads((ROOT/following).read_text())
        screened=collect_remote('observed-batch-bound',ROOT/follow['output'],following)
        status.update(state='joint_references_admitted' if screened['numerical_containment_passed'] else 'containment_hold',
            remaining_totals=screened['totals'],followup=follow['output']+'/compact_summary.json')
        if screened['numerical_containment_passed']:
            priced=collect_remote('observed-batch-next-plan',ROOT/follow['output'],following)
            status.update(next_cost_admitted=priced['cost_admitted'],next_priced_cases=len(priced['next_joint_cases']))
    else:
        status.update(state='precision_hold',failed_records=[r['record'] for r in summary['groups'] if not r['candidate_passed']])
except Exception as error:
    status.update(state='failed',error=str(error));raise
finally:
    path.write_text(json.dumps(status,indent=2)+'\n')
print('POSITIVE_RISK_COMPETITION_DONE '+json.dumps(status),flush=True)

if status["state"]=="joint_references_admitted" and not status["next_cost_admitted"]:
    import subprocess
    subprocess.run([sys.executable,str(ROOT/"paper_artifacts/historical_positive_risk_full_grid_continuation_v1/close_phase.py"),"--plan",spec["followup_bound_plan"]],cwd=ROOT,check=True)
