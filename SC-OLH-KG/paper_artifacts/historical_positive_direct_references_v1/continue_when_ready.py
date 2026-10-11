"""Run all sixteen priced direct-reference jobs, then re-screen the same bank."""
import json

from performance.launch_historical_component_bank import ROOT, launch, collect_remote

plan='performance/manifests/historical_positive_direct_references_v1.json'
following='performance/manifests/historical_positive_risk_screen_with_direct_v1.json'
spec=json.loads((ROOT/plan).read_text());out=ROOT/spec['output']
prep=json.loads((out/'preparation.json').read_text());assert prep['estimated_cost_admitted']
status=dict(state='running',registered_workers=len(prep['jobs']),maximum_live_workers=8,
    cumulative_phase_estimated_cpu_seconds=prep['estimated_cpu_seconds_including_reserves'],
    original_screen600reserve_retained=True,independent_confirmation=False)
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
launch('observed-batch-risk',plan)
summary=collect_remote('positive-direct-reference',out,plan)
if summary['local_risk_candidate_passed']:
    follow=json.loads((ROOT/following).read_text())
    result=collect_remote('observed-batch-bound',ROOT/follow['output'],following)
    status.update(state='direct_references_admitted' if result['numerical_containment_passed'] else 'containment_hold',
        remaining_totals=result['totals'],followup=follow['output']+'/compact_summary.json')
else:
    status.update(state='precision_hold',failed_records=[r['record'] for r in summary['groups'] if not r['candidate_passed']])
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print('POSITIVE_DIRECT_REFERENCE_DONE '+json.dumps(status),flush=True)
