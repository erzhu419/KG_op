"""Check only the two cached guarded/direct competition conflicts."""
import json
import numpy as np

from performance.complete_historical_source_heads import ROOT,read,save
from performance.diagnose_historical_component_guarded import compressed_read,mixed_values
from performance.validate_historical_component_bank import group_checks
from performance.validate_historical_component_transfer import record_bundle,part_path
from performance.validate_historical_label_control import tables
from performance.validate_historical_label_bank import valid_tables

out=ROOT/'paper_artifacts/historical_decision_regions_v1';work=read(str(out/'pilot_plan.json'))
contexts=read(str(out/'contexts.json'));direct_contexts=read(str(out/'direct_contexts.json'))
layout=read('paper_artifacts/historical_cdf_bank_v1/bank_layout.json');results=[];tasks=[];cpu=0.
shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals/063.json.gz')
for shard,row in enumerate(work['selected']):
    context=contexts[row['record']];bundle=record_bundle(shared,context)
    parts={h:read(str(part_path(out,row,h)))['rows'] for h in bundle['utility_identities']}
    data={r:tables(bundle,{h:parts[h][r] for h in parts}) for r in row['rules']}
    checks=group_checks(bundle,[dict(identity=h,rows=parts[h]) for h in parts],'projected',row['pairs'],1e-4)
    checks['valid_unclipped_ranges']=all(valid_tables(bundle,d) for d in data.values())
    checks['passed']=bool(checks['passed'] and checks['valid_unclipped_ranges'])
    direct=direct_contexts[row['record']]['groups']['136'];direct_tables=direct['baseline']['projected']
    opposite=[]
    for g,d in zip(bundle['opposite'],direct['opposite']):
        assert g['head_log_scores']==d['head_log_scores']
        opposite.append(dict(head_log_scores=g['head_log_scores'],projected=np.concatenate([g['projected'],d['projected']],axis=1).tolist()))
    merged=dict(bundle,opposite=opposite)
    statistics=np.concatenate([data['E8'][0],direct_tables['statistics']],axis=1)
    costs=np.concatenate([data['E8'][1],direct_tables['costs']],axis=1)
    values=mixed_values(merged,statistics,costs,'projected')
    indices=layout['groups'][63]['action_indices']+layout['groups'][136]['action_indices']
    recommendations=np.asarray(indices)[values.argmax(1)]
    results.append(dict(record=row['record'],checks=checks,
        admitted=checks['passed'],recommendations=recommendations.tolist(),
        guarded_recommendation_count=int(np.count_nonzero(np.isin(recommendations,indices[:3]))),
        minimum_direct_advantage=float((values[:,3:].max(1)-values[:,:3].max(1)).min())))
    for h in parts:
        cpu+=sum(parts[h][r]['cpu_seconds'] for r in row['rules'])
        tasks.append(read(str(out/'tasks/pilot'/f'{h}_projected_{shard}.json')))
result=dict(complete=all(t['complete'] for t in tasks),all_references_admitted=all(r['admitted'] for r in results),
    records=results,new_integral_cpu_seconds=cpu,successful_worker_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),
    maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
    independent_confirmation=False,full_predictive_integration_admitted=False,
    scope='Two development records,all32 cached objective heads; only guarded group63 and analytic native direct group136 compared. Original365-action class and all8 identity scores/weights preserved,other groups unvalidated. A failed reference remains HOLD. No full KG or independent decision claim.')
save(out/'native_reference_summary.json',result);print(json.dumps(result))
