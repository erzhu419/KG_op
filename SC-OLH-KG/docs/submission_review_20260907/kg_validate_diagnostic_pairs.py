import json, math
from pathlib import Path
import numpy as np
root=Path('/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
p=Path('/tmp/kg_standardized_generic_comparison.json');result=json.loads(p.read_text())
summaries=json.loads((root/'paper_artifacts/or_review/randomized_profile_primary.json').read_text())['aggregate_analysis']['summaries']
key=lambda r:(r['regime'],r['target_seed'])
d={key(r):r for r in json.loads(Path('/tmp/kg_standardized_generic_diagnostic.json').read_text())['rows']}
validation={}
for arm in ['source_atlas','generic_dct_maximin']:
 rows=[json.loads(p.read_text()) for p in (root/'results/or_review_v2_evalpatch4/profile_primary').glob('**/*d1000__*'+arm+'*.json')]
 mapped={key(r):r for r in rows}
 assert len(rows)==160 and set(mapped)==set(d)
 assert all(r['nominal_dimension']==1000 and r['design_seed']==d[k]['design_seed'] for k,r in mapped.items())
 for ref in summaries:
  if ref['arm']!=arm or ref['nominal_dimension']!=1000:continue
  group=[r for r in rows if r['regime']==ref['regime']]
  assert sum(r['initial_design_contains_true_feasible'] for r in group)==ref['initial_design_true_feasible_coverage_count']
  assert sum(r['independently_certified'] and not r['false_certificate'] for r in group)==ref['certified_true_feasible_deployment_count']
  assert sum(r['false_certificate'] for r in group)==ref['false_certificate_count']
  assert math.isclose(np.mean([r['penalized_loss'] for r in group]),ref['mean_penalized_loss'],rel_tol=1e-12,abs_tol=1e-12)
 validation[arm]={'local_raw_rows':160,'same_task_and_algorithm_seed_as_diagnostic':True,'all_8_regime_coverage_certification_false_counts_and_mean_loss_match_frozen_compact':True}
result['raw_evidence_validation']=validation
result['cost_diagnostic']={'dimension':1000,'source_target_plus_verification_mean':178.,'source_call_count':384,'source_vs_frozen_generic_call_break_even_targets':math.ceil(384/(233-178)),'source_vs_standardized_generic_call_break_even_targets':math.ceil(384/(203.5-178)),'source_calls_per_true_certificate_at_M20':197.2/(74/160),'standardized_generic_calls_per_true_certificate':203.5/(49/160)}
p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'raw_evidence_validation':validation,'cost_diagnostic':result['cost_diagnostic']},indent=2))
