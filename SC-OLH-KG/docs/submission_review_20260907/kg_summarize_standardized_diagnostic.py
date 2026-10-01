import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
root=Path('/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
diag=json.loads(Path('/tmp/kg_standardized_generic_diagnostic.json').read_text())
key=lambda r:(r['regime'],int(r['target_seed']))
control={key(r):r for r in diag['rows']}
source={}
for p in (root/'results/or_review_v2_evalpatch4/profile_primary').glob('**/*d1000*source_atlas*.json'):
 r=json.loads(p.read_text())
 if r['nominal_dimension']!=1000:continue
 source[key(r)]={k:r[k] for k in ['regime','target_seed','design_seed','initial_design_contains_true_feasible','independently_certified','false_certificate','penalized_loss','source_calls','target_search_calls','verification_calls','all_in_calls_unamortized','all_in_calls_amortized','deployed_truth']}
if len(source)!=160 or set(source)!=set(control):raise RuntimeError('unpaired cells')
for k in source:
 if source[k]['design_seed']!=control[k]['design_seed']:raise RuntimeError('algorithm seed mismatch')
success=lambda r:bool(r['independently_certified'] and not r['false_certificate'] and r.get('deployed_truth',{}).get('feasible',False))
regimes=sorted({k[0] for k in source})
paired=np.asarray([[int(success(source[k]))-int(success(control[k])) for k in sorted(source) if k[0]==regime] for regime in regimes])
rng=np.random.default_rng(20260907)
idx=rng.integers(0,20,size=(10000,8,20))
boot=np.mean(paired[np.arange(8)[None,:,None],idx],axis=(1,2))
wins=int(np.sum(paired==1)); losses=int(np.sum(paired==-1)); ties=int(np.sum(paired==0))
compact=json.loads((root/'paper_artifacts/or_review/randomized_profile_primary.json').read_text())['aggregate_analysis']['summaries']
old=[r for r in compact if r['nominal_dimension']==1000 and r['arm']=='generic_dct_maximin']
def summarize(rows):
 return {'n':len(rows),'coverage_count':sum(r['initial_design_contains_true_feasible'] for r in rows),'certified_true_count':sum(success(r) for r in rows),'false_certificate_count':sum(r['false_certificate'] for r in rows),'mean_penalized_loss':float(np.mean([r['penalized_loss'] for r in rows])),'mean_all_in_calls_unamortized':float(np.mean([r['all_in_calls_unamortized'] for r in rows]))}
s=summarize(list(source.values())); c=summarize(list(control.values()))
s['mean_all_in_calls_amortized_M20']=float(np.mean([r['all_in_calls_amortized'] for r in source.values()]))
old_s={'n':sum(r['independent_task_count'] for r in old),'coverage_count':sum(r['initial_design_true_feasible_coverage_count'] for r in old),'certified_true_count':sum(r['certified_true_feasible_deployment_count'] for r in old),'false_certificate_count':sum(r['false_certificate_count'] for r in old),'mean_penalized_loss':float(np.mean([r['mean_penalized_loss'] for r in old])),'mean_all_in_calls_unamortized':float(np.mean([r['mean_all_in_calls_unamortized'] for r in old]))}
per=[]
for regime in regimes:
 per.append({'regime':regime,'source':summarize([r for r in source.values() if r['regime']==regime]),'standardized_generic':summarize([r for r in control.values() if r['regime']==regime]),'frozen_generic':[{k:r[k] for k in ['initial_design_true_feasible_coverage_count','certified_true_feasible_deployment_count','false_certificate_count']} for r in old if r['regime']==regime][0]})
result={'interpretation':'Post-hoc diagnostic on existing 160 task seeds at d=1000. Frozen source rows read from local results; outcome-free standardized control newly evaluated. Not independent confirmatory evidence. No frozen source or manuscript files edited.','control_change':'Apply source-identical z-standardization of the 18 profile coordinates before the generic medoid and farthest-first. Source labels absent. Other run_task logic unchanged.','source':s,'standardized_generic':c,'frozen_generic':old_s,'paired_source_minus_standardized_success':{'mean':float(np.mean(paired)),'fixed_regime_stratified_paired_bootstrap_95CI':[float(v) for v in np.quantile(boot,[.025,.975])],'wins':wins,'losses':losses,'ties':ties,'two_sided_exact_discordant_sign_p_unadjusted_posthoc':float(binomtest(wins,wins+losses).pvalue)},'per_regime':per}
Path('/tmp/kg_standardized_generic_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='per_regime'},indent=2))
