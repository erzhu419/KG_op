import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
import performance.benchmark_profile_stress_suite as b
from performance.run_profile_stress_matrix import derived_target_seed,derived_design_seed
from core.profile_atlas import profile_cosine_coordinate,farthest_first_indices
freeze='da8f1e5c594dace1cc667a2e4b87956b1001b67b'

def standardized_control(profiles,count,**kw):
 c=np.vstack([profile_cosine_coordinate(p,**kw) for p in profiles])
 scale=np.std(c,axis=0); scale=np.where(scale>1e-10,scale,1.0)
 c=(c-np.mean(c,axis=0,keepdims=True))/scale[None,:]
 dist=np.linalg.norm(c[:,None,:]-c[None,:,:],axis=2)
 first=int(np.argmin(np.mean(dist,axis=1)))
 ids=farthest_first_indices(c,int(count),initial_index=first)
 return ids,{'contract_id':'posthoc_standardized_outcome_free_medoid_diagnostic','selected_indices':list(ids),'source_outcomes_used':False,'target_outcomes_used':False}

b.generic_dct_maximin=standardized_control
rows=[]; t=time.time()
for regime in b.PROFILE_STRESS_REGIMES:
 for replicate in range(20):
  r=b.run_task(regime=regime,target_seed=derived_target_seed(freeze,regime,replicate),design_seed=derived_design_seed(freeze,regime,replicate),arm='generic_dct_maximin',dimension=1000)
  rows.append({k:r[k] for k in ['regime','target_seed','design_seed','initial_design_contains_true_feasible','independently_certified','false_certificate','penalized_loss','source_calls','target_search_calls','verification_calls','all_in_calls_unamortized','deployed_truth','selected_profile_ids']})
  if len(rows)==1: print('first_task_seconds',round(time.time()-t,3),flush=True)
 print(regime, 'n',len(rows),'elapsed_seconds',round(time.time()-t,2),flush=True)
 Path('/tmp/kg_standardized_generic_diagnostic.json').write_text(json.dumps({'interpretation':'post-hoc diagnostic on existing confirmatory task seeds; not new confirmatory evidence','freeze_commit':freeze,'dimension':1000,'rows':rows},indent=2)+'\n')
print(json.dumps({'n':len(rows),'feasible':sum(r['initial_design_contains_true_feasible'] for r in rows),'certified_true':sum(r['independently_certified'] and not r['false_certificate'] for r in rows),'false_certificate':sum(r['false_certificate'] for r in rows),'mean_penalized_loss':float(np.mean([r['penalized_loss'] for r in rows]))}),flush=True)
