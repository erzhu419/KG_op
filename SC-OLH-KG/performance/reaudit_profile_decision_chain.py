"""Read-only reconstruction of fixed confirmation decisions; no new simulations."""
import os,sys,json
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
from problems.randomized_profiles import RandomizedOrderedProfileProblem,generate_structural_profile_library
from performance.benchmark_profile_stress_suite import _stable_seed
records=[r for r in json.loads((root/'paper_artifacts/submission_revision_20260930/profile_rows.json').read_text())['rows'] if r['study']=='confirmation']
libraries={f:{p.profile_id:p for p in generate_structural_profile_library(64,dimension=128,seed=f+991,maximum_frequency=40)} for f in {r['family_seed'] for r in records}}
rows=[];cache={}
for row in records:
 key=(row['family_seed'],row['regime'],row['target_seed'])
 if key not in cache:
  target=RandomizedOrderedProfileProblem(regime=row['regime'],role='target',task_seed=row['target_seed'],family_seed=row['family_seed']+_stable_seed(row['regime']),d=1000)
  cache[key]=(target,{})
 target,truth=cache[key];points=[];probabilities=[];margins=[];observations=[]
 for i,pid in enumerate(row['selected_profile_ids']):
  if pid not in truth:
   point=target.point_from_structural_profile(libraries[row['family_seed']][pid]);truth[pid]=(point,target.true_outputs(point),target.true_sigma(point),target.true_feasibility_probability(point),target.true_chance_margin(point))
  point,mean,scale,p,margin=truth[pid];points.append(point);probabilities.append(p);margins.append(margin)
  rng=np.random.default_rng(np.random.SeedSequence([row['design_seed'],i,2017]));observations.append(mean+rng.normal(0.,scale))
 obs=np.array(observations);ps=np.array(probabilities);safe=np.array(margins)<=0
 assert bool(safe.any())==row['initial_design_contains_true_feasible'],key
 eligible=np.flatnonzero(obs[:,1]<=target.tau).tolist();roles=[]
 if eligible:
  roles += [min(eligible,key=lambda i:obs[i,0]), min(eligible,key=lambda i:obs[i,1])]
 roles += [min(range(10),key=lambda i:obs[i,0]+2*max(obs[i,1]-target.tau,0))]
 roles += sorted(range(10),key=lambda i:obs[i,0])
 chosen=[]
 for i in roles:
  if i not in chosen:chosen.append(i)
  if len(chosen)==3:break
 # Reconstruct only the original verification streams to validate the diagnostic.
 verified=False;false=False;calls=0
 for candidate_index,i in enumerate(chosen):
  mean=truth[row['selected_profile_ids'][i]][1];scale=truth[row['selected_profile_ids'][i]][2]
  all_success=True
  for replication in range(80):
   rng=np.random.default_rng(np.random.SeedSequence([row['design_seed']+71003,candidate_index,replication,0x42494E4F]))
   sample=mean+rng.normal(0.,scale)
   if sample[1]>target.tau:all_success=False
  calls+=80
  if all_success:
   verified=True;false=not bool(safe[i]);break
 assert verified==row['independently_certified'] and false==row['false_certificate'] and calls==row['verification_calls'],('verification reconstruction mismatch',key,row['arm'])
 survival=1.;expected_true=0.;expected_false=0.
 for i in chosen:
  q=float(ps[i]**80)
  if safe[i]:expected_true+=survival*q
  else:expected_false+=survival*q
  survival*=1-q
 rows.append(dict(arm=row['arm'],family_seed=row['family_seed'],regime=row['regime'],target_seed=row['target_seed'],design_coverage=bool(safe.any()),shortlist_coverage=bool(safe[chosen].any()),design_has_p99=bool(np.any(ps>=.99)),shortlist_has_p99=bool(np.any(ps[chosen]>=.99)),expected_true_certificate=expected_true,expected_false_certificate=expected_false,observed_true_certificate=bool(row['certified_true_feasible']),first_safe=bool(safe[0]),first_expected_true_certificate=float(ps[0]**80) if safe[0] else 0.,first_in_shortlist=bool(0 in chosen)))
def total(selected):
 return dict(runs=len(selected),**{k:sum(r[k] for r in selected) for k in ('design_coverage','shortlist_coverage','design_has_p99','shortlist_has_p99','expected_true_certificate','expected_false_certificate','observed_true_certificate','first_safe','first_in_shortlist')})
summary=dict(role='post hoc diagnostic conditional on the frozen task law, designs and original search observations; no method choice or new verification outcomes',new_source_simulations=0,new_target_simulations=0,verified_original_decisions=len(rows),reconstructed_original_search_observations=14400,groups={arm:total([r for r in rows if r['arm']==arm]) for arm in sorted({r['arm'] for r in rows})},source_atlas_by_regime={regime:total([r for r in rows if r['arm']=='source_atlas' and r['regime']==regime]) for regime in sorted({r['regime'] for r in rows})})
summary['source_first_only_analytic']=dict(runs=240,true_feasible_count=sum(r['first_safe'] for r in rows if r['arm']=='source_atlas'),expected_true_certificates_with_80_draws=sum(r['first_expected_true_certificate'] for r in rows if r['arm']=='source_atlas'),role='hypothetical single-center analytic expectation; original error allocation retained conservatively; no observed singleton experimental outcomes')
summary['code']='performance/reaudit_profile_decision_chain.py'
summary['expected_certificate_scope']='Expectation over fresh iid verification streams conditional on frozen design and reconstructed original noisy search/shortlist; no averaging over new task or search outcomes'
summary['p99_scope']='Descriptive deep-safety diagnostic only; certification threshold remains .95'
out=root/'paper_artifacts/decision_chain_reaudit_20261002';out.mkdir(exist_ok=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(summary['groups'],indent=2))
