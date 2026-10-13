import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
from performance.run_meta_prior_mechanism import observe,task_truth,moments
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from representation.source_function_prior import FixedFunctionBasis,SourceFunctionPrior,head_batches
from representation.transfer_mixture import TransferMixtureGPR
from core.verification_kg import verification_utility,predictive_pass_probability
root=Path('/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
p=json.loads((root/'performance/meta_prior_total_budget_protocol.json').read_text())
cases=json.loads((root/'paper_artifacts/meta_prior_total_budget_v6_20261002/failure_cases.json').read_text());output=[]
for case in cases:
 row=case['row']
 if row['arm']!='risk_ard_mixture' or row['allocation']!='n24_v928':continue
 seed,reg=row['seed'],row['regime'];protocol={**p,**p['allocations'][row['allocation']]}
 psi=np.random.default_rng(51000+seed).uniform(-2,2,(384,8));reference=psi[:96]
 tasks,supports=prepare_sources(seed,reg,protocol);batches=[head_batches(tasks,h) for h in range(3)];noises=[[t['noise'][:,h] for t in tasks] for h in range(3)]
 full=[SourceFunctionPrior.fit(batches[h],noises[h],reference,'fixed_source') for h in range(3)]
 structural=[SourceFunctionPrior.from_basis(batches[h],noises[h],FixedFunctionBasis(reference,case['source_info'][h]['selected_indices']),'conditional_hard') for h in range(2)]
 basis=FixedFunctionBasis(reference);tau=np.array(case['source_info'][2]['source_function_variance']);signal=np.mean([max(float(np.var(b.signals)-np.mean(v)),.02) for b,v in zip(batches[2],noises[2])]);mu=np.zeros(69);mu[0]=np.mean([np.mean(b.signals) for b in batches[2]])
 structural.append(SourceFunctionPrior(basis,mu,np.diag(np.r_[.25,tau+.01*signal/68]),.02,{}))
 models=[TransferMixtureGPR([structural[h],full[h]],psi,reference) for h in range(3)]
 target_seed=61000+100*seed+p['regimes'].index(reg);truth=task_truth(psi,supports,target_seed);truth[:,1]+=1
 visits=np.zeros(384,dtype=int);obs=[]
 for step,i in enumerate(row['queries']):
  y,n=observe(truth[i],target_seed,i,int(visits[i]),12)
  for h,m in enumerate(models):m.update([i],y[h],n[h])
  visits[i]+=1
  if i==row['recommendation']:obs.append({'step':step+1,'response':y.tolist(),'noise':n.tolist(),'response_minus_truth':(y-truth[i]).tolist()})
 candidates=np.arange(384)[:,None];mean,var=moments(models,candidates);reward,power=verification_utility(mean,var,protocol);i=row['recommendation']
 assert abs(float(power[i])-row['predicted_verifier_power'])<1e-8
 for m,saved in zip(models,row['final_model_state']):assert abs(m.diagnostics()['structural_weight']-saved['structural_weight'])<1e-8
 diagnostic={'seed':seed,'regime':reg,'allocation':row['allocation'],'recommendation':i,'truth':truth[i].tolist(),'posterior_mean':mean[i].tolist(),'posterior_variance':var[i].tolist(),'predicted_pass':float(power[i]),'true_pass':row['true_verifier_power'],'replay_power_error':float(power[i]-row['predicted_verifier_power']),'observations_at_recommendation':obs,'component_moments':[]}
 for h,m in enumerate(models):diagnostic['component_moments'].append([{'mean':float(x.posterior_mean([i])),'variance':float(x.posterior_var([i])),'weight':float(np.exp(m.log_weights[j]))} for j,x in enumerate(m.models)])
 for h,name in [(1,'g_mean_replaced_by_truth'),(2,'eta_mean_replaced_by_truth')]:
  mm=mean[i].copy();mm[h]=truth[i,h];diagnostic[name]=float(predictive_pass_probability(mm,var[i],protocol))
 replacement=mean[i].copy();replacement[1:]=truth[i,1:];diagnostic['both_means_replaced_by_truth_keep_epistemic_variance']=float(predictive_pass_probability(replacement,var[i],protocol))
 initial_case=next(c['row'] for c in cases if c['row']['seed']==seed and c['row']['regime']==reg and c['row']['arm']=='risk_ard_mixture' and c['row']['allocation']=='n08_v1120');j=initial_case['recommendation'];true_reward,_=verification_utility(truth,np.zeros_like(truth),protocol)
 diagnostic['old_initial_choice_at_final_posterior']={'candidate':j,'predicted_reward':float(reward[j]),'predicted_pass':float(power[j]),'true_reward':float(true_reward[j]),'true_margin':float(truth[j,1]+1.6448536269514722*np.exp(.5*truth[j,2]))}
 diagnostic['recommended_predicted_reward']=float(reward[i]);output.append(diagnostic)
 print(json.dumps(diagnostic))
(root/'paper_artifacts/meta_prior_total_budget_v6_20261002/selection_failure_replay.json').write_text(json.dumps({'scope':'Offline reconstruction of already paid deterministic observations at fixed logged query paths; no KG, new search, validation or dispatch. Truth replacements are post-scoring attribution, not deployable controls.','cases':output},indent=2)+'\n')
