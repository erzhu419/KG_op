# Run from the KG_op repository root in the project Python environment.
# Small simulator/selection probes only; not a target benchmark rerun.
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,inspect,textwrap
sys.path.insert(0,'/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
import numpy as np
from core.profile_atlas import SourceScoredProfileAtlas,ProfileAtlasConfig
from problems.randomized_profiles import PROFILE_STRESS_REGIMES,RandomizedOrderedProfileProblem,generate_structural_profile_library,source_profile_records
library=generate_structural_profile_library(64,dimension=128,seed=20261799,maximum_frequency=40)
res={'scope':'read-only selection and physical-boundary diagnostics, no new target benchmark matrix','selection':[]}
for regime in PROFILE_STRESS_REGIMES:
 stable=0
 for c in regime.encode(): stable=(stable*257+c)%(2**31-1)
 sources=[RandomizedOrderedProfileProblem(regime=regime,role='source',task_seed=10000+i,family_seed=20260808+stable,d=256) for i in range(2)]
 records=source_profile_records(sources,library,replications=3,seed=20260808+1237)
 a=SourceScoredProfileAtlas(ProfileAtlasConfig(n0=10)).fit(records).selection
 b=SourceScoredProfileAtlas(ProfileAtlasConfig(n0=10,safety_metric_weight=0,objective_metric_weight=0)).fit(records).selection
 ia=[x.profile_id for x in a.members];ib=[x.profile_id for x in b.members]
 row={'regime':regime,'same_ordered_design':ia==ib,'overlap':len(set(ia)&set(ib)),'full':ia,'source_best_structural_only':ib}
 res['selection'].append(row)
 print(regime,'same_ordered',ia==ib,'overlap',row['overlap'],flush=True)
from problems.energy_forecast_policy import OPSDForecastIndexedStorageProblem
p=OPSDForecastIndexedStorageProblem('SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz',market='SE_1')
method=inspect.getsource(OPSDForecastIndexedStorageProblem._evaluate_start_batch)
method=textwrap.dedent(method).replace('soc = np.clip(first_target * capacity, 0.0, capacity)','soc = np.full(count, 0.5 * capacity, dtype=float)')
ns={'np':np};exec(method,ns);fixed=ns['_evaluate_start_batch']
starts=p._starts['verification'][np.linspace(0,len(p._starts['verification'])-1,24,dtype=int)]
old=[];new=[]
for prof in library:
 x=p.continuous_to_int(np.interp(p.nodes,prof.nodes,prof.values))
 old.append(p._evaluate_start_batch(x,starts));new.append(fixed(p,x,starts))
old=np.asarray(old);new=np.asarray(new);om=old[:,:,0].mean(1);nm=new[:,:,0].mean(1)
res['energy']={'market':'SE_1','profile_count':64,'verification_windows_per_profile':24,'max_constraint_change':float(abs(new[:,:,1]-old[:,:,1]).max()),'objective_change_min':float((new[:,:,0]-old[:,:,0]).min()),'objective_change_max':float((new[:,:,0]-old[:,:,0]).max()),'mean_objective_rank_reversals':int(sum((om[i]-om[j])*(nm[i]-nm[j])<0 for i in range(64) for j in range(i))),'meaning':'50 percent fixed initial SOC; same target physics and window starts; this is not a rerun of Energy certification comparisons'}
print(res['energy'],flush=True)
from problems.rzdt import InventorySupplyChainProblem,QueueResourceControlProblem
res['adapter']=[]
for cls in (InventorySupplyChainProblem,QueueResourceControlProblem):
 p=cls(d=6,L=100)
 x=(50,50,50,50,30,30);y=(50,50,50,50,70,70)
 res['adapter'].append({'problem':cls.__name__,'summary_x':p._policy_summary(x),'summary_y':p._policy_summary(y),'max_feature_difference':float(abs(p.gpr_basis_map().features(x)-p.gpr_basis_map().features(y)).max()),'objective_x':list(p.true_objectives(x)),'objective_y':list(p.true_objectives(y))})
print(res['adapter'],flush=True)
with open('SC-OLH-KG/docs/submission_review_20260929/diagnostic_rerun.json','w') as f: json.dump(res,f,indent=2)
