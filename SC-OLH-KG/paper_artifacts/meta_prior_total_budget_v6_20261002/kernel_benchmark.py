import argparse,importlib.util,json,resource,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
from performance.run_meta_prior_mechanism import observe,task_truth
from performance.run_meta_prior_total_budget import fit_priors,simulator_supports
p=json.load(open('/home/erzhu419/mine_code/KG_op/SC-OLH-KG/performance/meta_prior_candidate384_protocol.json'))
a=argparse.ArgumentParser();a.add_argument('--old',action='store_true');a.add_argument('--bank',type=int,default=384);a.add_argument('--output',required=True);args=a.parse_args()
psi=np.random.default_rng(51040).uniform(-2,2,(args.bank,8));priors,_=fit_priors(40,'shared_structure','fixed_source',psi[:96],p)
models=[prior.target_model(psi) for prior in priors]
rng=np.random.default_rng(51040);rng.uniform(-2,2,(96,8));initial=rng.choice(96,8,replace=False)
truth=task_truth(psi,simulator_supports(40,'shared_structure',8),65000);truth[:,1]+=1
for i in initial:
 y,n=observe(truth[i],65000,int(i),0,12)
 for h,m in enumerate(models):m.update([int(i)],y[h],n[h])
if args.old:
 spec=importlib.util.spec_from_file_location('old_verification','/home/erzhu419/mine_code/KG_op_meta_prior_coverage_deploy/SC-OLH-KG/core/verification_kg.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
else:
 import core.verification_kg as mod
Path('/proc/self/clear_refs').write_text('5\n');start=time.perf_counter();scores=mod.verification_kg_scores(models,np.arange(args.bank)[:,None],p);elapsed=time.perf_counter()-start
result={'variant':'frozen_v5' if args.old else 'blocked_v6','bank':args.bank,'scores':scores.tolist(),'choice':int(np.argmax(scores)),'seconds':elapsed,'peak_rss_mb':int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))/1024}
Path(args.output).write_text(json.dumps(result)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='scores'}))
