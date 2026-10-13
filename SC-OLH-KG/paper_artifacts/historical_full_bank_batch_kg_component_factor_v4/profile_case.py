import cProfile,json,pstats,time
import numpy as np
from scipy.special import digamma
from performance.historical_full_bank_batch_kg import restore
from performance.complete_historical_source_heads import read,save,ROOT
from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_raw_terminal import raw_terminal_values
spec=read("performance/manifests/historical_full_bank_batch_kg_component_factor_v4.json")
model,terminal,state,_=restore(spec,"learned_prior_independent_02_source_common_task_rbf_layout6")
import copy
future=copy.deepcopy(model);b=next(r for r in read(spec["observations"])["rows"] if r["task_id"]==state["task_id"] and r["candidate"]==63)
v=np.asarray(b["paired_observations"]);v[:,1]-=b["tau"];scores=future.update_replicates(63,v)["head_log_scores"];eta=np.log(v.var(axis=0,ddof=1))-digamma(1.5)+np.log(1.5)
small=ConditionalTerminalActions([[28,45,10]],[1],terminal.protocol);profile=cProfile.Profile();profile.enable();tick=time.process_time()
raw_terminal_values(model,future,small,356,63,v.mean(axis=0),scores,power=12,seed=spec["seed"],replicates=4,eta_labels=eta,guarded_mean_position=0)
profile.disable();elapsed=time.process_time()-tick
entries=[{"function":k[2],"file":k[0],"calls":r[1],"self_seconds":r[2],"cumulative_seconds":r[3]} for k,r in pstats.Stats(profile).stats.items()]
result={"cpu_seconds":elapsed,"action_count":5,"entries":sorted(entries,key=lambda x:x["cumulative_seconds"],reverse=True)[:20],"new_physical_calls":0}
save(ROOT/spec["output"]/"guarded_profile.json",result);print("SCHEDULEURM_DONE "+json.dumps(result))
