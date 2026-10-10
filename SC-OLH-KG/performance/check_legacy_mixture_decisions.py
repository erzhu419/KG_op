"""Resolve numerical ranking and drift in the frozen paid-evidence mixture."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.guarded_deployment_kg import _root
from core.mixture_guarded_kg import mixture_raw_lookahead
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix


def check():
    spec=json.loads((ROOT/"performance/manifests/legacy_mixture_precision_v1.json").read_text())
    replay=json.loads((ROOT/spec["replay"]).read_text()); protocol=replay["protocol"]
    source_spec=replay["source_fit_contract"]
    old=json.loads((ROOT/protocol["paid_paths"]).read_text()); base=old["protocol"]
    started=time.process_time(); rows=[]; truth_jobs=[]; diagnostic_jobs=[]
    for domain in dict.fromkeys(c["domain"] for c in spec["cases"]):
        problem,initial,candidates,model,_=build_models(domain,source_spec,base,protocol["hypotheses"])
        paid=next(r for r in old["rows"] if r["domain"]==domain and r["arm"]=="source_orthogonal" and r["search_calls"]==13)
        for prefix,(index,value) in enumerate(zip(paid["paid_candidate_indices"],paid["paid_observations"]),1):
            model.update_raw(index,[value[0],value[1]-problem.tau])
            if {"domain":domain,"prefix":prefix} not in spec["cases"]: continue
            terminal,_=terminal_at_prefix(problem,initial,paid,prefix,source_spec,len(candidates))
            remaining=protocol["total_target_call_cap"]-prefix
            original=next(r for r in replay["rows"] if r["domain"]==domain and r["prefix"]==prefix)
            row={"domain":domain,"prefix":prefix,"model_weights":model.weights.tolist(),
                "weight_replay_max_error":float(np.max(np.abs(model.weights-original["model_weights"]))),
                "head_log_score_sums":np.sum([h["head_log_scores"] for h in model.history],axis=0).tolist(),
                "integrations":[],"candidate_diagnostics":{}}
            # Once per state, sample each component's two-dimensional g/eta marginal.
            z=ndtri(qmc.Sobol(2,scramble=True,seed=spec["seeds"][0]).random_base2(spec["draw_power"]))
            gmom=[g.predict(full_cov=True) for _,g in model.components]
            integration_values=[]
            for seed in spec["seeds"]:
                rewards,costs,actions=model.terminal_components(terminal,remaining,spec["draw_power"],seed)
                views={"observable_source_full":(rewards[1],costs[1]),
                    "fixed_equal_mixture":(rewards.mean(axis=0),costs.mean(axis=0)),
                    "paid_evidence_mixture":(model.weights@rewards,model.weights@costs)}
                integration_values.append({name:values[0] for name,values in views.items()})
                integration={"seed":seed,"decisions":{}}
                for view in spec["views"]:
                    r,c=views[view]; best=int(np.lexsort((c,-r))[0]); primary=actions[best]["primary"]
                    other=[i for i,a in enumerate(actions) if a["primary"]!=primary]
                    decision={"action":actions[best],"action_index":best,"predicted_reward":float(r[best]),
                        "predicted_verification_calls":float(c[best]),
                        "reward_gap_to_other_primary":float(r[best]-r[other].max())}
                    integration["decisions"][view]=decision
                    truth_jobs.append((decision,terminal,remaining,best,problem,candidates))
                    if str(primary) not in row["candidate_diagnostics"]:
                        components=[]
                        for mean,cov in gmom:
                            local=mean[primary]+z@_root(cov[primary,:,primary,:]).T
                            feasible=float(np.mean(local[:,0]+ndtri(1-problem.alpha)*np.exp(.5*local[:,1])<=0))
                            components.append({"mean_g":float(mean[primary,0]),"mean_eta":float(mean[primary,1]),
                                "sd_g":float(np.sqrt(cov[primary,0,primary,0])),
                                "sd_eta":float(np.sqrt(cov[primary,1,primary,1])),"probability_feasible":feasible})
                        row["candidate_diagnostics"][str(primary)]={"components":components,
                            "mixture_probability_feasible":float(model.weights@[c["probability_feasible"] for c in components]),
                            "observed":primary in paid["paid_candidate_indices"][:prefix]}
                row["integrations"].append(integration)
            row["max_reward_difference"]={view:float(np.max(np.abs(integration_values[0][view]-integration_values[-1][view])))
                for view in spec["views"]}
            probe=spec["lookahead_probe"]
            if domain==probe["domain"] and prefix==probe["prefix"]:
                print(domain,prefix,"increased-fantasy probe starts",flush=True)
                row["lookahead"]=mixture_raw_lookahead(model,terminal,remaining,probe["candidates"],
                    draw_power=probe["draw_power"],fantasy_power=probe["fantasy_power"],seed=probe["seed"])
                print(domain,prefix,"increased-fantasy probe complete",flush=True)
            rows.append(row); diagnostic_jobs.append((row,problem,candidates))
            print(domain,prefix,"precision decisions frozen",flush=True)
    for decision,terminal,remaining,best,problem,candidates in truth_jobs:
        truth=np.array([[problem.true_objective(p),np.log(problem.true_sigma(p)[0]**2),
            problem.true_constraint_mean(p)-problem.tau,np.log(problem.true_sigma(p)[1]**2)] for p in candidates])[None]
        reward,cost,_=terminal.evaluate(truth,remaining)
        decision["truth"]={"reward":float(reward[best]),"total_calls":float(cost[best]+protocol["total_target_call_cap"]-remaining),
            "primary_feasible":bool(problem.is_truly_feasible(candidates[decision["action"]["primary"]]))}
    for row,problem,candidates in diagnostic_jobs:
        for primary,diagnostic in row["candidate_diagnostics"].items():
            point=candidates[int(primary)]; sigma=problem.true_sigma(point)[1]
            mean=problem.true_constraint_mean(point)-problem.tau
            diagnostic["truth"]={"mean_g":float(mean),"mean_eta":float(np.log(sigma**2)),
                "chance_margin":float(mean+ndtri(1-problem.alpha)*sigma)}
    result={"protocol":spec,"rows":rows,"new_calls":0,"cpu_seconds":time.process_time()-started}
    (ROOT/"paper_artifacts/legacy_mixture_budget_v1"/spec["output"]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"states":len(rows),"new_calls":0,"cpu_seconds":result["cpu_seconds"]}),flush=True)


if __name__=="__main__":
    check()
