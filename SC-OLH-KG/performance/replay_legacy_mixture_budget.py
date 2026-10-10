"""Withdraw source hypotheses using paid joint predictive evidence."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive,resample_normalized_profiles
from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.designs import load_frozen_source_informed_design
from core.mixture_guarded_kg import mixture_raw_lookahead
from performance.benchmark_frozen_proposal_only import _select_shortlist
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.replay_guarded_deployment import shortlist_indices
from performance.replay_legacy_observable_budget import fit_arms
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_function_prior import legacy_public_coordinates,legacy_source_labels
from representation.legacy_observable_prior import legacy_exposure_descriptors

OUT=ROOT/"paper_artifacts/legacy_mixture_budget_v1"


def build_models(domain,source_spec,base,names):
    inputs=ROOT/"paper_artifacts/legacy_joint_bridge_v1/inputs"/domain
    archive=FrozenTransferArchive.load(inputs/"archive.json")
    initial,_=load_frozen_source_informed_design(inputs/"design.json",heldout=domain,
        seed=base["target_seed"],n0=10,dimension=base["target_dimension"])
    problem=build_scalarized_problem(domain,base["target_dimension"],base["L"],base["sigma"],base["alpha"],base["weights"])
    lifted=resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]),problem.d)
    candidates=list(dict.fromkeys(list(initial)+[tuple(map(int,problem.continuous_to_int(x))) for x in lifted]))
    desc=[]
    for task in archive.tasks:
        provider=build_scalarized_problem(task.name,task.X.shape[1],base["L"],base["sigma"],base["alpha"],base["weights"])
        desc.extend(legacy_exposure_descriptors(provider,task.X_integer,source_spec["dynamics_grid_dimension"]))
    coordinate,arms=fit_arms(legacy_source_labels(archive),np.asarray(desc),source_spec)
    observable_psi=coordinate.transform(legacy_exposure_descriptors(problem,candidates,source_spec["dynamics_grid_dimension"]))
    legacy_psi=legacy_public_coordinates(np.asarray(candidates)/problem.L)
    components=[]
    for name in names:
        psi=legacy_psi if name=="legacy_source_orthogonal" else observable_psi
        ps=arms[name]
        components.append((JointFunctionRiskPosterior([ps[0]],[ps[3]],psi),
                           JointFunctionRiskPosterior([ps[1]],[ps[2]],psi)))
    sources={"coordinate":coordinate.diagnostics(),
        "fit_diagnostics":{a:[p.diagnostics for p in ps] for a,ps in arms.items()}}
    return problem,initial,candidates,JointResponseMixture(names,components),sources


def terminal_at_prefix(problem,initial,paid,prefix,source_spec,candidate_count):
    n0=min(prefix,10); obs=paid["paid_observations"][:n0]
    original,_=_select_shortlist(problem,initial[:n0],obs)
    positions=[initial[:n0].index(tuple(r["point"])) for r in original]
    table,inc=shortlist_indices(positions,list(range(n0)),obs,candidate_count,range(10))
    terminal=BudgetedTerminalActions(table,inc,{"objective_reference":float(np.dot(problem.weights,problem.ref_point)),
        "familywise_delta":source_spec["familywise_delta"],"chance_alpha":problem.alpha,
        "comparison_budget":source_spec["comparison_budget"],"comparison_delta":source_spec["comparison_delta"]})
    return terminal,positions


def run():
    spec=json.loads((ROOT/"performance/manifests/legacy_mixture_budget_v1.json").read_text())
    source_spec=json.loads((ROOT/spec["source_fit_contract"]).read_text())
    old=json.loads((ROOT/spec["paid_paths"]).read_text()); base=old["protocol"]
    started=time.process_time(); rows=[]; sources={}; histories={}; truth_jobs=[]
    OUT.mkdir(parents=True,exist_ok=True)
    for domain in source_spec["domains"]:
        problem,initial,candidates,model,sources[domain]=build_models(domain,source_spec,base,spec["hypotheses"])
        prior_mean,prior_cov=model.predict_joint()
        paid=next(r for r in old["rows"] if r["domain"]==domain and r["arm"]=="source_orthogonal" and r["search_calls"]==13)
        for prefix,(index,value) in enumerate(zip(paid["paid_candidate_indices"],paid["paid_observations"]),1):
            model.update_raw(index,[value[0],value[1]-problem.tau])
            if prefix not in spec["prefixes"]: continue
            terminal,positions=terminal_at_prefix(problem,initial,paid,prefix,source_spec,len(candidates))
            remaining=spec["total_target_call_cap"]-prefix
            r,c,actions=model.terminal_components(terminal,remaining,spec["terminal_draw_power"],spec["integration_seed"])
            views={name:(r[i],c[i]) for i,name in enumerate(spec["hypotheses"])}
            views["fixed_equal_mixture"]=(np.mean(r,axis=0),np.mean(c,axis=0))
            views["paid_evidence_mixture"]=(model.weights@r,model.weights@c)
            record={"domain":domain,"prefix":prefix,"model_weights":model.weights.tolist(),"decisions":{}}
            for name,(reward,cost) in views.items():
                best=int(np.lexsort((cost,-reward))[0])
                record["decisions"][name]={"action":actions[best],"predicted_reward":float(reward[best]),
                    "predicted_verification_calls":float(cost[best]),"action_index":best}
            mean,C=model.predict_joint(); support=positions[1]
            record["risk_support"]={"candidate":support,"joint_mean":mean[support].tolist(),
                "prior_eta_mean":float(prior_mean[support,3]),
                "eta_variance_reduction":float(1-C[support,3,support,3]/prior_cov[support,3,support,3]),
                "f_g_posterior_covariance":float(C[support,0,support,2])}
            probe=spec["lookahead_probe"]
            if domain==probe["domain"] and prefix==probe["prefix"]:
                primary=record["decisions"]["paid_evidence_mixture"]["action"]["primary"]
                points=list(dict.fromkeys([support,positions[0],primary]))
                print("joint mixture raw lookahead starts",points,flush=True)
                record["lookahead"]=mixture_raw_lookahead(model,terminal,remaining,points,
                    draw_power=probe["draw_power"],fantasy_power=probe["fantasy_power"],seed=spec["integration_seed"])
                record["lookahead"]["point_observed"]={str(i):i in paid["paid_candidate_indices"][:prefix] for i in points}
                print("joint mixture raw lookahead complete",flush=True)
            rows.append(record); truth_jobs.append((record,problem,candidates,terminal,remaining))
            print(domain,prefix,"mixture weights and decisions frozen",flush=True)
        histories[domain]=model.history
    # Every recommendation and fantasy has frozen before truth is consulted.
    for record,problem,candidates,terminal,remaining in truth_jobs:
        truth=np.array([[problem.true_objective(p),np.log(problem.true_sigma(p)[0]**2),
            problem.true_constraint_mean(p)-problem.tau,np.log(problem.true_sigma(p)[1]**2)] for p in candidates])[None]
        r,c,_=terminal.evaluate(truth,remaining)
        for decision in record["decisions"].values():
            best=decision["action_index"]; primary=decision["action"]["primary"]
            decision["truth"]={"reward":float(r[best]),"verification_calls":float(c[best]),
                "total_calls":float(c[best]+record["prefix"]),
                "primary_feasible":bool(problem.is_truly_feasible(candidates[primary]))}
        point=candidates[record["risk_support"]["candidate"]]
        record["risk_support"]["truth_g_eta"]=[float(problem.true_constraint_mean(point)-problem.tau),
                                               float(np.log(problem.true_sigma(point)[1]**2))]
    result={"protocol":spec,"source_fit_contract":source_spec,"rows":rows,"sources":sources,
        "paid_history":histories,"new_calls":0,"cpu_seconds":time.process_time()-started}
    (OUT/"replay.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"states":len(rows),"decisions":sum(len(r["decisions"]) for r in rows),
        "new_calls":0,"cpu_seconds":result["cpu_seconds"]}),flush=True)


if __name__=="__main__":
    run()
