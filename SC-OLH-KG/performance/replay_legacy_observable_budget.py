"""Frozen-source observable representation and shared-budget attribution."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive, resample_normalized_profiles
from core.budgeted_guarded_kg import BudgetedTerminalActions, budgeted_raw_lookahead
from core.designs import load_frozen_source_informed_design
from core.guarded_deployment_kg import posterior_draws
from performance.benchmark_frozen_proposal_only import _select_shortlist
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.replay_guarded_deployment import shortlist_indices
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from representation.legacy_observable_prior import LegacyObservableCoordinate, legacy_exposure_descriptors
from representation.orthogonal_function_prior import (
    OrthogonalLowOrderBasis, fit_orthogonal_function_prior, same_source_orthogonal_full_prior)
from representation.source_function_prior import head_batches

OUT=ROOT/"paper_artifacts/legacy_observable_budget_v1"


def fit_arms(tasks,descriptors,spec):
    coordinate=LegacyObservableCoordinate().fit(descriptors,tasks)
    observable=coordinate.transform(descriptors)
    adapted=[]; offset=0
    for task in tasks:
        n=len(task["response"])
        adapted.append(dict(task,psi=observable[offset:offset+n])); offset+=n
    arms={}
    for name,data in [("legacy_source_orthogonal",tasks),("observable_source_orthogonal",adapted)]:
        basis=OrthogonalLowOrderBasis(np.vstack([t["psi"] for t in data]))
        priors=[fit_orthogonal_function_prior(head_batches(data,h),[t["noise"][:,h] for t in data],
            basis,max_iterations=spec["source_max_iterations"]) for h in range(4)]
        arms[name]=priors
    arms["observable_source_full"]=[same_source_orthogonal_full_prior(p) for p in arms["observable_source_orthogonal"]]
    return coordinate,arms


def source_validation(tasks,descriptors,spec):
    reduced,training_inputs=[],[]; offset=0
    for t in tasks:
        n=len(t["response"]); train=np.arange(n)%4!=3
        reduced.append({**t,"psi":t["psi"][train],"response":t["response"][train],"noise":t["noise"][train]})
        training_inputs.extend(descriptors[offset:offset+n][train]); offset+=n
    coordinate,arms=fit_arms(reduced,np.asarray(training_inputs),spec)
    transformed=coordinate.transform(descriptors)
    records=[]
    for arm,priors in arms.items():
        offset=0
        for t in tasks:
            n=len(t["response"]); train=np.arange(n)%4!=3; test=np.flatnonzero(~train)
            psi=t["psi"] if arm=="legacy_source_orthogonal" else transformed[offset:offset+n]
            offset+=n
            f=JointFunctionRiskPosterior([priors[0]],[priors[3]],psi)
            g=JointFunctionRiskPosterior([priors[1]],[priors[2]],psi)
            for i in np.flatnonzero(train):
                y=t["response"][i]
                f.update_batch(i,y[0],y[3],3); g.update_batch(i,y[1],y[2],3)
            fm,_=f.predict(); gm,_=g.predict()
            score_f=[joint_batch_log_score(f.local_components(i),t["response"][i,0],t["response"][i,3],3) for i in test]
            score_g=[joint_batch_log_score(g.local_components(i),t["response"][i,1],t["response"][i,2],3) for i in test]
            predicted=np.column_stack([fm[:,0],gm[:,0],gm[:,1],fm[:,1]])
            records.append({"arm":arm,"source_domain":t["name"],"training_sites":int(train.sum()),
                "heldout_sites":len(test),"joint_log_score_f":float(np.mean(score_f)),
                "joint_log_score_g":float(np.mean(score_g)),
                "head_mse":np.mean((predicted[test]-t["response"][test])**2,axis=0).tolist()})
    return {"records":records,"fit_diagnostics":{a:[p.diagnostics for p in ps] for a,ps in arms.items()},
            "coordinate":coordinate.diagnostics()}


def run():
    spec=json.loads((ROOT/"performance/manifests/legacy_observable_budget_v1.json").read_text())
    old=json.loads((ROOT/spec["paid_paths"]).read_text()); base=old["protocol"]
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.process_time(); rows=[]; source={}; truth_jobs=[]
    for domain in spec["domains"]:
        inputs=ROOT/"paper_artifacts/legacy_joint_bridge_v1/inputs"/domain
        archive=FrozenTransferArchive.load(inputs/"archive.json")
        initial,_=load_frozen_source_informed_design(inputs/"design.json",heldout=domain,
            seed=base["target_seed"],n0=10,dimension=base["target_dimension"])
        problem=build_scalarized_problem(domain,base["target_dimension"],base["L"],base["sigma"],base["alpha"],base["weights"])
        lifted=resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]),problem.d)
        candidates=list(dict.fromkeys(list(initial)+[tuple(map(int,problem.continuous_to_int(x))) for x in lifted]))
        tasks=legacy_source_labels(archive); descriptor_parts=[]
        for t in archive.tasks:
            provider=build_scalarized_problem(t.name,t.X.shape[1],base["L"],base["sigma"],base["alpha"],base["weights"])
            descriptor_parts.append(legacy_exposure_descriptors(provider,t.X_integer,spec["dynamics_grid_dimension"]))
        descriptors=np.vstack(descriptor_parts)
        validation=source_validation(tasks,descriptors,spec)
        print(domain,"source-site validation complete",flush=True)
        coordinate,arms=fit_arms(tasks,descriptors,spec)
        target_desc=legacy_exposure_descriptors(problem,candidates,spec["dynamics_grid_dimension"])
        target_psi=coordinate.transform(target_desc)
        legacy_psi=legacy_public_coordinates(np.asarray(candidates)/problem.L)
        source[domain]={"validation":validation,"coordinate":coordinate.diagnostics(),
            "fit_diagnostics":{a:[p.diagnostics for p in ps] for a,ps in arms.items()},
            "source_max_abs_latent":float(np.abs(coordinate.transform(descriptors)).max()),
            "target_max_abs_latent":float(np.abs(target_psi).max())}
        # All arms receive exactly this same opened legacy path, not an
        # arm-specific search outcome selected after observing its reward.
        paid=next(r for r in old["rows"] if r["domain"]==domain and r["arm"]=="source_orthogonal" and r["search_calls"]==13)
        for arm in spec["arms"]:
            priors=arms[arm]; psi=legacy_psi if arm=="legacy_source_orthogonal" else target_psi
            f=JointFunctionRiskPosterior([priors[0]],[priors[3]],psi)
            g=JointFunctionRiskPosterior([priors[1]],[priors[2]],psi)
            prior_mean,prior_cov=g.predict(full_cov=True)
            for prefix,(index,value) in enumerate(zip(paid["paid_candidate_indices"],paid["paid_observations"]),1):
                f.update_batch(index,value[0],None,1); g.update_batch(index,value[1]-problem.tau,None,1)
                if prefix not in spec["prefixes"]: continue
                n0=min(prefix,10); observed=paid["paid_observations"][:n0]
                original,_=_select_shortlist(problem,initial[:n0],observed)
                positions=[initial[:n0].index(tuple(r["point"])) for r in original]
                table,inc=shortlist_indices(positions,list(range(n0)),observed,len(candidates),range(10))
                protocol={"objective_reference":float(np.dot(problem.weights,problem.ref_point)),
                    "familywise_delta":spec["familywise_delta"],"chance_alpha":problem.alpha,
                    "comparison_budget":spec["comparison_budget"],"comparison_delta":spec["comparison_delta"]}
                terminal=BudgetedTerminalActions(table,inc,protocol); remaining=spec["total_target_call_cap"]-prefix
                draws=posterior_draws(f,g,spec["terminal_draw_power"],spec["integration_seed"])
                rewards,costs,actions=terminal.evaluate(draws,remaining)
                best=int(np.lexsort((costs,-rewards))[0]); mean,cov=g.predict(full_cov=True)
                support=positions[1]
                record={"domain":domain,"arm":arm,"prefix":prefix,"terminal_action":actions[best],
                    "predicted_reward":float(rewards[best]),"predicted_verification_calls":float(costs[best]),
                    "risk_support":{"candidate":support,"g_eta":mean[support].tolist(),
                        "eta_prior_mean":float(prior_mean[support,1]),
                        "eta_variance_reduction":float(1-cov[support,1,support,1]/prior_cov[support,1,support,1])}}
                probe=spec["lookahead_probe"]
                if domain==probe["domain"] and arm==probe["arm"] and prefix==probe["prefix"]:
                    search_points=list(dict.fromkeys([support,positions[0],actions[best]["primary"]]))
                    print("budgeted new/repeat lookahead starts",search_points,flush=True)
                    record["lookahead"]=budgeted_raw_lookahead(f,g,terminal,remaining,search_points,
                        draw_power=probe["draw_power"],fantasy_power=probe["fantasy_power"],seed=spec["integration_seed"])
                    record["lookahead"]["point_observed"]={str(i):i in paid["paid_candidate_indices"][:prefix] for i in search_points}
                    print("budgeted lookahead complete",flush=True)
                rows.append(record); truth_jobs.append((record,problem,candidates,terminal,remaining,best,protocol))
            print(domain,arm,"target prefixes complete",flush=True)
    for record,problem,candidates,terminal,remaining,best,protocol in truth_jobs:
        truth=np.array([[problem.true_objective(p),np.log(problem.true_sigma(p)[0]**2),
            problem.true_constraint_mean(p)-problem.tau,np.log(problem.true_sigma(p)[1]**2)] for p in candidates])[None]
        r,c,actions=terminal.evaluate(truth,remaining)
        chosen=record["terminal_action"]["primary"]
        record["truth"]={"expected_reward":float(r[best]),"expected_verification_calls":float(c[best]),
            "total_expected_calls":float(c[best]+record["prefix"]),"primary_feasible":bool(problem.is_truly_feasible(candidates[chosen])),
            "g_eta_support":[float(problem.true_constraint_mean(candidates[record["risk_support"]["candidate"]])-problem.tau),
                float(np.log(problem.true_sigma(candidates[record["risk_support"]["candidate"]])[1]**2))]}
    result={"protocol":spec,"source":source,"rows":rows,"new_calls":0,"cpu_seconds":time.process_time()-started}
    (OUT/"replay.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"rows":len(rows),"new_calls":0,"cpu_seconds":result["cpu_seconds"]}),flush=True)


if __name__=="__main__":
    run()
