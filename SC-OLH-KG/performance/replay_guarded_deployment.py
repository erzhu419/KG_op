"""Align deployment reward and replay early prefixes without new observations."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive, resample_normalized_profiles
from core.designs import load_frozen_source_informed_design
from core.guarded_deployment_kg import deployment_values, guarded_raw_kg, posterior_draws, _root
from core.legacy_joint_kg import legacy_joint_utility
from performance.benchmark_frozen_proposal_only import _select_shortlist
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.run_legacy_joint_bridge import function_moments, update_objective
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from representation.orthogonal_function_prior import (
    OrthogonalLowOrderBasis, fit_orthogonal_function_prior, same_source_orthogonal_full_prior)
from representation.source_function_prior import head_batches

OUT = ROOT / "paper_artifacts/guarded_deployment_replay_v1"


def shortlist_indices(initial_rows, initial_indices, observations, candidate_count, frozen_ten):
    """Freeze supports from the paid prefix; complete from label-free proposals."""
    incumbent, safe = initial_indices[initial_rows[0]], initial_indices[initial_rows[1]]
    fillers = [initial_indices[int(i)] for i in np.argsort(np.asarray(observations)[:, 1], kind="stable")]
    fillers += list(frozen_ten)
    rows = [list(dict.fromkeys([candidate, incumbent, safe] + fillers))[:3]
            for candidate in range(candidate_count)]
    return np.asarray(rows), np.array([row.index(incumbent) for row in rows])


def replay():
    spec = json.loads((ROOT / "performance/manifests/guarded_deployment_replay_v1.json").read_text())
    old = json.loads((ROOT / spec["input"]).read_text())
    base = old["protocol"]
    OUT.mkdir(parents=True, exist_ok=True)
    started, rows, sources = time.process_time(), [], {}
    for domain in base["domains"]:
        inputs = ROOT / "paper_artifacts/legacy_joint_bridge_v1/inputs" / domain
        archive = FrozenTransferArchive.load(inputs / "archive.json")
        initial, _ = load_frozen_source_informed_design(inputs / "design.json", heldout=domain,
            seed=base["target_seed"], n0=10, dimension=base["target_dimension"])
        problem = build_scalarized_problem(domain, base["target_dimension"], base["L"],
                                           base["sigma"], base["alpha"], base["weights"])
        lifted = resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]), problem.d)
        candidates = list(dict.fromkeys(list(initial) + [tuple(map(int, problem.continuous_to_int(x))) for x in lifted]))
        psi = legacy_public_coordinates(np.asarray(candidates) / problem.L)
        tasks = legacy_source_labels(archive)
        basis = OrthogonalLowOrderBasis(np.vstack([t["psi"] for t in tasks]))
        priors = [fit_orthogonal_function_prior(head_batches(tasks, h), [t["noise"][:, h] for t in tasks],
            basis, max_iterations=spec["source_max_iterations"]) for h in range(4)]
        sources[domain] = [p.diagnostics for p in priors]
        protocol = {"objective_reference": float(np.dot(problem.weights, problem.ref_point)),
            "candidate_budgets": spec["terminal_candidate_budgets"], "chance_alpha": problem.alpha,
            "familywise_delta": spec["terminal_familywise_delta"],
            "comparison_budget": spec["terminal_objective_comparison_budget"],
            "comparison_delta": spec["terminal_objective_comparison_delta"]}
        for arm in spec["arms"]:
            fitted = [same_source_orthogonal_full_prior(p) for p in priors] if arm == "same_source_full" else priors
            paid = next(r for r in old["rows"] if r["domain"] == domain and r["arm"] == arm and r["search_calls"] == 13)
            f_model = JointFunctionRiskPosterior([fitted[0]], [fitted[3]], psi)
            g_model = JointFunctionRiskPosterior([fitted[1]], [fitted[2]], psi)
            before, before_cov = g_model.predict(full_cov=True)
            old_f, old_C = function_moments(fitted[0], psi)
            f_noise = float(np.mean(np.concatenate([t.replicate_variance[:, 0] for t in archive.tasks])))
            for calls, (index, value) in enumerate(zip(paid["paid_candidate_indices"], paid["paid_observations"]), start=1):
                f_model.update_batch(index, value[0], None, 1)
                g_model.update_batch(index, value[1] - problem.tau, None, 1)
                update_objective(old_f, old_C, index, value[0], f_noise)
                if calls not in spec["prefixes"]:
                    continue
                n0 = min(calls, 10)
                observed = paid["paid_observations"][:n0]
                prefix_points = initial[:n0]
                old_shortlist, _ = _select_shortlist(problem, prefix_points, observed)
                support_positions = [prefix_points.index(tuple(r["point"])) for r in old_shortlist]
                shortlists, incumbents = shortlist_indices(support_positions, list(range(n0)), observed, len(candidates), range(10))
                draws = posterior_draws(f_model,g_model,spec["recommendation_draw_power"],spec["integration_seed"])
                values, costs = deployment_values(draws,shortlists,incumbents,protocol)
                ref_draws = posterior_draws(f_model,g_model,spec["numerical_reference_draw_power"],spec["integration_seed"]+1)
                ref_values, ref_costs = deployment_values(ref_draws,shortlists,incumbents,protocol)
                recommendation = int(values.argmax())
                mean, cov = g_model.predict(full_cov=True)
                point_cov = cov[np.arange(len(candidates)),:,np.arange(len(candidates)),:]
                proxy = legacy_joint_utility(old_f,np.diag(old_C),mean,point_cov,
                    {"objective_reference":protocol["objective_reference"],"verification_replicates":80,
                     "chance_alpha":problem.alpha,"candidate_delta":protocol["familywise_delta"]/3})[0]
                # Isolate the terminal rule change while holding the previous
                # fixed-noise objective posterior and current g posterior.
                u = ndtri(qmc.Sobol(len(psi),scramble=True,
                    seed=spec["integration_seed"]+2).random_base2(spec["recommendation_draw_power"]))
                fixed_noise_draws = draws.copy()
                fixed_noise_draws[:,:,0] = old_f + u @ _root(old_C).T
                fixed_noise_draws[:,:,1] = np.log(f_noise)
                terminal_only, _ = deployment_values(fixed_noise_draws,shortlists,incumbents,protocol)
                selected = {"old_primary_proxy":int(proxy.argmax()), "terminal_rule_only":int(terminal_only.argmax()),
                            "full_deployment":recommendation, "reference_integration":int(ref_values.argmax())}
                row = {"domain":domain,"arm":arm,"prefix":calls,"selected":selected,
                    "predicted_deployment_reward":float(values[recommendation]),
                    "predicted_verification_calls":float(costs[recommendation]),
                    "numerical_max_reward_difference":float(np.max(np.abs(values-ref_values))),
                    "reference_reward_loss_of_selected":float(ref_values.max()-ref_values[recommendation]),
                    "old_empirical_incumbent":support_positions[0],"shortlist":shortlists[recommendation].tolist(),
                    "all_values":values.tolist(),"all_expected_calls":costs.tolist(),"risk_points":[]}
                for role, i in [("new_primary",recommendation),("initial_primary",support_positions[0]),("initial_safe",support_positions[1])]:
                    C = point_cov[i]
                    conditional = C[0,0]-C[0,1]**2/C[1,1]
                    loading = C[0,1]/C[1,1]
                    environment = np.exp(mean[i,1])
                    fisher = loading**2/(conditional+environment)+.5*(environment/(conditional+environment))**2
                    row["risk_points"].append({"role":role,"candidate":i,"g_eta":mean[i].tolist(),
                        "eta_prior_mean":float(before[i,1]),"eta_mean_update":float(mean[i,1]-before[i,1]),
                        "eta_variance_reduction_fraction":float(1-C[1,1]/before_cov[i,1,i,1]),
                        "eta_conditional_likelihood_information_at_mean":float(fisher)})
                probe = spec["kg_probe"]
                if domain == probe["domain"] and arm == probe["arm"] and calls == probe["prefix"]:
                    print("KG probe starts",flush=True)
                    kg=guarded_raw_kg(f_model,g_model,shortlists,incumbents,protocol,
                        draw_power=probe["draw_power"],fantasy_power=probe["fantasy_power"],seed=spec["integration_seed"])
                    row["kg_probe"]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in kg.items()}
                    print("KG probe complete",flush=True)
                rows.append(row)
            print(domain,arm,"prefixes replayed",flush=True)
    # Truth is scored only after every recommendation/fantasy has been frozen.
    for domain in base["domains"]:
        inputs=ROOT/"paper_artifacts/legacy_joint_bridge_v1/inputs"/domain
        archive=FrozenTransferArchive.load(inputs/"archive.json")
        initial,_=load_frozen_source_informed_design(inputs/"design.json",heldout=domain,seed=80,n0=10,dimension=1000)
        problem=build_scalarized_problem(domain,1000,base["L"],base["sigma"],base["alpha"],base["weights"])
        lifted=resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]),1000)
        candidates=list(dict.fromkeys(list(initial)+[tuple(map(int,problem.continuous_to_int(x))) for x in lifted]))
        truth_draws=np.array([[problem.true_objective(p),np.log(problem.true_sigma(p)[0]**2),
            problem.true_constraint_mean(p)-problem.tau,np.log(problem.true_sigma(p)[1]**2)] for p in candidates])[None]
        protocol={"objective_reference":float(np.dot(problem.weights,problem.ref_point)),
            "candidate_budgets":spec["terminal_candidate_budgets"],"chance_alpha":problem.alpha,
            "familywise_delta":spec["terminal_familywise_delta"],"comparison_budget":spec["terminal_objective_comparison_budget"],
            "comparison_delta":spec["terminal_objective_comparison_delta"]}
        for row in [r for r in rows if r["domain"]==domain]:
            paid=next(r for r in old["rows"] if r["domain"]==domain and r["arm"]==row["arm"] and r["search_calls"]==13)
            n0=min(row["prefix"],10); observed=paid["paid_observations"][:n0]
            original,_=_select_shortlist(problem,initial[:n0],observed)
            positions=[initial[:n0].index(tuple(r["point"])) for r in original]
            table,inc=shortlist_indices(positions,list(range(n0)),observed,len(candidates),range(10))
            row["selected_truth"]={}
            for label,i in row["selected"].items():
                reward,cost=deployment_values(truth_draws,table[i:i+1],inc[i:i+1],protocol)
                row["selected_truth"][label]={"objective":float(problem.true_objective(candidates[i])),
                    "true_feasible":bool(problem.is_truly_feasible(candidates[i])),
                    "expected_deployment_reward":float(reward[0]),"expected_verification_calls":float(cost[0])}
    result={"protocol":spec,"source_diagnostics":sources,"rows":rows,"new_calls":0,"cpu_seconds":time.process_time()-started}
    (OUT/"replay.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"rows":len(rows),"new_calls":0,"cpu_seconds":result["cpu_seconds"]}),flush=True)
    return result


if __name__ == "__main__":
    replay()
