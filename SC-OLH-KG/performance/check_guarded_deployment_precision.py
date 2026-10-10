"""Resolve terminal-integration sensitivity on four predeclared saved states.

No search or verification observations are generated. The incumbent-128
control uses the same three budgets in a different order, to distinguish
policy improvement from purchasing a stronger incumbent certificate.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive, resample_normalized_profiles
from core.designs import load_frozen_source_informed_design
from core.guarded_deployment_kg import deployment_values, posterior_draws
from performance.benchmark_frozen_proposal_only import _select_shortlist
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.replay_guarded_deployment import shortlist_indices
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.source_function_prior import head_batches


def check():
    spec = json.loads((ROOT / "performance/manifests/guarded_deployment_precision_v1.json").read_text())
    old = json.loads((ROOT / spec["paid_paths"]).read_text())
    replay = json.loads((ROOT / spec["replay"]).read_text())
    base = old["protocol"]
    started, results = time.process_time(), []
    for domain in dict.fromkeys(c["domain"] for c in spec["cases"]):
        inputs = ROOT / "paper_artifacts/legacy_joint_bridge_v1/inputs" / domain
        archive = FrozenTransferArchive.load(inputs / "archive.json")
        initial, _ = load_frozen_source_informed_design(inputs / "design.json", heldout=domain,
            seed=base["target_seed"], n0=10, dimension=base["target_dimension"])
        problem = build_scalarized_problem(domain, base["target_dimension"], base["L"],
                                           base["sigma"], base["alpha"], base["weights"])
        lifted = resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]), problem.d)
        candidates = list(dict.fromkeys(list(initial) +
            [tuple(map(int, problem.continuous_to_int(x))) for x in lifted]))
        psi = legacy_public_coordinates(np.asarray(candidates) / problem.L)
        tasks = legacy_source_labels(archive)
        basis = OrthogonalLowOrderBasis(np.vstack([t["psi"] for t in tasks]))
        priors = [fit_orthogonal_function_prior(head_batches(tasks, h), [t["noise"][:, h] for t in tasks],
            basis, max_iterations=replay["protocol"]["source_max_iterations"]) for h in range(4)]
        paid = next(r for r in old["rows"] if r["domain"] == domain and
                    r["arm"] == "source_orthogonal" and r["search_calls"] == 13)
        f = JointFunctionRiskPosterior([priors[0]], [priors[3]], psi)
        g = JointFunctionRiskPosterior([priors[1]], [priors[2]], psi)
        for prefix, (index, observed) in enumerate(zip(paid["paid_candidate_indices"], paid["paid_observations"]), 1):
            f.update_batch(index, observed[0], None, 1)
            g.update_batch(index, observed[1] - problem.tau, None, 1)
            if not any(c["domain"] == domain and c["prefix"] == prefix for c in spec["cases"]):
                continue
            n0 = min(prefix, 10)
            original, _ = _select_shortlist(problem, initial[:n0], paid["paid_observations"][:n0])
            positions = [initial[:n0].index(tuple(r["point"])) for r in original]
            table, inc = shortlist_indices(positions, list(range(n0)), paid["paid_observations"][:n0],
                                           len(candidates), range(10))
            incumbent = positions[0]
            protocol = {"objective_reference": float(np.dot(problem.weights, problem.ref_point)),
                "candidate_budgets": [80,128,128], "chance_alpha": problem.alpha,
                "familywise_delta": .05, "comparison_budget": 8, "comparison_delta": .05}
            control = dict(protocol, candidate_budgets=spec["incumbent_control_budgets"])
            record = next(r for r in replay["rows"] if r["domain"] == domain and
                          r["arm"] == "source_orthogonal" and r["prefix"] == prefix)
            integrations = []
            all_values = []
            for power, seed in spec["integrations"]:
                draws = posterior_draws(f,g,power,seed)
                values, costs = deployment_values(draws,table,inc,protocol)
                control_value, control_cost = deployment_values(draws,table[incumbent:incumbent+1],
                                                                inc[incumbent:incumbent+1],control)
                chosen = int(values.argmax())
                integrations.append({"draws":len(draws),"seed":seed,"recommended":chosen,
                    "predicted_reward":float(values[chosen]),"predicted_calls":float(costs[chosen]),
                    "reward_of_original_512_recommendation":float(values[record["selected"]["full_deployment"]]),
                    "incumbent_128_reward":float(control_value[0]),"incumbent_128_calls":float(control_cost[0])})
                all_values.append(values)
            truth = np.array([[problem.true_objective(p), np.log(problem.true_sigma(p)[0]**2),
                problem.true_constraint_mean(p)-problem.tau,np.log(problem.true_sigma(p)[1]**2)]
                for p in candidates])[None]
            truth_reward, truth_cost = deployment_values(truth,table,inc,protocol)
            cr, cc = deployment_values(truth,table[incumbent:incumbent+1],inc[incumbent:incumbent+1],control)
            for result in integrations:
                i = result["recommended"]
                result["truth_reward"] = float(truth_reward[i])
                result["truth_calls"] = float(truth_cost[i])
                result["primary_truly_feasible"] = bool(problem.is_truly_feasible(candidates[i]))
            results.append({"domain":domain,"prefix":prefix,"old_512_recommendation":record["selected"]["full_deployment"],
                "integrations":integrations,"max_cross_seed_reward_difference":float(np.abs(all_values[-1]-all_values[-2]).max()),
                "last_seed_loss_of_other_seed_recommendation":float(all_values[-1].max()-all_values[-1][integrations[-2]["recommended"]]),
                "truth_incumbent_128_reward":float(cr[0]),"truth_incumbent_128_calls":float(cc[0]),
                "truth_old_proxy_reward":record["selected_truth"]["old_primary_proxy"]["expected_deployment_reward"],
                "truth_old_proxy_calls":record["selected_truth"]["old_primary_proxy"]["expected_verification_calls"]})
            print(domain,prefix,"precision and allocation scored",flush=True)
    output = {"protocol":spec,"rows":results,"new_calls":0,"cpu_seconds":time.process_time()-started}
    (ROOT / "paper_artifacts/guarded_deployment_replay_v1/precision.json").write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({"rows":len(results),"new_calls":0,"cpu_seconds":output["cpu_seconds"]}),flush=True)


if __name__ == "__main__":
    check()
