"""Diagnose the paid bridge trajectories; no new simulation or validation."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive, resample_normalized_profiles
from core.designs import load_frozen_source_informed_design
from core.legacy_joint_kg import legacy_joint_utility
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.run_legacy_joint_bridge import function_moments, update_objective
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from representation.orthogonal_function_prior import (
    OrthogonalLowOrderBasis, fit_orthogonal_function_prior, same_source_orthogonal_full_prior)
from representation.source_function_prior import head_batches

OUT = ROOT / "paper_artifacts/legacy_joint_bridge_v1"


def analyze():
    result = json.loads((OUT / "results.json").read_text())
    spec = result["protocol"]
    diagnostics, headroom = [], []
    for domain in spec["domains"]:
        inputs = OUT / "inputs" / domain
        archive = FrozenTransferArchive.load(inputs / "archive.json")
        initial, _ = load_frozen_source_informed_design(inputs / "design.json", heldout=domain,
            seed=spec["target_seed"], n0=spec["n0"], dimension=spec["target_dimension"])
        problem = build_scalarized_problem(domain, spec["target_dimension"], spec["L"],
                                           spec["sigma"], spec["alpha"], spec["weights"])
        lifted = resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]), problem.d)
        candidates = list(dict.fromkeys(list(initial) + [tuple(map(int, problem.continuous_to_int(x))) for x in lifted]))
        psi = legacy_public_coordinates(np.asarray(candidates) / problem.L)
        tasks = legacy_source_labels(archive)
        basis = OrthogonalLowOrderBasis(np.vstack([t["psi"] for t in tasks]))
        priors = [fit_orthogonal_function_prior(head_batches(tasks, h), [t["noise"][:, h] for t in tasks],
            basis, max_iterations=spec["source_max_iterations"]) for h in range(3)]
        old_deployed = next(r["deployed"] for r in result["rows"] if r["domain"] == domain and r["arm"] == "legacy_proposal_only")
        # Post-decision upper bound: distinguishes a bad ranking from a bank
        # with no remaining utility gain. This never chooses a search point.
        truth_f = np.array([problem.true_objective(p) for p in candidates])
        truth_risk = np.array([[problem.true_constraint_mean(p) - problem.tau,
            np.log(problem.true_sigma(p)[1] ** 2)] for p in candidates])
        scoring = {"objective_reference": float(np.dot(problem.weights, problem.ref_point)),
            "verification_replicates": spec["terminal_candidate_budgets"][0],
            "chance_alpha": problem.alpha, "candidate_delta": spec["terminal_familywise_delta"] / 3}
        bank_utility = legacy_joint_utility(truth_f, np.zeros(len(candidates)), truth_risk,
                                          np.zeros((len(candidates), 2, 2)), scoring)[0]
        old_index = candidates.index(tuple(old_deployed))
        headroom.append({"domain": domain, "candidate_count": len(candidates),
            "best_primary_utility": float(bank_utility.max()), "best_candidate": int(bank_utility.argmax()),
            "legacy_deployed_primary_utility": float(bank_utility[old_index]),
            "remaining_primary_utility": float(bank_utility.max() - bank_utility[old_index])})
        for arm in ["same_source_full", "source_orthogonal"]:
            arm_priors = [same_source_orthogonal_full_prior(p) for p in priors] if arm == "same_source_full" else priors
            for row in [r for r in result["rows"] if r["domain"] == domain and r["arm"] == arm]:
                model = JointFunctionRiskPosterior([arm_priors[1]], [arm_priors[2]], psi)
                fm, fc = function_moments(arm_priors[0], psi)
                f_noise = float(np.mean(np.concatenate([t.replicate_variance[:, 0] for t in archive.tasks])))
                for index, value in zip(row["paid_candidate_indices"], row["paid_observations"]):
                    model.update_batch(index, value[1] - problem.tau, None, 1)
                    update_objective(fm, fc, index, value[0], f_noise)
                mean, covariance = model.predict(full_cov=True)
                for role, index in [("model_primary", row["primary_index"]),
                                    ("legacy_deployed_support", candidates.index(tuple(old_deployed)))]:
                    point = candidates[index]
                    truth = np.array([problem.true_constraint_mean(point) - problem.tau,
                                      np.log(problem.true_sigma(point)[1] ** 2)])
                    C = covariance[index, :, index, :]
                    protocol = {"objective_reference": float(np.dot(problem.weights, problem.ref_point)),
                        "verification_replicates": spec["terminal_candidate_budgets"][0],
                        "chance_alpha": problem.alpha, "candidate_delta": spec["terminal_familywise_delta"] / 3}
                    base_power = float(legacy_joint_utility(np.array([fm[index]]), np.array([fc[index,index]]),
                        mean[index:index+1], C[None], protocol)[1][0])
                    counterfactual_powers = {}
                    for h, name in [(0, "replace_g_mean_by_truth"), (1, "replace_eta_mean_by_truth")]:
                        fixed = mean[index].copy(); fixed[h] = truth[h]
                        counterfactual_powers[name] = float(legacy_joint_utility(np.array([fm[index]]),
                            np.array([fc[index,index]]), fixed[None], C[None], protocol)[1][0])
                    diagnostics.append({"domain": domain, "arm": arm, "search_calls": row["search_calls"],
                        "role": role, "candidate": index, "observed_at_this_budget": index in row["paid_candidate_indices"],
                        "posterior_g_eta": mean[index].tolist(), "truth_g_eta": truth.tolist(),
                        "error_in_posterior_sd": ((mean[index] - truth) / np.sqrt(np.diag(C))).tolist(),
                        "predicted_primary_power": base_power,
                        "diagnostic_oracle_mean_substitution": counterfactual_powers})
    summary = {"role": "post-decision failure diagnosis on the same opened development paths; truth substitutions are not an algorithm",
        "new_calls": 0, "diagnostics": diagnostics, "post_decision_bank_headroom": headroom,
        "all_source_heads_converged": all(h["optimizer_converged"] for d in result["source_diagnostics"].values() for h in d["fitted_heads"]),
        "extra_search_changes_deployed_policy": sum(r["deployed"] != next(b["deployed"] for b in result["rows"]
            if b["domain"] == r["domain"] and b["arm"] == "legacy_proposal_only") for r in result["rows"] if r["search_calls"] == 13),
        "certified_rows": sum(r["verification"]["certified"] for r in result["rows"]),
        "false_certificates": sum(r["verification"]["certified"] and not r["deployed_truth"]["true_feasible"] for r in result["rows"])}
    (OUT / "diagnosis.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in diagnostics:
        if r["domain"] == "InventorySupplyChain":
            print(json.dumps(r))
    print(json.dumps({k:v for k,v in summary.items() if k != "diagnostics"}))


if __name__ == "__main__":
    analyze()
