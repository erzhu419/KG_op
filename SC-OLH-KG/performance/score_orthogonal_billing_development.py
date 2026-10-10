"""Analytic scoring after choices are fixed; opened development laws only."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.verification_kg import verification_utility
from core.joint_verification_kg import joint_verification_utility
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law
from performance.replay_orthogonal_meta_prior import prepare_replay
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_inventory_prior import OrthogonalInventoryBasis, embed_inventory_prior
from representation.source_function_prior import SourceFunctionPrior


def diagnose_component_sharing(directory, replay, failures):
    """Reconstruct fixed models from source artifacts, without any oracle input."""
    p = replay["contract"]
    _, recorded, psi, reference, _, u1, u1_full = prepare_replay(p)
    source = json.loads((directory / "source_check.json").read_text())
    raw = source["all_source_priors"][1]
    original = SourceFunctionPrior(InventoryFunctionBasis(reference, raw["diagnostics"]["selected_indices"]),
        np.asarray(raw["mean"]), np.asarray(raw["covariance"]), raw["discrepancy"], raw["diagnostics"])
    selected = embed_inventory_prior(original, OrthogonalInventoryBasis(reference), reference)
    covariance = np.diag(np.r_[selected.covariance[0, 0],
        np.full(selected.basis.feature_dim, source["source_signal_scales"][1] / selected.basis.feature_dim)])
    full = SourceFunctionPrior(selected.basis, selected.mean, covariance, selected.discrepancy_variance,
                               {"mode": "same_source_orthogonal_full"})
    for failure in failures:
        row = next(r for r in recorded["rows"] if r["law_seed"] == failure["law_seed"] and r["arm"] == p["recorded_path_arm"])
        joint = JointFunctionRiskPosterior([selected, full], [u1[2], u1_full[2]], psi)
        observations = row["search_path"][:failure["paid_points"]]
        for observation in observations:
            joint.update_batch(observation["index"], observation["response"][1], observation["response"][2], p["replicates"])
        failure["early_joint_diagnostics"] = joint.diagnostics()
        failure["components_at_recommendation"] = [{"weight": w, "mean": m.tolist(), "covariance": c.tolist()}
            for w, m, c in joint.local_components(failure["recommendation"])]
        failure["recorded_batches_at_recommendation"] = [r for r in observations if r["index"] == failure["recommendation"]]
        design = np.column_stack([np.ones(len(psi)), original.basis.transform(psi)])
        singular = np.linalg.svd(design[[r["index"] for r in observations]], compute_uv=False)
        failure["initial_billing_design_singular_values"] = singular.tolist()
        failure["candidate_prior_covariance_sharing"] = float((design @ original.covariance @ design.T)[failure["recommendation"], failure["recommendation"]])


def score():
    directory = ROOT / "paper_artifacts/orthogonal_billing_replay_v1"
    replay = json.loads((directory / "replay.json").read_text())
    p = replay["contract"]
    archive = json.loads((ROOT / p["source_archive"]).read_text())
    base = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    previous = json.loads((ROOT / "paper_artifacts/orthogonal_joint_replay_v1/development_scoring.json").read_text())
    policies = np.asarray(archive["public_policies"])
    rows, reproduction_error = [], 0.
    for seed in sorted({r["law_seed"] for r in replay["rows"]}):
        world = InventoryCostSimulator(InventoryPlan(**archive["contexts"][0]),
            sample_cost_law(seed, base["target_context"]["constraint_price_range"]))
        for row in replay["rows"]:
            if row["law_seed"] != seed:
                continue
            for stage in row["stages"]:
                truth = world.oracle_moments(policies[stage["recommendation"]])
                utility, power = verification_utility(truth[None], np.zeros((1, 3)), p)
                reward = float(utility[0])
                if row["arm"] == "u1_joint_batch":
                    old = next(r for r in previous["rows"] if r["law_seed"] == seed and r["arm"] == row["arm"])
                    old_stage = next(s for s in old["stages"] if s["paid_points"] == stage["paid_points"])
                    reproduction_error = max(reproduction_error, abs(reward - old_stage["true_expected_verified_reward"]))
                rows.append({"law_seed": seed, "arm": row["arm"], "paid_points": stage["paid_points"],
                    "recommendation": stage["recommendation"], "true_reward": reward,
                    "true_verifier_power": float(power[0]), "predicted_verifier_power": stage["predicted_verifier_power"],
                    "absolute_power_error": abs(stage["predicted_verifier_power"] - float(power[0]))})
    if reproduction_error > 1e-10:
        raise ValueError("Analytic evaluator changed the previous development task contract")
    arms = p["arms"] + ["u1_joint_batch"]
    summary = [{"arm": arm, "paid_points": points,
        "mean_true_reward": float(np.mean([r["true_reward"] for r in rows if r["arm"] == arm and r["paid_points"] == points])),
        "mean_absolute_power_error": float(np.mean([r["absolute_power_error"] for r in rows if r["arm"] == arm and r["paid_points"] == points]))}
        for arm in arms for points in p["stages"]]
    seeds = sorted({r["law_seed"] for r in rows})
    bootstrap = np.random.default_rng(1005).integers(len(seeds), size=(5000, len(seeds)))
    paired = []
    for control in ("u1_joint_batch", "billing_joint_residual_002", "billing_full_joint"):
        for points in p["stages"]:
            selected = {r["law_seed"]: r["true_reward"] for r in rows if r["arm"] == "billing_joint" and r["paid_points"] == points}
            baseline = {r["law_seed"]: r["true_reward"] for r in rows if r["arm"] == control and r["paid_points"] == points}
            difference = np.array([selected[s] - baseline[s] for s in seeds])
            paired.append({"control": control, "paid_points": points,
                "paired_mean_reward_difference": float(difference.mean()),
                "development_bootstrap_95_interval": np.percentile(difference[bootstrap].mean(axis=1), [2.5, 97.5]).tolist(),
                "wins_ties_losses": [int(np.sum(difference > 1e-10)), int(np.sum(np.abs(difference) <= 1e-10)), int(np.sum(difference < -1e-10))],
                "per_seed_difference": {str(s): float(d) for s, d in zip(seeds, difference)}})
    failures = []
    early = sorted([r for r in rows if r["arm"] == "billing_joint" and r["paid_points"] == 8],
                   key=lambda r: r["absolute_power_error"], reverse=True)[:2]
    for row in early:
        stage = next(r for r in replay["rows"] if r["arm"] == row["arm"] and r["law_seed"] == row["law_seed"])["stages"][0]
        world = InventoryCostSimulator(InventoryPlan(**archive["contexts"][0]),
            sample_cost_law(row["law_seed"], base["target_context"]["constraint_price_range"]))
        truth = world.oracle_moments(policies[row["recommendation"]])[1:]
        mean = np.asarray(stage["recommendation_g_eta_mean"])
        covariance = np.asarray(stage["recommendation_g_eta_covariance"])
        replacements = {}
        for name, heads in [("unchanged", []), ("g_mean_only", [0]), ("eta_mean_only", [1]), ("both_means", [0, 1])]:
            modified = mean.copy()
            modified[heads] = truth[heads]
            _, power = joint_verification_utility(np.zeros(1), np.zeros(1), modified[None], covariance[None], p)
            replacements[name] = float(power[0])
        failures.append({**row, "posterior_g_eta_mean": mean.tolist(), "truth_g_eta": truth.tolist(),
            "posterior_g_eta_covariance": covariance.tolist(),
            "mean_error_in_posterior_sd": ((mean - truth) / np.sqrt(np.diag(covariance))).tolist(),
            "mean_replacement_power": replacements})
    diagnose_component_sharing(directory, replay, failures)
    result = {"study_role": "post-inference analytic scoring of already-open development laws",
        "target_truth_used_for_scoring_only": True, "independent_confirmation": False,
        "new_episode_calls": 0, "new_verification_calls": 0,
        "baseline_reproduction_error": reproduction_error, "rows": rows, "summary": summary, "paired": paired,
        "largest_two_early_power_errors": failures}
    (directory / "development_scoring.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    result = score()
    print(json.dumps({"summary": result["summary"], "paired": result["paired"],
                      "early_failure_diagnosis": result["largest_two_early_power_errors"]}, indent=2))
