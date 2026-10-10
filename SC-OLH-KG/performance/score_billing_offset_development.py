"""Score fixed offset-linkage recommendations on opened development laws."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.joint_verification_kg import joint_verification_utility
from core.verification_kg import verification_utility
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


def score():
    directory = ROOT / "paper_artifacts/billing_offset_replay_v1"
    replay = json.loads((directory / "replay.json").read_text())
    p = replay["contract"]
    archive = json.loads((ROOT / p["source_archive"]).read_text())
    base = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    previous = json.loads((ROOT / "paper_artifacts/orthogonal_billing_replay_v1/development_scoring.json").read_text())
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
                if row["arm"] not in p["arms"]:
                    old = next(r for r in previous["rows"] if r["law_seed"] == seed and r["arm"] == row["arm"]
                               and r["paid_points"] == stage["paid_points"])
                    reproduction_error = max(reproduction_error, abs(reward - old["true_reward"]))
                record = {"law_seed": seed, "arm": row["arm"], "paid_points": stage["paid_points"],
                    "recommendation": stage["recommendation"], "true_reward": reward,
                    "true_verifier_power": float(power[0]), "predicted_verifier_power": stage["predicted_verifier_power"],
                    "absolute_power_error": abs(stage["predicted_verifier_power"] - float(power[0]))}
                if row["arm"] == "linked_billing_joint" and stage["paid_points"] == p["initial_points"]:
                    mean = np.asarray(stage["recommendation_g_eta_mean"])
                    covariance = np.asarray(stage["recommendation_g_eta_covariance"])
                    replaced = {}
                    for name, heads in [("unchanged", []), ("g_mean_only", [0]), ("eta_mean_only", [1]), ("both_means", [0, 1])]:
                        modified = mean.copy()
                        modified[heads] = truth[1:][heads]
                        _, probability = joint_verification_utility(np.zeros(1), np.zeros(1), modified[None], covariance[None], p)
                        replaced[name] = float(probability[0])
                    record.update({"posterior_g_eta_mean": mean.tolist(), "truth_g_eta": truth[1:].tolist(),
                        "mean_error_in_posterior_sd": ((mean - truth[1:]) / np.sqrt(np.diag(covariance))).tolist(),
                        "mean_replacement_power": replaced})
                rows.append(record)
    if reproduction_error > 1e-10:
        raise ValueError("Offset scorer changed the original development task contract")
    arms = p["arms"] + ["billing_joint", "billing_full_joint", "u1_joint_batch"]
    summary = [{"arm": arm, "paid_points": points,
        "mean_true_reward": float(np.mean([r["true_reward"] for r in rows if r["arm"] == arm and r["paid_points"] == points])),
        "mean_absolute_power_error": float(np.mean([r["absolute_power_error"] for r in rows if r["arm"] == arm and r["paid_points"] == points]))}
        for arm in arms for points in p["stages"]]
    seeds = sorted({r["law_seed"] for r in rows})
    bootstrap = np.random.default_rng(1005).integers(len(seeds), size=(5000, len(seeds)))
    paired = []
    for control in ("billing_joint", "linked_old_covariance_joint", "linked_pooled_joint",
                    "linked_old_covariance_old_full_joint", "billing_full_joint", "u1_joint_batch"):
        for points in p["stages"]:
            selected = {r["law_seed"]: r["true_reward"] for r in rows if r["arm"] == "linked_billing_joint" and r["paid_points"] == points}
            baseline = {r["law_seed"]: r["true_reward"] for r in rows if r["arm"] == control and r["paid_points"] == points}
            difference = np.array([selected[s] - baseline[s] for s in seeds])
            paired.append({"control": control, "paid_points": points,
                "paired_mean_reward_difference": float(difference.mean()),
                "development_bootstrap_95_interval": np.percentile(difference[bootstrap].mean(axis=1), [2.5, 97.5]).tolist(),
                "wins_ties_losses": [int(np.sum(difference > 1e-10)), int(np.sum(np.abs(difference) <= 1e-10)), int(np.sum(difference < -1e-10))],
                "per_seed_difference": {str(s): float(d) for s, d in zip(seeds, difference)}})
    failure = sorted([r for r in rows if r["arm"] == "linked_billing_joint" and r["paid_points"] == 8],
                     key=lambda r: r["absolute_power_error"], reverse=True)[:2]
    attribution = []
    for control in ("billing_joint", "billing_full_joint", "u1_joint_batch"):
        for points in p["stages"]:
            selected = {r["law_seed"]: r["true_reward"] for r in rows
                        if r["arm"] == "linked_old_covariance_old_full_joint" and r["paid_points"] == points}
            baseline = {r["law_seed"]: r["true_reward"] for r in rows if r["arm"] == control and r["paid_points"] == points}
            difference = np.array([selected[s] - baseline[s] for s in seeds])
            attribution.append({"control": control, "paid_points": points,
                "paired_mean_reward_difference": float(difference.mean()),
                "development_bootstrap_95_interval": np.percentile(difference[bootstrap].mean(axis=1), [2.5, 97.5]).tolist(),
                "wins_ties_losses": [int(np.sum(difference > 1e-10)), int(np.sum(np.abs(difference) <= 1e-10)), int(np.sum(difference < -1e-10))],
                "per_seed_difference": {str(s): float(d) for s, d in zip(seeds, difference)}})
    result = {"study_role": "post-inference analytic scoring of opened development laws",
        "target_truth_used_for_scoring_only": True, "independent_confirmation": False,
        "new_source_calls": 0, "new_target_calls": 0, "new_verification_calls": 0,
        "baseline_reproduction_error": reproduction_error, "rows": rows, "summary": summary,
        "paired": paired, "offset_only_attribution": attribution,
        "largest_two_early_power_errors": failure}
    (directory / "development_scoring.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    result = score()
    print(json.dumps({"summary": result["summary"], "paired": result["paired"],
        "early_failure_diagnosis": result["largest_two_early_power_errors"]}, indent=2))
