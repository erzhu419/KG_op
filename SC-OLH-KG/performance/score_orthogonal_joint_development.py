"""Post-inference analytic scoring on the eight already-open development laws.

Private oracle moments are used only here, after the replay output is fixed.
No episode observations or certificates are generated and no priors are fit.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.verification_kg import verification_utility
from core.joint_verification_kg import joint_verification_utility
from performance.replay_orthogonal_meta_prior import prepare_replay
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law
from representation.joint_function_risk import JointFunctionRiskPosterior


def diagnose_initial_failure(replay, scoring_rows, base):
    """Inspect the worst observed early loss; this never changes the algorithm."""
    contrasts = []
    for seed in sorted({r["law_seed"] for r in scoring_rows}):
        task = {r["arm"]: r for r in scoring_rows if r["law_seed"] == seed}
        a, b = task["u1_joint_batch"]["stages"][0], task["u1_plugin_independent"]["stages"][0]
        contrasts.append((a["true_expected_verified_reward"] - b["true_expected_verified_reward"], seed,
                          a["recommendation"], a["paid_points"]))
    _, seed, index, points = min(contrasts)
    p = replay["contract"]
    archive, recorded, psi, _, _, priors, full = prepare_replay(p)
    old = next(r for r in recorded["rows"] if r["law_seed"] == seed and r["arm"] == p["recorded_path_arm"])
    model = JointFunctionRiskPosterior([priors[1], full[1]], [priors[2], full[2]], psi)
    for observation in old["search_path"][:points]:
        model.update_batch(observation["index"], observation["response"][1],
                           observation["response"][2], p["replicates"])
    means, covariance = model.predict(full_cov=True)
    point_covariance = covariance[index, :, index, :]
    world = InventoryCostSimulator(InventoryPlan(**archive["contexts"][0]),
        sample_cost_law(seed, base["target_context"]["constraint_price_range"]))
    truth = world.oracle_moments(np.asarray(archive["public_policies"])[index])
    observed = [r for r in old["search_path"][:points] if r["index"] == index]
    result = {"scope": "post-inference diagnosis of the worst already-open early development loss",
        "law_seed": seed, "paid_points": points, "candidate": index,
        "observed_candidate": bool(observed), "recorded_batches_at_candidate": observed,
        "posterior_g_eta_mean": means[index].tolist(),
        "posterior_point_covariance": point_covariance.tolist(), "truth_g_eta": truth[1:].tolist(),
        "new_episode_calls": 0, "target_truth_used_for_diagnostics_only": True, "mean_replacement_power": {}}
    for name, replacements in [("unchanged", []), ("g_mean_only", [0]), ("eta_mean_only", [1]), ("both_means", [0, 1])]:
        mean = means[index].copy()
        for head in replacements:
            mean[head] = truth[head + 1]
        _, power = joint_verification_utility(np.zeros(1), np.zeros(1), mean[None], point_covariance[None], p)
        result["mean_replacement_power"][name] = float(power[0])
    (ROOT / "paper_artifacts/orthogonal_joint_replay_v1/initial_failure_diagnosis.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def score():
    replay = json.loads((ROOT / "paper_artifacts/orthogonal_joint_replay_v1/replay.json").read_text())
    p = replay["contract"]
    archive = json.loads((ROOT / p["source_archive"]).read_text())
    old = json.loads((ROOT / p["target_response_archive"]).read_text())
    base = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    plan = InventoryPlan(**archive["contexts"][0])
    policies = np.asarray(archive["public_policies"])
    rows, max_original_error, oracle_evaluations = [], 0., 0
    for seed in sorted({row["law_seed"] for row in replay["rows"]}):
        world = InventoryCostSimulator(plan, sample_cost_law(seed, base["target_context"]["constraint_price_range"]))
        cache = {}
        def evaluate(index):
            nonlocal oracle_evaluations
            if index not in cache:
                truth = world.oracle_moments(policies[index])
                utility, power = verification_utility(truth[None], np.zeros((1, 3)), p)
                cache[index] = float(utility[0]), float(power[0])
                oracle_evaluations += 1
            return cache[index]
        for previous in old["rows"]:
            if previous["law_seed"] == seed:
                max_original_error = max(max_original_error,
                    abs(evaluate(previous["choice"])[0] - previous["final_true_expected_reward"]),
                    abs(evaluate(previous["initial_choice"])[0] - previous["initial_true_expected_reward"]))
        for row in replay["rows"]:
            if row["law_seed"] != seed:
                continue
            stages = []
            for stage in row["stages"]:
                utility, power = evaluate(stage["recommendation"])
                stages.append({"paid_points": stage["paid_points"], "recommendation": stage["recommendation"],
                    "true_expected_verified_reward": utility, "true_verifier_power": power,
                    "predicted_verifier_power": stage["predicted_verifier_power"],
                    "power_error": stage["predicted_verifier_power"] - power})
            rows.append({"law_seed": seed, "arm": row["arm"], "stages": stages})
    if max_original_error > 1e-10:
        raise ValueError(f"Development scoring does not reproduce the original task contract: {max_original_error}")
    summaries = []
    for arm in p["arms"]:
        for stage in p["stages"]:
            selected = [next(s for s in r["stages"] if s["paid_points"] == stage)
                        for r in rows if r["arm"] == arm]
            summaries.append({"arm": arm, "paid_points": stage, "task_blocks": len(selected),
                "mean_true_expected_verified_reward": float(np.mean([s["true_expected_verified_reward"] for s in selected])),
                "mean_abs_verifier_power_error": float(np.mean([abs(s["power_error"]) for s in selected]))})
    paired = []
    for seed in sorted({r["law_seed"] for r in rows}):
        task = {r["arm"]: r for r in rows if r["law_seed"] == seed}
        paired.append({"law_seed": seed, "final_joint_minus_plugin_reward":
            task["u1_joint_batch"]["stages"][-1]["true_expected_verified_reward"]
            - task["u1_plugin_independent"]["stages"][-1]["true_expected_verified_reward"]})
    result = {"study_role": "analytic post-inference scoring of previously opened development laws",
        "algorithm_output": "paper_artifacts/orthogonal_joint_replay_v1/replay.json",
        "target_truth_used_for_scoring_only": True, "independent_confirmation": False,
        "new_episode_calls": 0, "new_observations": 0, "new_verification_calls": 0,
        "analytic_oracle_evaluations": oracle_evaluations,
        "max_original_reward_reproduction_error": max_original_error,
        "rows": rows, "summary": summaries, "paired_final_reward": paired}
    (ROOT / "paper_artifacts/orthogonal_joint_replay_v1/development_scoring.json").write_text(
        json.dumps(result, indent=2) + "\n")
    diagnose_initial_failure(replay, rows, base)
    return result


if __name__ == "__main__":
    result = score()
    print(json.dumps({"summary": result["summary"], "paired": result["paired_final_reward"],
        "original_reward_error": result["max_original_reward_reproduction_error"]}, indent=2))
