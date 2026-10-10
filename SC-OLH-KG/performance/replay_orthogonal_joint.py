"""Replay the full paid prefixes with U1 independent and joint batch updates."""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.joint_verification_kg import joint_verification_kg, joint_verification_utility
from core.verification_kg import posterior_covariances
from performance.replay_orthogonal_meta_prior import prepare_replay
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.transfer_mixture import TransferMixtureGPR


def independent_pair_moments(models, candidates):
    means = np.column_stack([m.posterior_mean_many(candidates) for m in models])
    head_covariance = posterior_covariances(models, candidates)
    covariance = np.zeros((len(candidates), 2, len(candidates), 2))
    covariance[:, 0, :, 0], covariance[:, 1, :, 1] = head_covariance[..., 0], head_covariance[..., 1]
    return means, covariance


def independent_local_components(models, index):
    return [(float(np.exp(gw + ew)), np.array([g.posterior_mean([index]), e.posterior_mean([index])]),
             np.diag([g.posterior_var([index]), e.posterior_var([index])]))
            for (gw, g), (ew, e) in itertools.product(
                zip(models[0].log_weights, models[0].models), zip(models[1].log_weights, models[1].models))]


def run():
    started = time.process_time()
    p = json.loads((ROOT / "performance/orthogonal_joint_replay_v1.json").read_text())
    archive, recorded, psi, reference, basis, priors, full = prepare_replay(p)
    candidates = np.arange(len(psi))[:, None]
    rows = []
    for old in recorded["rows"]:
        if old["arm"] != p["recorded_path_arm"]:
            continue
        for arm in p["arms"]:
            if arm == "full_joint_batch":
                objective = full[0].target_model(psi)
                joint = JointFunctionRiskPosterior([full[1]], [full[2]], psi)
            else:
                objective = TransferMixtureGPR([priors[0], full[0]], psi, reference)
                if arm == "u1_joint_batch":
                    joint = JointFunctionRiskPosterior([priors[1], full[1]], [priors[2], full[2]], psi)
                else:
                    independent = [TransferMixtureGPR([priors[h], full[h]], psi, reference) for h in (1, 2)]
            errors, scores, stages = [], [], []
            for step, observation in enumerate(old["search_path"]):
                index = int(observation["index"])
                response, noise = np.asarray(observation["response"]), np.asarray(observation["noise"])
                if arm == "u1_plugin_independent":
                    before = np.array([m.posterior_mean_many([[index]])[0] for m in independent])
                    if step >= p["initial_points"]:
                        scores.append(joint_batch_log_score(independent_local_components(independent, index),
                            response[1], response[2], p["replicates"]))
                    for h, model in enumerate(independent, 1):
                        model.update([index], float(response[h]), float(noise[h]))
                else:
                    before = joint.predict()[0][index]
                    score = joint.update_batch(index, float(response[1]), float(response[2]), p["replicates"])
                    if step >= p["initial_points"]:
                        scores.append(score)
                if step >= p["initial_points"]:
                    errors.append((before - response[1:]) ** 2)
                objective.update([index], float(response[0]), float(noise[0]))
                if step + 1 in p["stages"]:
                    pair_mean, covariance = (independent_pair_moments(independent, candidates)
                        if arm == "u1_plugin_independent" else joint.predict(full_cov=True))
                    indices = np.arange(len(psi))
                    point_covariance = covariance[indices, :, indices, :]
                    f_mean = objective.posterior_mean_many(candidates)
                    f_covariance = posterior_covariances([objective], candidates)[..., 0]
                    utility, power = joint_verification_utility(f_mean, np.diag(f_covariance),
                        pair_mean, point_covariance, p)
                    measurement_noise = np.column_stack([np.full(len(psi), .12 ** 2 / p["replicates"]),
                        np.exp(pair_mean[:, 1] + .5 * point_covariance[:, 1, 1]) / p["replicates"],
                        np.full(len(psi), float(polygamma(1, (p["replicates"] - 1) / 2)))])
                    kg = joint_verification_kg(f_mean, f_covariance, pair_mean, covariance, measurement_noise, p)
                    choice = int(np.argmax(utility))
                    correlation = point_covariance[:, 0, 1] / np.sqrt(
                        point_covariance[:, 0, 0] * point_covariance[:, 1, 1])
                    stages.append({"paid_points": step + 1, "recommendation": choice,
                        "predicted_utility": float(utility[choice]), "predicted_verifier_power": float(power[choice]),
                        "next_kg_probe": int(np.argmax(kg)), "max_kg": float(kg.max()),
                        "g_eta_correlation_at_recommendation": float(correlation[choice]),
                        "max_abs_g_eta_correlation": float(np.max(np.abs(correlation)))})
            rows.append({"law_seed": old["law_seed"], "arm": arm,
                "recorded_policy_indices": [item["index"] for item in old["search_path"]],
                "label_rmse_after_initial": np.sqrt(np.mean(errors, axis=0)).tolist(),
                "mean_joint_log_predictive_density_after_initial": float(np.mean(scores)),
                "stages": stages, "joint_diagnostics": joint.diagnostics()
                    if arm != "u1_plugin_independent" else None})
    summary = []
    for arm in p["arms"]:
        selected = [r for r in rows if r["arm"] == arm]
        summary.append({"arm": arm, "task_blocks": len(selected),
            "mean_label_rmse_after_initial": np.mean([r["label_rmse_after_initial"] for r in selected], axis=0).tolist(),
            "mean_joint_log_predictive_density_after_initial": float(np.mean([
                r["mean_joint_log_predictive_density_after_initial"] for r in selected]))})
    paired = []
    for seed in sorted({r["law_seed"] for r in rows}):
        task = {r["arm"]: r for r in rows if r["law_seed"] == seed}
        left, right = task["u1_joint_batch"], task["u1_plugin_independent"]
        paired.append({"law_seed": seed,
            "joint_minus_plugin_log_score": left["mean_joint_log_predictive_density_after_initial"]
                - right["mean_joint_log_predictive_density_after_initial"],
            "recommendation_changes": [a["recommendation"] != b["recommendation"]
                for a, b in zip(left["stages"], right["stages"])],
            "kg_probe_changes": [a["next_kg_probe"] != b["next_kg_probe"]
                for a, b in zip(left["stages"], right["stages"])]})
    result = {"contract": p, "basis": basis.diagnostics(), "summary": summary,
        "rows": rows, "paired_joint_vs_plugin": paired,
        "source_calls_in_reused_archive": archive["source_episode_calls"],
        "target_calls_in_reused_paths": sum(len(r["search_path"]) * p["replicates"]
            for r in recorded["rows"] if r["arm"] == p["recorded_path_arm"]),
        "head_order": ["constraint_mean", "log_variance"],
        "new_source_calls": 0, "new_target_calls": 0, "new_verification_calls": 0,
        "target_truth_used": False, "independent_confirmation": False,
        "local_cpu_seconds": time.process_time() - started}
    output = ROOT / p["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "paired": result["paired_joint_vs_plugin"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}, indent=2))
