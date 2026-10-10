"""Source-held-out mean repair, then joint filtering of saved paid prefixes."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.joint_verification_kg import joint_verification_kg, joint_verification_utility
from core.verification_kg import posterior_covariances
from performance.replay_orthogonal_meta_prior import prepare_replay
from problems.inventory_cost_plan import InventoryPlan
from representation.inventory_mean_prior import InventoryFunctionBasis, fit_inventory_posterior_group_prior
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_function_prior import fit_orthogonal_function_prior
from representation.orthogonal_inventory_prior import OrthogonalInventoryBasis, embed_inventory_prior
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.transfer_mixture import TransferMixtureGPR


def fit_source_means(archive, reference, old_basis, p):
    plans = [InventoryPlan(**c) for c in archive["contexts"]]
    policies = np.asarray(archive["public_policies"])
    psi = np.array([plans[a["context"]].trajectory(policies[a["policy_index"]])
                    .coordinates(plans[a["context"]].capacity) for a in archive["assignments"]])
    data = [{"psi": psi, "response": np.asarray(r["response"]), "noise": np.asarray(r["noise"])}
            for r in archive["source_records"]]
    rows, raw_priors = [], []
    test = p["source_test_positions"]
    for head in range(2):
        for held_index, held in enumerate(data):
            keep = [r for i, r in enumerate(data) if i != held_index]
            batches, noises = head_batches(keep, head), [r["noise"][:, head] for r in keep]
            selected = fit_inventory_posterior_group_prior(batches, noises, reference,
                p["source_adaptation_positions"], test)
            u1 = fit_orthogonal_function_prior(batches, noises, old_basis)
            for name, prior in [("u1", u1), ("billing_selected", selected),
                                ("billing_residual_002", replace(selected, discrepancy_variance=.02))]:
                model = prior.target_model(psi)
                for index in p["source_adaptation_positions"]:
                    model.update([index], held["response"][index, head], held["noise"][index, head])
                mean = model.posterior_mean_many(np.asarray(test)[:, None])
                variance = model.posterior_var_many(np.asarray(test)[:, None]) + held["noise"][test, head]
                error = held["response"][test, head] - mean
                rows.append({"head": head, "held_source": held_index, "arm": name,
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "mean_log_density": float(np.mean(norm.logpdf(error, scale=np.sqrt(variance)))),
                    "standardized_residual_rms": float(np.sqrt(np.mean(error ** 2 / variance))),
                    "selected_names": prior.basis.names, "discrepancy": prior.discrepancy_variance})
        raw_priors.append(fit_inventory_posterior_group_prior(head_batches(data, head),
            [r["noise"][:, head] for r in data], reference, p["source_adaptation_positions"], test))
    summaries = [{"head": head, "arm": arm, **{metric: float(np.mean([
        r[metric] for r in rows if r["head"] == head and r["arm"] == arm]))
        for metric in ("rmse", "mean_log_density", "standardized_residual_rms")}}
        for head in range(2) for arm in ("u1", "billing_selected", "billing_residual_002")]
    result = {"selection": p["source_selection"], "target_response_labels_used": False,
        "rows": rows, "summary": summaries,
        "all_source_priors": [{"mean": r.mean.tolist(), "covariance": r.covariance.tolist(),
            "discrepancy": r.discrepancy_variance, "diagnostics": r.diagnostics} for r in raw_priors],
        "source_signal_scales": [float(np.mean([max(float(np.var(r["response"][:, h])
            - np.mean(r["noise"][:, h])), .02) for r in data])) for h in range(2)]}
    return raw_priors, result


def run():
    start = time.process_time()
    p = json.loads((ROOT / "performance/orthogonal_billing_replay_v1.json").read_text())
    archive, recorded, psi, reference, old_basis, u1, u1_full = prepare_replay(p)
    output = ROOT / p["output"]
    output.mkdir(parents=True, exist_ok=True)
    raw, source_result = fit_source_means(archive, reference, old_basis, p)
    basis = OrthogonalInventoryBasis(reference)
    selected = [embed_inventory_prior(r, basis, reference) for r in raw]
    full = []
    for h, prior in enumerate(selected):
        covariance = np.diag(np.r_[prior.covariance[0, 0],
            np.full(basis.feature_dim, source_result["source_signal_scales"][h] / basis.feature_dim)])
        full.append(replace(prior, covariance=covariance,
            diagnostics={**basis.diagnostics(), "mode": "same_source_orthogonal_full"}))
    source_result["orthogonal_basis"] = basis.diagnostics()
    source_result["local_cpu_seconds"] = time.process_time() - start
    (output / "source_check.json").write_text(json.dumps(source_result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source_summary": source_result["summary"],
        "selected": [(r.basis.names, r.discrepancy_variance) for r in raw],
        "source_cpu_seconds": source_result["local_cpu_seconds"]}), flush=True)
    rows = []
    candidates = np.arange(len(psi))[:, None]
    for old in recorded["rows"]:
        if old["arm"] != p["recorded_path_arm"]:
            continue
        for arm in p["arms"]:
            selected_arm, full_arm = selected, full
            if arm == "billing_joint_residual_002":
                selected_arm = [replace(r, discrepancy_variance=.02) for r in selected]
                full_arm = [replace(r, discrepancy_variance=.02) for r in full]
            objective = (full_arm[0].target_model(psi) if arm == "billing_full_joint"
                         else TransferMixtureGPR([selected_arm[0], full_arm[0]], psi, reference))
            g_priors = [full_arm[1]] if arm == "billing_full_joint" else [selected_arm[1], full_arm[1]]
            joint = JointFunctionRiskPosterior(g_priors, [u1[2], u1_full[2]], psi)
            errors, scores, stages = [], [], []
            for step, observation in enumerate(old["search_path"]):
                index, response, noise = observation["index"], observation["response"], observation["noise"]
                before = joint.predict()[0][index]
                score = joint.update_batch(index, response[1], response[2], p["replicates"])
                if step >= p["initial_points"]:
                    errors.append((before - np.asarray(response)[1:]) ** 2)
                    scores.append(score)
                objective.update([index], response[0], noise[0])
                if step + 1 in p["stages"]:
                    mean, covariance = joint.predict(full_cov=True)
                    indices = np.arange(len(psi))
                    point_covariance = covariance[indices, :, indices, :]
                    fm = objective.posterior_mean_many(candidates)
                    fc = posterior_covariances([objective], candidates)[..., 0]
                    utility, power = joint_verification_utility(fm, np.diag(fc), mean, point_covariance, p)
                    measurement_noise = np.column_stack([np.full(len(psi), .12 ** 2 / p["replicates"]),
                        np.exp(mean[:, 1] + .5 * point_covariance[:, 1, 1]) / p["replicates"],
                        np.full(len(psi), float(polygamma(1, (p["replicates"] - 1) / 2)))])
                    kg = joint_verification_kg(fm, fc, mean, covariance, measurement_noise, p)
                    choice = int(np.argmax(utility))
                    stages.append({"paid_points": step + 1, "recommendation": choice,
                        "predicted_utility": float(utility[choice]), "predicted_verifier_power": float(power[choice]),
                        "recommendation_g_eta_mean": mean[choice].tolist(),
                        "recommendation_g_eta_covariance": point_covariance[choice].tolist(),
                        "next_kg_probe": int(np.argmax(kg)), "max_kg": float(kg.max())})
            rows.append({"law_seed": old["law_seed"], "arm": arm,
                "recorded_policy_indices": [r["index"] for r in old["search_path"]],
                "label_rmse_after_initial": np.sqrt(np.mean(errors, axis=0)).tolist(),
                "mean_joint_log_predictive_density_after_initial": float(np.mean(scores)),
                "stages": stages, "joint_diagnostics": joint.diagnostics()})
    # Existing U1 rows are fixed references, not silently regenerated evidence.
    baseline = json.loads((ROOT / p["baseline_replay"]).read_text())
    baseline_rows = [r for r in baseline["rows"] if r["arm"] == "u1_joint_batch"]
    for row in baseline_rows:
        current = next(r for r in rows if r["law_seed"] == row["law_seed"])
        if row["recorded_policy_indices"] != current["recorded_policy_indices"]:
            raise ValueError("U1 reference does not have the same paid policy history")
    summaries = [{"arm": arm, "task_blocks": len(baseline_rows),
        "mean_label_rmse_after_initial": np.mean([r["label_rmse_after_initial"]
            for r in rows + baseline_rows if r["arm"] == arm], axis=0).tolist(),
        "mean_joint_log_predictive_density_after_initial": float(np.mean([
            r["mean_joint_log_predictive_density_after_initial"] for r in rows + baseline_rows if r["arm"] == arm]))}
        for arm in p["arms"] + ["u1_joint_batch"]]
    result = {"contract": p, "source_priors": [r.diagnostics for r in selected],
        "basis": basis.diagnostics(), "rows": rows + baseline_rows, "summary": summaries,
        "target_truth_used": False, "independent_confirmation": False,
        "new_source_calls": 0, "new_target_calls": 0, "new_verification_calls": 0,
        "source_calls_in_reused_archive": archive["source_episode_calls"],
        "target_calls_in_reused_paths": len(baseline_rows) * 24 * p["replicates"],
        "local_cpu_seconds": time.process_time() - start}
    (output / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "local_cpu_seconds": result["local_cpu_seconds"]}, indent=2))
