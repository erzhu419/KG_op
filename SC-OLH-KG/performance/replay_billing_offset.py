"""Public-offset linkage: source validation and joint paid-prefix replay."""

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
from representation.billing_offset_prior import fit_billing_offset_prior, link_billing_coefficients, pooled_linked_prior
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_inventory_prior import OrthogonalInventoryBasis, embed_inventory_prior
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.transfer_mixture import TransferMixtureGPR


def source_check(archive, reference, p):
    plans = [InventoryPlan(**c) for c in archive["contexts"]]
    policies = np.asarray(archive["public_policies"])
    psi = np.array([plans[a["context"]].trajectory(policies[a["policy_index"]])
        .coordinates(plans[a["context"]].capacity) for a in archive["assignments"]])
    offsets = np.array([plans[a["context"]].mean_offsets for a in archive["assignments"]])
    data = [{"psi": psi, "response": np.asarray(r["response"]), "noise": np.asarray(r["noise"])}
            for r in archive["source_records"]]
    old_source = json.loads((ROOT / p["baseline_source"]).read_text())
    rows, priors = [], []
    for head in range(2):
        for held_index, held in enumerate(data):
            keep = [r for i, r in enumerate(data) if i != held_index]
            prior = fit_billing_offset_prior(head_batches(keep, head), [r["noise"][:, head] for r in keep],
                reference, offsets[:, head], p["source_adaptation_positions"], p["source_test_positions"])
            for arm, candidate in [("linked", prior), ("linked_pooled", pooled_linked_prior(prior))]:
                model = candidate.target_model(psi)
                for index in p["source_adaptation_positions"]:
                    model.update([index], held["response"][index, head] - offsets[index, head], held["noise"][index, head])
                test = p["source_test_positions"]
                mean = model.posterior_mean_many(np.asarray(test)[:, None]) + offsets[test, head]
                variance = model.posterior_var_many(np.asarray(test)[:, None]) + held["noise"][test, head]
                error = held["response"][test, head] - mean
                rows.append({"head": head, "held_source": held_index, "arm": arm,
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "mean_log_density": float(np.mean(norm.logpdf(error, scale=np.sqrt(variance)))),
                    "standardized_residual_rms": float(np.sqrt(np.mean(error ** 2 / variance))),
                    "selected_discrepancy": prior.discrepancy_variance})
        prior = fit_billing_offset_prior(head_batches(data, head), [r["noise"][:, head] for r in data],
            reference, offsets[:, head], p["source_adaptation_positions"], p["source_test_positions"])
        mean = prior.mean.copy()
        mean[0] = plans[0].mean_offsets[head]
        priors.append(replace(prior, mean=mean,
            diagnostics={**prior.diagnostics, "public_mean_offset": float(mean[0]), "coefficient_mean": mean.tolist()}))
    rows += [{**r, "arm": "previous_free_intercept"} for r in old_source["rows"] if r["arm"] == "billing_selected"]
    summary = [{"head": h, "arm": arm, **{metric: float(np.mean([r[metric]
        for r in rows if r["head"] == h and r["arm"] == arm]))
        for metric in ("rmse", "mean_log_density", "standardized_residual_rms")}}
        for h in range(2) for arm in ("linked", "linked_pooled", "previous_free_intercept")]
    result = {"study_role": "source-law held-out conditional prediction, known public offsets removed",
        "target_labels_used": False, "support": p["support"], "rows": rows, "summary": summary,
        "all_source_priors": [{"mean": r.mean.tolist(), "covariance": r.covariance.tolist(),
            "discrepancy": r.discrepancy_variance, "diagnostics": r.diagnostics} for r in priors]}
    return priors, old_source, result


def run():
    start = time.process_time()
    p = json.loads((ROOT / "performance/billing_offset_replay_v1.json").read_text())
    archive, recorded, psi, reference, _, eta, eta_full = prepare_replay(p)
    priors, old_source, source = source_check(archive, reference, p)
    basis = OrthogonalInventoryBasis(reference)
    old_linked, old_full = [], []
    for head, prior in enumerate(priors):
        old = old_source["all_source_priors"][head]
        old_basis = InventoryFunctionBasis(reference, old["diagnostics"]["selected_indices"])
        old_linked.append(link_billing_coefficients(old_basis, np.asarray(old["covariance"])[1:, 1:],
            prior.discrepancy_variance, {"mode": "source_old_covariance_linked"}, prior.mean[0]))
        previous = SourceFunctionPrior(old_basis, np.asarray(old["mean"]), np.asarray(old["covariance"]),
                                       old["discrepancy"], old["diagnostics"])
        embedded = embed_inventory_prior(previous, basis, reference)
        full_covariance = np.diag(np.r_[embedded.covariance[0, 0],
            np.full(basis.feature_dim, old_source["source_signal_scales"][head] / basis.feature_dim)])
        old_full.append(replace(embedded, covariance=full_covariance,
                               diagnostics={**basis.diagnostics(), "mode": "same_source_orthogonal_full"}))
    sets = {"linked_billing_joint": priors, "linked_old_covariance_joint": old_linked,
            "linked_pooled_joint": [pooled_linked_prior(prior) for prior in priors],
            "linked_old_covariance_old_full_joint": old_linked}
    output = ROOT / p["output"]
    output.mkdir(parents=True, exist_ok=True)
    source["local_cpu_seconds"] = time.process_time() - start
    (output / "source_check.json").write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source_summary": source["summary"],
        "selected_residuals": [r.discrepancy_variance for r in priors],
        "source_cpu_seconds": source["local_cpu_seconds"]}), flush=True)
    rows, candidates = [], np.arange(len(psi))[:, None]
    for old in recorded["rows"]:
        if old["arm"] != p["recorded_path_arm"]:
            continue
        for arm, raw in sets.items():
            selected = [embed_inventory_prior(r, basis, reference) for r in raw]
            pooled = [embed_inventory_prior(pooled_linked_prior(r), basis, reference) for r in raw]
            pooled = [replace(r, diagnostics={**r.diagnostics, "mode": "same_source_linked_pooled"}) for r in pooled]
            if arm == "linked_old_covariance_old_full_joint":
                pooled = old_full
            if arm == "linked_pooled_joint":
                objective = pooled[0].target_model(psi)
                g_priors = [pooled[1]]
            else:
                objective = TransferMixtureGPR([selected[0], pooled[0]], psi, reference)
                g_priors = [selected[1], pooled[1]]
            joint = JointFunctionRiskPosterior(g_priors, [eta[2], eta_full[2]], psi)
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
                    point = covariance[indices, :, indices, :]
                    fm = objective.posterior_mean_many(candidates)
                    fc = posterior_covariances([objective], candidates)[..., 0]
                    utility, power = joint_verification_utility(fm, np.diag(fc), mean, point, p)
                    measurement = np.column_stack([np.full(len(psi), .12 ** 2 / p["replicates"]),
                        np.exp(mean[:, 1] + .5 * point[:, 1, 1]) / p["replicates"],
                        np.full(len(psi), float(polygamma(1, (p["replicates"] - 1) / 2)))])
                    kg = joint_verification_kg(fm, fc, mean, covariance, measurement, p)
                    choice = int(np.argmax(utility))
                    stages.append({"paid_points": step + 1, "recommendation": choice,
                        "predicted_utility": float(utility[choice]), "predicted_verifier_power": float(power[choice]),
                        "recommendation_g_eta_mean": mean[choice].tolist(),
                        "recommendation_g_eta_covariance": point[choice].tolist(),
                        "next_kg_probe": int(np.argmax(kg)), "max_kg": float(kg.max())})
            rows.append({"law_seed": old["law_seed"], "arm": arm,
                "recorded_policy_indices": [r["index"] for r in old["search_path"]],
                "label_rmse_after_initial": np.sqrt(np.mean(errors, axis=0)).tolist(),
                "mean_joint_log_predictive_density_after_initial": float(np.mean(scores)),
                "stages": stages, "joint_diagnostics": joint.diagnostics()})
    previous = json.loads((ROOT / p["baseline_replay"]).read_text())
    baseline = [r for r in previous["rows"] if r["arm"] in ("billing_joint", "billing_full_joint", "u1_joint_batch")]
    for row in baseline:
        current = next(r for r in rows if r["law_seed"] == row["law_seed"])
        if row["recorded_policy_indices"] != current["recorded_policy_indices"]:
            raise ValueError("Offset comparison does not share the previous paid history")
    rows += baseline
    summary = [{"arm": arm, "task_blocks": 8,
        "mean_label_rmse_after_initial": np.mean([r["label_rmse_after_initial"] for r in rows if r["arm"] == arm], axis=0).tolist(),
        "mean_joint_log_predictive_density_after_initial": float(np.mean([
            r["mean_joint_log_predictive_density_after_initial"] for r in rows if r["arm"] == arm]))}
        for arm in p["arms"] + ["billing_joint", "billing_full_joint", "u1_joint_batch"]]
    kg_pairs = []
    for seed in sorted({r["law_seed"] for r in rows}):
        task = {r["arm"]: r for r in rows if r["law_seed"] == seed}
        for left, right in zip(task["linked_billing_joint"]["stages"], task["linked_pooled_joint"]["stages"]):
            kg_pairs.append({"law_seed": seed, "paid_points": left["paid_points"],
                "source_kg_probe": left["next_kg_probe"], "pooled_kg_probe": right["next_kg_probe"],
                "kg_probe_changed": left["next_kg_probe"] != right["next_kg_probe"]})
    kg_result = {"study_role": "cached KG comparison; probes not executed", "rows": kg_pairs,
        "changes_by_stage": {str(k): sum(r["kg_probe_changed"] for r in kg_pairs if r["paid_points"] == k) for k in p["stages"]}}
    (output / "kg_pairing.json").write_text(json.dumps(kg_result, indent=2, allow_nan=False) + "\n")
    result = {"contract": p, "rows": rows, "summary": summary,
        "paired_kg_vs_pooled": kg_result,
        "source_priors": [embed_inventory_prior(r, basis, reference).diagnostics for r in priors],
        "target_truth_used": False, "independent_confirmation": False,
        "new_source_calls": 0, "new_target_calls": 0, "new_verification_calls": 0,
        "source_calls_in_reused_archive": archive["source_episode_calls"],
        "target_calls_in_reused_paths": 8 * 24 * p["replicates"],
        "local_cpu_seconds": time.process_time() - start}
    (output / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "local_cpu_seconds": result["local_cpu_seconds"]}, indent=2))
