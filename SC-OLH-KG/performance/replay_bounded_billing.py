"""Bounded pricing: source validation, then already-paid development prefixes."""

from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma
from scipy.stats import multivariate_normal, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.joint_verification_kg import joint_verification_kg, joint_verification_utility
from core.verification_kg import posterior_covariances
from performance.replay_orthogonal_meta_prior import prepare_replay
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan
from representation.billing_offset_prior import source_price_moments
from representation.bounded_billing import bounded_joint, bounded_objective, mix_moments
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_function_prior import fit_orthogonal_function_prior, same_source_orthogonal_full_prior
from representation.orthogonal_inventory_prior import OrthogonalInventoryBasis, embed_inventory_prior
from representation.source_function_prior import SourceFunctionPrior, head_batches


def price_covariances(data, basis, offsets):
    out = []
    for h, weights in enumerate((InventoryCostSimulator.objective_weights, InventoryCostSimulator.constraint_weights)):
        beta_cov = source_price_moments(head_batches(data, h), [r["noise"][:, h] for r in data], basis, offsets[:, h])
        transform = basis.scale * weights
        out.append((beta_cov / np.outer(transform, transform),
            np.eye(4) * np.trace(beta_cov) / 4 / np.outer(transform, transform)))
    return out


def prior_geometry(covariances, p):
    rows = []
    for head, key in enumerate(("objective_price_bounds", "constraint_price_bounds")):
        bounds = p[key]
        center = np.full(4, sum(bounds) / 2)
        points = np.array(list(itertools.product(bounds, repeat=4)) + [center.tolist()])
        covariance = covariances[head][0]
        density = multivariate_normal.logpdf(points, center, covariance)
        rows.append({"head": head, "price_covariance_diagonal": np.diag(covariance).tolist(),
            "corner_and_midpoint_density_ratio": float(np.exp(density.max() - density.min()))})
    return rows


def source_validation(archive, reference, eta_basis, p):
    plans = [InventoryPlan(**c) for c in archive["contexts"]]
    policies = np.asarray(archive["public_policies"])
    psi = np.array([plans[a["context"]].trajectory(policies[a["policy_index"]]).coordinates(plans[a["context"]].capacity)
                    for a in archive["assignments"]])
    offsets = np.array([plans[a["context"]].mean_offsets for a in archive["assignments"]])
    data = [{"psi": psi, "response": np.asarray(r["response"]), "noise": np.asarray(r["noise"])} for r in archive["source_records"]]
    basis = InventoryFunctionBasis(reference, InventoryFunctionBasis.billing_indices)
    q = psi[:, [3, 5, 4, 6]]
    rows = []
    for held, record in enumerate(data):
        training = [r for i, r in enumerate(data) if i != held]
        covariances = price_covariances(training, basis, offsets)
        eta = fit_orthogonal_function_prior(head_batches(training, 2), [r["noise"][:, 2] for r in training], eta_basis)
        eta_priors = [eta, same_source_orthogonal_full_prior(eta)]
        observations = [{"index": i, "response": record["response"][i].tolist(), "noise": record["noise"][i].tolist(),
                         "replicates": p["replicates"]} for i in p["source_adaptation_positions"]]
        for arm, index in (("bounded_source", 0), ("bounded_pooled_prices", 1), ("bounded_uniform_prices", None)):
            for head in range(2):
                covariance = None if index is None else covariances[head][index]
                weights = (InventoryCostSimulator.objective_weights, InventoryCostSimulator.constraint_weights)[head]
                bounds = p[("objective_price_bounds", "constraint_price_bounds")[head]]
                if head == 0:
                    result = bounded_objective(q * weights, offsets[:, head], observations, covariance, bounds, p["price_quadrature_order"])
                    mean, variance = result["mean"], np.diag(result["covariance"])
                else:
                    result = bounded_joint(q * weights, offsets[:, head], observations, covariance, bounds, eta_priors, psi,
                        p["price_quadrature_order"], p["eta_quadrature_order"])
                    mean, variance = result["mean"][:, 0], np.diag(result["covariance"][:, 0, :, 0])
                test = p["source_test_positions"]
                error = record["response"][test, head] - mean[test]
                variance = variance[test] + record["noise"][test, head]
                rows.append({"held_source": held, "arm": arm, "head": head,
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "working_gaussian_log_density": float(np.mean(norm.logpdf(error, scale=np.sqrt(variance)))),
                    "standardized_residual_rms": float(np.sqrt(np.mean(error ** 2 / variance)))})
    summary = [{"arm": arm, "head": h, **{key: float(np.mean([r[key] for r in rows if r["arm"] == arm and r["head"] == h]))
        for key in ("rmse", "working_gaussian_log_density", "standardized_residual_rms")}}
        for arm in ("bounded_source", "bounded_pooled_prices", "bounded_uniform_prices") for h in range(2)]
    return price_covariances(data, basis, offsets), {"study_role": "outer source-law held-out prediction; four billing support and zero discrepancy previously frozen",
        "target_labels_used": False, "rows": rows, "summary": summary,
        "score": "working Gaussian mean density, matching earlier mean prediction scores; eta fitted without the held source law"}


def original_full_priors(reference, p):
    old = json.loads((ROOT / p["baseline_source"]).read_text())
    basis = OrthogonalInventoryBasis(reference)
    priors = []
    for h in range(2):
        saved = old["all_source_priors"][h]
        raw = SourceFunctionPrior(InventoryFunctionBasis(reference, saved["diagnostics"]["selected_indices"]),
            np.asarray(saved["mean"]), np.asarray(saved["covariance"]), saved["discrepancy"], saved["diagnostics"])
        embedded = embed_inventory_prior(raw, basis, reference)
        priors.append(replace(embedded, covariance=np.diag(np.r_[embedded.covariance[0, 0],
            np.full(basis.feature_dim, old["source_signal_scales"][h] / basis.feature_dim)]),
            diagnostics={"mode": "same_source_orthogonal_full"}))
    return priors


def full_prefix(priors, eta_priors, psi, observations):
    objective = priors[0].target_model(psi)
    joint = JointFunctionRiskPosterior([priors[1]], eta_priors, psi)
    f_evidence = g_evidence = 0.
    for o in observations:
        i, y, v = [o["index"]], o["response"], o["noise"]
        f_evidence += norm.logpdf(y[0], objective.posterior_mean(i), np.sqrt(objective.posterior_var(i) + v[0]))
        objective.update(i, y[0], v[0])
        g_evidence += joint.update_batch(o["index"], y[1], y[2], o["replicates"])
    candidates = np.arange(len(psi))[:, None]
    mean, covariance = joint.predict(full_cov=True)
    return {"mean": objective.posterior_mean_many(candidates), "covariance": posterior_covariances([objective], candidates)[..., 0],
            "log_evidence": float(f_evidence)}, {"mean": mean, "covariance": covariance, "log_evidence": float(g_evidence)}


def recommendation(objective, joint, p):
    mean, covariance = joint["mean"], joint["covariance"]
    indices = np.arange(len(mean))
    point = covariance[indices, :, indices, :]
    utility, power = joint_verification_utility(objective["mean"], np.diag(objective["covariance"]), mean, point, p)
    choice = int(np.argmax(utility))
    return {"recommendation": choice, "predicted_utility": float(utility[choice]),
        "predicted_verifier_power": float(power[choice]), "recommendation_g_eta_mean": mean[choice].tolist(),
        "recommendation_g_eta_covariance": point[choice].tolist()}, utility


def evaluate(psi, q, offsets, observations, covariances, eta, eta_full, fallback, p, arm, order, kg):
    covariance_index = {"bounded_source": 0, "bounded_pooled_prices": 1}.get(arm)
    fc, gc = (None, None) if covariance_index is None else [c[covariance_index] for c in covariances]
    eta_priors = [eta_full] if arm == "bounded_uniform_pooled_risk" else [eta, eta_full]
    f = bounded_objective(q * InventoryCostSimulator.objective_weights, offsets[:, 0], observations,
                         fc, p["objective_price_bounds"], order)
    g = bounded_joint(q * InventoryCostSimulator.constraint_weights, offsets[:, 1], observations,
                      gc, p["constraint_price_bounds"], eta_priors, psi, order, p["eta_quadrature_order"])
    fm, gm = [mix_moments([{**bounded, "log_evidence": bounded["log_evidence"] - np.log(2)},
        {**full, "log_evidence": full["log_evidence"] - np.log(2)}]) for bounded, full in zip((f, g), fallback)]
    stage, utility = recommendation(fm, gm, p)
    physical, _ = recommendation(f, g, p)
    stage.update({"paid_points": len(observations), "physical_only": physical,
        "physical_component_weights": [fm["component_weights"][0], gm["component_weights"][0]],
        "physical_price_means": [f["price_mean"].tolist(), g["price_mean"].tolist()],
        "physical_price_sds": [np.sqrt(np.diag(f["price_covariance"])).tolist(), np.sqrt(np.diag(g["price_covariance"])).tolist()],
        "price_quadrature_order": order})
    if kg:
        mean, covariance = gm["mean"], gm["covariance"]
        indices = np.arange(len(mean))
        point = covariance[indices, :, indices, :]
        measurement = np.column_stack([np.full(len(psi), .12 ** 2 / p["replicates"]),
            np.exp(mean[:, 1] + .5 * point[:, 1, 1]) / p["replicates"],
            np.full(len(psi), polygamma(1, (p["replicates"] - 1) / 2))])
        scores = joint_verification_kg(fm["mean"], fm["covariance"], mean, covariance, measurement, p)
        stage.update(next_kg_probe=int(np.argmax(scores)), max_kg=float(scores.max()))
    return stage, utility


def run():
    start = time.process_time()
    p = json.loads((ROOT / "performance/bounded_billing_replay_v1.json").read_text())
    archive, recorded, psi, reference, eta_basis, priors, full = prepare_replay(p)
    output = ROOT / p["output"]
    output.mkdir(parents=True, exist_ok=True)
    covariances, source = source_validation(archive, reference, eta_basis, p)
    source["price_prior_covariances"] = [[c.tolist() for c in pair] for pair in covariances]
    source["public_box_prior_geometry"] = prior_geometry(covariances, p)
    source["local_cpu_seconds"] = time.process_time() - start
    (output / "source_check.json").write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source_summary": source["summary"], "cpu_seconds": source["local_cpu_seconds"]}), flush=True)
    q = psi[:, [3, 5, 4, 6]]
    plan = InventoryPlan(**archive["contexts"][0])
    offsets = np.broadcast_to(plan.mean_offsets, (len(psi), 2))
    mean_full = original_full_priors(reference, p)
    rows, numerical = [], []
    for recorded_row in recorded["rows"]:
        if recorded_row["arm"] != p["recorded_path_arm"]:
            continue
        stages = {a: [] for a in p["arms"]}
        for points in p["stages"]:
            observations = [{**r, "replicates": p["replicates"]} for r in recorded_row["search_path"][:points]]
            fallback = full_prefix(mean_full, [priors[2], full[2]], psi, observations)
            pooled_fallback = full_prefix(mean_full, [full[2]], psi, observations)
            for arm in p["arms"]:
                baseline = pooled_fallback if arm == "bounded_uniform_pooled_risk" else fallback
                low, low_utility = evaluate(psi, q, offsets, observations, covariances, priors[2], full[2],
                    baseline, p, arm, p["price_quadrature_order"], False)
                high, high_utility = evaluate(psi, q, offsets, observations, covariances, priors[2], full[2],
                    baseline, p, arm, p["refinement_order"], True)
                check = {"law_seed": recorded_row["law_seed"], "arm": arm, "paid_points": points,
                    "recommendation_equal": low["recommendation"] == high["recommendation"],
                    "maximum_utility_difference": abs(low["predicted_utility"] - high["predicted_utility"]),
                    "all_candidate_maximum_utility_difference": float(np.max(np.abs(low_utility - high_utility))),
                    "physical_recommendation_equal": low["physical_only"]["recommendation"] == high["physical_only"]["recommendation"]}
                check["gate_passed"] = check["recommendation_equal"] and check["maximum_utility_difference"] <= .003
                numerical.append(check)
                stages[arm].append(high)
        rows += [{"law_seed": recorded_row["law_seed"], "arm": arm,
            "recorded_policy_indices": [r["index"] for r in recorded_row["search_path"]], "stages": stages[arm]} for arm in p["arms"]]
        print(json.dumps({"law_seed": recorded_row["law_seed"], "cpu_seconds": time.process_time() - start,
            "recommendations": {a: [r["recommendation"] for r in stages[a]] for a in p["arms"]}}), flush=True)
    previous = json.loads((ROOT / p["baseline_replay"]).read_text())
    baselines = [r for r in previous["rows"] if r["arm"] in ("linked_old_covariance_old_full_joint", "linked_billing_joint", "billing_full_joint", "u1_joint_batch")]
    for row in baselines:
        paired = next(r for r in rows if r["law_seed"] == row["law_seed"])
        if row["recorded_policy_indices"] != paired["recorded_policy_indices"]:
            raise ValueError("Bounded replay does not share the earlier paid history")
    result = {"contract": p, "rows": rows + baselines, "numerical_refinement": numerical,
        "numerical_gate_passed": all(r["gate_passed"] for r in numerical),
        "target_truth_used": False, "independent_confirmation": False,
        "new_source_calls": 0, "new_target_calls": 0, "new_verification_calls": 0,
        "source_calls_in_reused_archive": archive["source_episode_calls"],
        "target_calls_in_reused_paths": len({r["law_seed"] for r in rows}) * 24 * p["replicates"],
        "local_cpu_seconds": time.process_time() - start}
    (output / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"numerical_gate_passed": result["numerical_gate_passed"], "cpu_seconds": result["local_cpu_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    run()
