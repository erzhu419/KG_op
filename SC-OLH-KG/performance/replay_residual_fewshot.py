"""Public-center source covariance and same-history few-shot attribution."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.replay_bounded_billing import evaluate, full_prefix, original_full_priors, prior_geometry
from performance.replay_orthogonal_meta_prior import prepare_replay
from problems.inventory_cost_plan import InventoryPlan, InventoryCostSimulator
from representation.bounded_billing import bounded_joint, bounded_objective
from representation.orthogonal_function_prior import fit_orthogonal_function_prior, same_source_orthogonal_full_prior
from representation.source_function_prior import head_batches
from representation.source_price_residuals import source_price_residual_covariance, pooled_residual_covariance


def fit_covariances(data, q, offsets, p):
    covariance_pairs, diagnostics = [], []
    for head, weights in enumerate((InventoryCostSimulator.objective_weights, InventoryCostSimulator.constraint_weights)):
        covariance, info = source_price_residual_covariance(q * weights,
            [r["response"][:, head] for r in data], [r["noise"][:, head] for r in data], offsets[:, head],
            p[("objective_price_bounds", "constraint_price_bounds")[head]])
        covariance_pairs.append((covariance, pooled_residual_covariance(covariance)))
        diagnostics.append(info)
    return covariance_pairs, diagnostics


def source_validation(archive, reference, eta_basis, p):
    plans = [InventoryPlan(**c) for c in archive["contexts"]]
    policies = np.asarray(archive["public_policies"])
    psi = np.array([plans[a["context"]].trajectory(policies[a["policy_index"]]).coordinates(plans[a["context"]].capacity)
                    for a in archive["assignments"]])
    offsets = np.array([plans[a["context"]].mean_offsets for a in archive["assignments"]])
    data = [{"psi": psi, "response": np.asarray(r["response"]), "noise": np.asarray(r["noise"])} for r in archive["source_records"]]
    q = psi[:, [3, 5, 4, 6]]
    rows = []
    for held, record in enumerate(data):
        training = [r for i, r in enumerate(data) if i != held]
        covariances, _ = fit_covariances(training, q, offsets, p)
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
                    result = bounded_objective(q * weights, offsets[:, head], observations, covariance, bounds, p["refinement_order"])
                    mean, variance = result["mean"], np.diag(result["covariance"])
                else:
                    result = bounded_joint(q * weights, offsets[:, head], observations, covariance, bounds, eta_priors, psi,
                                          p["refinement_order"], p["eta_quadrature_order"])
                    mean, variance = result["mean"][:, 0], np.diag(result["covariance"][:, 0, :, 0])
                test = p["source_test_positions"]
                error = record["response"][test, head] - mean[test]
                variance = variance[test] + record["noise"][test, head]
                rows.append({"held_source": held, "arm": arm, "head": head,
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "working_gaussian_log_density": float(np.mean(norm.logpdf(error, scale=np.sqrt(variance)))),
                    "standardized_residual_rms": float(np.sqrt(np.mean(error ** 2 / variance)))})
    covariances, diagnostics = fit_covariances(data, q, offsets, p)
    summary = [{"arm": arm, "head": head, **{key: float(np.mean([r[key] for r in rows if r["arm"] == arm and r["head"] == head]))
        for key in ("rmse", "working_gaussian_log_density", "standardized_residual_rms")}}
        for arm in p["arms"][:3] for head in range(2)]
    result = {"study_role": "outer held-source mean prediction, held source excluded from price and eta fitting",
        "target_labels_used": False, "summary": summary, "rows": rows,
        "price_prior_covariances": [[c.tolist() for c in pair] for pair in covariances],
        "source_price_fit": diagnostics, "public_box_prior_geometry": prior_geometry(covariances, p)}
    return covariances, result


def run():
    start = time.process_time()
    p = json.loads((ROOT / "performance/residual_fewshot_replay_v1.json").read_text())
    archive, recorded, psi, reference, eta_basis, priors, full = prepare_replay(p)
    output = ROOT / p["output"]
    output.mkdir(parents=True, exist_ok=True)
    covariances, source = source_validation(archive, reference, eta_basis, p)
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
    result = {"contract": p, "rows": rows, "numerical_refinement": numerical,
        "numerical_gate_passed": all(r["gate_passed"] for r in numerical),
        "target_truth_used": False, "independent_confirmation": False,
        "new_source_calls": 0, "new_target_calls": 0, "new_verification_calls": 0,
        "source_calls_in_reused_archive": archive["source_episode_calls"],
        "target_calls_in_reused_prefixes": len({r["law_seed"] for r in rows}) * max(p["stages"]) * p["replicates"],
        "local_cpu_seconds": time.process_time() - start}
    (output / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"numerical_gate_passed": result["numerical_gate_passed"], "cpu_seconds": result["local_cpu_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    run()
