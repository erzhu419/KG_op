"""One fixed distribution-complexity repair; reuse every paid record."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from core.verification_kg import posterior_covariances, verification_kg_from_moments
from problems.inventory_cost_plan import InventoryPlan
from representation.physical_risk_prior import RiskBatchData, fit_source_risk_families
from representation.source_function_prior import SourceFunctionPrior, head_batches


def run(p):
    start = time.process_time()
    primary = json.loads((ROOT / p["primary"]).read_text())
    protocol = primary["protocol"]
    old = json.loads((ROOT / protocol["existing_records"]).read_text())
    design = json.loads((ROOT / protocol["source_design"]).read_text())
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    policies = np.asarray(old["public_policies"])
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    X_source = np.asarray(design["risk_design"])
    reps = old["protocol"]["replicates"]
    tasks = [RiskBatchData.from_log_labels(X_source, np.asarray(task["response"])[:, 2], reps)
             for task in primary["source_records"]]
    reference = np.vstack([X_source, X])
    config = protocol["source_risk_distribution"]
    prior = fit_source_risk_families(tasks, reference, **config)
    holdout = []
    for t in range(len(tasks)):
        fold = fit_source_risk_families([task for j, task in enumerate(tasks) if j != t], reference, **config)
        logp = fold.target_model().log_predictive_density(tasks[t])
        holdout.append({"task": t, "per_label_log_density_gain":
            (logp - primary["source_task_holdout"][t]["pooled_log_density"]) / len(X_source),
            "optimizer_converged": fold.diagnostics["source_optimizer_converged"]})
    initial = old["target_policy_indices"]
    y, noise = np.asarray(old["target_response"]), np.asarray(old["target_noise"])
    posterior = prior.target_model()
    before = posterior.diagnostics()
    posterior.update_batch(RiskBatchData.from_log_labels(X[initial], y[:, 2], reps))
    mean, covariance = posterior.predict(X, full_cov=True)
    source_psi = np.array([plans[row["context"]].trajectory(policies[row["policy_index"]]).coordinates(
        plans[row["context"]].capacity) for row in design["assignments"]])
    source = [{"psi": source_psi, "response": np.asarray(task["response"]), "noise": np.asarray(task["noise"])}
              for task in primary["source_records"]]
    target_psi, candidates = plans[0].coordinates_many(policies), np.arange(len(policies))[:, None]
    means = []
    for h in range(2):
        m = SourceFunctionPrior.fit(head_batches(source, h), [task["noise"][:, h] for task in source],
                                    target_psi, "fixed_source").target_model(target_psi)
        for index, label, variance in zip(initial, y[:, h], noise[:, h]):
            m.update([index], label, variance)
        means.append(m)
    joint_mean = np.column_stack([*[m.posterior_mean_many(candidates) for m in means], mean])
    joint_cov = np.concatenate([posterior_covariances(means, candidates), covariance[..., None]], axis=2)
    measurement_noise = np.column_stack([np.full(len(X), .12 ** 2 / reps),
        np.exp(mean + .5 * np.diag(covariance)) / reps, np.full(len(X), polygamma(1, (reps - 1) / 2))])
    scores = verification_kg_from_moments(joint_mean, joint_cov, measurement_noise, old["protocol"])
    # Evaluate only after fitting; these private coefficients are not inputs
    # to any prior, update, source validation, or acquisition call.
    truth_beta = np.asarray(primary["target_oracle_coefficients_diagnostic_only"])
    truth_eta = np.log(X @ truth_beta)
    mask = np.ones(len(X), dtype=bool)
    mask[initial] = False
    beta = np.asarray(posterior.diagnostics()["coefficient_mean"])
    return {"protocol": p, "source_info": prior.diagnostics, "source_task_holdout": holdout,
        "source_holdout_mean_per_label_gain": float(np.mean([r["per_label_log_density_gain"] for r in holdout])),
        "before_target": before, "after_target": posterior.diagnostics(),
        "held_out_eta_rmse": float(np.sqrt(np.mean((mean[mask] - truth_eta[mask]) ** 2))),
        "held_out_average_log_risk_variance": float(np.diag(covariance)[mask].mean()),
        "component_rmse": {name: float(np.sqrt(np.mean((X[:, block] @ (beta[block] - truth_beta[block])) ** 2)))
            for name, block in (("floor", slice(0, 1)), ("local", slice(1, 4)),
                                ("shared", slice(4, 7)), ("volume", slice(7, 9)))},
        "finite_moments_and_kg": bool(np.isfinite(scores).all() and np.isfinite(covariance).all()),
        "new_episode_calls": 0, "decision_performance_established": False,
        "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    p = json.loads((ROOT / "performance/inventory_physical_risk_family_check_protocol.json").read_text())
    result = run(p)
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/family_check.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("protocol", "source_info")}, allow_nan=False))
