"""Frozen physical-risk learning check with existing target labels reused."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector, params_to_beta
from core.verification_kg import posterior_covariances, verification_kg_from_moments
from problems.inventory_cost_plan import InventoryPlan, InventoryCostSimulator, sample_cost_law
from representation.physical_risk_prior import RiskBatchData, fit_source_physical_risk
from representation.source_function_prior import SourceFunctionPrior, head_batches


def run(p):
    start = time.process_time()
    old = json.loads((ROOT / p["existing_records"]).read_text())
    design = json.loads((ROOT / p["source_design"]).read_text())
    original = old["protocol"]
    policies = np.asarray(old["public_policies"])
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    source_psi = np.array([plans[row["context"]].trajectory(policies[row["policy_index"]]).coordinates(
                           plans[row["context"]].capacity) for row in design["assignments"]])
    X_source = np.asarray(design["risk_design"])
    assert [row["policy_index"] for row in design["assignments"]] == old["source_policy_indices"]
    assert design["contexts"][0] == json.loads(json.dumps(original["context"]))
    source, new_calls = [], 0
    for task in range(original["source_tasks"]):
        law = sample_cost_law(original["source_law_seed"] + task)
        rng = np.random.default_rng(p["source_noise_seed_for_new_contexts"] + task)
        response = [np.asarray(y) for y in old["source_records"][task]["response"][:4]]
        noise = [np.asarray(v) for v in old["source_records"][task]["noise"][:4]]
        for row in design["assignments"][4:]:
            world = InventoryCostSimulator(plans[row["context"]], law)
            y, v = world.observe(policies[row["policy_index"]], original["replicates"], rng)
            response.append(y)
            noise.append(v)
            new_calls += world.episode_calls
        source.append({"psi": source_psi, "response": np.asarray(response), "noise": np.asarray(noise)})
    tasks = [RiskBatchData.from_log_labels(X_source, task["response"][:, 2], original["replicates"])
             for task in source]
    X_target = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    reference = np.vstack([X_source, X_target])
    learned, pooled = fit_source_physical_risk(tasks, reference, **p["source_risk_distribution"])
    holdout = []
    for left_out in range(len(tasks)):
        keep = [task for i, task in enumerate(tasks) if i != left_out]
        train_prior, control_prior = fit_source_physical_risk(keep, reference, **p["source_risk_distribution"])
        logp = [prior.target_model().log_predictive_density(tasks[left_out]) for prior in (train_prior, control_prior)]
        holdout.append({"task": left_out, "learned_log_density": logp[0], "pooled_log_density": logp[1],
            "per_label_log_density_gain": (logp[0] - logp[1]) / len(X_source),
            "optimizer_converged": train_prior.diagnostics["source_optimizer_converged"],
            "optimizer_iterations": train_prior.diagnostics["source_optimizer_iterations"]})
    psi_target = plans[0].coordinates_many(policies)
    initial = old["target_policy_indices"]
    y, noise = np.asarray(old["target_response"]), np.asarray(old["target_noise"])
    candidates = np.arange(len(policies))[:, None]
    means = []
    for head in range(2):
        prior = SourceFunctionPrior.fit(head_batches(source, head), [task["noise"][:, head] for task in source],
                                        psi_target, "fixed_source")
        model = prior.target_model(psi_target)
        for index, label, variance in zip(initial, y[:, head], noise[:, head]):
            model.update([index], label, variance)
        means.append(model)
    m = np.column_stack([model.posterior_mean_many(candidates) for model in means])
    covariance = posterior_covariances(means, candidates)
    target_task = RiskBatchData.from_log_labels(X_target[initial], y[:, 2], original["replicates"])
    posteriors = []
    results = {}
    # Oracle evaluation comes after every prior fit and paid posterior update.
    # It never supplies coefficients, active groups, or responses to a learner.
    for name, prior in (("source_learned_physical", learned), ("same_source_pooled_physical", pooled)):
        posterior = prior.target_model()
        before = posterior.diagnostics()
        posterior.update_batch(target_task)
        risk_mean, risk_covariance = posterior.predict(X_target, full_cov=True)
        mean = np.column_stack([m, risk_mean])
        cov = np.concatenate([covariance, risk_covariance[..., None]], axis=2)
        variance = np.einsum('iih->ih', cov)
        fantasy_noise = np.column_stack([np.full(len(policies), .12 ** 2 / original["replicates"]),
            np.exp(risk_mean + .5 * variance[:, 2]) / original["replicates"],
            np.full(len(policies), polygamma(1, (original["replicates"] - 1) / 2))])
        scores = verification_kg_from_moments(mean, cov, fantasy_noise, original)
        results[name] = {"before_target": before, "after_target": posterior.diagnostics(),
            "finite_moments_and_kg": bool(np.isfinite(scores).all() and np.isfinite(cov).all()),
            "moment_gaussian_kg_min": float(scores.min()), "moment_gaussian_kg_max": float(scores.max())}
        posteriors.append(posterior)
    diagnostic_world = InventoryCostSimulator(plans[0], sample_cost_law(original["target_law_seed"]))
    truth = np.array([diagnostic_world.oracle_moments(x) for x in policies])
    true_coefficients = params_to_beta(diagnostic_world._law.risk)
    held_out = np.ones(len(policies), dtype=bool)
    held_out[initial] = False
    for name, posterior in zip(results, posteriors):
        predicted_eta, predicted_variance = posterior.predict(X_target)
        beta = np.asarray(posterior.diagnostics()["coefficient_mean"])
        results[name].update(held_out_eta_rmse=float(np.sqrt(np.mean((predicted_eta[held_out] - truth[held_out, 2]) ** 2))),
            held_out_average_log_risk_variance=float(predicted_variance[held_out].mean()),
            component_rmse={label: float(np.sqrt(np.mean((X_target[:, block] @ (beta[block] - true_coefficients[block])) ** 2)))
                for label, block in (("floor", slice(0, 1)), ("local", slice(1, 4)),
                                     ("shared", slice(4, 7)), ("volume", slice(7, 9)))})
    assert new_calls == p["new_source_episode_calls"]
    assert diagnostic_world.episode_calls == 0
    return {"protocol": p, "new_source_episode_calls": new_calls, "reused_source_episode_calls": p["reused_source_episode_calls"],
        "new_target_episode_calls": 0, "reused_target_episode_calls": p["reused_target_episode_calls"],
        "verification_calls": 0, "source_info": learned.diagnostics, "source_task_holdout": holdout,
        "source_holdout_mean_per_label_gain": float(np.mean([row["per_label_log_density_gain"] for row in holdout])),
        "target_results": results, "target_oracle_coefficients_diagnostic_only": true_coefficients.tolist(),
        "source_records": [{"response": task["response"].tolist(), "noise": task["noise"].tolist()} for task in source],
        "decision_performance_established": False, "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    protocol = json.loads((ROOT / "performance/inventory_physical_risk_protocol.json").read_text())
    result = run(protocol)
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("protocol", "source_records", "source_info")}, allow_nan=False))
