"""Cached joint-likelihood attribution; preserve all paths and negative cases."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from core.verification_kg import objective_quality, particle_pass_probability, verification_utility
from performance.check_inventory_mean_interface import source_data
from problems.inventory_cost_plan import InventoryCostSimulator, sample_cost_law
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_mean_risk_prior import JointMeanRiskPosterior
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import SourceFunctionPrior


def run(p):
    start = time.process_time()
    saved = json.loads((ROOT / p["closed_loop_records"]).read_text())
    group = json.loads((ROOT / p["mean_priors"]).read_text())
    source, design, policies, plans, data = source_data(p["source_records"])
    protocol = saved["protocol"]
    reps = protocol["replicates"]
    source_psi, psi = data[0]["psi"], plans[0].coordinates_many(policies)
    means = []
    for spec in group["all_source_priors"]:
        assert spec["discrepancy_variance"] == 0., "Conditional coefficient model requires no policy discrepancy"
        means.append(SourceFunctionPrior(InventoryFunctionBasis(source_psi, spec["diagnostics"]["selected_indices"]),
            np.asarray(spec["mean"]), np.asarray(spec["covariance"]), 0., spec["diagnostics"]))
    Z = np.column_stack([np.ones(len(psi)), means[1].basis.transform(psi)])
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    Xs = np.asarray(design["risk_design"])
    risk = fit_source_risk_families([RiskBatchData.from_log_labels(Xs, t["response"][:, 2], reps) for t in data],
                                   np.vstack([Xs, X]), **source["protocol"]["source_risk_distribution"])
    pooled = PhysicalRiskPrior(risk.coefficients, np.full(len(risk.weights), 1. / len(risk.weights)), risk.families, risk.family_names, {})
    risk_variances = X @ risk.coefficients.T
    # Set the common difficulty levels from source observations before replay.
    default = [i for i, r in enumerate(design["assignments"]) if r["context"] == 0]
    source_minima = [float(np.min(t["response"][default, 1] + plans[0].cost_budget
                      + norm.ppf(.95) * np.sqrt(t["noise"][default, 1] * reps))) for t in data]
    levels = [1., *np.quantile(source_minima, [.25, .5]).tolist()]
    rows = []
    candidates = np.arange(len(policies))[:, None]
    for cell in saved["rows"]:
        if time.process_time() - start > p["local_cpu_limit_seconds"]:
            raise RuntimeError("Cached replay exceeded local CPU budget")
        f, g = [prior.target_model(psi) for prior in means]
        separate = risk.target_model()
        joint = {"repaired_joint": JointMeanRiskPosterior(means[1].mean, means[1].covariance, risk),
                 "repaired_joint_pooled": JointMeanRiskPosterior(means[1].mean, means[1].covariance, pooled)}
        observed, task_rows = [], []
        for step, record in enumerate(cell["search_path"], 1):
            index, y, noise = record["index"], np.asarray(record["response"]), record["noise"]
            observed.append(index)
            f.update([index], y[0], noise[0])
            g.update([index], y[1], noise[1])
            batch = RiskBatchData.from_log_labels(X[index:index+1], y[2:3], reps)
            separate.update_batch(batch)
            for model in joint.values():
                model.update_batch(Z[index:index+1], y[1:2], batch)
            if step not in p["checkpoints"]:
                continue
            f_mean, f_variance = f.posterior_mean_many(candidates), f.posterior_var_many(candidates)
            quality = objective_quality(f_mean[:, None], f_variance[:, None])
            for arm in p["arms"]:
                if arm == "repaired_mean_separate_risk":
                    g_mean, g_variance = g.posterior_mean_many(candidates), g.posterior_var_many(candidates)
                    eta_mean, eta_variance = separate.predict(X)
                    power = particle_pass_probability(g_mean, g_variance, risk_variances, np.exp(separate.log_weights), protocol)
                    cross_max = 0.
                else:
                    m, covariance = joint[arm].predict(Z, X, full_cov=True)
                    g_mean, eta_mean = m.T
                    diagonal = np.diag(covariance.reshape(len(policies) * 2, len(policies) * 2)).reshape(len(policies), 2)
                    g_variance, eta_variance = diagonal.T
                    power = joint[arm].verifier_probability(Z, X, protocol)
                    cross_max = float(np.max(np.abs(covariance[:, 0, :, 1])))
                utility = quality * power
                choice = int(np.argmax(utility))
                collapsed, _ = verification_utility(np.column_stack([f_mean, g_mean, eta_mean]),
                    np.column_stack([f_variance, g_variance, eta_variance]), protocol)
                gaussian_choice = int(np.argmax(collapsed))
                task_rows.append({"law_seed": cell["law_seed"], "old_path_arm": cell["arm"], "step": step, "arm": arm,
                    "choice": choice, "predicted_verifier_power": float(power[choice]),
                    "gaussian_independent_choice": gaussian_choice,
                    "joint_utility_gap_of_gaussian_choice": float(utility[choice] - utility[gaussian_choice]),
                    "max_absolute_mean_risk_cross_covariance": cross_max,
                    "g_mean": g_mean.tolist(), "eta_mean": eta_mean.tolist(), "observed_indices": list(observed)})
        # Private truth evaluates already-frozen recommendations only.
        world = InventoryCostSimulator(plans[0], sample_cost_law(cell["law_seed"]))
        truth = np.array([world.oracle_moments(x) for x in policies])
        reward, true_power = verification_utility(truth, np.zeros_like(truth), protocol)
        margin = truth[:, 1] + norm.ppf(1 - protocol["chance_alpha"]) * np.exp(.5 * truth[:, 2])
        for row in task_rows:
            held = np.ones(len(policies), dtype=bool)
            held[row.pop("observed_indices")] = False
            prediction_g, prediction_eta = np.asarray(row.pop("g_mean")), np.asarray(row.pop("eta_mean"))
            old_choice = cell["initial_choice"] if row["step"] == p["checkpoints"][0] else cell["choice"]
            row.update(held_candidate_g_rmse=float(np.sqrt(np.mean((prediction_g[held] - truth[held, 1]) ** 2))),
                held_candidate_eta_rmse=float(np.sqrt(np.mean((prediction_eta[held] - truth[held, 2]) ** 2))),
                true_expected_reward=float(reward[row["choice"]]), old_true_expected_reward=float(reward[old_choice]),
                true_verifier_power=float(true_power[row["choice"]]), true_margin=float(margin[row["choice"]]),
                true_feasible_candidates=cell["true_feasible_candidates"])
        rows.extend(task_rows)
        assert all(m.paid_episodes == reps * len(cell["search_path"]) for m in joint.values())
    summary = {str(step): {arm: {metric: float(np.mean([r[metric] for r in rows if r["step"] == step and r["arm"] == arm]))
        for metric in ("held_candidate_g_rmse", "held_candidate_eta_rmse", "true_expected_reward", "old_true_expected_reward",
                       "joint_utility_gap_of_gaussian_choice", "max_absolute_mean_risk_cross_covariance")}
        for arm in p["arms"]} for step in p["checkpoints"]}
    return {"protocol": p, "rows": rows, "summary": summary,
        "source_only_next_budget_levels": {"levels": levels, "source_task_minimum_estimates": source_minima,
            "source_calls_in_threshold_subset": len(default) * len(data) * reps, "target_truth_used": False},
        "new_episode_calls": 0, "independent_validation_calls": 0, "new_searches": 0,
        "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/joint_mean_risk_replay.json"
    if output.exists():
        print("Existing cached joint replay retained.")
        sys.exit(0)
    p = json.loads((ROOT / "performance/inventory_joint_mean_risk_replay_protocol.json").read_text())
    result = run(p)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": result["summary"], "budget_levels": result["source_only_next_budget_levels"]["levels"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}))
