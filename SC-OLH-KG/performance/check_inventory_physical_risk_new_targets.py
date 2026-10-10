"""Eight frozen new tasks; shared initial observations, no search or verifier."""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector, params_to_beta
from problems.inventory_cost_plan import InventoryPlan, InventoryCostSimulator, sample_cost_law
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families


def run(p):
    start = time.process_time()
    source = json.loads((ROOT / p["source_records"]).read_text())
    source_protocol = source["protocol"]
    old = json.loads((ROOT / source_protocol["existing_records"]).read_text())
    design = json.loads((ROOT / source_protocol["source_design"]).read_text())
    policies, initial = np.asarray(old["public_policies"]), old["target_policy_indices"]
    plan = InventoryPlan(**old["protocol"]["context"])
    X = np.array([cumulative_feature_vector(plan.trajectory(x).risk_exposures()) for x in policies])
    X_source = np.asarray(design["risk_design"])
    tasks = [RiskBatchData.from_log_labels(X_source, np.asarray(task["response"])[:, 2], p["replicates"])
             for task in source["source_records"]]
    learned = fit_source_risk_families(tasks, np.vstack([X_source, X]), **source_protocol["source_risk_distribution"])
    pooled = PhysicalRiskPrior(learned.coefficients, np.full(len(learned.weights), 1. / len(learned.weights)),
        learned.families, learned.family_names, {"mode": "same_source_pooled_scale"})
    assert learned.diagnostics["source_optimizer_converged"]
    priors = {"factor_family": learned, "pooled": pooled}
    frozen_weights = {name: prior.weights.copy() for name, prior in priors.items()}
    mask = np.ones(len(policies), dtype=bool)
    mask[initial] = False
    rows, new_calls = [], 0
    for law_seed, noise_seed in zip(p["target_law_seeds"], p["target_noise_seeds"]):
        world = InventoryCostSimulator(plan, sample_cost_law(law_seed))
        rng = np.random.default_rng(noise_seed)
        labels = [world.observe(policies[index], p["replicates"], rng) for index in initial]
        response, noise = np.array([y for y, _ in labels]), np.array([v for _, v in labels])
        evidence = RiskBatchData.from_log_labels(X[initial], response[:, 2], p["replicates"])
        arms = {}
        for name, prior in priors.items():
            posterior = prior.target_model()
            posterior.update_batch(evidence)
            mean, variance = posterior.predict(X)
            assert posterior.paid_episodes == world.episode_calls == len(initial) * p["replicates"]
            assert np.array_equal(prior.weights, frozen_weights[name])
            arms[name] = {"mean": mean, "variance": variance, "diagnostics": posterior.diagnostics()}
        # Private truth is first evaluated after both models finish updating.
        true_beta = params_to_beta(world._law.risk)
        true_eta = np.log(X @ true_beta)
        for name, answer in arms.items():
            mean, variance = answer.pop("mean"), answer.pop("variance")
            beta = np.asarray(answer["diagnostics"]["coefficient_mean"])
            answer.update(held_out_eta_rmse=float(np.sqrt(np.mean((mean[mask] - true_eta[mask]) ** 2))),
                held_out_average_log_risk_variance=float(variance[mask].mean()),
                component_rmse={label: float(np.sqrt(np.mean((X[mask, block] @ (beta[block] - true_beta[block])) ** 2)))
                    for label, block in (("floor", slice(0, 1)), ("local", slice(1, 4)),
                                         ("shared", slice(4, 7)), ("volume", slice(7, 9)))})
        gain = arms["pooled"]["held_out_eta_rmse"] - arms["factor_family"]["held_out_eta_rmse"]
        rows.append({"law_seed": law_seed, "noise_seed": noise_seed,
            "initial_policy_indices": initial, "target_episode_calls": world.episode_calls,
            "response": response.tolist(), "noise": noise.tolist(), "arms": arms,
            "paired_rmse_gain": gain, "true_coefficients_diagnostic_only": true_beta.tolist()})
        new_calls += world.episode_calls
    assert len(rows) == len(p["target_law_seeds"]) == len(p["target_noise_seeds"])
    assert new_calls == p["new_target_episode_calls"]
    return {"protocol": p, "rows": rows, "source_info": learned.diagnostics,
        "summary": {"tasks": len(rows), "factor_family_wins": sum(r["paired_rmse_gain"] > 0 for r in rows),
            "mean_paired_rmse_gain": float(np.mean([r["paired_rmse_gain"] for r in rows])),
            "rmse_gain_range": [float(min(r["paired_rmse_gain"] for r in rows)), float(max(r["paired_rmse_gain"] for r in rows))],
            "mean_eta_rmse": {name: float(np.mean([r["arms"][name]["held_out_eta_rmse"] for r in rows])) for name in priors},
            "mean_log_risk_variance": {name: float(np.mean([r["arms"][name]["held_out_average_log_risk_variance"] for r in rows])) for name in priors},
            "mean_component_rmse": {name: {component: float(np.mean([r["arms"][name]["component_rmse"][component] for r in rows]))
                for component in ("floor", "local", "shared", "volume")} for name in priors}},
        "new_source_episode_calls": 0, "new_target_episode_calls": new_calls,
        "source_archive_episode_calls_per_prior": p["source_archive_episode_calls_per_prior"],
        "decision_performance_established": False, "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    p = json.loads((ROOT / "performance/inventory_physical_risk_new_targets_protocol.json").read_text())
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/new_targets.json"
    if output.exists():
        print("Existing frozen results retained; read new_targets.json instead of regenerating observations.")
        sys.exit(0)
    result = run(p)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": result["summary"], "new_target_episode_calls": result["new_target_episode_calls"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}, allow_nan=False))
