"""Cached recommendation screen; no observations and no method selection."""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from core.verification_kg import objective_quality, particle_pass_probability, verification_utility
from problems.inventory_cost_plan import InventoryPlan, InventoryCostSimulator, sample_cost_law
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import SourceFunctionPrior, head_batches


def run():
    start = time.process_time()
    folder = ROOT / "paper_artifacts/inventory_physical_risk_20261004"
    new = json.loads((folder / "new_targets.json").read_text())
    source = json.loads((ROOT / new["protocol"]["source_records"]).read_text())
    source_p = source["protocol"]
    old = json.loads((ROOT / source_p["existing_records"]).read_text())
    design = json.loads((ROOT / source_p["source_design"]).read_text())
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    policies, initial = np.asarray(old["public_policies"]), old["target_policy_indices"]
    psi = plans[0].coordinates_many(policies)
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    X_source = np.asarray(design["risk_design"])
    reps = new["protocol"]["replicates"]
    risk_tasks = [RiskBatchData.from_log_labels(X_source, np.asarray(task["response"])[:, 2], reps) for task in source["source_records"]]
    prior = fit_source_risk_families(risk_tasks, np.vstack([X_source, X]), **source_p["source_risk_distribution"])
    pooled = PhysicalRiskPrior(prior.coefficients, np.full(len(prior.weights), 1. / len(prior.weights)), prior.families, prior.family_names, {})
    source_psi = np.array([plans[row["context"]].trajectory(policies[row["policy_index"]]).coordinates(plans[row["context"]].capacity)
                           for row in design["assignments"]])
    source_rows = [{"psi": source_psi, "response": np.asarray(task["response"]), "noise": np.asarray(task["noise"])} for task in source["source_records"]]
    means = [SourceFunctionPrior.fit(head_batches(source_rows, h), [task["noise"][:, h] for task in source_rows], psi, "fixed_source") for h in range(2)]
    candidates = np.arange(len(policies))[:, None]
    protocol, rows = old["protocol"], []
    risk_variances = X @ prior.coefficients.T
    for saved in new["rows"]:
        response, noise = np.asarray(saved["response"]), np.asarray(saved["noise"])
        models = [p.target_model(psi) for p in means]
        for index, y, v in zip(initial, response, noise):
            for h, model in enumerate(models):
                model.update([index], y[h], v[h])
        m = np.column_stack([model.posterior_mean_many(candidates) for model in models])
        v = np.column_stack([model.posterior_var_many(candidates) for model in models])
        quality = objective_quality(m, v)
        arms = {}
        for name, p in (("factor_family", prior), ("pooled", pooled)):
            risk = p.target_model()
            risk.update_batch(RiskBatchData.from_log_labels(X[initial], response[:, 2], reps))
            eta_m, eta_v = risk.predict(X)
            gaussian, _ = verification_utility(np.column_stack([m, eta_m]), np.column_stack([v, eta_v]), protocol)
            particle = quality * particle_pass_probability(m[:, 1], v[:, 1], risk_variances,
                                                           np.exp(risk.log_weights), protocol)
            a, b = int(np.argmax(gaussian)), int(np.argmax(particle))
            arms[name] = {"gaussian_choice": a, "particle_choice": b,
                "utility_max_absolute_error": float(np.max(np.abs(gaussian - particle))),
                "particle_posterior_choice_gap": float(particle[b] - particle[a])}
        # True deployment reward is diagnostic only; no independent verifier
        # is run and no choice or prior is changed in response to this reward.
        world = InventoryCostSimulator(plans[0], sample_cost_law(saved["law_seed"]))
        truth = np.array([world.oracle_moments(x) for x in policies])
        reward, _ = verification_utility(truth, np.zeros_like(truth), protocol)
        for result in arms.values():
            result.update(true_reward_gaussian_choice=float(reward[result["gaussian_choice"]]),
                          true_reward_particle_choice=float(reward[result["particle_choice"]]))
        rows.append({"law_seed": saved["law_seed"], "arms": arms})
    return {"purpose": "exploratory cached decision interface; preserve frozen priors and no new observations",
        "rows": rows, "summary": {name: {"changed_choices": sum(r["arms"][name]["gaussian_choice"] != r["arms"][name]["particle_choice"] for r in rows),
            "mean_posterior_choice_gap": float(np.mean([r["arms"][name]["particle_posterior_choice_gap"] for r in rows])),
            "max_utility_error": float(max(r["arms"][name]["utility_max_absolute_error"] for r in rows)),
            "mean_true_initial_reward_gaussian": float(np.mean([r["arms"][name]["true_reward_gaussian_choice"] for r in rows])),
            "mean_true_initial_reward_particle": float(np.mean([r["arms"][name]["true_reward_particle_choice"] for r in rows]))}
            for name in ("factor_family", "pooled")},
        "new_episode_calls": 0, "verification_calls": 0, "decision_performance_established": False,
        "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/decision_interface.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": result["summary"], "local_cpu_seconds": result["local_cpu_seconds"]}, allow_nan=False))
