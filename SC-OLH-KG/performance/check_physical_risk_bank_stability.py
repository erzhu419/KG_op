"""One frozen numerical sensitivity check after the primary target regression.

Uses a second Sobol scramble at the same size; never selects the better seed.
No observation, design, likelihood or regularization changes.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from problems.inventory_cost_plan import InventoryPlan, InventoryCostSimulator, sample_cost_law
from representation.physical_risk_prior import RiskBatchData, fit_source_physical_risk


def run():
    start = time.process_time()
    folder = ROOT / "paper_artifacts/inventory_physical_risk_20261004"
    primary = json.loads((folder / "check.json").read_text())
    p = primary["protocol"]
    old = json.loads((ROOT / p["existing_records"]).read_text())
    design = json.loads((ROOT / p["source_design"]).read_text())
    X_source = np.asarray(design["risk_design"])
    reps = old["protocol"]["replicates"]
    tasks = [RiskBatchData.from_log_labels(X_source, np.asarray(task["response"])[:, 2], reps)
             for task in primary["source_records"]]
    plan = InventoryPlan(**old["protocol"]["context"])
    policies = np.asarray(old["public_policies"])
    X = np.array([cumulative_feature_vector(plan.trajectory(x).risk_exposures()) for x in policies])
    labels = RiskBatchData.from_log_labels(X[old["target_policy_indices"]], np.asarray(old["target_response"])[:, 2], reps)
    held_out = np.ones(len(X), dtype=bool)
    held_out[old["target_policy_indices"]] = False
    world = InventoryCostSimulator(plan, sample_cost_law(old["protocol"]["target_law_seed"]))
    truth = np.array([world.oracle_moments(x)[2] for x in policies])
    config = p["source_risk_distribution"]
    # Fixed before this script executes; alternate seed is a numerical check,
    # not a new selected method or an independent target experiment.
    seeds = [config["seed"], 105043]
    rows, predictions = [], []
    for seed in seeds:
        priors = fit_source_physical_risk(tasks, np.vstack([X_source, X]), **{**config, "seed": seed})
        means = []
        row = {"seed": seed, "optimizer_converged": priors[0].diagnostics["source_optimizer_converged"]}
        for name, prior in zip(("learned", "pooled"), priors):
            posterior = prior.target_model()
            posterior.update_batch(labels)
            mean, _ = posterior.predict(X)
            row[name] = {"held_out_eta_rmse": float(np.sqrt(np.mean((mean[held_out] - truth[held_out]) ** 2))),
                         "effective_particles": posterior.diagnostics()["effective_particles"]}
            means.append(mean)
        predictions.append(means)
        rows.append(row)
    return {"purpose": "numerical bank sensitivity; keep original seed and negative primary result",
            "new_episode_calls": 0, "particle_count": 4096, "rows": rows,
            "held_out_prediction_rms_change": {name: float(np.sqrt(np.mean((predictions[1][h][held_out] - predictions[0][h][held_out]) ** 2)))
                for h, name in enumerate(("learned", "pooled"))}, "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/bank_stability.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
