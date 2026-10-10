"""Run the frozen official HyperBO initial-posterior replay.

The source and target records already exist from the independent confirmation.
This script adds no simulator calls and scores only the common eight-point
posterior across the 96 public candidates.
"""

import json
import copy
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_external_models import OfficialHyperBOSurrogate
from baselines.transfer_models import ScalarTaskData
from core.verification_kg import verification_utility
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


PROTOCOL_PATH = ROOT / "performance/inventory_official_hyperbo_initial_protocol.json"
INDEPENDENT = ROOT / "paper_artifacts/inventory_physical_risk_20261004/independent_source_confirmation.json"
BASE = ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json"
DESIGN = ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json"


def run(p):
    started = time.process_time()
    archive = json.loads(INDEPENDENT.read_text())
    base = json.loads(BASE.read_text())
    design = json.loads(DESIGN.read_text())
    policies = np.asarray(base["public_policies"], dtype=float)
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    target_X = plans[0].coordinates_many(policies)
    source_tasks = []
    for index, task in enumerate(archive["source_records"]):
        source_tasks.append({
            "psi": np.asarray(task["psi"], dtype=float),
            "response": np.asarray(task["response"], dtype=float),
            "noise": np.asarray(task["noise"], dtype=float),
        })
    target_records = archive["target_records"]
    heads = ["objective", "constraint_mean", "log_variance"]
    trained = []
    for head, name in enumerate(heads):
        tasks = [ScalarTaskData(
            f"source_{index}", task["psi"], task["response"][:, head], task["noise"][:, head]
        ) for index, task in enumerate(source_tasks)]
        model = OfficialHyperBOSurrogate(seed=100 + head, source_steps=p["source_training_steps"])
        model.meta_fit(tasks)
        trained.append(model)

    rows = []
    for target in target_records:
        indices = np.asarray(target["initial_policy_indices"], dtype=int)
        response = np.asarray(target["response"], dtype=float)
        noise = np.asarray(target["noise"], dtype=float)
        models = []
        for head, source_model in enumerate(trained):
            # Copy the frozen source posterior so target conditioning remains
            # isolated across target laws without retraining the archive.
            model = copy.deepcopy(source_model)
            model.adapt(target_X[indices], response[:, head], noise[:, head])
            models.append(model)
        predictions = [model.predict(target_X, full_cov=True) for model in models]
        mean = np.column_stack([prediction[0] for prediction in predictions])
        variance = np.column_stack([np.diag(prediction[1]) for prediction in predictions])
        truth_world = InventoryCostSimulator(plans[0], sample_cost_law(target["law_seed"]))
        truth = np.array([truth_world.oracle_moments(policy) for policy in policies])
        errors = mean - truth
        for budget in p["evaluation_budgets"]:
            shifted = mean.copy()
            shifted_truth = truth.copy()
            shifted[:, 1] += plans[0].cost_budget - budget
            shifted_truth[:, 1] += plans[0].cost_budget - budget
            protocol = json.loads((ROOT / "performance/inventory_meta_prior_closed_loop_protocol.json").read_text())
            protocol = {**protocol, "verification_replicates": p["verification_replicates"]}
            utility, power = verification_utility(shifted, variance, protocol)
            true_utility, true_power = verification_utility(shifted_truth, np.zeros_like(shifted_truth), protocol)
            choice = int(np.argmax(utility))
            rows.append({
                "law_seed": int(target["law_seed"]), "evaluation_budget": float(budget),
                "choice": choice, "predicted_utility": float(utility[choice]),
                "predicted_verifier_power": float(power[choice]),
                "true_expected_verified_reward": float(true_utility[choice]),
                "best_true_expected_verified_reward": float(true_utility.max()),
                "true_shortfall": float(true_utility.max() - true_utility[choice]),
                "true_verifier_power": float(true_power[choice]),
                "rmse_f": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                "rmse_g": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
                "rmse_eta": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
                "source_noise_variances": [float(model.observation_noise_variance()) for model in models],
                "adapted_points_per_head": [int(model.n_target) for model in models],
            })
    return {
        "protocol": p,
        "rows": rows,
        "source_tasks": len(source_tasks),
        "target_tasks": len(target_records),
        "new_source_episode_calls": 0,
        "new_target_episode_calls": 0,
        "official_fallback_used": False,
        "local_cpu_seconds": time.process_time() - started,
    }


if __name__ == "__main__":
    p = json.loads(PROTOCOL_PATH.read_text())
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/official_hyperbo_initial.json"
    result = run(p)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "source_tasks": result["source_tasks"],
                      "target_tasks": result["target_tasks"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}, indent=2))
