"""Generate the frozen source archive for the feasible-deployment contract."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


def run():
    contract = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    base = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    source = contract["source_archive"]
    price_range = tuple(source["constraint_price_range"])
    contexts = design["contexts"]
    policies = np.asarray(base["public_policies"], dtype=float)
    assignments = design["assignments"]
    replicates = int(source["replicates_per_site"])
    task_count = int(source["task_count"])
    rows = []
    episode_calls = 0
    for task in range(task_count):
        law = sample_cost_law(int(source["law_seed_start"]) + task, price_range)
        rng = np.random.default_rng(int(source["noise_seed_start"]) + task)
        response, noise = [], []
        for row in assignments:
            context = dict(contexts[int(row["context"])])
            plan = InventoryPlan(**context)
            world = InventoryCostSimulator(plan, law)
            y, v = world.observe(policies[int(row["policy_index"])], replicates, rng)
            response.append(y)
            noise.append(v)
            episode_calls += world.episode_calls
        rows.append({"response": np.asarray(response).tolist(), "noise": np.asarray(noise).tolist()})
    result = {
        "contract_id": contract["contract_id"],
        "purpose": "source observations only; no target observations, target oracle, search, or verification",
        "constraint_price_range": list(price_range),
        "contexts": contexts,
        "assignments": assignments,
        "public_policies": policies.tolist(),
        "source_policy_indices": [int(row["policy_index"]) for row in assignments],
        "source_law_seed_start": int(source["law_seed_start"]),
        "source_noise_seed_start": int(source["noise_seed_start"]),
        "source_tasks": task_count,
        "replicates": replicates,
        "source_episode_calls": episode_calls,
        "source_records": rows,
        "target_observations_used": 0,
        "verification_calls": 0,
    }
    return result


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_feasible_deployment_v1/source_archive.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source_tasks": result["source_tasks"],
                      "source_sites_per_task": len(result["assignments"]),
                      "source_episode_calls": result["source_episode_calls"],
                      "target_observations_used": result["target_observations_used"],
                      "verification_calls": result["verification_calls"]}))
