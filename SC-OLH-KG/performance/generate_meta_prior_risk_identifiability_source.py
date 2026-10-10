"""Generate source observations for the multi-context risk geometry contract."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


def run():
    contract = json.loads((ROOT / "performance/meta_prior_risk_identifiability_contract_v1.json").read_text())
    geometry = json.loads((ROOT / "paper_artifacts/meta_prior_risk_identifiability_v1/geometry_preflight.json").read_text())
    base = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    policies = np.asarray(base["public_policies"], dtype=float)
    contexts = design["contexts"]
    assignments = geometry["selected_assignments"]
    reps = int(contract["replicates_per_site"])
    price_range = tuple(contract["constraint_price_range"])
    records = []
    calls = 0
    for task in range(int(contract["source_tasks"])):
        law = sample_cost_law(int(contract["source_law_seed_start"]) + task, price_range)
        rng = np.random.default_rng(int(contract["source_noise_seed_start"]) + task)
        response, noise = [], []
        for row in assignments:
            plan = InventoryPlan(**contexts[int(row["context"])])
            world = InventoryCostSimulator(plan, law)
            y, v = world.observe(policies[int(row["policy_index"])], reps, rng)
            response.append(y)
            noise.append(v)
            calls += world.episode_calls
        records.append({"response": np.asarray(response).tolist(), "noise": np.asarray(noise).tolist()})
    return {
        "contract_id": contract["contract_id"],
        "purpose": "multi-context source risk archive only; no target observations or verification",
        "contexts": contexts,
        "assignments": assignments,
        "public_policies": policies.tolist(),
        "source_law_seed_start": int(contract["source_law_seed_start"]),
        "source_noise_seed_start": int(contract["source_noise_seed_start"]),
        "constraint_price_range": list(price_range),
        "source_tasks": int(contract["source_tasks"]),
        "replicates": reps,
        "source_episode_calls": calls,
        "source_records": records,
        "target_observations_used": 0,
        "verification_calls": 0,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_risk_identifiability_v1/source_archive.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source_tasks": result["source_tasks"],
                      "sites_per_task": len(result["assignments"]),
                      "source_episode_calls": result["source_episode_calls"],
                      "target_observations_used": result["target_observations_used"]}))
