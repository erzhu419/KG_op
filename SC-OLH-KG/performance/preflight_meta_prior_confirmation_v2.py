"""Oracle-only fixed-seed preflight for the independent v2 confirmation block."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


def run():
    contract = json.loads((ROOT / "performance/meta_prior_feasible_confirmation_v2.json").read_text())
    base = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    archive = json.loads((ROOT / contract["source_archive"]).read_text())
    policies = np.asarray(archive["public_policies"], dtype=float)
    context = dict(base["target_context"])
    price_range = tuple(context.pop("constraint_price_range"))
    plan = InventoryPlan(**context)
    z = norm.ppf(1.0 - float(base["decision_protocol"]["chance_alpha"]))
    rows = []
    for seed in contract["target_law_seeds"]:
        world = InventoryCostSimulator(plan, sample_cost_law(int(seed), price_range))
        truth = np.asarray([world.oracle_moments(policy) for policy in policies])
        margin = truth[:, 1] + z * np.exp(0.5 * truth[:, 2])
        feasible = margin <= 0.0
        objective = (5.0 - truth[:, 0]) / 10.0
        rows.append({
            "law_seed": int(seed),
            "feasible_count": int(feasible.sum()),
            "infeasible_count": int((~feasible).sum()),
            "objective_span_among_feasible": float(np.ptp(objective[feasible])) if feasible.any() else None,
            "minimum_margin": float(margin.min()),
            "maximum_margin": float(margin.max()),
        })
    return {
        "contract_id": contract["contract_id"],
        "purpose": "oracle-only fixed-seed preflight; no paid target observations or verification",
        "rows": rows,
        "all_seeds_retained": True,
        "original_confirmation_block_read": False,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_feasible_deployment_v1/confirmation_v2_preflight.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "all_seeds_retained": result["all_seeds_retained"],
                      "original_confirmation_block_read": result["original_confirmation_block_read"]}))
