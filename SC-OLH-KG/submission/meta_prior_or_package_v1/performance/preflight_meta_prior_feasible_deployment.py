"""Oracle-only preflight for the feasible-deployment task contract.

This script reads no paid target observations and runs no verifier.  It checks
only whether the declared public candidate bank and development law block can
support a meaningful end-to-end comparison.  Its output is a contract gate,
not a method result.
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


def _evaluate_block(contract, seeds):
    context = dict(contract["target_context"])
    price_range = tuple(context.pop("constraint_price_range"))
    plan = InventoryPlan(**context)
    base = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    policies = np.asarray(base["public_policies"], dtype=float)
    alpha = float(contract["decision_protocol"]["chance_alpha"])
    z = norm.ppf(1.0 - alpha)
    rows = []
    for seed in seeds:
        world = InventoryCostSimulator(plan, sample_cost_law(seed, price_range))
        truth = np.asarray([world.oracle_moments(policy) for policy in policies])
        margin = truth[:, 1] + z * np.exp(0.5 * truth[:, 2])
        feasible = np.flatnonzero(margin <= 0.0)
        objective = (5.0 - truth[:, 0]) / 10.0
        rows.append({
            "law_seed": int(seed),
            "candidate_count": int(len(policies)),
            "feasible_count": int(len(feasible)),
            "minimum_margin": float(np.min(margin)),
            "maximum_margin": float(np.max(margin)),
            "objective_span_among_feasible": (
                float(np.ptp(objective[feasible])) if len(feasible) else None
            ),
            "best_feasible_objective": (
                float(np.max(objective[feasible])) if len(feasible) else None
            ),
            "infeasible_candidate_count": int(np.sum(margin > 0.0)),
        })
    return rows


def run():
    contract = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    block = contract["development_block"]
    seeds = [int(seed) for seed in block["law_seeds"]]
    rows = _evaluate_block(contract, seeds)
    gate = block["gate"]
    failures = []
    for row in rows:
        if row["feasible_count"] < int(gate["minimum_feasible_candidates"]):
            failures.append({"law_seed": row["law_seed"], "reason": "too_few_feasible_candidates"})
        if (row["objective_span_among_feasible"] is None
                or row["objective_span_among_feasible"] < float(gate["minimum_objective_span_among_feasible"])):
            failures.append({"law_seed": row["law_seed"], "reason": "insufficient_objective_separation"})
        if gate["must_retain_an_infeasible_candidate"] and row["infeasible_candidate_count"] == 0:
            failures.append({"law_seed": row["law_seed"], "reason": "constraint_not_discriminating"})
    return {
        "contract_id": contract["contract_id"],
        "purpose": "oracle-only development preflight; no paid observations or verification calls",
        "development_rows": rows,
        "gate_failures": failures,
        "passed": not failures,
        "heldout_confirmation_block_read": False,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_feasible_deployment_v1/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "rows": len(result["development_rows"]),
                      "gate_failures": len(result["gate_failures"]),
                      "heldout_confirmation_block_read": result["heldout_confirmation_block_read"]}))
