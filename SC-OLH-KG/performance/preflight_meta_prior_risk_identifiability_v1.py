"""Outcome-free geometry preflight for the risk-factor contract."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from problems.inventory_cost_plan import InventoryPlan


def run():
    contract = json.loads((ROOT / "performance/meta_prior_risk_identifiability_contract_v1.json").read_text())
    base = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    policies = np.asarray(base["public_policies"], dtype=float)
    rows = []
    context_summaries = []
    for context_index, context in enumerate(design["contexts"]):
        plan = InventoryPlan(**context)
        X = np.asarray([
            cumulative_feature_vector(plan.trajectory(policy).risk_exposures())
            for policy in policies
        ])
        _, _, pivots = qr(X.T, pivoting=True, mode="economic")
        selected = np.sort(np.asarray(pivots[:int(contract["source_sites_per_context"])], dtype=int))
        rows.extend({
            "context": int(context_index),
            "policy_index": int(policy_index),
            "risk_features": X[policy_index].tolist(),
        } for policy_index in selected)
        context_summaries.append({
            "context": int(context_index),
            "selected_sites": int(len(selected)),
            "rank": int(np.linalg.matrix_rank(X[selected])),
            "condition_number": float(np.linalg.cond(X[selected])),
        })
    matrix = np.asarray([row["risk_features"] for row in rows])
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    gate = contract["geometry_gate"]
    passed = bool(
        len(rows) == int(contract["source_sites_total"])
        and np.linalg.matrix_rank(matrix) >= int(gate["required_rank"])
        and float(np.linalg.cond(matrix)) <= float(gate["maximum_condition_number"])
        and all(row["selected_sites"] >= int(gate["minimum_context_sites"])
                for row in context_summaries)
    )
    return {
        "contract_id": contract["contract_id"],
        "target_oracle_used": False,
        "selected_assignments": rows,
        "context_summaries": context_summaries,
        "joint_rank": int(np.linalg.matrix_rank(matrix)),
        "joint_condition_number": float(np.linalg.cond(matrix)),
        "joint_singular_values": singular.tolist(),
        "passed": passed,
        "target_observations_used": 0,
        "verification_calls": 0,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_risk_identifiability_v1/geometry_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "joint_rank": result["joint_rank"],
                      "joint_condition_number": result["joint_condition_number"],
                      "target_observations_used": result["target_observations_used"]}))
