"""Decode the fixed Gaussian mean posterior against public price bounds."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.replay_orthogonal_meta_prior import prepare_replay
from problems.inventory_cost_plan import InventoryCostSimulator
from representation.billing_offset_prior import pooled_linked_prior
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_inventory_prior import OrthogonalInventoryBasis, embed_inventory_prior
from representation.source_function_prior import SourceFunctionPrior


def run():
    directory = ROOT / "paper_artifacts/billing_offset_replay_v1"
    p = json.loads((ROOT / "performance/billing_offset_replay_v1.json").read_text())
    archive, recorded, psi, reference, _, eta, eta_full = prepare_replay(p)
    source = json.loads((directory / "source_check.json").read_text())["all_source_priors"][1]
    raw = SourceFunctionPrior(InventoryFunctionBasis(reference, InventoryFunctionBasis.billing_indices),
        np.asarray(source["mean"]), np.asarray(source["covariance"]), source["discrepancy"], source["diagnostics"])
    basis = OrthogonalInventoryBasis(reference)
    selected = embed_inventory_prior(raw, basis, reference)
    pooled = embed_inventory_prior(pooled_linked_prior(raw), basis, reference)
    pooled = replace(pooled, diagnostics={**pooled.diagnostics, "mode": "same_source_linked_pooled"})
    scoring = json.loads((directory / "development_scoring.json").read_text())
    failures = scoring["largest_two_early_power_errors"]
    quantity = psi[:, [3, 5, 4, 6]]
    decoder = np.linalg.pinv(quantity)
    weights = InventoryCostSimulator.constraint_weights
    lower, upper = archive["constraint_price_range"]
    rows = []
    for failure in failures:
        old = next(r for r in recorded["rows"] if r["law_seed"] == failure["law_seed"] and r["arm"] == p["recorded_path_arm"])
        model = JointFunctionRiskPosterior([selected, pooled], [eta[2], eta_full[2]], psi)
        for observation in old["search_path"][:failure["paid_points"]]:
            model.update_batch(observation["index"], observation["response"][1], observation["response"][2], p["replicates"])
        mean, covariance = model.predict(full_cov=True)
        cost_mean = mean[:, 0] - raw.mean[0]
        coefficient_mean = decoder @ cost_mean
        coefficient_covariance = decoder @ covariance[:, 0, :, 0] @ decoder.T
        price_mean = coefficient_mean / weights
        price_sd = np.sqrt(np.diag(coefficient_covariance)) / weights
        rows.append({"law_seed": failure["law_seed"], "paid_points": failure["paid_points"],
            "recommendation": failure["recommendation"], "price_order": raw.basis.names,
            "public_constraint_price_range": [lower, upper], "posterior_price_mean": price_mean.tolist(),
            "posterior_price_sd": price_sd.tolist(),
            "posterior_price_marginal_95_intervals": np.column_stack([price_mean - 1.96 * price_sd, price_mean + 1.96 * price_sd]).tolist(),
            "posterior_mean_outside_public_bounds": ((price_mean < lower) | (price_mean > upper)).tolist(),
            "reconstructed_mean_max_error": float(np.max(np.abs(quantity @ coefficient_mean - cost_mean)))})
    result = {"scope": "posterior decoding using public quantities and fixed observed labels; no target price truth",
        "case_selection": "two largest early errors from already-open analytic development scoring",
        "target_truth_used_for_case_selection": True, "target_price_truth_used": False,
        "new_episode_calls": 0, "rows": rows}
    (directory / "price_bounds_diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
