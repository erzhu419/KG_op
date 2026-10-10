"""Initial-observation risk-factor diagnostic for the multi-context contract."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector, params_to_beta
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families


def _pooled(prior):
    return PhysicalRiskPrior(
        prior.coefficients,
        np.full(len(prior.weights), 1.0 / len(prior.weights)),
        prior.families,
        prior.family_names,
        {"mode": "same_source_pooled_factor_families"},
    )


def run():
    contract = json.loads((ROOT / "performance/meta_prior_risk_identifiability_contract_v1.json").read_text())
    archive = json.loads((ROOT / contract["source_archive_output"]).read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    base = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    policies = np.asarray(archive["public_policies"], dtype=float)
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    assignments = archive["assignments"]
    source_X = np.asarray([
        row["risk_features"] for row in assignments
    ], dtype=float)
    target_X = np.asarray([
        cumulative_feature_vector(plans[0].trajectory(policy).risk_exposures())
        for policy in policies
    ])
    tasks = [
        RiskBatchData.from_log_labels(
            source_X, np.asarray(row["response"])[:, 2], archive["replicates"]
        )
        for row in archive["source_records"]
    ]
    source_lodo = []
    for left in range(len(tasks)):
        keep = [task for index, task in enumerate(tasks) if index != left]
        learned = fit_source_risk_families(keep, np.vstack([source_X, target_X]))
        pooled = _pooled(learned)
        source_lodo.append({
            "left_out": left,
            "learned_minus_pooled_log_density": float(
                learned.target_model().log_predictive_density(tasks[left])
                - pooled.target_model().log_predictive_density(tasks[left])
            ),
            "learned_converged": bool(learned.diagnostics["source_optimizer_converged"]),
            "pooled_control_reference_mode": pooled.diagnostics.get("mode", ""),
        })
    initial = np.asarray(base["target_policy_indices"], dtype=int)
    target_rows = []
    z = norm.ppf(0.95)
    for offset, seed in enumerate(contract["development_target_law_seeds"]):
        seed = int(seed)
        world = InventoryCostSimulator(
            plans[0], sample_cost_law(seed, tuple(contract["constraint_price_range"]))
        )
        observations = []
        for index in initial:
            y, v = world.observe(
                policies[index], archive["replicates"],
                np.random.default_rng(int(contract["source_noise_seed_start"]) + 1000 + offset * 97 + int(index)),
            )
            observations.append((index, y, v))
        true_eta = np.asarray([world.oracle_moments(policy)[2] for policy in policies])
        true_beta = params_to_beta(world._law.risk)
        rows = {"law_seed": seed}
        learned = fit_source_risk_families(tasks, np.vstack([source_X, target_X]))
        for label, prior in (("learned", learned), ("pooled", _pooled(learned))):
            posterior = prior.target_model()
            posterior.update_batch(RiskBatchData.from_log_labels(
                target_X[initial], np.asarray([y[2] for _, y, _ in observations]), archive["replicates"]
            ))
            eta, eta_var = posterior.predict(target_X)
            holdout = np.ones(len(policies), dtype=bool)
            holdout[initial] = False
            mean_beta = np.asarray(posterior.diagnostics()["coefficient_mean"])
            rows[label] = {
                "eta_rmse_holdout": float(np.sqrt(np.mean((eta[holdout] - true_eta[holdout]) ** 2))),
                "eta_interval_coverage_holdout": float(np.mean(np.abs(eta[holdout] - true_eta[holdout]) <= 1.645 * np.sqrt(np.maximum(eta_var[holdout], 0.0)))),
                "factor_beta_rmse": float(np.sqrt(np.mean((mean_beta - true_beta) ** 2))),
                "effective_particles": float(posterior.diagnostics()["effective_particles"]),
            }
        target_rows.append(rows)
    return {
        "contract_id": contract["contract_id"],
        "purpose": "source LODO and eight-point target risk calibration; no search or verification",
        "confirmation_block_read": False,
        "source_lodo": source_lodo,
        "source_lodo_mean_gain": float(np.mean([row["learned_minus_pooled_log_density"] for row in source_lodo])),
        "target_rows": target_rows,
        "target_observations": len(target_rows) * len(initial) * archive["replicates"],
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_risk_identifiability_v1/initial_calibration.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source_lodo_mean_gain": result["source_lodo_mean_gain"],
                      "target_rows": len(result["target_rows"]),
                      "target_observations": result["target_observations"],
                      "confirmation_block_read": result["confirmation_block_read"]}))
