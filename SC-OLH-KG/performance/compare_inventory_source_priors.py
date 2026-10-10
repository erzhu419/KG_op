"""Replay one paid target interface under three mean-prior choices.

This is an attribution check, not a new target experiment.  The target's
existing eight observations are reused exactly.  Risk priors are held fixed
within each context so the comparison isolates the source-learned mean
structure from a same-source full dictionary and a target-only dictionary.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from core.joint_verification_kg import joint_verification_utility
from core.verification_kg import posterior_covariances, verification_utility
from performance.check_inventory_mean_interface import source_data
from problems.inventory_cost_plan import InventoryCostSimulator, sample_cost_law
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_mean_risk_prior import JointMeanRiskPosterior
from representation.physical_risk_prior import (
    PhysicalRiskPrior,
    RiskBatchData,
    fit_source_risk_families,
)
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches


PROTOCOL = {
    "status": "frozen_replay_existing_target_eight_points",
    "source_records": "paper_artifacts/inventory_physical_risk_20261004/check.json",
    "mean_arms": ["source_billing", "same_source_full_dictionary", "target_only_full_dictionary"],
    "risk_contexts": ["source_learned_factor_families", "same_source_pooled_scale"],
    "target_record_file": "paper_artifacts/inventory_physical_risk_20261004/new_targets.json",
    "target_law_seeds": [95001, 95002, 95003, 95004, 95005, 95006, 95007, 95008],
    "initial_points": 8,
    "replicates": 12,
    "verification_replicates": 928,
    "new_target_episode_calls": 0,
    "oracle_use": "post-decision scoring only; never used for fitting or recommendation",
    "scientific_status": "interface attribution; not independent confirmation or mature-baseline comparison",
}


def _load():
    group = json.loads((ROOT / "paper_artifacts/inventory_physical_risk_20261004/mean_posterior_group.json").read_text())
    source, design, policies, plans, data = source_data(group["protocol"]["source_records"])
    old = json.loads((ROOT / source["protocol"]["existing_records"]).read_text())
    source_psi = data[0]["psi"]
    target_psi = plans[0].coordinates_many(policies)
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    Xs = np.asarray(design["risk_design"])
    risk_tasks = [RiskBatchData.from_log_labels(Xs, task["response"][:, 2], 12) for task in data]
    learned_risk = fit_source_risk_families(
        risk_tasks, np.vstack([Xs, X]), **source["protocol"]["source_risk_distribution"]
    )
    pooled_risk = PhysicalRiskPrior(
        learned_risk.coefficients,
        np.full(len(learned_risk.weights), 1.0 / len(learned_risk.weights)),
        learned_risk.families,
        learned_risk.family_names,
        {**learned_risk.diagnostics, "mode": "same_source_pooled_scale"},
    )

    learned_means = []
    for spec in group["all_source_priors"]:
        basis = InventoryFunctionBasis(source_psi, spec["diagnostics"]["selected_indices"])
        learned_means.append(SourceFunctionPrior(
            basis, np.asarray(spec["mean"]), np.asarray(spec["covariance"]),
            float(spec["discrepancy_variance"]), spec["diagnostics"],
        ))
    full_means, target_only_means = [], []
    for head in range(2):
        batches = head_batches(data, head)
        noises = [task["noise"][:, head] for task in data]
        full_means.append(SourceFunctionPrior.fit(
            batches, noises, source_psi, "fixed_source"
        ))
        target_only_means.append(SourceFunctionPrior.fit(
            [], [], source_psi, "target_only"
        ))
    target_records = json.loads((ROOT / PROTOCOL["target_record_file"]).read_text())["rows"]
    return {
        "source": source,
        "policies": policies,
        "plans": plans,
        "data": data,
        "target_psi": target_psi,
        "X": X,
        "means": {
            "source_billing": learned_means,
            "same_source_full_dictionary": full_means,
            "target_only_full_dictionary": target_only_means,
        },
        "risks": {
            "source_learned_factor_families": learned_risk,
            "same_source_pooled_scale": pooled_risk,
        },
        "target_records": target_records,
    }


def run():
    start = time.process_time()
    p = {**PROTOCOL, "verification_replicates": PROTOCOL["verification_replicates"]}
    loaded = _load()
    candidates = np.arange(len(loaded["policies"]))[:, None]
    psi = loaded["target_psi"]
    X = loaded["X"]
    protocol = json.loads((ROOT / "performance/inventory_meta_prior_closed_loop_protocol.json").read_text())
    protocol = {**protocol, "verification_replicates": p["verification_replicates"]}
    rows = []
    for target in loaded["target_records"]:
        law_seed = int(target["law_seed"])
        world = InventoryCostSimulator(loaded["plans"][0], sample_cost_law(law_seed))
        truth = np.array([world.oracle_moments(x) for x in loaded["policies"]])
        true_utility, true_power = verification_utility(truth, np.zeros_like(truth), protocol)
        target_indices = np.asarray(target["initial_policy_indices"], dtype=int)
        target_response = np.asarray(target["response"], dtype=float)
        target_noise = np.asarray(target["noise"], dtype=float)
        for mean_name, mean_priors in loaded["means"].items():
            for risk_name, risk_prior in loaded["risks"].items():
                f_model = mean_priors[0].target_model(psi)
                g_prior = mean_priors[1]
                joint = JointMeanRiskPosterior(g_prior.mean, g_prior.covariance, risk_prior)
                Z = np.column_stack([np.ones(len(psi)), g_prior.basis.transform(psi)])
                for index, response, noise in zip(target_indices, target_response, target_noise):
                    f_model.update([int(index)], response[0], noise[0])
                    joint.update_batch(
                        Z[int(index):int(index) + 1], response[1:2],
                        RiskBatchData.from_log_labels(X[int(index):int(index) + 1], response[2:3], p["replicates"]),
                    )
                f_mean = f_model.posterior_mean_many(candidates)
                f_cov = posterior_covariances([f_model], candidates)[..., 0]
                pair_mean, pair_cov = joint.predict(Z, X, full_cov=True)
                diagonal = pair_cov[np.arange(len(psi)), :, np.arange(len(psi)), :]
                utility, power = joint_verification_utility(
                    f_mean, np.diag(f_cov), pair_mean, diagonal, protocol
                )
                choice = int(np.argmax(utility))
                errors = np.column_stack([
                    f_mean - truth[:, 0], pair_mean[:, 0] - truth[:, 1], pair_mean[:, 1] - truth[:, 2]
                ])
                rows.append({
                    "law_seed": law_seed,
                    "mean_arm": mean_name,
                    "risk_context": risk_name,
                    "choice": choice,
                    "predicted_utility": float(utility[choice]),
                    "predicted_verifier_power": float(power[choice]),
                    "true_expected_verified_reward": float(true_utility[choice]),
                    "best_true_expected_verified_reward": float(true_utility.max()),
                    "true_shortfall": float(true_utility.max() - true_utility[choice]),
                    "true_verifier_power": float(true_power[choice]),
                    "rmse_f": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                    "rmse_g": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
                    "rmse_eta": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
                    "source_calls_new": 0,
                    "target_calls_new": 0,
                })
    return {
        "protocol": p,
        "rows": rows,
        "tasks": len(loaded["target_records"]),
        "target_records_used": True,
        "oracle_used_only_after_recommendation": True,
        "new_source_episode_calls": 0,
        "new_target_episode_calls": 0,
        "local_cpu_seconds": time.process_time() - start,
    }


if __name__ == "__main__":
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/source_prior_attribution_replay.json"
    result = run()
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"tasks": result["tasks"], "rows": len(result["rows"]),
                      "local_cpu_seconds": result["local_cpu_seconds"]}, indent=2))
