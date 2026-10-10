"""Small initial-observation representation diagnostic.

The diagnostic compares the fixed public policy coordinate with the declared
observable inventory exposure after the same eight paid target points.  It is
not a deployment confirmation and does not read the held-out confirmation
seed block.
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law
from representation.physical_risk_prior import RiskBatchData, fit_source_risk_families
from representation.source_function_prior import SourceFunctionPrior
from representation.transferable_spectral import SourceDomainBatch


def _coordinates(archive, design):
    policies = np.asarray(archive["public_policies"], dtype=float)
    assignments = archive["assignments"]
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    policy = np.asarray([policies[row["policy_index"]] for row in assignments])
    observable = np.asarray([
        plans[row["context"]].trajectory(policies[row["policy_index"]]).coordinates(
            plans[row["context"]].capacity
        )
        for row in assignments
    ])
    return policies, plans, policy, observable


def _fit_mean_models(archive, source_psi, candidate_psi):
    heads = []
    for head in (0, 1):
        batches = [
            SourceDomainBatch(str(task), source_psi, np.asarray(row["response"])[:, head])
            for task, row in enumerate(archive["source_records"])
        ]
        noises = [np.asarray(row["noise"])[:, head] for row in archive["source_records"]]
        prior = SourceFunctionPrior.fit(
            batches, noises, candidate_psi, "learned", active_dim=4
        )
        heads.append(prior.target_model(candidate_psi))
    return heads


def run():
    contract = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    archive = json.loads((ROOT / contract["source_archive"]["output"]).read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    policies, plans, source_policy_psi, source_observable_psi = _coordinates(archive, design)
    target_plan = plans[0]
    target_policy_psi = policies
    target_observable_psi = np.asarray([
        target_plan.trajectory(policy).coordinates(target_plan.capacity)
        for policy in policies
    ])
    risk_design = np.asarray(design["risk_design"], dtype=float)
    risk_tasks = [
        RiskBatchData.from_log_labels(
            risk_design, np.asarray(row["response"])[:, 2], archive["replicates"]
        )
        for row in archive["source_records"]
    ]
    risk_prior = fit_source_risk_families(
        risk_tasks, np.vstack([risk_design, np.asarray([
            cumulative_feature_vector(target_plan.trajectory(policy).risk_exposures())
            for policy in policies
        ])]),
    )
    initial_indices = np.asarray(
        json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())["target_policy_indices"],
        dtype=int,
    )
    z = norm.ppf(1.0 - float(contract["decision_protocol"]["chance_alpha"]))
    outputs = {"policy": [], "observable": []}
    for seed in contract["development_block"]["law_seeds"]:
        seed = int(seed)
        noise_seed = int(contract["development_block"]["noise_seed_start"]) + (seed - int(contract["development_block"]["law_seeds"][0]))
        world = InventoryCostSimulator(
            target_plan,
            sample_cost_law(seed, tuple(contract["target_context"]["constraint_price_range"])),
        )
        observations = []
        for index in initial_indices:
            y, v = world.observe(policies[index], archive["replicates"], np.random.default_rng(noise_seed + int(index)))
            observations.append((int(index), y, v))
        truth = np.asarray([world.oracle_moments(policy) for policy in policies])
        target_X = np.asarray([
            cumulative_feature_vector(target_plan.trajectory(policy).risk_exposures())
            for policy in policies
        ])
        risk_post = risk_prior.target_model()
        risk_post.update_batch(RiskBatchData.from_log_labels(
            target_X[initial_indices],
            np.asarray([row[1][2] for row in observations]),
            archive["replicates"],
        ))
        eta_pred, eta_var = risk_post.predict(target_X)
        for label, source_psi, target_psi in (
            ("policy", source_policy_psi, target_policy_psi),
            ("observable", source_observable_psi, target_observable_psi),
        ):
            # Refit with the chosen source coordinate so the two arms differ
            # only in representation, while sharing observations and risk.
            f_model, g_model = _fit_mean_models(archive, source_psi, target_psi)
            for index, y, v in observations:
                f_model.update([index], y[0], v[0])
                g_model.update([index], y[1], v[1])
            f_pred = f_model.posterior_mean_many(np.arange(len(policies))[:, None])
            g_pred = g_model.posterior_mean_many(np.arange(len(policies))[:, None])
            predicted_margin = g_pred + z * np.exp(0.5 * eta_pred)
            predicted_feasible = predicted_margin <= 0.0
            choice_pool = np.flatnonzero(predicted_feasible)
            choice = int(choice_pool[np.argmin(f_pred[choice_pool])]) if len(choice_pool) else int(np.argmin(predicted_margin))
            true_margin = truth[:, 1] + z * np.exp(0.5 * truth[:, 2])
            holdout = np.ones(len(policies), dtype=bool)
            holdout[initial_indices] = False
            outputs[label].append({
                "law_seed": seed,
                "f_rmse_holdout": float(np.sqrt(np.mean((f_pred[holdout] - truth[holdout, 0]) ** 2))),
                "g_rmse_holdout": float(np.sqrt(np.mean((g_pred[holdout] - truth[holdout, 1]) ** 2))),
                "eta_rmse_holdout": float(np.sqrt(np.mean((eta_pred[holdout] - truth[holdout, 2]) ** 2))),
                "g_rank_spearman_holdout": float(spearmanr(g_pred[holdout], truth[holdout, 1]).statistic),
                "predicted_feasible_count": int(np.sum(predicted_feasible)),
                "choice": choice,
                "choice_true_feasible": bool(true_margin[choice] <= 0.0),
                "choice_true_reward": float(np.clip((5.0 - truth[choice, 0]) / 10.0, 0.0, 1.0) if true_margin[choice] <= 0.0 else 0.0),
                "true_feasible_count": int(np.sum(true_margin <= 0.0)),
                "target_episode_calls": int(world.episode_calls),
            })
    return {
        "contract_id": contract["contract_id"],
        "purpose": "initial eight-point representation diagnostic; not confirmation",
        "confirmation_block_read": False,
        "target_observations": 8 * len(initial_indices) * archive["replicates"],
        "arms": outputs,
        "source_archive": contract["source_archive"]["output"],
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_feasible_deployment_v1/representation_initial.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"target_observations": result["target_observations"], "confirmation_block_read": result["confirmation_block_read"], "arms": {k: len(v) for k, v in result["arms"].items()}}))
