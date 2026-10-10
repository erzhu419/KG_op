"""Development closed loop for the feasible-deployment contract.

This is a new, isolated runner.  It does not overwrite the original
Inventory closed-loop artifacts and never reads the held-out confirmation
seed block.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma
from scipy.stats import chi2, norm, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from core.verification_kg import posterior_covariances, verification_kg_from_moments, verification_utility
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import SourceFunctionPrior
from representation.transferable_spectral import SourceDomainBatch


def _fit_mean_priors(archive, source_psi, target_psi):
    models = []
    for head in (0, 1):
        batches = [
            SourceDomainBatch(str(task), source_psi, np.asarray(row["response"])[:, head])
            for task, row in enumerate(archive["source_records"])
        ]
        noises = [np.asarray(row["noise"])[:, head] for row in archive["source_records"]]
        prior = SourceFunctionPrior.fit(
            batches, noises, target_psi, "learned", active_dim=4
        )
        models.append(prior.target_model(target_psi))
    return models


def _terminal_verification(world, policy, p, rng):
    samples = world.episodes(policy, p["verification_replicates"], rng)[:, 1]
    n = len(samples)
    df = n - 1
    mean = float(samples.mean())
    sd = float(samples.std(ddof=1))
    upper = mean + sd * (
        t.ppf(.975, df) / np.sqrt(n)
        + norm.ppf(1.0 - p["chance_alpha"]) * np.sqrt(df / chi2.ppf(.025, df))
    )
    return bool(upper <= 0.0), float(upper)


def run():
    start = time.process_time()
    contract = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    archive = json.loads((ROOT / contract["source_archive"]["output"]).read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    base = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    policies = np.asarray(archive["public_policies"], dtype=float)
    assignments = archive["assignments"]
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    target_plan = plans[0]
    initial = np.asarray(base["target_policy_indices"], dtype=int)
    reps = int(contract["decision_protocol"]["common_initial_points"])
    target_points = int(contract["decision_protocol"]["target_search_points"])
    n_reps = int(archive["replicates"])
    p = {
        "replicates": n_reps,
        "chance_alpha": float(contract["decision_protocol"]["chance_alpha"]),
        "verification_replicates": int(contract["decision_protocol"]["verification_replicates"]),
    }
    source_policy_psi = np.asarray([policies[row["policy_index"]] for row in assignments])
    source_observable_psi = np.asarray([
        plans[row["context"]].trajectory(policies[row["policy_index"]]).coordinates(
            plans[row["context"]].capacity
        )
        for row in assignments
    ])
    target_policy_psi = policies
    target_observable_psi = np.asarray([
        target_plan.trajectory(policy).coordinates(target_plan.capacity)
        for policy in policies
    ])
    source_X = np.asarray(design["risk_design"], dtype=float)
    target_X = np.asarray([
        cumulative_feature_vector(target_plan.trajectory(policy).risk_exposures())
        for policy in policies
    ])
    risk_tasks = [
        RiskBatchData.from_log_labels(
            source_X, np.asarray(row["response"])[:, 2], n_reps
        )
        for row in archive["source_records"]
    ]
    learned_risk = fit_source_risk_families(
        risk_tasks, np.vstack([source_X, target_X])
    )
    pooled_risk = PhysicalRiskPrior(
        learned_risk.coefficients,
        np.full(len(learned_risk.weights), 1.0 / len(learned_risk.weights)),
        learned_risk.families,
        learned_risk.family_names,
        {"mode": "same_source_pooled_factor_families"},
    )
    arms = {
        "policy_pooled_risk": (source_policy_psi, target_policy_psi, pooled_risk),
        "observable_pooled_risk": (source_observable_psi, target_observable_psi, pooled_risk),
        "observable_learned_risk": (source_observable_psi, target_observable_psi, learned_risk),
    }
    rows = []
    price_range = tuple(contract["target_context"]["constraint_price_range"])
    target_seeds = [int(seed) for seed in contract["development_block"]["law_seeds"]]
    noise_start = int(contract["development_block"]["noise_seed_start"])
    for seed_offset, law_seed in enumerate(target_seeds):
        noise_seed = noise_start + seed_offset
        world = InventoryCostSimulator(
            target_plan, sample_cost_law(law_seed, price_range)
        )
        observation_cache = {}
        verifier_cache = {}
        task_rows = []
        for arm_name, (source_psi, target_psi, risk_prior) in arms.items():
            means = _fit_mean_priors(archive, source_psi, target_psi)
            risk_model = risk_prior.target_model()
            visits = np.zeros(len(policies), dtype=int)
            path = []
            initial_choice = None

            def moments():
                eta, eta_cov = risk_model.predict(target_X, full_cov=True)
                mean = np.column_stack([
                    means[0].posterior_mean_many(np.arange(len(policies))[:, None]),
                    means[1].posterior_mean_many(np.arange(len(policies))[:, None]),
                    eta,
                ])
                cov = np.concatenate([
                    posterior_covariances(means, np.arange(len(policies))[:, None]),
                    eta_cov[..., None],
                ], axis=2)
                return mean, cov

            for step in range(target_points):
                if step < reps:
                    index = int(initial[step])
                else:
                    mean, covariance = moments()
                    variance = np.einsum("iih->ih", covariance)
                    fantasy_noise = np.column_stack([
                        np.full(len(policies), .12 ** 2 / n_reps),
                        np.exp(mean[:, 2] + .5 * variance[:, 2]) / n_reps,
                        np.full(len(policies), polygamma(1, (n_reps - 1) / 2)),
                    ])
                    scores = verification_kg_from_moments(
                        mean, covariance, fantasy_noise, p
                    )
                    index = int(np.argmax(scores))
                key = (index, int(visits[index]))
                visits[index] += 1
                if key not in observation_cache:
                    rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 0, *key]))
                    observation_cache[key] = world.observe(policies[index], n_reps, rng)
                response, noise = observation_cache[key]
                for head, model in enumerate(means):
                    model.update([index], response[head], noise[head])
                risk_model.update_batch(RiskBatchData.from_log_labels(
                    target_X[index:index + 1], response[2:3], n_reps
                ))
                path.append({
                    "index": int(index), "visit": int(key[1]),
                    "response": response.tolist(), "noise": noise.tolist(),
                })
                if step + 1 == reps:
                    mean, covariance = moments()
                    utility, _ = verification_utility(
                        mean, np.einsum("iih->ih", covariance), p
                    )
                    initial_choice = int(np.argmax(utility))
            mean, covariance = moments()
            utility, power = verification_utility(
                mean, np.einsum("iih->ih", covariance), p
            )
            choice = int(np.argmax(utility))
            if choice not in verifier_cache:
                rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 1, choice]))
                verifier_cache[choice] = _terminal_verification(
                    world, policies[choice], p, rng
                )
            verified, upper = verifier_cache[choice]
            task_rows.append({
                "law_seed": law_seed, "noise_seed": noise_seed, "arm": arm_name,
                "initial_choice": initial_choice, "choice": choice,
                "predicted_verifier_power": float(power[choice]),
                "verified": bool(verified), "validation_upper_margin": float(upper),
                "target_search_calls": int(target_points * n_reps),
                "search_path": path,
                "risk_diagnostics": risk_model.diagnostics(),
            })
        truth = np.asarray([world.oracle_moments(policy) for policy in policies])
        truth_reward, _ = verification_utility(truth, np.zeros_like(truth), p)
        true_margin = truth[:, 1] + norm.ppf(1.0 - p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
        for row in task_rows:
            choice = row["choice"]
            row.update({
                "initial_true_expected_reward": float(truth_reward[row["initial_choice"]]),
                "final_true_expected_reward": float(truth_reward[choice]),
                "best_available_reward": float(np.max(truth_reward[true_margin <= 0])) if np.any(true_margin <= 0) else 0.0,
                "true_feasible_candidates": int(np.sum(true_margin <= 0)),
                "true_final_margin": float(true_margin[choice]),
                "false_certificate": bool(row["verified"] and true_margin[choice] > 0),
                "realized_verified_reward": float(row["verified"] * np.clip((5.0 - truth[choice, 0]) / 10.0, 0.0, 1.0)),
            })
        rows.extend(task_rows)
        print(json.dumps({"finished_tasks": seed_offset + 1, "law_seed": law_seed,
                          "cells": len(rows), "local_cpu_seconds": time.process_time() - start}), flush=True)
    summaries = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {
            "cells": len(arm_rows),
            "verified": int(sum(row["verified"] for row in arm_rows)),
            "false_certificates": int(sum(row["false_certificate"] for row in arm_rows)),
            "mean_initial_reward": float(np.mean([row["initial_true_expected_reward"] for row in arm_rows])),
            "mean_final_reward": float(np.mean([row["final_true_expected_reward"] for row in arm_rows])),
            "mean_realized_verified_reward": float(np.mean([row["realized_verified_reward"] for row in arm_rows])),
        }
    return {
        "contract_id": contract["contract_id"],
        "purpose": "development 24-point paired closed loop; not heldout confirmation",
        "confirmation_block_read": False,
        "source_episode_calls": int(archive["source_episode_calls"]),
        "target_search_episode_calls": int(sum(row["target_search_calls"] for row in rows)),
        "rows": rows,
        "summary": summaries,
        "local_cpu_seconds": time.process_time() - start,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_feasible_deployment_v1/closed_loop.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cells": len(result["rows"]), "confirmation_block_read": result["confirmation_block_read"], "summary": result["summary"], "local_cpu_seconds": result["local_cpu_seconds"]}))
