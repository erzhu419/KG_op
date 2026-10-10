"""Risk-only development closed loop for the multi-context contract."""

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


def _fit_mean_models(archive, source_psi, target_psi):
    models = []
    for head in (0, 1):
        batches = [
            SourceDomainBatch(str(task), source_psi, np.asarray(row["response"])[:, head])
            for task, row in enumerate(archive["source_records"])
        ]
        noises = [np.asarray(row["noise"])[:, head] for row in archive["source_records"]]
        prior = SourceFunctionPrior.fit(batches, noises, target_psi, "learned", active_dim=4)
        models.append(prior.target_model(target_psi))
    return models


def _verify(world, policy, p, rng):
    values = world.episodes(policy, p["verification_replicates"], rng)[:, 1]
    n = len(values)
    df = n - 1
    upper = float(values.mean() + values.std(ddof=1) * (
        t.ppf(.975, df) / np.sqrt(n)
        + norm.ppf(1.0 - p["chance_alpha"]) * np.sqrt(df / chi2.ppf(.025, df))
    ))
    return bool(upper <= 0.0), upper


def run():
    start = time.process_time()
    contract = json.loads((ROOT / "performance/meta_prior_risk_identifiability_contract_v1.json").read_text())
    archive = json.loads((ROOT / contract["source_archive_output"]).read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    base = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    policies = np.asarray(archive["public_policies"], dtype=float)
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    target_plan = plans[0]
    assignments = archive["assignments"]
    source_psi = np.asarray([
        plans[row["context"]].trajectory(policies[row["policy_index"]]).coordinates(
            plans[row["context"]].capacity
        )
        for row in assignments
    ])
    target_psi = np.asarray([
        target_plan.trajectory(policy).coordinates(target_plan.capacity)
        for policy in policies
    ])
    source_X = np.asarray([row["risk_features"] for row in assignments], dtype=float)
    target_X = np.asarray([
        cumulative_feature_vector(target_plan.trajectory(policy).risk_exposures())
        for policy in policies
    ])
    reps = int(contract["replicates_per_site"])
    initial = np.asarray(base["target_policy_indices"], dtype=int)
    p = {
        "replicates": reps,
        "chance_alpha": 0.05,
        "verification_replicates": 928,
    }
    tasks = [RiskBatchData.from_log_labels(source_X, np.asarray(row["response"])[:, 2], reps)
             for row in archive["source_records"]]
    learned = fit_source_risk_families(tasks, np.vstack([source_X, target_X]))
    pooled = PhysicalRiskPrior(
        learned.coefficients,
        np.full(len(learned.weights), 1.0 / len(learned.weights)),
        learned.families,
        learned.family_names,
        {"mode": "same_source_pooled_factor_families"},
    )
    arms = {"observable_pooled_risk": pooled, "observable_learned_risk": learned}
    rows = []
    verifier_cache = {}
    price_range = tuple(contract["constraint_price_range"])
    for offset, law_seed in enumerate(contract["development_target_law_seeds"]):
        law_seed = int(law_seed)
        noise_seed = 97500 + offset
        world = InventoryCostSimulator(target_plan, sample_cost_law(law_seed, price_range))
        observation_cache = {}
        task_rows = []
        for arm_name, risk_prior in arms.items():
            means = _fit_mean_models(archive, source_psi, target_psi)
            risk_model = risk_prior.target_model()
            visits = np.zeros(len(policies), dtype=int)
            path = []
            initial_choice = None

            def moments():
                eta, eta_cov = risk_model.predict(target_X, full_cov=True)
                mean = np.column_stack([
                    means[0].posterior_mean_many(np.arange(len(policies))[:, None]),
                    means[1].posterior_mean_many(np.arange(len(policies))[:, None]), eta,
                ])
                covariance = np.concatenate([
                    posterior_covariances(means, np.arange(len(policies))[:, None]),
                    eta_cov[..., None],
                ], axis=2)
                return mean, covariance

            for step in range(24):
                if step < len(initial):
                    index = int(initial[step])
                else:
                    mean, covariance = moments()
                    variance = np.einsum("iih->ih", covariance)
                    fantasy_noise = np.column_stack([
                        np.full(len(policies), .12 ** 2 / reps),
                        np.exp(mean[:, 2] + .5 * variance[:, 2]) / reps,
                        np.full(len(policies), polygamma(1, (reps - 1) / 2)),
                    ])
                    index = int(np.argmax(verification_kg_from_moments(mean, covariance, fantasy_noise, p)))
                key = (index, int(visits[index]))
                visits[index] += 1
                if key not in observation_cache:
                    rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 0, *key]))
                    observation_cache[key] = world.observe(policies[index], reps, rng)
                response, noise = observation_cache[key]
                for head, model in enumerate(means):
                    model.update([index], response[head], noise[head])
                risk_model.update_batch(RiskBatchData.from_log_labels(target_X[index:index + 1], response[2:3], reps))
                path.append({"index": int(index), "visit": int(key[1]), "response": response.tolist(), "noise": noise.tolist()})
                if step + 1 == len(initial):
                    mean, covariance = moments()
                    utility, _ = verification_utility(mean, np.einsum("iih->ih", covariance), p)
                    initial_choice = int(np.argmax(utility))
            mean, covariance = moments()
            utility, power = verification_utility(mean, np.einsum("iih->ih", covariance), p)
            choice = int(np.argmax(utility))
            if choice not in verifier_cache:
                verifier_cache[choice] = _verify(
                    world, policies[choice], p,
                    np.random.default_rng(np.random.SeedSequence([noise_seed, 1, choice]))
                )
            verified, upper = verifier_cache[choice]
            task_rows.append({
                "law_seed": law_seed, "noise_seed": noise_seed, "arm": arm_name,
                "initial_choice": initial_choice, "choice": choice,
                "predicted_verifier_power": float(power[choice]), "verified": bool(verified),
                "validation_upper_margin": float(upper), "search_path": path,
                "target_search_calls": 24 * reps,
                "risk_diagnostics": risk_model.diagnostics(),
            })
        truth = np.asarray([world.oracle_moments(policy) for policy in policies])
        reward, _ = verification_utility(truth, np.zeros_like(truth), p)
        margin = truth[:, 1] + norm.ppf(.95) * np.exp(.5 * truth[:, 2])
        for row in task_rows:
            choice = row["choice"]
            row.update({
                "initial_true_expected_reward": float(reward[row["initial_choice"]]),
                "final_true_expected_reward": float(reward[choice]),
                "best_available_reward": float(np.max(reward[margin <= 0])) if np.any(margin <= 0) else 0.0,
                "true_feasible_candidates": int(np.sum(margin <= 0)),
                "true_final_margin": float(margin[choice]),
                "false_certificate": bool(row["verified"] and margin[choice] > 0),
                "realized_verified_reward": float(row["verified"] * np.clip((5.0 - truth[choice, 0]) / 10.0, 0.0, 1.0)),
            })
        rows.extend(task_rows)
        print(json.dumps({"finished_tasks": offset + 1, "law_seed": law_seed, "cells": len(rows),
                          "local_cpu_seconds": time.process_time() - start}), flush=True)
    summary = {}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "cells": len(selected), "verified": int(sum(row["verified"] for row in selected)),
            "false_certificates": int(sum(row["false_certificate"] for row in selected)),
            "mean_initial_reward": float(np.mean([row["initial_true_expected_reward"] for row in selected])),
            "mean_final_reward": float(np.mean([row["final_true_expected_reward"] for row in selected])),
            "mean_realized_verified_reward": float(np.mean([row["realized_verified_reward"] for row in selected])),
        }
    return {
        "contract_id": contract["contract_id"],
        "purpose": "risk-only development closed loop; confirmation block not read",
        "confirmation_block_read": False,
        "source_episode_calls": int(archive["source_episode_calls"]),
        "rows": rows,
        "summary": summary,
        "local_cpu_seconds": time.process_time() - start,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_risk_identifiability_v1/closed_loop.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cells": len(result["rows"]), "summary": result["summary"],
                      "confirmation_block_read": result["confirmation_block_read"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}))
