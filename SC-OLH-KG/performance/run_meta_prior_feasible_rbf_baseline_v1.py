"""Target-only generic RBF constrained-KG baseline on the feasible contract.

The kernel, lengthscale, and outputscale are frozen before target labels.  No
source response or source risk archive is read by this runner.  It shares the
candidate bank, initial design, paid budget, keyed noise streams, and fresh
928-repetition verifier with the source-structure arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma
from scipy.stats import chi2, norm, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_models import ExactGPSurrogate, ScalarTaskData
from core.verification_kg import verification_kg_from_moments, verification_utility
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


def _verify(world, policy, p, rng):
    samples = world.episodes(policy, p["verification_replicates"], rng)[:, 1]
    n = len(samples)
    df = n - 1
    upper = float(samples.mean() + samples.std(ddof=1) * (
        t.ppf(.975, df) / np.sqrt(n)
        + norm.ppf(1.0 - p["chance_alpha"]) * np.sqrt(df / chi2.ppf(.025, df))))
    return bool(upper <= 0.0), upper


def _state(models, X):
    predictions = [model.predict(X, full_cov=True) for model in models]
    mean = np.column_stack([item[0] for item in predictions])
    covariance = np.stack([item[1] for item in predictions], axis=-1)
    variance = np.maximum(np.einsum("iih->ih", covariance), 0.0)
    return mean, covariance, variance


def run(contract_path=None, output_path=None):
    started = time.process_time()
    contract_path = contract_path or "performance/meta_prior_feasible_rbf_baseline_v1.json"
    contract = json.loads((ROOT / contract_path).read_text())
    base = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    archive = json.loads((ROOT / contract["source_archive"]).read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    check = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    policies = np.asarray(archive["public_policies"], dtype=float)
    plans = [InventoryPlan(**ctx) for ctx in design["contexts"]]
    target_plan = plans[0]
    target_psi = np.asarray([target_plan.trajectory(policy).coordinates(target_plan.capacity)
                             for policy in policies])
    initial = np.asarray(check["target_policy_indices"], dtype=int)
    reps = int(contract["replicates"])
    p = {"replicates": reps, "chance_alpha": float(base["decision_protocol"]["chance_alpha"]),
         "verification_replicates": int(contract["verification_replicates"])}
    rows = []
    for offset, law_seed in enumerate(contract["target_law_seeds"]):
        noise_seed = int(contract["target_noise_seed_start"]) + offset
        world = InventoryCostSimulator(target_plan, sample_cost_law(int(law_seed), tuple(base["target_context"]["constraint_price_range"])))
        observation_cache, verifier_cache = {}, {}
        models = [ExactGPSurrogate(kernel="rbf", lengthscale=float(contract["lengthscale"]),
                                   outputscale=float(contract["outputscale"])) for _ in range(3)]
        obs_X, obs_y, obs_noise = [], [], []
        path = []
        initial_choice = None
        for step in range(int(contract["target_search_points"])):
            if step < len(initial):
                index = int(initial[step])
            else:
                Xobs = np.asarray(obs_X, dtype=float)
                Yobs = np.asarray(obs_y, dtype=float)
                Nobs = np.asarray(obs_noise, dtype=float)
                mean, covariance, variance = _state(models, target_psi)
                fantasy_noise = np.column_stack([
                    np.full(len(policies), .12 ** 2 / reps),
                    np.exp(mean[:, 2] + .5 * variance[:, 2]) / reps,
                    np.full(len(policies), polygamma(1, (reps - 1) / 2)),
                ])
                scores = verification_kg_from_moments(mean, covariance, fantasy_noise, p)
                index = int(np.argmax(scores))
            visit = sum(1 for item in path if item["index"] == index)
            key = (index, visit)
            if key not in observation_cache:
                rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 0, index, visit]))
                observation_cache[key] = world.observe(policies[index], reps, rng)
            response, noise = observation_cache[key]
            obs_X.append(target_psi[index])
            obs_y.append(response)
            obs_noise.append(noise)
            Xobs, Yobs, Nobs = np.asarray(obs_X), np.asarray(obs_y), np.asarray(obs_noise)
            for head, model in enumerate(models):
                model.fit(Xobs, Yobs[:, head], Nobs[:, head])
            path.append({"index": int(index), "visit": int(visit),
                         "response": response.tolist(), "noise": noise.tolist()})
            if step + 1 == len(initial):
                mean, covariance, variance = _state(models, target_psi)
                utility, _ = verification_utility(mean, variance, p)
                initial_choice = int(np.argmax(utility))
        mean, covariance, variance = _state(models, target_psi)
        utility, power = verification_utility(mean, variance, p)
        choice = int(np.argmax(utility))
        if choice not in verifier_cache:
            rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 1, choice]))
            verifier_cache[choice] = _verify(world, policies[choice], p, rng)
        verified, upper = verifier_cache[choice]
        truth = np.asarray([world.oracle_moments(policy) for policy in policies])
        true_utility, _ = verification_utility(truth, np.zeros_like(truth), p)
        margin = truth[:, 1] + norm.ppf(1.0 - p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
        rows.append({
            "law_seed": int(law_seed), "noise_seed": noise_seed,
            "arm": "target_only_rbf_kg", "initial_choice": initial_choice, "choice": choice,
            "predicted_verifier_power": float(power[choice]), "verified": bool(verified),
            "validation_upper_margin": float(upper), "target_search_calls": int(len(path) * reps),
            "search_path": path, "initial_true_expected_reward": float(true_utility[initial_choice]),
            "final_true_expected_reward": float(true_utility[choice]),
            "best_available_reward": float(np.max(true_utility[margin <= 0])) if np.any(margin <= 0) else 0.0,
            "true_feasible_candidates": int(np.sum(margin <= 0)),
            "true_final_margin": float(margin[choice]),
            "false_certificate": bool(verified and margin[choice] > 0),
            "realized_verified_reward": float(verified * np.clip((5.0 - truth[choice, 0]) / 10.0, 0.0, 1.0)),
            "model_diagnostics": [model.diagnostics() for model in models],
        })
        print(json.dumps({"finished_tasks": offset + 1, "law_seed": int(law_seed),
                          "local_cpu_seconds": time.process_time() - started}), flush=True)
    result = {
        "contract": contract, "confirmation_block_read": False,
        "original_confirmation_block_read": False, "independent_block_read": False,
        "rows": rows, "summary": {
            "target_only_rbf_kg": {
                "cells": len(rows), "verified": int(sum(r["verified"] for r in rows)),
                "false_certificates": int(sum(r["false_certificate"] for r in rows)),
                "mean_final_true_reward": float(np.mean([r["final_true_expected_reward"] for r in rows])),
                "mean_realized_verified_reward": float(np.mean([r["realized_verified_reward"] for r in rows])),
            }
        },
        "local_cpu_seconds": time.process_time() - started,
        "decision_performance_established": False,
    }
    output = ROOT / (output_path or "paper_artifacts/meta_prior_feasible_deployment_v1/target_only_rbf_v1.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": result["summary"], "local_cpu_seconds": result["local_cpu_seconds"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="performance/meta_prior_feasible_rbf_baseline_v1.json")
    parser.add_argument("--output", default="paper_artifacts/meta_prior_feasible_deployment_v1/target_only_rbf_v1.json")
    args = parser.parse_args()
    run(args.contract, args.output)
