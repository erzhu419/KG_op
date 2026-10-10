"""Paired mean-prior controls on the feasible deployment contract.

The contract may compare source-selected structure, the same-source pooled
full dictionary, and a target-only full dictionary under the same keyed target
and verification streams. The original confirmation interval is never run as
a block by this entry point.
"""

import json
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma
from scipy.stats import chi2, norm, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from core.verification_kg import (
    posterior_covariances, verification_kg_from_moments, verification_utility,
)
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import SourceFunctionPrior
from representation.transferable_spectral import SourceDomainBatch


def _fit_means(archive, source_psi, target_psi, mode):
    models = []
    for head in (0, 1):
        batches = [
            SourceDomainBatch(str(task), source_psi, np.asarray(row["response"])[:, head])
            for task, row in enumerate(archive["source_records"])
        ]
        noises = [np.asarray(row["noise"])[:, head] for row in archive["source_records"]]
        prior = SourceFunctionPrior.fit(batches, noises, target_psi, mode, active_dim=4)
        models.append(prior.target_model(target_psi))
    return models


def _verify(world, policy, p, rng):
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


def run(contract_path=None, output_path=None):
    started = time.process_time()
    contract_path = contract_path or "performance/meta_prior_feasible_source_arm_v1.json"
    contract = json.loads((ROOT / contract_path).read_text())
    base = json.loads((ROOT / "performance/meta_prior_feasible_deployment_contract.json").read_text())
    archive = json.loads((ROOT / contract["source_archive"]).read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    check = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    policies = np.asarray(archive["public_policies"], dtype=float)
    plans = [InventoryPlan(**ctx) for ctx in design["contexts"]]
    target_plan = plans[0]
    initial = np.asarray(check["target_policy_indices"], dtype=int)
    n_reps = int(contract["replicates"])
    p = {"replicates": n_reps, "chance_alpha": float(base["decision_protocol"]["chance_alpha"]),
         "verification_replicates": int(contract["verification_replicates"])}
    assignments = archive["assignments"]
    source_psi = np.asarray([
        plans[row["context"]].trajectory(policies[row["policy_index"]]).coordinates(
            plans[row["context"]].capacity
        ) for row in assignments
    ])
    target_psi = np.asarray([
        target_plan.trajectory(policy).coordinates(target_plan.capacity) for policy in policies
    ])
    source_X = np.asarray(design["risk_design"], dtype=float)
    target_X = np.asarray([
        cumulative_feature_vector(target_plan.trajectory(policy).risk_exposures())
        for policy in policies
    ])
    risk_tasks = [RiskBatchData.from_log_labels(source_X, np.asarray(row["response"])[:, 2], n_reps)
                  for row in archive["source_records"]]
    learned = fit_source_risk_families(risk_tasks, np.vstack([source_X, target_X]))
    pooled = PhysicalRiskPrior(
        learned.coefficients, np.full(len(learned.weights), 1.0 / len(learned.weights)),
        learned.families, learned.family_names,
        {**learned.diagnostics, "mode": "same_source_pooled_factor_families"},
    )
    mode_map = {
        "source_selected_structure": "learned",
        "same_source_full_dictionary": "fixed_source",
        "target_only_full_dictionary": "target_only",
    }
    unknown = [arm for arm in contract["mean_arms"] if arm not in mode_map]
    if unknown:
        raise ValueError(f"unsupported mean arm(s): {unknown}")
    price_range = tuple(base["target_context"]["constraint_price_range"])
    rows = []
    for offset, law_seed in enumerate(contract["target_law_seeds"]):
        noise_seed = int(contract["target_noise_seed_start"]) + offset
        world = InventoryCostSimulator(target_plan, sample_cost_law(int(law_seed), price_range))
        observation_cache, verifier_cache = {}, {}
        task_rows = []
        for arm_name in contract["mean_arms"]:
            # Refit each arm so no posterior state is shared across methods.
            mode = mode_map[arm_name]
            means = _fit_means(archive, source_psi, target_psi, mode)
            risk_model = pooled.target_model()
            path = []
            initial_choice = None

            def moments():
                eta, eta_cov = risk_model.predict(target_X, full_cov=True)
                mean = np.column_stack([
                    means[0].posterior_mean_many(np.arange(len(policies))[:, None]),
                    means[1].posterior_mean_many(np.arange(len(policies))[:, None]), eta,
                ])
                cov = np.concatenate([
                    posterior_covariances(means, np.arange(len(policies))[:, None]),
                    eta_cov[..., None],
                ], axis=2)
                return mean, cov

            for step in range(int(contract["target_search_points"])):
                if step < len(initial):
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
                key = (index, int(sum(1 for row in path if row["index"] == index)))
                if key not in observation_cache:
                    rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 0, *key]))
                    observation_cache[key] = world.observe(policies[index], n_reps, rng)
                response, noise = observation_cache[key]
                for head, model in enumerate(means):
                    model.update([index], response[head], noise[head])
                risk_model.update_batch(RiskBatchData.from_log_labels(
                    target_X[index:index + 1], response[2:3], n_reps
                ))
                path.append({"index": int(index), "visit": int(key[1]),
                             "response": response.tolist(), "noise": noise.tolist()})
                if step + 1 == len(initial):
                    mean, covariance = moments()
                    initial_utility, _ = verification_utility(
                        mean, np.einsum("iih->ih", covariance), p
                    )
                    initial_choice = int(np.argmax(initial_utility))
            mean, covariance = moments()
            utility, power = verification_utility(
                mean, np.einsum("iih->ih", covariance), p
            )
            choice = int(np.argmax(utility))
            if choice not in verifier_cache:
                rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 1, choice]))
                verifier_cache[choice] = _verify(world, policies[choice], p, rng)
            verified, upper = verifier_cache[choice]
            task_rows.append({"law_seed": int(law_seed), "noise_seed": noise_seed,
                "arm": arm_name, "initial_choice": initial_choice, "choice": choice,
                "predicted_verifier_power": float(power[choice]), "verified": bool(verified),
                "validation_upper_margin": float(upper), "target_search_calls": int(len(path) * n_reps),
                "search_path": path, "risk_diagnostics": risk_model.diagnostics()})
        truth = np.asarray([world.oracle_moments(policy) for policy in policies])
        true_utility, _ = verification_utility(truth, np.zeros_like(truth), p)
        margin = truth[:, 1] + norm.ppf(1.0 - p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
        for row in task_rows:
            choice = row["choice"]
            row.update({
                "initial_true_expected_reward": float(true_utility[row["initial_choice"]]),
                "final_true_expected_reward": float(true_utility[choice]),
                "best_available_reward": float(np.max(true_utility[margin <= 0])) if np.any(margin <= 0) else 0.0,
                "true_feasible_candidates": int(np.sum(margin <= 0)),
                "true_final_margin": float(margin[choice]),
                "false_certificate": bool(row["verified"] and margin[choice] > 0),
                "realized_verified_reward": float(row["verified"] * np.clip((5.0 - truth[choice, 0]) / 10.0, 0.0, 1.0)),
            })
        rows.extend(task_rows)
        print(json.dumps({"finished_tasks": offset + 1, "law_seed": int(law_seed),
                          "cells": len(rows), "local_cpu_seconds": time.process_time() - started}), flush=True)
    summaries = {}
    for arm in contract["mean_arms"]:
        ar = [r for r in rows if r["arm"] == arm]
        summaries[arm] = {
            "cells": len(ar), "verified": int(sum(r["verified"] for r in ar)),
            "false_certificates": int(sum(r["false_certificate"] for r in ar)),
            "mean_final_true_reward": float(np.mean([r["final_true_expected_reward"] for r in ar])),
            "mean_realized_verified_reward": float(np.mean([r["realized_verified_reward"] for r in ar])),
        }
    original_confirmation_block_read = bool(
        contract.get("original_confirmation_block_read", contract.get("confirmation_block_read", False))
    )
    independent_block_read = contract.get("contract_id") == "inventory_feasible_independent_confirmation_v2"
    result = {"contract": contract,
              # ``confirmation_block_read`` retains the legacy meaning: the
              # original held-out block.  The v2 contract is a separate block.
              "confirmation_block_read": original_confirmation_block_read,
              "original_confirmation_block_read": original_confirmation_block_read,
              "independent_block_read": independent_block_read,
              "new_source_episode_calls": 0,
              "target_search_episode_calls": int(sum(r["target_search_calls"] for r in rows)),
              "rows": rows, "summary": summaries,
              "local_cpu_seconds": time.process_time() - started,
              "decision_performance_established": False}
    output = ROOT / (output_path or "paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_arm.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": summaries,
                      "original_confirmation_block_read": original_confirmation_block_read,
                      "independent_block_read": independent_block_read,
                      "local_cpu_seconds": result["local_cpu_seconds"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="performance/meta_prior_feasible_source_arm_v1.json")
    parser.add_argument("--output", default="paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_arm.json")
    args = parser.parse_args()
    run(args.contract, args.output)
