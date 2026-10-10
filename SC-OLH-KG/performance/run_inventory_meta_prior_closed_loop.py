"""Bounded prospective operational loop; preserve every outcome and source cost."""

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
from core.verification_kg import (objective_quality, particle_pass_probability, posterior_covariances,
                                  verification_kg_from_moments, verification_utility)
from problems.inventory_cost_plan import InventoryPlan, InventoryCostSimulator, sample_cost_law
from representation.conditional_function_prior import conditional_selection
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.transfer_mixture import TransferMixtureGPR


def summarize(rows, arms):
    return {arm: {"cells": sum(row["arm"] == arm for row in rows),
        "verified": sum(row["verified"] for row in rows if row["arm"] == arm),
        "false_certificates": sum(row["false_certificate"] for row in rows if row["arm"] == arm),
        "mean_initial_reward": float(np.mean([row["initial_true_expected_reward"] for row in rows if row["arm"] == arm])),
        "mean_final_reward": float(np.mean([row["final_true_expected_reward"] for row in rows if row["arm"] == arm])),
        "mean_realized_reward": float(np.mean([row["realized_verified_reward"] for row in rows if row["arm"] == arm]))}
        for arm in arms if any(row["arm"] == arm for row in rows)}


def run(p):
    start = time.process_time()
    source = json.loads((ROOT / p["source_records"]).read_text())
    s = source["protocol"]
    old = json.loads((ROOT / s["existing_records"]).read_text())
    design = json.loads((ROOT / s["source_design"]).read_text())
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    policies, initial = np.asarray(old["public_policies"]), old["target_policy_indices"]
    psi = plans[0].coordinates_many(policies)
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    Xs = np.asarray(design["risk_design"])
    source_psi = np.array([plans[row["context"]].trajectory(policies[row["policy_index"]]).coordinates(plans[row["context"]].capacity)
                           for row in design["assignments"]])
    data = [{"psi": source_psi, "response": np.asarray(task["response"]), "noise": np.asarray(task["noise"])} for task in source["source_records"]]
    full, structural = [], []
    for h in range(2):
        batches, noises = head_batches(data, h), [task["noise"][:, h] for task in data]
        full.append(SourceFunctionPrior.fit(batches, noises, psi, "fixed_source"))
        support, history = conditional_selection(batches, noises, FixedFunctionBasis(psi), 4)
        prior = SourceFunctionPrior.from_basis(batches, noises, FixedFunctionBasis(psi, support), "conditional_hard")
        prior.diagnostics.update(selected_indices=support, selection_cv=history)
        structural.append(prior)
    risk = fit_source_risk_families([RiskBatchData.from_log_labels(Xs, task["response"][:, 2], p["replicates"]) for task in data],
                                   np.vstack([Xs, X]), **s["source_risk_distribution"])
    assert risk.diagnostics["source_optimizer_converged"]
    pooled = PhysicalRiskPrior(risk.coefficients, np.full(len(risk.weights), 1. / len(risk.weights)), risk.families, risk.family_names, {})
    candidates = np.arange(len(policies))[:, None]
    risk_variances = X @ risk.coefficients.T
    rows, actually_simulated = [], 0
    complete = True
    for law_seed, noise_seed in zip(p["target_law_seeds"], p["target_noise_seeds"]):
        world = InventoryCostSimulator(plans[0], sample_cost_law(law_seed))
        cache, verification_cache = {}, {}
        task_rows = []
        for arm in p["arms"]:
            if time.process_time() - start >= p["local_cpu_limit_seconds"]:
                complete = False
                break
            means = ([prior.target_model(psi) for prior in full] if arm == "same_source_general" else
                     [TransferMixtureGPR([structural[h], full[h]], psi, psi) for h in range(2)])
            risk_model = (risk if arm == "structural_complete" else pooled).target_model()
            visits = np.zeros(len(policies), dtype=int)
            path, initial_choice = [], None

            def moments():
                eta, eta_cov = risk_model.predict(X, full_cov=True)
                m = np.column_stack([*[model.posterior_mean_many(candidates) for model in means], eta])
                cov = np.concatenate([posterior_covariances(means, candidates), eta_cov[..., None]], axis=2)
                return m, cov

            for step in range(p["target_points"]):
                if step < p["initial_points"]:
                    index = initial[step]
                else:
                    m, cov = moments()
                    variance = np.einsum('iih->ih', cov)
                    noise = np.column_stack([np.full(len(X), .12 ** 2 / p["replicates"]),
                        np.exp(m[:, 2] + .5 * variance[:, 2]) / p["replicates"],
                        np.full(len(X), polygamma(1, (p["replicates"] - 1) / 2))])
                    index = int(np.argmax(verification_kg_from_moments(m, cov, noise, p)))
                key = (index, int(visits[index]))
                visits[index] += 1
                if key not in cache:
                    rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 0, *key]))
                    cache[key] = world.observe(policies[index], p["replicates"], rng)
                y, noise = cache[key]
                for h, model in enumerate(means):
                    model.update([index], y[h], noise[h])
                risk_model.update_batch(RiskBatchData.from_log_labels(X[index:index+1], y[2:3], p["replicates"]))
                path.append({"index": int(index), "visit": key[1], "response": y.tolist(), "noise": noise.tolist()})
                if step + 1 == p["initial_points"]:
                    m, cov = moments()
                    utility, _ = verification_utility(m, np.einsum('iih->ih', cov), p)
                    initial_choice = int(np.argmax(utility))
            m, cov = moments()
            variance = np.einsum('iih->ih', cov)
            utility, power = verification_utility(m, variance, p)
            choice = int(np.argmax(utility))
            particle = objective_quality(m, variance) * particle_pass_probability(m[:, 1], variance[:, 1],
                risk_variances, np.exp(risk_model.log_weights), p)
            alternative = int(np.argmax(particle))
            if choice not in verification_cache:
                rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 1, choice]))
                verification_cache[choice] = world.episodes(policies[choice], p["verification_replicates"], rng)[:, 1]
            validation = verification_cache[choice]
            n, df = len(validation), len(validation) - 1
            sample_mean, sample_sd = float(validation.mean()), float(validation.std(ddof=1))
            upper = sample_mean + sample_sd * (t.ppf(.975, df) / np.sqrt(n)
                    + norm.ppf(1 - p["chance_alpha"]) * np.sqrt(df / chi2.ppf(.025, df)))
            total = len(path) * p["replicates"] + len(validation)
            assert total == p["total_target_calls_per_arm"]
            assert risk_model.paid_episodes == p["target_points"] * p["replicates"]
            task_rows.append({"law_seed": law_seed, "noise_seed": noise_seed, "arm": arm,
                "initial_choice": initial_choice, "choice": choice, "particle_choice": alternative,
                "terminal_particle_choice_gap": float(particle[alternative] - particle[choice]),
                "predicted_verifier_power": float(power[choice]), "verified": bool(upper <= 0),
                "validation_upper_margin": float(upper), "target_calls": total, "search_path": path,
                "risk_diagnostics": risk_model.diagnostics(),
                "mean_model_weights": [model.diagnostics() for model in means] if arm != "same_source_general" else []})
            print(json.dumps({"finished_cells": len(rows) + len(task_rows), "law_seed": law_seed,
                              "arm": arm, "local_cpu_seconds": time.process_time() - start}), flush=True)
        # Evaluate private truth after all recommendations and validations for
        # this task. Oracle rewards never enter learning or decision calls.
        truth = np.array([world.oracle_moments(x) for x in policies])
        reward, _ = verification_utility(truth, np.zeros_like(truth), p)
        margin = truth[:, 1] + norm.ppf(1 - p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
        for row in task_rows:
            choice = row["choice"]
            row.update(initial_true_expected_reward=float(reward[row["initial_choice"]]),
                final_true_expected_reward=float(reward[choice]), best_available_reward=float(reward.max()),
                true_feasible_candidates=int(np.sum(margin <= 0)), true_final_margin=float(margin[choice]),
                false_certificate=bool(row["verified"] and margin[choice] > 0),
                realized_verified_reward=float(row["verified"] * np.clip((5 - truth[choice, 0]) / 10, 0, 1)))
        rows.extend(task_rows)
        actually_simulated += world.episode_calls
        if not complete:
            break
    summary = summarize(rows, p["arms"])
    return {"protocol": p, "complete": complete, "rows": rows, "summary": summary,
        "source_mean_info": [prior.diagnostics for prior in structural],
        "source_risk_info": risk.diagnostics, "new_source_calls": 0,
        "allocated_target_calls": sum(row["target_calls"] for row in rows),
        "actually_simulated_episode_calls_with_paired_reuse": actually_simulated,
        "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    p = json.loads((ROOT / "performance/inventory_meta_prior_closed_loop_protocol.json").read_text())
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/closed_loop.json"
    previous = None
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["complete"]:
            print("Existing complete closed-loop results retained; no observations regenerated.")
            sys.exit(0)
        done = {row["law_seed"] for row in previous["rows"]}
        assert all({row["arm"] for row in previous["rows"] if row["law_seed"] == seed} == set(p["arms"]) for seed in done)
        remaining = [(law, noise) for law, noise in zip(p["target_law_seeds"], p["target_noise_seeds"]) if law not in done]
        segment = output.with_name("closed_loop_segment1.json")
        if not segment.exists():
            segment.write_text(output.read_text())
        next_p = {**p, "target_law_seeds": [law for law, _ in remaining], "target_noise_seeds": [noise for _, noise in remaining]}
        result = run(next_p)
        result["protocol"] = p
        result["rows"] = previous["rows"] + result["rows"]
        result["summary"] = summarize(result["rows"], p["arms"])
        result["execution_segment_cpu_seconds"] = [previous["local_cpu_seconds"], result["local_cpu_seconds"]]
        result["local_cpu_seconds"] += previous["local_cpu_seconds"]
        result["allocated_target_calls"] += previous["allocated_target_calls"]
        result["actually_simulated_episode_calls_with_paired_reuse"] += previous["actually_simulated_episode_calls_with_paired_reuse"]
        result["complete"] = {(row["law_seed"], row["arm"]) for row in result["rows"]} == {
            (seed, arm) for seed in p["target_law_seeds"] for arm in p["arms"]}
    else:
        result = run(p)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": result["complete"], "summary": result["summary"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}), flush=True)
