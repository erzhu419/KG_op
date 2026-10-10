"""Bounded prospective joint-KG pilot, with matched total target budgets."""

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
from core.joint_verification_kg import budgeted_joint_kg, joint_verification_kg, joint_verification_utility
from core.verification_kg import objective_quality, posterior_covariances, verification_utility
from performance.check_inventory_mean_interface import source_data
from problems.inventory_cost_plan import InventoryCostSimulator, sample_cost_law
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_mean_risk_prior import JointMeanRiskPosterior
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import SourceFunctionPrior


def prepare():
    folder = ROOT / "paper_artifacts/inventory_physical_risk_20261004"
    group = json.loads((folder / "mean_posterior_group.json").read_text())
    source, design, policies, plans, data = source_data(group["protocol"]["source_records"])
    old = json.loads((ROOT / source["protocol"]["existing_records"]).read_text())
    source_psi, psi = data[0]["psi"], plans[0].coordinates_many(policies)
    means = []
    for spec in group["all_source_priors"]:
        assert spec["discrepancy_variance"] == 0.
        means.append(SourceFunctionPrior(InventoryFunctionBasis(source_psi, spec["diagnostics"]["selected_indices"]),
            np.asarray(spec["mean"]), np.asarray(spec["covariance"]), 0., spec["diagnostics"]))
    Z = np.column_stack([np.ones(len(psi)), means[1].basis.transform(psi)])
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    Xs = np.asarray(design["risk_design"])
    risk = fit_source_risk_families([RiskBatchData.from_log_labels(Xs, t["response"][:, 2], 12) for t in data],
                                   np.vstack([Xs, X]), **source["protocol"]["source_risk_distribution"])
    pooled = PhysicalRiskPrior(risk.coefficients, np.full(len(risk.weights), 1. / len(risk.weights)), risk.families, risk.family_names, {})
    return {"means": means, "risk": risk, "pooled": pooled, "X": X, "Z": Z, "psi": psi,
            "policies": policies, "plan": plans[0], "initial": old["target_policy_indices"],
            "risk_variances": X @ risk.coefficients.T, "candidates": np.arange(len(policies))[:, None]}


def models(prepared, prior_name):
    f = prepared["means"][0].target_model(prepared["psi"])
    g = prepared["means"][1]
    joint = JointMeanRiskPosterior(g.mean, g.covariance, prepared[prior_name])
    return f, joint


def update(prepared, f, joint, index, response, noise, replicates):
    f.update([index], response[0], noise[0])
    joint.update_batch(prepared["Z"][index:index+1], response[1:2],
        RiskBatchData.from_log_labels(prepared["X"][index:index+1], response[2:3], replicates))


def moments(prepared, f, joint, budget):
    f_mean = f.posterior_mean_many(prepared["candidates"])
    f_cov = posterior_covariances([f], prepared["candidates"])[..., 0]
    pair_mean, pair_cov = joint.predict(prepared["Z"], prepared["X"], full_cov=True)
    # All paid learning labels remain residual cost relative to source budget
    # one. Changing a declared cost budget is a known location shift only.
    pair_mean[:, 0] += prepared["plan"].cost_budget - budget
    return f_mean, f_cov, pair_mean, pair_cov


def recommend(state, protocol):
    f_mean, f_cov, pair_mean, pair_cov = state
    ix = np.arange(len(f_mean))
    utility, power = joint_verification_utility(f_mean, np.diag(f_cov), pair_mean, pair_cov[ix, :, ix, :], protocol)
    return int(np.argmax(utility)), utility, power


def measurement_noise(prepared, joint, replicates):
    expected_variance = prepared["risk_variances"] @ np.exp(joint.log_weights)
    return np.column_stack([np.full(len(expected_variance), .12 ** 2 / replicates), expected_variance / replicates,
                            np.full(len(expected_variance), polygamma(1, (replicates - 1) / 2))])


def summarize(rows, p):
    result = []
    for budget in p["cost_budgets"]:
        for arm in p["arms"]:
            subset = [r for r in rows if r["cost_budget"] == budget and r["arm"] == arm]
            if not subset:
                continue
            result.append({"cost_budget": budget, "arm": arm, "cells": len(subset),
                "verified": sum(r["verified"] for r in subset), "false_certificates": sum(r["false_certificate"] for r in subset),
                "mean_true_expected_reward": float(np.mean([r["true_expected_reward"] for r in subset])),
                "mean_initial_true_expected_reward": float(np.mean([r["initial_true_expected_reward"] for r in subset])),
                "mean_realized_reward": float(np.mean([r["realized_verified_reward"] for r in subset]))})
    return result


def run(p, output):
    start = time.process_time()
    prepared = prepare()
    if output.exists():
        result = json.loads(output.read_text())
        assert result["protocol"] == p
        if result["complete"]:
            return result
    else:
        result = {"protocol": p, "rows": [], "complete": False, "new_source_calls": 0,
                  "actually_simulated_episode_calls": 0, "execution_segment_cpu_seconds": []}
    done = {(r["law_seed"], r["cost_budget"], r["arm"]) for r in result["rows"]}
    # Cache within a law, across budgets and arms. Already-finished laws are
    # skipped on resume; all cells of a law form one persistent segment unit.
    for law_seed, noise_seed in zip(p["target_law_seeds"], p["target_noise_seeds"]):
        expected = {(law_seed, budget, arm) for budget in p["cost_budgets"] for arm in p["arms"]}
        if expected.issubset(done):
            continue
        assert not (expected & done), "Resume only complete task-law blocks"
        if time.process_time() - start >= p["local_cpu_limit_seconds"]:
            break
        world = InventoryCostSimulator(prepared["plan"], sample_cost_law(law_seed))
        observations, validation_cache, law_rows = {}, {}, []
        for budget in p["cost_budgets"]:
            for arm, spec in p["arms"].items():
                cell_start = time.process_time()
                f, joint = models(prepared, spec["risk_prior"])
                visits, path = np.zeros(len(prepared["policies"]), dtype=int), []
                budget_aware = spec.get("allocation") == "one_step_net_value"
                initial_verification = (p["total_target_calls_per_arm"] - p["initial_points"] * p["replicates"]
                                        if budget_aware else spec["verification_replicates"])
                active_protocol = {**p, "verification_replicates": initial_verification}
                allocation_decisions = []
                initial_state = None
                initial_choice = None
                for step in range(spec["target_points"]):
                    if step < p["initial_points"]:
                        index = prepared["initial"][step]
                    else:
                        state = moments(prepared, f, joint, budget)
                        if budget_aware:
                            remaining = p["total_target_calls_per_arm"] - len(path) * p["replicates"]
                            scores = budgeted_joint_kg(*state, measurement_noise(prepared, joint, p["replicates"]),
                                p, remaining, order=p["quadrature_order"])
                            allocation_decisions.append({"paid_batches": len(path), "remaining_calls": remaining,
                                "maximum_net_value": float(scores.max()), "action": int(np.argmax(scores)),
                                "stop": bool(scores.max() <= 0.)})
                            if scores.max() <= 0.:
                                break
                        else:
                            scores = joint_verification_kg(*state, measurement_noise(prepared, joint, p["replicates"]),
                                                          active_protocol, order=p["quadrature_order"])
                        index = int(np.argmax(scores))
                    key = (int(index), int(visits[index]))
                    visits[index] += 1
                    if key not in observations:
                        rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 0, *key]))
                        observations[key] = world.observe(prepared["policies"][index], p["replicates"], rng)
                    response, noise = observations[key]
                    update(prepared, f, joint, index, response, noise, p["replicates"])
                    path.append({"index": int(index), "visit": key[1], "source_budget_response": response.tolist(), "noise": noise.tolist()})
                    if step + 1 == p["initial_points"]:
                        initial_state = moments(prepared, f, joint, budget)
                        initial_choice = recommend(initial_state, active_protocol)[0]
                n = (p["total_target_calls_per_arm"] - len(path) * p["replicates"]
                     if budget_aware else spec["verification_replicates"])
                active_protocol = {**p, "verification_replicates": n}
                initial_allocation_choice = initial_choice
                initial_choice = recommend(initial_state, active_protocol)[0]
                state = moments(prepared, f, joint, budget)
                choice, utility, power = recommend(state, active_protocol)
                conditional_mean, conditional_variance = joint.conditional_moments(prepared["Z"])
                from core.verification_kg import particle_pass_probability
                exact_power = particle_pass_probability(conditional_mean + prepared["plan"].cost_budget - budget,
                    conditional_variance, prepared["risk_variances"], np.exp(joint.log_weights), active_protocol)
                exact_utility = objective_quality(state[0][:, None], np.diag(state[1])[:, None]) * exact_power
                particle_choice = int(np.argmax(exact_utility))
                validation_key = (choice, n)
                if validation_key not in validation_cache:
                    rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 1, choice, n]))
                    validation_cache[validation_key] = world.episodes(prepared["policies"][choice], n, rng)[:, 1]
                validation = validation_cache[validation_key] + prepared["plan"].cost_budget - budget
                df = n - 1
                upper = float(validation.mean() + validation.std(ddof=1) * (t.ppf(.975, df) / np.sqrt(n)
                    + norm.ppf(1-p["chance_alpha"]) * np.sqrt(df / chi2.ppf(.025, df))))
                calls = len(path) * p["replicates"] + n
                assert calls == p["total_target_calls_per_arm"]
                assert joint.paid_episodes == len(path) * p["replicates"]
                law_rows.append({"law_seed": law_seed, "noise_seed": noise_seed, "cost_budget": budget, "arm": arm,
                    "initial_choice": initial_choice, "choice": choice, "particle_choice": particle_choice,
                    "particle_utility_gap": float(exact_utility[particle_choice] - exact_utility[choice]),
                    "predicted_verifier_power": float(power[choice]), "particle_verifier_power": float(exact_power[choice]),
                    "verified": bool(upper <= 0), "validation_upper_margin": upper, "search_path": path,
                    "verification_replicates": n, "allocation_decisions": allocation_decisions,
                    "initial_allocation_choice": initial_allocation_choice, "initial_verification_replicates": initial_verification,
                    "target_calls": calls, "risk_diagnostics": joint.diagnostics(), "local_cpu_seconds": time.process_time() - cell_start})
        # Private truth is used only after the law's complete decisions.
        truth_base = np.array([world.oracle_moments(x) for x in prepared["policies"]])
        for row in law_rows:
            truth = truth_base.copy()
            truth[:, 1] += prepared["plan"].cost_budget - row["cost_budget"]
            protocol = {**p, "verification_replicates": row["verification_replicates"]}
            reward, _ = verification_utility(truth, np.zeros_like(truth), protocol)
            initial_reward, _ = verification_utility(truth, np.zeros_like(truth),
                {**p, "verification_replicates": row["initial_verification_replicates"]})
            margin = truth[:, 1] + norm.ppf(1-p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
            row.update(true_expected_reward=float(reward[row["choice"]]), initial_true_expected_reward=float(reward[row["initial_choice"]]),
                initial_allocated_true_expected_reward=float(initial_reward[row["initial_allocation_choice"]]),
                best_available_reward=float(reward.max()), true_feasible_candidates=int(np.sum(margin <= 0)),
                true_final_margin=float(margin[row["choice"]]), false_certificate=bool(row["verified"] and margin[row["choice"]] > 0),
                realized_verified_reward=float(row["verified"] * np.clip((5-truth[row["choice"], 0])/10, 0, 1)))
        result["rows"].extend(law_rows)
        result["actually_simulated_episode_calls"] += world.episode_calls
        done.update(expected)
        result["summary"] = summarize(result["rows"], p)
        result["complete"] = len(done) == len(p["target_law_seeds"]) * len(p["cost_budgets"]) * len(p["arms"])
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"completed_laws": len({r["law_seed"] for r in result["rows"]}),
            "completed_cells": len(result["rows"]), "segment_cpu_seconds": time.process_time()-start}), flush=True)
    result["execution_segment_cpu_seconds"].append(time.process_time() - start)
    result["local_cpu_seconds"] = sum(result["execution_segment_cpu_seconds"])
    result["allocated_target_calls"] = sum(r["target_calls"] for r in result["rows"])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = json.loads((ROOT / "performance/inventory_joint_kg_protocol.json").read_text())
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/joint_kg.json"
    result = run(p, output)
    print(json.dumps({"complete": result["complete"], "summary": result["summary"], "local_cpu_seconds": result["local_cpu_seconds"]}))
