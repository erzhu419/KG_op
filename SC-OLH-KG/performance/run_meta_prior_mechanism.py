"""Run a frozen, paired source-function-prior mechanism shard (one source seed).

Simulator truth stays in observation generation and post-decision scoring.
Acquisition receives only the three target GPRs and the public protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import digamma, expit, polygamma
from scipy.stats import chi2, norm, spearmanr, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.kg import compute_h  # noqa: E402
from representation.source_function_prior import (  # noqa: E402
    PublicCoordinateSpectralBasis, SourceFunctionPrior, head_batches,
)

PROTOCOL = ROOT / "performance/meta_prior_mechanism_protocol.json"


def dictionary(psi):
    return PublicCoordinateSpectralBasis(max_library_size=64)._library(psi)[0]


def shared_support(seed, dim):
    rng = np.random.default_rng(14001 + seed)
    order = rng.permutation(dim)
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    pair = tuple(sorted(order[:2]))
    # One linear component, one quadratic, and one low-order interaction.
    return [int(order[0]), int(dim + order[2]), int(4 * dim + pairs.index(pair))]


def task_truth(psi, supports, task_seed):
    """Independent signed coefficients move optima while support is shared."""
    rng = np.random.default_rng(task_seed)
    raw = dictionary(psi)
    truth = np.empty((len(psi), 3))
    for head, indices in enumerate(supports):
        magnitude = rng.uniform(0.8, 1.4, len(indices))
        beta = magnitude * rng.choice([-1.0, 1.0], len(indices))
        if head == 2:
            beta *= 0.5
        truth[:, head] = raw[:, indices] @ beta
    truth[:, 1] -= 0.6
    truth[:, 2] -= 2.5
    return truth


def observe(truth, seed, candidate, visit, reps):
    rng = np.random.default_rng(np.random.SeedSequence([seed, candidate, visit, 71]))
    # Cumulative response has one common path shock plus independent local
    # increments; their total conditional variance is exp(log_variance).
    common = rng.normal(size=reps)
    local = rng.normal(size=(reps, 4)).sum(axis=1) / 2.0
    cumulative = truth[1] + np.exp(0.5 * truth[2]) * (
        np.sqrt(0.7) * common + np.sqrt(0.3) * local)
    objective = truth[0] + 0.12 * rng.normal(size=reps)
    sample_var = float(np.var(cumulative, ddof=1))
    df = reps - 1
    log_bias = float(digamma(df / 2.0) + np.log(2.0 / df))
    response = np.array([objective.mean(), cumulative.mean(), np.log(sample_var) - log_bias])
    noise = np.array([0.12 ** 2 / reps, sample_var / reps, float(polygamma(1, df / 2.0))])
    return response, noise


def source_data(seed, protocol):
    dim, count = protocol["psi_dim"], protocol["source_points"]
    support = shared_support(seed, dim)
    supports = [support] * 3
    tasks = []
    for task in range(protocol["source_tasks"]):
        rng = np.random.default_rng(21000 + 100 * seed + task)
        psi = rng.uniform(-2.0, 2.0, (count, dim))
        truth = task_truth(psi, supports, 31000 + 100 * seed + task)
        observations = [observe(row, 41000 + 100 * seed + task, i, 0,
                                protocol["replicates"]) for i, row in enumerate(truth)]
        tasks.append({"psi": psi, "response": np.vstack([r for r, _ in observations]),
                      "noise": np.vstack([v for _, v in observations])})
    return tasks, support


def target_supports(support, seed, regime, dim):
    if regime == "support_mismatch":
        # Shift each underlying coordinate by half the input dimension.
        pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
        a, b = pairs[support[2] - 4 * dim]
        shift = dim // 2
        pair = tuple(sorted(((a + shift) % dim, (b + shift) % dim)))
        changed = [(support[0] + shift) % dim,
                   dim + (support[1] - dim + shift) % dim,
                   4 * dim + pairs.index(pair)]
        return [changed] * 3
    if regime == "high_frequency_risk":
        start = 4 * dim + dim * (dim - 1) // 2
        return [support, support, [start + support[0], start + (support[0] + 1) % dim]]
    if regime == "separate_risk_support":
        alternate = target_supports(support, seed, "support_mismatch", dim)[0]
        return [support, support, alternate]
    return [support] * 3


def moments(models, candidates):
    mean = np.column_stack([m.posterior_mean_many(candidates) for m in models])
    variance = np.column_stack([m.posterior_var_many(candidates) for m in models])
    return mean, variance


def loss_and_derivatives(mean, variance, protocol):
    z = norm.ppf(1.0 - protocol["chance_alpha"])
    sigma = np.exp(0.5 * (mean[:, 2] + 0.5 * variance[:, 2]))
    margin = mean[:, 1] + z * sigma
    temp, penalty = protocol["penalty_temperature"], protocol["risk_penalty"]
    loss = mean[:, 0] + penalty * temp * np.logaddexp(0.0, margin / temp)
    weight = penalty * expit(margin / temp)
    derivative = np.column_stack([np.ones(len(mean)), weight, weight * z * sigma * 0.5])
    return loss, derivative


def kg_scores(models, candidates, protocol):
    """Independent-head KG for a locally linearized penalized decision loss.

    The line-envelope expectation uses the existing exact compute_h. The
    nonlinear joint chance-constrained KG is not claimed to be exact here.
    """
    mean, variance = moments(models, candidates)
    loss, derivative = loss_and_derivatives(mean, variance, protocol)
    reps = protocol["replicates"]
    future_noise = [np.full(len(candidates), 0.12 ** 2 / reps),
                    np.exp(mean[:, 2] + 0.5 * variance[:, 2]) / reps,
                    np.full(len(candidates), float(polygamma(1, (reps - 1) / 2.0)))]
    score = np.zeros(len(candidates))
    for head, model in enumerate(models):
        A = model.augmented_feature_matrix(candidates)
        covariance = A @ model.C @ A.T
        unseen = [tuple(x) not in model.sol_to_idx for x in candidates]
        covariance[np.diag_indices(len(candidates))] += model.lambda_i * np.asarray(unseen)
        scale = np.sqrt(np.maximum(np.diag(covariance) + future_noise[head], 1e-12))
        slope = derivative[:, head, None] * covariance / scale[None, :]
        score += np.array([compute_h(-loss, -slope[:, j]) for j in range(len(candidates))])
    return score


def safety_margin(mean, variance, alpha):
    # Two marginal nominal 95% bounds give a nominal 90% model-based statement.
    bound = norm.ppf(0.95)
    return (mean[:, 1] + bound * np.sqrt(variance[:, 1])
            + norm.ppf(1.0 - alpha) * np.exp(0.5 * (
                mean[:, 2] + bound * np.sqrt(variance[:, 2]))))


def diagnostics(models, candidates, truth, protocol):
    mean, variance = moments(models, candidates)
    error = mean - truth
    z = norm.ppf(0.95)
    alpha = protocol["chance_alpha"]
    margin = truth[:, 1] + norm.ppf(1.0 - alpha) * np.exp(0.5 * truth[:, 2])
    pred_margin = mean[:, 1] + norm.ppf(1.0 - alpha) * np.exp(0.5 * mean[:, 2])
    safe = safety_margin(mean, variance, alpha) <= 0.0
    return {"rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "nominal90_coverage": np.mean(np.abs(error) <= z * np.sqrt(variance), axis=0).tolist(),
            "risk_variance_relative_mae": float(np.mean(np.abs(
                np.exp(mean[:, 2]) - np.exp(truth[:, 2])) / np.exp(truth[:, 2]))),
            "margin_rank_correlation": float(spearmanr(margin, pred_margin).statistic),
            "posterior_safe_count": int(safe.sum()),
            "false_safe_count": int(np.sum(safe & (margin > 0.0)))}


def verify(truth, seed, candidate, reps, alpha):
    rng = np.random.default_rng(np.random.SeedSequence([seed, candidate, 99, 93]))
    values = truth[1] + np.exp(0.5 * truth[2]) * rng.normal(size=reps)
    variance = float(np.var(values, ddof=1))
    df = reps - 1
    mean_upper = values.mean() + t.ppf(0.975, df) * np.sqrt(variance / reps)
    sigma_upper = np.sqrt(df * variance / chi2.ppf(0.025, df))
    upper = float(mean_upper + norm.ppf(1.0 - alpha) * sigma_upper)
    return {"verified": bool(upper <= 0.0), "margin_upper": upper,
            "confidence": 0.95, "replicates": reps}


def run_arm(priors, initial, candidates, candidate_psi, truth, target_seed, protocol,
            on_observation=None, models=None, acquisition=None, recommendation_rule=None):
    if models is None:
        models = [prior.target_model(candidate_psi) for prior in priors]
    if acquisition is None:
        acquisition = kg_scores
    visits = np.zeros(len(candidates), dtype=int)
    queries, first_scores, initial_diagnostics = [], None, None
    for step in range(protocol["target_points"]):
        if step < len(initial):
            index = initial[step]
        else:
            scores = acquisition(models, candidates, protocol)
            if first_scores is None:
                first_scores = scores.tolist()
            index = int(np.argmax(scores))
        response, noise = observe(truth[index], target_seed, index, int(visits[index]),
                                  protocol["replicates"])
        for head, model in enumerate(models):
            model.update(candidates[index], response[head], noise[head])
        if on_observation is not None:
            on_observation(models, step, index, response, noise)
        visits[index] += 1
        queries.append(int(index))
        if step + 1 == len(initial):
            initial_diagnostics = diagnostics(models, candidates, truth, protocol)
    mean, variance = moments(models, candidates)
    margin = safety_margin(mean, variance, protocol["chance_alpha"])
    safe = np.flatnonzero(margin <= 0.0)
    loss, _ = loss_and_derivatives(mean, variance, protocol)
    recommendation = int(safe[np.argmin(mean[safe, 0])]) if len(safe) else int(np.argmin(loss))
    if recommendation_rule is not None:
        recommendation = int(recommendation_rule(mean, variance, protocol))
    true_margin = (truth[:, 1] + norm.ppf(1.0 - protocol["chance_alpha"])
                   * np.exp(0.5 * truth[:, 2]))
    feasible = np.flatnonzero(true_margin <= 0.0)
    true_loss, _ = loss_and_derivatives(truth, np.zeros_like(truth), protocol)
    is_feasible = bool(true_margin[recommendation] <= 0.0)
    result = {"initial": initial_diagnostics,
            "final": diagnostics(models, candidates, truth, protocol),
            "queries": queries, "first_online_kg_scores": first_scores,
            "recommendation": recommendation, "true_feasible": is_feasible,
            "posterior_safe": bool(margin[recommendation] <= 0.0),
            "true_margin": float(true_margin[recommendation]),
            "true_objective": float(truth[recommendation, 0]),
            "feasible_regret": (float(truth[recommendation, 0] - truth[feasible, 0].min())
                                if is_feasible and len(feasible) else None),
            "penalized_regret": float(true_loss[recommendation] - true_loss.min()),
            "verification": verify(truth[recommendation], target_seed, recommendation,
                                   protocol["verification_replicates"], protocol["chance_alpha"]),
            "target_search_calls": protocol["target_points"] * protocol["replicates"],
            "target_verification_calls": protocol["verification_replicates"],
            "priors": [p.diagnostics for p in priors]}
    if on_observation is not None:
        result["target_support"] = [m.adaptive_sparsity_diagnostics() for m in models
                                    if hasattr(m, "adaptive_sparsity_diagnostics")]
    return result


def run_seed(seed, protocol):
    start = time.monotonic()
    tasks, support = source_data(seed, protocol)
    rng = np.random.default_rng(51000 + seed)
    psi = rng.uniform(-2.0, 2.0, (protocol["candidates"], protocol["psi_dim"]))
    candidates = np.arange(len(psi), dtype=int)[:, None]
    initial = rng.choice(len(psi), protocol["initial_points"], replace=False).tolist()
    batches = [head_batches(tasks, head) for head in range(3)]
    noise = [[task["noise"][:, head] for task in tasks] for head in range(3)]
    common_priors = {}
    for arm in protocol["arms"]:
        if arm == "oracle_structure":
            continue
        common_priors[arm] = [SourceFunctionPrior.fit(
            batches[head], noise[head], psi, arm,
            active_dim=protocol["learned_active_dim_per_head"]) for head in range(3)]
    rows = []
    for regime_index, regime in enumerate(protocol["regimes"]):
        supports = target_supports(support, seed, regime, protocol["psi_dim"])
        target_seed = 61000 + seed * 100 + regime_index
        truth = task_truth(psi, supports, target_seed)
        oracle_priors = [SourceFunctionPrior.fit(batches[head], noise[head], psi,
                          "oracle_structure", oracle_indices=supports[head]) for head in range(3)]
        # No target truth is present in either prior-fitting input.
        for arm in protocol["arms"]:
            priors = oracle_priors if arm == "oracle_structure" else common_priors[arm]
            result = run_arm(priors, initial, candidates, psi, truth, target_seed, protocol)
            source_calls = (0 if arm == "target_only" else protocol["source_tasks"]
                            * protocol["source_points"] * protocol["replicates"])
            result.update({"seed": seed, "regime": regime, "arm": arm,
                           "initial_design": initial, "oracle_support": supports,
                           "source_calls": source_calls,
                           "source_calls_amortized_four_targets": source_calls / len(protocol["regimes"])})
            rows.append(result)
            print(json.dumps({"seed": seed, "regime": regime, "arm": arm,
                              "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    return {"protocol": protocol, "rows": rows,
            "elapsed_seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.seed not in protocol["seeds"]:
        raise ValueError("seed is outside the frozen protocol")
    result = run_seed(args.seed, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
