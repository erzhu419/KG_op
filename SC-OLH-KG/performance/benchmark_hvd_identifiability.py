#!/usr/bin/env python3
"""Identify when cumulative HVD adds value beyond pooled variance.

This benchmark isolates the variance model from objective search.  It sweeps
shared-shock strength and within-policy replication, fits each HVD variant to
ordinary noisy simulator observations, and audits variance calibration and
chance-feasibility certificates on a held-out policy grid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.rzdt import FactorShockStatePolicyRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from variance.orthogonal_hvd import OrthogonalHVD  # noqa: E402


MODE_CONFIG = {
    "pooled": ("pooled", False),
    "class": ("class", False),
    "orthogonal_pointwise": ("orthogonal", False),
    "factor_pointwise": ("factor", False),
    "factor_cumulative": ("factor", True),
}


def _policy(problem, u, q, spread, phase=0.0):
    d = int(problem.d)
    if d <= 1:
        return problem.continuous_to_int([u])
    positions = (np.arange(d - 1, dtype=float) + 0.5) / float(d - 1)
    tail = q + np.sqrt(2.0) * spread * np.cos(
        2.0 * np.pi * positions + float(phase))
    return problem.continuous_to_int(np.r_[u, np.clip(tail, 0.0, 1.0)])


def training_policies(problem, n, rng):
    rows = []
    seen = set()
    while len(rows) < int(n):
        row = _policy(
            problem,
            rng.uniform(0.02, 0.98),
            rng.uniform(0.15, 0.92),
            rng.uniform(0.0, 0.24),
            rng.uniform(0.0, 2.0 * np.pi),
        )
        if row not in seen:
            rows.append(row)
            seen.add(row)
    return rows


def evaluation_policies(problem):
    rows = []
    for u in np.linspace(0.04, 0.96, 7):
        for q in np.linspace(0.20, 0.90, 8):
            for spread in (0.0, 0.06, 0.14, 0.22):
                rows.append(_policy(problem, u, q, spread))
    return list(dict.fromkeys(rows))


def _safe_spearman(first, second):
    if len(first) < 3 or np.std(first) <= 1e-15 or np.std(second) <= 1e-15:
        return None
    value = spearmanr(first, second).statistic
    return None if not np.isfinite(value) else float(value)


def run_cell(args):
    mode, use_provider = MODE_CONFIG[args.mode]
    base = FactorShockStatePolicyRZDT1(
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
        shared_shock_scale=args.shock_scale,
    )
    problem = ScalarizedProblem(base, weights=(0.5, 0.5))
    rng = np.random.default_rng(args.seed)
    train = training_policies(problem, args.n_train, rng)
    hvd = OrthogonalHVD(
        mode=mode,
        n_outputs=2,
        use_cumulative_provider=use_provider,
        activation_min_records=min(args.activation_min_records, args.n_train),
        residual_tail_delta=args.delta,
        ridge_alpha=args.ridge,
    )

    for x in train:
        replicates = np.asarray([
            problem.simulate(x, rng)[1]
            for _ in range(args.replicates)
        ], dtype=float)
        replicate_variance = float(np.var(replicates, ddof=1))
        hvd.update(
            1,
            x,
            float(np.mean(replicates)),
            problem.true_constraint_mean(x),
            problem=problem,
            replicate_variance=replicate_variance,
        )

    test = evaluation_policies(problem)
    true_variance = np.asarray([
        float(problem.true_sigma(x)[1] ** 2) for x in test
    ])
    predicted = hvd.predict_variance_many(1, test, problem)
    certified = hvd.predict_certification_variance_many(1, test, problem)
    true_shared = np.asarray([
        float(problem.true_cumulative_risk_decomposition(
            x, output_index=1)["shared"])
        for x in test
    ])
    fitted_shared = []
    for x in test:
        decomposition = hvd.predict_decomposition(1, x, problem)
        cumulative = decomposition.get("cumulative") or {}
        blocks = cumulative.get("fitted_blocks") or {}
        fitted_shared.append(blocks.get("shared"))
    fitted_shared_array = np.asarray([
        np.nan if value is None else float(value) for value in fitted_shared
    ])

    z_alpha = float(norm.ppf(1.0 - problem.alpha))
    mean_constraint = np.asarray([
        problem.true_constraint_mean(x) for x in test
    ], dtype=float)
    true_margin = mean_constraint + z_alpha * np.sqrt(true_variance) - problem.tau
    certified_margin = mean_constraint + z_alpha * np.sqrt(certified) - problem.tau
    true_feasible = true_margin <= 0.0
    posterior_feasible = certified_margin <= 0.0
    false_feasible = posterior_feasible & ~true_feasible
    missed_feasible = ~posterior_feasible & true_feasible
    log_error = np.log(np.maximum(predicted, 1e-12)) - np.log(
        np.maximum(true_variance, 1e-12))
    valid_shared = np.isfinite(fitted_shared_array)

    return {
        "schema_version": 1,
        "status": "ok",
        "experiment": "hvd_identifiability",
        "mode": args.mode,
        "seed": int(args.seed),
        "d": int(args.d),
        "n_train_policies": int(args.n_train),
        "replicates_per_policy": int(args.replicates),
        "simulator_calls": int(args.n_train * args.replicates),
        "shared_shock_scale": float(args.shock_scale),
        "evaluation_count": int(len(test)),
        "log_variance_rmse": float(np.sqrt(np.mean(log_error ** 2))),
        "variance_spearman": _safe_spearman(true_variance, predicted),
        "shared_risk_spearman": (
            _safe_spearman(true_shared[valid_shared], fitted_shared_array[valid_shared])
            if int(np.sum(valid_shared)) >= 3 else None
        ),
        "median_predicted_true_ratio": float(np.median(
            predicted / np.maximum(true_variance, 1e-12))),
        "median_certified_true_ratio": float(np.median(
            certified / np.maximum(true_variance, 1e-12))),
        "variance_upper_coverage": float(np.mean(certified >= true_variance)),
        "posterior_feasible_count": int(np.sum(posterior_feasible)),
        "true_feasible_count": int(np.sum(true_feasible)),
        "false_feasible_count": int(np.sum(false_feasible)),
        "false_feasible_rate": float(np.mean(false_feasible)),
        "missed_feasible_count": int(np.sum(missed_feasible)),
        "missed_feasible_rate": float(np.mean(missed_feasible)),
        "certificate_nonvacuous": bool(np.any(posterior_feasible)),
        "hvd_diagnostics": hvd.diagnostics(),
        "information_contract": {
            "oracle_used_for_fit": False,
            "oracle_used_for_post_run_audit": True,
            "fit_inputs": "ordinary_replicated_simulator_observations",
            "objective_search_involved": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODE_CONFIG), required=True)
    parser.add_argument("--shock-scale", type=float, required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--activation-min-records", type=int, default=16)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.shock_scale < 0.0:
        raise ValueError("shock scale must be nonnegative")
    if args.replicates < 2:
        raise ValueError("replicate variance needs at least two replications")
    started = time.perf_counter()
    result = run_cell(args)
    result["wall_time_sec"] = float(time.perf_counter() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "mode": result["mode"],
        "seed": result["seed"],
        "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
