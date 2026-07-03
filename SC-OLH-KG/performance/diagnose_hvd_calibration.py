"""Diagnose variance and chance-bound calibration for one OLH-KG run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


def json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    return obj


def parse_weights(value):
    items = [float(v.strip()) for v in str(value).split(",") if v.strip()]
    if len(items) != 2:
        raise ValueError("--weights must contain two comma-separated numbers")
    return tuple(items)


def finite(values):
    vals = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            vals.append(value)
    return vals


def stats(values):
    vals = finite(values)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(vals)),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def build_problem(args):
    base = make_problem(
        args.problem,
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
    )
    return ScalarizedProblem(base, weights=parse_weights(args.weights))


def build_config(args):
    return SingleOLHKGConfig(
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        K2=args.K2,
        posterior_pool_size=args.posterior_pool_size,
        posterior_keep=args.posterior_keep,
        n_thr=args.n_thr,
        variance_mode=args.variance_mode,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_coupling=args.lambda_coupling if args.use_state_coupling else 0.0,
        use_state_coupling=args.use_state_coupling,
        use_state_basis=args.use_state_basis,
        eval_pool_size=args.eval_pool_size,
        seed=args.seed,
    )


def candidate_grid(problem):
    if hasattr(problem, "all_axis_solutions"):
        return [tuple(map(int, x)) for x in problem.all_axis_solutions()]
    lo, hi = problem.int_bounds()
    rows = []
    for x1 in range(int(lo[0]), int(hi[0]) + 1):
        rows.append(tuple([x1] + [int(lo[j]) for j in range(1, problem.d)]))
    return rows


def observed_counts(history):
    counts = {}
    for x, _ in history:
        x_tuple = tuple(int(v) for v in x)
        counts[x_tuple] = counts.get(x_tuple, 0) + 1
    return counts


def grid_rows(alg, problem, result, args):
    X = candidate_grid(problem)
    rec_x = tuple(int(v) for v in result["x_recommended"])
    z = norm.ppf(1 - problem.alpha)
    mu_obj = alg.gpr[0].posterior_mean_many(X)
    mu_g = alg.gpr[1].posterior_mean_many(X)
    var_pred = alg.variance_model.predict_variance_many(1, X, problem)
    var_cert = alg.variance_model.predict_certification_variance_many(1, X, problem)
    obs_counts = observed_counts(alg.history)
    true_best_x, true_best_obj = problem.true_best_feasible()
    true_best_x = None if true_best_x is None else tuple(int(v) for v in true_best_x)

    rows = []
    for idx, x in enumerate(X):
        true_obj = float(problem.true_objective(x))
        true_g = float(problem.true_constraint_mean(x))
        true_sig = float(problem.true_sigma(x)[1])
        true_var = float(true_sig ** 2)
        true_margin = true_g + z * true_sig - problem.tau
        post_margin = float(mu_g[idx] + z * math.sqrt(max(var_cert[idx], 1e-12)) - problem.tau)
        post_feasible = post_margin <= 0.0
        true_feasible = true_margin <= 0.0
        pred_ratio = float(var_pred[idx] / max(true_var, 1e-12))
        cert_ratio = float(var_cert[idx] / max(true_var, 1e-12))
        rows.append({
            "x": list(map(int, x)),
            "x1": int(x[0]),
            "risk_class": int(problem.risk_class(x)),
            "is_recommendation": bool(x == rec_x),
            "is_true_best": bool(true_best_x is not None and x == true_best_x),
            "observed_count": int(obs_counts.get(x, 0)),
            "true_objective": true_obj,
            "posterior_mu_objective": float(mu_obj[idx]),
            "true_best_objective": float(true_best_obj),
            "simple_regret": float(true_obj - true_best_obj),
            "true_constraint_mean": true_g,
            "posterior_mu_constraint": float(mu_g[idx]),
            "constraint_mu_error": float(mu_g[idx] - true_g),
            "true_constraint_variance": true_var,
            "predicted_constraint_variance": float(var_pred[idx]),
            "certification_constraint_variance": float(var_cert[idx]),
            "predicted_true_variance_ratio": pred_ratio,
            "certification_true_variance_ratio": cert_ratio,
            "true_chance_margin": float(true_margin),
            "posterior_chance_margin": post_margin,
            "posterior_feasible": bool(post_feasible),
            "true_feasible": bool(true_feasible),
            "false_feasible": bool(post_feasible and not true_feasible),
            "missed_feasible": bool((not post_feasible) and true_feasible),
            "near_true_boundary": bool(abs(true_margin) <= args.boundary_window),
        })
    return rows


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault("all", []).append(row)
        grouped.setdefault(f"class_{row['risk_class']}", []).append(row)
        if row["near_true_boundary"]:
            grouped.setdefault("near_boundary", []).append(row)
    summary = {}
    for key, group in grouped.items():
        false_rows = [row for row in group if row["false_feasible"]]
        missed_rows = [row for row in group if row["missed_feasible"]]
        summary[key] = {
            "n": int(len(group)),
            "posterior_feasible_count": int(sum(row["posterior_feasible"] for row in group)),
            "true_feasible_count": int(sum(row["true_feasible"] for row in group)),
            "false_feasible_count": int(len(false_rows)),
            "missed_feasible_count": int(len(missed_rows)),
            "mu_error": stats(row["constraint_mu_error"] for row in group),
            "abs_mu_error": stats(abs(row["constraint_mu_error"]) for row in group),
            "predicted_true_variance_ratio": stats(
                row["predicted_true_variance_ratio"] for row in group),
            "certification_true_variance_ratio": stats(
                row["certification_true_variance_ratio"] for row in group),
            "posterior_margin": stats(row["posterior_chance_margin"] for row in group),
            "true_margin": stats(row["true_chance_margin"] for row in group),
            "false_feasible_x1": [int(row["x1"]) for row in false_rows],
            "missed_feasible_x1": [int(row["x1"]) for row in missed_rows],
        }
    return summary


def recommendation_diagnostics(rows):
    recs = [row for row in rows if row["is_recommendation"]]
    if not recs:
        return None
    row = recs[0]
    return {
        "x": row["x"],
        "true_feasible": row["true_feasible"],
        "posterior_feasible": row["posterior_feasible"],
        "false_feasible": row["false_feasible"],
        "simple_regret": row["simple_regret"],
        "true_objective": row["true_objective"],
        "true_chance_margin": row["true_chance_margin"],
        "posterior_chance_margin": row["posterior_chance_margin"],
        "constraint_mu_error": row["constraint_mu_error"],
        "true_constraint_variance": row["true_constraint_variance"],
        "predicted_constraint_variance": row["predicted_constraint_variance"],
        "certification_constraint_variance": row["certification_constraint_variance"],
        "predicted_true_variance_ratio": row["predicted_true_variance_ratio"],
        "certification_true_variance_ratio": row["certification_true_variance_ratio"],
        "risk_class": row["risk_class"],
        "observed_count": row["observed_count"],
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            writer.writerow(flat)


def run_diagnostic(args):
    problem = build_problem(args)
    config = build_config(args)
    alg = SingleOLHKGAlgorithm(problem, config)
    result = alg.run(verbose=args.verbose)
    rows = grid_rows(alg, problem, result, args)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "problem": args.problem,
            "variance_mode": args.variance_mode,
            "seed": args.seed,
            "N": args.N,
            "n0": args.n0,
            "K1": args.K1,
            "K2": args.K2,
            "use_state_coupling": args.use_state_coupling,
            "use_state_basis": args.use_state_basis,
            "lambda_feas": args.lambda_feas,
            "lambda_var": args.lambda_var,
            "lambda_coupling": args.lambda_coupling,
            "boundary_window": args.boundary_window,
        },
        "run_result": result,
        "variance_diagnostics": alg.variance_model.diagnostics(),
        "recommendation": recommendation_diagnostics(rows),
        "summary": summarize_rows(rows),
        "grid_rows": rows,
    }


def output_paths(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "sc" if args.use_state_coupling else "raw"
    prefix = args.out_prefix or (
        f"hvd_calibration_{args.variance_mode}_{suffix}_seed{args.seed}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_grid.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["grid_rows"])
    return {"json": str(json_path), "grid_csv": str(rows_path)}


def compact_print(result, paths):
    return {
        "paths": paths,
        "recommendation": result["recommendation"],
        "summary": {
            key: {
                "n": val["n"],
                "false_feasible_count": val["false_feasible_count"],
                "missed_feasible_count": val["missed_feasible_count"],
                "mu_error_mean": val["mu_error"]["mean"],
                "abs_mu_error_mean": val["abs_mu_error"]["mean"],
                "pred_var_ratio_median": val["predicted_true_variance_ratio"]["median"],
                "cert_var_ratio_median": val["certification_true_variance_ratio"]["median"],
                "false_feasible_x1": val["false_feasible_x1"],
            }
            for key, val in result["summary"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="RegimeRZDT1")
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=5)
    parser.add_argument("--K1", type=int, default=15)
    parser.add_argument("--K2", type=int, default=1)
    parser.add_argument("--posterior_pool_size", type=int, default=300)
    parser.add_argument("--posterior_keep", type=int, default=15)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--eval_pool_size", type=int, default=500)
    parser.add_argument("--variance_mode", default="orthogonal",
                        choices=["pooled", "oracle", "class", "orthogonal", "factor"])
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_coupling", type=float, default=0.15)
    parser.add_argument("--use-state-coupling", action="store_true")
    parser.add_argument("--use_state_basis", action="store_true")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--boundary_window", type=float, default=0.08)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_diagnostic(args)
    paths = output_paths(args, result)
    print(json.dumps(json_safe(compact_print(result, paths)), indent=2))


if __name__ == "__main__":
    main()
