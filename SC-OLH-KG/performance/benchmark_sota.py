"""Compare OLH-KG variants with strong lightweight baselines.

The local environment does not include BoTorch/GPyTorch, so `turbo_lite` and
`scbo_lite` are dependency-light trust-region baselines rather than exact
published implementations.  They are useful for budget-matched regression
tests and can be replaced by external adapters later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from baselines.baseline_algorithms import BaselineConfig, SequentialBaseline  # noqa: E402
from benchmark_quality import json_safe, nested_get, parse_csv, parse_weights, write_csv  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


def finite(values):
    out = []
    for value in values:
        if value is None:
            continue
        val = float(value)
        if val == val and abs(val) < float("inf"):
            out.append(val)
    return out


def stats(values):
    vals = finite(values)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def make_wrapped_problem(args):
    base = make_problem(
        args.problem,
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
    )
    return ScalarizedProblem(base, weights=parse_weights(args.weights))


def row_from_result(variant, seed, args, result):
    true_feasible = bool(result["true_feasible"])
    posterior_feasible = bool(result.get("posterior_feasible", False))
    feasible_regret = float(result["simple_regret"]) if true_feasible else None
    return {
        "variant": variant,
        "seed": int(seed),
        "problem": args.problem,
        "N": int(args.N),
        "n0": int(args.n0),
        "x_recommended": result["x_recommended"],
        "true_feasible": true_feasible,
        "posterior_feasible": posterior_feasible,
        "false_feasible": bool(posterior_feasible and not true_feasible),
        "true_objective": float(result["true_objective"]),
        "true_best_objective": float(result["true_best_objective"]),
        "simple_regret": float(result["simple_regret"]),
        "feasible_simple_regret": feasible_regret,
        "true_chance_margin": float(result["true_chance_margin"]),
        "constraint_violation": max(float(result["true_chance_margin"]), 0.0),
        "posterior_chance_margin": float(result.get("posterior_chance_margin", 0.0)),
        "wall_time_sec": float(result["total_time_sec"]),
        "n_simulations": int(result["n_simulations"]),
        "n_distinct_solutions": int(result["n_distinct_solutions"]),
    }


def run_olhkg(args, seed, use_sc):
    problem = make_wrapped_problem(args)
    config = SingleOLHKGConfig(
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        K2=args.K2,
        posterior_pool_size=args.posterior_pool_size,
        posterior_keep=args.posterior_keep,
        structured_candidate_count=args.structured_candidate_count,
        state_candidate_count=args.state_candidate_count,
        state_inverse_pool_size=args.state_inverse_pool_size,
        state_inverse_neighbors=args.state_inverse_neighbors,
        variance_mode=args.variance_mode,
        lambda_feas=args.lambda_feas,
        lambda_var=args.lambda_var,
        lambda_coupling=args.lambda_coupling if use_sc else 0.0,
        use_state_coupling=use_sc,
        seed=seed,
    )
    result = SingleOLHKGAlgorithm(problem, config).run(verbose=False)
    return row_from_result("olhkg_sc" if use_sc else "olhkg", seed, args, result)


def run_baseline(args, seed, method):
    problem = make_wrapped_problem(args)
    config = BaselineConfig(
        N=args.N,
        n0=args.n0,
        seed=seed,
        method=method,
        batch_candidates=args.baseline_batch_candidates,
        tr_radius_init=args.tr_radius_init,
    )
    result = SequentialBaseline(problem, config).run()
    return row_from_result(method, seed, args, result)


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for variant, items in grouped.items():
        summaries[variant] = {
            "variant": variant,
            "n_runs": len(items),
            "true_feasible_rate": sum(bool(r["true_feasible"]) for r in items) / len(items),
            "false_feasible_rate": sum(bool(r["false_feasible"]) for r in items) / len(items),
            "feasible_simple_regret": stats(r["feasible_simple_regret"] for r in items),
            "constraint_violation": stats(r["constraint_violation"] for r in items),
            "wall_time_sec": stats(r["wall_time_sec"] for r in items),
        }
    baseline = summaries.get("olhkg")
    if baseline is not None:
        base_regret = nested_get(baseline, ["feasible_simple_regret", "median"])
        base_false = baseline["false_feasible_rate"]
        base_feas = baseline["true_feasible_rate"]
        for summary in summaries.values():
            regret = nested_get(summary, ["feasible_simple_regret", "median"])
            summary["vs_olhkg"] = {
                "feasible_rate_delta": float(summary["true_feasible_rate"] - base_feas),
                "false_feasible_rate_delta": float(summary["false_feasible_rate"] - base_false),
                "feasible_regret_median_delta": (
                    None if regret is None or base_regret is None
                    else float(regret - base_regret)
                ),
            }
    return summaries


def flatten_summary(summary):
    row = {
        "variant": summary["variant"],
        "n_runs": summary["n_runs"],
        "true_feasible_rate": summary["true_feasible_rate"],
        "false_feasible_rate": summary["false_feasible_rate"],
    }
    for metric in ("feasible_simple_regret", "constraint_violation", "wall_time_sec"):
        for key, value in summary[metric].items():
            row[f"{metric}_{key}"] = value
    for key, value in summary.get("vs_olhkg", {}).items():
        row[f"vs_olhkg_{key}"] = value
    return row


def run_benchmark(args):
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    rows = []
    variants = []
    if args.include_olhkg:
        variants.append(("olhkg", None))
    if args.include_sc:
        variants.append(("olhkg_sc", None))
    for method in parse_csv(args.baselines):
        variants.append((method, method))
    for name, method in variants:
        for seed in seeds:
            print(f"[sota] variant={name} seed={seed}", flush=True)
            if name == "olhkg":
                rows.append(run_olhkg(args, seed, use_sc=False))
            elif name == "olhkg_sc":
                rows.append(run_olhkg(args, seed, use_sc=True))
            else:
                rows.append(run_baseline(args, seed, method))
    summaries = summarize(rows)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(args),
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"sota_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, [flatten_summary(s) for s in result["summary"].values()])
    return {"json": str(json_path), "rows_csv": str(rows_path), "summary_csv": str(summary_path)}


def printable_summary(result):
    rows = []
    for variant, summary in result["summary"].items():
        rows.append({
            "variant": variant,
            "true_feasible_rate": summary["true_feasible_rate"],
            "false_feasible_rate": summary["false_feasible_rate"],
            "feasible_regret_median": nested_get(
                summary, ["feasible_simple_regret", "median"]),
            "violation_mean": nested_get(summary, ["constraint_violation", "mean"]),
            "wall_time_median_sec": nested_get(summary, ["wall_time_sec", "median"]),
            "vs_olhkg": summary.get("vs_olhkg", {}),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="StatePolicyRZDT1")
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
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=-1)
    parser.add_argument("--state_inverse_pool_size", type=int, default=500)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--variance_mode", default="orthogonal")
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--baselines", default="sobol,random,turbo_lite,scbo_lite")
    parser.add_argument("--baseline_batch_candidates", type=int, default=64)
    parser.add_argument("--tr_radius_init", type=float, default=0.35)
    parser.add_argument("--include_olhkg", action="store_true", default=True)
    parser.add_argument("--include_sc", action="store_true", default=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()

    result = run_benchmark(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "summary": printable_summary(result),
    }), indent=2))


if __name__ == "__main__":
    main()
