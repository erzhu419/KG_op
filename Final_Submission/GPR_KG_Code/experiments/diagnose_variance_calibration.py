"""Variance and feasibility calibration diagnostics for variance-study runs.

The script restores saved checkpoints and evaluates each method on fixed
candidate pools.  It is intended for debugging whether VEPM fails because the
benchmark lacks a variance signal, because the variance estimator is poorly
calibrated, or because final feasibility classification is miscalibrated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.problem_registry import make_problem  # noqa: E402
from gpr_kg import GPRKR_Algorithm  # noqa: E402


def load_results(run_dir: Path, problems: set[str] | None):
    records = []
    for path in sorted(run_dir.glob("*/*/result.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if problems is not None and data["problem"] not in problems:
            continue
        data["result_path"] = str(path)
        data["run_dir"] = str(path.parent)
        records.append(data)
    return records


def deterministic_random_grid(problem, n_points: int, seed: int):
    rng = np.random.RandomState(seed)
    lo, hi = problem.int_bounds()
    lo = np.asarray(lo, dtype=int)
    hi = np.asarray(hi, dtype=int)
    out = []
    seen = set()
    attempts = 0
    max_attempts = max(1000, n_points * 20)
    while len(out) < n_points and attempts < max_attempts:
        attempts += 1
        x = tuple(int(rng.randint(lo[j], hi[j] + 1))
                  for j in range(problem.d))
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def axis_scan(problem):
    lo, hi = problem.int_bounds()
    out = []
    for x1 in range(int(lo[0]), int(hi[0]) + 1):
        x = [int(lo[j]) for j in range(problem.d)]
        x[0] = int(x1)
        out.append(tuple(x))
    return out


def observed_points(result):
    pts = []
    for obs in result.get("observation_history", []):
        pts.append(tuple(int(v) for v in obs["x"]))
    return pts


def build_pools(problem, result, random_pool_size: int, seed: int):
    pools = {}
    pools["axis"] = axis_scan(problem)
    pools["random"] = deterministic_random_grid(problem, random_pool_size, seed)
    combined = set(pools["axis"])
    combined.update(pools["random"])
    combined.update(observed_points(result))
    pools["combined"] = sorted(combined)
    return pools


def t1_region(problem, x):
    t1 = float(problem.normalize(np.asarray(x))[0])
    if t1 < 1.0 / 3.0:
        return "low_t1"
    if t1 < 2.0 / 3.0:
        return "mid_t1"
    return "high_t1"


def variance_metrics(values):
    if not values:
        return {}
    pred = np.array([v["pred"] for v in values], dtype=float)
    true = np.array([v["true"] for v in values], dtype=float)
    eps = 1e-12
    err = pred - true
    rel = err / np.maximum(true, eps)
    ratio = pred / np.maximum(true, eps)
    return {
        "n_eval": int(len(values)),
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(true)),
        "bias_mean": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "rel_mae": float(np.mean(np.abs(rel))),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_median": float(np.median(ratio)),
        "log_rmse": float(np.sqrt(np.mean(
            (np.log(np.maximum(pred, eps)) -
             np.log(np.maximum(true, eps))) ** 2))),
    }


def feasibility_metrics(flags):
    if not flags:
        return {}
    pred = np.array([v["posterior_feasible"] for v in flags], dtype=bool)
    true = np.array([v["true_feasible"] for v in flags], dtype=bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    tn = int(np.sum(~pred & ~true))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "n_eval": int(len(flags)),
        "posterior_feasible_count": int(np.sum(pred)),
        "true_feasible_count": int(np.sum(true)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_feasible_rate": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "false_infeasible_rate": float(fn / (fn + tp)
                                       if (fn + tp) else 0.0),
    }


def cell_diagnostics(alg, problem, candidates):
    vepm = getattr(alg, "vepm", None)
    if vepm is None:
        return {}
    partition_sols = getattr(vepm, "partition_sols", {}) or {}
    sol_count = getattr(vepm, "sol_count", {}) or {}
    total_partitions = int(getattr(vepm, "total_partitions", 0) or 0)
    visited_cells = list(partition_sols)
    cell_solution_sizes = [len(partition_sols[c]) for c in visited_cells]
    cell_obs_sizes = [
        int(sum(sol_count.get(tuple(s), 1) for s in partition_sols[c]))
        for c in visited_cells
    ]

    eval_cells = [vepm.partition_index(x) for x in candidates]
    eval_unseen = [c not in partition_sols for c in eval_cells]
    eval_cell_sizes = [len(partition_sols.get(c, [])) for c in eval_cells]
    eval_singleton = [size == 1 for size in eval_cell_sizes]

    def stat(vals, fn, default=0.0):
        return float(fn(vals)) if vals else float(default)

    return {
        "partition_method": getattr(vepm, "partition_method", ""),
        "partition_features": " ".join(
            str(j) for j in getattr(vepm, "feature_indices", [])),
        "n_features": int(getattr(vepm, "n_features", 0) or 0),
        "total_partitions": total_partitions,
        "visited_cells": int(len(visited_cells)),
        "visited_cell_fraction": (
            float(len(visited_cells) / total_partitions)
            if total_partitions else 0.0),
        "mean_solutions_per_visited_cell": stat(cell_solution_sizes, np.mean),
        "median_solutions_per_visited_cell": stat(
            cell_solution_sizes, np.median),
        "singleton_visited_cell_fraction": (
            float(np.mean([s == 1 for s in cell_solution_sizes]))
            if cell_solution_sizes else 0.0),
        "max_solutions_per_visited_cell": stat(cell_solution_sizes, np.max),
        "mean_obs_per_visited_cell": stat(cell_obs_sizes, np.mean),
        "median_obs_per_visited_cell": stat(cell_obs_sizes, np.median),
        "max_obs_per_visited_cell": stat(cell_obs_sizes, np.max),
        "eval_unseen_cell_fraction": float(np.mean(eval_unseen))
        if eval_unseen else 0.0,
        "eval_singleton_cell_fraction": float(np.mean(eval_singleton))
        if eval_singleton else 0.0,
    }


def evaluate_record(result, args):
    problem = make_problem(
        result["problem"],
        d=int(result["d"]),
        sigma=float(result["sigma"]),
        alpha=float(result["alpha"]),
    )
    checkpoint_path = Path(result["run_dir"]) / "checkpoint.pkl"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint missing for {result['run_dir']}: {checkpoint_path}")
    alg = GPRKR_Algorithm.restore_checkpoint(str(checkpoint_path), problem)
    pools = build_pools(
        problem,
        result,
        random_pool_size=args.random_pool_size,
        seed=args.random_seed_base + int(result["rep"]),
    )
    q = norm.ppf(1.0 - problem.alpha)

    variance_rows = []
    feasibility_rows = []
    cell_rows = []

    for pool_name, candidates in pools.items():
        by_obj = {i: [] for i in range(3)}
        by_obj_region = {(i, region): [] for i in range(3)
                         for region in ("low_t1", "mid_t1", "high_t1")}
        feas_flags = []

        for x in candidates:
            x_arr = np.asarray(x, dtype=int)
            true_vars = np.asarray(problem.true_sigma(x_arr), dtype=float) ** 2
            pred_vars = []
            for i in range(3):
                pred, _details = alg._effective_variance_with_details(i, x_arr)
                pred = max(float(pred), 1e-12)
                true = max(float(true_vars[i]), 1e-12)
                item = {"pred": pred, "true": true}
                by_obj[i].append(item)
                by_obj_region[(i, t1_region(problem, x))].append(item)
                pred_vars.append(pred)

            mu3 = float(alg.gpr[2].posterior_mean(x_arr))
            posterior_feasible = bool(
                mu3 + q * math.sqrt(max(pred_vars[2], 1e-12)) <= problem.tau)
            true_feasible = bool(problem.is_truly_feasible(tuple(x)))
            feas_flags.append({
                "posterior_feasible": posterior_feasible,
                "true_feasible": true_feasible,
            })

        base = {
            "problem": result["problem"],
            "method": result["method"],
            "variance_mode": result.get("variance_mode", ""),
            "rep": int(result["rep"]),
            "seed": int(result["seed"]),
            "pool": pool_name,
        }
        for i in range(3):
            metrics = variance_metrics(by_obj[i])
            variance_rows.append({
                **base,
                "objective_index": i,
                "region": "all",
                **metrics,
            })
            for region in ("low_t1", "mid_t1", "high_t1"):
                metrics = variance_metrics(by_obj_region[(i, region)])
                variance_rows.append({
                    **base,
                    "objective_index": i,
                    "region": region,
                    **metrics,
                })
        feasibility_rows.append({**base, **feasibility_metrics(feas_flags)})
        cell_rows.append({**base, **cell_diagnostics(alg, problem, candidates)})

    return variance_rows, feasibility_rows, cell_rows


def write_csv(rows, path: Path):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, group_cols, metric_cols):
    grouped = {}
    for row in rows:
        key = tuple(row[c] for c in group_cols)
        grouped.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(grouped.items()):
        rec = {c: key[i] for i, c in enumerate(group_cols)}
        rec["n"] = len(group)
        for metric in metric_cols:
            vals = []
            for row in group:
                val = row.get(metric, "")
                if val == "" or val is None:
                    continue
                vals.append(float(val))
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            rec[f"{metric}_mean"] = float(np.mean(arr))
            rec[f"{metric}_std"] = (
                float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
            rec[f"{metric}_se"] = (
                rec[f"{metric}_std"] / math.sqrt(len(arr))
                if len(arr) else 0.0)
        out.append(rec)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--problems", nargs="+", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--random_pool_size", type=int, default=1000)
    parser.add_argument("--random_seed_base", type=int, default=92000)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    problems = set(args.problems) if args.problems else None
    output_dir = Path(args.output_dir) if args.output_dir else (
        run_dir / "variance_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(run_dir, problems)
    all_variance = []
    all_feas = []
    all_cells = []
    for result in records:
        var_rows, feas_rows, cell_rows = evaluate_record(result, args)
        all_variance.extend(var_rows)
        all_feas.extend(feas_rows)
        all_cells.extend(cell_rows)

    write_csv(all_variance, output_dir / "variance_calibration_rows.csv")
    write_csv(all_feas, output_dir / "feasibility_calibration_rows.csv")
    write_csv(all_cells, output_dir / "vepm_cell_diagnostics_rows.csv")

    variance_summary = summarize(
        all_variance,
        ["problem", "method", "variance_mode", "pool",
         "objective_index", "region"],
        ["pred_mean", "true_mean", "bias_mean", "rmse", "mae",
         "rel_mae", "ratio_mean", "ratio_median", "log_rmse"],
    )
    feasibility_summary = summarize(
        all_feas,
        ["problem", "method", "variance_mode", "pool"],
        ["posterior_feasible_count", "true_feasible_count", "tp", "fp",
         "fn", "tn", "precision", "recall", "f1",
         "false_feasible_rate", "false_infeasible_rate"],
    )
    cell_summary = summarize(
        all_cells,
        ["problem", "method", "variance_mode", "pool",
         "partition_method", "partition_features",
         "n_features", "total_partitions"],
        ["visited_cells", "visited_cell_fraction",
         "mean_solutions_per_visited_cell",
         "median_solutions_per_visited_cell",
         "singleton_visited_cell_fraction",
         "max_solutions_per_visited_cell",
         "mean_obs_per_visited_cell",
         "median_obs_per_visited_cell",
         "max_obs_per_visited_cell",
         "eval_unseen_cell_fraction",
         "eval_singleton_cell_fraction"],
    )
    write_csv(variance_summary, output_dir / "variance_calibration_summary.csv")
    write_csv(feasibility_summary,
              output_dir / "feasibility_calibration_summary.csv")
    write_csv(cell_summary, output_dir / "vepm_cell_diagnostics_summary.csv")

    print(f"[diagnose] loaded {len(records)} result files")
    print(f"[diagnose] wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
