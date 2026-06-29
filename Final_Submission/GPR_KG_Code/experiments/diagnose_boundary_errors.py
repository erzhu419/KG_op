"""Chance-boundary diagnostics for variance-study runs.

This script restores saved checkpoints and evaluates whether poor performance
comes from variance calibration, posterior mean error, feasible-boundary
classification, or sampling allocation.  It is intentionally offline: it does
not rerun the sequential policy.
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
    rows = []
    for path in sorted(run_dir.glob("*/*/result.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if problems is not None and data["problem"] not in problems:
            continue
        data["result_path"] = str(path)
        data["run_dir"] = str(path.parent)
        rows.append(data)
    return rows


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
            seen.add(x)
            out.append(x)
    return out


def axis_scan(problem):
    lo, hi = problem.int_bounds()
    out = []
    for x1 in range(int(lo[0]), int(hi[0]) + 1):
        x = [int(lo[j]) for j in range(problem.d)]
        x[0] = int(x1)
        out.append(tuple(x))
    return out


def observed_points(result, unique: bool = False):
    pts = [tuple(int(v) for v in obs["x"])
           for obs in result.get("observation_history", [])]
    if unique:
        return sorted(set(pts))
    return pts


def build_pools(problem, result, random_pool_size: int, seed: int):
    pools = {
        "axis": axis_scan(problem),
        "random": deterministic_random_grid(problem, random_pool_size, seed),
    }
    combined = set(pools["axis"])
    combined.update(pools["random"])
    combined.update(observed_points(result, unique=True))
    pools["combined"] = sorted(combined)
    return pools


def t1_value(problem, x):
    return float(problem.normalize(np.asarray(x, dtype=int))[0])


def t1_region(problem, x):
    t1 = t1_value(problem, x)
    if t1 < 1.0 / 3.0:
        return "low_t1"
    if t1 < 2.0 / 3.0:
        return "mid_t1"
    return "high_t1"


def classify(pred_feasible: bool, true_feasible: bool):
    if pred_feasible and true_feasible:
        return "tp"
    if pred_feasible and not true_feasible:
        return "fp"
    if (not pred_feasible) and true_feasible:
        return "fn"
    return "tn"


def evaluate_point(alg, problem, x, kappa: float):
    x_tuple = tuple(int(v) for v in x)
    x_arr = np.asarray(x_tuple, dtype=int)
    q = norm.ppf(1.0 - problem.alpha)
    true_obj = np.asarray(problem.true_objectives(x_tuple), dtype=float)
    true_sigma = np.asarray(problem.true_sigma(x_tuple), dtype=float)
    true_var = np.maximum(true_sigma ** 2, 1e-12)
    mu = np.array([
        float(alg.gpr[i].posterior_mean(x_arr)) for i in range(3)
    ], dtype=float)
    pred_var = []
    for i in range(3):
        v, _details = alg._effective_variance_with_details(i, x_arr)
        pred_var.append(max(float(v), 1e-12))
    pred_var = np.asarray(pred_var, dtype=float)

    true_margin = float(true_obj[2] + q * true_sigma[2] - problem.tau)
    post_margin = float(mu[2] + kappa * q * math.sqrt(pred_var[2])
                        - problem.tau)
    true_feasible = bool(true_margin <= 0.0)
    posterior_feasible = bool(post_margin <= 0.0)
    label = classify(posterior_feasible, true_feasible)
    return {
        "x": x_tuple,
        "x1": int(x_tuple[0]),
        "t1": t1_value(problem, x_tuple),
        "t1_region": t1_region(problem, x_tuple),
        "true_feasible": true_feasible,
        "posterior_feasible": posterior_feasible,
        "class": label,
        "true_margin": true_margin,
        "post_margin": post_margin,
        "abs_true_margin": abs(true_margin),
        "margin_error": post_margin - true_margin,
        "true_mu": true_obj,
        "pred_mu": mu,
        "mu_error": mu - true_obj,
        "true_var": true_var,
        "pred_var": pred_var,
    }


def finite_values(items, key):
    vals = []
    for item in items:
        val = item.get(key)
        if val is None:
            continue
        vals.append(float(val))
    return vals


def mean_or_zero(values):
    return float(np.mean(values)) if len(values) else 0.0


def rmse(values):
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0


def subset_metrics(points):
    if not points:
        out = {
            "n_eval": 0,
            "posterior_feasible_count": 0,
            "true_feasible_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_feasible_rate": 0.0,
            "false_infeasible_rate": 0.0,
            "true_margin_mean": 0.0,
            "abs_true_margin_mean": 0.0,
            "post_margin_mean": 0.0,
            "abs_post_margin_mean": 0.0,
            "margin_bias": 0.0,
            "margin_mae": 0.0,
            "margin_rmse": 0.0,
            "fp_count": 0,
            "fn_count": 0,
            "fp_true_margin_mean": 0.0,
            "fp_post_margin_mean": 0.0,
            "fp_var2_ratio_mean": 0.0,
            "fn_true_margin_mean": 0.0,
            "fn_post_margin_mean": 0.0,
            "fn_var2_ratio_mean": 0.0,
        }
        for i in range(3):
            out.update({
                f"mu{i}_bias": 0.0,
                f"mu{i}_mae": 0.0,
                f"mu{i}_rmse": 0.0,
                f"var{i}_pred_mean": 0.0,
                f"var{i}_true_mean": 0.0,
                f"var{i}_bias": 0.0,
                f"var{i}_rmse": 0.0,
                f"var{i}_ratio_mean": 0.0,
                f"var{i}_ratio_median": 0.0,
                f"var{i}_log_rmse": 0.0,
            })
        return out
    n = len(points)
    pred = np.array([p["posterior_feasible"] for p in points], dtype=bool)
    true = np.array([p["true_feasible"] for p in points], dtype=bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    tn = int(np.sum(~pred & ~true))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    out = {
        "n_eval": int(n),
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

    true_margin = np.array([p["true_margin"] for p in points], dtype=float)
    post_margin = np.array([p["post_margin"] for p in points], dtype=float)
    margin_err = post_margin - true_margin
    out.update({
        "true_margin_mean": float(np.mean(true_margin)),
        "abs_true_margin_mean": float(np.mean(np.abs(true_margin))),
        "post_margin_mean": float(np.mean(post_margin)),
        "abs_post_margin_mean": float(np.mean(np.abs(post_margin))),
        "margin_bias": float(np.mean(margin_err)),
        "margin_mae": float(np.mean(np.abs(margin_err))),
        "margin_rmse": float(np.sqrt(np.mean(margin_err ** 2))),
    })

    for i in range(3):
        mu_err = np.array([p["mu_error"][i] for p in points], dtype=float)
        pred_var = np.array([p["pred_var"][i] for p in points], dtype=float)
        true_var = np.array([p["true_var"][i] for p in points], dtype=float)
        var_err = pred_var - true_var
        ratio = pred_var / np.maximum(true_var, 1e-12)
        out.update({
            f"mu{i}_bias": float(np.mean(mu_err)),
            f"mu{i}_mae": float(np.mean(np.abs(mu_err))),
            f"mu{i}_rmse": float(np.sqrt(np.mean(mu_err ** 2))),
            f"var{i}_pred_mean": float(np.mean(pred_var)),
            f"var{i}_true_mean": float(np.mean(true_var)),
            f"var{i}_bias": float(np.mean(var_err)),
            f"var{i}_rmse": float(np.sqrt(np.mean(var_err ** 2))),
            f"var{i}_ratio_mean": float(np.mean(ratio)),
            f"var{i}_ratio_median": float(np.median(ratio)),
            f"var{i}_log_rmse": float(np.sqrt(np.mean(
                (np.log(np.maximum(pred_var, 1e-12)) -
                 np.log(np.maximum(true_var, 1e-12))) ** 2))),
        })

    for label in ("fp", "fn"):
        selected = [p for p in points if p["class"] == label]
        out[f"{label}_count"] = int(len(selected))
        out[f"{label}_true_margin_mean"] = mean_or_zero(
            np.array([p["true_margin"] for p in selected], dtype=float))
        out[f"{label}_post_margin_mean"] = mean_or_zero(
            np.array([p["post_margin"] for p in selected], dtype=float))
        out[f"{label}_var2_ratio_mean"] = mean_or_zero(
            np.array([p["pred_var"][2] / max(p["true_var"][2], 1e-12)
                      for p in selected], dtype=float))
    return out


def boundary_threshold(points, quantile: float, fixed_width: float | None):
    if fixed_width is not None:
        return float(fixed_width)
    vals = np.array([p["abs_true_margin"] for p in points], dtype=float)
    if len(vals) == 0:
        return 0.0
    return float(np.quantile(vals, quantile))


def add_subset(rows, base, name, points):
    rows.append({**base, "subset": name, **subset_metrics(points)})


def sampling_allocation(result, problem, points_combined, threshold):
    obs = observed_points(result, unique=False)
    obs_unique = observed_points(result, unique=True)
    point_map = {p["x"]: p for p in points_combined}
    q75_var2 = float(np.quantile(
        [p["true_var"][2] for p in points_combined], 0.75))

    def summarize_obs(seq, prefix):
        if not seq:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_near_boundary_count": 0,
                f"{prefix}_near_boundary_fraction": 0.0,
                f"{prefix}_true_feasible_fraction": 0.0,
                f"{prefix}_high_var2_fraction": 0.0,
                f"{prefix}_low_t1_fraction": 0.0,
                f"{prefix}_mid_t1_fraction": 0.0,
                f"{prefix}_high_t1_fraction": 0.0,
                f"{prefix}_pareto_axis_fraction": 0.0,
            }
        recs = []
        lo, _hi = problem.int_bounds()
        lo_tail = tuple(int(lo[j]) for j in range(1, problem.d))
        for x in seq:
            rec = point_map.get(x)
            if rec is None:
                rec = evaluate_point_dummy(problem, x)
            recs.append(rec)
        near = [r["abs_true_margin"] <= threshold for r in recs]
        true_feas = [r["true_feasible"] for r in recs]
        high_var = [r["true_var"][2] >= q75_var2 for r in recs]
        regions = [r["t1_region"] for r in recs]
        pareto_axis = [
            tuple(int(v) for v in r["x"][1:]) == lo_tail for r in recs
        ]
        return {
            f"{prefix}_n": int(len(seq)),
            f"{prefix}_near_boundary_count": int(np.sum(near)),
            f"{prefix}_near_boundary_fraction": float(np.mean(near)),
            f"{prefix}_true_feasible_fraction": float(np.mean(true_feas)),
            f"{prefix}_high_var2_fraction": float(np.mean(high_var)),
            f"{prefix}_low_t1_fraction": float(np.mean(
                [r == "low_t1" for r in regions])),
            f"{prefix}_mid_t1_fraction": float(np.mean(
                [r == "mid_t1" for r in regions])),
            f"{prefix}_high_t1_fraction": float(np.mean(
                [r == "high_t1" for r in regions])),
            f"{prefix}_pareto_axis_fraction": float(np.mean(pareto_axis)),
        }

    candidate_near = [p["abs_true_margin"] <= threshold
                      for p in points_combined]
    candidate_feas = [p["true_feasible"] for p in points_combined]
    row = {
        "candidate_pool_size": int(len(points_combined)),
        "boundary_abs_threshold": float(threshold),
        "candidate_near_boundary_fraction": float(np.mean(candidate_near)),
        "candidate_true_feasible_fraction": float(np.mean(candidate_feas)),
    }
    row.update(summarize_obs(obs, "obs"))
    row.update(summarize_obs(obs_unique, "unique_obs"))
    row["observation_duplicate_fraction"] = (
        1.0 - len(obs_unique) / len(obs) if obs else 0.0)
    return row


def evaluate_point_dummy(problem, x):
    x_tuple = tuple(int(v) for v in x)
    q = norm.ppf(1.0 - problem.alpha)
    true_obj = np.asarray(problem.true_objectives(x_tuple), dtype=float)
    true_sigma = np.asarray(problem.true_sigma(x_tuple), dtype=float)
    true_margin = float(true_obj[2] + q * true_sigma[2] - problem.tau)
    return {
        "x": x_tuple,
        "t1_region": t1_region(problem, x_tuple),
        "true_feasible": bool(true_margin <= 0.0),
        "abs_true_margin": abs(true_margin),
        "true_var": np.maximum(true_sigma ** 2, 1e-12),
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

    subset_rows = []
    detail_rows = []
    sampling_rows = []
    evaluated = {}
    for pool_name, candidates in pools.items():
        points = [evaluate_point(alg, problem, x, args.kappa)
                  for x in candidates]
        evaluated[pool_name] = points
        threshold = boundary_threshold(
            points, args.boundary_quantile, args.boundary_abs_width)
        base = {
            "problem": result["problem"],
            "method": result["method"],
            "variance_mode": result.get("variance_mode", ""),
            "rep": int(result["rep"]),
            "seed": int(result["seed"]),
            "pool": pool_name,
            "kappa": float(args.kappa),
            "boundary_abs_threshold": float(threshold),
        }
        add_subset(subset_rows, base, "all", points)
        add_subset(
            subset_rows, base, "near_true_boundary",
            [p for p in points if p["abs_true_margin"] <= threshold])
        add_subset(
            subset_rows, base, "far_true_boundary",
            [p for p in points if p["abs_true_margin"] > threshold])
        for region in ("low_t1", "mid_t1", "high_t1"):
            add_subset(
                subset_rows, base, region,
                [p for p in points if p["t1_region"] == region])
        for label in ("tp", "fp", "fn", "tn"):
            add_subset(
                subset_rows, base, label,
                [p for p in points if p["class"] == label])

        if pool_name == args.detail_pool:
            for p in points:
                if (args.detail_all or
                        p["abs_true_margin"] <= threshold or
                        p["class"] in {"fp", "fn"}):
                    detail_rows.append(point_detail_row(base, p))

    combined = evaluated["combined"]
    combined_threshold = boundary_threshold(
        combined, args.boundary_quantile, args.boundary_abs_width)
    sampling_rows.append({
        "problem": result["problem"],
        "method": result["method"],
        "variance_mode": result.get("variance_mode", ""),
        "rep": int(result["rep"]),
        "seed": int(result["seed"]),
        **sampling_allocation(result, problem, combined, combined_threshold),
    })
    return subset_rows, sampling_rows, detail_rows


def point_detail_row(base, point):
    row = {
        **base,
        "x": " ".join(str(v) for v in point["x"]),
        "x1": int(point["x1"]),
        "t1": float(point["t1"]),
        "t1_region": point["t1_region"],
        "class": point["class"],
        "true_feasible": int(point["true_feasible"]),
        "posterior_feasible": int(point["posterior_feasible"]),
        "true_margin": float(point["true_margin"]),
        "post_margin": float(point["post_margin"]),
        "abs_true_margin": float(point["abs_true_margin"]),
        "margin_error": float(point["margin_error"]),
    }
    for i in range(3):
        row[f"true_mu{i}"] = float(point["true_mu"][i])
        row[f"pred_mu{i}"] = float(point["pred_mu"][i])
        row[f"mu{i}_error"] = float(point["mu_error"][i])
        row[f"true_var{i}"] = float(point["true_var"][i])
        row[f"pred_var{i}"] = float(point["pred_var"][i])
    return row


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
    parser.add_argument("--random_seed_base", type=int, default=97000)
    parser.add_argument("--boundary_quantile", type=float, default=0.25)
    parser.add_argument("--boundary_abs_width", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--detail_pool", default="axis",
                        choices=["axis", "random", "combined"])
    parser.add_argument("--detail_all", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    problems = set(args.problems) if args.problems else None
    out_dir = Path(args.output_dir) if args.output_dir else (
        run_dir / "boundary_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(run_dir, problems)
    all_subset = []
    all_sampling = []
    all_detail = []
    for result in records:
        subset_rows, sampling_rows, detail_rows = evaluate_record(result, args)
        all_subset.extend(subset_rows)
        all_sampling.extend(sampling_rows)
        all_detail.extend(detail_rows)

    write_csv(all_subset, out_dir / "boundary_subset_rows.csv")
    write_csv(all_sampling, out_dir / "sampling_allocation_rows.csv")
    write_csv(all_detail, out_dir / "boundary_point_rows.csv")

    subset_metrics_cols = [
        "boundary_abs_threshold", "n_eval",
        "posterior_feasible_count", "true_feasible_count",
        "tp", "fp", "fn", "tn", "precision", "recall", "f1",
        "false_feasible_rate", "false_infeasible_rate",
        "abs_true_margin_mean", "abs_post_margin_mean",
        "margin_bias", "margin_mae", "margin_rmse",
        "mu0_mae", "mu0_rmse", "mu1_mae", "mu1_rmse",
        "mu2_mae", "mu2_rmse",
        "var2_pred_mean", "var2_true_mean", "var2_rmse",
        "var2_ratio_mean", "var2_ratio_median", "var2_log_rmse",
        "fp_count", "fn_count",
        "fp_true_margin_mean", "fp_post_margin_mean",
        "fn_true_margin_mean", "fn_post_margin_mean",
    ]
    sampling_metrics_cols = [
        "candidate_pool_size", "boundary_abs_threshold",
        "candidate_near_boundary_fraction",
        "candidate_true_feasible_fraction",
        "obs_n", "obs_near_boundary_count",
        "obs_near_boundary_fraction", "obs_true_feasible_fraction",
        "obs_high_var2_fraction", "obs_low_t1_fraction",
        "obs_mid_t1_fraction", "obs_high_t1_fraction",
        "obs_pareto_axis_fraction",
        "unique_obs_n", "unique_obs_near_boundary_count",
        "unique_obs_near_boundary_fraction",
        "unique_obs_true_feasible_fraction",
        "unique_obs_high_var2_fraction",
        "unique_obs_low_t1_fraction", "unique_obs_mid_t1_fraction",
        "unique_obs_high_t1_fraction", "unique_obs_pareto_axis_fraction",
        "observation_duplicate_fraction",
    ]
    write_csv(
        summarize(
            all_subset,
            ["problem", "method", "variance_mode", "pool", "subset"],
            subset_metrics_cols,
        ),
        out_dir / "boundary_subset_summary.csv",
    )
    write_csv(
        summarize(
            all_sampling,
            ["problem", "method", "variance_mode"],
            sampling_metrics_cols,
        ),
        out_dir / "sampling_allocation_summary.csv",
    )

    print(f"[boundary] loaded {len(records)} result files")
    print(f"[boundary] wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
