"""Offline final-recommendation post-processing for checkpointed RZDT runs.

The sequential GPR-KG policy is not rerun.  This script loads each saved
checkpoint, evaluates the final posterior over an expanded finite candidate
pool, applies a conservative posterior chance-feasibility filter, and then
Pareto-filters the survivors by posterior objective means.

Two pools are reported:
  generic: sampled solutions, all saved iteration candidates, and local
           integer-grid neighborhoods around sampled/reported Pareto points.
  axis_scan: generic plus the benchmark diagnostic line
             (x1, 0, ..., 0), x1 = 0,...,100.
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

from gpr_kg import (  # noqa: E402
    GPRKR_Algorithm,
    RZDT1,
    RZDT2,
    RZDT5_RR,
    compute_hypervolume_2d,
    pareto_filter,
)
from metrics import compute_cvr, compute_igd  # noqa: E402


def make_problem(name: str, d: int, sigma: float, alpha: float, tau: float):
    kwargs = dict(d=d, L=100, sigma=sigma, heteroscedastic=True, alpha=alpha)
    if name == "RZDT1":
        problem = RZDT1(**kwargs)
    elif name == "RZDT2":
        problem = RZDT2(**kwargs)
    elif name == "RZDT5_RR":
        problem = RZDT5_RR(**kwargs)
    else:
        raise ValueError(f"Unknown problem: {name}")
    problem.tau = tau
    problem.ref_point = np.array([1.5, 1.5], dtype=float)
    return problem


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    return obj


def add_neighbors(pool: set[tuple[int, ...]], problem, centers, radius: int):
    if radius <= 0:
        return
    lo, hi = problem.int_bounds()
    for x in centers:
        x = tuple(int(v) for v in x)
        pool.add(x)
        for j in range(problem.d):
            for step in range(1, radius + 1):
                for sign in (-1, 1):
                    y = list(x)
                    y[j] = int(np.clip(y[j] + sign * step, lo[j], hi[j]))
                    pool.add(tuple(y))


def build_candidate_pool(result, problem, radius: int, include_axis_scan: bool,
                         include_iteration_candidates: bool):
    pool: set[tuple[int, ...]] = set()

    for obs in result.get("observation_history", []):
        pool.add(tuple(int(v) for v in obs["x"]))

    for x in result.get("pareto_solutions", []):
        pool.add(tuple(int(v) for v in x))

    for log in result.get("iteration_log", []):
        if include_iteration_candidates:
            for x in log.get("candidate_set", []) or []:
                pool.add(tuple(int(v) for v in x))
        x_sel = log.get("x_selected")
        if x_sel:
            pool.add(tuple(int(v) for v in x_sel))

    centers = list(pool)
    add_neighbors(pool, problem, centers, radius)

    if include_axis_scan:
        lo, hi = problem.int_bounds()
        for x1 in range(int(lo[0]), int(hi[0]) + 1):
            x = [0] * problem.d
            x[0] = x1
            for j in range(1, problem.d):
                x[j] = int(lo[j])
            pool.add(tuple(x))

    return sorted(pool)


def tpos_solution_set(problem):
    lo, hi = problem.int_bounds()
    sols = []
    for x1 in range(int(lo[0]), int(hi[0]) + 1):
        x = tuple([x1] + [int(lo[j]) for j in range(1, problem.d)])
        if problem.is_truly_feasible(x):
            sols.append(x)
    return set(sols)


def evaluate_recommendation(alg, problem, candidates, kappa: float):
    q = norm.ppf(1.0 - problem.alpha)
    feasible_sols = []
    posterior_objs = []
    for x in candidates:
        x_arr = np.array(x, dtype=int)
        mu1 = alg.gpr[0].posterior_mean(x_arr)
        mu2 = alg.gpr[1].posterior_mean(x_arr)
        mu3 = alg.gpr[2].posterior_mean(x_arr)
        sig3 = math.sqrt(max(float(alg._effective_variance(2, x_arr)), 0.0))
        if mu3 + kappa * q * sig3 <= problem.tau:
            feasible_sols.append(tuple(int(v) for v in x))
            posterior_objs.append([mu1, mu2])

    if posterior_objs:
        pf_post, idx = pareto_filter(np.array(posterior_objs),
                                     return_indices=True)
        rec_sols = [feasible_sols[i] for i in idx]
    else:
        pf_post = np.empty((0, 2))
        rec_sols = []

    true_objs = []
    for x in rec_sols:
        f1, f2, _ = problem.true_objectives(x)
        true_objs.append([f1, f2])
    true_objs = np.array(true_objs) if true_objs else np.empty((0, 2))

    if len(true_objs) > 0:
        pf_true, true_idx = pareto_filter(true_objs, return_indices=True)
        rec_true_sols = [rec_sols[i] for i in true_idx]
    else:
        pf_true = np.empty((0, 2))
        rec_true_sols = []

    tpos = tpos_solution_set(problem)
    tpos_hits = sum(1 for x in rec_true_sols if tuple(x) in tpos)

    return {
        "candidate_pool_size": int(len(candidates)),
        "posterior_feasible_count": int(len(feasible_sols)),
        "posterior_nd_count": int(len(rec_sols)),
        "pareto_solutions": [[int(v) for v in x] for x in rec_true_sols],
        "pareto_objectives_true": pf_true.tolist(),
        "hv_final": float(compute_hypervolume_2d(pf_true, problem.ref_point)),
        "igd_final": float(compute_igd(pf_true, problem.true_pareto_front())),
        "cvr_final": float(compute_cvr(rec_true_sols, problem)),
        "n_pareto_solutions": int(len(rec_true_sols)),
        "tpos_hits": int(tpos_hits),
    }


def summarize(rows, output_dir: Path):
    summary = {}
    keys = [
        "hv_final",
        "igd_final",
        "cvr_final",
        "n_pareto_solutions",
        "tpos_hits",
        "candidate_pool_size",
        "posterior_feasible_count",
        "posterior_nd_count",
    ]
    groups = {}
    for row in rows:
        groups.setdefault((row["problem"], row["pool"], row["kappa"]), []).append(row)

    for group, vals in groups.items():
        problem, pool, kappa = group
        summary.setdefault(problem, {}).setdefault(pool, {})[str(kappa)] = {}
        for key in keys:
            arr = np.array([float(v[key]) for v in vals], dtype=float)
            summary[problem][pool][str(kappa)][key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "se": float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
                if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": int(len(arr)),
            }

    with (output_dir / "recommendation_summary.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--kappas", nargs="+", type=float,
                        default=[1.0, 1.25, 1.5])
    parser.add_argument("--neighbor_radius", type=int, default=1)
    parser.add_argument("--include_iteration_candidates", action="store_true",
                        help=("Include every saved per-iteration candidate. "
                              "This is slower; the default compact pool uses "
                              "sampled/selected solutions plus neighborhoods."))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (
        results_dir / "postprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    detail_dir = output_dir / "details"
    detail_dir.mkdir(exist_ok=True)

    for result_path in sorted(results_dir.glob("*/*/result.json")):
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        problem = make_problem(
            result["problem"],
            int(result["d"]),
            float(result["sigma"]),
            float(result["alpha"]),
            float(result["tau"]),
        )
        checkpoint_path = result_path.with_name("checkpoint.pkl")
        alg = GPRKR_Algorithm.restore_checkpoint(str(checkpoint_path), problem)

        pools = {
            "generic": build_candidate_pool(
                result, problem, args.neighbor_radius, include_axis_scan=False,
                include_iteration_candidates=args.include_iteration_candidates),
            "axis_scan": build_candidate_pool(
                result, problem, args.neighbor_radius, include_axis_scan=True,
                include_iteration_candidates=args.include_iteration_candidates),
        }

        run_detail = {
            "problem": result["problem"],
            "rep": int(result["rep"]),
            "sample_pareto": {
                "hv_final": result["hv_final"],
                "igd_final": result["igd_final"],
                "cvr_final": result["cvr_final"],
                "n_pareto_solutions": result["n_pareto_solutions"],
            },
            "recommendations": {},
        }

        for pool_name, candidates in pools.items():
            for kappa in args.kappas:
                rec = evaluate_recommendation(alg, problem, candidates, kappa)
                row = {
                    "problem": result["problem"],
                    "rep": int(result["rep"]),
                    "seed": int(result["seed"]),
                    "pool": pool_name,
                    "kappa": float(kappa),
                    **{k: rec[k] for k in [
                        "candidate_pool_size",
                        "posterior_feasible_count",
                        "posterior_nd_count",
                        "hv_final",
                        "igd_final",
                        "cvr_final",
                        "n_pareto_solutions",
                        "tpos_hits",
                    ]},
                    "sample_hv_final": result["hv_final"],
                    "sample_igd_final": result["igd_final"],
                    "sample_cvr_final": result["cvr_final"],
                    "sample_n_pareto_solutions":
                        result["n_pareto_solutions"],
                }
                rows.append(row)
                run_detail["recommendations"][f"{pool_name}_k{kappa}"] = rec

        detail_path = detail_dir / (
            f"{result['problem']}_rep{int(result['rep']):02d}.json")
        with detail_path.open("w", encoding="utf-8") as f:
            json.dump(json_safe(run_detail), f, indent=2)

    csv_path = output_dir / "recommendation_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows, output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'recommendation_summary.json'}")


if __name__ == "__main__":
    main()
