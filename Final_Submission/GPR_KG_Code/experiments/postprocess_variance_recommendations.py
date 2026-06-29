"""Common-pool final recommendation for variance-study runs.

This script does not rerun the sequential policy.  It restores each saved
checkpoint and asks every method to recommend a final Pareto set on the same
deterministic candidate pool for a given (problem, replication).

Pools:
  common_generic:
      union of all methods' observed/reported/candidate points for the same
      (problem, rep), local neighborhoods, and a deterministic random grid.
  common_axis:
      common_generic plus the diagnostic Pareto-axis scan
      (x1, lo_2, ..., lo_d).  This is useful for synthetic benchmark
      diagnosis and should be reported as a diagnostic pool if used.
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
from gpr_kg import GPRKR_Algorithm, compute_hypervolume_2d, pareto_filter  # noqa: E402
from metrics import compute_cvr, compute_igd  # noqa: E402


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


def load_results(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(run_dir.glob("*/*/result.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["result_path"] = str(path)
        data["run_dir"] = str(path.parent)
        records.append(data)
    return records


def add_neighbors(pool: set[tuple[int, ...]], problem, centers, radius: int):
    if radius <= 0:
        return
    lo, hi = problem.int_bounds()
    lo = np.asarray(lo, dtype=int)
    hi = np.asarray(hi, dtype=int)
    for x in list(centers):
        x0 = np.asarray(x, dtype=int)
        pool.add(tuple(int(v) for v in x0))
        for j in range(problem.d):
            for step in range(1, radius + 1):
                for sign in (-1, 1):
                    y = x0.copy()
                    y[j] = int(np.clip(y[j] + sign * step, lo[j], hi[j]))
                    pool.add(tuple(int(v) for v in y))


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


def build_common_pools(records: list[dict], problem, args):
    """Build method-independent pools for one problem/rep group."""
    base_pool: set[tuple[int, ...]] = set()
    for result in records:
        for obs in result.get("observation_history", []):
            base_pool.add(tuple(int(v) for v in obs["x"]))
        for x in result.get("pareto_solutions", []):
            base_pool.add(tuple(int(v) for v in x))
        for log in result.get("iteration_log", []):
            x_sel = log.get("x_selected")
            if x_sel:
                base_pool.add(tuple(int(v) for v in x_sel))
            if args.include_iteration_candidates:
                for x in log.get("candidate_set", []) or []:
                    base_pool.add(tuple(int(v) for v in x))

    add_neighbors(base_pool, problem, list(base_pool), args.neighbor_radius)
    seed = int(args.random_seed_base + records[0]["rep"])
    for x in deterministic_random_grid(problem, args.random_pool_size, seed):
        base_pool.add(x)

    pools = {"common_generic": sorted(base_pool)}
    axis_pool = set(base_pool)
    for x in axis_scan(problem):
        axis_pool.add(x)
    pools["common_axis"] = sorted(axis_pool)
    return pools


def tpos_solution_set(problem):
    lo, hi = problem.int_bounds()
    sols = set()
    for x1 in range(int(lo[0]), int(hi[0]) + 1):
        x = tuple([int(x1)] + [int(lo[j]) for j in range(1, problem.d)])
        if problem.is_truly_feasible(x):
            sols.add(x)
    return sols


def evaluate_recommendation(alg, problem, candidates, kappa: float):
    q = norm.ppf(1.0 - problem.alpha)
    feasible_sols = []
    posterior_objs = []
    feasible_truth = []
    posterior_feasible_flags = []
    true_feasible_flags = []

    for x in candidates:
        x_tuple = tuple(int(v) for v in x)
        x_arr = np.array(x_tuple, dtype=int)
        mu1 = alg.gpr[0].posterior_mean(x_arr)
        mu2 = alg.gpr[1].posterior_mean(x_arr)
        mu3 = alg.gpr[2].posterior_mean(x_arr)
        sig3 = math.sqrt(max(float(alg._effective_variance(2, x_arr)), 0.0))
        posterior_feasible = bool(mu3 + kappa * q * sig3 <= problem.tau)
        true_feasible = bool(problem.is_truly_feasible(x_tuple))
        posterior_feasible_flags.append(posterior_feasible)
        true_feasible_flags.append(true_feasible)
        if posterior_feasible:
            feasible_sols.append(x_tuple)
            posterior_objs.append([mu1, mu2])
            feasible_truth.append(true_feasible)

    if posterior_objs:
        _, idx = pareto_filter(np.array(posterior_objs), return_indices=True)
        rec_sols = [feasible_sols[i] for i in idx]
    else:
        rec_sols = []

    true_objs = []
    for x in rec_sols:
        f1, f2, _ = problem.true_objectives(x)
        true_objs.append([f1, f2])
    true_objs = np.array(true_objs) if true_objs else np.empty((0, 2))

    if len(true_objs) > 0:
        pf_true, idx_true = pareto_filter(true_objs, return_indices=True)
        rec_true_sols = [rec_sols[i] for i in idx_true]
    else:
        pf_true = np.empty((0, 2))
        rec_true_sols = []

    posterior_feasible_flags = np.array(posterior_feasible_flags, dtype=bool)
    true_feasible_flags = np.array(true_feasible_flags, dtype=bool)
    tp = int(np.sum(posterior_feasible_flags & true_feasible_flags))
    fp = int(np.sum(posterior_feasible_flags & ~true_feasible_flags))
    fn = int(np.sum(~posterior_feasible_flags & true_feasible_flags))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0)

    tpos = tpos_solution_set(problem)
    tpos_hits = sum(1 for x in rec_true_sols if tuple(x) in tpos)

    return {
        "candidate_pool_size": int(len(candidates)),
        "posterior_feasible_count": int(np.sum(posterior_feasible_flags)),
        "true_feasible_count": int(np.sum(true_feasible_flags)),
        "posterior_nd_count": int(len(rec_sols)),
        "pareto_solutions": [[int(v) for v in x] for x in rec_true_sols],
        "pareto_objectives_true": pf_true.tolist(),
        "hv_final": float(compute_hypervolume_2d(pf_true, problem.ref_point)),
        "igd_final": float(compute_igd(pf_true, problem.true_pareto_front())),
        "cvr_final": float(compute_cvr(rec_true_sols, problem)),
        "n_pareto_solutions": int(len(rec_true_sols)),
        "tpos_hits": int(tpos_hits),
        "feas_precision": float(precision),
        "feas_recall": float(recall),
        "feas_f1": float(f1_score),
    }


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    metrics = [
        "candidate_pool_size",
        "posterior_feasible_count",
        "true_feasible_count",
        "posterior_nd_count",
        "hv_final",
        "igd_final",
        "cvr_final",
        "n_pareto_solutions",
        "tpos_hits",
        "feas_precision",
        "feas_recall",
        "feas_f1",
    ]
    grouped: dict[tuple[str, str, float, str], list[dict]] = {}
    for row in rows:
        key = (row["problem"], row["pool"], float(row["kappa"]),
               row["method"])
        grouped.setdefault(key, []).append(row)

    out_rows = []
    for (problem, pool, kappa, method), group in sorted(grouped.items()):
        out = {
            "problem": problem,
            "pool": pool,
            "kappa": kappa,
            "method": method,
            "n": len(group),
        }
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in group], dtype=float)
            out[f"{metric}_mean"] = float(np.mean(vals))
            out[f"{metric}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            out[f"{metric}_se"] = (
                out[f"{metric}_std"] / float(np.sqrt(len(vals)))
                if len(vals) else 0.0)
        out_rows.append(out)
    return out_rows


def oracle_gap_rows(rows: list[dict]) -> list[dict]:
    by_key = {}
    for row in rows:
        key = (row["problem"], int(row["rep"]), row["pool"],
               float(row["kappa"]), row["method"])
        by_key[key] = row

    out = []
    base_methods = ("GPR-KG-pooled-pre", "GPR-KG", "GPR-KG-oracleV")
    keys = sorted({(r["problem"], int(r["rep"]), r["pool"],
                    float(r["kappa"])) for r in rows})
    for problem, rep, pool, kappa in keys:
        pooled = by_key.get((problem, rep, pool, kappa, base_methods[0]))
        vepm = by_key.get((problem, rep, pool, kappa, base_methods[1]))
        oracle = by_key.get((problem, rep, pool, kappa, base_methods[2]))
        if pooled is None or vepm is None or oracle is None:
            continue
        hv_oracle_gain = oracle["hv_final"] - pooled["hv_final"]
        hv_vepm_gain = vepm["hv_final"] - pooled["hv_final"]
        igd_oracle_gain = pooled["igd_final"] - oracle["igd_final"]
        igd_vepm_gain = pooled["igd_final"] - vepm["igd_final"]
        out.append({
            "problem": problem,
            "rep": rep,
            "pool": pool,
            "kappa": kappa,
            "hv_oracle_gain": float(hv_oracle_gain),
            "hv_vepm_gain": float(hv_vepm_gain),
            "hv_captured_oracle_ratio": (
                float(hv_vepm_gain / hv_oracle_gain)
                if abs(hv_oracle_gain) > 1e-12 else ""),
            "hv_unrealized_gap": float(
                oracle["hv_final"] - vepm["hv_final"]),
            "igd_oracle_gain": float(igd_oracle_gain),
            "igd_vepm_gain": float(igd_vepm_gain),
            "igd_captured_oracle_ratio": (
                float(igd_vepm_gain / igd_oracle_gain)
                if abs(igd_oracle_gain) > 1e-12 else ""),
            "igd_unrealized_gap": float(
                vepm["igd_final"] - oracle["igd_final"]),
        })
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--kappas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--neighbor_radius", type=int, default=1)
    parser.add_argument("--random_pool_size", type=int, default=1000)
    parser.add_argument("--random_seed_base", type=int, default=91000)
    parser.add_argument("--include_iteration_candidates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (
        run_dir / "postprocessing_common")
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = output_dir / "details"
    detail_dir.mkdir(exist_ok=True)

    records = load_results(run_dir)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for result in records:
        grouped.setdefault((result["problem"], int(result["rep"])), []).append(
            result)

    rows = []
    for (problem_name, rep), group in sorted(grouped.items()):
        ref = group[0]
        problem = make_problem(
            problem_name,
            d=int(ref["d"]),
            sigma=float(ref["sigma"]),
            alpha=float(ref["alpha"]),
        )
        pools = build_common_pools(group, problem, args)
        pool_sizes = {name: len(pool) for name, pool in pools.items()}

        for result in group:
            checkpoint_path = Path(result["run_dir"]) / "checkpoint.pkl"
            alg = GPRKR_Algorithm.restore_checkpoint(
                str(checkpoint_path), problem)
            detail = {
                "problem": problem_name,
                "rep": int(result["rep"]),
                "method": result["method"],
                "pool_sizes": pool_sizes,
                "sample_metrics": {
                    "hv_final": result["hv_final"],
                    "igd_final": result["igd_final"],
                    "cvr_final": result["cvr_final"],
                    "n_pareto_solutions": result["n_pareto_solutions"],
                },
                "recommendations": {},
            }
            for pool_name, candidates in pools.items():
                for kappa in args.kappas:
                    rec = evaluate_recommendation(
                        alg, problem, candidates, float(kappa))
                    detail["recommendations"][f"{pool_name}_k{kappa}"] = rec
                    rows.append({
                        "problem": problem_name,
                        "rep": int(result["rep"]),
                        "seed": int(result["seed"]),
                        "method": result["method"],
                        "method_base": result.get("method_base", ""),
                        "variance_mode": result.get("variance_mode", ""),
                        "pool": pool_name,
                        "kappa": float(kappa),
                        "sample_hv_final": float(result["hv_final"]),
                        "sample_igd_final": float(result["igd_final"]),
                        "sample_cvr_final": float(result["cvr_final"]),
                        "sample_n_pareto_solutions": int(
                            result["n_pareto_solutions"]),
                        **{k: rec[k] for k in [
                            "candidate_pool_size",
                            "posterior_feasible_count",
                            "true_feasible_count",
                            "posterior_nd_count",
                            "hv_final",
                            "igd_final",
                            "cvr_final",
                            "n_pareto_solutions",
                            "tpos_hits",
                            "feas_precision",
                            "feas_recall",
                            "feas_f1",
                        ]},
                    })
            detail_path = detail_dir / (
                f"{problem_name}_{result['method']}_rep{int(rep):02d}.json")
            with detail_path.open("w", encoding="utf-8") as f:
                json.dump(json_safe(detail), f, indent=2)

    write_csv(rows, output_dir / "common_recommendation_rows.csv")
    summary_rows = summarize(rows)
    write_csv(summary_rows, output_dir / "common_recommendation_summary.csv")
    gaps = oracle_gap_rows(rows)
    write_csv(gaps, output_dir / "common_oracle_gap.csv")
    print(f"[postprocess] loaded {len(records)} result files")
    print(f"[postprocess] wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
