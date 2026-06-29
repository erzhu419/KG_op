"""Checkpointed same-budget structured baselines for RZDT experiments.

This driver reruns the surrogate baselines under the same synthetic problem
configuration used by the checkpointed GPR-KG/GPR-KG-nV experiments:

* RZDT1, RZDT2, and RZDT5_RR
* N=150, n0=30, sigma=0.04, tau=0, alpha=0.05
* 10 macro-replications by default
* a common structured finite-grid initial design shared across methods

Each run writes result.json, run_meta.json, and iteration_snapshots.jsonl.  The
method implementations remain backward compatible: if no initial design is
provided, they retain their original random pre-sampling behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable is not None else 0)
            self.count = 0
            self.desc = desc or ""
            self.unit = unit or "it"
            self.start = time.time()

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n=1):
            self.count += n
            elapsed = max(time.time() - self.start, 1e-9)
            rate = self.count / elapsed
            eta = ((self.total - self.count) / rate
                   if self.total and rate > 0 else float("nan"))
            print(f"{self.desc}: {self.count}/{self.total} {self.unit}, "
                  f"elapsed={elapsed:.1f}s, eta={eta:.1f}s", flush=True)

        def set_postfix(self, **kwargs):
            if kwargs:
                print(", ".join(f"{k}={v}" for k, v in kwargs.items()),
                      flush=True)

        def close(self):
            pass


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_D,
    DEFAULT_N,
    DEFAULT_N0,
    HV_EVAL_INTERVAL,
    N_REPS,
    REF_POINT,
    SEED_BASE,
)
from experiments.structured_design import (  # noqa: E402
    common_random_initial_samples,
    structured_initial_samples,
)
from experiments.problem_registry import (  # noqa: E402
    ALL_RZDT_PROBLEMS,
    ORIGINAL_RZDT_PROBLEMS,
    make_problem as registry_make_problem,
)
from gpr_kg import compute_hypervolume_2d  # noqa: E402
from methods.cehvi_method import cEHVIMethod  # noqa: E402
from methods.cparego_method import cParEGOMethod  # noqa: E402
from methods.nsga2_kriging import NSGA2Kriging  # noqa: E402


PROBLEMS = ORIGINAL_RZDT_PROBLEMS
METHODS = {
    "cEHVI": cEHVIMethod,
    "cParEGO": cParEGOMethod,
    "NSGA-II-K": NSGA2Kriging,
}


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


def make_problem(name: str, d: int, sigma: float, alpha: float):
    problem = registry_make_problem(name, d=d, sigma=sigma, alpha=alpha)
    problem.ref_point = np.array(REF_POINT, dtype=float)
    return problem


def true_pareto_hits(pareto_solutions, problem) -> int:
    true_set = set()
    lo, hi = problem.int_bounds()
    for x1 in range(int(lo[0]), int(hi[0]) + 1):
        x = tuple([x1] + [int(lo[j]) for j in range(1, problem.d)])
        if problem.is_truly_feasible(x):
            true_set.add(x)
    return int(sum(tuple(int(v) for v in x) in true_set
                   for x in pareto_solutions))


def run_one(problem_name: str, method_name: str, rep: int, args):
    seed = args.seed_base + rep
    run_dir = (args.output_dir / problem_name / method_name
               / f"rep{rep:02d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    meta_path = run_dir / "run_meta.json"
    snapshot_path = run_dir / "iteration_snapshots.jsonl"

    if result_path.exists() and not args.force:
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f), "skipped"

    if args.restart:
        for path in (result_path, meta_path, snapshot_path):
            if path.exists():
                path.unlink()

    np.random.seed(seed)
    problem = make_problem(problem_name, args.d, args.sigma, args.alpha)
    if args.initial_design == "common_random":
        initial_samples = common_random_initial_samples(problem, args.n0, seed)
    else:
        initial_samples = structured_initial_samples(problem, args.n0)
    method = METHODS[method_name]()
    meta = {
        "problem": problem_name,
        "method": method_name,
        "rep": rep,
        "seed": seed,
        "N": args.N,
        "n0": args.n0,
        "d": args.d,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "tau": problem.tau,
        "initial_design": args.initial_design,
        "initial_samples": initial_samples,
        "hv_eval_interval": args.hv_eval_interval,
        "snapshot_path": str(snapshot_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(meta), f, indent=2)

    t0 = time.time()
    result = method.run(
        problem,
        N=args.N,
        n0=args.n0,
        seed=seed,
        hv_eval_interval=args.hv_eval_interval,
        initial_samples=initial_samples,
        snapshot_path=str(snapshot_path),
    )
    wall = time.time() - t0

    true_pf = problem.true_pareto_front()
    hv_upper = (float(compute_hypervolume_2d(true_pf, problem.ref_point))
                if len(true_pf) > 0 else 0.0)
    result.update(meta)
    result.update({
        "completed": True,
        "wall_time_sec": float(wall),
        "hv_upper": hv_upper,
        "hv_ratio": (float(result["hv_final"]) / hv_upper
                     if hv_upper > 0 else 0.0),
        "n_true_pareto_hits": true_pareto_hits(
            result.get("pareto_solutions", []), problem),
        "logging_detail": (
            "iteration_snapshots.jsonl records all initial evaluations and "
            "adaptive evaluations; result.json stores full observation and "
            "iteration histories returned by the method."
        ),
    })

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)
    return result, "completed"


def write_summary(results, output_dir: Path):
    rows = []
    for result in results:
        rows.append({
            "problem": result["problem"],
            "method": result["method"],
            "rep": result["rep"],
            "seed": result["seed"],
            "hv_final": result["hv_final"],
            "hv_ratio": result["hv_ratio"],
            "igd_final": result["igd_final"],
            "cvr_final": result["cvr_final"],
            "n_pareto_solutions": result["n_pareto_solutions"],
            "n_true_pareto_hits": result["n_true_pareto_hits"],
            "n_simulations": result["n_simulations"],
            "total_time_sec": result.get("total_time_sec", 0.0),
            "wall_time_sec": result["wall_time_sec"],
        })
    if not rows:
        return

    with (output_dir / "summary_runs.csv").open("w", newline="",
                                                encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault((row["problem"], row["method"]), []).append(row)
    summary = {}
    for (problem, method), group in grouped.items():
        summary.setdefault(problem, {})[method] = {}
        for key in ("hv_final", "hv_ratio", "igd_final", "cvr_final",
                    "n_pareto_solutions", "n_true_pareto_hits",
                    "wall_time_sec"):
            vals = np.array([r[key] for r in group], dtype=float)
            summary[problem][method][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            }
    with (output_dir / "summary_by_problem_method.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run structured same-budget RZDT surrogate baselines.")
    parser.add_argument("--run_id", default=os.environ.get(
        "RUN_ID", "structured_baselines_full"))
    parser.add_argument("--results_root",
                        default="results/rzdt_structured_baselines")
    parser.add_argument(
        "--initial_design",
        choices=["common_structured", "common_random"],
        default="common_structured")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        choices=list(METHODS))
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS),
                        choices=list(ALL_RZDT_PROBLEMS))
    parser.add_argument("--n_reps", type=int, default=N_REPS)
    parser.add_argument("--seed_base", type=int, default=SEED_BASE)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--n0", type=int, default=DEFAULT_N0)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--hv_eval_interval", type=int,
                        default=HV_EVAL_INTERVAL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.methods = ["cParEGO", "NSGA-II-K"]
        args.problems = ["RZDT1"]
        args.n_reps = 1
        args.N = 12
        args.n0 = 6
        args.hv_eval_interval = 3
        args.run_id = args.run_id + "_smoke"

    args.output_dir = Path(args.results_root) / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w",
                                                     encoding="utf-8") as f:
        json.dump({k: json_safe(v) for k, v in vars(args).items()
                   if k != "output_dir"}, f, indent=2)

    units = [(p, m, r)
             for p in args.problems
             for m in args.methods
             for r in range(args.n_reps)]
    results = []
    pbar = tqdm(units, total=len(units),
                desc=f"structured baselines {args.run_id}", unit="run")
    for problem, method, rep in pbar:
        pbar.set_postfix(problem=problem, method=method, rep=rep)
        result, status = run_one(problem, method, rep, args)
        results.append(result)
        print(f"[{status}] {problem} {method} rep={rep} "
              f"HV={result['hv_final']:.6f} "
              f"IGD={result['igd_final']:.6f} "
              f"CVR={result['cvr_final']:.3f} "
              f"ND={result['n_pareto_solutions']}", flush=True)
    pbar.close()
    write_summary(results, args.output_dir)
    print(f"[summary] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
