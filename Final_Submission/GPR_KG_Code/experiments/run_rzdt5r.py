"""
RZDT5_R Experiment: revised RZDT5 with extended space and loosened constraint.

Problem settings:
  - x_1 in {0,...,100}, x_j in {0,...,50} for j>=2  (d=5)
  - f1 = x1/100
  - g  = 1 + sum_{j>=2} x_j/50
  - f2 = g / (x1 + 1)
  - f3 = x1/100 - 0.5
  - sigma_i(x) = sigma * (0.3 + 2*t_1^2),  sigma=0.04, t_1 = x1/100
  - Chance constraint: P(f3 <= tau) >= 0.95,  tau = 0.5, alpha = 0.05
  - Heteroscedastic = True

Results saved to results/rzdt5r/RZDT5_R/ (same format as results/d5_v2/).

Usage:
    cd GPR_KG_Code
    python -m experiments.run_rzdt5r
    python -m experiments.run_rzdt5r --method GPR-KG
    python -m experiments.run_rzdt5r --n_reps 10
"""

import sys
import os
import json
import time
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gpr_kg import RZDT5_R, compute_hypervolume_2d
from metrics import compute_igd, compute_cvr
from experiments.config import (
    DEFAULT_N, DEFAULT_N0, DEFAULT_D, DEFAULT_ALPHA,
    N_REPS, SEED_BASE, REF_POINT, HV_EVAL_INTERVAL,
)

from methods.random_search import RandomSearch
from methods.gpr_kg_method import GPRKGMethod
from methods.gpr_kg_nv import GPRKGnVMethod
from methods.cehvi_method import cEHVIMethod
from methods.cparego_method import cParEGOMethod
from methods.nsga2_kriging import NSGA2Kriging
from methods.nsga2_direct import NSGA2Direct

# ── output root ──────────────────────────────────────────────
RESULTS_ROOT = os.path.join(BASE_DIR, "results", "rzdt5r")

# ── method registry ──────────────────────────────────────────
ALL_METHODS = {
    "GPR-KG":    GPRKGMethod,
    "GPR-KG-nV": GPRKGnVMethod,
    "cEHVI":     cEHVIMethod,
    "cParEGO":   cParEGOMethod,
    "NSGA-II-K": NSGA2Kriging,
    "NSGA-II-D": NSGA2Direct,
    "RS":        RandomSearch,
}

# ── problem settings (RZDT5_R) ───────────────────────────────
RZDT5R_SIGMA = 0.04
RZDT5R_TAU   = 0.5

def make_problem(problem_name="RZDT5_R", d=DEFAULT_D,
                 sigma=RZDT5R_SIGMA, alpha=DEFAULT_ALPHA):
    """Instantiate RZDT5_R with tau=0.5."""
    if problem_name != "RZDT5_R":
        raise ValueError(f"Unknown problem: {problem_name}")
    prob = RZDT5_R(d=d, sigma=sigma, heteroscedastic=True, alpha=alpha)
    prob.tau = RZDT5R_TAU
    return prob

ALL_PROBLEM_NAMES = ["RZDT5_R"]

def _safe(name):
    return name.replace("-", "_")

def rep_path(problem_name, method_name, rep):
    return os.path.join(
        RESULTS_ROOT, problem_name,
        f"{_safe(method_name)}_rep{rep+1:02d}.json"
    )


def save_problem_meta(problem_name):
    """Save problem metadata (tau, true PF, etc.) to results/rzdt5r/problems/."""
    meta_dir = os.path.join(RESULTS_ROOT, "problems")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f"{problem_name}_meta.json")

    if os.path.exists(meta_path):
        return

    prob = make_problem(problem_name)
    true_pf = prob.true_pareto_front()
    ref = REF_POINT

    all_pf_pts = []
    lo, hi = prob.int_bounds()
    for x1 in range(lo[0], hi[0] + 1):
        x = tuple([x1] + [lo[j] for j in range(1, prob.d)])
        f1, f2, _ = prob.true_objectives(x)
        is_feas = prob.is_truly_feasible(x)
        all_pf_pts.append({"x1": x1, "f1": f1, "f2": f2, "feasible": bool(is_feas)})

    if hasattr(prob, 'true_pareto_curve'):
        f1_curve, f2_curve = prob.true_pareto_curve()
    else:
        f1_curve = [p["f1"] for p in all_pf_pts]
        f2_curve = [p["f2"] for p in all_pf_pts]

    meta = {
        "problem": problem_name,
        "d": DEFAULT_D,
        "sigma": RZDT5R_SIGMA,
        "alpha": DEFAULT_ALPHA,
        "tau": float(prob.tau),
        "true_pf_feasible": true_pf.tolist() if len(true_pf) > 0 else [],
        "n_feasible_pf": int(len(true_pf)),
        "all_pf_grid": all_pf_pts,
        "pf_curve_f1": f1_curve.tolist() if hasattr(f1_curve, 'tolist') else f1_curve,
        "pf_curve_f2": f2_curve.tolist() if hasattr(f2_curve, 'tolist') else f2_curve,
        "hv_true_pf": float(compute_hypervolume_2d(true_pf, ref)) if len(true_pf) > 0 else 0.0,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [meta] {problem_name}: tau={prob.tau:.4f}, "
          f"feasible_PF={len(true_pf)}, "
          f"HV_true={meta['hv_true_pf']:.4f}")


def run_one(method_name, problem_name, rep, n_reps_total):
    """Run one replication. Skip if result file already exists."""
    path = rep_path(problem_name, method_name, rep)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(f"    [skip] rep {rep+1:02d}/{n_reps_total} already exists")
        return json.load(open(path))

    seed = SEED_BASE + rep
    prob = make_problem(problem_name)
    method = ALL_METHODS[method_name]()

    t0 = time.time()
    result = method.run(prob, N=DEFAULT_N, n0=DEFAULT_N0, seed=seed)
    wall = time.time() - t0

    result["method"]          = method_name
    result["problem"]         = problem_name
    result["rep"]             = rep + 1
    result["seed"]            = seed
    result["N"]               = DEFAULT_N
    result["n0"]              = DEFAULT_N0
    result["d"]               = DEFAULT_D
    result["heteroscedastic"] = True
    result["tau"]             = float(prob.tau)
    result["sigma"]           = RZDT5R_SIGMA
    result["wall_time_sec"]   = float(wall)

    ref = REF_POINT
    true_pf = prob.true_pareto_front()
    hv_upper = float(compute_hypervolume_2d(true_pf, ref)) if len(true_pf) > 0 else 1.0
    result["hv_upper"] = hv_upper
    result["hv_ratio"] = result["hv_final"] / hv_upper if hv_upper > 0 else 0.0

    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def aggregate(problem_name, method_name, n_reps):
    records = []
    for rep in range(n_reps):
        path = rep_path(problem_name, method_name, rep)
        if os.path.exists(path):
            try:
                records.append(json.load(open(path)))
            except Exception:
                pass

    if not records:
        return None

    def _stat(key):
        vals = [r[key] for r in records if key in r]
        if not vals:
            return None, None
        arr = np.array(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

    hv_m, hv_se = _stat("hv_final")
    igd_m, igd_se = _stat("igd_final")
    cvr_m, cvr_se = _stat("cvr_final")
    wall_m, wall_se = _stat("wall_time_sec")
    ratio_m, _ = _stat("hv_ratio")

    return {
        "method": method_name, "problem": problem_name,
        "n_reps": len(records),
        "hv_mean": hv_m, "hv_se": hv_se,
        "igd_mean": igd_m, "igd_se": igd_se,
        "cvr_mean": cvr_m, "cvr_se": cvr_se,
        "hv_ratio_mean": ratio_m,
        "wall_time_mean": wall_m, "wall_time_se": wall_se,
        "hv_values": [r["hv_final"] for r in records],
        "igd_values": [r["igd_final"] for r in records],
        "cvr_values": [r["cvr_final"] for r in records],
    }


def run_rzdt5r(methods=None, problems=None, n_reps=N_REPS):
    if methods is None:
        methods = list(ALL_METHODS.keys())
    if problems is None:
        problems = ALL_PROBLEM_NAMES

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print("\n" + "=" * 70)
    print("  RZDT5_R Experiments (revised RZDT5, extended space)")
    print(f"  Problems : {problems}")
    print(f"  Methods  : {methods}")
    print(f"  Reps     : {n_reps}  |  N={DEFAULT_N}  n0={DEFAULT_N0}")
    print(f"  tau={RZDT5R_TAU}, heteroscedastic=True, sigma={RZDT5R_SIGMA}")
    print(f"  Decision space: x1 in {{0..100}}, x_j in {{0..50}} (j>=2)")
    print(f"  Results  : {RESULTS_ROOT}")
    print("=" * 70)

    print("\n[1/3] Saving problem metadata...")
    for pname in problems:
        save_problem_meta(pname)

    print("\n[2/3] Running experiments...")
    t_total_start = time.time()

    for pname in problems:
        for mname in methods:
            print(f"\n  {mname:12s} on {pname} ({n_reps} reps)")
            for rep in range(n_reps):
                print(f"    rep {rep+1:02d}/{n_reps} ...", end=" ", flush=True)
                t0 = time.time()
                r = run_one(mname, pname, rep, n_reps)
                elapsed = time.time() - t0
                print(f"HV={r['hv_final']:.4f}  "
                      f"IGD={r['igd_final']:.4f}  "
                      f"CVR={r['cvr_final']:.4f}  "
                      f"({elapsed:.1f}s)")

    total_elapsed = time.time() - t_total_start
    print(f"\n  Total wall time: {total_elapsed/60:.1f} min")

    print("\n[3/3] Aggregating results...")
    summary_all = {}
    for pname in problems:
        problem_summary = {}
        for mname in ALL_METHODS:
            agg = aggregate(pname, mname, n_reps)
            if agg:
                problem_summary[mname] = agg
        if problem_summary:
            summary_all[pname] = problem_summary
            summary_path = os.path.join(RESULTS_ROOT, pname, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(problem_summary, f, indent=2)
            print(f"  Saved {pname}/summary.json")

    all_summary_path = os.path.join(RESULTS_ROOT, "summary_all.json")
    with open(all_summary_path, "w") as f:
        json.dump(summary_all, f, indent=2)

    _print_table(summary_all, problems, list(ALL_METHODS.keys()))
    return summary_all


def _print_table(summary, problems, methods):
    col = 22
    h2 = f"{'':14s}"
    for p in problems:
        h2 += f"  {p:^{col}s}"
    print("\n" + "=" * (14 + (col + 2) * len(problems)))
    print(h2)
    h3 = f"{'':14s}"
    for _ in problems:
        h3 += f"  {'HV':>6s} {'IGD':>6s} {'CVR':>6s}"
    print(h3)
    print("-" * (14 + (col + 2) * len(problems)))
    for mname in methods:
        row = f"{mname:<14s}"
        for p in problems:
            if p in summary and mname in summary[p]:
                s = summary[p][mname]
                row += (f"  {s['hv_mean']:>6.4f}"
                        f" {s['igd_mean']:>6.4f}"
                        f" {s['cvr_mean']:>6.4f}")
            else:
                row += f"  {'---':>6s} {'---':>6s} {'---':>6s}"
        print(row)
    print("=" * (14 + (col + 2) * len(problems)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RZDT5_R experiment (revised RZDT5)")
    parser.add_argument("--method",  type=str, default=None)
    parser.add_argument("--problem", type=str, nargs="+", default=None)
    parser.add_argument("--n_reps",  type=int, default=N_REPS)
    args = parser.parse_args()

    methods  = [args.method] if args.method else None
    problems = args.problem

    run_rzdt5r(methods=methods, problems=problems, n_reps=args.n_reps)
