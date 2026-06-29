"""
d=5 Heteroscedastic Benchmark Experiments (Section 6.2 of paper).

Runs all 7 methods on RZDT3 / RZDT4 / RZDT6 with d=5, heteroscedastic
noise, moderate constraint (50% feasible), for N_REPS replications.

Results are saved to:
    results/d5_hetero/
        problems/
            RZDT3_meta.json   <- tau, theoretical PF, feasible PF, PF curve
            RZDT4_meta.json
            RZDT6_meta.json
        RZDT3/
            GPR_KG_rep01.json <- full per-rep data (hv_history, metrics, timing)
            GPR_KG_rep02.json
            ...
            summary.json      <- aggregated stats across reps
        RZDT4/ ...
        RZDT6/ ...
        summary_all.json      <- cross-problem table (for Table 1)

Usage:
    cd GPR_KG_Code
    python -m experiments.run_d5                        # all methods, all problems
    python -m experiments.run_d5 --method GPR-KG        # single method
    python -m experiments.run_d5 --problem RZDT4        # single problem
    python -m experiments.run_d5 --method GPR-KG --problem RZDT4
"""

import sys
import os
import json
import time
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gpr_kg import RZDT1, RZDT2, RZDT3, RZDT4, RZDT5, RZDT6, compute_hypervolume_2d
from metrics import compute_igd, compute_cvr
from experiments.config import (
    DEFAULT_N, DEFAULT_N0, DEFAULT_D, DEFAULT_L, DEFAULT_SIGMA,
    DEFAULT_ALPHA, FEASIBILITY_MODERATE, N_REPS, SEED_BASE,
    REF_POINT, REF_POINT_RZDT5, HV_EVAL_INTERVAL,
)

def get_ref_point(problem_name):
    """Return the appropriate HV reference point for the problem."""
    return REF_POINT_RZDT5 if problem_name == "RZDT5" else REF_POINT
from methods.random_search import RandomSearch
from methods.gpr_kg_method import GPRKGMethod
from methods.gpr_kg_nv import GPRKGnVMethod
from methods.cehvi_method import cEHVIMethod
from methods.cparego_method import cParEGOMethod
from methods.nsga2_kriging import NSGA2Kriging
from methods.nsga2_direct import NSGA2Direct

# ── output root ──────────────────────────────────────────────
RESULTS_ROOT = os.path.join(BASE_DIR, "results", "d5_hetero")

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

# ── problem registry ─────────────────────────────────────────
PROBLEM_CLASSES = {
    "RZDT1": RZDT1,
    "RZDT2": RZDT2,
    "RZDT3": RZDT3,
    "RZDT4": RZDT4,
    "RZDT5": RZDT5,
    "RZDT6": RZDT6,
}

# Safe file name for a method
def _safe(name):
    return name.replace("-", "_")

def rep_path(problem_name, method_name, rep):
    """Path for a single replication result file."""
    return os.path.join(
        RESULTS_ROOT, problem_name,
        f"{_safe(method_name)}_rep{rep+1:02d}.json"
    )


# ─────────────────────────────────────────────────────────────
# 1.  Save problem metadata (theoretical Pareto front etc.)
# ─────────────────────────────────────────────────────────────

def save_problem_meta(problem_name):
    """Compute and save theoretical Pareto front + problem metadata.

    Saved once per problem to results/d5_hetero/problems/<name>_meta.json.
    Used by the plotting scripts to overlay the true PF on convergence plots.
    """
    meta_dir = os.path.join(RESULTS_ROOT, "problems")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f"{problem_name}_meta.json")

    if os.path.exists(meta_path):
        return  # already saved

    cls = PROBLEM_CLASSES[problem_name]
    prob = cls(d=DEFAULT_D, L=DEFAULT_L, sigma=DEFAULT_SIGMA,
               heteroscedastic=True, alpha=DEFAULT_ALPHA)
    prob.calibrate_constraint(FEASIBILITY_MODERATE)

    # True Pareto front: discrete feasible points (f1, f2)
    true_pf = prob.true_pareto_front()          # shape (n, 2)

    # Full PF on all 20 grid points (ignoring constraint) for reference
    all_pf_pts = []
    for x1 in range(1, prob.L + 1):
        x = tuple([x1] + [1] * (prob.d - 1))
        f1, f2, _ = prob.true_objectives(x)
        is_feas = prob.is_truly_feasible(x)
        cv_val = f1  # placeholder, just collect feasibility flag
        all_pf_pts.append({"x1": x1, "f1": f1, "f2": f2,
                            "feasible": bool(is_feas)})

    # Continuous reference curve for plotting (if available)
    if hasattr(prob, 'true_pareto_curve'):
        f1_curve, f2_curve = prob.true_pareto_curve()
    else:
        # Discrete PF (e.g., RZDT5) — use the grid points as the curve
        f1_curve = [p["f1"] for p in all_pf_pts]
        f2_curve = [p["f2"] for p in all_pf_pts]

    meta = {
        "problem": problem_name,
        "d": DEFAULT_D,
        "L": DEFAULT_L,
        "sigma": DEFAULT_SIGMA,
        "alpha": DEFAULT_ALPHA,
        "tau": float(prob.tau),
        "feasibility_ratio": FEASIBILITY_MODERATE,
        "ref_point": get_ref_point(problem_name).tolist(),
        # Feasible discrete Pareto front (the "ground truth" for HV/IGD)
        "true_pf_feasible": true_pf.tolist() if len(true_pf) > 0 else [],
        "n_feasible_pf": int(len(true_pf)),
        # All 20 grid Pareto points with feasibility flag
        "all_pf_grid": all_pf_pts,
        # Dense curve for plotting (100 continuous points)
        "pf_curve_f1": f1_curve if isinstance(f1_curve, list) else f1_curve.tolist(),
        "pf_curve_f2": f2_curve if isinstance(f2_curve, list) else f2_curve.tolist(),
        # HV of the full feasible true Pareto front (upper bound)
        "hv_true_pf": float(compute_hypervolume_2d(true_pf, get_ref_point(problem_name)))
                      if len(true_pf) > 0 else 0.0,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [meta] {problem_name}: tau={prob.tau:.4f}, "
          f"feasible_PF={len(true_pf)}/20, "
          f"HV_true={meta['hv_true_pf']:.4f}")


# ─────────────────────────────────────────────────────────────
# 2.  Run a single (method, problem, rep)
# ─────────────────────────────────────────────────────────────

def run_one(method_name, problem_name, rep, n_reps_total):
    """Run one replication.  Skip if result file already exists (resume)."""
    path = rep_path(problem_name, method_name, rep)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(f"    [skip] rep {rep+1:02d}/{n_reps_total} already exists")
        return json.load(open(path))

    seed = SEED_BASE + rep

    # Instantiate fresh problem each rep (same tau via same seed=42 inside calibrate)
    cls = PROBLEM_CLASSES[problem_name]
    prob = cls(d=DEFAULT_D, L=DEFAULT_L, sigma=DEFAULT_SIGMA,
               heteroscedastic=True, alpha=DEFAULT_ALPHA)
    prob.calibrate_constraint(FEASIBILITY_MODERATE)

    method = ALL_METHODS[method_name]()

    t0 = time.time()
    result = method.run(prob, N=DEFAULT_N, n0=DEFAULT_N0, seed=seed)
    wall = time.time() - t0

    # Augment with experiment metadata
    result["method"]          = method_name
    result["problem"]         = problem_name
    result["rep"]             = rep + 1
    result["seed"]            = seed
    result["N"]               = DEFAULT_N
    result["n0"]              = DEFAULT_N0
    result["d"]               = DEFAULT_D
    result["heteroscedastic"] = True
    result["tau"]             = float(prob.tau)
    result["wall_time_sec"]   = float(wall)

    # Normalised HV: fraction of true-PF HV achieved
    ref = get_ref_point(problem_name)
    true_pf = prob.true_pareto_front()
    hv_upper = float(compute_hypervolume_2d(true_pf, ref)) if len(true_pf) > 0 else 1.0
    result["hv_upper"] = hv_upper
    result["hv_ratio"] = result["hv_final"] / hv_upper if hv_upper > 0 else 0.0

    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ─────────────────────────────────────────────────────────────
# 3.  Aggregate results for one (method, problem) pair
# ─────────────────────────────────────────────────────────────

def aggregate(problem_name, method_name, n_reps):
    """Load completed reps and compute mean ± SE for each metric."""
    records = []
    for rep in range(n_reps):
        path = rep_path(problem_name, method_name, rep)
        if os.path.exists(path):
            records.append(json.load(open(path)))

    if not records:
        return None

    def _stat(key):
        vals = [r[key] for r in records if key in r]
        if not vals:
            return None, None
        arr = np.array(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

    # Convergence curves: align by budget step, average HV across reps
    # Each hv_history is a list of [budget, hv] pairs
    hv_histories = [r.get("hv_history", []) for r in records]
    # Find common budget steps
    if hv_histories and hv_histories[0]:
        steps = [pair[0] for pair in hv_histories[0]]
        hv_at_step = {}
        for step in steps:
            vals = []
            for hist in hv_histories:
                match = [h[1] for h in hist if h[0] == step]
                if match:
                    vals.append(match[0])
            if vals:
                hv_at_step[step] = {
                    "mean": float(np.mean(vals)),
                    "se":   float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0,
                    "n":    len(vals),
                }
    else:
        hv_at_step = {}

    hv_m,   hv_se   = _stat("hv_final")
    igd_m,  igd_se  = _stat("igd_final")
    cvr_m,  cvr_se  = _stat("cvr_final")
    tpi_m,  tpi_se  = _stat("time_per_iter_mean")
    wall_m, wall_se = _stat("wall_time_sec")
    ratio_m, _      = _stat("hv_ratio")

    return {
        "method":              method_name,
        "problem":             problem_name,
        "n_reps":              len(records),
        "hv_mean":             hv_m,   "hv_se":  hv_se,
        "igd_mean":            igd_m,  "igd_se": igd_se,
        "cvr_mean":            cvr_m,  "cvr_se": cvr_se,
        "hv_ratio_mean":       ratio_m,
        "time_per_iter_mean":  tpi_m,  "time_per_iter_se": tpi_se,
        "wall_time_mean":      wall_m, "wall_time_se": wall_se,
        "hv_values":  [r["hv_final"]  for r in records],
        "igd_values": [r["igd_final"] for r in records],
        "cvr_values": [r["cvr_final"] for r in records],
        "hv_convergence":      hv_at_step,   # {budget_step: {mean, se, n}}
    }


# ─────────────────────────────────────────────────────────────
# 4.  Master runner
# ─────────────────────────────────────────────────────────────

def run_d5(methods=None, problems=None, n_reps=N_REPS):
    """Run all (method, problem) combinations for d=5."""
    if methods  is None: methods  = list(ALL_METHODS.keys())
    if problems is None: problems = list(PROBLEM_CLASSES.keys())

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print("\n" + "=" * 70)
    print("  d=5 Heteroscedastic Benchmark Experiments")
    print(f"  Problems : {problems}")
    print(f"  Methods  : {methods}")
    print(f"  Reps     : {n_reps}  |  N={DEFAULT_N}  n0={DEFAULT_N0}")
    print(f"  Results  : {RESULTS_ROOT}")
    print("=" * 70)

    # ── Save theoretical Pareto fronts once per problem ──────
    print("\n[1/3] Saving problem metadata and theoretical Pareto fronts...")
    for pname in problems:
        save_problem_meta(pname)

    # ── Run experiments ──────────────────────────────────────
    print("\n[2/3] Running experiments...")
    t_total_start = time.time()

    for pname in problems:
        for mname in methods:
            print(f"\n  ── {mname:12s} on {pname} ({n_reps} reps) ──")
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

    # ── Aggregate and save summaries ─────────────────────────
    print("\n[3/3] Aggregating results...")
    summary_all = {}

    for pname in problems:
        problem_summary = {}
        for mname in ALL_METHODS:   # always aggregate all methods if data exists
            agg = aggregate(pname, mname, n_reps)
            if agg:
                problem_summary[mname] = agg

        if problem_summary:
            summary_all[pname] = problem_summary
            summary_path = os.path.join(RESULTS_ROOT, pname, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(problem_summary, f, indent=2)
            print(f"  Saved {pname}/summary.json")

    # Cross-problem summary
    all_summary_path = os.path.join(RESULTS_ROOT, "summary_all.json")
    with open(all_summary_path, "w") as f:
        json.dump(summary_all, f, indent=2)
    print(f"  Saved summary_all.json")

    _print_table(summary_all, problems, list(ALL_METHODS.keys()))
    return summary_all


# ─────────────────────────────────────────────────────────────
# 5.  Console table
# ─────────────────────────────────────────────────────────────

def _print_table(summary, problems, methods):
    """Print a compact results table to console."""
    col = 22
    header = f"{'Method':<14s}"
    for p in problems:
        header += f"  {'':^{col}s}"
    print("\n" + "=" * (14 + (col + 2) * len(problems)))
    # Problem header
    h2 = f"{'':14s}"
    for p in problems:
        h2 += f"  {p:^{col}s}"
    print(h2)
    # Metric header
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


# ─────────────────────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run d=5 heteroscedastic benchmark experiments")
    parser.add_argument("--method",  type=str, default=None,
                        help="Single method name, e.g. GPR-KG")
    parser.add_argument("--problem", type=str, nargs="+", default=None,
                        help="Problem name(s), e.g. RZDT1 RZDT2 RZDT5")
    parser.add_argument("--n_reps",  type=int, default=N_REPS,
                        help=f"Number of replications (default: {N_REPS})")
    args = parser.parse_args()

    methods  = [args.method]  if args.method  else None
    problems = args.problem  # already a list or None

    run_d5(methods=methods, problems=problems, n_reps=args.n_reps)
