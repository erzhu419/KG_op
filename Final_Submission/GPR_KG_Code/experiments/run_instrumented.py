"""
Instrumented re-run of GPR-KG and GPR-KG-nV on RZDT1/RZDT2/RZDT5_RR.

Adds per-iteration (stride=10) recording of:
  * posterior mean mu_i(x) at a fixed eval_set    -> Fig.3 (RMSE history)
  * posterior feasible Pareto set (x tuples)      -> Fig.4 (#infeasible trace)
  * VEPM/pooled variance estimate at ref TPOS x*  -> Fig.5(a) (variance tracking)

Uses the SAME problem configs as run_d5_v2 and run_rzdt5rr (d=5, sigma=0.04,
tau=0, heteroscedastic=True, N=150, n0=30).  Results go to a separate
folder so existing 10-rep results are untouched.

Usage:
    python -m experiments.run_instrumented                # all, 10 reps
    python -m experiments.run_instrumented --n_reps 3
    python -m experiments.run_instrumented --problem RZDT2
"""

import os
import sys
import json
import time
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gpr_kg import RZDT1, RZDT2, RZDT5_RR, compute_hypervolume_2d
from methods.gpr_kg_method import GPRKGMethod
from methods.gpr_kg_nv    import GPRKGnVMethod
from experiments.config   import DEFAULT_N, DEFAULT_N0, DEFAULT_D, DEFAULT_ALPHA, SEED_BASE


RESULTS_ROOT = os.path.join(BASE_DIR, "results", "instrumented")
N_REPS_DEFAULT = 10
STRIDE = 10
SIGMA  = 0.04
TAU    = 0.0

METHOD_CLASSES = {
    "GPR-KG":    GPRKGMethod,
    "GPR-KG-nV": GPRKGnVMethod,
}


def _safe(name):
    return name.replace("-", "_")


def make_problem(problem_name):
    if problem_name == "RZDT1":
        p = RZDT1(d=DEFAULT_D, sigma=SIGMA, heteroscedastic=True, alpha=DEFAULT_ALPHA)
    elif problem_name == "RZDT2":
        p = RZDT2(d=DEFAULT_D, sigma=SIGMA, heteroscedastic=True, alpha=DEFAULT_ALPHA)
    elif problem_name == "RZDT5_RR":
        p = RZDT5_RR(d=DEFAULT_D, sigma=SIGMA, heteroscedastic=True, alpha=DEFAULT_ALPHA)
    else:
        raise ValueError(f"Unknown problem: {problem_name}")
    p.tau = TAU
    return p


def _tpos_points(prob):
    """Return the list of true Pareto-optimal integer points (x_j=0 for j>=2)."""
    lo, hi = prob.int_bounds()
    tpos = []
    for x1 in range(lo[0], hi[0] + 1):
        x = tuple([x1] + [0] * (prob.d - 1))
        if prob.is_truly_feasible(x):
            tpos.append(x)
    return tpos


def _random_non_tpos_points(prob, n_rand=30, seed=0):
    """Some off-axis points for RMSE breadth."""
    rng = np.random.default_rng(seed)
    pts = set()
    lo, hi = prob.int_bounds()
    while len(pts) < n_rand:
        x = tuple(int(rng.integers(lo[j], hi[j] + 1)) for j in range(prob.d))
        # ensure at least one x_j != 0 for j>=2 (off the TPOS axis)
        if any(x[j] != 0 for j in range(1, prob.d)):
            pts.add(x)
    return list(pts)


def build_instrument_config(prob):
    """eval_x (for RMSE) and ref_x (for variance tracking) per problem.

    eval_x = all TPOS points (on-axis) + 30 off-axis random integer points.
    ref_x  = a *middle* TPOS point that is unlikely to be sampled early
             (sits between left and right feasible tails for RZDT1/RZDT2,
             or mid feasible range for RZDT5_RR).
    """
    tpos = _tpos_points(prob)
    off_axis = _random_non_tpos_points(prob, n_rand=30, seed=42)
    eval_x = tpos + off_axis

    # Pick ref_x as a TPOS point with non-extreme x_1
    if len(tpos) == 0:
        ref_x = tpos[0] if tpos else None
    else:
        # Sort by x_1 and pick one in the middle of TPOS range
        tpos_sorted = sorted(tpos, key=lambda t: t[0])
        ref_x = tpos_sorted[len(tpos_sorted) // 2]

    true_means = np.array([prob.true_objectives(x) for x in eval_x])  # (K,3)
    true_sigmas = np.array([prob.true_sigma(x) for x in eval_x])  # (K,3)
    true_sigma_ref = [float(s) for s in prob.true_sigma(ref_x)]

    return {
        'stride': STRIDE,
        'eval_x': [tuple(int(v) for v in x) for x in eval_x],
        'ref_x':  tuple(int(v) for v in ref_x) if ref_x is not None else None,
        '_true_means':  true_means.tolist(),
        '_true_sigmas': true_sigmas.tolist(),
        '_true_sigma_ref': true_sigma_ref,
        '_n_tpos': len(tpos),
    }


def _rep_path(problem_name, method_name, rep):
    return os.path.join(RESULTS_ROOT, problem_name,
                        f"{_safe(method_name)}_rep{rep+1:02d}.json")


def run_one(problem_name, method_name, rep, n_reps_total):
    path = _rep_path(problem_name, method_name, rep)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return json.load(open(path))

    seed  = SEED_BASE + rep
    prob  = make_problem(problem_name)
    instr = build_instrument_config(prob)

    method = METHOD_CLASSES[method_name]()
    method.instrument = {
        'stride': instr['stride'],
        'eval_x': instr['eval_x'],
        'ref_x':  instr['ref_x'],
    }

    t0 = time.time()
    result = method.run(prob, N=DEFAULT_N, n0=DEFAULT_N0, seed=seed)
    wall = time.time() - t0

    result['method']  = method_name
    result['problem'] = problem_name
    result['rep']     = rep + 1
    result['seed']    = seed
    result['N']       = DEFAULT_N
    result['n0']      = DEFAULT_N0
    result['d']       = DEFAULT_D
    result['tau']     = float(prob.tau)
    result['sigma']   = SIGMA
    result['wall_time_sec'] = float(wall)
    # Ground truth shared across reps
    result['instrument_config'] = {
        'stride':          instr['stride'],
        'eval_x':          instr['eval_x'],
        'ref_x':           instr['ref_x'],
        'true_means':      instr['_true_means'],
        'true_sigmas':     instr['_true_sigmas'],
        'true_sigma_ref':  instr['_true_sigma_ref'],
        'n_tpos':          instr['_n_tpos'],
    }
    result['instrument_log'] = method.instrument_log

    with open(path, "w") as f:
        json.dump(result, f, default=str)

    print(f"    [done] {problem_name}/{method_name} rep{rep+1:02d}  "
          f"HV={result['hv_final']:.4f}  "
          f"snaps={len(method.instrument_log)}  "
          f"t={wall:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, nargs="+",
                        default=["RZDT1", "RZDT2", "RZDT5_RR"])
    parser.add_argument("--method", type=str, nargs="+",
                        default=["GPR-KG", "GPR-KG-nV"])
    parser.add_argument("--n_reps", type=int, default=N_REPS_DEFAULT)
    args = parser.parse_args()

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print("=" * 70)
    print(f"  Instrumented re-run  (stride={STRIDE}, n_reps={args.n_reps})")
    print(f"  Problems : {args.problem}")
    print(f"  Methods  : {args.method}")
    print(f"  Results  : {RESULTS_ROOT}")
    print("=" * 70)

    t_start = time.time()
    for pname in args.problem:
        print(f"\n[+] Problem {pname}")
        for mname in args.method:
            print(f"  -- {mname} --")
            for rep in range(args.n_reps):
                run_one(pname, mname, rep, args.n_reps)

    total_min = (time.time() - t_start) / 60.0
    print(f"\n[done] total wall time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
