"""
Phase B: VEPM partition-scheme ablation on RZDT1, RZDT2, RZDT5_RR (d=5).

Runs GPR-KG with three VEPM partition schemes:
  - 'medoid_K'        K-means clustering with K = ceil(sqrt(n0*4)) ~ 20
  - 'binary_bin_raw'  original Zheng-style raw 2d-feature scheme (1024 cells
                      at d=5, intentionally degenerate — paper baseline)
  - 'aggregate'       four-feature [mean, std, max, min] ablation

GPR-KG-nV (pooled variance) is unaffected by the partition scheme; we
reuse the existing R=10 results in results/d5_v2 / results/rzdt5rr.

Usage:
    python -u -m experiments.run_phaseB
    python -u -m experiments.run_phaseB --method medoid_K --problem RZDT1
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
from experiments.config import (
    DEFAULT_N, DEFAULT_N0, REF_POINT, REF_POINT_RZDT5,
    DEFAULT_ALPHA, FEASIBILITY_MODERATE, SEED_BASE,
)

# Map RZDT5_RR to its specific reference point (denser grid).
PROBLEM_CLASSES = {
    'RZDT1':   RZDT1,
    'RZDT2':   RZDT2,
    'RZDT5_RR': RZDT5_RR,
}

PARTITION_SCHEMES = ['medoid_K', 'binary_bin_raw', 'aggregate']
# - 'binary_bin_raw' is an alias for the new 'binary_bin' (paper Zheng raw,
#   no fallback); kept for backward compat with previously-saved JSONs.
# - 'aggregate' is the 4-feature [mean,std,max,min] ablation that destroys
#   per-coordinate alignment (see paper Section 4.3.1 / Table tab:alignment).
DEFAULT_D = 5
DEFAULT_L = 100
DEFAULT_SIGMA = 0.04
N_REPS = 10


def get_ref_point(name):
    return REF_POINT_RZDT5 if name in ('RZDT5', 'RZDT5_R', 'RZDT5_RR') else REF_POINT


def rep_path(problem, scheme, rep):
    return os.path.join(BASE_DIR, 'results', 'phaseB', problem,
                        f'GPR_KG_{scheme}_rep{rep:02d}.json')


def run_one(problem_name, scheme, rep):
    path = rep_path(problem_name, scheme, rep)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return json.load(open(path))

    seed = SEED_BASE + rep

    cls = PROBLEM_CLASSES[problem_name]
    if problem_name == 'RZDT5_RR':
        prob = cls(d=DEFAULT_D, sigma=DEFAULT_SIGMA, heteroscedastic=True,
                   alpha=DEFAULT_ALPHA)
    else:
        prob = cls(d=DEFAULT_D, L=DEFAULT_L, sigma=DEFAULT_SIGMA,
                   heteroscedastic=True, alpha=DEFAULT_ALPHA)
    # Paper Section 6 (lines 1737/1752 etc.) uses strict tau=0 for the three
    # primary RZDT problems (RZDT1, RZDT2, RZDT5_RR); the moderate
    # FEASIBILITY_MODERATE path is for RZDT3/4/6 only.  Match the paper.
    prob.tau = 0.0

    method = GPRKGMethod(partition_method=scheme)

    t0 = time.time()
    result = method.run(prob, N=DEFAULT_N, n0=DEFAULT_N0, seed=seed)
    wall = time.time() - t0

    result.update({
        'method':           f'GPR-KG-{scheme}',
        'problem':          problem_name,
        'rep':              rep + 1,
        'seed':             seed,
        'N':                DEFAULT_N,
        'n0':               DEFAULT_N0,
        'd':                DEFAULT_D,
        'partition_method': scheme,
        'tau':              float(prob.tau),
        'wall_time_sec':    float(wall),
    })

    ref = get_ref_point(problem_name)
    true_pf = prob.true_pareto_front()
    hv_upper = float(compute_hypervolume_2d(true_pf, ref)) if len(true_pf) > 0 else 1.0
    result['hv_upper'] = hv_upper
    result['hv_ratio'] = result['hv_final'] / hv_upper if hv_upper > 0 else 0.0

    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    return result


def aggregate(problem_name, scheme, n_reps=N_REPS):
    records = []
    for rep in range(n_reps):
        path = rep_path(problem_name, scheme, rep)
        if os.path.exists(path):
            records.append(json.load(open(path)))
    if not records:
        return None

    def stat(key):
        vals = [r[key] for r in records if key in r]
        if not vals:
            return None, None
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    hv_m,  hv_se  = stat('hv_final')
    igd_m, igd_se = stat('igd_final')
    cvr_m, cvr_se = stat('cvr_final')
    wt_m,  wt_se  = stat('wall_time_sec')
    hvr_m, hvr_se = stat('hv_ratio')

    return {
        'method':         f'GPR-KG-{scheme}',
        'problem':        problem_name,
        'partition_method': scheme,
        'n_reps':         len(records),
        'hv_mean':        hv_m,  'hv_se':  hv_se,
        'igd_mean':       igd_m, 'igd_se': igd_se,
        'cvr_mean':       cvr_m, 'cvr_se': cvr_se,
        'hv_ratio_mean':  hvr_m, 'hv_ratio_se': hvr_se,
        'wall_time_mean': wt_m,  'wall_time_se': wt_se,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', nargs='+', default=list(PROBLEM_CLASSES.keys()),
                        choices=list(PROBLEM_CLASSES.keys()))
    parser.add_argument('--method',  nargs='+', default=PARTITION_SCHEMES,
                        choices=PARTITION_SCHEMES)
    parser.add_argument('--reps',    type=int,  default=N_REPS)
    args = parser.parse_args()

    print('=' * 70)
    print(f' Phase B: VEPM partition ablation')
    print(f'   Problems:  {args.problem}')
    print(f'   Schemes:   {args.method}')
    print(f'   Reps:      {args.reps}    N={DEFAULT_N}  n0={DEFAULT_N0}  d={DEFAULT_D}')
    print('=' * 70, flush=True)

    t_start = time.time()
    for pname in args.problem:
        for scheme in args.method:
            print(f'\n  ── GPR-KG-{scheme} on {pname} ({args.reps} reps) ──', flush=True)
            for rep in range(args.reps):
                t_rep = time.time()
                r = run_one(pname, scheme, rep)
                dt = time.time() - t_rep
                print(f'    rep {rep+1:02d}/{args.reps}  HV={r["hv_final"]:.3f}  '
                      f'IGD={r["igd_final"]:.3f}  CVR={r["cvr_final"]:.3f}  '
                      f'wall={dt:.0f}s', flush=True)

    print('\n' + '=' * 70)
    print(' Aggregate ')
    print('=' * 70)
    print(f'{"problem":10s} | {"method":24s} | {"HV±SE":>14s} | {"IGD±SE":>14s} | {"CVR±SE":>14s}')
    print('-' * 90)
    summary = {}
    for pname in args.problem:
        summary[pname] = {}
        for scheme in args.method:
            agg = aggregate(pname, scheme, args.reps)
            if agg is None:
                continue
            summary[pname][scheme] = agg
            print(f'{pname:10s} | GPR-KG-{scheme:18s}'
                  f'| {agg["hv_mean"]:>5.3f}±{agg["hv_se"]:>5.3f} '
                  f'| {agg["igd_mean"]:>5.3f}±{agg["igd_se"]:>5.3f} '
                  f'| {agg["cvr_mean"]:>5.3f}±{agg["cvr_se"]:>5.3f}')

    out = os.path.join(BASE_DIR, 'results', 'phaseB', 'summary_phaseB.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved {out}')
    print(f'Total wall: {(time.time()-t_start)/60:.1f}min')


if __name__ == '__main__':
    main()
