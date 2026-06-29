"""
Phase 2 main experiment runner for the ingolstadt21 case study.

Mirror of experiments.intas.run_main with Ingolstadt21Problem.

Output layout under results/ingolstadt21/:
    baseline.json                         -- Phase 0
    hetero_test.json                      -- Phase 1
    {METHOD}_run/                         -- Phase 2, one dir per method
        checkpoint.pkl
        snapshots.jsonl
        summary.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from experiments.ingolstadt21.ingolstadt21_problem import Ingolstadt21Problem
from experiments.ingolstadt21.config import RESULTS_DIR, DEFAULT_N, DEFAULT_N0
from methods.gpr_kg_method import GPRKGMethod
from methods.gpr_kg_nv import GPRKGnVMethod
from gpr_kg import GPRKR_Algorithm
from experiments.config import GPR_KG_PARAMS

METHODS = {
    "GPR-KG":    {"class": GPRKGMethod,   "w_vepm": GPR_KG_PARAMS['w_vepm']},
    "GPR-KG-nV": {"class": GPRKGnVMethod, "w_vepm": None},
}

# Default VEPM partition scheme on ingolstadt21 (d=44).  After the namespace
# cleanup, "binary_bin" is the paper-faithful raw Zheng scheme; at d=44 it
# enters the saturation regime (Corollary partition-saturation), as
# documented in Section 8 of the manuscript.  Override at the CLI with
# --partition {binary_bin, aggregate, medoid_K}.
DEFAULT_PARTITION = "binary_bin"


def _method_dir(method_name: str, partition: str = DEFAULT_PARTITION,
                seed: int = None) -> str:
    """Per-(method, partition, seed) results directory.

    For backward compatibility, when partition == DEFAULT_PARTITION and
    seed is None or 100 we keep the original ``{METHOD}_run/`` layout so
    pre-cleanup checkpoints/summaries remain in place.  Otherwise we use
    ``{METHOD}_{partition}_seed{seed}/`` to avoid collisions.
    """
    safe = method_name.replace("-", "_")
    if partition == DEFAULT_PARTITION and (seed is None or seed == 100):
        sub = f"{safe}_run"
    else:
        seed_tag = f"seed{seed}" if seed is not None else "seedDefault"
        sub = f"{safe}_{partition}_{seed_tag}"
    d = os.path.join(RESULTS_DIR, sub)
    os.makedirs(d, exist_ok=True)
    return d


def _apply_nv_patch_if_needed(alg, method_name: str):
    if method_name != 'GPR-KG-nV':
        return
    alg.vepm._pooled_only = True
    print(f"  [nV] VEPM set to pooled-only mode", flush=True)


def _select_reference_plan(prob: Ingolstadt21Problem):
    het_path = os.path.join(RESULTS_DIR, 'hetero_test.json')
    if os.path.exists(het_path):
        het = json.load(open(het_path))
        plans = het['results']
        best = min(plans, key=lambda r: r['mean_f1'])
        return tuple(int(v) for v in best['plan']), best['plan_id']
    return tuple(int(v) for v in prob.default_x), -1


def _build_instrument_config(prob: Ingolstadt21Problem):
    ref_x, ref_src_id = _select_reference_plan(prob)
    true_sigma2_ref = None
    het_path = os.path.join(RESULTS_DIR, 'hetero_test.json')
    if os.path.exists(het_path) and ref_src_id >= 0:
        het = json.load(open(het_path))
        rec = het['results'][ref_src_id]
        true_sigma2_ref = [rec['var_f1'], rec['var_f2'], rec['var_f3']]
    return {
        'stride': 10,
        'eval_x': [],
        'ref_x':  ref_x,
        '_ref_x_source_id': ref_src_id,
        '_true_sigma2_ref': true_sigma2_ref,
    }


def run_one(method_name: str, N: int, n0: int, seed: int,
            partition: str = DEFAULT_PARTITION, verbose=True):
    out_dir = _method_dir(method_name, partition=partition, seed=seed)
    ckpt_path = os.path.join(out_dir, 'checkpoint.pkl')
    snap_path = os.path.join(out_dir, 'snapshots.jsonl')
    summary_path = os.path.join(out_dir, 'summary.json')

    if os.path.exists(summary_path):
        print(f"[skip] {method_name} (part={partition}, seed={seed}): summary.json exists",
              flush=True)
        return json.load(open(summary_path))

    prob = Ingolstadt21Problem()
    instr = _build_instrument_config(prob)

    if os.path.exists(ckpt_path):
        print(f"[resume] {method_name} (part={partition}, seed={seed}): "
              f"loading checkpoint...", flush=True)
        alg = GPRKR_Algorithm.restore_checkpoint(
            ckpt_path, problem=prob, instrument=instr)
        alg._checkpoint_path = ckpt_path
        alg._snapshot_jsonl_path = snap_path
        _apply_nv_patch_if_needed(alg, method_name)
        print(f"[resume] {method_name}: presampling_done={alg._presampling_done}  "
              f"main_iter_completed={alg._main_iter_completed}/{N-n0}", flush=True)
    else:
        params = dict(GPR_KG_PARAMS)
        alg = GPRKR_Algorithm(
            problem=prob, N=N, n0=n0,
            K1=params['K1'], K2=params['K2'],
            lambda_i=params['lambda_i'],
            prior_var=params['prior_var'],
            w_vepm=params['w_vepm'],
            n_thr=params['n_thr'],
            seed=seed,
            partition_method=partition,
        )
        alg.instrument = instr
        alg._checkpoint_path = ckpt_path
        alg._snapshot_jsonl_path = snap_path
        _apply_nv_patch_if_needed(alg, method_name)
        print(f"[fresh] {method_name}: N={N}, n0={n0}, seed={seed}", flush=True)

    t_start = time.time()
    final_pareto = alg.run_resumable(verbose=verbose)
    wall = time.time() - t_start

    summary = {
        'method':           method_name,
        'partition_method': partition,
        'problem': 'ingolstadt21',
        'N':       N, 'n0': n0, 'seed': seed,
        'd':       prob.d, 'tau': prob.tau, 'alpha': prob.alpha,
        'T0':      prob.T0, 'A0': prob.A0, 'E0': prob.E0,
        'wall_time_sec':       float(wall),
        'final_pareto_set':    [list(x) for x in final_pareto],
        'final_true_objs':     alg.final_log.get('true_objectives', []) if alg.final_log else [],
        'hv_history':          [(int(n), float(hv)) for n, hv in alg.hv_history],
        'n_iterations':        alg._main_iter_completed,
        'n_unique_solutions':  len(alg.gpr[0].sampled_set) if alg.gpr else 0,
        'instrument_config': {
            'stride':           instr['stride'],
            'ref_x':            list(instr['ref_x']),
            '_ref_x_source_id': instr['_ref_x_source_id'],
            '_true_sigma2_ref': instr['_true_sigma2_ref'],
        },
        'instrument_log':   alg.instrument_log,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, default=str)

    print(f"[done] {method_name}: LNSs={len(final_pareto)}, "
          f"HV_final={summary['hv_history'][-1][1] if summary['hv_history'] else 0:.4f}, "
          f"wall={wall/3600:.2f}h", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", nargs="+", default=list(METHODS.keys()),
                        choices=list(METHODS.keys()))
    parser.add_argument("--partition", default=DEFAULT_PARTITION,
                        choices=["binary_bin", "aggregate", "medoid_K"],
                        help="VEPM partition scheme (default: %(default)s)")
    parser.add_argument("--N",    type=int, default=DEFAULT_N)
    parser.add_argument("--n0",   type=int, default=DEFAULT_N0)
    parser.add_argument("--seed", type=int, nargs="+", default=[100],
                        help="One or more macro-replication seeds")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"  Phase 2: ingolstadt21 main experiment", flush=True)
    print(f"  Methods  : {args.method}", flush=True)
    print(f"  Partition: {args.partition}", flush=True)
    print(f"  Seeds    : {args.seed}", flush=True)
    print(f"  Budget   : n0={args.n0}, N={args.N} per (method, seed)", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    for seed in args.seed:
        for mname in args.method:
            print(f"\n[+] Starting {mname}  partition={args.partition}  seed={seed}",
                  flush=True)
            run_one(mname, N=args.N, n0=args.n0, seed=seed,
                    partition=args.partition, verbose=True)
    total_h = (time.time() - t_start) / 3600
    print(f"\n[all done] total wall time: {total_h:.2f}h", flush=True)


if __name__ == "__main__":
    main()
