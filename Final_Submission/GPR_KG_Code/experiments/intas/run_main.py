"""
Phase 2 main experiment runner for the InTAS engineering case study.

Runs GPR-KG (with VEPM) and GPR-KG-nV (pooled-variance ablation) on
InTAS with n0=100 pre-samples plus N=300 sequential KG decisions per
method (Zheng et al. 2019 §5.2 protocol: single run, no macro-reps).

Per-iteration instrumentation:
  * pickle checkpoint of full GPRKR_Algorithm state after each iter
    (enables seamless resume after system-level process termination)
  * JSONL snapshot (one line per iter): timestamp, HV, Pareto set size,
    sim/compute timing, x_selected, Y_observed
  * stride=10 instrument snapshots with sigma^2 at a fixed reference plan
    (for Fig 6.6 VEPM variance tracking) — uses Phase 1 output if present

Usage:
    python -u -m experiments.intas.run_main
    # Options (all optional):
    python -u -m experiments.intas.run_main --method GPR-KG
    python -u -m experiments.intas.run_main --N 100   # for testing

Output layout under results/intas/:
    baseline.json                            -- Phase 0
    hetero_test.json                         -- Phase 1
    {METHOD}_run/                            -- Phase 2, one dir per method
        checkpoint.pkl                       -- full algorithm state
        snapshots.jsonl                      -- per-iteration log
        summary.json                         -- final summary after run
"""

import os
import sys
import json
import time
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from experiments.intas.intas_problem import InTASProblem
from experiments.intas.config import RESULTS_DIR, DEFAULT_N, DEFAULT_N0
from methods.gpr_kg_method import GPRKGMethod
from methods.gpr_kg_nv import GPRKGnVMethod
from gpr_kg import GPRKR_Algorithm
from experiments.config import GPR_KG_PARAMS

METHODS = {
    "GPR-KG":    {"class": GPRKGMethod,   "w_vepm": GPR_KG_PARAMS['w_vepm']},
    "GPR-KG-nV": {"class": GPRKGnVMethod, "w_vepm": None},  # pooled fallback
}


def _method_dir(method_name: str) -> str:
    safe = method_name.replace("-", "_")
    d = os.path.join(RESULTS_DIR, f"{safe}_run")
    os.makedirs(d, exist_ok=True)
    return d


def _apply_nv_patch_if_needed(alg, method_name: str):
    """Flip VEPM into pooled-only mode for the GPR-KG-nV ablation.

    With ``_pooled_only=True``, VEPM.get_variance(i, x) always returns
    the pooled global variance regardless of x — replicating the
    ablation in methods/gpr_kg_nv.py while reusing GPRKR_Algorithm's
    checkpoint/resume infrastructure.
    """
    if method_name != 'GPR-KG-nV':
        return
    alg.vepm._pooled_only = True
    print(f"  [nV] VEPM set to pooled-only mode (global_var fallback)",
          flush=True)


def _select_reference_plan(intas_prob: InTASProblem):
    """Pick a low-congestion reference plan x_ref for VEPM variance
    tracking (Fig 6.6).  If Phase 1 results exist, use the plan with
    lowest mean_f1 (least congested).  Otherwise default to x_0.
    """
    het_path = os.path.join(RESULTS_DIR, 'hetero_test.json')
    if os.path.exists(het_path):
        het = json.load(open(het_path))
        plans = het['results']
        best = min(plans, key=lambda r: r['mean_f1'])
        return tuple(int(v) for v in best['plan']), best['plan_id']
    # Fallback: field plan
    return tuple(int(v) for v in intas_prob.default_x), -1


def _build_instrument_config(prob: InTASProblem):
    """Build instrument dict and record ground-truth values needed for
    post-hoc Fig 6.6 (variance tracking at x_ref)."""
    ref_x, ref_src_id = _select_reference_plan(prob)
    # Ground-truth sigma^2 at ref_x: if Phase 1 has this plan, use its
    # 10-rep variance; else run a quick 10-rep estimate now (skipped).
    true_sigma2_ref = None
    het_path = os.path.join(RESULTS_DIR, 'hetero_test.json')
    if os.path.exists(het_path) and ref_src_id >= 0:
        het = json.load(open(het_path))
        rec = het['results'][ref_src_id]
        true_sigma2_ref = [rec['var_f1'], rec['var_f2'], rec['var_f3']]
    return {
        'stride': 10,
        'eval_x': [],          # no RMSE eval set for InTAS (no TPOS known)
        'ref_x':  ref_x,
        '_ref_x_source_id':    ref_src_id,
        '_true_sigma2_ref':    true_sigma2_ref,
    }


def run_one(method_name: str, N: int, n0: int, seed: int, verbose=True):
    """Run one method end-to-end with checkpoint + resume."""
    out_dir = _method_dir(method_name)
    ckpt_path = os.path.join(out_dir, 'checkpoint.pkl')
    snap_path = os.path.join(out_dir, 'snapshots.jsonl')
    summary_path = os.path.join(out_dir, 'summary.json')

    if os.path.exists(summary_path):
        print(f"[skip] {method_name}: summary.json already exists", flush=True)
        return json.load(open(summary_path))

    prob = InTASProblem()
    instr = _build_instrument_config(prob)

    # Resume from checkpoint if it exists
    if os.path.exists(ckpt_path):
        print(f"[resume] {method_name}: loading checkpoint...", flush=True)
        alg = GPRKR_Algorithm.restore_checkpoint(
            ckpt_path, problem=prob, instrument=instr)
        alg._checkpoint_path = ckpt_path
        alg._snapshot_jsonl_path = snap_path
        _apply_nv_patch_if_needed(alg, method_name)
        print(f"[resume] {method_name}: "
              f"presampling_done={alg._presampling_done}  "
              f"main_iter_completed={alg._main_iter_completed}/{N-n0}",
              flush=True)
    else:
        # Fresh start
        params = dict(GPR_KG_PARAMS)
        alg = GPRKR_Algorithm(
            problem=prob, N=N, n0=n0,
            K1=params['K1'], K2=params['K2'],
            lambda_i=params['lambda_i'],
            prior_var=params['prior_var'],
            w_vepm=params['w_vepm'],
            n_thr=params['n_thr'],
            seed=seed,
        )
        alg.instrument = instr
        alg._checkpoint_path = ckpt_path
        alg._snapshot_jsonl_path = snap_path
        _apply_nv_patch_if_needed(alg, method_name)
        print(f"[fresh] {method_name}: N={N}, n0={n0}, seed={seed}",
              flush=True)

    # Run (resumable)
    t_start = time.time()
    final_pareto = alg.run_resumable(verbose=verbose)
    wall = time.time() - t_start

    # Build summary
    summary = {
        'method':  method_name,
        'problem': 'InTAS',
        'N':       N,
        'n0':      n0,
        'seed':    seed,
        'd':       prob.d,
        'tau':     prob.tau,
        'alpha':   prob.alpha,
        'T0':      prob.T0,
        'A0':      prob.A0,
        'E0':      prob.E0,
        'wall_time_sec':    float(wall),
        'final_pareto_set': [list(x) for x in final_pareto],
        'final_true_objs':  alg.final_log.get('true_objectives', []) if alg.final_log else [],
        'hv_history':       [(int(n), float(hv)) for n, hv in alg.hv_history],
        'n_iterations':     alg._main_iter_completed,
        'n_unique_solutions': len(alg.gpr[0].sampled_set) if alg.gpr else 0,
        'instrument_config': {
            'stride':            instr['stride'],
            'ref_x':             list(instr['ref_x']),
            '_ref_x_source_id':  instr['_ref_x_source_id'],
            '_true_sigma2_ref':  instr['_true_sigma2_ref'],
        },
        'instrument_log':   alg.instrument_log,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, default=str)

    print(f"[done] {method_name}: "
          f"LNSs={len(final_pareto)}, "
          f"HV_final={summary['hv_history'][-1][1] if summary['hv_history'] else 0:.4f}, "
          f"wall={wall/3600:.1f}h",
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", nargs="+", default=list(METHODS.keys()),
                        choices=list(METHODS.keys()))
    parser.add_argument("--N",    type=int, default=DEFAULT_N,
                        help="Sequential KG iterations (default %(default)s)")
    parser.add_argument("--n0",   type=int, default=DEFAULT_N0,
                        help="Pre-samples (default %(default)s)")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"  Phase 2: InTAS main experiment (Zheng 2019 §5.2 protocol)",
          flush=True)
    print(f"  Methods : {args.method}", flush=True)
    print(f"  Budget  : n0={args.n0}, N={args.N} per method "
          f"(total {args.n0 + args.N} sims)", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    for mname in args.method:
        print(f"\n[+] Starting method {mname}", flush=True)
        run_one(mname, N=args.N, n0=args.n0, seed=args.seed, verbose=True)
    total_h = (time.time() - t_start) / 3600
    print(f"\n[all done] total wall time: {total_h:.2f}h", flush=True)


if __name__ == "__main__":
    main()
