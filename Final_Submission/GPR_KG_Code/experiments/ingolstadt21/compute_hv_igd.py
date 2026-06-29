"""
Post-hoc HV + IGD analysis for the ingolstadt21 Phase 2 runs.

Reads:
  results/ingolstadt21/{GPR_KG,GPR_KG_nV}_run/snapshots.jsonl
  results/ingolstadt21/{GPR_KG,GPR_KG_nV}_run/summary.json (for instrument_log
    which contains posterior pareto_set every stride iters)

Computes:
  Per-iter Pareto set (running, sample-derived) and HV
  Reference Pareto set = union of all observed (Y_observed) across both
    methods, restricted to f3 <= tau, then non-dominated on (f1, f2).
  IGD(t) = 1/|R| * sum over r in R of min_{p in P(t)} ||r - p||
  where P(t) is the algorithm's posterior Pareto set at iter t (from
    instrument_log) projected to (f1, f2).

Saves:
  results/ingolstadt21/hv_igd_summary.json
  results/figures/fig_hv_ingolstadt21.{pdf,png}
  results/figures/fig_igd_ingolstadt21.{pdf,png}

Usage:
    python -m experiments.ingolstadt21.compute_hv_igd
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ingolstadt21.config import RESULTS_DIR, TAU_EMISSION

FIG_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

REF_POINT = np.array([1.5, 1.5])    # for HV in (f1, f2) space


def load_method(method_name: str):
    """Returns dict with snapshot data and instrument_log Pareto sets."""
    safe = method_name.replace("-", "_")
    run_dir = os.path.join(RESULTS_DIR, f"{safe}_run")
    snap_path = os.path.join(run_dir, "snapshots.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")

    iters, Ys, hvs = [], [], []
    with open(snap_path) as f:
        for line in f:
            r = json.loads(line)
            if 'Y_observed' not in r:
                continue
            iters.append(r['iter'])
            Ys.append(r['Y_observed'])
            hvs.append(r['hv'])

    summary = json.load(open(summary_path))
    instrument_log = summary.get('instrument_log', [])

    return {
        'method':        method_name,
        'iters':         np.array(iters),
        'Y_observed':    np.array(Ys),
        'hv_alg':        np.array(hvs),
        'instrument':    instrument_log,
        'final_pareto':  summary.get('final_pareto_set', []),
        'final_true':    summary.get('final_true_objs', []),
        'tau':           summary.get('tau', TAU_EMISSION),
    }


def is_pareto(pts: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (minimisation in all dims)."""
    n = len(pts)
    eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not eff[i]:
            continue
        dom = (pts <= pts[i]).all(axis=1) & (pts < pts[i]).any(axis=1)
        dom[i] = False
        if dom.any():
            eff[i] = False
    return eff


def pareto_hv_2d(P: np.ndarray, ref: np.ndarray) -> float:
    """2-D hypervolume relative to ref (assumes minimisation)."""
    if len(P) == 0:
        return 0.0
    Pclip = P.copy()
    Pclip[:, 0] = np.minimum(Pclip[:, 0], ref[0])
    Pclip[:, 1] = np.minimum(Pclip[:, 1], ref[1])
    Pclip = Pclip[(Pclip[:, 0] < ref[0]) & (Pclip[:, 1] < ref[1])]
    if len(Pclip) == 0:
        return 0.0
    P_sorted = Pclip[Pclip[:, 0].argsort()]
    hv = 0.0
    prev_y = ref[1]
    for x, y in P_sorted:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return hv


def igd(P: np.ndarray, R: np.ndarray) -> float:
    """Inverted Generational Distance: avg over R of min distance to P."""
    if len(P) == 0:
        return float('nan')
    if len(R) == 0:
        return 0.0
    # Pairwise Euclidean distance
    D = np.sqrt(((R[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1))
    return float(D.min(axis=1).mean())


def build_running_pareto(Y: np.ndarray, tau: float) -> list:
    """Per-iter running Pareto set in (f1, f2) for points with f3 <= tau.
    Returns list of np.ndarray (each can be empty if no feasible yet)."""
    feasible_mask = Y[:, 2] <= tau
    pf_list = []
    for t in range(len(Y)):
        seen = Y[:t+1]
        seen_f = seen[feasible_mask[:t+1]]
        if len(seen_f) == 0:
            pf_list.append(np.empty((0, 2)))
            continue
        pts = seen_f[:, :2]
        eff = is_pareto(pts)
        pf_list.append(pts[eff])
    return pf_list


def reference_pareto_from_runs(runs: dict, tau: float) -> np.ndarray:
    """Union of all observed solutions across methods, filtered for
    f3 <= tau, projected to (f1, f2), kept non-dominated.  This is
    'the best Pareto front anyone has seen' — a proxy for the true PF."""
    pool = []
    for r in runs.values():
        Y = r['Y_observed']
        feasible = Y[Y[:, 2] <= tau, :2]
        if len(feasible) > 0:
            pool.append(feasible)
    if not pool:
        return np.empty((0, 2))
    pool = np.vstack(pool)
    eff = is_pareto(pool)
    return pool[eff]


def main():
    runs = {}
    for m in ['GPR-KG', 'GPR-KG-nV']:
        try:
            runs[m] = load_method(m)
            print(f"loaded {m}: {len(runs[m]['Y_observed'])} obs, "
                  f"{len(runs[m]['instrument'])} instrument snaps, "
                  f"tau={runs[m]['tau']}")
        except FileNotFoundError as e:
            print(f"skip {m}: {e}")

    if len(runs) == 0:
        print("No completed runs found.")
        return

    tau_used = list(runs.values())[0]['tau']
    R_pareto = reference_pareto_from_runs(runs, tau_used)
    print(f"\nReference Pareto (union, f3<=tau): {len(R_pareto)} points")
    if len(R_pareto) > 0:
        print(f"  f1 range: [{R_pareto[:,0].min():.3f}, {R_pareto[:,0].max():.3f}]")
        print(f"  f2 range: [{R_pareto[:,1].min():.3f}, {R_pareto[:,1].max():.3f}]")

    # ── Per-method per-iter HV/IGD curves ───────────────────────────────
    summaries = {}
    for m, r in runs.items():
        pf_list = build_running_pareto(r['Y_observed'], tau_used)
        hv_curve = np.array([pareto_hv_2d(pf, REF_POINT) for pf in pf_list])
        igd_curve = np.array([igd(pf, R_pareto) for pf in pf_list])
        feasible_count = (r['Y_observed'][:, 2] <= tau_used).sum()

        summaries[m] = {
            'n_iters':             int(len(r['Y_observed'])),
            'n_feasible':          int(feasible_count),
            'hv_final':            float(hv_curve[-1]),
            'igd_final':           float(igd_curve[-1]) if not np.isnan(igd_curve[-1]) else None,
            'hv_max':              float(np.max(hv_curve)),
            'pareto_size_final':   int(len(pf_list[-1])),
            'best_f1':             float(r['Y_observed'][:, 0].min()),
            'best_f2':             float(r['Y_observed'][:, 1].min()),
            'best_f3':             float(r['Y_observed'][:, 2].min()),
        }
        print(f"\n=== {m} ===")
        for k, v in summaries[m].items():
            print(f"  {k}: {v}")

    # ── Plot HV curve ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {'GPR-KG': 'tab:red', 'GPR-KG-nV': 'tab:blue'}
    for m, r in runs.items():
        pf_list = build_running_pareto(r['Y_observed'], tau_used)
        hv_curve = np.array([pareto_hv_2d(pf, REF_POINT) for pf in pf_list])
        ax.plot(r['iters'], hv_curve, color=colors.get(m, 'k'), label=m, lw=1.6)
    ax.set_xlabel('KG iteration')
    ax.set_ylabel('Hypervolume (running sample Pareto, ref=(1.5,1.5))')
    ax.set_title(rf'ingolstadt21 Phase 2: HV vs iteration ($\tau$={tau_used})')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'fig_hv_ingolstadt21.{ext}'),
                    dpi=180, bbox_inches='tight')
    plt.close(fig)

    # ── Plot IGD curve ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, r in runs.items():
        pf_list = build_running_pareto(r['Y_observed'], tau_used)
        igd_curve = np.array([igd(pf, R_pareto) for pf in pf_list])
        # Replace NaN with NaN for plotting (early iters before feasibility)
        ax.plot(r['iters'], igd_curve, color=colors.get(m, 'k'), label=m, lw=1.6)
    ax.set_xlabel('KG iteration')
    ax.set_ylabel('IGD (vs union-Pareto reference)')
    ax.set_title(rf'ingolstadt21 Phase 2: IGD vs iteration ($\tau$={tau_used})')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'fig_igd_ingolstadt21.{ext}'),
                    dpi=180, bbox_inches='tight')
    plt.close(fig)

    # ── Save numerical summary ───────────────────────────────────────────
    out = {
        'tau_used':              tau_used,
        'reference_pareto_size': int(len(R_pareto)),
        'reference_pareto_f1_range': (
            [float(R_pareto[:, 0].min()), float(R_pareto[:, 0].max())]
            if len(R_pareto) > 0 else None),
        'reference_pareto_f2_range': (
            [float(R_pareto[:, 1].min()), float(R_pareto[:, 1].max())]
            if len(R_pareto) > 0 else None),
        'methods': summaries,
    }
    with open(os.path.join(RESULTS_DIR, 'hv_igd_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved hv_igd_summary.json + fig_hv_ingolstadt21 + fig_igd_ingolstadt21")


if __name__ == "__main__":
    main()
