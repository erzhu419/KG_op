"""
VEPM vs pooled-variance convergence analysis (Path C / Fig 6.6 style).

Reads instrument_log['sigma2_ref'] from both methods' summary.json, plots how
each method's variance estimate at the reference plan x_ref evolves over
KG iterations, against:
  * the Phase 1 ground-truth variance at x_ref (10-rep estimate)
  * the Phase 1 global-mean variance across all 20 plans (the asymptotic
    target of pooled variance — what GPR-KG-nV converges to)

Saves:
  results/figures/fig_vepm_convergence_ingolstadt21.{pdf,png}
  results/ingolstadt21/vepm_convergence_summary.json

Usage:
    python -m experiments.ingolstadt21.compute_vepm_convergence
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ingolstadt21.config import RESULTS_DIR

FIG_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

OBJS = ['f1', 'f2', 'f3']


def load_method(method_name: str):
    safe = method_name.replace("-", "_")
    summary_path = os.path.join(RESULTS_DIR, f"{safe}_run", "summary.json")
    s = json.load(open(summary_path))
    instrument_log = s.get('instrument_log', [])
    iters    = np.array([e['n'] for e in instrument_log])
    sigma2   = np.array([e['sigma2_ref'] for e in instrument_log])  # (n_snap, 3)
    true_s2  = s['instrument_config']['_true_sigma2_ref']
    return {
        'method':  method_name,
        'iters':   iters,
        'sigma2':  sigma2,
        'true_s2': np.array(true_s2),
    }


def load_phase1_global_mean_var():
    het = json.load(open(os.path.join(RESULTS_DIR, 'hetero_test.json')))
    plans = het['results']
    return np.array([
        np.mean([p['var_f1'] for p in plans]),
        np.mean([p['var_f2'] for p in plans]),
        np.mean([p['var_f3'] for p in plans]),
    ])


def main():
    runs = {}
    for m in ['GPR-KG', 'GPR-KG-nV']:
        runs[m] = load_method(m)
    global_mean = load_phase1_global_mean_var()

    true_s2 = runs['GPR-KG']['true_s2']
    print(f"Reference plan ground-truth variance (10-rep Phase 1):")
    for j, name in enumerate(OBJS):
        print(f"  true sigma^2_{name} = {true_s2[j]:.5f}  (global mean: {global_mean[j]:.5f}, "
              f"ratio = {true_s2[j]/global_mean[j]:.2f}x)")
    print()

    # ── Plot 3-panel: one per objective ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = {'GPR-KG': 'tab:red', 'GPR-KG-nV': 'tab:blue'}
    for j, name in enumerate(OBJS):
        ax = axes[j]
        for m, r in runs.items():
            ax.plot(r['iters'], r['sigma2'][:, j],
                    color=colors[m], lw=1.8, marker='o', markersize=3.5,
                    label=m, alpha=0.85)
        ax.axhline(true_s2[j], color='black', linestyle='--', lw=1.4,
                   label=r'True $\sigma^2$ (Phase 1)')
        ax.axhline(global_mean[j], color='gray', linestyle=':', lw=1.2,
                   label=r'Global mean $\sigma^2$ (pooled target)')
        ax.set_title(rf'${name}$ at reference plan')
        ax.set_xlabel('Total observations $n$')
        if j == 0:
            ax.set_ylabel(r'Estimated $\sigma^2(x_{\mathrm{ref}})$')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
    plt.suptitle(r'VEPM vs.\ pooled variance estimation at reference plan $x_{\mathrm{ref}}$',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fp = os.path.join(FIG_DIR, f'fig_vepm_convergence_ingolstadt21.{ext}')
        fig.savefig(fp, dpi=180, bbox_inches='tight')
        print(f"  saved {fp}")
    plt.close(fig)

    # ── Numerical summary: final estimate vs true & vs global mean ──────
    summary = {
        'true_sigma2_ref':       true_s2.tolist(),
        'global_mean_sigma2':    global_mean.tolist(),
        'methods': {},
    }
    print()
    print(f"{'Method':12s} | "
          f"{'final sig^2 f1':14s} | "
          f"{'rel_err vs true f1':18s} | "
          f"{'final sig^2 f3':14s} | "
          f"{'rel_err vs true f3':18s}")
    print("-" * 95)
    for m, r in runs.items():
        final_s2 = r['sigma2'][-1]
        rel_err = np.abs(final_s2 - true_s2) / true_s2
        print(f"{m:12s} | "
              f"{final_s2[0]:.5f}        | "
              f"{rel_err[0]*100:6.1f}%             | "
              f"{final_s2[2]:.5f}        | "
              f"{rel_err[2]*100:6.1f}%")
        summary['methods'][m] = {
            'final_sigma2':     final_s2.tolist(),
            'rel_err_vs_true':  rel_err.tolist(),
            'rel_err_vs_global': (np.abs(final_s2 - global_mean) / global_mean).tolist(),
        }
    with open(os.path.join(RESULTS_DIR, 'vepm_convergence_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
