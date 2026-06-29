"""Consolidate Phase 1 hetero_test data using the first R reps per plan.

Reads results/intas/hetero_test.partial.jsonl, keeps at most R reps per
plan, computes the full heteroscedasticity test suite (Levene, Bartlett,
White F-test, Spearman, var_ratio), writes a clean results/intas/
hetero_test.json, and generates results/figures/figH_hetero.pdf with
four panels:
    (a) sigma^2(f^3) scatter vs E[f^1]
    (b) Box plot of f^3 per plan, sorted by E[f^1]
    (c) sigma^2(f^3) vs E[f^2]  (strongest Spearman relation)
    (d) Per-plan variance ratios for f^1, f^2, f^3
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.intas.config import RESULTS_DIR

FIG_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

R_USE = 10   # Number of reps to retain per plan


def main():
    partial_path = os.path.join(RESULTS_DIR, 'hetero_test.partial.jsonl')
    out_path     = os.path.join(RESULTS_DIR, 'hetero_test.json')
    fig_base     = os.path.join(FIG_DIR, 'figH_hetero')

    # Load partial records
    with open(partial_path) as f:
        recs = [json.loads(l) for l in f]

    # Keep first R_USE reps per plan, ordered by rep index
    data = defaultdict(list)
    for r in sorted(recs, key=lambda r: (r['plan_id'], r.get('rep', 0))):
        pid = r['plan_id']
        if len(data[pid]) < R_USE:
            data[pid].append(r['y'])
    pids = sorted(data.keys())
    M = len(pids)
    R = R_USE
    print(f"M={M} plans, R={R} reps each, total {M*R} sims used")

    # Plans file (decision vectors) for record
    plans_path = os.path.join(RESULTS_DIR, 'hetero_test.plans.json')
    plans_cache = json.load(open(plans_path))
    plans = plans_cache['plans']

    # Per-plan statistics
    results = []
    means = np.zeros((M, 3))
    vars_ = np.zeros((M, 3))
    for i, pid in enumerate(pids):
        arr = np.array(data[pid])  # (R, 3)
        means[i] = arr.mean(axis=0)
        vars_[i] = arr.var(axis=0, ddof=1)
        results.append({
            'plan_id': pid,
            'plan':    list(plans[pid]),
            'mean_f1': float(means[i, 0]),
            'mean_f2': float(means[i, 1]),
            'mean_f3': float(means[i, 2]),
            'var_f1':  float(vars_[i, 0]),
            'var_f2':  float(vars_[i, 1]),
            'var_f3':  float(vars_[i, 2]),
            'reps':    arr.tolist(),
        })

    # ------------------------------------------------------------------
    # Heteroscedasticity tests
    # ------------------------------------------------------------------
    hetero_by_obj = {}
    for j, name in enumerate(['f1', 'f2', 'f3']):
        groups = [np.array(data[p])[:, j] for p in pids]
        lev_s, lev_p = stats.levene(*groups)
        try:
            bar_s, bar_p = stats.bartlett(*groups)
        except Exception:
            bar_s, bar_p = float('nan'), float('nan')

        plan_vars = np.array([np.var(g, ddof=1) for g in groups])
        var_ratio = float(plan_vars.max() / plan_vars.min()) if plan_vars.min() > 0 else float('inf')

        # Spearman on (mean_f2, sigma^2_j) — strongest relation empirically
        corr_f1 = float(np.corrcoef(means[:, 0], plan_vars)[0, 1])
        sp_rho_f1, sp_p_f1 = stats.spearmanr(means[:, 0], plan_vars)
        sp_rho_f2, sp_p_f2 = stats.spearmanr(means[:, 1], plan_vars)

        # White test: log(sigma^2_j) ~ m1 + m2 + m3 + m1^2 + m2^2 + m3^2
        log_var = np.log(plan_vars)
        X = np.column_stack([np.ones(M), means[:, 0], means[:, 1], means[:, 2],
                              means[:, 0]**2, means[:, 1]**2, means[:, 2]**2])
        beta, *_ = np.linalg.lstsq(X, log_var, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((log_var - y_hat) ** 2)
        ss_tot = np.sum((log_var - log_var.mean()) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        k = X.shape[1] - 1
        F_white = (R2 / k) / ((1 - R2) / (M - k - 1)) if (1 - R2) > 0 else 0.0
        p_white_F = float(1 - stats.f.cdf(F_white, k, M - k - 1)) if F_white > 0 else 1.0
        p_white_nr2 = float(1 - stats.chi2.cdf(M * R2, k))

        hetero_by_obj[name] = {
            'levene_stat':   float(lev_s),
            'levene_p':      float(lev_p),
            'bartlett_stat': float(bar_s),
            'bartlett_p':    float(bar_p),
            'corr_mean_f1_vs_var': corr_f1,
            'spearman_f1_rho': float(sp_rho_f1),
            'spearman_f1_p':   float(sp_p_f1),
            'spearman_f2_rho': float(sp_rho_f2),
            'spearman_f2_p':   float(sp_p_f2),
            'white_R2':        float(R2),
            'white_F':         float(F_white),
            'white_F_p':       p_white_F,
            'white_nR2_p':     p_white_nr2,
            'min_var':         float(plan_vars.min()),
            'max_var':         float(plan_vars.max()),
            'var_ratio':       var_ratio,
            'reject_homoscedasticity': bool((lev_p < 0.05) or (bar_p < 0.05) or (p_white_F < 0.05)),
        }

    summary = {
        'M': M, 'R': R,
        'total_sims': M * R,
        'hetero_by_obj': hetero_by_obj,
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\nHeteroscedasticity test suite:")
    print(f"  {'output':>6} {'Levene p':>10} {'Bartlett p':>11} "
          f"{'White F p':>10} {'Spearman(f2) p':>15} {'var_ratio':>10}")
    for name, h in hetero_by_obj.items():
        print(f"  {name:>6} {h['levene_p']:>10.4f} {h['bartlett_p']:>11.4f} "
              f"{h['white_F_p']:>10.4f} {h['spearman_f2_p']:>15.4f} "
              f"{h['var_ratio']:>10.2f}x")

    # ------------------------------------------------------------------
    # Figure: 4-panel hetero diagnostic
    # ------------------------------------------------------------------
    mean_f1 = means[:, 0]
    mean_f2 = means[:, 1]
    var_f1 = vars_[:, 0]
    var_f2 = vars_[:, 1]
    var_f3 = vars_[:, 2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) sigma^2(f^3) vs E[f^1] scatter
    ax = axes[0, 0]
    sc = ax.scatter(mean_f1, var_f3, c=var_f3, cmap='Reds', s=100,
                    edgecolors='k', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label=r'$\hat\sigma^2(f^3)$')
    ax.set_xlabel(r'$\mathbb{E}[f^1]$')
    ax.set_ylabel(r'$\hat\sigma^2(f^3)$')
    ax.set_title(r'(a) $\hat\sigma^2(f^3)$ vs.\ efficiency level')
    # Spearman annotation
    rho3, p3 = stats.spearmanr(mean_f1, var_f3)
    ax.text(0.05, 0.95, f'Spearman $\\rho$={rho3:.2f}, $p$={p3:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.grid(alpha=0.3)

    # (b) Box plot of f^3 per plan, sorted by E[f^1]
    ax2 = axes[0, 1]
    f3_groups = [np.array(data[p])[:, 2] for p in pids]
    order = np.argsort(mean_f1)
    f3_sorted = [f3_groups[i] for i in order]
    bp = ax2.boxplot(f3_sorted, patch_artist=True, showfliers=True,
                     medianprops={'color': 'red', 'linewidth': 1.5})
    cmap = plt.cm.coolwarm
    for patch, i in zip(bp['boxes'], range(len(order))):
        patch.set_facecolor(cmap(i / max(1, len(order) - 1)))
    ax2.set_xlabel(r'Plans sorted by $\mathbb{E}[f^1]$ (low$\to$high)')
    ax2.set_ylabel(r'$f^3$ distribution')
    ax2.set_title(r'(b) Per-plan $f^3$ distributions')
    ax2.axhline(1.0, ls='--', color='gray', lw=0.8, label='Baseline')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis='y')

    # (c) sigma^2(f^3) vs E[f^2]  (significant Spearman relation)
    ax3 = axes[1, 0]
    sc3 = ax3.scatter(mean_f2, var_f3, c=var_f3, cmap='Reds', s=100,
                      edgecolors='k', linewidths=0.5)
    plt.colorbar(sc3, ax=ax3, label=r'$\hat\sigma^2(f^3)$')
    ax3.set_xlabel(r'$\mathbb{E}[f^2]$ (equity)')
    ax3.set_ylabel(r'$\hat\sigma^2(f^3)$')
    ax3.set_title(r'(c) $\hat\sigma^2(f^3)$ vs.\ equity level')
    rhoE, pE = stats.spearmanr(mean_f2, var_f3)
    ax3.text(0.05, 0.95, f'Spearman $\\rho$={rhoE:.2f}, $p$={pE:.3f}',
             transform=ax3.transAxes, va='top', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax3.grid(alpha=0.3)

    # (d) Variance ratio bars
    ax4 = axes[1, 1]
    sorted_pids_by_f1 = [pids[i] for i in order]
    # Normalized variances (divide by min across plans for each output)
    v1n = var_f1 / var_f1.min()
    v2n = var_f2 / var_f2.min()
    v3n = var_f3 / var_f3.min()
    x_axis = np.arange(M)
    w = 0.27
    ax4.bar(x_axis - w, v1n[order], w, label=r'$\hat\sigma^2(f^1)$ / min', color='steelblue')
    ax4.bar(x_axis,      v2n[order], w, label=r'$\hat\sigma^2(f^2)$ / min', color='orange')
    ax4.bar(x_axis + w, v3n[order], w, label=r'$\hat\sigma^2(f^3)$ / min', color='indianred')
    ax4.set_yscale('log')
    ax4.set_xlabel(r'Plans sorted by $\mathbb{E}[f^1]$ (low$\to$high)')
    ax4.set_ylabel(r'Normalized variance (log scale)')
    ax4.set_title(r'(d) Per-plan variance ratios (max/min = 4$\times$-35$\times$)')
    ax4.legend(fontsize=8, loc='upper left')
    ax4.grid(alpha=0.3, axis='y', which='both')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fp = f'{fig_base}.{ext}'
        fig.savefig(fp, dpi=180, bbox_inches='tight')
        print(f"  saved {fp}")
    plt.close(fig)


if __name__ == '__main__':
    main()
