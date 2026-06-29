"""
Analysis script for d=5 heteroscedastic benchmark experiments.

Reads from results/d5_hetero/ and generates:
  1. HV convergence plots (one subplot per problem, 7 methods)
  2. Pareto front comparison plots (one subplot per problem)
  3. Console results table (HV / IGD / CVR mean ± SE)
  4. LaTeX table rows ready for the paper

Usage:
    cd GPR_KG_Code
    python -m analysis.analyse_d5_hetero
    python -m analysis.analyse_d5_hetero --problems RZDT1 RZDT2  # subset
"""

import sys
import os
import json
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

RESULTS_ROOT = os.path.join(BASE_DIR, "results", "d5_hetero")
FIGURES_DIR  = os.path.join(BASE_DIR, "figures", "d5_hetero")

# ── Style ────────────────────────────────────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 150

METHOD_ORDER = ['GPR-KG', 'GPR-KG-nV', 'cEHVI', 'cParEGO',
                'NSGA-II-K', 'NSGA-II-D', 'RS']

METHOD_COLORS = {
    'GPR-KG':    '#1f77b4',   # blue
    'GPR-KG-nV': '#ff7f0e',   # orange
    'cEHVI':     '#2ca02c',   # green
    'cParEGO':   '#d62728',   # red
    'NSGA-II-K': '#9467bd',   # purple
    'NSGA-II-D': '#8c564b',   # brown
    'RS':        '#7f7f7f',   # gray
}

METHOD_LINESTYLES = {
    'GPR-KG':    '-',
    'GPR-KG-nV': '--',
    'cEHVI':     '-.',
    'cParEGO':   ':',
    'NSGA-II-K': '-',
    'NSGA-II-D': '--',
    'RS':        ':',
}

METHOD_MARKERS = {
    'GPR-KG':    'o',
    'GPR-KG-nV': 's',
    'cEHVI':     '^',
    'cParEGO':   'v',
    'NSGA-II-K': 'D',
    'NSGA-II-D': 'P',
    'RS':        'x',
}

PROBLEM_TITLES = {
    'RZDT1': 'RZDT1 (convex, sqrt-noise)',
    'RZDT2': 'RZDT2 (concave, bell-noise)',
    'RZDT5': 'RZDT5 (hyperbolic, quad-noise)',
    'RZDT3': 'RZDT3 (discontinuous)',
    'RZDT4': 'RZDT4 (multi-modal)',
    'RZDT6': 'RZDT6 (degenerate)',
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _safe(name):
    return name.replace('-', '_')


def load_reps(problem, method, max_reps=10):
    """Load all available rep JSON files for (problem, method)."""
    records = []
    prob_dir = os.path.join(RESULTS_ROOT, problem)
    for rep in range(1, max_reps + 1):
        path = os.path.join(prob_dir, f"{_safe(method)}_rep{rep:02d}.json")
        if os.path.exists(path):
            try:
                records.append(json.load(open(path)))
            except json.JSONDecodeError:
                pass
    return records


def load_meta(problem):
    """Load problem metadata (tau, true PF, curve)."""
    path = os.path.join(RESULTS_ROOT, "problems", f"{problem}_meta.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None


def aggregate_metric(records, key):
    vals = [r[key] for r in records if key in r and r[key] is not None]
    if not vals:
        return None, None
    arr = np.array(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def build_hv_curves(records):
    """Align HV histories and return (stages, mean_hv, se_hv)."""
    histories = [r.get('hv_history', []) for r in records]
    histories = [h for h in histories if h]
    if not histories:
        return None, None, None

    all_stages = sorted(set(pair[0] for h in histories for pair in h))
    mat = np.full((len(histories), len(all_stages)), np.nan)
    for i, hist in enumerate(histories):
        s2hv = {pair[0]: pair[1] for pair in hist}
        for j, s in enumerate(all_stages):
            if s in s2hv:
                mat[i, j] = s2hv[s]

    # Forward-fill NaN within each row
    for i in range(mat.shape[0]):
        last = np.nan
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                last = mat[i, j]
            elif not np.isnan(last):
                mat[i, j] = last

    valid = ~np.all(np.isnan(mat), axis=0)
    stages = np.array(all_stages)[valid]
    mat = mat[:, valid]
    mean_hv = np.nanmean(mat, axis=0)
    se_hv = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return stages, mean_hv, se_hv


# ── Figure 1: HV convergence ─────────────────────────────────────────────────

def plot_hv_convergence(problems, n_reps=10, save_dir=None):
    """One subplot per problem showing HV convergence curves for all methods."""
    n_probs = len(problems)
    fig, axes = plt.subplots(1, n_probs, figsize=(5.5 * n_probs, 5), sharey=False)
    if n_probs == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems):
        meta = load_meta(problem)
        hv_upper = meta['hv_true_pf'] if meta else None

        for method in METHOD_ORDER:
            records = load_reps(problem, method, max_reps=n_reps)
            if not records:
                continue
            stages, mean_hv, se_hv = build_hv_curves(records)
            if stages is None:
                continue

            color = METHOD_COLORS.get(method, 'black')
            ls    = METHOD_LINESTYLES.get(method, '-')
            lw    = 2.2 if method == 'GPR-KG' else 1.6
            zord  = 10 if method == 'GPR-KG' else 5

            ax.plot(stages, mean_hv, color=color, linestyle=ls,
                    linewidth=lw, label=method, zorder=zord)
            ax.fill_between(stages,
                            mean_hv - se_hv,
                            mean_hv + se_hv,
                            color=color, alpha=0.12, zorder=zord - 1)

        # Reference line: HV of true feasible Pareto front
        if hv_upper and hv_upper > 0:
            ax.axhline(hv_upper, color='black', linestyle=':', linewidth=1,
                       alpha=0.5, label='True PF HV')

        ax.set_xlabel('Simulation Evaluations')
        ax.set_ylabel('Hypervolume')
        ax.set_title(PROBLEM_TITLES.get(problem, problem))
        ax.grid(True, alpha=0.25)

    # Legend on last panel
    axes[-1].legend(loc='lower right', framealpha=0.9,
                    fontsize=8, ncol=1)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        tag = '_'.join(problems)
        for ext in ('pdf', 'png'):
            p = os.path.join(save_dir, f'fig_hv_convergence_{tag}.{ext}')
            plt.savefig(p, dpi=200, bbox_inches='tight')
            print(f'  Saved: {p}')
    plt.close()


# ── Figure 2: Pareto fronts ───────────────────────────────────────────────────

def plot_pareto_fronts(problems, n_reps=10, save_dir=None):
    """One subplot per problem showing best-rep Pareto fronts vs true PF."""
    n_probs = len(problems)
    fig, axes = plt.subplots(1, n_probs, figsize=(5.5 * n_probs, 5), sharey=False)
    if n_probs == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems):
        meta = load_meta(problem)

        # True Pareto curve
        if meta:
            f1c = np.array(meta['pf_curve_f1'])
            f2c = np.array(meta['pf_curve_f2'])
            # Filter to feasible region
            feas_pts = [(p['f1'], p['f2']) for p in meta['all_pf_grid'] if p['feasible']]
            ax.plot(f1c, f2c, 'k-', linewidth=1.5, alpha=0.4, label='True PF curve',
                    zorder=0)
            if feas_pts:
                fp = np.array(feas_pts)
                ax.scatter(fp[:, 0], fp[:, 1], c='black', s=30, zorder=1,
                           marker='*', label='Feasible true PF')

        for method in METHOD_ORDER:
            records = load_reps(problem, method, max_reps=n_reps)
            if not records:
                continue
            # Use median-HV rep
            hvs = [r.get('hv_final', 0) for r in records]
            best_idx = int(np.argsort(hvs)[len(hvs) // 2])  # median
            rec = records[best_idx]
            pf_pts = rec.get('pareto_objectives_true', [])
            if not pf_pts:
                pf_pts = rec.get('pareto_objectives', [])
            if not pf_pts:
                continue
            pf = np.array(pf_pts)
            color = METHOD_COLORS.get(method, 'black')
            marker = METHOD_MARKERS.get(method, 'o')
            ax.scatter(pf[:, 0], pf[:, 1], c=color, marker=marker, s=25,
                       label=f'{method}', alpha=0.75, linewidths=0.3,
                       edgecolors='none', zorder=5)

        ax.set_xlabel('$f^1$')
        ax.set_ylabel('$f^2$')
        ax.set_title(PROBLEM_TITLES.get(problem, problem))
        ax.grid(True, alpha=0.25)

    axes[-1].legend(loc='best', fontsize=7, framealpha=0.9)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        tag = '_'.join(problems)
        for ext in ('pdf', 'png'):
            p = os.path.join(save_dir, f'fig_pareto_{tag}.{ext}')
            plt.savefig(p, dpi=200, bbox_inches='tight')
            print(f'  Saved: {p}')
    plt.close()


# ── Console + LaTeX table ─────────────────────────────────────────────────────

def print_results_table(problems, n_reps=10):
    """Print console + LaTeX results table."""
    metrics = ['hv_final', 'igd_final', 'cvr_final']
    metric_labels = ['HV', 'IGD', 'CVR']

    print('\n' + '=' * 100)
    print(f"{'Method':<14}", end='')
    for p in problems:
        for m in metric_labels:
            print(f'  {p}/{m:<8}', end='')
    print()
    print('-' * 100)

    latex_rows = []

    for method in METHOD_ORDER:
        row = f'{method:<14}'
        latex_parts = [method.replace('-', '\\textendash{}')]

        for problem in problems:
            records = load_reps(problem, method, max_reps=n_reps)
            for mkey in metrics:
                mean, se = aggregate_metric(records, mkey)
                if mean is None:
                    row += f'  {"--":>9}'
                    latex_parts.append('--')
                else:
                    row += f'  {mean:.3f}±{se:.3f}'
                    latex_parts.append(f'{mean:.3f} ({se:.3f})')

        print(row)
        latex_rows.append(latex_parts)

    print('=' * 100)

    # LaTeX table
    print('\n── LaTeX table rows ──')
    print(r'\hline')
    for parts in latex_rows:
        print(' & '.join(parts) + r' \\')
    print(r'\hline')


def print_summary(problems, n_reps=10):
    """Print rank summary: which method wins on each metric/problem."""
    print('\n── Method ranking by HV (higher is better) ──')
    for problem in problems:
        hvs = {}
        for method in METHOD_ORDER:
            records = load_reps(problem, method, max_reps=n_reps)
            m, _ = aggregate_metric(records, 'hv_final')
            if m is not None:
                hvs[method] = m
        ranked = sorted(hvs.items(), key=lambda x: -x[1])
        print(f'\n  {problem}:')
        for rank, (m, v) in enumerate(ranked, 1):
            marker = ' *' if m == 'GPR-KG' else ''
            print(f'    {rank}. {m:<14} HV={v:.4f}{marker}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problems', nargs='+',
                        default=['RZDT1', 'RZDT2', 'RZDT5'],
                        help='Which problems to analyse')
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()

    problems = args.problems
    n_reps   = args.n_reps

    # Check available data
    print('\n── Available results ──')
    for p in problems:
        prob_dir = os.path.join(RESULTS_ROOT, p)
        if not os.path.isdir(prob_dir):
            print(f'  {p}: NO DATA')
            continue
        counts = {}
        for m in METHOD_ORDER:
            n = sum(1 for r in range(1, n_reps + 1)
                    if os.path.exists(os.path.join(prob_dir,
                                                   f'{_safe(m)}_rep{r:02d}.json')))
            counts[m] = n
        print(f'  {p}: ' + ', '.join(f'{m}={counts[m]}' for m in METHOD_ORDER))

    # Table
    print_results_table(problems, n_reps)
    print_summary(problems, n_reps)

    if not args.no_figures:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        print(f'\n── Generating figures → {FIGURES_DIR} ──')
        print('  HV convergence...')
        plot_hv_convergence(problems, n_reps, save_dir=FIGURES_DIR)
        print('  Pareto fronts...')
        plot_pareto_fronts(problems, n_reps, save_dir=FIGURES_DIR)


if __name__ == '__main__':
    main()
