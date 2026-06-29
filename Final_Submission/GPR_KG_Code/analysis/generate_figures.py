"""
Generate publication-quality figures for Section 6 of the paper.

Figure 1: HV convergence curves on RZDT1 (7 methods, mean ± 1SE)
Figure 2: Computation time vs dimension (log scale)
Figure 3: VEPM variance convergence
"""

import sys
import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from experiments.config import METHOD_NAMES, REF_POINT

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 15
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150
rcParams['text.usetex'] = False  # Set True if LaTeX available

# Color scheme for 7 methods
METHOD_COLORS = {
    'GPR-KG':    '#1f77b4',   # blue
    'GPR-KG-nV': '#ff7f0e',   # orange
    'cEHVI':     '#2ca02c',   # green
    'cParEGO':   '#d62728',   # red
    'NSGA-II-K': '#9467bd',   # purple
    'NSGA-II-D': '#8c564b',   # brown
    'RS':        '#7f7f7f',   # gray
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

METHOD_LINESTYLES = {
    'GPR-KG':    '-',
    'GPR-KG-nV': '--',
    'cEHVI':     '-.',
    'cParEGO':   ':',
    'NSGA-II-K': '-',
    'NSGA-II-D': '--',
    'RS':        ':',
}


def load_hv_histories(section_dir, method_name, problem_name, n_reps):
    """Load HV history from individual rep files."""
    histories = []
    safe_name = method_name.replace('-', '_')
    for rep in range(n_reps):
        fname = f"{problem_name}_{safe_name}_rep{rep+1}.json"
        path = os.path.join(section_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        if 'hv_history' in data:
            histories.append(data['hv_history'])
    return histories


def generate_figure1(save_dir, n_reps=30):
    """Figure 1: HV convergence curves on RZDT1.

    7 methods, solid lines = mean over n_reps, shaded = ±1 SE.
    """
    section_dir = os.path.join(BASE_DIR, "results", "sec62")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    nsga2d_final_hv = None

    for method in METHOD_NAMES:
        histories = load_hv_histories(section_dir, method, 'RZDT1', n_reps)
        if len(histories) == 0:
            continue

        # Align all histories to common x-axis
        all_stages = sorted(set(s for h in histories for s, _ in h))
        if len(all_stages) == 0:
            continue

        hv_matrix = np.full((len(histories), len(all_stages)), np.nan)
        for i, hist in enumerate(histories):
            stage_to_hv = dict(hist)
            for j, s in enumerate(all_stages):
                if s in stage_to_hv:
                    hv_matrix[i, j] = stage_to_hv[s]

        # Fill forward NaN values
        for i in range(hv_matrix.shape[0]):
            for j in range(1, hv_matrix.shape[1]):
                if np.isnan(hv_matrix[i, j]):
                    hv_matrix[i, j] = hv_matrix[i, j-1]

        valid = ~np.isnan(hv_matrix).any(axis=0)
        stages = np.array(all_stages)[valid]
        hv_vals = hv_matrix[:, valid]

        mean_hv = np.mean(hv_vals, axis=0)
        se_hv = np.std(hv_vals, axis=0) / np.sqrt(len(histories))

        color = METHOD_COLORS.get(method, 'black')
        ls = METHOD_LINESTYLES.get(method, '-')

        ax.plot(stages, mean_hv, color=color, linestyle=ls, linewidth=2,
                label=method)
        ax.fill_between(stages, mean_hv - se_hv, mean_hv + se_hv,
                         color=color, alpha=0.15)

        if method == 'NSGA-II-D' and len(mean_hv) > 0:
            nsga2d_final_hv = mean_hv[-1]

    # Add NSGA-II-D reference line
    if nsga2d_final_hv is not None:
        ax.axhline(y=nsga2d_final_hv, color=METHOD_COLORS['NSGA-II-D'],
                   linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Number of Simulation Evaluations')
    ax.set_ylabel('Dominated Hypervolume')
    ax.set_title('HV Convergence on RZDT1 ($d=5$, $\\sigma=0.1$)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'fig1_convergence.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path_png = os.path.join(save_dir, 'fig1_convergence.png')
    plt.savefig(save_path_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    print(f"  Saved: {save_path_png}")


def generate_figure2(save_dir, n_reps=30):
    """Figure 2: Computation time per iteration vs dimension (log scale)."""
    import glob
    section_dir = os.path.join(BASE_DIR, "results", "sec63")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    dims = [5, 10, 20]

    for method in METHOD_NAMES:
        safe_name = method.replace('-', '_')
        times = []
        valid_dims = []
        for d in dims:
            prob_name = f"RZDT1_d{d}"
            files = sorted(glob.glob(os.path.join(section_dir, f'{prob_name}_{safe_name}_rep*.json')))
            if not files:
                continue
            t_vals = []
            for f in files:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                t_vals.append(data.get('time_per_iter_mean', 0))
            if t_vals:
                times.append(np.mean(t_vals))
                valid_dims.append(d)

        if len(times) > 0:
            # Replace zero times with small value for log scale
            times = [max(t, 0.001) for t in times]
            color = METHOD_COLORS.get(method, 'black')
            marker = METHOD_MARKERS.get(method, 'o')
            ls = METHOD_LINESTYLES.get(method, '-')
            ax.plot(valid_dims, times, color=color, marker=marker,
                    linestyle=ls, linewidth=2, markersize=8, label=method)

    ax.set_xlabel('Decision-space Dimension $d$')
    ax.set_ylabel('Computation Time per Iteration (s)')
    ax.set_title('Computational Scalability')
    ax.set_yscale('log')
    ax.set_xticks(dims)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'fig2_scalability.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path_png = os.path.join(save_dir, 'fig2_scalability.png')
    plt.savefig(save_path_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_figure3(save_dir):
    """Figure 3: VEPM variance convergence at unsampled solution.

    Compares variance estimate with/without VEPM vs true variance.
    """
    # This requires special data from ablation runs
    # For now, generate from a dedicated run
    print("  Figure 3 requires special VEPM tracking data (to be generated).")


def generate_pareto_plots(save_dir, n_reps=30):
    """Bonus: Pareto front plots for all 3 problems (best rep per method)."""
    section_dir = os.path.join(BASE_DIR, "results", "sec62")

    for prob_name in ['RZDT1', 'RZDT2', 'RZDT5']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for method in METHOD_NAMES:
            safe_name = method.replace('-', '_')
            best_hv = -1
            best_pf = None

            for rep in range(n_reps):
                fname = f"{prob_name}_{safe_name}_rep{rep+1}.json"
                path = os.path.join(section_dir, fname)
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    continue
                if data['hv_final'] > best_hv:
                    best_hv = data['hv_final']
                    best_pf = np.array(data['pareto_objectives_true'])

            if best_pf is not None and len(best_pf) > 0:
                color = METHOD_COLORS.get(method, 'black')
                marker = METHOD_MARKERS.get(method, 'o')
                ax.scatter(best_pf[:, 0], best_pf[:, 1], c=color,
                          marker=marker, s=40, label=f'{method} (HV={best_hv:.3f})',
                          alpha=0.8, edgecolors='gray', linewidths=0.3)

        ax.set_xlabel('$f^1(x)$')
        ax.set_ylabel('$f^2(x)$')
        ax.set_title(f'{prob_name}: Best Pareto Front by Method')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'pareto_{prob_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


def generate_all_figures(n_reps=30):
    """Generate all figures."""
    save_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(save_dir, exist_ok=True)

    print("\nGenerating Figure 1 (HV convergence)...")
    generate_figure1(save_dir, n_reps)

    print("\nGenerating Figure 2 (Scalability)...")
    generate_figure2(save_dir, n_reps)

    print("\nGenerating Figure 3 (VEPM effect)...")
    generate_figure3(save_dir)

    print("\nGenerating Pareto front plots...")
    generate_pareto_plots(save_dir, n_reps)


if __name__ == '__main__':
    generate_all_figures()
