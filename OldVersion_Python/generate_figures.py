"""Generate figures for old paper Section 5.1 reproduction.

Figures:
  Fig 1: Iterative solutions projected on true Pareto front (per problem)
  Fig 2: IGD convergence over iterations (per problem, both methods)
  Fig 3: RMSE convergence for f1, f2, f3 (per problem)
  Fig 4: Number of infeasible solutions in non-dominated set
  Fig 5: VEPM variance convergence at a reference point
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
PROBLEMS = ['RZDT1', 'RZDT2', 'RZDT5']
METHODS = {'GPR_KG': 'GPR-KG (with VEPM)', 'GPR_KG_nV': 'GPR-KG-nV (no VEPM)'}
METHOD_COLORS = {'GPR_KG': 'blue', 'GPR_KG_nV': 'red'}
METHOD_STYLES = {'GPR_KG': '-', 'GPR_KG_nV': '--'}


def load_results():
    """Load all result JSON files."""
    results = {}
    for method_key in METHODS:
        for problem in PROBLEMS:
            pattern = os.path.join(RESULTS_DIR, f'{problem}_{method_key}_seed*.json')
            files = sorted(glob.glob(pattern))
            key = (problem, method_key)
            results[key] = []
            for f in files:
                try:
                    with open(f) as fh:
                        results[key].append(json.load(fh))
                except (json.JSONDecodeError, IOError):
                    pass
    return results


def get_true_pf(problem_name, d=5):
    """Get true Pareto front for a problem."""
    from core.test_problems import PROBLEMS as PROB_CLASSES
    prob = PROB_CLASSES[problem_name](d=d)
    return prob.true_pareto_front()


def fig1_iterative_solutions(results):
    """Fig 1: Iterative solutions projected on true PF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for col, problem in enumerate(PROBLEMS):
        ax = axes[col]
        true_pf = get_true_pf(problem)
        ax.plot(true_pf[:, 0], true_pf[:, 1], 'k-', linewidth=1.5,
                label='True PF', zorder=1)

        for method_key, method_label in METHODS.items():
            key = (problem, method_key)
            data_list = results.get(key, [])
            if not data_list:
                continue
            # Use first replicate for scatter
            d0 = data_list[0]
            iters = d0.get('iter_solutions', [])
            if not iters:
                continue
            feasible = [s for s in iters if s['feasible']]
            infeasible = [s for s in iters if not s['feasible']]
            if feasible:
                f_arr = np.array([s['f_true'][:2] for s in feasible])
                ax.scatter(f_arr[:, 0], f_arr[:, 1], s=10, alpha=0.5,
                          color=METHOD_COLORS[method_key],
                          label=f'{method_label} (feas.)', zorder=2)
            if infeasible:
                f_arr = np.array([s['f_true'][:2] for s in infeasible])
                ax.scatter(f_arr[:, 0], f_arr[:, 1], s=10, alpha=0.3,
                          marker='x', color=METHOD_COLORS[method_key],
                          label=f'{method_label} (infeas.)', zorder=2)

        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.set_title(problem)
        if col == 0:
            ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_iterative_solutions.pdf'), dpi=200)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_iterative_solutions.png'), dpi=200)
    plt.close()
    print("  Fig 1: Iterative solutions saved")


def _extract_convergence(data_list, history_key, val_idx=None):
    """Extract convergence data: returns (x_vals, mean_vals, se_vals)."""
    if not data_list:
        return None, None, None

    all_series = []
    for d in data_list:
        hist = d.get(history_key, [])
        if not hist:
            continue
        if val_idx is not None:
            series = [(h[0], h[val_idx]) for h in hist]
        else:
            series = [(h[0], h[1]) for h in hist]
        all_series.append(series)

    if not all_series:
        return None, None, None

    # Find common x-axis (use first replicate's x values)
    x_vals = np.array([s[0] for s in all_series[0]])
    n_points = len(x_vals)

    # Collect values at each x point
    vals_matrix = []
    for series in all_series:
        if len(series) == n_points:
            vals_matrix.append([s[1] for s in series])
    if not vals_matrix:
        return None, None, None

    vals_matrix = np.array(vals_matrix)
    mean_vals = np.mean(vals_matrix, axis=0)
    se_vals = np.std(vals_matrix, axis=0) / np.sqrt(len(vals_matrix))
    return x_vals, mean_vals, se_vals


def fig2_igd_convergence(results):
    """Fig 2: IGD convergence over iterations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for col, problem in enumerate(PROBLEMS):
        ax = axes[col]
        for method_key, method_label in METHODS.items():
            key = (problem, method_key)
            x, mean, se = _extract_convergence(results.get(key, []), 'igd_history')
            if x is None:
                continue
            color = METHOD_COLORS[method_key]
            ax.plot(x, mean, METHOD_STYLES[method_key], color=color,
                    label=method_label, linewidth=1.5)
            ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)

        ax.set_xlabel('Number of Evaluations')
        ax.set_ylabel('IGD')
        ax.set_title(problem)
        ax.set_yscale('log')
        if col == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_igd_convergence.pdf'), dpi=200)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_igd_convergence.png'), dpi=200)
    plt.close()
    print("  Fig 2: IGD convergence saved")


def fig3_rmse_convergence(results):
    """Fig 3: RMSE convergence for f1, f2, f3."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    obj_labels = ['$f_1$', '$f_2$', '$f_3$']

    for col, problem in enumerate(PROBLEMS):
        for row, (obj_name, val_idx) in enumerate(
            [('rmse_f1', 1), ('rmse_f2', 2), ('rmse_f3', 3)]
        ):
            ax = axes[row, col]
            for method_key, method_label in METHODS.items():
                key = (problem, method_key)
                x, mean, se = _extract_convergence(
                    results.get(key, []), 'rmse_history', val_idx=val_idx
                )
                if x is None:
                    continue
                color = METHOD_COLORS[method_key]
                ax.plot(x, mean, METHOD_STYLES[method_key], color=color,
                        label=method_label, linewidth=1.5)
                ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)

            if row == 0:
                ax.set_title(problem)
            ax.set_ylabel(f'RMSE({obj_labels[row]})')
            if row == 2:
                ax.set_xlabel('Number of Evaluations')
            if col == 0 and row == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_rmse_convergence.pdf'), dpi=200)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_rmse_convergence.png'), dpi=200)
    plt.close()
    print("  Fig 3: RMSE convergence saved")


def fig4_infeasible_count(results):
    """Fig 4: Number of infeasible solutions in non-dominated set."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for col, problem in enumerate(PROBLEMS):
        ax = axes[col]
        for method_key, method_label in METHODS.items():
            key = (problem, method_key)
            x, mean, se = _extract_convergence(
                results.get(key, []), 'infeasible_history'
            )
            if x is None:
                continue
            color = METHOD_COLORS[method_key]
            ax.plot(x, mean, METHOD_STYLES[method_key], color=color,
                    label=method_label, linewidth=1.5)
            ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)

        ax.set_xlabel('Number of Evaluations')
        ax.set_ylabel('# Infeasible in Non-dominated Set')
        ax.set_title(problem)
        if col == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_infeasible.pdf'), dpi=200)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_infeasible.png'), dpi=200)
    plt.close()
    print("  Fig 4: Infeasible count saved")


def fig5_vepm_variance(results):
    """Fig 5: VEPM variance convergence at reference point."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    obj_colors = ['blue', 'green', 'red']
    obj_labels = ['$f_1$', '$f_2$', '$f_3$']

    for col, problem in enumerate(PROBLEMS):
        ax = axes[col]
        for method_key, method_label in METHODS.items():
            key = (problem, method_key)
            data_list = results.get(key, [])
            if not data_list:
                continue

            # Average variance trajectories across reps
            all_var_series = []
            for d in data_list:
                vh = d.get('variance_history', [])
                if vh:
                    all_var_series.append(vh)

            if not all_var_series:
                continue

            # Use first rep's x values
            x_vals = np.array([v[0] for v in all_var_series[0]])
            n_pts = len(x_vals)
            for obj_i in range(3):
                vals = []
                for vs in all_var_series:
                    if len(vs) == n_pts:
                        vals.append([v[1][obj_i] for v in vs])
                if not vals:
                    continue
                vals = np.array(vals)
                mean_v = np.mean(vals, axis=0)
                linestyle = METHOD_STYLES[method_key]
                label = f'{method_label} {obj_labels[obj_i]}'
                ax.plot(x_vals, mean_v, linestyle, color=obj_colors[obj_i],
                        label=label, linewidth=1.0,
                        alpha=1.0 if method_key == 'GPR_KG' else 0.5)

        ax.set_xlabel('Number of Evaluations')
        ax.set_ylabel('Variance Estimate')
        ax.set_title(problem)
        ax.set_yscale('log')
        if col == 0:
            ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_vepm_variance.pdf'), dpi=200)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_vepm_variance.png'), dpi=200)
    plt.close()
    print("  Fig 5: VEPM variance convergence saved")


def fig_hv_convergence(results):
    """Bonus: HV convergence over iterations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for col, problem in enumerate(PROBLEMS):
        ax = axes[col]
        for method_key, method_label in METHODS.items():
            key = (problem, method_key)
            x, mean, se = _extract_convergence(results.get(key, []), 'hv_history')
            if x is None:
                continue
            color = METHOD_COLORS[method_key]
            ax.plot(x, mean, METHOD_STYLES[method_key], color=color,
                    label=method_label, linewidth=1.5)
            ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)

        ax.set_xlabel('Number of Evaluations')
        ax.set_ylabel('Hypervolume')
        ax.set_title(problem)
        if col == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_hv_convergence.pdf'), dpi=200)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_hv_convergence.png'), dpi=200)
    plt.close()
    print("  HV convergence saved")


def fig_final_pareto(results):
    """Bonus: Final estimated Pareto fronts vs true PF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for col, problem in enumerate(PROBLEMS):
        ax = axes[col]
        true_pf = get_true_pf(problem)
        ax.plot(true_pf[:, 0], true_pf[:, 1], 'k-', linewidth=2,
                label='True PF', zorder=1)

        for method_key, method_label in METHODS.items():
            key = (problem, method_key)
            data_list = results.get(key, [])
            if not data_list:
                continue
            # Overlay all replicates' final Pareto fronts
            for i, d in enumerate(data_list):
                pf = d.get('pareto_objectives_true', [])
                if pf:
                    pf = np.array(pf)
                    label = method_label if i == 0 else None
                    ax.scatter(pf[:, 0], pf[:, 1], s=15, alpha=0.4,
                              color=METHOD_COLORS[method_key],
                              label=label, zorder=2)

        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.set_title(problem)
        if col == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_final_pareto.pdf'), dpi=200)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_final_pareto.png'), dpi=200)
    plt.close()
    print("  Final Pareto fronts saved")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    results = load_results()
    total = sum(len(v) for v in results.values())
    print(f"Loaded {total} result files")
    if total == 0:
        print("No results. Run experiments first.")
        return

    print("Generating figures...")
    fig1_iterative_solutions(results)
    fig2_igd_convergence(results)
    fig3_rmse_convergence(results)
    fig4_infeasible_count(results)
    fig5_vepm_variance(results)
    fig_hv_convergence(results)
    fig_final_pareto(results)
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
