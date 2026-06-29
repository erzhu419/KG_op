"""Quick re-run to regenerate convergence curves with all 5 lines visible."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import time

from gpr_kg import (RZDT1, RZDT2, RZDT5, GPRKR_Algorithm,
                     pareto_filter, compute_hypervolume_2d)

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 15
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_and_collect(problem_class, problem_name, d=5, L=20, sigma=0.1,
                    N=300, n0=30, n_reps=5, seed_base=42):
    """Run GPR-KG and collect HV histories + Pareto fronts."""
    print(f"\n{'='*60}")
    print(f"  {problem_name}: d={d}, L={L}, sigma={sigma}, N={N}, reps={n_reps}")
    print(f"{'='*60}")

    all_hv_histories = []
    all_pareto_fronts = []
    all_hv_final = []
    all_times = []

    for rep in range(n_reps):
        print(f"  Rep {rep+1}/{n_reps}...", end=" ", flush=True)
        t0 = time.time()

        prob = problem_class(d=d, L=L, sigma=sigma, alpha=0.05)
        prob.calibrate_constraint(feasibility_ratio=0.5)

        alg = GPRKR_Algorithm(
            problem=prob, N=N, n0=n0,
            K1=20, K2=2,
            lambda_i=0.1, prior_var=100.0, w_vepm=5.0,
            n_thr=20, seed=seed_base + rep
        )

        pareto_set = alg.run(verbose=False)
        elapsed = time.time() - t0

        true_pf = alg.get_true_objectives_of_estimate(pareto_set)
        ref_point = np.array([1.5, 1.5])
        hv = compute_hypervolume_2d(true_pf, ref_point) if len(true_pf) > 0 else 0.0

        all_hv_histories.append(alg.hv_history)
        all_pareto_fronts.append(true_pf)
        all_hv_final.append(hv)
        all_times.append(elapsed)
        print(f"HV={hv:.4f}, |PF|={len(true_pf)}, {elapsed:.0f}s")

    return {
        'problem_name': problem_name,
        'hv_histories': all_hv_histories,
        'pareto_fronts': all_pareto_fronts,
        'hv_final': all_hv_final,
        'times': all_times,
        'N': N, 'n0': n0, 'd': d, 'L': L, 'sigma': sigma
    }


def get_true_pareto_front(problem_class, d=5, L=20, sigma=0.1):
    prob = problem_class(d=d, L=L, sigma=sigma, alpha=0.05)
    prob.calibrate_constraint(feasibility_ratio=0.5)
    return prob.true_pareto_front(), prob


def plot_convergence(all_results, save_path):
    """Plot HV convergence with all 5 replications clearly visible."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, results in enumerate(all_results):
        ax = axes[idx]
        problem_name = results['problem_name']

        # Plot each replication as a distinct line
        for rep, hv_hist in enumerate(results['hv_histories']):
            if len(hv_hist) > 0:
                stages, hvs = zip(*hv_hist)
                ax.plot(stages, hvs, color=rep_colors[rep % len(rep_colors)],
                        alpha=0.7, linewidth=1.2, label=f'Rep {rep+1}')

        # Mean curve
        if len(results['hv_histories']) > 0:
            all_stages = results['hv_histories'][0]
            if len(all_stages) > 0:
                stages_ref = [s for s, _ in all_stages]
                hv_matrix = []
                for hv_hist in results['hv_histories']:
                    if len(hv_hist) == len(stages_ref):
                        hv_matrix.append([h for _, h in hv_hist])
                if len(hv_matrix) > 1:
                    hv_matrix = np.array(hv_matrix)
                    mean_hv = np.mean(hv_matrix, axis=0)
                    ax.plot(stages_ref, mean_hv, 'k-', linewidth=2.5,
                            label='Mean', zorder=10)

        ax.set_xlabel('Simulation Budget')
        ax.set_ylabel('Hypervolume Indicator')
        ax.set_title(f'{problem_name}')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('HV Convergence Curves ($d=5$, $N=300$, $\\sigma=0.1$)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_individual_pareto(results, problem_class, save_path):
    """Individual Pareto front plot with all 5 reps."""
    problem_name = results['problem_name']
    d, L, sigma = results['d'], results['L'], results['sigma']
    true_pf, prob = get_true_pareto_front(problem_class, d, L, sigma)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if hasattr(prob, 'true_pareto_curve'):
        f1_c, f2_c = prob.true_pareto_curve()
        ax.plot(f1_c, f2_c, 'k--', linewidth=1.0, alpha=0.3,
                label='Continuous ZDT curve')

    if len(true_pf) > 0:
        ax.scatter(true_pf[:, 0], true_pf[:, 1], c='blue', s=80, marker='*',
                   zorder=5, edgecolors='navy', linewidths=0.5,
                   label=f'True Pareto front ({len(true_pf)} pts)')

    rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    rep_markers = ['o', 's', '^', 'v', 'P']
    for rep, pf in enumerate(results['pareto_fronts']):
        if len(pf) > 0:
            ax.scatter(pf[:, 0], pf[:, 1],
                       c=[rep_colors[rep]], s=40, marker=rep_markers[rep],
                       alpha=0.7, edgecolors='gray', linewidths=0.3,
                       label=f'GPR-KG Rep {rep+1} (HV={results["hv_final"][rep]:.4f})')

    best_rep = np.argmax(results['hv_final'])
    best_pf = results['pareto_fronts'][best_rep]
    if len(best_pf) > 0:
        idx_sort = np.argsort(best_pf[:, 0])
        ax.plot(best_pf[idx_sort, 0], best_pf[idx_sort, 1],
                color=rep_colors[best_rep], linestyle='-', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('$f^1(x)$')
    ax.set_ylabel('$f^2(x)$')
    ax.set_title(f'{problem_name}: True vs GPR-KG Pareto Front\n'
                 f'($d={d}$, $L={L}$, $\\sigma={sigma}$, $N={results["N"]}$)')
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined_pareto(all_results, problem_classes, save_path):
    """Combined 1x3 Pareto front figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (results, prob_cls) in enumerate(zip(all_results, problem_classes)):
        ax = axes[idx]
        problem_name = results['problem_name']
        d, L, sigma = results['d'], results['L'], results['sigma']
        true_pf, prob = get_true_pareto_front(prob_cls, d, L, sigma)

        if hasattr(prob, 'true_pareto_curve'):
            f1_c, f2_c = prob.true_pareto_curve()
            ax.plot(f1_c, f2_c, 'b--', linewidth=1.0, alpha=0.3,
                    label='Continuous ZDT curve')

        if len(true_pf) > 0:
            ax.scatter(true_pf[:, 0], true_pf[:, 1], c='blue', s=80, marker='*',
                       zorder=5, edgecolors='navy', linewidths=0.5,
                       label=f'True PF ({len(true_pf)} pts)')

        best_rep = np.argmax(results['hv_final'])
        best_pf = results['pareto_fronts'][best_rep]
        if len(best_pf) > 0:
            s_idx = np.argsort(best_pf[:, 0])
            ax.plot(best_pf[s_idx, 0], best_pf[s_idx, 1],
                    'r-', linewidth=1.2, alpha=0.5)
            ax.scatter(best_pf[:, 0], best_pf[:, 1], c='red', s=50, marker='D',
                       zorder=6, edgecolors='darkred', linewidths=0.5,
                       label=f'GPR-KG Best (HV={results["hv_final"][best_rep]:.4f})')

        for rep, pf in enumerate(results['pareto_fronts']):
            if rep != best_rep and len(pf) > 0:
                ax.scatter(pf[:, 0], pf[:, 1],
                           c=[rep_colors[rep]], s=25, marker='o',
                           alpha=0.5, edgecolors='gray', linewidths=0.2,
                           label=f'Rep {rep+1}')

        ax.set_xlabel('$f^1(x)$')
        ax.set_ylabel('$f^2(x)$')
        ax.set_title(f'{problem_name}')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'GPR-KG Pareto Front Approximation ($d={d}$, $N={results["N"]}$, '
                 f'$\\sigma={sigma}$)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def save_summary(all_results, save_path):
    import json
    summary = {}
    for results in all_results:
        name = results['problem_name']
        hv = results['hv_final']
        summary[name] = {
            'HV_mean': float(np.mean(hv)),
            'HV_std': float(np.std(hv)),
            'HV_best': float(np.max(hv)),
            'HV_worst': float(np.min(hv)),
            'PF_sizes': [len(pf) for pf in results['pareto_fronts']],
            'Time_mean_sec': float(np.mean(results['times'])),
            'N': results['N'], 'd': results['d'],
            'L': results['L'], 'sigma': results['sigma'],
        }
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Problem':<10} {'HV Mean':>9} {'HV Std':>9} {'HV Best':>9} "
          f"{'|PF| Avg':>9} {'Time(s)':>9}")
    print(f"{'-'*70}")
    for name, s in summary.items():
        pf_mean = np.mean(s['PF_sizes'])
        print(f"{name:<10} {s['HV_mean']:>9.4f} {s['HV_std']:>9.4f} "
              f"{s['HV_best']:>9.4f} {pf_mean:>9.1f} {s['Time_mean_sec']:>9.1f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    print("GPR-KG: Full experiment with corrected plots")

    d, L, sigma = 5, 20, 0.1
    N, n0, n_reps = 300, 30, 5

    configs = [(RZDT1, 'RZDT1'), (RZDT2, 'RZDT2'), (RZDT5, 'RZDT5')]
    all_results = []
    prob_classes = []

    for cls, name in configs:
        res = run_and_collect(cls, name, d=d, L=L, sigma=sigma,
                              N=N, n0=n0, n_reps=n_reps, seed_base=42)
        all_results.append(res)
        prob_classes.append(cls)
        plot_individual_pareto(res, cls,
                               os.path.join(FIGURES_DIR, f'pareto_front_{name}.png'))

    plot_combined_pareto(all_results, prob_classes,
                         os.path.join(FIGURES_DIR, 'pareto_fronts_combined.png'))
    plot_convergence(all_results,
                     os.path.join(FIGURES_DIR, 'convergence_curves.png'))
    save_summary(all_results, os.path.join(RESULTS_DIR, 'results_summary.json'))

    print("\nAll done!")
