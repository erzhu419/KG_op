"""
Run GPR-KG experiments on RZDT1, RZDT2, RZDT5 test problems.

This script:
  1. Runs GPR-KG on each test problem for 5 independent replications.
  2. Saves ALL intermediate results to JSON files per problem per replication:
     - Per-iteration timing breakdown (posterior solve, KG computation, etc.)
     - Per-iteration sampling decisions and KG factor values
     - Per-iteration posterior state (visited solutions, parameter dimension)
     - Periodic hypervolume snapshots and Pareto front estimates
     - Pre-sampling and final solution details
     - Full observation history
  3. Generates publication-quality figures:
     - Individual Pareto front plots (true PF vs algorithm output, all 5 reps)
     - Combined 1x3 Pareto front comparison
     - HV convergence curves (all 5 reps + mean)
  4. Saves a summary JSON with aggregate statistics.

Output directory structure:
  GPR_KG_Code/
    results/
      RZDT1_rep1_full.json   # complete intermediate results for RZDT1 rep 1
      RZDT1_rep2_full.json   # ...
      ...
      RZDT5_rep5_full.json
      results_summary.json   # aggregate summary across all experiments
    figures/
      pareto_front_RZDT1.png
      pareto_front_RZDT2.png
      pareto_front_RZDT5.png
      pareto_fronts_combined.png
      convergence_curves.png

JSON structure for each *_full.json:
  {
    "config": {N, n0, K1, K2, lambda_i, prior_var, ...},
    "pre_sampling": {n0, time_sec, observations: [{x, Y}, ...], ...},
    "iterations": [
      {
        "iteration": 0,            # 0-based main-loop index
        "stage": 30,               # absolute stage (including pre-sampling)
        "t_posterior_solve": 0.01,  # seconds for posterior problem
        "t_candidate_gen": 0.001,  # seconds for candidate set generation
        "t_kg_compute": 0.05,      # seconds for KG factor computation
        "t_simulate": 0.0001,      # seconds for simulation
        "t_belief_update": 0.002,  # seconds for GPR Kalman update
        "t_vepm_update": 0.0001,   # seconds for VEPM update
        "t_hv_eval": 0.01,         # seconds for HV evaluation (0 if skipped)
        "t_total": 0.08,           # total wall-clock for this iteration
        "x_selected": [3,1,5,2,8], # selected solution
        "is_new_solution": true,   # first visit?
        "n_candidates": 70,        # |A_n|
        "n_pareto_kg": 5,          # non-dominated in KG space
        "Y_observed": [0.1, 0.8, 0.6],  # simulation output
        "kg1_selected": 0.03,      # KG factor for obj 1
        "kg2_selected": 0.01,      # KG factor for obj 2
        "n_visited": 31,           # distinct solutions visited
        "theta_dim": 42,           # dim of augmented parameter
        "hv": 0.85,                # HV (null if not evaluated this iter)
        "pareto_set_size": 6,      # |PF| (null if not evaluated)
        "pareto_front": [[0.1,0.9],...],  # PF points (null if not evaluated)
      },
      ...
    ],
    "final": {pareto_set_size, pareto_solutions: [{x,f1,f2,f3},...], ...},
    "hv_history": [[30, 0.5], [40, 0.7], ...],
    "all_observations": [{x, Y}, ...],
  }
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import time
import json

from gpr_kg import (RZDT1, RZDT2, RZDT5, GPRKR_Algorithm,
                     pareto_filter, compute_hypervolume_2d)

# ---- Plot style ----
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 15
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# Experiment runner
# =============================================================================

def run_single_experiment(problem_class, problem_name, d=5, L=20, sigma=0.1,
                          N=300, n0=30, n_reps=5, seed_base=42):
    """Run GPR-KG on a problem for multiple replications, saving all intermediate results.

    For each replication:
      - Runs the full GPR-KG algorithm with detailed per-iteration logging
      - Saves the complete results (config, iterations, observations) to a JSON file
      - Collects Pareto fronts and HV histories for plotting

    Args:
        problem_class: RZDT1, RZDT2, or RZDT5 class.
        problem_name (str): Name for file/plot labeling.
        d, L, sigma: Problem parameters.
        N, n0: Budget parameters.
        n_reps: Number of independent replications.
        seed_base: Starting random seed (rep k uses seed_base + k).

    Returns:
        dict with aggregated results for plotting and summary.
    """
    print(f"\n{'='*60}")
    print(f"  {problem_name}: d={d}, L={L}, sigma={sigma}, N={N}, reps={n_reps}")
    print(f"{'='*60}")

    all_pareto_fronts = []       # true objectives of algorithm output per rep
    all_estimated_fronts = []    # posterior mean objectives per rep
    all_hv_histories = []        # HV convergence data per rep
    all_times = []               # total time per rep
    all_hv_final = []            # final HV per rep

    for rep in range(n_reps):
        print(f"\n  Rep {rep+1}/{n_reps}...", end=" ", flush=True)
        t0 = time.time()

        # Create problem instance
        prob = problem_class(d=d, L=L, sigma=sigma, alpha=0.05)
        prob.calibrate_constraint(feasibility_ratio=0.5)

        # Run algorithm with full logging
        alg = GPRKR_Algorithm(
            problem=prob, N=N, n0=n0,
            K1=20, K2=2,
            lambda_i=0.1, prior_var=100.0, w_vepm=5.0,
            n_thr=20, seed=seed_base + rep
        )
        pareto_set = alg.run(verbose=False)
        elapsed = time.time() - t0
        all_times.append(elapsed)

        # Get true objectives of estimated Pareto set
        true_pf = alg.get_true_objectives_of_estimate(pareto_set)
        est_pf = alg.get_estimated_pareto_front(pareto_set)
        all_pareto_fronts.append(true_pf)
        all_estimated_fronts.append(est_pf)

        # Final HV
        ref_point = np.array([1.5, 1.5])
        hv = compute_hypervolume_2d(true_pf, ref_point) if len(true_pf) > 0 else 0.0
        all_hv_final.append(hv)
        all_hv_histories.append(alg.hv_history)

        print(f"HV={hv:.4f}, |PF|={len(true_pf)}, {elapsed:.0f}s")

        # ---- Save complete intermediate results for this replication ----
        full_results = alg.get_full_results()
        full_results['replication'] = rep + 1
        full_results['problem_name'] = problem_name
        full_results['total_time_sec'] = elapsed
        full_results['final_hv'] = hv
        full_results['true_pareto_front'] = true_pf.tolist() if len(true_pf) > 0 else []
        full_results['estimated_pareto_front'] = est_pf.tolist() if len(est_pf) > 0 else []

        # Convert tuple keys to strings for JSON serialization
        save_path = os.path.join(RESULTS_DIR, f'{problem_name}_rep{rep+1}_full.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, default=_json_serializer)
        print(f"    Saved: {save_path}")

    return {
        'problem_name': problem_name,
        'pareto_fronts': all_pareto_fronts,
        'estimated_fronts': all_estimated_fronts,
        'hv_final': all_hv_final,
        'hv_histories': all_hv_histories,
        'times': all_times,
        'N': N, 'n0': n0, 'd': d, 'L': L, 'sigma': sigma,
    }


def _json_serializer(obj):
    """Custom JSON serializer for numpy types and tuples."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# Plotting functions
# =============================================================================

def get_true_pareto_front(problem_class, d=5, L=20, sigma=0.1):
    """Compute the true discrete Pareto front for a problem."""
    prob = problem_class(d=d, L=L, sigma=sigma, alpha=0.05)
    prob.calibrate_constraint(feasibility_ratio=0.5)
    return prob.true_pareto_front(), prob


def plot_pareto_front(results, problem_class, save_path):
    """Plot true vs algorithm Pareto fronts with all 5 replications.

    Blue stars: true discrete Pareto front.
    Colored markers (circle, square, triangle, etc.): each replication.
    Best replication highlighted with connecting line.
    """
    problem_name = results['problem_name']
    d, L, sigma = results['d'], results['L'], results['sigma']
    true_pf, prob = get_true_pareto_front(problem_class, d, L, sigma)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Continuous reference curve
    if hasattr(prob, 'true_pareto_curve'):
        f1_c, f2_c = prob.true_pareto_curve()
        ax.plot(f1_c, f2_c, 'k--', linewidth=1.0, alpha=0.3,
                label='Continuous ZDT curve')

    # True discrete PF
    if len(true_pf) > 0:
        ax.scatter(true_pf[:, 0], true_pf[:, 1], c='blue', s=80, marker='*',
                   zorder=5, edgecolors='navy', linewidths=0.5,
                   label=f'True Pareto front ({len(true_pf)} pts)')

    # Each replication with distinct color and marker
    rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    rep_markers = ['o', 's', '^', 'v', 'P']
    for rep, pf in enumerate(results['pareto_fronts']):
        if len(pf) > 0:
            ax.scatter(pf[:, 0], pf[:, 1],
                       c=[rep_colors[rep]], s=40, marker=rep_markers[rep],
                       alpha=0.7, edgecolors='gray', linewidths=0.3,
                       label=f'GPR-KG Rep {rep+1} (HV={results["hv_final"][rep]:.4f})')

    # Highlight best replication with connecting line
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


def plot_convergence(all_results, save_path):
    """Plot HV convergence curves for all 3 problems.

    Each subplot shows 5 colored lines (one per replication) plus
    a black mean curve. All reps are clearly labeled.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, results in enumerate(all_results):
        ax = axes[idx]
        problem_name = results['problem_name']

        # Plot each replication
        for rep, hv_hist in enumerate(results['hv_histories']):
            if len(hv_hist) > 0:
                stages, hvs = zip(*hv_hist)
                ax.plot(stages, hvs, color=rep_colors[rep % len(rep_colors)],
                        alpha=0.7, linewidth=1.2, label=f'Rep {rep+1}')

        # Mean curve across replications
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


def plot_combined_pareto(all_results, problem_classes, save_path):
    """Combined 1x3 figure of all Pareto fronts."""
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

        # Best replication highlighted
        best_rep = np.argmax(results['hv_final'])
        best_pf = results['pareto_fronts'][best_rep]
        if len(best_pf) > 0:
            s_idx = np.argsort(best_pf[:, 0])
            ax.plot(best_pf[s_idx, 0], best_pf[s_idx, 1],
                    'r-', linewidth=1.2, alpha=0.5)
            ax.scatter(best_pf[:, 0], best_pf[:, 1], c='red', s=50, marker='D',
                       zorder=6, edgecolors='darkred', linewidths=0.5,
                       label=f'GPR-KG Best (HV={results["hv_final"][best_rep]:.4f})')

        # Other reps
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


def save_results_summary(all_results, save_path):
    """Save aggregate summary statistics to JSON and print table."""
    summary = {}
    for results in all_results:
        name = results['problem_name']
        hv = results['hv_final']
        summary[name] = {
            'HV_mean': float(np.mean(hv)),
            'HV_std': float(np.std(hv)),
            'HV_best': float(np.max(hv)),
            'HV_worst': float(np.min(hv)),
            'HV_per_rep': [float(h) for h in hv],
            'PF_sizes': [len(pf) for pf in results['pareto_fronts']],
            'Time_per_rep_sec': [float(t) for t in results['times']],
            'Time_mean_sec': float(np.mean(results['times'])),
            'N': results['N'], 'n0': results['n0'],
            'd': results['d'], 'L': results['L'], 'sigma': results['sigma'],
        }

    with open(save_path, 'w', encoding='utf-8') as f:
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
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  GPR-KG Algorithm: Comprehensive Evaluation")
    print("  Test Problems: RZDT1, RZDT2, RZDT5")
    print("  All intermediate results saved to JSON")
    print("=" * 60)

    # ---- Experiment parameters (matching paper Appendix A4) ----
    d = 5           # decision space dimension
    L = 20          # levels per dimension, x_j in {1,...,20}
    sigma = 0.1     # homoscedastic noise std
    N = 300         # total simulation budget
    n0 = 30         # pre-sampling budget
    n_reps = 5      # independent replications per problem

    problem_configs = [
        (RZDT1, 'RZDT1'),
        (RZDT2, 'RZDT2'),
        (RZDT5, 'RZDT5'),
    ]

    all_results = []
    problem_classes = []

    for prob_cls, prob_name in problem_configs:
        results = run_single_experiment(
            problem_class=prob_cls,
            problem_name=prob_name,
            d=d, L=L, sigma=sigma,
            N=N, n0=n0, n_reps=n_reps,
            seed_base=42
        )
        all_results.append(results)
        problem_classes.append(prob_cls)

        # Individual Pareto front plot
        plot_pareto_front(
            results, prob_cls,
            os.path.join(FIGURES_DIR, f'pareto_front_{prob_name}.png'))

    # Combined plots
    plot_combined_pareto(
        all_results, problem_classes,
        os.path.join(FIGURES_DIR, 'pareto_fronts_combined.png'))

    plot_convergence(
        all_results,
        os.path.join(FIGURES_DIR, 'convergence_curves.png'))

    # Summary
    save_results_summary(
        all_results,
        os.path.join(RESULTS_DIR, 'results_summary.json'))

    print(f"\n  All experiments completed!")
    print(f"  Intermediate results: {RESULTS_DIR}/<problem>_rep<k>_full.json")
    print(f"  Figures: {FIGURES_DIR}/")
