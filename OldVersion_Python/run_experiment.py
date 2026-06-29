"""Main GPR-KG algorithm loop and experiment runner.

Translated from: sim_test.m + example_with_post_process.m

Algorithm flow:
  1. Pre-sample N0 solutions via LHD
  2. For n = 1 to N:
     a. Generate candidate solutions (K1 random + K2 posterior)
     b. Compute noise variance for each candidate (VEPM)
     c. Compute KG factor for each candidate
     d. Select best candidate via Pareto+weighted average
     e. Simulate at selected solution
     f. Kalman update: b, B
     g. VEPM update: Lem, Lem_s
  3. Post-processing: find Pareto front, compute metrics

Default parameters (from example_with_post_process.m):
  N=100, d=5, N0=50, m=40, n_thr=20, s0=1, var0=0.01
  K1=20, K2=2, stdev=0.05, tau_e=-0.04, alpha=1.645
"""
import os
import sys
import json
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.basis_functions import compute_features, num_features
from core.initialization import presample
from core.kalman_update import kalman_update, posterior_mean
from core.kg_computation import select_kg_solution
from core.candidate_gen import generate_candidates
from core.vepm import VEPM
from core.utils import x_in_s, batch_x_in_s
from core.pareto_utils import (pareto_front_indices, compute_hypervolume_2d,
                                compute_igd)
from core.test_problems import PROBLEMS


def run_gpr_kg(problem_name, seed, N=100, d=5, N0=50, m=40, n_thr=20,
               K1=20, K2=2, use_vepm=True, verbose=True):
    """Run the old-version GPR-KG algorithm.

    Parameters
    ----------
    problem_name : str
        'RZDT1', 'RZDT2', or 'RZDT5'.
    seed : int
        Random seed.
    N : int
        Number of iterations (sampling decisions).
    d : int
        Decision dimension.
    N0 : int
        Number of pre-samples.
    m : int
        Number of recent solutions for candidate generation.
    n_thr : int
        Iteration threshold for constraint activation.
    K1 : int
        Number of random candidates.
    K2 : int
        Number of posterior draws for candidates.
    use_vepm : bool
        If False, use prior variance (no VEPM update). This is GPR-KG-nV.
    verbose : bool

    Returns
    -------
    dict
        Complete result dictionary with metrics and history.
    """
    rng = np.random.RandomState(seed)
    problem = PROBLEMS[problem_name](d=d)
    p = num_features(d)
    n_obj = 3

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running GPR-KG {'(with VEPM)' if use_vepm else '(no VEPM)'}")
        print(f"Problem: {problem_name}, d={d}, N0={N0}, N={N}, seed={seed}")
        print(f"{'='*60}")

    t_start = time.time()

    # ===== Phase 1: Pre-sampling =====
    # MATLAB: pre_sample returns b0, B0, z0 but does NOT store pre-sampled
    # solutions in 'sampled'. sampled starts empty in the main loop.
    b, B, z0, presample_norms, presample_orig, Y_pre = presample(
        N0, d, problem, rng=rng
    )

    # Main-loop sampled solutions (starts EMPTY, matching MATLAB sim_test.m)
    sampled_norms = []  # only main-loop samples, for KG/Kalman
    sampled_orig = []
    num_samples = np.array([], dtype=int)
    n_dev = 0  # number of deviation terms in b (grows by 1 per iteration)

    # Initialize VEPM with prior values only (MATLAB: Lem=Lem0, Lem_s=[])
    vepm = VEPM(d=d, n_obj=n_obj, s0=1.0, var0=0.01, n_thr=n_thr)

    # Track histories for all figures
    hv_history = []
    igd_history = []         # Fig 2: IGD convergence
    rmse_history = []        # Fig 3: RMSE of f1, f2, f3
    infeasible_history = []  # Fig 4: number of infeasible in non-dominated set
    variance_history = []    # Fig 5: VEPM variance estimates at reference points
    iter_solutions = []      # Fig 1: iterative solutions (true objectives)
    ref_point = np.array([1.5, 1.5])
    iter_times = []

    # Select a fixed reference point on the true PF for Fig 5 variance tracking
    true_pf_solutions = []
    for x1_val in range(int(problem.x_U[0]) + 1):
        x_ref = np.copy(problem.x_L)
        x_ref[0] = x1_val
        if problem.is_truly_feasible(x_ref):
            x_ref_norm = problem.normalize(x_ref)
            true_pf_solutions.append(x_ref_norm)
    vepm_ref_point = true_pf_solutions[len(true_pf_solutions) // 2] if true_pf_solutions else None

    if verbose:
        print(f"Pre-sampling done: {N0} samples, p={p}")

    # ===== Phase 2: Main loop =====
    for iteration in range(1, N + 1):
        t_iter_start = time.time()

        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}/{N}...")

        # Recent sampled solutions (last m) — main-loop only, matching MATLAB
        n_sampled = len(sampled_norms)
        start_idx = max(0, n_sampled - m)
        recent_norms = sampled_norms[start_idx:]

        # Generate candidate solutions
        x_range = problem.x_U - problem.x_L
        candidates = generate_candidates(
            iteration, n_thr, K1, K2, d, b, B,
            recent_norms, vepm,
            problem.tau_e, problem.alpha_z, problem.stdev,
            rng=rng, x_range=x_range
        )
        K = len(candidates)

        if K == 0:
            if verbose:
                print(f"  Warning: no candidates at iteration {iteration}")
            continue

        # Compute variance estimate for each candidate
        # MATLAB: var_x always uses Lem (partition-based) since x_in_s never matches
        lem_all = np.zeros((n_obj, K))
        for k in range(K):
            lem_all[:, k] = vepm.get_variance(candidates[k], None)

        if not use_vepm:
            lem_all[:, :] = 0.01

        # Select sampling decision via KG
        best_idx = select_kg_solution(
            candidates, sampled_norms, b, B, z0, lem_all, d
        )
        x_star_norm = candidates[best_idx]
        x_star_orig = problem.denormalize(x_star_norm)
        x_star_norm = problem.normalize(x_star_orig)  # canonical integer-grid form

        # Simulate
        y = problem.simulate(x_star_orig, rng=rng)

        # Record iterative solution's true objectives (Fig 1)
        f_true_star = problem.true_objectives(x_star_orig)
        is_feasible_star = problem.is_truly_feasible(x_star_orig)
        iter_solutions.append({
            'iteration': iteration,
            'f_true': f_true_star.tolist(),
            'feasible': bool(is_feasible_star),
        })

        # MATLAB-compatible: always treat as new (x_in_s never matches in MATLAB)
        sol_idx = n_dev  # deviation term index (0, 1, 2, ...)
        is_new = True
        n_dev += 1
        sampled_norms.append(x_star_norm.copy())
        sampled_orig.append(x_star_orig.copy())
        num_samples = np.append(num_samples, 1)

        # Kalman update
        phi_x = compute_features(x_star_norm)
        lem_x = lem_all[:, best_idx]
        b, B = kalman_update(
            b, B, z0, phi_x, lem_x, sol_idx, y, n_dev - 1, is_new
        )

        # VEPM update
        if use_vepm:
            theta_pred = np.array([
                posterior_mean(b[i], phi_x, sol_idx, n_dev)
                for i in range(n_obj)
            ])
            vepm.update(x_star_norm, y, theta_pred, sol_idx,
                        num_samples[sol_idx], sampled_norms, num_samples)

        # Track timing
        t_iter = time.time() - t_iter_start
        iter_times.append(t_iter)

        # Track VEPM variance at reference point (Fig 5)
        if vepm_ref_point is not None:
            var_est = vepm.get_variance(vepm_ref_point, None)
            variance_history.append((N0 + iteration, var_est.tolist()))

        # Compute full metrics every 5 iterations (Figs 2-4)
        if iteration % 5 == 0 or iteration == N:
            # Use ALL solutions (pre-samples + main-loop) for metrics
            all_norms = presample_norms + sampled_norms
            all_orig = presample_orig + sampled_orig
            nd_data = _compute_iterative_metrics(
                b, all_norms, all_orig, N0, n_dev, problem, ref_point, d
            )
            igd_history.append((N0 + iteration, nd_data['igd']))
            rmse_history.append((N0 + iteration, nd_data['rmse_f1'],
                                 nd_data['rmse_f2'], nd_data['rmse_f3']))
            infeasible_history.append((N0 + iteration, nd_data['n_infeasible']))
            hv_history.append((N0 + iteration, float(nd_data['hv'])))

    # ===== Phase 3: Post-processing =====
    t_total = time.time() - t_start

    # Find final Pareto front from ALL sampled solutions (pre + main)
    all_norms = presample_norms + sampled_norms
    all_orig = presample_orig + sampled_orig
    result = _postprocess(b, all_norms, all_orig, N0, n_dev,
                          problem, ref_point, d)

    result.update({
        'problem': problem_name,
        'seed': seed,
        'method': 'GPR-KG' if use_vepm else 'GPR-KG-nV',
        'N': N,
        'N0': N0,
        'd': d,
        'K1': K1,
        'K2': K2,
        'use_vepm': use_vepm,
        'time_total': t_total,
        'time_per_iter_mean': float(np.mean(iter_times)) if iter_times else 0,
        'n_sampled_unique': len(presample_norms) + len(sampled_norms),
        'n_fes': len(presample_norms) + len(sampled_norms),
        # Intermediate data for figures
        'hv_history': hv_history,
        'igd_history': igd_history,
        'rmse_history': rmse_history,
        'infeasible_history': infeasible_history,
        'variance_history': variance_history,
        'iter_solutions': iter_solutions,
    })

    if verbose:
        print(f"\nResults:")
        print(f"  HV = {result['hv_final']:.4f}")
        print(f"  IGD = {result['igd_final']:.4f}")
        print(f"  #LPOS = {result['n_lpos']}")
        print(f"  #Infeasible = {result['n_infeasible']}")
        print(f"  Time = {t_total:.1f}s ({np.mean(iter_times):.2f}s/iter)")

    return result


def _get_dev_idx(k, n_presample):
    """Map all-solutions index k to deviation term index.

    Pre-samples (k < n_presample) have no deviation terms → return None.
    Main-loop samples (k >= n_presample) → return k - n_presample.
    """
    if k < n_presample:
        return None
    return k - n_presample


def _compute_iterative_metrics(b, all_norms, all_orig, n_presample, n_dev,
                               problem, ref_point, d):
    """Compute all iterative metrics for figures.

    Parameters
    ----------
    all_norms, all_orig : lists of ALL solutions (pre-samples + main-loop)
    n_presample : int, number of pre-sample solutions
    n_dev : int, number of deviation terms in b
    """
    n_obj = 3
    n_total = len(all_norms)
    true_pf = problem.true_pareto_front()

    # Compute true objectives and feasibility for all solutions
    true_obj = np.zeros((n_total, 3))
    feasible = np.zeros(n_total, dtype=bool)
    for k in range(n_total):
        true_obj[k] = problem.true_objectives(all_orig[k])
        feasible[k] = problem.is_truly_feasible(all_orig[k])

    # Find non-dominated set among feasible solutions (on TRUE objectives)
    hv = 0.0
    igd = float('inf')
    n_infeasible_nd = 0

    if np.any(feasible):
        feasible_idx = np.where(feasible)[0]
        feasible_obj_2d = true_obj[feasible_idx, :2]
        pf_local_idx = pareto_front_indices(feasible_obj_2d)
        if pf_local_idx:
            pf_idx = feasible_idx[pf_local_idx]
            pareto_obj = true_obj[pf_idx, :2]
            hv = compute_hypervolume_2d(pareto_obj, ref_point)
            igd = compute_igd(pareto_obj, true_pf)

    # Compute posterior means for RMSE (only for main-loop solutions with deviations)
    pred_obj = np.zeros((n_total, 3))
    for k in range(n_total):
        phi_k = compute_features(all_norms[k])
        dev_idx = _get_dev_idx(k, n_presample)
        for i in range(n_obj):
            pred_obj[k, i] = posterior_mean(b[i], phi_k, dev_idx, n_dev)

    # Count infeasible in non-dominated set (using posterior predictions)
    pred_2d = pred_obj[:, :2]
    nd_idx = pareto_front_indices(pred_2d)
    if nd_idx:
        n_infeasible_nd = sum(1 for k in nd_idx if not feasible[k])

    # RMSE for each objective on the non-dominated set
    rmse_f1 = rmse_f2 = rmse_f3 = 0.0
    if nd_idx:
        nd_arr = np.array(nd_idx)
        rmse_f1 = float(np.sqrt(np.mean((pred_obj[nd_arr, 0] - true_obj[nd_arr, 0])**2)))
        rmse_f2 = float(np.sqrt(np.mean((pred_obj[nd_arr, 1] - true_obj[nd_arr, 1])**2)))
        rmse_f3 = float(np.sqrt(np.mean((pred_obj[nd_arr, 2] - true_obj[nd_arr, 2])**2)))

    return {
        'hv': float(hv),
        'igd': float(igd),
        'rmse_f1': rmse_f1,
        'rmse_f2': rmse_f2,
        'rmse_f3': rmse_f3,
        'n_infeasible': n_infeasible_nd,
    }


def _postprocess(b, all_norms, all_orig, n_presample, n_dev,
                 problem, ref_point, d):
    """Post-processing: compute all metrics.

    Translated from example_with_post_process.m:
      P = perato_con(Mean{N}, sampled_short, ...) -- NSGA-II on posterior model
      P = x_L'+P.*(x_U-x_L)'; P = unique(P, 'rows');
      for i: obj_sol(:,i) = sim_func(P(i,:), n, zeros(3,1));  -- true objectives
    """
    from core.candidate_gen import _PosteriorBiObjProblemFast
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination

    p_dim = num_features(d)
    x_range = problem.x_U - problem.x_L

    # Run NSGA-II on final posterior model (matching MATLAB perato_con)
    # Use only parametric part of b (MATLAB behavior: deviation terms not used)
    bb_param = [b[i][:p_dim] for i in range(3)]

    nsga_problem = _PosteriorBiObjProblemFast(
        bb_param, p_dim, d,
        tau_e=problem.tau_e, alpha_z=problem.alpha_z, sigma_3=problem.stdev[2]
    )
    # MATLAB: gamultiobj default MaxGenerations = 200*n_var = 1000 for d=5
    algorithm = NSGA2(pop_size=200)
    try:
        res = pymoo_minimize(
            nsga_problem, algorithm,
            get_termination("n_gen", 1000),
            seed=42, verbose=False
        )
        if res.X is not None:
            X_result = res.X.copy()
            if X_result.ndim == 1:
                X_result = X_result.reshape(1, -1)
            # Round to problem's integer grid
            for i in range(d):
                X_result[:, i] = np.round(X_result[:, i] * x_range[i]) / x_range[i]
            X_result = np.clip(X_result, 0, 1)
            X_result = np.unique(X_result, axis=0)

            # Evaluate true objectives at posterior-optimal solutions
            pareto_solutions_model = []
            pareto_obj_model = []
            for row in X_result:
                x_orig = problem.denormalize(row)
                f_true = problem.true_objectives(x_orig)
                if problem.is_truly_feasible(x_orig):
                    pareto_solutions_model.append(x_orig)
                    pareto_obj_model.append(f_true[:2])

            pareto_obj_model = np.array(pareto_obj_model) if pareto_obj_model else np.empty((0, 2))
        else:
            pareto_obj_model = np.empty((0, 2))
            pareto_solutions_model = []
    except Exception:
        pareto_obj_model = np.empty((0, 2))
        pareto_solutions_model = []

    # Find Pareto front among the model-optimized solutions
    if len(pareto_obj_model) > 0:
        pf_local_idx = pareto_front_indices(pareto_obj_model)
        pareto_obj = pareto_obj_model[pf_local_idx]
        pareto_solutions = [pareto_solutions_model[i].tolist() for i in pf_local_idx]
    else:
        pareto_obj = np.empty((0, 2))
        pareto_solutions = []

    # HV
    hv = compute_hypervolume_2d(pareto_obj, ref_point) if len(pareto_obj) > 0 else 0.0

    # IGD
    true_pf = problem.true_pareto_front()
    igd = compute_igd(pareto_obj, true_pf)

    # Number of Learned Pareto-Optimal Solutions (#LPOS)
    # Count TRUE PF points that have at least one close estimated point
    n_lpos = 0
    if len(pareto_obj) > 0 and len(true_pf) > 0:
        for p_true in true_pf:
            min_dist = np.min(np.linalg.norm(pareto_obj - p_true, axis=1))
            if min_dist < 0.05:
                n_lpos += 1

    # Infeasible count from all actually-sampled solutions
    n_total = len(all_orig)
    n_infeasible = sum(1 for k in range(n_total) if not problem.is_truly_feasible(all_orig[k]))

    pareto_objectives = pareto_obj.tolist() if len(pareto_obj) > 0 else []

    return {
        'hv_final': float(hv),
        'igd_final': float(igd),
        'n_lpos': int(n_lpos),
        'n_infeasible': int(n_infeasible),
        'n_pareto': len(pareto_obj),
        'pareto_solutions': pareto_solutions,
        'pareto_objectives_true': pareto_objectives,
    }


def main():
    """Run a single experiment from command line."""
    import argparse
    parser = argparse.ArgumentParser(description='Run GPR-KG (Old Version)')
    parser.add_argument('--problem', default='RZDT2', choices=['RZDT1', 'RZDT2', 'RZDT5'])
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--N0', type=int, default=50)
    parser.add_argument('--K1', type=int, default=20)
    parser.add_argument('--K2', type=int, default=2)
    parser.add_argument('--no-vepm', action='store_true')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    result = run_gpr_kg(
        problem_name=args.problem,
        seed=args.seed,
        N=args.N,
        d=args.d,
        N0=args.N0,
        K1=args.K1,
        K2=args.K2,
        use_vepm=not args.no_vepm,
    )

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    method_name = 'GPR_KG' if not args.no_vepm else 'GPR_KG_nV'
    fname = f"{args.problem}_{method_name}_rep{args.seed}.json"
    fpath = os.path.join(args.output_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {fpath}")


if __name__ == '__main__':
    main()
