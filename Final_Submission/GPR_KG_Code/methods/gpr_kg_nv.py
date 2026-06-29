"""
GPR-KG-nV: GPR-KG with VEPM disabled (ablation variant).

This is an ablation study that isolates the contribution of VEPM by replacing
it with a simple variance estimator while keeping EVERY other algorithmic
component identical to GPR-KG:

  * Same parametric GPR model (ParametricGPR)
  * Same KG factor computation (compute_kg_factor / compute_h)
  * Same candidate generation: K1 LHD + K2 NSGA-II posterior-sampling runs
  * Same Pareto-KG selection criterion (crowding-distance among non-dominated KG pairs)
  * Same HV evaluation on true objectives

The only difference:
  VEPM  →  simple per-solution sample variance
  Variance at x:
    - If ≥ 2 observations at x: sample variance of residuals (Bessel-corrected)
    - Otherwise: global_var_hat[i] = mean of all per-solution sample variances
                 (or fixed 0.01 if no replications exist anywhere)

This design guarantees that any performance difference between GPR-KG and
GPR-KG-nV is attributable solely to VEPM, not to differences in candidate
generation or acquisition structure.
"""

import numpy as np
import time
import sys
import os
from scipy.stats import norm, qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpr_kg import (ParametricGPR, compute_kg_factor, compute_h,
                     pareto_filter, crowding_distance_select,
                     compute_hypervolume_2d,
                     _PosteriorBiObjProblem, _pareto_front_indices)
from methods.base_method import BaseMethod
from experiments.config import HV_EVAL_INTERVAL, GPR_KG_PARAMS

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False


def _json_safe(obj):
    """Convert numpy/tuple objects in detailed logs to JSON-safe values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj


class GPRKGnVMethod(BaseMethod):
    """GPR-KG-nV: ablation of GPR-KG without VEPM.

    Identical to GPR-KG except that observation noise variance is estimated
    directly from replications at each sampled solution rather than using
    the VEPM partition-based extrapolation.

    This ensures the ablation isolates only the effect of VEPM.
    """

    name = "GPR-KG-nV"

    def __init__(self, **override_params):
        self.params = dict(GPR_KG_PARAMS)
        self.params.update(override_params)
        # Optional instrumentation (set before calling run):
        #   {'eval_x': [tuples], 'ref_x': tuple, 'stride': int}
        self.instrument = None
        self.instrument_log = []

    def run(self, problem, N, n0, seed, hv_eval_interval=HV_EVAL_INTERVAL):
        if seed is not None:
            np.random.seed(seed)

        t_start = time.time()
        d = problem.d
        L = problem.L
        K1 = self.params['K1']
        K2 = self.params['K2']
        n_thr = self.params['n_thr']
        p = 2 * d + 1  # basis dimension: [1, x, x^2]
        use_boundary_initial_design = self.params.get(
            'use_boundary_initial_design', True)
        use_archive_candidates = self.params.get('use_archive_candidates', False)
        archive_neighbor_radius = int(self.params.get('archive_neighbor_radius', 0) or 0)
        kg_selection_tiebreak = self.params.get(
            'kg_selection_tiebreak', 'crowding_distance')
        variance_floor = float(self.params.get('variance_floor', 1e-8) or 0.0)

        # Three parametric GPR models (identical to GPR-KG)
        gpr = [ParametricGPR(d, self.params['lambda_i'], self.params['prior_var'])
               for _ in range(3)]

        observations = {}   # x_tuple -> list of Y arrays
        history = []
        hv_history = []
        time_per_iter = []
        iteration_log = []

        # ------------------------------------------------------------------
        # Variance estimation helpers (replace VEPM)
        # ------------------------------------------------------------------

        def _local_variance(i, x_tuple):
            """Sample variance of residuals at x (Bessel-corrected)."""
            obs_list = observations.get(x_tuple, [])
            if len(obs_list) < 2:
                return None
            resids = [obs[i] - gpr[i].posterior_mean(np.array(x_tuple))
                      for obs in obs_list]
            return max(float(np.var(resids, ddof=1)), 1e-8)

        def _global_var_hat(i):
            """Mean of all available per-solution sample variances.

            Falls back to 0.01 when no solution has ≥2 observations.
            This is used for unvisited solutions and single-observation
            solutions in place of VEPM's partition extrapolation.
            """
            local_vars = []
            for x_t, obs_list in observations.items():
                if len(obs_list) >= 2:
                    v = _local_variance(i, x_t)
                    if v is not None:
                        local_vars.append(v)
            return float(np.mean(local_vars)) if local_vars else 0.01

        def get_variance_nv(i, x):
            """Variance estimate without VEPM.

            Returns local sample variance when ≥2 observations are available;
            otherwise returns the global empirical mean variance across all
            replicated solutions (or 0.01 prior if no replicates exist).
            """
            x_tuple = tuple(x)
            v = _local_variance(i, x_tuple)
            if v is not None:
                return max(v, variance_floor)
            return max(_global_var_hat(i), variance_floor)

        def boundary_solutions():
            lo, hi = problem.int_bounds()
            lo = np.asarray(lo, dtype=int)
            hi = np.asarray(hi, dtype=int)
            center = np.round((lo + hi) / 2.0).astype(int)
            seeds = {tuple(lo), tuple(hi), tuple(center)}
            for j in range(d):
                x_hi = lo.copy()
                x_hi[j] = hi[j]
                seeds.add(tuple(x_hi))
                x_mid = lo.copy()
                x_mid[j] = center[j]
                seeds.add(tuple(x_mid))
            return list(seeds)

        def neighbor_solutions(x_tuple):
            if archive_neighbor_radius <= 0:
                return []
            lo, hi = problem.int_bounds()
            lo = np.asarray(lo, dtype=int)
            hi = np.asarray(hi, dtype=int)
            x0 = np.asarray(x_tuple, dtype=int)
            neigh = set()
            for j in range(d):
                for step in range(1, archive_neighbor_radius + 1):
                    for sign in (-1, 1):
                        x = x0.copy()
                        x[j] = int(np.clip(x[j] + sign * step, lo[j], hi[j]))
                        neigh.add(tuple(x))
            neigh.discard(tuple(x0))
            return list(neigh)

        def quality_select(candidate_set, pareto_kg_idx, kg_pairs):
            if len(pareto_kg_idx) == 0:
                return int(np.random.randint(len(candidate_set)))
            if kg_selection_tiebreak != 'posterior_quality':
                nd_kg = kg_pairs[pareto_kg_idx]
                local_idx = crowding_distance_select(nd_kg)
                return int(pareto_kg_idx[local_idx])
            q_alpha = norm.ppf(1 - problem.alpha)
            scores = []
            for idx in pareto_kg_idx:
                x_arr = np.array(candidate_set[idx])
                mu1 = gpr[0].posterior_mean(x_arr)
                mu2 = gpr[1].posterior_mean(x_arr)
                mu3 = gpr[2].posterior_mean(x_arr)
                sig3 = np.sqrt(get_variance_nv(2, x_arr))
                margin = mu3 + q_alpha * sig3 - problem.tau
                scores.append((mu1 + mu2 + max(0.0, margin), idx))
            scores.sort(key=lambda z: z[0])
            return int(scores[0][1])

        # ------------------------------------------------------------------
        # Candidate generation (identical to GPR-KG, using nV variance for NSGA-II)
        # ------------------------------------------------------------------

        def _basis_matrix(X_values):
            """[1, x, x^2] feature matrix.  Identical to GPRKR_Algorithm._basis_matrix."""
            N_pts = len(X_values)
            Phi = np.ones((N_pts, 2 * d + 1))
            Phi[:, 1:d + 1] = X_values
            Phi[:, d + 1:] = X_values ** 2
            return Phi

        def generate_candidate_set(iteration, pareto_set=None):
            """LHD + NSGA-II candidate generation — same as GPR-KG.

            Continuous NSGA-II search points are mapped to the integer grid
            before posterior evaluation. The NSGA-II constraint threshold
            uses the nV variance estimate instead of VEPM.
            """
            candidates = set()

            # Part 1: K1 LHD samples
            try:
                sampler = qmc.LatinHypercube(d=d, seed=np.random.randint(100000))
                lhd = sampler.random(K1)
            except Exception:
                lhd = np.random.rand(K1, d)
            for row in lhd:
                candidates.add(problem.continuous_to_int(row))

            if use_archive_candidates:
                archive = pareto_set or []
                for x_tuple in archive:
                    candidates.add(tuple(x_tuple))
                    for x_nb in neighbor_solutions(x_tuple):
                        candidates.add(x_nb)

            # Part 2: K2 posterior-sampled NSGA-II runs
            use_constraint = (iteration > n_thr)

            for _ in range(K2):
                # Sample posterior parametric coefficients (same as GPR-KG)
                bb_param = []
                for i in range(3):
                    b_param = gpr[i].a[:p].copy()
                    B_param = gpr[i].C[:p, :p].copy()
                    B_param = (B_param + B_param.T) / 2
                    eigvals = np.linalg.eigvalsh(B_param)
                    if np.min(eigvals) < 0:
                        B_param -= 1.1 * np.min(eigvals) * np.eye(p)
                    try:
                        theta_i = np.random.multivariate_normal(b_param, B_param)
                    except np.linalg.LinAlgError:
                        theta_i = b_param + np.random.randn(p) * 0.01
                    bb_param.append(theta_i)

                # Build NSGA-II problem using the nV variance estimate instead
                # of VEPM, while keeping candidate generation aligned with the
                # main GPR-KG implementation.
                if use_constraint and HAS_PYMOO:
                    nsga_prob = _PosteriorBiObjProblem(
                        bb_param, p, d, L,
                        to_int_func=problem.continuous_to_int,
                        tau_e=problem.tau,
                        alpha_z=norm.ppf(1 - problem.alpha),
                        variance_lookup=lambda x: get_variance_nv(2, x))
                elif HAS_PYMOO:
                    nsga_prob = _PosteriorBiObjProblem(
                        bb_param, p, d, L,
                        to_int_func=problem.continuous_to_int)
                else:
                    nsga_prob = None

                if nsga_prob is not None and HAS_PYMOO:
                    algorithm = NSGA2(pop_size=100)
                    try:
                        res = pymoo_minimize(
                            nsga_prob, algorithm,
                            get_termination("n_gen", 50),
                            seed=int(np.random.randint(100000)),
                            verbose=False)
                        if res.X is not None:
                            X_result = res.X
                            if X_result.ndim == 1:
                                X_result = X_result.reshape(1, -1)
                            for row in X_result:
                                candidates.add(problem.continuous_to_int(row))
                    except Exception:
                        # Fallback: random search on sampled posterior
                        X_rand = np.random.rand(500, d)
                        X_int = np.array([
                            problem.continuous_to_int(row)
                            for row in X_rand
                        ], dtype=float)
                        Phi = _basis_matrix(X_int)
                        obj = np.column_stack([
                            np.round(Phi @ bb_param[0] * 100) / 100.0,
                            np.round(Phi @ bb_param[1] * 100) / 100.0])
                        pf_idx = _pareto_front_indices(obj)
                        for idx in pf_idx:
                            candidates.add(problem.continuous_to_int(X_rand[idx]))

            return list(candidates) if candidates else [problem.sample_random()]

        # ------------------------------------------------------------------
        # Posterior feasible Pareto set (same structure as GPR-KG)
        # ------------------------------------------------------------------

        def solve_posterior():
            candidates = set()
            for x_tuple in gpr[0].sampled_set:
                candidates.add(x_tuple)
            for _ in range(500):
                candidates.add(problem.sample_random())

            feasible_objs = []
            feasible_sols = []
            q_alpha = norm.ppf(1 - problem.alpha)
            for x_tuple in candidates:
                x_arr = np.array(x_tuple)
                mu1 = gpr[0].posterior_mean(x_arr)
                mu2 = gpr[1].posterior_mean(x_arr)
                mu3 = gpr[2].posterior_mean(x_arr)
                # Use nV variance estimate for constraint check
                sigma3 = np.sqrt(get_variance_nv(2, x_arr))
                if mu3 + q_alpha * sigma3 <= problem.tau:
                    feasible_objs.append([mu1, mu2])
                    feasible_sols.append(x_tuple)

            if len(feasible_objs) == 0:
                return []
            feasible_objs = np.array(feasible_objs)
            _, pareto_idx = pareto_filter(feasible_objs, return_indices=True)
            return [feasible_sols[i] for i in pareto_idx]

        # ------------------------------------------------------------------
        # Phase 1: Pre-sampling
        # ------------------------------------------------------------------

        pre_sample_set = set()
        if use_boundary_initial_design:
            for x_tuple in boundary_solutions():
                if len(pre_sample_set) >= n0:
                    break
                pre_sample_set.add(tuple(x_tuple))
        while len(pre_sample_set) < n0:
            pre_sample_set.add(problem.sample_random())
        pre_samples = list(pre_sample_set)

        for x_tuple in pre_samples:
            x_arr = np.array(x_tuple)
            Y = problem.simulate(x_arr)
            observations.setdefault(x_tuple, []).append(Y)
            history.append((x_tuple, Y))

        # Data-driven initialization (same approach as GPR-KG v2)
        Phi_pre = np.array([
            np.concatenate([[1.0], np.array(x, dtype=float),
                            np.array(x, dtype=float) ** 2])
            for x in pre_samples
        ])
        lambda_data = []
        prior_var_data = []
        beta_hat_data = []
        for i in range(3):
            Y_i = np.array([observations[x][0][i] for x in pre_samples])
            try:
                beta_hat_i = np.linalg.lstsq(Phi_pre, Y_i, rcond=None)[0]
            except Exception:
                beta_hat_i = np.zeros(p)
            residuals_i = Y_i - Phi_pre @ beta_hat_i
            z0_i = max(float(np.var(residuals_i)), 1e-6)
            b_var_i = max(float(np.var(beta_hat_i)), 1e-6)
            lambda_data.append(z0_i)
            prior_var_data.append(b_var_i)
            beta_hat_data.append(beta_hat_i)

        # Reinitialize GPR with data-driven params and OLS mean
        gpr = [ParametricGPR(d, lambda_data[i], prior_var_data[i]) for i in range(3)]
        for i in range(3):
            gpr[i].a[:p] = beta_hat_data[i]

        # Dimension augment all pre-samples (no Kalman update needed)
        for x_tuple in pre_samples:
            for i in range(3):
                gpr[i].dimension_augment(np.array(x_tuple))

        # ------------------------------------------------------------------
        # Phase 2: Main loop (identical to GPR-KG except VEPM → get_variance_nv)
        # ------------------------------------------------------------------

        for n in range(n0, N):
            t_iter = time.time()
            iter_idx = n - n0
            iter_log = {
                "n": int(n),
                "iteration": int(iter_idx),
            }

            t0 = time.time()
            pareto_set = solve_posterior()
            iter_log["t_posterior_solve"] = time.time() - t0

            # LHD + NSGA-II candidate generation (same as GPR-KG)
            t0 = time.time()
            candidate_set = generate_candidate_set(
                iteration=n - n0, pareto_set=pareto_set)
            if not candidate_set:
                candidate_set = [problem.sample_random()]
            candidate_arrays = [np.array(c) for c in candidate_set]
            iter_log["t_candidate_gen"] = time.time() - t0
            iter_log["n_candidates"] = len(candidate_set)
            iter_log["candidate_set"] = [
                [int(v) for v in x_tuple] for x_tuple in candidate_set
            ]

            # Compute Pareto-KG factors using nV variance estimate
            t0 = time.time()
            kg_pairs = []
            for x_tuple in candidate_set:
                x_arr = np.array(x_tuple)
                kg1 = compute_kg_factor(gpr[0], candidate_arrays, x_arr,
                                        get_variance_nv(0, x_arr))
                kg2 = compute_kg_factor(gpr[1], candidate_arrays, x_arr,
                                        get_variance_nv(1, x_arr))
                kg_pairs.append([kg1, kg2])
            kg_pairs = np.array(kg_pairs)
            iter_log["t_kg_compute"] = time.time() - t0
            iter_log["kg_pairs"] = kg_pairs.tolist()

            # Pareto-KG selection (same as GPR-KG)
            _, pareto_kg_idx = pareto_filter(-kg_pairs, return_indices=True)
            iter_log["pareto_kg_indices"] = [int(idx) for idx in pareto_kg_idx]
            if len(pareto_kg_idx) == 0:
                selected_idx = np.random.randint(len(candidate_set))
            else:
                selected_idx = quality_select(candidate_set, pareto_kg_idx, kg_pairs)

            x_selected = candidate_set[selected_idx]
            x_arr = np.array(x_selected)
            iter_log["x_selected"] = [int(v) for v in x_selected]
            iter_log["is_new_solution"] = x_selected not in observations
            iter_log["n_pareto_kg"] = len(pareto_kg_idx)
            iter_log["kg1_selected"] = float(kg_pairs[selected_idx, 0])
            iter_log["kg2_selected"] = float(kg_pairs[selected_idx, 1])

            mu_before = [gpr[i].posterior_mean(x_arr) for i in range(3)]
            sigma2_before = [get_variance_nv(i, x_arr) for i in range(3)]
            iter_log["mu_before_update"] = [float(v) for v in mu_before]
            iter_log["sigma2_before_update"] = [float(v) for v in sigma2_before]

            # Simulate
            t0 = time.time()
            Y = problem.simulate(x_arr)
            iter_log["t_simulate"] = time.time() - t0
            iter_log["Y_observed"] = Y.tolist()
            observations.setdefault(x_selected, []).append(Y)
            history.append((x_selected, Y))

            # Update GPR using nV variance estimate
            t0 = time.time()
            for i in range(3):
                gpr[i].update(x_arr, Y[i], sigma2_before[i])
            iter_log["t_belief_update"] = time.time() - t0

            t_compute = time.time() - t_iter
            time_per_iter.append(t_compute)
            iter_log["n_visited"] = len(gpr[0].sampled_set)
            iter_log["theta_dim"] = len(gpr[0].a)

            # HV on true objectives (same as GPR-KG)
            if (n - n0) % hv_eval_interval == 0 or n == N - 1:
                t0 = time.time()
                ps = solve_posterior()
                if len(ps) > 0:
                    true_objs = np.array([
                        problem.true_objectives(x)[:2] for x in ps])
                    pf = pareto_filter(true_objs)
                    hv = compute_hypervolume_2d(pf, problem.ref_point)
                else:
                    hv = 0.0
                hv_history.append((n, float(hv)))
                iter_log["t_hv_eval"] = time.time() - t0
                iter_log["hv"] = float(hv)
                iter_log["pareto_set_size"] = len(ps)
            else:
                iter_log["t_hv_eval"] = 0.0
                iter_log["hv"] = None
                iter_log["pareto_set_size"] = None

            iter_log["t_total"] = time.time() - t_iter
            iteration_log.append(iter_log)

            # Optional instrumentation snapshot (mirrors gpr_kg.py hook)
            if self.instrument is not None:
                stride = self.instrument.get('stride', 10)
                iter_idx = n - n0
                if (iter_idx % stride == 0) or (n == N - 1):
                    snap = {'n': int(n), 'iteration': int(iter_idx)}
                    eval_x = self.instrument.get('eval_x')
                    if eval_x:
                        eval_arr = [np.array(x) for x in eval_x]
                        for i in range(3):
                            mu_vals = np.array(
                                [gpr[i].posterior_mean(xa) for xa in eval_arr])
                            snap[f'mu{i}_eval'] = mu_vals.tolist()
                    ref_x = self.instrument.get('ref_x')
                    if ref_x is not None:
                        ref_arr = np.array(ref_x)
                        snap['sigma2_ref'] = [
                            float(get_variance_nv(i, ref_arr))
                            for i in range(3)]
                    ps_snap = solve_posterior()
                    snap['pareto_set'] = [tuple(int(v) for v in x) for x in ps_snap]
                    self.instrument_log.append(snap)

        # ------------------------------------------------------------------
        # Phase 3: Final output
        # ------------------------------------------------------------------

        total_time = time.time() - t_start
        final_pareto = solve_posterior()

        result = self._make_result(
            pareto_solutions=final_pareto,
            problem=problem,
            hv_history=hv_history,
            total_time=total_time,
            time_per_iter=time_per_iter,
            n_simulations=N,
            ref_point=problem.ref_point,
        )
        result["iteration_log"] = _json_safe(iteration_log)
        result["observation_history"] = [
            {"x": [int(v) for v in x], "Y": [float(y) for y in Y]}
            for x, Y in history
        ]
        result["instrument_log"] = _json_safe(self.instrument_log)
        result["logging_detail"] = (
            "Per adaptive iteration: candidate_set, kg_pairs, selected point, "
            "observation, pre-update posterior mean/variance, timing, and HV "
            "snapshots when evaluated."
        )
        return result
