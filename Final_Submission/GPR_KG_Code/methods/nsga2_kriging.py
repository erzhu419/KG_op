"""
NSGA-II-K: NSGA-II with Kriging (GP) surrogates.

Three GP surrogates (one per output) are fitted to the observed data.
NSGA-II runs on the f1/f2 surrogates for offspring generation; the
constraint surrogate (f3 GP) provides a probabilistic constraint violation
used inside the constrained EA.

Constraint handling in the surrogate EA follows:
  Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
  "A fast and elitist multiobjective genetic algorithm: NSGA-II."
  IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
  https://doi.org/10.1109/4235.996017

Probabilistic constraint violation in the surrogate space:
  CV(x) = max(0, mu3(x) + z_{1-alpha} * sigma3(x) - tau)
which mirrors the probabilistic constraint used by GPR-KG.

GP surrogate (SimpleKriging) uses a squared-exponential (RBF) kernel.
predict_with_std returns the posterior predictive standard deviation,
enabling uncertainty-aware constraint handling.
"""

import numpy as np
import time
import sys
import os
import json
from scipy.spatial.distance import cdist
from scipy.stats import norm as sp_norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpr_kg import pareto_filter, compute_hypervolume_2d
from methods.base_method import BaseMethod
from methods.nsga2_utils import (non_dominated_sort, crowding_distance,
                                  constrained_nsga2_one_generation,
                                  constrained_environmental_selection)
from experiments.config import HV_EVAL_INTERVAL, NSGA2_PARAMS


class SimpleKriging:
    """Simple Kriging (GP) surrogate with squared-exponential kernel.

    Lightweight implementation for NSGA-II-K.  Fits a GP with constant mean
    and RBF kernel to observed data; predicts both mean and posterior
    predictive standard deviation at new points.

    Reference for GP regression:
      Rasmussen, C. E., & Williams, C. K. I. (2006).
      "Gaussian Processes for Machine Learning." MIT Press.
      http://www.gaussianprocess.org/gpml/
    """

    def __init__(self, length_scale=5.0, noise_var=0.01):
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.X_train = None
        self.y_train = None
        self.K_inv_y = None
        self._L = None   # Cholesky factor stored for predict_with_std
        self._K = None

    def fit(self, X, y):
        """Fit GP to training data.

        Args:
            X: np.array (n, d) training inputs.
            y: np.array (n,) training outputs.
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        n = len(X)

        K = self._kernel(self.X_train, self.X_train)
        K += (self.noise_var + 1e-6) * np.eye(n)
        self._K = K

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            self._L = L
        except np.linalg.LinAlgError:
            alpha = np.linalg.solve(K + 1e-4 * np.eye(n), self.y_train)
            self._L = None

        self.K_inv_y = alpha

    def predict(self, X_new):
        """Predict posterior mean at new points.

        Args:
            X_new: np.array (m, d).

        Returns:
            np.array (m,) predicted means.
        """
        if self.X_train is None:
            return np.zeros(len(X_new))

        K_star = self._kernel(np.array(X_new, dtype=float), self.X_train)
        return K_star @ self.K_inv_y

    def predict_with_std(self, X_new):
        """Predict posterior mean and predictive standard deviation.

        Posterior variance:  k(x*,x*) - k_*^T (K + sigma^2 I)^{-1} k_*
        For RBF kernel: k(x,x) = 1.0 (prior variance = 1).

        Args:
            X_new: np.array (m, d).

        Returns:
            mu: np.array (m,) posterior means.
            sigma: np.array (m,) posterior standard deviations (>= 1e-4).
        """
        if self.X_train is None:
            n = len(X_new)
            return np.zeros(n), np.ones(n)

        X_new_arr = np.array(X_new, dtype=float)
        K_star = self._kernel(X_new_arr, self.X_train)  # (m, n_train)
        mu = K_star @ self.K_inv_y

        # Prior variance for RBF kernel is k(x,x) = 1.0
        prior_var = 1.0
        if self._L is not None:
            # v = L^{-1} k_star^T, shape (n_train, m)
            v = np.linalg.solve(self._L, K_star.T)
            var = np.maximum(prior_var - np.sum(v ** 2, axis=0), 1e-8)
        else:
            var = np.ones(len(X_new_arr)) * prior_var

        return mu, np.sqrt(var)

    def _kernel(self, X1, X2):
        """Squared-exponential (RBF) kernel with unit signal variance."""
        dists = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-0.5 * dists / (self.length_scale ** 2))


class NSGA2Kriging(BaseMethod):
    """NSGA-II with Kriging surrogates (NSGA-II-K in the paper).

    At each iteration:
      1. Fit GPs for f1, f2 (objectives) and f3 (constraint).
      2. Run constrained NSGA-II (Deb et al. 2002) on the surrogates.
         Constraint violation in the surrogate EA:
           CV(x) = max(0, mu3(x) + z_{1-alpha}*sigma3(x) - tau)
         This is the same probabilistic formulation used in GPR-KG.
      3. Pick the best unvisited point from the EA population and simulate it.
    """

    name = "NSGA-II-K"

    def run(self, problem, N, n0, seed, hv_eval_interval=HV_EVAL_INTERVAL,
            initial_samples=None, snapshot_path=None):
        ref = problem.ref_point
        if seed is not None:
            np.random.seed(seed)

        t_start = time.time()
        d = problem.d
        lo, hi = problem.int_bounds()
        L = int(hi[0])
        pop_size = min(NSGA2_PARAMS['pop_size'], 30)  # Reduced for tractability
        n_ea_gens = 10   # EA generations on surrogate per simulation step
        q_alpha = sp_norm.ppf(1.0 - problem.alpha)  # z_{1-alpha}

        X_data = []
        Y_data = []
        hv_history = []
        time_per_iter = []
        iteration_log = []

        if snapshot_path and os.path.exists(snapshot_path):
            os.remove(snapshot_path)

        # --- Phase 1: Pre-sampling ---
        if initial_samples is not None:
            pre_samples = [tuple(int(v) for v in x)
                           for x in initial_samples[:n0]]
        else:
            pre_samples = []
            seen = set()
            while len(pre_samples) < n0:
                x = problem.sample_random()
                if x not in seen:
                    pre_samples.append(x)
                    seen.add(x)

        for k, x_tuple in enumerate(pre_samples):
            x_arr = np.array(x_tuple)
            Y = problem.simulate(x_arr)
            X_data.append(x_arr)
            Y_data.append(Y)
            self._append_snapshot(snapshot_path, {
                "stage": "initial",
                "index": int(k),
                "x": [int(v) for v in x_tuple],
                "Y": [float(v) for v in Y],
            })

        n_sims = n0

        # --- Phase 2: Main loop ---
        for n in range(n0, N):
            t_iter = time.time()

            X_arr = np.array(X_data)
            Y_arr = np.array(Y_data)

            # Fit 3 GP surrogates: f1, f2 (objectives) + f3 (constraint)
            surrogates = []
            for i in range(3):
                gp = SimpleKriging(length_scale=max(3.0, d * 0.5))
                gp.fit(X_arr, Y_arr[:, i])
                surrogates.append(gp)

            # Initialize population (include some observed points)
            pop = np.random.randint(1, L + 1, size=(pop_size, d))
            n_include = min(len(X_data), pop_size // 2)
            if n_include > 0:
                idx = np.random.choice(len(X_data), n_include, replace=False)
                pop[:n_include] = X_arr[idx].astype(int)

            # Evaluate initial population on surrogates
            pop_obj = np.column_stack([
                surrogates[0].predict(pop),
                surrogates[1].predict(pop),
            ])
            mu3, sigma3 = surrogates[2].predict_with_std(pop)
            # Probabilistic constraint violation (mirrors GPR-KG formulation)
            pop_cv = np.maximum(mu3 + q_alpha * sigma3 - problem.tau, 0.0)

            # Run constrained EA generations on surrogates (Deb et al. 2002)
            for _ in range(n_ea_gens):
                offspring = constrained_nsga2_one_generation(
                    pop, pop_obj, pop_cv, 1, L, pop_size,
                    crossover_eta=NSGA2_PARAMS['crossover_eta'],
                    mutation_eta=NSGA2_PARAMS['mutation_eta'],
                    crossover_prob=NSGA2_PARAMS['crossover_prob'],
                )
                off_obj = np.column_stack([
                    surrogates[0].predict(offspring),
                    surrogates[1].predict(offspring),
                ])
                mu3_off, sigma3_off = surrogates[2].predict_with_std(offspring)
                off_cv = np.maximum(mu3_off + q_alpha * sigma3_off - problem.tau, 0.0)

                combined_pop = np.vstack([pop, offspring])
                combined_obj = np.vstack([pop_obj, off_obj])
                combined_cv = np.concatenate([pop_cv, off_cv])
                pop, pop_obj, pop_cv = constrained_environmental_selection(
                    combined_pop, combined_obj, combined_cv, pop_size)

            # Select best unvisited solution from EA population
            # Rank by constrained front then crowding distance
            fronts = non_dominated_sort(pop_obj)   # use plain sort for ranking here
            cd = {}
            for front in fronts:
                cd.update(crowding_distance(pop_obj, front))

            ranked = []
            for r, front in enumerate(fronts):
                for idx in front:
                    ranked.append((pop_cv[idx], r, -cd.get(idx, 0), idx))
            ranked.sort()  # feasible (cv=0) first, then by rank, then crowding

            x_selected = None
            sampled_set = set(tuple(x) for x in X_data)
            for _, _, _, idx in ranked:
                x_cand = tuple(pop[idx])
                if x_cand not in sampled_set:
                    x_selected = pop[idx]
                    break

            if x_selected is None:
                x_selected = np.array(list(problem.sample_random()))

            # Simulate selected solution
            Y = problem.simulate(x_selected)
            X_data.append(x_selected.copy())
            Y_data.append(Y)
            n_sims += 1

            t_compute = time.time() - t_iter
            time_per_iter.append(t_compute)

            # HV evaluation
            if (n - n0) % hv_eval_interval == 0 or n == N - 1:
                pf_objs, pf_sols = self._current_pareto(X_data, Y_data, problem)
                hv = compute_hypervolume_2d(pf_objs, ref) \
                    if len(pf_objs) > 0 else 0.0
                hv_history.append((n, float(hv)))
            else:
                hv = None

            log = {
                "stage": "adaptive",
                "iteration": int(n),
                "x": [int(v) for v in x_selected],
                "Y": [float(v) for v in Y],
                "time_sec": float(t_compute),
                "pop_size": int(pop_size),
                "n_ea_gens": int(n_ea_gens),
                "hv": None if hv is None else float(hv),
            }
            iteration_log.append(log)
            self._append_snapshot(snapshot_path, log)

        total_time = time.time() - t_start
        pf_objs, pf_sols = self._current_pareto(X_data, Y_data, problem)

        result = self._make_result(
            pareto_solutions=pf_sols,
            problem=problem,
            hv_history=hv_history,
            total_time=total_time,
            time_per_iter=time_per_iter,
            n_simulations=n_sims,
            ref_point=ref,
        )
        result["initial_samples"] = [[int(v) for v in x] for x in pre_samples]
        result["observation_history"] = [
            {"x": [int(v) for v in x], "Y": [float(y) for y in Y]}
            for x, Y in zip(X_data, Y_data)
        ]
        result["iteration_log"] = iteration_log
        result["surrogate_ea_pop_size"] = int(pop_size)
        result["surrogate_ea_generations"] = int(n_ea_gens)
        return result

    def _append_snapshot(self, snapshot_path, payload):
        if not snapshot_path:
            return
        with open(snapshot_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _current_pareto(self, X_data, Y_data, problem):
        """Extract feasible Pareto front from simulated solutions.

        Feasibility is determined by the observed (noisy) constraint
        Y[2] <= tau, consistent with direct NSGA-II having no probabilistic
        constraint model for the final output.
        """
        feasible_objs = []
        feasible_sols = []

        for x_arr, Y in zip(X_data, Y_data):
            x_tuple = tuple(x_arr)
            if Y[2] <= problem.tau:
                feasible_objs.append([Y[0], Y[1]])
                feasible_sols.append(x_tuple)

        if len(feasible_objs) == 0:
            return np.empty((0, 2)), []

        objs = np.array(feasible_objs)
        pf, pf_idx = pareto_filter(objs, return_indices=True)
        return pf, [feasible_sols[i] for i in pf_idx]
