"""
cEHVI: Constrained Expected Hypervolume Improvement.

Uses GP surrogate models (Matern-5/2 via BoTorch when available; RBF via
SimpleKriging as fallback) for all three outputs.  At each iteration the
acquisition score is:

    score(x) = E[HVI(x)] * PoF(x)

where
  * E[HVI(x)] = Monte-Carlo expected hypervolume improvement, estimated by
    sampling (f1*, f2*) from the GP posterior at x (Daulton et al. 2020).
  * PoF(x) = P(f3(x) <= tau) = Phi((tau - mu3(x)) / sigma3(x))
    is the probability of feasibility obtained from the constraint GP
    (Gelbart et al. 2014).

References
----------
Daulton, S., Balandat, M., & Bakshy, E. (2020).
  "Differentiable Expected Hypervolume Improvement for Parallel
  Multi-Objective Bayesian Optimization." NeurIPS 2020.
  https://arxiv.org/abs/2006.05078

Gelbart, M. A., Snoek, J., & Adams, R. P. (2014).
  "Bayesian Optimization with Unknown Constraints." UAI 2014.
  arXiv:1403.5607  https://arxiv.org/abs/1403.5607

BoTorch open-source implementation:
  https://github.com/pytorch/botorch
  botorch.acquisition.multi_objective.logei (qLogNoisyExpectedHypervolume...)
"""

import numpy as np
import time
import sys
import os
import json
import warnings
from scipy.stats import norm as sp_norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpr_kg import pareto_filter, compute_hypervolume_2d
from methods.base_method import BaseMethod
from experiments.config import HV_EVAL_INTERVAL

try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.utils.transforms import normalize
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False


class cEHVIMethod(BaseMethod):
    """Constrained Expected Hypervolume Improvement (cEHVI).

    The main entry point always uses the lightweight GP implementation for
    tractable 30-repetition experiments.  The BoTorch path (_run_botorch) is
    retained for reference and spot-checking.
    """

    name = "cEHVI"

    def run(self, problem, N, n0, seed, hv_eval_interval=HV_EVAL_INTERVAL,
            initial_samples=None, snapshot_path=None):
        self._ref = problem.ref_point
        return self._run_fallback(problem, N, n0, seed, hv_eval_interval,
                                  initial_samples=initial_samples,
                                  snapshot_path=snapshot_path)

    # ------------------------------------------------------------------
    # BoTorch-based path (reference implementation)
    # ------------------------------------------------------------------

    def _run_botorch(self, problem, N, n0, seed, hv_eval_interval):
        """BoTorch-based implementation."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        t_start = time.time()
        d = problem.d
        lo, hi = problem.int_bounds()
        L = int(hi[0])  # for backward compat
        bounds = torch.tensor([lo.tolist(), hi.tolist()], dtype=torch.float64)

        X_data = []
        Y_data = []
        hv_history = []
        time_per_iter = []

        # Pre-sampling
        pre_samples = set()
        while len(pre_samples) < n0:
            pre_samples.add(problem.sample_random())

        for x_tuple in pre_samples:
            x_arr = np.array(x_tuple)
            Y = problem.simulate(x_arr)
            X_data.append(list(x_tuple))
            Y_data.append(Y.tolist())

        # Main loop
        for n in range(n0, N):
            t_iter = time.time()

            try:
                x_next = self._select_next_botorch(
                    X_data, Y_data, d, L, bounds, problem)
            except Exception:
                x_next = problem.sample_random()

            x_arr = np.array(x_next)
            Y = problem.simulate(x_arr)
            X_data.append(list(x_next))
            Y_data.append(Y.tolist())

            t_compute = time.time() - t_iter
            time_per_iter.append(t_compute)

            if (n - n0) % hv_eval_interval == 0 or n == N - 1:
                pf_objs, pf_sols = self._current_pareto_list(X_data, Y_data, problem)
                hv = compute_hypervolume_2d(pf_objs, self._ref) \
                    if len(pf_objs) > 0 else 0.0
                hv_history.append((n, float(hv)))

        total_time = time.time() - t_start
        pf_objs, pf_sols = self._current_pareto_list(X_data, Y_data, problem)

        return self._make_result(
            pareto_solutions=pf_sols,
            problem=problem,
            hv_history=hv_history,
            total_time=total_time,
            time_per_iter=time_per_iter,
            n_simulations=N,
            ref_point=self._ref,
        )

    def _select_next_botorch(self, X_data, Y_data, d, L, bounds, problem):
        """Select next point using BoTorch GP + MC-EHVI with proper PoF.

        Probability of feasibility follows Gelbart et al. (2014):
            PoF(x) = Phi((tau - mu3(x)) / sigma3(x))
        using the Normal CDF (not a sigmoid approximation).
        """
        X_tensor = torch.tensor(X_data, dtype=torch.float64)
        Y_arr = np.array(Y_data)
        X_norm = normalize(X_tensor, bounds)

        models = []
        for i in range(3):
            y_i = torch.tensor(Y_arr[:, i], dtype=torch.float64).unsqueeze(-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SingleTaskGP(X_norm, y_i, outcome_transform=Standardize(m=1))
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
            models.append(model)

        n_candidates = 100
        X_cand_np = np.array([problem.sample_random() for _ in range(n_candidates)], dtype=float)
        X_cand = torch.tensor(X_cand_np, dtype=torch.float64)
        X_cand_norm = normalize(X_cand, bounds)

        with torch.no_grad():
            pred_f1 = models[0].posterior(X_cand_norm)
            pred_f2 = models[1].posterior(X_cand_norm)
            pred_f3 = models[2].posterior(X_cand_norm)

            mu1 = pred_f1.mean.squeeze().numpy()
            mu2 = pred_f2.mean.squeeze().numpy()
            mu3 = pred_f3.mean.squeeze().numpy()
            std1 = pred_f1.variance.squeeze().sqrt().numpy()
            std2 = pred_f2.variance.squeeze().sqrt().numpy()
            std3 = pred_f3.variance.squeeze().sqrt().numpy()

        # PoF = Phi((tau - mu3) / sigma3)  [Gelbart et al. 2014]
        pof = sp_norm.cdf((problem.tau - mu3) / np.maximum(std3, 1e-8))

        # Current Pareto front from feasible observations
        Y_arr_np = np.array(Y_data)
        feas_mask = Y_arr_np[:, 2] <= problem.tau
        if feas_mask.any():
            current_pf = pareto_filter(Y_arr_np[feas_mask][:, :2])
            hv_current = compute_hypervolume_2d(current_pf, self._ref.astype(float))
        else:
            current_pf = np.empty((0, 2))
            hv_current = 0.0

        # MC-EHVI: sample (f1*, f2*) from GP posterior (Daulton et al. 2020)
        n_mc = 32
        scores = np.zeros(n_candidates)
        for ci in range(n_candidates):
            if pof[ci] < 0.01:
                continue
            f1_s = np.random.normal(mu1[ci], std1[ci], n_mc)
            f2_s = np.random.normal(mu2[ci], std2[ci], n_mc)
            total_impr = 0.0
            for s in range(n_mc):
                new_pt = np.array([[f1_s[s], f2_s[s]]])
                combined = np.vstack([current_pf, new_pt]) \
                    if len(current_pf) > 0 else new_pt
                hv_new = compute_hypervolume_2d(
                    pareto_filter(combined), self._ref.astype(float))
                total_impr += max(0.0, hv_new - hv_current)
            scores[ci] = (total_impr / n_mc) * pof[ci]

        best_idx = int(np.argmax(scores))
        return tuple(int(v) for v in X_cand[best_idx])

    # ------------------------------------------------------------------
    # Lightweight fallback (used by default)
    # ------------------------------------------------------------------

    def _run_fallback(self, problem, N, n0, seed, hv_eval_interval,
                      initial_samples=None, snapshot_path=None):
        """SimpleKriging-based MC-EHVI with proper PoF (Gelbart et al. 2014).

        Acquisition:
            score(x) = E[HVI(x | current_PF)] * PoF(x)

        PoF(x) = Phi((tau - mu3(x)) / sigma3(x))
            Probability of feasibility from the constraint GP.
            (Gelbart et al. 2014, arXiv:1403.5607)

        E[HVI(x)] estimated by Monte-Carlo sampling from the GP posterior of
        f1 and f2 at x.  (Daulton et al. 2020, NeurIPS)
        """
        from methods.nsga2_kriging import SimpleKriging

        if seed is not None:
            np.random.seed(seed)

        t_start = time.time()
        d = problem.d
        lo, hi = problem.int_bounds()
        L = int(hi[0])

        X_data = []
        Y_data = []
        hv_history = []
        time_per_iter = []
        iteration_log = []

        if snapshot_path and os.path.exists(snapshot_path):
            os.remove(snapshot_path)

        # Pre-sampling
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
            Y = problem.simulate(np.array(x_tuple))
            X_data.append(np.array(x_tuple))
            Y_data.append(Y)
            self._append_snapshot(snapshot_path, {
                "stage": "initial",
                "index": int(k),
                "x": [int(v) for v in x_tuple],
                "Y": [float(v) for v in Y],
            })

        # Main loop
        for n in range(n0, N):
            t_iter = time.time()

            X_arr = np.array(X_data)
            Y_arr = np.array(Y_data)

            # Fit three GPs (f1, f2 objectives + f3 constraint)
            surrogates = []
            for i in range(3):
                gp = SimpleKriging(length_scale=max(3.0, d * 0.5))
                gp.fit(X_arr, Y_arr[:, i])
                surrogates.append(gp)

            # Generate discrete candidates
            n_cand = 200
            X_cand = np.array([problem.sample_random() for _ in range(n_cand)])

            # GP posterior at candidates
            mu1, std1 = surrogates[0].predict_with_std(X_cand)
            mu2, std2 = surrogates[1].predict_with_std(X_cand)
            mu3, std3 = surrogates[2].predict_with_std(X_cand)

            # PoF(x) = Phi((tau - mu3) / sigma3)  [Gelbart et al. 2014]
            pof = sp_norm.cdf((problem.tau - mu3) / np.maximum(std3, 1e-8))

            # Current Pareto front from feasible observed data
            feas_mask = Y_arr[:, 2] <= problem.tau
            if feas_mask.any():
                current_pf = pareto_filter(Y_arr[feas_mask][:, :2])
                hv_current = compute_hypervolume_2d(current_pf, self._ref)
            else:
                current_pf = np.empty((0, 2))
                hv_current = 0.0

            # MC-EHVI: sample GP posterior of f1, f2 to estimate E[HVI]
            # (Daulton et al. 2020 — MC approximation of EHVI)
            n_mc = 32
            scores = np.zeros(n_cand)
            for ci in range(n_cand):
                if pof[ci] < 0.01:
                    continue
                f1_s = np.random.normal(mu1[ci], std1[ci], n_mc)
                f2_s = np.random.normal(mu2[ci], std2[ci], n_mc)
                total_impr = 0.0
                for s in range(n_mc):
                    new_pt = np.array([[f1_s[s], f2_s[s]]])
                    combined = np.vstack([current_pf, new_pt]) \
                        if len(current_pf) > 0 else new_pt
                    hv_new = compute_hypervolume_2d(
                        pareto_filter(combined), self._ref)
                    total_impr += max(0.0, hv_new - hv_current)
                scores[ci] = (total_impr / n_mc) * pof[ci]

            best_idx = int(np.argmax(scores))
            x_next = tuple(X_cand[best_idx])

            Y = problem.simulate(np.array(x_next))
            X_data.append(np.array(x_next))
            Y_data.append(Y)

            t_compute = time.time() - t_iter
            time_per_iter.append(t_compute)

            if (n - n0) % hv_eval_interval == 0 or n == N - 1:
                pf_objs, pf_sols = self._current_pareto_np(X_data, Y_data, problem)
                hv = compute_hypervolume_2d(pf_objs, self._ref) \
                    if len(pf_objs) > 0 else 0.0
                hv_history.append((n, float(hv)))
            else:
                hv = None

            log = {
                "stage": "adaptive",
                "iteration": int(n),
                "x": [int(v) for v in x_next],
                "Y": [float(v) for v in Y],
                "time_sec": float(t_compute),
                "n_candidates": int(n_cand),
                "best_score": float(scores[best_idx]),
                "hv": None if hv is None else float(hv),
            }
            iteration_log.append(log)
            self._append_snapshot(snapshot_path, log)

        total_time = time.time() - t_start
        pf_objs, pf_sols = self._current_pareto_np(X_data, Y_data, problem)

        result = self._make_result(
            pareto_solutions=pf_sols,
            problem=problem,
            hv_history=hv_history,
            total_time=total_time,
            time_per_iter=time_per_iter,
            n_simulations=N,
            ref_point=self._ref,
        )
        result["initial_samples"] = [[int(v) for v in x] for x in pre_samples]
        result["observation_history"] = [
            {"x": [int(v) for v in x], "Y": [float(y) for y in Y]}
            for x, Y in zip(X_data, Y_data)
        ]
        result["iteration_log"] = iteration_log
        return result

    def _append_snapshot(self, snapshot_path, payload):
        if not snapshot_path:
            return
        with open(snapshot_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _current_pareto_list(self, X_data, Y_data, problem):
        """Extract feasible Pareto front from list-format data."""
        feasible_objs, feasible_sols = [], []
        for x_list, Y_list in zip(X_data, Y_data):
            if Y_list[2] <= problem.tau:
                feasible_objs.append([Y_list[0], Y_list[1]])
                feasible_sols.append(tuple(int(v) for v in x_list))
        if not feasible_objs:
            return np.empty((0, 2)), []
        objs = np.array(feasible_objs)
        pf, pf_idx = pareto_filter(objs, return_indices=True)
        return pf, [feasible_sols[i] for i in pf_idx]

    def _current_pareto_np(self, X_data, Y_data, problem):
        """Extract feasible Pareto front from numpy-format data."""
        feasible_objs, feasible_sols = [], []
        for x_arr, Y in zip(X_data, Y_data):
            if Y[2] <= problem.tau:
                feasible_objs.append([Y[0], Y[1]])
                feasible_sols.append(tuple(int(v) for v in x_arr))
        if not feasible_objs:
            return np.empty((0, 2)), []
        objs = np.array(feasible_objs)
        pf, pf_idx = pareto_filter(objs, return_indices=True)
        return pf, [feasible_sols[i] for i in pf_idx]
