"""
cParEGO: Constrained ParEGO.

Combines random augmented Chebyshev scalarization with constrained Expected
Improvement.  Each iteration:
  1. Draw a weight vector w uniformly from the unit simplex (Dirichlet(1,1)).
  2. Scalarize all observations: s(x) = max_i(w_i*f_i(x)) + rho*sum_i(w_i*f_i(x))
  3. Fit a GP on the scalarized values (scalar surrogate) and a second GP on
     the constraint output f3.
  4. Select the next point by maximising the Constrained Expected Improvement:
       cEI(x) = EI(x) * PoF(x)
     where
       EI(x)  = (s_best - mu_s(x)) * Phi(z) + sigma_s(x) * phi(z),
                z = (s_best - mu_s(x)) / sigma_s(x)    [minimisation]
       PoF(x) = Phi((tau - mu_c(x)) / sigma_c(x))
                probability of satisfying the constraint

References
----------
Knowles, J. (2006).
  "ParEGO: A Hybrid Algorithm with On-Line Landscape Approximation for
  Expensive Multiobjective Optimization Problems."
  IEEE Transactions on Evolutionary Computation, 10(1), 50-66.
  https://doi.org/10.1109/TEVC.2005.851274

Gelbart, M. A., Snoek, J., & Adams, R. P. (2014).
  "Bayesian Optimization with Unknown Constraints."
  UAI 2014. arXiv:1403.5607  https://arxiv.org/abs/1403.5607
  (Section 3 defines cEI = EI * PoF for independent constraints)

BoTorch qParEGO reference implementation:
  https://github.com/pytorch/botorch
  botorch.acquisition.multi_objective.parego
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


class cParEGOMethod(BaseMethod):
    """Constrained ParEGO (cParEGO in the paper).

    Parameters
    ----------
    rho : float
        Augmentation coefficient for Chebyshev scalarization (Knowles 2006
        recommends rho in [1e-6, 0.05]; default 0.05).
    """

    name = "cParEGO"

    def __init__(self, rho=0.05):
        # rho = 0.05 follows Knowles (2006) recommendation
        self.rho = rho

    def run(self, problem, N, n0, seed, hv_eval_interval=HV_EVAL_INTERVAL,
            initial_samples=None, snapshot_path=None):
        ref = problem.ref_point
        if seed is not None:
            np.random.seed(seed)
            if HAS_BOTORCH:
                torch.manual_seed(seed)

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

        # --- Pre-sampling ---
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
            X_data.append(np.array(x_tuple, dtype=float))
            Y_data.append(Y)
            self._append_snapshot(snapshot_path, {
                "stage": "initial",
                "index": int(k),
                "x": [int(v) for v in x_tuple],
                "Y": [float(v) for v in Y],
            })

        # --- Main loop ---
        for n in range(n0, N):
            t_iter = time.time()

            X_arr = np.array(X_data)
            Y_arr = np.array(Y_data)

            # Draw weight from uniform distribution on the unit simplex
            # (Knowles 2006; BoTorch qParEGO uses the same Dirichlet(1,...,1) draw)
            w = np.random.dirichlet([1.0, 1.0])

            # Augmented Chebyshev scalarization (Knowles 2006, Eq. 1)
            #   s(x) = max_i(w_i * f_i(x)) + rho * sum_i(w_i * f_i(x))
            s_values = self._chebyshev_scalarize(Y_arr[:, :2], w)

            # Generate discrete candidates
            n_cand = 200
            X_cand = np.array([list(problem.sample_random()) for _ in range(n_cand)], dtype=float)

            x_next = self._select_cei(X_arr, s_values, Y_arr[:, 2],
                                      X_cand, problem)

            # Simulate
            Y = problem.simulate(np.array(x_next))
            X_data.append(np.array(x_next, dtype=float))
            Y_data.append(Y)

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
                "x": [int(v) for v in x_next],
                "Y": [float(v) for v in Y],
                "time_sec": float(t_compute),
                "n_candidates": int(n_cand),
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
            n_simulations=N,
            ref_point=ref,
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
    # Core acquisition
    # ------------------------------------------------------------------

    def _chebyshev_scalarize(self, obj_values, w):
        """Augmented Chebyshev scalarization (minimisation).

        s(x) = max_i(w_i * f_i(x)) + rho * sum_i(w_i * f_i(x))

        Reference: Knowles (2006), Eq. (1).
        """
        weighted = obj_values * w[np.newaxis, :]
        return np.max(weighted, axis=1) + self.rho * np.sum(weighted, axis=1)

    def _select_cei(self, X_arr, s_values, f3_values, X_cand, problem):
        """Select next point by maximising cEI = EI * PoF (Gelbart et al. 2014).

        Both EI and PoF are computed from GP posterior mean and std.

        EI(x) = (s_best - mu_s(x)) * Phi(z) + sigma_s(x) * phi(z)
                z = (s_best - mu_s(x)) / sigma_s(x)      [minimisation EI]

        PoF(x) = Phi((tau - mu_c(x)) / sigma_c(x))
                probability that f3(x) <= tau given the constraint GP.

        References: Gelbart et al. (2014) UAI, arXiv:1403.5607.
        """
        from methods.nsga2_kriging import SimpleKriging

        d = X_arr.shape[1]

        # GP on scalarized objective
        gp_s = SimpleKriging(length_scale=max(3.0, d * 0.5))
        gp_s.fit(X_arr, s_values)

        # GP on constraint output f3
        gp_c = SimpleKriging(length_scale=max(3.0, d * 0.5))
        gp_c.fit(X_arr, f3_values)

        mu_s, std_s = gp_s.predict_with_std(X_cand)
        mu_c, std_c = gp_c.predict_with_std(X_cand)

        # Expected Improvement on scalarized objective (minimisation)
        # If no feasible point has been found yet, fall back to PoF-only
        feas_obs = f3_values <= problem.tau
        if feas_obs.any():
            # s_best: best scalarized value among feasible observations
            s_best = np.min(s_values[feas_obs])
            ei = self._expected_improvement(mu_s, std_s, s_best)
        else:
            # No feasible point yet: use EI over all observations (Gelbart 2014)
            s_best = np.min(s_values)
            ei = self._expected_improvement(mu_s, std_s, s_best)

        # PoF(x) = Phi((tau - mu_c) / sigma_c)  [Gelbart et al. 2014]
        pof = sp_norm.cdf((problem.tau - mu_c) / np.maximum(std_c, 1e-8))

        # cEI = EI * PoF  (Gelbart et al. 2014, Eq. 4)
        scores = ei * pof

        best_idx = int(np.argmax(scores))
        return tuple(int(v) for v in X_cand[best_idx])

    def _expected_improvement(self, mu, sigma, f_best):
        """Expected Improvement for minimisation.

        EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
                z = (f_best - mu) / sigma

        Reference: Jones, D. R., Schonlau, M., & Welch, W. J. (1998).
          "Efficient Global Optimization of Expensive Black-Box Functions."
          Journal of Global Optimization, 13(4), 455-492.
          https://doi.org/10.1023/A:1008306431147
        """
        sigma = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma
        return (f_best - mu) * sp_norm.cdf(z) + sigma * sp_norm.pdf(z)

    # ------------------------------------------------------------------
    # BoTorch-based path (retained for reference)
    # ------------------------------------------------------------------

    def _select_botorch(self, X_arr, s_values, f3_values, X_cand, problem):
        """Select next point using BoTorch GP + constrained EI (Gelbart 2014)."""
        try:
            d = X_arr.shape[1]
            L_max = int(X_cand.max())
            bounds = torch.tensor([[1.0] * d, [float(L_max)] * d],
                                   dtype=torch.float64)
            X_tensor = torch.tensor(X_arr, dtype=torch.float64)
            X_norm = normalize(X_tensor, bounds)
            X_cand_tensor = torch.tensor(X_cand, dtype=torch.float64)
            X_cand_norm = normalize(X_cand_tensor, bounds)

            s_tensor = torch.tensor(s_values, dtype=torch.float64).unsqueeze(-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_s = SingleTaskGP(X_norm, s_tensor,
                                       outcome_transform=Standardize(m=1))
                mll_s = ExactMarginalLogLikelihood(model_s.likelihood, model_s)
                fit_gpytorch_mll(mll_s)

            f3_tensor = torch.tensor(f3_values, dtype=torch.float64).unsqueeze(-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_c = SingleTaskGP(X_norm, f3_tensor,
                                       outcome_transform=Standardize(m=1))
                mll_c = ExactMarginalLogLikelihood(model_c.likelihood, model_c)
                fit_gpytorch_mll(mll_c)

            with torch.no_grad():
                pred_s = model_s.posterior(X_cand_norm)
                pred_c = model_c.posterior(X_cand_norm)
                mu_s = pred_s.mean.squeeze().numpy()
                std_s = pred_s.variance.squeeze().sqrt().numpy()
                mu_c = pred_c.mean.squeeze().numpy()
                std_c = pred_c.variance.squeeze().sqrt().numpy()

            feas_obs = f3_values <= problem.tau
            s_best = np.min(s_values[feas_obs]) if feas_obs.any() \
                else np.min(s_values)
            ei = self._expected_improvement(mu_s, std_s, s_best)
            # PoF via Normal CDF (Gelbart et al. 2014)
            pof = sp_norm.cdf((problem.tau - mu_c) / np.maximum(std_c, 1e-8))
            scores = ei * pof
            best_idx = int(np.argmax(scores))
            return tuple(int(v) for v in X_cand[best_idx])

        except Exception:
            return self._select_cei(X_arr, s_values, f3_values, X_cand, problem)

    # ------------------------------------------------------------------
    # Pareto front helper
    # ------------------------------------------------------------------

    def _current_pareto(self, X_data, Y_data, problem):
        """Extract feasible Pareto front from observed data."""
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
