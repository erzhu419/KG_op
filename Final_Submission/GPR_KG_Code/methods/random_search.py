"""
RS (Random Search) baseline method.

Uniformly samples N solutions from the discrete space {1,...,L}^d,
simulates each once, and returns the observed Pareto front of feasible solutions.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpr_kg import pareto_filter, compute_hypervolume_2d
from methods.base_method import BaseMethod
from experiments.config import HV_EVAL_INTERVAL


class RandomSearch(BaseMethod):
    """Random Search baseline (RS in the paper)."""

    name = "RS"

    def run(self, problem, N, n0, seed, hv_eval_interval=HV_EVAL_INTERVAL):
        if seed is not None:
            np.random.seed(seed)

        t_start = time.time()
        d, L = problem.d, problem.L
        tau = problem.tau

        ref = problem.ref_point
        sampled = {}       # x_tuple -> [Y1, Y2, Y3]
        hv_history = []
        time_per_iter = []

        for n in range(N):
            t0 = time.time()
            x = problem.sample_random()
            Y = problem.simulate(np.array(x))
            sampled[x] = Y
            time_per_iter.append(time.time() - t0)

            # HV on TRUE objectives at intervals (consistent with other methods)
            if (n + 1) % hv_eval_interval == 0 or n == N - 1:
                _, pf_sols = self._current_pareto(sampled, problem)
                if pf_sols:
                    true_objs = np.array([problem.true_objectives(x)[:2]
                                          for x in pf_sols])
                    true_objs_pf = pareto_filter(true_objs)
                    hv = compute_hypervolume_2d(true_objs_pf, ref)
                else:
                    hv = 0.0
                hv_history.append((n + 1, float(hv)))

        total_time = time.time() - t_start

        # Final Pareto front from observed values
        pf_objs, pf_sols = self._current_pareto(sampled, problem)

        return self._make_result(
            pareto_solutions=pf_sols,
            problem=problem,
            hv_history=hv_history,
            total_time=total_time,
            time_per_iter=time_per_iter,
            n_simulations=N,
            ref_point=ref,
        )

    def _current_pareto(self, sampled, problem):
        """Extract current Pareto front from ESTIMATED feasible solutions.

        Uses observed Y[2] (noisy constraint) for feasibility check,
        NOT the true feasibility. This allows CVR to capture methods'
        ability to correctly identify feasible solutions.
        """
        feasible_objs = []
        feasible_sols = []

        for x_tuple, Y in sampled.items():
            # Estimated feasibility: Y[2] <= tau (observed noisy constraint)
            if Y[2] <= problem.tau:
                feasible_objs.append([Y[0], Y[1]])
                feasible_sols.append(x_tuple)

        if len(feasible_objs) == 0:
            return np.empty((0, 2)), []

        objs = np.array(feasible_objs)
        pf, pf_idx = pareto_filter(objs, return_indices=True)
        return pf, [feasible_sols[i] for i in pf_idx]
