"""
Base class defining the uniform interface for all comparison methods.

Every method must implement run() and return a standardized result dict.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpr_kg import pareto_filter, compute_hypervolume_2d
from metrics import compute_igd, compute_cvr


class BaseMethod(ABC):
    """Abstract base class for all comparison methods."""

    name = "BaseMethod"  # Override in subclass

    @abstractmethod
    def run(self, problem, N, n0, seed, hv_eval_interval=10):
        """Run the optimization method.

        Args:
            problem: TestProblem instance (already calibrated with tau set).
            N: Total simulation budget.
            n0: Pre-sampling budget.
            seed: Random seed for reproducibility.
            hv_eval_interval: Compute HV every k iterations for convergence.

        Returns:
            dict with standardized keys (see _make_result).
        """
        pass

    def _make_result(self, pareto_solutions, problem, hv_history,
                     total_time, time_per_iter, n_simulations, ref_point):
        """Create standardized result dict.

        Args:
            pareto_solutions: list of solution tuples (decision vectors).
            problem: TestProblem instance.
            hv_history: list of (budget_used, hv_value) tuples.
            total_time: total wall-clock time in seconds.
            time_per_iter: list of per-iteration computation times.
            n_simulations: total number of simulation calls.
            ref_point: reference point for HV computation.

        Returns:
            dict with all metrics and data.
        """
        # Compute TRUE objectives for the output Pareto set
        true_objs = []
        for x in pareto_solutions:
            f1, f2, _ = problem.true_objectives(x)
            true_objs.append([f1, f2])
        true_objs = np.array(true_objs) if true_objs else np.empty((0, 2))

        # Filter to actual Pareto-optimal among true objectives
        if len(true_objs) > 0:
            pf_true, pf_idx = pareto_filter(true_objs, return_indices=True)
            pareto_solutions_filtered = [pareto_solutions[i] for i in pf_idx]
        else:
            pf_true = np.empty((0, 2))
            pareto_solutions_filtered = []

        # Compute metrics on true objectives
        true_pf = problem.true_pareto_front()
        hv = compute_hypervolume_2d(pf_true, ref_point)
        igd = compute_igd(pf_true, true_pf)
        cvr = compute_cvr(pareto_solutions_filtered, problem)

        return {
            'method': self.name,
            'pareto_solutions': [[int(v) for v in x] for x in pareto_solutions_filtered],
            'pareto_objectives_true': pf_true.tolist(),
            'hv_final': float(hv),
            'igd_final': float(igd),
            'cvr_final': float(cvr),
            'hv_history': [(int(s), float(v)) for s, v in hv_history] if hv_history else [],
            'total_time_sec': float(total_time),
            'time_per_iter': [float(t) for t in time_per_iter] if time_per_iter else [],
            'time_per_iter_mean': float(np.mean(time_per_iter)) if time_per_iter else 0.0,
            'n_simulations': n_simulations,
            'n_pareto_solutions': len(pareto_solutions_filtered),
        }
