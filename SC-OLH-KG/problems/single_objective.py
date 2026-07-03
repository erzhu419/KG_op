"""Single-objective chance-constrained wrappers."""

from __future__ import annotations

import numpy as np


class ScalarizedProblem:
    """Turn a vector RZDT problem into `(objective, constraint)` output.

    The objective is a scalarization of `(f1, f2)`, while the chance constraint
    uses the original `f3` output.
    """

    problem_name = "Scalarized"

    def __init__(self, base_problem, weights=(0.5, 0.5)):
        self.base = base_problem
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / max(float(np.sum(self.weights)), 1e-12)
        self.d = base_problem.d
        self.L = base_problem.L
        self.alpha = base_problem.alpha
        self.tau = base_problem.tau
        self.sigma_level = base_problem.sigma_level
        self.variance_features = getattr(base_problem, "variance_features", (0,))
        self.recommended_partition_features = getattr(
            base_problem, "recommended_partition_features", self.variance_features)
        self.ref_point = getattr(base_problem, "ref_point", None)
        self.problem_name = f"{base_problem.problem_name}_scalar"

    def int_bounds(self):
        return self.base.int_bounds()

    def normalize(self, x):
        return self.base.normalize(x)

    def continuous_to_int(self, x_norm):
        return self.base.continuous_to_int(x_norm)

    def sample_random(self, rng=None):
        return self.base.sample_random(rng)

    def risk_class(self, x):
        return self.base.risk_class(x)

    def true_vector_objectives(self, x):
        return self.base.true_objectives(x)

    def true_objective(self, x):
        f1, f2, _ = self.base.true_objectives(x)
        return float(self.weights[0] * f1 + self.weights[1] * f2)

    def true_constraint_mean(self, x):
        return float(self.base.true_objectives(x)[2])

    def true_outputs(self, x):
        return np.array([self.true_objective(x), self.true_constraint_mean(x)], dtype=float)

    def true_sigma(self, x):
        sig = self.base.true_sigma(x)
        obj_sig = np.sqrt((self.weights[0] * sig[0]) ** 2 + (self.weights[1] * sig[1]) ** 2)
        return np.array([float(obj_sig), float(sig[2])], dtype=float)

    def simulate(self, x, rng=None):
        rng = rng or np.random.default_rng()
        Y = self.base.simulate(x, rng)
        obj = float(self.weights[0] * Y[0] + self.weights[1] * Y[1])
        return np.array([obj, float(Y[2])], dtype=float)

    def is_truly_feasible(self, x):
        return self.base.is_truly_feasible(x)

    def all_axis_solutions(self):
        if hasattr(self.base, "all_axis_solutions"):
            return self.base.all_axis_solutions()
        lo, hi = self.int_bounds()
        return [tuple([x1] + [lo[j] for j in range(1, self.d)])
                for x1 in range(lo[0], hi[0] + 1)]

    def structured_candidates(self, n=10, rng=None):
        if hasattr(self.base, "structured_candidates"):
            return self.base.structured_candidates(n=n, rng=rng)
        return []

    def initial_samples(self, n=5, rng=None):
        if hasattr(self.base, "initial_samples"):
            return self.base.initial_samples(n=n, rng=rng)
        return []

    def hvd_residual_variance_cap(self, output_index=0):
        if hasattr(self.base, "hvd_residual_variance_cap"):
            return self.base.hvd_residual_variance_cap(output_index=output_index)
        return None

    def true_best_feasible(self):
        best_x = None
        best_y = np.inf
        for x in self.all_axis_solutions():
            if self.is_truly_feasible(x):
                y = self.true_objective(x)
                if y < best_y:
                    best_x, best_y = x, y
        return best_x, float(best_y)
