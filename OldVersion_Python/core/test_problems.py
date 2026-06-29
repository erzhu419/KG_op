"""Test problem definitions: RZDT1, RZDT2, RZDT5.

Translated from: sim_func.m, sim_test.m (partial)

Old version parameters:
  - Decision space: x in [0, 100]^d (continuous, rounded to integers)
  - Normalized: x_norm = x / 100, so x_norm in {0, 0.01, 0.02, ..., 1.0}
  - Noise: additive Gaussian N(0, stdev^2), stdev = 0.05 for all objectives
  - Constraint: f3(x) = -(x1/100 - 0.5)^2, tau = -0.04, alpha = 1.645

sim_func.m only implements RZDT2. RZDT1 and RZDT5 are derived from
standard ZDT1 and ZDT3 respectively, with the same constraint function.
"""
import numpy as np


class OldTestProblem:
    """Base class for old-version test problems."""

    def __init__(self, d=5, x_L=None, x_U=None, stdev=None):
        self.d = d
        self.x_L = x_L if x_L is not None else np.zeros(d)
        self.x_U = x_U if x_U is not None else 100.0 * np.ones(d)
        self.stdev = stdev if stdev is not None else 0.05 * np.ones(3)
        # Constraint parameters
        self.tau_e = -0.04
        self.alpha_z = 1.645  # Phi^{-1}(0.95) for one-sided 5% test

    def normalize(self, x):
        """Normalize x from [x_L, x_U] to [0, 1]."""
        return (np.asarray(x, dtype=float) - self.x_L) / (self.x_U - self.x_L)

    def denormalize(self, x_norm):
        """Convert from [0, 1] to [x_L, x_U] and round to integers."""
        x = self.x_L + np.asarray(x_norm, dtype=float) * (self.x_U - self.x_L)
        return np.round(x)

    def true_objectives(self, x):
        """Compute true (noiseless) objectives. x in original scale [x_L, x_U].

        Returns
        -------
        np.ndarray of shape (3,)
            [f1, f2, f3] where f3 is the constraint function.
        """
        raise NotImplementedError

    def simulate(self, x, rng=None):
        """Run one noisy simulation at solution x (original scale).

        Translated from sim_func.m:
          y(i) = f_i(x) + normrnd(0, stdev(i))

        Parameters
        ----------
        x : np.ndarray of shape (d,)
            Solution in original scale [x_L, x_U].
        rng : np.random.RandomState, optional

        Returns
        -------
        np.ndarray of shape (3,)
        """
        f = self.true_objectives(x)
        if rng is None:
            noise = np.random.randn(3) * self.stdev
        else:
            noise = rng.randn(3) * self.stdev
        return f + noise

    def is_truly_feasible(self, x):
        """Check if x satisfies the probabilistic constraint.

        Constraint: E[f3(x)] + alpha * sigma_3(x) <= tau_e
        """
        f = self.true_objectives(x)
        return f[2] + self.alpha_z * self.stdev[2] <= self.tau_e

    def true_pareto_front(self, n_points=1000):
        """Compute the true Pareto front (feasible, noiseless).

        Returns (N, 2) array of (f1, f2) values.
        """
        raise NotImplementedError


class RZDT1(OldTestProblem):
    """RZDT1: Convex Pareto front based on ZDT1.

    f1(x) = x1 / 100
    g(x)  = 1 + 9/(d-1) * sum(x_j/100, j=2..d)
    f2(x) = g * (1 - sqrt(f1/g))
    f3(x) = -(x1/100 - 0.5)^2     (constraint)

    Pareto optimal: x_j = 0 for j >= 2.
    True PF: convex curve f2 = 1 - sqrt(f1), parameterized by x1.
    """

    def true_objectives(self, x):
        x = np.asarray(x, dtype=float)
        d = self.d
        f1 = x[0] / 100.0
        g = 1.0 + 9.0 / (d - 1) * np.sum(x[1:] / 100.0) if d > 1 else 1.0
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0, None)))
        f3 = -(x[0] / 100.0 - 0.5) ** 2
        return np.array([f1, f2, f3])

    def true_pareto_front(self, n_points=101):
        """Discrete Pareto front: x1 in {0,1,...,100}, x_j=0 for j>=2."""
        points = []
        for x1 in range(101):
            x = np.zeros(self.d)
            x[0] = x1
            f = self.true_objectives(x)
            if self.is_truly_feasible(x):
                points.append([f[0], f[1]])
        return np.array(points) if points else np.empty((0, 2))


class RZDT2(OldTestProblem):
    """RZDT2: Concave Pareto front based on ZDT2.

    Translated directly from sim_func.m:
      y(1) = x(1)/100 + noise
      g = 1 + 9/(n-1)*sum(x(2:n)/100)
      y(2) = g*(1-(x(1)/g/100)^2) + noise
      y(3) = -(x(1)/100-0.5)^2 + noise

    Pareto optimal: x_j = 0 for j >= 2.
    True PF: concave curve f2 = 1 - f1^2.
    """

    def true_objectives(self, x):
        x = np.asarray(x, dtype=float)
        d = self.d
        f1 = x[0] / 100.0
        g = 1.0 + 9.0 / (d - 1) * np.sum(x[1:] / 100.0) if d > 1 else 1.0
        f2 = g * (1.0 - (x[0] / g / 100.0) ** 2)
        f3 = -(x[0] / 100.0 - 0.5) ** 2
        return np.array([f1, f2, f3])

    def true_pareto_front(self, n_points=101):
        points = []
        for x1 in range(101):
            x = np.zeros(self.d)
            x[0] = x1
            f = self.true_objectives(x)
            if self.is_truly_feasible(x):
                points.append([f[0], f[1]])
        return np.array(points) if points else np.empty((0, 2))


class RZDT5(OldTestProblem):
    """RZDT5: Discrete Pareto front based on ZDT5.

    Translated from Draft_1017-Bao.docx Appendix B:
      g1(x,ξ) = x1/30 + ξ
      g2(x,ξ) = [Σ_{t=2}^d (x_t/5 + 1)] / (x1 + 1) + ξ
      g3(x,ξ) = x1/30 - 0.5 + ξ

    Decision bounds: x1 ∈ {0,...,30}, x_t ∈ {0,...,5} for t=2,...,d
    Pareto optimal: x_t = 0 for t >= 2, varying x1.
    Constraint with tau_e=-0.04: x1 ≤ 11 → 12 TPOS.
    """

    def __init__(self, d=5):
        x_L = np.zeros(d)
        x_U = np.zeros(d)
        x_U[0] = 30.0
        x_U[1:] = 5.0
        super().__init__(d=d, x_L=x_L, x_U=x_U)

    def true_objectives(self, x):
        x = np.asarray(x, dtype=float)
        d = self.d
        f1 = x[0] / 30.0
        g = np.sum(x[1:] / 5.0 + 1.0) if d > 1 else 1.0
        f2 = g / (x[0] + 1.0)
        f3 = x[0] / 30.0 - 0.5
        return np.array([f1, f2, f3])

    def true_pareto_front(self, n_points=None):
        """Discrete Pareto front: x_t=0 for t>=2, x1 varies over feasible range."""
        points = []
        for x1 in range(int(self.x_U[0]) + 1):
            x = np.copy(self.x_L)
            x[0] = x1
            f = self.true_objectives(x)
            if self.is_truly_feasible(x):
                points.append([f[0], f[1]])
        return np.array(points) if points else np.empty((0, 2))


PROBLEMS = {
    'RZDT1': RZDT1,
    'RZDT2': RZDT2,
    'RZDT5': RZDT5,
}
