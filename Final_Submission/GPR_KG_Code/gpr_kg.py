"""
GPR-KG: Parametric Gaussian Process Regression with Knowledge Gradient
for Bi-Objective Constrained Simulation Optimization.

Implementation based on the paper submitted to Operations Research.

Algorithm Overview:
    The GPR-KG algorithm solves bi-objective simulation optimization problems
    with probabilistic constraints over large discrete decision spaces. It
    combines three key components:

    1. Parametric GPR Belief Model (Section 3 of the paper):
       - Models each objective f^i(x) = phi(x)^T * beta + zeta(x)
       - Uses quadratic polynomial basis without cross-terms
       - Augmented parameter vector theta = (beta, zeta) grows as new
         solutions are visited
       - Bayesian posterior updates via rank-one Kalman filtering

    2. VEPM - Variance Estimation Parametric Model (Section 3.3):
       - Partitions decision space into cells based on variable ranges
       - Shares variance estimates within partition cells
       - Enables variance extrapolation to unsampled solutions
       - Recursive O(1) updates per iteration

    3. Pareto-KG Sampling Policy (Section 4):
       - Knowledge Gradient factor measures expected reduction in best
         posterior estimate per objective
       - Bi-objective selection via Pareto non-dominance in KG-factor space
       - Crowding distance tie-breaking for diversity

Module Structure:
    - TestProblem, RZDT3, RZDT4, RZDT6: Heteroscedastic test problems
    - ParametricGPR: Parametric belief model with augmented features
    - VEPM: Partition-based variance estimation
    - compute_h, compute_kg_factor: KG computation (Frazier & Powell 2009)
    - pareto_filter, crowding_distance_select: Multi-objective utilities
    - compute_hypervolume_2d: Hypervolume indicator computation
    - GPRKR_Algorithm: Complete algorithm with full intermediate logging
"""

import numpy as np
import time
from scipy.stats import norm, qmc
from itertools import product as cart_product

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination


# =============================================================================
# 1. Test Problem Definitions (RZDT3, RZDT4, RZDT6 + legacy RZDT1, RZDT2, RZDT5)
#
#    All problems use integer decision vectors x in {1,...,L}^d.
#    Each has two objectives f1, f2 (to minimize) and one constraint
#    function f3 with probabilistic constraint P(g(x,xi) <= tau) >= 1-alpha.
#    Simulation outputs: Y^i(x) = f^i(x) + N(0, sigma^i(x)^2).
# =============================================================================

class TestProblem:
    """Base class for RZDT test problems.

    Attributes:
        d (int): Decision space dimension.
        L (int): Number of levels per dimension, x_j in {1,...,L}.
        sigma_level (float): Homoscedastic noise std (if not heteroscedastic).
        heteroscedastic (bool): Whether to use response-proportional noise.
        alpha (float): Constraint confidence level (default 0.05 for 95%).
        tau (float): Constraint threshold, set by calibrate_constraint().
    """

    def __init__(self, d=5, L=20, sigma=0.1, heteroscedastic=False, alpha=0.05):
        self.d = d
        self.L = L
        self.sigma_level = sigma
        self.heteroscedastic = heteroscedastic
        self.alpha = alpha
        self.tau = None  # set by calibrate_constraint
        self.ref_point = np.array([1.5, 1.5])  # default HV reference point

    def normalize(self, x):
        """Map integer x in {1,...,L} to [0,1] via (x-1)/(L-1)."""
        return (np.asarray(x, dtype=float) - 1.0) / (self.L - 1.0)

    def int_bounds(self):
        """Return (lo_array, hi_array) for the integer decision space.

        Default: x_j in {1,...,L} (1-indexed).
        Override in subclasses with different ranges (e.g., 0-indexed or mixed L).
        """
        lo = np.ones(self.d, dtype=int)
        hi = np.full(self.d, self.L, dtype=int)
        return lo, hi

    def sample_random(self):
        """Sample a random solution uniformly from the integer decision grid."""
        lo, hi = self.int_bounds()
        return tuple(int(np.random.randint(lo[j], hi[j] + 1)) for j in range(self.d))

    def continuous_to_int(self, x_norm):
        """Convert normalized [0,1]^d to integer grid using int_bounds.

        Args:
            x_norm: array-like of shape (d,) in [0,1]^d.
        Returns:
            tuple of d integers within [lo, hi] per dimension.
        """
        lo, hi = self.int_bounds()
        x_norm = np.asarray(x_norm, dtype=float)
        x_int = np.zeros(self.d, dtype=int)
        for j in range(self.d):
            x_int[j] = int(round(x_norm[j] * (hi[j] - lo[j]))) + lo[j]
            x_int[j] = int(np.clip(x_int[j], lo[j], hi[j]))
        return tuple(x_int)

    def sigma_func(self, x, f_val):
        """Noise standard deviation at solution x for a given true value f_val.

        Homoscedastic: sigma(x) = sigma_level (constant).
        Heteroscedastic: sigma(x) = 0.1 + 0.9*|f(x)| (response-proportional).
        """
        if self.heteroscedastic:
            return 0.1 + 0.9 * np.abs(f_val)
        else:
            return self.sigma_level

    def simulate(self, x):
        """Run one simulation replication at solution x.

        Returns:
            np.array of shape (3,): noisy observations [Y1, Y2, Y3].
        """
        f1, f2, f3 = self.true_objectives(x)
        s1 = self.sigma_func(x, f1)
        s2 = self.sigma_func(x, f2)
        s3 = self.sigma_func(x, f3)
        Y1 = f1 + np.random.randn() * s1
        Y2 = f2 + np.random.randn() * s2
        Y3 = f3 + np.random.randn() * s3
        return np.array([Y1, Y2, Y3])

    def true_sigma(self, x):
        """Return true noise std [sigma1, sigma2, sigma3] at solution x."""
        f1, f2, f3 = self.true_objectives(x)
        return np.array([self.sigma_func(x, f1),
                         self.sigma_func(x, f2),
                         self.sigma_func(x, f3)])

    def calibrate_constraint(self, feasibility_ratio=0.5):
        """Set threshold tau so ~feasibility_ratio fraction of solutions are feasible.

        Feasibility is defined by: f3(x) + Phi^{-1}(1-alpha)*sigma3(x) <= tau.
        We compute this quantile over all (or sampled) solutions to set tau.
        """
        if self.d <= 3 or self.L <= 5:
            all_sols = list(cart_product(range(1, self.L + 1), repeat=self.d))
        else:
            rng = np.random.RandomState(42)
            all_sols = [tuple(rng.randint(1, self.L + 1, size=self.d))
                        for _ in range(50000)]

        q_vals = []
        for x in all_sols:
            f1, f2, f3 = self.true_objectives(x)
            s3 = self.sigma_func(x, f3)
            q = f3 + norm.ppf(1 - self.alpha) * s3
            q_vals.append(q)

        self.tau = np.percentile(q_vals, feasibility_ratio * 100)
        return self.tau

    def is_truly_feasible(self, x):
        """Check if solution x satisfies the true probabilistic constraint."""
        _, _, f3 = self.true_objectives(x)
        s3 = self.sigma_func(x, f3)
        return f3 + norm.ppf(1 - self.alpha) * s3 <= self.tau

    def true_pareto_front(self):
        """Return the true discrete Pareto front (objective values).
        Must be implemented by subclass.
        """
        raise NotImplementedError


class RZDT1(TestProblem):
    """RZDT1: Convex Pareto front based on ZDT1.

    Decision space: x_j in {0,...,L}^d, L=100.  Normalized t_j = x_j/L.

    Objectives (original paper Bao et al.):
        f1(x) = x1/100
        g(x)  = 1 + 9/(d-1) * sum_{j>=2} x_j/100
        f2(x) = g * (1 - sqrt(f1/g))
        f3(x) = -(x1/100 - 0.5)^2 + 0.04

    Constraint (tau=0):
        Pr[f3(x,xi) <= 0] >= 1 - alpha.
        With homoscedastic sigma=0.05 gives 32 TPOS.

    Pareto-optimal: x_j=0 for j>=2 => g=1.

    Heteroscedastic noise:
        sigma_i(x) = sigma * (0.5 + 2*sqrt(t_1))
    """

    def __init__(self, d=5, L=100, sigma=0.1, heteroscedastic=False, alpha=0.05):
        super().__init__(d, L, sigma, heteroscedastic, alpha)

    def int_bounds(self):
        """x_j in {0,...,L} (0-indexed)."""
        lo = np.zeros(self.d, dtype=int)
        hi = np.full(self.d, self.L, dtype=int)
        return lo, hi

    def normalize(self, x):
        """Map x in {0,...,L} to [0,1] via x/L."""
        return np.asarray(x, dtype=float) / float(self.L)

    def continuous_to_int(self, x_norm):
        """Convert [0,1]^d to {0,...,L}^d."""
        x_norm = np.asarray(x_norm, dtype=float)
        x_int = np.round(x_norm * self.L).astype(int)
        return tuple(np.clip(x_int, 0, self.L))

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        g = 1.0 + 9.0 / (self.d - 1) * np.sum(t[1:]) if self.d > 1 else 1.0
        f2 = g * (1.0 - np.sqrt(f1 / g))
        f3 = -(t[0] - 0.5) ** 2 + 0.04
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        if self.heteroscedastic:
            t1 = float(x[0]) / float(self.L)
            return self.sigma_level * (0.5 + 2.0 * np.sqrt(max(t1, 0.0)))
        return self.sigma_level

    def true_pareto_front(self):
        points = []
        for x1 in range(0, self.L + 1):
            x = tuple([x1] + [0] * (self.d - 1))
            if self.is_truly_feasible(x):
                f1, f2, _ = self.true_objectives(x)
                points.append((f1, f2))
        return np.array(points) if points else np.empty((0, 2))

    def true_pareto_curve(self):
        t = np.linspace(0, 1, 200)
        return t, 1.0 - np.sqrt(t)


class RZDT2(TestProblem):
    """RZDT2: Concave Pareto front based on ZDT2.

    Decision space: x_j in {0,...,L}^d, L=100.  Normalized t_j = x_j/L.

    Objectives:
        f1(x) = x1/100
        g(x)  = 1 + 9/(d-1) * sum_{j>=2} x_j/100
        f2(x) = g * (1 - (f1/g)^2)
        f3(x) = -(x1/100 - 0.5)^2 + 0.04

    Pareto-optimal: x_j=0 for j>=2.

    Heteroscedastic noise:
        sigma_i(x) = sigma * (0.5 + 2.5*sin^2(pi*t_1))
    """

    def __init__(self, d=5, L=100, sigma=0.1, heteroscedastic=False, alpha=0.05):
        super().__init__(d, L, sigma, heteroscedastic, alpha)

    def int_bounds(self):
        """x_j in {0,...,L} (0-indexed)."""
        lo = np.zeros(self.d, dtype=int)
        hi = np.full(self.d, self.L, dtype=int)
        return lo, hi

    def normalize(self, x):
        """Map x in {0,...,L} to [0,1] via x/L."""
        return np.asarray(x, dtype=float) / float(self.L)

    def continuous_to_int(self, x_norm):
        """Convert [0,1]^d to {0,...,L}^d."""
        x_norm = np.asarray(x_norm, dtype=float)
        x_int = np.round(x_norm * self.L).astype(int)
        return tuple(np.clip(x_int, 0, self.L))

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        g = 1.0 + 9.0 / (self.d - 1) * np.sum(t[1:]) if self.d > 1 else 1.0
        f2 = g * (1.0 - (f1 / g) ** 2)
        f3 = -(t[0] - 0.5) ** 2 + 0.04
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        if self.heteroscedastic:
            t1 = float(x[0]) / float(self.L)
            return self.sigma_level * (0.5 + 2.5 * np.sin(np.pi * t1) ** 2)
        return self.sigma_level

    def true_pareto_front(self):
        points = []
        for x1 in range(0, self.L + 1):
            x = tuple([x1] + [0] * (self.d - 1))
            if self.is_truly_feasible(x):
                f1, f2, _ = self.true_objectives(x)
                points.append((f1, f2))
        return np.array(points) if points else np.empty((0, 2))

    def true_pareto_curve(self):
        t = np.linspace(0, 1, 200)
        return t, 1.0 - t ** 2


class RZDT5(TestProblem):
    """RZDT5: Hyperbolic Pareto front based on ZDT5.

    Original bounds (Bao et al.): x1 in {0,...,30}, x_j in {0,...,5} for j>=2.

    Objectives:
        f1(x) = x1/30
        g(x)  = 1 + sum_{j>=2} x_j/5
        f2(x) = g / (x1 + 1)
        f3(x) = x1/30 - 0.5

    Constraint (tau=0):
        With heteroscedastic sigma=0.1 gives ~12 TPOS (x1 in {0,...,11}).

    Pareto-optimal: x_j=0 for j>=2 => g=1.

    Heteroscedastic noise:
        sigma_i(x) = sigma * (0.3 + 2*t_1^2)  where t_1 = x1/30
    """

    def __init__(self, d=5, L=30, sigma=0.1, heteroscedastic=False, alpha=0.05):
        # L=30 is the range for x1; x_j in {0,...,5} for j>=2
        super().__init__(d, L, sigma, heteroscedastic, alpha)
        self.L1 = 30   # x1 range: {0,...,30}
        self.L2 = 5    # x_j range: {0,...,5} for j>=2
        self.ref_point = np.array([1.5, 1.5])

    def int_bounds(self):
        """x1 in {0,...,30}, x_j in {0,...,5} for j>=2."""
        lo = np.zeros(self.d, dtype=int)
        hi = np.full(self.d, self.L2, dtype=int)
        hi[0] = self.L1
        return lo, hi

    def normalize(self, x):
        """Map to [0,1]: t1 = x1/30, t_j = x_j/5 for j>=2."""
        x = np.asarray(x, dtype=float)
        t = np.zeros(self.d)
        t[0] = x[0] / float(self.L1)
        if self.d > 1:
            t[1:] = x[1:] / float(self.L2)
        return t

    def continuous_to_int(self, x_norm):
        """Convert [0,1]^d to integer grid with mixed L."""
        x_norm = np.asarray(x_norm, dtype=float)
        x_int = np.zeros(self.d, dtype=int)
        x_int[0] = int(np.clip(round(x_norm[0] * self.L1), 0, self.L1))
        for j in range(1, self.d):
            x_int[j] = int(np.clip(round(x_norm[j] * self.L2), 0, self.L2))
        return tuple(x_int)

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        g = 1.0 + np.sum(t[1:]) if self.d > 1 else 1.0
        f2 = g / (30.0 * t[0] + 1.0)
        f3 = t[0] - 0.5
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        if self.heteroscedastic:
            t1 = float(x[0]) / float(self.L1)
            return self.sigma_level * (0.3 + 2.0 * t1 ** 2)
        return self.sigma_level

    def true_pareto_front(self):
        points = []
        for x1 in range(0, self.L1 + 1):
            x = tuple([x1] + [0] * (self.d - 1))
            if self.is_truly_feasible(x):
                f1, f2, _ = self.true_objectives(x)
                points.append((f1, f2))
        return np.array(points) if points else np.empty((0, 2))

    def true_pareto_curve(self):
        """Continuous reference curve f2 = 1/(30*t1+1) for plotting."""
        t = np.linspace(0, 1, 200)
        return t, 1.0 / (30.0 * t + 1.0)


class RZDT5_R(TestProblem):
    """RZDT5_R: Revised RZDT5 with extended decision space and loosened constraint.

    Decision space: x_1 in {0,...,100}, x_j in {0,...,50} for j>=2.

    Objectives:
        f1(x) = x1/100
        g(x)  = 1 + sum_{j>=2} x_j/50
        f2(x) = g / (x1 + 1)          (hyperbolic)
        f3(x) = x1/100 - 0.5           (constraint output)

    Chance constraint (tau=0.5, alpha=0.05):
        P(f3(x,xi) <= 0.5) >= 0.95
        With heteroscedastic sigma=0.04: 88 TPOS (x1 in {0,...,87}, x_j=0).

    Pareto-optimal (unconstrained): x_j=0 for j>=2 => g=1.

    Heteroscedastic noise:
        sigma_i(x) = sigma * (0.3 + 2*t_1^2),  t_1 = x1/100
        Global variance ratio: ~58.8x;  TPOS variance ratio: ~36.6x.
    """

    def __init__(self, d=5, L=100, sigma=0.04, heteroscedastic=True, alpha=0.05):
        super().__init__(d, L, sigma, heteroscedastic, alpha)
        self.L1 = 100  # x1 range: {0,...,100}
        self.L2 = 50   # x_j range: {0,...,50} for j>=2
        self.ref_point = np.array([1.5, 1.5])

    def int_bounds(self):
        """x1 in {0,...,100}, x_j in {0,...,50} for j>=2."""
        lo = np.zeros(self.d, dtype=int)
        hi = np.full(self.d, self.L2, dtype=int)
        hi[0] = self.L1
        return lo, hi

    def normalize(self, x):
        """Map to [0,1]: t1 = x1/100, t_j = x_j/50 for j>=2."""
        x = np.asarray(x, dtype=float)
        t = np.zeros(self.d)
        t[0] = x[0] / float(self.L1)
        if self.d > 1:
            t[1:] = x[1:] / float(self.L2)
        return t

    def continuous_to_int(self, x_norm):
        """Convert [0,1]^d to integer grid with mixed L."""
        x_norm = np.asarray(x_norm, dtype=float)
        x_int = np.zeros(self.d, dtype=int)
        x_int[0] = int(np.clip(round(x_norm[0] * self.L1), 0, self.L1))
        for j in range(1, self.d):
            x_int[j] = int(np.clip(round(x_norm[j] * self.L2), 0, self.L2))
        return tuple(x_int)

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        g = 1.0 + np.sum(t[1:]) if self.d > 1 else 1.0
        f2 = g / (float(x[0]) + 1.0)
        f3 = t[0] - 0.5
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        if self.heteroscedastic:
            t1 = float(x[0]) / float(self.L1)
            return self.sigma_level * (0.3 + 2.0 * t1 ** 2)
        return self.sigma_level

    def true_pareto_front(self):
        points = []
        for x1 in range(0, self.L1 + 1):
            x = tuple([x1] + [0] * (self.d - 1))
            if self.is_truly_feasible(x):
                f1, f2, _ = self.true_objectives(x)
                points.append((f1, f2))
        return np.array(points) if points else np.empty((0, 2))

    def true_pareto_curve(self):
        """Continuous reference curve f2 = 1/(100*t1+1) for plotting."""
        t = np.linspace(0, 1, 200)
        return t, 1.0 / (100.0 * t + 1.0)


class RZDT5_RR(RZDT5_R):
    """RZDT5_RR: RZDT5_R geometry but with the original strict constraint tau=0.

    Same decision space, objectives, and noise as RZDT5_R:
        x_1 in {0,...,100}, x_j in {0,...,50} for j>=2.
        f1 = x1/100,  g = 1 + sum_{j>=2} x_j/50,  f2 = g/(x1+1),
        f3 = x1/100 - 0.5,  sigma_i(x) = sigma * (0.3 + 2*t_1^2).

    Chance constraint (tau=0, alpha=0.05):
        P(f3(x,xi) <= 0.0) >= 0.95.
        With sigma=0.04: 46 TPOS (x_1 in {0,...,45}, x_j=0).
        TPOS variance ratio: ~5.5x.
    """
    pass


class RZDT1_VC(RZDT1):
    """Variance-critical RZDT1.

    The objective geometry is identical to RZDT1.  The constraint mean is
    constructed so that true chance feasibility along the Pareto set is
    controlled by a deterministic margin, while the observed constraint noise
    remains the same monotone heteroscedastic field as RZDT1.

    With tau=0, true feasibility is equivalent to margin(t1) <= 0.  A pooled
    variance model instead uses margin(t1) + z_alpha*(sigma_pool-sigma(t1)),
    so it tends to reject low-noise feasible points and accept high-noise
    infeasible points near the boundary.
    """

    variance_features = (0,)
    recommended_partition_features = (0,)

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        g = 1.0 + 9.0 / (self.d - 1) * np.sum(t[1:]) if self.d > 1 else 1.0
        f2 = g * (1.0 - np.sqrt(f1 / g))
        margin = 0.015 * ((t[0] - 0.25) * (t[0] - 0.75) / 0.0625)
        sigma3 = self.sigma_func(x, 0.0)
        f3 = margin - norm.ppf(1 - self.alpha) * sigma3
        return f1, f2, f3


class RZDT2_VC(RZDT2):
    """Variance-critical RZDT2.

    The objective geometry and bell-shaped heteroscedastic noise are inherited
    from RZDT2.  The chance constraint is calibrated through a deterministic
    margin, making variance miscalibration visible in feasible-set
    classification rather than only in posterior uncertainty.
    """

    variance_features = (0,)
    recommended_partition_features = (0,)

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        g = 1.0 + 9.0 / (self.d - 1) * np.sum(t[1:]) if self.d > 1 else 1.0
        f2 = g * (1.0 - (f1 / g) ** 2)
        margin = -0.012 * np.cos(2.0 * np.pi * t[0])
        sigma3 = self.sigma_func(x, 0.0)
        f3 = margin - norm.ppf(1 - self.alpha) * sigma3
        return f1, f2, f3


class RZDT5_RR_VC(RZDT5_RR):
    """Variance-critical RZDT5_RR.

    This variant keeps the enlarged RZDT5_RR grid and hyperbolic front, but
    changes the chance-constraint mean so that the feasible Pareto segment
    spans both lower- and higher-variance regions.  It is intended for
    oracle-gap diagnostics rather than as a replacement for the original
    moderate-heteroscedastic RZDT5_RR case.
    """

    variance_features = (0,)
    recommended_partition_features = (0,)

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        g = 1.0 + np.sum(t[1:]) if self.d > 1 else 1.0
        f2 = g / (float(x[0]) + 1.0)
        margin = 0.012 * ((t[0] - 0.20) * (t[0] - 0.80) / 0.0600)
        sigma3 = self.sigma_func(x, 0.0)
        f3 = margin - norm.ppf(1 - self.alpha) * sigma3
        return f1, f2, f3


class RCZDTBase(TestProblem):
    """Base class for variance-critical curved-discrete benchmarks.

    These problems keep an integer box x_j in {0,...,L} but deliberately use
    multi-coordinate Pareto manifolds and chance constraints whose feasibility
    is sensitive to local noise calibration.
    """

    variance_features = (0,)
    recommended_partition_features = (0,)

    def __init__(self, d=5, L=100, sigma=0.04, heteroscedastic=True,
                 alpha=0.05):
        super().__init__(d, L, sigma, heteroscedastic, alpha)
        self.ref_point = np.array([1.5, 1.5])

    def int_bounds(self):
        lo = np.zeros(self.d, dtype=int)
        hi = np.full(self.d, self.L, dtype=int)
        return lo, hi

    def normalize(self, x):
        return np.asarray(x, dtype=float) / float(self.L)

    def continuous_to_int(self, x_norm):
        x_norm = np.asarray(x_norm, dtype=float)
        x_int = np.round(x_norm * self.L).astype(int)
        return tuple(np.clip(x_int, 0, self.L))

    def _tail_penalty(self, t, start):
        if self.d <= start:
            return 0.0
        return float(np.sum(t[start:] ** 2))

    def true_pareto_solutions(self):
        raise NotImplementedError

    def true_pareto_front(self):
        sols = []
        seen = set()
        for x in self.true_pareto_solutions():
            x = tuple(int(v) for v in x)
            if x in seen:
                continue
            seen.add(x)
            if self.is_truly_feasible(x):
                sols.append(x)
        if not sols:
            return np.empty((0, 2))
        objs = np.array([self.true_objectives(x)[:2] for x in sols],
                        dtype=float)
        return pareto_filter(objs)


class RCZDT_Curve2D(RCZDTBase):
    """Curved two-coordinate Pareto set with center-peaked variance.

    The unconstrained Pareto set is approximately
    x = (x1, L-x1, 0, 0, ...).  The chance boundary cuts through the
    high-noise middle part of this curve, making pooled variance prone to
    low-noise false rejections and high-noise false acceptances.
    """

    variance_features = (0, 1)
    recommended_partition_features = (0, 1)

    def _pareto_x(self, x1):
        x = [0] * self.d
        x[0] = int(x1)
        if self.d > 1:
            x[1] = int(np.clip(round(self.L - x1), 0, self.L))
        return tuple(x)

    def true_pareto_solutions(self):
        return [self._pareto_x(x1) for x1 in range(0, self.L + 1)]

    def true_objectives(self, x):
        t = self.normalize(x)
        u = float(t[0])
        v = float(t[1]) if self.d > 1 else 1.0 - u
        penalty = 2.0 * (v - (1.0 - u)) ** 2 + 2.0 * self._tail_penalty(t, 2)
        f1 = u + penalty
        f2 = 1.0 - np.sqrt(max(u, 0.0)) + penalty
        margin = 0.018 * ((u - 0.30) * (u - 0.70) / 0.040)
        sigma3 = self.sigma_func(x, 0.0)
        f3 = margin - norm.ppf(1 - self.alpha) * sigma3
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        if self.heteroscedastic:
            t = self.normalize(x)
            u = float(t[0])
            v = float(t[1]) if self.d > 1 else 1.0 - u
            bump = np.exp(-((u - 0.5) ** 2 + (v - 0.5) ** 2) / 0.035)
            return self.sigma_level * (0.25 + 2.75 * bump)
        return self.sigma_level


class RCZDT_MisalignedV(RCZDTBase):
    """Multi-coordinate Pareto set whose variance is mainly controlled by x3."""

    variance_features = (2,)
    recommended_partition_features = (2,)

    @staticmethod
    def _v_star(u):
        return 0.15 + 0.70 * u ** 2

    @staticmethod
    def _r_star(u):
        return 0.50 + 0.35 * np.sin(2.0 * np.pi * u)

    def _pareto_x(self, x1):
        u = float(x1) / float(self.L)
        x = [0] * self.d
        x[0] = int(x1)
        if self.d > 1:
            x[1] = int(np.clip(round(self.L * self._v_star(u)), 0, self.L))
        if self.d > 2:
            x[2] = int(np.clip(round(self.L * self._r_star(u)), 0, self.L))
        return tuple(x)

    def true_pareto_solutions(self):
        return [self._pareto_x(x1) for x1 in range(0, self.L + 1)]

    def true_objectives(self, x):
        t = self.normalize(x)
        u = float(t[0])
        v = float(t[1]) if self.d > 1 else self._v_star(u)
        r = float(t[2]) if self.d > 2 else self._r_star(u)
        penalty = (
            2.0 * (v - self._v_star(u)) ** 2
            + 2.0 * (r - self._r_star(u)) ** 2
            + 2.0 * self._tail_penalty(t, 3))
        f1 = u + penalty
        f2 = (1.0 - u) ** 2 + penalty
        margin = 0.014 * np.sin(2.0 * np.pi * u)
        sigma3 = self.sigma_func(x, 0.0)
        f3 = margin - norm.ppf(1 - self.alpha) * sigma3
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        if self.heteroscedastic:
            t = self.normalize(x)
            v = float(t[1]) if self.d > 1 else 0.5
            r = float(t[2]) if self.d > 2 else 0.5
            logistic = 1.0 / (1.0 + np.exp(-10.0 * (r - 0.5)))
            return self.sigma_level * (
                0.30 + 2.70 * logistic + 0.30 * np.sin(np.pi * v) ** 2)
        return self.sigma_level


class RCZDT_StepV(RCZDTBase):
    """Piecewise Pareto manifold with region-type heteroscedasticity."""

    variance_features = (2,)
    recommended_partition_features = (2,)

    @staticmethod
    def _r_star(u):
        return 0.25 if u < 0.5 else 0.75

    def _pareto_x(self, x1):
        u = float(x1) / float(self.L)
        x = [0] * self.d
        x[0] = int(x1)
        if self.d > 1:
            x[1] = int(np.clip(round(self.L * (1.0 - u)), 0, self.L))
        if self.d > 2:
            x[2] = int(np.clip(round(self.L * self._r_star(u)), 0, self.L))
        return tuple(x)

    def true_pareto_solutions(self):
        return [self._pareto_x(x1) for x1 in range(0, self.L + 1)]

    def true_objectives(self, x):
        t = self.normalize(x)
        u = float(t[0])
        v = float(t[1]) if self.d > 1 else 1.0 - u
        r = float(t[2]) if self.d > 2 else self._r_star(u)
        penalty = (
            2.0 * (v - (1.0 - u)) ** 2
            + 2.0 * (r - self._r_star(u)) ** 2
            + 2.0 * self._tail_penalty(t, 3))
        f1 = u + penalty
        f2 = 1.0 - u + penalty
        margin = 0.030 * (u - 0.5)
        sigma3 = self.sigma_func(x, 0.0)
        f3 = margin - norm.ppf(1 - self.alpha) * sigma3
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        if self.heteroscedastic:
            t = self.normalize(x)
            r = float(t[2]) if self.d > 2 else 0.0
            step = 1.0 / (1.0 + np.exp(-25.0 * (r - 0.5)))
            return self.sigma_level * (0.50 + 2.50 * step)
        return self.sigma_level


class RZDT3(TestProblem):
    """RZDT3: Five-band discontinuous Pareto front based on ZDT3.

    Decision space: x_j in {1,...,L}^d.  Normalized t_j = (x_j-1)/(L-1).

    Objectives:
        f1(x) = t_1
        g(x)  = 1 + 9/(d-1) * sum_{j>=2} t_j
        f2(x) = g * (1 - sqrt(f1/g) - (f1/g) * sin(10*pi*f1))
        f3(x) = 0.5 * |sin(10*pi*t_1)| + 0.5  (constraint output)

    Constraint:
        The f3 formula mirrors the five-band sinusoidal structure of f2: peaks
        of |sin(10*pi*t_1)| align with the peaks of each Pareto band, making
        the band peaks probabilistically infeasible.  At tau_50 calibration
        (50% overall feasibility), ~11/20 Pareto-optimal solutions are feasible
        and the infeasible ones are scattered across all five bands.  At tight
        calibration (~10%), only ~3/20 remain feasible.

    Structure:
        The ZDT3 discontinuous terms create a Pareto front consisting of five
        disjoint bands in objective space (Zitzler et al. 2000, Eq. 6).  The
        sinusoidal term in f2 causes f2 to dip below 0 near the top of each
        band, producing non-dominated gaps.

    Heteroscedastic noise (active when heteroscedastic=True):
        sigma_i(x) = sigma_level * (1 + 3 * (2*t_1 - 1)^2)

        U-shaped noise profile: high at the two extremes (t_1=0 and t_1=1)
        and low at the center (t_1=0.5).  Variance at the endpoints is 16x
        higher than at the center.  With VEPM's default 4-bin partition on
        x_1, the two outer bins (t_1 near 0 and 1) contain high-variance
        solutions while the two inner bins contain low-variance solutions.
        A pooled variance estimator over-estimates noise in the inner bins
        by ~3x and under-estimates it in the outer bins by ~1.7x.

        This U-shaped pattern is complementary to the exponential gradient
        (RZDT4) and linear gradient (RZDT6) noise: all three provide
        distinct spatial structures for testing VEPM's partition-based
        variance estimation.

    References:
        Zitzler, E., Deb, K., and Thiele, L. (2000). "Comparison of
        Multiobjective Evolutionary Algorithms: Empirical Results."
        Evolutionary Computation, 8(2), 173-195.
        https://doi.org/10.1162/106365600568202
    """

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        g = 1.0 + 9.0 / (self.d - 1) * np.sum(t[1:]) if self.d > 1 else 1.0
        ratio = f1 / g if g > 1e-12 else 0.0
        f2 = g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1))
        f3 = 0.5 * np.abs(np.sin(10.0 * np.pi * t[0])) + 0.5
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        """U-shaped position-dependent noise.

        sigma(x) = sigma_level * (1 + 3 * (2*t_1 - 1)^2)

        Noise is highest at the two extremes of the first decision dimension
        (t_1=0 and t_1=1, sigma=4*sigma_level) and lowest at the center
        (t_1=0.5, sigma=sigma_level).  The variance ratio is 16x.

        Physical interpretation for ZDT3: the five Pareto-optimal bands span
        f1 in [0, ~0.85], so the outer bands (near f1=0 and f1=0.85) carry
        higher noise and the middle bands carry lower noise.  VEPM's partition
        correctly identifies this pattern; a pooled estimator does not.
        """
        if self.heteroscedastic:
            t1 = (float(x[0]) - 1.0) / (self.L - 1.0)
            return self.sigma_level * (1.0 + 3.0 * (2.0 * t1 - 1.0) ** 2)
        return self.sigma_level

    def true_pareto_front(self):
        """Return feasible Pareto-optimal front points (f1, f2).

        The true Pareto front is a subset of the ZDT3 curve
            f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1),  f1 in [0,1]
        restricted to the feasible discrete solutions.
        """
        points = []
        for x1 in range(1, self.L + 1):
            x = tuple([x1] + [1] * (self.d - 1))
            if self.is_truly_feasible(x):
                f1, f2, _ = self.true_objectives(x)
                points.append((f1, f2))
        return np.array(points) if points else np.empty((0, 2))

    def true_pareto_curve(self):
        """Continuous reference curve: ZDT3 front on [0,1]."""
        t = np.linspace(0, 1, 1000)
        return t, 1.0 - np.sqrt(t) - t * np.sin(10.0 * np.pi * t)


class RZDT4(TestProblem):
    """RZDT4: Multi-modal landscape with convex Pareto front.

    Decision space: x_j in {1,...,L}^d.  Normalized t_j = (x_j-1)/(L-1).

    Objectives:
        f1(x) = t_1
        g(x)  = 1 + (1.5/(d-1)) * sum_{j>=2} |sin(2*pi*t_j)|
        f2(x) = g * (1 - sqrt(f1/g))
        f3(x) = t_1 + 1/(d-1) * sum_{j>=2} |sin(2*pi*t_j)| + 0.3

    Constraint:
        The f3 formula mirrors the g-function structure: the multi-modal
        |sin| terms add a constraint penalty for non-Pareto solutions (t_j≠0),
        while on the Pareto front (t_j=0) f3 simplifies to t_1 + 0.3.  The
        constraint thus increases linearly with t_1 along the Pareto front.
        Combined with exponential noise, high-t_1 Pareto solutions carry both
        larger f3 values AND larger safety margins, cutting the last 5 solutions
        (x_1=16-20) under tau_50 calibration.

    Structure:
        The absolute-sine g function creates (d-1) independent multi-modal
        dimensions.  Each t_j in {0, 0.5, 1.0} gives |sin| = 0 (a local
        optimum for g); all other values give positive contributions.  With
        L=20, these exact grid optima are unavailable (L/2=10 → t=0.474, not
        0.5), so the true minimum of g is at x_j=1 (t_j=0) for all j≥2.

        The global Pareto front is convex (same as RZDT1) but surrounded by
        a deceptive landscape with ~2^(d-1) apparent local optima.  Methods
        without effective exploration easily stagnate at suboptimal plateaus.

        g range: [1, 1 + 1.5] = [1, 2.5] for any d, keeping f2 in [0, 2.5].

    Heteroscedastic noise (active when heteroscedastic=True):
        sigma_i(x) = sigma_level * exp(gamma * t_1),  gamma = 2.2

        Noise grows exponentially across the first decision dimension from
        sigma_level (t_1=0) to sigma_level*exp(2.2) ≈ 9x at t_1=1.  This
        creates a strong directional gradient in variance that is spatially
        independent of the objective values, making response-based variance
        estimators ineffective.  A pooled variance estimator over-estimates
        noise in the low-t_1 region by ~20x, causing severe mis-scoring of
        KG factors for solutions near the low-noise end of the Pareto front.

    References:
        Zitzler, E., Deb, K., and Thiele, L. (2000). "Comparison of
        Multiobjective Evolutionary Algorithms: Empirical Results."
        Evolutionary Computation, 8(2), 173-195.
        https://doi.org/10.1162/106365600568202
    """

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0]
        if self.d > 1:
            g = 1.0 + (1.5 / (self.d - 1)) * np.sum(np.abs(np.sin(2.0 * np.pi * t[1:])))
        else:
            g = 1.0
        ratio = f1 / g if g > 1e-12 else 0.0
        f2 = g * (1.0 - np.sqrt(ratio))
        f3 = t[0] + (1.0 / (self.d - 1)) * np.sum(np.abs(np.sin(2.0 * np.pi * t[1:]))) + 0.3 if self.d > 1 else t[0] + 0.3
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        """Exponential noise along the first decision dimension.

        sigma(x) = sigma_level * exp(gamma * t_1),  gamma = 2.2

        Rationale: the exponential profile is spatially independent of the
        objective landscape, ensuring that only partition-aware estimators
        such as VEPM can correctly capture the noise structure.  Simple
        response-proportional estimators (sigma ∝ |f|) will systematically
        misestimate noise in this problem.
        """
        if self.heteroscedastic:
            t1 = (float(x[0]) - 1.0) / (self.L - 1.0)
            return self.sigma_level * np.exp(2.2 * t1)
        return self.sigma_level

    def true_pareto_front(self):
        """Return feasible Pareto-optimal front (f1, f2).

        The global Pareto front lies at g=1, achieved when x_j=1 for all j>=2
        (giving t_j=0, |sin(0)|=0).  Shape is convex, identical to RZDT1 but
        the multi-modal landscape makes it hard to discover.
        """
        points = []
        for x1 in range(1, self.L + 1):
            x = tuple([x1] + [1] * (self.d - 1))
            if self.is_truly_feasible(x):
                f1, f2, _ = self.true_objectives(x)
                points.append((f1, f2))
        return np.array(points) if points else np.empty((0, 2))

    def true_pareto_curve(self):
        """Continuous reference curve f2 = 1 - sqrt(f1) (at g=1)."""
        t = np.linspace(0, 1, 200)
        return t, 1.0 - np.sqrt(t)


class RZDT6(TestProblem):
    """RZDT6: Non-uniform Pareto density with non-convex front.

    Decision space: x_j in {1,...,L}^d.  Normalized t_j = (x_j-1)/(L-1).

    Objectives:
        f1(x) = t_1^3                          (non-uniform: dense near t_1=0)
        g(x)  = 1 + 9/(d-1) * sum_{j>=2} t_j  (same as RZDT1/2)
        f2(x) = g * (1 - (f1/g)^2)            (non-convex: ZDT2-like)
        f3(x) = 3 * t_1^2 + 0.1               (constraint output)

    Constraint:
        The f3 formula uses t_1^2, consistent with the cubic f1=t_1^3 transform:
        both are convex functions concentrated near t_1=1.  On the Pareto front
        f3 = 3*t_1^2 + 0.1, growing from 0.1 (t_1=0) to 3.1 (t_1=1).  With
        linear noise sigma*(0.3+2.2*t_1), high-t_1 solutions carry both large
        f3 and large safety margins.  Under tau_50 calibration ~11/20 Pareto
        solutions are feasible (x_1=1-11), cutting the sparsely-sampled
        high-f1 end of the non-convex front.

    Structure:
        The cubic transform of t_1 compresses many solutions into a small f1
        range near 0 (e.g., t_1=0.5 gives f1=0.125), creating highly
        non-uniform density on the Pareto front.  Algorithms that sample
        uniformly in decision space will under-explore the high-f1 region
        (t_1 near 1) and over-sample the low-f1 region.

        The non-convex f2 = g*(1-(f1/g)^2) curve (ZDT2 family) ensures the
        problem cannot be solved by simple weighted-sum scalarization.

        f1 range: [0, 1], f2 at Pareto (g=1): 1 - t_1^6 in [0, 1].

    Heteroscedastic noise (active when heteroscedastic=True):
        sigma_i(x) = sigma_level * (0.3 + 2.2 * t_1)

        Noise increases linearly with t_1 from 0.3*sigma_level (low-f1 region,
        densely sampled) to 2.5*sigma_level (high-f1 region, sparsely sampled).
        This is the most adversarial pattern for methods without spatial variance
        estimation: the region with fewest samples has the highest noise, and a
        naive pooled-variance estimator will severely under-estimate noise for
        solutions with large t_1.

    References:
        Zitzler, E., Deb, K., and Thiele, L. (2000). "Comparison of
        Multiobjective Evolutionary Algorithms: Empirical Results."
        Evolutionary Computation, 8(2), 173-195.
        https://doi.org/10.1162/106365600568202
    """

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = t[0] ** 3
        g = 1.0 + 9.0 / (self.d - 1) * np.sum(t[1:]) if self.d > 1 else 1.0
        ratio = f1 / g if g > 1e-12 else 0.0
        f2 = g * (1.0 - ratio ** 2)
        f3 = 3.0 * t[0] ** 2 + 0.1
        return f1, f2, f3

    def sigma_func(self, x, f_val):
        """Linear gradient noise along the first decision dimension.

        sigma(x) = sigma_level * (0.3 + 2.2 * t_1)

        This creates a ~8.3x difference in noise between the two extremes of
        the first dimension.  Since the non-uniform f1=t_1^3 transform places
        most Pareto-optimal solutions near small t_1 values, the low-noise
        region is densely sampled while the high-noise region is sparse — a
        scenario where VEPM's partition-based extrapolation has the largest
        advantage over sample-mean variance estimators.
        """
        if self.heteroscedastic:
            t1 = (float(x[0]) - 1.0) / (self.L - 1.0)
            return self.sigma_level * (0.3 + 2.2 * t1)
        return self.sigma_level

    def true_pareto_front(self):
        """Return feasible Pareto-optimal front (f1, f2).

        PF lies at g=1 (x_j=1 for j>=2), with f2 = 1 - f1^2 = 1 - t_1^6.
        Non-convex curve with non-uniform point spacing.
        """
        points = []
        for x1 in range(1, self.L + 1):
            x = tuple([x1] + [1] * (self.d - 1))
            if self.is_truly_feasible(x):
                f1, f2, _ = self.true_objectives(x)
                points.append((f1, f2))
        return np.array(points) if points else np.empty((0, 2))

    def true_pareto_curve(self):
        """Continuous reference curve f2 = 1 - f1^2 (at g=1, f1=t^3)."""
        t = np.linspace(0, 1, 200)
        f1 = t ** 3
        f2 = 1.0 - f1 ** 2
        return f1, f2


# =============================================================================
# 2. Parametric GPR Belief Model
#
#    For each evaluation index i in {1,2,3}:
#      f^i(x) = phi(x)^T * beta^i + zeta^i(x)
#
#    - beta^i ~ N(a_beta_0, C_beta_0)  [parametric coefficients]
#    - zeta^i(x) ~ N(0, lambda_i)      [solution-specific deviation]
#
#    The augmented parameter theta^i = (beta^i, zeta^i_sampled) and
#    augmented feature tilde_phi(x) enable a unified Kalman update.
# =============================================================================

class ParametricGPR:
    """Parametric Gaussian Process Regression with augmented features.

    Maintains posterior N(a, C) for the augmented parameter theta^i.
    Supports dimension augmentation when new solutions are first visited.

    Attributes:
        d (int): Decision space dimension.
        p (int): Number of basis functions = 2d+1 (quadratic without cross-terms).
        lambda_i (float): Prior variance for deviation terms zeta(x).
        a (np.array): Posterior mean of theta (size p + |X_n|).
        C (np.array): Posterior covariance of theta (size (p+|X_n|) x (p+|X_n|)).
        sampled_set (list): Ordered list of visited solution tuples.
        sol_to_idx (dict): Mapping from solution tuple to deviation index.
    """

    def __init__(self, d, lambda_i=0.1, prior_var=100.0):
        self.d = d
        self.p = 2 * d + 1  # quadratic basis: (1, x1,...,xd, x1^2,...,xd^2)
        self.lambda_i = lambda_i

        # Prior: beta ~ N(0, prior_var * I_p)
        self.a = np.zeros(self.p)
        self.C = prior_var * np.eye(self.p)

        self.sampled_set = []   # ordered list of solution tuples
        self.sol_to_idx = {}    # solution tuple -> index in deviation part

    def basis(self, x):
        """Quadratic basis without cross-terms: (1, x1,...,xd, x1^2,...,xd^2).

        This gives p = 2d+1 basis functions, keeping dimensionality linear in d.
        See Eq. (quadratic_basis) in the paper.
        """
        x = np.asarray(x, dtype=float)
        return np.concatenate([[1.0], x, x ** 2])

    def augmented_feature(self, x):
        """Augmented feature vector tilde_phi(x) in R^{p + |X_n|}.

        If x has been visited: tilde_phi = (phi(x), e_x)  where e_x is unit vector.
        If x is new:           tilde_phi = (phi(x), 0)    (deviation not yet active).
        See Eq. (augmented_feature) in the paper.
        """
        x_tuple = tuple(x)
        phi = self.basis(x)
        n_sampled = len(self.sampled_set)
        if x_tuple in self.sol_to_idx:
            e = np.zeros(n_sampled)
            e[self.sol_to_idx[x_tuple]] = 1.0
            return np.concatenate([phi, e])
        else:
            return np.concatenate([phi, np.zeros(n_sampled)])

    def posterior_mean(self, x):
        """Posterior mean mu(x) = tilde_phi(x)^T * a.  See Eq. (posterior_mean)."""
        phi_tilde = self.augmented_feature(x)
        return float(phi_tilde @ self.a)

    def posterior_var(self, x):
        """Posterior variance of f(x).  See Eq. (posterior_var).

        For unvisited x, adds lambda_i for the prior uncertainty of zeta(x).
        """
        x_tuple = tuple(x)
        phi_tilde = self.augmented_feature(x)
        var = float(phi_tilde @ self.C @ phi_tilde)
        if x_tuple not in self.sol_to_idx:
            var += self.lambda_i
        return max(var, 1e-12)

    def dimension_augment(self, x):
        """Augment state when solution x is visited for the first time.

        Appends zeta(x) ~ N(0, lambda_i) to theta and expands C accordingly.
        See Eq. (augment_mean) in Proposition 3.1.
        """
        x_tuple = tuple(x)
        if x_tuple not in self.sol_to_idx:
            idx = len(self.sampled_set)
            self.sampled_set.append(x_tuple)
            self.sol_to_idx[x_tuple] = idx

            self.a = np.concatenate([self.a, [0.0]])
            n = len(self.a)
            C_new = np.zeros((n, n))
            C_new[:n - 1, :n - 1] = self.C
            C_new[n - 1, n - 1] = self.lambda_i
            self.C = C_new

    def update(self, x, y, sigma2_hat):
        """Bayesian rank-one Kalman update after observing Y=y at solution x.

        Implements Proposition 3.1 (posterior update):
            a_{n+1} = a_n + (C_n * e_tilde) / denom * (y - e_tilde^T * a_n)
            C_{n+1} = C_n - (C_n * e_tilde * e_tilde^T * C_n) / denom
        where denom = sigma2_hat + e_tilde^T * C_n * e_tilde.

        Args:
            x: Solution vector (array-like).
            y (float): Observed value Y^i.
            sigma2_hat (float): Estimated noise variance at x (from VEPM).
        """
        # Step (a): Dimension augmentation if x is new
        self.dimension_augment(x)

        # Step (b): Rank-one Kalman update
        e_tilde = self.augmented_feature(x)
        Ce = self.C @ e_tilde
        denom = sigma2_hat + e_tilde @ Ce
        if denom < 1e-15:
            return

        innovation = y - e_tilde @ self.a
        gain = Ce / denom

        self.a = self.a + gain * innovation
        self.C = self.C - np.outer(gain, Ce)
        self.C = 0.5 * (self.C + self.C.T)  # enforce symmetry


# =============================================================================
# 3. VEPM: Variance Estimation Parametric Model
#
#    Partition-based variance sharing scheme for heteroscedastic noise.
#    Key idea: solutions with similar features share similar variances.
#    Each dimension j is split into m_j bins; the partition combination
#    c(x) = (c_1(x_1), ..., c_d(x_d)) groups solutions with similar structure.
#
#    For visited solutions: individual weighted estimate (prior + residuals).
#    For unvisited solutions: use the common partition estimate.
#    See Section 3.3 and Lemma VEPM_recursive in the paper.
# =============================================================================

class VEPM:
    """Variance Estimation Parametric Model.

    Uses feature-based partition scheme: for each solution x, compute
    2d partition features [x_norm, x_norm^2] where x_norm is the normalized
    decision vector. Each feature is binned at threshold 0.5, yielding
    2^(2d) total partition combinations.

    Attributes:
        d (int): Decision dimension.
        L (int): Levels per dimension.
        w (float): Prior weight (pseudo-sample size for initial estimate).
        n_features (int): Number of partition features (2d).
        total_partitions (int): Total partition combinations (2^(2d)).
        partition_common (dict): (i, c) -> common variance estimate for partition c.
        sol_variance (dict): (i, x_tuple) -> individual variance estimate.
        sol_count (dict): x_tuple -> number of times solution was sampled.
        partition_sols (dict): c -> set of sampled solution tuples in partition.
        initial_estimates (dict): (i, c) -> initial variance from pre-sampling.
    """

    def __init__(self, d, L, w=5.0, normalize_func=None,
                 partition_method='binary_bin', K=None,
                 feature_indices=None,
                 adaptive_feature_selection=False,
                 adaptive_max_features=None,
                 adaptive_min_score=0.0,
                 shrinkage_kappa=0.0,
                 robust_update=False,
                 residual_clip_factor=None,
                 new_point_weight=1.0,
                 partition_weight_floor=0.0):
        """
        partition_method (paper-faithful, no auto-fallback by d):
            'binary_bin' (default, paper Sec.4.3): Zheng 2019 raw 2d-feature
                [x_norm, x_norm^2] binary-bin scheme.  Each feature is binned
                against threshold 0.5, giving 2^(2d) cells.  At d <= 4 this
                yields a manageable cardinality; at d >= 5 the cell count
                quickly exceeds typical sample budgets and VEPM enters the
                saturation regime (Corollary partition-saturation in the
                paper).
            'aggregate':  4-feature aggregate scheme using
                [mean, std, max, min] of the normalized x.  Cardinality is
                fixed at 2^4 = 16 regardless of d; intended as a high-d
                ablation that destroys the per-coordinate alignment of
                Zheng features.
            'medoid_K':  k-means clustering on the 2d-feature vector with K
                centroids fitted at initialize() time; partition_index(x)
                returns the nearest-centroid index.  K is auto-tuned to
                max(8, min(64, ceil(sqrt(4 * N0)))) from the pre-sample
                size when ``K=None``.
        """
        self.d = d
        self.L = L
        self.w = w
        self._normalize_func = normalize_func  # optional override from problem
        if feature_indices is None:
            feature_indices = tuple(range(d))
        self.feature_indices = tuple(int(j) for j in feature_indices)
        if not self.feature_indices:
            self.feature_indices = tuple(range(d))
        for j in self.feature_indices:
            if j < 0 or j >= d:
                raise ValueError(
                    f"VEPM feature index {j} is outside dimension d={d}")
        self.active_d = len(self.feature_indices)

        # Backward-compatibility alias: earlier code distinguished
        # 'binary_bin_raw' (forced raw Zheng, no fallback) from a
        # 'binary_bin' that auto-aggregated at d>4.  After the cleanup
        # 'binary_bin' itself is always raw, so 'binary_bin_raw' is the
        # same scheme and is accepted as an alias.
        if partition_method == 'binary_bin_raw':
            partition_method = 'binary_bin'
        if partition_method not in ('binary_bin', 'aggregate', 'medoid_K'):
            raise ValueError(f"unknown partition_method: {partition_method}")
        self.partition_method = partition_method
        self._K_param = K   # explicit K override; None => auto-tune

        if partition_method == 'binary_bin':
            # Paper's Eq. (partition_combination): Zheng raw 2d-feature
            # binary bins, no d-dependent fallback.  At d=2 -> 16 cells,
            # d=5 -> 1024 cells, d=44 -> 2^88 cells (saturation regime).
            self._use_aggregate = False
            self.n_features = 2 * self.active_d
            self.total_partitions = 2 ** self.n_features
            self._feature_thresholds = np.full(self.n_features, 0.5)
        elif partition_method == 'aggregate':
            # 4-feature aggregate scheme: cardinality 16 regardless of d.
            # Thresholds are calibrated to per-feature pre-sample medians
            # in _calibrate_thresholds (called from initialize()).
            self._use_aggregate = True
            self.n_features = 4
            self.total_partitions = 2 ** self.n_features
            self._feature_thresholds = np.full(self.n_features, 0.5)
        else:   # medoid_K
            self._use_aggregate = False
            self.n_features = 2 * self.active_d
            self._feature_thresholds = None    # not used in medoid_K mode
            self.total_partitions = K if K is not None else 0

        self._centroids = None   # populated for medoid_K after initialize()
        self.adaptive_feature_selection = bool(adaptive_feature_selection)
        self.adaptive_max_features = (
            None if adaptive_max_features is None
            else max(1, int(adaptive_max_features)))
        self.adaptive_min_score = max(float(adaptive_min_score), 0.0)
        self.adaptive_feature_scores = {}
        self.adaptive_feature_selected = tuple(self.feature_indices)
        self.shrinkage_kappa = max(float(shrinkage_kappa), 0.0)
        self.robust_update = bool(robust_update)
        self.residual_clip_factor = (
            None if residual_clip_factor is None
            else float(residual_clip_factor))
        self.new_point_weight = max(float(new_point_weight), 0.0)
        self.partition_weight_floor = max(float(partition_weight_floor), 0.0)

        self.partition_common = {}
        self.sol_variance = {}
        self.sol_resid_weight = {}
        self.sol_count = {}
        self.partition_sols = {}
        self.initial_estimates = {}
        self.partition_resid_count = {}
        self.global_var = {}

    def _normalize(self, x):
        """Normalize x to [0,1]^d using problem's normalize function if provided."""
        if self._normalize_func is not None:
            return self._normalize_func(np.asarray(x, dtype=float))
        x = np.asarray(x, dtype=float)
        if self.L > 1:
            return (x - 1.0) / (self.L - 1.0)
        return np.zeros_like(x)

    def _active_normalized(self, x):
        """Return normalized coordinates used by the variance partition."""
        x_norm = self._normalize(x)
        feature_indices = getattr(
            self, 'feature_indices', tuple(range(self.d)))
        return np.asarray(x_norm, dtype=float)[list(feature_indices)]

    def _features(self, x):
        """Compute the partition feature vector.

        Raw mode (low-d binary_bin / medoid_K): 2d-vector [x_norm, x_norm^2]
            clipped to [0, 0.9999].  This matches the paper's
            $\\phi^{\\mathrm{part}}(x)$ definition.
        Aggregate mode: [mean, std, max, min] of
            x_norm — scalar summaries that stay constant in d.
        """
        x_norm = self._active_normalized(x)
        if self._use_aggregate:
            if len(x_norm) == 0:
                return np.full(self.n_features, 0.5)
            return np.array([
                float(np.mean(x_norm)),
                float(np.std(x_norm)),
                float(np.max(x_norm)),
                float(np.min(x_norm)),
            ])
        feats = np.concatenate([x_norm, x_norm ** 2])
        return np.clip(feats, 0.0, 0.9999)

    def partition_index(self, x):
        """Map a solution x to its partition label.

        binary_bin mode: each feature is binned against its calibrated
            threshold (median of pre-sample features in aggregate mode;
            0.5 in raw mode).  Returns a tuple of n_features binary indices.
        medoid_K mode: returns the integer index of the nearest centroid
            (Euclidean) in the 2d-feature space, in {0,...,K-1}.  Falls
            back to binary_bin behaviour before centroids are fitted.
        """
        feats = self._features(x)
        if self.partition_method == 'medoid_K' and self._centroids is not None:
            d = np.linalg.norm(self._centroids - feats[None, :], axis=1)
            return int(np.argmin(d))
        bins = tuple(int(feats[k] > self._feature_thresholds[k])
                     for k in range(self.n_features))
        return bins

    def _calibrate_thresholds(self, pre_samples):
        """Set per-feature thresholds to the median of pre-sample features
        so the resulting partitions are roughly balanced.  Raw mode keeps
        the 0.5 default to preserve original Zheng 2019 behavior."""
        if not self._use_aggregate or not pre_samples:
            return
        feats = np.array([self._features(np.asarray(x)) for x in pre_samples])
        self._feature_thresholds = np.median(feats, axis=0)

    def _fit_medoid_K(self, pre_samples):
        """Fit K cluster centroids on the 2d-feature vectors of pre_samples.

        K is chosen so that the expected per-cell occupancy is at least ~10
        for the eventual budget N: K = max(8, min(64, ceil(sqrt(n_pre*4)))).
        For typical (n0, N) = (100, 400) this gives K = 20.  Override via
        the ``K`` argument to VEPM.__init__.

        Uses sklearn.cluster.KMeans if available; falls back to a minimal
        Lloyd's-algorithm implementation otherwise (deterministic seeded).
        """
        n_pre = max(1, len(pre_samples))
        if self._K_param is not None:
            K = int(self._K_param)
        else:
            K = max(8, min(64, int(np.ceil(np.sqrt(n_pre * 4)))))
        K = min(K, n_pre)   # can't have more clusters than samples
        self.total_partitions = K

        feats = np.array([self._features(np.asarray(x)) for x in pre_samples])

        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=K, n_init=10, random_state=0)
            km.fit(feats)
            self._centroids = km.cluster_centers_
            return
        except ImportError:
            pass

        # Fallback: minimal Lloyd's algorithm with k-means++ seeding.
        rng = np.random.default_rng(0)
        idx = [int(rng.integers(n_pre))]
        for _ in range(K - 1):
            d2 = np.min(((feats - feats[idx, None, :]) ** 2).sum(axis=2), axis=0)
            probs = d2 / max(d2.sum(), 1e-12)
            idx.append(int(rng.choice(n_pre, p=probs)))
        centroids = feats[idx].copy()
        for _ in range(50):
            assign = np.argmin(np.linalg.norm(
                feats[:, None, :] - centroids[None, :, :], axis=2), axis=1)
            new = np.array([
                feats[assign == k].mean(axis=0) if (assign == k).any() else centroids[k]
                for k in range(K)
            ])
            if np.allclose(new, centroids, atol=1e-9):
                break
            centroids = new
        self._centroids = centroids
        feats = np.array([self._features(x) for x in pre_samples])
        self._feature_thresholds = np.median(feats, axis=0)

    def _reset_binary_geometry_for_features(self, feature_indices):
        """Reset raw binary-bin geometry after adaptive feature screening."""
        feature_indices = tuple(int(j) for j in feature_indices)
        if not feature_indices:
            feature_indices = tuple(range(self.d))
        for j in feature_indices:
            if j < 0 or j >= self.d:
                raise ValueError(
                    f"VEPM feature index {j} is outside dimension d={self.d}")
        self.feature_indices = feature_indices
        self.active_d = len(feature_indices)
        if self.partition_method == 'binary_bin':
            self.n_features = 2 * self.active_d
            self.total_partitions = 2 ** self.n_features
            self._feature_thresholds = np.full(self.n_features, 0.5)
        elif self.partition_method == 'medoid_K':
            self.n_features = 2 * self.active_d
            self._centroids = None

    @staticmethod
    def _safe_r2_from_feature(values, target):
        """Return a robust one-dimensional association score in [0, 1]."""
        values = np.asarray(values, dtype=float)
        target = np.asarray(target, dtype=float)
        if len(values) < 3:
            return 0.0
        if np.std(values) <= 1e-12 or np.std(target) <= 1e-12:
            return 0.0
        corr = float(np.corrcoef(values, target)[0, 1])
        corr2 = 0.0 if not np.isfinite(corr) else corr ** 2

        threshold = float(np.median(values))
        left = target[values <= threshold]
        right = target[values > threshold]
        split_r2 = 0.0
        if len(left) > 0 and len(right) > 0:
            total = float(np.sum((target - np.mean(target)) ** 2))
            within = (
                float(np.sum((left - np.mean(left)) ** 2))
                + float(np.sum((right - np.mean(right)) ** 2)))
            if total > 1e-12:
                split_r2 = max(0.0, 1.0 - within / total)
        return float(max(corr2, split_r2))

    def _apply_adaptive_feature_selection(self, pre_samples, observations,
                                          gpr_models):
        """Select low-dimensional variance-relevant coordinates from data.

        The selector uses only pre-sample residuals computed with the current
        pre-update mean model.  It ranks coordinates by how well their
        normalized linear/quadratic values explain log squared residuals.
        """
        if not self.adaptive_feature_selection:
            return
        if self.partition_method not in ('binary_bin', 'medoid_K'):
            return
        if not pre_samples:
            return

        x_rows = []
        y_rows = []
        eps = 1e-12
        for x_tuple in pre_samples:
            obs_list = observations.get(x_tuple, [])
            if not obs_list:
                continue
            x_arr = np.asarray(x_tuple, dtype=float)
            resid2_vals = []
            for obs in obs_list:
                for i in range(3):
                    mu = float(gpr_models[i].posterior_mean(x_arr))
                    resid2_vals.append((float(obs[i]) - mu) ** 2)
            if resid2_vals:
                x_rows.append(self._normalize(x_arr))
                y_rows.append(np.log(float(np.mean(resid2_vals)) + eps))
        if len(y_rows) < 4:
            return

        X = np.asarray(x_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float)
        scores = {}
        candidate_indices = tuple(
            int(j) for j in getattr(self, 'feature_indices',
                                    tuple(range(self.d))))
        if not candidate_indices:
            candidate_indices = tuple(range(self.d))
        for j in candidate_indices:
            z = X[:, j]
            scores[j] = max(
                self._safe_r2_from_feature(z, y),
                self._safe_r2_from_feature(z ** 2, y))
        ordered = sorted(scores.items(), key=lambda item: item[1],
                         reverse=True)
        max_features = self.adaptive_max_features or min(2, self.d)
        selected = [
            j for j, score in ordered[:max_features]
            if score >= self.adaptive_min_score
        ]
        if not selected and ordered:
            selected = [ordered[0][0]]
        if selected:
            self._reset_binary_geometry_for_features(selected)
            self.adaptive_feature_selected = tuple(selected)
            self.adaptive_feature_scores = {
                int(j): float(score) for j, score in ordered
            }

    def _partition_total_weight(self, i, c):
        """Evidence weight available in a partition cell."""
        sols = self.partition_sols.get(c, set())
        return float(sum(self._partition_weight(i, s) for s in sols))

    def _shrink_variance(self, i, raw_variance, evidence_weight):
        """Shrink sparse local/cell variance estimates toward pooled variance."""
        raw_variance = max(float(raw_variance), 1e-12)
        kappa = max(float(getattr(self, 'shrinkage_kappa', 0.0)), 0.0)
        if kappa <= 0:
            return raw_variance
        target = max(float(self.global_var.get(i, raw_variance)), 1e-12)
        evidence_weight = max(float(evidence_weight), 0.0)
        weight = evidence_weight / (evidence_weight + kappa)
        return float(weight * raw_variance + (1.0 - weight) * target)

    def initialize(self, pre_samples, observations, gpr_models):
        """Initialize VEPM from pre-sampling data.

        Computes initial partition variance estimates from pre-sample squared
        residuals. For partitions with fewer than 2 residuals, uses the global
        mean squared residual as fallback.

        Args:
            pre_samples: List of solution tuples sampled in pre-sampling phase.
            observations: Dict x_tuple -> list of observation arrays [Y1,Y2,Y3].
            gpr_models: List of 3 ParametricGPR models (already updated with pre-samples).
        """
        # Calibrate the partition geometry BEFORE assigning anything to cells.
        self._apply_adaptive_feature_selection(
            pre_samples, observations, gpr_models)
        if self.partition_method == 'medoid_K':
            self._fit_medoid_K(pre_samples)
        else:
            self._calibrate_thresholds(pre_samples)

        partition_residuals = {}
        global_residuals = {i: [] for i in range(3)}

        for x_tuple in pre_samples:
            c = self.partition_index(x_tuple)
            x_arr = np.array(x_tuple)
            for i in range(3):
                mu = gpr_models[i].posterior_mean(x_arr)
                for obs in observations[x_tuple]:
                    resid2 = (obs[i] - mu) ** 2
                    key = (i, c)
                    if key not in partition_residuals:
                        partition_residuals[key] = []
                    partition_residuals[key].append(resid2)
                    global_residuals[i].append(resid2)

        self.global_var = {i: np.mean(global_residuals[i]) if global_residuals[i] else 0.01
                           for i in range(3)}

        for i in range(3):
            for key, resids in partition_residuals.items():
                if key[0] == i:
                    c = key[1]
                    self.partition_resid_count[(i, c)] = len(resids)
                    self.initial_estimates[(i, c)] = (
                        np.mean(resids) if len(resids) >= 2 else self.global_var[i])

        for x_tuple in set(pre_samples):
            c = self.partition_index(x_tuple)
            for i in range(3):
                init_est = self.initial_estimates.get((i, c), self.global_var[i])
                x_arr = np.array(x_tuple)
                mu = gpr_models[i].posterior_mean(x_arr)
                sum_resid = sum((obs[i] - mu) ** 2 for obs in observations[x_tuple])
                count = len(observations[x_tuple])
                self.sol_variance[(i, x_tuple)] = (self.w * init_est + sum_resid) / (self.w + count)
                self.sol_resid_weight[(i, x_tuple)] = float(count)
                self.sol_count[x_tuple] = count
                if c not in self.partition_sols:
                    self.partition_sols[c] = set()
                self.partition_sols[c].add(x_tuple)

        for c in self.partition_sols:
            for i in range(3):
                sols_in_c = self.partition_sols[c]
                total_weight = 0.0
                weighted_sum = 0.0
                for s in sols_in_c:
                    n_s = self._partition_weight(i, s)
                    weighted_sum += self.sol_variance.get((i, s), self.global_var[i]) * n_s
                    total_weight += n_s
                self.partition_common[(i, c)] = (
                    weighted_sum / total_weight if total_weight > 0 else self.global_var[i])

    def _partition_weight(self, i, x_tuple):
        """Evidence weight used when averaging solution variances in a cell."""
        fallback = float(self.sol_count.get(x_tuple, 1))
        weight = float(self.sol_resid_weight.get((i, x_tuple), fallback))
        if self.partition_weight_floor > 0:
            weight = max(weight, self.partition_weight_floor)
        return max(weight, 1e-12)

    def _robust_residual(self, i, x_tuple, c, resid2, is_new, old_var):
        """Return residual contribution and diagnostics for a VEPM update.

        The default path is the original recursive update.  When robust mode is
        enabled, one-shot residuals at newly sampled points can be downweighted
        and large residuals are clipped relative to the current cell/solution
        variance scale.  This protects finite-budget runs from treating
        surrogate mean misspecification as simulation noise.
        """
        resid2 = max(float(resid2), 0.0)
        ref_candidates = [
            old_var,
            self.partition_common.get((i, c), None),
            self.initial_estimates.get((i, c), None),
            self.global_var.get(i, None),
        ]
        ref_vals = [
            float(v) for v in ref_candidates
            if v is not None and np.isfinite(v) and float(v) > 0
        ]
        ref_var = max(ref_vals) if ref_vals else 0.01
        clipped = False
        clip_threshold = None
        resid_eff = resid2
        if self.robust_update and self.residual_clip_factor is not None:
            clip_threshold = max(
                float(self.residual_clip_factor) * ref_var, 1e-12)
            if resid_eff > clip_threshold:
                resid_eff = clip_threshold
                clipped = True
        if self.robust_update and is_new:
            update_weight = self.new_point_weight
        else:
            update_weight = 1.0
        update_weight = max(float(update_weight), 0.0)
        return resid_eff, update_weight, {
            "resid2_raw": float(resid2),
            "resid2_effective": float(resid_eff),
            "resid2_ref_var": float(ref_var),
            "resid2_clip_threshold": (
                None if clip_threshold is None else float(clip_threshold)),
            "resid2_clipped": bool(clipped),
            "resid2_update_weight": float(update_weight),
            "is_new_solution": bool(is_new),
        }

    def get_variance(self, i, x):
        """Get variance estimate for objective i at solution x.

        Visited solutions use their individual estimates.
        Unvisited solutions use the common partition estimate.

        If ``self._pooled_only`` is True (nV ablation mode), always
        return the pooled global variance regardless of x — replicating
        methods/gpr_kg_nv.py behaviour while keeping the VEPM object
        picklable (no method-level monkey-patching).
        """
        if getattr(self, '_pooled_only', False):
            return max(self.global_var.get(i, 0.01), 1e-8)
        x_tuple = tuple(x)
        if (i, x_tuple) in self.sol_variance:
            raw = self.sol_variance[(i, x_tuple)]
            evidence = self.sol_resid_weight.get(
                (i, x_tuple), self.sol_count.get(x_tuple, 1))
            return max(self._shrink_variance(i, raw, evidence), 1e-8)
        c = self.partition_index(x)
        if (i, c) in self.partition_common:
            raw = self.partition_common[(i, c)]
            return max(self._shrink_variance(
                i, raw, self._partition_total_weight(i, c)), 1e-8)
        if (i, c) in self.initial_estimates:
            raw = self.initial_estimates[(i, c)]
            evidence = self.partition_resid_count.get((i, c), 0)
            return max(self._shrink_variance(i, raw, evidence), 1e-8)
        return max(self.global_var.get(i, 0.01), 1e-8)

    def update(self, i, x, y, mu, gpr_model):
        """VEPM recursive update for objective i after observing Y=y at x.

        Step (a): Update individual variance estimate (Eq. VEPM_rec_a/c).
        Step (b): Recompute partition common variance as sample-count weighted
                  average of all individual estimates in the partition.

        Args:
            i (int): Objective index (0, 1, or 2).
            x: Solution vector.
            y (float): Observed value Y^i.
            mu (float): Current posterior mean mu^i(x).
            gpr_model: The ParametricGPR model for objective i.
        """
        x_tuple = tuple(x)
        c = self.partition_index(x)
        resid2 = (y - mu) ** 2

        if c not in self.partition_sols:
            self.partition_sols[c] = set()

        old_count = self.sol_count.get(x_tuple, 0)
        new_count = old_count + 1
        is_new = old_count == 0

        old_weight = float(
            self.sol_resid_weight.get((i, x_tuple), float(old_count)))
        old_var = self.sol_variance.get((i, x_tuple), self.get_variance(i, x))
        resid_eff, update_weight, details = self._robust_residual(
            i, x_tuple, c, resid2, is_new, old_var)
        new_weight = old_weight + update_weight

        if not is_new:
            # Step (a) revisit: Eq. (VEPM_rec_a)
            if update_weight > 0:
                new_var = ((self.w + old_weight) * old_var
                           + update_weight * resid_eff) / (self.w + new_weight)
            else:
                new_var = old_var
            self.sol_variance[(i, x_tuple)] = new_var
        else:
            # Step (a) new solution: Eq. (VEPM_rec_c)
            init_est = self.initial_estimates.get((i, c), self.global_var.get(i, 0.01))
            if update_weight > 0:
                new_var = (self.w * init_est + update_weight * resid_eff) / (
                    self.w + update_weight)
            else:
                new_var = init_est
            self.sol_variance[(i, x_tuple)] = new_var
            self.partition_sols[c].add(x_tuple)
        self.sol_resid_weight[(i, x_tuple)] = new_weight

        # Step (b): Recompute partition common variance as sample-count
        # weighted average over all solutions in the same partition.
        total_weight = 0.0
        weighted_sum = 0.0
        for s in self.partition_sols[c]:
            n_s = self._partition_weight(i, s)
            weighted_sum += self.sol_variance.get((i, s), self.global_var.get(i, 0.01)) * n_s
            total_weight += n_s
        if total_weight > 0:
            self.partition_common[(i, c)] = weighted_sum / total_weight

        # Update sample count once per vector-valued simulation observation.
        # The algorithm calls update() sequentially for objectives i=0,1,2;
        # delaying the shared count update until i=2 keeps all three variance
        # channels on the same pre-update visit count.
        if i == 2:
            self.sol_count[x_tuple] = new_count
        details.update({
            "objective_index": int(i),
            "old_sample_count": int(old_count),
            "new_sample_count": int(new_count),
            "old_resid_weight": float(old_weight),
            "new_resid_weight": float(new_weight),
            "old_variance": float(old_var),
            "new_variance": float(self.sol_variance[(i, x_tuple)]),
            "partition_index": c,
            "partition_common": float(
                self.partition_common.get((i, c), self.global_var.get(i, 0.01))),
        })
        return details


class RidgeLogVarianceSurrogate:
    """Lightweight surrogate for a smooth log-variance field.

    The model fits ridge regression to log(residual^2) using normalized
    decision-space features.  It stabilizes VEPM in sparse partitions without
    replacing the partition estimator asymptotically.
    """

    def __init__(self, d, alpha=1e-3, floor=1e-8, min_samples=20):
        self.d = int(d)
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.min_samples = int(min_samples)
        self.records = {i: [] for i in range(3)}
        self.coef = {i: None for i in range(3)}
        self.fitted_n = {i: 0 for i in range(3)}

    def _features(self, x, problem):
        x_norm = np.asarray(problem.normalize(np.asarray(x, dtype=float)),
                            dtype=float)
        center = np.full_like(x_norm, 0.5)
        stats = np.array([
            float(np.mean(x_norm)),
            float(np.std(x_norm)),
            float(np.min(x_norm)),
            float(np.max(x_norm)),
            float(np.linalg.norm(x_norm - center) / np.sqrt(max(1, self.d))),
        ])
        return np.concatenate([[1.0], x_norm, x_norm ** 2, stats])

    def add(self, i, x, resid2):
        self.records[int(i)].append((
            tuple(int(v) for v in np.asarray(x, dtype=int)),
            float(max(resid2, self.floor)),
        ))

    def fit(self, problem):
        for i in range(3):
            recs = self.records[i]
            if len(recs) < self.min_samples:
                continue
            X = np.vstack([self._features(x, problem) for x, _ in recs])
            y = np.log(np.array([r for _, r in recs], dtype=float))
            reg = self.alpha * np.eye(X.shape[1])
            reg[0, 0] = 0.0
            try:
                beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X.T @ X + reg, X.T @ y,
                                       rcond=None)[0]
            self.coef[i] = beta
            self.fitted_n[i] = len(recs)

    def predict(self, i, x, problem):
        beta = self.coef.get(int(i))
        if beta is None:
            return None
        z = self._features(x, problem)
        return float(max(np.exp(float(z @ beta)), self.floor))

    def diagnostics(self):
        return {
            str(i): {
                "n_records": int(len(self.records[i])),
                "fitted_n": int(self.fitted_n[i]),
                "is_fitted": self.coef[i] is not None,
            }
            for i in range(3)
        }


# =============================================================================
# 4. KG Factor Computation
#
#    The h-function: h(a, b) = E[max_j(a_j + b_j Z)] - max_j a_j, Z ~ N(0,1)
#    is computed using the sort-based algorithm of Frazier & Powell (2009).
#
#    For minimization, we use: nu^KG = h(-mu, -sigma_tilde).
#    See Lemma KG_distribution and Algorithm 3 (h-function) in the paper.
# =============================================================================

def compute_h(a, b):
    """Compute h(a, b) = E[max_j(a_j + b_j Z)] - max_j a_j, Z ~ N(0,1).

    Implements the sort-and-prune algorithm of Frazier & Powell (2009).
    Complexity: O(M log M) where M = len(a).

    Steps:
        1. Sort by slopes b.
        2. Remove dominated alternatives (lower envelope pruning).
        3. Compute crossover points between adjacent lines.
        4. Integrate the piecewise-linear expectation analytically.

    Args:
        a (np.array): Intercepts of length M.
        b (np.array): Slopes of length M.

    Returns:
        float: h(a, b) >= 0.
    """
    M = len(a)
    if M == 0:
        return 0.0

    # Step 1: Sort by slope
    idx = np.argsort(b)
    a_sorted = a[idx].copy()
    b_sorted = b[idx].copy()

    # Step 2: Remove dominated alternatives
    keep = []
    for j in range(M):
        dominated = False
        for k in range(M):
            if k == j:
                continue
            if a_sorted[k] >= a_sorted[j] and b_sorted[k] >= b_sorted[j]:
                if a_sorted[k] > a_sorted[j] or b_sorted[k] > b_sorted[j]:
                    dominated = True
                    break
        if not dominated:
            keep.append(j)

    if len(keep) <= 1:
        return 0.0

    a_k = a_sorted[keep]
    b_k = b_sorted[keep]

    # Upper envelope construction via convex hull pruning
    stack = [0]
    for j in range(1, len(a_k)):
        while len(stack) >= 2:
            j1 = stack[-2]
            j2 = stack[-1]
            if b_k[j] == b_k[j2]:
                if a_k[j] >= a_k[j2]:
                    stack.pop()
                else:
                    break
            else:
                z_cross = ((a_k[j2] - a_k[j1]) / (b_k[j1] - b_k[j2])
                           if b_k[j1] != b_k[j2] else -np.inf)
                z_new = ((a_k[j] - a_k[j2]) / (b_k[j2] - b_k[j])
                         if b_k[j2] != b_k[j] else np.inf)
                if z_new <= z_cross:
                    stack.pop()
                else:
                    break
        stack.append(j)

    if len(stack) <= 1:
        return 0.0

    a_f = a_k[stack]
    b_f = b_k[stack]

    # Step 3: Compute crossover points
    n = len(a_f)
    z_cross = np.zeros(n - 1)
    for j in range(n - 1):
        db = b_f[j] - b_f[j + 1]
        z_cross[j] = (a_f[j + 1] - a_f[j]) / db if abs(db) > 1e-15 else -np.inf

    # Step 4: Integrate analytically
    # h = sum_j [ a_j * P(segment_j) + b_j * (phi(z_lo) - phi(z_hi)) ] - max(a)
    h_val = 0.0
    for j in range(n):
        z_lo = -np.inf if j == 0 else z_cross[j - 1]
        z_hi = np.inf if j == n - 1 else z_cross[j]

        p = norm.cdf(z_hi) - norm.cdf(z_lo)
        if p < 1e-15:
            continue

        phi_lo = norm.pdf(z_lo) if not np.isinf(z_lo) else 0.0
        phi_hi = norm.pdf(z_hi) if not np.isinf(z_hi) else 0.0
        h_val += a_f[j] * p + b_f[j] * (phi_lo - phi_hi)

    h_val -= np.max(a)
    return max(h_val, 0.0)


def compute_kg_factor(gpr, candidate_set, x, sigma2_hat):
    """Compute KG factor nu^{KG,i}(x) for one objective.

    Uses Lemma KG_distribution:
        sigma_tilde_{x'}(x) = phi(x')^T C e_x / sqrt(sigma2 + e_x^T C e_x)
        nu^KG = h(-mu, -sigma_tilde)

    Args:
        gpr (ParametricGPR): Belief model for objective i.
        candidate_set (list): List of candidate solution arrays.
        x (np.array): Solution being evaluated for sampling.
        sigma2_hat (float): Estimated noise variance at x.

    Returns:
        float: KG factor >= 0.
    """
    n_cand = len(candidate_set)
    if n_cand == 0:
        return 0.0

    try:
        e_tilde = gpr.augmented_feature(x)
        Ce = gpr.C @ e_tilde
        denom_val = sigma2_hat + float(e_tilde @ Ce)
        denom_sqrt = np.sqrt(max(denom_val, 1e-15))

        mu_vec = np.array([gpr.posterior_mean(xp) for xp in candidate_set])
        sigma_tilde = np.array([
            float(gpr.augmented_feature(xp) @ Ce) / denom_sqrt
            for xp in candidate_set
        ])

        return compute_h(-mu_vec, -sigma_tilde)
    except Exception:
        return 0.0


# =============================================================================
# 5. Pareto and Hypervolume Utilities
# =============================================================================

def pareto_filter(points, return_indices=False):
    """Return Pareto-optimal points under minimization.

    A point p is Pareto-optimal if no other point q satisfies
    q_i <= p_i for all i and q_j < p_j for some j.

    Args:
        points (np.array): Shape (n, m) array of objective values.
        return_indices (bool): If True, also return indices of Pareto points.

    Returns:
        Pareto-optimal points array, and optionally their indices.
    """
    if len(points) == 0:
        empty = np.empty((0, 2))
        return (empty, np.array([], dtype=int)) if return_indices else empty

    points = np.array(points)
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_pareto[i] = False
                break

    if return_indices:
        return points[is_pareto], np.where(is_pareto)[0]
    return points[is_pareto]


def crowding_distance_select(kg_pairs):
    """Select solution with largest crowding distance from KG pairs.

    Used as tie-breaking rule among non-dominated solutions in KG-factor space.
    Solutions at the boundary (extreme KG values) get infinite distance.
    """
    if len(kg_pairs) == 1:
        return 0

    n = len(kg_pairs)
    distances = np.zeros(n)

    for obj in range(2):
        vals = kg_pairs[:, obj]
        sorted_idx = np.argsort(vals)
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
        if val_range < 1e-15:
            continue

        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / val_range

    return np.argmax(distances)


def compute_hypervolume_2d(points, ref_point):
    """Compute hypervolume indicator for 2D minimization.

    The hypervolume is the area of the region dominated by the Pareto front
    and bounded above by the reference point.

    Args:
        points (np.array): Shape (n, 2) Pareto front points.
        ref_point (np.array): Shape (2,) reference point.

    Returns:
        float: Hypervolume value.
    """
    if len(points) == 0:
        return 0.0

    points = np.array(points)
    mask = np.all(points < ref_point, axis=1)
    points = points[mask]
    if len(points) == 0:
        return 0.0

    pf = pareto_filter(points)
    if len(pf) == 0:
        return 0.0
    pf = pf[np.argsort(pf[:, 0])]

    hv = 0.0
    for k in range(len(pf)):
        x_hi = ref_point[0] if k == len(pf) - 1 else pf[k + 1, 0]
        y_lo = pf[k, 1]
        if x_hi > pf[k, 0] and ref_point[1] > y_lo:
            hv += (x_hi - pf[k, 0]) * (ref_point[1] - y_lo)

    return hv


# =============================================================================
# 6. Complete GPR-KG Algorithm with Full Intermediate Logging
#
#    The algorithm records detailed per-iteration data for post-hoc analysis:
#    - Sampling decisions (which solution was selected and why)
#    - Simulation observations
#    - KG factor values
#    - Posterior Pareto front snapshots
#    - Hypervolume trajectory
#    - Computation time breakdown per iteration
# =============================================================================


class _PosteriorBiObjProblem(PymooProblem):
    """Bi-objective problem on sampled posterior for candidate generation.

    Given sampled parametric coefficients bb[0], bb[1] (and bb[2] for
    constraint), maps each continuous search point to the integer decision
    grid and evaluates obj_i(x) = round(Psi(x) @ bb[i] * 100) / 100
    (2-decimal rounding matching MATLAB bi_obj.m).
    """

    def __init__(self, bb_param, p, d, L, to_int_func,
                 tau_e=None, alpha_z=None, variance_lookup=None):
        n_ieq = 1 if tau_e is not None else 0
        super().__init__(n_var=d, n_obj=2, n_ieq_constr=n_ieq,
                         xl=np.zeros(d), xu=np.ones(d))
        self.bb_param = bb_param
        self.p = p
        self.to_int_func = to_int_func
        self.tau_e = tau_e
        self.alpha_z = alpha_z
        self.variance_lookup = variance_lookup

    def _basis_matrix(self, X):
        """Compute [1, x, x^2] feature matrix on the GPR integer scale."""
        N, d = X.shape
        Psi = np.ones((N, 2 * d + 1))
        Psi[:, 1:d+1] = X
        Psi[:, d+1:] = X ** 2
        return Psi

    def _evaluate(self, X, out, *args, **kwargs):
        X_int = np.array([self.to_int_func(row) for row in X], dtype=float)
        Psi = self._basis_matrix(X_int)
        f1 = np.round(Psi @ self.bb_param[0] * 100) / 100.0
        f2 = np.round(Psi @ self.bb_param[1] * 100) / 100.0
        out["F"] = np.column_stack([f1, f2])
        if self.tau_e is not None:
            f3 = Psi @ self.bb_param[2]
            sigma3 = np.array([
                np.sqrt(max(self.variance_lookup(tuple(int(v) for v in row)), 1e-8))
                for row in X_int
            ])
            out["G"] = (f3 - self.tau_e + self.alpha_z * sigma3).reshape(-1, 1)


def _pareto_front_indices(obj):
    """Return indices of non-dominated rows in obj (N x 2, minimization)."""
    N = len(obj)
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        for j in range(N):
            if i == j or not is_pareto[j]:
                continue
            if np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


class GPRKR_Algorithm:
    """GPR-KG: Complete algorithm with comprehensive intermediate result logging.

    All intermediate results are stored in self.iteration_log, a list of
    dictionaries (one per iteration), each containing:

        {
            'iteration': int,          # iteration index (0-based from main loop start)
            'stage': int,              # absolute stage index n (including pre-sampling)

            # --- Timing breakdown (seconds) ---
            't_posterior_solve': float, # time to solve posterior optimization problem
            't_candidate_gen': float,  # time to generate candidate set
            't_kg_compute': float,     # time to compute KG factors for all candidates
            't_simulate': float,       # time to run simulation
            't_belief_update': float,  # time for GPR posterior update (3 objectives)
            't_vepm_update': float,    # time for VEPM variance update (3 objectives)
            't_hv_eval': float,        # time for hypervolume evaluation (0 if skipped)
            't_total': float,          # total wall-clock time for this iteration

            # --- Sampling decision ---
            'x_selected': tuple,       # solution selected for simulation
            'is_new_solution': bool,   # whether x was visited for the first time
            'n_candidates': int,       # size of candidate set |A_n|
            'n_pareto_kg': int,        # number of non-dominated solutions in KG space

            # --- Observations ---
            'Y_observed': list,        # [Y1, Y2, Y3] simulation outputs

            # --- KG factors of selected solution ---
            'kg1_selected': float,     # KG factor for objective 1 of selected solution
            'kg2_selected': float,     # KG factor for objective 2 of selected solution

            # --- Posterior state ---
            'n_visited': int,          # number of distinct solutions visited so far
            'theta_dim': int,          # current dimension of augmented parameter vector

            # --- Performance snapshot (every 10 iterations + final) ---
            'hv': float or None,       # hypervolume indicator (None if not evaluated)
            'pareto_set_size': int or None,  # |PF| of estimated Pareto set
            'pareto_front': list or None,    # [(f1,f2),...] estimated PF objective values
        }

    Additionally:
        self.pre_sampling_log: dict with pre-sampling phase details
        self.final_log: dict with final solution details
        self.hv_history: list of (stage, hv) tuples for convergence plotting
        self.history: list of (x_tuple, Y_array) for all observations
        self.observations: dict x_tuple -> list of Y arrays
    """

    def _resolve_partition_features(self, partition_features):
        """Resolve VEPM partition feature indices.

        The theory requires the variance partition to be aligned with the
        features governing the noise field.  For registered benchmarks,
        ``auto`` uses ``problem.recommended_partition_features``; ``all`` keeps
        the historical full-coordinate partition for sensitivity analysis.
        """
        if partition_features is None:
            partition_features = 'auto'
        if isinstance(partition_features, str):
            value = partition_features.strip().lower()
            if value == 'auto':
                features = getattr(
                    self.problem, 'recommended_partition_features', None)
                if features is None:
                    features = getattr(self.problem, 'variance_features', None)
                if features is None:
                    features = tuple(range(self.d))
            elif value in ('all', 'full'):
                features = tuple(range(self.d))
            else:
                features = tuple(
                    int(part.strip()) for part in value.split(',')
                    if part.strip())
        else:
            features = tuple(int(j) for j in partition_features)
        if not features:
            features = tuple(range(self.d))
        bad = [j for j in features if j < 0 or j >= self.d]
        if bad:
            raise ValueError(
                f"invalid partition feature indices {bad} for d={self.d}")
        return tuple(features)

    def __init__(self, problem, N=150, n0=30, K1=50, K2=2,
                 lambda_i=0.1, prior_var=100.0, w_vepm=1.0,
                 n_thr=20, seed=None,
                 partition_method='binary_bin', partition_K=None,
                 partition_features='auto',
                 use_boundary_initial_design=True,
                 initial_samples=None,
                 use_archive_candidates=False,
                 archive_neighbor_radius=0,
                 kg_selection_tiebreak='crowding_distance',
                 variance_shrinkage_rho0=0.0,
                 variance_floor=1e-8,
                 variance_surrogate='none',
                 variance_surrogate_rho0=0.0,
                 variance_surrogate_alpha=1e-3,
                 variance_surrogate_min_samples=20,
                 variance_surrogate_only_constraint=False,
                 variance_surrogate_clip_low=0.5,
                 variance_surrogate_clip_high=2.0,
                 robust_vepm=False,
                 vepm_residual_clip_factor=None,
                 vepm_new_point_weight=1.0,
                 vepm_partition_weight_floor=0.0,
                 adaptive_vepm=False,
                 adaptive_vepm_max_features=2,
                 adaptive_vepm_min_score=0.0,
                 vepm_shrinkage_kappa=0.0,
                 variance_mode='vepm',
                 replication_policy='none',
                 replication_max_per_solution=3,
                 replication_score_threshold=5e-4,
                 replication_boundary_scale=1.0,
                 replication_budget_fraction=1.0,
                 boundary_candidate_policy='none',
                 boundary_candidate_count=0,
                 boundary_candidate_pool_size=500,
                 boundary_candidate_margin_scale=1.0,
                 boundary_candidate_feasibility_buffer=0.0,
                 boundary_acquisition_weight=0.0,
                 boundary_acquisition_margin_scale=1.0,
                 boundary_acquisition_decay_power=0.0,
                 exploration_epsilon0=0.0,
                 exploration_epsilon_min=0.0,
                 exploration_decay_power=1.0):
        """Initialize the GPR-KG algorithm.

        Args:
            problem (TestProblem): Test problem instance with calibrated constraint.
            N (int): Total simulation budget (including pre-sampling).
            n0 (int): Number of pre-sampling solutions (each simulated once).
            K1 (int): Number of LHD random candidates.
            K2 (int): Number of posterior sampling + NSGA-II iterations.
            lambda_i (float): Prior variance for deviation terms zeta(x).
            prior_var (float): Prior variance for beta coefficients.
            w_vepm (float): VEPM prior weight (pseudo-sample size).
            n_thr (int): Threshold for activating constraint in candidate gen.
            seed (int or None): Random seed for reproducibility.
            partition_method (str): VEPM partition scheme. One of:
                'binary_bin' (default): Zheng 2019 raw 2d-feature scheme,
                    with no dimension-dependent fallback.
                'aggregate': 4-feature aggregate ablation.
                'medoid_K': k-means clustering with K=O(sqrt(N)) centroids.
            partition_K (int or None): override for K when partition_method
                is 'medoid_K'.  Default None auto-tunes from pre-sample size.
            partition_features: 'auto', 'all', comma-separated indices, or an
                iterable of coordinate indices. 'auto' uses the benchmark's
                recommended variance features, e.g. (0,) for the RZDT suite.
            replication_policy (str): Optional adaptive replication rule.
                'none' preserves the original pure exploration behavior.
                'boundary' may replicate visited solutions near the posterior
                chance-constraint boundary.
            replication_max_per_solution (int): Maximum total observations
                allowed at any replicated solution under the boundary rule.
            replication_score_threshold (float): Minimum boundary score needed
                to switch from the KG exploration candidate to replication.
            replication_boundary_scale (float): Scale multiplier in the
                chance-boundary proximity score.
            replication_budget_fraction (float): Maximum fraction of the
                adaptive budget that may be spent on replication.  The default
                value of 1.0 preserves the historical unbounded policy.
            boundary_candidate_policy (str): Optional new-point boundary
                exploration rule.  'none' preserves the original candidate
                set.  'chance_margin' adds unvisited finite-grid points whose
                posterior chance margin is close to zero.  'chance_feasible'
                only adds the posterior-feasible side of that boundary.
            boundary_candidate_count (int): Maximum number of boundary
                candidates to add when the policy is enabled.
            boundary_candidate_pool_size (int): Number of random unvisited
                finite-grid points screened for boundary candidates.
            boundary_candidate_margin_scale (float): Scale multiplier used to
                normalize the absolute posterior chance margin.
            boundary_candidate_feasibility_buffer (float): For
                ``chance_feasible`` candidates, require
                ``mu_3(x) + z_(1-alpha) sigma_3(x) - tau <=
                -buffer * sigma_3(x)``.  Zero preserves the plain posterior
                feasible-side rule.
            robust_vepm (bool): Enable finite-sample robustification of VEPM
                residual updates.
            vepm_residual_clip_factor (float or None): If set, clip squared
                residuals at this multiple of the current local/cell variance
                scale before updating VEPM.
            vepm_new_point_weight (float): Fractional residual weight for the
                first observation at a newly sampled point. Revisits keep
                weight 1.0.
            vepm_partition_weight_floor (float): Minimum evidence weight when
                averaging solution-level variances into a partition.
            adaptive_vepm (bool): If True, select a small set of
                variance-relevant coordinates from pre-sample residuals before
                constructing VEPM partitions.
            vepm_shrinkage_kappa (float): Pseudo-count for shrinking sparse
                VEPM cell/solution variance estimates toward pooled
                pre-sample variance.
            boundary_acquisition_weight (float): Additive chance-boundary
                score weight in Pareto-KG selection. Zero preserves the
                historical KG rule.
            exploration_epsilon0 (float): Initial probability of drawing a
                fully random finite-grid solution. Zero preserves the
                historical deterministic acquisition path.
        """
        self.problem = problem
        self.d = problem.d
        self.L = problem.L
        self.N = N
        self.n0 = n0
        self.K1 = K1
        self.K2 = K2
        self.lambda_i = lambda_i
        self.prior_var = prior_var
        self.w_vepm = w_vepm
        self.n_thr = n_thr
        self.seed = seed
        self.partition_method = partition_method
        self.partition_K = partition_K
        self.partition_features_mode = partition_features
        self.partition_features = self._resolve_partition_features(
            partition_features)
        self.use_boundary_initial_design = bool(use_boundary_initial_design)
        self.initial_samples = (
            [tuple(int(v) for v in x) for x in initial_samples]
            if initial_samples is not None else None)
        self.use_archive_candidates = bool(use_archive_candidates)
        self.archive_neighbor_radius = int(archive_neighbor_radius or 0)
        self.kg_selection_tiebreak = kg_selection_tiebreak
        self.variance_shrinkage_rho0 = float(variance_shrinkage_rho0 or 0.0)
        self.variance_floor = float(variance_floor or 0.0)
        self.variance_surrogate = str(variance_surrogate or 'none')
        self.variance_surrogate_rho0 = float(
            variance_surrogate_rho0 or 0.0)
        self.variance_surrogate_alpha = float(
            variance_surrogate_alpha or 1e-3)
        self.variance_surrogate_min_samples = int(
            variance_surrogate_min_samples or 20)
        self.variance_surrogate_only_constraint = bool(
            variance_surrogate_only_constraint)
        self.variance_surrogate_clip_low = float(
            variance_surrogate_clip_low or 0.5)
        self.variance_surrogate_clip_high = float(
            variance_surrogate_clip_high or 2.0)
        self.robust_vepm = bool(robust_vepm)
        self.vepm_residual_clip_factor = (
            None if vepm_residual_clip_factor is None
            else float(vepm_residual_clip_factor))
        self.vepm_new_point_weight = max(float(vepm_new_point_weight), 0.0)
        self.vepm_partition_weight_floor = max(
            float(vepm_partition_weight_floor), 0.0)
        self.adaptive_vepm = bool(adaptive_vepm)
        self.adaptive_vepm_max_features = max(
            1, int(adaptive_vepm_max_features or 1))
        self.adaptive_vepm_min_score = max(
            float(adaptive_vepm_min_score or 0.0), 0.0)
        self.vepm_shrinkage_kappa = max(float(vepm_shrinkage_kappa or 0.0),
                                        0.0)
        self.variance_mode = str(variance_mode or 'vepm')
        valid_variance_modes = {'vepm', 'pooled_pre', 'oracle'}
        if self.variance_mode not in valid_variance_modes:
            raise ValueError(
                f"unknown variance_mode {self.variance_mode!r}; "
                f"expected one of {sorted(valid_variance_modes)}")
        self.pooled_pre_variance = {}
        self.replication_policy = str(replication_policy or 'none')
        self.replication_max_per_solution = int(
            replication_max_per_solution or 1)
        self.replication_score_threshold = float(
            replication_score_threshold
            if replication_score_threshold is not None else 5e-4)
        self.replication_boundary_scale = float(
            replication_boundary_scale or 1.0)
        self.replication_budget_fraction = max(
            0.0, float(replication_budget_fraction
                       if replication_budget_fraction is not None else 1.0))
        self.boundary_candidate_policy = str(
            boundary_candidate_policy or 'none')
        self.boundary_candidate_count = max(
            0, int(boundary_candidate_count or 0))
        self.boundary_candidate_pool_size = max(
            0, int(boundary_candidate_pool_size or 0))
        self.boundary_candidate_margin_scale = max(
            float(boundary_candidate_margin_scale or 1.0), 1e-8)
        self.boundary_candidate_feasibility_buffer = max(
            float(boundary_candidate_feasibility_buffer or 0.0), 0.0)
        self.boundary_acquisition_weight = max(
            float(boundary_acquisition_weight or 0.0), 0.0)
        self.boundary_acquisition_margin_scale = max(
            float(boundary_acquisition_margin_scale or 1.0), 1e-8)
        self.boundary_acquisition_decay_power = max(
            float(boundary_acquisition_decay_power or 0.0), 0.0)
        self.exploration_epsilon0 = max(
            float(exploration_epsilon0 or 0.0), 0.0)
        self.exploration_epsilon_min = max(
            float(exploration_epsilon_min or 0.0), 0.0)
        self.exploration_decay_power = max(
            float(exploration_decay_power or 1.0), 0.0)
        self._last_boundary_candidate_log = {
            "policy": self.boundary_candidate_policy,
            "n_screened": 0,
            "n_added": 0,
        }
        self.var_surrogate_model = (
            RidgeLogVarianceSurrogate(
                self.d,
                alpha=self.variance_surrogate_alpha,
                floor=max(self.variance_floor, 1e-12),
                min_samples=self.variance_surrogate_min_samples)
            if self.variance_surrogate == 'ridge_logvar' else None)

        if seed is not None:
            np.random.seed(seed)

        # Three GPR models: one per objective/constraint
        self.gpr = [ParametricGPR(self.d, lambda_i, prior_var) for _ in range(3)]

        # Shared VEPM instance with the chosen partition scheme.
        _norm_func = lambda x: problem.normalize(np.asarray(x, dtype=float))
        self.vepm = VEPM(self.d, self.L, w_vepm, normalize_func=_norm_func,
                         partition_method=partition_method, K=partition_K,
                         feature_indices=self.partition_features,
                         adaptive_feature_selection=self.adaptive_vepm,
                         adaptive_max_features=(
                             self.adaptive_vepm_max_features),
                         adaptive_min_score=(
                             self.adaptive_vepm_min_score),
                         shrinkage_kappa=self.vepm_shrinkage_kappa,
                         robust_update=self.robust_vepm,
                         residual_clip_factor=(
                             self.vepm_residual_clip_factor),
                         new_point_weight=self.vepm_new_point_weight,
                         partition_weight_floor=(
                             self.vepm_partition_weight_floor))

        # ---- Data storage ----
        self.observations = {}    # x_tuple -> list of [Y1, Y2, Y3] arrays
        self.history = []         # chronological list of (x_tuple, Y_array)
        self.hv_history = []      # list of (stage, hv) for convergence plot

        # ---- Intermediate result logs ----
        self.iteration_log = []       # per-iteration detailed log (main loop)
        self.pre_sampling_log = None  # pre-sampling phase summary
        self.final_log = None         # final solution summary

        # ---- Optional instrumentation (set by run_instrumented.py) ----
        # When `instrument` is not None it must be a dict with keys
        #   'eval_x'   : list of tuples, points at which to record posterior
        #                means for RMSE computation
        #   'ref_x'    : tuple, reference point for variance tracking
        #   'stride'   : int, record every `stride` iterations
        # Recorded snapshots are appended to self.instrument_log.
        self.instrument = None
        self.instrument_log = []

        # ---- Optional checkpointing (used by InTAS long runs) --------------
        # When _checkpoint_path is set, save_checkpoint() pickles the whole
        # algorithm state after every main-loop iteration, enabling resume
        # from the last completed iteration after an external kill.
        # When _snapshot_jsonl_path is set, each iteration's iter_log is
        # appended to the JSONL file for lightweight streaming observation.
        self._checkpoint_path      = None
        self._snapshot_jsonl_path  = None
        # Marks whether pre-sampling (Phase 1) has completed.  Used by
        # run_resumable() to skip pre-sampling on restart.
        self._presampling_done     = False
        # Number of main-loop iterations completed so far (0-based count).
        # Runs from 0 up to (N - n0).
        self._main_iter_completed  = 0

    # ---- Pickle compatibility: exclude runtime-only attributes ------------
    _UNPICKLABLE_ATTRS = ('problem', 'instrument')

    def __getstate__(self):
        """Exclude attributes that hold references to C extensions (libsumo)
        or user-supplied callbacks (lambda), so the algorithm state can be
        pickled independently of the simulator / instrumentation context.

        Also clears ``vepm._normalize_func`` because it is a lambda closure
        over ``self.problem`` (unpicklable); it is re-attached in
        ``restore_checkpoint``.
        """
        state = self.__dict__.copy()
        for k in self._UNPICKLABLE_ATTRS:
            state[k] = None
        # Deep-copy vepm into a shallow container whose normalize_func is None
        if state.get('vepm') is not None:
            import copy as _copy
            vepm_copy = _copy.copy(state['vepm'])
            vepm_copy._normalize_func = None
            state['vepm'] = vepm_copy
        return state

    def save_checkpoint(self, path=None):
        """Atomically pickle the full algorithm state to ``path`` (or
        ``self._checkpoint_path`` if path is None).  Safe to call from the
        main loop; on Windows, rename is atomic within the same volume."""
        import pickle as _pickle
        import os as _os
        target = path or self._checkpoint_path
        if target is None:
            return
        tmp = target + '.tmp'
        with open(tmp, 'wb') as f:
            _pickle.dump(self, f, protocol=_pickle.HIGHEST_PROTOCOL)
        _os.replace(tmp, target)

    @classmethod
    def restore_checkpoint(cls, path, problem, instrument=None):
        """Load a pickled algorithm state and re-attach the problem and
        (optional) instrumentation dict.  Also re-attaches the VEPM's
        ``_normalize_func`` lambda (cleared by ``__getstate__``).

        Returns an algorithm instance ready to resume via
        ``run_resumable()``.
        """
        import pickle as _pickle
        with open(path, 'rb') as f:
            alg = _pickle.load(f)
        alg.problem = problem
        alg.instrument = instrument
        if not hasattr(alg, 'variance_mode'):
            alg.variance_mode = 'vepm'
        if not hasattr(alg, 'pooled_pre_variance'):
            alg.pooled_pre_variance = {}
        if not hasattr(alg, 'robust_vepm'):
            alg.robust_vepm = False
        if not hasattr(alg, 'vepm_residual_clip_factor'):
            alg.vepm_residual_clip_factor = None
        if not hasattr(alg, 'vepm_new_point_weight'):
            alg.vepm_new_point_weight = 1.0
        if not hasattr(alg, 'vepm_partition_weight_floor'):
            alg.vepm_partition_weight_floor = 0.0
        if not hasattr(alg, 'adaptive_vepm'):
            alg.adaptive_vepm = False
        if not hasattr(alg, 'adaptive_vepm_max_features'):
            alg.adaptive_vepm_max_features = 2
        if not hasattr(alg, 'adaptive_vepm_min_score'):
            alg.adaptive_vepm_min_score = 0.0
        if not hasattr(alg, 'vepm_shrinkage_kappa'):
            alg.vepm_shrinkage_kappa = 0.0
        if not hasattr(alg, 'replication_budget_fraction'):
            alg.replication_budget_fraction = 1.0
        if not hasattr(alg, 'boundary_candidate_policy'):
            alg.boundary_candidate_policy = 'none'
        if not hasattr(alg, 'boundary_candidate_count'):
            alg.boundary_candidate_count = 0
        if not hasattr(alg, 'boundary_candidate_pool_size'):
            alg.boundary_candidate_pool_size = 0
        if not hasattr(alg, 'boundary_candidate_margin_scale'):
            alg.boundary_candidate_margin_scale = 1.0
        if not hasattr(alg, 'boundary_candidate_feasibility_buffer'):
            alg.boundary_candidate_feasibility_buffer = 0.0
        if not hasattr(alg, 'boundary_acquisition_weight'):
            alg.boundary_acquisition_weight = 0.0
        if not hasattr(alg, 'boundary_acquisition_margin_scale'):
            alg.boundary_acquisition_margin_scale = 1.0
        if not hasattr(alg, 'boundary_acquisition_decay_power'):
            alg.boundary_acquisition_decay_power = 0.0
        if not hasattr(alg, 'exploration_epsilon0'):
            alg.exploration_epsilon0 = 0.0
        if not hasattr(alg, 'exploration_epsilon_min'):
            alg.exploration_epsilon_min = 0.0
        if not hasattr(alg, 'exploration_decay_power'):
            alg.exploration_decay_power = 1.0
        if not hasattr(alg, '_last_boundary_candidate_log'):
            alg._last_boundary_candidate_log = {
                "policy": alg.boundary_candidate_policy,
                "n_screened": 0,
                "n_added": 0,
            }
        # Re-attach VEPM's normalize_func (same lambda as in __init__)
        if alg.vepm is not None:
            alg.vepm._normalize_func = (
                lambda x: problem.normalize(np.asarray(x, dtype=float)))
            if not hasattr(alg.vepm, 'robust_update'):
                alg.vepm.robust_update = bool(alg.robust_vepm)
            if not hasattr(alg.vepm, 'residual_clip_factor'):
                alg.vepm.residual_clip_factor = alg.vepm_residual_clip_factor
            if not hasattr(alg.vepm, 'new_point_weight'):
                alg.vepm.new_point_weight = alg.vepm_new_point_weight
            if not hasattr(alg.vepm, 'partition_weight_floor'):
                alg.vepm.partition_weight_floor = (
                    alg.vepm_partition_weight_floor)
            if not hasattr(alg.vepm, 'adaptive_feature_selection'):
                alg.vepm.adaptive_feature_selection = False
            if not hasattr(alg.vepm, 'adaptive_feature_scores'):
                alg.vepm.adaptive_feature_scores = {}
            if not hasattr(alg.vepm, 'adaptive_feature_selected'):
                alg.vepm.adaptive_feature_selected = tuple(
                    getattr(alg.vepm, 'feature_indices',
                            tuple(range(getattr(alg.vepm, 'd', 0)))))
            if not hasattr(alg.vepm, 'partition_resid_count'):
                alg.vepm.partition_resid_count = {}
            if not hasattr(alg.vepm, 'shrinkage_kappa'):
                alg.vepm.shrinkage_kappa = alg.vepm_shrinkage_kappa
            if not hasattr(alg.vepm, 'sol_resid_weight'):
                alg.vepm.sol_resid_weight = {
                    (i, x): float(alg.vepm.sol_count.get(x, 1))
                    for (i, x) in getattr(alg.vepm, 'sol_variance', {})
                }
        return alg

    def _append_snapshot_jsonl(self, record):
        """Append one record (dict) to the snapshot JSONL, if enabled."""
        if self._snapshot_jsonl_path is None:
            return
        try:
            import json as _json
            with open(self._snapshot_jsonl_path, 'a') as f:
                f.write(_json.dumps(record, default=str) + '\n')
        except Exception as e:
            # Never let instrumentation kill a long run
            print(f"[warn] snapshot JSONL append failed: {e}", flush=True)

    def random_solution(self):
        """Sample a solution uniformly at random from the problem's integer grid."""
        return self.problem.sample_random()

    def _boundary_solutions(self):
        """Problem-independent structured seeds on an integer box.

        These points use only the design-space bounds, not objective or
        constraint values. They provide a small structured finite-grid initial
        design (center plus coordinate-axis endpoints) while preserving the
        random-grid component of the pre-sampling phase.
        """
        lo, hi = self.problem.int_bounds()
        lo = np.asarray(lo, dtype=int)
        hi = np.asarray(hi, dtype=int)
        center = np.round((lo + hi) / 2.0).astype(int)
        seeds = {tuple(lo), tuple(hi), tuple(center)}
        for j in range(self.d):
            x_hi = lo.copy()
            x_hi[j] = hi[j]
            seeds.add(tuple(x_hi))
            x_mid = lo.copy()
            x_mid[j] = center[j]
            seeds.add(tuple(x_mid))
        return list(seeds)

    def _neighbor_solutions(self, x_tuple, radius=None):
        """Coordinate-wise integer neighbors clipped to the decision bounds."""
        radius = self.archive_neighbor_radius if radius is None else int(radius)
        if radius <= 0:
            return []
        lo, hi = self.problem.int_bounds()
        lo = np.asarray(lo, dtype=int)
        hi = np.asarray(hi, dtype=int)
        x0 = np.asarray(x_tuple, dtype=int)
        neigh = set()
        for j in range(self.d):
            for step in range(1, radius + 1):
                for sign in (-1, 1):
                    x = x0.copy()
                    x[j] = int(np.clip(x[j] + sign * step, lo[j], hi[j]))
                    neigh.add(tuple(x))
        neigh.discard(tuple(x0))
        return list(neigh)

    def _effective_variance(self, i, x):
        """Effective observation-noise variance for objective/constraint i.

        ``variance_mode='vepm'`` keeps the current VEPM path.  ``pooled_pre``
        freezes the pre-sample pooled residual variance, and ``oracle`` uses
        the benchmark's true variance function for upper-bound diagnostics.
        """
        return self._effective_variance_with_details(i, x)[0]

    def _freeze_pooled_pre_variance(self):
        """Freeze pooled pre-sample residual variances after VEPM init."""
        global_var = getattr(self.vepm, 'global_var', {}) or {}
        self.pooled_pre_variance = {
            int(i): max(float(global_var.get(i, 0.01)),
                        max(self.variance_floor, 1e-12))
            for i in range(3)
        }

    def _pooled_pre_variance(self, i):
        vals = getattr(self, 'pooled_pre_variance', {}) or {}
        if int(i) in vals:
            return max(float(vals[int(i)]), max(self.variance_floor, 1e-12))
        if hasattr(self.vepm, 'global_var'):
            return max(float(self.vepm.global_var.get(i, 0.01)),
                       max(self.variance_floor, 1e-12))
        return max(0.01, max(self.variance_floor, 1e-12))

    def _oracle_variance(self, i, x):
        sigma = np.asarray(self.problem.true_sigma(np.asarray(x)), dtype=float)
        return max(float(sigma[int(i)] ** 2), max(self.variance_floor, 1e-12))

    def _effective_variance_with_details(self, i, x):
        """Return effective variance plus diagnostic components."""
        mode = getattr(self, 'variance_mode', 'vepm')
        if mode == 'oracle':
            v_oracle = self._oracle_variance(i, x)
            return v_oracle, {
                "mode": "oracle",
                "oracle": float(v_oracle),
            }
        if mode == 'pooled_pre':
            v_pre = self._pooled_pre_variance(i)
            return v_pre, {
                "mode": "pooled_pre",
                "pooled_pre": float(v_pre),
            }

        v_local = float(self.vepm.get_variance(i, x))
        v_base = max(v_local, self.variance_floor)
        n_obs = max(1, len(self.history))
        rho_shrink = 0.0
        if self.variance_shrinkage_rho0 > 0:
            rho_shrink = min(1.0, self.variance_shrinkage_rho0 / np.sqrt(n_obs))
            v_global = float(self.vepm.global_var.get(i, v_local))
            v_base = (1.0 - rho_shrink) * v_local + rho_shrink * v_global
            v_base = max(v_base, self.variance_floor)

        v_sur = None
        rho_sur = 0.0
        v_eff = v_base
        if (self.var_surrogate_model is not None
                and self.variance_surrogate_rho0 > 0):
            v_sur = self.var_surrogate_model.predict(i, x, self.problem)
            if v_sur is not None:
                if self.variance_surrogate_only_constraint and int(i) != 2:
                    v_sur = None
                else:
                    lo = self.variance_surrogate_clip_low * v_base
                    hi = self.variance_surrogate_clip_high * v_base
                    v_sur = float(np.clip(v_sur, lo, hi))
            if v_sur is not None:
                rho_sur = min(0.8, self.variance_surrogate_rho0
                              / np.sqrt(n_obs))
                v_eff = (1.0 - rho_sur) * v_base + rho_sur * v_sur
        return max(v_eff, self.variance_floor), {
            "mode": "vepm",
            "vepm": float(v_local),
            "base": float(v_base),
            "surrogate": None if v_sur is None else float(v_sur),
            "rho_shrink": float(rho_shrink),
            "rho_surrogate": float(rho_sur),
        }

    def _seed_variance_surrogate(self, pre_samples):
        if self.var_surrogate_model is None:
            return
        for x_tuple in pre_samples:
            x_arr = np.asarray(x_tuple, dtype=int)
            for i in range(3):
                y_val = float(self.observations[x_tuple][0][i])
                mu_val = float(self.gpr[i].posterior_mean(x_arr))
                self.var_surrogate_model.add(
                    i, x_arr, (y_val - mu_val) ** 2)
        self.var_surrogate_model.fit(self.problem)

    def _update_variance_surrogate(self, x_arr, Y, mu_before):
        if self.var_surrogate_model is None:
            return
        for i in range(3):
            self.var_surrogate_model.add(
                i, x_arr, (float(Y[i]) - float(mu_before[i])) ** 2)
        self.var_surrogate_model.fit(self.problem)

    def _observation_count(self, x_tuple):
        return len(self.observations.get(tuple(int(v) for v in x_tuple), []))

    def _replication_candidate_score(self, x_tuple, pareto_lookup):
        """Score a visited solution for theory-compatible replication.

        The score is high when the solution is close to the posterior
        chance-constraint boundary and still has non-negligible local noise.
        It uses only posterior quantities and VEPM/nV variance estimates, so
        it does not leak true objective or constraint values.
        """
        x_tuple = tuple(int(v) for v in x_tuple)
        count = self._observation_count(x_tuple)
        if count <= 0 or count >= self.replication_max_per_solution:
            return None

        x_arr = np.asarray(x_tuple, dtype=int)
        v3 = max(float(self._effective_variance(2, x_arr)),
                 max(self.variance_floor, 1e-12))
        sig3 = float(np.sqrt(v3))
        q_alpha = norm.ppf(1 - self.problem.alpha)
        mu3 = float(self.gpr[2].posterior_mean(x_arr))
        margin = float(mu3 + q_alpha * sig3 - self.problem.tau)
        scale = max(self.replication_boundary_scale * sig3, 1e-8)
        boundary_weight = float(np.exp(-abs(margin) / scale))
        pareto_weight = 2.0 if x_tuple in pareto_lookup else 1.0
        score = pareto_weight * boundary_weight * v3 / (1.0 + count)
        return {
            "x": [int(v) for v in x_tuple],
            "score": float(score),
            "count": int(count),
            "mu3": float(mu3),
            "sigma3": float(sig3),
            "chance_margin": float(margin),
            "boundary_weight": float(boundary_weight),
            "pareto_weight": float(pareto_weight),
        }

    def _select_replication_candidate(self, pareto_set):
        if self.replication_policy != 'boundary':
            return None, None
        if self.replication_max_per_solution <= 1:
            return None, None

        pareto_lookup = {
            tuple(int(v) for v in x_tuple) for x_tuple in (pareto_set or [])
        }
        best = None
        for x_tuple in self.observations:
            info = self._replication_candidate_score(x_tuple, pareto_lookup)
            if info is None:
                continue
            if best is None or info["score"] > best["score"]:
                best = info
        if best is None:
            return None, None
        return tuple(best["x"]), best

    def _replication_budget_cap(self):
        adaptive_budget = max(0, int(self.N) - int(self.n0))
        if adaptive_budget <= 0:
            return 0
        return int(np.floor(self.replication_budget_fraction * adaptive_budget))

    def _replication_events_used(self):
        return int(sum(
            1 for log in getattr(self, 'iteration_log', [])
            if bool(log.get('selected_by_replication', False))))

    def _apply_replication_policy(self, candidate_set, candidate_arrays,
                                  selected_idx, kg_pairs, pareto_set,
                                  iteration=None):
        """Return the final sampling point after optional replication."""
        kg_x = tuple(int(v) for v in candidate_set[selected_idx])
        kg1 = float(kg_pairs[selected_idx, 0])
        kg2 = float(kg_pairs[selected_idx, 1])
        rep_x, rep_info = self._select_replication_candidate(pareto_set)
        used_reps = self._replication_events_used()
        rep_cap = self._replication_budget_cap()
        budget_available = used_reps < rep_cap
        score_passed = (
            rep_x is not None
            and rep_info is not None
            and rep_info["score"] >= self.replication_score_threshold)
        use_replication = (
            rep_x is not None
            and rep_info is not None
            and score_passed
            and budget_available)
        final_x = tuple(rep_x) if use_replication else kg_x
        eps = self._exploration_probability(iteration)
        selected_by_exploration = False
        if eps > 0 and np.random.rand() < eps:
            final_x = tuple(int(v) for v in self.random_solution())
            use_replication = False
            selected_by_exploration = True

        final_arr = np.asarray(final_x, dtype=int)
        if use_replication or selected_by_exploration:
            kg1_final = compute_kg_factor(
                self.gpr[0], candidate_arrays, final_arr,
                self._effective_variance(0, final_arr))
            kg2_final = compute_kg_factor(
                self.gpr[1], candidate_arrays, final_arr,
                self._effective_variance(1, final_arr))
        else:
            kg1_final = kg1
            kg2_final = kg2

        selection_log = {
            "replication_policy": self.replication_policy,
            "selected_by_replication": bool(use_replication),
            "replication_score_threshold": float(
                self.replication_score_threshold),
            "replication_budget_fraction": float(
                self.replication_budget_fraction),
            "replication_budget_cap": int(rep_cap),
            "replication_events_used_before": int(used_reps),
            "replication_budget_available": bool(budget_available),
            "replication_score_passed": bool(score_passed),
            "x_kg_selected": [int(v) for v in kg_x],
            "kg_selected_index": int(selected_idx),
            "kg1_kg_selected": float(kg1),
            "kg2_kg_selected": float(kg2),
            "best_replication_candidate": rep_info,
            "kg1_final_selected": float(kg1_final),
            "kg2_final_selected": float(kg2_final),
            "exploration_epsilon": float(eps),
            "selected_by_exploration": bool(selected_by_exploration),
        }
        return final_x, selection_log

    def _exploration_probability(self, iteration):
        """Probability of a fully random finite-grid exploration step."""
        eps0 = float(getattr(self, 'exploration_epsilon0', 0.0))
        eps_min = float(getattr(self, 'exploration_epsilon_min', 0.0))
        if eps0 <= 0.0 and eps_min <= 0.0:
            return 0.0
        t = max(1.0, float((0 if iteration is None else iteration) + 1))
        power = max(float(getattr(self, 'exploration_decay_power', 1.0)), 0.0)
        eps = eps0 / (t ** power) if eps0 > 0.0 else 0.0
        return float(np.clip(max(eps, eps_min), 0.0, 1.0))

    def _boundary_acquisition_effective_weight(self, iteration):
        """Return the iteration-specific chance-boundary acquisition weight."""
        weight0 = float(getattr(self, 'boundary_acquisition_weight', 0.0))
        if weight0 <= 0.0:
            return 0.0
        if iteration is not None and iteration <= self.n_thr:
            return 0.0
        t = max(1.0, float(
            (0 if iteration is None else iteration) - self.n_thr + 1))
        power = max(
            float(getattr(self, 'boundary_acquisition_decay_power', 0.0)),
            0.0)
        return float(weight0 / (t ** power))

    def _boundary_acquisition_scores(self, candidate_set, iteration):
        """Score candidates by chance-boundary proximity and variance scale.

        The score is intentionally posterior-only: it uses the current GPR
        mean and effective observation variance, never true objective values.
        """
        weight_eff = self._boundary_acquisition_effective_weight(iteration)
        log = {
            "enabled": bool(weight_eff > 0.0),
            "base_weight": float(getattr(
                self, 'boundary_acquisition_weight', 0.0)),
            "effective_weight": float(weight_eff),
            "margin_scale": float(getattr(
                self, 'boundary_acquisition_margin_scale', 1.0)),
            "decay_power": float(getattr(
                self, 'boundary_acquisition_decay_power', 0.0)),
            "max_score": 0.0,
            "mean_score": 0.0,
            "best_index": None,
            "best_margin": None,
        }
        if weight_eff <= 0.0 or len(candidate_set) == 0:
            return np.zeros(len(candidate_set), dtype=float), log

        q_alpha = norm.ppf(1 - self.problem.alpha)
        raw_scores = []
        margins = []
        for x_tuple in candidate_set:
            x_arr = np.asarray(x_tuple, dtype=int)
            mu3 = float(self.gpr[2].posterior_mean(x_arr))
            v3 = max(float(self._effective_variance(2, x_arr)),
                     max(self.variance_floor, 1e-12))
            sig3 = float(np.sqrt(v3))
            margin = float(mu3 + q_alpha * sig3 - self.problem.tau)
            scale = max(self.boundary_acquisition_margin_scale * sig3, 1e-8)
            proximity = float(np.exp(-0.5 * (margin / scale) ** 2))
            raw_scores.append(proximity * sig3)
            margins.append(margin)
        scores = np.asarray(raw_scores, dtype=float)
        if np.max(scores) > 0:
            scores = scores / float(np.max(scores))
        best = int(np.argmax(scores)) if len(scores) else None
        log.update({
            "max_score": float(np.max(scores)) if len(scores) else 0.0,
            "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
            "best_index": best,
            "best_margin": (
                None if best is None else float(margins[best])),
        })
        return scores, log

    def _posterior_quality_select(self, candidate_set, pareto_kg_idx, kg_pairs):
        """Select within the KG-efficient set using posterior solution quality.

        The KG non-dominated set is still the primary selection rule.  This
        tie-break prefers candidates with lower posterior objective values and
        a safer posterior chance-constraint margin.
        """
        if len(pareto_kg_idx) == 0:
            return int(np.random.randint(len(candidate_set)))
        if self.kg_selection_tiebreak != 'posterior_quality':
            nd_kg = kg_pairs[pareto_kg_idx]
            local_idx = crowding_distance_select(nd_kg)
            return int(pareto_kg_idx[local_idx])

        q_alpha = norm.ppf(1 - self.problem.alpha)
        scores = []
        for idx in pareto_kg_idx:
            x_arr = np.array(candidate_set[idx])
            mu1 = self.gpr[0].posterior_mean(x_arr)
            mu2 = self.gpr[1].posterior_mean(x_arr)
            mu3 = self.gpr[2].posterior_mean(x_arr)
            sig3 = np.sqrt(self._effective_variance(2, x_arr))
            margin = mu3 + q_alpha * sig3 - self.problem.tau
            feasibility_penalty = max(0.0, margin)
            scores.append((mu1 + mu2 + feasibility_penalty, idx))
        scores.sort(key=lambda z: z[0])
        return int(scores[0][1])

    def _solve_posterior_problem(self):
        """Solve the posterior optimization problem (Proposition 3.2).

        Evaluates posterior means at all visited solutions + random sample,
        filters by posterior feasibility, and returns Pareto-optimal set.

        Returns:
            list of solution tuples forming the estimated Pareto set.
        """
        candidates = set()
        for x_tuple in self.gpr[0].sampled_set:
            candidates.add(x_tuple)
        for _ in range(500):
            candidates.add(self.random_solution())

        candidates = list(candidates)
        feasible_objs = []
        feasible_sols = []
        q_alpha = norm.ppf(1 - self.problem.alpha)

        for x_tuple in candidates:
            x_arr = np.array(x_tuple)
            mu1 = self.gpr[0].posterior_mean(x_arr)
            mu2 = self.gpr[1].posterior_mean(x_arr)
            mu3 = self.gpr[2].posterior_mean(x_arr)
            sigma3 = np.sqrt(self._effective_variance(2, x_arr))

            if mu3 + q_alpha * sigma3 <= self.problem.tau:
                feasible_objs.append([mu1, mu2])
                feasible_sols.append(x_tuple)

        if len(feasible_objs) == 0:
            return []

        feasible_objs = np.array(feasible_objs)
        _, pareto_idx = pareto_filter(feasible_objs, return_indices=True)
        return [feasible_sols[i] for i in pareto_idx]

    def _boundary_candidate_solutions(self, existing_candidates, iteration):
        """Return unvisited candidates near the posterior chance boundary.

        This augments the candidate set only.  It does not force selection and
        does not replicate old points; the KG-Pareto rule still chooses the
        final sampling decision.
        """
        log = {
            "policy": self.boundary_candidate_policy,
            "iteration": int(iteration),
            "n_screened": 0,
            "n_added": 0,
            "skipped_observed": 0,
            "skipped_duplicate": 0,
            "skipped_posterior_infeasible": 0,
            "skipped_buffer_unsafe": 0,
            "feasibility_buffer": float(
                getattr(self, 'boundary_candidate_feasibility_buffer', 0.0)),
            "best_abs_scaled_margin": None,
            "worst_added_abs_scaled_margin": None,
        }
        if self.boundary_candidate_policy == 'none':
            self._last_boundary_candidate_log = log
            return []
        if self.boundary_candidate_policy not in (
                'chance_margin', 'chance_feasible'):
            self._last_boundary_candidate_log = log
            return []
        if self.boundary_candidate_count <= 0:
            self._last_boundary_candidate_log = log
            return []
        if self.boundary_candidate_pool_size <= 0:
            self._last_boundary_candidate_log = log
            return []
        if iteration <= self.n_thr:
            self._last_boundary_candidate_log = log
            return []

        existing = {tuple(int(v) for v in x) for x in existing_candidates}
        observed = set(self.observations)
        seen = set(existing)
        scored = []
        q_alpha = norm.ppf(1 - self.problem.alpha)
        attempts = 0
        max_attempts = max(1000, 10 * self.boundary_candidate_pool_size)

        while (log["n_screened"] < self.boundary_candidate_pool_size
               and attempts < max_attempts):
            attempts += 1
            x_tuple = tuple(int(v) for v in self.random_solution())
            if x_tuple in observed:
                log["skipped_observed"] += 1
                continue
            if x_tuple in seen:
                log["skipped_duplicate"] += 1
                continue
            seen.add(x_tuple)
            x_arr = np.asarray(x_tuple, dtype=int)
            mu1 = float(self.gpr[0].posterior_mean(x_arr))
            mu2 = float(self.gpr[1].posterior_mean(x_arr))
            mu3 = float(self.gpr[2].posterior_mean(x_arr))
            sig3 = float(np.sqrt(max(
                self._effective_variance(2, x_arr),
                max(self.variance_floor, 1e-12))))
            margin = float(mu3 + q_alpha * sig3 - self.problem.tau)
            log["n_screened"] += 1
            target_margin = 0.0
            if self.boundary_candidate_policy == 'chance_feasible':
                if margin > 0.0:
                    log["skipped_posterior_infeasible"] += 1
                    continue
                buffer = float(getattr(
                    self, 'boundary_candidate_feasibility_buffer', 0.0))
                target_margin = -buffer * sig3
                if margin > target_margin:
                    log["skipped_buffer_unsafe"] += 1
                    continue
            scaled_margin = abs(margin - target_margin) / max(
                self.boundary_candidate_margin_scale * sig3, 1e-8)
            obj_quality = mu1 + mu2
            scored.append((scaled_margin, obj_quality, x_tuple))

        scored.sort(key=lambda z: (z[0], z[1]))
        chosen = scored[:self.boundary_candidate_count]
        if chosen:
            log["best_abs_scaled_margin"] = float(chosen[0][0])
            log["worst_added_abs_scaled_margin"] = float(chosen[-1][0])
        log["n_added"] = int(len(chosen))
        self._last_boundary_candidate_log = log
        return [x for _scaled, _obj, x in chosen]

    def _generate_candidate_set(self, pareto_set, iteration):
        """Generate candidate set A_n = A_LHD ∪ A_post.

        Part 1 (A_LHD): K1 Latin Hypercube Design samples, rounded to
            the integer grid {1,...,L}^d then normalized to [0,1]^d.
        Part 2 (A_post): K2 iterations of posterior coefficient sampling +
            NSGA-II on the sampled posterior objectives, yielding Pareto-
            optimal solutions that exploit the current belief structure.

        Args:
            pareto_set: Current estimated Pareto set (unused by LHD+NSGA-II
                approach, kept for interface compatibility).
            iteration: Current iteration count (for constraint activation).

        Returns:
            list of solution tuples.
        """
        p = 2 * self.d + 1  # number of basis features
        candidates = set()

        # Part 1: K1 LHD samples on integer grid
        try:
            sampler = qmc.LatinHypercube(d=self.d, seed=np.random.randint(100000))
            lhd = sampler.random(self.K1)  # (K1, d) in [0,1]
        except Exception:
            lhd = np.random.rand(self.K1, self.d)
        # Convert to problem's integer grid using problem's continuous_to_int
        for row in lhd:
            candidates.add(self.problem.continuous_to_int(row))

        if self.use_archive_candidates:
            archive = set(pareto_set or [])
            for x_tuple in archive:
                candidates.add(tuple(x_tuple))
                for x_nb in self._neighbor_solutions(x_tuple):
                    candidates.add(x_nb)

        # Part 2: K2 posterior-sampled NSGA-II runs
        use_constraint = (iteration > self.n_thr)

        for _ in range(self.K2):
            # Sample posterior parametric coefficients
            bb_param = []
            for i in range(3):
                b_param = self.gpr[i].a[:p].copy()
                B_param = self.gpr[i].C[:p, :p].copy()
                B_param = (B_param + B_param.T) / 2
                eigvals = np.linalg.eigvalsh(B_param)
                if np.min(eigvals) < 0:
                    B_param -= 1.1 * np.min(eigvals) * np.eye(p)
                try:
                    theta_i = np.random.multivariate_normal(b_param, B_param)
                except np.linalg.LinAlgError:
                    theta_i = b_param + np.random.randn(p) * 0.01
                bb_param.append(theta_i)

            # Build NSGA-II problem on sampled posterior
            if use_constraint:
                problem = _PosteriorBiObjProblem(
                    bb_param, p, self.d, self.L,
                    to_int_func=self.problem.continuous_to_int,
                    tau_e=self.problem.tau,
                    alpha_z=norm.ppf(1 - self.problem.alpha),
                    variance_lookup=lambda x: self._effective_variance(2, x))
            else:
                problem = _PosteriorBiObjProblem(
                    bb_param, p, self.d, self.L,
                    to_int_func=self.problem.continuous_to_int)

            algorithm = NSGA2(pop_size=100)
            try:
                res = pymoo_minimize(
                    problem, algorithm,
                    get_termination("n_gen", 50),
                    seed=int(np.random.randint(100000)),
                    verbose=False)
                if res.X is not None:
                    X_result = res.X
                    if X_result.ndim == 1:
                        X_result = X_result.reshape(1, -1)
                    # Convert from [0,1]^d to problem's integer grid
                    for row in X_result:
                        candidates.add(self.problem.continuous_to_int(row))
            except Exception:
                # Fallback: random search on sampled posterior
                X_rand = np.random.rand(500, self.d)
                X_int = np.array([
                    self.problem.continuous_to_int(row)
                    for row in X_rand
                ], dtype=float)
                Phi = self._basis_matrix(X_int)
                obj = np.column_stack([
                    np.round(Phi @ bb_param[0] * 100) / 100.0,
                    np.round(Phi @ bb_param[1] * 100) / 100.0])
                pf_idx = _pareto_front_indices(obj)
                for idx in pf_idx:
                    candidates.add(self.problem.continuous_to_int(X_rand[idx]))

        for x_tuple in self._boundary_candidate_solutions(
                candidates, iteration):
            candidates.add(tuple(x_tuple))

        return list(candidates)

    def _basis_matrix(self, X_values):
        """Compute basis feature matrix on the GPR input scale.

        Features: [1, x_1, ..., x_d, x_1^2, ..., x_d^2] of shape (N, 2d+1).
        The main GPR model is trained on integer-grid decision vectors, so
        posterior candidate evaluations first map continuous search points to
        the integer grid before calling this helper.
        """
        N = len(X_values)
        Phi = np.ones((N, 2 * self.d + 1))
        Phi[:, 1:self.d+1] = X_values
        Phi[:, self.d+1:] = X_values ** 2
        return Phi

    def _evaluate_hv(self, use_true_objectives=True):
        """Evaluate current hypervolume indicator.

        Args:
            use_true_objectives (bool): If True, compute HV on true objective
                values (for correct performance evaluation). If False, use
                posterior means (for algorithm-internal tracking).

        Returns:
            tuple: (hv_value, pareto_set_size, pareto_front_points)
        """
        pareto_est = self._solve_posterior_problem()
        if len(pareto_est) > 0:
            if use_true_objectives:
                objs = np.array([
                    self.problem.true_objectives(np.array(x))[:2]
                    for x in pareto_est
                ])
            else:
                objs = np.array([
                    [self.gpr[0].posterior_mean(np.array(x)),
                     self.gpr[1].posterior_mean(np.array(x))]
                    for x in pareto_est
                ])
            pf = pareto_filter(objs)
            hv = compute_hypervolume_2d(pf, self.problem.ref_point)
            return hv, len(pf), pf.tolist()
        return 0.0, 0, []

    def run(self, verbose=True):
        """Run the complete GPR-KG algorithm with full intermediate logging.

        This method executes:
            Phase 1: Pre-sampling (n0 random solutions, each simulated once)
            Phase 2: Main loop (N - n0 sequential iterations with KG-guided sampling)
            Phase 3: Final posterior problem solve

        All timing and intermediate results are saved to self.iteration_log,
        self.pre_sampling_log, and self.final_log.

        Args:
            verbose (bool): If True, print progress updates.

        Returns:
            list: Estimated Pareto-optimal solution set (list of tuples).
        """

        # =================================================================
        # Phase 1: Pre-sampling
        # =================================================================
        t_pre_start = time.time()

        if verbose:
            print(f"Pre-sampling phase: {self.n0} solutions...")

        pre_sample_set = set()
        if self.initial_samples is not None:
            for x_tuple in self.initial_samples:
                if len(pre_sample_set) >= self.n0:
                    break
                pre_sample_set.add(tuple(int(v) for v in x_tuple))
        if self.use_boundary_initial_design:
            for x_tuple in self._boundary_solutions():
                if len(pre_sample_set) >= self.n0:
                    break
                pre_sample_set.add(tuple(x_tuple))
        while len(pre_sample_set) < self.n0:
            pre_sample_set.add(self.random_solution())
        pre_samples = list(pre_sample_set)

        # Simulate each pre-sample once
        pre_obs_list = []
        for x_tuple in pre_samples:
            x_arr = np.array(x_tuple)
            Y = self.problem.simulate(x_arr)
            if x_tuple not in self.observations:
                self.observations[x_tuple] = []
            self.observations[x_tuple].append(Y)
            self.history.append((x_tuple, Y))
            pre_obs_list.append({'x': list(x_tuple), 'Y': Y.tolist()})

        # ── Data-driven initialization (following Bao et al. pre_sample.m) ──
        # Build basis matrix from pre-sampled INTEGER inputs (same scale as GPR)
        p = 2 * self.d + 1
        Phi_pre = np.array([
            np.concatenate([[1.0],
                            np.array(x, dtype=float),
                            np.array(x, dtype=float) ** 2])
            for x in pre_samples
        ])  # shape (n0, p)

        lambda_data = []
        prior_var_data = []
        beta_hat_data = []

        for i in range(3):
            Y_i = np.array([self.observations[x][0][i] for x in pre_samples])
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

        # Re-initialize GPR models with data-driven lambda and prior_var
        _norm_func = lambda x: self.problem.normalize(np.asarray(x, dtype=float))
        self.gpr = [ParametricGPR(self.d, lambda_data[i], prior_var_data[i])
                    for i in range(3)]
        # Set initial parametric mean to OLS estimate (matches old MATLAB init)
        for i in range(3):
            self.gpr[i].a[:p] = beta_hat_data[i]

        # Dimension augment all pre-sampled solutions
        # (adds deviation terms; NOT Kalman-updated — OLS provides the estimate)
        for x_tuple in pre_samples:
            x_arr = np.array(x_tuple)
            for i in range(3):
                self.gpr[i].dimension_augment(x_arr)

        # Initialize VEPM using GPR posterior means (from OLS estimate)
        self.vepm.initialize(pre_samples, self.observations, self.gpr)
        self.partition_features = tuple(getattr(
            self.vepm, 'feature_indices', self.partition_features))
        self._freeze_pooled_pre_variance()
        self._seed_variance_surrogate(pre_samples)

        t_pre_end = time.time()

        # Save pre-sampling log
        self.pre_sampling_log = {
            'n0': self.n0,
            'n_unique_solutions': len(pre_samples),
            'time_sec': t_pre_end - t_pre_start,
            'observations': pre_obs_list,
            'theta_dim_after': len(self.gpr[0].a),
            'variance_mode': self.variance_mode,
            'pooled_pre_variance': {
                str(k): float(v)
                for k, v in self.pooled_pre_variance.items()
            },
            'robust_vepm': self.robust_vepm,
            'vepm_residual_clip_factor': self.vepm_residual_clip_factor,
            'vepm_new_point_weight': self.vepm_new_point_weight,
            'vepm_partition_weight_floor': self.vepm_partition_weight_floor,
            'adaptive_vepm': self.adaptive_vepm,
            'adaptive_vepm_max_features': self.adaptive_vepm_max_features,
            'adaptive_vepm_min_score': self.adaptive_vepm_min_score,
            'adaptive_feature_selected': list(getattr(
                self.vepm, 'adaptive_feature_selected',
                getattr(self.vepm, 'feature_indices', ()))),
            'adaptive_feature_scores': getattr(
                self.vepm, 'adaptive_feature_scores', {}),
            'vepm_shrinkage_kappa': self.vepm_shrinkage_kappa,
            'variance_surrogate': (
                None if self.var_surrogate_model is None
                else self.var_surrogate_model.diagnostics()),
        }

        # =================================================================
        # Phase 2: Main loop (sequential KG-guided sampling)
        # =================================================================
        if verbose:
            print(f"Main loop: {self.N - self.n0} iterations...")

        for n in range(self.n0, self.N):
            iter_idx = n - self.n0   # 0-based iteration index
            t_iter_start = time.time()
            iter_log = {
                'iteration': iter_idx,
                'stage': n,
                'variance_mode': self.variance_mode,
            }

            if verbose and iter_idx % 50 == 0:
                print(f"  Iteration {iter_idx + 1}/{self.N - self.n0}")

            # ---- Step 1: Solve posterior problem ----
            t0 = time.time()
            pareto_set = self._solve_posterior_problem()
            iter_log['t_posterior_solve'] = time.time() - t0

            # ---- Step 2: Generate candidate set ----
            t0 = time.time()
            candidate_set = self._generate_candidate_set(pareto_set, iter_idx)
            if len(candidate_set) == 0:
                candidate_set = [self.random_solution()]
            candidate_arrays = [np.array(c) for c in candidate_set]
            iter_log['t_candidate_gen'] = time.time() - t0
            iter_log['n_candidates'] = len(candidate_set)
            iter_log['boundary_candidate_log'] = dict(
                getattr(self, '_last_boundary_candidate_log', {}))
            iter_log['candidate_set'] = [
                [int(v) for v in x_tuple] for x_tuple in candidate_set
            ]

            # ---- Step 3: Compute KG factors ----
            t0 = time.time()
            kg_pairs = []
            for x_tuple in candidate_set:
                x_arr = np.array(x_tuple)
                kg1 = compute_kg_factor(
                    self.gpr[0], candidate_arrays, x_arr,
                    self._effective_variance(0, x_arr))
                kg2 = compute_kg_factor(
                    self.gpr[1], candidate_arrays, x_arr,
                    self._effective_variance(1, x_arr))
                kg_pairs.append([kg1, kg2])
            kg_pairs = np.array(kg_pairs)
            iter_log['t_kg_compute'] = time.time() - t0
            iter_log['kg_pairs'] = kg_pairs.tolist()
            boundary_scores, boundary_acq_log = (
                self._boundary_acquisition_scores(candidate_set, iter_idx))
            acquisition_pairs = (
                kg_pairs
                + boundary_acq_log["effective_weight"]
                * boundary_scores[:, None])
            iter_log['boundary_acquisition_log'] = boundary_acq_log
            iter_log['boundary_acquisition_scores'] = (
                boundary_scores.tolist())

            # ---- Step 4: Pareto-KG selection ----
            _, pareto_kg_idx = pareto_filter(
                -acquisition_pairs, return_indices=True)
            iter_log['pareto_kg_indices'] = [
                int(idx) for idx in pareto_kg_idx
            ]
            iter_log['pareto_acquisition_indices'] = [
                int(idx) for idx in pareto_kg_idx
            ]

            selected_idx = self._posterior_quality_select(
                candidate_set, pareto_kg_idx, acquisition_pairs)
            iter_log['boundary_acquisition_selected'] = float(
                boundary_scores[selected_idx])

            x_selected, selection_log = self._apply_replication_policy(
                candidate_set, candidate_arrays, selected_idx, kg_pairs,
                pareto_set, iter_idx)
            x_arr = np.array(x_selected)
            iter_log['x_selected'] = x_selected
            iter_log['is_new_solution'] = x_selected not in self.observations
            iter_log['n_pareto_kg'] = len(pareto_kg_idx)
            iter_log.update(selection_log)
            iter_log['kg1_selected'] = selection_log['kg1_final_selected']
            iter_log['kg2_selected'] = selection_log['kg2_final_selected']

            # Cache pre-update posterior means and variance estimates.  VEPM's
            # residual recursion is defined with mu_n(x^n), while the Kalman
            # update uses the same stage-n plug-in noise variance.
            mu_before = [
                self.gpr[i].posterior_mean(x_arr)
                for i in range(3)
            ]
            sigma2_before = [
                self._effective_variance(i, x_arr)
                for i in range(3)
            ]
            sigma2_details = [
                self._effective_variance_with_details(i, x_arr)[1]
                for i in range(3)
            ]
            iter_log['mu_before_update'] = [float(v) for v in mu_before]
            iter_log['sigma2_before_update'] = [
                float(v) for v in sigma2_before
            ]
            iter_log['sigma2_components_before_update'] = sigma2_details

            # ---- Step 5: Simulate ----
            t0 = time.time()
            Y = self.problem.simulate(x_arr)
            iter_log['t_simulate'] = time.time() - t0
            iter_log['Y_observed'] = Y.tolist()

            if x_selected not in self.observations:
                self.observations[x_selected] = []
            self.observations[x_selected].append(Y)
            self.history.append((x_selected, Y))

            # ---- Step 6: Update posterior beliefs (3 GPR models) ----
            t0 = time.time()
            for i in range(3):
                self.gpr[i].update(x_arr, Y[i], sigma2_before[i])
            iter_log['t_belief_update'] = time.time() - t0

            # ---- Step 7: Update VEPM (3 objectives) ----
            t0 = time.time()
            vepm_update_details = []
            for i in range(3):
                detail = self.vepm.update(
                    i, x_arr, Y[i], mu_before[i], self.gpr[i])
                if detail is not None:
                    vepm_update_details.append(detail)
            iter_log['t_vepm_update'] = time.time() - t0
            iter_log['vepm_update_details'] = vepm_update_details
            t0 = time.time()
            self._update_variance_surrogate(x_arr, Y, mu_before)
            iter_log['t_variance_surrogate_update'] = time.time() - t0
            iter_log['variance_surrogate'] = (
                None if self.var_surrogate_model is None
                else self.var_surrogate_model.diagnostics())

            # ---- Posterior state ----
            iter_log['n_visited'] = len(self.gpr[0].sampled_set)
            iter_log['theta_dim'] = len(self.gpr[0].a)

            # ---- Performance snapshot (every 10 iterations + final) ----
            if iter_idx % 10 == 0 or n == self.N - 1:
                t0 = time.time()
                hv, pf_size, pf_points = self._evaluate_hv()
                iter_log['t_hv_eval'] = time.time() - t0
                iter_log['hv'] = hv
                iter_log['pareto_set_size'] = pf_size
                iter_log['pareto_front'] = pf_points
                self.hv_history.append((n, hv))
            else:
                iter_log['t_hv_eval'] = 0.0
                iter_log['hv'] = None
                iter_log['pareto_set_size'] = None
                iter_log['pareto_front'] = None

            iter_log['t_total'] = time.time() - t_iter_start
            self.iteration_log.append(iter_log)

            # ---- Optional instrumentation snapshot ----
            if self.instrument is not None:
                stride = self.instrument.get('stride', 10)
                if (iter_idx % stride == 0) or (n == self.N - 1):
                    snap = {'n': int(n), 'iteration': int(iter_idx)}
                    eval_x = self.instrument.get('eval_x')
                    if eval_x:
                        eval_arr = [np.array(x) for x in eval_x]
                        for i in range(3):
                            mu_vals = np.array(
                                [self.gpr[i].posterior_mean(xa) for xa in eval_arr])
                            snap[f'mu{i}_eval'] = mu_vals.tolist()
                    ref_x = self.instrument.get('ref_x')
                    if ref_x is not None:
                        ref_arr = np.array(ref_x)
                        snap['sigma2_ref'] = [
                            float(self.vepm.get_variance(i, ref_arr))
                            for i in range(3)]
                    # Posterior Pareto set (list of x tuples) — reuse the one
                    # computed at the start of this iteration
                    snap['pareto_set'] = [tuple(int(v) for v in x)
                                          for x in pareto_set]
                    self.instrument_log.append(snap)

        # =================================================================
        # Phase 3: Final posterior problem
        # =================================================================
        if verbose:
            print("Solving final posterior problem...")

        t0 = time.time()
        final_pareto = self._solve_posterior_problem()
        t_final = time.time() - t0

        # Compute final true objectives
        final_true_objs = []
        for x in final_pareto:
            f1, f2, f3 = self.problem.true_objectives(np.array(x))
            final_true_objs.append({'x': list(x), 'f1': f1, 'f2': f2, 'f3': f3})

        self.final_log = {
            'time_solve_sec': t_final,
            'pareto_set_size': len(final_pareto),
            'pareto_solutions': final_true_objs,
            'total_observations': len(self.history),
            'n_distinct_solutions': len(self.gpr[0].sampled_set),
        }

        return final_pareto

    def get_estimated_pareto_front(self, pareto_set):
        """Get posterior mean objective values for estimated Pareto set."""
        if len(pareto_set) == 0:
            return np.empty((0, 2))
        objs = np.array([
            [self.gpr[0].posterior_mean(np.array(x)),
             self.gpr[1].posterior_mean(np.array(x))]
            for x in pareto_set
        ])
        return pareto_filter(objs)

    def get_true_objectives_of_estimate(self, pareto_set):
        """Get TRUE objective values for the estimated Pareto solutions."""
        if len(pareto_set) == 0:
            return np.empty((0, 2))
        objs = []
        for x in pareto_set:
            f1, f2, _ = self.problem.true_objectives(np.array(x))
            objs.append([f1, f2])
        return pareto_filter(np.array(objs))

    def run_resumable(self, verbose=True):
        """Like run() but saves a checkpoint after every main-loop iteration
        and skips pre-sampling + already-completed iterations on restart.

        Requires ``self._checkpoint_path`` to be set before calling.  Users
        typically set it via a setter or directly after construction.  When
        the checkpoint file already exists and has been loaded via
        ``GPRKR_Algorithm.restore_checkpoint()``, this method picks up from
        ``self._main_iter_completed`` and skips Phase 1 (pre-sampling).

        Optionally, ``self._snapshot_jsonl_path`` if set receives one JSONL
        line per iteration for streaming monitoring independent of the
        pickle checkpoint.

        Phase 2 iterations mirror run() exactly; only the progress
        bookkeeping differs.  The final posterior solve (Phase 3) is still
        done once at the end as in run().
        """
        import time as _time
        import json as _json

        # =================================================================
        # Phase 1: Pre-sampling (skipped on resume)
        # =================================================================
        if not self._presampling_done:
            t_pre_start = _time.time()
            # ── Resumable pre-sampling logic ──────────────────────────────
            # If a previous run was interrupted partway through pre-sampling,
            # `self.history` already contains some (x, Y) observations.
            # Three cases:
            #   (a) history empty → fresh start, generate n0 new pre-samples
            #   (b) history has 0 < k < n0 → keep those k as pre-samples,
            #        generate (n0 - k) MORE random distinct samples, simulate
            #        only those, append to history.
            #   (c) history has >= n0 → use the first n0 as pre-samples, do
            #        OLS init only (no new simulations).
            existing_pre = [x for x, _ in self.history]
            existing_pre_set = set(existing_pre)
            n_have = len(existing_pre)
            if verbose:
                if n_have == 0:
                    print(f"[resumable] Fresh pre-sampling: generating "
                          f"{self.n0} solutions...", flush=True)
                elif n_have < self.n0:
                    print(f"[resumable] Resuming pre-sampling: have "
                          f"{n_have}/{self.n0} from previous run; "
                          f"generating {self.n0 - n_have} more...",
                          flush=True)
                else:
                    print(f"[resumable] Pre-sampling already covered: "
                          f"history has {n_have} observations >= "
                          f"n0={self.n0}; skipping simulation, doing OLS "
                          f"init only.", flush=True)

            # Build pre_samples list by combining existing + (newly-generated)
            pre_samples = list(existing_pre[:self.n0])
            pre_obs_list = [{'x': list(x), 'Y': self.history[i][1].tolist()}
                            for i, x in enumerate(existing_pre[:self.n0])]
            if self.initial_samples is not None:
                for x_seed in self.initial_samples:
                    if len(pre_samples) >= self.n0:
                        break
                    x_seed = tuple(int(v) for v in x_seed)
                    if x_seed in existing_pre_set or x_seed in pre_samples:
                        continue
                    pre_samples.append(x_seed)
            if self.use_boundary_initial_design:
                for x_seed in self._boundary_solutions():
                    if len(pre_samples) >= self.n0:
                        break
                    if x_seed in existing_pre_set or x_seed in pre_samples:
                        continue
                    pre_samples.append(x_seed)
            while len(pre_samples) < self.n0:
                x_new = self.random_solution()
                if x_new in existing_pre_set or x_new in pre_samples:
                    continue
                pre_samples.append(x_new)
            # Simulate the not-yet-observed ones (skip already in history)
            for x_tuple in pre_samples:
                if x_tuple in existing_pre_set:
                    continue   # already simulated in previous run
                x_arr = np.array(x_tuple)
                Y = self.problem.simulate(x_arr)
                if x_tuple not in self.observations:
                    self.observations[x_tuple] = []
                self.observations[x_tuple].append(Y)
                self.history.append((x_tuple, Y))
                pre_obs_list.append({'x': list(x_tuple), 'Y': Y.tolist()})
                # Save intermediate after each pre-sample too, so kill during
                # pre-sampling doesn't lose the whole phase.
                self.save_checkpoint()

            # Data-driven initialisation (same as run())
            p = 2 * self.d + 1
            Phi_pre = np.array([
                np.concatenate([[1.0],
                                np.array(x, dtype=float),
                                np.array(x, dtype=float) ** 2])
                for x in pre_samples
            ])
            lambda_data, prior_var_data, beta_hat_data = [], [], []
            for i in range(3):
                Y_i = np.array([self.observations[x][0][i] for x in pre_samples])
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

            _norm_func = lambda x: self.problem.normalize(np.asarray(x, dtype=float))
            self.gpr = [ParametricGPR(self.d, lambda_data[i], prior_var_data[i])
                        for i in range(3)]
            for i in range(3):
                self.gpr[i].a[:p] = beta_hat_data[i]
            for x_tuple in pre_samples:
                x_arr = np.array(x_tuple)
                for i in range(3):
                    self.gpr[i].dimension_augment(x_arr)
            self.vepm.initialize(pre_samples, self.observations, self.gpr)
            self.partition_features = tuple(getattr(
                self.vepm, 'feature_indices', self.partition_features))
            self._freeze_pooled_pre_variance()
            self._seed_variance_surrogate(pre_samples)

            self.pre_sampling_log = {
                'n0': self.n0,
                'n_unique_solutions': len(pre_samples),
                'time_sec': _time.time() - t_pre_start,
                'observations': pre_obs_list,
                'theta_dim_after': len(self.gpr[0].a),
                'variance_mode': self.variance_mode,
                'pooled_pre_variance': {
                    str(k): float(v)
                    for k, v in self.pooled_pre_variance.items()
                },
                'robust_vepm': self.robust_vepm,
                'vepm_residual_clip_factor': self.vepm_residual_clip_factor,
                'vepm_new_point_weight': self.vepm_new_point_weight,
                'vepm_partition_weight_floor': (
                    self.vepm_partition_weight_floor),
                'adaptive_vepm': self.adaptive_vepm,
                'adaptive_vepm_max_features': (
                    self.adaptive_vepm_max_features),
                'adaptive_vepm_min_score': self.adaptive_vepm_min_score,
                'adaptive_feature_selected': list(getattr(
                    self.vepm, 'adaptive_feature_selected',
                    getattr(self.vepm, 'feature_indices', ()))),
                'adaptive_feature_scores': getattr(
                    self.vepm, 'adaptive_feature_scores', {}),
                'vepm_shrinkage_kappa': self.vepm_shrinkage_kappa,
                'variance_surrogate': (
                    None if self.var_surrogate_model is None
                    else self.var_surrogate_model.diagnostics()),
            }
            self._presampling_done = True
            self.save_checkpoint()
        else:
            if verbose:
                print(f"[resumable] Pre-sampling already done; "
                      f"skipping to iter {self._main_iter_completed}", flush=True)

        # =================================================================
        # Phase 2: Main loop — resume-aware
        # =================================================================
        total_main_iters = self.N - self.n0
        if verbose:
            print(f"[resumable] Main loop: iter "
                  f"{self._main_iter_completed}/{total_main_iters}", flush=True)

        while self._main_iter_completed < total_main_iters:
            iter_idx = self._main_iter_completed
            n = self.n0 + iter_idx
            t_iter_start = _time.time()
            iter_log = {'iteration': iter_idx, 'stage': n,
                        'variance_mode': self.variance_mode,
                        'timestamp_iso': _time.strftime('%Y-%m-%dT%H:%M:%S')}

            # Reuse the inner body of run() exactly — copy/paste of lines
            # 1846-1963 in the original run().  For maintainability we
            # extract into _main_loop_body() below.
            pareto_set = self._main_loop_body(iter_idx, n, iter_log,
                                              t_iter_start)

            self.iteration_log.append(iter_log)

            # ---- Optional instrument snapshot (existing behaviour) --------
            if self.instrument is not None:
                stride = self.instrument.get('stride', 10)
                if (iter_idx % stride == 0) or (iter_idx == total_main_iters - 1):
                    snap = {'n': int(n), 'iteration': int(iter_idx),
                            'timestamp_iso': iter_log['timestamp_iso']}
                    eval_x = self.instrument.get('eval_x')
                    if eval_x:
                        eval_arr = [np.array(x) for x in eval_x]
                        for i in range(3):
                            mu_vals = np.array([self.gpr[i].posterior_mean(xa)
                                                for xa in eval_arr])
                            snap[f'mu{i}_eval'] = mu_vals.tolist()
                    ref_x = self.instrument.get('ref_x')
                    if ref_x is not None:
                        ref_arr = np.array(ref_x)
                        snap['sigma2_ref'] = [float(self._effective_variance(i, ref_arr))
                                              for i in range(3)]
                    snap['pareto_set'] = [tuple(int(v) for v in x) for x in pareto_set]
                    self.instrument_log.append(snap)

            # ---- Lightweight JSONL snapshot every iteration ---------------
            self._append_snapshot_jsonl({
                'iter':           iter_idx,
                'stage':          n,
                'timestamp_iso':  iter_log['timestamp_iso'],
                't_iter':         iter_log.get('t_total'),
                't_simulate':     iter_log.get('t_simulate'),
                't_compute':      (iter_log.get('t_posterior_solve', 0) +
                                   iter_log.get('t_candidate_gen',   0) +
                                   iter_log.get('t_kg_compute',      0) +
                                   iter_log.get('t_belief_update',   0) +
                                   iter_log.get('t_vepm_update',     0)),
                'x_selected':     list(iter_log.get('x_selected', [])),
                'Y_observed':     iter_log.get('Y_observed'),
                'hv':             iter_log.get('hv'),
                'pareto_set_size': iter_log.get('pareto_set_size'),
                'n_visited':      iter_log.get('n_visited'),
            })

            # ---- Advance and checkpoint ----------------------------------
            self._main_iter_completed += 1
            self.save_checkpoint()

            if verbose and iter_idx % 10 == 0:
                print(f"  iter {iter_idx+1}/{total_main_iters}  "
                      f"HV={iter_log.get('hv','--')}  "
                      f"t_iter={iter_log.get('t_total',0):.1f}s",
                      flush=True)

        # =================================================================
        # Phase 3: Final posterior solve
        # =================================================================
        if verbose:
            print("[resumable] Solving final posterior problem...", flush=True)
        t0 = _time.time()
        final_pareto = self._solve_posterior_problem()
        t_final = _time.time() - t0

        final_true_objs = []
        for x in final_pareto:
            f1, f2, f3 = self.problem.true_objectives(np.array(x))
            final_true_objs.append({'x': list(x), 'f1': f1, 'f2': f2, 'f3': f3})

        self.final_log = {
            'pareto_set_size': len(final_pareto),
            'time_sec': t_final,
            'pareto_solutions': [list(x) for x in final_pareto],
            'true_objectives': final_true_objs,
        }
        self.save_checkpoint()
        return final_pareto

    def _main_loop_body(self, iter_idx, n, iter_log, t_iter_start):
        """Inner body of one main-loop iteration, extracted so that run()
        and run_resumable() share identical simulation logic."""
        import time as _time

        # ---- Step 1: Solve posterior problem ----
        t0 = _time.time()
        pareto_set = self._solve_posterior_problem()
        iter_log['t_posterior_solve'] = _time.time() - t0

        # ---- Step 2: Generate candidate set ----
        t0 = _time.time()
        candidate_set = self._generate_candidate_set(pareto_set, iter_idx)
        if len(candidate_set) == 0:
            candidate_set = [self.random_solution()]
        candidate_arrays = [np.array(c) for c in candidate_set]
        iter_log['t_candidate_gen'] = _time.time() - t0
        iter_log['n_candidates'] = len(candidate_set)
        iter_log['boundary_candidate_log'] = dict(
            getattr(self, '_last_boundary_candidate_log', {}))
        iter_log['candidate_set'] = [
            [int(v) for v in x_tuple] for x_tuple in candidate_set
        ]

        # ---- Step 3: KG factors ----
        t0 = _time.time()
        kg_pairs = []
        for x_tuple in candidate_set:
            x_arr = np.array(x_tuple)
            kg1 = compute_kg_factor(self.gpr[0], candidate_arrays, x_arr,
                                    self._effective_variance(0, x_arr))
            kg2 = compute_kg_factor(self.gpr[1], candidate_arrays, x_arr,
                                    self._effective_variance(1, x_arr))
            kg_pairs.append([kg1, kg2])
        kg_pairs = np.array(kg_pairs)
        iter_log['t_kg_compute'] = _time.time() - t0
        iter_log['kg_pairs'] = kg_pairs.tolist()
        boundary_scores, boundary_acq_log = (
            self._boundary_acquisition_scores(candidate_set, iter_idx))
        acquisition_pairs = (
            kg_pairs
            + boundary_acq_log["effective_weight"]
            * boundary_scores[:, None])
        iter_log['boundary_acquisition_log'] = boundary_acq_log
        iter_log['boundary_acquisition_scores'] = boundary_scores.tolist()

        # ---- Step 4: Pareto-KG selection ----
        _, pareto_kg_idx = pareto_filter(-acquisition_pairs,
                                         return_indices=True)
        iter_log['pareto_kg_indices'] = [int(idx) for idx in pareto_kg_idx]
        iter_log['pareto_acquisition_indices'] = [
            int(idx) for idx in pareto_kg_idx
        ]
        selected_idx = self._posterior_quality_select(
            candidate_set, pareto_kg_idx, acquisition_pairs)
        iter_log['boundary_acquisition_selected'] = float(
            boundary_scores[selected_idx])

        x_selected, selection_log = self._apply_replication_policy(
            candidate_set, candidate_arrays, selected_idx, kg_pairs,
            pareto_set, iter_idx)
        x_arr = np.array(x_selected)
        iter_log['x_selected']     = x_selected
        iter_log['is_new_solution'] = x_selected not in self.observations
        iter_log['n_pareto_kg']    = len(pareto_kg_idx)
        iter_log.update(selection_log)
        iter_log['kg1_selected']   = selection_log['kg1_final_selected']
        iter_log['kg2_selected']   = selection_log['kg2_final_selected']

        # Cache pre-update posterior means and variance estimates.  This keeps
        # the resumable path identical to run(): GPR uses sigma_n^2(x^n), and
        # VEPM residuals use mu_n(x^n), as in Lemma VEPM_recursive.
        mu_before = [
            self.gpr[i].posterior_mean(x_arr)
            for i in range(3)
        ]
        sigma2_before = [
            self._effective_variance(i, x_arr)
            for i in range(3)
        ]
        sigma2_details = [
            self._effective_variance_with_details(i, x_arr)[1]
            for i in range(3)
        ]
        iter_log['mu_before_update'] = [float(v) for v in mu_before]
        iter_log['sigma2_before_update'] = [float(v) for v in sigma2_before]
        iter_log['sigma2_components_before_update'] = sigma2_details

        # ---- Step 5: Simulate ----
        t0 = _time.time()
        Y = self.problem.simulate(x_arr)
        iter_log['t_simulate']  = _time.time() - t0
        iter_log['Y_observed']  = Y.tolist()

        if x_selected not in self.observations:
            self.observations[x_selected] = []
        self.observations[x_selected].append(Y)
        self.history.append((x_selected, Y))

        # ---- Step 6: GPR update ----
        t0 = _time.time()
        for i in range(3):
            self.gpr[i].update(x_arr, Y[i], sigma2_before[i])
        iter_log['t_belief_update'] = _time.time() - t0

        # ---- Step 7: VEPM update ----
        t0 = _time.time()
        vepm_update_details = []
        for i in range(3):
            detail = self.vepm.update(
                i, x_arr, Y[i], mu_before[i], self.gpr[i])
            if detail is not None:
                vepm_update_details.append(detail)
        iter_log['t_vepm_update'] = _time.time() - t0
        iter_log['vepm_update_details'] = vepm_update_details
        t0 = _time.time()
        self._update_variance_surrogate(x_arr, Y, mu_before)
        iter_log['t_variance_surrogate_update'] = _time.time() - t0
        iter_log['variance_surrogate'] = (
            None if self.var_surrogate_model is None
            else self.var_surrogate_model.diagnostics())

        iter_log['n_visited'] = len(self.gpr[0].sampled_set)
        iter_log['theta_dim'] = len(self.gpr[0].a)

        # ---- HV evaluation (every 10 iter, or final) ----
        total_main = self.N - self.n0
        if iter_idx % 10 == 0 or iter_idx == total_main - 1:
            t0 = _time.time()
            hv, pf_size, pf_points = self._evaluate_hv()
            iter_log['t_hv_eval']       = _time.time() - t0
            iter_log['hv']              = hv
            iter_log['pareto_set_size'] = pf_size
            iter_log['pareto_front']    = pf_points
            self.hv_history.append((n, hv))
        else:
            iter_log['t_hv_eval']       = 0.0
            iter_log['hv']              = None
            iter_log['pareto_set_size'] = None
            iter_log['pareto_front']    = None

        iter_log['t_total'] = _time.time() - t_iter_start
        return pareto_set

    def get_full_results(self):
        """Return a comprehensive results dictionary for saving.

        Returns:
            dict with keys:
                'config': algorithm configuration parameters
                'pre_sampling': pre-sampling phase log
                'iterations': list of per-iteration logs
                'final': final solution log
                'hv_history': list of (stage, hv) tuples
                'all_observations': list of (x, Y) tuples
        """
        return {
            'config': {
                'N': self.N, 'n0': self.n0,
                'K1': self.K1, 'K2': self.K2,
                'lambda_i': self.lambda_i, 'prior_var': self.prior_var,
                'w_vepm': self.w_vepm, 'n_thr': self.n_thr,
                'seed': self.seed,
                'd': self.d, 'L': self.L,
                'tau': self.problem.tau, 'alpha': self.problem.alpha,
                'vepm_partitions': self.vepm.total_partitions,
                'partition_method': self.partition_method,
                'partition_features_mode': self.partition_features_mode,
                'partition_features': list(self.partition_features),
                'vepm_n_features': getattr(self.vepm, 'n_features', None),
                'variance_mode': self.variance_mode,
                'pooled_pre_variance': {
                    str(k): float(v)
                    for k, v in getattr(self, 'pooled_pre_variance', {}).items()
                },
                'variance_surrogate': self.variance_surrogate,
                'variance_surrogate_rho0': self.variance_surrogate_rho0,
                'variance_surrogate_alpha': self.variance_surrogate_alpha,
                'variance_surrogate_min_samples': (
                    self.variance_surrogate_min_samples),
                'variance_surrogate_only_constraint': (
                    self.variance_surrogate_only_constraint),
                'variance_surrogate_clip_low': (
                    self.variance_surrogate_clip_low),
                'variance_surrogate_clip_high': (
                    self.variance_surrogate_clip_high),
                'robust_vepm': self.robust_vepm,
                'vepm_residual_clip_factor': self.vepm_residual_clip_factor,
                'vepm_new_point_weight': self.vepm_new_point_weight,
                'vepm_partition_weight_floor': (
                    self.vepm_partition_weight_floor),
                'adaptive_vepm': self.adaptive_vepm,
                'adaptive_vepm_max_features': (
                    self.adaptive_vepm_max_features),
                'adaptive_vepm_min_score': self.adaptive_vepm_min_score,
                'adaptive_feature_selected': list(getattr(
                    self.vepm, 'adaptive_feature_selected',
                    getattr(self.vepm, 'feature_indices', ()))),
                'adaptive_feature_scores': getattr(
                    self.vepm, 'adaptive_feature_scores', {}),
                'vepm_shrinkage_kappa': self.vepm_shrinkage_kappa,
                'replication_policy': self.replication_policy,
                'replication_max_per_solution': (
                    self.replication_max_per_solution),
                'replication_score_threshold': (
                    self.replication_score_threshold),
                'replication_boundary_scale': (
                    self.replication_boundary_scale),
                'replication_budget_fraction': (
                    self.replication_budget_fraction),
                'boundary_candidate_policy': (
                    self.boundary_candidate_policy),
                'boundary_candidate_count': (
                    self.boundary_candidate_count),
                'boundary_candidate_pool_size': (
                    self.boundary_candidate_pool_size),
                'boundary_candidate_margin_scale': (
                    self.boundary_candidate_margin_scale),
                'boundary_candidate_feasibility_buffer': (
                    self.boundary_candidate_feasibility_buffer),
                'boundary_acquisition_weight': (
                    self.boundary_acquisition_weight),
                'boundary_acquisition_margin_scale': (
                    self.boundary_acquisition_margin_scale),
                'boundary_acquisition_decay_power': (
                    self.boundary_acquisition_decay_power),
                'exploration_epsilon0': self.exploration_epsilon0,
                'exploration_epsilon_min': self.exploration_epsilon_min,
                'exploration_decay_power': self.exploration_decay_power,
            },
            'pre_sampling': self.pre_sampling_log,
            'iterations': self.iteration_log,
            'final': self.final_log,
            'hv_history': self.hv_history,
            'all_observations': [
                {'x': list(x), 'Y': Y.tolist()} for x, Y in self.history
            ],
        }


class _NoOpVEPM:
    """No-op placeholder used by the GPR-KG-nV ablation.

    It preserves the checkpointed GPRKR_Algorithm control flow while ensuring
    that no VEPM partition estimate is used by the ablation variant.
    """

    total_partitions = 0

    def initialize(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None


class GPRKRnV_Algorithm(GPRKR_Algorithm):
    """Checkpointable GPR-KG-nV ablation.

    This variant inherits the theory-aligned engineering path of GPR-KG
    (structured finite-grid initial design, integer-grid candidate mapping,
    KG-Pareto candidate selection, checkpoints, and JSONL snapshots), but
    replaces VEPM by a direct pooled/local residual variance estimate.
    """

    def __init__(self, *args, nv_fallback_variance=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.nv_fallback_variance = float(nv_fallback_variance)
        self.vepm = _NoOpVEPM()

    def _local_residual_variance(self, i, x_tuple):
        obs_list = self.observations.get(tuple(x_tuple), [])
        if len(obs_list) < 2:
            return None
        x_arr = np.asarray(x_tuple, dtype=float)
        mu = float(self.gpr[i].posterior_mean(x_arr))
        resids = [float(obs[i]) - mu for obs in obs_list]
        return max(float(np.var(resids, ddof=1)), self.variance_floor)

    def _global_residual_variance(self, i):
        local_vars = []
        for x_tuple, obs_list in self.observations.items():
            if len(obs_list) < 2:
                continue
            v = self._local_residual_variance(i, x_tuple)
            if v is not None and np.isfinite(v):
                local_vars.append(v)
        if local_vars:
            return max(float(np.mean(local_vars)), self.variance_floor)
        return max(self.nv_fallback_variance, self.variance_floor)

    def _effective_variance(self, i, x):
        """Direct non-VEPM variance estimate for GPR-KG-nV.

        Use the Bessel-corrected residual variance at a replicated solution;
        otherwise use the pooled mean of all currently available replicated
        residual variances, with a fixed fallback before any replication exists.
        """
        x_tuple = tuple(int(v) for v in x)
        v_local = self._local_residual_variance(i, x_tuple)
        if v_local is not None:
            return max(v_local, self.variance_floor)
        return self._global_residual_variance(i)

    def _effective_variance_with_details(self, i, x):
        x_tuple = tuple(int(v) for v in x)
        v_local = self._local_residual_variance(i, x_tuple)
        v_global = self._global_residual_variance(i)
        v_eff = v_local if v_local is not None else v_global
        v_eff = max(float(v_eff), self.variance_floor)
        return v_eff, {
            "local_replicate": (
                None if v_local is None else float(v_local)),
            "pooled_replicate": float(v_global),
            "fallback": float(self.nv_fallback_variance),
            "source": "local_replicate" if v_local is not None
            else "pooled_or_fallback",
        }

    def get_full_results(self):
        results = super().get_full_results()
        results["config"]["method"] = "GPR-KG-nV"
        results["config"]["nv_fallback_variance"] = self.nv_fallback_variance
        results["config"]["vepm_partitions"] = 0
        return results
