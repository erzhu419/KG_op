"""RZDT-style synthetic benchmarks used by the SC-OLH-KG prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from core.metrics import pareto_filter


@dataclass
class ProblemInfo:
    name: str
    d: int
    L: int
    sigma: float
    alpha: float
    tau: float


class TestProblem:
    """Base vector-output test problem `(f1, f2, f3)`."""

    problem_name = "base"
    variance_features = (0,)
    recommended_partition_features = (0,)

    def __init__(self, d=5, L=100, sigma=0.04, heteroscedastic=True, alpha=0.05):
        self.d = int(d)
        self.L = int(L)
        self.sigma_level = float(sigma)
        self.heteroscedastic = bool(heteroscedastic)
        self.alpha = float(alpha)
        self.tau = 0.0
        self.ref_point = np.array([1.5, 1.5], dtype=float)

    def info(self):
        return ProblemInfo(
            name=self.problem_name,
            d=self.d,
            L=self.L,
            sigma=self.sigma_level,
            alpha=self.alpha,
            tau=self.tau,
        )

    def int_bounds(self):
        return np.zeros(self.d, dtype=int), np.full(self.d, self.L, dtype=int)

    def normalize(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip(x / float(self.L), 0.0, 1.0)

    def continuous_to_int(self, x_norm):
        lo, hi = self.int_bounds()
        x_norm = np.asarray(x_norm, dtype=float)
        x_int = np.round(lo + x_norm * (hi - lo)).astype(int)
        return tuple(np.clip(x_int, lo, hi))

    def sample_random(self, rng=None):
        rng = rng or np.random.default_rng()
        lo, hi = self.int_bounds()
        return tuple(int(rng.integers(lo[j], hi[j] + 1)) for j in range(self.d))

    def true_objectives(self, x):
        raise NotImplementedError

    def sigma_func(self, x, f_val=0.0):
        return self.sigma_level

    def true_sigma(self, x):
        sig = float(self.sigma_func(x, 0.0))
        return np.array([sig, sig, sig], dtype=float)

    def simulate(self, x, rng=None):
        rng = rng or np.random.default_rng()
        means = np.asarray(self.true_objectives(x), dtype=float)
        sig = self.true_sigma(x)
        return means + rng.normal(0.0, sig, size=3)

    def is_truly_feasible(self, x):
        f3 = float(self.true_objectives(x)[2])
        sigma3 = float(self.true_sigma(x)[2])
        return f3 + norm.ppf(1 - self.alpha) * sigma3 <= self.tau

    def true_pareto_solutions(self):
        lo, hi = self.int_bounds()
        return [tuple([x1] + [lo[j] for j in range(1, self.d)])
                for x1 in range(lo[0], hi[0] + 1)]

    def true_pareto_front(self):
        objs = []
        for x in self.true_pareto_solutions():
            if self.is_truly_feasible(x):
                objs.append(self.true_objectives(x)[:2])
        if not objs:
            return np.empty((0, 2))
        return pareto_filter(np.asarray(objs, dtype=float))

    def risk_class(self, x):
        u = float(self.normalize(x)[0])
        if u < 1.0 / 3.0:
            return 0
        if u < 2.0 / 3.0:
            return 1
        return 2


class RZDT1(TestProblem):
    """Convex front with monotone heteroscedasticity."""

    problem_name = "RZDT1"

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = float(t[0])
        g = 1.0 + 9.0 / max(self.d - 1, 1) * float(np.sum(t[1:]))
        f2 = g * (1.0 - np.sqrt(max(f1 / g, 0.0)))
        f3 = f1 - 0.5
        return f1, float(f2), float(f3)

    def sigma_func(self, x, f_val=0.0):
        if not self.heteroscedastic:
            return self.sigma_level
        u = float(self.normalize(x)[0])
        return self.sigma_level * (0.5 + 2.5 * np.sqrt(max(u, 0.0)))


class RZDT2(TestProblem):
    """Concave front with bell-shaped heteroscedasticity."""

    problem_name = "RZDT2"

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = float(t[0])
        g = 1.0 + 9.0 / max(self.d - 1, 1) * float(np.sum(t[1:]))
        f2 = g * (1.0 - (f1 / g) ** 2)
        f3 = -(f1 - 0.5) ** 2 + 0.04
        return f1, float(f2), float(f3)

    def sigma_func(self, x, f_val=0.0):
        if not self.heteroscedastic:
            return self.sigma_level
        u = float(self.normalize(x)[0])
        return self.sigma_level * (0.5 + 2.5 * np.sin(np.pi * u) ** 2)


class RZDT5RR(TestProblem):
    """Enlarged-grid hyperbolic front from the current repo."""

    problem_name = "RZDT5_RR"

    def __init__(self, d=5, L=100, sigma=0.04, heteroscedastic=True, alpha=0.05):
        super().__init__(d=d, L=L, sigma=sigma, heteroscedastic=heteroscedastic, alpha=alpha)
        self.L1 = 100
        self.L2 = 50

    def int_bounds(self):
        hi = np.full(self.d, self.L2, dtype=int)
        hi[0] = self.L1
        return np.zeros(self.d, dtype=int), hi

    def normalize(self, x):
        x = np.asarray(x, dtype=float)
        z = np.zeros(self.d, dtype=float)
        z[0] = x[0] / float(self.L1)
        if self.d > 1:
            z[1:] = x[1:] / float(self.L2)
        return np.clip(z, 0.0, 1.0)

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = float(t[0])
        g = 1.0 + float(np.sum(t[1:]))
        f2 = g / (float(np.asarray(x)[0]) + 1.0)
        f3 = f1 - 0.5
        return f1, float(f2), float(f3)

    def sigma_func(self, x, f_val=0.0):
        if not self.heteroscedastic:
            return self.sigma_level
        u = float(self.normalize(x)[0])
        return self.sigma_level * (0.3 + 2.0 * u ** 2)


class RegimeRZDT1(RZDT1):
    """RZDT1 with explicit low/medium/high variance regimes.

    This is the first sanity benchmark for class-HVD.  The objective geometry
    remains simple while the simulation noise is piecewise by `x_1`.
    """

    problem_name = "RegimeRZDT1"

    def sigma_func(self, x, f_val=0.0):
        if not self.heteroscedastic:
            return self.sigma_level
        multipliers = (0.35, 1.0, 2.8)
        return self.sigma_level * multipliers[self.risk_class(x)]


class StatePolicyRZDT1(TestProblem):
    """State-policy synthetic with a feasible occupancy pocket.

    The objective and chance constraint depend on a low-dimensional occupancy
    summary, not only on the first decision coordinate.  This gives SC coupling
    a controlled benchmark where covering new policy states can reveal a good
    feasible region that raw axis-only RZDT tests do not emphasize.
    """

    problem_name = "StatePolicyRZDT1"
    variance_features = (0, 1)
    recommended_partition_features = (0, 1)

    def _policy_state(self, x):
        t = self.normalize(x)
        u = float(t[0])
        tail = t[1:] if self.d > 1 else np.array([0.0])
        q = float(np.mean(tail))
        spread = float(np.std(tail))
        return u, q, spread

    def true_objectives(self, x):
        u, q, spread = self._policy_state(x)
        t = self.normalize(x)
        tail = t[1:] if self.d > 1 else np.array([q])
        tail_loss = float(np.mean((tail - 0.70) ** 2))
        state_loss = (
            2.2 * (u - 0.25) ** 2
            + 5.0 * tail_loss
            + 0.05 * spread ** 2
        )
        f1 = 0.35 + state_loss
        f2 = 0.35 + state_loss + 0.08 * (u + 0.5 * q)
        tail_pocket = float(np.mean(((tail - 0.70) / 0.28) ** 2))
        pocket = ((u - 0.25) / 0.28) ** 2 + tail_pocket
        f3 = 0.12 * (pocket - 1.5) + 0.02 * spread
        return float(f1), float(f2), float(f3)

    def sigma_func(self, x, f_val=0.0):
        if not self.heteroscedastic:
            return self.sigma_level
        u, q, spread = self._policy_state(x)
        q_gap = min(abs(q - 0.70) / 0.70, 1.0)
        return self.sigma_level * (
            0.35 + 1.4 * q_gap + 0.8 * np.sin(np.pi * u) ** 2 + 0.8 * spread
        )

    def risk_class(self, x):
        _, q, _ = self._policy_state(x)
        if q < 0.45:
            return 0
        if q < 0.80:
            return 1
        return 2

    def all_axis_solutions(self):
        lo, hi = self.int_bounds()
        rows = []
        for x0 in range(int(lo[0]), int(hi[0]) + 1):
            for q in range(int(lo[0]), int(hi[0]) + 1, 5):
                x = [x0]
                x.extend([q] * (self.d - 1))
                rows.append(tuple(x))
        return rows


PROBLEM_REGISTRY = {
    "RZDT1": RZDT1,
    "RZDT2": RZDT2,
    "RZDT5_RR": RZDT5RR,
    "RegimeRZDT1": RegimeRZDT1,
    "StatePolicyRZDT1": StatePolicyRZDT1,
}


def make_problem(name="RZDT1", d=5, L=100, sigma=0.04, alpha=0.05):
    cls = PROBLEM_REGISTRY[name]
    problem = cls(d=d, L=L, sigma=sigma, heteroscedastic=True, alpha=alpha)
    problem.tau = 0.0
    return problem
