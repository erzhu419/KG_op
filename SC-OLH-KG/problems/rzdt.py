"""RZDT-style synthetic benchmarks used by the SC-OLH-KG prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from core.cumulative_risk import (
    CumulativeRiskFeatureProvider,
    CumulativeRiskParameters,
    RiskExposure,
    cumulative_feature_names,
    cumulative_feature_vector,
    decompose_cumulative_risk,
)
from core.metrics import pareto_filter
from representation.observable_exposure import grouped_policy_state_exposure


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


def _task_geometry(shift=None, radius_scale=1.0):
    """Validate a domain-independent normalized task-context draw."""

    values = np.zeros(3, dtype=float) if shift is None else np.asarray(
        shift, dtype=float).reshape(-1)
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise ValueError("task_geometry_shift must contain three finite values")
    scale = float(radius_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "task_geometry_radius_scale must be positive and finite")
    return values.copy(), float(np.clip(scale, 0.50, 2.0))


def _constant_three_block_grid(problem, first, second, third):
    """Return exact summaries induced by integer three-block policies."""

    def quantized(values):
        return np.asarray([
            int(np.clip(round(float(value) * problem.L), 0, problem.L))
            / float(problem.L)
            for value in values
        ], dtype=float)

    grids = np.meshgrid(
        quantized(first),
        quantized(second),
        quantized(third),
        indexing="ij",
    )
    controls = [np.asarray(grid, dtype=float).reshape(-1) for grid in grids]
    counts = np.asarray([
        len(group)
        for group in np.array_split(np.arange(problem.d), 3)
    ], dtype=float)
    weighted_mean = sum(
        counts[index] * controls[index] for index in range(3)
    ) / float(problem.d)
    dispersion = np.sqrt(sum(
        counts[index] * (controls[index] - weighted_mean) ** 2
        for index in range(3)
    ) / float(problem.d))
    return (*controls, dispersion)


def _batched_cumulative_variance(local, shared, params):
    local = np.asarray(local, dtype=float)
    shared = np.asarray(shared, dtype=float)
    independent = np.sum(
        np.asarray(params.Lambda, dtype=float)[:, None] * local ** 2,
        axis=0,
    )
    common = np.einsum(
        "in,ij,jn->n",
        shared,
        np.asarray(params.B, dtype=float),
        shared,
    )
    linear = (
        np.asarray(params.omega, dtype=float) @ shared
    )
    return np.maximum(
        float(params.floor) + independent + common + linear,
        0.0,
    )


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


class HyperbolicAxisFeatureMap:
    """Feature map for RZDT5-style fronts with `g(x_tail) / (x1 + 1)`.

    The default quadratic basis cannot represent the steep reciprocal term near
    the low-risk boundary, which makes the posterior overvalue x1 around 25.
    These features are still problem-structure only; they do not inspect true
    objective values or the feasible set.
    """

    def __init__(self, problem):
        self.problem = problem
        self.feature_dim = 2 * int(problem.d) + 4

    def features(self, x):
        z = np.asarray(self.problem.normalize(x), dtype=float)
        raw = np.asarray(x, dtype=float)
        lo, hi = self.problem.int_bounds()
        denom = max(float(raw[0] - lo[0] + 1.0), 1.0)
        inv = 1.0 / denom
        tail_sum = float(np.sum(z[1:])) if len(z) > 1 else 0.0
        x0_span = max(float(hi[0] - lo[0]), 1.0)
        x0_log = np.log1p(max(float(raw[0] - lo[0]), 0.0)) / np.log1p(x0_span)
        return np.concatenate([
            z,
            z ** 2,
            np.array([
                inv,
                tail_sum * inv,
                (1.0 + tail_sum) * inv,
                x0_log,
            ], dtype=float),
        ])

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        lo, hi = self.problem.int_bounds()
        scale = np.maximum(np.asarray(hi - lo, dtype=float), 1.0)
        z = np.clip((X_arr - lo[None, :]) / scale[None, :], 0.0, 1.0)
        denom = np.maximum(X_arr[:, 0] - float(lo[0]) + 1.0, 1.0)
        inv = 1.0 / denom
        tail_sum = np.sum(z[:, 1:], axis=1) if z.shape[1] > 1 else np.zeros(len(z))
        x0_span = max(float(hi[0] - lo[0]), 1.0)
        x0_log = np.log1p(np.maximum(X_arr[:, 0] - float(lo[0]), 0.0)) / np.log1p(x0_span)
        extra = np.column_stack([
            inv,
            tail_sum * inv,
            (1.0 + tail_sum) * inv,
            x0_log,
        ])
        return np.hstack([z, z ** 2, extra])


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

    def surrogate_basis_map(self):
        return HyperbolicAxisFeatureMap(self)

    def recommendation_refinement_candidates(self):
        lo, hi = self.int_bounds()
        upper = min(int(hi[0]), int(lo[0]) + 30)
        return [
            tuple([x1] + [int(lo[j]) for j in range(1, self.d)])
            for x1 in range(int(lo[0]), upper + 1)
        ]


class PaperRZDT1(RZDT1):
    """RZDT1 as used by the submitted paper's checkpointed pipeline.

    This keeps the paper chance constraint output
    ``-(x1/L - 0.5)^2 + 0.04`` and the slightly milder monotone noise field
    from ``Final_Submission/GPR_KG_Code/gpr_kg.py``.  It is intentionally
    separate from the prototype ``RZDT1`` so earlier smoke tests remain stable.
    """

    problem_name = "PaperRZDT1"

    def true_objectives(self, x):
        t = self.normalize(x)
        f1 = float(t[0])
        g = 1.0 + 9.0 / max(self.d - 1, 1) * float(np.sum(t[1:]))
        f2 = g * (1.0 - np.sqrt(max(f1 / g, 0.0)))
        f3 = -((f1 - 0.5) ** 2) + 0.04
        return f1, float(f2), float(f3)

    def sigma_func(self, x, f_val=0.0):
        if not self.heteroscedastic:
            return self.sigma_level
        u = float(self.normalize(x)[0])
        return self.sigma_level * (0.5 + 2.0 * np.sqrt(max(u, 0.0)))


class PaperRZDT2(RZDT2):
    """Paper RZDT2 alias with explicit problem name."""

    problem_name = "PaperRZDT2"


class PaperRZDT5RR(RZDT5RR):
    """Paper RZDT5_RR alias with explicit problem name."""

    problem_name = "PaperRZDT5_RR"


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

    def hvd_residual_variance_cap(self, output_index=0):
        del output_index
        return float((2.5 * self.sigma_level) ** 2)

    def risk_class(self, x):
        _, q, spread = self._policy_state(x)
        if q < 0.45:
            q_bin = 0
        elif q < 0.60:
            q_bin = 1
        elif q < 0.80:
            q_bin = 2
        else:
            q_bin = 3
        if spread < 0.08:
            spread_bin = 0
        elif spread < 0.22:
            spread_bin = 1
        else:
            spread_bin = 2
        return 10 * q_bin + spread_bin

    def structured_candidates(self, n=10, rng=None):
        rng = rng or np.random.default_rng()
        lo, hi = self.int_bounds()
        u_anchors = np.array([0, 10, 25, 40, 60, 80, 100], dtype=int)
        q_anchors = np.array([0, 50, 60, 70, 80, 100], dtype=int)
        rows = []
        for u in u_anchors:
            for q in q_anchors:
                x = [int(np.clip(u, lo[0], hi[0]))]
                x.extend([int(np.clip(q, lo[j], hi[j])) for j in range(1, self.d)])
                rows.append(tuple(x))
        order = rng.permutation(len(rows))
        return [rows[int(idx)] for idx in order[: max(0, int(n))]]

    def state_anchor_points(self, n=10, rng=None):
        rng = rng or np.random.default_rng()
        anchors = []
        u_vals = np.array([0.05, 0.15, 0.30, 0.40, 0.55, 0.75, 0.95])
        q_vals = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
        for u in u_vals:
            for q in q_vals:
                anchors.append(np.array([u, q], dtype=float))
        order = rng.permutation(len(anchors))
        return [anchors[int(idx)] for idx in order[: max(0, int(n))]]

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        rng = rng or np.random.default_rng()
        anchor = np.asarray(anchor, dtype=float)
        u = float(anchor[0])
        q = float(anchor[1]) if len(anchor) > 1 else 0.5
        lo, hi = self.int_bounds()
        rows = []
        for rep in range(max(1, int(n))):
            if rep == 0:
                u_i = int(np.round(self.L * np.clip(u, 0.0, 1.0)))
                q_i = int(np.round(self.L * np.clip(q, 0.0, 1.0)))
                x = [int(np.clip(u_i, lo[0], hi[0]))]
                x.extend([
                    int(np.clip(q_i, lo[j], hi[j]))
                    for j in range(1, self.d)
                ])
                rows.append(tuple(x))
                continue
            u_j = int(np.round(self.L * np.clip(u + rng.normal(0.0, 0.035), 0.0, 1.0)))
            q_j = int(np.round(self.L * np.clip(q + rng.normal(0.0, 0.035), 0.0, 1.0)))
            x = [int(np.clip(u_j, lo[0], hi[0]))]
            for j in range(1, self.d):
                tail = q_j
                if j == 1 and rng.random() < 0.35:
                    tail = int(np.round(q_j + rng.normal(0.0, 5.0)))
                x.append(int(np.clip(tail, lo[j], hi[j])))
            rows.append(tuple(x))
        return rows

    def initial_samples(self, n=5, rng=None):
        del rng
        anchors = [
            (0, 70, 70, 70, 70),
            (50, 70, 70, 70, 70),
            (25, 50, 50, 50, 50),
            (25, 90, 90, 90, 90),
            (75, 70, 70, 70, 70),
            (0, 100, 100, 100, 100),
            (100, 0, 0, 0, 0),
        ]
        lo, hi = self.int_bounds()
        rows = []
        for anchor in anchors[: max(0, int(n))]:
            x = tuple(
                int(np.clip(anchor[j] if j < len(anchor) else anchor[-1], lo[j], hi[j]))
                for j in range(self.d)
            )
            rows.append(x)
        return rows

    def all_axis_solutions(self):
        lo, hi = self.int_bounds()
        rows = []
        for x0 in range(int(lo[0]), int(hi[0]) + 1):
            for q in range(int(lo[0]), int(hi[0]) + 1, 5):
                x = [x0]
                x.extend([q] * (self.d - 1))
                rows.append(tuple(x))
        return rows


class StatePolicyMetaFeatureMap:
    """Low-dimensional basis for high-dimensional policy-state problems."""

    feature_dim = 11

    def __init__(self, problem):
        self.problem = problem

    def features(self, x):
        u, q, spread = self.problem.policy_state(x)
        reference_q = float(getattr(self.problem, "reference_q", 0.70))
        return np.array([
            u,
            q,
            spread,
            u ** 2,
            q ** 2,
            spread ** 2,
            u * q,
            u * spread,
            q * spread,
            np.sin(np.pi * u),
            abs(q - reference_q),
        ], dtype=float)

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(x) for x in X])


class HighDimStatePolicyRZDT1(TestProblem):
    """High-dimensional raw policy with a low-dimensional state optimum.

    Raw decisions can have thousands of coordinates, but the objective and
    chance constraint depend on a policy-state summary `(u, q, spread)`.
    This is the synthetic stress test for SC-style search: state candidates can
    invert useful meta anchors into raw policies, while raw random/trust-region
    candidates must discover the same low-spread tail pattern in a huge space.
    """

    problem_name = "HighDimStatePolicyRZDT1"
    variance_features = (0, 1, 2)
    recommended_partition_features = (0, 1, 2)

    def __init__(self, d=1000, L=100, sigma=0.04, heteroscedastic=True, alpha=0.05):
        super().__init__(d=d, L=L, sigma=sigma, heteroscedastic=heteroscedastic, alpha=alpha)
        self.u_star = 0.22
        self.q_star = 0.72
        self.constraint_u_center = 0.25
        self.constraint_q_center = self.q_star
        self.constraint_spread_center = 0.0
        self.constraint_radius_scale = 1.0

    def policy_state(self, x):
        z = self.normalize(x)
        u = float(z[0]) if len(z) else 0.0
        tail = z[1:] if len(z) > 1 else np.array([self.q_star])
        q = float(np.mean(tail))
        spread = float(np.std(tail))
        return u, q, spread

    def observable_state_exposure(self, x):
        """Observable control/state summary, independent of hidden targets."""

        z = self.normalize(x)
        head = z[:1]
        tail = z[1:] if len(z) > 1 else z
        return grouped_policy_state_exposure(
            z,
            (head, tail),
            channel_names=("primary_control", "aggregate_tail_control"),
            provider="state_policy_control_schema",
        )

    def _objectives_from_state(self, u, q, spread):
        state_loss = (
            2.5 * (u - self.u_star) ** 2
            + 5.5 * (q - self.q_star) ** 2
            + 0.6 * spread ** 2
        )
        f1 = 0.28 + state_loss
        f2 = 0.32 + state_loss + 0.05 * u + 0.02 * abs(q - self.q_star)
        if getattr(self, "task_geometry_is_nominal", True):
            pocket = (
                ((u - 0.25) / 0.18) ** 2
                + ((q - self.q_star) / 0.12) ** 2
                + 0.7 * (spread / 0.12) ** 2
            )
        else:
            radius = float(self.constraint_radius_scale)
            pocket = (
                ((u - self.constraint_u_center) / (0.18 * radius)) ** 2
                + ((q - self.constraint_q_center) / (0.12 * radius)) ** 2
                + 0.7 * (
                    (spread - self.constraint_spread_center) / (0.12 * radius)
                ) ** 2
            )
        f3 = 0.09 * (pocket - 1.0)
        return float(f1), float(f2), float(f3)

    def true_objectives(self, x):
        return self._objectives_from_state(*self.policy_state(x))

    def _sigma_from_state(self, u, q, spread):
        if not self.heteroscedastic:
            return self.sigma_level
        q_gap = min(abs(q - self.q_star) / max(self.q_star, 1e-8), 1.0)
        return self.sigma_level * (
            0.30 + 0.75 * q_gap + 0.45 * np.sin(np.pi * u) ** 2
            + 0.70 * min(spread / 0.35, 1.0)
        )

    def sigma_func(self, x, f_val=0.0):
        del f_val
        return self._sigma_from_state(*self.policy_state(x))

    def risk_class(self, x):
        _, q, spread = self.policy_state(x)
        q_bin = int(np.clip(np.floor(q * 5.0), 0, 4))
        spread_bin = 0 if spread < 0.04 else (1 if spread < 0.12 else 2)
        return 10 * q_bin + spread_bin

    def hvd_residual_variance_cap(self, output_index=0):
        del output_index
        return float((2.8 * self.sigma_level) ** 2)

    def hvd_features(self, x):
        u, q, spread = self.policy_state(x)
        return np.array([
            1.0,
            u,
            q,
            spread,
            u ** 2,
            q ** 2,
            spread ** 2,
            abs(q - self.q_star),
            np.sin(np.pi * u),
        ], dtype=float)

    def gpr_basis_map(self):
        return StatePolicyMetaFeatureMap(self)

    def recommendation_random_pool_size(self):
        return 0

    def surrogate_basis_map(self):
        return StatePolicyMetaFeatureMap(self)

    def _constant_tail_x(self, u, q):
        lo, hi = self.int_bounds()
        u_i = int(np.clip(round(float(u) * self.L), lo[0], hi[0]))
        q_i = int(np.clip(round(float(q) * self.L), lo[-1], hi[-1]))
        if self.d <= 1:
            return (u_i,)
        return tuple([u_i] + [q_i] * (self.d - 1))

    def scalarized_true_best_feasible(self, weights):
        weights = np.asarray(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        z_alpha = norm.ppf(1 - self.alpha)
        best = None
        for u_i in range(0, self.L + 1):
            u = u_i / float(self.L)
            for q_i in range(0, self.L + 1):
                q = q_i / float(self.L)
                f1, f2, f3 = self._objectives_from_state(u, q, 0.0)
                x = self._constant_tail_x(u, q)
                # Subclasses can replace pointwise noise with cumulative
                # shared-shock risk, so the oracle must use the same public
                # truth interface as feasibility evaluation.
                margin = f3 + z_alpha * float(self.true_sigma(x)[2]) - self.tau
                if margin > 0.0:
                    continue
                obj = float(weights[0] * f1 + weights[1] * f2)
                item = (obj, u, q)
                if best is None or item < best:
                    best = item
        if best is None:
            return None, float("inf")
        return self._constant_tail_x(best[1], best[2]), float(best[0])

    def initial_samples(self, n=5, rng=None):
        del rng
        anchors = [
            (0.05, 0.50),
            (0.50, 0.50),
            (0.25, 0.55),
            (0.25, 0.90),
            (0.75, 0.50),
            (0.10, 0.82),
            (0.90, 0.20),
            (0.40, 0.65),
        ]
        rows = [self._constant_tail_x(u, q) for u, q in anchors[: max(0, int(n))]]
        return rows

    def structured_candidates(self, n=10, rng=None):
        rng = rng or np.random.default_rng()
        u_vals = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.65, 0.90])
        q_vals = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        rows = [self._constant_tail_x(u, q) for u in u_vals for q in q_vals]
        order = rng.permutation(len(rows))
        return [rows[int(idx)] for idx in order[: max(0, int(n))]]

    def recommendation_refinement_candidates(self):
        """Dense, non-oracle meta grid for final high-dimensional selection.

        The raw search space may have thousands of coordinates, but this
        synthetic family declares that policies are judged through `(u, q,
        spread)`.  The refinement pool therefore covers that state-policy
        manifold directly instead of falling back to raw random points whose
        tail spread is almost surely large.
        """
        cache = getattr(self, "_recommendation_refinement_cache", None)
        if cache is not None:
            return list(cache)
        # Fixed meta grid, deliberately not expressed in terms of the hidden
        # synthetic optimum.  It is dense around the feasible transition but
        # remains a generic state-policy refinement rule.
        u_vals = np.round(np.arange(0.10, 0.361, 0.02), 2)
        q_vals = np.round(np.arange(0.60, 0.841, 0.02), 2)
        rows = [self._constant_tail_x(u, q) for u in u_vals for q in q_vals]
        rows = tuple(dict.fromkeys(rows))
        self._recommendation_refinement_cache = rows
        return list(rows)

    def all_axis_solutions(self):
        """No raw axis oracle for the high-dimensional state-policy case."""
        return []

    def state_anchor_points(self, n=10, rng=None):
        rng = rng or np.random.default_rng()
        anchors = []
        u_vals = np.array([0.06, 0.14, 0.22, 0.30, 0.38, 0.50, 0.66, 0.84])
        q_vals = np.array([0.52, 0.60, 0.68, 0.72, 0.76, 0.84, 0.92])
        for u in u_vals:
            for q in q_vals:
                anchors.append(np.array([u, q], dtype=float))
        order = rng.permutation(len(anchors))
        return [anchors[int(idx)] for idx in order[: max(0, int(n))]]

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        rng = rng or np.random.default_rng()
        anchor = np.asarray(anchor, dtype=float)
        u = float(anchor[0])
        q = float(anchor[1]) if len(anchor) > 1 else 0.70
        rows = []
        for rep in range(max(1, int(n))):
            if rep == 0:
                rows.append(self._constant_tail_x(u, q))
                continue
            u_j = np.clip(u + rng.normal(0.0, 0.025), 0.0, 1.0)
            q_j = np.clip(q + rng.normal(0.0, 0.025), 0.0, 1.0)
            rows.append(self._constant_tail_x(u_j, q_j))
        return rows


class FactorShockStatePolicyRZDT1(CumulativeRiskFeatureProvider, HighDimStatePolicyRZDT1):
    """State-policy benchmark with an explicit cumulative risk decomposition.

    The chance-constraint noise variance is generated by

        A(x)^T Lambda A(x) + N(x)^T B N(x) + N(x)^T omega + floor,

    where ``A`` are idiosyncratic regime exposures and ``N`` are shared-shock
    exposures.  This is the first synthetic where factor-HVD has a real target
    beyond pointwise residual smoothing.
    """

    problem_name = "FactorShockStatePolicyRZDT1"
    variance_features = (0, 1, 2)
    recommended_partition_features = (0, 1, 2)

    def __init__(
        self,
        d=1000,
        L=100,
        sigma=0.04,
        heteroscedastic=True,
        alpha=0.05,
        shared_shock_scale=1.0,
        task_geometry_shift=None,
        task_geometry_radius_scale=1.0,
    ):
        super().__init__(
            d=d,
            L=L,
            sigma=sigma,
            heteroscedastic=heteroscedastic,
            alpha=alpha,
        )
        self.shared_shock_scale = max(float(shared_shock_scale), 0.0)
        shift, radius = _task_geometry(
            task_geometry_shift, task_geometry_radius_scale)
        self.task_geometry_shift = shift
        self.task_geometry_radius_scale = radius
        self.task_geometry_is_nominal = bool(
            np.all(shift == 0.0) and radius == 1.0)
        if not self.task_geometry_is_nominal:
            self.u_star = float(np.clip(0.22 + shift[0], 0.05, 0.55))
            self.q_star = float(np.clip(0.72 + shift[1], 0.35, 0.95))
            self.constraint_u_center = float(np.clip(
                self.u_star + 0.03, 0.05, 0.60))
            self.constraint_q_center = self.q_star
            self.constraint_spread_center = float(np.clip(
                shift[2], 0.0, 0.12))
            self.constraint_radius_scale = radius

    def _risk_exposures_from_state(self, u, q, spread):
        q_gap = abs(q - self.q_star)
        u_gap = abs(u - self.u_star)
        spread_scaled = min(spread / 0.35, 1.0)
        a = np.array([
            0.25 + min(u_gap / 0.45, 1.0),
            0.20 + min(q_gap / 0.35, 1.0),
            0.10 + spread_scaled,
        ], dtype=float)
        n = np.array([
            0.10 + np.sin(np.pi * u) ** 2,
            0.15 + min(q_gap / 0.30, 1.0) + 0.50 * spread_scaled,
        ], dtype=float)
        return a, n

    def _reference_risk_exposure(self):
        return RiskExposure(
            np.zeros(3, dtype=float),
            np.zeros(2, dtype=float),
            local_names=("u_gap", "q_gap", "spread"),
            shared_names=("oscillation", "tail_shock"),
        )

    def risk_exposures(self, x, output_index=1):
        del output_index
        a, n = self._risk_exposures_from_state(*self.policy_state(x))
        return RiskExposure(
            a,
            n,
            local_names=("u_gap", "q_gap", "spread"),
            shared_names=("oscillation", "tail_shock"),
            meta={
                "provider": "FactorShockStatePolicyRZDT1",
                "shared_shock_scale": float(self.shared_shock_scale),
            },
        )

    def state_anchor_points(self, n=10, rng=None):
        return CumulativeRiskFeatureProvider.state_anchor_points(self, n=n, rng=rng)

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        return CumulativeRiskFeatureProvider.inverse_state_anchor(
            self,
            anchor,
            rng=rng,
            n=n,
        )

    def cumulative_risk_parameters(self, output_index=1):
        """Return the ground-truth cumulative variance parameters.

        ``output_index`` follows the scalarized convention: output 1 is the
        chance constraint.  Output 0 keeps a simpler objective-noise model.
        """
        scale = float(self.sigma_level) ** 2
        if int(output_index) not in (1, 2):
            return None
        return CumulativeRiskParameters(
            Lambda=scale * np.array([0.10, 0.25, 0.70], dtype=float),
            B=scale * self.shared_shock_scale * np.array([
                [0.85, 0.42],
                [0.42, 0.60],
            ], dtype=float),
            omega=scale * np.array([0.08, 0.16], dtype=float),
            floor=float((0.10 * self.sigma_level) ** 2),
        )

    def cumulative_risk_feature_names(self, output_index=1):
        if self.cumulative_risk_parameters(output_index=output_index) is None:
            return None
        return cumulative_feature_names(self._reference_risk_exposure())

    def cumulative_risk_features(self, x, output_index=1):
        if self.cumulative_risk_parameters(output_index=output_index) is None:
            return None
        return cumulative_feature_vector(self.risk_exposures(x, output_index=output_index))

    def hvd_features(self, x):
        u, q, spread = self.policy_state(x)
        a, n = self.risk_exposures(x)
        return np.array([
            1.0,
            u,
            q,
            spread,
            u ** 2,
            q ** 2,
            spread ** 2,
            abs(q - self.q_star),
            np.sin(np.pi * u),
            a[0],
            a[1],
            a[2],
            n[0],
            n[1],
        ], dtype=float)

    def true_sigma(self, x):
        u, q, spread = self.policy_state(x)
        obj_scale = 0.55 + 0.35 * np.sin(np.pi * u) ** 2 + 0.25 * min(spread / 0.35, 1.0)
        obj_sig = float(max(0.35 * self.sigma_level * obj_scale, 1e-8))
        risk = self.true_cumulative_risk_decomposition(x, output_index=1)
        con_sig = float(np.sqrt(max(risk["total"], 1e-12)))
        return np.array([obj_sig, obj_sig, con_sig], dtype=float)

    def sigma_func(self, x, f_val=0.0):
        del f_val
        return float(self.true_sigma(x)[2])

    def hvd_residual_variance_cap(self, output_index=0):
        if int(output_index) in (1, 2):
            scale = float(self.sigma_level) ** 2
            return float(8.0 * max(1.0, self.shared_shock_scale) * scale)
        return float((2.0 * self.sigma_level) ** 2)


class InventorySupplyChainProblem(CumulativeRiskFeatureProvider, TestProblem):
    """Multi-node inventory control with local and shared demand risk."""

    problem_name = "InventorySupplyChain"
    variance_features = (0, 1, 2)
    recommended_partition_features = (0, 1, 2)

    def __init__(
        self,
        d=6,
        L=100,
        sigma=0.04,
        heteroscedastic=True,
        alpha=0.05,
        task_geometry_shift=None,
        task_geometry_radius_scale=1.0,
    ):
        super().__init__(d=max(4, int(d)), L=L, sigma=sigma, heteroscedastic=heteroscedastic, alpha=alpha)
        shift, radius = _task_geometry(
            task_geometry_shift, task_geometry_radius_scale)
        self.task_geometry_shift = shift
        self.task_geometry_radius_scale = radius
        self.task_geometry_is_nominal = bool(
            np.all(shift == 0.0) and radius == 1.0)
        if self.task_geometry_is_nominal:
            self.target_stock = 0.58
            self.target_reorder = 0.36
            self.target_safety = 0.42
        else:
            self.target_stock = float(np.clip(0.58 + shift[0], 0.25, 0.85))
            self.target_reorder = float(np.clip(0.36 + shift[1], 0.15, 0.75))
            self.target_safety = float(np.clip(0.42 + shift[2], 0.20, 0.80))
        self.tau = 0.0

    def _policy_summary(self, x):
        z = self.normalize(x)
        thirds = np.array_split(z, 3)
        stock = float(np.mean(thirds[0]))
        reorder = float(np.mean(thirds[1]))
        safety = float(np.mean(thirds[2]))
        dispersion = float(np.std(z))
        return stock, reorder, safety, dispersion

    def observable_state_exposure(self, x):
        """Observable inventory-control channels before any outcome query."""

        z = self.normalize(x)
        groups = tuple(np.array_split(z, 3))
        return grouped_policy_state_exposure(
            z,
            groups,
            channel_names=("stock_control", "reorder_control", "safety_control"),
            provider="inventory_control_schema",
        )

    def true_objectives(self, x):
        stock, reorder, safety, dispersion = self._policy_summary(x)
        holding = 0.7 * max(stock - self.target_stock, 0.0) ** 2
        backlog = 1.8 * max(self.target_stock - stock, 0.0) ** 2
        reorder_loss = 1.4 * (reorder - self.target_reorder) ** 2
        safety_loss = 1.1 * (safety - self.target_safety) ** 2
        f1 = 0.25 + holding + backlog + reorder_loss + 0.3 * dispersion ** 2
        f2 = 0.30 + safety_loss + 0.5 * reorder_loss + 0.5 * backlog
        if self.task_geometry_is_nominal:
            service_gap = (
                ((stock - 0.56) / 0.20) ** 2
                + ((reorder - 0.34) / 0.22) ** 2
                + ((safety - 0.44) / 0.18) ** 2
                + 0.4 * (dispersion / 0.25) ** 2
            )
        else:
            radius = self.task_geometry_radius_scale
            service_gap = (
                ((stock - (self.target_stock - 0.02)) / (0.20 * radius)) ** 2
                + ((reorder - (self.target_reorder - 0.02)) / (0.22 * radius)) ** 2
                + ((safety - (self.target_safety + 0.02)) / (0.18 * radius)) ** 2
                + 0.4 * (dispersion / (0.25 * radius)) ** 2
            )
        f3 = 0.10 * (service_gap - 1.0)
        return float(f1), float(f2), float(f3)

    def _reference_risk_exposure(self):
        return RiskExposure(
            np.zeros(3, dtype=float),
            np.zeros(2, dtype=float),
            local_names=("backlog", "holding", "stockout"),
            shared_names=("demand_level", "demand_volatility"),
        )

    def risk_exposures(self, x, output_index=1):
        del output_index
        stock, reorder, safety, dispersion = self._policy_summary(x)
        if self.task_geometry_is_nominal:
            backlog = max(0.0, 0.62 - stock)
            holding = max(0.0, stock - 0.58)
            stockout = max(0.0, 0.48 - safety) + 0.35 * max(
                0.0, 0.30 - reorder)
            demand_level = 0.20 + max(
                0.0, 0.55 - stock) + 0.35 * abs(reorder - 0.35)
            demand_volatility = 0.15 + dispersion + 0.5 * max(
                0.0, 0.50 - safety)
        else:
            backlog = max(0.0, self.target_stock + 0.04 - stock)
            holding = max(0.0, stock - self.target_stock)
            stockout = max(0.0, self.target_safety + 0.06 - safety) + 0.35 * max(
                0.0, self.target_reorder - 0.06 - reorder)
            demand_level = 0.20 + max(
                0.0, self.target_stock - 0.03 - stock) + 0.35 * abs(
                    reorder - (self.target_reorder - 0.01))
            demand_volatility = 0.15 + dispersion + 0.5 * max(
                0.0, self.target_safety + 0.08 - safety)
        return RiskExposure(
            [0.20 + backlog, 0.15 + holding, 0.10 + stockout],
            [demand_level, demand_volatility],
            local_names=("backlog", "holding", "stockout"),
            shared_names=("demand_level", "demand_volatility"),
            meta={"provider": self.problem_name},
        )

    def cumulative_risk_parameters(self, output_index=1):
        if int(output_index) not in (1, 2):
            return None
        scale = float(self.sigma_level) ** 2
        return CumulativeRiskParameters(
            Lambda=scale * np.array([0.60, 0.18, 0.95], dtype=float),
            B=scale * np.array([[0.90, 0.35], [0.35, 0.70]], dtype=float),
            omega=scale * np.array([0.18, 0.24], dtype=float),
            floor=float((0.12 * self.sigma_level) ** 2),
        )

    def cumulative_risk_features(self, x, output_index=1):
        if self.cumulative_risk_parameters(output_index=output_index) is None:
            return None
        return cumulative_feature_vector(self.risk_exposures(x, output_index=output_index))

    def cumulative_risk_feature_names(self, output_index=1):
        if self.cumulative_risk_parameters(output_index=output_index) is None:
            return None
        return cumulative_feature_names(self._reference_risk_exposure())

    def true_cumulative_risk_decomposition(self, x, output_index=1):
        params = self.cumulative_risk_parameters(output_index=output_index)
        if params is None:
            return None
        return decompose_cumulative_risk(self.risk_exposures(x, output_index), params)

    def true_sigma(self, x):
        stock, reorder, safety, dispersion = self._policy_summary(x)
        obj_sig = float(max(self.sigma_level * (0.25 + 0.2 * dispersion + 0.2 * abs(stock - reorder)), 1e-8))
        risk = self.true_cumulative_risk_decomposition(x, output_index=1)
        return np.array([obj_sig, obj_sig, float(np.sqrt(max(risk["total"], 1e-12)))])

    def risk_class(self, x):
        stock, reorder, safety, _ = self._policy_summary(x)
        stock_bin = 0 if stock < 0.45 else (1 if stock < 0.65 else 2)
        service_bin = 0 if safety < 0.35 else (1 if reorder < 0.45 else 2)
        return 10 * stock_bin + service_bin

    def hvd_features(self, x):
        stock, reorder, safety, dispersion = self._policy_summary(x)
        A, N = self.risk_exposures(x)
        return np.array([
            1.0, stock, reorder, safety, dispersion,
            stock ** 2, reorder ** 2, safety ** 2,
            A[0], A[1], A[2], N[0], N[1],
        ], dtype=float)

    def gpr_basis_map(self):
        return StatePolicyMetaFeatureMap(_SummaryAdapter(self, self._policy_summary))

    def surrogate_basis_map(self):
        return self.gpr_basis_map()

    def _constant_policy(self, stock, reorder, safety):
        vals = [stock, reorder, safety]
        rows = []
        for j in range(self.d):
            v = vals[min(2, int(3 * j / max(self.d, 1)))]
            rows.append(int(np.clip(round(v * self.L), 0, self.L)))
        return tuple(rows)

    def initial_samples(self, n=5, rng=None):
        del rng
        anchors = [(0.35, 0.25, 0.30), (0.55, 0.35, 0.45), (0.75, 0.45, 0.50),
                   (0.60, 0.55, 0.30), (0.45, 0.30, 0.65), (0.85, 0.20, 0.20)]
        return [self._constant_policy(*a) for a in anchors[: max(0, int(n))]]

    def structured_candidates(self, n=10, rng=None):
        rng = rng or np.random.default_rng()
        anchors = [(s, r, q) for s in [0.35, 0.50, 0.60, 0.75]
                   for r in [0.25, 0.35, 0.45, 0.60]
                   for q in [0.30, 0.45, 0.60]]
        order = rng.permutation(len(anchors))
        return [self._constant_policy(*anchors[int(i)]) for i in order[: max(0, int(n))]]

    def state_anchor_points(self, n=10, rng=None):
        return CumulativeRiskFeatureProvider.state_anchor_points(self, n=n, rng=rng)

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        return CumulativeRiskFeatureProvider.inverse_state_anchor(
            self,
            anchor,
            rng=rng,
            n=n,
        )

    def _scalarized_true_best_feasible_on_grid(
        self, weights, stock_axis, reorder_axis, safety_axis,
    ):
        weights = np.asarray(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        z_alpha = norm.ppf(1 - self.alpha)
        stock, reorder, safety, dispersion = _constant_three_block_grid(
            self, stock_axis, reorder_axis, safety_axis)
        holding = 0.7 * np.maximum(
            stock - self.target_stock, 0.0) ** 2
        backlog_loss = 1.8 * np.maximum(
            self.target_stock - stock, 0.0) ** 2
        reorder_loss = 1.4 * (reorder - self.target_reorder) ** 2
        safety_loss = 1.1 * (safety - self.target_safety) ** 2
        f1 = (
            0.25 + holding + backlog_loss + reorder_loss
            + 0.3 * dispersion ** 2
        )
        f2 = (
            0.30 + safety_loss + 0.5 * reorder_loss
            + 0.5 * backlog_loss
        )
        if self.task_geometry_is_nominal:
            service_gap = (
                ((stock - 0.56) / 0.20) ** 2
                + ((reorder - 0.34) / 0.22) ** 2
                + ((safety - 0.44) / 0.18) ** 2
                + 0.4 * (dispersion / 0.25) ** 2
            )
            backlog = np.maximum(0.0, 0.62 - stock)
            holding_exposure = np.maximum(0.0, stock - 0.58)
            stockout = (
                np.maximum(0.0, 0.48 - safety)
                + 0.35 * np.maximum(0.0, 0.30 - reorder)
            )
            demand_level = (
                0.20 + np.maximum(0.0, 0.55 - stock)
                + 0.35 * np.abs(reorder - 0.35)
            )
            demand_volatility = (
                0.15 + dispersion
                + 0.5 * np.maximum(0.0, 0.50 - safety)
            )
        else:
            radius = self.task_geometry_radius_scale
            service_gap = (
                (
                    (stock - (self.target_stock - 0.02))
                    / (0.20 * radius)
                ) ** 2
                + (
                    (reorder - (self.target_reorder - 0.02))
                    / (0.22 * radius)
                ) ** 2
                + (
                    (safety - (self.target_safety + 0.02))
                    / (0.18 * radius)
                ) ** 2
                + 0.4 * (dispersion / (0.25 * radius)) ** 2
            )
            backlog = np.maximum(
                0.0, self.target_stock + 0.04 - stock)
            holding_exposure = np.maximum(
                0.0, stock - self.target_stock)
            stockout = (
                np.maximum(
                    0.0, self.target_safety + 0.06 - safety)
                + 0.35 * np.maximum(
                    0.0, self.target_reorder - 0.06 - reorder)
            )
            demand_level = (
                0.20
                + np.maximum(
                    0.0, self.target_stock - 0.03 - stock)
                + 0.35 * np.abs(
                    reorder - (self.target_reorder - 0.01))
            )
            demand_volatility = (
                0.15 + dispersion
                + 0.5 * np.maximum(
                    0.0, self.target_safety + 0.08 - safety)
            )
        f3 = 0.10 * (service_gap - 1.0)
        local = np.vstack([
            0.20 + backlog,
            0.15 + holding_exposure,
            0.10 + stockout,
        ])
        shared = np.vstack([demand_level, demand_volatility])
        variance = _batched_cumulative_variance(
            local,
            shared,
            self.cumulative_risk_parameters(output_index=1),
        )
        objective = weights[0] * f1 + weights[1] * f2
        feasible = f3 + z_alpha * np.sqrt(variance) <= self.tau
        if not np.any(feasible):
            return None, float("inf")
        ranked = np.where(feasible, objective, np.inf)
        best_index = int(np.argmin(ranked))
        best_x = self._constant_policy(
            stock[best_index],
            reorder[best_index],
            safety[best_index],
        )
        return best_x, float(objective[best_index])

    def scalarized_true_best_feasible(self, weights):
        return self._scalarized_true_best_feasible_on_grid(
            weights,
            np.linspace(0.35, 0.80, 46),
            np.linspace(0.22, 0.60, 39),
            np.linspace(0.28, 0.68, 41),
        )

    def recommendation_refinement_candidates(self):
        rows = []
        for stock in np.linspace(0.45, 0.72, 15):
            for reorder in np.linspace(0.28, 0.50, 12):
                for safety in np.linspace(0.34, 0.58, 13):
                    rows.append(self._constant_policy(stock, reorder, safety))
        return list(dict.fromkeys(rows))

    def all_axis_solutions(self):
        return []

    def hvd_residual_variance_cap(self, output_index=0):
        if int(output_index) in (1, 2):
            return float(8.0 * self.sigma_level ** 2)
        return float(2.0 * self.sigma_level ** 2)


class QueueResourceControlProblem(CumulativeRiskFeatureProvider, TestProblem):
    """Network queue/resource allocation with bursty shared load."""

    problem_name = "QueueResourceControl"
    variance_features = (0, 1, 2)
    recommended_partition_features = (0, 1, 2)

    def __init__(
        self,
        d=6,
        L=100,
        sigma=0.04,
        heteroscedastic=True,
        alpha=0.05,
        task_geometry_shift=None,
        task_geometry_radius_scale=1.0,
    ):
        super().__init__(d=max(4, int(d)), L=L, sigma=sigma, heteroscedastic=heteroscedastic, alpha=alpha)
        shift, radius = _task_geometry(
            task_geometry_shift, task_geometry_radius_scale)
        self.task_geometry_shift = shift
        self.task_geometry_radius_scale = radius
        self.task_geometry_is_nominal = bool(
            np.all(shift == 0.0) and radius == 1.0)
        if self.task_geometry_is_nominal:
            self.target_capacity = 0.64
            self.target_priority = 0.38
            self.target_smoothing = 0.52
        else:
            self.target_capacity = float(np.clip(0.64 + shift[0], 0.30, 0.90))
            self.target_priority = float(np.clip(0.38 + shift[1], 0.15, 0.80))
            self.target_smoothing = float(np.clip(0.52 + shift[2], 0.20, 0.85))

    def _policy_summary(self, x):
        z = self.normalize(x)
        thirds = np.array_split(z, 3)
        capacity = float(np.mean(thirds[0]))
        priority = float(np.mean(thirds[1]))
        smoothing = float(np.mean(thirds[2]))
        imbalance = float(np.std(z))
        return capacity, priority, smoothing, imbalance

    def observable_state_exposure(self, x):
        """Observable resource-control channels before any outcome query."""

        z = self.normalize(x)
        groups = tuple(np.array_split(z, 3))
        return grouped_policy_state_exposure(
            z,
            groups,
            channel_names=(
                "capacity_control", "priority_control", "smoothing_control"
            ),
            provider="queue_control_schema",
        )

    def true_objectives(self, x):
        capacity, priority, smoothing, imbalance = self._policy_summary(x)
        if self.task_geometry_is_nominal:
            wait_loss = (
                2.0 * max(0.0, 0.58 - capacity) ** 2
                + 0.8 * (priority - 0.36) ** 2
            )
            resource_loss = (
                0.7 * max(0.0, capacity - 0.72) ** 2
                + 0.5 * (smoothing - 0.50) ** 2
            )
        else:
            wait_loss = 2.0 * max(
                0.0, self.target_capacity - 0.06 - capacity) ** 2 + 0.8 * (
                    priority - (self.target_priority - 0.02)) ** 2
            resource_loss = 0.7 * max(
                0.0, capacity - (self.target_capacity + 0.08)) ** 2 + 0.5 * (
                    smoothing - (self.target_smoothing - 0.02)) ** 2
        f1 = 0.24 + wait_loss + 0.25 * imbalance ** 2
        f2 = 0.30 + resource_loss + 0.35 * wait_loss
        if self.task_geometry_is_nominal:
            pocket = (
                ((capacity - 0.64) / 0.18) ** 2
                + ((priority - 0.38) / 0.20) ** 2
                + ((smoothing - 0.52) / 0.20) ** 2
                + 0.45 * (imbalance / 0.28) ** 2
            )
        else:
            radius = self.task_geometry_radius_scale
            pocket = (
                ((capacity - self.target_capacity) / (0.18 * radius)) ** 2
                + ((priority - self.target_priority) / (0.20 * radius)) ** 2
                + ((smoothing - self.target_smoothing) / (0.20 * radius)) ** 2
                + 0.45 * (imbalance / (0.28 * radius)) ** 2
            )
        f3 = 0.095 * (pocket - 1.0)
        return float(f1), float(f2), float(f3)

    def _reference_risk_exposure(self):
        return RiskExposure(
            np.zeros(3, dtype=float),
            np.zeros(2, dtype=float),
            local_names=("queue", "wait", "utilization"),
            shared_names=("arrival_burst", "common_load"),
        )

    def risk_exposures(self, x, output_index=1):
        del output_index
        capacity, priority, smoothing, imbalance = self._policy_summary(x)
        if self.task_geometry_is_nominal:
            queue = max(0.0, 0.62 - capacity) + 0.25 * imbalance
            wait = max(0.0, 0.44 - priority) + 0.15 * max(
                0.0, 0.45 - smoothing)
            utilization = max(0.0, capacity - 0.70) + 0.20 * imbalance
            arrival_burst = 0.15 + max(
                0.0, 0.60 - smoothing) + 0.45 * imbalance
            common_load = 0.20 + abs(capacity - 0.64) + 0.35 * abs(
                priority - 0.38)
        else:
            queue = max(
                0.0, self.target_capacity - 0.02 - capacity) + 0.25 * imbalance
            wait = max(0.0, self.target_priority + 0.06 - priority) + 0.15 * max(
                0.0, self.target_smoothing - 0.07 - smoothing)
            utilization = max(
                0.0, capacity - (self.target_capacity + 0.06)) + 0.20 * imbalance
            arrival_burst = 0.15 + max(
                0.0, self.target_smoothing + 0.08 - smoothing) + 0.45 * imbalance
            common_load = 0.20 + abs(capacity - self.target_capacity) + 0.35 * abs(
                priority - self.target_priority)
        return RiskExposure(
            [0.10 + queue, 0.12 + wait, 0.10 + utilization],
            [arrival_burst, common_load],
            local_names=("queue", "wait", "utilization"),
            shared_names=("arrival_burst", "common_load"),
            meta={"provider": self.problem_name},
        )

    def cumulative_risk_parameters(self, output_index=1):
        if int(output_index) not in (1, 2):
            return None
        scale = float(self.sigma_level) ** 2
        return CumulativeRiskParameters(
            Lambda=scale * np.array([0.75, 0.80, 0.22], dtype=float),
            B=scale * np.array([[0.80, 0.45], [0.45, 0.78]], dtype=float),
            omega=scale * np.array([0.22, 0.18], dtype=float),
            floor=float((0.12 * self.sigma_level) ** 2),
        )

    def cumulative_risk_features(self, x, output_index=1):
        if self.cumulative_risk_parameters(output_index=output_index) is None:
            return None
        return cumulative_feature_vector(self.risk_exposures(x, output_index=output_index))

    def cumulative_risk_feature_names(self, output_index=1):
        if self.cumulative_risk_parameters(output_index=output_index) is None:
            return None
        return cumulative_feature_names(self._reference_risk_exposure())

    def true_cumulative_risk_decomposition(self, x, output_index=1):
        params = self.cumulative_risk_parameters(output_index=output_index)
        if params is None:
            return None
        return decompose_cumulative_risk(self.risk_exposures(x, output_index), params)

    def true_sigma(self, x):
        capacity, _, _, imbalance = self._policy_summary(x)
        obj_sig = float(max(self.sigma_level * (0.25 + 0.25 * imbalance + 0.1 * abs(capacity - 0.64)), 1e-8))
        risk = self.true_cumulative_risk_decomposition(x, output_index=1)
        return np.array([obj_sig, obj_sig, float(np.sqrt(max(risk["total"], 1e-12)))])

    def risk_class(self, x):
        capacity, priority, smoothing, _ = self._policy_summary(x)
        cap_bin = 0 if capacity < 0.50 else (1 if capacity < 0.70 else 2)
        burst_bin = 0 if smoothing > 0.58 else (1 if priority > 0.34 else 2)
        return 10 * cap_bin + burst_bin

    def hvd_features(self, x):
        capacity, priority, smoothing, imbalance = self._policy_summary(x)
        A, N = self.risk_exposures(x)
        return np.array([
            1.0, capacity, priority, smoothing, imbalance,
            capacity ** 2, priority ** 2, smoothing ** 2,
            A[0], A[1], A[2], N[0], N[1],
        ], dtype=float)

    def gpr_basis_map(self):
        return StatePolicyMetaFeatureMap(_SummaryAdapter(self, self._policy_summary))

    def surrogate_basis_map(self):
        return self.gpr_basis_map()

    def _constant_policy(self, capacity, priority, smoothing):
        vals = [capacity, priority, smoothing]
        return tuple(
            int(np.clip(round(vals[min(2, int(3 * j / max(self.d, 1)))] * self.L), 0, self.L))
            for j in range(self.d)
        )

    def initial_samples(self, n=5, rng=None):
        del rng
        anchors = [(0.45, 0.30, 0.42), (0.62, 0.38, 0.52), (0.78, 0.42, 0.58),
                   (0.55, 0.55, 0.35), (0.70, 0.25, 0.65), (0.35, 0.60, 0.30)]
        return [self._constant_policy(*a) for a in anchors[: max(0, int(n))]]

    def structured_candidates(self, n=10, rng=None):
        rng = rng or np.random.default_rng()
        anchors = [(c, p, s) for c in [0.45, 0.58, 0.66, 0.78]
                   for p in [0.25, 0.36, 0.45, 0.58]
                   for s in [0.36, 0.50, 0.62]]
        order = rng.permutation(len(anchors))
        return [self._constant_policy(*anchors[int(i)]) for i in order[: max(0, int(n))]]

    def state_anchor_points(self, n=10, rng=None):
        return CumulativeRiskFeatureProvider.state_anchor_points(self, n=n, rng=rng)

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        return CumulativeRiskFeatureProvider.inverse_state_anchor(
            self,
            anchor,
            rng=rng,
            n=n,
        )

    def _scalarized_true_best_feasible_on_grid(
        self, weights, capacity_axis, priority_axis, smoothing_axis,
    ):
        weights = np.asarray(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        z_alpha = norm.ppf(1 - self.alpha)
        capacity, priority, smoothing, imbalance = (
            _constant_three_block_grid(
                self,
                capacity_axis,
                priority_axis,
                smoothing_axis,
            )
        )
        if self.task_geometry_is_nominal:
            wait_loss = (
                2.0 * np.maximum(0.0, 0.58 - capacity) ** 2
                + 0.8 * (priority - 0.36) ** 2
            )
            resource_loss = (
                0.7 * np.maximum(0.0, capacity - 0.72) ** 2
                + 0.5 * (smoothing - 0.50) ** 2
            )
            pocket = (
                ((capacity - 0.64) / 0.18) ** 2
                + ((priority - 0.38) / 0.20) ** 2
                + ((smoothing - 0.52) / 0.20) ** 2
                + 0.45 * (imbalance / 0.28) ** 2
            )
            queue = (
                np.maximum(0.0, 0.62 - capacity)
                + 0.25 * imbalance
            )
            wait = (
                np.maximum(0.0, 0.44 - priority)
                + 0.15 * np.maximum(0.0, 0.45 - smoothing)
            )
            utilization = (
                np.maximum(0.0, capacity - 0.70)
                + 0.20 * imbalance
            )
            arrival_burst = (
                0.15 + np.maximum(0.0, 0.60 - smoothing)
                + 0.45 * imbalance
            )
            common_load = (
                0.20 + np.abs(capacity - 0.64)
                + 0.35 * np.abs(priority - 0.38)
            )
        else:
            wait_loss = (
                2.0 * np.maximum(
                    0.0,
                    self.target_capacity - 0.06 - capacity,
                ) ** 2
                + 0.8 * (
                    priority - (self.target_priority - 0.02)
                ) ** 2
            )
            resource_loss = (
                0.7 * np.maximum(
                    0.0,
                    capacity - (self.target_capacity + 0.08),
                ) ** 2
                + 0.5 * (
                    smoothing - (self.target_smoothing - 0.02)
                ) ** 2
            )
            radius = self.task_geometry_radius_scale
            pocket = (
                (
                    (capacity - self.target_capacity)
                    / (0.18 * radius)
                ) ** 2
                + (
                    (priority - self.target_priority)
                    / (0.20 * radius)
                ) ** 2
                + (
                    (smoothing - self.target_smoothing)
                    / (0.20 * radius)
                ) ** 2
                + 0.45 * (imbalance / (0.28 * radius)) ** 2
            )
            queue = (
                np.maximum(
                    0.0,
                    self.target_capacity - 0.02 - capacity,
                )
                + 0.25 * imbalance
            )
            wait = (
                np.maximum(
                    0.0,
                    self.target_priority + 0.06 - priority,
                )
                + 0.15 * np.maximum(
                    0.0,
                    self.target_smoothing - 0.07 - smoothing,
                )
            )
            utilization = (
                np.maximum(
                    0.0,
                    capacity - (self.target_capacity + 0.06),
                )
                + 0.20 * imbalance
            )
            arrival_burst = (
                0.15
                + np.maximum(
                    0.0,
                    self.target_smoothing + 0.08 - smoothing,
                )
                + 0.45 * imbalance
            )
            common_load = (
                0.20 + np.abs(capacity - self.target_capacity)
                + 0.35 * np.abs(
                    priority - self.target_priority)
            )
        f1 = 0.24 + wait_loss + 0.25 * imbalance ** 2
        f2 = 0.30 + resource_loss + 0.35 * wait_loss
        f3 = 0.095 * (pocket - 1.0)
        local = np.vstack([
            0.10 + queue,
            0.12 + wait,
            0.10 + utilization,
        ])
        shared = np.vstack([arrival_burst, common_load])
        variance = _batched_cumulative_variance(
            local,
            shared,
            self.cumulative_risk_parameters(output_index=1),
        )
        objective = weights[0] * f1 + weights[1] * f2
        feasible = f3 + z_alpha * np.sqrt(variance) <= self.tau
        if not np.any(feasible):
            return None, float("inf")
        ranked = np.where(feasible, objective, np.inf)
        best_index = int(np.argmin(ranked))
        best_x = self._constant_policy(
            capacity[best_index],
            priority[best_index],
            smoothing[best_index],
        )
        return best_x, float(objective[best_index])

    def scalarized_true_best_feasible(self, weights):
        return self._scalarized_true_best_feasible_on_grid(
            weights,
            np.linspace(0.45, 0.82, 38),
            np.linspace(0.24, 0.58, 35),
            np.linspace(0.34, 0.68, 35),
        )

    def recommendation_refinement_candidates(self):
        rows = []
        for capacity in np.linspace(0.52, 0.76, 13):
            for priority in np.linspace(0.28, 0.50, 12):
                for smoothing in np.linspace(0.42, 0.64, 12):
                    rows.append(self._constant_policy(capacity, priority, smoothing))
        return list(dict.fromkeys(rows))

    def all_axis_solutions(self):
        return []

    def hvd_residual_variance_cap(self, output_index=0):
        if int(output_index) in (1, 2):
            return float(8.0 * self.sigma_level ** 2)
        return float(2.0 * self.sigma_level ** 2)


class _SummaryAdapter:
    """Adapter exposing ``policy_state`` to the existing meta feature map."""

    reference_q = 0.5

    def __init__(self, problem, summary_fn):
        self.problem = problem
        self.summary_fn = summary_fn

    def policy_state(self, x):
        values = self.summary_fn(x)
        return float(values[0]), float(values[1]), float(values[3])


PROBLEM_REGISTRY = {
    "RZDT1": RZDT1,
    "RZDT2": RZDT2,
    "RZDT5_RR": RZDT5RR,
    "PaperRZDT1": PaperRZDT1,
    "PaperRZDT2": PaperRZDT2,
    "PaperRZDT5_RR": PaperRZDT5RR,
    "RegimeRZDT1": RegimeRZDT1,
    "StatePolicyRZDT1": StatePolicyRZDT1,
    "HighDimStatePolicyRZDT1": HighDimStatePolicyRZDT1,
    "FactorShockStatePolicyRZDT1": FactorShockStatePolicyRZDT1,
    "InventorySupplyChain": InventorySupplyChainProblem,
    "InventorySupplyChainProblem": InventorySupplyChainProblem,
    "QueueResourceControl": QueueResourceControlProblem,
    "QueueResourceControlProblem": QueueResourceControlProblem,
}


def make_problem(
    name="RZDT1", d=5, L=100, sigma=0.04, alpha=0.05, **problem_kwargs,
):
    cls = PROBLEM_REGISTRY[name]
    problem = cls(
        d=d,
        L=L,
        sigma=sigma,
        heteroscedastic=True,
        alpha=alpha,
        **problem_kwargs,
    )
    problem.tau = 0.0
    return problem
