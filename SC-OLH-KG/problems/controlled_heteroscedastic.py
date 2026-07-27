"""Controlled heteroscedastic optimization problems.

The suite varies the location and geometry of the constraint noise while
holding the objective and constraint-mean surface fixed.  Hidden truth is used
only by the simulator and post-run oracle audit.  The declared cumulative-risk
provider is observable from a policy; one scenario deliberately adds an
undeclared high-frequency residual to test model misspecification.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core.cumulative_risk import (
    CumulativeRiskFeatureProvider,
    CumulativeRiskParameters,
    RiskExposure,
    decompose_cumulative_risk,
)
from representation.observable_exposure import grouped_policy_state_exposure


CONTROLLED_HETERO_SCENARIOS = {
    "homoscedastic": {
        "location": "global",
        "geometry": "constant",
        "provider_exact": True,
    },
    "smooth_boundary": {
        "location": "chance_boundary",
        "geometry": "smooth_local_bump",
        "provider_exact": True,
    },
    "optimum_hotspot": {
        "location": "near_feasible_optimum",
        "geometry": "smooth_local_bump",
        "provider_exact": True,
    },
    "safe_interior_hotspot": {
        "location": "deep_safe_interior",
        "geometry": "smooth_local_bump",
        "provider_exact": True,
    },
    "regime_step": {
        "location": "boundary_crossing_regime",
        "geometry": "discontinuous_step",
        "provider_exact": True,
    },
    "sparse_axis": {
        "location": "single_control_channel",
        "geometry": "sparse_monotone",
        "provider_exact": True,
    },
    "shared_factor": {
        "location": "low_control_and_imbalanced",
        "geometry": "correlated_low_rank_factor",
        "provider_exact": True,
    },
    "hidden_periodic": {
        "location": "distributed",
        "geometry": "high_frequency_interaction",
        "provider_exact": False,
    },
}


class ControlledLatentFeatureMap:
    """Dimension-equivariant observable mean coordinate."""

    feature_dim = 10

    def __init__(self, problem):
        self.problem = problem

    def features(self, x):
        z0, z1, z2, dispersion = self.problem.policy_state(x)
        return np.asarray([
            z0,
            z1,
            z2,
            dispersion,
            z0 ** 2,
            z1 ** 2,
            z2 ** 2,
            z0 * z1,
            z0 * z2,
            z1 * z2,
        ], dtype=float)

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(x) for x in X])


class ControlledHeteroscedasticProblem(CumulativeRiskFeatureProvider):
    """High-dimensional policy problem with controlled noise geometry."""

    problem_name = "ControlledHeteroscedastic"
    simulation_noise_model = "iid_gaussian"
    variance_features = (0, 1, 2)
    recommended_partition_features = (0, 1, 2)
    prefer_direct_gpr_basis = True
    scenario = "homoscedastic"

    def __init__(
        self,
        d=1000,
        L=100,
        sigma=0.04,
        heteroscedastic=True,
        alpha=0.05,
        scenario=None,
        **unused,
    ):
        del unused
        self.d = max(3, int(d))
        self.L = int(L)
        self.sigma_level = float(sigma)
        self.heteroscedastic = bool(heteroscedastic)
        self.alpha = float(alpha)
        self.tau = 0.0
        self.ref_point = np.array([1.5, 1.5], dtype=float)
        selected = str(scenario or self.scenario)
        if selected not in CONTROLLED_HETERO_SCENARIOS:
            raise ValueError(f"unknown controlled heteroscedastic scenario: {selected}")
        self.scenario = selected
        self.problem_name = f"ControlledHetero_{selected}"
        self._oracle_cache = {}

    def int_bounds(self):
        return np.zeros(self.d, dtype=int), np.full(self.d, self.L, dtype=int)

    def normalize(self, x):
        return np.clip(np.asarray(x, dtype=float) / float(self.L), 0.0, 1.0)

    def continuous_to_int(self, x_norm):
        values = np.asarray(x_norm, dtype=float).reshape(-1)
        if len(values) != self.d:
            raise ValueError(f"expected {self.d} normalized coordinates")
        return tuple(np.clip(np.rint(values * self.L), 0, self.L).astype(int))

    def sample_random(self, rng=None):
        rng = rng or np.random.default_rng()
        return tuple(int(v) for v in rng.integers(0, self.L + 1, size=self.d))

    def policy_state(self, x):
        z = self.normalize(x)
        groups = np.array_split(z, 3)
        means = [float(np.mean(group)) for group in groups]
        dispersion = float(np.sqrt(np.mean([
            np.var(group) for group in groups
        ])))
        return means[0], means[1], means[2], dispersion

    @staticmethod
    def _mean_surfaces(z0, z1, z2, dispersion):
        imbalance = (
            (z0 - z1) ** 2 + (z1 - z2) ** 2 + (z0 - z2) ** 2
        )
        control = 0.50 * z0 + 0.30 * z1 + 0.20 * z2
        f1 = (
            0.12 + 0.45 * z0 + 0.30 * z1 + 0.25 * z2
            + 0.18 * imbalance + 0.08 * dispersion ** 2
        )
        f2 = (
            0.12 + 0.35 * z0 + 0.40 * z1 + 0.25 * z2
            + 0.18 * imbalance + 0.08 * dispersion ** 2
        )
        constraint = (
            0.055 - 0.120 * control
            + 0.012 * imbalance + 0.025 * dispersion ** 2
        )
        return f1, f2, constraint, control, imbalance

    def true_objectives(self, x):
        values = self._mean_surfaces(*self.policy_state(x))
        return float(values[0]), float(values[1]), float(values[2])

    def observable_state_exposure(self, x):
        z = self.normalize(x)
        groups = tuple(np.array_split(z, 3))
        return grouped_policy_state_exposure(
            z,
            groups,
            channel_names=("control_0", "control_1", "control_2"),
            provider="controlled_heteroscedastic_policy_schema",
        )

    @staticmethod
    def _broadcast_state(z0, z1, z2, dispersion):
        values = np.broadcast_arrays(
            np.asarray(z0, dtype=float),
            np.asarray(z1, dtype=float),
            np.asarray(z2, dtype=float),
            np.asarray(dispersion, dtype=float),
        )
        return tuple(np.asarray(value, dtype=float) for value in values)

    def _risk_arrays(self, z0, z1, z2, dispersion):
        z0, z1, z2, dispersion = self._broadcast_state(
            z0, z1, z2, dispersion)
        _, _, constraint, control, imbalance = self._mean_surfaces(
            z0, z1, z2, dispersion)
        shape = np.shape(control)
        A = np.zeros((3,) + shape, dtype=float)
        N = np.zeros((2,) + shape, dtype=float)
        hidden = np.zeros(shape, dtype=float)

        if self.scenario == "homoscedastic":
            pass
        elif self.scenario == "smooth_boundary":
            bump = np.exp(-0.5 * (constraint / 0.018) ** 2)
            A[0] = bump
            A[1] = 0.25 * np.abs(z0 - z1)
            A[2] = 0.20 * dispersion
            N[0] = 0.10 * np.maximum(1.0 - control, 0.0)
            N[1] = 0.08 * np.sqrt(np.maximum(imbalance, 0.0))
        elif self.scenario == "optimum_hotspot":
            radius2 = (
                (z0 - 0.62) ** 2
                + (z1 - 0.66) ** 2
                + (z2 - 0.72) ** 2
            )
            A[0] = np.exp(-0.5 * radius2 / 0.14 ** 2)
            A[1] = 0.20 * np.sqrt(np.maximum(imbalance, 0.0))
            A[2] = 0.20 * dispersion
            N[0] = 0.08 * np.maximum(1.0 - control, 0.0)
        elif self.scenario == "safe_interior_hotspot":
            radius2 = (
                (z0 - 0.90) ** 2
                + (z1 - 0.86) ** 2
                + (z2 - 0.82) ** 2
            )
            A[0] = np.exp(-0.5 * radius2 / 0.13 ** 2)
            A[1] = 0.18 * np.sqrt(np.maximum(imbalance, 0.0))
            A[2] = 0.20 * dispersion
            N[0] = 0.06 * control
        elif self.scenario == "regime_step":
            A[0] = (
                (control >= 0.57) & (z1 <= 0.70)
            ).astype(float)
            A[1] = (z0 >= 0.60).astype(float)
            A[2] = 0.25 * dispersion
            N[0] = 0.10 * (z2 <= 0.55).astype(float)
            N[1] = 0.08 * np.sqrt(np.maximum(imbalance, 0.0))
        elif self.scenario == "sparse_axis":
            A[0] = np.clip((z0 - 0.20) / 0.80, 0.0, 1.0)
            A[1] = 0.04 * np.abs(z1 - 0.5)
            A[2] = 0.04 * dispersion
            N[0] = 0.03 * np.maximum(1.0 - control, 0.0)
        elif self.scenario == "shared_factor":
            A[0] = 0.12 * np.maximum(1.0 - control, 0.0)
            A[1] = 0.10 * np.sqrt(np.maximum(imbalance, 0.0))
            A[2] = 0.10 * dispersion
            N[0] = np.maximum(0.92 - control, 0.0)
            N[1] = 0.12 + np.abs(z0 - z1) + 0.35 * dispersion
        elif self.scenario == "hidden_periodic":
            A[0] = 0.15 * np.maximum(1.0 - control, 0.0)
            A[1] = 0.10 * np.sqrt(np.maximum(imbalance, 0.0))
            A[2] = 0.10 * dispersion
            N[0] = 0.05 * np.abs(z0 - z2)
            oscillation = (
                np.sin(8.0 * np.pi * z0)
                * np.sin(6.0 * np.pi * z1)
            )
            hidden = (0.45 * self.sigma_level) ** 2 * (
                0.15 + 0.85 * oscillation ** 2
            )
        return A, N, hidden

    def _reference_risk_exposure(self):
        return RiskExposure(
            np.zeros(3, dtype=float),
            np.zeros(2, dtype=float),
            local_names=("local_shape", "local_contrast", "local_dispersion"),
            shared_names=("common_level", "common_contrast"),
        )

    def risk_exposures(self, x, output_index=1):
        del output_index
        A, N, _ = self._risk_arrays(*self.policy_state(x))
        return RiskExposure(
            np.asarray(A, dtype=float).reshape(3),
            np.asarray(N, dtype=float).reshape(2),
            local_names=("local_shape", "local_contrast", "local_dispersion"),
            shared_names=("common_level", "common_contrast"),
            meta={
                "provider": type(self).__name__,
                "scenario": self.scenario,
                "provider_exact": bool(
                    CONTROLLED_HETERO_SCENARIOS[self.scenario][
                        "provider_exact"]),
            },
        )

    def cumulative_risk_parameters(self, output_index=1):
        if int(output_index) not in (1, 2):
            return None
        scale2 = float(self.sigma_level) ** 2
        if not self.heteroscedastic or self.scenario == "homoscedastic":
            return CumulativeRiskParameters(
                Lambda=np.zeros(3),
                B=np.zeros((2, 2)),
                omega=np.zeros(2),
                floor=0.04 * scale2,
            )
        if self.scenario == "shared_factor":
            return CumulativeRiskParameters(
                Lambda=scale2 * np.array([0.15, 0.12, 0.08]),
                B=scale2 * np.array([[0.62, 0.28], [0.28, 0.48]]),
                omega=scale2 * np.array([0.06, 0.05]),
                floor=0.04 * scale2,
            )
        return CumulativeRiskParameters(
            Lambda=scale2 * np.array([0.28, 0.16, 0.10]),
            B=scale2 * np.array([[0.20, 0.08], [0.08, 0.15]]),
            omega=scale2 * np.array([0.04, 0.03]),
            floor=0.04 * scale2,
        )

    def _true_constraint_variance_many(self, z0, z1, z2, dispersion):
        A, N, hidden = self._risk_arrays(z0, z1, z2, dispersion)
        params = self.cumulative_risk_parameters(output_index=1)
        independent = np.sum(
            np.asarray(params.Lambda)[:, None] * A.reshape(3, -1) ** 2,
            axis=0,
        )
        shared = np.einsum(
            "in,ij,jn->n",
            N.reshape(2, -1),
            np.asarray(params.B),
            N.reshape(2, -1),
        )
        linear = np.asarray(params.omega) @ N.reshape(2, -1)
        total = (
            float(params.floor) + independent + shared + linear
            + np.asarray(hidden, dtype=float).reshape(-1)
        )
        return np.maximum(total, 1e-12).reshape(np.shape(np.asarray(z0)))

    def true_cumulative_risk_decomposition(self, x, output_index=1):
        params = self.cumulative_risk_parameters(output_index=output_index)
        if params is None:
            return None
        exposure = self.risk_exposures(x, output_index=output_index)
        out = decompose_cumulative_risk(exposure, params)
        _, _, hidden = self._risk_arrays(*self.policy_state(x))
        hidden_value = float(np.asarray(hidden))
        out["provider_total"] = float(out["total"])
        out["unmodeled_residual"] = hidden_value
        out["total"] = float(max(out["provider_total"] + hidden_value, 0.0))
        out["provider_exact"] = bool(
            CONTROLLED_HETERO_SCENARIOS[self.scenario]["provider_exact"])
        return out

    def true_sigma(self, x):
        decomposition = self.true_cumulative_risk_decomposition(
            x, output_index=1)
        constraint_sigma = float(np.sqrt(max(decomposition["total"], 1e-12)))
        objective_sigma = float(max(0.20 * self.sigma_level, 1e-8))
        return np.array(
            [objective_sigma, objective_sigma, constraint_sigma],
            dtype=float,
        )

    def simulate(self, x, rng=None):
        rng = rng or np.random.default_rng()
        means = np.asarray(self.true_objectives(x), dtype=float)
        return means + rng.normal(0.0, self.true_sigma(x), size=3)

    def is_truly_feasible(self, x):
        margin = (
            float(self.true_objectives(x)[2])
            + norm.ppf(1.0 - self.alpha) * float(self.true_sigma(x)[2])
            - self.tau
        )
        return bool(margin <= 0.0)

    def risk_class(self, x):
        z0, z1, z2, _ = self.policy_state(x)
        return (
            (0 if z0 < 0.5 else 1)
            + 2 * (0 if z1 < 0.5 else 1)
            + 4 * (0 if z2 < 0.5 else 1)
        )

    def hvd_features(self, x):
        z0, z1, z2, dispersion = self.policy_state(x)
        exposure = self.risk_exposures(x)
        return np.concatenate([
            np.array([
                1.0, z0, z1, z2, dispersion,
                z0 ** 2, z1 ** 2, z2 ** 2,
                z0 * z1, z0 * z2, z1 * z2,
            ], dtype=float),
            exposure.A,
            exposure.N,
        ])

    def gpr_basis_map(self, output_index=0):
        del output_index
        return ControlledLatentFeatureMap(self)

    def surrogate_basis_map(self):
        return self.gpr_basis_map()

    def structured_candidates(self, n=10, rng=None):
        del n, rng
        return []

    def initial_samples(self, n=5, rng=None):
        del n, rng
        return []

    def recommendation_refinement_candidates(self):
        return []

    def all_axis_solutions(self):
        return []

    def _constant_policy(self, z0, z1, z2):
        values = (z0, z1, z2)
        out = np.empty(self.d, dtype=int)
        for index, group in enumerate(np.array_split(np.arange(self.d), 3)):
            out[group] = int(np.clip(np.rint(values[index] * self.L), 0, self.L))
        return tuple(int(value) for value in out)

    def _oracle_grid(self, weights, center=None, radius=None, points=41):
        if center is None:
            axes = [np.linspace(0.0, 1.0, int(points))] * 3
        else:
            axes = [
                np.linspace(
                    max(0.0, float(center[index]) - float(radius)),
                    min(1.0, float(center[index]) + float(radius)),
                    int(points),
                )
                for index in range(3)
            ]
        z0, z1, z2 = np.meshgrid(*axes, indexing="ij")
        flat = tuple(value.reshape(-1) for value in (z0, z1, z2))
        dispersion = np.zeros_like(flat[0])
        f1, f2, constraint, _, _ = self._mean_surfaces(
            *flat, dispersion)
        objective = float(weights[0]) * f1 + float(weights[1]) * f2
        variance = self._true_constraint_variance_many(
            *flat, dispersion)
        margin = (
            constraint
            + norm.ppf(1.0 - self.alpha) * np.sqrt(variance)
            - self.tau
        )
        feasible = margin <= 0.0
        if not np.any(feasible):
            return None
        ranked = np.where(feasible, objective, np.inf)
        best = int(np.argmin(ranked))
        return (
            np.array([flat[0][best], flat[1][best], flat[2][best]]),
            float(objective[best]),
        )

    def scalarized_true_best_feasible(self, weights):
        normalized = np.asarray(weights, dtype=float)
        normalized = normalized / max(float(np.sum(normalized)), 1e-12)
        key = tuple(np.round(normalized, 12))
        if key in self._oracle_cache:
            return self._oracle_cache[key]
        coarse = self._oracle_grid(normalized, points=41)
        if coarse is None:
            result = (None, float("inf"))
            self._oracle_cache[key] = result
            return result
        refined = self._oracle_grid(
            normalized,
            center=coarse[0],
            radius=0.04,
            points=41,
        )
        latent = coarse[0] if refined is None else refined[0]
        base = np.rint(latent * self.L).astype(int)
        candidates = []
        for delta0 in (-1, 0, 1):
            for delta1 in (-1, 0, 1):
                for delta2 in (-1, 0, 1):
                    values = np.clip(
                        base + np.array([delta0, delta1, delta2]),
                        0,
                        self.L,
                    ) / float(self.L)
                    candidates.append(self._constant_policy(*values))
        feasible = [x for x in candidates if self.is_truly_feasible(x)]
        if not feasible:
            result = (None, float("inf"))
        else:
            objectives = [
                float(
                    normalized[0] * self.true_objectives(x)[0]
                    + normalized[1] * self.true_objectives(x)[1]
                )
                for x in feasible
            ]
            index = int(np.argmin(objectives))
            result = (feasible[index], float(objectives[index]))
        self._oracle_cache[key] = result
        return result

    def hvd_residual_variance_cap(self, output_index=0):
        if int(output_index) in (1, 2):
            return float(4.0 * self.sigma_level ** 2)
        return float(self.sigma_level ** 2)

    def cumulative_risk_provider_status(self):
        spec = CONTROLLED_HETERO_SCENARIOS[self.scenario]
        return {
            "status": "available",
            "provider": type(self).__name__,
            "coordinate": "psi=(A,N)",
            "scenario": self.scenario,
            "variance_location": spec["location"],
            "variance_geometry": spec["geometry"],
            "provider_exact": bool(spec["provider_exact"]),
        }

    def oracle_contract(self):
        return {
            "available": True,
            "used_for_decision": False,
            "used_post_run_for_regret": True,
            "latent_dimension": 3,
            "raw_dimension": int(self.d),
            "coarse_grid_points_per_axis": 41,
            "local_refinement_points_per_axis": 41,
        }


class ControlledHeteroHomoscedastic(ControlledHeteroscedasticProblem):
    scenario = "homoscedastic"


class ControlledHeteroSmoothBoundary(ControlledHeteroscedasticProblem):
    scenario = "smooth_boundary"


class ControlledHeteroOptimumHotspot(ControlledHeteroscedasticProblem):
    scenario = "optimum_hotspot"


class ControlledHeteroSafeInterior(ControlledHeteroscedasticProblem):
    scenario = "safe_interior_hotspot"


class ControlledHeteroRegimeStep(ControlledHeteroscedasticProblem):
    scenario = "regime_step"


class ControlledHeteroSparseAxis(ControlledHeteroscedasticProblem):
    scenario = "sparse_axis"


class ControlledHeteroSharedFactor(ControlledHeteroscedasticProblem):
    scenario = "shared_factor"


class ControlledHeteroHiddenPeriodic(ControlledHeteroscedasticProblem):
    scenario = "hidden_periodic"


CONTROLLED_PROBLEM_REGISTRY = {
    "ControlledHeteroHomoscedastic": ControlledHeteroHomoscedastic,
    "ControlledHeteroSmoothBoundary": ControlledHeteroSmoothBoundary,
    "ControlledHeteroOptimumHotspot": ControlledHeteroOptimumHotspot,
    "ControlledHeteroSafeInterior": ControlledHeteroSafeInterior,
    "ControlledHeteroRegimeStep": ControlledHeteroRegimeStep,
    "ControlledHeteroSparseAxis": ControlledHeteroSparseAxis,
    "ControlledHeteroSharedFactor": ControlledHeteroSharedFactor,
    "ControlledHeteroHiddenPeriodic": ControlledHeteroHiddenPeriodic,
}
