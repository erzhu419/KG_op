"""Forecast-indexed storage policy optimization on pinned OPSD data.

The V2 energy experiment represented a policy by its position inside a random
historical window.  This module separates the decision discretization from the
simulation horizon.  A decision is instead an ordered response profile

    forecast stress in [0, 1] -> target state of charge in [0, 1].

The stress coordinate uses only day-ahead load forecasts and is frozen from
the search year.  Realized forecast errors are read only by the simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.cumulative_risk import (
    CumulativeRiskFeatureProvider,
    RiskExposure,
    cumulative_feature_names,
    cumulative_feature_vector,
)
from core.profile_atlas import regular_profile_nodes
from data.opsd import OPSDMarketSeries, load_opsd_market
from problems.energy_reliability import StoragePhysics


class OPSDForecastIndexedStorageProblem(CumulativeRiskFeatureProvider):
    """Storage reserve control using an observable forecast-stress policy.

    ``d`` controls only the resolution of the response profile.  ``horizon``
    controls the number of consecutive physical hours in one simulator call.
    Refining ``d`` therefore refines the same functional decision rather than
    lengthening the simulated operating period.
    """

    problem_name = "OPSDForecastIndexedStorage"
    simulation_noise_model = "iid_empirical_fixed_horizon_window_start"
    verification_distribution_scope = (
        "fixed_empirical_distribution_over_admissible_window_start_indices"
    )
    variance_features = (0, 1, 2, 3)
    recommended_partition_features = variance_features
    policy_semantics = "forecast_stress_to_target_state_of_charge"
    stress_coordinate_contract = "max_forecast_level_and_absolute_ramp_v1"

    def __init__(
        self,
        data_path,
        *,
        market="DK_2",
        year=2018,
        d=1000,
        horizon=168,
        L=100,
        alpha=0.05,
        sigma=0.05,
        heteroscedastic=True,
        physics=None,
        minimum_windows=32,
        required_splits=("search", "audit", "verification"),
        outcome_access=True,
    ):
        self.data_path = str(Path(data_path))
        self.outcome_access = bool(outcome_access)
        self.series: OPSDMarketSeries = load_opsd_market(
            self.data_path,
            market,
            include_outcomes=self.outcome_access,
        )
        self.market = str(market)
        self.year = int(year)
        self.d = int(d)
        self.horizon = int(horizon)
        self.verification_window_length = int(horizon)
        self.L = int(L)
        self.alpha = float(alpha)
        self.sigma_level = float(sigma)
        self.heteroscedastic = bool(heteroscedastic)
        self.physics = physics or StoragePhysics()
        self.tau = 0.0
        self.ref_point = None
        self.nodes = regular_profile_nodes(self.d)
        if self.d <= 1 or self.horizon <= 1 or self.L <= 1:
            raise ValueError("energy V3 requires d>1, horizon>1, and L>1")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

        self._periods = {
            "search": (
                f"{self.year - 1:04d}-01-01T00",
                f"{self.year:04d}-01-01T00",
            ),
            "audit": (
                f"{self.year:04d}-01-01T00",
                f"{self.year + 1:04d}-01-01T00",
            ),
            "verification": (
                f"{self.year + 1:04d}-01-01T00",
                f"{self.year + 2:04d}-01-01T00",
            ),
        }
        self._starts = {
            name: self.series.valid_window_starts(self.horizon, *period)
            for name, period in self._periods.items()
        }
        self.required_splits = tuple(dict.fromkeys(map(str, required_splits)))
        unknown = set(self.required_splits) - set(self._periods)
        if unknown:
            raise ValueError(f"unknown required OPSD splits: {sorted(unknown)}")
        if not self.required_splits:
            raise ValueError("at least one OPSD split must be required")
        too_small = {
            name: len(starts)
            for name, starts in self._starts.items()
            if name in self.required_splits
            and len(starts) < int(minimum_windows)
        }
        if too_small:
            raise ValueError(
                "OPSD split has insufficient contiguous windows: "
                f"{too_small}; horizon={self.horizon}"
            )
        self._prepare_observable_state()

    def _prepare_observable_state(self):
        series = self.series
        search_start = np.datetime64(
            self._periods["search"][0], "h").astype(np.int64)
        search_stop = np.datetime64(
            self._periods["search"][1], "h").astype(np.int64)
        search_mask = (
            (series.timestamp_hour >= search_start)
            & (series.timestamp_hour < search_stop)
        )
        if not np.any(search_mask):
            raise ValueError(
                f"OPSD archive has no {self.market}/{self.year} search rows")

        load_scale = float(np.median(series.load_forecast[search_mask]))
        if not np.isfinite(load_scale) or load_scale <= 0.0:
            raise ValueError("OPSD load scale is not positive")
        self._load_scale = load_scale
        self._net_error = (
            np.asarray(
                series.load_actual - series.load_forecast,
                dtype=float,
            ) / load_scale
            if self.outcome_access else None
        )

        price_scale = max(
            float(np.median(np.abs(series.price[search_mask]))), 1.0)
        self._normalized_price = np.clip(
            np.asarray(series.price, dtype=float) / price_scale,
            -2.0,
            6.0,
        )
        forecast = np.asarray(series.load_forecast, dtype=float) / load_scale
        lower, upper = np.quantile(forecast[search_mask], [0.05, 0.95])
        width = max(float(upper - lower), 1e-8)
        self._forecast_level = np.clip(
            (forecast - lower) / width, 0.0, 1.0)
        ramp = np.abs(np.diff(forecast, prepend=forecast[0]))
        ramp_scale = max(float(np.quantile(ramp[search_mask], 0.95)), 1e-8)
        self._forecast_ramp = np.clip(ramp / ramp_scale, 0.0, 1.0)
        self._forecast_stress = np.maximum(
            self._forecast_level, self._forecast_ramp)
        self._search_mask = search_mask

    def information_contract(self):
        return {
            "dataset": "OPSD time_series 2020-10-06",
            "market": self.market,
            "year": self.year,
            "search_period": list(self._periods["search"]),
            "audit_period": list(self._periods["audit"]),
            "verification_period": list(self._periods["verification"]),
            "normalization_fit_period": "search_only",
            "required_splits": list(self.required_splits),
            "policy_semantics": self.policy_semantics,
            "stress_coordinate_contract": self.stress_coordinate_contract,
            "stress_definition": "max(level_05_95, absolute_ramp_q95)",
            "stress_weights_tuned_from_target_outcomes": False,
            "policy_grid_dimension": int(self.d),
            "simulation_horizon_hours": int(self.horizon),
            "decision_dimension_changes_simulation_horizon": False,
            "actual_target_error_used_by_observable_coordinate": False,
            "actual_target_error_used_by_simulator": True,
            "outcome_access_enabled": self.outcome_access,
            "split_window_counts": {
                name: int(len(starts)) for name, starts in self._starts.items()
            },
            "window_sampling": "iid_start_indices_with_replacement",
            "underlying_windows_may_overlap": True,
            "certificate_scope": self.verification_distribution_scope,
            "future_time_series_iid_claimed": False,
        }

    def int_bounds(self):
        return np.zeros(self.d, dtype=int), np.full(self.d, self.L, dtype=int)

    def normalize(self, x):
        values = np.asarray(x, dtype=float).reshape(-1)
        if len(values) != self.d:
            raise ValueError(f"expected {self.d} policy entries")
        return np.clip(values / float(self.L), 0.0, 1.0)

    def continuous_to_int(self, x_norm):
        values = np.asarray(x_norm, dtype=float).reshape(-1)
        if len(values) != self.d:
            raise ValueError(f"expected {self.d} normalized entries")
        return tuple(np.clip(
            np.rint(values * self.L), 0, self.L,
        ).astype(int))

    def encode_semantic_profile(self, profile):
        return self.continuous_to_int(profile)

    def sample_random(self, rng=None):
        rng = rng or np.random.default_rng()
        return tuple(
            int(value) for value in rng.integers(0, self.L + 1, size=self.d)
        )

    def _policy_target_soc(self, x, stress):
        profile = self.normalize(x)
        response = np.interp(
            np.asarray(stress, dtype=float),
            self.nodes,
            profile,
            left=float(profile[0]),
            right=float(profile[-1]),
        )
        return (
            self.physics.reserve_floor
            + self.physics.reserve_span * np.clip(response, 0.0, 1.0)
        )

    def _evaluate_start_batch(self, x, starts, *, return_diagnostics=False):
        if self._net_error is None:
            raise RuntimeError(
                "OPSD outcome access is disabled for this problem instance")
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        if len(starts) == 0:
            empty = np.zeros((0, 2), dtype=float)
            return (empty, {}) if return_diagnostics else empty
        if np.any(
            (starts < 0) | (starts + self.horizon > len(self._net_error))
        ):
            raise ValueError("incomplete OPSD evaluation window")

        physics = self.physics
        capacity = float(physics.energy_capacity)
        power = float(physics.power_capacity)
        efficiency = float(physics.efficiency)
        count = len(starts)
        first_stress = self._forecast_stress[starts]
        first_target = self._policy_target_soc(x, first_stress)
        soc = np.clip(first_target * capacity, 0.0, capacity)
        grid_charge = np.zeros(count, dtype=float)
        throughput = np.zeros(count, dtype=float)
        unserved = np.zeros(count, dtype=float)
        spill = np.zeros(count, dtype=float)
        positive_error = np.zeros(count, dtype=float)
        target_sum = np.zeros(count, dtype=float)

        for offset in range(self.horizon):
            rows = starts + offset
            target = self._policy_target_soc(x, self._forecast_stress[rows])
            target_sum += target
            desired = target * capacity
            delta = desired - soc
            charging = delta > 0.0
            if np.any(charging):
                charge = np.minimum(delta[charging] / efficiency, power)
                soc[charging] = np.minimum(
                    capacity, soc[charging] + efficiency * charge)
                prices = self._normalized_price[rows[charging]]
                grid_charge[charging] += np.maximum(prices, 0.0) * charge
                throughput[charging] += charge
            releasing = delta < 0.0
            if np.any(releasing):
                release = np.minimum.reduce([
                    -delta[releasing] * efficiency,
                    np.full(np.sum(releasing), power, dtype=float),
                    soc[releasing] * efficiency,
                ])
                soc[releasing] = np.maximum(
                    0.0, soc[releasing] - release / efficiency)
                throughput[releasing] += release

            shock = self._net_error[rows]
            shortage = shock >= 0.0
            if np.any(shortage):
                positive = shock[shortage]
                positive_error[shortage] += positive
                discharge = np.minimum.reduce([
                    positive,
                    np.full(np.sum(shortage), power, dtype=float),
                    soc[shortage] * efficiency,
                ])
                soc[shortage] = np.maximum(
                    0.0, soc[shortage] - discharge / efficiency)
                throughput[shortage] += discharge
                unserved[shortage] += positive - discharge
            surplus_mask = ~shortage
            if np.any(surplus_mask):
                surplus = -shock[surplus_mask]
                charge = np.minimum.reduce([
                    surplus,
                    np.full(np.sum(surplus_mask), power, dtype=float),
                    np.maximum(capacity - soc[surplus_mask], 0.0) / efficiency,
                ])
                soc[surplus_mask] = np.minimum(
                    capacity, soc[surplus_mask] + efficiency * charge)
                throughput[surplus_mask] += charge
                spill[surplus_mask] += surplus - charge

        horizon = float(self.horizon)
        reserve = target_sum / horizon
        objective = (
            physics.reserve_cost * reserve
            + grid_charge / horizon
            + physics.cycle_cost * throughput / horizon
            + physics.unserved_cost * unserved / horizon
            + physics.spill_cost * spill / horizon
        )
        denominator = np.maximum(positive_error, 1e-4 * horizon)
        unserved_fraction = unserved / denominator
        constraint = (
            unserved_fraction - physics.maximum_unserved_fraction)
        values = np.column_stack([objective, constraint])
        if not return_diagnostics:
            return values
        return values, {
            "unserved_fraction": unserved_fraction,
            "positive_error": positive_error,
            "unserved_energy": unserved,
            "spill_energy": spill,
            "throughput": throughput,
            "mean_target_soc": reserve,
        }

    def _evaluate_start(self, x, start):
        values, diagnostics = self._evaluate_start_batch(
            x, [int(start)], return_diagnostics=True)
        return values[0], {
            key: float(np.asarray(value).reshape(-1)[0])
            for key, value in diagnostics.items()
        }

    def simulate_from_split(self, x, split, rng=None, *, return_diagnostics=False):
        split = str(split)
        if split not in self._starts:
            raise ValueError(f"unknown OPSD split {split!r}")
        rng = rng or np.random.default_rng()
        starts = self._starts[split]
        start = int(starts[int(rng.integers(0, len(starts)))])
        output, diagnostics = self._evaluate_start(x, start)
        if return_diagnostics:
            return output, {**diagnostics, "split": split, "start_index": start}
        return output

    def simulate(self, x, rng=None):
        return self.simulate_from_split(x, "search", rng)

    def split_window_starts(self, split):
        split = str(split)
        if split not in self._starts:
            raise ValueError(f"unknown OPSD split {split!r}")
        return np.asarray(self._starts[split], dtype=np.int64).copy()

    def evaluate_window_starts(self, x, starts, *, batch_size=512):
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(starts) == 0:
            return np.zeros((0, 2), dtype=float)
        return np.vstack([
            self._evaluate_start_batch(x, starts[begin:begin + batch_size])
            for begin in range(0, len(starts), batch_size)
        ])

    def split_population(self, x, split, *, maximum_windows=None, batch_size=512):
        starts = self.split_window_starts(split)
        if maximum_windows is not None and len(starts) > int(maximum_windows):
            indices = np.linspace(
                0, len(starts) - 1, int(maximum_windows)
            ).round().astype(int)
            starts = starts[indices]
        return self.evaluate_window_starts(x, starts, batch_size=batch_size)

    def true_outputs(self, x):
        return np.mean(self.split_population(x, "audit"), axis=0)

    def true_objective(self, x):
        return float(self.true_outputs(x)[0])

    def true_constraint_mean(self, x):
        return float(self.true_outputs(x)[1])

    def true_sigma(self, x):
        values = self.split_population(x, "audit")
        return np.std(values, axis=0, ddof=1)

    def is_truly_feasible(self, x):
        values = self.split_population(x, "audit")[:, 1]
        return bool(np.mean(values <= self.tau) >= 1.0 - self.alpha)

    def true_best_feasible(self):
        return None, float("inf")

    def _reference_risk_exposure(self):
        return RiskExposure(
            np.zeros(4, dtype=float),
            np.zeros(3, dtype=float),
            local_names=(
                "low_reserve", "reserve_edge", "policy_slope",
                "policy_curvature",
            ),
            shared_names=(
                "forecast_level_exposure", "forecast_ramp_exposure",
                "price_exposure",
            ),
        )

    def risk_exposures(self, x, output_index=1):
        del output_index
        target = self._policy_target_soc(x, self._forecast_stress)
        search_target = target[self._search_mask]
        deficit = 1.0 - search_target
        profile = self._policy_target_soc(x, self.nodes)
        slope = np.diff(profile, prepend=profile[0])
        curvature = np.diff(slope, prepend=slope[0])
        edge = (
            np.maximum(0.20 - search_target, 0.0)
            + np.maximum(search_target - 0.95, 0.0)
        )
        local = np.asarray([
            np.sqrt(np.mean(deficit ** 2)),
            np.sqrt(np.mean(edge ** 2)),
            np.sqrt(np.mean(slope ** 2)),
            np.sqrt(np.mean(curvature ** 2)),
        ])
        shared = np.asarray([
            np.mean(deficit * self._forecast_level[self._search_mask]),
            np.mean(deficit * self._forecast_ramp[self._search_mask]),
            np.mean(deficit * np.maximum(
                self._normalized_price[self._search_mask], 0.0)),
        ])
        reference = self._reference_risk_exposure()
        return RiskExposure(
            local,
            shared,
            local_names=reference.local_names,
            shared_names=reference.shared_names,
            meta={
                "provider": self.problem_name,
                "market": self.market,
                "policy_semantics": self.policy_semantics,
                "target_outcomes_used": False,
            },
        )

    def cumulative_risk_features(self, x, output_index=1):
        return cumulative_feature_vector(self.risk_exposures(x, output_index))

    def cumulative_risk_feature_names(self, output_index=1):
        del output_index
        return cumulative_feature_names(self._reference_risk_exposure())

    def cumulative_risk_parameters(self, output_index=1):
        del output_index
        return None

    def cumulative_risk_provider_status(self):
        return {
            "status": "available_empirical_no_oracle_parameters",
            "provider": type(self).__name__,
            "coordinate": "psi=(A,N)",
            "policy_semantics": self.policy_semantics,
            "target_outcomes_used": False,
        }
