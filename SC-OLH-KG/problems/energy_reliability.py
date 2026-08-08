"""Real-data storage reserve optimization on pinned OPSD time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.cumulative_risk import (
    CumulativeRiskFeatureProvider,
    RiskExposure,
    cumulative_feature_names,
    cumulative_feature_vector,
)
from data.opsd import OPSDMarketSeries, load_opsd_market
from representation.observable_exposure import grouped_policy_state_exposure


@dataclass(frozen=True)
class StoragePhysics:
    """Target-independent battery and operating-cost constants."""

    energy_capacity: float = 0.40
    power_capacity: float = 0.40
    efficiency: float = 0.92
    reserve_floor: float = 0.10
    reserve_span: float = 0.85
    reserve_cost: float = 0.018
    cycle_cost: float = 0.030
    unserved_cost: float = 8.0
    spill_cost: float = 0.08
    maximum_unserved_fraction: float = 0.11

    def __post_init__(self):
        if self.energy_capacity <= 0.0 or self.power_capacity <= 0.0:
            raise ValueError("storage capacities must be positive")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("storage efficiency must lie in (0, 1]")
        if not 0.0 <= self.reserve_floor < 1.0:
            raise ValueError("reserve floor must lie in [0, 1)")
        if not 0.0 < self.reserve_span <= 1.0 - self.reserve_floor:
            raise ValueError("reserve span exceeds storage bounds")
        if not 0.0 < self.maximum_unserved_fraction < 1.0:
            raise ValueError("reliability threshold must lie in (0, 1)")


class OPSDStorageReliabilityProblem(CumulativeRiskFeatureProvider):
    """High-dimensional reserve/SOC policy with empirical chance risk.

    Actual load and renewable realization errors are read only by
    :meth:`simulate` or an explicitly selected post-search split.  The
    observable state and cumulative-risk coordinates use policy, forecast,
    price, calendar, and declared storage physics only.
    """

    problem_name = "OPSDStorageReliability"
    simulation_noise_model = "iid_empirical_hourly_window"
    verification_distribution_scope = (
        "fixed_empirical_distribution_over_admissible_window_start_indices"
    )
    variance_features = (0, 1, 2, 3)
    recommended_partition_features = variance_features

    def __init__(
        self,
        data_path,
        *,
        market="DK_2",
        year=2018,
        d=1000,
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
        self.L = int(L)
        self.alpha = float(alpha)
        self.sigma_level = float(sigma)
        self.heteroscedastic = bool(heteroscedastic)
        self.physics = physics or StoragePhysics()
        self.tau = 0.0
        self.ref_point = None
        if self.d <= 1 or self.L <= 1:
            raise ValueError("energy problem requires d>1 and L>1")
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
            name: self.series.valid_window_starts(self.d, *period)
            for name, period in self._periods.items()
        }
        self.required_splits = tuple(dict.fromkeys(map(str, required_splits)))
        unknown_splits = set(self.required_splits) - set(self._periods)
        if unknown_splits:
            raise ValueError(
                f"unknown required OPSD splits: {sorted(unknown_splits)}")
        if not self.required_splits:
            raise ValueError("at least one OPSD split must be required")
        too_small = {
            name: len(starts) for name, starts in self._starts.items()
            if name in self.required_splits
            and len(starts) < int(minimum_windows)
        }
        if too_small:
            raise ValueError(
                "OPSD split has insufficient contiguous windows: "
                f"{too_small}; horizon={self.d}"
            )
        self._prepare_time_series()
        self._prepare_observable_profiles()

    def _prepare_time_series(self):
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

        # The pinned release contains a day-ahead load forecast but no aligned
        # day-ahead wind/solar forecast.  Adding a hand-built persistence error
        # would change the uncertainty model and made the fixed storage asset
        # vacuously unsafe.  The registered V1 domain therefore balances only
        # the observed load forecast error.
        self._net_error = (
            np.asarray(
                series.load_actual - series.load_forecast,
                dtype=float,
            ) / load_scale
            if self.outcome_access else None
        )

        price_scale = float(np.median(np.abs(series.price[search_mask])))
        price_scale = max(price_scale, 1.0)
        self._normalized_price = np.clip(
            np.asarray(series.price, dtype=float) / price_scale,
            -2.0,
            6.0,
        )
        forecast = np.asarray(series.load_forecast, dtype=float) / load_scale
        lower, upper = np.quantile(forecast[search_mask], [0.05, 0.95])
        width = max(float(upper - lower), 1e-8)
        self._forecast_level = np.clip((forecast - lower) / width, 0.0, 1.0)
        forecast_ramp = np.abs(np.diff(forecast, prepend=forecast[0]))
        ramp_scale = max(float(np.quantile(
            forecast_ramp[search_mask], 0.95)), 1e-8)
        self._forecast_ramp = np.clip(forecast_ramp / ramp_scale, 0.0, 3.0)

    def _prepare_observable_profiles(self):
        """Average declared exogenous forecasts over search-window positions."""

        starts = self._starts["search"]
        if len(starts) > 512:
            indices = np.linspace(0, len(starts) - 1, 512).round().astype(int)
            starts = starts[indices]
        relative = np.arange(self.d, dtype=np.int64)
        rows = starts[:, None] + relative[None, :]
        self._relative_forecast_level = np.mean(
            self._forecast_level[rows], axis=0)
        self._relative_forecast_ramp = np.mean(
            self._forecast_ramp[rows], axis=0)
        price = np.mean(self._normalized_price[rows], axis=0)
        lo, hi = np.quantile(price, [0.05, 0.95])
        self._relative_price = np.clip(
            (price - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)

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
        return tuple(np.clip(np.rint(values * self.L), 0, self.L).astype(int))

    def sample_random(self, rng=None):
        rng = rng or np.random.default_rng()
        return tuple(
            int(value) for value in rng.integers(0, self.L + 1, size=self.d)
        )

    def _target_soc(self, x):
        z = self.normalize(x)
        return (
            self.physics.reserve_floor + self.physics.reserve_span * z
        )

    def _evaluate_start(self, x, start):
        if self._net_error is None:
            raise RuntimeError(
                "OPSD outcome access is disabled for this problem instance")
        target = self._target_soc(x)
        physics = self.physics
        start = int(start)
        stop = start + self.d
        error = self._net_error[start:stop]
        price = self._normalized_price[start:stop]
        if len(error) != self.d:
            raise ValueError("incomplete OPSD evaluation window")

        capacity = float(physics.energy_capacity)
        power = float(physics.power_capacity)
        efficiency = float(physics.efficiency)
        soc = float(np.clip(target[0] * capacity, 0.0, capacity))
        grid_charge = 0.0
        throughput = 0.0
        unserved = 0.0
        spill = 0.0
        positive_error = 0.0

        for index in range(self.d):
            desired = float(target[index] * capacity)
            delta = desired - soc
            if delta > 0.0:
                charge = min(delta / efficiency, power)
                soc = min(capacity, soc + efficiency * charge)
                grid_charge += max(float(price[index]), 0.0) * charge
                throughput += charge
            elif delta < 0.0:
                release = min(-delta * efficiency, power, soc * efficiency)
                soc = max(0.0, soc - release / efficiency)
                throughput += release

            shock = float(error[index])
            if shock >= 0.0:
                positive_error += shock
                discharge = min(shock, power, soc * efficiency)
                soc = max(0.0, soc - discharge / efficiency)
                throughput += discharge
                unserved += shock - discharge
            else:
                surplus = -shock
                charge = min(
                    surplus,
                    power,
                    max(capacity - soc, 0.0) / efficiency,
                )
                soc = min(capacity, soc + efficiency * charge)
                throughput += charge
                spill += surplus - charge

        horizon = float(self.d)
        reserve = float(np.mean(target))
        objective = (
            physics.reserve_cost * reserve
            + grid_charge / horizon
            + physics.cycle_cost * throughput / horizon
            + physics.unserved_cost * unserved / horizon
            + physics.spill_cost * spill / horizon
        )
        denominator = max(positive_error, 1e-4 * horizon)
        unserved_fraction = float(unserved / denominator)
        constraint = (
            unserved_fraction - physics.maximum_unserved_fraction
        )
        return np.asarray([objective, constraint], dtype=float), {
            "unserved_fraction": unserved_fraction,
            "positive_error": float(positive_error),
            "unserved_energy": float(unserved),
            "spill_energy": float(spill),
            "throughput": float(throughput),
            "mean_target_soc": reserve,
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

    def split_population(self, x, split, *, maximum_windows=None):
        starts = self._starts[str(split)]
        if maximum_windows is not None and len(starts) > int(maximum_windows):
            selected = np.linspace(
                0, len(starts) - 1, int(maximum_windows)
            ).round().astype(int)
            starts = starts[selected]
        return np.vstack([self._evaluate_start(x, start)[0] for start in starts])

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
        # The real high-dimensional domain has no exhaustive optimizer oracle.
        return None, float("inf")

    def _reference_risk_exposure(self):
        return RiskExposure(
            np.zeros(4, dtype=float),
            np.zeros(3, dtype=float),
            local_names=(
                "low_reserve", "reserve_edge", "policy_ramp",
                "sustained_low_reserve",
            ),
            shared_names=(
                "forecast_peak", "forecast_ramp", "price_system",
            ),
        )

    def risk_exposures(self, x, output_index=1):
        del output_index
        target = self._target_soc(x)
        deficit = 1.0 - target
        ramp = np.abs(np.diff(target, prepend=target[0]))
        edge = np.maximum(0.20 - target, 0.0) + np.maximum(target - 0.95, 0.0)
        block = max(1, min(24, self.d))
        kernel = np.ones(block, dtype=float) / float(block)
        sustained = np.convolve(deficit, kernel, mode="same")
        local = np.asarray([
            np.sqrt(np.mean(deficit ** 2)),
            np.sqrt(np.mean(edge ** 2)),
            np.sqrt(np.mean(ramp ** 2)),
            np.sqrt(np.mean(sustained ** 2)),
        ])
        shared = np.asarray([
            np.mean(deficit * self._relative_forecast_level),
            np.mean(deficit * self._relative_forecast_ramp),
            np.mean(deficit * self._relative_price),
        ])
        return RiskExposure(
            local,
            shared,
            local_names=self._reference_risk_exposure().local_names,
            shared_names=self._reference_risk_exposure().shared_names,
            meta={
                "provider": self.problem_name,
                "market": self.market,
                "target_outcomes_used": False,
            },
        )

    def cumulative_risk_features(self, x, output_index=1):
        return cumulative_feature_vector(self.risk_exposures(x, output_index))

    def cumulative_risk_feature_names(self, output_index=1):
        del output_index
        return cumulative_feature_names(self._reference_risk_exposure())

    def cumulative_risk_parameters(self, output_index=1):
        # Real OPSD errors do not expose oracle variance parameters.
        del output_index
        return None

    def cumulative_risk_provider_status(self):
        return {
            "status": "available_empirical_no_oracle_parameters",
            "provider": type(self).__name__,
            "coordinate": "psi=(A,N)",
            "target_outcomes_used": False,
        }

    def observable_state_exposure(self, x):
        z = self.normalize(x)
        target = self._target_soc(x)
        groups = (
            target,
            1.0 - target,
            target * self._relative_forecast_level,
            (1.0 - target) * np.clip(self._relative_forecast_ramp, 0.0, 1.0),
        )
        return grouped_policy_state_exposure(
            z,
            groups,
            channel_names=(
                "target_soc", "reserve_deficit", "forecast_peak_reserve",
                "forecast_ramp_deficit",
            ),
            provider="opsd_storage_forecast_schema",
        )

    def hvd_features(self, x):
        exposure = self.risk_exposures(x)
        return np.concatenate([
            np.asarray([1.0]),
            exposure.A,
            exposure.N,
            exposure.A ** 2,
            exposure.N ** 2,
        ])

    def risk_class(self, x):
        target = self._target_soc(x)
        reserve_bin = int(np.clip(np.floor(3.0 * np.mean(target)), 0, 2))
        ramp_bin = int(np.mean(np.abs(np.diff(target))) > 0.08)
        return 10 * reserve_bin + ramp_bin

    def initial_samples(self, n=5, rng=None):
        del n, rng
        return []

    def structured_candidates(self, n=10, rng=None):
        del n, rng
        return []

    def recommendation_refinement_candidates(self):
        return []

    def all_axis_solutions(self):
        return []
