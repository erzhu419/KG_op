"""Same storage asset and 2017 observable scales across historical reuse tasks."""
from __future__ import annotations

import numpy as np

from problems.energy_forecast_policy import OPSDForecastIndexedStorageProblem


def month_period(year, month):
    start = f"{year:04d}-{month:02d}-01T00"
    stop = f"{year + 1:04d}-01-01T00" if month == 12 else f"{year:04d}-{month + 1:02d}-01T00"
    return start, stop


class TemporalArchiveReuseStorageProblem(OPSDForecastIndexedStorageProblem):
    """Reuse the market's two 2017 source periods for twelve later decisions.

    The base preparation fits all observable scales in 2017. Restricting the
    simulation starts afterwards keeps physical capacity and the interpretation
    of a policy fixed across seasons. Audit is a postdecision alias for the
    finite verification population, not an additional selection data set.
    """

    def __init__(self, data_path, *, market, search_period, verification_period,
                 outcome_access=True):
        super().__init__(data_path, market=market, year=2018, d=1000, horizon=168,
                         required_splits=("search",), outcome_access=outcome_access,
                         initial_soc_fraction=0.5)
        self._periods = {"search": tuple(search_period),
                         "verification": tuple(verification_period),
                         "audit": tuple(verification_period)}
        self._starts = {name: self.series.valid_window_starts(self.horizon, *period)
                        for name, period in self._periods.items()}
        self.required_splits = ("search", "verification")
        missing = {name: len(self._starts[name]) for name in self.required_splits
                   if len(self._starts[name]) < 32}
        if missing:
            raise ValueError(f"temporal reuse periods lack complete 168-hour windows: {missing}")

    def information_contract(self):
        contract = super().information_contract()
        contract.update(
            normalization_fit_period=["2017-01-01T00", "2018-01-01T00"],
            audit_role="postdecision alias of finite verification population",
            physical_capacity_changes_across_months=False,
            source_matching_rule="same market, historical 2017 archive",
        )
        return contract


def sampled_window_starts(problem, split, seeds):
    """Draw the same one iid start per seed as simulate_from_split."""
    starts = problem.split_window_starts(split)
    return np.array([starts[int(np.random.default_rng(np.random.SeedSequence(seed)).integers(0, len(starts)))]
                     for seed in seeds], dtype=np.int64)
