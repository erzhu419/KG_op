"""Frozen empirical reserves from the already executed 90-minute BOA history."""
import numpy as np

from problems.public_battery_dispatch import positive_integral
from problems.public_battery_units import unit_planning_terms


def reserve_planning_terms(plans, history, asset, lookback_minutes=90, tolerance=1e-8):
    """History begins one lookback before the first nomination clock time."""
    terms = unit_planning_terms(plans, asset)
    first = np.where(history["active"], history["first_MW"], 0)
    last = np.where(history["active"], history["last_MW"], 0)
    exported = positive_integral(first, last, 1 / 60)
    imported = positive_integral(-first, -last, 1 / 60)
    ends = np.arange(len(plans["active"])) * 30 + lookback_minutes
    begins = ends - lookback_minutes

    def totals(values):
        cumulative = np.vstack((np.zeros(2), np.cumsum(values, axis=0)))
        return cumulative[ends] - cumulative[begins]

    lower = totals(exported) / asset["discharge_efficiency"]
    upper = np.asarray(asset["unit_energy_MWh"]) - asset["charge_efficiency"] * totals(imported)
    peak = np.maximum(np.abs(first), np.abs(last))
    recent_peak = np.array([peak[a:b].max(axis=0) for a, b in zip(begins, ends)])
    observed_bound = np.maximum(asset["power_MW"] - recent_peak[:, ::-1], 0)
    terms.update({"target_lower_MWh": lower, "target_upper_MWh": upper,
                  "past_BOA_peak_MW": recent_peak,
                  "observed_power_bound_MW": observed_bound,
                  "reserve_conflict": np.any(lower > upper + tolerance, axis=-1),
                  "future_power_bound_MW": np.minimum(terms["future_power_bound_MW"], observed_bound)})
    return terms


def inventory_reserve_planning_terms(plans, history, asset, lookback_minutes=90, tolerance=1e-8):
    terms = reserve_planning_terms(plans, history, asset, lookback_minutes, tolerance)
    terms["future_power_bound_MW"] = unit_planning_terms(plans, asset)["future_power_bound_MW"]
    return terms


def power_reserve_planning_terms(plans, history, asset, lookback_minutes=90, tolerance=1e-8):
    terms = reserve_planning_terms(plans, history, asset, lookback_minutes, tolerance)
    original = unit_planning_terms(plans, asset)
    for field in ("target_lower_MWh", "target_upper_MWh"):
        terms[field] = original[field]
    terms["reserve_conflict"] = np.zeros(len(plans["active"]), bool)
    return terms
