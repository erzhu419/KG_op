"""Causal receipt-sign forecasts for delayed PN; hypothetical pulses stay in planning."""
import numpy as np

from problems.public_battery_dispatch import inventory_rate
from problems.public_battery_pending_availability import reference_pulse_power
from problems.public_battery_units import nominate_units, unit_planning_terms
from problems.public_battery_visibility import trajectory


def prioritize_inventory(preferred, projected, idle_hours, bounds, lower, upper, asset):
    """Reserve minimum power for the forecast band, then recover target preferences."""
    def command_for_stock(stock):
        difference = projected - stock
        return np.where(difference >= 0, difference * asset["discharge_efficiency"],
                        difference / asset["charge_efficiency"]) / idle_hours

    low = np.maximum(command_for_stock(upper), -bounds)
    high = np.minimum(command_for_stock(lower), bounds)
    minimum = np.clip(np.zeros(2), low, high)
    required = float(np.abs(minimum).sum())
    feasible = bool(np.all(low <= high) and required <= asset["power_MW"])
    candidate = preferred.copy()
    fraction = None
    if feasible:
        desired = np.clip(preferred, low, high)
        total = float(np.abs(desired).sum())
        fraction = 1. if total <= asset["power_MW"] else (asset["power_MW"] - required) / (total - required)
        candidate = minimum + fraction * (desired - minimum)
    info = {"forecast_band_feasible": feasible, "forecast_power_interval_MW": np.column_stack((low, high)).tolist(),
            "forecast_minimum_command_MW": minimum.tolist(), "forecast_required_gross_power_MW": required,
            "forecast_preference_fraction": fraction, "forecast_stock_without_nomination_MWh": projected.tolist(),
            "forecast_stock_after_candidate_MWh": (projected + inventory_rate(candidate, asset) * idle_hours).tolist(),
            "forecast_stock_band_MWh": np.column_stack((lower, upper)).tolist()}
    return candidate, info


def nominate_receipt_forecast(soc, current, following, target, accepted, minute, cue,
                             request_peaks_MW, asset, tolerance=1e-8, *, execution_prefix=None):
    """Use only the current receipt cue; predict at the next whole-hour clock."""
    terms = unit_planning_terms(accepted, asset)
    ordinary = nominate_units(soc[None], current[None], following[None], target[None],
                              np.array([0]), terms, asset)[0]
    info = {"receipt_cue": cue, "forecast_used": False, "forecast_call_minute": None,
            "ordinary_nomination_MW": ordinary.tolist(), "analytic_nomination_calls": 1,
            "reference_safety_path_evaluations": 0, "reference_safety_scale": 1.,
            "reference_execution_prefix_minutes": 0 if execution_prefix is None else len(execution_prefix["first_MW"]),
            "known_reference_failure": False, "nomination_reason": "ordinary"}
    if minute % 60 != 30 or cue["direction"] is None:
        return ordinary, info

    baseline = np.repeat(np.vstack((current, following, ordinary)), 30, axis=0)
    reference = tuple(np.where(accepted["active"][0], accepted[key][0], baseline)
                      for key in ("first_MW", "last_MW"))
    first, last = reference_pulse_power(reference[0][30:], reference[1][30:], request_peaks_MW[cue["direction"]])
    forecast = {key: value.copy() for key, value in accepted.items()}
    forecast["active"][0, 30:62] = True
    forecast["first_MW"][0, 30:62], forecast["last_MW"][0, 30:62] = first[:32], last[:32]
    forecast_terms = unit_planning_terms(forecast, asset)
    preferred = nominate_units(soc[None], current[None], following[None], target[None],
                               np.array([0]), forecast_terms, asset)[0]
    idle = forecast_terms["idle_hours"][0]
    projected = (soc + forecast_terms["forced"][0].sum(axis=0)
                 + inventory_rate(current, asset) * idle[0] + inventory_rate(following, asset) * idle[1])
    capacity = np.array(asset["unit_energy_MWh"])
    magnitude = np.abs(request_peaks_MW[cue["direction"]])
    lower, upper = np.zeros(2), capacity.copy()
    if cue["direction"] == "export":
        lower = magnitude * 31 / 60 / asset["discharge_efficiency"]
    else:
        upper -= magnitude * 31 / 60 * asset["charge_efficiency"]
    candidate, allocation = prioritize_inventory(preferred, projected, idle[2],
                                                 forecast_terms["future_power_bound_MW"][0],
                                                 lower, upper, asset)
    info.update(allocation)
    info.update(forecast_used=True, forecast_call_minute=minute + 30, analytic_nomination_calls=2,
                forecast_preferred_nomination_MW=preferred.tolist(), forecast_candidate_MW=candidate.tolist())

    def known_physics(command):
        baseline = np.repeat(np.vstack((current, following, command)), 30, axis=0)
        reference = tuple(np.where(accepted["active"][0], accepted[key][0], baseline)
                          for key in ("first_MW", "last_MW"))
        info["reference_safety_path_evaluations"] += 1
        if execution_prefix is None:
            return trajectory(*reference, soc, 90, asset)
        # The accepted pulse must use the declaration/execution accumulation,
        # not a new origin at an already rounded half-hour inventory.
        first = np.vstack((execution_prefix["first_MW"], reference[0]))
        last = np.vstack((execution_prefix["last_MW"], reference[1]))
        return trajectory(first, last, execution_prefix["initial_SOC_MWh"], len(first), asset)

    def safe(metrics):
        return (max(metrics["unit_bound_violation_MWh"]) <= tolerance
                and metrics["power_bound_violation_MW"] <= tolerance)

    ordinary_metrics = known_physics(ordinary)
    info["ordinary_reference_physics"] = ordinary_metrics
    if not safe(ordinary_metrics):
        info.update(known_reference_failure=True, reference_safety_scale=0.,
                    nomination_reason="retained_unsafe_ordinary_reference")
        return ordinary, info
    candidate_metrics = known_physics(candidate)
    info["candidate_reference_physics"] = candidate_metrics
    if safe(candidate_metrics):
        info.update(nomination_reason="receipt_forecast", nominated_reference_physics=candidate_metrics)
        return candidate, info
    low, high, last_safe = 0., 1., ordinary_metrics
    for _ in range(48):
        middle = (low + high) / 2
        metrics = known_physics(ordinary + middle * (candidate - ordinary))
        if safe(metrics):
            low, last_safe = middle, metrics
        else:
            high = middle
    info.update(reference_safety_scale=low, nomination_reason="safe_interpolation",
                nominated_reference_physics=last_safe)
    return ordinary + low * (candidate - ordinary), info
