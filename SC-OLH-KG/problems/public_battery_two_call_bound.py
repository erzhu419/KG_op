"""Necessary stock band for four active two-call direction branches.

The common first-hour PN cancels between first-sign outcomes. Second-pulse
references are deliberately allowed to know its sign and use all site power.
"""
import numpy as np

from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_visibility import trajectory


def first_call_separation(peak_MW, charge_efficiency, discharge_efficiency):
    """Lower bound for one-minute ramps and a 30-minute absolute hold."""
    kappa = min(charge_efficiency, 1 / discharge_efficiency)
    return peak_MW * ((charge_efficiency + 1 / discharge_efficiency) / 2
                      + 2 * kappa / 60)


def two_call_bound(asset, peak_MW, unit=1, stock_tolerance=1e-8,
                   power_tolerance=1e-8):
    """Two continuous-extrema kernels; negative zero-origin stock is a delta.

    The enlarged power limit includes the existing physical tolerance. Only
    the export lower edge and import upper edge are imposed, so every feasible
    second-call starting stock must lie in this optimistic band.
    """
    reference_limit = asset["power_MW"] + power_tolerance
    kernels = {}
    for direction, sign in (("export", 1), ("import", -1)):
        locked = np.zeros((2, 2))
        locked[:, unit] = -sign * reference_limit
        peak = np.zeros(2)
        peak[unit] = sign * peak_MW
        first, last = pending_pulse_power(locked, peak)
        metrics = trajectory(first, last, np.zeros(2), 32, asset)
        kernels[direction] = {
            "reference_MW": float(locked[0, unit]),
            "peak_MW": float(peak[unit]),
            "minimum_delta_MWh": metrics["minimum_SOC_MWh"][unit],
            "maximum_delta_MWh": metrics["maximum_SOC_MWh"][unit],
            "final_delta_MWh": metrics["final_SOC_MWh"][unit],
        }
    lower = -kernels["export"]["minimum_delta_MWh"]
    upper = asset["unit_energy_MWh"][unit] - kernels["import"]["maximum_delta_MWh"]
    separation = first_call_separation(peak_MW, asset["charge_efficiency"],
                                       asset["discharge_efficiency"])
    width = upper - lower + 2 * stock_tolerance
    gap = separation - width
    return {"unit": unit + 1, "first_stock_separation_lower_bound_MWh": separation,
            "second_stock_band_without_stock_tolerance_MWh": [lower, upper],
            "second_stock_band_MWh": [lower - stock_tolerance, upper + stock_tolerance],
            "second_stock_band_width_MWh": width, "separation_excess_MWh": gap,
            "uniform_four_sign_guarantee_impossible": bool(gap > 0),
            "stock_tolerance_MWh": stock_tolerance, "power_tolerance_MW": power_tolerance,
            "optimistic_reference_limit_MW": reference_limit,
            "optimistic_second_pulse_paths": 2, "response_kernels": kernels}
