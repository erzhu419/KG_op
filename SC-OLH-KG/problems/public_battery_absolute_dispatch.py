"""Absolute common-direction pulse admission, with actual PN kept as reference.

The supported request activates both units at a common proportional amplitude.
No activation is a separate reference path, not an independent-unit rectangle.
"""
import numpy as np

from problems.public_battery_pending_availability import reference_pulse_power
from problems.public_battery_visibility import trajectory


def declare_absolute(initial_SOC_MWh, locked_baseline_MW, requested_peak_MW,
                     asset, tolerance=1e-8, *, reference_power=None):
    """Declare the safe zero-to-request interval before observing the mark.

Both the unmodified reference and the zero-peak revision must be safe for the
whole known hour. Stocks are monotone in common signed peak; gross power is
convex, so checking the interval endpoints covers intermediate amplitudes.
"""
    initial = np.asarray(initial_SOC_MWh, float)
    locked = np.asarray(locked_baseline_MW, float)
    peak = np.asarray(requested_peak_MW, float)
    baseline = np.repeat(locked, 30, axis=0)
    first, last = (reference_pulse_power(baseline, baseline) if reference_power is None
                   else reference_pulse_power(*reference_power))

    def physical(metrics):
        return (max(metrics["unit_bound_violation_MWh"]) <= tolerance
                and metrics["power_bound_violation_MW"] <= tolerance)

    baseline_physics = trajectory(first, last, initial, 60, asset)
    declaration = {"initial_SOC_MWh": initial.copy(), "locked_baseline_MW": locked.copy(),
                   "reference_first_MW": first, "reference_last_MW": last,
                   "offered_capacity_MW": np.abs(peak), "anchor_MW": np.zeros(2),
                   "guaranteed_peak_MW": None, "available_capacity_MW": np.zeros(2),
                   "baseline_physics": baseline_physics,
                   "known_commitment_failure": not physical(baseline_physics),
                   "envelope_path_evaluations": 1}
    if declaration["known_commitment_failure"]:
        declaration["reason"] = "known_commitment_failure"
        return declaration

    def sustainable(scale):
        pulse_first, pulse_last = reference_pulse_power(first, last, scale * peak)
        metrics = trajectory(pulse_first, pulse_last, initial, 60, asset)
        declaration["envelope_path_evaluations"] += 1
        return physical(metrics)

    if not sustainable(0.):
        declaration["reason"] = "no_sustainable_neutral_pulse"
        return declaration
    scale = 1.
    if not sustainable(scale):
        low, high = 0., 1.
        for _ in range(48):
            middle = (low + high) / 2
            if sustainable(middle):
                low = middle
            else:
                high = middle
        scale = low
    declaration["guaranteed_peak_MW"] = scale * peak
    declaration["available_capacity_MW"] = scale * np.abs(peak)
    declaration["reason"] = "available"
    return declaration
