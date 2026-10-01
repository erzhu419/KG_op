"""One-shot absolute dispatch under locked PN and already received BOAs.

Availability precedes request observation. A common proportional reservation
preserves the physical rectangle between the neutral peak and offered peak,
including no activation, independent unit activations and the mandatory tail.
This is a simulation admission rule, not an identified BM dispatch response.
"""
import numpy as np

from problems.public_battery_dispatch import positive_integral
from problems.public_battery_visibility import trajectory


def pending_pulse_power(locked_baseline_MW, peak_MW=None):
    """No instruction retains PN; even a zero-valued peak issues a BOA."""
    baseline = np.repeat(np.asarray(locked_baseline_MW, float), 30, axis=0)
    return reference_pulse_power(baseline, baseline, peak_MW)


def reference_pulse_power(reference_first_MW, reference_last_MW, peak_MW=None):
    """Replace minutes 0--32 explicitly, then resume the received reference."""
    first, last = np.asarray(reference_first_MW, float).copy(), np.asarray(reference_last_MW, float).copy()
    if peak_MW is not None:
        first[:32], last[:32] = peak_MW, peak_MW
        first[0], last[31] = reference_first_MW[0], reference_first_MW[32]
    return first, last


def _physical(metrics, tolerance):
    return (max(metrics["unit_bound_violation_MWh"]) <= tolerance
            and metrics["power_bound_violation_MW"] <= tolerance)


def declare_pending(initial_SOC_MWh, locked_baseline_MW, offered_capacity_MW,
                    direction, asset, tolerance=1e-8, *, reference_power=None):
    """Reserve capacity relative to PN, before observing a new absolute request.

    reference_power contains the 60-minute endpoint arrays built from the
    received receipt prefix. With no prior BOA, the locked PN is the reference.
    """
    initial = np.asarray(initial_SOC_MWh, float)
    locked = np.asarray(locked_baseline_MW, float)
    capacity = np.asarray(offered_capacity_MW, float)
    sign = 1 if direction == "export" else -1
    anchor = np.max(locked, axis=0) if sign == 1 else np.min(locked, axis=0)
    base_first, base_last = (pending_pulse_power(locked) if reference_power is None
                             else reference_pulse_power(*reference_power))
    baseline_physics = trajectory(base_first, base_last, initial, 60, asset)
    declaration = {"initial_SOC_MWh": initial.copy(), "locked_baseline_MW": locked.copy(),
                   "reference_first_MW": base_first, "reference_last_MW": base_last,
                   "direction": direction, "offered_capacity_MW": capacity.copy(),
                   "anchor_MW": anchor, "guaranteed_peak_MW": None,
                   "available_capacity_MW": np.zeros(2), "baseline_physics": baseline_physics,
                   "known_commitment_failure": not _physical(baseline_physics, tolerance),
                   "envelope_path_evaluations": 1}
    if declaration["known_commitment_failure"]:
        declaration["reason"] = "known_commitment_failure"
        return declaration
    anchor_first, anchor_last = reference_pulse_power(base_first, base_last, anchor)

    def sustainable(scale):
        peak = anchor + sign * scale * capacity
        first, last = reference_pulse_power(base_first, base_last, peak)
        metrics = trajectory(first, last, initial, 60, asset)
        declaration["envelope_path_evaluations"] += 1
        # Each unit may independently be idle or receive any peak in the range.
        worst_first = np.maximum.reduce([np.abs(base_first), np.abs(anchor_first), np.abs(first)])
        worst_last = np.maximum.reduce([np.abs(base_last), np.abs(anchor_last), np.abs(last)])
        gross = max(np.sum(worst_first, axis=1).max(), np.sum(worst_last, axis=1).max())
        return _physical(metrics, tolerance) and gross <= asset["power_MW"] + tolerance

    if not sustainable(0.):
        declaration["reason"] = "no_sustainable_neutral_pulse"
        return declaration
    scale = 1.
    if not sustainable(scale):
        low, high = 0., 1.
        # Stocks are monotone in peak; the gross-power constraint is convex.
        # Starting at the safe anchor, feasible scales form an interval.
        for _ in range(48):
            middle = (low + high) / 2
            if sustainable(middle):
                low = middle
            else:
                high = middle
        scale = low
    declaration["guaranteed_peak_MW"] = anchor + sign * scale * capacity
    declaration["available_capacity_MW"] = scale * capacity
    declaration["reason"] = "available"
    return declaration


def evaluate_pending(declaration, proposal_peak_MW, service_capacity_MW, asset, tolerance=1e-8):
    """Follow admitted absolute power; an unsafe known reference stays a failure."""
    reference = declaration["reference_first_MW"], declaration["reference_last_MW"]
    initial = declaration["initial_SOC_MWh"]
    guaranteed = declaration["guaranteed_peak_MW"]
    admitted = None
    if proposal_peak_MW is not None and guaranteed is not None:
        anchor = declaration["anchor_MW"]
        admitted = np.clip(np.asarray(proposal_peak_MW, float),
                           np.minimum(anchor, guaranteed), np.maximum(anchor, guaranteed))
    first, last = reference_pulse_power(*reference, admitted)
    requested_first, requested_last = reference_pulse_power(*reference, proposal_peak_MW)
    physics = trajectory(first, last, initial, 60, asset)
    exported = positive_integral(first, last, 1 / 60)
    imported = positive_integral(-first, -last, 1 / 60)
    # Instruction error is signed against the requested absolute path, not
    # against zero. An extra import can violate a request to operate at zero.
    difference_first = requested_first - first
    difference_last = requested_last - last
    unmet_increase = np.sum(positive_integral(difference_first, difference_last, 1 / 60), axis=0)
    unmet_decrease = np.sum(positive_integral(-difference_first, -difference_last, 1 / 60), axis=0)
    shortfall = np.maximum(np.asarray(service_capacity_MW) - declaration["available_capacity_MW"], 0.)
    physical_feasible = _physical(physics, tolerance)
    service_success = (physical_feasible and not declaration["known_commitment_failure"]
                       and np.max(shortfall) <= tolerance
                       and max(np.max(unmet_increase), np.max(unmet_decrease)) <= tolerance)
    delta = asset["charge_efficiency"] * imported - exported / asset["discharge_efficiency"]
    return {"request_observed": proposal_peak_MW is not None,
            "proposal_peak_MW": None if proposal_peak_MW is None else np.asarray(proposal_peak_MW, float),
            "admitted_peak_MW": admitted, "service_capacity_MW": np.asarray(service_capacity_MW),
            "availability_shortfall_MW": shortfall,
            "unmet_increase_MWh": unmet_increase, "unmet_decrease_MWh": unmet_decrease,
            "total_unmet_energy_MWh": float(np.sum(unmet_increase + unmet_decrease)),
            "gross_export_MWh": np.sum(exported, axis=0), "gross_import_MWh": np.sum(imported, axis=0),
            "physical_feasible": bool(physical_feasible), "service_success": bool(service_success),
            "known_commitment_failure": declaration["known_commitment_failure"],
            "physics": physics, "SOC_trace_MWh": np.vstack((initial, initial + np.cumsum(delta, axis=0))),
            "power_first_MW": first, "power_last_MW": last}
