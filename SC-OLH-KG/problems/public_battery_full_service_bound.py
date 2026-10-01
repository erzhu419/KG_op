"""Generous aggregate-stock bound under full bidirectional power saturation."""
import numpy as np


def pulse_area_minutes(age_minutes):
    """Integrated unit-height 1/30/1-minute pulse, including future zero area."""
    age = np.clip(np.asarray(age_minutes, float), 0., 32.)
    return np.where(age <= 1., age**2 / 2,
                    np.where(age <= 31., age - .5, 30.5 + (age - 31.) - (age - 31.)**2 / 2))


def full_service_bound(marks, initial_SOC_MWh, site_peak_MW, charge_efficiency,
                       discharge_efficiency, horizon_minutes, power_tolerance_MW,
                       stock_tolerance_MWh):
    """No stock-capacity clipping: a violated optimistic bound proves necessity.

    The caller establishes the saturated full-service hypothesis. Zero-net PN
    can only dissipate stock. Lossless circulation and unrestricted upper stock
    are deliberate relaxations; a nonnegative bound is inconclusive.
    """
    minutes = np.arange(horizon_minutes + 1)
    energies = {d: np.zeros(len(minutes)) for d in ("export", "import")}
    started = {d: np.zeros(len(minutes), int) for d in energies}
    completed = {d: np.zeros(len(minutes), int) for d in energies}
    for mark in marks:
        direction = mark["direction"]
        if direction is None:
            continue
        age = minutes - mark["minute"]
        energies[direction] += site_peak_MW * pulse_area_minutes(age) / 60
        started[direction] += age >= 0
        completed[direction] += age >= 32
    correction = power_tolerance_MW / discharge_efficiency * minutes / 60
    initial = float(np.sum(initial_SOC_MWh))
    upper = (initial + charge_efficiency * energies["import"]
             - energies["export"] / discharge_efficiency + correction)
    threshold = -len(initial_SOC_MWh) * stock_tolerance_MWh
    violations = np.flatnonzero(upper < threshold)
    first = None if len(violations) == 0 else int(violations[0])
    prefix = {"minute": minutes, "ideal_net_import_MWh": energies["import"],
              "ideal_net_export_MWh": energies["export"], "power_tolerance_correction_MWh": correction,
              "total_SOC_upper_bound_MWh": upper, "import_calls_started": started["import"],
              "export_calls_started": started["export"], "import_calls_completed": completed["import"],
              "export_calls_completed": completed["export"]}
    summary = {"conditional_full_service_infeasible": first is not None,
               "initial_total_SOC_MWh": initial, "physical_total_SOC_lower_bound_MWh": threshold,
               "first_inspected_violating_minute": first,
               "first_violation": None if first is None else {k: v[first].item() for k, v in prefix.items()},
               "minimum_upper_bound_MWh": float(upper.min()),
               "minimum_upper_bound_minute": int(minutes[np.argmin(upper)]),
               "final_upper_bound_MWh": float(upper[-1]), "prefix_rows": len(minutes),
               "import_calls": int(started["import"][-1]), "export_calls": int(started["export"][-1]),
               "no_activation_clocks": sum(m["direction"] is None for m in marks)}
    return summary, prefix
