"""Counterfactual inventory control with delayed nominations and received BOAs.

The workload is exogenous. Prices enter metered-energy accounting, while the
nomination rule sees SOC, pending commitments and an available forecast signal.
"""
from __future__ import annotations

import numpy as np

from performance.inspect_public_battery_dispatch import instant, reconstruct


def positive_integral(first, last, hours):
    """Integral of the positive part of a linear power segment."""
    first, last = np.asarray(first), np.asarray(last)
    a, b = np.maximum(first, 0), np.maximum(last, 0)
    crossing = first * last < 0
    triangle = np.divide(a * a + b * b, 2 * np.abs(last - first),
                         out=np.zeros_like(a, dtype=float), where=crossing)
    return np.where(crossing, triangle, (a + b) / 2) * hours


def minute_workload(segments, start, stop, power_MW):
    """Convert the exact causal ledger to whole-minute, chronological energy legs."""
    ledger = reconstruct(segments, start, stop)
    count = int((stop - start).total_seconds() / 60)
    active = np.zeros(count, dtype=bool)
    first, last = np.zeros(count), np.zeros(count)
    for row in ledger:
        if not row["active_BOA"]:
            continue
        left = int((instant(row["from_utc"]) - start).total_seconds() / 60)
        right = int((instant(row["to_utc"]) - start).total_seconds() / 60)
        weights = np.arange(right - left + 1) / (right - left)
        levels = row["from_MW"] + weights * (row["to_MW"] - row["from_MW"])
        active[left:right] = True
        first[left:right], last[left:right] = levels[:-1], levels[1:]
    export = positive_integral(first, last, 1 / 60)
    imported = positive_integral(-first, -last, 1 / 60)
    bounded_export = export - positive_integral(first - power_MW, last - power_MW, 1 / 60)
    bounded_import = imported - positive_integral(-first - power_MW, -last - power_MW, 1 / 60)
    import_first = (first < 0) | ((first == 0) & (last < 0))
    result = {"active": active, "peak_MW": np.maximum(np.abs(first), np.abs(last))}
    for label, outgoing, incoming in (("requested", export, imported),
                                      ("bounded", bounded_export, bounded_import)):
        result[label + "_out_1"] = np.where(import_first, 0, outgoing)
        result[label + "_in_1"] = np.where(import_first, incoming, 0)
        result[label + "_out_2"] = np.where(import_first, outgoing, 0)
        result[label + "_in_2"] = np.where(import_first, 0, incoming)
    return result


def inventory_rate(power, asset):
    """Positive power exports electricity; negative power imports it."""
    return np.where(power >= 0, -power / asset["discharge_efficiency"],
                    -power * asset["charge_efficiency"])


def nominate(soc, current, following, target, asset):
    # Both pending half-hours are already committed, before the new delivery.
    projected = soc + .5 * (inventory_rate(current, asset) + inventory_rate(following, asset))
    difference = projected - target
    command = np.where(difference >= 0, difference * asset["discharge_efficiency"] / .5,
                       difference / (asset["charge_efficiency"] * .5))
    return np.clip(command, -asset["power_MW"], asset["power_MW"])


def simulate(workload, prices, targets, starts, horizon_minutes, asset, initial_fraction=.5):
    """Batch policies/windows; targets and prices are indexed by half-hour.

    ``targets`` has one row per fixed policy. Starts index the minute workload.
    The first two baseline half-hours are common zero nominations. Every later
    command is fixed an hour before delivery, and BOAs take dispatch priority.
    """
    starts = np.asarray(starts, dtype=int)
    shape = (len(targets), len(starts))
    initial = asset["energy_MWh"] * initial_fraction
    soc = np.full(shape, initial)
    current, following, future = (np.zeros(shape) for _ in range(3))
    cash, outgoing, incoming, boa_missing, baseline_missing = (np.zeros(shape) for _ in range(5))
    minimum, maximum = soc.copy(), soc.copy()
    trace = []
    for minute in range(horizon_minutes):
        indices = starts + minute
        if minute % 30 == 0:
            if minute:
                current, following = following, future
            if minute + 60 < horizon_minutes:
                target = targets[:, (indices + 60) // 30] * asset["energy_MWh"]
                future = nominate(soc, current, following, target, asset)
                trace.append({"decision_minute": minute, "delivery_minute": minute + 60,
                              "first_policy_window_SOC_MWh": float(soc[0, 0]),
                              "first_policy_window_nomination_MW": float(future[0, 0])})
        active = workload["active"][indices]
        price = prices[indices // 30]
        baseline_out, baseline_in = np.maximum(current, 0) / 60, np.maximum(-current, 0) / 60
        for leg in (1, 2):
            requested_out = np.where(active, workload[f"requested_out_{leg}"][indices],
                                     baseline_out if leg == 1 else 0)
            requested_in = np.where(active, workload[f"requested_in_{leg}"][indices],
                                    baseline_in if leg == 1 else 0)
            bounded_out = np.where(active, workload[f"bounded_out_{leg}"][indices],
                                   baseline_out if leg == 1 else 0)
            bounded_in = np.where(active, workload[f"bounded_in_{leg}"][indices],
                                  baseline_in if leg == 1 else 0)
            delivered_out = np.minimum(bounded_out, np.maximum(soc, 0) * asset["discharge_efficiency"])
            soc -= delivered_out / asset["discharge_efficiency"]
            delivered_in = np.minimum(bounded_in, np.maximum(asset["energy_MWh"] - soc, 0)
                                      / asset["charge_efficiency"])
            soc += delivered_in * asset["charge_efficiency"]
            missing = requested_out + requested_in - delivered_out - delivered_in
            boa_missing += np.where(active, missing, 0)
            baseline_missing += np.where(active, 0, missing)
            outgoing += delivered_out
            incoming += delivered_in
            cash += price * (delivered_in - delivered_out)
            minimum, maximum = np.minimum(minimum, soc), np.maximum(maximum, soc)
    terminal_value = asset["discharge_efficiency"] * prices[(starts + horizon_minutes - 1) // 30]
    adjustment = terminal_value * (initial - soc)
    error = soc - (initial + asset["charge_efficiency"] * incoming - outgoing / asset["discharge_efficiency"])
    return {"cost_GBP": cash + adjustment, "metered_cash_GBP": cash,
            "inventory_adjustment_GBP": adjustment, "final_SOC_MWh": soc,
            "gross_import_MWh": incoming, "gross_export_MWh": outgoing,
            "BOA_undelivered_MWh": boa_missing, "baseline_undelivered_MWh": baseline_missing,
            "minimum_SOC_MWh": minimum, "maximum_SOC_MWh": maximum,
            "energy_balance_error_MWh": error, "nomination_trace": trace}
