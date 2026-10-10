"""Two BM-unit workloads with a pooled site inventory and delayed nominations.

This is a development abstraction. Gross unit flows retain conversion losses;
published net connection power is conservatively also imposed on gross power.
Failed windows stop and have no full-window economic outcome.
Historical BOAs are fixed requests; simulated availability does not alter them.
"""
from datetime import timedelta

import numpy as np

from performance.inspect_public_battery_dispatch import instant, reconstruct
from problems.public_battery_dispatch import inventory_rate, positive_integral


def site_workload(units, start, stop):
    count = int((stop - start).total_seconds() / 60)
    shape = (count, len(units))
    result = {"active": np.zeros(shape, bool), "first_MW": np.zeros(shape),
              "last_MW": np.zeros(shape)}
    for unit, segments in enumerate(units):
        for row in reconstruct(segments, start, stop):
            if not row["active_BOA"]:
                continue
            left = int((instant(row["from_utc"]) - start).total_seconds() / 60)
            right = int((instant(row["to_utc"]) - start).total_seconds() / 60)
            values = np.linspace(row["from_MW"], row["to_MW"], right - left + 1)
            result["active"][left:right, unit] = True
            result["first_MW"][left:right, unit] = values[:-1]
            result["last_MW"][left:right, unit] = values[1:]
    return result


def receipt_plans(units, start, count):
    """Fixed half-hour clock; each 90-minute plan uses its received prefix only."""
    shape = (count // 30, 90, len(units))
    result = {"active": np.zeros(shape, bool), "first_MW": np.zeros(shape),
              "last_MW": np.zeros(shape)}
    for decision in range(shape[0]):
        now = start + timedelta(minutes=30 * decision)
        end = now + timedelta(minutes=90)
        for unit, segments in enumerate(units):
            visible = sorted((s for s in segments if s.received <= now
                              and s.start < end and s.stop > now),
                             key=lambda s: (s.received, s.number))
            for s in visible:
                left, right = max(now, s.start), min(end, s.stop)
                lo = int((left - now).total_seconds() / 60)
                hi = int((right - now).total_seconds() / 60)
                values = np.linspace(s.power(left), s.power(right), hi - lo + 1)
                result["active"][decision, lo:hi, unit] = True
                result["first_MW"][decision, lo:hi, unit] = values[:-1]
                result["last_MW"][decision, lo:hi, unit] = values[1:]
    return result


def planning_terms(plans, asset):
    """Compress visible plans to forced SOC changes, idle exposure, power bounds."""
    active = plans["active"].reshape(-1, 3, 30, 2)
    first = plans["first_MW"].reshape(active.shape)
    last = plans["last_MW"].reshape(active.shape)
    forced = (asset["charge_efficiency"] * positive_integral(-first, -last, 1 / 60)
              - positive_integral(first, last, 1 / 60) / asset["discharge_efficiency"])
    idle = np.sum(~active, axis=(2, 3)) / 120  # total nomination is split in two
    gross = np.maximum(np.sum(np.abs(first), axis=3), np.sum(np.abs(last), axis=3))
    idle_units = np.sum(~active, axis=3)
    bound = np.divide(2 * np.maximum(asset["power_MW"] - gross, 0), idle_units,
                      out=np.full_like(gross, asset["power_MW"]), where=idle_units > 0)
    return {"forced": np.sum(forced, axis=(2, 3)), "idle_hours": idle,
            "future_power_bound_MW": np.minimum(asset["power_MW"], np.min(bound[:, 2], axis=1))}


def nominate_site(soc, current, following, target, decisions, terms, asset):
    forced, idle = terms["forced"][decisions], terms["idle_hours"][decisions]
    projected = (soc + forced.sum(axis=1) + inventory_rate(current, asset) * idle[:, 0]
                 + inventory_rate(following, asset) * idle[:, 1])
    difference = projected - target
    command = np.where(difference >= 0, difference * asset["discharge_efficiency"],
                       difference / asset["charge_efficiency"])
    command = np.divide(command, idle[:, 2], out=np.zeros_like(command), where=idle[:, 2] > 0)
    bound = terms["future_power_bound_MW"][decisions]
    return np.clip(command, -bound, bound)


def minute_pieces(first, last):
    """Split at each unit's zero crossing; each resulting SOC rate is linear."""
    crossing = first * last < 0
    if not np.any(crossing):
        yield first, last, 1 / 60
        return
    cuts = np.divide(-first, last - first, out=np.ones_like(first), where=crossing)
    cuts = np.sort(np.concatenate((np.zeros((*first.shape[:-1], 1)), cuts,
                                   np.ones((*first.shape[:-1], 1))), axis=-1), axis=-1)
    for left, right in zip(np.moveaxis(cuts[..., :-1], -1, 0),
                           np.moveaxis(cuts[..., 1:], -1, 0)):
        yield first + (last - first) * left[..., None], \
              first + (last - first) * right[..., None], (right - left) / 60


def inventory_piece(first, last, hours, asset):
    outgoing = np.sum(np.maximum(first, 0) + np.maximum(last, 0), axis=-1) * hours / 2
    incoming = np.sum(np.maximum(-first, 0) + np.maximum(-last, 0), axis=-1) * hours / 2
    a, b = (np.sum(inventory_rate(power, asset), axis=-1) for power in (first, last))
    delta = asset["charge_efficiency"] * incoming - outgoing / asset["discharge_efficiency"]
    crossing = a * b < 0
    fraction = np.divide(-a, b - a, out=np.zeros_like(a), where=crossing)
    extremum = hours * (a * fraction + (b - a) * fraction ** 2 / 2)
    lower = np.minimum(np.minimum(delta, 0), np.where(crossing, extremum, 0))
    upper = np.maximum(np.maximum(delta, 0), np.where(crossing, extremum, 0))
    gross_peak = np.maximum(np.sum(np.abs(first), axis=-1), np.sum(np.abs(last), axis=-1))
    return delta, lower, upper, outgoing, incoming, gross_peak


def simulate_site(workload, prices, targets, starts, horizon_minutes, asset, terms,
                  initial_fraction=0.5, tolerance=1e-8):
    starts = np.asarray(starts, int)
    shape = (len(targets), len(starts))
    initial = asset["energy_MWh"] * initial_fraction
    soc = np.full(shape, initial)
    current, following, future = (np.zeros(shape) for _ in range(3))
    cash, outgoing, incoming = (np.zeros(shape) for _ in range(3))
    minimum, maximum = soc.copy(), soc.copy()
    unit_delta = np.zeros((*shape, 2))
    unit_minimum, unit_maximum = unit_delta.copy(), unit_delta.copy()
    failure_minute = np.full(shape, -1, int)
    power_failure, inventory_failure = (np.zeros(shape, bool) for _ in range(2))
    trace = []
    for minute in range(horizon_minutes):
        indices = starts + minute
        if minute % 30 == 0:
            if minute:
                current, following = following, future
            if minute + 60 < horizon_minutes:
                target = targets[:, (indices + 60) // 30] * asset["energy_MWh"]
                future = nominate_site(soc, current, following, target, indices // 30, terms, asset)
                trace.append({"decision_minute": minute, "delivery_minute": minute + 60,
                              "first_policy_window_active": bool(failure_minute[0, 0] < 0),
                              "first_policy_window_nomination_MW": float(future[0, 0])})
        active = workload["active"][indices]
        first = np.where(active, workload["first_MW"][indices], current[..., None] / 2)
        last = np.where(active, workload["last_MW"][indices], current[..., None] / 2)
        for a, b, hours in minute_pieces(first, last):
            delta, low, high, exported, imported, peak = inventory_piece(a, b, hours, asset)
            alive = failure_minute < 0
            bad_power = peak > asset["power_MW"] + tolerance
            bad_inventory = (soc + low < -tolerance) | (soc + high > asset["energy_MWh"] + tolerance)
            failed = alive & (bad_power | bad_inventory)
            failure_minute[failed] = minute
            power_failure |= alive & bad_power
            inventory_failure |= alive & bad_inventory
            delivered = alive & ~failed
            unit_step = (asset["charge_efficiency"] * (np.maximum(-a, 0) + np.maximum(-b, 0))
                         - (np.maximum(a, 0) + np.maximum(b, 0)) / asset["discharge_efficiency"])
            unit_step *= np.asarray(hours)[..., None] / 2
            unit_delta += np.where(delivered[..., None], unit_step, 0)
            unit_minimum = np.minimum(unit_minimum, unit_delta)
            unit_maximum = np.maximum(unit_maximum, unit_delta)
            minimum = np.minimum(minimum, np.where(delivered, soc + low, soc))
            maximum = np.maximum(maximum, np.where(delivered, soc + high, soc))
            soc += np.where(delivered, delta, 0)
            outgoing += np.where(delivered, exported, 0)
            incoming += np.where(delivered, imported, 0)
            cash += np.where(delivered, prices[indices // 30] * (imported - exported), 0)
    success = failure_minute < 0
    adjustment = asset["discharge_efficiency"] * prices[(starts + horizon_minutes - 1) // 30] * (initial - soc)
    error = soc - (initial + asset["charge_efficiency"] * incoming - outgoing / asset["discharge_efficiency"])
    completed = {"cost_GBP": cash + adjustment, "metered_cash_GBP": cash,
                 "inventory_adjustment_GBP": adjustment, "final_SOC_MWh": soc,
                 "gross_import_MWh": incoming, "gross_export_MWh": outgoing}
    return {**{key: np.where(success, value, np.nan) for key, value in completed.items()},
            "unit_minimum_SOC_change_MWh": unit_minimum,
            "unit_maximum_SOC_change_MWh": unit_maximum,
            "equal_partition_success": success & np.all((unit_minimum >= -asset["energy_MWh"] / 4 - tolerance)
                                    & (unit_maximum <= asset["energy_MWh"] / 4 + tolerance), axis=-1),
            "success": success, "first_failure_minute": failure_minute,
            "power_failure": power_failure, "inventory_failure": inventory_failure,
            "minimum_SOC_MWh": minimum, "maximum_SOC_MWh": maximum,
            "energy_balance_error_MWh": error, "nomination_trace": trace}
