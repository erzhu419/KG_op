"""Tail PN from two already announced active calls and exact inventory flows."""
import numpy as np

from problems.public_battery_announcements import released_announcements
from problems.public_battery_dispatch import inventory_rate, positive_integral
from problems.public_battery_visibility import trajectory


def ramp_inventory(first, last, asset):
    return (asset["charge_efficiency"] * positive_integral(-np.asarray(first), -np.asarray(last), 1 / 60)
            - positive_integral(first, last, 1 / 60) / asset["discharge_efficiency"])


def tail_stock_projection(soc, current, following, first_peak, command, asset):
    """Inventory at the second call, 90 minutes after this nomination."""
    return (np.asarray(soc) + inventory_rate(np.asarray(current), asset) / 2
            + ramp_inventory(following, first_peak, asset) + inventory_rate(np.asarray(first_peak), asset) / 2
            + ramp_inventory(first_peak, command, asset) + inventory_rate(np.asarray(command), asset) * 28 / 60)


def nominate_announced_pair(soc, current, following, target_fractions, minute, announcements, asset):
    """A component command exists only for two fully released active calls."""
    known = {row["delivery_minute"]: row for row in released_announcements(announcements, minute)}
    pair = [known.get(minute + offset) for offset in (30, 90)]
    complete = all(row is not None and row["direction"] is not None for row in pair)
    info = {"known_pair_complete": complete, "scalar_stock_projection_evaluations": 0,
            "nomination_feasible": False, "nomination_reason": "incomplete_or_inactive_announced_pair"}
    if not complete:
        return None, info
    first_peak = np.asarray(pair[0]["peak_MW"], float)
    base = (np.asarray(soc) + inventory_rate(np.asarray(current), asset) / 2
            + ramp_inventory(following, first_peak, asset) + inventory_rate(first_peak, asset) / 2)
    return stock_interval_command(base, first_peak, pair[0], pair[1], target_fractions, asset, info)


def stock_interval_command(base, first_peak, first_row, second_row, target_fractions, asset, info):
    """Solve the frozen monotone stock intervals and shared-power allocation."""
    capacity = np.asarray(asset["unit_energy_MWh"])
    lower, upper = np.zeros(2), capacity.copy()
    if second_row["direction"] == "export":
        lower = np.abs(second_row["peak_MW"]) * 31 / 60 / asset["discharge_efficiency"]
    elif second_row["direction"] == "import":
        upper -= asset["charge_efficiency"] * np.abs(second_row["peak_MW"]) * 31 / 60

    def project(unit, command):
        info["scalar_stock_projection_evaluations"] += 1
        tail = (inventory_rate(np.asarray(command), asset) / 2 if first_peak is None else
                ramp_inventory(first_peak[unit], command, asset)
                + inventory_rate(np.asarray(command), asset) * 28 / 60)
        return float(base[unit] + tail)

    power = asset["power_MW"]
    attainable_upper = np.array([project(u, -power) for u in range(2)])
    attainable_lower = np.array([project(u, power) for u in range(2)])
    feasible_lower, feasible_upper = np.maximum(lower, attainable_lower), np.minimum(upper, attainable_upper)
    info.update(first_direction=first_row["direction"], second_direction=second_row["direction"],
                second_stock_band_MWh=np.column_stack((lower, upper)).tolist(),
                attainable_stock_band_MWh=np.column_stack((attainable_lower, attainable_upper)).tolist(),
                feasible_stock_band_MWh=np.column_stack((feasible_lower, feasible_upper)).tolist())
    if np.any(feasible_lower > feasible_upper):
        info["nomination_reason"] = "empty_stock_band"
        return None, info

    def inverse_bracket(unit, stock):
        if stock >= attainable_upper[unit]:
            return -power, -power
        if stock <= attainable_lower[unit]:
            return power, power
        left, right = -power, power
        for _ in range(48):
            middle = (left + right) / 2
            if project(unit, middle) > stock:
                left = middle
            else:
                right = middle
        return left, right

    command_lower = np.array([inverse_bracket(u, feasible_upper[u])[1] for u in range(2)])
    command_upper = np.array([inverse_bracket(u, feasible_lower[u])[0] for u in range(2)])
    target = np.clip(np.asarray(target_fractions) * capacity, feasible_lower, feasible_upper)
    preferred = np.array([sum(inverse_bracket(u, target[u])) / 2 for u in range(2)])
    preferred = np.clip(preferred, command_lower, command_upper)
    minimum = np.clip(np.zeros(2), command_lower, command_upper)
    required, desired = float(np.abs(minimum).sum()), float(np.abs(preferred).sum())
    info.update(command_interval_MW=np.column_stack((command_lower, command_upper)).tolist(),
                clipped_target_stock_MWh=target.tolist(), preferred_command_MW=preferred.tolist(),
                minimum_command_MW=minimum.tolist(), minimum_required_gross_power_MW=required,
                preferred_gross_power_MW=desired)
    if required > power:
        info["nomination_reason"] = "shared_power_infeasible"
        return None, info
    fraction = 1. if desired <= power else (power - required) / (desired - required)
    command = minimum + fraction * (preferred - minimum)
    projected = [project(u, command[u]) for u in range(2)]
    info.update(nomination_feasible=True, nomination_reason="announced_pair_stock_interval",
                preference_fraction=fraction, nominated_gross_power_MW=float(np.abs(command).sum()),
                projected_second_call_SOC_MWh=projected)
    return command, info


def nominate_announced(soc, current, following, target_fractions, accepted, minute,
                       announcements, horizon_minutes, asset, tolerance=1e-8, *, execution_prefix=None):
    """Complete announced policy; only own admitted pulses become mandatory paths."""
    zero = np.zeros(2)
    info = {"nomination_feasible": False, "scalar_stock_projection_evaluations": 0, "reference_safety_path_evaluations": 0,
            "known_reference_failure": False, "planning_failure": False,
            "reference_execution_prefix_minutes": 0 if execution_prefix is None
            else len(execution_prefix["first_MW"]), "nomination_reason": "whole_hour_zero"}
    if minute % 60 == 0:
        return zero, info
    known = {row["delivery_minute"]: row for row in released_announcements(announcements, minute)}
    pair = []
    for delivery in (minute + 30, minute + 90):
        if delivery in (0, 60) or delivery >= horizon_minutes:
            pair.append({"delivery_minute": delivery, "direction": None, "peak_MW": None})
        elif delivery in known:
            pair.append(known[delivery])
        else:
            raise ValueError(f"missing released in-horizon announcement for {delivery} at {minute}")
    baseline = np.repeat(np.asarray(current)[None], 30, axis=0)
    pending = tuple(np.where(accepted["active"][0, :30], accepted[key][0, :30], baseline)
                    for key in ("first_MW", "last_MW"))
    delta = (asset["charge_efficiency"] * positive_integral(-pending[0], -pending[1], 1 / 60)
             - positive_integral(pending[0], pending[1], 1 / 60) / asset["discharge_efficiency"])
    pending_delta = delta.sum(axis=0)
    first_peak = None if pair[0]["direction"] is None else np.asarray(pair[0]["peak_MW"])
    base = np.asarray(soc) + pending_delta
    if first_peak is None:
        base = base + inventory_rate(np.asarray(following), asset) / 2
    else:
        base = base + ramp_inventory(following, first_peak, asset) + inventory_rate(first_peak, asset) / 2
    info.update(known_pair_complete=True, pending_inventory_change_MWh=pending_delta.tolist(),
                known_delivery_slots=[row["delivery_minute"] for row in pair])
    candidate, info = stock_interval_command(base, first_peak, pair[0], pair[1], target_fractions, asset, info)
    planning_reason = info["nomination_reason"]

    def known_physics(command):
        baseline = np.repeat(np.vstack((current, following, command)), 30, axis=0)
        reference = tuple(np.where(accepted["active"][0], accepted[key][0], baseline)
                          for key in ("first_MW", "last_MW"))
        initial = np.asarray(soc)
        hour_offset = minute % 60
        if execution_prefix is not None:
            reference = tuple(np.vstack((execution_prefix[key], endpoint))
                              for key, endpoint in zip(("first_MW", "last_MW"), reference))
            initial = np.asarray(execution_prefix["initial_SOC_MWh"])
            hour_offset = (minute - len(execution_prefix["first_MW"])) % 60
        info["reference_safety_path_evaluations"] += 1
        return trajectory(*reference, initial, len(reference[0]), asset,
                          hour_break_minutes=range(60 - hour_offset, len(reference[0]), 60))

    def safe(metrics):
        return (max(metrics["unit_bound_violation_MWh"]) <= tolerance
                and metrics["power_bound_violation_MW"] <= tolerance)

    zero_metrics = known_physics(zero)
    info["zero_reference_physics"] = zero_metrics
    command, selected = zero, zero_metrics
    if not safe(zero_metrics):
        info.update(known_reference_failure=True, nomination_reason="retained_unsafe_zero_reference")
    elif candidate is None:
        info.update(planning_failure=True, nomination_reason=planning_reason)
    else:
        minimum = np.asarray(info["minimum_command_MW"])
        minimum_metrics = known_physics(minimum)
        info["minimum_reference_physics"] = minimum_metrics
        if not safe(minimum_metrics):
            info.update(planning_failure=True, nomination_reason="unsafe_minimum_band_command")
        else:
            candidate_metrics = known_physics(candidate)
            info["candidate_reference_physics"] = candidate_metrics
            command, selected = candidate, candidate_metrics
            info["reference_safety_scale"] = 1.
            if not safe(candidate_metrics):
                low, high, selected = 0., 1., minimum_metrics
                for _ in range(48):
                    middle = (low + high) / 2
                    metrics = known_physics(minimum + middle * (candidate - minimum))
                    if safe(metrics):
                        low, selected = middle, metrics
                    else:
                        high = middle
                command = minimum + low * (candidate - minimum)
                info.update(reference_safety_scale=low, nomination_reason="safe_interpolation")
    info["nominated_reference_physics"] = selected
    info["nominated_gross_power_MW"] = float(np.abs(command).sum())
    tail = (inventory_rate(command, asset) / 2 if first_peak is None else
            ramp_inventory(first_peak, command, asset) + inventory_rate(command, asset) * 28 / 60)
    info["scalar_stock_projection_evaluations"] += 2
    info["projected_second_call_SOC_MWh"] = (base + tail).tolist()
    return command, info
