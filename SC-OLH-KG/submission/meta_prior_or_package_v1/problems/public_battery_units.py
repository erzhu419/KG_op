"""Independent inventories and delayed baselines under fixed historical requests.

BOA levels remain exogenous when the simulated baseline or inventory changes.
This is a dispatch-trace stress test, not a counterfactual BM dispatch model.
"""
import numpy as np

from problems.public_battery_dispatch import inventory_rate, positive_integral
from problems.public_battery_site import minute_pieces


def unit_planning_terms(plans, asset):
    active = plans["active"].reshape(-1, 3, 30, 2)
    first = plans["first_MW"].reshape(active.shape)
    last = plans["last_MW"].reshape(active.shape)
    forced = (asset["charge_efficiency"] * positive_integral(-first, -last, 1 / 60)
              - positive_integral(first, last, 1 / 60) / asset["discharge_efficiency"])
    gross = np.maximum(np.sum(np.abs(first[:, 2]), axis=-1),
                       np.sum(np.abs(last[:, 2]), axis=-1))
    headroom = np.maximum(asset["power_MW"] - gross, 0)
    bounds = np.where(~active[:, 2], headroom[..., None], asset["power_MW"])
    return {"forced": np.sum(forced, axis=2),
            "idle_hours": np.sum(~active, axis=2) / 60,
            "future_power_bound_MW": np.min(bounds, axis=1),
            "target_lower_MWh": np.zeros((len(active), 2)),
            "target_upper_MWh": np.broadcast_to(asset["unit_energy_MWh"], (len(active), 2)).copy()}


def nominate_units(soc, current, following, target, decisions, terms, asset):
    forced = terms["forced"][decisions]
    idle = terms["idle_hours"][decisions]
    projected = (soc + forced.sum(axis=1) + inventory_rate(current, asset) * idle[:, 0]
                 + inventory_rate(following, asset) * idle[:, 1])
    difference = projected - target
    command = np.where(difference >= 0, difference * asset["discharge_efficiency"],
                       difference / asset["charge_efficiency"])
    command = np.divide(command, idle[:, 2], out=np.zeros_like(command), where=idle[:, 2] > 0)
    bound = terms["future_power_bound_MW"][decisions]
    command = np.clip(command, -bound, bound)
    total = np.sum(np.abs(command), axis=-1, keepdims=True)
    scale = np.divide(asset["power_MW"], total, out=np.ones_like(total), where=total > asset["power_MW"])
    return command * scale


def forced_run_bounds(workload, start, horizon_minutes, asset):
    """Necessary energy bounds during contiguous BOAs, with no idle replenishment.

    The initial inventory of each run is free, so passing this bound does not
    establish feasibility under the actual initial state or delayed commitments.
    """
    active = workload["active"][start:start + horizon_minutes]
    first = workload["first_MW"][start:start + horizon_minutes]
    last = workload["last_MW"][start:start + horizon_minutes]
    delta = (asset["charge_efficiency"] * positive_integral(-first, -last, 1 / 60)
             - positive_integral(first, last, 1 / 60) / asset["discharge_efficiency"])
    crossing = first * last < 0
    fraction = np.divide(-first, last - first, out=np.zeros_like(first), where=crossing)
    turn = inventory_rate(first, asset) * fraction / 120
    low, high = np.minimum(np.minimum(delta, 0), turn), np.maximum(np.maximum(delta, 0), turn)
    result = []
    for u in (0, 1):
        edges = np.diff(np.r_[False, active[:, u], False].astype(int))
        worst = {"unit": u + 1, "required_capacity_MWh": 0., "from_minute": None, "to_minute": None}
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            prefix = np.r_[0., np.cumsum(delta[left:right, u])]
            minimum = float(np.minimum(0, np.min(prefix[:-1] + low[left:right, u])))
            maximum = float(np.maximum(0, np.max(prefix[:-1] + high[left:right, u])))
            required = maximum - minimum
            if required > worst["required_capacity_MWh"]:
                worst = {"unit": u + 1, "required_capacity_MWh": required,
                         "from_minute": int(start + left), "to_minute": int(start + right)}
        result.append(worst)
    return result


def simulate_units(workload, prices, targets, starts, horizon_minutes, asset, terms,
                   initial_fraction=.5, tolerance=1e-8):
    """Shared or per-unit targets; stop at first physical failure, price completions."""
    starts = np.asarray(starts, int)
    targets = np.asarray(targets)
    if targets.ndim == 2:
        targets = targets[..., None]
    shape = (len(targets), len(starts))
    capacity = np.asarray(asset["unit_energy_MWh"])
    initial = capacity * initial_fraction
    soc = np.broadcast_to(initial, (*shape, 2)).copy()
    current, following, future = (np.zeros_like(soc) for _ in range(3))
    outgoing, incoming = np.zeros_like(soc), np.zeros_like(soc)
    cash = np.zeros(shape)
    minimum, maximum = soc.copy(), soc.copy()
    failure_minute = np.full(shape, -1, int)
    power_failure = np.zeros(shape, bool)
    unit_inventory_failure = np.zeros_like(soc, bool)
    boa_inventory_failure, baseline_inventory_failure = (np.zeros(shape, bool) for _ in range(2))
    trace, failure_contexts = [], []
    current_origin = following_origin = future_origin = None

    def decision_snapshot(origin, p, w):
        return {"nomination_decision_minute": origin["decision_minute"],
                "decision_SOC_MWh": origin["SOC_MWh"][p, w].tolist(),
                "pending_MW": origin["pending_MW"][p, w].tolist(),
                "nomination_MW": origin["nomination_MW"][p, w].tolist()}

    for minute in range(horizon_minutes):
        indices = starts + minute
        if minute % 30 == 0:
            if minute:
                current, following = following, future
                current_origin, following_origin = following_origin, future_origin
            if minute + 60 < horizon_minutes:
                decisions = indices // 30
                fraction = targets[:, (indices + 60) // 30]
                lower, upper = terms["target_lower_MWh"][decisions], terms["target_upper_MWh"][decisions]
                target = lower + fraction * (upper - lower)
                future = nominate_units(soc, current, following, target, decisions, terms, asset)
                previous_origin = ({key: future_origin[key] for key in
                                    ("decision_minute", "SOC_MWh", "pending_MW", "nomination_MW")}
                                   if future_origin is not None else None)
                future_origin = {"decision_minute": minute, "SOC_MWh": soc.copy(),
                                 "pending_MW": np.stack((current, following), axis=2),
                                 "nomination_MW": future.copy(), "previous_origin": previous_origin}
                trace.append({"decision_minute": minute, "delivery_minute": minute + 60,
                              "first_policy_window_active": bool(failure_minute[0, 0] < 0),
                              "first_policy_window_SOC_MWh": soc[0, 0].tolist(),
                              "first_policy_window_target_SOC_MWh": target[0, 0].tolist(),
                              "first_policy_window_nomination_MW": future[0, 0].tolist()})
        active = workload["active"][indices]
        first = np.where(active, workload["first_MW"][indices], current)
        last = np.where(active, workload["last_MW"][indices], current)
        piece_stop = np.zeros(shape)
        for a, b, hours in minute_pieces(first, last):
            piece_stop += np.asarray(hours) * 60
            duration = np.asarray(hours)[..., None]
            exported = (np.maximum(a, 0) + np.maximum(b, 0)) * duration / 2
            imported = (np.maximum(-a, 0) + np.maximum(-b, 0)) * duration / 2
            delta = asset["charge_efficiency"] * imported - exported / asset["discharge_efficiency"]
            end = soc + delta  # each unit's power has fixed sign within a piece
            bad_unit = (end < -tolerance) | (end > capacity + tolerance)
            peak = np.maximum(np.sum(np.abs(a), axis=-1), np.sum(np.abs(b), axis=-1))
            bad_power = peak > asset["power_MW"] + tolerance
            alive = failure_minute < 0
            failed = alive & (bad_power | np.any(bad_unit, axis=-1))
            for p, w in zip(*np.nonzero(failed)):
                context = {"policy_index": int(p), "window_index": int(w), "failure_minute": minute,
                           "failure_piece_stop_fraction": float(min(1, piece_stop[p, w])),
                           "nomination_decision_minute": None, "previous_nomination": None}
                if current_origin is not None:
                    context.update(decision_snapshot(current_origin, p, w))
                    if current_origin["previous_origin"] is not None:
                        context["previous_nomination"] = decision_snapshot(current_origin["previous_origin"], p, w)
                failure_contexts.append(context)
            failure_minute[failed] = minute
            power_failure |= alive & bad_power
            unit_inventory_failure |= alive[..., None] & bad_unit
            boa_inventory_failure |= alive & np.any(bad_unit & active, axis=-1)
            baseline_inventory_failure |= alive & np.any(bad_unit & ~active, axis=-1)
            delivered = alive & ~failed
            soc = np.where(delivered[..., None], end, soc)
            minimum, maximum = np.minimum(minimum, soc), np.maximum(maximum, soc)
            outgoing += np.where(delivered[..., None], exported, 0)
            incoming += np.where(delivered[..., None], imported, 0)
            cash += np.where(delivered, prices[indices // 30] * np.sum(imported - exported, axis=-1), 0)
    success = failure_minute < 0
    terminal = asset["discharge_efficiency"] * prices[(starts + horizon_minutes - 1) // 30]
    adjustment = terminal * np.sum(initial - soc, axis=-1)
    error = soc - (initial + asset["charge_efficiency"] * incoming - outgoing / asset["discharge_efficiency"])
    completed = {"cost_GBP": cash + adjustment, "metered_cash_GBP": cash,
                 "inventory_adjustment_GBP": adjustment, "final_SOC_MWh": soc.sum(axis=-1),
                 "gross_import_MWh": incoming.sum(axis=-1), "gross_export_MWh": outgoing.sum(axis=-1)}
    return {**{key: np.where(success, value, np.nan) for key, value in completed.items()},
            "unit_final_SOC_MWh": np.where(success[..., None], soc, np.nan),
            "unit_gross_import_MWh": np.where(success[..., None], incoming, np.nan),
            "unit_gross_export_MWh": np.where(success[..., None], outgoing, np.nan),
            "minimum_unit_SOC_MWh": minimum, "maximum_unit_SOC_MWh": maximum,
            "energy_balance_error_MWh": error,
            "success": success, "first_failure_minute": failure_minute,
            "power_failure": power_failure, "unit_inventory_failure": unit_inventory_failure,
            "BOA_inventory_failure": boa_inventory_failure,
            "baseline_inventory_failure": baseline_inventory_failure, "nomination_trace": trace,
            "first_failure_contexts": failure_contexts}
