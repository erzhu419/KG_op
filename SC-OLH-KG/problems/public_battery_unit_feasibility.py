"""Joint perfect-information feasibility with two independent battery stocks."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from problems.public_battery_dispatch import inventory_rate, positive_integral
from problems.public_battery_feasibility import power_from_rate


def replay_schedule(workload, start, powers, protocol, nomination_segment_MW=None):
    """Rebuild minute powers and their exact chronological inventory extrema.

    No LP state, auxiliary power variable or compressed inventory constraint is
    used in this replay. Baselines are physical signed powers, one per unit.
    """
    asset = protocol["asset"]
    capacity = np.asarray(asset["unit_energy_MWh"])
    initial = capacity * protocol["controls"]["initial_soc_fraction"]
    end = start + len(powers) * 30
    baseline = np.repeat(powers, 30, axis=0)
    active = workload["active"][start:end]
    first = np.where(active, workload["first_MW"][start:end], baseline)
    last = np.where(active, workload["last_MW"][start:end], baseline)
    crossing = first * last < 0
    # Integrate trapezoids or the two sign-separated triangles directly.
    denominator = 120 * np.abs(last - first)
    exported = np.where(crossing, np.divide(np.maximum(first, 0) ** 2 + np.maximum(last, 0) ** 2,
        denominator, out=np.zeros_like(first), where=crossing),
        (np.maximum(first, 0) + np.maximum(last, 0)) / 120)
    imported = np.where(crossing, np.divide(np.maximum(-first, 0) ** 2 + np.maximum(-last, 0) ** 2,
        denominator, out=np.zeros_like(first), where=crossing),
        (np.maximum(-first, 0) + np.maximum(-last, 0)) / 120)
    delta = asset["charge_efficiency"] * imported - exported / asset["discharge_efficiency"]
    prefix = np.vstack((np.zeros(2), np.cumsum(delta, axis=0)))
    fraction = np.divide(-first, last - first, out=np.zeros_like(first), where=crossing)
    rate = np.where(first >= 0, -first / asset["discharge_efficiency"],
                    -first * asset["charge_efficiency"])
    at_zero = rate * fraction / 120
    low = np.minimum(np.minimum(0, delta), at_zero)
    high = np.maximum(np.maximum(0, delta), at_zero)
    minimum = initial + np.minimum(0, np.min(prefix[:-1] + low, axis=0))
    maximum = initial + np.maximum(0, np.max(prefix[:-1] + high, axis=0))
    final = initial + prefix[-1]
    incoming, outgoing = imported.sum(axis=0), exported.sum(axis=0)
    error = final - (initial + asset["charge_efficiency"] * incoming - outgoing / asset["discharge_efficiency"])
    gross_peak = max(np.max(np.sum(np.abs(first), axis=1)), np.max(np.sum(np.abs(last), axis=1)))
    nominal_peak = np.max(np.sum(np.abs(powers), axis=1))
    bootstrap = powers[:protocol["controls"]["initial_nomination_minutes"] // 30]
    locked = np.broadcast_to(protocol["controls"]["initial_nomination_MW"], bootstrap.shape)
    result = {"minimum_unit_SOC_MWh": minimum.tolist(), "maximum_unit_SOC_MWh": maximum.tolist(),
            "final_unit_SOC_MWh": final.tolist(), "gross_import_MWh": incoming.tolist(),
            "gross_export_MWh": outgoing.tolist(), "energy_balance_error_MWh": error.tolist(),
            "maximum_inventory_bound_violation_MWh": float(max(0, -minimum.min(), np.max(maximum - capacity))),
            "peak_gross_execution_MW": float(gross_peak), "peak_gross_nomination_MW": float(nominal_peak),
            "maximum_gross_power_violation_MW": float(max(0, gross_peak - asset["power_MW"])),
            "maximum_nomination_power_violation_MW": float(max(0, nominal_peak - asset["power_MW"])),
            "maximum_initial_nomination_deviation_MW": float(np.max(np.abs(bootstrap - locked))),
            "first_hour_maximum_nomination_MW": float(np.max(np.abs(bootstrap)))}
    if nomination_segment_MW is not None:
        left, right = np.asarray(nomination_segment_MW)
        direction = right - left
        denominator = direction @ direction
        command = powers[len(bootstrap)]
        fraction = float(np.clip((command - left) @ direction / denominator, 0, 1)) if denominator else 0.
        result.update({"nomination_segment_fraction": fraction,
                       "maximum_nomination_segment_deviation_MW": float(np.max(np.abs(command - left - fraction * direction)))})
    return result


def unit_physical_oracle(workload, start, horizon_minutes, protocol, nomination_segment_MW=None):
    """Optionally constrain the next unlocked nomination to a sign-fixed segment."""
    asset = protocol["asset"]
    energy_tol = protocol["replay"]["energy_tolerance_MWh"]
    power_tol = protocol["replay"]["power_tolerance_MW"]
    active = workload["active"][start:start + horizon_minutes]
    first = workload["first_MW"][start:start + horizon_minutes]
    last = workload["last_MW"][start:start + horizon_minutes]
    gross = np.maximum(np.sum(np.abs(first), axis=1), np.sum(np.abs(last), axis=1))
    peak = float(gross.max())
    base = {"peak_forced_gross_MW": peak, "LP_solved": False, "witness_replayed": False}
    if peak > asset["power_MW"] + power_tol:
        return {**base, "status": "forced_power_infeasible"}, None
    delta = (asset["charge_efficiency"] * positive_integral(-first, -last, 1 / 60)
             - positive_integral(first, last, 1 / 60) / asset["discharge_efficiency"])
    crossing = first * last < 0
    fraction = np.divide(-first, last - first, out=np.zeros_like(first), where=crossing)
    turn = inventory_rate(first, asset) * fraction / 120
    low, high = np.minimum(np.minimum(0, delta), turn), np.maximum(np.maximum(0, delta), turn)
    slots = horizon_minutes // 30
    rates_count = 2 * slots
    variables = 6 * slots + 2
    rows, cols, values, limits = [], [], [], []
    eq_rows, eq_cols, eq_values, offsets = [], [], [], []
    locked_slots = protocol["controls"]["initial_nomination_minutes"] // 30
    locked = np.broadcast_to(protocol["controls"]["initial_nomination_MW"], (locked_slots, 2))
    locked_rates = inventory_rate(locked, asset)
    bounds = [(locked_rates[j, u], locked_rates[j, u]) if j < locked_slots else (None, None)
              for j in range(slots) for u in (0, 1)]
    power_bounds = []

    def inequality(entries, limit):
        row = len(limits)
        for column, value in entries:
            rows.append(row)
            cols.append(column)
            values.append(value)
        limits.append(limit)

    for slot in range(slots):
        left, right = slot * 30, (slot + 1) * 30
        for unit in (0, 1):
            rate_col = slot * 2 + unit
            abs_col = rates_count + rate_col
            stock_col = 2 * rates_count + slot * 2 + unit
            idle, offset = 0, 0.
            groups = {0: [0., 0.]}
            for minute in range(left, right):
                if active[minute, unit]:
                    extrema = groups.setdefault(idle, [offset, offset])
                    extrema[0] = min(extrema[0], offset + low[minute, unit])
                    extrema[1] = max(extrema[1], offset + high[minute, unit])
                    offset += delta[minute, unit]
                else:
                    idle += 1
            extrema = groups.setdefault(idle, [offset, offset])
            extrema[0], extrema[1] = min(extrema[0], offset), max(extrema[1], offset)
            for idle_minutes, (minimum, maximum) in groups.items():
                inequality([(stock_col, 1), (rate_col, idle_minutes / 60)], asset["unit_energy_MWh"][unit] - maximum)
                inequality([(stock_col, -1), (rate_col, -idle_minutes / 60)], minimum)
            eq_row = len(offsets)
            eq_rows.extend((eq_row,) * 3)
            eq_cols.extend((rate_col, stock_col, stock_col + 2))
            eq_values.extend((-idle / 60, -1, 1))
            offsets.append(offset)
            inequality([(rate_col, 1 / asset["charge_efficiency"]), (abs_col, -1)], 0)
            inequality([(rate_col, -asset["discharge_efficiency"]), (abs_col, -1)], 0)
            free = ~active[left:right, unit]
            headroom = asset["power_MW"] - np.max(gross[left:right][free]) if np.any(free) else asset["power_MW"]
            power_bounds.append((0, max(0, headroom)))
        inequality([(rates_count + 2 * slot, 1), (rates_count + 2 * slot + 1, 1)], asset["power_MW"])
    bounds += power_bounds
    initial = np.asarray(asset["unit_energy_MWh"]) * protocol["controls"]["initial_soc_fraction"]
    bounds += [(initial[u], initial[u]) if j == 0 else (0, asset["unit_energy_MWh"][u])
               for j in range(slots + 1) for u in (0, 1)]
    if nomination_segment_MW is not None:
        left, right = inventory_rate(np.asarray(nomination_segment_MW), asset)
        for unit in (0, 1):
            row = len(offsets)
            eq_rows.extend((row, row))
            eq_cols.extend((2 * locked_slots + unit, variables))
            eq_values.extend((1., left[unit] - right[unit]))
            offsets.append(left[unit])
        bounds.append((0, 1))
        variables += 1
    matrix = coo_matrix((values, (rows, cols)), shape=(len(limits), variables)).tocsr()
    equalities = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(len(offsets), variables)).tocsr()
    options = {k: protocol["solver"][k] for k in ("primal_feasibility_tolerance", "dual_feasibility_tolerance")}
    options["time_limit"] = protocol["solver"]["time_limit_seconds_per_window"]
    solution = linprog(np.zeros(variables), A_ub=matrix, b_ub=limits, A_eq=equalities,
                       b_eq=offsets, bounds=bounds, method="highs", options=options)
    base.update({"LP_solved": True, "solver_status": int(solution.status), "solver_message": solution.message,
                 "inequality_rows": len(limits), "variables": variables})
    if solution.status == 2:
        return {**base, "status": "energy_or_commitment_infeasible"}, None
    if solution.status != 0:
        return {**base, "status": "solver_unresolved"}, None
    powers = power_from_rate(solution.x[:rates_count].reshape(slots, 2), asset)
    witness = replay_schedule(workload, start, powers, protocol, nomination_segment_MW)
    valid = (witness["maximum_inventory_bound_violation_MWh"] <= energy_tol
             and np.max(np.abs(witness["energy_balance_error_MWh"])) <= energy_tol
             and witness["maximum_gross_power_violation_MW"] <= power_tol
             and witness["maximum_nomination_power_violation_MW"] <= power_tol
             and witness["maximum_initial_nomination_deviation_MW"] <= power_tol)
    if nomination_segment_MW is not None:
        valid = valid and witness["maximum_nomination_segment_deviation_MW"] <= power_tol
    return {**base, "witness_replayed": True, **witness,
            "status": "replayed_feasible" if valid else "witness_unresolved"}, powers if valid else None
