"""Perfect-information physical feasibility for the fixed dispatch model.

One signed inventory rate corresponds to one physical import/export power.
Consequently the feasibility LP permits no simultaneous baseline flows.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack


def power_from_rate(rate, asset):
    return np.where(rate >= 0, -rate / asset["charge_efficiency"],
                    -rate * asset["discharge_efficiency"])


def replay(workload, start, powers, asset, initial_soc):
    """Replay oracle nominations; no policy forecasting or cost calculation."""
    soc, missing, incoming, outgoing = initial_soc, 0.0, 0.0, 0.0
    minimum, maximum = soc, soc
    for minute in range(len(powers) * 30):
        i = start + minute
        active = workload["active"][i]
        power = powers[minute // 30]
        for leg in (1, 2):
            if active:
                req_out, req_in = (workload[f"requested_{d}_{leg}"][i] for d in ("out", "in"))
                bound_out, bound_in = (workload[f"bounded_{d}_{leg}"][i] for d in ("out", "in"))
            else:
                req_out = bound_out = max(power, 0) / 60 if leg == 1 else 0
                req_in = bound_in = max(-power, 0) / 60 if leg == 1 else 0
            delivered_out = min(bound_out, max(soc, 0) * asset["discharge_efficiency"])
            soc -= delivered_out / asset["discharge_efficiency"]
            delivered_in = min(bound_in, max(asset["energy_MWh"] - soc, 0) / asset["charge_efficiency"])
            soc += delivered_in * asset["charge_efficiency"]
            missing += req_out + req_in - delivered_out - delivered_in
            incoming, outgoing = incoming + delivered_in, outgoing + delivered_out
            minimum, maximum = min(minimum, soc), max(maximum, soc)
    return {"undelivered_MWh": missing, "minimum_SOC_MWh": minimum,
            "maximum_SOC_MWh": maximum, "final_SOC_MWh": soc,
            "energy_balance_error_MWh": soc - (initial_soc + asset["charge_efficiency"] * incoming
                                              - outgoing / asset["discharge_efficiency"])}


def physical_oracle(workload, start, horizon_minutes, protocol):
    asset = protocol["asset"]
    tolerance = protocol["physical_tolerance_MWh"]
    end = start + horizon_minutes
    minimum_power_shortfall = sum(float(np.sum(workload[f"requested_{d}_{leg}"][start:end]
                                                - workload[f"bounded_{d}_{leg}"][start:end]))
                                   for d in ("out", "in") for leg in (1, 2))
    base = {"minimum_power_shortfall_MWh": minimum_power_shortfall,
            "LP_solved": False, "witness_replayed": False}
    if minimum_power_shortfall > tolerance:
        return {**base, "status": "power_infeasible"}, None
    slots = horizon_minutes // 30
    row_indices, columns, coefficients, lower, upper = [], [], [], [], []
    idle_hours, offsets = [], []
    for slot in range(slots):
        idle, offset = 0, 0.0
        groups = {0: [0.0, 0.0]}
        for i in range(start + slot * 30, start + (slot + 1) * 30):
            if workload["active"][i]:
                extrema = groups.setdefault(idle, [offset, offset])
                for leg in (1, 2):
                    offset += (asset["charge_efficiency"] * workload[f"requested_in_{leg}"][i]
                               - workload[f"requested_out_{leg}"][i] / asset["discharge_efficiency"])
                    extrema[0], extrema[1] = min(extrema[0], offset), max(extrema[1], offset)
            else:
                idle += 1
        extrema = groups.setdefault(idle, [offset, offset])
        extrema[0], extrema[1] = min(extrema[0], offset), max(extrema[1], offset)
        for minutes_idle, (minimum, maximum) in groups.items():
            row = len(lower)
            row_indices.extend((row, row))
            columns.extend((slot, slots + slot))
            coefficients.extend((minutes_idle / 60, 1))
            lower.append(-minimum)
            upper.append(asset["energy_MWh"] - maximum)
        idle_hours.append(idle / 60)
        offsets.append(offset)
    # x = [signed SOC rates, SOC at all half-hour boundaries].
    matrix = coo_matrix((coefficients, (row_indices, columns)), shape=(len(lower), 2 * slots + 1)).tocsr()
    eq_rows = np.repeat(np.arange(slots), 3)
    eq_cols = np.array([[j, slots + j, slots + j + 1] for j in range(slots)]).ravel()
    eq_values = np.array([[-idle_hours[j], -1, 1] for j in range(slots)]).ravel()
    equalities = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(slots, 2 * slots + 1)).tocsr()
    rate_bound = (-asset["power_MW"] / asset["discharge_efficiency"],
                  asset["power_MW"] * asset["charge_efficiency"])
    bounds = [(0, 0) if j * 30 < protocol["initial_zero_nomination_minutes"] else rate_bound for j in range(slots)]
    bounds += [(protocol["initial_SOC_MWh"], protocol["initial_SOC_MWh"])] + [(0, asset["energy_MWh"])] * slots
    options = {k: protocol["LP"][k] for k in ("primal_feasibility_tolerance", "dual_feasibility_tolerance")}
    options["time_limit"] = protocol["LP"]["time_limit_seconds_per_window"]
    solution = linprog(np.zeros(2 * slots + 1), A_ub=vstack((matrix, -matrix)).tocsr(),
                       b_ub=np.r_[upper, -np.array(lower)], A_eq=equalities, b_eq=offsets,
                       bounds=bounds, method=protocol["LP"]["method"], options=options)
    base.update({"LP_solved": True, "solver_status": int(solution.status),
                 "solver_message": solution.message, "inequality_rows": 2 * len(lower)})
    if solution.status == 2:
        return {**base, "status": "energy_or_commitment_infeasible"}, None
    if solution.status != 0:
        return {**base, "status": "solver_unresolved"}, None
    powers = power_from_rate(solution.x[:slots], asset)
    witness = replay(workload, start, powers, asset, protocol["initial_SOC_MWh"])
    valid = (witness["undelivered_MWh"] <= tolerance
             and witness["minimum_SOC_MWh"] >= -tolerance
             and witness["maximum_SOC_MWh"] <= asset["energy_MWh"] + tolerance
             and abs(witness["energy_balance_error_MWh"]) <= tolerance
             and np.max(np.abs(powers)) <= asset["power_MW"] + tolerance)
    return {**base, "witness_replayed": True, **witness,
            "status": "replayed_feasible" if valid else "witness_unresolved"}, powers if valid else None
