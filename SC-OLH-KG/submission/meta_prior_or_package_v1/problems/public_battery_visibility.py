"""Nomination-time reconstruction of the frozen controller's first failures."""
import numpy as np

from performance.inspect_public_battery_dispatch import instant, reconstruct
from problems.public_battery_dispatch import inventory_rate, positive_integral


def receipt_metadata(units, start, stop):
    shape = (int((stop - start).total_seconds() / 60), 2)
    result = {"received_minute": np.zeros(shape, int), "acceptance_number": np.zeros(shape, int)}
    for u, segments in enumerate(units):
        for row in reconstruct(segments, start, stop):
            if row["active_BOA"]:
                left = int((instant(row["from_utc"]) - start).total_seconds() / 60)
                right = int((instant(row["to_utc"]) - start).total_seconds() / 60)
                result["received_minute"][left:right, u] = int((instant(row["received_utc"]) - start).total_seconds() / 60)
                result["acceptance_number"][left:right, u] = row["acceptance_number"]
    return result


def trajectory(first, last, initial, length_minutes, asset, *, hour_break_minutes=()):
    """Exact energy/extrema, restarting accumulation at supplied execution hours."""
    count = int(np.ceil(length_minutes))
    first, last = first[:count].copy(), last[:count].copy()
    durations = np.ones(count)
    durations[-1] = length_minutes - (count - 1)
    last[-1] = first[-1] + (last[-1] - first[-1]) * durations[-1]
    hours = durations[:, None] / 60
    incoming = positive_integral(-first, -last, hours)
    outgoing = positive_integral(first, last, hours)
    delta = asset["charge_efficiency"] * incoming - outgoing / asset["discharge_efficiency"]
    crossing = first * last < 0
    fraction = np.divide(-first, last - first, out=np.zeros_like(first), where=crossing)
    turn = inventory_rate(first, asset) * fraction * hours / 2
    low, high = np.minimum(np.minimum(0, delta), turn), np.maximum(np.maximum(0, delta), turn)
    minimum, maximum, stock = np.asarray(initial).copy(), np.asarray(initial).copy(), np.asarray(initial).copy()
    left = 0
    for right in [t for t in hour_break_minutes if 0 < t < count] + [count]:
        prefix = np.vstack((np.zeros(2), np.cumsum(delta[left:right], axis=0)))
        minimum = np.minimum(minimum, stock + np.minimum(0, np.min(prefix[:-1] + low[left:right], axis=0)))
        maximum = np.maximum(maximum, stock + np.maximum(0, np.max(prefix[:-1] + high[left:right], axis=0)))
        stock = stock + prefix[-1]
        left = right
    capacity = np.asarray(asset["unit_energy_MWh"])
    violation = np.maximum(np.maximum(-minimum, maximum - capacity), 0)
    gross = max(np.sum(np.abs(first), axis=1).max(), np.sum(np.abs(last), axis=1).max())
    return {"minimum_SOC_MWh": minimum.tolist(), "maximum_SOC_MWh": maximum.tolist(),
            "final_SOC_MWh": stock.tolist(),
            "unit_bound_violation_MWh": violation.tolist(),
            "power_bound_violation_MW": float(max(gross - asset["power_MW"], 0))}


def classify_failure(context, window_start, plans, workload, receipts, asset, tolerance=1e-8):
    relative = context["nomination_decision_minute"]
    if relative is None:
        return {"classification": "unresolved_trace_discrepancy", "reason": "failure has no post-bootstrap nomination"}
    decision = window_start + relative
    index = decision // 30
    length = context["failure_minute"] - relative + context["failure_piece_stop_fraction"]
    baseline = np.repeat(np.vstack((context["pending_MW"], context["nomination_MW"])), 30, axis=0)
    initial = np.array(context["decision_SOC_MWh"])
    known_active = plans["active"][index]
    known_first = np.where(known_active, plans["first_MW"][index], baseline)
    known_last = np.where(known_active, plans["last_MW"][index], baseline)
    active = workload["active"][decision:decision + 90]
    actual_first = np.where(active, workload["first_MW"][decision:decision + 90], baseline)
    actual_last = np.where(active, workload["last_MW"][decision:decision + 90], baseline)
    known = trajectory(known_first, known_last, initial, length, asset)
    pending = trajectory(known_first, known_last, initial, min(length, 60), asset)
    actual = trajectory(actual_first, actual_last, initial, length, asset)

    def unsafe(metrics):
        return max(metrics["unit_bound_violation_MWh"]) > tolerance or metrics["power_bound_violation_MW"] > tolerance

    count = int(np.ceil(length))
    changed = ((np.abs(actual_first[:count] - known_first[:count]) > tolerance)
               | (np.abs(actual_last[:count] - known_last[:count]) > tolerance))
    later = set()
    for minute, unit in zip(*np.nonzero(changed & active[:count])):
        received = receipts["received_minute"][decision + minute, unit]
        if received > decision:
            later.add((int(unit + 1), int(receipts["acceptance_number"][decision + minute, unit]), int(received)))
    if not unsafe(actual):
        label = "unresolved_trace_discrepancy"
    elif unsafe(pending):
        label = "known_pending_hour_violation"
    elif unsafe(known):
        label = "known_new_delivery_violation"
    elif later:
        label = "later_receipt_path_change"
    else:
        label = "unresolved_trace_discrepancy"
    return {"classification": label, "decision_absolute_minute": int(decision),
            "predicted_prefix_minutes": float(length), "known_prefix": known,
            "known_pending_hour": pending, "actual_prefix": actual,
            "changed_power_minutes": int(np.any(changed, axis=1).sum()),
            "later_changed_receipts": [{"unit": u, "acceptance_number": n, "received_minute": r}
                                       for u, n, r in sorted(later)]}
