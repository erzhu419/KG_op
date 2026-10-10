"""Hourly fixed service calls, causal admission and delayed unit nominations.

Public receipts supply simulation direction marks. Accepted simulated pulses,
not historical BOA powers, define the controller's mandatory reference.
"""
import numpy as np

from problems.public_battery_dispatch import positive_integral
from problems.public_battery_announcements import released_announcements
from problems.public_battery_announced_nomination import nominate_announced
from problems.public_battery_absolute_dispatch import declare_absolute
from problems.public_battery_pending_availability import declare_pending, reference_pulse_power
from problems.public_battery_receipt_forecast import nominate_receipt_forecast
from problems.public_battery_units import nominate_units, unit_planning_terms
from problems.public_battery_visibility import trajectory


def receipt_marks(units, start, clock_minutes):
    """The latest nonzero announced endpoint in (t-60,t] supplies the sign."""
    ledger = []
    for unit, segments in enumerate(units):
        for segment in segments:
            if segment.last_MW == 0:
                continue
            relative = lambda time: (time - start).total_seconds() / 60
            key = (relative(segment.received), unit, segment.number,
                   relative(segment.stop), relative(segment.start))
            ledger.append((key, segment))
    marks = []
    for minute in clock_minutes:
        selected = max((row for row in ledger if minute - 60 < row[0][0] <= minute),
                       key=lambda row: row[0], default=None)
        source = None
        if selected is not None:
            key, segment = selected
            source = {"received_minute": key[0], "unit": key[1] + 1, "acceptance_number": key[2],
                      "stop_minute": key[3], "start_minute": key[4], "last_MW": segment.last_MW}
        marks.append({"minute": int(minute), "direction": None if selected is None
                      else "export" if selected[1].last_MW > 0 else "import", "source": source})
    return marks


def accepted_plan(pulse, minute):
    """Own received pulse on the next 90 minutes; PN applies outside its span."""
    shape = (1, 90, 2)
    plan = {"active": np.zeros(shape, bool), "first_MW": np.zeros(shape), "last_MW": np.zeros(shape)}
    if pulse is not None:
        offset = minute - pulse["start_minute"]
        remaining = 32 - offset
        if remaining > 0:
            plan["active"][0, :remaining] = True
            for key in ("first_MW", "last_MW"):
                plan[key][0, :remaining] = pulse[key][offset:32]
    return plan


def simulate_service(initial_SOC_MWh, initial_locked_PN_MW, target_fractions, marks,
                     dispatch_rule, horizon_minutes, service_capacity_MW, asset, tolerance=1e-8,
                     *, absolute_request_peaks_MW=None, nomination_cues=None, announcement_ledger=None,
                     stop_on_service_failure=False):
    """Execute the firm-headroom or prescribed absolute-request task.

Explicit absolute request data replaces PN-relative proposals and removes the
inactive-hour reserve obligation. Physics, event order and PN control are shared.
Targets are an ordered two-unit vector or one ordered row per eligible decision.
Population classification may stop at an irreversible service failure, keeping
the delivered prefix and all planned requests in its denominator.
    """
    soc = np.asarray(initial_SOC_MWh, float).copy()
    initial = soc.copy()
    current, following = np.asarray(initial_locked_PN_MW, float).copy()
    future = np.zeros(2)
    targets = np.asarray(target_fractions, float)
    decision_rows = len(range(0, max(0, horizon_minutes - 60), 30))
    if targets.shape not in ((2,), (decision_rows, 2)):
        raise ValueError("targets require two ordered units and one row per eligible nomination")
    capacity = np.asarray(service_capacity_MW, float)
    pulse = None
    hourly, nominations = [], []
    outgoing, incoming = np.zeros(2), np.zeros(2)
    minimum, maximum = soc.copy(), soc.copy()
    path_evaluations, physical_failure, completed = 0, None, 0
    first_service_failure = None
    stopped_for_service = False

    for minute in range(horizon_minutes):
        if minute % 30 == 0:
            if minute:
                current, following = following.copy(), future.copy()
            if minute % 60 == 0:
                plan = accepted_plan(pulse, minute)
                locked = np.vstack((current, following))
                baseline = np.repeat(locked, 30, axis=0)
                reference = tuple(np.where(plan["active"][0, :60], plan[key][0, :60], baseline)
                                  for key in ("first_MW", "last_MW"))
                # Neither upcoming mark nor its direction enters declaration.
                declarations = {direction: (declare_pending(soc, locked, capacity, direction, asset,
                                                            tolerance, reference_power=reference)
                                            if absolute_request_peaks_MW is None else
                                            declare_absolute(soc, locked, absolute_request_peaks_MW[direction],
                                                             asset, tolerance, reference_power=reference))
                                for direction in ("export", "import")}
                path_evaluations += sum(d["envelope_path_evaluations"] for d in declarations.values())
                mark = marks[minute // 60]
                proposal = admitted = None
                if mark["direction"] is not None:
                    declaration = declarations[mark["direction"]]
                    anchor = declaration["anchor_MW"]
                    proposal = (anchor + (1 if mark["direction"] == "export" else -1) * capacity
                                if absolute_request_peaks_MW is None else
                                np.asarray(absolute_request_peaks_MW[mark["direction"]], float))
                    if dispatch_rule == "full_request_reference":
                        admitted = proposal.copy()
                    elif declaration["guaranteed_peak_MW"] is not None:
                        guaranteed = declaration["guaranteed_peak_MW"]
                        admitted = np.clip(proposal, np.minimum(anchor, guaranteed), np.maximum(anchor, guaranteed))
                actual = reference_pulse_power(*reference, admitted)
                requested = reference_pulse_power(*reference, proposal)
                if admitted is not None:
                    pulse = {"start_minute": minute, "first_MW": actual[0][:32], "last_MW": actual[1][:32]}
                available = np.array([declarations[d]["available_capacity_MW"] for d in ("export", "import")])
                shortfall = np.maximum(capacity - available, 0.)
                known_failure = any(d["known_commitment_failure"] for d in declarations.values())
                difference_first, difference_last = requested[0] - actual[0], requested[1] - actual[1]
                anticipated_unmet = (positive_integral(difference_first, difference_last, 1 / 60)
                                     + positive_integral(-difference_first, -difference_last, 1 / 60)).sum()
                reasons = []
                if known_failure:
                    reasons.append("known_commitment_failure")
                if absolute_request_peaks_MW is None and np.max(shortfall) > tolerance:
                    reasons.append("firm_availability_shortfall")
                if anticipated_unmet > tolerance:
                    reasons.append("request_shortfall")
                record = {"minute": minute, "initial_SOC_MWh": soc.tolist(), "locked_PN_MW": locked.tolist(),
                          "available_capacity_MW": available.tolist(), "availability_shortfall_MW": shortfall.tolist(),
                          "declaration_reasons": {d: declarations[d]["reason"] for d in declarations},
                          "known_commitment_failure": known_failure, "direction": mark["direction"],
                          "proposal_peak_MW": None if proposal is None else proposal.tolist(),
                          "admitted_peak_MW": None if admitted is None else admitted.tolist(),
                          "unmet_increase_MWh": [0., 0.], "unmet_decrease_MWh": [0., 0.],
                          "delivery_prefix_minutes": 0, "delivery_accounting_complete": False,
                          "service_failure_reasons": reasons, "service_success": False}
                if absolute_request_peaks_MW is not None:
                    record.pop("service_success")
                    record.update(request_observed=proposal is not None,
                                  request_fulfilled=None if proposal is None else False,
                                  anticipated_unmet_energy_MWh=float(anticipated_unmet))
                hourly.append(record)
                if reasons and first_service_failure is None:
                    first_service_failure = {"minute": minute, "reasons": reasons.copy()}
                if stop_on_service_failure and first_service_failure is not None:
                    stopped_for_service = True
                    break
            if minute + 60 < horizon_minutes:
                fractions = targets if targets.ndim == 1 else targets[minute // 30]
                target = fractions * asset["unit_energy_MWh"]
                plan = accepted_plan(pulse, minute)
                forecast_info = {}
                if announcement_ledger is not None:
                    future, forecast_info = nominate_announced(
                        soc, current, following, fractions, plan, minute,
                        released_announcements(announcement_ledger, minute), horizon_minutes, asset, tolerance,
                        execution_prefix={"initial_SOC_MWh": np.array(record["initial_SOC_MWh"]),
                                          "first_MW": actual[0][:minute % 60],
                                          "last_MW": actual[1][:minute % 60]})
                    if forecast_info["known_reference_failure"] and first_service_failure is None:
                        first_service_failure = {"minute": minute, "reasons": ["known_commitment_failure_at_nomination"]}
                elif nomination_cues is None:
                    terms = unit_planning_terms(plan, asset)
                    future = nominate_units(soc[None], current[None], following[None], target[None],
                                            np.array([0]), terms, asset)[0]
                else:
                    future, forecast_info = nominate_receipt_forecast(
                        soc, current, following, target, plan, minute, nomination_cues[minute // 30],
                        absolute_request_peaks_MW, asset, tolerance,
                        execution_prefix={"initial_SOC_MWh": np.array(record["initial_SOC_MWh"]),
                                          "first_MW": actual[0][:minute % 60],
                                          "last_MW": actual[1][:minute % 60]})
                    if forecast_info["known_reference_failure"] and first_service_failure is None:
                        first_service_failure = {"minute": minute, "reasons": ["known_commitment_failure_at_nomination"]}
                nominations.append({"minute": minute, "delivery_minute": minute + 60,
                                    "target_fractions": fractions.tolist(),
                                    "SOC_MWh": soc.tolist(), "locked_PN_MW": [current.tolist(), following.tolist()],
                                    "nomination_MW": future.tolist(), **forecast_info})
                if stop_on_service_failure and first_service_failure is not None:
                    stopped_for_service = True
                    break

        actual_first = actual_last = current
        if pulse is not None and 0 <= minute - pulse["start_minute"] < 32:
            offset = minute - pulse["start_minute"]
            actual_first, actual_last = pulse["first_MW"][offset], pulse["last_MW"][offset]
        # Use the declaration's hour origin and accumulation order. Repeated
        # one-minute SOC updates can cross the same 1e-8 boundary by rounding.
        physics = trajectory(actual[0], actual[1], np.array(record["initial_SOC_MWh"]),
                             minute % 60 + 1, asset)
        if max(physics["unit_bound_violation_MWh"]) > tolerance or physics["power_bound_violation_MW"] > tolerance:
            physical_failure = {"minute": minute, "last_safe_SOC_MWh": soc.tolist(),
                                "power_first_MW": actual_first.tolist(), "power_last_MW": actual_last.tolist(),
                                **physics}
            record["service_failure_reasons"].append("physical_failure")
            if first_service_failure is None:
                first_service_failure = {"minute": minute, "reasons": ["physical_failure"]}
            break
        exported = positive_integral(actual_first, actual_last, 1 / 60)
        imported = positive_integral(-actual_first, -actual_last, 1 / 60)
        outgoing += exported
        incoming += imported
        minimum = np.minimum(minimum, physics["minimum_SOC_MWh"])
        maximum = np.maximum(maximum, physics["maximum_SOC_MWh"])
        soc = np.asarray(physics["final_SOC_MWh"])
        # Only delivered minutes count; a truncated requested pulse stays censored.
        offset = minute % 60
        error_first, error_last = requested[0][offset] - actual_first, requested[1][offset] - actual_last
        for key, sign in (("unmet_increase_MWh", 1), ("unmet_decrease_MWh", -1)):
            record[key] = (np.array(record[key]) + positive_integral(sign * error_first, sign * error_last, 1 / 60)).tolist()
        record["delivery_prefix_minutes"] += 1
        completed += 1
        if offset == 59:
            record["delivery_accounting_complete"] = True
            if absolute_request_peaks_MW is None:
                record["service_success"] = not record["service_failure_reasons"]
            elif record["request_observed"]:
                record["request_fulfilled"] = not record["service_failure_reasons"]

    physical_complete = physical_failure is None and completed == horizon_minutes
    expected_hours = horizon_minutes // 60
    summary = {"dispatch_rule": dispatch_rule, "planned_minutes": horizon_minutes,
               "completed_minutes": completed, "physical_complete": physical_complete,
               "window_success": physical_complete and len(hourly) == expected_hours and all(not r["service_failure_reasons"] for r in hourly),
               "scheduled_service_hours": expected_hours, "assessed_service_hours": len(hourly),
               "successful_service_hours": sum(r.get("service_success", False) for r in hourly),
               "service_failure_hours": sum(bool(r["service_failure_reasons"]) for r in hourly),
               "unassessed_service_hours": expected_hours - len(hourly),
               "activation_hours": sum(r["direction"] is not None for r in hourly),
               "capacity_shortfall_hours": sum(max(max(v) for v in r["availability_shortfall_MW"]) > tolerance for r in hourly),
               "request_shortfall_hours": sum("request_shortfall" in r["service_failure_reasons"] for r in hourly),
               "known_commitment_failure_hours": sum(r["known_commitment_failure"] for r in hourly),
               "unmet_increase_MWh": np.sum([r["unmet_increase_MWh"] for r in hourly], axis=0).tolist(),
               "unmet_decrease_MWh": np.sum([r["unmet_decrease_MWh"] for r in hourly], axis=0).tolist(),
               "delivery_accounting_complete": physical_complete,
               "first_service_failure": first_service_failure, "first_physical_failure": physical_failure,
               "final_or_last_safe_SOC_MWh": soc.tolist(), "minimum_accepted_SOC_MWh": minimum.tolist(),
               "maximum_accepted_SOC_MWh": maximum.tolist(), "gross_export_MWh": outgoing.tolist(),
               "gross_import_MWh": incoming.tolist(), "envelope_path_evaluations": path_evaluations,
               "nomination_decisions": len(nominations), "minute_physics_evaluations": completed + (physical_failure is not None),
               "energy_balance_error_MWh": (soc - initial - asset["charge_efficiency"] * incoming
                                           + outgoing / asset["discharge_efficiency"]).tolist()}
    if absolute_request_peaks_MW is not None:
        planned = sum(m["direction"] is not None for m in marks)
        observed = sum(r["request_observed"] for r in hourly)
        fulfilled = sum(r["request_fulfilled"] is True for r in hourly)
        for key in ("scheduled_service_hours", "assessed_service_hours", "successful_service_hours",
                    "service_failure_hours", "unassessed_service_hours", "activation_hours"):
            summary.pop(key)
        summary["limited_declaration_hours"] = summary.pop("capacity_shortfall_hours")
        summary["first_dispatch_failure"] = summary.pop("first_service_failure")
        summary.update(task="absolute_dispatch_tracking", scheduled_hourly_clocks=expected_hours,
                       assessed_hourly_clocks=len(hourly), inactive_hourly_clocks=len(hourly) - observed,
                       planned_requests=planned, observed_requests=observed, fulfilled_requests=fulfilled,
                       unfulfilled_observed_requests=observed - fulfilled, unobserved_requests=planned - observed,
                       unfulfilled_or_unassessed_requests=planned - fulfilled)
        summary["window_success"] = summary["window_success"] and fulfilled == planned
    if nomination_cues is not None:
        known_failures = sum(n["known_reference_failure"] for n in nominations)
        summary.update(nomination_controller="receipt_forecast_inventory_priority",
                       receipt_forecast_decisions=sum(n["forecast_used"] for n in nominations),
                       forecast_band_infeasible_decisions=sum(n.get("forecast_band_feasible") is False for n in nominations),
                       nomination_safety_interpolations=sum(n["nomination_reason"] == "safe_interpolation" for n in nominations),
                       nomination_known_reference_failure_decisions=known_failures,
                       nomination_safety_path_evaluations=sum(n["reference_safety_path_evaluations"] for n in nominations),
                       analytic_nomination_calls=sum(n["analytic_nomination_calls"] for n in nominations))
        summary["window_success"] = summary["window_success"] and known_failures == 0
    if announcement_ledger is not None:
        known_failures = sum(n["known_reference_failure"] for n in nominations)
        summary.update(task="announced_absolute_dispatch_tracking", nomination_controller="announced_inventory_priority",
                       half_hour_announced_decisions=sum(n["minute"] % 60 == 30 for n in nominations),
                       whole_hour_zero_decisions=sum(n["minute"] % 60 == 0 for n in nominations),
                       nomination_planning_failure_decisions=sum(n["planning_failure"] for n in nominations),
                       nomination_safety_interpolations=sum(n["nomination_reason"] == "safe_interpolation" for n in nominations),
                       nomination_known_reference_failure_decisions=known_failures,
                       nomination_safety_path_evaluations=sum(n["reference_safety_path_evaluations"] for n in nominations),
                       scalar_stock_projection_evaluations=sum(n["scalar_stock_projection_evaluations"] for n in nominations))
        summary["window_success"] = summary["window_success"] and known_failures == 0
    if stop_on_service_failure:
        summary["stopped_on_service_failure"] = stopped_for_service
        summary["physical_outcome_scope"] = "executed_prefix" if stopped_for_service else "executed_path"
    return summary, hourly, nominations
