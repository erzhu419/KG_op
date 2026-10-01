#!/usr/bin/env python3
"""Twenty-two frozen components, then four announced-task pilots if the gate passes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.public_battery_announced_nomination import nominate_announced
from problems.public_battery_causal_service import accepted_plan, simulate_service
from problems.public_battery_dispatch import positive_integral
from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_controller_pilot_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_controller_pilot_v1_20261001"


def read(relative):
    return json.loads((ROOT / relative).read_text())


def write(out, name, data):
    (out / name).write_text(json.dumps(data, indent=2) + "\n")


def inventory_delta(first, last, asset):
    return (asset["charge_efficiency"] * positive_integral(-first, -last, 1 / 60)
            - positive_integral(first, last, 1 / 60) / asset["discharge_efficiency"])


def fixture_rows(directions, minute, peaks):
    return [{"release_minute": delivery - 120, "delivery_minute": delivery,
             "direction": direction, "peak_MW": None if direction == "none" else peaks[direction],
             "source": None}
            for direction, delivery in zip(directions, (minute + 30, minute + 90))]


def inspect(protocol, out, components_only=False):
    basis = read(protocol["basis_protocol"])
    task = read(protocol["task_protocol"])["contract"]
    peaks = task["absolute_peak_MW"]
    retained = read(protocol["retained_asset_protocol"])["retained_asset"]
    asset = {"power_MW": retained["shared_gross_power_MW"],
             **{k: retained[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    targets = read(basis["saved_target_protocol"])
    receipt = read(targets["basis_protocol"])
    absolute = read(receipt["basis_protocol"])
    causal = read(absolute["basis_previous_pilot"])
    tolerance = read(causal["retained_controller_protocol"])["failure"]["numerical_energy_tolerance_MWh"]
    saved = read(protocol["basis_outputs"] + "/components.json")
    roster = protocol["sequential_pilot"]
    horizon = roster["execution_minutes"]
    rows = []
    out.mkdir(parents=True, exist_ok=True)

    for profile in roster["profiles"]:
        context = next(r for r in saved if r["profile_id"] == profile["id"])
        initial = np.asarray(context["initial_SOC_MWh"])
        current, following = np.asarray(context["locked_PN_MW"])
        for pair in protocol["component_preflight"]["base_sign_pairs"]:
            minute = context["nomination_minute"]
            announcements = fixture_rows(pair.split("_"), minute, peaks)
            for row in announcements:
                if row["direction"] == "none":
                    row["direction"] = None
            plan = accepted_plan(None, minute)
            pending = np.repeat(current[None], 30, axis=0)
            rows.append(component(profile, "base", pair, minute, initial, current, following, plan,
                                  announcements, initial, (pending, pending), None, 90, horizon, asset, tolerance))
            write(out, "components.json", rows)
        for pair in ("export_export", "import_import"):
            source = next(r for r in saved if r["profile_id"] == profile["id"] and r["sign_pair"] == pair)
            direction = pair.split("_")[0]
            old_tail = np.asarray(source["nomination_MW"])
            origin_SOC = initial + np.cumsum(inventory_delta(pending, pending, asset), axis=0)[-1]
            own_first, own_last = pending_pulse_power([following, old_tail], peaks[direction])
            soc = origin_SOC + np.cumsum(inventory_delta(own_first, own_last, asset), axis=0)[29]
            origin_minute = context["nomination_minute"] + 30
            minute = origin_minute + 30
            own = {"start_minute": origin_minute, "first_MW": own_first[:32], "last_MW": own_last[:32]}
            plan = accepted_plan(own, minute)
            execution_prefix = {"initial_SOC_MWh": origin_SOC,
                                "first_MW": own_first[:30], "last_MW": own_last[:30]}
            announcements = fixture_rows([direction, direction], minute, peaks)
            rows.append(component(profile, "own_prefix", pair, minute, soc, old_tail, np.zeros(2), plan,
                                  announcements, origin_SOC, (own_first, own_last), execution_prefix,
                                  120, horizon, asset, tolerance))
            write(out, "components.json", rows)

    gate = protocol["component_preflight"]
    tests = read(str(out.relative_to(ROOT)) + "/unit_test_accounting.json")
    passed = sum(row["probe_pass"] for row in rows)
    component_summary = {
        "nomination_probes": len(rows), "passed_probes": passed, "execution_trajectory_evaluations": len(rows),
        "reference_safety_trajectory_evaluations": sum(r["reference_safety_path_evaluations"] for r in rows),
        "scalar_stock_projection_evaluations": sum(r["scalar_stock_projection_evaluations"] for r in rows),
        "maximum_scalar_stock_evaluations_per_probe": max(r["scalar_stock_projection_evaluations"] for r in rows),
        "maximum_safety_trajectories_per_probe": max(r["reference_safety_path_evaluations"] for r in rows),
        "full_requested_pulses_tested": sum(r["full_requested_pulses_tested"] for r in rows),
        "minimum_SOC_MWh": min(min(r["physics"]["minimum_SOC_MWh"]) for r in rows),
        "minimum_capacity_headroom_MWh": min(float(np.min(np.asarray(asset["unit_energy_MWh"])
                                                        - r["physics"]["maximum_SOC_MWh"])) for r in rows),
        "maximum_projection_error_MWh": max(max(abs(e) for e in r["projection_error_MWh"]) for r in rows),
        "unit_tests_pass": tests["pytest_exit_code"] == 0,
        "component_gate_pass": bool(passed == gate["planned_half_hour_nomination_probes"] == len(rows)
                                    and len(rows) <= gate["maximum_execution_trajectory_evaluations"]
                                    and tests["pytest_exit_code"] == 0)}
    write(out, "component_summary.json", component_summary)
    print(json.dumps(component_summary), flush=True)
    if not component_summary["component_gate_pass"] or components_only:
        write(out, "summary.json", {"protocol_id": protocol["protocol_id"],
              "status": "component_gate_failure" if not component_summary["component_gate_pass"] else "components_complete",
              "components": component_summary, "controller_window_evaluations": 0,
              "decision": "retain_components_and_diagnose_before_pilots", **protocol["accounting_constraints"],
              "source_comparison_gate": "HOLD", "library_matrix_launch": False})
        return

    announcements = read(protocol["task_outputs"] + "/announcements.json")
    deliveries = read(protocol["task_outputs"] + "/delivery_clocks.json")
    marks = [{**row, "minute": row["delivery_minute"]} for row in deliveries]
    runs, hours, nominations = [], [], []
    for profile in roster["profiles"]:
        for rule in roster["dispatch_rules"]:
            result, records, decisions = simulate_service(
                roster["initial_SOC_MWh"], roster["initial_locked_PN_MW"], profile["target_fractions"],
                marks, rule, horizon, np.abs(peaks["export"]), asset, tolerance,
                absolute_request_peaks_MW=peaks, announcement_ledger=announcements)
            identity = {"profile_id": profile["id"], "dispatch_rule": rule, "period": roster["period"],
                        "window_start_minute": 0}
            runs.append({**identity, **result})
            hours.extend({**identity, **row} for row in records)
            nominations.extend({**identity, **row} for row in decisions)
            for name, data in (("runs.json", runs), ("hourly_dispatch.json", hours), ("nominations.json", nominations)):
                write(out, name, data)
            if (max(abs(e) for e in result["energy_balance_error_MWh"]) > tolerance
                    or result["planned_requests"] != task["expected_active_requests"]
                    or any(d["reference_safety_path_evaluations"] > 51
                           or d["scalar_stock_projection_evaluations"] > 300 for d in decisions)):
                raise ValueError("pilot accounting discrepancy: preserve completed records and stop")
            print(json.dumps({**identity, **{k: result[k] for k in ("completed_minutes", "window_success",
                  "fulfilled_requests", "unfulfilled_observed_requests", "unobserved_requests",
                  "nomination_planning_failure_decisions", "nomination_known_reference_failure_decisions",
                  "first_dispatch_failure", "first_physical_failure")}}), flush=True)
    summary = {"protocol_id": protocol["protocol_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "announced_controller_pilot_complete", "components": component_summary,
               "controller_window_evaluations": len(runs), "implementation_replays": 0, "runs": runs,
               "physically_completed_windows": sum(r["physical_complete"] for r in runs),
               "service_successful_windows": sum(r["window_success"] for r in runs),
               "hourly_dispatch_records": len(hours), "nomination_records": len(nominations),
               "pilot_envelope_path_evaluations": sum(r["envelope_path_evaluations"] for r in runs),
               "pilot_nomination_safety_path_evaluations": sum(r["nomination_safety_path_evaluations"] for r in runs),
               "pilot_scalar_stock_projection_evaluations": sum(r["scalar_stock_projection_evaluations"] for r in runs),
               "pilot_minute_physics_evaluations": sum(r["minute_physics_evaluations"] for r in runs),
               "unit_test_accounting": "unit_test_accounting.json (separate from all components and pilots)",
               "decision": "register_broader_announced_development" if any(r["window_success"] for r in runs)
               else "diagnose_first_failure_from_saved_inputs", "retained_failure_probability": .05,
               **protocol["accounting_constraints"], "source_comparison_gate": "HOLD", "library_matrix_launch": False,
               "limitations": protocol["limitations"]}
    write(out, "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("runs", "limitations")}), flush=True)


def component(profile, kind, pair, minute, soc, current, following, accepted, announcements,
              origin_SOC, existing_hour, execution_prefix, second_offset, horizon, asset, tolerance):
    command, info = nominate_announced(soc, current, following, profile["target_fractions"], accepted,
                                       minute, announcements, horizon, asset, tolerance,
                                       execution_prefix=execution_prefix)
    first_hour = pending_pulse_power([following, command], announcements[0]["peak_MW"])
    second_hour = pending_pulse_power(np.zeros((2, 2)), announcements[1]["peak_MW"])
    first, last = (np.concatenate(parts) for parts in zip(existing_hour, first_hour, second_hour))
    physics = trajectory(first, last, origin_SOC, len(first), asset)
    delta = inventory_delta(first, last, asset)
    second_soc = origin_SOC + np.cumsum(delta, axis=0)[second_offset - 1]
    error = second_soc - info["projected_second_call_SOC_MWh"]
    band = np.asarray(info["second_stock_band_MWh"])
    in_band = bool(np.all(second_soc >= band[:, 0] - tolerance) and np.all(second_soc <= band[:, 1] + tolerance))
    safe = bool(max(physics["unit_bound_violation_MWh"]) <= tolerance and physics["power_bound_violation_MW"] <= tolerance)
    balance = np.asarray(physics["final_SOC_MWh"]) - origin_SOC - delta.sum(axis=0)
    passed = bool(safe and in_band and not info["known_reference_failure"] and not info["planning_failure"]
                  and np.max(np.abs(error)) <= tolerance and np.max(np.abs(balance)) <= tolerance
                  and info["scalar_stock_projection_evaluations"] <= 300 and info["reference_safety_path_evaluations"] <= 51)
    row = {"profile_id": profile["id"], "kind": kind, "sign_pair": pair, "nomination_minute": minute,
           "SOC_MWh": np.asarray(soc).tolist(), "locked_PN_MW": [np.asarray(current).tolist(), np.asarray(following).tolist()],
           "target_fractions": profile["target_fractions"], "fixture_announcements": announcements,
           "execution_origin_SOC_MWh": origin_SOC.tolist(), "execution_minutes": len(first),
           "nomination_MW": command.tolist(), **info, "physics": physics, "physical_feasible": safe,
           "full_requested_pulses_tested": sum(a["direction"] is not None for a in announcements) + (kind == "own_prefix"),
           "second_call_initial_SOC_MWh": second_soc.tolist(), "projection_error_MWh": error.tolist(),
           "second_call_stock_in_required_band": in_band, "energy_balance_error_MWh": balance.tolist(), "probe_pass": passed}
    print(json.dumps({k: row[k] for k in ("profile_id", "kind", "sign_pair", "probe_pass", "nomination_reason", "physics")}), flush=True)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--components-only", action="store_true")
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output, args.components_only)
