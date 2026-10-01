#!/usr/bin/env python3
"""Frozen three-season announced screen: 872 new windows and four reused pilots."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.inspect_public_battery_dispatch import instant
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_announcements import announce_requests, delivery_clock_roster
from problems.public_battery_causal_service import receipt_marks, simulate_service

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_development_screen_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_development_screen_v1_20261001"
CALL_FIELDS = ("envelope_path_evaluations", "nomination_decisions", "half_hour_announced_decisions",
               "whole_hour_zero_decisions", "nomination_safety_path_evaluations",
               "scalar_stock_projection_evaluations", "minute_physics_evaluations")


def read(relative):
    return json.loads((ROOT / relative).read_text())


def write(out, name, data):
    (out / name).write_text(json.dumps(data, indent=2) + "\n")


def append(out, name, data):
    with (out / name).open("a") as stream:
        stream.write(json.dumps(data) + "\n")


def sample_marks(cached_protocol, case):
    start, stop = instant(case["start"]), instant(case["stop"])
    units = [cached_unit(cached_protocol, case, unit) for unit in (0, 1)]
    clocks = range(0, int((stop - start).total_seconds() / 60), 60)
    return receipt_marks(units, start, clocks)


def window_workload(marks, start_minute, contract):
    """Translate this source window while preserving its public receipt provenance."""
    source_span = contract["announcement_clocks"]["stop_exclusive_minute"]
    selected = deepcopy([mark for mark in marks if start_minute <= mark["minute"] < start_minute + source_span])
    for mark in selected:
        mark["minute"] -= start_minute
        if mark["source"] is not None:
            for key in ("received_minute", "start_minute", "stop_minute"):
                mark["source"][key] -= start_minute
    expected = list(range(0, source_span, contract["announcement_clocks"]["step_minutes"]))
    if [mark["minute"] for mark in selected] != expected:
        raise ValueError("source window does not contain every frozen announcement clock")
    announcements = announce_requests(selected, contract)
    deliveries = delivery_clock_roster(announcements, contract)
    metadata = {"source_window_start_minute": start_minute, "source_window_stop_exclusive_minute": start_minute + source_span,
                "announcement_records": len(announcements), "delivery_clock_records": len(deliveries),
                "planned_requests": sum(mark["direction"] is not None for mark in selected),
                "direction_counts": {direction: sum(mark["direction"] == direction for mark in selected)
                                     for direction in ("export", "import")},
                "provenance": "sample_request_marks.json retains sample-relative source timestamps; subtract source_window_start_minute to obtain this window's source timestamps"}
    return announcements, deliveries, metadata


def validate_pilot_workload(announcements, deliveries, pilot_protocol):
    old = pilot_protocol["task_outputs"]
    if (announcements != read(old + "/announcements.json")
            or deliveries != read(old + "/delivery_clocks.json")):
        raise ValueError("April start-zero workload differs from saved pilot; reuse is blocked")


def failure_context(result, hours, nominations):
    """Keep the earliest failure state and the two decisions immediately before it."""
    failures = [value for value in (result["first_dispatch_failure"], result["first_physical_failure"]) if value is not None]
    first = min(failures, key=lambda value: value["minute"], default=None)
    if first is None:
        return None
    minute = first["minute"]
    return {"first_failure": first, "hour_record": next((row for row in reversed(hours) if row["minute"] <= minute), None),
            "nomination_at_failure": next((row for row in nominations if row["minute"] == minute), None),
            "preceding_nominations": [row for row in nominations if row["minute"] < minute][-2:]}


def validate_result(result, metadata, nominations, tolerance):
    planned, observed, fulfilled = (result[key] for key in ("planned_requests", "observed_requests", "fulfilled_requests"))
    if (planned != metadata["planned_requests"] or observed + result["unobserved_requests"] != planned
            or fulfilled + result["unfulfilled_observed_requests"] != observed
            or max(abs(error) for error in result["energy_balance_error_MWh"]) > tolerance
            or any(row["scalar_stock_projection_evaluations"] > 300
                   or row["reference_safety_path_evaluations"] > 51 for row in nominations)):
        raise ValueError("window request/energy/call accounting discrepancy; preserve records and stop")


def summarize(protocol, rows, status):
    groups = []
    roster, controller = protocol["roster"], protocol["controller"]
    for sample in protocol["development_samples"]:
        for profile in controller["profiles"]:
            for rule in controller["dispatch_rules"]:
                selected = [row for row in rows if row["period"] == sample["start"][:10]
                            and row["profile_id"] == profile["id"] and row["dispatch_rule"] == rule]
                if not selected:
                    continue
                success = sum(row["window_success"] for row in selected)
                groups.append({"period": sample["start"][:10], "profile_id": profile["id"], "dispatch_rule": rule,
                               "assessed_windows": len(selected), "successful_windows": success,
                               "failed_windows": len(selected) - success,
                               "physically_completed_windows": sum(row["physical_complete"] for row in selected),
                               "request_failure_windows": sum(row["unfulfilled_or_unassessed_requests"] > 0 for row in selected),
                               "known_commitment_failure_windows": sum(row["known_commitment_failure_hours"] > 0
                                   or row["nomination_known_reference_failure_decisions"] > 0 for row in selected),
                               "planning_failure_windows": sum(row["nomination_planning_failure_decisions"] > 0 for row in selected),
                               "planned_requests": sum(row["planned_requests"] for row in selected),
                               "fulfilled_requests": sum(row["fulfilled_requests"] for row in selected),
                               "unobserved_requests": sum(row["unobserved_requests"] for row in selected),
                               "minimum_request_count_per_window": min(row["planned_requests"] for row in selected),
                               "maximum_request_count_per_window": max(row["planned_requests"] for row in selected),
                               "meets_development_criterion": len(selected) == roster["windows_per_sample"]
                               and success >= protocol["gate"]["required_successes_per_sample"]})
    complete = len(rows) == roster["reported_controller_window_evaluations"]
    common = [profile["id"] for profile in controller["profiles"] if complete
              and all(group["meets_development_criterion"] for group in groups if group["profile_id"] == profile["id"])]
    new = [row for row in rows if row["evaluation_origin"] == "new"]
    reused = [row for row in rows if row["evaluation_origin"] == "reused_pilot"]
    first_failure = next((row for row in rows if not row["window_success"]), None)
    return {"protocol_id": protocol["protocol_id"], "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "reported_controller_window_evaluations": len(rows),
            "new_controller_window_evaluations": len(new), "reused_controller_window_evaluations": len(reused),
            "implementation_replays": 0, "physically_completed_windows": sum(row["physical_complete"] for row in rows),
            "service_successful_windows": sum(row["window_success"] for row in rows), "groups": groups,
            "common_development_feasible_profiles": common, "development_gate_pass": complete and bool(common),
            "new_execution_calls": {key: sum(row[key] for row in new) for key in CALL_FIELDS},
            "reused_pilot_provenance_calls": {key: sum(row[key] for row in reused) for key in CALL_FIELDS},
            "maximum_energy_balance_error_MWh": max((max(abs(e) for e in row["energy_balance_error_MWh"]) for row in rows), default=0),
            "first_failed_window": None if first_failure is None else {key: first_failure[key] for key in
                 ("period", "window_start_minute", "profile_id", "dispatch_rule", "first_dispatch_failure", "first_physical_failure")},
            "decision": "register_announced_library_objective_stage" if complete and common else
                 "diagnose_earliest_failure_from_saved_records" if complete else "complete_frozen_screen",
            "unit_test_accounting": "unit_test_accounting.json (no physics or controller replay)",
            **protocol["accounting_constraints"], "source_comparison_gate": "HOLD", "library_matrix_launch": False,
            "limitations": protocol["limitations"]}


def inspect(protocol, out):
    import pytest

    out.mkdir(parents=True, exist_ok=True)
    if (out / "windows.jsonl").exists():
        raise ValueError("screen output already contains window records; select a separate output directory")

    class TestAccounting:
        def pytest_sessionfinish(self, session, exitstatus):
            counts = session.items[0].module.COUNTS
            write(out, "unit_test_accounting.json", {"pytest_exit_code": int(exitstatus),
                  "collected_tests": len(session.items), **counts, "test_controller_window_evaluations": 0,
                  "test_trajectory_evaluations": 0, "test_nomination_calls": 0})

    exit_code = pytest.main(["-q", str(ROOT / "tests/test_public_battery_announced_development.py")], plugins=[TestAccounting()])
    if exit_code:
        write(out, "summary.json", {"protocol_id": protocol["protocol_id"], "status": "interface_tests_failed",
                                   "new_controller_window_evaluations": 0, "source_comparison_gate": "HOLD"})
        return
    cached_protocol = read(protocol["cached_data_protocol"])
    pilot = read(protocol["basis_protocol"])
    pilot_runs = read(protocol["basis_outputs"] + "/runs.json")
    contract = read(protocol["task_protocol"])["contract"]
    retained = read(pilot["retained_asset_protocol"])["retained_asset"]
    asset = {"power_MW": retained["shared_gross_power_MW"],
             **{key: retained[key] for key in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    tolerance = cached_protocol["failure"]["numerical_energy_tolerance_MWh"]
    if protocol["controller"]["profiles"] != pilot["sequential_pilot"]["profiles"]:
        raise ValueError("retained target profiles differ from pilot")
    samples, workloads, rows = [], [], []
    try:
        for sample in protocol["development_samples"]:
            period = sample["start"][:10]
            case = next(case for case in cached_protocol["development"] if case["start"] == sample["start"])
            marks = sample_marks(cached_protocol, case)
            samples.append({"period": period, "source_start_utc": sample["start"], "source_stop_utc": sample["stop"], "marks": marks})
            write(out, "sample_request_marks.json", samples)
            for start in protocol["roster"]["source_window_start_minutes"]:
                announcements, deliveries, metadata = window_workload(marks, start, contract)
                workloads.append({"period": period, **metadata})
                write(out, "workloads.json", workloads)
                reuse = period == pilot["sequential_pilot"]["period"] and start == 0
                if reuse:
                    validate_pilot_workload(announcements, deliveries, pilot)
                    write(out, "pilot_reuse.json", {"matched_announcements": len(announcements),
                          "matched_delivery_clocks": len(deliveries), "reused_windows": len(pilot_runs),
                          "pilot_artifacts": protocol["basis_outputs"], "controller_replays": 0, "nomination_replays": 0})
                dispatcher_marks = [{**row, "minute": row["delivery_minute"]} for row in deliveries]
                for profile in protocol["controller"]["profiles"]:
                    for rule in protocol["controller"]["dispatch_rules"]:
                        identity = {"period": period, "window_start_minute": start, "profile_id": profile["id"], "dispatch_rule": rule}
                        if reuse:
                            result = next(row for row in pilot_runs if row["profile_id"] == profile["id"] and row["dispatch_rule"] == rule)
                            row = {**result, **identity, "evaluation_origin": "reused_pilot",
                                   "provenance": protocol["basis_outputs"] + "/runs.json"}
                        else:
                            result, hours, nominations = simulate_service(
                                pilot["sequential_pilot"]["initial_SOC_MWh"], pilot["sequential_pilot"]["initial_locked_PN_MW"],
                                profile["target_fractions"], dispatcher_marks, rule, contract["total_execution_horizon_minutes"],
                                np.abs(contract["absolute_peak_MW"]["export"]), asset, tolerance,
                                absolute_request_peaks_MW=contract["absolute_peak_MW"], announcement_ledger=announcements)
                            row = {**result, **identity, "evaluation_origin": "new"}
                            context = failure_context(result, hours, nominations)
                            if context is not None:
                                append(out, "first_failures.jsonl", {**identity, **context})
                            # Persist the result before any discrepancy can block later cells.
                            append(out, "windows.jsonl", row)
                            rows.append(row)
                            validate_result(result, metadata, nominations, tolerance)
                        if reuse:
                            validate_result(result, metadata, [], tolerance)
                            append(out, "windows.jsonl", row)
                            rows.append(row)
                        print(json.dumps({**identity, "evaluation_origin": row["evaluation_origin"],
                              "completed_cells": len(rows), "window_success": result["window_success"],
                              "completed_minutes": result["completed_minutes"], "planned_requests": result["planned_requests"],
                              "fulfilled_requests": result["fulfilled_requests"], "first_dispatch_failure": result["first_dispatch_failure"],
                              "first_physical_failure": result["first_physical_failure"]}), flush=True)
                write(out, "progress.json", summarize(protocol, rows, "running"))
        summary = summarize(protocol, rows, "announced_development_screen_complete")
        if (summary["new_controller_window_evaluations"] != protocol["roster"]["new_controller_window_evaluations"]
                or summary["reused_controller_window_evaluations"] != protocol["roster"]["reused_controller_window_evaluations"]):
            raise ValueError("completed roster differs from frozen new/reused counts")
        write(out, "summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        summary = summarize(protocol, rows, "implementation_or_interface_discrepancy")
        summary["error"] = str(error)
        write(out, "summary.json", summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
