#!/usr/bin/env python3
"""Six frozen dynamic-target components and 24 public-data development pilots."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.inspect_public_battery_announced_development import (
    CALL_FIELDS, failure_context, validate_result, window_workload, write,
)
from performance.inspect_public_battery_announced_library import (
    calibration, decode_library, forecast_stress, instant, read, read_csv,
    settlement_cash, target_fractions, terminal_adjustment,
)
from problems.public_battery_causal_service import simulate_service
from problems.public_battery_pending_availability import pending_pulse_power

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_dynamic_targets_pilot_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_dynamic_targets_pilot_v1_20261001"


def schedule(rows, start, horizon, nodes, pair, fit):
    stresses, leads = [], []
    for minute in range(0, max(0, horizon - 60), 30):
        decision = start + timedelta(minutes=minute)
        stress, lead = forecast_stress(rows, decision + timedelta(minutes=60), decision, fit)
        stresses.append(stress)
        leads.append(lead)
    return target_fractions(nodes, pair, stresses).T, {
        "eligible_decisions": len(stresses), "minimum_forecast_publication_lead_minutes": min(leads),
        "stress_range": [min(stresses), max(stresses)], "target_range_by_unit":
        [[float(np.min(channel)), float(np.max(channel))] for channel in target_fractions(nodes, pair, stresses)],
        "last_eligible_delivery_utc": (start + timedelta(minutes=horizon - 30)).isoformat()}


def economics(result, hours, start, prices, asset, tolerance=1e-8):
    if not result["window_success"] or not result["physical_complete"]:
        return {"cost_GBP": None, "status": "undefined_failed_or_interrupted_window"}
    horizon = result["planned_minutes"]
    if (result["completed_minutes"] != horizon or [h["minute"] for h in hours] != list(range(0, horizon, 60))
            or any(h["delivery_prefix_minutes"] != 60 or not h["delivery_accounting_complete"] for h in hours)):
        raise ValueError("successful-window economic record is incomplete")
    initial = np.asarray(hours[0]["initial_SOC_MWh"], float)
    soc, imported, exported, cash, errors = initial.copy(), np.zeros(2), np.zeros(2), 0., []
    for hour in hours:
        errors.append(float(np.max(np.abs(soc - hour["initial_SOC_MWh"]))))
        first, last = pending_pulse_power(hour["locked_PN_MW"], hour["admitted_peak_MW"])
        value, incoming, outgoing = settlement_cash(first, last, start + timedelta(minutes=hour["minute"]), prices)
        cash += value
        imported += incoming.sum(axis=0)
        exported += outgoing.sum(axis=0)
        soc += asset["charge_efficiency"] * incoming.sum(axis=0) - outgoing.sum(axis=0) / asset["discharge_efficiency"]
    errors += [float(np.max(np.abs(imported - result["gross_import_MWh"]))),
               float(np.max(np.abs(exported - result["gross_export_MWh"]))),
               float(np.max(np.abs(soc - result["final_or_last_safe_SOC_MWh"])))]
    if max(errors) > tolerance:
        raise ValueError(f"economic gross-flow/inventory discrepancy: {max(errors)}")
    final_clock = start + timedelta(minutes=horizon - 30)
    price = float(prices[final_clock]["price_GBP_per_MWh"])
    value, adjustment = terminal_adjustment(initial, result["final_or_last_safe_SOC_MWh"], price, asset["discharge_efficiency"])
    return {"status": "successful_complete_window", "metered_cash_GBP": cash,
            "last_execution_settlement_utc": final_clock.isoformat(), "last_execution_price_GBP_per_MWh": price,
            "terminal_value_GBP_per_MWh": value, "inventory_adjustment_GBP": adjustment,
            "cost_GBP": cash + adjustment, "maximum_accounting_error_MWh": max(errors)}


def call_totals(runs):
    return {key: sum(row[key] for row in runs) for key in CALL_FIELDS}


def components(protocol, pilot, nodes, pairs, asset, tolerance, out):
    peaks = {"export": [49., 49.], "import": [-49., -49.]}
    ledger = [{"release_minute": 0, "delivery_minute": 120, "direction": "export", "peak_MW": peaks["export"]},
              {"release_minute": 60, "delivery_minute": 180, "direction": "import", "peak_MW": peaks["import"]}]
    marks = [{"minute": minute, "direction": direction} for minute, direction in
             zip(range(0, 240, 60), (None, None, "export", "import"))]
    saved, comparisons = [], []

    def execute(profile_id, kind, targets):
        result, hours, nominations = simulate_service([49, 49], [[0, 0], [0, 0]], targets, marks,
            "conservative_admission", 240, [49, 49], asset, tolerance,
            absolute_request_peaks_MW=peaks, announcement_ledger=ledger)
        identity = {"profile_id": profile_id, "kind": kind}
        saved.append({**identity, "result": result, "hourly": hours, "nominations": nominations})
        write(out, "component_results.json", {"runs": saved, "constant_comparisons": comparisons, "interface_gate_pass": False})
        expected = np.broadcast_to(np.asarray(targets), (6, 2)) if np.asarray(targets).ndim == 1 else targets
        if any(n["target_fractions"] != expected[n["minute"] // 30].tolist() for n in nominations):
            raise ValueError("component targets reached the wrong units or nomination clock")
        validate_result(result, {"planned_requests": 2}, nominations, tolerance)
        return result, hours, nominations

    for profile in pilot["sequential_pilot"]["profiles"]:
        static = execute(profile["id"], "constant_static", profile["target_fractions"])
        dynamic = execute(profile["id"], "constant_schedule", np.tile(profile["target_fractions"], (6, 1)))
        if static != dynamic:
            raise ValueError("constant schedule changes physical or nomination behavior")
        comparisons.append({"profile_id": profile["id"], "exact_constant_equivalence": True})
    for profile_id in protocol["pilots"]["profiles"]:
        targets = target_fractions(nodes, pairs[profile_id], [0., .25, .5, .75, 1., .25]).T
        execute(profile_id, "functional_schedule", targets)
    output = {"runs": saved, "constant_comparisons": comparisons,
              "short_controller_evaluations": len(saved), "calls": call_totals([r["result"] for r in saved]),
              "interface_gate_pass": True, "functional_failures_block_pilots": False}
    write(out, "component_results.json", output)
    return output


def summarize(protocol, rows, components_result, status):
    groups = []
    for day in protocol["pilots"]["source_sample_start_dates"]:
        for profile in protocol["pilots"]["profiles"]:
            for rule in protocol["pilots"]["dispatch_rules"]:
                chosen = [r for r in rows if r["period"] == day and r["profile_id"] == profile and r["dispatch_rule"] == rule]
                groups.append({"period": day, "profile_id": profile, "dispatch_rule": rule,
                               "assessed_windows": len(chosen), "successful_windows": sum(r["window_success"] for r in chosen),
                               "physically_completed_windows": sum(r["physical_complete"] for r in chosen),
                               "planned_requests": sum(r["planned_requests"] for r in chosen),
                               "fulfilled_requests": sum(r["fulfilled_requests"] for r in chosen),
                               "unobserved_requests": sum(r["unobserved_requests"] for r in chosen)})
    return {"protocol_id": protocol["protocol_id"], "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "new_controller_window_evaluations": len(rows), "controller_window_replays": 0,
            "short_component_controller_evaluations": components_result["short_controller_evaluations"],
            "component_calls": components_result["calls"], "pilot_calls": call_totals(rows),
            "successful_windows": sum(r["window_success"] for r in rows),
            "physically_completed_windows": sum(r["physical_complete"] for r in rows), "groups": groups,
            "maximum_energy_balance_error_MWh": max((max(abs(e) for e in r["energy_balance_error_MWh"]) for r in rows), default=0),
            **protocol["accounting_constraints"], "source_comparison_gate": "HOLD", "library_matrix_launch": False,
            "development_reliability_gate_assessed": False, "limitations": protocol["limitations"]}


def inspect(protocol, out):
    import pytest

    out.mkdir(parents=True, exist_ok=True)
    if (out / "runs.json").exists() or (out / "component_results.json").exists():
        raise ValueError("existing dynamic execution records; use a separate output for a replay")
    write(out, "effective_protocol.json", protocol)

    class TestAccounting:
        def pytest_sessionfinish(self, session, exitstatus):
            write(out, "unit_test_accounting.json", {"pytest_exit_code": int(exitstatus),
                  "collected_tests": len(session.items), **session.items[0].module.COUNTS,
                  "test_controller_window_evaluations": 0, "test_nomination_policy_calls": 0,
                  "test_declaration_path_evaluations": 0, "test_trajectory_evaluations": 0})

    code = pytest.main(["-q", str(ROOT / "tests/test_public_battery_announced_dynamic.py")], plugins=[TestAccounting()])
    if code:
        write(out, "summary.json", {"status": "dynamic_interface_tests_failed", "new_controller_window_evaluations": 0,
                                   "source_comparison_gate": "HOLD"})
        return
    data = read(ROOT / protocol["data_protocol"])
    basis = read(ROOT / protocol["basis_protocol"])
    pilot = read(ROOT / protocol["controller_protocol"])
    roster, nodes, channels, _ = decode_library(basis)
    pairs = {r["profile_id"]: pair for r, pair in zip(roster, channels)}
    asset, tolerance = data["asset"], data["failure"]["numerical_energy_tolerance_MWh"]
    component = components(protocol, pilot, nodes, pairs, asset, tolerance, out)
    contract = read(ROOT / protocol["task_protocol"])["contract"]
    boundary = read(ROOT / protocol["basis_outputs"] / "boundary_half_hours.json")
    source_marks = read(ROOT / protocol["screen_outputs"] / "sample_request_marks.json")
    fit = calibration(data)
    runs, hours_saved, nominations_saved, failures, economic_rows, target_metadata = [], [], [], [], [], []
    write(out, "runs.json", runs)
    try:
        for day in protocol["pilots"]["source_sample_start_dates"]:
            case = next(c for c in data["development"] if c["start"][:10] == day)
            inputs = {instant(r["start_time_utc"]): r for r in read_csv(ROOT / case["csv"]) + boundary[day]}
            marks = next(s["marks"] for s in source_marks if s["period"] == day)
            for window in protocol["pilots"]["source_window_start_minutes"]:
                announcements, deliveries, metadata = window_workload(marks, window, contract)
                dispatch_marks = [{**r, "minute": r["delivery_minute"]} for r in deliveries]
                start = instant(case["start"]) + timedelta(minutes=window)
                for profile in protocol["pilots"]["profiles"]:
                    targets, target_info = schedule(inputs, start, protocol["pilots"]["execution_minutes"], nodes, pairs[profile], fit)
                    target_metadata.append({"period": day, "window_start_minute": window, "profile_id": profile, **target_info})
                    write(out, "target_metadata.json", target_metadata)
                    for rule in protocol["pilots"]["dispatch_rules"]:
                        identity = {"period": day, "window_start_minute": window, "profile_id": profile, "dispatch_rule": rule}
                        result, hours, nominations = simulate_service(protocol["pilots"]["initial_SOC_MWh"],
                            protocol["pilots"]["initial_locked_PN_MW"], targets, dispatch_marks, rule,
                            protocol["pilots"]["execution_minutes"], np.abs(contract["absolute_peak_MW"]["export"]),
                            asset, tolerance, absolute_request_peaks_MW=contract["absolute_peak_MW"], announcement_ledger=announcements)
                        runs.append({**identity, **result})
                        hours_saved.extend({**identity, **r} for r in hours)
                        nominations_saved.extend({**identity, **r} for r in nominations)
                        context = failure_context(result, hours, nominations)
                        if context is not None:
                            failures.append({**identity, **context})
                        # Keep each completed/failed cell before validating or moving on.
                        for filename, records in (("runs.json", runs), ("hourly_dispatch.json", hours_saved),
                                                  ("nominations.json", nominations_saved), ("first_failures.json", failures)):
                            write(out, filename, records)
                        validate_result(result, metadata, nominations, tolerance)
                        if any(n["target_fractions"] != targets[n["minute"] // 30].tolist() for n in nominations):
                            raise ValueError("pilot target routing discrepancy")
                        economic_rows.append({**identity, **economics(result, hours, start, inputs, asset, tolerance)})
                        write(out, "pilot_economics.json", economic_rows)
                        write(out, "progress.json", summarize(protocol, runs, component, "running"))
                        print(json.dumps({**identity, "completed_cells": len(runs), "window_success": result["window_success"],
                              "physical_complete": result["physical_complete"], "completed_minutes": result["completed_minutes"],
                              "planned_requests": result["planned_requests"], "fulfilled_requests": result["fulfilled_requests"],
                              "cost_GBP": economic_rows[-1]["cost_GBP"]}), flush=True)
        if len(runs) != protocol["pilots"]["planned_new_controller_window_evaluations"]:
            raise ValueError("dynamic pilot roster is incomplete")
        summary = summarize(protocol, runs, component, "dynamic_targets_pilots_complete")
        summary.update(successful_costs=len([r for r in economic_rows if r["cost_GBP"] is not None]),
                       maximum_economic_accounting_error_MWh=max((r["maximum_accounting_error_MWh"] for r in economic_rows
                                                                 if r["cost_GBP"] is not None), default=None),
                       decision="freeze_population_experiment_with_explicit_execution_budget")
        write(out, "summary.json", summary)
        write(out, "artifact_accounting.json", {"controller_windows": len(runs), "controller_window_replays": 0,
              "short_components": len(component["runs"]), "hour_records": len(hours_saved),
              "nomination_records": len(nominations_saved), "failed_window_contexts": len(failures),
              "full_cost_records": summary["successful_costs"], "undefined_cost_records": len(runs) - summary["successful_costs"],
              "target_schedules": len(target_metadata), "new_public_data_requests": 0, "minute_traces_saved": 0})
        print(json.dumps(summary), flush=True)
    except Exception as error:
        summary = summarize(protocol, runs, component, "implementation_or_interface_discrepancy")
        summary["error"] = str(error)
        write(out, "summary.json", summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(read(args.protocol), args.output)
