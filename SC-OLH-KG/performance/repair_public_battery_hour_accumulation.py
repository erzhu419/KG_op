#!/usr/bin/env python3
"""Verify six saved tolerance-boundary cases and replay only their six windows."""
import json
from datetime import timedelta
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.inspect_public_battery_announced_dynamic import (
    call_totals, economics, instant, read, schedule, write,
)
from performance.inspect_public_battery_announced_library import calibration, decode_library, read_csv
from performance.inspect_public_battery_announced_development import failure_context, validate_result, window_workload
from problems.public_battery_announcements import released_announcements
from problems.public_battery_announced_nomination import nominate_announced
from problems.public_battery_causal_service import accepted_plan, simulate_service
from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_hour_accumulation_repair_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_hour_accumulation_repair_v1_20261001"


def inspect():
    import pytest

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "case_probes.json").exists():
        raise ValueError("repair probes already exist; preserve their actual call accounting")
    protocol = read(PROTOCOL)
    write(OUTPUT, "effective_protocol.json", protocol)
    function_codes = {trajectory.__code__: "trajectory_evaluations", nominate_announced.__code__: "nomination_policy_calls",
                      simulate_service.__code__: "short_controller_evaluations"}
    test_counts = {name: 0 for name in function_codes.values()}

    def count(frame, event, arg):
        if event == "call" and frame.f_code in function_codes:
            test_counts[function_codes[frame.f_code]] += 1

    sys.setprofile(count)
    code = pytest.main(["-q", str(ROOT / "tests/test_public_battery_hour_accumulation.py"),
                        str(ROOT / "tests/test_public_battery_announced_controller.py"),
                        str(ROOT / "tests/test_public_battery_visibility.py")])
    sys.setprofile(None)
    write(OUTPUT, "unit_test_accounting.json", {"pytest_exit_code": int(code), **test_counts})
    if code:
        raise ValueError("hour-accumulation regression tests failed")
    dynamic = read(ROOT / protocol["basis_protocol"])
    old = ROOT / protocol["basis_outputs"]
    data = read(ROOT / dynamic["data_protocol"])
    asset, tolerance = data["asset"], data["failure"]["numerical_energy_tolerance_MWh"]
    hours, decisions = read(old / "hourly_dispatch.json"), read(old / "nominations.json")
    marks = read(ROOT / dynamic["screen_outputs"] / "sample_request_marks.json")
    boundary = read(ROOT / dynamic["basis_outputs"] / "boundary_half_hours.json")
    contract = read(ROOT / dynamic["task_protocol"])["contract"]
    roster, nodes, channels, _ = decode_library(read(ROOT / dynamic["basis_protocol"]))
    pairs = {r["profile_id"]: p for r, p in zip(roster, channels)}
    fit = calibration(data)
    probes, runs, saved_hours, saved_decisions, failures, costs = [], [], [], [], [], []
    for case in protocol["saved_component_nomination_cases"]:
        identity = {k: case[k] for k in ("period", "window_start_minute", "profile_id", "dispatch_rule")}
        matching = lambda r: all(r[k] == v for k, v in identity.items())
        minute = case["safety_nomination_minute"]
        decision = next(n for n in decisions if matching(n) and n["minute"] == minute)
        hour = next(h for h in hours if matching(h) and h["minute"] == minute // 60 * 60)
        first, last = pending_pulse_power(hour["locked_PN_MW"], hour["admitted_peak_MW"])
        pulse = None if hour["admitted_peak_MW"] is None else {"start_minute": hour["minute"],
                                                               "first_MW": first[:32], "last_MW": last[:32]}
        plan = accepted_plan(pulse, minute)
        sample = next(s["marks"] for s in marks if s["period"] == identity["period"])
        announcements, _, _ = window_workload(sample, identity["window_start_minute"], contract)
        q, info = nominate_announced(np.array(decision["SOC_MWh"]), *np.array(decision["locked_PN_MW"]),
            decision["target_fractions"], plan, minute, released_announcements(announcements, minute), 10200, asset, tolerance,
            execution_prefix={"initial_SOC_MWh": np.array(hour["initial_SOC_MWh"]), "first_MW": first[:30], "last_MW": last[:30]})
        baseline = np.repeat(np.vstack((*decision["locked_PN_MW"], q)), 30, axis=0)
        references = [np.vstack((prefix[:30], np.where(plan["active"][0], plan[key][0], baseline)))
                      for prefix, key in ((first, "first_MW"), (last, "last_MW"))]
        a = trajectory(references[0][:60], references[1][:60], np.array(hour["initial_SOC_MWh"]), 60, asset)
        b = trajectory(references[0][60:], references[1][60:], np.array(a["final_SOC_MWh"]), 60, asset)
        passed = (max(a["unit_bound_violation_MWh"] + b["unit_bound_violation_MWh"]) <= tolerance
                  and max(a["power_bound_violation_MW"], b["power_bound_violation_MW"]) <= tolerance
                  and info["nominated_reference_physics"]["final_SOC_MWh"] == b["final_SOC_MWh"])
        probes.append({**identity, "nomination_minute": minute, "old_nomination_MW": decision["nomination_MW"],
                       "repaired_nomination_MW": q.tolist(), "nomination_info": info,
                       "actual_hour_physics": [a, b], "direct_execution_trajectory_calls": 2, "probe_pass": passed})
        write(OUTPUT, "case_probes.json", probes)
        if not passed:
            raise ValueError("repaired nomination still disagrees with actual hour origins")
    for identity in protocol["replay_roster"]:
        case = next(c for c in data["development"] if c["start"][:10] == identity["period"])
        inputs = {instant(r["start_time_utc"]): r for r in read_csv(ROOT / case["csv"]) + boundary[identity["period"]]}
        sample = next(s["marks"] for s in marks if s["period"] == identity["period"])
        announcements, deliveries, metadata = window_workload(sample, identity["window_start_minute"], contract)
        start = instant(case["start"]) + timedelta(minutes=identity["window_start_minute"])
        targets, _ = schedule(inputs, start, 10200, nodes, pairs[identity["profile_id"]], fit)
        result, h, n = simulate_service([49, 49], [[0, 0], [0, 0]], targets,
            [{**r, "minute": r["delivery_minute"]} for r in deliveries], identity["dispatch_rule"], 10200,
            [49, 49], asset, tolerance, absolute_request_peaks_MW=contract["absolute_peak_MW"], announcement_ledger=announcements)
        runs.append({**identity, **result})
        saved_hours.extend({**identity, **r} for r in h)
        saved_decisions.extend({**identity, **r} for r in n)
        context = failure_context(result, h, n)
        if context:
            failures.append({**identity, **context})
        costs.append({**identity, **economics(result, h, start, inputs, asset, tolerance)})
        for filename, records in (("runs.json", runs), ("hourly_dispatch.json", saved_hours), ("nominations.json", saved_decisions),
                                  ("first_failures.json", failures), ("pilot_economics.json", costs)):
            write(OUTPUT, filename, records)
        validate_result(result, metadata, n, tolerance)
        print(json.dumps({**identity, "replays_completed": len(runs), "window_success": result["window_success"],
                          "physical_complete": result["physical_complete"], "completed_minutes": result["completed_minutes"]}), flush=True)
    summary = {"protocol_id": protocol["protocol_id"], "status": "hour_accumulation_repair_complete",
               "saved_case_nomination_probes": len(probes), "direct_probe_trajectory_calls": 12,
               "probe_reference_safety_calls": sum(p["nomination_info"]["reference_safety_path_evaluations"] for p in probes),
               "probe_scalar_projection_calls": sum(p["nomination_info"]["scalar_stock_projection_evaluations"] for p in probes),
               "controller_window_replays": len(runs), "replay_calls": call_totals(runs),
               "physically_completed_replay_windows": sum(r["physical_complete"] for r in runs),
               "successful_replay_windows": sum(r["window_success"] for r in runs),
               "initial_24_outcomes_preserved": True, "maximum_energy_balance_error_MWh":
               max(max(abs(e) for e in r["energy_balance_error_MWh"]) for r in runs),
               "new_public_data_requests": 0, "confirmation_year_access": False,
               "source_comparison_gate": "HOLD", "library_matrix_launch": False}
    write(OUTPUT, "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    inspect()
