#!/usr/bin/env python3
"""Frozen whole-site development diagnostic; second-unit responses only."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import requests

from performance.inspect_independent_energy_data import instant, retrieve, write_csv
from performance.inspect_public_battery_controller import calibration, load_signal, load_workload, policy_pool
from performance.inspect_public_battery_dispatch import decode
from problems.public_battery_site import planning_terms, receipt_plans, simulate_site, site_workload

PROTOCOL = ROOT / "performance/manifests/public_battery_site_development_v2_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_site_development_v2_20260930"
PREVIOUS = ROOT / "paper_artifacts/public_battery_controller_development_v1_20260930"


def second_unit(session, protocol, case, out):
    start, stop = instant(case["start"]), instant(case["stop"])
    left = start - timedelta(days=protocol["dispatch_prefix_days"])
    rows, logs = [], []
    unit = protocol["asset"]["workload_bm_units"][1]
    while left < stop:
        right = min(stop, left + timedelta(days=protocol["request_chunk_days"]))
        path = out / "samples" / f"{unit}_{left.date()}_{right.date()}.json"
        cached = path.exists()
        piece, meta = retrieve(session, protocol["api_base"], protocol["acceptances_endpoint"],
            {"bmUnit": unit, "from": left.isoformat(), "to": right.isoformat(), "format": "json"}, path)
        rows.extend(r for r in piece if left <= instant(r["timeFrom"]) < right)
        logs.append({**meta, "returned_rows": len(piece), "cached": cached})
        print(json.dumps({"unit": unit, "from": str(left.date()), "rows": len(piece),
                          "response_bytes": meta["response_bytes"], "cached": cached}), flush=True)
        left = right
    segments = decode(rows, unit)
    for s in segments:
        if s.start >= s.stop or s.received > s.start or any(t.second or t.microsecond for t in (s.start, s.stop, s.received)):
            raise ValueError("second-unit BOA violates the supported ordinary minute interface")
    return [s for s in segments if s.start < stop and s.stop > start], logs


def inspect(protocol, out):
    (out / "samples").mkdir(parents=True, exist_ok=True)
    session, fit = requests.Session(), calibration(protocol)
    ids, nodes, points = policy_pool(protocol)
    previous_protocol = json.loads((ROOT / "performance/manifests/public_battery_controller_development_v1_20260930.json").read_text())
    profiles, windows, cases, traces = [], [], [], []
    horizon = protocol["windows"]["hours"] * 60
    tolerance = protocol["failure"]["numerical_energy_tolerance_MWh"]
    for case in protocol["development"]:
        label = case["start"][:10]
        _, reused_logs, first = load_workload(session, previous_protocol, case, PREVIOUS)
        second, new_logs = second_unit(session, protocol, case, out)
        units = [first, second]
        start, stop = instant(case["start"]), instant(case["stop"])
        workload = site_workload(units, start, stop)
        plans = receipt_plans(units, start, len(workload["active"]))
        stress, prices = load_signal(protocol, case, fit)
        targets = np.array([np.interp(stress, nodes, point) for point in points])
        starts = np.arange(0, len(workload["active"]) - horizon + 1,
                           protocol["windows"]["start_stride_hours"] * 60)
        for condition in protocol["development_conditions"]:
            visible = plans if condition == "received_BOA_projection" else {k: np.zeros_like(v) for k, v in plans.items()}
            result = simulate_site(workload, prices, targets, starts, horizon, protocol["asset"],
                                   planning_terms(visible, protocol["asset"]),
                                   protocol["control"]["initial_soc_fraction"], tolerance)
            if np.max(np.abs(result["energy_balance_error_MWh"])) > tolerance:
                raise ValueError("whole-site gross flows do not conserve inventory")
            if np.min(result["minimum_SOC_MWh"]) < -tolerance or np.max(result["maximum_SOC_MWh"]) > protocol["asset"]["energy_MWh"] + tolerance:
                raise ValueError("accepted trajectory violates pooled inventory bounds")
            success = result["success"]
            for p, (profile_id, family, in_library) in enumerate(ids):
                fraction = float(np.mean(success[p]))
                completed_cost = result["cost_GBP"][p, success[p]]
                profiles.append({"period": label, "condition": condition, "profile_id": profile_id,
                    "family": family, "in_library": in_library, "window_success_fraction": fraction,
                    "empirically_feasible": fraction >= 1 - protocol["failure"]["chance_failure_probability"],
                    "mean_completed_window_cost_GBP": float(np.mean(completed_cost)) if len(completed_cost) else None,
                    "power_failure_windows": int(np.sum(result["power_failure"][p])),
                    "inventory_failure_windows": int(np.sum(result["inventory_failure"][p]))})
                for w, minute in enumerate(starts):
                    row = {"period": label, "condition": condition, "profile_id": profile_id,
                           "window_start_minute": int(minute), "success": bool(success[p, w]),
                           "first_failure_minute": int(result["first_failure_minute"][p, w]),
                           "power_failure": bool(result["power_failure"][p, w]),
                           "inventory_failure": bool(result["inventory_failure"][p, w])}
                    for key in ("cost_GBP", "metered_cash_GBP", "inventory_adjustment_GBP", "final_SOC_MWh", "gross_import_MWh", "gross_export_MWh"):
                        row[key] = float(result[key][p, w]) if success[p, w] else None
                    windows.append(row)
            library = success[[i for i, (_, _, inside) in enumerate(ids) if inside]]
            report = {"period": label, "condition": condition, "windows": len(starts),
                      "unit_segments": [len(s) for s in units],
                      "unit_acceptances": [len({s.number for s in seg}) for seg in units],
                      "library_feasible_profiles": sum(p["empirically_feasible"] for p in profiles[-len(ids):] if p["in_library"]),
                      "library_successful_windows_union": int(np.sum(np.any(library, axis=0))),
                      "extra_reference_feasible_profiles": sum(p["empirically_feasible"] for p in profiles[-len(ids):] if not p["in_library"]),
                      "max_energy_balance_error_MWh": float(np.max(np.abs(result["energy_balance_error_MWh"])))}
            cases.append(report)
            traces.append({"period": label, "condition": condition, "first_policy_first_window": result["nomination_trace"]})
            print(json.dumps(report), flush=True)
        (out / f"requests_{label}.json").write_text(json.dumps({"P1_reused": reused_logs, "P2": new_logs}, indent=2) + "\n")
    common = {}
    for condition in protocol["development_conditions"]:
        subset = [p for p in profiles if p["condition"] == condition and p["in_library"]]
        common[condition] = [name for name, _, inside in ids if inside
                             and all(p["empirically_feasible"] for p in subset if p["profile_id"] == name)]
    primary = protocol["development_conditions"][0]
    has_unsafe = any(not p["empirically_feasible"] for p in profiles if p["condition"] == primary and p["in_library"])
    logs = [log for case in protocol["development"] for log in json.loads((out / f"requests_{case['start'][:10]}.json").read_text())["P2"]]
    summary = {"protocol_id": protocol["protocol_id"], "status": "development_complete", "normalization": fit,
               "cases": cases, "library_profiles_feasible_in_all_periods": common,
               "pooled_development_gate": "proceed_to_source_archive_pilot" if common[primary] and has_unsafe else "hold_full_comparison",
               "decision": "require_unit_partition_diagnostic" if common[primary] and has_unsafe else "hold_full_comparison",
               "diagnostic_window_evaluations": len(windows),
               "P2_response_bytes": sum(log["response_bytes"] for log in logs),
               "new_API_response_bytes": sum(log["response_bytes"] for log in logs if not log["cached"]),
               "new_API_requests": sum(not log["cached"] for log in logs),
               "optimizer_calls": 0, "terminal_verifier_calls": 0, "confirmation_year_access": False,
               "limitations": protocol["model_assumptions"]}
    write_csv(out / "profiles.csv", profiles)
    write_csv(out / "windows.csv", windows)
    (out / "nomination_traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
