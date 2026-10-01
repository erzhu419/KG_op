#!/usr/bin/env python3
"""Physical upper bound using only the saved V1 development dispatch records."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_dispatch import decode
from problems.public_battery_dispatch import minute_workload
from problems.public_battery_feasibility import physical_oracle

PROTOCOL = ROOT / "performance/manifests/public_battery_physical_oracle_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_physical_oracle_v1_20260930"


def saved_workload(controller, case, input_dir):
    start, stop = instant(case["start"]), instant(case["stop"])
    left = start - timedelta(days=controller["dispatch_prefix_days"])
    rows = []
    while left < stop:
        right = min(stop, left + timedelta(days=controller["request_chunk_days"]))
        path = input_dir / "samples" / f"{left.date()}_{right.date()}_acceptances.json"
        if left == instant("2024-04-14T00:00:00Z"):
            path = ROOT / "paper_artifacts/public_battery_application_spec_20260930/pillswood_acceptances_20240414_15.json"
        data = json.loads(path.read_text())["payload"]["data"]
        rows.extend(r for r in data if left <= instant(r["timeFrom"]) < right)
        left = right
    segments = decode(rows, controller["asset"]["workload_bm_unit"])
    relevant = [s for s in segments if s.start < stop and s.stop > start]
    return minute_workload(relevant, start, stop, controller["asset"]["power_MW"])


def compare_controller_windows(oracle_rows, input_dir):
    previous = json.loads((input_dir / "summary.json").read_text())
    library = {r["profile_id"] for r in previous["profiles"] if r["in_library"]}
    oracle = {(r["period"], r["window_start_minute"]): r["status"] for r in oracle_rows}
    successful, library_successful = set(), set()
    with (input_dir / "windows.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["success"] != "True":
                continue
            key = (row["period"], int(row["window_start_minute"]))
            successful.add(key)
            if row["profile_id"] in library:
                library_successful.add(key)
    if any(oracle[key] != "replayed_feasible" for key in successful):
        raise ValueError("oracle excludes an already successful causal-control window")
    return [{"period": period,
             "oracle_replayed_feasible_windows": sum(status == "replayed_feasible" for key, status in oracle.items() if key[0] == period),
             "any_library_policy_successful_windows": sum(key[0] == period for key in library_successful),
             "oracle_feasible_windows_missed_by_entire_library": sum(status == "replayed_feasible" and key not in library_successful
                  for key, status in oracle.items() if key[0] == period),
             "already_successful_control_windows_included": True}
            for period in sorted({r["period"] for r in oracle_rows})]


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    controller = json.loads((ROOT / "performance/manifests" / protocol["controller_protocol"]).read_text())
    input_dir = ROOT / protocol["input_directory"]
    horizon = controller["windows"]["hours"] * 60
    rows, cases, witnesses = [], [], {}
    for case in controller["development"]:
        label = case["start"][:10]
        workload = saved_workload(controller, case, input_dir)
        starts = np.arange(0, len(workload["active"]) - horizon + 1,
                           controller["windows"]["start_stride_hours"] * 60)
        schedules = np.full((len(starts), horizon // 30), np.nan)
        case_rows = []
        for w, start in enumerate(starts):
            report, powers = physical_oracle(workload, int(start), horizon, protocol)
            row = {"period": label, "window_start_minute": int(start), **report}
            case_rows.append(row)
            if powers is not None:
                schedules[w] = powers
            if (w + 1) % 20 == 0:
                print(json.dumps({"period": label, "completed_windows": w + 1,
                                  "statuses": dict(Counter(r["status"] for r in case_rows))}), flush=True)
        rows.extend(case_rows)
        witnesses[label] = schedules
        statuses = Counter(r["status"] for r in case_rows)
        feasible_rows = [r for r in case_rows if r["status"] == "replayed_feasible"]
        report = {"period": label, "windows": len(starts), "statuses": dict(statuses),
            "LP_solves": sum(r["LP_solved"] for r in case_rows),
            "physical_witness_replays": sum(r["witness_replayed"] for r in case_rows),
            "unresolved_windows": sum(statuses[s] for s in ("solver_unresolved", "witness_unresolved")),
            "oracle_feasible_fraction": statuses["replayed_feasible"] / len(starts),
            "max_feasible_witness_undelivered_MWh": max((r["undelivered_MWh"] for r in feasible_rows), default=None),
            "max_feasible_witness_energy_balance_error_MWh": max((abs(r["energy_balance_error_MWh"]) for r in feasible_rows), default=None)}
        cases.append(report)
        print(json.dumps(report), flush=True)
    # A missing witness is represented by NaN, with its reason retained in windows.json.
    np.savez_compressed(out / "witnesses.npz", **witnesses)
    (out / "windows.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(out / "cases.csv", [{k: json.dumps(v) if isinstance(v, dict) else v for k, v in c.items()} for c in cases])
    contrast = compare_controller_windows(rows, input_dir)
    (out / "controller_oracle_contrast.json").write_text(json.dumps(contrast, indent=2) + "\n")
    summary = {"protocol_id": protocol["protocol_id"], "status": "offline_physical_diagnostic_complete",
        "cases": cases, "controller_comparison": contrast, "diagnostic_windows": len(rows),
        "diagnostic_LP_solves": sum(r["LP_solved"] for r in rows),
        "diagnostic_physical_witness_replays": sum(r["witness_replayed"] for r in rows),
        "new_data_requests": 0, "algorithm_optimizer_calls": 0, "terminal_verifier_calls": 0,
        "confirmation_year_access": False,
        "scope": "perfect-information physical feasibility; replayed witnesses are not causal policies or source-effect evidence"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("diagnostic_windows", "diagnostic_LP_solves", "diagnostic_physical_witness_replays")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
