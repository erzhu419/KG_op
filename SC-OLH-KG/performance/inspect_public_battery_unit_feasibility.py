#!/usr/bin/env python3
"""Joint physical upper bound on the same cached V3 development windows."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_site import site_workload
from problems.public_battery_unit_feasibility import unit_physical_oracle

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_physical_oracle_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_physical_oracle_v1_20260930"
CONTROLLER_OUTPUT = ROOT / "paper_artifacts/public_battery_unit_controller_development_v3_20260930"


def controller_contrast(rows):
    with (CONTROLLER_OUTPUT / "profiles.csv").open() as handle:
        library = {r["profile_id"] for r in csv.DictReader(handle) if r["in_library"] == "True"}
    oracle = {(r["period"], r["window_start_minute"]): r["status"] for r in rows}
    any_success, library_success = set(), set()
    with (CONTROLLER_OUTPUT / "windows.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["success"] == "True":
                key = row["period"], int(row["window_start_minute"])
                any_success.add(key)
                if row["profile_id"] in library:
                    library_success.add(key)
    if any(oracle[key] != "replayed_feasible" for key in any_success):
        raise ValueError("joint physical oracle excludes a V3 successful window")
    return [{"period": period,
             "oracle_replayed_feasible_windows": sum(v == "replayed_feasible" for k, v in oracle.items() if k[0] == period),
             "any_library_policy_successful_windows": sum(k[0] == period for k in library_success),
             "oracle_feasible_windows_missed_by_entire_library": sum(v == "replayed_feasible" and k not in library_success
                  for k, v in oracle.items() if k[0] == period),
             "V3_successful_windows_included": True} for period in sorted({r["period"] for r in rows})]


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    controller = json.loads((ROOT / protocol["controller_protocol"]).read_text())
    horizon = protocol["windows"]["hours"] * 60
    rows, cases, witnesses = [], [], {}
    for case in protocol["development"]:
        label = case["start"][:10]
        start, stop = instant(case["start"]), instant(case["stop"])
        workload = site_workload([cached_unit(controller, case, i) for i in (0, 1)], start, stop)
        starts = np.arange(0, len(workload["active"]) - horizon + 1,
                           protocol["windows"]["start_stride_hours"] * 60)
        schedules = np.full((len(starts), horizon // 30, 2), np.nan)
        case_rows = []
        for w, minute in enumerate(starts):
            report, powers = unit_physical_oracle(workload, int(minute), horizon, protocol)
            case_rows.append({"period": label, "window_start_minute": int(minute), **report})
            if powers is not None:
                schedules[w] = powers
            if (w + 1) % 20 == 0:
                print(json.dumps({"period": label, "completed_windows": w + 1,
                                  "statuses": dict(Counter(r["status"] for r in case_rows))}), flush=True)
        rows.extend(case_rows)
        witnesses[label] = schedules
        statuses = Counter(r["status"] for r in case_rows)
        feasible = [r for r in case_rows if r["status"] == "replayed_feasible"]
        report = {"period": label, "windows": len(starts), "statuses": dict(statuses),
                  "LP_solves": sum(r["LP_solved"] for r in case_rows),
                  "physical_witness_replays": sum(r["witness_replayed"] for r in case_rows),
                  "unresolved_windows": statuses["solver_unresolved"] + statuses["witness_unresolved"],
                  "oracle_feasible_fraction": len(feasible) / len(starts),
                  "max_inventory_bound_violation_MWh": max((r["maximum_inventory_bound_violation_MWh"] for r in feasible), default=None),
                  "max_unit_energy_balance_error_MWh": max((max(abs(v) for v in r["energy_balance_error_MWh"]) for r in feasible), default=None),
                  "max_execution_power_violation_MW": max((r["maximum_gross_power_violation_MW"] for r in feasible), default=None)}
        cases.append(report)
        print(json.dumps(report), flush=True)
    np.savez_compressed(out / "witnesses.npz", **witnesses)
    (out / "windows.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(out / "cases.csv", [{k: json.dumps(v) if isinstance(v, dict) else v for k, v in c.items()} for c in cases])
    comparison = controller_contrast(rows)
    (out / "controller_oracle_contrast.json").write_text(json.dumps(comparison, indent=2) + "\n")
    unresolved = sum(c["unresolved_windows"] for c in cases)
    chance = controller["failure"]["chance_failure_probability"]
    sufficient = all(c["oracle_feasible_fraction"] >= 1 - chance for c in cases)
    summary = {"protocol_id": protocol["protocol_id"], "status": "joint_physical_diagnostic_complete",
               "cases": cases, "controller_comparison": comparison,
               "diagnostic_windows": len(rows), "diagnostic_LP_solves": sum(r["LP_solved"] for r in rows),
               "diagnostic_physical_witness_replays": sum(r["witness_replayed"] for r in rows),
               "scipy_version": scipy.__version__, "unresolved_windows": unresolved,
               "decision": "resolve_physical_diagnostic" if unresolved else
                   ("address_causal_control_and_candidate_coverage" if sufficient else "hold_frozen_application_model"),
               "new_data_requests": 0, "algorithm_optimizer_calls": 0, "terminal_verifier_calls": 0,
               "confirmation_year_access": False, "scope": protocol["scope"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
