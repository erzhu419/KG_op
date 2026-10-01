#!/usr/bin/env python3
"""Run the four frozen April sequential service pilots from cached receipts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_public_battery_dispatch import instant
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_causal_service import receipt_marks, simulate_service

PROTOCOL = ROOT / "performance/manifests/public_battery_causal_service_pilot_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_causal_service_pilot_v1_20261001"


def inspect(protocol, out):
    roster = protocol["roster"]
    original = json.loads((ROOT / protocol["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": original["shared_gross_power_MW"],
             **{k: original[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    base = json.loads((ROOT / protocol["retained_controller_protocol"]).read_text())
    case = next(c for c in base["development"] if c["start"][:10] == roster["period"])
    start = instant(case["start"])
    units = [cached_unit(base, case, unit) for unit in (0, 1)]
    workload = protocol["request_workload"]
    clocks = range(workload["clock_start_minute"], workload["clock_stop_exclusive_minute"], workload["clock_minutes"])
    marks = receipt_marks(units, start, clocks)
    horizon = roster["horizon_hours"] * 60
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    runs, hours, nominations = [], [], []
    for profile in roster["profiles"]:
        for rule in roster["dispatch_rules"]:
            summary, records, decisions = simulate_service(
                roster["initial_SOC_MWh"], roster["initial_locked_PN_MW"], profile["target_fractions"],
                marks, rule, horizon, workload["service_capacity_MW"], asset, tolerance)
            # A broken gross-energy ledger would invalidate interpretation.
            if max(abs(v) for v in summary["energy_balance_error_MWh"]) > tolerance:
                raise ValueError("sequential service execution does not conserve unit energy")
            identity = {"period": roster["period"], "window_start_minute": 0,
                        "profile_id": profile["id"], "dispatch_rule": rule}
            runs.append({**identity, **summary})
            hours.extend({**identity, **row} for row in records)
            nominations.extend({**identity, **row} for row in decisions)
            print(json.dumps({**identity, **{k: summary[k] for k in (
                "completed_minutes", "physical_complete", "window_success", "successful_service_hours",
                "capacity_shortfall_hours", "request_shortfall_hours", "first_service_failure",
                "first_physical_failure")}}), flush=True)
    if len(runs) != roster["planned_controller_window_evaluations"]:
        raise ValueError("executed trial count differs from the frozen four-run roster")
    summary = {"protocol_id": protocol["protocol_id"], "status": "causal_service_pilot_complete",
               "controller_window_evaluations": len(runs), "physically_completed_windows": sum(r["physical_complete"] for r in runs),
               "service_successful_windows": sum(r["window_success"] for r in runs), "runs": runs,
               "request_clocks": len(marks), "request_activation_clocks": sum(m["direction"] is not None for m in marks),
               "hourly_service_records": len(hours), "nomination_records": len(nominations),
               "envelope_path_evaluations": sum(r["envelope_path_evaluations"] for r in runs),
               "minute_physics_evaluations": sum(r["minute_physics_evaluations"] for r in runs),
               "source_comparison_gate": protocol["source_comparison_gate"], **protocol["accounting_constraints"],
               "library_matrix_launch": False, "retained_failure_probability": protocol["retained_failure_probability"],
               "decision": "diagnose_first_service_or_physical_failure_before_any_larger_run"
                           if not all(r["window_success"] for r in runs) else "register_broader_development_population",
               "limitations": protocol["limitations"],
               "delivery_accounting": "completed windows report all delivered shortfall; interrupted windows report executed-prefix shortfall and unassessed hours, without imputing future delivery"}
    initial_run = out / "numerical_repair_initial_run/summary.json"
    initial_counts = (json.loads(initial_run.read_text()) if initial_run.exists()
                      else {"controller_window_evaluations": 0, "envelope_path_evaluations": 0, "minute_physics_evaluations": 0})
    summary["implementation_repair_controller_window_replays"] = initial_counts["controller_window_evaluations"]
    for key in ("controller_window_evaluations", "envelope_path_evaluations", "minute_physics_evaluations"):
        summary["total_" + key] = summary[key] + initial_counts[key]
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary.json", summary), ("request_marks.json", marks),
                       ("hourly_service.json", hours), ("nominations.json", nominations)):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("runs", "limitations")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
