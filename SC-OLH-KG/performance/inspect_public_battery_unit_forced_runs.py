#!/usr/bin/env python3
"""Control-independent necessary energy bound for the cached V3 unit workloads."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_partition import cached_unit
from performance.inspect_public_battery_units import OUTPUT, PROTOCOL
from problems.public_battery_site import site_workload
from problems.public_battery_units import forced_run_bounds


def inspect():
    protocol = json.loads(PROTOCOL.read_text())
    horizon = protocol["windows"]["hours"] * 60
    tolerance = protocol["failure"]["numerical_energy_tolerance_MWh"]
    rows, cases = [], []
    for case in protocol["development"]:
        start, stop = instant(case["start"]), instant(case["stop"])
        workload = site_workload([cached_unit(protocol, case, i) for i in (0, 1)], start, stop)
        starts = np.arange(0, len(workload["active"]) - horizon + 1,
                           protocol["windows"]["start_stride_hours"] * 60)
        for minute in starts:
            bounds = forced_run_bounds(workload, int(minute), horizon, protocol["asset"])
            row = {"period": case["start"][:10], "window_start_minute": int(minute)}
            for u, bound in enumerate(bounds):
                row[f"unit_{u + 1}_required_capacity_MWh"] = bound["required_capacity_MWh"]
                row[f"unit_{u + 1}_worst_run_from_minute"] = bound["from_minute"]
                row[f"unit_{u + 1}_worst_run_to_minute"] = bound["to_minute"]
            row["forced_run_proves_energy_infeasible"] = any(b["required_capacity_MWh"] > e + tolerance
                for b, e in zip(bounds, protocol["asset"]["unit_energy_MWh"]))
            rows.append(row)
        subset = rows[-len(starts):]
        report = {"period": case["start"][:10], "windows": len(starts),
                  "forced_run_energy_infeasible_windows": sum(r["forced_run_proves_energy_infeasible"] for r in subset),
                  "maximum_forced_run_capacity_MWh": [max(r[f"unit_{u + 1}_required_capacity_MWh"] for r in subset) for u in (0, 1)]}
        cases.append(report)
        print(json.dumps(report), flush=True)
    summary = {"basis_protocol": protocol["protocol_id"], "cases": cases, "diagnostic_windows": len(rows),
               "new_API_requests": 0, "optimizer_calls": 0, "terminal_verifier_calls": 0,
               "confirmation_year_access": False,
               "scope": "necessary condition with free inventory at the start of each forced run; no causal or full physical feasibility assertion"}
    write_csv(OUTPUT / "forced_run_windows.csv", rows)
    (OUTPUT / "forced_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    controller_summary = json.loads((OUTPUT / "summary.json").read_text())
    controller_summary["forced_run_diagnostic"] = {"file": "forced_run_summary.json", "diagnostic_windows": len(rows),
        "energy_infeasible_windows": sum(c["forced_run_energy_infeasible_windows"] for c in cases), "scope": summary["scope"]}
    controller_summary["next_protocol"] = "performance/manifests/public_battery_unit_physical_oracle_v1_20260930.json"
    (OUTPUT / "summary.json").write_text(json.dumps(controller_summary, indent=2) + "\n")


if __name__ == "__main__":
    inspect()
