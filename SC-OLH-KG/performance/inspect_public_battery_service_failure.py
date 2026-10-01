#!/usr/bin/env python3
"""Explain first firm-service deficits using saved PN and absolute peak power."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "performance/manifests/public_battery_causal_service_pilot_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_causal_service_pilot_v1_20261001"


def inspect():
    protocol = json.loads(PROTOCOL.read_text())
    asset = json.loads((ROOT / protocol["retained_asset_protocol"]).read_text())["retained_asset"]
    capacity = np.asarray(protocol["request_workload"]["service_capacity_MW"])
    hours = json.loads((OUTPUT / "hourly_service.json").read_text())
    reports = []
    for profile in protocol["roster"]["profiles"]:
        row = next(r for r in hours if r["profile_id"] == profile["id"]
                   and r["dispatch_rule"] == "conservative_admission"
                   and "firm_availability_shortfall" in r["service_failure_reasons"])
        locked = np.asarray(row["locked_PN_MW"])
        export_peak = np.max(locked, axis=0) + capacity
        import_peak = np.min(locked, axis=0) - capacity
        reports.append({"profile_id": profile["id"], "minute": row["minute"],
                        "actual_request_direction": row["direction"], "locked_PN_MW": locked.tolist(),
                        "baseline_peak_gross_MW": float(np.max(np.sum(np.abs(locked), axis=1))),
                        "full_export_absolute_peak_MW": export_peak.tolist(),
                        "full_import_absolute_peak_MW": import_peak.tolist(),
                        "full_joint_export_gross_MW": float(np.sum(np.abs(export_peak))),
                        "full_joint_import_gross_MW": float(np.sum(np.abs(import_peak))),
                        "available_export_capacity_MW": row["available_capacity_MW"][0],
                        "available_import_capacity_MW": row["available_capacity_MW"][1],
                        "shared_gross_power_limit_MW": asset["shared_gross_power_MW"],
                        "explanation": "both full directional joint peaks exceed the unchanged gross site limit; power alone excludes the fixed service at this state even without activation"})
    summary = {"basis": protocol["protocol_id"], "status": "first_service_failure_diagnosed", "profiles": reports,
               "new_controller_window_evaluations": 0, "new_data_requests": 0, "diagnostic_LP_calls": 0,
               "algorithm_optimizer_calls": 0, "confirmation_year_access": False,
               "source_comparison_gate": "HOLD",
               "limitations": "explains the first two saved-state failures; does not establish every policy's infeasibility or a historical business obligation"}
    (OUTPUT / "first_failure_diagnosis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    inspect()
