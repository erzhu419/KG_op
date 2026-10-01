#!/usr/bin/env python3
"""Replay the four frozen saved-state PN timing fixtures, originals first."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from problems.public_battery_causal_service import simulate_service
from problems.public_battery_dispatch import positive_integral
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_replenishment_timing_preflight_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_replenishment_timing_preflight_v1_20261001"


def absolute_times(record, offset):
    return {**record, **{key: record[key] + offset for key in ("minute", "delivery_minute") if key in record}}


def inspect(protocol, out):
    base = json.loads((ROOT / protocol["basis_protocol"]).read_text())
    original = json.loads((ROOT / base["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": original["shared_gross_power_MW"],
             **{k: original[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    previous = json.loads((ROOT / base["basis_previous_pilot"]).read_text())
    control = json.loads((ROOT / previous["retained_controller_protocol"]).read_text())
    tolerance = control["failure"]["numerical_energy_tolerance_MWh"]
    basis = ROOT / protocol["basis_outputs"]
    prior = json.loads((basis / "summary.json").read_text())
    roster = protocol["roster"]
    start, stop = roster["start_minute"], roster["stop_exclusive_minute"]
    replay_start = start + 30
    marks = [m for m in json.loads((basis / "request_marks.json").read_text()) if replay_start <= m["minute"] < stop]
    local_marks = [{**m, "minute": m["minute"] - replay_start} for m in marks]
    peaks = base["contract"]["absolute_request_peak_MW"]
    runs, hours, nominations = [], [], []
    out.mkdir(parents=True, exist_ok=True)
    for command in roster["commands"]:
        for profile in protocol["profiles"]:
            initial = np.array(profile["initial_SOC_at_minute_330_MWh"])
            locked = np.array(profile["initial_locked_PN_at_minute_330_MW"])
            power = np.tile(locked[0], (30, 1))
            prefix = trajectory(power, power, initial, 30, asset)
            if max(prefix["unit_bound_violation_MWh"]) > tolerance or prefix["power_bound_violation_MW"] > tolerance:
                raise ValueError("saved locked prefix is unsafe; do not interpret the timing intervention")
            nominated = (profile["retained_nomination_for_minute_390_MW"] if command["id"] == "retained"
                         else command["nomination_MW"])
            result, records, decisions = simulate_service(
                prefix["final_SOC_MWh"], [locked[1].tolist(), nominated], profile["target_fractions"],
                local_marks, roster["dispatch_rule"], stop - replay_start, np.abs(peaks["export"]),
                asset, tolerance, absolute_request_peaks_MW=peaks)
            for key in ("first_dispatch_failure", "first_physical_failure"):
                if result[key] is not None:
                    result[key] = absolute_times(result[key], replay_start)
            exported = positive_integral(power, power, 1 / 60).sum(axis=0)
            imported = positive_integral(-power, -power, 1 / 60).sum(axis=0)
            result["gross_export_MWh"] = (exported + result["gross_export_MWh"]).tolist()
            result["gross_import_MWh"] = (imported + result["gross_import_MWh"]).tolist()
            balance = (np.array(result["final_or_last_safe_SOC_MWh"]) - initial
                       - asset["charge_efficiency"] * np.array(result["gross_import_MWh"])
                       + np.array(result["gross_export_MWh"]) / asset["discharge_efficiency"])
            if max(abs(balance)) > tolerance:
                raise ValueError("short replay does not conserve each unit's energy")
            result["energy_balance_error_MWh"] = balance.tolist()
            result["minimum_accepted_SOC_MWh"] = np.minimum(prefix["minimum_SOC_MWh"], result["minimum_accepted_SOC_MWh"]).tolist()
            result["maximum_accepted_SOC_MWh"] = np.maximum(prefix["maximum_SOC_MWh"], result["maximum_accepted_SOC_MWh"]).tolist()
            result["planned_minutes"] = stop - start
            result["completed_minutes"] += 30
            result["prefix_path_evaluations"] = 1
            result["nomination_decisions"] += 1  # The explicitly retained/intervened minute-330 decision.
            reproduced = None
            if command["id"] == "retained":
                expected = next(r for r in prior["runs"] if r["profile_id"] == profile["id"]
                                and r["dispatch_rule"] == roster["dispatch_rule"])["first_physical_failure"]
                actual = result["first_physical_failure"]
                reproduced = (actual is not None and actual["minute"] == expected["minute"]
                              and np.allclose(actual["last_safe_SOC_MWh"], expected["last_safe_SOC_MWh"], atol=tolerance, rtol=0)
                              and np.allclose(actual["final_SOC_MWh"], expected["final_SOC_MWh"], atol=tolerance, rtol=0))
            identity = {"profile_id": profile["id"], "command_id": command["id"]}
            runs.append({**identity, "start_minute": start, "stop_exclusive_minute": stop,
                         "initial_SOC_MWh": initial.tolist(), "initial_locked_PN_MW": locked.tolist(),
                         "intervened_nomination_MW": nominated, "retained_failure_reproduced": reproduced, **result})
            hours.extend({**identity, **absolute_times(r, replay_start)} for r in records)
            nominations.append({**identity, "minute": start, "delivery_minute": roster["intervention_delivery_minute"],
                                "SOC_MWh": initial.tolist(), "locked_PN_MW": locked.tolist(), "nomination_MW": nominated})
            nominations.extend({**identity, **absolute_times(r, replay_start)} for r in decisions)
            # Keep executed evidence even if the original branch fails reproduction.
            for name, data in (("runs.json", runs), ("hourly_dispatch.json", hours), ("nominations.json", nominations)):
                (out / name).write_text(json.dumps(data, indent=2) + "\n")
            print(json.dumps({**identity, **{k: runs[-1][k] for k in (
                "completed_minutes", "physical_complete", "window_success", "fulfilled_requests",
                "retained_failure_reproduced", "first_dispatch_failure", "first_physical_failure")}}), flush=True)
            if reproduced is False:
                raise ValueError("original branch does not reproduce its saved failure; stop before intervention")
    interventions = [r for r in runs if r["command_id"] != "retained"]
    complete = all(r["window_success"] for r in interventions)
    summary = {"protocol_id": protocol["protocol_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "replenishment_timing_preflight_complete", "short_controller_replays": len(runs),
               "retained_failures_reproduced": sum(r["retained_failure_reproduced"] is True for r in runs),
               "service_successful_interventions": sum(r["window_success"] for r in interventions), "runs": runs,
               "prefix_path_evaluations": sum(r["prefix_path_evaluations"] for r in runs),
               "envelope_path_evaluations": sum(r["envelope_path_evaluations"] for r in runs),
               "minute_physics_evaluations": sum(r["minute_physics_evaluations"] for r in runs),
               "request_marks": marks, "hourly_dispatch_records": len(hours), "nomination_records": len(nominations),
               "source_comparison_gate": protocol["source_comparison_gate"], **protocol["accounting_constraints"],
               "library_matrix_launch": False,
               "decision": "freeze_causal_nomination_rule_and_bounded_pilot" if complete else "diagnose_remaining_physical_or_commitment_failure",
               "limitations": protocol["limitations"]}
    if len(runs) != roster["planned_short_controller_replays"]:
        raise ValueError("short replay count differs from the frozen roster")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("runs", "request_marks", "limitations")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
