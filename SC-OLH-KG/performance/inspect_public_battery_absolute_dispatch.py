#!/usr/bin/env python3
"""Run the frozen four absolute-dispatch fixtures, then four April pilots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from problems.public_battery_absolute_dispatch import declare_absolute
from problems.public_battery_causal_service import simulate_service
from problems.public_battery_pending_availability import evaluate_pending

PROTOCOL = ROOT / "performance/manifests/public_battery_absolute_dispatch_pilot_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_absolute_dispatch_pilot_v1_20261001"


def write(out, name, data):
    (out / name).write_text(json.dumps(data, indent=2) + "\n")


def inspect(protocol, out):
    original = json.loads((ROOT / protocol["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": original["shared_gross_power_MW"],
             **{k: original[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    previous = json.loads((ROOT / protocol["basis_previous_pilot"]).read_text())
    controller = json.loads((ROOT / previous["retained_controller_protocol"]).read_text())
    tolerance = controller["failure"]["numerical_energy_tolerance_MWh"]
    basis = ROOT / protocol["basis_previous_outputs"]
    marks = json.loads((basis / "request_marks.json").read_text())
    previous_summary = json.loads((basis / "summary.json").read_text())
    prior_hours = json.loads((basis / "hourly_service.json").read_text())
    peaks = protocol["contract"]["absolute_request_peak_MW"]
    capacity = np.abs(peaks["export"])
    roster = protocol["sequential_pilot"]
    horizon = roster["horizon_hours"] * 60
    if ([m["minute"] for m in marks] != list(range(0, horizon, 60))
            or sum(m["direction"] is not None for m in marks) != previous_summary["request_activation_clocks"]):
        raise ValueError("cached request clocks differ from the retained workload")
    out.mkdir(parents=True, exist_ok=True)
    fixtures, traces, fixture_paths = [], [], 0
    for profile in protocol["component_preflight"]["profiles"]:
        state, = [r for r in prior_hours if r["profile_id"] == profile
                  and r["dispatch_rule"] == "conservative_admission" and r["minute"] == 60]
        for direction in protocol["component_preflight"]["directions"]:
            declaration = declare_absolute(state["initial_SOC_MWh"], state["locked_PN_MW"],
                                           peaks[direction], asset, tolerance)
            result = evaluate_pending(declaration, peaks[direction], capacity, asset, tolerance)
            fixture_paths += declaration["envelope_path_evaluations"]
            first, last = result["power_first_MW"], result["power_last_MW"]
            balance = (np.array(result["physics"]["final_SOC_MWh"]) - state["initial_SOC_MWh"]
                       - asset["charge_efficiency"] * result["gross_import_MWh"]
                       + result["gross_export_MWh"] / asset["discharge_efficiency"])
            correct = (result["service_success"] and max(abs(balance)) <= tolerance
                       and np.allclose(result["admitted_peak_MW"], peaks[direction], atol=tolerance, rtol=0)
                       and np.allclose(first[0], state["locked_PN_MW"][0], atol=tolerance, rtol=0)
                       and np.allclose(last[31], state["locked_PN_MW"][1], atol=tolerance, rtol=0)
                       and np.allclose(first[32:], state["locked_PN_MW"][1], atol=tolerance, rtol=0))
            identity = {"profile_id": profile, "direction": direction, "basis_minute": 60}
            fixtures.append({**identity, "component_pass": bool(correct),
                             "initial_SOC_MWh": state["initial_SOC_MWh"], "locked_PN_MW": state["locked_PN_MW"],
                             "available_absolute_peak_magnitude_MW": declaration["available_capacity_MW"].tolist(),
                             "energy_balance_error_MWh": balance.tolist(),
                             **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
                                if k not in ("physics", "SOC_trace_MWh", "power_first_MW", "power_last_MW")},
                             **result["physics"]})
            traces.append({**identity, **{k: result[k].tolist() for k in
                           ("SOC_trace_MWh", "power_first_MW", "power_last_MW")}})
    component = {"protocol_id": protocol["protocol_id"], "fixture_replays": len(fixtures),
                 "passed_fixtures": sum(f["component_pass"] for f in fixtures),
                 "envelope_path_evaluations": fixture_paths, "final_path_evaluations": len(fixtures),
                 "fixtures": fixtures}
    write(out, "component_summary.json", component)
    write(out, "component_traces.json", traces)
    if (len(fixtures) != protocol["component_preflight"]["planned_fixture_replays"]
            or not all(f["component_pass"] for f in fixtures)):
        raise ValueError("absolute-dispatch component discrepancy; do not run the pilots")
    print(json.dumps({k: v for k, v in component.items() if k != "fixtures"}), flush=True)

    runs, hours, nominations = [], [], []
    for profile in roster["profiles"]:
        for rule in roster["dispatch_rules"]:
            result, records, decisions = simulate_service(
                roster["initial_SOC_MWh"], roster["initial_locked_PN_MW"], profile["target_fractions"],
                marks, rule, horizon, capacity, asset, tolerance, absolute_request_peaks_MW=peaks)
            if max(abs(v) for v in result["energy_balance_error_MWh"]) > tolerance:
                raise ValueError("absolute-dispatch rollout does not conserve unit energy")
            identity = {"period": roster["period"], "window_start_minute": 0,
                        "profile_id": profile["id"], "dispatch_rule": rule}
            runs.append({**identity, **result})
            hours.extend({**identity, **r} for r in records)
            nominations.extend({**identity, **r} for r in decisions)
            print(json.dumps({**identity, **{k: result[k] for k in (
                "completed_minutes", "physical_complete", "window_success", "planned_requests",
                "fulfilled_requests", "unfulfilled_observed_requests", "unobserved_requests",
                "first_dispatch_failure", "first_physical_failure")}}), flush=True)
    if len(runs) != roster["planned_controller_window_evaluations"]:
        raise ValueError("executed trial count differs from the frozen four-run roster")
    summary = {"protocol_id": protocol["protocol_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "absolute_dispatch_pilot_complete", "component_fixture_replays": len(fixtures),
               "component_passes": component["passed_fixtures"], "controller_window_evaluations": len(runs),
               "physically_completed_windows": sum(r["physical_complete"] for r in runs),
               "service_successful_windows": sum(r["window_success"] for r in runs), "runs": runs,
               "request_marks_origin": protocol["basis_previous_outputs"] + "/request_marks.json",
               "request_clocks": len(marks), "request_activation_clocks": sum(m["direction"] is not None for m in marks),
               "hourly_dispatch_records": len(hours), "nomination_records": len(nominations),
               "pilot_envelope_path_evaluations": sum(r["envelope_path_evaluations"] for r in runs),
               "total_envelope_path_evaluations": fixture_paths + sum(r["envelope_path_evaluations"] for r in runs),
               "component_final_path_evaluations": len(fixtures),
               "minute_physics_evaluations": sum(r["minute_physics_evaluations"] for r in runs),
               "source_comparison_gate": protocol["source_comparison_gate"], **protocol["accounting_constraints"],
               "library_matrix_launch": False, "retained_failure_probability": .05,
               "decision": "register_broader_development_population" if any(r["window_success"] for r in runs)
                           else "diagnose_first_dispatch_failure_before_any_larger_run",
               "delivery_accounting": "all 95 calls remain in each planned denominator; interrupted runs report executed-prefix error and unobserved calls; inactive clocks create no delivered-request observation",
               "limitations": protocol["limitations"]}
    for name, data in (("summary.json", summary), ("request_marks.json", marks),
                       ("hourly_dispatch.json", hours), ("nominations.json", nominations)):
        write(out, name, data)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("runs", "limitations")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
