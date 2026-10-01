#!/usr/bin/env python3
"""Four frozen nomination probes followed by four causal receipt-forecast pilots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_public_battery_dispatch import instant
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_causal_service import accepted_plan, receipt_marks, simulate_service
from problems.public_battery_receipt_forecast import nominate_receipt_forecast
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_receipt_forecast_pilot_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_receipt_forecast_pilot_v1_20261001"


def write(out, name, data):
    (out / name).write_text(json.dumps(data, indent=2) + "\n")


def inspect(protocol, out):
    base = json.loads((ROOT / protocol["basis_protocol"]).read_text())
    original = json.loads((ROOT / base["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": original["shared_gross_power_MW"],
             **{k: original[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    old_protocol = json.loads((ROOT / base["basis_previous_pilot"]).read_text())
    controller = json.loads((ROOT / old_protocol["retained_controller_protocol"]).read_text())
    tolerance = controller["failure"]["numerical_energy_tolerance_MWh"]
    roster = protocol["sequential_pilot"]
    horizon = roster["horizon_hours"] * 60
    basis = ROOT / protocol["basis_outputs"]
    marks = json.loads((basis / "request_marks.json").read_text())
    old_results = json.loads((basis / "summary.json").read_text())["runs"]
    case = next(c for c in controller["development"] if c["start"][:10] == roster["period"])
    units = [cached_unit(controller, case, u) for u in (0, 1)]
    start = instant(case["start"])
    if receipt_marks(units, start, range(0, horizon, 60)) != marks:
        raise ValueError("cue ledger differs from the retained request workload")
    cues = receipt_marks(units, start, range(0, horizon - 60, 30))
    timing = json.loads((ROOT / protocol["timing_protocol"]).read_text())
    audit = json.loads((ROOT / protocol["information_audit"]).read_text())
    observed = next(c for c in audit["observations"] if c["minute"] == 330)
    if cues[11] != observed:
        raise ValueError("nomination cue does not reproduce the frozen receipt-prefix observation")
    peaks = base["contract"]["absolute_request_peak_MW"]
    out.mkdir(parents=True, exist_ok=True)
    write(out, "nomination_cues.json", cues)
    probes, probe_extra_paths = [], 0
    for profile in timing["profiles"]:
        soc = np.array(profile["initial_SOC_at_minute_330_MWh"])
        current, following = np.array(profile["initial_locked_PN_at_minute_330_MW"])
        target = np.array(profile["target_fractions"]) * asset["unit_energy_MWh"]
        for cue in (observed, {"minute": 330, "direction": None, "source": None}):
            command, info = nominate_receipt_forecast(soc, current, following, target, accepted_plan(None, 330),
                                                     330, cue, peaks, asset, tolerance)
            physics = info.get("nominated_reference_physics")
            if physics is None:
                power = np.repeat(np.vstack((current, following, command)), 30, axis=0)
                physics = trajectory(power, power, soc, 90, asset)
                probe_extra_paths += 1
            correct = (max(physics["unit_bound_violation_MWh"]) <= tolerance
                       and physics["power_bound_violation_MW"] <= tolerance
                       and np.abs(command).sum() <= asset["power_MW"] + tolerance
                       and (cue["direction"] is not None or np.allclose(
                           command, profile["retained_nomination_for_minute_390_MW"], atol=tolerance, rtol=0)))
            probes.append({"profile_id": profile["id"], "probe_pass": bool(correct), "minute": 330,
                           "initial_SOC_MWh": soc.tolist(), "locked_PN_MW": [current.tolist(), following.tolist()],
                           "nomination_MW": command.tolist(), "known_reference_physics": physics, **info})
    component = {"protocol_id": protocol["protocol_id"], "nomination_probes": len(probes),
                 "passed_probes": sum(p["probe_pass"] for p in probes), "probes": probes,
                 "nomination_safety_path_evaluations": sum(p["reference_safety_path_evaluations"] for p in probes),
                 "additional_reference_path_evaluations": probe_extra_paths,
                 "analytic_nomination_calls": sum(p["analytic_nomination_calls"] for p in probes)}
    write(out, "component_summary.json", component)
    if (len(probes) != protocol["component_preflight"]["planned_nomination_probes"]
            or not all(p["probe_pass"] for p in probes)):
        raise ValueError("nomination probe discrepancy; do not interpret or run the pilots")
    print(json.dumps({k: v for k, v in component.items() if k != "probes"}), flush=True)
    runs, hours, nominations, comparisons = [], [], [], []
    for profile in roster["profiles"]:
        for rule in roster["dispatch_rules"]:
            result, records, decisions = simulate_service(
                roster["initial_SOC_MWh"], roster["initial_locked_PN_MW"], profile["target_fractions"],
                marks, rule, horizon, np.abs(peaks["export"]), asset, tolerance,
                absolute_request_peaks_MW=peaks, nomination_cues=cues)
            identity = {"profile_id": profile["id"], "dispatch_rule": rule, "period": roster["period"],
                        "window_start_minute": 0}
            runs.append({**identity, **result})
            hours.extend({**identity, **r} for r in records)
            nominations.extend({**identity, **d} for d in decisions)
            for name, data in (("runs.json", runs), ("hourly_dispatch.json", hours), ("nominations.json", nominations)):
                write(out, name, data)
            if max(abs(v) for v in result["energy_balance_error_MWh"]) > tolerance:
                raise ValueError("receipt-forecast rollout does not conserve unit energy")
            if any(d["reference_safety_path_evaluations"] > 50 for d in decisions):
                raise ValueError("nomination safety evaluations exceed the frozen per-decision budget")
            prior = next(r for r in old_results if r["profile_id"] == profile["id"] and r["dispatch_rule"] == rule)
            comparisons.append({**identity, "ordinary_completed_minutes": prior["completed_minutes"],
                                "forecast_completed_minutes": result["completed_minutes"],
                                "ordinary_fulfilled_requests": prior["fulfilled_requests"],
                                "forecast_fulfilled_requests": result["fulfilled_requests"],
                                "ordinary_unobserved_requests": prior["unobserved_requests"],
                                "forecast_unobserved_requests": result["unobserved_requests"],
                                "ordinary_window_success": prior["window_success"],
                                "forecast_window_success": result["window_success"]})
            print(json.dumps({**identity, **{k: result[k] for k in (
                "completed_minutes", "physical_complete", "window_success", "fulfilled_requests",
                "unfulfilled_observed_requests", "unobserved_requests", "receipt_forecast_decisions",
                "nomination_known_reference_failure_decisions", "first_dispatch_failure", "first_physical_failure")}}), flush=True)
    if len(runs) != roster["planned_controller_window_evaluations"]:
        raise ValueError("controller-window count differs from the frozen four-run roster")
    summary = {"protocol_id": protocol["protocol_id"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "receipt_forecast_pilot_complete", "nomination_probes": len(probes),
               "component_passes": component["passed_probes"], "controller_window_evaluations": len(runs),
               "physically_completed_windows": sum(r["physical_complete"] for r in runs),
               "service_successful_windows": sum(r["window_success"] for r in runs), "runs": runs,
               "request_clocks": len(marks), "request_activation_clocks": sum(m["direction"] is not None for m in marks),
               "cached_nomination_cue_clocks": len(cues), "hourly_dispatch_records": len(hours),
               "nomination_records": len(nominations), "comparisons": comparisons,
               "pilot_envelope_path_evaluations": sum(r["envelope_path_evaluations"] for r in runs),
               "pilot_nomination_safety_path_evaluations": sum(r["nomination_safety_path_evaluations"] for r in runs),
               "probe_nomination_safety_path_evaluations": component["nomination_safety_path_evaluations"],
               "probe_additional_reference_path_evaluations": probe_extra_paths,
               "minute_physics_evaluations": sum(r["minute_physics_evaluations"] for r in runs),
               "analytic_nomination_calls": component["analytic_nomination_calls"] + sum(r["analytic_nomination_calls"] for r in runs),
               "source_comparison_gate": protocol["source_comparison_gate"], **protocol["accounting_constraints"],
               "library_matrix_launch": False,
               "decision": "register_broader_development" if any(r["window_success"] for r in runs)
                           else "diagnose_forecast_or_allocation_or_remaining_physical_limit",
               "unit_test_accounting": "unit_test_accounting.json (separate from pilot and probe counts)",
               "retained_failure_probability": .05, "limitations": protocol["limitations"],
               "delivery_accounting": "all 95 requests remain in each planned denominator; interrupted runs record executed-prefix error and unobserved requests; inactive clocks are not delivered requests"}
    archived = out / "execution_origin_initial_run/summary.json"
    initial_counts = json.loads(archived.read_text()) if archived.exists() else {}
    summary["implementation_repair_controller_window_replays"] = initial_counts.get("controller_window_evaluations", 0)
    summary["implementation_repair_nomination_probes"] = initial_counts.get("nomination_probes", 0)
    for key in ("controller_window_evaluations", "nomination_probes", "pilot_envelope_path_evaluations",
                "pilot_nomination_safety_path_evaluations", "probe_nomination_safety_path_evaluations",
                "probe_additional_reference_path_evaluations", "minute_physics_evaluations", "analytic_nomination_calls"):
        summary["total_" + key] = summary[key] + initial_counts.get(key, 0)
    diagnosis = out / "numerical_origin_diagnosis.json"
    summary["numerical_diagnosis_path_evaluations"] = (json.loads(diagnosis.read_text())["additional_saved_state_path_evaluations"]
                                                       if diagnosis.exists() else 0)
    write(out, "summary.json", summary)
    write(out, "request_marks.json", marks)
    write(out, "comparisons.json", comparisons)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("runs", "comparisons", "limitations")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
