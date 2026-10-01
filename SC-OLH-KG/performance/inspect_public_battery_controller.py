#!/usr/bin/env python3
"""Frozen development-only feasibility of delayed battery nominations."""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import requests

from core.profile_atlas import regular_profile_nodes
from performance.inspect_independent_energy_data import instant, retrieve, write_csv
from performance.inspect_public_battery_dispatch import decode
from problems.public_battery_dispatch import minute_workload, simulate
from problems.randomized_profiles import generate_structural_profile_library

PROTOCOL = ROOT / "performance/manifests/public_battery_controller_development_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_controller_development_v1_20260930"


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def calibration(protocol):
    rows = read_csv(ROOT / protocol["calibration_csv"])
    times = [instant(r["start_time_utc"]) for r in rows]
    first, last = map(instant, protocol["signal_calibration_period_utc"])
    mask = np.array([first <= t < last for t in times])
    forecast = np.array([float(r["forecast_MW"]) for r in rows])
    lower, upper = np.quantile(forecast[mask], [.05, .95])
    ramp = np.abs(np.diff(forecast, prepend=forecast[0]))
    return {"forecast_q05_MW": float(lower), "forecast_q95_MW": float(upper),
            "absolute_ramp_q95_MW": float(np.quantile(ramp[mask], .95)),
            "fit_half_hours": int(mask.sum())}


def policy_pool(protocol):
    spec = protocol["library"]
    library = generate_structural_profile_library(spec["profiles"], dimension=spec["nodes"],
        seed=spec["seed"], maximum_frequency=spec["maximum_frequency"])
    nodes = regular_profile_nodes(protocol["policy_dimension"])
    levels = protocol["policy_integer_levels"]
    points = [np.rint(np.interp(nodes, p.nodes, p.values) * levels) / levels for p in library]
    ids = [(p.profile_id, p.family, True) for p in library]
    for value in protocol["reference_profiles"]:
        ids.append((f"constant_{value:.2f}", "extra_reference", False))
        points.append(np.full(len(nodes), value))
    return ids, nodes, np.array(points)


def load_workload(session, protocol, case, out):
    start, stop = instant(case["start"]), instant(case["stop"])
    left = start - timedelta(days=protocol["dispatch_prefix_days"])
    rows, logs = [], []
    while left < stop:
        right = min(stop, left + timedelta(days=protocol["request_chunk_days"]))
        path = out / "samples" / f"{left.date()}_{right.date()}_acceptances.json"
        if left == instant("2024-04-14T00:00:00Z"):
            path = ROOT / "paper_artifacts/public_battery_application_spec_20260930/pillswood_acceptances_20240414_15.json"
        piece, meta = retrieve(session, protocol["api_base"], protocol["acceptances_endpoint"],
            {"bmUnit": protocol["asset"]["workload_bm_unit"], "from": left.isoformat(),
             "to": right.isoformat(), "format": "json"}, path)
        # Inclusive API query endpoints are trimmed before joining chunks.
        rows.extend(r for r in piece if left <= instant(r["timeFrom"]) < right)
        logs.append({**meta, "returned_rows": len(piece),
                     "reused_previous_stage": left == instant("2024-04-14T00:00:00Z")})
        left = right
    segments = decode(rows, protocol["asset"]["workload_bm_unit"])
    for s in segments:
        if s.start >= s.stop or s.received > s.start:
            raise ValueError("ordinary BOA segment is empty or precedes instruction receipt")
        if any(t.second or t.microsecond for t in (s.start, s.stop, s.received)):
            raise ValueError("minute-step pilot requires ordinary whole-minute BOA times")
    relevant = [s for s in segments if s.start < stop and s.stop > start]
    return minute_workload(relevant, start, stop, protocol["asset"]["power_MW"]), logs, relevant


def load_signal(protocol, case, fit):
    rows = read_csv(ROOT / case["csv"])
    forecast = np.array([float(r["forecast_MW"]) for r in rows])
    level = np.clip((forecast - fit["forecast_q05_MW"]) /
                    (fit["forecast_q95_MW"] - fit["forecast_q05_MW"]), 0, 1)
    ramp = np.clip(np.abs(np.diff(forecast, prepend=forecast[0])) / fit["absolute_ramp_q95_MW"], 0, 1)
    stress = np.maximum(level, ramp)
    start, stop = instant(case["start"]), instant(case["stop"])
    selected = [i for i, r in enumerate(rows) if start <= instant(r["start_time_utc"]) < stop]
    for offset, i in enumerate(selected):
        delivery = start + timedelta(minutes=30 * offset)
        if instant(rows[i]["start_time_utc"]) != delivery:
            raise ValueError("development forecast/price CSV has a missing settlement interval")
        decision = delivery - timedelta(minutes=protocol["control"]["nomination_lead_minutes"])
        if max(instant(rows[k]["forecast_publish_time_utc"]) for k in (i - 1, i)) > decision:
            raise ValueError("forecast signal was unavailable at nomination time")
    if len(selected) != int((stop - start).total_seconds() / 1800):
        raise ValueError("development CSV does not cover the full frozen period")
    return stress[selected], np.array([float(rows[i]["price_GBP_per_MWh"]) for i in selected])


def inspect(protocol, out):
    (out / "samples").mkdir(parents=True, exist_ok=True)
    session, fit = requests.Session(), calibration(protocol)
    ids, nodes, points = policy_pool(protocol)
    profiles, window_rows, cases = [], [], []
    horizon = protocol["windows"]["hours"] * 60
    tolerance = protocol["failure"]["numerical_energy_tolerance_MWh"]
    for case in protocol["development"]:
        label = case["start"][:10]
        workload, logs, segments = load_workload(session, protocol, case, out)
        stress, prices = load_signal(protocol, case, fit)
        targets = np.array([np.interp(stress, nodes, point) for point in points])
        starts = np.arange(0, len(workload["active"]) - horizon + 1,
                           protocol["windows"]["start_stride_hours"] * 60)
        results = simulate(workload, prices, targets, starts, horizon, protocol["asset"],
                           protocol["control"]["initial_soc_fraction"])
        success = results["BOA_undelivered_MWh"] + results["baseline_undelivered_MWh"] <= tolerance
        if np.max(np.abs(results["energy_balance_error_MWh"])) > tolerance:
            raise ValueError("minute engine did not conserve battery energy")
        if np.min(results["minimum_SOC_MWh"]) < -tolerance or np.max(results["maximum_SOC_MWh"]) > protocol["asset"]["energy_MWh"] + tolerance:
            raise ValueError("minute engine exceeded the battery energy bounds")
        peaks = np.array([np.max(workload["peak_MW"][s:s + horizon]) for s in starts])
        for p, (profile_id, family, in_library) in enumerate(ids):
            fraction = float(np.mean(success[p]))
            profiles.append({"period": label, "profile_id": profile_id, "family": family,
                "in_library": in_library, "window_success_fraction": fraction,
                "empirically_feasible": fraction >= 1 - protocol["failure"]["chance_failure_probability"],
                "mean_cost_GBP": float(np.mean(results["cost_GBP"][p])),
                "max_BOA_undelivered_MWh": float(np.max(results["BOA_undelivered_MWh"][p])),
                "max_baseline_undelivered_MWh": float(np.max(results["baseline_undelivered_MWh"][p]))})
            for w, minute in enumerate(starts):
                window_rows.append({"period": label, "profile_id": profile_id, "window_start_minute": int(minute),
                    "success": bool(success[p, w]), **{key: float(results[key][p, w]) for key in
                    ("cost_GBP", "metered_cash_GBP", "inventory_adjustment_GBP", "final_SOC_MWh",
                     "gross_import_MWh", "gross_export_MWh", "BOA_undelivered_MWh", "baseline_undelivered_MWh")}})
        case_profiles = profiles[-len(ids):]
        report = {"period": label, "windows": len(starts), "instruction_segments": len(segments),
            "acceptances": len({s.number for s in segments}),
            "peak_requested_MW": float(np.max(peaks)),
            "windows_with_request_above_reference_power": int(np.sum(peaks > protocol["asset"]["power_MW"])),
            "power_only_window_success_upper_bound": float(np.mean(peaks <= protocol["asset"]["power_MW"])),
            "library_feasible_profiles": sum(p["empirically_feasible"] for p in case_profiles if p["in_library"]),
            "extra_reference_feasible_profiles": sum(p["empirically_feasible"] for p in case_profiles if not p["in_library"]),
            "forecast_stress_unique_values": len(np.unique(stress)),
            "forecast_stress_endpoint_fraction": float(np.mean((stress == 0) | (stress == 1))),
            "max_energy_balance_error_MWh": float(np.max(np.abs(results["energy_balance_error_MWh"]))),
            "request_logs": logs}
        cases.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != "request_logs"}), flush=True)
    library_ids = [i for i, _, inside in ids if inside]
    common_feasible = [i for i in library_ids if all(p["empirically_feasible"] for p in profiles if p["profile_id"] == i)]
    all_feasible = all(p["empirically_feasible"] for p in profiles if p["in_library"])
    decision = "proceed_to_source_archive_pilot" if common_feasible and not all_feasible else "hold_full_comparison"
    summary = {"protocol_id": protocol["protocol_id"], "status": "development_preflight_complete",
        "normalization": fit, "cases": cases, "profiles": profiles,
        "library_profiles_feasible_in_all_periods": common_feasible,
        "decision": decision, "diagnostic_window_evaluations": len(window_rows),
        "valid_API_response_bytes": sum(log["response_bytes"] for c in cases for log in c["request_logs"]),
        "new_API_response_bytes": sum(log["response_bytes"] for c in cases for log in c["request_logs"]
                                     if not log["reused_previous_stage"]),
        "optimizer_calls": 0, "terminal_verifier_calls": 0, "confirmation_year_access": False,
        "scope": "overlapping finite development populations; no source benefit or independent certificate estimate"}
    write_csv(out / "profiles.csv", profiles)
    write_csv(out / "windows.csv", window_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("decision", "diagnostic_window_evaluations", "valid_API_response_bytes")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
