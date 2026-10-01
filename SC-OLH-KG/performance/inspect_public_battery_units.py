#!/usr/bin/env python3
"""Frozen unit-controller development diagnostic using cached 2024 inputs only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_controller import calibration, load_signal, policy_pool
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_units import simulate_units, unit_planning_terms

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_controller_development_v3_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_controller_development_v3_20260930"


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    ids, nodes, points = policy_pool(protocol)
    fit, profiles, windows, cases, traces = calibration(protocol), [], [], [], []
    horizon = protocol["windows"]["hours"] * 60
    tolerance = protocol["failure"]["numerical_energy_tolerance_MWh"]
    capacity = np.array(protocol["asset"]["unit_energy_MWh"])
    for case in protocol["development"]:
        label = case["start"][:10]
        start, stop = instant(case["start"]), instant(case["stop"])
        units = [cached_unit(protocol, case, i) for i in (0, 1)]
        workload = site_workload(units, start, stop)
        stress, prices = load_signal(protocol, case, fit)
        targets = np.array([np.interp(stress, nodes, point) for point in points])
        starts = np.arange(0, len(workload["active"]) - horizon + 1,
                           protocol["windows"]["start_stride_hours"] * 60)
        terms = unit_planning_terms(receipt_plans(units, start, len(workload["active"])), protocol["asset"])
        result = simulate_units(workload, prices, targets, starts, horizon, protocol["asset"], terms,
                                protocol["control"]["initial_soc_fraction"], tolerance)
        if np.max(np.abs(result["energy_balance_error_MWh"])) > tolerance:
            raise ValueError("independent unit gross flows do not conserve energy")
        if np.min(result["minimum_unit_SOC_MWh"]) < -tolerance or np.any(result["maximum_unit_SOC_MWh"] > capacity + tolerance):
            raise ValueError("accepted trajectory exceeds a unit energy bound")
        success = result["success"]
        for p, (name, family, inside) in enumerate(ids):
            fraction = float(np.mean(success[p]))
            completed = result["cost_GBP"][p, success[p]]
            profiles.append({"period": label, "profile_id": name, "family": family, "in_library": inside,
                "window_success_fraction": fraction,
                "empirically_feasible": fraction >= 1 - protocol["failure"]["chance_failure_probability"],
                "all_windows_completed": bool(np.all(success[p])),
                "mean_completed_window_cost_GBP": float(np.mean(completed)) if len(completed) else None,
                "power_failure_windows": int(result["power_failure"][p].sum()),
                "unit_1_inventory_failure_windows": int(result["unit_inventory_failure"][p, :, 0].sum()),
                "unit_2_inventory_failure_windows": int(result["unit_inventory_failure"][p, :, 1].sum()),
                "BOA_inventory_failure_windows": int(result["BOA_inventory_failure"][p].sum()),
                "baseline_inventory_failure_windows": int(result["baseline_inventory_failure"][p].sum())})
            for w, minute in enumerate(starts):
                row = {"period": label, "profile_id": name, "window_start_minute": int(minute),
                       "success": bool(success[p, w]), "first_failure_minute": int(result["first_failure_minute"][p, w]),
                       "power_failure": bool(result["power_failure"][p, w]),
                       "unit_1_inventory_failure": bool(result["unit_inventory_failure"][p, w, 0]),
                       "unit_2_inventory_failure": bool(result["unit_inventory_failure"][p, w, 1]),
                       "BOA_inventory_failure": bool(result["BOA_inventory_failure"][p, w]),
                       "baseline_inventory_failure": bool(result["baseline_inventory_failure"][p, w])}
                for key in ("cost_GBP", "metered_cash_GBP", "inventory_adjustment_GBP", "final_SOC_MWh", "gross_import_MWh", "gross_export_MWh"):
                    row[key] = float(result[key][p, w]) if success[p, w] else None
                for u in (0, 1):
                    row[f"unit_{u + 1}_final_SOC_MWh"] = float(result["unit_final_SOC_MWh"][p, w, u]) if success[p, w] else None
                windows.append(row)
        library_indices = [i for i, (_, _, inside) in enumerate(ids) if inside]
        library = success[library_indices]
        gross = np.maximum(np.abs(workload["first_MW"]).sum(axis=-1), np.abs(workload["last_MW"]).sum(axis=-1))
        peaks = np.array([gross[s:s + horizon].max() for s in starts])
        report = {"period": label, "windows": len(starts), "unit_segments": [len(s) for s in units],
            "library_feasible_profiles": sum(p["empirically_feasible"] for p in profiles[-len(ids):] if p["in_library"]),
            "library_failed_windows": int(np.sum(~library)),
            "library_all_windows_completed_profiles": int(np.all(library, axis=1).sum()),
            "library_successful_windows_union": int(np.any(library, axis=0).sum()),
            "extra_reference_feasible_profiles": sum(p["empirically_feasible"] for p in profiles[-len(ids):] if not p["in_library"]),
            "peak_forced_gross_MW": float(peaks.max()),
            "power_only_window_success_upper_bound": float(np.mean(peaks <= protocol["asset"]["power_MW"])),
            "library_first_failure_causes": {
                "power_failure": int(result["power_failure"][library_indices].sum()),
                "BOA_inventory_failure": int(result["BOA_inventory_failure"][library_indices].sum()),
                "baseline_inventory_failure": int(result["baseline_inventory_failure"][library_indices].sum()),
                "unit_1_inventory_failure": int(result["unit_inventory_failure"][library_indices, :, 0].sum()),
                "unit_2_inventory_failure": int(result["unit_inventory_failure"][library_indices, :, 1].sum())},
            "library_first_hour_failure_windows": int(np.sum((result["first_failure_minute"][library_indices] >= 0)
                & (result["first_failure_minute"][library_indices] < 60))),
            "max_unit_energy_balance_error_MWh": float(np.max(np.abs(result["energy_balance_error_MWh"])))}
        cases.append(report)
        traces.append({"period": label, "first_policy_first_window": result["nomination_trace"]})
        print(json.dumps(report), flush=True)
    library_ids = [name for name, _, inside in ids if inside]
    common = [name for name in library_ids if all(p["empirically_feasible"] for p in profiles if p["profile_id"] == name)]
    strict_common = [name for name in library_ids if all(p["all_windows_completed"] for p in profiles if p["profile_id"] == name)]
    has_unsafe = any(not p["empirically_feasible"] for p in profiles if p["in_library"])
    summary = {"protocol_id": protocol["protocol_id"], "status": "development_complete", "normalization": fit,
               "cases": cases, "library_profiles_feasible_in_all_periods": common,
               "library_profiles_completing_all_windows_in_all_periods": strict_common,
               "decision": "proceed_to_source_archive_pilot" if common and has_unsafe else "hold_full_comparison",
               "diagnostic_window_evaluations": len(windows), "new_API_requests": 0, "new_API_response_bytes": 0,
               "optimizer_calls": 0, "terminal_verifier_calls": 0, "confirmation_year_access": False,
               "limitations": protocol["model_assumptions"],
               "cost_scope": protocol["cost"]["scope"]}
    write_csv(out / "profiles.csv", profiles)
    write_csv(out / "windows.csv", windows)
    failure_diagnostic = {"cases": [{"period": c["period"], "library_failed_windows": c["library_failed_windows"],
        "first_failure_causes": c["library_first_failure_causes"],
        "first_hour_failures": c["library_first_hour_failure_windows"]} for c in cases],
        "scope": "first failing piece; cause counts can overlap; no later outcomes assigned to stopped windows"}
    (out / "failure_diagnostic.json").write_text(json.dumps(failure_diagnostic, indent=2) + "\n")
    (out / "nomination_traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
