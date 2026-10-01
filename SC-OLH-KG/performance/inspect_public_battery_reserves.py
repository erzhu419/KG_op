#!/usr/bin/env python3
"""Frozen V4 empirical-reserve development, paired against saved V3 outcomes."""
import argparse
import csv
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_controller import calibration, load_signal, policy_pool
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_reserves import reserve_planning_terms
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_units import simulate_units
from problems.public_battery_visibility import classify_failure, receipt_metadata

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_reserve_development_v4_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_reserve_development_v4_20260930"


def inspect(protocol, out, planning_rule=reserve_planning_terms):
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / protocol["basis_controller_protocol"]).read_text())
    with (ROOT / protocol["basis_controller_outputs"] / "windows.csv").open() as handle:
        previous = {(r["period"], r["profile_id"], int(r["window_start_minute"])): r for r in csv.DictReader(handle)}
    ids, nodes, points = policy_pool(base)
    if len(ids) != protocol["roster"]["library_profiles"] + len(protocol["roster"]["reference_profiles"]):
        raise ValueError("profile roster differs from frozen V4")
    fit, profiles, windows, cases, reserves, traces, failure_examples = calibration(base), [], [], [], [], [], []
    horizon = base["windows"]["hours"] * 60
    lookback = protocol["control"]["reserve_lookback_minutes"]
    asset = base["asset"]
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    capacity = np.asarray(asset["unit_energy_MWh"])
    library_indices = [i for i, (_, _, inside) in enumerate(ids) if inside]
    reference_index = next(i for i, (name, _, _) in enumerate(ids) if name == "constant_0.50")
    for case in base["development"]:
        label = case["start"][:10]
        start, stop = instant(case["start"]), instant(case["stop"])
        units = [cached_unit(base, case, i) for i in (0, 1)]
        workload = site_workload(units, start, stop)
        history = site_workload(units, start - timedelta(minutes=lookback), stop)
        plans = receipt_plans(units, start, len(workload["active"]))
        terms = planning_rule(plans, history, asset, lookback, tolerance)
        stress, prices = load_signal(base, case, fit)
        targets = np.array([np.interp(stress, nodes, point) for point in points])
        starts = np.arange(0, len(workload["active"]) - horizon + 1, base["windows"]["start_stride_hours"] * 60)
        if len(starts) != protocol["roster"]["complete_windows_per_sample"]:
            raise ValueError("window population differs from frozen V4")
        used_decisions = int((starts[-1] + horizon - 90) // 30 + 1)
        conflicts = int(terms["reserve_conflict"][:used_decisions].sum())
        for d in range(used_decisions):
            row = {"period": label, "decision_minute": d * 30, "reserve_conflict": bool(terms["reserve_conflict"][d])}
            for u in (0, 1):
                for field in ("target_lower_MWh", "target_upper_MWh", "past_BOA_peak_MW", "future_power_bound_MW"):
                    row[f"unit_{u + 1}_{field}"] = float(terms[field][d, u])
            reserves.append(row)
        if conflicts:
            report = {"period": label, "status": "not_simulated_reserve_rule_conflict", "reserve_conflict_decisions": conflicts}
            cases.append(report)
            print(json.dumps(report), flush=True)
            continue
        result = simulate_units(workload, prices, targets, starts, horizon, asset, terms,
                                base["control"]["initial_soc_fraction"], tolerance)
        error = float(np.max(np.abs(result["energy_balance_error_MWh"])))
        if error > tolerance or np.min(result["minimum_unit_SOC_MWh"]) < -tolerance or np.any(result["maximum_unit_SOC_MWh"] > capacity + tolerance):
            raise ValueError("accepted V4 trajectory violates inventory or energy accounting")
        success = result["success"]
        old_success = np.array([[previous[label, name, int(s)]["success"] == "True" for s in starts] for name, _, _ in ids])
        for p, (name, family, inside) in enumerate(ids):
            fraction = float(success[p].mean())
            completed = result["cost_GBP"][p, success[p]]
            profiles.append({"period": label, "profile_id": name, "family": family, "in_library": inside,
                "window_success_fraction": fraction, "empirically_feasible": fraction >= 1 - base["failure"]["chance_failure_probability"],
                "all_windows_completed": bool(success[p].all()), "V3_window_success_fraction": float(old_success[p].mean()),
                "gained_windows": int((success[p] & ~old_success[p]).sum()), "lost_windows": int((~success[p] & old_success[p]).sum()),
                "mean_completed_window_cost_GBP": float(completed.mean()) if len(completed) else None})
            for w, s in enumerate(starts):
                old = previous[label, name, int(s)]
                row = {"period": label, "profile_id": name, "in_library": inside, "window_start_minute": int(s),
                    "success": bool(success[p, w]), "V3_success": bool(old_success[p, w]),
                    "first_failure_minute": int(result["first_failure_minute"][p, w]), "V3_first_failure_minute": int(old["first_failure_minute"]),
                    "power_failure": bool(result["power_failure"][p, w]), "V3_power_failure": old["power_failure"],
                    "unit_1_inventory_failure": bool(result["unit_inventory_failure"][p, w, 0]),
                    "unit_2_inventory_failure": bool(result["unit_inventory_failure"][p, w, 1]),
                    "BOA_inventory_failure": bool(result["BOA_inventory_failure"][p, w]),
                    "baseline_inventory_failure": bool(result["baseline_inventory_failure"][p, w])}
                for field in ("cost_GBP", "metered_cash_GBP", "inventory_adjustment_GBP", "final_SOC_MWh", "gross_import_MWh", "gross_export_MWh"):
                    row[field] = float(result[field][p, w]) if success[p, w] else None
                windows.append(row)
        library, old_library = success[library_indices], old_success[library_indices]
        report = {"period": label, "status": "evaluated", "windows": len(starts), "reserve_conflict_decisions": 0,
            "library_feasible_profiles": sum(r["empirically_feasible"] for r in profiles[-len(ids):] if r["in_library"]),
            "V3_library_feasible_profiles": int((old_library.mean(axis=1) >= .95).sum()),
            "library_all_windows_completed_profiles": int(library.all(axis=1).sum()),
            "library_successful_windows_union": int(library.any(axis=0).sum()),
            "library_failed_windows": int((~library).sum()),
            "library_gained_windows": int((library & ~old_library).sum()), "library_lost_windows": int((~library & old_library).sum()),
            "library_first_failure_causes": {"power_failure": int(result["power_failure"][library_indices].sum()),
                "BOA_inventory_failure": int(result["BOA_inventory_failure"][library_indices].sum()),
                "baseline_inventory_failure": int(result["baseline_inventory_failure"][library_indices].sum()),
                "unit_1_inventory_failure": int(result["unit_inventory_failure"][library_indices, :, 0].sum()),
                "unit_2_inventory_failure": int(result["unit_inventory_failure"][library_indices, :, 1].sum())},
            "reference_50_success_windows": int(success[reference_index].sum()),
            "max_unit_energy_balance_error_MWh": error}
        cases.append(report)
        reference_failure = next((c for c in result["first_failure_contexts"] if c["policy_index"] == reference_index and c["window_index"] == 0), None)
        if reference_failure is not None:
            failure_examples.append({"period": label, "profile_id": "constant_0.50", "window_start_minute": 0,
                "submission_context": reference_failure,
                "reconstruction": classify_failure(reference_failure, 0, plans, workload,
                                                    receipt_metadata(units, start, stop), asset, tolerance)})
        traces.append({"period": label, "first_policy_first_window": result["nomination_trace"],
                       "reference_50_first_window_failure": reference_failure})
        print(json.dumps(report), flush=True)
    complete = all(c["status"] == "evaluated" for c in cases)
    common = [name for name, _, inside in ids if inside and complete
              and all(p["empirically_feasible"] for p in profiles if p["profile_id"] == name)]
    strict = [name for name in common if all(p["all_windows_completed"] for p in profiles if p["profile_id"] == name)]
    gate_pass = complete and bool(common) and any(not p["empirically_feasible"] for p in profiles if p["in_library"])
    summary = {"protocol_id": protocol["protocol_id"], "status": "development_complete" if complete else "development_hold_reserve_conflict",
        "normalization": fit, "cases": cases, "library_profiles_feasible_in_all_periods": common,
        "library_profiles_completing_all_windows_in_all_periods": strict, "development_gate_passed": gate_pass,
        "decision": "assess_application_and_source_protocol" if gate_pass else "hold_full_comparison",
        "diagnostic_window_evaluations": len(windows), "planned_window_evaluations": protocol["roster"]["window_evaluations"],
        "new_data_requests": 0, "new_API_response_bytes": 0, "algorithm_optimizer_calls": 0,
        "terminal_verifier_calls": 0, "confirmation_year_access": False,
        "limitations": base["model_assumptions"] + [protocol["limitations"]], "cost_scope": base["cost"]["scope"]}
    write_csv(out / "profiles.csv", profiles)
    write_csv(out / "windows.csv", windows)
    write_csv(out / "reserves.csv", reserves)
    (out / "nomination_traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    (out / "reference_failure_examples.json").write_text(json.dumps(failure_examples, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
