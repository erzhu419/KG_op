#!/usr/bin/env python3
"""Frozen V5 causal two-channel development, with explicit V3 result reuse."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
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

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_controller_development_v5_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_controller_development_v5_20260930"
FLAGS = ("success", "power_failure", "unit_1_inventory_failure", "unit_2_inventory_failure",
         "BOA_inventory_failure", "baseline_inventory_failure")
ECONOMIC = ("cost_GBP", "metered_cash_GBP", "inventory_adjustment_GBP", "final_SOC_MWh",
            "gross_import_MWh", "gross_export_MWh", "unit_1_final_SOC_MWh", "unit_2_final_SOC_MWh")


def new_window_rows(result, candidates, starts, period):
    rows = []
    for p, candidate in enumerate(candidates):
        for w, minute in enumerate(starts):
            success = bool(result["success"][p, w])
            row = {"period": period, "profile_id": candidate["profile_id"],
                   "window_start_minute": int(minute), "success": success,
                   "first_failure_minute": int(result["first_failure_minute"][p, w]),
                   "power_failure": bool(result["power_failure"][p, w]),
                   "unit_1_inventory_failure": bool(result["unit_inventory_failure"][p, w, 0]),
                   "unit_2_inventory_failure": bool(result["unit_inventory_failure"][p, w, 1]),
                   "BOA_inventory_failure": bool(result["BOA_inventory_failure"][p, w]),
                   "baseline_inventory_failure": bool(result["baseline_inventory_failure"][p, w])}
            for name in ECONOMIC:
                if name.startswith("unit_"):
                    value = result["unit_final_SOC_MWh"][p, w, int(name[5]) - 1]
                else:
                    value = result[name][p, w]
                row[name] = float(value) if success else None
            rows.append({**row, "evaluation_origin": "new_V5_rollout", "source_profile_id": ""})
    return rows


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / protocol["basis_controller_protocol"]).read_text())
    with (ROOT / protocol["candidate_mapping"]["roster_csv"]).open() as handle:
        catalog = [{**r, "in_library": True} for r in csv.DictReader(handle)]
    if len(catalog) != protocol["roster"]["joint_library_profiles"]:
        raise ValueError("joint candidate roster differs from the frozen protocol")
    for value in protocol["roster"]["extra_shared_reference_profiles"]:
        name = f"constant_{value:.2f}"
        catalog.append({"profile_id": name, "family": "extra_reference", "unit_1_component": name,
                        "unit_2_component": name, "reused_V3_profile_id": name, "in_library": False})
    fresh = [c for c in catalog if not c["reused_V3_profile_id"]]
    reused = [c for c in catalog if c["reused_V3_profile_id"]]
    saved_names = {c["reused_V3_profile_id"] for c in reused}
    saved = {}
    with (ROOT / protocol["basis_controller_outputs"] / "windows.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["profile_id"] not in saved_names:
                continue
            for name in FLAGS:
                row[name] = row[name] == "True"
            for name in ("window_start_minute", "first_failure_minute"):
                row[name] = int(row[name])
            for name in ECONOMIC:
                row[name] = float(row[name]) if row[name] else None
            saved[row["period"], row["profile_id"], row["window_start_minute"]] = row
    ids, nodes, points = policy_pool(base)
    index = {name: i for i, (name, _, _) in enumerate(ids)}
    fit = calibration(base)
    horizon = base["windows"]["hours"] * 60
    asset = base["asset"]
    capacity = np.asarray(asset["unit_energy_MWh"])
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    profiles, windows, cases, traces = [], [], [], []
    with gzip.open(out / "failure_contexts.jsonl.gz", "wt") as failure_file:
        for case in base["development"]:
            label = case["start"][:10]
            print(json.dumps({"period": label, "stage": "starting_frozen_rollout", "new_profiles": len(fresh)}), flush=True)
            start, stop = instant(case["start"]), instant(case["stop"])
            units = [cached_unit(base, case, u) for u in (0, 1)]
            workload = site_workload(units, start, stop)
            stress, prices = load_signal(base, case, fit)
            component_targets = np.array([np.interp(stress, nodes, point) for point in points])
            targets = np.stack([np.stack([component_targets[index[c[f"unit_{u}_component"]]]
                                         for u in (1, 2)], axis=-1) for c in fresh])
            starts = np.arange(0, len(workload["active"]) - horizon + 1, base["windows"]["start_stride_hours"] * 60)
            if len(starts) != protocol["roster"]["complete_windows_per_sample"]:
                raise ValueError("development window roster changed")
            terms = unit_planning_terms(receipt_plans(units, start, len(workload["active"])), asset)
            result = simulate_units(workload, prices, targets, starts, horizon, asset, terms,
                                    base["control"]["initial_soc_fraction"], tolerance)
            energy_error = float(np.max(np.abs(result["energy_balance_error_MWh"])))
            if energy_error > tolerance or np.min(result["minimum_unit_SOC_MWh"]) < -tolerance or np.any(result["maximum_unit_SOC_MWh"] > capacity + tolerance):
                raise ValueError("accepted two-unit trajectory violates energy bounds or accounting")
            by_profile = defaultdict(list)
            new_rows = new_window_rows(result, fresh, starts, label)
            for row in new_rows:
                by_profile[row["profile_id"]].append(row)
            for candidate in reused:
                origin = "reused_V3_diagonal" if candidate["in_library"] else "reused_V3_shared_reference"
                for minute in starts:
                    row = saved[label, candidate["reused_V3_profile_id"], int(minute)]
                    by_profile[candidate["profile_id"]].append({**row, "profile_id": candidate["profile_id"],
                        "evaluation_origin": origin, "source_profile_id": candidate["reused_V3_profile_id"]})
            case_profiles, case_windows = [], []
            for candidate in catalog:
                rows = by_profile[candidate["profile_id"]]
                successes = sum(r["success"] for r in rows)
                completed = [r["cost_GBP"] for r in rows if r["success"]]
                profile = {"period": label, **candidate, "successful_windows": successes,
                    "window_success_fraction": successes / len(starts),
                    "empirically_feasible": successes / len(starts) >= 1 - base["failure"]["chance_failure_probability"],
                    "all_windows_completed": successes == len(starts),
                    "mean_completed_window_cost_GBP": float(np.mean(completed)) if completed else None,
                    **{name + "_windows": sum(r[name] for r in rows) for name in FLAGS if name != "success"}}
                case_profiles.append(profile)
                case_windows.extend(rows)
            profiles.extend(case_profiles)
            windows.extend(case_windows)
            library = [p for p in case_profiles if p["in_library"]]
            library_windows = [r for c in catalog if c["in_library"] for r in by_profile[c["profile_id"]]]
            union = np.any([[r["success"] for r in by_profile[c["profile_id"]]] for c in catalog if c["in_library"]], axis=0)
            report = {"period": label, "windows": len(starts), "new_window_evaluations": len(new_rows),
                "reused_window_evaluations": len(case_windows) - len(new_rows),
                "library_feasible_profiles": sum(p["empirically_feasible"] for p in library),
                "library_feasible_counts_by_family": dict(Counter(p["family"] for p in library if p["empirically_feasible"])),
                "library_best_successful_windows": max(p["successful_windows"] for p in library),
                "library_all_windows_completed_profiles": sum(p["all_windows_completed"] for p in library),
                "library_successful_windows_union": int(np.sum(union)),
                "library_failed_windows": sum(not r["success"] for r in library_windows),
                "library_first_failure_causes": {name: sum(r[name] for r in library_windows) for name in FLAGS if name != "success"},
                "max_new_unit_energy_balance_error_MWh": energy_error}
            cases.append(report)
            traces.append({"period": label, "profile_id": fresh[0]["profile_id"], "first_window_nomination_trace": result["nomination_trace"]})
            for context in result["first_failure_contexts"]:
                record = {"period": label, "profile_id": fresh[context["policy_index"]]["profile_id"],
                          "window_start_minute": int(starts[context["window_index"]]), **context}
                failure_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps(report), flush=True)
    origins = Counter(r["evaluation_origin"] for r in windows)
    if origins["new_V5_rollout"] != protocol["roster"]["new_window_evaluations"] or origins["reused_V3_diagonal"] != protocol["roster"]["reused_diagonal_constant_window_evaluations"] or origins["reused_V3_shared_reference"] != protocol["roster"]["reused_extra_reference_window_evaluations"] or len(windows) != protocol["roster"]["reported_total_window_evaluations"]:
        raise ValueError("completed new/reused window counts differ from the frozen protocol")
    common = [c["profile_id"] for c in catalog if c["in_library"] and all(p["empirically_feasible"] for p in profiles if p["profile_id"] == c["profile_id"])]
    strict = [c["profile_id"] for c in catalog if c["in_library"] and all(p["all_windows_completed"] for p in profiles if p["profile_id"] == c["profile_id"])]
    unsafe = any(not p["empirically_feasible"] for p in profiles if p["in_library"])
    gate = bool(common) and unsafe
    summary = {"protocol_id": protocol["protocol_id"], "status": "development_complete", "normalization": fit,
        "cases": cases, "library_profiles_feasible_in_all_periods": common,
        "library_profiles_completing_all_windows_in_all_periods": strict, "has_unsafe_library_profile": unsafe,
        "development_gate": "PASS" if gate else "HOLD",
        "decision": "register_source_archive_pilot" if gate else "hold_full_comparison",
        "source_comparison_gate": "HOLD", "reported_window_evaluations": len(windows), "window_evaluation_origins": dict(origins),
        "reused_V3_windows_file": str((ROOT / protocol["basis_controller_outputs"] / "windows.csv").relative_to(ROOT)),
        **protocol["accounting"], "comparison_scope": protocol["comparison_scope"], "cost_scope": protocol["cost_scope"],
        "limitations": [protocol["limitations"], *base["model_assumptions"]]}
    write_csv(out / "profiles.csv", profiles)
    write_csv(out / "windows.csv", windows)
    (out / "nomination_traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
