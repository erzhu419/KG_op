#!/usr/bin/env python3
"""Offline unit-inventory sensitivity for the ten common pooled-site profiles."""
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_controller import calibration, load_signal, policy_pool
from performance.inspect_public_battery_dispatch import decode
from problems.public_battery_site import planning_terms, receipt_plans, simulate_site, site_workload

SPEC = ROOT / "performance/manifests/public_battery_site_partition_diagnostic_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_site_development_v2_20260930"


def cached_unit(protocol, case, unit_index):
    start, stop = instant(case["start"]), instant(case["stop"])
    prefix_start = start - timedelta(days=protocol["dispatch_prefix_days"])
    left = prefix_start
    unit = protocol["asset"]["workload_bm_units"][unit_index]
    rows = []
    while left < stop:
        right = min(stop, left + timedelta(days=protocol["request_chunk_days"]))
        if unit_index:
            path = OUTPUT / "samples" / f"{unit}_{left.date()}_{right.date()}.json"
        elif left == instant("2024-04-14T00:00:00Z"):
            path = ROOT / "paper_artifacts/public_battery_application_spec_20260930/pillswood_acceptances_20240414_15.json"
        else:
            path = ROOT / "paper_artifacts/public_battery_controller_development_v1_20260930/samples" / f"{left.date()}_{right.date()}_acceptances.json"
        piece = json.loads(path.read_text())["payload"]["data"]
        rows.extend(r for r in piece if left <= instant(r["timeFrom"]) < right)
        left = right
    return [s for s in decode(rows, unit) if s.start < stop and s.stop > prefix_start]


def inspect():
    spec = json.loads(SPEC.read_text())
    protocol = json.loads((ROOT / spec["basis_protocol"]).read_text())
    ids, nodes, points = policy_pool(protocol)
    selected = [i for i, (name, _, _) in enumerate(ids) if name in spec["profiles"]]
    points = points[selected]
    names = [ids[i][0] for i in selected]
    fit, rows = calibration(protocol), []
    capacity_required = np.zeros((len(names), 2))
    horizon = protocol["windows"]["hours"] * 60
    tolerance = protocol["failure"]["numerical_energy_tolerance_MWh"]
    for case in protocol["development"]:
        start, stop = instant(case["start"]), instant(case["stop"])
        units = [cached_unit(protocol, case, i) for i in (0, 1)]
        workload = site_workload(units, start, stop)
        stress, prices = load_signal(protocol, case, fit)
        targets = np.array([np.interp(stress, nodes, point) for point in points])
        starts = np.arange(0, len(workload["active"]) - horizon + 1,
                           protocol["windows"]["start_stride_hours"] * 60)
        terms = planning_terms(receipt_plans(units, start, len(workload["active"])), protocol["asset"])
        result = simulate_site(workload, prices, targets, starts, horizon, protocol["asset"], terms,
                               protocol["control"]["initial_soc_fraction"], tolerance)
        for p, name in enumerate(names):
            pooled = result["success"][p]
            separate = result["equal_partition_success"][p]
            required = 2 * np.maximum(-result["unit_minimum_SOC_change_MWh"][p, pooled],
                                      result["unit_maximum_SOC_change_MWh"][p, pooled])
            capacity_required[p] = np.maximum(capacity_required[p], required.max(axis=0))
            rows.append({"period": case["start"][:10], "profile_id": name,
                         "pooled_success_windows": int(pooled.sum()),
                         "equal_partition_success_windows": int(separate.sum()),
                         "equal_partition_success_fraction": float(separate.mean()),
                         "equal_partition_empirically_feasible": float(separate.mean()) >= 1 - protocol["failure"]["chance_failure_probability"],
                         "additional_unit_failure_windows": int(np.sum(pooled & ~separate))})
        print(json.dumps({"period": case["start"][:10], "equal_partition_feasible_profiles":
                          sum(r["equal_partition_empirically_feasible"] for r in rows[-len(names):])}), flush=True)
    common = [name for name in names if all(r["equal_partition_empirically_feasible"] for r in rows if r["profile_id"] == name)]
    allocations = []
    for name, required in zip(names, capacity_required):
        lower, upper = float(required[0]), float(protocol["asset"]["energy_MWh"] - required[1])
        allocations.append({"profile_id": name, "unit_capacity_required_MWh": required.tolist(),
                            "all_pooled_success_windows_allow_static_partition": lower <= upper,
                            "unit_1_capacity_interval_MWh": [lower, upper] if lower <= upper else None})
    summary = {"protocol_id": spec["protocol_id"], "rows": rows, "capacity_diagnostic": allocations,
               "common_equal_partition_feasible_profiles": common,
               "decision": "source_archive_pilot_with_explicit_partition_assumption" if common else "hold_source_comparison_for_unit_controller",
               "diagnostic_window_evaluations": len(names) * len(protocol["development"]) * len(starts),
               "new_API_requests": 0, "optimizer_calls": 0, "terminal_verifier_calls": 0,
               "confirmation_year_access": False,
               "limitation": "98 MWh per unit is an equal-allocation sensitivity assumption; required capacities cover all completed pooled windows and are stricter than the 95% chance criterion"}
    write_csv(OUTPUT / "partition_profiles.csv", rows)
    (OUTPUT / "partition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    site_summary = json.loads((OUTPUT / "summary.json").read_text())
    site_summary.setdefault("pooled_development_gate", site_summary["decision"])
    site_summary["decision"] = summary["decision"]
    site_summary["partition_diagnostic"] = {"file": "partition_summary.json",
                                          "common_equal_partition_feasible_profiles": common}
    (OUTPUT / "summary.json").write_text(json.dumps(site_summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    inspect()
