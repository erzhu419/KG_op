#!/usr/bin/env python3
"""Trace nomination-time information for every frozen April V3 first failure."""
import argparse
from collections import Counter
import csv
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
from problems.public_battery_visibility import classify_failure, receipt_metadata

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_failure_visibility_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_failure_visibility_v1_20260930"


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    controller = json.loads((ROOT / protocol["basis_controller_protocol"]).read_text())
    ids, nodes, points = policy_pool(controller)
    selected = [i for i, (name, _, inside) in enumerate(ids) if inside or name == "constant_0.50"]
    ids, points = [ids[i] for i in selected], points[selected]
    with (ROOT / protocol["basis_controller_outputs"] / "windows.csv").open() as handle:
        previous = {(r["period"], r["profile_id"], int(r["window_start_minute"])): r for r in csv.DictReader(handle)}
    fit, rows, examples = calibration(controller), [], {}
    horizon = controller["windows"]["hours"] * 60
    tolerance = controller["failure"]["numerical_energy_tolerance_MWh"]
    for case in protocol["development"]:
        label = case["start"][:10]
        start, stop = instant(case["start"]), instant(case["stop"])
        units = [cached_unit(controller, case, i) for i in (0, 1)]
        workload = site_workload(units, start, stop)
        plans = receipt_plans(units, start, len(workload["active"]))
        receipts = receipt_metadata(units, start, stop)
        stress, prices = load_signal(controller, case, fit)
        targets = np.array([np.interp(stress, nodes, point) for point in points])
        starts = np.arange(0, len(workload["active"]) - horizon + 1, controller["windows"]["start_stride_hours"] * 60)
        result = simulate_units(workload, prices, targets, starts, horizon, controller["asset"],
                                unit_planning_terms(plans, controller["asset"]),
                                controller["control"]["initial_soc_fraction"], tolerance)
        for p, (name, _, _) in enumerate(ids):
            for w, minute in enumerate(starts):
                old = previous[label, name, int(minute)]
                if bool(result["success"][p, w]) != (old["success"] == "True") or int(result["first_failure_minute"][p, w]) != int(old["first_failure_minute"]):
                    raise ValueError("trace instrumentation changed a saved V3 success flag or first failure minute")
        for context in result["first_failure_contexts"]:
            p, w = context["policy_index"], context["window_index"]
            name, family, inside = ids[p]
            report = classify_failure(context, int(starts[w]), plans, workload, receipts, controller["asset"], tolerance)
            old = previous[label, name, int(starts[w])]
            known, actual = report.get("known_prefix"), report.get("actual_prefix")
            row = {"period": label, "profile_id": name, "family": family, "in_library": inside,
                   "window_start_minute": int(starts[w]), "first_failure_minute": context["failure_minute"],
                   "nomination_decision_minute": context["nomination_decision_minute"],
                   "classification": report["classification"],
                   "later_changed_receipts": len(report.get("later_changed_receipts", [])),
                   "power_failure": old["power_failure"], "unit_1_inventory_failure": old["unit_1_inventory_failure"],
                   "unit_2_inventory_failure": old["unit_2_inventory_failure"]}
            for u in (0, 1):
                row[f"known_unit_{u + 1}_bound_violation_MWh"] = known["unit_bound_violation_MWh"][u] if known else None
                row[f"actual_unit_{u + 1}_bound_violation_MWh"] = actual["unit_bound_violation_MWh"][u] if actual else None
            row["known_power_bound_violation_MW"] = known["power_bound_violation_MW"] if known else None
            row["known_violation_matches_actual_failure_cause"] = bool(known and
                ((known["power_bound_violation_MW"] > tolerance and old["power_failure"] == "True") or
                 any(known["unit_bound_violation_MWh"][u] > tolerance and old[f"unit_{u + 1}_inventory_failure"] == "True" for u in (0, 1))))
            rows.append(row)
            if report["classification"] not in examples:
                examples[report["classification"]] = {"row": row, "submission_context": context, "reconstruction": report}
        print(json.dumps({"period": label, "classified_failures": len(rows),
                          "classes": dict(Counter(r["classification"] for r in rows))}), flush=True)
    library = [r for r in rows if r["in_library"]]
    reference = [r for r in rows if not r["in_library"]]
    counts = Counter(r["classification"] for r in rows)
    summary = {"protocol_id": protocol["protocol_id"], "status": "failure_visibility_diagnostic_complete",
               "classified_failures": len(rows), "diagnostic_window_evaluations": len(ids) * len(starts) * len(protocol["development"]),
               "saved_V3_results_reproduced": True, "classification_counts": dict(counts),
               "library_classification_counts": dict(Counter(r["classification"] for r in library)),
               "reference_classification_counts": dict(Counter(r["classification"] for r in reference)),
               "known_violation_matching_actual_cause": sum(r["known_violation_matches_actual_failure_cause"] for r in rows),
               "decision": "resolve_trace_discrepancies" if counts["unresolved_trace_discrepancy"] else "freeze_evidence_based_control_repair",
               "new_data_requests": 0, "algorithm_optimizer_calls": 0, "terminal_verifier_calls": 0,
               "confirmation_year_access": False, "scope": protocol["limitations"]}
    write_csv(out / "failures.csv", rows)
    (out / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
