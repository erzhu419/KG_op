#!/usr/bin/env python3
"""Frozen V5 receipt-prefix diagnosis from retained stocks and commitments."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_visibility import classify_failure, receipt_metadata, trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_targets_failure_visibility_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_targets_failure_visibility_v1_20260930"


def first_failure_piece_matches(context, window, workload, asset, tolerance):
    decision = window + context["nomination_decision_minute"]
    baseline = np.repeat(np.vstack((context["pending_MW"], context["nomination_MW"])), 30, axis=0)
    active = workload["active"][decision:decision + 90]
    first = np.where(active, workload["first_MW"][decision:decision + 90], baseline)
    last = np.where(active, workload["last_MW"][decision:decision + 90], baseline)
    minute = context["failure_minute"] - context["nomination_decision_minute"]
    stop = context["failure_piece_stop_fraction"]
    a, b = first[minute], last[minute]
    cuts = [0., 1.] + [-a[u] / (b[u] - a[u]) for u in (0, 1) if a[u] * b[u] < 0]
    previous = max(t for t in cuts if t < stop - 1e-12)
    initial = np.array(context["decision_SOC_MWh"])
    before = trajectory(first, last, initial, minute + previous, asset)
    failed = trajectory(first, last, initial, minute + stop, asset)
    safe_before = max(before["unit_bound_violation_MWh"]) <= tolerance and before["power_bound_violation_MW"] <= tolerance
    return {"safe_before_failing_piece": safe_before,
            "piece_stop_is_reconstruction_boundary": min(abs(t - stop) for t in cuts) < 1e-10,
            "inventory_failure": [v > tolerance for v in failed["unit_bound_violation_MWh"]],
            "power_failure": failed["power_bound_violation_MW"] > tolerance}


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / protocol["basis_controller_protocol"]).read_text())
    basis = ROOT / protocol["basis_V5_outputs"]
    period, names = protocol["roster"]["period"], protocol["roster"]["profiles"]
    with (basis / "windows.csv").open() as handle:
        saved = {(r["profile_id"], int(r["window_start_minute"])): r for r in csv.DictReader(handle)
                 if r["period"] == period and r["profile_id"] in names}
    expected = {(name, minute) for name in names for minute in range(0, 4321, 60)}
    if set(saved) != expected or any(r["success"] != "False" for r in saved.values()):
        raise ValueError("saved failed-window roster differs from the frozen diagnosis")
    contexts = {}
    with gzip.open(basis / "failure_contexts.jsonl.gz", "rt") as handle:
        for line in handle:
            c = json.loads(line)
            if c["period"] == period and c["profile_id"] in names:
                key = c["profile_id"], c["window_start_minute"]
                if key in contexts:
                    raise ValueError("retained failure context is duplicated")
                contexts[key] = c
    if set(contexts) != expected or len(contexts) != protocol["roster"]["failed_contexts"]:
        raise ValueError("retained state roster differs from the frozen diagnosis")
    case = next(c for c in base["development"] if c["start"][:10] == period)
    start, stop = instant(case["start"]), instant(case["stop"])
    units = [cached_unit(base, case, u) for u in (0, 1)]
    workload = site_workload(units, start, stop)
    plans = receipt_plans(units, start, len(workload["active"]))
    receipts = receipt_metadata(units, start, stop)
    asset, tolerance = base["asset"], base["failure"]["numerical_energy_tolerance_MWh"]
    rows, probes, examples = [], [], {}
    for key in sorted(expected):
        name, window = key
        c, old = contexts[key], saved[key]
        if c["failure_minute"] != int(old["first_failure_minute"]):
            raise ValueError("saved failure minute and retained context disagree")
        report = classify_failure(c, window, plans, workload, receipts, asset, tolerance)
        match = first_failure_piece_matches(c, window, workload, asset, tolerance)
        reproduced = (match["safe_before_failing_piece"] and match["piece_stop_is_reconstruction_boundary"]
                      and match["power_failure"] == (old["power_failure"] == "True")
                      and match["inventory_failure"] == [old[f"unit_{u}_inventory_failure"] == "True" for u in (1, 2)])
        if not reproduced:
            report["classification"] = "unresolved_trace_discrepancy"
        known, actual = report["known_prefix"], report["actual_prefix"]
        row = {"period": period, "profile_id": name, "window_start_minute": window,
               "first_failure_minute": c["failure_minute"], "failure_absolute_minute": window + c["failure_minute"],
               "decision_absolute_minute": report["decision_absolute_minute"],
               "classification": report["classification"], "first_failure_piece_reproduced": reproduced,
               "later_changed_receipts": len(report["later_changed_receipts"]),
               "changed_power_minutes": report["changed_power_minutes"],
               **{k: old[k] == "True" for k in ("power_failure", "unit_1_inventory_failure", "unit_2_inventory_failure")},
               "known_power_bound_violation_MW": known["power_bound_violation_MW"],
               "actual_power_bound_violation_MW": actual["power_bound_violation_MW"]}
        for u in (0, 1):
            row[f"known_unit_{u + 1}_bound_violation_MWh"] = known["unit_bound_violation_MWh"][u]
            row[f"actual_unit_{u + 1}_bound_violation_MWh"] = actual["unit_bound_violation_MWh"][u]
        rows.append(row)
        probes.append({"row": row, "submission_context": c, "first_piece_match": match, "reconstruction": report})
        examples.setdefault((name, report["classification"]), probes[-1])
    groups = [{"profile_id": name, "contexts": sum(r["profile_id"] == name for r in rows),
               "classification_counts": dict(Counter(r["classification"] for r in rows if r["profile_id"] == name)),
               "distinct_absolute_failure_minutes": len({r["failure_absolute_minute"] for r in rows if r["profile_id"] == name})}
              for name in names]
    counts = Counter(r["classification"] for r in rows)
    summary = {"protocol_id": protocol["protocol_id"],
               "status": "failure_visibility_diagnostic_complete" if not counts["unresolved_trace_discrepancy"] else "failure_visibility_diagnostic_unresolved",
               "classified_failures": len(rows), "classification_counts": dict(counts), "profiles": groups,
               "saved_first_failure_pieces_reproduced": sum(r["first_failure_piece_reproduced"] for r in rows),
               "distinct_absolute_failure_minutes": len({r["failure_absolute_minute"] for r in rows}),
               "decision": "resolve_trace_discrepancies" if counts["unresolved_trace_discrepancy"] else "assess_causal_physical_projection" if counts["known_new_delivery_violation"] else "assess_information_and_commitment_model",
               "source_comparison_gate": "HOLD", **protocol["accounting"], "limitations": protocol["limitations"]}
    write_csv(out / "failures.csv", rows)
    (out / "probes.json").write_text(json.dumps(probes, indent=2) + "\n")
    (out / "examples.json").write_text(json.dumps(list(examples.values()), indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
