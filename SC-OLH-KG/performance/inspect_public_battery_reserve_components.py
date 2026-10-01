#!/usr/bin/env python3
"""Serial frozen inventory-only and power-only ablations, paired with V3/V4."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.inspect_independent_energy_data import write_csv
from performance.inspect_public_battery_reserves import inspect as evaluate
from problems.public_battery_reserves import inventory_reserve_planning_terms, power_reserve_planning_terms

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_reserve_components_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_reserve_components_v1_20260930"


def read_windows(path):
    with path.open() as handle:
        return {(r["period"], r["profile_id"], int(r["window_start_minute"])): r for r in csv.DictReader(handle)}


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    references = {name: read_windows(ROOT / path / "windows.csv")
                  for name, path in protocol["retained_comparators"].items()}
    rules = {"inventory_reserve_only": inventory_reserve_planning_terms,
             "power_reserve_only": power_reserve_planning_terms}
    summaries, paired = [], []
    for condition in protocol["conditions"]:
        name = condition["name"]
        resolved = json.loads((ROOT / protocol["basis_protocol"]).read_text())
        resolved.update({"protocol_id": protocol["protocol_id"] + "/" + name,
                         "role": protocol["role"], "registration_context": protocol["registration_context"],
                         "limitations": protocol["limitations"], "component_condition": condition})
        resolved["control"]["target"] = condition["target"]
        resolved["control"]["power_reserve"] = condition["power"]
        child_out = out / name
        child_out.mkdir(parents=True, exist_ok=True)
        (child_out / "resolved_protocol.json").write_text(json.dumps(resolved, indent=2) + "\n")
        evaluate(resolved, child_out, rules[name])
        child = json.loads((child_out / "summary.json").read_text())
        summaries.append({"condition": name, **child})
        current = read_windows(child_out / "windows.csv")
        for period in sorted({key[0] for key in current}):
            selected = [(key, r) for key, r in current.items() if key[0] == period and r["in_library"] == "True"]
            for reference, old in references.items():
                categories = {"both_success": 0, "gained_windows": 0, "lost_windows": 0, "both_failed": 0}
                for key, row in selected:
                    new_success, old_success = row["success"] == "True", old[key]["success"] == "True"
                    label = ("both_success" if old_success else "gained_windows") if new_success else ("lost_windows" if old_success else "both_failed")
                    categories[label] += 1
                paired.append({"condition": name, "period": period, "reference": reference,
                    "library_window_evaluations": len(selected), **categories,
                    "new_power_failure_windows": sum(r["power_failure"] == "True" for _, r in selected),
                    "reference_power_failure_windows": sum(old[key]["power_failure"] == "True" for key, _ in selected)})
        print(json.dumps({"condition": name, "common_feasible_library": child["library_profiles_feasible_in_all_periods"],
                          "development_gate_passed": child["development_gate_passed"]}), flush=True)
    total = sum(s["diagnostic_window_evaluations"] for s in summaries)
    complete = total == protocol["accounting"]["planned_new_window_evaluations"] and all(s["status"] == "development_complete" for s in summaries)
    passing = [s["condition"] for s in summaries if s["development_gate_passed"]]
    summary = {"protocol_id": protocol["protocol_id"], "status": "component_development_complete" if complete else "component_development_incomplete",
        "conditions": summaries, "paired_library_outcomes": paired, "passing_conditions": passing,
        "decision": "assess_passing_control_and_source_protocol" if complete and passing else "hold_90_minute_reserve_family",
        "diagnostic_window_evaluations": total, "planned_window_evaluations": protocol["accounting"]["planned_new_window_evaluations"],
        "retained_comparator_reruns": 0, "new_data_requests": 0, "algorithm_optimizer_calls": 0,
        "terminal_verifier_calls": 0, "confirmation_year_access": False, "limitations": protocol["limitations"]}
    write_csv(out / "paired_comparison.csv", paired)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("conditions", "paired_library_outcomes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
