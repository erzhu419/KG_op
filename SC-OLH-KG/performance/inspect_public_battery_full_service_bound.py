#!/usr/bin/env python3
"""Apply the frozen aggregate-energy bound to existing pilot request marks."""
import argparse
import csv
from datetime import timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.inspect_public_battery_dispatch import instant
from problems.public_battery_full_service_bound import full_service_bound

PROTOCOL = ROOT / "performance/manifests/public_battery_full_service_energy_bound_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_full_service_energy_bound_v1_20261001"


def inspect(protocol, out):
    pilot = json.loads((ROOT / protocol["basis_protocol"]).read_text())
    asset = json.loads((ROOT / protocol["retained_asset_protocol"]).read_text())["retained_asset"]
    base = json.loads((ROOT / pilot["retained_controller_protocol"]).read_text())
    scope = protocol["scope"]
    # The necessity proof requires full site saturation and the unchanged state.
    if (scope["service_capacity_MW"] != pilot["request_workload"]["service_capacity_MW"]
            or scope["initial_SOC_MWh"] != pilot["roster"]["initial_SOC_MWh"]
            or sum(scope["service_capacity_MW"]) != scope["shared_gross_power_MW"]
            or scope["shared_gross_power_MW"] != asset["shared_gross_power_MW"]):
        raise ValueError("energy-bound hypotheses differ from the saturated frozen pilot")
    marks = json.loads((ROOT / protocol["basis_outputs"] / "request_marks.json").read_text())
    workload = pilot["request_workload"]
    if [m["minute"] for m in marks] != list(range(workload["clock_start_minute"],
                                                workload["clock_stop_exclusive_minute"], workload["clock_minutes"])):
        raise ValueError("saved request-clock roster differs from the frozen pilot")
    horizon = pilot["roster"]["horizon_hours"] * 60
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    result, prefix = full_service_bound(marks, scope["initial_SOC_MWh"], scope["shared_gross_power_MW"],
                                        asset["charge_efficiency"], asset["discharge_efficiency"],
                                        horizon, tolerance, tolerance)
    case = next(c for c in base["development"] if c["start"][:10] == pilot["roster"]["period"])
    first = result["first_inspected_violating_minute"]
    summary = {"protocol_id": protocol["protocol_id"], "status": "conditional_full_service_infeasible"
               if result["conditional_full_service_infeasible"] else "full_service_bound_inconclusive",
               **result, "first_violation_utc": None if first is None else
               (instant(case["start"]) + timedelta(minutes=first)).isoformat(),
               "request_clocks": len(marks), "site_peak_MW": scope["shared_gross_power_MW"],
               "charge_efficiency": asset["charge_efficiency"], "discharge_efficiency": asset["discharge_efficiency"],
               "power_tolerance_MW": tolerance, "stock_tolerance_per_unit_MWh": tolerance,
               "bound_precision": "optimistic power-tolerance correction retained; upper bound is not clipped at physical capacity",
               "proof": "full joint directional power forces near-zero net PN; conversion losses and prescribed pulses bound total stock from above",
               "decision": "review_business_duty_and_capacity_before_more_controller_search"
               if result["conditional_full_service_infeasible"] else "analyze_per_unit_stock_and_delayed_PN_scheduling",
               "source_comparison_gate": protocol["source_comparison_gate"], **protocol["accounting"],
               "library_matrix_launch": False, "limitations": protocol["limitations"]}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "prefix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(prefix)
        writer.writerows(zip(*prefix.values()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
