#!/usr/bin/env python3
"""Evaluate the frozen two-call bound and four saved first-sign branches."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_two_call_bound import two_call_bound
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_two_call_information_bound_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_two_call_information_bound_v1_20261001"


def read(relative):
    return json.loads((ROOT / relative).read_text())


def inspect(protocol, out):
    pilot = read(protocol["basis_protocol"])
    absolute = read(pilot["basis_protocol"])
    causal = read(absolute["basis_previous_pilot"])
    base = read(causal["retained_controller_protocol"])
    retained = read(protocol["retained_asset_protocol"])["retained_asset"]
    asset = {"power_MW": retained["shared_gross_power_MW"],
             **{k: retained[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    spec = protocol["component_probes"]
    unit = spec["unit_for_bound"] - 1
    peak = absolute["contract"]["absolute_request_peak_MW"]["export"]
    profiles = {p["id"]: p["target_fractions"] for p in pilot["sequential_pilot"]["profiles"]}
    diagnosis = {r["profile_id"]: r for r in read(protocol["basis_diagnosis"])["rows"]}
    hours = read(Path(protocol["basis_outputs"]) / "hourly_dispatch.json")
    if (base["control"]["decision_minutes"] != 30
            or base["control"]["nomination_lead_minutes"] != 60
            or spec["first_prefix_duration_minutes"] != 60
            or spec["first_call_branches"] != ["export", "import"]
            or peak != [49, 49]
            or absolute["contract"]["absolute_request_peak_MW"]["import"] != [-49, -49]):
        raise ValueError("two-call bound requires the frozen clock and full absolute pulse")
    # A context mismatch blocks interpretation before any component calculation.
    for context in spec["contexts"]:
        saved = next(h for h in hours if h["profile_id"] == context["profile_id"]
                     and h["dispatch_rule"] == "conservative_admission"
                     and h["minute"] == context["first_call_minute"])
        prior = diagnosis[context["profile_id"]]["earlier_tail_nomination"]
        if (saved["initial_SOC_MWh"] != context["first_call_initial_SOC_MWh"]
                or saved["locked_PN_MW"] != context["first_hour_locked_PN_MW"]
                or prior["minute"] != context["first_nomination_minute"]
                or prior["receipt_cue"] != context["current_cue"]
                or context["target_fractions"] != profiles[context["profile_id"]]
                or context["second_call_minute"] - context["first_call_minute"] != 60
                or context["first_call_minute"] - context["first_nomination_minute"] != 30):
            raise ValueError("frozen first-call context differs from the retained pilot")
    result = two_call_bound(asset, peak[unit], unit, tolerance, tolerance)
    rows = []
    for context in spec["contexts"]:
        branches = {}
        for direction, sign in (("export", 1), ("import", -1)):
            first, last = pending_pulse_power(context["first_hour_locked_PN_MW"],
                                              sign * np.array(peak))
            metrics = trajectory(first, last, np.array(context["first_call_initial_SOC_MWh"]), 60, asset)
            branches[direction] = {"physics": metrics,
                                   "physical_feasible": max(metrics["unit_bound_violation_MWh"]) <= tolerance
                                   and metrics["power_bound_violation_MW"] <= tolerance}
        separation = np.array(branches["import"]["physics"]["final_SOC_MWh"]) - np.array(
            branches["export"]["physics"]["final_SOC_MWh"])
        if np.min(separation) < result["first_stock_separation_lower_bound_MWh"] - tolerance:
            raise ValueError("saved branch kernel contradicts the analytic separation")
        rows.append({"profile_id": context["profile_id"], "branches": branches,
                     "first_stock_separation_MWh": separation.tolist(),
                     "unit_separation_excess_over_second_band_MWh":
                     float(separation[unit] - result["second_stock_band_width_MWh"])})
    count = 2 * len(rows)
    if (count != spec["planned_saved_context_prefix_paths"]
            or result["optimistic_second_pulse_paths"] != spec["planned_optimistic_second_pulse_paths"]):
        raise ValueError("component path accounting differs from the frozen six-path roster")
    summary = {"protocol_id": protocol["protocol_id"],
               "status": "conditional_uniform_two_call_guarantee_impossible"
               if result["uniform_four_sign_guarantee_impossible"] else "two_call_bound_inconclusive",
               **result, "saved_context_prefix_paths": count, "total_component_paths": count + 2,
               "all_saved_first_branches_physical_feasible": all(
                   b["physical_feasible"] for r in rows for b in r["branches"].values()),
               "decision": "reassess_task_and_nomination_information_before_another_controller_change",
               **protocol["accounting_constraints"], "library_matrix_launch": False,
               "source_comparison_gate": protocol["source_comparison_gate"],
               "limitations": protocol["limitations"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "saved_context_branches.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "branches"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
