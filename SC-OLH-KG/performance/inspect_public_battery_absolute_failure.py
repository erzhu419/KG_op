#!/usr/bin/env python3
"""Explain the first absolute-request deficit from retained states and PN."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from problems.public_battery_pending_availability import reference_pulse_power
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_absolute_dispatch_pilot_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_absolute_dispatch_pilot_v1_20261001"


def inspect():
    protocol = json.loads(PROTOCOL.read_text())
    original = json.loads((ROOT / protocol["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": original["shared_gross_power_MW"],
             **{k: original[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    summary = json.loads((OUTPUT / "summary.json").read_text())
    hours = json.loads((OUTPUT / "hourly_dispatch.json").read_text())
    nominations = json.loads((OUTPUT / "nominations.json").read_text())
    rows = []
    for profile in protocol["sequential_pilot"]["profiles"]:
        selected = [h for h in hours if h["profile_id"] == profile["id"]
                    and h["dispatch_rule"] == "conservative_admission"]
        failure = next(h for h in selected if "request_shortfall" in h["service_failure_reasons"])
        minute = failure["minute"]
        baseline = np.repeat(failure["locked_PN_MW"], 30, axis=0)
        initial = np.array(failure["initial_SOC_MWh"])
        requested = reference_pulse_power(baseline, baseline, failure["proposal_peak_MW"])
        admitted = reference_pulse_power(baseline, baseline, failure["admitted_peak_MW"])
        metrics = {"reference": trajectory(baseline, baseline, initial, 60, asset),
                   "requested": trajectory(*requested, initial, 60, asset),
                   "admitted": trajectory(*admitted, initial, 60, asset)}
        if (failure["known_commitment_failure"] or max(metrics["reference"]["unit_bound_violation_MWh"]) > 1e-8
                or max(metrics["admitted"]["unit_bound_violation_MWh"]) > 1e-8
                or metrics["requested"]["power_bound_violation_MW"] > 1e-8):
            raise ValueError("first failure has another mechanism; do not label it delayed inventory replenishment")
        depleted = [i + 1 for i, v in enumerate(metrics["requested"]["unit_bound_violation_MWh"]) if v > 1e-8]
        if not depleted or min(metrics["requested"]["minimum_SOC_MWh"]) >= -1e-8:
            raise ValueError("requested trajectory does not reproduce depletion")
        raw = next(r for r in summary["runs"] if r["profile_id"] == profile["id"]
                   and r["dispatch_rule"] == "full_request_reference")
        prior = [r for r in nominations if r["profile_id"] == profile["id"]
                 and r["dispatch_rule"] == "conservative_admission" and minute - 90 <= r["minute"] < minute]
        rows.append({"profile_id": profile["id"], "first_request_deficit_minute": minute,
                     "direction": failure["direction"], "initial_SOC_MWh": initial.tolist(),
                     "locked_PN_MW": failure["locked_PN_MW"], "requested_peak_MW": failure["proposal_peak_MW"],
                     "admitted_peak_MW": failure["admitted_peak_MW"], "depleted_units_if_full_request": depleted,
                     "required_starting_SOC_for_requested_path_MWh": np.maximum(
                         initial - metrics["requested"]["minimum_SOC_MWh"], 0.).tolist(),
                     "unmet_increase_MWh": failure["unmet_increase_MWh"],
                     "unmet_decrease_MWh": failure["unmet_decrease_MWh"], "paths": metrics,
                     "preceding_nominations": prior,
                     "full_request_first_physical_failure": raw["first_physical_failure"],
                     "mechanism": "the prior export drains unit 1 before its delayed recharge starts; the next absolute export overrides that recharge during the pulse, and the later safe reference tail cannot prevent earlier depletion"})
    result = {"protocol_id": protocol["protocol_id"], "diagnosed_profiles": len(rows),
              "additional_saved_state_path_evaluations": 3 * len(rows), "controller_window_evaluations": 0,
              "rows": rows, "source_comparison_gate": "HOLD", "confirmation_year_access": False,
              "decision": "test_whether_one_earlier_PN_nomination_can_avoid_the_saved_first_failure_before_designing_a_new_causal_rule"}
    (OUTPUT / "first_failure_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    for row in rows:
        print(json.dumps({k: row[k] for k in ("profile_id", "initial_SOC_MWh", "admitted_peak_MW",
                                             "required_starting_SOC_for_requested_path_MWh")}))


if __name__ == "__main__":
    inspect()
