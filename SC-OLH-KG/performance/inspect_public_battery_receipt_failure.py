#!/usr/bin/env python3
"""Diagnose the receipt-forecast pilot's first unmet call from saved records."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from problems.public_battery_pending_availability import reference_pulse_power
from problems.public_battery_visibility import trajectory

PROTOCOL = ROOT / "performance/manifests/public_battery_receipt_forecast_pilot_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_receipt_forecast_pilot_v1_20261001"


def inspect():
    protocol = json.loads(PROTOCOL.read_text())
    base = json.loads((ROOT / protocol["basis_protocol"]).read_text())
    original = json.loads((ROOT / base["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": original["shared_gross_power_MW"],
             **{k: original[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    summary = json.loads((OUTPUT / "summary.json").read_text())
    hours = json.loads((OUTPUT / "hourly_dispatch.json").read_text())
    nominations = json.loads((OUTPUT / "nominations.json").read_text())
    rows = []
    for profile in protocol["sequential_pilot"]["profiles"]:
        selected = [h for h in hours if h["profile_id"] == profile["id"] and h["dispatch_rule"] == "conservative_admission"]
        failure = next(h for h in selected if "request_shortfall" in h["service_failure_reasons"])
        minute = failure["minute"]
        prior = [n for n in nominations if n["profile_id"] == profile["id"]
                 and n["dispatch_rule"] == "conservative_admission" and minute - 90 <= n["minute"] < minute]
        forecast = next(n for n in prior if n["minute"] == minute - 30)
        prefix_nomination = next(n for n in prior if n["minute"] == minute - 90)
        baseline = np.repeat(failure["locked_PN_MW"], 30, axis=0)
        initial = np.array(failure["initial_SOC_MWh"])
        requested = reference_pulse_power(baseline, baseline, failure["proposal_peak_MW"])
        admitted = reference_pulse_power(baseline, baseline, failure["admitted_peak_MW"])
        metrics = {"reference": trajectory(baseline, baseline, initial, 60, asset),
                   "requested": trajectory(*requested, initial, 60, asset),
                   "admitted": trajectory(*admitted, initial, 60, asset)}
        if (forecast["receipt_cue"]["direction"] != failure["direction"]
                or metrics["requested"]["power_bound_violation_MW"] > 1e-8
                or max(metrics["reference"]["unit_bound_violation_MWh"]) > 1e-8
                or max(metrics["admitted"]["unit_bound_violation_MWh"]) > 1e-8
                or max(metrics["requested"]["maximum_SOC_MWh"]) <= 98. + 1e-8):
            raise ValueError("first failure does not support the expected correct-but-late forecast diagnosis")
        raw = next(r for r in summary["runs"] if r["profile_id"] == profile["id"] and r["dispatch_rule"] == "full_request_reference")
        rows.append({"profile_id": profile["id"], "first_unmet_request_minute": minute,
                     "initial_SOC_MWh": initial.tolist(), "locked_PN_MW": failure["locked_PN_MW"],
                     "direction": failure["direction"], "admitted_peak_MW": failure["admitted_peak_MW"],
                     "latest_forecast_direction_correct": True,
                     "forecast_nomination_minute": forecast["minute"], "forecast_delivery_minute": forecast["delivery_minute"],
                     "earlier_tail_nomination": prefix_nomination,
                     "late_forecast_nomination": forecast,
                     "maximum_starting_SOC_for_full_requested_path_MWh": (np.array(asset["unit_energy_MWh"])
                         - np.array(metrics["requested"]["maximum_SOC_MWh"]) + initial).tolist(),
                     "paths": metrics, "full_request_first_physical_failure": raw["first_physical_failure"],
                     "mechanism": "correct import forecast, but the new release PN starts during the final part of the next pulse; inventory exceeds its ceiling before that recovery can act. The earlier free PN slot was committed when its public cue was empty"})
    result = {"protocol_id": protocol["protocol_id"], "diagnosed_profiles": len(rows),
              "additional_saved_state_path_evaluations": 3 * len(rows), "controller_window_evaluations": 0,
              "rows": rows, "source_comparison_gate": "HOLD", "confirmation_year_access": False,
              "decision": "derive_a_two_call_information_necessity_bound_before_any_further_controller_change"}
    (OUTPUT / "first_failure_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    for row in rows:
        print(json.dumps({k: row[k] for k in ("profile_id", "initial_SOC_MWh", "admitted_peak_MW",
                                             "maximum_starting_SOC_for_full_requested_path_MWh")}))


if __name__ == "__main__":
    inspect()
