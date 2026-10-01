#!/usr/bin/env python3
"""The six frozen nonzero-PN and locked-plan availability fixtures."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from problems.public_battery_pending_availability import declare_pending, evaluate_pending

PROTOCOL = ROOT / "performance/manifests/public_battery_availability_pending_preflight_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_availability_pending_preflight_v1_20260930"


def inspect():
    protocol = json.loads(PROTOCOL.read_text())
    original = json.loads((ROOT / protocol["retained_asset_protocol"]).read_text())["retained_asset"]
    asset = {"power_MW": original["shared_gross_power_MW"],
             **{k: original[k] for k in ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    rows, traces, path_evaluations = [], [], 0
    for fixture in protocol["fixtures"]:
        declaration = declare_pending(fixture["initial_SOC_MWh"], fixture["locked_baseline_MW"],
                                      fixture["service_capacity_MW"], fixture["direction"], asset)
        proposal = None if fixture["request"] is None else fixture["request"]["peak_MW"]
        result = evaluate_pending(declaration, proposal, fixture["service_capacity_MW"], asset)
        path_evaluations += declaration["envelope_path_evaluations"]
        row = {"fixture": fixture["id"], "initial_SOC_MWh": fixture["initial_SOC_MWh"],
               "locked_baseline_MW": fixture["locked_baseline_MW"], "direction": fixture["direction"],
               "declaration_reason": declaration["reason"],
               "available_capacity_MW": declaration["available_capacity_MW"].tolist(),
               "guaranteed_peak_MW": None if declaration["guaranteed_peak_MW"] is None
                                      else declaration["guaranteed_peak_MW"].tolist(),
               **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
                  if k not in ("physics", "SOC_trace_MWh", "power_first_MW", "power_last_MW")},
               **result["physics"]}
        rows.append(row)
        traces.append({"fixture": fixture["id"], **{k: result[k].tolist() for k in
                      ("SOC_trace_MWh", "power_first_MW", "power_last_MW")}})
    summary = {"protocol_id": protocol["protocol_id"], "status": "pending_availability_component_complete",
               "fixture_replays": len(rows), "physically_feasible_fixtures": sum(r["physical_feasible"] for r in rows),
               "service_successes": sum(r["service_success"] for r in rows),
               "service_failures": sum(not r["service_success"] for r in rows),
               "known_commitment_failures": sum(r["known_commitment_failure"] for r in rows),
               "envelope_path_evaluations": path_evaluations, "fixtures": rows,
               "source_comparison_gate": protocol["source_comparison_gate"], **protocol["accounting"],
               "decision": "define_already_received_BOA_commitments_before_a_bounded_causal_pilot",
               "limitations": "six component cases with two locked PN half-hours; no prior BOA commitments or historical-week reliability claim; common capacity scaling is a simulation admission rule"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "fixtures"}), flush=True)


if __name__ == "__main__":
    inspect()
