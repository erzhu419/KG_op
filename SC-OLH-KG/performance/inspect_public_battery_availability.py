#!/usr/bin/env python3
"""Run only the frozen zero-baseline availability component fixtures."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from problems.public_battery_availability import declare_pulse, evaluate_pulse

PROTOCOL = ROOT / "performance/manifests/public_battery_availability_contract_preflight_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_availability_contract_preflight_v1_20260930"


def inspect():
    protocol = json.loads(PROTOCOL.read_text())
    fixture = protocol["fixtures"]
    asset = {"power_MW": protocol["retained_asset"]["shared_gross_power_MW"],
             **{k: protocol["retained_asset"][k] for k in
                ("unit_energy_MWh", "charge_efficiency", "discharge_efficiency")}}
    if fixture["baseline_MW"] != [0., 0.]:
        raise ValueError("this frozen component supports zero baselines only")
    rows, traces = [], []
    for state_index, initial in enumerate(fixture["states_MWh"]):
        for direction in fixture["directions"]:
            obligation = np.array(fixture["obligation_MW"]) * (1 if direction == "export" else -1)
            declaration = declare_pulse(initial, obligation, asset)
            result = evaluate_pulse(declaration, obligation, obligation, asset)
            row = {"fixture": f"stock_{state_index}_{direction}", "initial_SOC_MWh": initial,
                   "guaranteed_MW": declaration["guaranteed_MW"].tolist(),
                   **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
                      if k not in ("SOC_trace_MWh", "power_first_MW", "power_last_MW", "physics")},
                   **result["physics"]}
            rows.append(row)
            traces.append({"fixture": row["fixture"],
                           **{k: result[k].tolist() for k in
                              ("SOC_trace_MWh", "power_first_MW", "power_last_MW")}})
    summary = {"protocol_id": protocol["protocol_id"],
               "status": "availability_component_preflight_complete" if all(r["physical_feasible"] for r in rows)
                         else "availability_component_physics_failed",
               "stock_scenarios": len(fixture["states_MWh"]), "pulse_replays": len(rows),
               "physically_feasible_pulses": sum(r["physical_feasible"] for r in rows),
               "service_successes": sum(r["service_success"] for r in rows),
               "service_failures": sum(not r["service_success"] for r in rows),
               "fixtures": rows, "source_comparison_gate": protocol["source_comparison_gate"],
               **protocol["accounting"],
               "decision": "specify_nonzero_baselines_and_pending_commitments_before_a_bounded_pilot",
               "limitations": fixture["scope"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "fixtures"}), flush=True)


if __name__ == "__main__":
    inspect()
