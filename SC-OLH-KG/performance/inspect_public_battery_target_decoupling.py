#!/usr/bin/env python3
"""Frozen inverse-target reconstruction using already replayed suffixes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_action_continuum import inverse_unit_targets, nomination_for_fraction
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_units import unit_planning_terms

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_target_decoupling_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_target_decoupling_v1_20260930"


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / protocol["controller_protocol"]).read_text())
    physical = json.loads((ROOT / protocol["physical_oracle_protocol"]).read_text())
    prior = json.loads((ROOT / protocol["basis_scalar_continuum_outputs"] / "summary.json").read_text())
    if prior["status"] != "scalar_continuum_diagnostic_complete" or prior["classification_counts"] != {"shared_target_action_gap": 23}:
        raise ValueError("prior continuum result differs from the frozen reconstruction basis")
    period, basis = protocol["roster"]["period"], ROOT / protocol["basis_recoverability_outputs"]
    contexts = {r["window_start_minute"]: r["context"] for r in json.loads((basis / "failure_contexts.json").read_text())
                if r["method"] == "V3" and r["period"] == period}
    reports = {r["window_start_minute"]: r for r in json.loads((basis / "probes.json").read_text())
               if r["method"] == "V3" and r["period"] == period and r["probe"] == "current"}
    case = next(c for c in base["development"] if c["start"][:10] == period)
    start, stop = instant(case["start"]), instant(case["stop"])
    units = [cached_unit(base, case, u) for u in (0, 1)]
    workload = site_workload(units, start, stop)
    asset = base["asset"]
    terms = unit_planning_terms(receipt_plans(units, start, len(workload["active"])), asset)
    tolerance = physical["replay"]["power_tolerance_MW"]
    rows, details = [], []
    witness_path = basis / "witnesses.npz"
    with np.load(witness_path) as witnesses:
        for window in range(0, 1321, 60):
            state = contexts[window]
            absolute = window + state["nomination_decision_minute"]
            if absolute != protocol["roster"]["absolute_nomination_decision_minute"] or reports[window]["status"] != "replayed_feasible":
                raise ValueError("reference state lacks the registered feasible suffix")
            key = f"V3_{period}_w{window}_current"
            desired = witnesses[key][2]
            raw, fractions = inverse_unit_targets(desired, state, absolute // 30, terms, asset)
            actual = nomination_for_fraction(fractions, state, absolute // 30, terms, asset)
            error = float(np.max(np.abs(actual - desired)))
            represented = error <= tolerance
            rows.append({"window_start_minute": window, "decision_relative_minute": state["nomination_decision_minute"],
                         "decision_absolute_minute": absolute, "physical_witness_key": key,
                         "raw_target_fraction_unit_1": float(raw[0]), "raw_target_fraction_unit_2": float(raw[1]),
                         "target_fraction_unit_1": float(fractions[0]), "target_fraction_unit_2": float(fractions[1]),
                         "desired_nomination_unit_1_MW": float(desired[0]), "desired_nomination_unit_2_MW": float(desired[1]),
                         "reconstructed_nomination_unit_1_MW": float(actual[0]), "reconstructed_nomination_unit_2_MW": float(actual[1]),
                         "original_rule_deviation_MW": error, "saved_action_represented": represented})
            details.append({"window_start_minute": window, "actual_initial_SOC_MWh": state["decision_SOC_MWh"],
                            "locked_pending_MW": state["pending_MW"], "physical_witness_key": key,
                            "saved_physical_status": reports[window]["status"],
                            "saved_physical_witness_replayed": reports[window]["witness_replayed"],
                            "saved_energy_balance_error_MWh": reports[window]["energy_balance_error_MWh"],
                            "raw_target_fractions": raw.tolist(), "target_fractions": fractions.tolist(),
                            "desired_new_nomination_MW": desired.tolist(), "reconstructed_new_nomination_MW": actual.tolist(),
                            "original_rule_deviation_MW": error, "saved_action_represented": represented})
    if len(rows) != protocol["roster"]["snapshots"]:
        raise ValueError("reconstruction roster differs from registered count")
    represented = sum(r["saved_action_represented"] for r in rows)
    summary = {"protocol_id": protocol["protocol_id"], "status": "target_reconstruction_complete",
               "reference_snapshots": len(rows), "represented_saved_actions": represented,
               "unrepresented_saved_action_windows": [r["window_start_minute"] for r in rows if not r["saved_action_represented"]],
               "maximum_original_rule_deviation_MW": max(r["original_rule_deviation_MW"] for r in rows),
               "target_fraction_ranges": [[min(r[f"target_fraction_unit_{u}"] for r in rows),
                                            max(r[f"target_fraction_unit_{u}"] for r in rows)] for u in (1, 2)],
               "reused_physical_witnesses": len(rows), "saved_witness_file": str(witness_path.relative_to(ROOT)),
               "decision": "register_causal_unit_specific_controller_development" if represented == len(rows) else "identify_saved_witness_representation_obstructions",
               "source_comparison_gate": "HOLD", **protocol["accounting"], "limitations": protocol["limitations"]}
    write_csv(out / "reconstruction.csv", rows)
    (out / "reconstruction.json").write_text(json.dumps(details, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
