#!/usr/bin/env python3
"""Frozen exact continuous shared-target action coverage at reference states."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_action_continuum import (
    nomination_for_fraction, scalar_action_segments,
    segment_fraction_at_theta, theta_at_segment_fraction,
)
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_unit_feasibility import unit_physical_oracle
from problems.public_battery_units import unit_planning_terms

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_scalar_action_continuum_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_scalar_action_continuum_v1_20260930"
INFEASIBLE = {"forced_power_infeasible", "energy_or_commitment_infeasible"}


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / protocol["controller_protocol"]).read_text())
    physical = json.loads((ROOT / protocol["physical_oracle_protocol"]).read_text())
    prior = json.loads((ROOT / protocol["basis_action_coverage_outputs"] / "summary.json").read_text())
    starts = prior["zero_coverage_window_starts"]
    if prior["status"] != "action_coverage_diagnostic_complete" or starts != list(range(0, 1321, 60)):
        raise ValueError("prior evidence does not match the frozen zero-coverage roster")
    period = protocol["roster"]["period"]
    contexts = {r["window_start_minute"]: r["context"] for r in json.loads(
        (ROOT / protocol["basis_recoverability_outputs"] / "failure_contexts.json").read_text())
        if r["method"] == "V3" and r["period"] == period}
    case = next(c for c in base["development"] if c["start"][:10] == period)
    start, stop = instant(case["start"]), instant(case["stop"])
    units = [cached_unit(base, case, u) for u in (0, 1)]
    workload = site_workload(units, start, stop)
    asset = base["asset"]
    terms = unit_planning_terms(receipt_plans(units, start, len(workload["active"])), asset)
    capacity = np.asarray(asset["unit_energy_MWh"])
    tolerance = physical["replay"]["power_tolerance_MW"]
    horizon = base["windows"]["hours"] * 60
    rows, windows, probes, witnesses = [], [], [], {}
    for window in starts:
        state = contexts[window]
        relative = state["nomination_decision_minute"]
        absolute = window + relative
        if absolute != protocol["roster"]["absolute_nomination_decision_minute"]:
            raise ValueError("selected snapshot is outside the frozen common nomination event")
        decision = absolute // 30
        reference_error = float(np.max(np.abs(nomination_for_fraction(.5, state, decision, terms, asset)
                                              - np.asarray(state["nomination_MW"]))))
        if reference_error > tolerance:
            raise ValueError("original rule does not reconstruct the saved reference action")
        segments = scalar_action_segments(state, decision, terms, asset)
        if len(segments) > protocol["roster"]["maximum_segments_per_snapshot"]:
            raise ValueError("analytic segment count exceeds the frozen bound")
        feasible, unresolved = [], []
        max_construction_error = 0.
        controls = {**physical["controls"],
                    "initial_soc_fraction": (np.asarray(state["decision_SOC_MWh"]) / capacity).tolist(),
                    "initial_nomination_minutes": 60, "initial_nomination_MW": state["pending_MW"]}
        for i, segment in enumerate(segments):
            endpoints = np.asarray(segment["nomination_endpoints_MW"])
            if np.any((endpoints[0] < -tolerance) & (endpoints[1] > tolerance)) or np.any(
                    (endpoints[1] < -tolerance) & (endpoints[0] > tolerance)):
                raise ValueError("action segment crosses an unsplit nomination sign change")
            construction_error = 0.
            for position in (0., .25, .5, .75, 1.):
                theta = segment["theta_left"] + position * (segment["theta_right"] - segment["theta_left"])
                fraction = segment_fraction_at_theta(theta, segment)
                expected = endpoints[0] + fraction * (endpoints[1] - endpoints[0])
                actual = nomination_for_fraction(theta, state, decision, terms, asset)
                construction_error = max(construction_error, float(np.max(np.abs(actual - expected))))
            if construction_error > tolerance:
                raise ValueError("analytic segment does not agree with the original nomination rule")
            max_construction_error = max(max_construction_error, construction_error)
            report, powers = unit_physical_oracle(workload, absolute, horizon - relative,
                                                  {**physical, "controls": controls}, endpoints)
            key = f"w{window}_segment{i}"
            theta = None
            rule_error = None
            if powers is not None:
                theta = theta_at_segment_fraction(report["nomination_segment_fraction"], segment)
                expected = nomination_for_fraction(theta, state, decision, terms, asset)
                rule_error = float(np.max(np.abs(expected - powers[2])))
                if rule_error > tolerance:
                    report["status"] = "scalar_rule_witness_unresolved"
                else:
                    witnesses[key] = powers
            if report["status"] == "replayed_feasible":
                feasible.append(i)
            elif report["status"] not in INFEASIBLE:
                unresolved.append(i)
            rows.append({"window_start_minute": window, "segment_index": i,
                         "theta_left": segment["theta_left"], "theta_right": segment["theta_right"],
                         "shared_power_normalized": segment["shared_power_normalized"],
                         "status": report["status"], "construction_error_MW": construction_error,
                         "witness_theta": theta, "original_rule_deviation_MW": rule_error,
                         "probe_id": key, "witness_id": key if key in witnesses else ""})
            probes.append({"probe_id": key, "window_start_minute": window,
                           "decision_relative_minute": relative, "decision_absolute_minute": absolute,
                           "remaining_minutes": horizon - relative,
                           "actual_initial_SOC_MWh": state["decision_SOC_MWh"],
                           "locked_pending_MW": state["pending_MW"], **segment, **report,
                           "reconstructed_theta": theta, "maximum_original_rule_deviation_MW": rule_error})
        classification = "finite_library_sampling_gap" if feasible else "unresolved" if unresolved else "shared_target_action_gap"
        windows.append({"window_start_minute": window, "analytic_segments": len(segments),
                        "feasible_segments": len(feasible), "unresolved_segments": len(unresolved),
                        "classification": classification, "reference_reconstruction_error_MW": reference_error,
                        "maximum_construction_error_MW": max_construction_error})
        print(json.dumps({"windows_completed": len(windows), "segment_probes": len(probes),
                          "window_start_minute": window, "classification": classification}), flush=True)
    if len(windows) != protocol["roster"]["snapshots"] or len(probes) > protocol["roster"]["maximum_diagnostic_LP_calls"]:
        raise ValueError("completed continuum diagnostic differs from frozen roster or budget")
    unresolved = sum(r["unresolved_segments"] for r in windows)
    gap = sum(r["classification"] == "shared_target_action_gap" for r in windows)
    summary = {"protocol_id": protocol["protocol_id"],
               "status": "scalar_continuum_diagnostic_complete" if not unresolved else "scalar_continuum_diagnostic_unresolved",
               "reference_snapshots": len(windows), "analytic_action_segments": len(rows),
               "classification_counts": dict(Counter(r["classification"] for r in windows)),
               "segment_statuses": dict(Counter(p["status"] for p in probes)),
               "unresolved_segments": unresolved, "diagnostic_LP_solves": sum(p["LP_solved"] for p in probes),
               "diagnostic_physical_witness_replays": sum(p["witness_replayed"] for p in probes),
               "original_rule_witnesses": len(witnesses),
               "maximum_construction_error_MW": max(r["maximum_construction_error_MW"] for r in windows),
               "maximum_reference_reconstruction_error_MW": max(r["reference_reconstruction_error_MW"] for r in windows),
               "reused_preceding_covered_states": prior["covered_windows"],
               "reused_preceding_negative_controls": prior["negative_controls"],
               "preceding_diagnostic_LPs_rerun": 0,
               "decision": "resolve_continuum_discrepancies" if unresolved else "assess_unit_specific_targets_or_earlier_planning" if gap else "address_finite_candidate_sampling",
               "source_comparison_gate": "HOLD", "scipy_version": scipy.__version__,
               **protocol["accounting"], "limitations": protocol["limitations"]}
    write_csv(out / "segments.csv", rows)
    write_csv(out / "windows.csv", windows)
    (out / "probes.json").write_text(json.dumps(probes, indent=2) + "\n")
    np.savez_compressed(out / "witnesses.npz", **witnesses)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inspect(json.loads(args.protocol.read_text()), args.output)
