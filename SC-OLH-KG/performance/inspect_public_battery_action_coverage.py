#!/usr/bin/env python3
"""Frozen original-library action coverage at recoverable reference states."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from performance.inspect_independent_energy_data import instant, write_csv
from performance.inspect_public_battery_controller import calibration, load_signal, policy_pool
from performance.inspect_public_battery_partition import cached_unit
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_unit_feasibility import unit_physical_oracle
from problems.public_battery_units import nominate_units, unit_planning_terms

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_action_coverage_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_action_coverage_v1_20260930"
INFEASIBLE = {"forced_power_infeasible", "energy_or_commitment_infeasible"}


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / protocol["controller_protocol"]).read_text())
    physical = json.loads((ROOT / protocol["physical_oracle_protocol"]).read_text())
    basis = ROOT / protocol["basis_recoverability_outputs"]
    period = protocol["roster"]["period"]
    with (basis / "windows.csv").open() as handle:
        recovery = {int(r["window_start_minute"]): r for r in csv.DictReader(handle)
                    if r["method"] == "V3" and r["period"] == period}
    contexts = {r["window_start_minute"]: r["context"]
                for r in json.loads((basis / "failure_contexts.json").read_text())
                if r["method"] == "V3" and r["period"] == period}
    horizon = base["windows"]["hours"] * 60
    case = next(c for c in base["development"] if c["start"][:10] == period)
    start, stop = instant(case["start"]), instant(case["stop"])
    units = [cached_unit(base, case, u) for u in (0, 1)]
    workload = site_workload(units, start, stop)
    expected_starts = set(range(0, len(workload["active"]) - horizon + 1,
                                base["windows"]["start_stride_hours"] * 60))
    if set(recovery) != expected_starts or set(contexts) != expected_starts:
        raise ValueError("action coverage must retain every frozen April reference window")
    selected = []
    for window in sorted(expected_starts):
        kind = "current" if recovery[window]["current_status"] == "replayed_feasible" else "previous"
        if recovery[window][f"{kind}_status"] != "replayed_feasible":
            raise ValueError("selected reference snapshot was not replayed recoverable")
        selected.append((window, kind, contexts[window] if kind == "current" else contexts[window]["previous_nomination"]))
    counts = Counter(kind for _, kind, _ in selected)
    if len(selected) != protocol["roster"]["reference_windows"] or any(
            counts[kind] != protocol["roster"][f"selected_{kind}_snapshots"] for kind in ("current", "previous")):
        raise ValueError("selected snapshot roster differs from the frozen protocol")
    asset, capacity = base["asset"], np.asarray(base["asset"]["unit_energy_MWh"])
    terms = unit_planning_terms(receipt_plans(units, start, len(workload["active"])), asset)
    stress, _ = load_signal(base, case, calibration(base))
    ids, nodes, points = policy_pool(base)
    library = [i for i, (_, _, inside) in enumerate(ids) if inside]
    if len(library) != protocol["roster"]["library_candidates_per_snapshot"]:
        raise ValueError("original library roster changed")
    actions, windows, negatives, probes, witnesses = [], [], [], [], {}
    cache_hits = 0
    for window, kind, snapshot in selected:
        relative = snapshot["nomination_decision_minute"]
        absolute = window + relative
        decisions = np.array([absolute // 30])
        soc = np.array(snapshot["decision_SOC_MWh"])[None, None, :]
        pending = np.array(snapshot["pending_MW"])
        current, following = pending[0][None, None, :], pending[1][None, None, :]
        theta = np.array([np.interp(stress[(absolute + 60) // 30], nodes, points[i]) for i in library])
        commands = nominate_units(soc, current, following, theta[:, None, None] * capacity,
                                  decisions, terms, asset)[:, 0]
        reference = np.array(snapshot["nomination_MW"])
        recomputed = nominate_units(soc, current, following, .5 * capacity,
                                    decisions, terms, asset)[0, 0]
        if np.max(np.abs(reference - recomputed)) > physical["replay"]["power_tolerance_MW"]:
            raise ValueError("nomination reconstruction does not reproduce the reference action")
        cache = {}

        def probe(command):
            nonlocal cache_hits
            pair = tuple(command)
            if pair in cache:
                cache_hits += 1
                return (*cache[pair], True)
            controls = {**physical["controls"],
                        "initial_soc_fraction": (soc[0, 0] / capacity).tolist(),
                        "initial_nomination_minutes": 90,
                        "initial_nomination_MW": np.vstack((pending, command)).tolist()}
            report, powers = unit_physical_oracle(workload, absolute, horizon - relative,
                                                  {**physical, "controls": controls})
            key = f"w{window}_{kind}_a{len(cache)}"
            probes.append({"probe_id": key, "window_start_minute": window, "snapshot": kind,
                           "decision_relative_minute": relative, "decision_absolute_minute": absolute,
                           "remaining_minutes": horizon - relative,
                           "actual_initial_SOC_MWh": snapshot["decision_SOC_MWh"],
                           "locked_nominations_MW": controls["initial_nomination_MW"], **report})
            witness = key if powers is not None else ""
            if powers is not None:
                witnesses[key] = powers
            cache[pair] = report, key, witness
            return report, key, witness, False

        report, key, witness, reused = probe(reference)
        negatives.append({"window_start_minute": window, "snapshot": kind,
                          "nomination_unit_1_MW": float(reference[0]), "nomination_unit_2_MW": float(reference[1]),
                          "status": report["status"], "expected_infeasible": report["status"] in INFEASIBLE,
                          "probe_id": key, "witness_id": witness})
        feasible_ids, unresolved_ids = [], []
        for i, fraction, command in zip(library, theta, commands):
            report, key, witness, reused = probe(command)
            name, family, _ = ids[i]
            if report["status"] == "replayed_feasible":
                feasible_ids.append(name)
            elif report["status"] not in INFEASIBLE:
                unresolved_ids.append(name)
            actions.append({"window_start_minute": window, "snapshot": kind,
                            "decision_relative_minute": relative, "decision_absolute_minute": absolute,
                            "profile_id": name, "family": family, "target_fraction": float(fraction),
                            "nomination_unit_1_MW": float(command[0]), "nomination_unit_2_MW": float(command[1]),
                            "status": report["status"], "probe_id": key, "witness_id": witness,
                            "reused_identical_nomination": reused})
        windows.append({"window_start_minute": window, "snapshot": kind,
                        "decision_relative_minute": relative, "feasible_candidate_actions": len(feasible_ids),
                        "unresolved_candidate_actions": len(unresolved_ids),
                        "feasible_profile_ids": json.dumps(feasible_ids), "unresolved_profile_ids": json.dumps(unresolved_ids)})
        print(json.dumps({"windows_completed": len(windows), "unique_probes": len(probes),
                          "window_start_minute": window, "snapshot": kind,
                          "feasible_candidates": len(feasible_ids), "unresolved_candidates": len(unresolved_ids),
                          "reference_control": negatives[-1]["status"]}), flush=True)
    if len(actions) != protocol["roster"]["candidate_action_evaluations"] or len(negatives) != protocol["roster"]["reference_action_negative_controls"]:
        raise ValueError("completed action roster differs from frozen counts")
    unresolved = sum(r["unresolved_candidate_actions"] for r in windows)
    bad_controls = sum(not r["expected_infeasible"] for r in negatives)
    zero = [r["window_start_minute"] for r in windows if not r["feasible_candidate_actions"]]
    profile_counts = {ids[i][0]: sum(a["profile_id"] == ids[i][0] and a["status"] == "replayed_feasible" for a in actions) for i in library}
    complete = not (unresolved or bad_controls)
    decision = ("resolve_action_or_negative_control_discrepancies" if not complete else
                "address_original_library_action_gaps" if zero else "address_causal_selection_and_cross_window_coherence")
    summary = {"protocol_id": protocol["protocol_id"],
               "status": "action_coverage_diagnostic_complete" if complete else "action_coverage_diagnostic_unresolved",
               "period": period, "reference_windows": len(windows), "snapshot_counts": dict(counts),
               "candidate_action_evaluations": len(actions), "candidate_action_statuses": dict(Counter(a["status"] for a in actions)),
               "negative_controls": len(negatives), "negative_control_statuses": dict(Counter(r["status"] for r in negatives)),
               "unexpected_negative_controls": bad_controls, "unresolved_candidate_actions": unresolved,
               "covered_windows": sum(r["feasible_candidate_actions"] > 0 for r in windows),
               "zero_coverage_window_starts": zero,
               "minimum_feasible_candidates_per_window": min(r["feasible_candidate_actions"] for r in windows),
               "maximum_feasible_candidates_per_window": max(r["feasible_candidate_actions"] for r in windows),
               "snapshot_coverage": [{"snapshot": kind,
                   "windows": sum(r["snapshot"] == kind for r in windows),
                   "covered_windows": sum(r["snapshot"] == kind and r["feasible_candidate_actions"] > 0 for r in windows)}
                   for kind in ("current", "previous")],
               "zero_coverage_absolute_decision_minutes": sorted({a["decision_absolute_minute"] for a in actions if a["window_start_minute"] in zero}),
               "profile_feasible_action_counts": profile_counts,
               "profiles_covering_every_reference_snapshot": [name for name, count in profile_counts.items() if count == len(windows)],
               "unique_action_probes": len(probes), "identical_nomination_cache_hits": cache_hits,
               "diagnostic_LP_solves": sum(p["LP_solved"] for p in probes),
               "diagnostic_physical_witness_replays": sum(p["witness_replayed"] for p in probes),
               "maximum_feasible_witness_residuals": {name: max(p[name] for p in probes if p["status"] == "replayed_feasible")
                   for name in ("maximum_inventory_bound_violation_MWh", "maximum_gross_power_violation_MW",
                                "maximum_nomination_power_violation_MW", "maximum_initial_nomination_deviation_MW")},
               "scipy_version": scipy.__version__, "decision": decision, "source_comparison_gate": "HOLD",
               **protocol["accounting"], "limitations": protocol["limitations"]}
    for name, rows in (("actions", actions), ("windows", windows), ("negative_controls", negatives)):
        write_csv(out / f"{name}.csv", rows)
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
