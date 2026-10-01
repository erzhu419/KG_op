#!/usr/bin/env python3
"""Frozen suffix feasibility from actual stocks and two locked nominations."""
import argparse
from collections import Counter
import csv
from datetime import timedelta
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
from problems.public_battery_reserves import reserve_planning_terms
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_unit_feasibility import unit_physical_oracle
from problems.public_battery_units import simulate_units, unit_planning_terms

PROTOCOL = ROOT / "performance/manifests/public_battery_unit_recoverability_v1_20260930.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_unit_recoverability_v1_20260930"
INFEASIBLE = {"forced_power_infeasible", "energy_or_commitment_infeasible"}


def classify(current, previous):
    if current["status"] not in INFEASIBLE | {"replayed_feasible"} or (previous is not None and previous["status"] not in INFEASIBLE | {"replayed_feasible"}):
        return "unresolved_probe"
    if current["status"] == "replayed_feasible":
        return "unresolved_monotonicity_discrepancy" if previous is not None and previous["status"] in INFEASIBLE else "current_feasible"
    if previous is None:
        # At decision zero the state and locked zeros equal the already replayed
        # whole-window physical oracle, so a contradictory result needs repair.
        return "unresolved_initial_probe_infeasible"
    return "lost_between_snapshots" if previous["status"] == "replayed_feasible" else "already_infeasible_at_previous"


def inspect(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / protocol["controller_protocols"]["V3"]).read_text())
    reserve = json.loads((ROOT / protocol["controller_protocols"]["V4"]).read_text())
    physical = json.loads((ROOT / protocol["physical_oracle_protocol"]).read_text())
    saved = {}
    name = protocol["roster"]["profile_id"]
    for method, path in protocol["saved_controller_outputs"].items():
        with (ROOT / path / "windows.csv").open() as handle:
            saved[method] = {(r["period"], int(r["window_start_minute"])): r for r in csv.DictReader(handle) if r["profile_id"] == name}
    ids, nodes, points = policy_pool(base)
    reference = next(i for i, (profile, _, _) in enumerate(ids) if profile == name)
    fit = calibration(base)
    horizon = base["windows"]["hours"] * 60
    asset = base["asset"]
    tolerance = base["failure"]["numerical_energy_tolerance_MWh"]
    capacity = np.asarray(asset["unit_energy_MWh"])
    traces, contexts, workloads = [], [], {}
    for case in base["development"]:
        label = case["start"][:10]
        start, stop = instant(case["start"]), instant(case["stop"])
        units = [cached_unit(base, case, u) for u in (0, 1)]
        workload = site_workload(units, start, stop)
        workloads[label] = workload
        plans = receipt_plans(units, start, len(workload["active"]))
        lookback = reserve["control"]["reserve_lookback_minutes"]
        history = site_workload(units, start - timedelta(minutes=lookback), stop)
        terms = {"V3": unit_planning_terms(plans, asset),
                 "V4": reserve_planning_terms(plans, history, asset, lookback, tolerance)}
        stress, prices = load_signal(base, case, fit)
        targets = np.interp(stress, nodes, points[reference])[None, :]
        starts = np.arange(0, len(workload["active"]) - horizon + 1, base["windows"]["start_stride_hours"] * 60)
        for method in ("V3", "V4"):
            result = simulate_units(workload, prices, targets, starts, horizon, asset, terms[method],
                                    base["control"]["initial_soc_fraction"], tolerance)
            for w, s in enumerate(starts):
                old = saved[method][label, int(s)]
                success, failure = bool(result["success"][0, w]), int(result["first_failure_minute"][0, w])
                if success != (old["success"] == "True") or failure != int(old["first_failure_minute"]):
                    raise ValueError("reference instrumentation changed a saved success flag or failure minute")
                traces.append({"method": method, "period": label, "profile_id": name,
                               "window_start_minute": int(s), "success": success, "first_failure_minute": failure})
            for c in result["first_failure_contexts"]:
                relative = c["nomination_decision_minute"]
                if relative != c["failure_minute"] // 30 * 30 - 60:
                    raise ValueError("failure snapshot belongs to a different nomination")
                previous = c["previous_nomination"]
                if previous is not None and previous["nomination_decision_minute"] != relative - 30:
                    raise ValueError("previous snapshot is not the preceding decision")
                contexts.append({"method": method, "period": label, "window_start_minute": int(starts[c["window_index"]]), "context": c})
            print(json.dumps({"method": method, "period": label, "reference_windows_reproduced": len(starts),
                              "failed_windows": len(result["first_failure_contexts"])}), flush=True)
    if len(traces) != protocol["roster"]["controller_reference_window_evaluations"] or any(
            sum(c["method"] == method for c in contexts) != count for method, count in protocol["roster"]["saved_failed_reference_windows"].items()):
        raise ValueError("reference roster differs from frozen recoverability protocol")
    write_csv(out / "reference_trace.csv", traces)
    (out / "failure_contexts.json").write_text(json.dumps(contexts, indent=2) + "\n")
    probes, rows, witnesses = [], [], {}
    for record in contexts:
        context = record["context"]
        reports = {}
        for kind, snapshot in (("current", context), ("previous", context["previous_nomination"])):
            if snapshot is None:
                reports[kind] = None
                continue
            relative = snapshot["nomination_decision_minute"]
            controls = {**physical["controls"],
                        "initial_soc_fraction": (np.array(snapshot["decision_SOC_MWh"]) / capacity).tolist(),
                        "initial_nomination_MW": snapshot["pending_MW"]}
            probe_protocol = {**physical, "controls": controls}
            absolute = record["window_start_minute"] + relative
            report, powers = unit_physical_oracle(workloads[record["period"]], absolute, horizon - relative, probe_protocol)
            reports[kind] = report
            probes.append({"method": record["method"], "period": record["period"],
                "window_start_minute": record["window_start_minute"], "probe": kind,
                "decision_relative_minute": relative, "decision_absolute_minute": absolute,
                "remaining_minutes": horizon - relative, "actual_initial_SOC_MWh": snapshot["decision_SOC_MWh"],
                "locked_pending_MW": snapshot["pending_MW"], **report})
            if powers is not None:
                key = f"{record['method']}_{record['period']}_w{record['window_start_minute']}_{kind}"
                witnesses[key] = powers
            if len(probes) % 20 == 0:
                print(json.dumps({"suffix_probes_completed": len(probes), "statuses": dict(Counter(p["status"] for p in probes))}), flush=True)
        rows.append({"method": record["method"], "period": record["period"], "profile_id": name,
            "window_start_minute": record["window_start_minute"], "first_failure_minute": context["failure_minute"],
            "nomination_decision_minute": context["nomination_decision_minute"],
            "current_status": reports["current"]["status"],
            "previous_status": reports["previous"]["status"] if reports["previous"] is not None else "no_previous_decision",
            "classification": classify(reports["current"], reports["previous"])})
    cases = []
    for method in ("V3", "V4"):
        for case in base["development"]:
            label = case["start"][:10]
            selected = [r for r in rows if r["method"] == method and r["period"] == label]
            cases.append({"method": method, "period": label, "failed_reference_windows": len(selected),
                          "classification_counts": dict(Counter(r["classification"] for r in selected))})
    unresolved = sum(r["classification"].startswith("unresolved") for r in rows)
    summary = {"protocol_id": protocol["protocol_id"], "status": "recoverability_diagnostic_complete" if not unresolved else "recoverability_diagnostic_unresolved",
        "reference_window_evaluations": len(traces), "saved_reference_results_reproduced": True,
        "failed_reference_windows": len(rows), "cases": cases,
        "classification_counts": dict(Counter(r["classification"] for r in rows)),
        "suffix_probe_statuses": dict(Counter(p["status"] for p in probes)), "diagnostic_LP_solves": sum(p["LP_solved"] for p in probes),
        "diagnostic_physical_witness_replays": sum(p["witness_replayed"] for p in probes),
        "no_previous_decision_windows": sum(r["previous_status"] == "no_previous_decision" for r in rows),
        "unresolved_windows": unresolved, "scipy_version": scipy.__version__,
        "decision": "resolve_recoverability_discrepancies" if unresolved else "use_recovery_labels_before_any_new_controller",
        "new_data_requests": 0, "algorithm_optimizer_calls": 0, "terminal_verifier_calls": 0,
        "confirmation_year_access": False, "limitations": protocol["limitations"]}
    write_csv(out / "windows.csv", rows)
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
