#!/usr/bin/env python3
"""Evaluate the preregistered lower-envelope frontend promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

try:
    from paper_result_audit import extract_result_record
except ModuleNotFoundError:
    from .paper_result_audit import extract_result_record


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_regret(record):
    value = record.get("feasible_regret")
    return None if value is None else float(value)


def _empirical_false_certificate_count(traffic):
    return sum(
        bool(row.get("deployed_certified"))
        and float(row.get("deployed_feasible_probability", 0.0)) < 0.95
        for row in traffic.get("rows", ())
    )


def evaluate_traffic(manifest, traffic):
    contract = manifest["traffic_development_gate"]
    information = traffic.get("information_contract", {})
    failures = []
    observed_seeds = sorted(map(int, traffic.get("seeds", ())))
    expected_seeds = sorted(map(int, contract["search_seeds"]))
    if traffic.get("status") != "complete":
        failures.append("traffic audit is incomplete")
    if observed_seeds != expected_seeds:
        failures.append("traffic development seeds differ from registration")
    if information.get("evidence_phase") != contract["evidence_phase"]:
        failures.append("traffic evidence phase differs from registration")
    if (
        information.get("source_selection_mode")
        != contract["source_selection_mode"]
    ):
        failures.append("traffic source-selection mode drifted")
    if int(traffic.get("source_calls_per_run", -1)) != int(
        contract["source_calls"]
    ):
        failures.append("traffic source budget drifted")
    if int(traffic.get("target_initial_design_calls_per_run", -1)) != int(
        contract["n0"]
    ):
        failures.append("traffic n0 drifted")
    if int(traffic.get("target_search_calls_per_run", -1)) != int(
        contract["target_search_calls"]
    ):
        failures.append("traffic search budget drifted")
    if traffic.get("policy_vectors_exported") is not False:
        failures.append("traffic release audit exported policy vectors")
    if information.get("target_oracle_used") is not False:
        failures.append("traffic development used the target oracle")
    if information.get("historical_target_anchor_used") is not False:
        failures.append("traffic development used a historical target anchor")

    empirical_false = _empirical_false_certificate_count(traffic)
    certified = int(traffic.get("certified_seed_count", 0))
    outcome_pass = bool(
        certified >= int(contract["minimum_certified_seed_count"])
        and empirical_false
        <= int(contract["maximum_empirical_false_certificate_count"])
    )
    return {
        "status": "pass" if not failures and outcome_pass else "fail",
        "contract_failures": failures,
        "certified_seed_count": certified,
        "seed_count": len(observed_seeds),
        "empirical_false_certificate_count": int(empirical_false),
        "minimum_certified_seed_count": int(
            contract["minimum_certified_seed_count"]),
        "outcome_pass": outcome_pass,
    }


def evaluate_synthetic(manifest, records):
    contract = manifest["synthetic_noninferiority_gate"]
    domains = tuple(map(str, contract["domains"]))
    backends = tuple(map(str, contract["backends"]))
    seeds = tuple(map(int, contract["seeds"]))
    expected = {
        (frontend, backend, domain, seed)
        for frontend in ("v1", "lower_envelope_v2")
        for backend in backends
        for domain in domains
        for seed in seeds
    }
    by_key = {}
    duplicates = []
    for source in records:
        row = dict(source)
        key = (
            str(row["frontend"]),
            str(row["backend"]),
            str(row["domain"]),
            int(row["seed"]),
        )
        if key in by_key:
            duplicates.append(key)
        by_key[key] = row
    missing = sorted(expected - set(by_key))
    unexpected = sorted(set(by_key) - expected)
    cells = {}
    global_failures = []
    if duplicates:
        global_failures.append("duplicate synthetic result cells")
    if missing:
        global_failures.append("missing synthetic result cells")
    if unexpected:
        global_failures.append("unexpected synthetic result cells")

    for backend in backends:
        for domain in domains:
            baseline = [
                by_key.get(("v1", backend, domain, seed)) for seed in seeds
            ]
            challenger = [
                by_key.get(("lower_envelope_v2", backend, domain, seed))
                for seed in seeds
            ]
            complete = all(row is not None for row in (*baseline, *challenger))
            all_ok = bool(
                complete
                and all(row.get("status") == "ok" for row in (*baseline, *challenger))
            )
            baseline_feasible = sum(
                bool(row and row.get("true_feasible")) for row in baseline
            )
            challenger_feasible = sum(
                bool(row and row.get("true_feasible")) for row in challenger
            )
            baseline_false = sum(
                bool(row and row.get("false_certificate")) for row in baseline
            )
            challenger_false = sum(
                bool(row and row.get("false_certificate")) for row in challenger
            )
            regret_deltas = []
            contract_mismatches = []
            for base, new in zip(baseline, challenger):
                if base is None or new is None:
                    continue
                for field in (
                    "source_archive_fingerprint",
                    "problem_contract_fingerprint",
                    "verifier_signature",
                    "source_calls",
                    "target_initial_calls",
                    "target_search_calls",
                ):
                    if base.get(field) != new.get(field):
                        contract_mismatches.append({
                            "seed": int(base["seed"]),
                            "field": field,
                        })
                if base.get("true_feasible") and new.get("true_feasible"):
                    base_regret = _finite_regret(base)
                    new_regret = _finite_regret(new)
                    if base_regret is not None and new_regret is not None:
                        regret_deltas.append(new_regret - base_regret)
            median_regret_increase = (
                None if not regret_deltas else float(statistics.median(regret_deltas))
            )
            feasibility_pass = bool(
                not contract["require_no_feasibility_count_loss"]
                or challenger_feasible >= baseline_feasible
            )
            regret_pass = bool(
                median_regret_increase is not None
                and median_regret_increase
                <= float(contract["maximum_median_paired_regret_increase"])
            )
            false_certificate_pass = bool(
                challenger_false - baseline_false
                <= int(contract["maximum_false_certificate_count_increase"])
            )
            cell_pass = bool(
                complete
                and all_ok
                and not contract_mismatches
                and feasibility_pass
                and regret_pass
                and false_certificate_pass
            )
            cells[f"{backend}/{domain}"] = {
                "status": "pass" if cell_pass else "fail",
                "complete": complete,
                "all_rows_ok": all_ok,
                "contract_mismatches": contract_mismatches,
                "baseline_true_feasible_count": baseline_feasible,
                "challenger_true_feasible_count": challenger_feasible,
                "baseline_false_certificate_count": baseline_false,
                "challenger_false_certificate_count": challenger_false,
                "median_paired_regret_increase": median_regret_increase,
                "feasibility_pass": feasibility_pass,
                "regret_pass": regret_pass,
                "false_certificate_pass": false_certificate_pass,
            }
    all_cells_pass = bool(
        cells and all(row["status"] == "pass" for row in cells.values())
    )
    return {
        "status": (
            "pass" if not global_failures and all_cells_pass else "fail"
        ),
        "global_failures": global_failures,
        "expected_record_count": len(expected),
        "observed_record_count": len(records),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "cells": cells,
    }


def evaluate_gate(manifest, traffic, records):
    traffic_gate = evaluate_traffic(manifest, traffic)
    synthetic_gate = evaluate_synthetic(manifest, records)
    promote = bool(
        traffic_gate["status"] == "pass"
        and synthetic_gate["status"] == "pass"
    )
    return {
        "schema_version": 1,
        "gate_id": manifest["gate_id"],
        "status": "complete",
        "traffic_development_gate": traffic_gate,
        "synthetic_noninferiority_gate": synthetic_gate,
        "promote_lower_envelope_v2": promote,
        "decision": (
            manifest["promotion_rule"]["success_action"]
            if promote
            else manifest["promotion_rule"]["failure_action"]
        ),
        "saas_used": False,
        "gpu_used": False,
    }


def load_synthetic_records(root, manifest):
    root = Path(root)
    contract = manifest["synthetic_noninferiority_gate"]
    records = []
    for frontend in ("v1", "lower_envelope_v2"):
        for backend in contract["backends"]:
            for domain in contract["domains"]:
                for seed in contract["seeds"]:
                    path = (
                        root / frontend / backend / domain
                        / f"seed{int(seed):04d}" / "result.json"
                    )
                    if not path.is_file():
                        continue
                    record = extract_result_record(
                        path,
                        track_id="lower_envelope_synthetic_gate",
                    )
                    record.update({
                        "frontend": frontend,
                        "backend": backend,
                    })
                    records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--traffic-development", required=True)
    parser.add_argument("--synthetic-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = _read_json(args.manifest)
    traffic = _read_json(args.traffic_development)
    records = load_synthetic_records(args.synthetic_root, manifest)
    payload = evaluate_gate(manifest, traffic, records)
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
