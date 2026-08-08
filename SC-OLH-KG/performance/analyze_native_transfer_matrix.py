#!/usr/bin/env python3
"""Fail-closed analysis of the native end-to-end transfer comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


CONTRACT_ID = "or_review_native_transfer_analysis_v1"
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
METHODS = (
    "safe_fpacoh_cbo",
    "rgpe_cbo",
    "stacked_transfer_gp_cbo",
    "mtgp_cbo",
    "fsbo_cbo",
    "hyperbo_cbo",
    "metabo_cbo",
    "malibo_cbo",
)
SEEDS = tuple(range(80, 100))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exact_interval(successes, total):
    if total <= 0:
        return [None, None]
    interval = binomtest(int(successes), int(total)).proportion_ci(
        confidence_level=0.95, method="exact")
    return [float(interval.low), float(interval.high)]


def _median(values):
    return None if not values else float(np.median(values))


def _mean(values):
    return None if not values else float(np.mean(values))


def _validate_row(payload, path):
    failures = []
    comparison = payload.get("comparison_contract", {})
    result = payload.get("result", {})
    verification = result.get("terminal_verification", {})
    required = {
        "status": payload.get("status"),
        "implementation": payload.get("implementation"),
        "target_initial_design": comparison.get("target_initial_design"),
        "source_calls": comparison.get("source_simulator_calls"),
        "target_dimension": comparison.get("target_dimension"),
        "n0": comparison.get("target_initial_calls_n0"),
        "target_search_calls": comparison.get("target_search_calls"),
        "source_oracle_aided": comparison.get("source_oracle_aided"),
        "source_scored_atlas_used": comparison.get(
            "source_scored_atlas_used"),
        "source_informed_initial_proposal": comparison.get(
            "source_informed_initial_proposal"),
        "source_archive_identical_across_methods": comparison.get(
            "source_archive_identical_across_methods"),
        "terminal_verification_identical_across_methods": comparison.get(
            "terminal_verification_identical_across_methods"),
        "shortlist_frozen": result.get(
            "terminal_shortlist_frozen_before_truth_metrics"),
        "verification_protocol": verification.get("protocol"),
        "verification_target_oracle_used": verification.get(
            "target_oracle_used"),
        "verification_updates_optimizer": verification.get(
            "posterior_updated_from_verification"),
    }
    expected = {
        "status": "ok",
        "implementation": "official",
        "target_initial_design": "native_source_sequential",
        "source_calls": 384,
        "target_dimension": 1000,
        "n0": 10,
        "target_search_calls": 13,
        "source_oracle_aided": False,
        "source_scored_atlas_used": False,
        "source_informed_initial_proposal": False,
        "source_archive_identical_across_methods": True,
        "terminal_verification_identical_across_methods": True,
        "shortlist_frozen": True,
        "verification_protocol": "ordered_frozen_shortlist",
        "verification_target_oracle_used": False,
        "verification_updates_optimizer": False,
    }
    for field, expected_value in expected.items():
        if required[field] != expected_value:
            failures.append({
                "kind": "contract_mismatch",
                "path": str(path),
                "field": field,
                "expected": expected_value,
                "observed": required[field],
            })
    return failures


def analyze(
    paths,
    *,
    expected_domains=DOMAINS,
    expected_methods=METHODS,
    expected_seeds=SEEDS,
):
    expected_domains = tuple(map(str, expected_domains))
    expected_methods = tuple(map(str, expected_methods))
    expected_seeds = tuple(map(int, expected_seeds))
    expected_keys = {
        (domain, method, seed, "official")
        for domain in expected_domains
        for method in expected_methods
        for seed in expected_seeds
    }
    rows = []
    failures = []
    seen = set()
    source_fingerprints = defaultdict(set)
    source_domain_sets = defaultdict(set)
    for path in map(Path, paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append({
                "kind": "unreadable_json",
                "path": str(path),
                "error_type": type(error).__name__,
                "error": str(error),
            })
            continue
        failures.extend(_validate_row(payload, path))
        key = (
            str(payload.get("heldout_target_domain")),
            str(payload.get("method")),
            int(payload.get("seed", -1)),
            str(payload.get("implementation")),
        )
        if key in seen:
            failures.append({
                "kind": "duplicate_cell",
                "path": str(path),
                "key": list(key),
            })
            continue
        seen.add(key)
        comparison = payload.get("comparison_contract", {})
        result = payload.get("result", {})
        verification = result.get("terminal_verification", {})
        domain = key[0]
        source_fingerprints[domain].add(
            str(comparison.get("source_archive_fingerprint")))
        source_domain_sets[domain].add(tuple(payload.get("source_domains", ())))
        certified = bool(verification.get("certified"))
        true_feasible = bool(result.get("true_feasible"))
        feasible_regret = result.get("feasible_regret")
        initial_audit = result.get("initial_truth_audit", {})
        rows.append({
            "heldout_target_domain": domain,
            "method": key[1],
            "seed": key[2],
            "implementation": key[3],
            "initial_design_contains_true_feasible": bool(
                int(initial_audit.get("true_feasible_count", 0)) > 0),
            "deployed_true_feasible": true_feasible,
            "independently_certified": certified,
            "certified_safe": bool(certified and true_feasible),
            "false_certificate": bool(certified and not true_feasible),
            "feasible_regret": (
                None if feasible_regret is None else float(feasible_regret)),
            "certified_epsilon_optimal_001": bool(
                certified and true_feasible and feasible_regret is not None
                and float(feasible_regret) <= 0.01),
            "certified_epsilon_optimal_005": bool(
                certified and true_feasible and feasible_regret is not None
                and float(feasible_regret) <= 0.05),
            "source_calls": int(comparison.get("source_simulator_calls", 0)),
            "target_search_calls": int(result.get("n_search_simulations", 0)),
            "verification_calls": int(
                result.get("n_verification_simulations", 0)),
            "target_total_calls": int(
                result.get("n_target_simulations_total", 0)),
            "all_in_calls_unamortized": int(
                comparison.get("total_source_plus_target_verification_calls", 0)),
            "wall_time_sec": float(payload.get("wall_time_sec", 0.0)),
            "raw_result": str(path),
            "raw_sha256": _sha256(path),
        })

    missing = sorted(expected_keys - seen)
    unexpected = sorted(seen - expected_keys)
    if missing:
        failures.append({"kind": "missing_cells", "count": len(missing),
                         "first_cells": [list(key) for key in missing[:20]]})
    if unexpected:
        failures.append({"kind": "unexpected_cells", "count": len(unexpected),
                         "first_cells": [list(key) for key in unexpected[:20]]})
    for domain in expected_domains:
        if len(source_fingerprints[domain]) != 1:
            failures.append({
                "kind": "source_archive_fingerprint_mismatch",
                "heldout_target_domain": domain,
                "observed": sorted(source_fingerprints[domain]),
            })
        if len(source_domain_sets[domain]) != 1:
            failures.append({
                "kind": "source_domain_set_mismatch",
                "heldout_target_domain": domain,
                "observed": [list(value) for value in source_domain_sets[domain]],
            })

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["heldout_target_domain"], row["method"])].append(row)
    domain_method_summaries = []
    for (domain, method), group in sorted(grouped.items()):
        certified_safe = sum(row["certified_safe"] for row in group)
        certified_regrets = [
            row["feasible_regret"] for row in group
            if row["certified_safe"] and row["feasible_regret"] is not None
        ]
        domain_method_summaries.append({
            "heldout_target_domain": domain,
            "method": method,
            "algorithmic_seed_count": len(group),
            "initial_design_coverage_count": int(sum(
                row["initial_design_contains_true_feasible"] for row in group)),
            "deployed_true_feasible_count": int(sum(
                row["deployed_true_feasible"] for row in group)),
            "certified_safe_count": int(certified_safe),
            "certified_safe_rate_exact_95ci": _exact_interval(
                certified_safe, len(group)),
            "false_certificate_count": int(sum(
                row["false_certificate"] for row in group)),
            "certified_epsilon_optimal_001_count": int(sum(
                row["certified_epsilon_optimal_001"] for row in group)),
            "certified_epsilon_optimal_005_count": int(sum(
                row["certified_epsilon_optimal_005"] for row in group)),
            "median_certified_feasible_regret": _median(certified_regrets),
            "mean_verification_calls": _mean([
                row["verification_calls"] for row in group]),
            "mean_all_in_calls_unamortized": _mean([
                row["all_in_calls_unamortized"] for row in group]),
            "median_wall_time_sec": _median([
                row["wall_time_sec"] for row in group]),
            "task_population_inference_claimed": False,
        })

    method_summaries = []
    for method in expected_methods:
        group = [row for row in rows if row["method"] == method]
        by_domain = {
            row["heldout_target_domain"]: row
            for row in domain_method_summaries if row["method"] == method
        }
        method_summaries.append({
            "method": method,
            "fixed_domain_count": len(by_domain),
            "algorithmic_seed_count": len(group),
            "certified_safe_count": int(sum(
                row["certified_safe"] for row in group)),
            "false_certificate_count": int(sum(
                row["false_certificate"] for row in group)),
            "minimum_domain_certified_safe_rate": (
                None if not by_domain else float(min(
                    summary["certified_safe_count"]
                    / summary["algorithmic_seed_count"]
                    for summary in by_domain.values()))),
            "domain_counts": {
                domain: by_domain[domain]["certified_safe_count"]
                for domain in sorted(by_domain)
            },
            "task_population_inference_claimed": False,
        })

    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "complete" if not failures else "incomplete",
        "comparison_type": "native_end_to_end_transfer",
        "common_atlas_backend_comparison": False,
        "source_archive_calls": 384,
        "target_search_budget": 13,
        "target_initialization_budget": 10,
        "primary_task_generalization_unit": "heldout_target_domain",
        "algorithmic_repeatability_unit": "seed_within_fixed_domain",
        "task_population_inference_claimed": False,
        "expected_cell_count": len(expected_keys),
        "row_count": len(rows),
        "domain_method_summaries": domain_method_summaries,
        "method_summaries": method_summaries,
        "source_archive_fingerprints": {
            domain: sorted(values)
            for domain, values in sorted(source_fingerprints.items())
        },
        "compact_rows": rows,
        "failures": failures,
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = []
    for value in args.inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(path.rglob("result.json"))
        elif path.is_file():
            paths.append(path)
    payload = analyze(paths)
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "row_count": payload["row_count"],
        "expected_cell_count": payload["expected_cell_count"],
        "failure_count": len(payload["failures"]),
    }, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise SystemExit(1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
