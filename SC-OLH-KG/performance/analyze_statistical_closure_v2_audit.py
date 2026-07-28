#!/usr/bin/env python3
"""Audit V51 finite-sample theorem assumptions without relabeling old runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
IMPLEMENTATION_CONTRACT_ID = "promoted_v51_observed_terminal_closure"
THEORY_CONTRACT_ID = "v51_statistical_closure_v2"
VARIANT = "statistical_closure_v2"


def _median(values):
    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite.append(value)
    return None if not finite else float(statistics.median(finite))


def _minimum(values):
    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite.append(value)
    return None if not finite else float(min(finite))


def load_rows(root):
    rows = []
    errors = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            experiment = str(payload.get("experiment_variant", ""))
            if f"/{VARIANT}/" not in f"/{experiment}/":
                continue
            for raw in payload["rows"]:
                row = dict(raw)
                row["result_path"] = str(path)
                rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return rows, errors


def _design(row):
    variance = dict(row.get("variance_diagnostics") or {})
    designs = dict(variance.get("cumulative_statistical_design") or {})
    return dict(designs.get("1") or {})


def _contract_ok(row):
    return bool(
        str(row.get("implementation_contract_id"))
        == IMPLEMENTATION_CONTRACT_ID
        and str(row.get("theory_contract_id")) == THEORY_CONTRACT_ID
        and str(row.get("theory_contract_timing"))
        == "declared_before_target_evaluation"
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _configuration_ok(row, profile):
    return bool(
        int(row.get("exact_kg_mc_samples", -1))
        == int(profile["exact_mc_samples"])
        and int(row.get("evaluate_or_replicate_new_action_count", -1))
        == int(profile["exact_shortlist_size"])
        and str(row.get("exact_kg_sampling_mode"))
        == str(profile["exact_sampling_mode"])
        and int(row.get("replication_max_per_solution", -1))
        == int(profile["replication_cap"])
    )


def _summarize(rows):
    designs = [_design(row) for row in rows]
    audits = [dict(row.get("certificate_outcome_audit") or {}) for row in rows]
    certified_counts = [
        int(audit.get("posterior_certified_count", 0) or 0)
        for audit in audits
    ]
    projections = Counter(str(item.get("projection", "missing"))
                          for item in designs)
    return {
        "run_count": len(rows),
        "contract_count": sum(_contract_ok(row) for row in rows),
        "diagnostic_count": sum(bool(item) for item in designs),
        "diagnostic_contract_count": sum(
            item.get("theory_contract") == THEORY_CONTRACT_ID
            for item in designs
        ),
        "active_identifiable_count": sum(bool(item.get(
            "active_identifiable", False)) for item in designs),
        "positive_replication_dof_count": sum(
            float(item.get("effective_replication_dof", 0.0) or 0.0) > 0.0
            for item in designs
        ),
        "finite_sample_hvd_applicable_count": sum(bool(item.get(
            "active_identifiable", False)) and float(item.get(
                "effective_replication_dof", 0.0) or 0.0) > 0.0
            for item in designs),
        "median_active_dimension": _median(
            item.get("active_calibration_dimension") for item in designs),
        "median_active_rank": _median(
            (item.get("active_geometry") or {}).get("rank")
            for item in designs),
        "median_lean_excitation_kappa": _median(
            item.get("lean_excitation_kappa") for item in designs),
        "minimum_lean_excitation_kappa": _minimum(
            item.get("lean_excitation_kappa") for item in designs),
        "median_target_evidence_count": _median(
            item.get("target_evidence_solution_count") for item in designs),
        "median_replicated_solution_count": _median(
            item.get("replicated_solution_count") for item in designs),
        "projection_counts": dict(sorted(projections.items())),
        "posterior_certified_count": sum(certified_counts),
        "nonvacuous_run_count": sum(value > 0 for value in certified_counts),
        "vacuous_run_count": sum(value == 0 for value in certified_counts),
        "false_certificate_count": sum(int(audit.get(
            "false_certificate_count", 0) or 0) for audit in audits),
        "true_feasible_recommendation_count": sum(bool(row.get(
            "true_feasible", False)) for row in rows),
        "median_feasible_regret": _median(
            row.get("feasible_simple_regret")
            for row in rows if row.get("true_feasible", False)),
    }


def analyze(root, registration, expected_count=15):
    rows, errors = load_rows(root)
    profile = dict(registration["profile"])
    selected = [row for row in rows if row.get("heldout") in DOMAINS]
    overall = _summarize(selected)
    by_domain = {
        domain: _summarize([
            row for row in selected if row.get("heldout") == domain
        ])
        for domain in DOMAINS
    }
    contract_complete = bool(
        len(selected) == int(expected_count)
        and not errors
        and all(_contract_ok(row) for row in selected)
        and all(_configuration_ok(row, profile) for row in selected)
        and all(_design(row).get("theory_contract") == THEORY_CONTRACT_ID
                for row in selected)
    )
    assumption_complete = bool(
        selected
        and overall["finite_sample_hvd_applicable_count"] == len(selected)
    )
    finite_sample_audit_passed = bool(contract_complete and assumption_complete)
    nonvacuity_in_every_domain = bool(
        selected
        and all(by_domain[domain]["posterior_certified_count"] > 0
                for domain in DOMAINS)
    )
    return {
        "scope": "v51_statistical_closure_v2_audit",
        "implementation_contract_id": IMPLEMENTATION_CONTRACT_ID,
        "theory_contract_id": THEORY_CONTRACT_ID,
        "expected_count": int(expected_count),
        "parsed_count": len(selected),
        "errors": errors,
        "frozen_configuration": {
            key: profile[key] for key in (
                "exact_mc_samples",
                "exact_shortlist_size",
                "exact_sampling_mode",
                "replication_cap",
            )
        },
        "overall": overall,
        "by_domain": by_domain,
        "contract_complete": contract_complete,
        "finite_sample_hvd_assumptions_hold_for_all_runs": assumption_complete,
        "finite_sample_hvd_audit_passed": finite_sample_audit_passed,
        "certificate_nonvacuity_observed_in_this_audit": bool(
            overall["posterior_certified_count"] > 0
        ),
        "certificate_nonvacuity_observed_in_every_domain": (
            nonvacuity_in_every_domain
        ),
        "publication_eligible": bool(
            finite_sample_audit_passed
            and nonvacuity_in_every_domain
            and overall["false_certificate_count"] == 0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=15)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    result = analyze(args.root, registration, args.expected_count)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
