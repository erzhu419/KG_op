#!/usr/bin/env python3
"""Analyze V16 partial-role transport and epistemic-calibration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_mean_alignment_v8_gate import SCENARIOS, _scenario
    from .analyze_mean_alignment_v15_gate import (
        _cell as _v15_cell,
        _variance_head_signature,
    )
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import SCENARIOS, _scenario
    from analyze_mean_alignment_v15_gate import (
        _cell as _v15_cell,
        _variance_head_signature,
    )


VARIANTS = (
    "v15_tanh_control",
    "adaptive_uncertainty",
    "hard_role",
    "hard_role_uncertainty",
    "transport",
    "transport_uncertainty",
    "transport_uncertainty_contrast",
)
CHALLENGERS = VARIANTS[1:]
TRANSPORT_VARIANTS = set(VARIANTS[4:])
UNCERTAINTY_VARIANTS = {
    "adaptive_uncertainty",
    "hard_role_uncertainty",
    "transport_uncertainty",
    "transport_uncertainty_contrast",
}


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            item = dict(row)
            experiment = str(payload.get("experiment_variant", ""))
            item["gate_variant"] = next(
                (name for name in VARIANTS if f"/{name}/" in f"/{experiment}/"),
                "missing",
            )
            rows.append(item)
    return [row for row in rows if row["gate_variant"] in VARIANTS]


def _find_alignment(value):
    if isinstance(value, dict):
        if "partial_transport" in value and "source_domains" in value:
            return value
        for child in value.values():
            found = _find_alignment(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_alignment(child)
            if found is not None:
                return found
    return None


def _constraint_prior(row):
    numerics = list(row.get("gpr_numerics") or [])
    return (
        dict(numerics[1].get("source_parametric_prior") or {})
        if len(numerics) > 1 else {}
    )


def _cell(rows, variant, scenario, expected_seeds):
    value = _v15_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    audits = [
        dict(row.get("boundary_raw_pool_truth_diagnostics") or {})
        for row in group
    ]
    priors = [_constraint_prior(row) for row in group]
    calibrations = [
        dict(prior.get("source_role_epistemic_calibration") or {})
        for prior in priors
    ]
    alignments = [
        _find_alignment(dict(row.get("meta_basis") or {}))
        for row in group
    ]
    alignments = [item for item in alignments if item is not None]
    source_components = [
        component
        for prior in priors
        for component in (prior.get("component_deviation_diagnostics") or [])
        if str(component.get("name")) != "target:null"
    ]
    target_matches = [
        match
        for alignment in alignments
        for match in (alignment.get("target_matches") or {}).values()
    ]
    transport_selections = [
        dict(alignment.get("transport_selection") or {})
        for alignment in alignments
    ]
    value.update({
        "audit_pool_full_certificate_count": int(sum(int(
            item.get("boundary_raw_pool_full_certified_count", 0))
            for item in audits)),
        "audit_pool_true_certificate_count": int(sum(int(
            item.get("boundary_raw_pool_true_certified_count", 0))
            for item in audits)),
        "descriptor_modes": sorted(set(str(
            row.get("meta_observable_mean_descriptor_mode", "missing"))
            for row in group)),
        "partial_transport_flags": sorted(set(bool(
            item.get("partial_transport", False)) for item in alignments)),
        "transport_selection_statuses": sorted(set(str(
            item.get("status", "missing"))
            for item in transport_selections)),
        "transport_selection_source_only": bool(
            transport_selections and all(
                not item.get("target_data_used", True)
                and not item.get("target_oracle_used", True)
                for item in transport_selections)),
        "transport_target_outcome_free": bool(
            target_matches and all(
                not item.get("target_labels_used", True)
                and not item.get("target_oracle_used", True)
                for item in target_matches)),
        "transport_weight_shapes": sorted(set(
            (
                len(item.get("transport_weights") or []),
                len((item.get("transport_weights") or [[None]])[0]),
            )
            for item in target_matches
            if item.get("transport_weights") is not None
        )),
        "role_epistemic_modes": sorted(set(str(
            item.get("mode", "none")) for item in calibrations)),
        "role_epistemic_scales": sorted(set(float(
            item.get("epistemic_covariance_scale", 1.0))
            for item in calibrations)),
        "role_epistemic_outcome_free": bool(
            calibrations and all(
                not item.get("target_labels_used", True)
                and not item.get("target_oracle_used", True)
                for item in calibrations)),
        "role_epistemic_monotone": bool(
            source_components and all(
                component.get(
                    "role_matching_uncertainty_monotone", False)
                and float(component.get(
                    "role_matching_epistemic_covariance_scale", 0.0)) >= 1.0
                and not component.get(
                    "role_matching_target_labels_used", True)
                and not component.get(
                    "role_matching_target_oracle_used", True)
                for component in source_components)),
    })
    return value


def _row_key(row):
    return (*_scenario(row), int(row["seed"]))


def _numeric_equal(left, right, key, tolerance=1e-12):
    a = left.get(key)
    b = right.get(key)
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=tolerance, atol=tolerance))


def _variance_head_exact(rows, variant):
    reference = {
        _row_key(row): row for row in rows
        if row["gate_variant"] == "v15_tanh_control"
    }
    candidate = {
        _row_key(row): row for row in rows
        if row["gate_variant"] == variant
    }
    if not reference or set(reference) != set(candidate):
        return False
    metrics = (
        "variance_log_rmse",
        "certified_variance_log_rmse",
        "median_predicted_true_variance_ratio",
        "median_certified_true_variance_ratio",
        "variance_upper_coverage",
    )
    return all(
        all(_numeric_equal(reference[key], candidate[key], metric)
            for metric in metrics)
        and _variance_head_signature(reference[key])
        == _variance_head_signature(candidate[key])
        for key in reference
    )


def _count_lower_nonworse(left, right, key, multiplier=1.0, offset=0.0):
    return sum(
        item[key] is not None and reference[key] is not None
        and item[key] <= multiplier * reference[key] + offset
        for item, reference in zip(left, right)
    )


def _count_higher_nonworse(left, right, key, offset=0.02):
    return sum(
        item[key] is not None and reference[key] is not None
        and item[key] + offset >= reference[key]
        for item, reference in zip(left, right)
    )


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in SCENARIOS]
        for variant in VARIANTS
    }
    paired = all(
        len({tuple(cell["initial_design_fingerprints"])
             for cell in scenario_cells}) == 1
        and len({tuple(cell["source_archive_fingerprints"])
                 for cell in scenario_cells}) == 1
        for scenario_cells in zip(*(grouped[value] for value in VARIANTS))
    )
    totals = {
        variant: {
            "true_feasible": int(sum(
                item["true_feasible_count"] for item in values)),
            "audit_pool_full_certificates": int(sum(
                item["audit_pool_full_certificate_count"] for item in values)),
            "audit_pool_true_certificates": int(sum(
                item["audit_pool_true_certificate_count"] for item in values)),
            "audit_pool_false_certificates": int(sum(
                item["audit_pool_false_certificate_count"] for item in values)),
        }
        for variant, values in grouped.items()
    }
    global_checks = {
        "all_cells_complete": all(
            item["complete"] for item in cells.values()),
        "oracle_free": all(item["oracle_free"] for item in cells.values()),
        "paired_initial_design_and_archive": bool(paired),
    }
    reference = grouped["v15_tanh_control"]
    reference_totals = totals["v15_tanh_control"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        independent_variance = all(
            item["variance_task_posterior_modes"] == ["replication_only"]
            and item["variance_task_posterior_statuses"]
            == ["frozen_source_prior"]
            and item["variance_task_evidence_counts"] == [0]
            and item["variance_task_effective_dofs"] == [0]
            and item["variance_task_uses_target_mean"] == [False]
            for item in values
        )
        transport_contract = (
            all(
                item["descriptor_modes"] == ["role_transport"]
                and item["partial_transport_flags"] == [True]
                and item["transport_selection_statuses"]
                == ["source_domain_dropout_selected"]
                and item["transport_selection_source_only"]
                and item["transport_target_outcome_free"]
                for item in values
            )
            if variant in TRANSPORT_VARIANTS else True
        )
        uncertainty_contract = (
            all(
                item["role_epistemic_modes"] == ["matching_uncertainty"]
                and min(item["role_epistemic_scales"]) >= 1.0
                and item["role_epistemic_outcome_free"]
                and item["role_epistemic_monotone"]
                for item in values
            )
            if variant in UNCERTAINTY_VARIANTS else True
        )
        checks[variant] = {
            **global_checks,
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_coordinate": bool(
                _variance_head_exact(rows, variant)),
            "source_only_partial_transport_contract": bool(
                transport_contract),
            "monotone_mean_epistemic_calibration_contract": bool(
                uncertainty_contract),
            "true_feasible_nonworse": bool(
                totals[variant]["true_feasible"]
                >= reference_totals["true_feasible"]),
            "true_certificate_support_nonworse": bool(
                totals[variant]["audit_pool_true_certificates"]
                >= reference_totals["audit_pool_true_certificates"]),
            "strict_false_certification_gain": bool(
                totals[variant]["audit_pool_false_certificates"]
                < reference_totals["audit_pool_false_certificates"]),
            "factor_shock_scale4_false_certification_gain": bool(
                values[1]["audit_pool_false_certificate_count"]
                < reference[1]["audit_pool_false_certificate_count"]),
            "mean_mae_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, reference, "median_mean_abs_error",
                    multiplier=1.05, offset=0.005) >= 3),
            "mean_rank_nonworse_at_least_3_of_4": bool(
                _count_higher_nonworse(
                    values, reference,
                    "median_constraint_mean_rank_correlation") >= 3),
            "feasible_regret_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, reference, "median_feasible_regret") >= 3),
        }
    eligible = [
        variant for variant in CHALLENGERS
        if all(checks[variant].values())
    ]
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "totals": totals,
        "global_checks": global_checks,
        "variant_checks": checks,
        "sequential_gate_eligible": eligible,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or args.root / "mean_alignment_v16_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
