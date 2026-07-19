#!/usr/bin/env python3
"""Analyze V14 singleton-HVD/constraint-mean separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _scenario,
        load_rows as _load_rows,
    )
    from .analyze_mean_alignment_v12_gate import _cell as _v12_cell
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _scenario,
        load_rows as _load_rows,
    )
    from analyze_mean_alignment_v12_gate import _cell as _v12_cell


VARIANTS = (
    "v4_mixture_control",
    "v8_mixture_control",
    "bounded_tanh_control",
    "isolated_v8_mixture",
    "isolated_tanh_mixture",
    "isolated_tanh_aggregate",
)
ISOLATED_VARIANTS = set(VARIANTS[3:])
CHALLENGERS = VARIANTS[4:]


def load_rows(root):
    return _load_rows(root, variants=VARIANTS)


def _cell(rows, variant, scenario, expected_seeds):
    value = _v12_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    diagnostics = [dict(row.get("variance_diagnostics") or {}) for row in group]
    value.update({
        "singleton_evidence_modes": sorted(set(str(
            item.get("singleton_evidence_mode", "missing"))
            for item in diagnostics)),
        "cumulative_target_evidence_modes": sorted(set(str(
            item.get("cumulative_target_evidence_mode", "missing"))
            for item in diagnostics)),
        "source_prior_singleton_counts": sorted(set(int(count)
            for item in diagnostics
            for count in (item.get("source_prior_singleton_count") or {}).values())),
        "replicated_solution_counts": sorted(set(int(count)
            for item in diagnostics
            for count in (item.get("replicated_solution_count") or {}).values())),
        "prequential_solution_counts": sorted(set(int(count)
            for item in diagnostics
            for count in (item.get("prequential_upper_solution_count") or {}).values())),
        "residual_effective_dofs": sorted(set(float(
            detail.get("effective_dof", -1.0))
            for item in diagnostics
            for detail in (item.get("residual_square_tail") or {}).values())),
        "cumulative_prior_target_weights": sorted(set(int(count)
            for item in diagnostics
            for count in (item.get("cumulative_prior_target_weight") or {}).values())),
        "hvd_source_task_weight_modes": sorted(set(str(
            row.get("hvd_source_task_weight_mode", "missing"))
            for row in group)),
    })
    return value


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


def _row_key(row):
    return (*_scenario(row), int(row["seed"]))


def _numeric_equal(left, right, key, tolerance=1e-12):
    a = left.get(key)
    b = right.get(key)
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=tolerance, atol=tolerance))


def _variance_head_signature(row):
    diagnostics = dict(row.get("variance_diagnostics") or {})
    return {
        key: diagnostics.get(key)
        for key in (
            "global_var",
            "class_count",
            "cumulative_prior_scale",
            "cumulative_prior_scale_se",
            "cumulative_prior_component_weights",
            "cumulative_prior_upper_scale",
            "source_prior_singleton_count",
            "replicated_solution_count",
            "prequential_upper_solution_count",
        )
    }


def _separation_exact(rows, variant):
    reference = {
        _row_key(row): row for row in rows
        if row["gate_variant"] == "isolated_v8_mixture"
    }
    candidate = {
        _row_key(row): row for row in rows
        if row["gate_variant"] == variant
    }
    if set(reference) != set(candidate) or not reference:
        return False
    numeric_keys = (
        "variance_log_rmse",
        "certified_variance_log_rmse",
        "median_predicted_true_variance_ratio",
        "median_certified_true_variance_ratio",
        "variance_upper_coverage",
    )
    return all(
        all(_numeric_equal(reference[key], candidate[key], metric)
            for metric in numeric_keys)
        and _variance_head_signature(reference[key])
        == _variance_head_signature(candidate[key])
        for key in reference
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
    all_cells = list(cells.values())
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
            "audit_pool_false_certificates": int(sum(
                item["audit_pool_false_certificate_count"]
                for item in values)),
        }
        for variant, values in grouped.items()
    }
    global_checks = {
        "all_cells_complete": all(item["complete"] for item in all_cells),
        "oracle_free": all(item["oracle_free"] for item in all_cells),
        "paired_initial_design_and_archive": bool(paired),
    }
    v8 = grouped["v8_mixture_control"]
    bounded = grouped["bounded_tanh_control"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        isolation_contract = all(
            item["singleton_evidence_modes"] == ["source_prior"]
            and item["cumulative_target_evidence_modes"]
            == ["replication_only"]
            and item["source_prior_singleton_counts"] == [10]
            and item["replicated_solution_counts"] == [0]
            and item["prequential_solution_counts"] == [0]
            and item["residual_effective_dofs"] == [0.0]
            and item["cumulative_prior_target_weights"] == [0]
            and item["hvd_source_task_weight_modes"] == ["independent"]
            for item in values
        )
        checks[variant] = {
            **global_checks,
            "singleton_hvd_isolation_contract": bool(isolation_contract),
            "variance_head_exactly_invariant_to_mean_coordinate": bool(
                _separation_exact(rows, variant)),
            "true_feasible_nonworse_than_v8": bool(
                totals[variant]["true_feasible"]
                >= totals["v8_mixture_control"]["true_feasible"]),
            "false_certification_nonworse_than_bounded_tanh": bool(
                totals[variant]["audit_pool_false_certificates"]
                <= totals["bounded_tanh_control"][
                    "audit_pool_false_certificates"]),
            "variance_rmse_nonworse_than_v8_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_variance_log_rmse",
                    multiplier=1.10) >= 3),
            "mean_mae_nonworse_than_bounded_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, bounded, "median_mean_abs_error",
                    multiplier=1.05, offset=0.005) >= 3),
            "mean_rank_nonworse_than_bounded_at_least_3_of_4": bool(
                _count_higher_nonworse(
                    values, bounded,
                    "median_constraint_mean_rank_correlation") >= 3),
            "posterior_extrapolation_nonworse_than_bounded": bool(
                _count_lower_nonworse(
                    values, bounded, "median_max_abs_posterior_mean_seen",
                    multiplier=1.05, offset=0.01) >= 3),
            "feasible_regret_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_feasible_regret") >= 3),
            "strict_combined_gain": bool(
                totals[variant]["audit_pool_false_certificates"]
                < totals["v8_mixture_control"][
                    "audit_pool_false_certificates"]),
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
        "cells": all_cells,
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
    output = args.out or args.root / "mean_alignment_v14_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
