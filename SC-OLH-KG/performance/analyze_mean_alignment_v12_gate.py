#!/usr/bin/env python3
"""Analyze V12 bounded-coordinate/function-space posterior calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _cell as _base_cell,
        _prior,
        _scenario,
        load_rows as _load_rows,
    )
    from .analyze_mean_alignment_v9_gate import _median
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _cell as _base_cell,
        _prior,
        _scenario,
        load_rows as _load_rows,
    )
    from analyze_mean_alignment_v9_gate import _median


VARIANTS = (
    "v4_mixture_control",
    "v8_mixture_control",
    "v11_aggregate_control",
    "bounded_mixture",
    "geometry_mixture",
    "bounded_geometry_mixture",
    "bounded_geometry_aggregate",
)
CHALLENGERS = VARIANTS[3:]
TRANSFORM_VARIANTS = {
    "bounded_mixture",
    "bounded_geometry_mixture",
    "bounded_geometry_aggregate",
}
GEOMETRY_VARIANTS = {
    "geometry_mixture",
    "bounded_geometry_mixture",
    "bounded_geometry_aggregate",
}


def load_rows(root):
    return _load_rows(root, variants=VARIANTS)


def _mean_basis(row):
    values = row.get("meta_basis") or {}
    return dict(values.get("1") or values.get(1) or {})


def _selected_coordinate_diagnostics(row):
    basis = _mean_basis(row)
    return dict(basis.get("selected_coordinate_diagnostics") or basis)


def _constraint_numerics(row):
    values = list(row.get("gpr_numerics") or [])
    return dict(values[1]) if len(values) > 1 else {}


def _cell(rows, variant, scenario, expected_seeds):
    value = _base_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    raw = [row.get("boundary_raw_pool_truth_diagnostics") or {}
           for row in group]
    coordinate = [_selected_coordinate_diagnostics(row) for row in group]
    transform = [dict(item.get("latent_transform_diagnostics") or {})
                 for item in coordinate]
    priors = [_prior(row) for row in group]
    null_rows = [
        dict(component)
        for prior in priors
        for component in prior.get("component_deviation_diagnostics", [])
        if component.get("name") == "target:null"
    ]
    numerics = [_constraint_numerics(row) for row in group]
    value.update({
        "median_constraint_mean_rank_correlation": _median([
            item.get("boundary_raw_pool_constraint_mean_rank_correlation")
            for item in raw]),
        "median_chance_margin_rank_correlation": _median([
            item.get("boundary_raw_pool_chance_margin_rank_correlation")
            for item in raw]),
        "median_max_abs_posterior_mean_seen": _median([
            item.get("max_abs_posterior_mean_seen") for item in numerics]),
        "latent_transforms": sorted(set(str(
            item.get("latent_transform", "missing"))
            for item in coordinate)),
        "latent_transform_statuses": sorted(set(str(
            item.get("status", "missing")) for item in transform)),
        "latent_transform_source_only": bool(group and all(
            item
            and not item.get("selection_uses_target_data", True)
            and not item.get("selection_uses_target_oracle", True)
            for item in transform)),
        "latent_transform_temperatures": sorted(set(float(
            item["selected_temperature"])
            for item in transform if item.get("selected_temperature")
            is not None)),
        "maximum_abs_source_latent_feature": _median([
            (item.get("alignment") or {}).get(
                "maximum_abs_source_latent_feature")
            for item in coordinate]),
        "null_geometry_modes": sorted(set(str(
            item.get("null_geometry_mode", "missing"))
            for item in null_rows)),
        "null_geometry_outcome_free": bool(group and null_rows and all(
            not item.get("target_labels_used_for_null_geometry", True)
            and not item.get("target_oracle_used_for_null_geometry", True)
            for item in null_rows)),
        "null_geometry_scale_preserved": bool(group and null_rows and all(
            item.get("average_predictive_scale_preserved", False)
            for item in null_rows)),
        "null_geometry_psd": bool(group and null_rows and all(
            float(item.get("minimum_covariance_eigenvalue", -1.0)) >= -1e-10
            for item in null_rows)),
        "null_geometry_pool_sources": sorted(set(str(
            item.get("target_geometry_pool_source", "missing"))
            for item in null_rows)),
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
    v4 = grouped["v4_mixture_control"]
    v8 = grouped["v8_mixture_control"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        expects_transform = variant in TRANSFORM_VARIANTS
        expects_geometry = variant in GEOMETRY_VARIANTS
        transform_contract = all(
            item["latent_transforms"] == [
                "source_tanh" if expects_transform else "identity"]
            and (
                item["latent_transform_statuses"]
                == (["source_lodo_selected"] if expects_transform
                    else ["identity"])
            )
            and (
                item["latent_transform_source_only"]
                if expects_transform else True)
            and (
                item["maximum_abs_source_latent_feature"] is not None
                and item["maximum_abs_source_latent_feature"] <= 1.0 + 1e-10
                if expects_transform else True)
            for item in values
        )
        geometry_contract = all(
            item["null_geometry_modes"] == [
                "target_pool" if expects_geometry else "isotropic"]
            and item["null_geometry_outcome_free"]
            and (
                item["null_geometry_scale_preserved"]
                and item["null_geometry_psd"]
                and item["null_geometry_pool_sources"]
                == ["deterministic_unlabeled_role_matching_pool"]
                if expects_geometry else True)
            for item in values
        )
        strict_diagnostic_gain = (
            totals[variant]["audit_pool_false_certificates"]
            < totals["v8_mixture_control"]["audit_pool_false_certificates"]
            or any(
                left["median_max_abs_posterior_mean_seen"] is not None
                and right["median_max_abs_posterior_mean_seen"] is not None
                and left["median_max_abs_posterior_mean_seen"] + 1e-12
                < right["median_max_abs_posterior_mean_seen"]
                for left, right in zip(values, v8)
            )
        )
        checks[variant] = {
            **global_checks,
            "coordinate_selection_is_outcome_free": all(
                item["selection_present"] and item["selection_outcome_free"]
                for item in values),
            "bounded_transform_contract": bool(transform_contract),
            "target_null_geometry_contract": bool(geometry_contract),
            "true_feasible_nonworse_than_v8": bool(
                totals[variant]["true_feasible"]
                >= totals["v8_mixture_control"]["true_feasible"]),
            "false_certification_nonworse_than_v8_and_better_than_v4": bool(
                totals[variant]["audit_pool_false_certificates"]
                <= totals["v8_mixture_control"][
                    "audit_pool_false_certificates"]
                and totals[variant]["audit_pool_false_certificates"]
                < totals["v4_mixture_control"][
                    "audit_pool_false_certificates"]),
            "mean_mae_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_mean_abs_error",
                    multiplier=1.10, offset=0.01) >= 3),
            "mean_rank_nonworse_at_least_3_of_4": bool(
                _count_higher_nonworse(
                    values, v8,
                    "median_constraint_mean_rank_correlation") >= 3),
            "posterior_extrapolation_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_max_abs_posterior_mean_seen",
                    multiplier=1.10, offset=0.01) >= 3),
            "variance_rmse_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_variance_log_rmse",
                    multiplier=1.10) >= 3),
            "feasible_regret_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_feasible_regret") >= 3),
            "strict_diagnostic_gain": bool(strict_diagnostic_gain),
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
    output = args.out or args.root / "mean_alignment_v12_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
