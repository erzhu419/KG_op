#!/usr/bin/env python3
"""Analyze V13 source-support projection and discrepancy coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _prior,
        _scenario,
        load_rows as _load_rows,
    )
    from .analyze_mean_alignment_v9_gate import _median
    from .analyze_mean_alignment_v12_gate import _cell as _v12_cell
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _prior,
        _scenario,
        load_rows as _load_rows,
    )
    from analyze_mean_alignment_v9_gate import _median
    from analyze_mean_alignment_v12_gate import _cell as _v12_cell


VARIANTS = (
    "v4_mixture_control",
    "v8_mixture_control",
    "v11_aggregate_control",
    "bounded_tanh_control",
    "support_clip_mixture",
    "support_residual_mixture",
    "support_clip_aggregate",
    "support_residual_aggregate",
)
CHALLENGERS = VARIANTS[4:]
RESIDUAL_VARIANTS = {
    "support_residual_mixture", "support_residual_aggregate"
}


def load_rows(root):
    return _load_rows(root, variants=VARIANTS)


def _mean_basis(row):
    values = row.get("meta_basis") or {}
    return dict(values.get("1") or values.get(1) or {})


def _selected_coordinate_diagnostics(row):
    basis = _mean_basis(row)
    return dict(basis.get("selected_coordinate_diagnostics") or basis)


def _cell(rows, variant, scenario, expected_seeds):
    value = _v12_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    coordinate = [_selected_coordinate_diagnostics(row) for row in group]
    transforms = [
        dict(item.get("latent_transform_diagnostics") or {})
        for item in coordinate
    ]
    bounds = [
        np.asarray(item.get("support_bounds") or [], dtype=float)
        for item in transforms
    ]
    value.update({
        "latent_transform_quantiles": sorted(set(float(
            item["selected_quantile"])
            for item in transforms
            if item.get("selected_quantile") is not None)),
        "support_bounds_present_positive": bool(group and all(
            len(item) > 0 and np.all(np.isfinite(item))
            and np.all(item > 0.0) for item in bounds)),
        "maximum_support_bound": _median([
            float(np.max(item)) for item in bounds if len(item)]),
        "support_residual_channels": sorted(set(bool(
            item.get("residual_channel", False))
            for item in transforms)),
        "support_residual_channel_indices": sorted(set(int(
            item["residual_channel_index"])
            for item in transforms
            if item.get("residual_channel_index") is not None)),
        "coordinate_feature_dims": sorted(set(int(
            item.get("feature_dim", -1)) for item in coordinate)),
        "coordinate_latent_dims": sorted(set(int(
            (item.get("alignment") or {}).get(
                "alignment_latent_dim", -1)) for item in coordinate)),
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
        residual = variant in RESIDUAL_VARIANTS
        expected_transform = (
            "source_support_residual" if residual
            else "source_support_clip")
        support_contract = all(
            item["latent_transforms"] == [expected_transform]
            and item["latent_transform_statuses"] == ["source_lodo_selected"]
            and item["latent_transform_source_only"]
            and item["support_bounds_present_positive"]
            and item["latent_transform_quantiles"]
            and item["support_residual_channels"] == [residual]
            and item["coordinate_latent_dims"]
            and item["coordinate_feature_dims"]
            and item["coordinate_feature_dims"][0]
            == item["coordinate_latent_dims"][0] + int(residual)
            and item["maximum_abs_source_latent_feature"] is not None
            and item["maximum_support_bound"] is not None
            and item["maximum_abs_source_latent_feature"]
            <= max(item["maximum_support_bound"], float(residual)) + 1e-10
            for item in values
        )
        checks[variant] = {
            **global_checks,
            "coordinate_selection_is_outcome_free": all(
                item["selection_present"] and item["selection_outcome_free"]
                for item in values),
            "source_support_transform_contract": bool(support_contract),
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
            "posterior_extrapolation_improves_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_max_abs_posterior_mean_seen",
                    multiplier=0.90) >= 3),
            "variance_rmse_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_variance_log_rmse",
                    multiplier=1.10) >= 3),
            "feasible_regret_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v8, "median_feasible_regret") >= 3),
            "strict_certification_or_mean_gain": bool(
                totals[variant]["audit_pool_false_certificates"]
                < totals["v8_mixture_control"][
                    "audit_pool_false_certificates"]
                or any(
                    left["median_mean_abs_error"] is not None
                    and right["median_mean_abs_error"] is not None
                    and left["median_mean_abs_error"]
                    < right["median_mean_abs_error"]
                    for left, right in zip(values, v8)
                )),
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
    output = args.out or args.root / "mean_alignment_v13_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
