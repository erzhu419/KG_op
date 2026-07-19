#!/usr/bin/env python3
"""Analyze V10 aggregate-transferability latent posterior calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _cell as _base_cell,
        _prior,
        _scenario,
        load_rows as _load_rows,
    )
    from .analyze_mean_alignment_v9_gate import (
        _count_higher_nonworse,
        _count_lower_nonworse,
        _median,
    )
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import (
        SCENARIOS,
        _cell as _base_cell,
        _prior,
        _scenario,
        load_rows as _load_rows,
    )
    from analyze_mean_alignment_v9_gate import (
        _count_higher_nonworse,
        _count_lower_nonworse,
        _median,
    )


VARIANTS = (
    "v4_mixture_control",
    "v8_mixture_control",
    "aggregate_latent",
    "aggregate_latent_scalar",
    "aggregate_latent_directional",
)
CHALLENGERS = VARIANTS[2:]
MISSPECIFICATION = {
    "aggregate_latent": "none",
    "aggregate_latent_scalar": "predictive_scale",
    "aggregate_latent_directional": "predictive_scale_directional",
}


def load_rows(root):
    return _load_rows(root, variants=VARIANTS)


def _cell(rows, variant, scenario, expected_seeds, target_count):
    value = _base_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_prior(row) for row in group]
    raw = [row.get("boundary_raw_pool_truth_diagnostics") or {}
           for row in group]
    source_rows = [
        item
        for prior in priors
        for item in prior.get("component_deviation_diagnostics", [])
        if item.get("name") == "source:aggregate"
    ]
    weights = [np.asarray(
        prior.get("component_posterior_weights", []), dtype=float)
        for prior in priors]
    value.update({
        "median_constraint_mean_rank_correlation": _median([
            item.get("boundary_raw_pool_constraint_mean_rank_correlation")
            for item in raw]),
        "median_chance_margin_rank_correlation": _median([
            item.get("boundary_raw_pool_chance_margin_rank_correlation")
            for item in raw]),
        "adaptation_modes": sorted(set(str(
            prior.get("adaptation_mode", "missing")) for prior in priors)),
        "misspecification_modes": sorted(set(str(
            prior.get("source_mean_misspecification_mode", "missing"))
            for prior in priors)),
        "component_names": sorted(set(tuple(
            prior.get("component_names", [])) for prior in priors)),
        "aggregate_latent_enabled": bool(group and all(bool(
            prior.get("aggregate_transferability_latent", False))
            for prior in priors)),
        "source_prior_frozen_target_posterior_updated": bool(group and all(
            not prior.get("prior_target_data_used", True)
            and prior.get("posterior_target_data_used", False)
            and int(prior.get("target_observation_count", -1))
            == int(target_count)
            and not prior.get("target_oracle_used", True)
            for prior in priors)),
        "aggregate_source_only": bool(
            len(source_rows) == len(group)
            and all(
                item.get("aggregate_contains_within_source_uncertainty", False)
                and item.get(
                    "aggregate_contains_between_source_disagreement", False)
                and not item.get(
                    "target_data_used_to_define_aggregate", True)
                and not item.get(
                    "target_oracle_used_to_define_aggregate", True)
                for item in source_rows)),
        "posterior_weights_valid": bool(group and all(
            vector.shape == (2,)
            and np.all(np.isfinite(vector))
            and np.all(vector >= 0.0)
            and abs(float(np.sum(vector)) - 1.0) <= 1e-10
            for vector in weights)),
        "misspecification_applied": bool(group and all(bool(
            item.get("source_mean_misspecification_applied", False))
            for item in source_rows)),
        "misspecification_uncertainty_noncontracting": bool(group and all(
            float(item.get(
                "source_mean_prior_covariance_trace_after", 0.0)) + 1e-12
            >= float(item.get(
                "source_mean_prior_covariance_trace_before", 0.0))
            and float(item.get(
                "source_mean_residual_floor_after", 0.0)) + 1e-12
            >= float(item.get(
                "source_mean_residual_floor_before", 0.0))
            and bool(item.get(
                "misspecification_uncertainty_can_only_increase", False))
            for item in source_rows)),
    })
    return value


def summarize(rows, expected_seeds=5, *, target_count=10):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds, target_count)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in SCENARIOS]
        for variant in VARIANTS
    }
    all_cells = list(cells.values())
    v4 = grouped["v4_mixture_control"]
    v8 = grouped["v8_mixture_control"]
    paired = all(
        len({tuple(cell["initial_design_fingerprints"])
             for cell in scenario_cells}) == 1
        and len({tuple(cell["source_archive_fingerprints"])
                for cell in scenario_cells}) == 1
        for scenario_cells in zip(*(grouped[variant] for variant in VARIANTS))
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
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        expected_mode = MISSPECIFICATION[variant]
        uncertainty_contract = all(
            item["misspecification_modes"] == [expected_mode]
            and (
                True if expected_mode == "none" else (
                    item["misspecification_applied"]
                    and item["misspecification_uncertainty_noncontracting"]
                )
            )
            for item in values)
        checks[variant] = {
            **global_checks,
            "aggregate_transferability_contract": all(
                item["adaptation_modes"]
                == ["sequential_aggregate_target_evidence_mixture"]
                and item["component_names"]
                == [("source:aggregate", "target:null")]
                and item["aggregate_latent_enabled"]
                and item["source_prior_frozen_target_posterior_updated"]
                and item["aggregate_source_only"]
                and item["posterior_weights_valid"]
                for item in values),
            "misspecification_uncertainty_contract": uncertainty_contract,
            "selection_is_outcome_free": all(
                item["selection_present"]
                and item["selection_outcome_free"] for item in values),
            "unsupported_factor_uses_ordered_fallback": all(
                item["cardinality_support"] == [False]
                and item["selected_coordinates"] == ["ordered"]
                for item in values[:2]),
            "supported_inventory_queue_use_roles": all(
                item["cardinality_support"] == [True]
                and item["selected_coordinates"] == ["role_aligned"]
                for item in values[2:]),
            "true_feasible_nonworse_than_both_controls": bool(
                totals[variant]["true_feasible"] >= max(
                    totals["v4_mixture_control"]["true_feasible"],
                    totals["v8_mixture_control"]["true_feasible"])),
            "false_certification_strictly_below_both_controls": bool(
                totals[variant]["audit_pool_false_certificates"] < min(
                    totals["v4_mixture_control"][
                        "audit_pool_false_certificates"],
                    totals["v8_mixture_control"][
                        "audit_pool_false_certificates"])),
            "mean_mae_nonworse_than_both_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v4, "median_mean_abs_error",
                    multiplier=1.10, offset=0.01) >= 3
                and _count_lower_nonworse(
                    values, v8, "median_mean_abs_error",
                    multiplier=1.10, offset=0.01) >= 3),
            "mean_rank_nonworse_than_both_at_least_3_of_4": bool(
                _count_higher_nonworse(
                    values, v4, "median_constraint_mean_rank_correlation",
                    offset=0.05) >= 3
                and _count_higher_nonworse(
                    values, v8, "median_constraint_mean_rank_correlation",
                    offset=0.05) >= 3),
            "variance_rmse_nonworse_than_both_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v4, "median_variance_log_rmse",
                    multiplier=1.10) >= 3
                and _count_lower_nonworse(
                    values, v8, "median_variance_log_rmse",
                    multiplier=1.10) >= 3),
            "feasible_regret_nonworse_than_both_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, v4, "median_feasible_regret") >= 3
                and _count_lower_nonworse(
                    values, v8, "median_feasible_regret") >= 3),
        }
    eligible = [
        variant for variant in CHALLENGERS if all(checks[variant].values())
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
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(
        load_rows(args.root), args.expected_seeds,
        target_count=args.target_count)
    output = args.out or args.root / "mean_alignment_v10_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "global_checks": result["global_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
