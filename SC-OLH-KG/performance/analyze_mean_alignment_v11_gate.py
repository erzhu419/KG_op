#!/usr/bin/env python3
"""Analyze V11 support-adaptive aggregate-transfer calibration."""

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
        _count_lower_nonworse,
        _median,
    )


VARIANTS = (
    "v4_mixture_control",
    "v8_mixture_control",
    "support_adaptive_aggregate",
)
CHALLENGER = "support_adaptive_aggregate"


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
    support = [dict(
        prior.get("support_adaptive_aggregate_selection") or {})
        for prior in priors]
    component_rows = [
        item
        for prior in priors
        for item in prior.get("component_deviation_diagnostics", [])
    ]
    source_aggregate = [
        item for item in component_rows
        if item.get("name") == "source:aggregate"
    ]
    posterior_weights = [np.asarray(
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
        "effective_source_adaptations": sorted(set(str(
            prior.get("effective_source_adaptation", "missing"))
            for prior in priors)),
        "aggregate_latent_flags": sorted(set(bool(
            prior.get("aggregate_transferability_latent", False))
            for prior in priors)),
        "support_adaptive_requested": bool(group and all(bool(
            prior.get("support_adaptive_aggregate_requested", False))
            for prior in priors)),
        "support_cardinality": sorted(set(bool(
            item.get("channel_cardinality_supported", False))
            for item in support)),
        "support_selection_outcome_free": bool(group and all(
            item
            and not item.get("target_labels_used", True)
            and not item.get("target_oracle_used", True)
            and not item.get("selection_uses_target_labels", True)
            and not item.get("selection_uses_target_oracle", True)
            for item in support)),
        "component_names": sorted(set(tuple(
            prior.get("component_names", [])) for prior in priors)),
        "posterior_weights_valid": bool(group and all(
            len(vector) >= 2
            and np.all(np.isfinite(vector))
            and np.all(vector >= 0.0)
            and abs(float(np.sum(vector)) - 1.0) <= 1e-10
            for vector in posterior_weights)),
        "target_posterior_updated": bool(group and all(
            not prior.get("prior_target_data_used", True)
            and prior.get("posterior_target_data_used", False)
            and int(prior.get("target_observation_count", -1))
            == int(target_count)
            and not prior.get("target_oracle_used", True)
            for prior in priors)),
        "aggregate_source_only": bool(
            len(source_aggregate) == len(group)
            and all(
                item.get("aggregate_contains_within_source_uncertainty", False)
                and item.get(
                    "aggregate_contains_between_source_disagreement", False)
                and not item.get(
                    "target_data_used_to_define_aggregate", True)
                and not item.get(
                    "target_oracle_used_to_define_aggregate", True)
                for item in source_aggregate)),
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
    challenger = grouped[CHALLENGER]
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
    factor = challenger[:2]
    supported = challenger[2:]
    supported_v8 = v8[2:]
    supported_mae_nonworse = all(
        left["median_mean_abs_error"]
        <= 1.10 * right["median_mean_abs_error"] + 0.01
        for left, right in zip(supported, supported_v8))
    supported_mae_strict = any(
        left["median_mean_abs_error"] + 1e-12
        < right["median_mean_abs_error"]
        for left, right in zip(supported, supported_v8))
    checks = {
        **global_checks,
        "coordinate_selection_is_outcome_free": all(
            item["selection_present"] and item["selection_outcome_free"]
            for item in challenger),
        "adaptation_selection_is_outcome_free": all(
            item["support_adaptive_requested"]
            and item["support_selection_outcome_free"]
            and item["target_posterior_updated"]
            and item["posterior_weights_valid"]
            for item in challenger),
        "unsupported_factor_uses_domain_mixture": all(
            item["support_cardinality"] == [False]
            and item["cardinality_support"] == [False]
            and item["selected_coordinates"] == ["ordered"]
            and item["effective_source_adaptations"] == ["domain_mixture"]
            and item["aggregate_latent_flags"] == [False]
            and item["adaptation_modes"]
            == ["sequential_target_evidence_mixture"]
            and all("source:aggregate" not in names
                    for names in item["component_names"])
            for item in factor),
        "supported_domains_use_aggregate_latent": all(
            item["support_cardinality"] == [True]
            and item["cardinality_support"] == [True]
            and item["selected_coordinates"] == ["role_aligned"]
            and item["effective_source_adaptations"] == ["aggregate_latent"]
            and item["aggregate_latent_flags"] == [True]
            and item["adaptation_modes"]
            == ["sequential_aggregate_target_evidence_mixture"]
            and item["component_names"]
            == [("source:aggregate", "target:null")]
            and item["aggregate_source_only"]
            for item in supported),
        "true_feasible_nonworse_than_both_controls": bool(
            totals[CHALLENGER]["true_feasible"] >= max(
                totals["v4_mixture_control"]["true_feasible"],
                totals["v8_mixture_control"]["true_feasible"])),
        "false_certification_nonworse_than_v8_and_better_than_v4": bool(
            totals[CHALLENGER]["audit_pool_false_certificates"]
            <= totals["v8_mixture_control"][
                "audit_pool_false_certificates"]
            and totals[CHALLENGER]["audit_pool_false_certificates"]
            < totals["v4_mixture_control"][
                "audit_pool_false_certificates"]),
        "supported_mean_mae_improves": bool(
            supported_mae_nonworse and supported_mae_strict),
        "variance_rmse_nonworse_at_least_3_of_4": bool(
            _count_lower_nonworse(
                challenger, v8, "median_variance_log_rmse",
                multiplier=1.10) >= 3),
        "feasible_regret_nonworse_at_least_3_of_4": bool(
            _count_lower_nonworse(
                challenger, v8, "median_feasible_regret") >= 3),
    }
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "cells": all_cells,
        "totals": totals,
        "global_checks": global_checks,
        "checks": checks,
        "sequential_gate_eligible": (
            [CHALLENGER] if all(checks.values()) else []),
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
    output = args.out or args.root / "mean_alignment_v11_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "checks": result["checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
