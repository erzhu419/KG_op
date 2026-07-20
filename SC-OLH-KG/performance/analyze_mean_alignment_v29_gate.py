#!/usr/bin/env python3
"""Analyze the V29 mean-calibration and incumbent-preservation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from .analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from .analyze_mean_alignment_v28_gate import _cell as _v28_cell
    from .analyze_mean_alignment_v8_gate import _scenario
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from analyze_mean_alignment_v28_gate import _cell as _v28_cell
    from analyze_mean_alignment_v8_gate import _scenario


VARIANTS = (
    "v28_direct_control",
    "v29_direct_dominance",
    "v29_scale_dominance",
    "v29_directional_dominance",
)
CHALLENGERS = VARIANTS[1:]
MISSPECIFICATION_MODES = {
    "v28_direct_control": "none",
    "v29_direct_dominance": "none",
    "v29_scale_dominance": "predictive_scale",
    "v29_directional_dominance": "predictive_scale_directional",
}


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        experiment = str(payload.get("experiment_variant", ""))
        variant = next(
            (name for name in VARIANTS if f"/{name}/" in f"/{experiment}/"),
            None,
        )
        if variant is None:
            continue
        for raw in payload.get("rows", []):
            row = dict(raw)
            row["gate_variant"] = variant
            rows.append(row)
    return rows


def _dominance_record_contract(row, expected_enabled):
    state = dict(row.get("posterior_dominance") or {})
    enabled = bool(row.get("posterior_dominance_enabled", False))
    if enabled != bool(expected_enabled):
        return False
    if bool(state.get("enabled", False)) != bool(expected_enabled):
        return False
    if state.get("target_oracle_used", True):
        return False
    if state.get("method") != "cantelli_covariance_free":
        return False
    if expected_enabled:
        history = list(state.get("history") or [])
        return bool(
            row.get("posterior_dominance_terminal_used", False)
            and state.get("incumbent") is not None
            and history
            and history[0].get("status") == "initialized"
            and not any(item.get("target_oracle_used", True)
                        for item in history)
            and int(row.get("posterior_dominance_switch_count", -1))
            == int(state.get("switch_count", -2))
            and np.isclose(float(state.get("delta_switch", np.nan)), 0.05)
        )
    return bool(
        not row.get("posterior_dominance_terminal_used", False)
        and state.get("incumbent") is None
        and not list(state.get("history") or [])
        and int(state.get("switch_count", 0)) == 0
    )


def _misspecification_contract(prior, expected_mode):
    mode = str(prior.get("source_mean_misspecification_mode", "missing"))
    if mode != expected_mode or prior.get("target_oracle_used", True):
        return False
    applied = bool(prior.get("source_mean_misspecification_applied", False))
    scale = float(prior.get("source_mean_misspecification_scale", 1.0))
    if expected_mode == "none":
        return bool(not applied and np.isclose(scale, 1.0))
    return bool(
        applied
        and scale >= 1.0
        and float(prior.get(
            "source_mean_prior_covariance_trace_after", 0.0)) + 1e-12
        >= float(prior.get(
            "source_mean_prior_covariance_trace_before", 0.0))
        and float(prior.get("source_mean_residual_floor_after", 0.0)) + 1e-12
        >= float(prior.get("source_mean_residual_floor_before", 0.0))
        and prior.get("misspecification_uncertainty_can_only_increase", False)
        and not prior.get("target_oracle_used_for_misspecification", True)
    )


def _cell(rows, variant, scenario, expected_seeds):
    value = _v28_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    dominance = [dict(row.get("posterior_dominance") or {})
                 for row in group]
    expected_dominance = variant != "v28_direct_control"
    expected_misspecification = MISSPECIFICATION_MODES[variant]
    value.update({
        "posterior_dominance_contract": bool(group) and all(
            _dominance_record_contract(row, expected_dominance)
            for row in group),
        "posterior_dominance_enabled_flags": sorted(set(bool(
            row.get("posterior_dominance_enabled", False))
            for row in group)),
        "posterior_dominance_terminal_used_count": int(sum(bool(
            row.get("posterior_dominance_terminal_used", False))
            for row in group)),
        "posterior_dominance_switch_count": int(sum(int(
            row.get("posterior_dominance_switch_count", 0) or 0)
            for row in group)),
        "posterior_dominance_methods": sorted(set(str(
            item.get("method", "missing")) for item in dominance)),
        "misspecification_modes": sorted(set(str(
            prior.get("source_mean_misspecification_mode", "missing"))
            for prior in priors)),
        "misspecification_contract": bool(priors) and all(
            _misspecification_contract(prior, expected_misspecification)
            for prior in priors),
        "misspecification_scales": [float(
            prior.get("source_mean_misspecification_scale", 1.0))
            for prior in priors],
    })
    return value


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in MEAN_SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in MEAN_SCENARIOS]
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
            "adaptive_losses": int(sum(
                item["adaptive_loss_count"] for item in values)),
            "audit_pool_true_certificates": int(sum(
                item["audit_pool_true_certificate_count"]
                for item in values)),
            "audit_pool_false_certificates": int(sum(
                item["audit_pool_false_certificate_count"]
                for item in values)),
            "oracle_both_certificates": int(sum(
                item["oracle_both_certified_total"] for item in values)),
            "posterior_dominance_switches": int(sum(
                item["posterior_dominance_switch_count"]
                for item in values)),
        }
        for variant, values in grouped.items()
    }
    global_checks = {
        "all_cells_complete": all(
            item["complete"] for item in cells.values()),
        "oracle_free": all(item["oracle_free"] for item in cells.values()),
        "paired_initial_design_and_archive": bool(paired),
    }
    control = grouped["v28_direct_control"]
    control_totals = totals["v28_direct_control"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        queue = cells[(variant, *MEAN_SCENARIOS[-1])]
        checks[variant] = {
            **global_checks,
            "single_empirical_bayes_hyperlaw_contract": bool(all(
                item["single_hyperlaw_contract"]
                and item["posterior_target_data_used"]
                for item in values)),
            "independent_variance_task_posterior_contract": bool(all(
                item["hvd_source_task_weight_modes"] == ["independent"]
                and item["variance_task_posterior_modes"]
                == ["replication_only"]
                and item["variance_task_posterior_statuses"]
                == ["frozen_source_prior"]
                and item["variance_task_evidence_counts"] == [0]
                and item["variance_task_effective_dofs"] == [0]
                and item["variance_task_uses_target_mean"] == [False]
                for item in values)),
            "split_cumulative_authority_contract": bool(all(
                item["certification_head_authorities"]
                == ["split_gpr_cumulative_hvd"]
                and item["decomposition_head_authorities"]
                == ["split_gpr_cumulative_hvd"]
                and item["recommendation_certificate_sources"]
                == ["split_aggregate_gpr_cumulative_hvd"]
                for item in values)),
            "mean_misspecification_contract": bool(all(
                item["misspecification_contract"] for item in values)),
            "posterior_dominance_contract": bool(all(
                item["posterior_dominance_contract"] for item in values)),
            "all_initially_safe_incumbents_preserved": bool(
                totals[variant]["true_feasible"]
                == len(MEAN_SCENARIOS) * expected_seeds
                and totals[variant]["adaptive_losses"] == 0),
            "queue_seed3_failure_repaired": bool(
                queue["true_feasible_count"] == expected_seeds
                and queue["adaptive_loss_count"] == 0),
            "audit_pool_false_certificates_eliminated": bool(
                totals[variant]["audit_pool_false_certificates"] == 0),
            "nonvacuous_true_certificate_coverage": bool(
                totals[variant]["audit_pool_true_certificates"] > 0),
            "strictly_safer_than_v28_direct": bool(
                totals[variant]["true_feasible"]
                > control_totals["true_feasible"]
                and totals[variant]["adaptive_losses"]
                < control_totals["adaptive_losses"]
                and totals[variant]["audit_pool_false_certificates"]
                < control_totals["audit_pool_false_certificates"]),
            "mean_rank_preserved_at_least_2_of_3": bool(
                _count_higher_nonworse(
                    values, control,
                    "median_constraint_mean_rank_correlation") >= 2),
            "mean_mae_preserved_at_least_2_of_3": bool(
                _count_lower_nonworse(
                    values, control, "median_mean_abs_error",
                    multiplier=1.02, offset=0.002) >= 2),
        }
    eligible = [
        variant for variant in CHALLENGERS
        if all(checks[variant].values())
    ]
    complexity_order = {
        "v29_direct_dominance": 0,
        "v29_scale_dominance": 1,
        "v29_directional_dominance": 2,
    }
    recommended = (
        min(
            eligible,
            key=lambda variant: (
                -totals[variant]["audit_pool_true_certificates"],
                complexity_order[variant],
            ),
        )
        if eligible else None
    )
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(MEAN_SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "totals": totals,
        "global_checks": global_checks,
        "variant_checks": checks,
        "promotion_eligible": eligible,
        "post_gate_baseline_recommendation": recommended,
        "post_gate_baseline_selection_rule": (
            "preserve every initially safe incumbent; eliminate all raw-pool "
            "false certificates while retaining nonvacuous true coverage; "
            "preserve mean fit; then prefer the simplest calibration"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or args.root / "mean_alignment_v29_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "promotion_eligible": result["promotion_eligible"],
        "post_gate_baseline_recommendation": result[
            "post_gate_baseline_recommendation"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
