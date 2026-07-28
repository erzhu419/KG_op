#!/usr/bin/env python3
"""Analyze the shared-observable independent mean/variance head gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "legacy_policy_control",
    "observable_mean_only",
    "observable_variance_only",
    "observable_dual_head",
)
SCENARIOS = (
    ("FactorShockStatePolicyRZDT1", 0.0),
    ("FactorShockStatePolicyRZDT1", 4.0),
    ("InventorySupplyChain", 1.0),
    ("QueueResourceControl", 1.0),
)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _median(values):
    values = [value for value in (_number(item) for item in values)
              if value is not None]
    return None if not values else float(median(values))


def _variant(payload, row):
    label = str(row.get("experiment_variant") or payload.get(
        "experiment_variant", ""))
    for variant in VARIANTS:
        if f"/{variant}/" in f"/{label}/" or label.endswith(f"/{variant}"):
            return variant
    return None


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for raw in payload.get("rows", []):
            row = dict(raw)
            variant = _variant(payload, row)
            if variant is None:
                continue
            row["gate_variant"] = variant
            rows.append(row)
    return rows


def _scenario(row):
    return (
        str(row.get("heldout")),
        float(row.get("target_shared_shock_scale", 1.0)),
    )


def _cell(rows, variant, scenario, expected_seeds):
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    seeds = {int(row["seed"]) for row in group if row.get("seed") is not None}
    raw = [row.get("boundary_raw_pool_truth_diagnostics") or {}
           for row in group]
    contracts = [row.get("mean_risk_coordinate_contract") or {}
                 for row in group]
    audits = [row.get("audit") or {} for row in group]
    failure_layers = {}
    for value in raw:
        layer = str(value.get(
            "boundary_raw_pool_failure_layer", "unknown"))
        failure_layers[layer] = failure_layers.get(layer, 0) + 1
    epistemic_to_safety_depth = []
    for value in raw:
        epistemic = _number(value.get(
            "boundary_raw_pool_best_feasible_epistemic_radius"))
        true_margin = _number(value.get(
            "boundary_raw_pool_best_feasible_true_margin"))
        if epistemic is not None and true_margin is not None:
            epistemic_to_safety_depth.append(
                epistemic / max(-true_margin, 1e-12))
    return {
        "variant": variant,
        "domain": scenario[0],
        "shock_scale": scenario[1],
        "n": int(len(group)),
        "complete": bool(
            len(group) == int(expected_seeds)
            and len(seeds) == int(expected_seeds)),
        "oracle_free": bool(group and all(
            bool(value.get("admissible_oracle_free_transfer", False))
            for value in audits)),
        "shared_observable_dual_head": bool(group and all(
            bool(value.get("shared_observable_exposure_input", False))
            for value in contracts)),
        "median_mean_rank_correlation": _median([
            value.get("boundary_raw_pool_constraint_mean_rank_correlation")
            for value in raw]),
        "median_mean_abs_error": _median([
            value.get("boundary_raw_pool_constraint_mean_median_abs_error")
            for value in raw]),
        "median_variance_log_rmse": _median([
            row.get("variance_log_rmse") for row in group]),
        "median_certified_variance_log_rmse": _median([
            row.get("certified_variance_log_rmse") for row in group]),
        "median_variance_upper_coverage": _median([
            row.get("variance_upper_coverage") for row in group]),
        "true_feasible_support_seeds": int(sum(
            bool(value.get("boundary_raw_pool_has_true_feasible", False))
            for value in raw)),
        "median_true_feasible_count": _median([
            value.get("boundary_raw_pool_true_feasible_count")
            for value in raw]),
        "median_posterior_certified_count": _median([
            value.get("boundary_raw_pool_full_certified_count")
            for value in raw]),
        "median_oracle_both_certified_count": _median([
            value.get("boundary_raw_pool_oracle_mean_variance_certified_count")
            for value in raw]),
        "median_best_feasible_epistemic_radius": _median([
            value.get("boundary_raw_pool_best_feasible_epistemic_radius")
            for value in raw]),
        "median_best_feasible_true_margin": _median([
            value.get("boundary_raw_pool_best_feasible_true_margin")
            for value in raw]),
        "median_best_feasible_oracle_mean_variance_margin": _median([
            value.get(
                "boundary_raw_pool_best_feasible_oracle_mean_variance_margin")
            for value in raw]),
        "median_epistemic_to_safety_depth_ratio": _median(
            epistemic_to_safety_depth),
        "failure_layer_counts": failure_layers,
    }


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    dual = [cells[("observable_dual_head", *scenario)]
            for scenario in SCENARIOS]
    mean_only = [cells[("observable_mean_only", *scenario)]
                 for scenario in SCENARIOS]
    legacy = [cells[("legacy_policy_control", *scenario)]
              for scenario in SCENARIOS]
    rmse_nonworse = 0
    rmse_strict_gain = 0
    coverage_nonworse = 0
    mean_rank_nonworse = 0
    mean_mae_noncatastrophic = 0
    oracle_both_nonvacuous = 0
    for left, right, baseline in zip(dual, mean_only, legacy):
        left_rmse = left["median_variance_log_rmse"]
        right_rmse = right["median_variance_log_rmse"]
        if left_rmse is not None and right_rmse is not None:
            rmse_nonworse += int(left_rmse <= 1.05 * right_rmse + 1e-12)
            rmse_strict_gain += int(left_rmse <= 0.98 * right_rmse)
        left_coverage = left["median_variance_upper_coverage"]
        right_coverage = right["median_variance_upper_coverage"]
        if left_coverage is not None and right_coverage is not None:
            coverage_nonworse += int(left_coverage >= right_coverage - 0.05)
        left_rank = left["median_mean_rank_correlation"]
        baseline_rank = baseline["median_mean_rank_correlation"]
        if left_rank is not None and baseline_rank is not None:
            mean_rank_nonworse += int(left_rank >= baseline_rank - 0.02)
        left_mae = left["median_mean_abs_error"]
        baseline_mae = baseline["median_mean_abs_error"]
        if left_mae is not None and baseline_mae is not None:
            mean_mae_noncatastrophic += int(
                left_mae <= max(2.0 * baseline_mae, baseline_mae + 0.05))
        oracle_both_nonvacuous += int(
            (left["median_oracle_both_certified_count"] or 0.0) > 0.0)
    fs4 = cells[(
        "observable_dual_head",
        "FactorShockStatePolicyRZDT1",
        4.0,
    )]
    promotion = {
        "all_cells_complete": bool(all(
            value["complete"] for value in cells.values())),
        "dual_head_track_is_oracle_free": bool(
            dual and all(value["oracle_free"] for value in dual)),
        "both_heads_share_observable_exposure": bool(
            dual and all(value["shared_observable_dual_head"] for value in dual)),
        "mean_rank_nonworse_in_at_least_3_of_4": bool(
            mean_rank_nonworse >= 3),
        "mean_mae_noncatastrophic_in_all_4": bool(
            mean_mae_noncatastrophic == 4),
        "variance_rmse_nonworse_in_at_least_3_of_4": bool(
            rmse_nonworse >= 3),
        "variance_rmse_strict_gain_in_at_least_1_of_4": bool(
            rmse_strict_gain >= 1),
        "variance_upper_coverage_nonworse_in_at_least_3_of_4": bool(
            coverage_nonworse >= 3),
        "factor_shock_scale4_support_in_at_least_4_of_5": bool(
            fs4["true_feasible_support_seeds"]
            >= max(1, int(expected_seeds) - 1)),
        "oracle_both_nonvacuous_in_at_least_3_of_4": bool(
            oracle_both_nonvacuous >= 3),
    }
    promotion["advance_to_sequential"] = bool(all(promotion.values()))
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "promotion_gate": promotion,
        "scientific_contract": {
            "shared_input": "observable_state_trajectory_exposure_e(x)",
            "mean_head": "phi_mu=h_mu(e)",
            "variance_head": "psi_v=h_v(e)=(A,N)",
            "head_parameters_shared": False,
            "variance_supervision": "ordinary_replicated_source_simulations",
            "target_truth_timing": "post_run_audit_only",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or (
        args.root / "observable_dual_head_offline_gate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["promotion_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
