#!/usr/bin/env python3
"""Analyze the V4 mean semantic-alignment and epistemic-calibration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "v3_control",
    "semantic_invariant",
    "boundary_quadratic",
    "epistemic_latent",
    "v4_ordered_joint",
    "v4_invariant_joint",
)
JOINT_VARIANTS = ("v4_ordered_joint", "v4_invariant_joint")
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


def _source_prior(row):
    numerics = row.get("gpr_numerics") or []
    if len(numerics) < 2:
        return {}
    return dict(numerics[1].get("source_parametric_prior") or {})


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
    priors = [_source_prior(row) for row in group]
    failure_layers = {}
    for value in raw:
        layer = str(value.get(
            "boundary_raw_pool_failure_layer", "unknown"))
        failure_layers[layer] = failure_layers.get(layer, 0) + 1
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
        "descriptor_modes": sorted(set(str(row.get(
            "meta_observable_mean_descriptor_mode", "unknown"))
            for row in group)),
        "feature_modes": sorted(set(str(row.get(
            "meta_observable_mean_feature_mode", "unknown"))
            for row in group)),
        "deviation_modes": sorted(set(str(row.get(
            "source_constraint_mean_deviation_mode", "unknown"))
            for row in group)),
        "adaptation_modes": sorted(set(str(row.get(
            "source_constraint_mean_adaptation_mode", "unknown"))
            for row in group)),
        "median_mean_rank_correlation": _median([
            value.get("boundary_raw_pool_constraint_mean_rank_correlation")
            for value in raw]),
        "median_mean_abs_error": _median([
            value.get("boundary_raw_pool_constraint_mean_median_abs_error")
            for value in raw]),
        "median_variance_log_rmse": _median([
            row.get("variance_log_rmse") for row in group]),
        "median_variance_upper_coverage": _median([
            row.get("variance_upper_coverage") for row in group]),
        "true_feasible_support_seeds": int(sum(
            bool(value.get("boundary_raw_pool_has_true_feasible", False))
            for value in raw)),
        "median_posterior_certified_count": _median([
            value.get("boundary_raw_pool_full_certified_count")
            for value in raw]),
        "median_oracle_both_certified_count": _median([
            value.get("boundary_raw_pool_oracle_mean_variance_certified_count")
            for value in raw]),
        "total_false_certified_count": int(sum(
            int(value.get("boundary_raw_pool_false_certified_count", 0) or 0)
            for value in raw)),
        "median_certificate_precision": _median([
            value.get("boundary_raw_pool_certificate_precision")
            for value in raw]),
        "median_best_feasible_epistemic_radius": _median([
            value.get("boundary_raw_pool_best_feasible_epistemic_radius")
            for value in raw]),
        "median_source_posterior_weight": _median([
            value.get("source_posterior_weight") for value in priors]),
        "median_target_null_weight": _median([
            value.get("target_only_posterior_weight") for value in priors]),
        "failure_layer_counts": failure_layers,
    }


def _nonworse(value, baseline, *, relative=1.0, additive=0.0):
    return bool(
        value is not None
        and baseline is not None
        and value <= relative * baseline + additive
    )


def _candidate_gate(cells, variant, expected_seeds):
    challenger = [cells[(variant, *scenario)] for scenario in SCENARIOS]
    control = [cells[("v3_control", *scenario)] for scenario in SCENARIOS]
    rank_nonworse = 0
    mae_noncatastrophic = 0
    variance_nonworse = 0
    coverage_nonworse = 0
    oracle_nonvacuous = 0
    for left, right in zip(challenger, control):
        left_rank = left["median_mean_rank_correlation"]
        right_rank = right["median_mean_rank_correlation"]
        if left_rank is not None and right_rank is not None:
            rank_nonworse += int(left_rank >= right_rank - 0.02)
        mae_noncatastrophic += int(_nonworse(
            left["median_mean_abs_error"],
            right["median_mean_abs_error"],
            relative=2.0,
            additive=0.05,
        ))
        variance_nonworse += int(_nonworse(
            left["median_variance_log_rmse"],
            right["median_variance_log_rmse"],
            relative=1.05,
        ))
        left_coverage = left["median_variance_upper_coverage"]
        right_coverage = right["median_variance_upper_coverage"]
        if left_coverage is not None and right_coverage is not None:
            coverage_nonworse += int(left_coverage >= right_coverage - 0.05)
        oracle_nonvacuous += int(
            (left["median_oracle_both_certified_count"] or 0.0) > 0.0)
    inventory = cells[(variant, "InventorySupplyChain", 1.0)]
    inventory_control = cells[("v3_control", "InventorySupplyChain", 1.0)]
    fs4 = cells[(variant, "FactorShockStatePolicyRZDT1", 4.0)]
    checks = {
        "all_cells_complete": bool(all(
            value["complete"] for value in challenger)),
        "oracle_free": bool(challenger and all(
            value["oracle_free"] for value in challenger)),
        "shared_observable_dual_head": bool(challenger and all(
            value["shared_observable_dual_head"] for value in challenger)),
        "factor_shock_scale4_support_at_least_4_of_5": bool(
            fs4["true_feasible_support_seeds"]
            >= max(1, int(expected_seeds) - 1)),
        "mean_rank_nonworse_at_least_3_of_4": bool(rank_nonworse >= 3),
        "mean_mae_noncatastrophic_all_4": bool(mae_noncatastrophic == 4),
        "inventory_mean_mae_gain_at_least_30_percent": bool(_nonworse(
            inventory["median_mean_abs_error"],
            inventory_control["median_mean_abs_error"],
            relative=0.70,
        )),
        "variance_rmse_nonworse_at_least_3_of_4": bool(
            variance_nonworse >= 3),
        "variance_coverage_nonworse_at_least_3_of_4": bool(
            coverage_nonworse >= 3),
        "oracle_both_certificate_nonvacuous_at_least_3_of_4": bool(
            oracle_nonvacuous >= 3),
        "zero_false_certificates": bool(all(
            value["total_false_certified_count"] == 0
            for value in challenger)),
    }
    return {
        "variant": variant,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "check_count": int(sum(checks.values())),
    }


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    candidates = [
        _candidate_gate(cells, variant, expected_seeds)
        for variant in JOINT_VARIANTS
    ]
    candidates.sort(key=lambda value: (
        int(value["pass"]), value["check_count"]), reverse=True)
    selected = candidates[0]
    all_cells_complete = bool(all(
        value["complete"] for value in cells.values()))
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "all_matrix_cells_complete": all_cells_complete,
        "cells": list(cells.values()),
        "joint_candidate_gates": candidates,
        "selected_joint_variant": selected["variant"],
        "advance_to_sequential": bool(
            all_cells_complete and selected["pass"]),
        "scientific_contract": {
            "shared_observable_input": "state/trajectory exposure e(x)",
            "mean_coordinate": "phi_mu=h_mu(e)",
            "variance_coordinate": "psi_v=h_v(e)=(A,N)",
            "source_discrepancy": (
                "latent coefficient covariance plus finite residual floor"),
            "reference_variance_preserved": True,
            "hvd_variance_head_changed": False,
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
        args.root / "mean_semantic_epistemic_offline_gate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_joint_variant": result["selected_joint_variant"],
        "advance_to_sequential": result["advance_to_sequential"],
        "joint_candidate_gates": result["joint_candidate_gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
