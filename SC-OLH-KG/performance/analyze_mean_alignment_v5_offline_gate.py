#!/usr/bin/env python3
"""Analyze V5 role alignment and source-mean misspecification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "v4_latent_control",
    "role_match_only",
    "misspec_scalar",
    "misspec_directional",
    "v5_role_scalar",
    "v5_role_directional",
)
CHALLENGERS = VARIANTS[1:]
ROLE_VARIANTS = {
    "role_match_only", "v5_role_scalar", "v5_role_directional"}
MISSPEC_VARIANTS = {
    "misspec_scalar", "misspec_directional",
    "v5_role_scalar", "v5_role_directional",
}
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
            if variant is not None:
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


def _component_diagnostics(prior):
    return [
        value for value in prior.get("component_deviation_diagnostics", [])
        if str(value.get("name", "")).startswith("source:")
    ]


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
    source_components = [
        component
        for prior in priors
        for component in _component_diagnostics(prior)
    ]
    role_required = variant in ROLE_VARIANTS
    misspec_required = variant in MISSPEC_VARIANTS
    misspec_monotone = all(
        bool(value.get(
            "misspecification_uncertainty_can_only_increase", False))
        and float(value.get("source_mean_prior_covariance_trace_after", 0.0))
        + 1e-12 >= float(value.get(
            "source_mean_prior_covariance_trace_before", 0.0))
        and float(value.get("source_mean_residual_floor_after", 0.0))
        + 1e-12 >= float(value.get(
            "source_mean_residual_floor_before", 0.0))
        for value in source_components
    ) if misspec_required and source_components else not misspec_required
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
        "role_matching_oracle_free": bool(
            not role_required or (
                group and all(
                    bool(value.get("channel_role_alignment_used", False))
                    and not bool(value.get(
                        "channel_role_target_matching_uses_labels", True))
                    and not bool(value.get(
                        "channel_role_target_matching_uses_oracle", True))
                    for value in contracts
                )
            )
        ),
        "misspecification_uncertainty_monotone": bool(misspec_monotone),
        "median_misspecification_scale": _median([
            value.get("source_mean_misspecification_scale")
            for value in source_components]),
        "median_misspecification_directional_mass": _median([
            value.get("source_mean_misspecification_directional_mass")
            for value in source_components]),
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
        "median_best_feasible_true_margin": _median([
            value.get("boundary_raw_pool_best_feasible_true_margin")
            for value in raw]),
        "median_source_posterior_weight": _median([
            value.get("source_posterior_weight") for value in priors]),
        "median_target_null_weight": _median([
            value.get("target_only_posterior_weight") for value in priors]),
    }


def _candidate_gate(cells, variant, expected_seeds):
    challenger = [cells[(variant, *scenario)] for scenario in SCENARIOS]
    control = [
        cells[("v4_latent_control", *scenario)] for scenario in SCENARIOS]
    rank_nonworse = sum(
        left["median_mean_rank_correlation"] is not None
        and right["median_mean_rank_correlation"] is not None
        and left["median_mean_rank_correlation"]
        >= right["median_mean_rank_correlation"] - 0.02
        for left, right in zip(challenger, control)
    )
    mae_nonworse = sum(
        left["median_mean_abs_error"] is not None
        and right["median_mean_abs_error"] is not None
        and left["median_mean_abs_error"]
        <= 1.25 * right["median_mean_abs_error"] + 0.02
        for left, right in zip(challenger, control)
    )
    variance_nonworse = sum(
        left["median_variance_log_rmse"] is not None
        and right["median_variance_log_rmse"] is not None
        and left["median_variance_log_rmse"]
        <= 1.05 * right["median_variance_log_rmse"]
        for left, right in zip(challenger, control)
    )
    challenger_false = sum(
        value["total_false_certified_count"] for value in challenger)
    control_false = sum(
        value["total_false_certified_count"] for value in control)
    fs4 = cells[(variant, "FactorShockStatePolicyRZDT1", 4.0)]
    checks = {
        "all_cells_complete": all(value["complete"] for value in challenger),
        "oracle_free": all(value["oracle_free"] for value in challenger),
        "shared_observable_dual_head": all(
            value["shared_observable_dual_head"] for value in challenger),
        "role_matching_oracle_free": all(
            value["role_matching_oracle_free"] for value in challenger),
        "misspecification_uncertainty_monotone": all(
            value["misspecification_uncertainty_monotone"]
            for value in challenger),
        "factor_shock_scale4_support_at_least_4_of_5": bool(
            fs4["true_feasible_support_seeds"]
            >= max(1, int(expected_seeds) - 1)),
        "mean_rank_nonworse_at_least_3_of_4": bool(rank_nonworse >= 3),
        "mean_mae_nonworse_at_least_3_of_4": bool(mae_nonworse >= 3),
        "variance_head_nonworse_at_least_3_of_4": bool(
            variance_nonworse >= 3),
        "strict_false_certification_reduction": bool(
            challenger_false < control_false),
    }
    return {
        "variant": variant,
        "challenger_false_certified_count": int(challenger_false),
        "control_false_certified_count": int(control_false),
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
        for variant in CHALLENGERS
    ]
    candidates.sort(key=lambda value: (
        int(value["pass"]), value["check_count"],
        -value["challenger_false_certified_count"],
    ), reverse=True)
    selected = candidates[0]
    complete = all(value["complete"] for value in cells.values())
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "all_matrix_cells_complete": bool(complete),
        "cells": list(cells.values()),
        "candidate_gates": candidates,
        "selected_variant": selected["variant"],
        "advance_to_sequential": bool(complete and selected["pass"]),
        "scientific_contract": {
            "shared_observable_input": "state/trajectory exposure e(x)",
            "mean_coordinate": "phi_mu=h_mu(role(e))",
            "variance_coordinate": "psi_v=h_v(e)=(A,N)",
            "target_role_matching": "unlabelled deterministic policy pool",
            "source_mean_misspecification": (
                "conservative scale plus optional PSD directional covariance"),
            "misspecification_covariance_can_only_increase": True,
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
    output = args.out or args.root / "mean_alignment_v5_offline_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_variant": result["selected_variant"],
        "advance_to_sequential": result["advance_to_sequential"],
        "candidate_gates": result["candidate_gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
