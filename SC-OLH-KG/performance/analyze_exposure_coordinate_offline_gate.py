#!/usr/bin/env python3
"""Analyze the preregistered observable-exposure sufficiency gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "latent_control",
    "profile_phi_r2",
    "learned_exposure_phi_r2",
    "learned_exposure_phi_r4",
    "learned_exposure_phi_r8",
    "provider_exposure_phi_r2",
    "provider_exposure_phi_r4",
    "provider_exposure_phi_r8",
)
EXPECTED_SCENARIOS = (
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
            row["result_path"] = str(path)
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
    audit = [row.get("boundary_raw_pool_truth_diagnostics") or {}
             for row in group]
    return {
        "variant": variant,
        "domain": scenario[0],
        "shock_scale": scenario[1],
        "n": int(len(group)),
        "unique_seed_count": int(len(seeds)),
        "complete": bool(
            len(group) == int(expected_seeds)
            and len(seeds) == int(expected_seeds)),
        "oracle_free": bool(all(
            not bool(row.get("audit", {}).get(
                "tcb_target_structural_provider_used", False))
            for row in group
        )),
        "raw_pool_true_feasible_seed_count": int(sum(
            bool(value.get("boundary_raw_pool_has_true_feasible", False))
            for value in audit)),
        "median_raw_pool_feasible_rate": _median([
            value.get("boundary_raw_pool_true_feasible_rate")
            for value in audit]),
        "median_mean_rank_correlation": _median([
            value.get("boundary_raw_pool_constraint_mean_rank_correlation")
            for value in audit]),
        "median_chance_rank_correlation": _median([
            value.get("boundary_raw_pool_chance_margin_rank_correlation")
            for value in audit]),
        "median_mean_abs_error": _median([
            value.get("boundary_raw_pool_constraint_mean_median_abs_error")
            for value in audit]),
        "median_variance_log_error": _median([
            value.get("boundary_raw_pool_variance_median_abs_log_error")
            for value in audit]),
        "median_full_certified_count": _median([
            value.get("boundary_raw_pool_full_certified_count")
            for value in audit]),
        "median_oracle_both_certified_count": _median([
            value.get(
                "boundary_raw_pool_oracle_mean_variance_certified_count")
            for value in audit]),
        "failure_layers": {
            layer: int(sum(
                value.get("boundary_raw_pool_failure_layer") == layer
                for value in audit
            ))
            for layer in (
                "candidate_support", "constraint_mean",
                "cumulative_variance", "epistemic_or_safety_depth", "closed",
            )
        },
    }


def _metric_nonworse(challenger, control, metric, tolerance=0.0):
    left = challenger.get(metric)
    right = control.get(metric)
    return bool(
        left is not None and right is not None
        and float(left) >= float(right) - float(tolerance)
    )


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds)
        for variant in VARIANTS
        for scenario in EXPECTED_SCENARIOS
    }
    rank_candidates = [
        variant for variant in VARIANTS
        if variant.startswith("learned_exposure_phi_")
    ]
    rank_scores = {}
    for variant in rank_candidates:
        nonworse = 0
        mae_safe = 0
        mean_rank = []
        for scenario in EXPECTED_SCENARIOS:
            challenger = cells[(variant, *scenario)]
            control = cells[("latent_control", *scenario)]
            nonworse += int(_metric_nonworse(
                challenger, control, "median_mean_rank_correlation", 0.02))
            left_mae = challenger["median_mean_abs_error"]
            right_mae = control["median_mean_abs_error"]
            mae_safe += int(
                left_mae is not None and right_mae is not None
                and left_mae <= max(2.0 * right_mae, right_mae + 0.05)
            )
            if challenger["median_mean_rank_correlation"] is not None:
                mean_rank.append(challenger["median_mean_rank_correlation"])
        rank_scores[variant] = {
            "rank_nonworse_scenarios": int(nonworse),
            "mae_noncatastrophic_scenarios": int(mae_safe),
            "median_across_scenario_mean_rank": _median(mean_rank),
        }
    eligible = [
        variant for variant, score in rank_scores.items()
        if score["rank_nonworse_scenarios"] >= 3
        and score["mae_noncatastrophic_scenarios"] == 4
    ]
    selected = (
        max(
            eligible,
            key=lambda variant: (
                rank_scores[variant]["median_across_scenario_mean_rank"]
                if rank_scores[variant]["median_across_scenario_mean_rank"]
                is not None else -float("inf"),
                -int(variant.rsplit("r", 1)[1]),
            ),
        )
        if eligible else None
    )
    fs4 = (
        cells[(selected, "FactorShockStatePolicyRZDT1", 4.0)]
        if selected is not None else None
    )
    oracle_both_scenarios = 0
    if selected is not None:
        oracle_both_scenarios = int(sum(
            (cells[(selected, *scenario)][
                "median_oracle_both_certified_count"] or 0.0) > 0.0
            for scenario in EXPECTED_SCENARIOS
        ))
    all_complete = all(cell["complete"] for cell in cells.values())
    provider_cells = [
        cells[(variant, *scenario)]
        for variant in VARIANTS if variant.startswith("provider_exposure_phi_")
        for scenario in EXPECTED_SCENARIOS
    ]
    bottleneck_rows = []
    for scenario in EXPECTED_SCENARIOS:
        learned_rows = [
            cells[(variant, *scenario)] for variant in rank_candidates]
        provider_rows = [
            cells[(variant, *scenario)]
            for variant in VARIANTS
            if variant.startswith("provider_exposure_phi_")
        ]

        def best(rows):
            finite = [row for row in rows
                      if row["median_mean_rank_correlation"] is not None]
            return (
                None if not finite else max(
                    finite,
                    key=lambda row: row["median_mean_rank_correlation"],
                )
            )

        learned_best = best(learned_rows)
        provider_best = best(provider_rows)
        learned_rank = (
            None if learned_best is None
            else learned_best["median_mean_rank_correlation"])
        provider_rank = (
            None if provider_best is None
            else provider_best["median_mean_rank_correlation"])
        gap = (
            None if learned_rank is None or provider_rank is None
            else float(provider_rank - learned_rank)
        )
        bottleneck_rows.append({
            "domain": scenario[0],
            "shock_scale": scenario[1],
            "best_learned_variant": (
                None if learned_best is None else learned_best["variant"]),
            "best_provider_variant": (
                None if provider_best is None else provider_best["variant"]),
            "best_learned_mean_rank": learned_rank,
            "best_provider_mean_rank": provider_rank,
            "provider_minus_learned_rank": gap,
            "representation_bottleneck_signal": bool(
                gap is not None and gap > 0.10),
            "head_or_calibration_bottleneck_signal": bool(
                provider_rank is not None
                and cells[("latent_control", *scenario)][
                    "median_mean_rank_correlation"] is not None
                and provider_rank
                < cells[("latent_control", *scenario)][
                    "median_mean_rank_correlation"] - 0.02
            ),
        })
    promotion = {
        "all_cells_complete": bool(all_complete),
        "oracle_free_challenger_selected": selected is not None,
        "selected_variant": selected,
        "mean_rank_nonworse_in_at_least_3_of_4": bool(selected is not None),
        "mean_mae_noncatastrophic_in_all_4": bool(selected is not None),
        "factor_shock_scale4_shared_pool_support_in_at_least_4_of_5": bool(
            fs4 is not None
            and fs4["raw_pool_true_feasible_seed_count"]
            >= max(1, int(expected_seeds) - 1)
        ),
        "oracle_both_nonvacuous_in_at_least_3_of_4": bool(
            oracle_both_scenarios >= 3),
        "provider_track_is_upper_bound": bool(
            provider_cells and all(not cell["oracle_free"]
                                   for cell in provider_cells)),
    }
    promotion["advance_to_sequential"] = bool(all(
        value for key, value in promotion.items()
        if key != "selected_variant"
    ))
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(EXPECTED_SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "rank_scores": rank_scores,
        "bottleneck_diagnosis": bottleneck_rows,
        "promotion_gate": promotion,
        "scientific_contract": {
            "shared_audit_pool": (
                "universal_low_frequency_no_source_templates"),
            "main_challenger_input": "frozen_source_learned_exposure",
            "provider_input_role": "structure_aware_upper_bound_only",
            "mean_and_variance_heads": "independent",
            "target_truth_timing": "post_run_audit_only",
            "target_oracle_used_for_decision": False,
        },
    }


def markdown(summary):
    lines = [
        "# Exposure Coordinate Offline Gate",
        "",
        f"Rows: `{summary['row_count']}`",
        "",
        "| variant | domain | shock | n | support seeds | mean rank | chance rank | mean MAE | oracle-both cert |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda value: "NA" if value is None else f"{value:.4g}"
    for row in summary["cells"]:
        lines.append(
            f"| {row['variant']} | {row['domain']} | {row['shock_scale']:g} "
            f"| {row['n']} | {row['raw_pool_true_feasible_seed_count']} "
            f"| {fmt(row['median_mean_rank_correlation'])} "
            f"| {fmt(row['median_chance_rank_correlation'])} "
            f"| {fmt(row['median_mean_abs_error'])} "
            f"| {fmt(row['median_oracle_both_certified_count'])} |"
        )
    lines.extend([
        "",
        "## Promotion Gate",
        "",
        "```json",
        json.dumps(summary["promotion_gate"], indent=2, sort_keys=True),
        "```",
        "",
        "Target truth is used only after the frozen recommendation and never changes a query.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or (args.root / "exposure_coordinate_offline_gate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(markdown(summary))
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
