#!/usr/bin/env python3
"""Analyze V2's preregistered observable state/trajectory coordinate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "latent_control",
    "learned_exposure_v1_phi_r4",
    "observable_state_phi_r2",
    "observable_state_phi_r4",
    "observable_state_phi_r8",
    "provider_exposure_phi_r2",
    "provider_exposure_phi_r4",
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
    audits = [row.get("boundary_raw_pool_truth_diagnostics") or {}
              for row in group]
    seeds = {int(row["seed"]) for row in group if row.get("seed") is not None}
    target_provider_used = [bool(
        row.get("audit", {}).get(
            "tcb_target_structural_provider_used", False))
        for row in group
    ]
    observable_used = [bool(
        row.get("audit", {}).get("observable_state_exposure_used", False))
        for row in group
    ]
    return {
        "variant": variant,
        "domain": scenario[0],
        "shock_scale": scenario[1],
        "n": int(len(group)),
        "complete": bool(
            len(group) == int(expected_seeds)
            and len(seeds) == int(expected_seeds)),
        "oracle_free": bool(group and not any(target_provider_used)),
        "observable_state_input": bool(group and all(observable_used)),
        "raw_pool_true_feasible_seed_count": int(sum(
            bool(value.get("boundary_raw_pool_has_true_feasible", False))
            for value in audits)),
        "median_raw_pool_feasible_rate": _median([
            value.get("boundary_raw_pool_true_feasible_rate")
            for value in audits]),
        "median_mean_rank_correlation": _median([
            value.get("boundary_raw_pool_constraint_mean_rank_correlation")
            for value in audits]),
        "median_chance_rank_correlation": _median([
            value.get("boundary_raw_pool_chance_margin_rank_correlation")
            for value in audits]),
        "median_mean_abs_error": _median([
            value.get("boundary_raw_pool_constraint_mean_median_abs_error")
            for value in audits]),
        "median_oracle_both_certified_count": _median([
            value.get("boundary_raw_pool_oracle_mean_variance_certified_count")
            for value in audits]),
        "median_best_feasible_oracle_both_margin": _median([
            value.get("boundary_raw_pool_best_feasible_oracle_mean_variance_margin")
            for value in audits]),
        "median_best_feasible_epistemic_radius": _median([
            value.get("boundary_raw_pool_best_feasible_epistemic_radius")
            for value in audits]),
        "failure_layers": {
            layer: int(sum(
                value.get("boundary_raw_pool_failure_layer") == layer
                for value in audits))
            for layer in (
                "candidate_support", "constraint_mean",
                "cumulative_variance", "epistemic_or_safety_depth", "closed",
            )
        },
    }


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    challengers = [
        variant for variant in VARIANTS
        if variant.startswith("observable_state_phi_")
    ]
    scores = {}
    for variant in challengers:
        nonworse = 0
        mae_safe = 0
        ranks = []
        for scenario in SCENARIOS:
            row = cells[(variant, *scenario)]
            control = cells[("latent_control", *scenario)]
            left_rank = row["median_mean_rank_correlation"]
            right_rank = control["median_mean_rank_correlation"]
            nonworse += int(
                left_rank is not None and right_rank is not None
                and left_rank >= right_rank - 0.02)
            left_mae = row["median_mean_abs_error"]
            right_mae = control["median_mean_abs_error"]
            mae_safe += int(
                left_mae is not None and right_mae is not None
                and left_mae <= max(2.0 * right_mae, right_mae + 0.05))
            if left_rank is not None:
                ranks.append(left_rank)
        scores[variant] = {
            "rank_nonworse_scenarios": int(nonworse),
            "mae_noncatastrophic_scenarios": int(mae_safe),
            "median_across_scenario_mean_rank": _median(ranks),
        }
    eligible = [
        variant for variant, score in scores.items()
        if score["rank_nonworse_scenarios"] >= 3
        and score["mae_noncatastrophic_scenarios"] == 4
    ]
    selected = (
        max(eligible, key=lambda variant: (
            scores[variant]["median_across_scenario_mean_rank"],
            -int(variant.rsplit("r", 1)[1]),
        ))
        if eligible else None
    )
    oracle_both_scenarios = 0
    if selected is not None:
        oracle_both_scenarios = int(sum(
            (cells[(selected, *scenario)][
                "median_oracle_both_certified_count"] or 0.0) > 0.0
            for scenario in SCENARIOS
        ))
    fs4_candidates = [
        cells[(
            variant,
            "FactorShockStatePolicyRZDT1",
            4.0,
        )]
        for variant in challengers
    ]
    provider_cells = [
        cells[(variant, *scenario)]
        for variant in VARIANTS if variant.startswith("provider_exposure_phi_")
        for scenario in SCENARIOS
    ]
    selected_cells = (
        [] if selected is None
        else [cells[(selected, *scenario)] for scenario in SCENARIOS]
    )
    promotion = {
        "all_cells_complete": bool(all(
            value["complete"] for value in cells.values())),
        "oracle_free_observable_challenger_selected": selected is not None,
        "selected_variant": selected,
        "mean_rank_nonworse_in_at_least_3_of_4": selected is not None,
        "mean_mae_noncatastrophic_in_all_4": selected is not None,
        "factor_shock_scale4_support_in_at_least_4_of_5": bool(
            fs4_candidates
            and max(row["raw_pool_true_feasible_seed_count"]
                    for row in fs4_candidates)
            >= max(1, int(expected_seeds) - 1)),
        "oracle_both_nonvacuous_in_at_least_3_of_4": bool(
            oracle_both_scenarios >= 3),
        "observable_state_track_is_oracle_free": bool(
            challengers
            and all(
                cells[(variant, *scenario)]["oracle_free"]
                and cells[(variant, *scenario)]["observable_state_input"]
                for variant in challengers for scenario in SCENARIOS
            )),
        "provider_track_is_upper_bound": bool(
            provider_cells and all(not row["oracle_free"]
                                   for row in provider_cells)),
    }
    promotion["advance_to_sequential"] = bool(all(
        value for key, value in promotion.items()
        if key != "selected_variant"
    ))
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "rank_scores": scores,
        "promotion_gate": promotion,
        "scientific_contract": {
            "shared_raw_pool": "dense_universal_plus_low_frequency",
            "main_input": "target_observable_state_trajectory_exposure",
            "independent_heads": True,
            "target_outcomes_used_to_build_exposure": False,
            "target_risk_provider_used_by_main_track": False,
            "target_truth_timing": "post_run_audit_only",
        },
    }


def markdown(summary):
    lines = [
        "# Observable State Coordinate Offline Gate",
        "",
        f"Rows: `{summary['row_count']}`",
        "",
        "| variant | domain | shock | n | support seeds | mean rank | chance rank | mean MAE | oracle-both | best oracle margin | epi radius |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda value: "NA" if value is None else f"{value:.4g}"
    for row in summary["cells"]:
        lines.append(
            f"| {row['variant']} | {row['domain']} | {row['shock_scale']:g} "
            f"| {row['n']} | {row['raw_pool_true_feasible_seed_count']} "
            f"| {fmt(row['median_mean_rank_correlation'])} "
            f"| {fmt(row['median_chance_rank_correlation'])} "
            f"| {fmt(row['median_mean_abs_error'])} "
            f"| {fmt(row['median_oracle_both_certified_count'])} "
            f"| {fmt(row['median_best_feasible_oracle_both_margin'])} "
            f"| {fmt(row['median_best_feasible_epistemic_radius'])} |"
        )
    lines.extend([
        "", "## Promotion Gate", "", "```json",
        json.dumps(summary["promotion_gate"], indent=2, sort_keys=True),
        "```", "",
        "Target truth is used only after every target decision is frozen.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or (
        args.root / "observable_state_coordinate_offline_gate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(markdown(summary))
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
