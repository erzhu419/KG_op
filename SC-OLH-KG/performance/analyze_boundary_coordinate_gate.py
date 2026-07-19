#!/usr/bin/env python3
"""Summarize the source-aligned boundary-coordinate causal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "latent_control",
    "phi_mean_only",
    "phi_mean_proposal",
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
    domain = str(row.get("heldout"))
    shock = float(row.get("target_shared_shock_scale", 1.0))
    return domain, shock


def summarize(rows, expected_seeds=5):
    cells = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["gate_variant"] == variant]
        for scenario in EXPECTED_SCENARIOS:
            group = [row for row in selected if _scenario(row) == scenario]
            seeds = {
                int(row["seed"])
                for row in group
                if row.get("seed") is not None
            }
            truth = [row.get("truth_pool_diagnostics") or {} for row in group]
            cert = [row.get("certificate_outcome_audit") or {} for row in group]
            adaptive = [row.get("adaptive_outcome_audit") or {} for row in group]
            proposal = [row.get("boundary_coordinate_proposal") or {}
                        for row in group]
            cells[(variant, *scenario)] = {
                "variant": variant,
                "domain": scenario[0],
                "shock_scale": scenario[1],
                "n": int(len(group)),
                "unique_seed_count": int(len(seeds)),
                "complete": bool(
                    len(group) == int(expected_seeds)
                    and len(seeds) == int(expected_seeds)),
                "true_feasible_count": int(sum(
                    bool(row.get("true_feasible", False)) for row in group)),
                "median_feasible_regret": _median([
                    row.get("feasible_simple_regret") for row in group
                    if row.get("true_feasible", False)
                ]),
                "posterior_certified_count": int(sum(
                    int(value.get("posterior_certified_count", 0))
                    for value in cert)),
                "false_certificate_count": int(sum(
                    int(value.get("false_certificate_count", 0))
                    for value in cert)),
                "adaptive_loss_count": int(sum(
                    bool(value.get("adaptive_loss", False))
                    for value in adaptive)),
                "median_final_margin": _median([
                    row.get("posterior_chance_margin") for row in group]),
                "median_mean_rank_correlation": _median([
                    value.get("mean_constraint_mean_rank_correlation")
                    for value in truth]),
                "median_chance_rank_correlation": _median([
                    value.get("mean_chance_margin_rank_correlation")
                    for value in truth]),
                "median_mean_abs_error": _median([
                    value.get("mean_constraint_mean_median_abs_error")
                    for value in truth]),
                "median_variance_log_error": _median([
                    value.get("mean_variance_median_abs_log_error")
                    for value in truth]),
                "median_phi_true_feasible_iteration_rate": _median([
                    value.get("phi_candidate_true_feasible_iteration_rate")
                    for value in truth]),
                "phi_generated_iterations": int(sum(
                    int(value.get("generated_iteration_count", 0))
                    for value in proposal)),
                "failure_layer_counts": {
                    layer: int(sum(
                        int((value.get("failure_layer_counts") or {}).get(
                            layer, 0))
                        for value in truth
                    ))
                    for layer in (
                        "candidate_support", "constraint_mean",
                        "cumulative_variance", "epistemic_or_safety_depth",
                        "closed",
                    )
                },
            }

    paired = {}
    scenarios = EXPECTED_SCENARIOS
    for challenger, control in (
        ("phi_mean_only", "latent_control"),
        ("phi_mean_proposal", "phi_mean_only"),
        ("phi_mean_proposal", "latent_control"),
    ):
        name = f"{challenger}_vs_{control}"
        wins = {
            "mean_rank": 0,
            "mean_rank_losses": 0,
            "mean_rank_pairs": 0,
            "final_margin": 0,
            "final_margin_losses": 0,
            "final_margin_pairs": 0,
            "feasibility": 0,
            "feasibility_losses": 0,
        }
        for scenario in scenarios:
            left = cells.get((challenger, *scenario))
            right = cells.get((control, *scenario))
            if left is None or right is None:
                continue
            lrank = left["median_mean_rank_correlation"]
            rrank = right["median_mean_rank_correlation"]
            if lrank is not None and rrank is not None:
                wins["mean_rank_pairs"] += 1
                wins["mean_rank"] += int(lrank > rrank + 1e-12)
                wins["mean_rank_losses"] += int(lrank < rrank - 1e-12)
            lmargin = left["median_final_margin"]
            rmargin = right["median_final_margin"]
            if lmargin is not None and rmargin is not None:
                wins["final_margin_pairs"] += 1
                wins["final_margin"] += int(lmargin < rmargin - 1e-12)
                wins["final_margin_losses"] += int(
                    lmargin > rmargin + 1e-12)
            wins["feasibility"] += int(
                left["true_feasible_count"] > right["true_feasible_count"])
            wins["feasibility_losses"] += int(
                left["true_feasible_count"] < right["true_feasible_count"])
        paired[name] = wins

    expected_cell_count = len(VARIANTS) * len(EXPECTED_SCENARIOS)
    all_complete = bool(len(cells) == expected_cell_count) and all(
        value["complete"] for value in cells.values())
    joint = [
        value for key, value in cells.items()
        if key[0] == "phi_mean_proposal"
    ]
    fs4 = [
        value for value in joint
        if value["domain"] == "FactorShockStatePolicyRZDT1"
        and abs(value["shock_scale"] - 4.0) <= 1e-12
    ]
    sound_certificate_count = int(sum(
        value["posterior_certified_count"] for value in joint))
    false_certificate_count = int(sum(
        value["false_certificate_count"] for value in joint))
    adaptive_loss_count = int(sum(
        value["adaptive_loss_count"] for value in joint))
    fs4_support = bool(fs4 and (
        fs4[0]["median_phi_true_feasible_iteration_rate"] or 0.0) > 0.0)
    mean_pair = paired.get(
        "phi_mean_only_vs_latent_control", {})
    promotion = {
        "all_cells_complete": all_complete,
        "factor_shock_scale4_phi_support": fs4_support,
        "nonzero_sound_certificate": bool(sound_certificate_count > 0),
        "zero_false_certificates": bool(false_certificate_count == 0),
        "zero_joint_adaptive_losses": bool(adaptive_loss_count == 0),
        "mean_rank_nonworse_across_scenarios": bool(
            mean_pair.get("mean_rank_pairs", 0) == len(EXPECTED_SCENARIOS)
            and mean_pair.get("mean_rank", 0)
            >= mean_pair.get("mean_rank_losses", 0)),
    }
    promotion["advance"] = bool(all(promotion.values()))
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            expected_cell_count * int(expected_seeds)),
        "expected_cell_count": int(expected_cell_count),
        "cells": list(cells.values()),
        "paired": paired,
        "promotion_gate": promotion,
        "scientific_contract": {
            "phi_representation_target": "source_chance_margin_bins",
            "phi_coefficient_prior_target": "source_constraint_mean",
            "target_adaptation": "budgeted_constraint_observations_only",
            "variance_coordinate": "psi=(A,N)",
            "target_oracle_used_for_decision": False,
        },
    }


def markdown(summary):
    lines = [
        "# Boundary Coordinate Gate",
        "",
        f"Rows: `{summary['row_count']}`",
        "",
        "| variant | domain | shock | n | feasible | cert | false | loss | margin | mean rank | chance rank | mean MAE | var log err | phi feasible iterations |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cells"]:
        fmt = lambda value: "NA" if value is None else f"{value:.4g}"
        lines.append(
            f"| {row['variant']} | {row['domain']} | {row['shock_scale']:g} "
            f"| {row['n']} | {row['true_feasible_count']} "
            f"| {row['posterior_certified_count']} "
            f"| {row['false_certificate_count']} "
            f"| {row['adaptive_loss_count']} "
            f"| {fmt(row['median_final_margin'])} "
            f"| {fmt(row['median_mean_rank_correlation'])} "
            f"| {fmt(row['median_chance_rank_correlation'])} "
            f"| {fmt(row['median_mean_abs_error'])} "
            f"| {fmt(row['median_variance_log_error'])} "
            f"| {fmt(row['median_phi_true_feasible_iteration_rate'])} |"
        )
    lines.extend([
        "",
        "## Promotion Gate",
        "",
        "```json",
        json.dumps(summary["promotion_gate"], indent=2, sort_keys=True),
        "```",
        "",
        "The oracle decomposition is post-run only and never changes a query or recommendation.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(load_rows(args.root), args.expected_seeds)
    out = args.out or (args.root / "boundary_coordinate_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(markdown(summary))
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
