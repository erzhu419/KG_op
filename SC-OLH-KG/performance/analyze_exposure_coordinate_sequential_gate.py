#!/usr/bin/env python3
"""Summarize the sequential observable-exposure coordinate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "latent_control",
    "exposure_mean_only",
    "exposure_mean_proposal",
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
            rows.append(row)
    return rows


def _scenario(row):
    return (
        str(row.get("heldout")),
        float(row.get("target_shared_shock_scale", 1.0)),
    )


def summarize(rows, expected_seeds=5):
    cells = {}
    for variant in VARIANTS:
        for scenario in EXPECTED_SCENARIOS:
            group = [row for row in rows if row["gate_variant"] == variant
                     and _scenario(row) == scenario]
            seeds = {int(row["seed"]) for row in group
                     if row.get("seed") is not None}
            raw = [row.get("boundary_raw_pool_truth_diagnostics") or {}
                   for row in group]
            cert = [row.get("certificate_outcome_audit") or {}
                    for row in group]
            adaptive = [row.get("adaptive_outcome_audit") or {}
                        for row in group]
            truth = [row.get("truth_pool_diagnostics") or {}
                     for row in group]
            cells[(variant, *scenario)] = {
                "variant": variant,
                "domain": scenario[0],
                "shock_scale": scenario[1],
                "n": int(len(group)),
                "complete": bool(
                    len(group) == expected_seeds
                    and len(seeds) == expected_seeds),
                "true_feasible_count": int(sum(
                    bool(row.get("true_feasible", False)) for row in group)),
                "median_feasible_regret": _median([
                    row.get("feasible_simple_regret") for row in group
                    if row.get("true_feasible", False)]),
                "certified_count": int(sum(
                    int(value.get("posterior_certified_count", 0))
                    for value in cert)),
                "false_certificate_count": int(sum(
                    int(value.get("false_certificate_count", 0))
                    for value in cert)),
                "adaptive_loss_count": int(sum(
                    bool(value.get("adaptive_loss", False))
                    for value in adaptive)),
                "mean_rank": _median([
                    value.get("mean_constraint_mean_rank_correlation")
                    for value in truth]),
                "phi_support_rate": _median([
                    value.get("phi_candidate_true_feasible_iteration_rate")
                    for value in truth]),
                "raw_pool_mean_rank": _median([
                    value.get(
                        "boundary_raw_pool_constraint_mean_rank_correlation")
                    for value in raw]),
                "raw_pool_oracle_both_count": _median([
                    value.get(
                        "boundary_raw_pool_oracle_mean_variance_certified_count")
                    for value in raw]),
            }

    joint = [cells[("exposure_mean_proposal", *scenario)]
             for scenario in EXPECTED_SCENARIOS]
    mean_only = [cells[("exposure_mean_only", *scenario)]
                 for scenario in EXPECTED_SCENARIOS]
    control = [cells[("latent_control", *scenario)]
               for scenario in EXPECTED_SCENARIOS]
    no_feasibility_losses = all(
        left["true_feasible_count"] >= right["true_feasible_count"]
        for left, right in zip(joint, control)
    )
    at_least_one_feasibility_gain = any(
        left["true_feasible_count"] > right["true_feasible_count"]
        for left, right in zip(joint, control)
    )
    mean_rank_nonworse = int(sum(
        left["raw_pool_mean_rank"] is not None
        and right["raw_pool_mean_rank"] is not None
        and left["raw_pool_mean_rank"] >= right["raw_pool_mean_rank"] - 0.02
        for left, right in zip(mean_only, control)
    ))
    fs4 = cells[(
        "exposure_mean_proposal",
        "FactorShockStatePolicyRZDT1",
        4.0,
    )]
    promotion = {
        "all_cells_complete": all(cell["complete"] for cell in cells.values()),
        "mean_rank_nonworse_in_at_least_3_of_4": mean_rank_nonworse >= 3,
        "joint_has_no_scenario_feasibility_loss": no_feasibility_losses,
        "joint_has_at_least_one_scenario_feasibility_gain": (
            at_least_one_feasibility_gain),
        "factor_shock_scale4_phi_support_nonzero": bool(
            (fs4["phi_support_rate"] or 0.0) > 0.0),
        "nonzero_sound_certificate": bool(sum(
            cell["certified_count"] for cell in joint) > 0),
        "zero_false_certificates": bool(sum(
            cell["false_certificate_count"] for cell in joint) == 0),
        "zero_joint_adaptive_losses": bool(sum(
            cell["adaptive_loss_count"] for cell in joint) == 0),
    }
    promotion["promote"] = bool(all(promotion.values()))
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(EXPECTED_SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "promotion_gate": promotion,
        "target_oracle_used_for_decision": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or (args.root / "exposure_coordinate_sequential_gate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["promotion_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
