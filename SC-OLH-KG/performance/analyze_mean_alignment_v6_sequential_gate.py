#!/usr/bin/env python3
"""Analyze the V6 online hierarchical misspecification gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "v4_latent_control",
    "misspec_scalar_static",
    "misspec_hierarchical",
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
            if variant is not None:
                row["gate_variant"] = variant
                rows.append(row)
    return rows


def _scenario(row):
    return (
        str(row.get("heldout")),
        float(row.get("target_shared_shock_scale", 1.0)),
    )


def _constraint_mean_diagnostics(row):
    diagnostics = list(row.get("gpr_numerics") or [])
    if len(diagnostics) < 2:
        return {}
    return dict(diagnostics[1].get("source_parametric_prior") or {})


def _source_scales(diagnostics):
    values = {}
    for item in diagnostics.get("component_deviation_diagnostics", []):
        name = str(item.get("name", ""))
        scale = _number(item.get("source_mean_misspecification_scale"))
        if name and scale is not None:
            values[name] = scale
    return values


def _cell(rows, variant, scenario, expected_seeds):
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    seeds = {int(row["seed"]) for row in group if row.get("seed") is not None}
    raw = [row.get("boundary_raw_pool_truth_diagnostics") or {}
           for row in group]
    mean_diagnostics = [_constraint_mean_diagnostics(row) for row in group]
    feasible_regrets = [
        row.get("feasible_simple_regret") for row in group
        if bool(row.get("true_feasible", False))
    ]
    return {
        "variant": variant,
        "domain": scenario[0],
        "shock_scale": scenario[1],
        "n": int(len(group)),
        "complete": bool(
            len(group) == int(expected_seeds)
            and len(seeds) == int(expected_seeds)),
        "oracle_free": bool(group and all(bool(
            (row.get("audit") or {}).get(
                "admissible_oracle_free_transfer", False))
            for row in group)),
        "true_feasible_count": int(sum(bool(
            row.get("true_feasible", False)) for row in group)),
        "median_feasible_regret": _median(feasible_regrets),
        "adaptive_loss_count": int(sum(bool(
            row.get("adaptive_loss", False)) for row in group)),
        "adaptive_rescue_count": int(sum(bool(
            row.get("adaptive_rescue", False)) for row in group)),
        "posterior_vacuous_count": int(sum(bool(
            row.get("posterior_certificate_vacuous", False))
            for row in group)),
        "evaluated_false_certificate_count": int(sum(
            int(row.get("false_certificate_count", 0) or 0)
            for row in group)),
        "audit_pool_false_certificate_count": int(sum(
            int(value.get("boundary_raw_pool_false_certified_count", 0) or 0)
            for value in raw)),
        "median_mean_abs_error": _median([
            value.get("boundary_raw_pool_constraint_mean_median_abs_error")
            for value in raw]),
        "median_variance_log_rmse": _median([
            row.get("variance_log_rmse") for row in group]),
        "online_update_counts": sorted(set(int(value.get(
            "online_mixture_update_count", -1)) for value in mean_diagnostics)),
        "target_observation_counts": sorted(set(int(value.get(
            "target_observation_count", -1)) for value in mean_diagnostics)),
        "scale_trajectory_lengths": sorted(set(len(value.get(
            "source_mean_misspecification_scale_trajectory", []))
            for value in mean_diagnostics)),
        "hierarchical_online": bool(group and all(bool(value.get(
            "source_mean_misspecification_online", False))
            for value in mean_diagnostics)),
        "target_null_scales": sorted(set(
            _source_scales(value).get("target:null")
            for value in mean_diagnostics
            if _source_scales(value).get("target:null") is not None
        )),
        "source_scale_changed_count": int(sum(
            len({
                round(float(scale), 12)
                for event in value.get(
                    "source_mean_misspecification_scale_trajectory", [])
                for name, scale in event.get("component_scales", {}).items()
                if not str(name).startswith("target:")
            }) > 1
            for value in mean_diagnostics
        )),
        "initial_design_fingerprints": sorted(set(str(
            (row.get("task_initial_design") or {}).get(
                "fingerprint", "missing"))
            for row in group)),
        "source_archive_fingerprints": sorted(set(str(
            (row.get("source_target_adaptation_contract") or {}).get(
                "source_archive_fingerprint", "missing"))
            for row in group)),
    }


def _count_nonworse(challenger, reference, key, *, multiplier=1.0, offset=0.0):
    count = 0
    for left, right in zip(challenger, reference):
        if left[key] is not None and right[key] is not None:
            count += int(left[key] <= multiplier * right[key] + offset)
    return count


def summarize(rows, expected_seeds=5, *, N=20, n0=10):
    cells = {
        (variant, *scenario): _cell(rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in SCENARIOS]
        for variant in VARIANTS
    }
    control = grouped["v4_latent_control"]
    static = grouped["misspec_scalar_static"]
    dynamic = grouped["misspec_hierarchical"]
    all_cells = list(cells.values())
    paired = all(
        dynamic_cell["initial_design_fingerprints"]
        == static_cell["initial_design_fingerprints"]
        == control_cell["initial_design_fingerprints"]
        and dynamic_cell["source_archive_fingerprints"]
        == static_cell["source_archive_fingerprints"]
        == control_cell["source_archive_fingerprints"]
        for dynamic_cell, static_cell, control_cell in zip(
            dynamic, static, control)
    )
    totals = {}
    for variant, values in grouped.items():
        totals[variant] = {
            "true_feasible": int(sum(
                value["true_feasible_count"] for value in values)),
            "adaptive_loss": int(sum(
                value["adaptive_loss_count"] for value in values)),
            "evaluated_false_certificates": int(sum(
                value["evaluated_false_certificate_count"] for value in values)),
            "audit_pool_false_certificates": int(sum(
                value["audit_pool_false_certificate_count"] for value in values)),
            "posterior_vacuous": int(sum(
                value["posterior_vacuous_count"] for value in values)),
        }
    dynamic_total = totals["misspec_hierarchical"]
    static_total = totals["misspec_scalar_static"]
    control_total = totals["v4_latent_control"]
    expected_online = int(N) - int(n0)
    checks = {
        "all_cells_complete": all(value["complete"] for value in all_cells),
        "oracle_free": all(value["oracle_free"] for value in all_cells),
        "paired_initial_design_and_archive": bool(paired),
        "hierarchical_state_updated_every_online_call": all(
            value["online_update_counts"] == [expected_online]
            and value["target_observation_counts"] == [int(N)]
            and value["scale_trajectory_lengths"] == [expected_online + 1]
            and value["hierarchical_online"]
            for value in dynamic
        ),
        "target_null_never_inflated": all(
            value["target_null_scales"] == [1.0] for value in dynamic),
        "source_scale_responds_to_online_data": sum(
            value["source_scale_changed_count"] for value in dynamic) > 0,
        "true_feasible_nonworse_than_both": bool(
            dynamic_total["true_feasible"]
            >= max(static_total["true_feasible"], control_total["true_feasible"])),
        "adaptive_loss_nonworse_than_both": bool(
            dynamic_total["adaptive_loss"]
            <= min(static_total["adaptive_loss"], control_total["adaptive_loss"])),
        "evaluated_false_certification_nonworse_than_both": bool(
            dynamic_total["evaluated_false_certificates"] <= min(
                static_total["evaluated_false_certificates"],
                control_total["evaluated_false_certificates"],
            )),
        "audit_pool_false_certification_lower_than_static": bool(
            dynamic_total["audit_pool_false_certificates"]
            < static_total["audit_pool_false_certificates"]),
        "audit_pool_false_certification_nonworse_than_control": bool(
            dynamic_total["audit_pool_false_certificates"]
            <= control_total["audit_pool_false_certificates"]),
        "mean_mae_nonworse_at_least_3_of_4": bool(
            _count_nonworse(
                dynamic, control, "median_mean_abs_error",
                multiplier=1.10, offset=0.01,
            ) >= 3),
        "variance_rmse_nonworse_at_least_3_of_4": bool(
            _count_nonworse(
                dynamic, control, "median_variance_log_rmse",
                multiplier=1.10,
            ) >= 3),
        "feasible_regret_nonworse_at_least_3_of_4": bool(
            _count_nonworse(
                dynamic, control, "median_feasible_regret") >= 3),
    }
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "totals": totals,
        "checks": checks,
        "promote_misspec_hierarchical": bool(all(checks.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(
        load_rows(args.root), args.expected_seeds, N=args.N, n0=args.n0)
    output = args.out or args.root / "mean_alignment_v6_sequential_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "promote_misspec_hierarchical": result[
            "promote_misspec_hierarchical"],
        "totals": result["totals"],
        "checks": result["checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
