#!/usr/bin/env python3
"""Analyze V8 support-adaptive role-coordinate calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


VARIANTS = (
    "v4_latent_control",
    "role_match_raw",
    "adaptive_role_ordered",
    "adaptive_role_set",
    "adaptive_role_ordered_contrast",
    "adaptive_role_set_contrast",
)
ADAPTIVE_VARIANTS = set(VARIANTS[2:])
CONTRAST_VARIANTS = {
    "adaptive_role_ordered_contrast",
    "adaptive_role_set_contrast",
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


def _variant(payload, row, variants=VARIANTS):
    label = str(row.get("experiment_variant") or payload.get(
        "experiment_variant", ""))
    for variant in variants:
        if f"/{variant}/" in f"/{label}/" or label.endswith(f"/{variant}"):
            return variant
    return None


def load_rows(root, variants=VARIANTS):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for raw in payload.get("rows", []):
            row = dict(raw)
            variant = _variant(payload, row, variants)
            if variant is not None:
                row["gate_variant"] = variant
                rows.append(row)
    return rows


def _scenario(row):
    return (
        str(row.get("heldout")),
        float(row.get("target_shared_shock_scale", 1.0)),
    )


def _prior(row):
    numerics = list(row.get("gpr_numerics") or [])
    if len(numerics) < 2:
        return {}
    return dict(numerics[1].get("source_parametric_prior") or {})


def _source_components(prior):
    return [
        dict(value)
        for value in prior.get("component_deviation_diagnostics", [])
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
    priors = [_prior(row) for row in group]
    selections = [dict(value.get("role_coordinate_selection") or {})
                  for value in priors]
    source = [item for value in priors for item in _source_components(value)]
    regrets = [row.get("feasible_simple_regret") for row in group
               if bool(row.get("true_feasible", False))]
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
        "median_feasible_regret": _median(regrets),
        "audit_pool_false_certificate_count": int(sum(int(
            value.get("boundary_raw_pool_false_certified_count", 0) or 0)
            for value in raw)),
        "median_mean_abs_error": _median([
            value.get("boundary_raw_pool_constraint_mean_median_abs_error")
            for value in raw]),
        "median_variance_log_rmse": _median([
            row.get("variance_log_rmse") for row in group]),
        "selected_coordinates": sorted(set(str(value.get(
            "selected_coordinate", "missing")) for value in selections)),
        "cardinality_support": sorted(set(bool(value.get(
            "channel_cardinality_supported", False))
            for value in selections)),
        "selection_outcome_free": bool(group and all(
            not value.get("target_labels_used", True)
            and not value.get("target_oracle_used", True)
            and not value.get("selection_uses_target_labels", True)
            and not value.get("selection_uses_target_oracle", True)
            for value in selections)),
        "selection_present": bool(group and all(bool(value)
                                                for value in selections)),
        "contrast_uncertainty_monotone": bool(source and all(
            float(value.get("source_mean_prior_covariance_trace_after", 0.0))
            + 1e-12 >= float(value.get(
                "source_mean_prior_covariance_trace_before", 0.0))
            and bool(value.get(
                "misspecification_uncertainty_can_only_increase", False))
            for value in source)),
        "contrast_rank_bounded": bool(source and all(
            0 <= int(value.get("source_contrast_rank", -1))
            <= int(value.get("source_contrast_rank_bound", -2))
            for value in source)),
        "contrast_source_only": bool(source and all(
            not value.get("source_contrast_uses_target_data", True)
            and not value.get(
                "target_oracle_used_for_misspecification", True)
            for value in source)),
        "initial_design_fingerprints": sorted(set(str(
            (row.get("task_initial_design") or {}).get(
                "fingerprint", "missing")) for row in group)),
        "source_archive_fingerprints": sorted(set(str(
            (row.get("source_target_adaptation_contract") or {}).get(
                "source_archive_fingerprint", "missing")) for row in group)),
    }


def _count_nonworse(challenger, reference, key, *, multiplier=1.0, offset=0.0):
    count = 0
    for left, right in zip(challenger, reference):
        if left[key] is not None and right[key] is not None:
            count += int(left[key] <= multiplier * right[key] + offset)
    return count


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in SCENARIOS]
        for variant in VARIANTS
    }
    control = grouped["v4_latent_control"]
    all_cells = list(cells.values())
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
                value["true_feasible_count"] for value in values)),
            "audit_pool_false_certificates": int(sum(
                value["audit_pool_false_certificate_count"]
                for value in values)),
        }
        for variant, values in grouped.items()
    }
    global_checks = {
        "all_cells_complete": all(value["complete"] for value in all_cells),
        "oracle_free": all(value["oracle_free"] for value in all_cells),
        "paired_initial_design_and_archive": bool(paired),
    }
    checks = {}
    for variant in VARIANTS[2:]:
        values = grouped[variant]
        contrast_required = variant in CONTRAST_VARIANTS
        factor_cells = values[:2]
        supported_cells = values[2:]
        variant_checks = {
            **global_checks,
            "true_feasible_nonworse_than_control": bool(
                totals[variant]["true_feasible"]
                >= totals["v4_latent_control"]["true_feasible"]),
            "false_certification_strictly_below_control": bool(
                totals[variant]["audit_pool_false_certificates"]
                < totals["v4_latent_control"][
                    "audit_pool_false_certificates"]),
            "mean_mae_nonworse_at_least_3_of_4": bool(_count_nonworse(
                values, control, "median_mean_abs_error",
                multiplier=1.10, offset=0.01) >= 3),
            "variance_rmse_nonworse_at_least_3_of_4": bool(_count_nonworse(
                values, control, "median_variance_log_rmse",
                multiplier=1.10) >= 3),
            "feasible_regret_nonworse_at_least_3_of_4": bool(_count_nonworse(
                values, control, "median_feasible_regret") >= 3),
            "selection_is_outcome_free": bool(all(
                value["selection_present"]
                and value["selection_outcome_free"] for value in values)),
            "unsupported_factor_uses_fallback": bool(all(
                value["cardinality_support"] == [False]
                and "role_aligned" not in value["selected_coordinates"]
                for value in factor_cells)),
            "supported_inventory_queue_use_roles": bool(all(
                value["cardinality_support"] == [True]
                and value["selected_coordinates"] == ["role_aligned"]
                for value in supported_cells)),
            "contrast_is_low_rank_psd_source_only": bool(
                all(value["contrast_uncertainty_monotone"]
                    and value["contrast_rank_bounded"]
                    and value["contrast_source_only"] for value in values)
                if contrast_required else True),
        }
        checks[variant] = variant_checks
    eligible = [
        variant for variant in VARIANTS[2:]
        if all(checks[variant].values())
    ]
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "totals": totals,
        "global_checks": global_checks,
        "variant_checks": checks,
        "sequential_gate_eligible": eligible,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or args.root / "mean_alignment_v8_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "global_checks": result["global_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
