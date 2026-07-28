#!/usr/bin/env python3
"""Analyze the V18 target-orthogonal residual mean-coordinate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_mean_alignment_v8_gate import _scenario
    from .analyze_mean_alignment_v16_gate import (
        _cell as _v16_cell,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from .analyze_mean_alignment_v17_gate import _variance_head_exact
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import _scenario
    from analyze_mean_alignment_v16_gate import (
        _cell as _v16_cell,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v17_gate import _variance_head_exact


VARIANTS = (
    "v15_tanh_control",
    "residual_rank1_quarter",
    "residual_rank1_unit",
    "residual_rank2_quarter",
    "residual_rank2_unit",
)
CHALLENGERS = VARIANTS[1:]
REQUESTED_RANK = {
    "residual_rank1_quarter": 1,
    "residual_rank1_unit": 1,
    "residual_rank2_quarter": 2,
    "residual_rank2_unit": 2,
}
MEAN_SCENARIOS = (
    ("FactorShockStatePolicyRZDT1", 0.0),
    ("InventorySupplyChain", 1.0),
    ("QueueResourceControl", 1.0),
)


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


def _find_target_residual(value):
    if isinstance(value, dict):
        candidate = value.get("target_orthogonal_residual")
        if isinstance(candidate, dict):
            return candidate
        for child in value.values():
            found = _find_target_residual(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_target_residual(child)
            if found is not None:
                return found
    return None


def _cell(rows, variant, scenario, expected_seeds):
    value = _v16_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    residuals = [
        _find_target_residual(dict(row.get("meta_basis") or {}))
        for row in group
    ]
    residuals = [item for item in residuals if item is not None]
    value.update({
        "target_residual_statuses": sorted(set(str(
            item.get("status", "missing")) for item in residuals)),
        "target_residual_requested_ranks": sorted(set(int(
            item.get("requested_rank", -1)) for item in residuals)),
        "target_residual_effective_ranks": sorted(set(int(
            item.get("effective_rank", -1)) for item in residuals)),
        "target_residual_maximum_cross_moment": max(
            (float(item.get("maximum_base_cross_moment", float("inf")))
             for item in residuals),
            default=None,
        ),
        "target_residual_outcome_free": bool(
            residuals and all(
                not item.get("target_labels_used", True)
                and not item.get("target_oracle_used", True)
                and not item.get(
                    "target_labels_used_to_define_coordinate", True)
                and not item.get(
                    "target_oracle_used_to_define_coordinate", True)
                for item in residuals)),
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
            "audit_pool_full_certificates": int(sum(
                item["audit_pool_full_certificate_count"] for item in values)),
            "audit_pool_true_certificates": int(sum(
                item["audit_pool_true_certificate_count"] for item in values)),
            "audit_pool_false_certificates": int(sum(
                item["audit_pool_false_certificate_count"] for item in values)),
        }
        for variant, values in grouped.items()
    }
    global_checks = {
        "all_cells_complete": all(
            item["complete"] for item in cells.values()),
        "oracle_free": all(item["oracle_free"] for item in cells.values()),
        "paired_initial_design_and_archive": bool(paired),
    }
    reference = grouped["v15_tanh_control"]
    reference_totals = totals["v15_tanh_control"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        rank = REQUESTED_RANK[variant]
        residual_contract = all(
            item["target_residual_statuses"] == ["fit"]
            and item["target_residual_requested_ranks"] == [rank]
            and item["target_residual_effective_ranks"] == [rank]
            and item["target_residual_maximum_cross_moment"] is not None
            and item["target_residual_maximum_cross_moment"] <= 1e-8
            and item["target_residual_outcome_free"]
            for item in values
        )
        independent_variance = all(
            item["variance_task_posterior_modes"] == ["replication_only"]
            and item["variance_task_posterior_statuses"]
            == ["frozen_source_prior"]
            and item["variance_task_evidence_counts"] == [0]
            and item["variance_task_effective_dofs"] == [0]
            and item["variance_task_uses_target_mean"] == [False]
            for item in values
        )
        checks[variant] = {
            **global_checks,
            "target_orthogonal_residual_contract": bool(residual_contract),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_coordinate": bool(
                _variance_head_exact(rows, variant)),
            "true_feasible_nonworse": bool(
                totals[variant]["true_feasible"]
                >= reference_totals["true_feasible"]),
            "true_certificate_support_nonworse": bool(
                totals[variant]["audit_pool_true_certificates"]
                >= reference_totals["audit_pool_true_certificates"]),
            "strict_false_certification_gain": bool(
                totals[variant]["audit_pool_false_certificates"]
                < reference_totals["audit_pool_false_certificates"]),
            "mean_mae_nonworse_at_least_2_of_3": bool(
                _count_lower_nonworse(
                    values, reference, "median_mean_abs_error",
                    multiplier=1.05, offset=0.005) >= 2),
            "mean_rank_nonworse_at_least_2_of_3": bool(
                _count_higher_nonworse(
                    values, reference,
                    "median_constraint_mean_rank_correlation") >= 2),
            "feasible_regret_nonworse_at_least_2_of_3": bool(
                _count_lower_nonworse(
                    values, reference, "median_feasible_regret") >= 2),
        }
    eligible = [
        variant for variant in CHALLENGERS
        if all(checks[variant].values())
    ]
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(MEAN_SCENARIOS) * expected_seeds),
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
    output = args.out or args.root / "mean_alignment_v18_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
