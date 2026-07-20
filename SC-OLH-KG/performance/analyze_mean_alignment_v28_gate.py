#!/usr/bin/env python3
"""Analyze the V28 constraint-head authority separation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_mean_alignment_v16_gate import (
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from .analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from .analyze_mean_alignment_v27_gate import _cell as _v27_cell
    from .analyze_mean_alignment_v8_gate import _scenario
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v16_gate import (
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from analyze_mean_alignment_v27_gate import _cell as _v27_cell
    from analyze_mean_alignment_v8_gate import _scenario


VARIANTS = (
    "v27_task_joint",
    "v28_split_task_hvd",
    "v28_split_cumulative_hvd",
)
CHALLENGERS = VARIANTS[1:]
AUTHORITIES = {
    "v27_task_joint": "task_joint",
    "v28_split_task_hvd": "split_gpr_task_hvd",
    "v28_split_cumulative_hvd": "split_gpr_cumulative_hvd",
}
CERTIFICATE_SOURCES = {
    "v27_task_joint": "task_joint_kl_hvd",
    "v28_split_task_hvd": "split_aggregate_gpr_task_hvd",
    "v28_split_cumulative_hvd": "split_aggregate_gpr_cumulative_hvd",
}


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


def _cell(rows, variant, scenario, expected_seeds):
    value = _v27_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    decompositions = [
        dict(row.get("certification_margin_decomposition") or {})
        for row in group
    ]
    value.update({
        "certification_head_authorities": sorted(set(str(
            row.get("certification_head_authority", "missing"))
            for row in group)),
        "recommendation_certificate_sources": sorted(set(str(
            row.get("posterior_certification_source", "missing"))
            for row in group)),
        "decomposition_head_authorities": sorted(set(str(
            item.get("certification_head_authority", "missing"))
            for item in decompositions)),
        "adaptive_loss_count": int(sum(bool(
            row.get("adaptive_loss", False)) for row in group)),
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
            "audit_pool_true_certificates": int(sum(
                item["audit_pool_true_certificate_count"] for item in values)),
            "audit_pool_false_certificates": int(sum(
                item["audit_pool_false_certificate_count"] for item in values)),
            "oracle_both_certificates": int(sum(
                item["oracle_both_certified_total"] for item in values)),
            "adaptive_losses": int(sum(
                item["adaptive_loss_count"] for item in values)),
        }
        for variant, values in grouped.items()
    }
    global_checks = {
        "all_cells_complete": all(
            item["complete"] for item in cells.values()),
        "oracle_free": all(item["oracle_free"] for item in cells.values()),
        "paired_initial_design_and_archive": bool(paired),
    }
    control = grouped["v27_task_joint"]
    control_totals = totals["v27_task_joint"]
    checks = {}
    promotion_checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        authority = AUTHORITIES[variant]
        source = CERTIFICATE_SOURCES[variant]
        independent_variance = all(
            item["hvd_source_task_weight_modes"] == ["independent"]
            and item["variance_task_posterior_modes"] == ["replication_only"]
            and item["variance_task_posterior_statuses"]
            == ["frozen_source_prior"]
            and item["variance_task_evidence_counts"] == [0]
            and item["variance_task_effective_dofs"] == [0]
            and item["variance_task_uses_target_mean"] == [False]
            for item in values
        )
        checks[variant] = {
            **global_checks,
            "single_empirical_bayes_hyperlaw_contract": bool(all(
                item["single_hyperlaw_contract"]
                and item["posterior_target_data_used"]
                for item in values)),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "authority_is_explicit_and_consistent": bool(all(
                item["certification_head_authorities"] == [authority]
                and item["decomposition_head_authorities"] == [authority]
                and item["recommendation_certificate_sources"] == [source]
                for item in values)),
            "true_feasible_nonworse_than_v27": bool(
                totals[variant]["true_feasible"]
                >= control_totals["true_feasible"]),
            "false_certification_nonworse_than_v27": bool(
                totals[variant]["audit_pool_false_certificates"]
                <= control_totals["audit_pool_false_certificates"]),
            "adaptive_loss_nonworse_than_v27": bool(
                totals[variant]["adaptive_losses"]
                <= control_totals["adaptive_losses"]),
            "mean_rank_preserved_at_least_2_of_3": bool(
                _count_higher_nonworse(
                    values, control,
                    "median_constraint_mean_rank_correlation") >= 2),
            "mean_mae_preserved_at_least_2_of_3": bool(
                _count_lower_nonworse(
                    values, control, "median_mean_abs_error",
                    multiplier=1.02, offset=0.002) >= 2),
        }
        promotion_checks[variant] = {
            "actual_certificate_coverage_improves": bool(
                totals[variant]["audit_pool_true_certificates"]
                > control_totals["audit_pool_true_certificates"]),
            "oracle_substitution_certifiability_improves": bool(
                totals[variant]["oracle_both_certificates"]
                > control_totals["oracle_both_certificates"]),
        }
    authority_eligible = [
        variant for variant in CHALLENGERS
        if all(checks[variant].values())
    ]
    promotion_eligible = [
        variant for variant in authority_eligible
        if any(promotion_checks[variant].values())
    ]
    preference = {
        "v28_split_cumulative_hvd": 0,
        "v28_split_task_hvd": 1,
    }
    recommended = (
        min(
            promotion_eligible,
            key=lambda variant: (
                -totals[variant]["audit_pool_true_certificates"],
                -totals[variant]["oracle_both_certificates"],
                preference[variant],
            ),
        )
        if promotion_eligible else None
    )
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "expected_row_count": int(
            len(VARIANTS) * len(MEAN_SCENARIOS) * expected_seeds),
        "cells": list(cells.values()),
        "totals": totals,
        "global_checks": global_checks,
        "variant_checks": checks,
        "promotion_checks": promotion_checks,
        "authority_gate_eligible": authority_eligible,
        "promotion_eligible": promotion_eligible,
        "post_gate_baseline_recommendation": recommended,
        "post_gate_baseline_selection_rule": (
            "preserve the V27 mean posterior and safety; require a strict "
            "certificate or oracle-substitution certifiability gain; then "
            "prefer direct cumulative-HVD authority"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), args.expected_seeds)
    output = args.out or args.root / "mean_alignment_v28_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "authority_gate_eligible": result["authority_gate_eligible"],
        "promotion_eligible": result["promotion_eligible"],
        "post_gate_baseline_recommendation": result[
            "post_gate_baseline_recommendation"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
