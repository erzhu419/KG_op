#!/usr/bin/env python3
"""Analyze the V22 source-geometry assignment-prior gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_mean_alignment_v8_gate import _scenario
    from .analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from .analyze_mean_alignment_v17_gate import _variance_head_exact
    from .analyze_mean_alignment_v20_gate import (
        MEAN_SCENARIOS,
        _find_role_assignment,
    )
    from .analyze_mean_alignment_v21_gate import _cell as _v21_cell
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import _scenario
    from analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v17_gate import _variance_head_exact
    from analyze_mean_alignment_v20_gate import (
        MEAN_SCENARIOS,
        _find_role_assignment,
    )
    from analyze_mean_alignment_v21_gate import _cell as _v21_cell


VARIANTS = (
    "v15_tanh_control",
    "v21_uniform_loo_t20",
    "geometry_marginal_s025",
    "geometry_loo_s025_t20",
    "geometry_loo_s100_t20",
)
CHALLENGERS = VARIANTS[2:]


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


def _initial_assignment_audit(row):
    return dict((((row.get("task_initial_design") or {}).get(
        "truth_audit") or {}).get(
            "initial_design_role_assignment_oracle_expressivity") or {}))


def _pool_assignment_audit(row):
    return dict((row.get("boundary_raw_pool_truth_diagnostics") or {}).get(
        "boundary_raw_pool_role_assignment_oracle_expressivity") or {})


def _cell(rows, variant, scenario, expected_seeds):
    value = _v21_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    role_diagnostics = [
        _find_role_assignment(dict(row.get("meta_basis") or {}))
        for row in group
    ]
    role_diagnostics = [item for item in role_diagnostics if item is not None]
    geometry = [
        dict(item.get("assignment_prior_diagnostics") or {})
        for item in role_diagnostics
    ]
    selected = [str(prior.get(
        "target_role_assignment_selected", "missing")) for prior in priors]
    hard = [str(item.get("hard_assignment", "missing")) for item in geometry]
    initial_audits = [_initial_assignment_audit(row) for row in group]
    pool_audits = [_pool_assignment_audit(row) for row in group]
    value.update({
        "assignment_prior_modes": sorted(set(str(
            item.get("mode", "missing")) for item in geometry)),
        "assignment_prior_oracle_free": bool(geometry) and all(
            not item.get("target_labels_used", True)
            and not item.get("target_oracle_used", True)
            for item in geometry),
        "assignment_prior_permutation_equivariant": bool(geometry) and all(
            item.get("permutation_equivariant", False) for item in geometry),
        "assignment_prior_maximum_matches_hard": bool(geometry) and all(
            item.get("maximum_prior_matches_hard_assignment", False)
            for item in geometry),
        "assignment_prior_effective_counts": [float(
            item["effective_assignment_count"]) for item in geometry
            if item.get("effective_assignment_count") is not None],
        "posterior_selected_matches_geometry_hard": int(sum(
            choice == prior_hard
            for choice, prior_hard in zip(selected, hard))),
        "posterior_selected_matches_initial_mae": int(sum(
            choice == str(audit.get("best_mae_assignment", "missing"))
            for choice, audit in zip(selected, initial_audits))),
        "posterior_selected_matches_initial_rank": int(sum(
            choice == str(audit.get("best_rank_assignment", "missing"))
            for choice, audit in zip(selected, initial_audits))),
        "posterior_selected_matches_pool_mae": int(sum(
            choice == str(audit.get("best_mae_assignment", "missing"))
            for choice, audit in zip(selected, pool_audits))),
        "posterior_selected_matches_pool_rank": int(sum(
            choice == str(audit.get("best_rank_assignment", "missing"))
            for choice, audit in zip(selected, pool_audits))),
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
            "selected_matches_geometry_hard": int(sum(
                item["posterior_selected_matches_geometry_hard"]
                for item in values)),
            "selected_matches_pool_mae": int(sum(
                item["posterior_selected_matches_pool_mae"]
                for item in values)),
            "selected_matches_pool_rank": int(sum(
                item["posterior_selected_matches_pool_rank"]
                for item in values)),
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
    uniform_totals = totals["v21_uniform_loo_t20"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        geometry_contract = all(
            item["assignment_prior_modes"] == ["source_geometry"]
            and item["assignment_prior_oracle_free"]
            and item["assignment_prior_permutation_equivariant"]
            and item["assignment_prior_maximum_matches_hard"]
            and item["assignment_prior_effective_counts"]
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
        selected_pool = (
            totals[variant]["selected_matches_pool_mae"]
            + totals[variant]["selected_matches_pool_rank"]
        )
        uniform_pool = (
            uniform_totals["selected_matches_pool_mae"]
            + uniform_totals["selected_matches_pool_rank"]
        )
        score_contract = all(
            (
                item["structure_score_modes"] == ["loo_predictive"]
                and item["structure_score_cross_fitted"] == [True]
                and item["structure_score_oracle_free"]
                and item["loo_counts_match_target_history"]
            ) if "_loo_" in variant else (
                item["structure_score_modes"] == ["marginal_likelihood"]
                and item["structure_score_cross_fitted"] == [False]
            )
            for item in values
        )
        checks[variant] = {
            **global_checks,
            "source_geometry_prior_contract": bool(geometry_contract),
            "registered_structure_score_contract": bool(score_contract),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_structure": bool(
                _variance_head_exact(rows, variant)),
            "strict_assignment_selection_gain_over_uniform": bool(
                selected_pool > uniform_pool),
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
    output = args.out or args.root / "mean_alignment_v22_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
