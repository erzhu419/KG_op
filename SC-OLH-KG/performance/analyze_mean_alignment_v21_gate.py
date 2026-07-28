#!/usr/bin/env python3
"""Analyze the V21 cross-fitted role-assignment posterior gate."""

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
        _cell as _v20_cell,
    )
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
        _cell as _v20_cell,
    )


VARIANTS = (
    "v15_tanh_control",
    "v20_assignment_marginal",
    "assignment_loo_t05",
    "assignment_loo_t10",
    "assignment_loo_t20",
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


def _cell(rows, variant, scenario, expected_seeds):
    value = _v20_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    component_diagnostics = [
        list(prior.get("component_loo_predictive_diagnostics") or [])
        for prior in priors
    ]
    value.update({
        "structure_score_modes": sorted(set(str(
            prior.get("structure_score_mode", "missing"))
            for prior in priors)),
        "structure_score_cross_fitted": sorted(set(bool(
            prior.get("structure_score_cross_fitted", False))
            for prior in priors)),
        "structure_score_oracle_free": bool(priors) and all(
            not prior.get("target_oracle_used_for_structure_score", False)
            and not prior.get("target_oracle_used", False)
            for prior in priors),
        "loo_component_diagnostics_present": bool(priors) and all(
            diagnostics for diagnostics in component_diagnostics),
        "loo_counts_match_target_history": bool(priors) and all(
            diagnostics
            and all(
                int(item.get("loo_count", -1))
                == int(prior.get("target_observation_count", -2))
                for item in diagnostics
            )
            for prior, diagnostics in zip(priors, component_diagnostics)),
        "loo_scores_finite": bool(priors) and all(
            diagnostics
            and all(
                abs(float(item.get("loo_mean_log_score", float("inf"))))
                < float("inf")
                for item in diagnostics
            )
            for diagnostics in component_diagnostics),
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
            "selected_matches_oracle_mae": int(sum(
                item["role_assignment_selected_matches_oracle_mae"]
                for item in values)),
            "selected_matches_oracle_rank": int(sum(
                item["role_assignment_selected_matches_oracle_rank"]
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
    marginal_totals = totals["v20_assignment_marginal"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        assignment_contract = all(
            item["role_assignment_statuses"] == ["fit"]
            and item["role_assignment_counts"]
            and min(item["role_assignment_counts"]) > 1
            and item["role_assignment_outcome_free_hypotheses"]
            and item["role_assignment_component_coverage"]
            and item["role_assignment_posterior_active"]
            and item["role_assignment_target_evidence_used"]
            and item["role_assignment_oracle_free_decision"]
            and item["role_assignment_source_null_orbit_matched"]
            and item["role_assignment_oracle_audit_contract"]
            for item in values
        )
        loo_contract = all(
            item["structure_score_modes"] == ["loo_predictive"]
            and item["structure_score_cross_fitted"] == [True]
            and item["structure_score_oracle_free"]
            and item["loo_component_diagnostics_present"]
            and item["loo_counts_match_target_history"]
            and item["loo_scores_finite"]
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
        assignment_match_count = (
            totals[variant]["selected_matches_oracle_mae"]
            + totals[variant]["selected_matches_oracle_rank"]
        )
        marginal_match_count = (
            marginal_totals["selected_matches_oracle_mae"]
            + marginal_totals["selected_matches_oracle_rank"]
        )
        checks[variant] = {
            **global_checks,
            "finite_role_assignment_posterior_contract": bool(
                assignment_contract),
            "cross_fitted_predictive_structure_contract": bool(loo_contract),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_structure": bool(
                _variance_head_exact(rows, variant)),
            "strict_assignment_selection_gain_over_marginal": bool(
                assignment_match_count > marginal_match_count),
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
    output = args.out or args.root / "mean_alignment_v21_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
