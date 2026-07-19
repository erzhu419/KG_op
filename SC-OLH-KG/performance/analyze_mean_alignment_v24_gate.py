#!/usr/bin/env python3
"""Analyze the V24 sequential hierarchical mean-calibration gate."""

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
    from .analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from .analyze_mean_alignment_v23_gate import _cell as _v23_cell
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import _scenario
    from analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v17_gate import _variance_head_exact
    from analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from analyze_mean_alignment_v23_gate import _cell as _v23_cell


VARIANTS = (
    "v15_tanh_control",
    "v23_factorized_none",
    "factorized_hierarchical_df4",
    "factorized_hierarchical_df16",
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


def _source_scale_contract(prior):
    components = list(prior.get("component_deviation_diagnostics") or [])
    if not components:
        return False
    source = [
        item for item in components
        if not str(item.get("name", "")).startswith("target:")
    ]
    null = [
        item for item in components
        if str(item.get("name", "")).startswith("target:")
    ]
    return bool(source and null) and all(
        item.get("source_mean_misspecification_applied", False)
        and float(item.get("source_mean_misspecification_scale", 0.0)) >= 1.0
        and item.get("misspecification_uncertainty_can_only_increase", False)
        and not item.get("target_oracle_used_for_misspecification", True)
        for item in source
    ) and all(
        not item.get("source_mean_misspecification_applied", True)
        and float(item.get("source_mean_misspecification_scale", 0.0)) == 1.0
        and not item.get("target_oracle_used_for_misspecification", True)
        for item in null
    )


def _cell(rows, variant, scenario, expected_seeds):
    value = _v23_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    raw = [
        dict(row.get("boundary_raw_pool_truth_diagnostics") or {})
        for row in group
    ]
    value.update({
        "hierarchical_scale_contract": bool(priors) and all(
            _source_scale_contract(prior) for prior in priors),
        "hierarchical_refit_from_frozen_law": bool(priors) and all(
            prior.get(
                "source_mean_misspecification_refit_from_frozen_law", False)
            for prior in priors),
        "sequential_target_counts": sorted(set(int(
            prior.get("target_observation_count", -1)) for prior in priors)),
        "sequential_update_counts": sorted(set(int(
            prior.get("online_mixture_update_count", -1)) for prior in priors)),
        "scale_trajectory_matches_updates": bool(priors) and all(
            len(prior.get(
                "source_mean_misspecification_scale_trajectory") or [])
            == int(prior.get("online_mixture_update_count", -1)) + 1
            for prior in priors),
        "oracle_both_certified_total": int(sum(int(
            item.get(
                "boundary_raw_pool_oracle_mean_variance_certified_count", 0)
            or 0) for item in raw)),
        "median_best_feasible_epistemic_radius": value.get(
            "median_best_feasible_epistemic_radius"),
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
                item["audit_pool_false_certificate_count"]
                for item in values)),
            "oracle_both_certificates": int(sum(
                item["oracle_both_certified_total"] for item in values)),
            "selected_matches_geometry_hard": int(sum(
                item["posterior_selected_matches_geometry_hard"]
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
    no_scale = grouped["v23_factorized_none"]
    no_scale_totals = totals["v23_factorized_none"]
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        hierarchy_contract = all(
            item["adaptation_modes"] == [
                "sequential_assignment_prior_conditional_hierarchical_"
                "expert_mixture"]
            and item["assignment_group_masses_fixed"]
            and item["assignment_marginal_matches_geometry_prior"]
            and item["assignment_target_label_update_disabled"]
            and item["conditional_expert_target_label_update_enabled"]
            and item["conditional_expert_updated"]
            and item["group_mass_oracle_free"]
            and item["posterior_target_data_used"]
            and item["hierarchical_scale_contract"]
            and item["hierarchical_refit_from_frozen_law"]
            and item["sequential_target_counts"] == [20]
            and item["sequential_update_counts"] == [10]
            and item["scale_trajectory_matches_updates"]
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
            "factorized_hierarchical_misspecification_contract": bool(
                hierarchy_contract),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_structure": bool(
                _variance_head_exact(rows, variant)),
            "geometry_hard_assignment_retained_all_seeds": bool(
                totals[variant]["selected_matches_geometry_hard"]
                == len(MEAN_SCENARIOS) * expected_seeds),
            "strict_oracle_certifiability_gain_over_no_scale": bool(
                totals[variant]["oracle_both_certificates"]
                > no_scale_totals["oracle_both_certificates"]),
            "true_feasible_nonworse": bool(
                totals[variant]["true_feasible"]
                >= reference_totals["true_feasible"]),
            "true_certificate_support_nonworse": bool(
                totals[variant]["audit_pool_true_certificates"]
                >= reference_totals["audit_pool_true_certificates"]),
            "false_certification_nonworse": bool(
                totals[variant]["audit_pool_false_certificates"]
                <= reference_totals["audit_pool_false_certificates"]),
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
            "epistemic_radius_nonworse_than_no_scale_at_least_2_of_3": bool(
                _count_lower_nonworse(
                    values, no_scale,
                    "median_best_feasible_epistemic_radius",
                    multiplier=1.05, offset=0.005) >= 2),
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
    output = args.out or args.root / "mean_alignment_v24_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
