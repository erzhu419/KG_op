#!/usr/bin/env python3
"""Analyze the V25 charged-pilot boundary-role posterior gate."""

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
    from .analyze_mean_alignment_v20_gate import (
        MEAN_SCENARIOS,
        _find_role_assignment,
    )
    from .analyze_mean_alignment_v24_gate import _cell as _v24_cell
    from .analyze_mean_alignment_v23_gate import _same_mass
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import _scenario
    from analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v20_gate import (
        MEAN_SCENARIOS,
        _find_role_assignment,
    )
    from analyze_mean_alignment_v24_gate import _cell as _v24_cell
    from analyze_mean_alignment_v23_gate import _same_mass


VARIANTS = (
    "v15_tanh_control",
    "v23_factorized_none",
    "boundary_geometry_s100",
    "boundary_geometry_s400",
    "boundary_geometry_s100_contrast",
    "boundary_geometry_s400_contrast",
)
CHALLENGERS = VARIANTS[2:]
CONTRAST_VARIANTS = set(VARIANTS[4:])


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


def _boundary_calibration(row):
    role = _find_role_assignment(dict(row.get("meta_basis") or {}))
    return dict((role or {}).get("boundary_calibration") or {})


def _source_contrast_contract(prior):
    source = [
        item for item in prior.get("component_deviation_diagnostics") or []
        if str(item.get("name", "")).startswith("source:")
    ]
    return bool(source) and all(
        item.get("source_mean_misspecification_applied", False)
        and item.get("source_contrast_assignment_conditional", False)
        and int(item.get("source_contrast_rank", -1))
        <= int(item.get("source_contrast_rank_bound", -2))
        and int(item.get("source_contrast_group_component_count", 0)) >= 1
        and not item.get("source_contrast_uses_target_data", True)
        and not item.get("target_oracle_used_for_misspecification", True)
        for item in source
    )


def _cell(rows, variant, scenario, expected_seeds):
    value = _v24_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    calibrations = [_boundary_calibration(row) for row in group]
    value.update({
        "pilot_boundary_calibration_contract": bool(calibrations) and all(
            calibration.get("status") == "fit"
            and calibration.get("mode") == "source_geometry_boundary"
            and 0 < int(calibration.get("target_observation_count", -1)) <= 10
            and calibration.get("target_labels_used", False)
            and not calibration.get("target_oracle_used", True)
            and calibration.get("permutation_equivariant", False)
            and float(calibration.get(
                "effective_assignment_count_after", 0.0)) > 0.0
            for calibration in calibrations
        ),
        "assignment_marginal_matches_boundary_posterior": bool(priors) and all(
            _same_mass(
                dict(prior.get("assignment_group_masses") or {}),
                dict(prior.get("target_role_assignment_posterior_mass") or {}),
            )
            for prior in priors
        ),
        "pilot_assignment_prior_then_frozen": bool(priors) and all(
            prior.get("target_labels_used_for_group_masses", False)
            and prior.get(
                "target_role_assignment_target_labels_used_for_prior", False)
            and not prior.get(
                "target_role_assignment_target_labels_used_for_online_update",
                True,
            )
            and prior.get("target_role_assignment_update_scope") == (
                "charged_pilot_assignment_prior_then_frozen_"
                "conditional_expert_only")
            and not prior.get("target_oracle_used_for_group_masses", True)
            for prior in priors
        ),
        "source_contrast_contract": bool(priors) and all(
            _source_contrast_contract(prior) for prior in priors),
        "boundary_assignment_changed_count": int(sum(
            calibration.get("maximum_prior_assignment")
            != calibration.get("maximum_posterior_assignment")
            for calibration in calibrations
        )),
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
            "selected_matches_pool_rank": int(sum(
                item["posterior_selected_matches_pool_rank"]
                for item in values)),
            "boundary_assignment_changed": int(sum(
                item["boundary_assignment_changed_count"] for item in values)),
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
        factorized_contract = all(
            item["adaptation_modes"] == [
                "sequential_assignment_prior_conditional_expert_mixture"]
            and item["assignment_group_masses_fixed"]
            and item["assignment_marginal_matches_boundary_posterior"]
            and item["conditional_expert_target_label_update_enabled"]
            and item["conditional_expert_updated"]
            and item["posterior_target_data_used"]
            and item["pilot_boundary_calibration_contract"]
            and item["pilot_assignment_prior_then_frozen"]
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
            "charged_pilot_factorized_role_contract": bool(
                factorized_contract),
            "assignment_conditional_source_contrast_contract": bool(
                all(item["source_contrast_contract"] for item in values)
                if variant in CONTRAST_VARIANTS else True),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "assignment_pool_rank_agreement_improves": bool(
                totals[variant]["selected_matches_pool_rank"]
                > no_scale_totals["selected_matches_pool_rank"]),
            "true_feasible_nonworse": bool(
                totals[variant]["true_feasible"]
                >= reference_totals["true_feasible"]),
            "true_certificate_support_nonworse": bool(
                totals[variant]["audit_pool_true_certificates"]
                >= reference_totals["audit_pool_true_certificates"]),
            "false_certification_nonworse": bool(
                totals[variant]["audit_pool_false_certificates"]
                <= reference_totals["audit_pool_false_certificates"]),
            "oracle_certifiability_nonworse_than_factorized": bool(
                totals[variant]["oracle_both_certificates"]
                >= no_scale_totals["oracle_both_certificates"]),
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
    output = args.out or args.root / "mean_alignment_v25_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
