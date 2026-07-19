#!/usr/bin/env python3
"""Analyze the V23 factorized assignment/expert posterior gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
    from .analyze_mean_alignment_v22_gate import _cell as _v22_cell
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
    from analyze_mean_alignment_v22_gate import _cell as _v22_cell


VARIANTS = (
    "v15_tanh_control",
    "v22_joint_geometry_loo",
    "factorized_geometry_s005_t20",
    "factorized_geometry_s025_t20",
    "factorized_geometry_s100_t20",
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


def _role_prior_mass(row):
    role = _find_role_assignment(dict(row.get("meta_basis") or {}))
    if role is None:
        return {}
    labels = [str(value) for value in role.get("assignments") or []]
    weights = list(role.get("assignment_prior_weights") or [])
    if len(labels) != len(weights):
        return {}
    return {label: float(weight) for label, weight in zip(labels, weights)}


def _same_mass(left, right, atol=1e-10):
    if set(left) != set(right) or not left:
        return False
    return all(np.isclose(
        float(left[key]), float(right[key]), rtol=0.0, atol=atol)
        for key in left)


def _conditional_expert_updated(prior):
    names = [str(value) for value in prior.get("component_names") or []]
    before = np.asarray(
        prior.get("component_prior_weights") or [], dtype=float)
    after = np.asarray(
        prior.get("component_posterior_weights") or [], dtype=float)
    marker = "|role_assignment="
    if len(names) == 0 or len(before) != len(names) or len(after) != len(names):
        return False
    labels = [
        name.rsplit(marker, 1)[1].split("|", 1)[0]
        if marker in name else "missing"
        for name in names
    ]
    for label in sorted(set(labels)):
        indices = np.asarray([
            index for index, value in enumerate(labels) if value == label
        ], dtype=int)
        if len(indices) < 2:
            continue
        prior_total = float(np.sum(before[indices]))
        posterior_total = float(np.sum(after[indices]))
        if prior_total <= 0.0 or posterior_total <= 0.0:
            continue
        prior_conditional = before[indices] / prior_total
        posterior_conditional = after[indices] / posterior_total
        if not np.allclose(
            prior_conditional, posterior_conditional, rtol=0.0, atol=1e-8
        ):
            return True
    return False


def _cell(rows, variant, scenario, expected_seeds):
    value = _v22_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    geometry_masses = [_role_prior_mass(row) for row in group]
    fixed_masses = [dict(
        prior.get("assignment_group_masses") or {}) for prior in priors]
    posterior_masses = [dict(
        prior.get("target_role_assignment_posterior_mass") or {})
        for prior in priors]
    value.update({
        "adaptation_modes": sorted(set(str(
            prior.get("adaptation_mode", "missing")) for prior in priors)),
        "assignment_group_masses_fixed": bool(priors) and all(
            prior.get("assignment_group_masses_fixed", False)
            for prior in priors),
        "assignment_marginal_matches_geometry_prior": bool(priors) and all(
            _same_mass(geometry, fixed)
            and _same_mass(geometry, posterior)
            for geometry, fixed, posterior in zip(
                geometry_masses, fixed_masses, posterior_masses)
        ),
        "assignment_target_label_update_disabled": bool(priors) and all(
            not prior.get(
                "target_role_assignment_target_labels_used_for_update", True)
            for prior in priors),
        "conditional_expert_target_label_update_enabled": bool(priors) and all(
            prior.get(
                "target_role_assignment_conditional_expert_uses_target_labels",
                False,
            )
            for prior in priors),
        "conditional_expert_updated": bool(priors) and all(
            _conditional_expert_updated(prior) for prior in priors),
        "group_mass_oracle_free": bool(priors) and all(
            not prior.get("target_oracle_used_for_group_masses", True)
            and not prior.get("target_oracle_used", True)
            for prior in priors),
        "posterior_target_data_used": bool(priors) and all(
            prior.get("posterior_target_data_used", False)
            for prior in priors),
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
    checks = {}
    for variant in CHALLENGERS:
        values = grouped[variant]
        factorization_contract = all(
            item["adaptation_modes"] == [
                "sequential_assignment_prior_conditional_expert_mixture"]
            and item["assignment_group_masses_fixed"]
            and item["assignment_marginal_matches_geometry_prior"]
            and item["assignment_target_label_update_disabled"]
            and item["conditional_expert_target_label_update_enabled"]
            and item["conditional_expert_updated"]
            and item["group_mass_oracle_free"]
            and item["posterior_target_data_used"]
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
            "factorized_assignment_expert_posterior_contract": bool(
                factorization_contract),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_structure": bool(
                _variance_head_exact(rows, variant)),
            "geometry_hard_assignment_retained_all_seeds": bool(
                totals[variant]["selected_matches_geometry_hard"]
                == len(MEAN_SCENARIOS) * expected_seeds),
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
    output = args.out or args.root / "mean_alignment_v23_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
