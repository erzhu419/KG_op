#!/usr/bin/env python3
"""Analyze the V20 finite channel-role assignment posterior gate."""

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
    from .analyze_mean_alignment_v18_gate import (
        MEAN_SCENARIOS,
        _cell as _v18_cell,
    )
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import _scenario
    from analyze_mean_alignment_v16_gate import (
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v17_gate import _variance_head_exact
    from analyze_mean_alignment_v18_gate import (
        MEAN_SCENARIOS,
        _cell as _v18_cell,
    )


VARIANTS = (
    "v15_tanh_control",
    "v17_hard_intervention",
    "v18_rank2_fixed",
    "role_assignment_plain",
    "role_assignment_hierarchical",
)
CHALLENGERS = VARIANTS[3:]


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


def _find_role_assignment(value):
    if isinstance(value, dict):
        candidate = value.get("role_assignment_posterior")
        if isinstance(candidate, dict):
            return candidate
        for child in value.values():
            found = _find_role_assignment(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_role_assignment(child)
            if found is not None:
                return found
    return None


def _assignment_mass(names, weights):
    marker = "|role_assignment="
    mass = {}
    for name, weight in zip(names, weights):
        name = str(name)
        if marker not in name:
            continue
        assignment = name.rsplit(marker, 1)[1].split("|", 1)[0]
        mass[assignment] = mass.get(assignment, 0.0) + float(weight)
    return mass


def _entropy_and_effective_count(mass):
    values = np.asarray(list(mass.values()), dtype=float)
    values = np.maximum(values, 0.0)
    if float(np.sum(values)) <= 0.0:
        return None, None
    values /= float(np.sum(values))
    positive = values[values > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return entropy, float(np.exp(entropy))


def _cell(rows, variant, scenario, expected_seeds):
    value = _v18_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    assignments = [
        _find_role_assignment(dict(row.get("meta_basis") or {}))
        for row in group
    ]
    assignments = [item for item in assignments if item is not None]
    audits = [
        dict((row.get("boundary_raw_pool_truth_diagnostics") or {}).get(
            "boundary_raw_pool_role_assignment_oracle_expressivity") or {})
        for row in group
    ]
    audits = [item for item in audits if item]

    posterior_l1 = []
    posterior_entropies = []
    posterior_effective_counts = []
    source_masses = []
    null_masses = []
    orbit_contracts = []
    selected = []
    for prior in priors:
        names = list(prior.get("component_names") or [])
        before = _assignment_mass(
            names, list(prior.get("component_prior_weights") or []))
        after = {
            str(key): float(mass)
            for key, mass in dict(prior.get(
                "target_role_assignment_posterior_mass") or {}).items()
        }
        orbit = set(after)
        source_orbit = set(dict(prior.get(
            "target_role_assignment_conditional_source_mass") or {}))
        null_orbit = set(dict(prior.get(
            "target_role_assignment_conditional_null_mass") or {}))
        orbit_contracts.append(bool(orbit and source_orbit == orbit == null_orbit))
        if before or after:
            keys = set(before) | set(after)
            posterior_l1.append(float(sum(
                abs(after.get(key, 0.0) - before.get(key, 0.0))
                for key in keys)))
        entropy, effective = _entropy_and_effective_count(after)
        if entropy is not None:
            posterior_entropies.append(entropy)
            posterior_effective_counts.append(effective)
        source_masses.append(float(prior.get(
            "target_role_assignment_structured_source_mass", 0.0)))
        null_masses.append(float(prior.get(
            "target_role_assignment_structured_null_mass", 0.0)))
        selected.append(str(prior.get(
            "target_role_assignment_selected", "missing")))

    oracle_mae_matches = []
    oracle_rank_matches = []
    for choice, audit in zip(selected, audits):
        oracle_mae_matches.append(choice == str(
            audit.get("best_mae_assignment", "missing")))
        oracle_rank_matches.append(choice == str(
            audit.get("best_rank_assignment", "missing")))

    value.update({
        "role_assignment_statuses": sorted(set(str(
            item.get("status", "missing")) for item in assignments)),
        "role_assignment_counts": sorted(set(int(
            item.get("assignment_count", 0)) for item in assignments)),
        "role_assignment_active_dims": sorted(set(int(
            item.get("active_feature_dim_per_atom", 0))
            for item in assignments)),
        "role_assignment_total_dims": sorted(set(int(
            item.get("total_stored_feature_dim", 0))
            for item in assignments)),
        "role_assignment_outcome_free_hypotheses": bool(
            assignments and all(
                not item.get(
                    "target_labels_used_to_define_assignments", True)
                and not item.get(
                    "target_oracle_used_to_define_assignments", True)
                and item.get("permutation_equivariant", False)
                for item in assignments)),
        "role_assignment_component_coverage": bool(priors) and all(
            prior.get("component_names")
            and all("|role_assignment=" in str(name)
                    for name in prior["component_names"])
            for prior in priors),
        "role_assignment_posterior_active": bool(priors) and all(
            prior.get("target_role_assignment_posterior_active", False)
            for prior in priors),
        "role_assignment_target_evidence_used": bool(priors) and all(
            prior.get(
                "target_role_assignment_target_labels_used_for_update", False)
            and int(prior.get("target_observation_count", 0)) > 0
            for prior in priors),
        "role_assignment_oracle_free_decision": bool(priors) and all(
            not prior.get("target_role_assignment_target_oracle_used", True)
            for prior in priors),
        "role_assignment_source_null_orbit_matched": bool(
            orbit_contracts and all(orbit_contracts)),
        "role_assignment_maximum_l1_update": (
            None if not posterior_l1 else max(posterior_l1)),
        "role_assignment_median_entropy": (
            None if not posterior_entropies
            else float(np.median(posterior_entropies))),
        "role_assignment_median_effective_count": (
            None if not posterior_effective_counts
            else float(np.median(posterior_effective_counts))),
        "role_assignment_source_masses": source_masses,
        "role_assignment_null_masses": null_masses,
        "role_assignment_selected": sorted(set(selected)),
        "role_assignment_oracle_audit_contract": bool(audits) and all(
            item.get("status") == "audited"
            and item.get("post_run_only", False)
            and item.get("target_oracle_used", False)
            and not item.get("target_oracle_used_for_decision", True)
            and int(item.get("assignment_count", 0)) > 1
            for item in audits),
        "role_assignment_oracle_best_mae": [
            float(item["best_median_abs_error"]) for item in audits
            if item.get("best_median_abs_error") is not None],
        "role_assignment_oracle_best_rank": [
            float(item["best_rank_correlation"]) for item in audits
            if item.get("best_rank_correlation") is not None],
        "role_assignment_selected_matches_oracle_mae": int(sum(
            oracle_mae_matches)),
        "role_assignment_selected_matches_oracle_rank": int(sum(
            oracle_rank_matches)),
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
        assignment_contract = all(
            item["role_assignment_statuses"] == ["fit"]
            and item["role_assignment_counts"]
            and min(item["role_assignment_counts"]) > 1
            and item["role_assignment_active_dims"]
            and max(item["role_assignment_active_dims"]) <= 8
            and all(
                total == count * active
                for total, count, active in zip(
                    item["role_assignment_total_dims"],
                    item["role_assignment_counts"],
                    item["role_assignment_active_dims"],
                ))
            and item["role_assignment_outcome_free_hypotheses"]
            and item["role_assignment_component_coverage"]
            and item["role_assignment_posterior_active"]
            and item["role_assignment_target_evidence_used"]
            and item["role_assignment_oracle_free_decision"]
            and item["role_assignment_source_null_orbit_matched"]
            and item["role_assignment_maximum_l1_update"] is not None
            and item["role_assignment_maximum_l1_update"] > 1e-8
            and item["role_assignment_oracle_audit_contract"]
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
            "finite_role_assignment_posterior_contract": bool(
                assignment_contract),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_structure": bool(
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
    output = args.out or args.root / "mean_alignment_v20_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
