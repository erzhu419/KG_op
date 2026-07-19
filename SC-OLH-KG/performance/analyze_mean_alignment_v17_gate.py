#!/usr/bin/env python3
"""Analyze the V17 intervention-response and mean-calibration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_mean_alignment_v8_gate import SCENARIOS, _scenario
    from .analyze_mean_alignment_v15_gate import _variance_head_signature
    from .analyze_mean_alignment_v16_gate import (
        _cell as _v16_cell,
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
        _numeric_equal,
    )
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import SCENARIOS, _scenario
    from analyze_mean_alignment_v15_gate import _variance_head_signature
    from analyze_mean_alignment_v16_gate import (
        _cell as _v16_cell,
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
        _numeric_equal,
    )


VARIANTS = (
    "v15_tanh_control",
    "ordered_hierarchical",
    "intervention_transport",
    "intervention_hierarchical",
)
CHALLENGERS = VARIANTS[1:]
INTERVENTION_VARIANTS = {
    "intervention_transport", "intervention_hierarchical"}
HIERARCHICAL_VARIANTS = {
    "ordered_hierarchical", "intervention_hierarchical"}


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


def _find_role_alignment(value):
    if isinstance(value, dict):
        if "signature_mode" in value and "source_domains" in value:
            return value
        for child in value.values():
            found = _find_role_alignment(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_role_alignment(child)
            if found is not None:
                return found
    return None


def _cell(rows, variant, scenario, expected_seeds):
    value = _v16_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    alignments = [
        _find_role_alignment(dict(row.get("meta_basis") or {}))
        for row in group
    ]
    alignments = [item for item in alignments if item is not None]
    target_matches = [
        match
        for alignment in alignments
        for match in (alignment.get("target_matches") or {}).values()
    ]
    trajectories = [
        list(prior.get("source_mean_misspecification_scale_trajectory") or [])
        for prior in priors
    ]
    components = [
        component
        for prior in priors
        for component in (prior.get("component_deviation_diagnostics") or [])
    ]
    source_components = [
        item for item in components
        if not str(item.get("name", "")).startswith("target:")
    ]
    null_components = [
        item for item in components
        if str(item.get("name", "")).startswith("target:")
    ]
    value.update({
        "role_signature_modes": sorted(set(str(
            item.get("signature_mode", "missing")) for item in alignments)),
        "barycentric_transport_flags": sorted(set(bool(
            item.get("barycentric_transport", False)) for item in alignments)),
        "source_signature_pools": sorted(set(str(
            item.get("source_signature_pool", "missing"))
            for item in alignments)),
        "target_transport_geometries": sorted(set(str(
            item.get("transport_geometry", "missing"))
            for item in target_matches)),
        "target_transport_outcome_free": bool(
            target_matches and all(
                not item.get("target_labels_used", True)
                and not item.get("target_oracle_used", True)
                for item in target_matches)),
        "hierarchical_modes": sorted(set(str(
            prior.get("source_mean_misspecification_mode", "none"))
            for prior in priors)),
        "hierarchical_online_flags": sorted(set(bool(
            prior.get("source_mean_misspecification_online", False))
            for prior in priors)),
        "hierarchical_refit_flags": sorted(set(bool(
            prior.get(
                "source_mean_misspecification_refit_from_frozen_law", False))
            for prior in priors)),
        "hierarchical_trajectory_lengths": sorted(set(
            len(items) for items in trajectories)),
        "source_misspecification_scales": sorted(set(float(
            item.get("source_mean_misspecification_scale", 0.0))
            for item in source_components)),
        "source_misspecification_monotone": bool(
            source_components and all(
                float(item.get("source_mean_misspecification_scale", 0.0))
                >= 1.0
                and item.get(
                    "misspecification_uncertainty_can_only_increase", False)
                and not item.get(
                    "target_oracle_used_for_misspecification", True)
                for item in source_components)),
        "target_null_uninflated": bool(
            null_components and all(
                float(item.get("source_mean_misspecification_scale", 1.0))
                == 1.0 for item in null_components)),
    })
    return value


def _row_key(row):
    return (*_scenario(row), int(row["seed"]))


def _variance_head_exact(rows, variant):
    reference = {
        _row_key(row): row for row in rows
        if row["gate_variant"] == "v15_tanh_control"
    }
    candidate = {
        _row_key(row): row for row in rows
        if row["gate_variant"] == variant
    }
    if not reference or set(reference) != set(candidate):
        return False
    metrics = (
        "variance_log_rmse",
        "certified_variance_log_rmse",
        "median_predicted_true_variance_ratio",
        "median_certified_true_variance_ratio",
        "variance_upper_coverage",
    )
    return all(
        all(_numeric_equal(reference[key], candidate[key], metric)
            for metric in metrics)
        and _variance_head_signature(reference[key])
        == _variance_head_signature(candidate[key])
        for key in reference
    )


def summarize(rows, expected_seeds=5):
    cells = {
        (variant, *scenario): _cell(
            rows, variant, scenario, expected_seeds)
        for variant in VARIANTS for scenario in SCENARIOS
    }
    grouped = {
        variant: [cells[(variant, *scenario)] for scenario in SCENARIOS]
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
        independent_variance = all(
            item["variance_task_posterior_modes"] == ["replication_only"]
            and item["variance_task_posterior_statuses"]
            == ["frozen_source_prior"]
            and item["variance_task_evidence_counts"] == [0]
            and item["variance_task_effective_dofs"] == [0]
            and item["variance_task_uses_target_mean"] == [False]
            for item in values
        )
        intervention_contract = (
            all(
                item["descriptor_modes"]
                == ["role_intervention_transport"]
                and item["role_signature_modes"]
                == ["intervention_response"]
                and item["barycentric_transport_flags"] == [True]
                and item["source_signature_pools"]
                == ["deterministic_unlabeled_intervention_pool"]
                and item["target_transport_geometries"]
                == ["barycentric_response"]
                and item["target_transport_outcome_free"]
                for item in values
            ) if variant in INTERVENTION_VARIANTS else True
        )
        hierarchical_contract = (
            all(
                item["hierarchical_modes"]
                == ["hierarchical_predictive_scale"]
                and item["hierarchical_online_flags"] == [True]
                and item["hierarchical_refit_flags"] == [True]
                and item["hierarchical_trajectory_lengths"] == [1]
                and min(item["source_misspecification_scales"]) >= 1.0
                and item["source_misspecification_monotone"]
                and item["target_null_uninflated"]
                for item in values
            ) if variant in HIERARCHICAL_VARIANTS else True
        )
        checks[variant] = {
            **global_checks,
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_coordinate": bool(
                _variance_head_exact(rows, variant)),
            "source_only_intervention_transport_contract": bool(
                intervention_contract),
            "hierarchical_mean_misspecification_contract": bool(
                hierarchical_contract),
            "true_feasible_nonworse": bool(
                totals[variant]["true_feasible"]
                >= reference_totals["true_feasible"]),
            "true_certificate_support_nonworse": bool(
                totals[variant]["audit_pool_true_certificates"]
                >= reference_totals["audit_pool_true_certificates"]),
            "strict_false_certification_gain": bool(
                totals[variant]["audit_pool_false_certificates"]
                < reference_totals["audit_pool_false_certificates"]),
            "factor_shock_scale4_false_certification_gain": bool(
                values[1]["audit_pool_false_certificate_count"]
                < reference[1]["audit_pool_false_certificate_count"]),
            "mean_mae_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, reference, "median_mean_abs_error",
                    multiplier=1.05, offset=0.005) >= 3),
            "mean_rank_nonworse_at_least_3_of_4": bool(
                _count_higher_nonworse(
                    values, reference,
                    "median_constraint_mean_rank_correlation") >= 3),
            "feasible_regret_nonworse_at_least_3_of_4": bool(
                _count_lower_nonworse(
                    values, reference, "median_feasible_regret") >= 3),
        }
    eligible = [
        variant for variant in CHALLENGERS
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
    output = args.out or args.root / "mean_alignment_v17_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
