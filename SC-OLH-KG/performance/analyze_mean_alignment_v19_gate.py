#!/usr/bin/env python3
"""Analyze the V19 Bayesian residual-rank mean-coordinate gate."""

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
    from .analyze_mean_alignment_v18_gate import (
        MEAN_SCENARIOS,
        _cell as _v18_cell,
        _find_target_residual,
    )
except ImportError:
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
        _find_target_residual,
    )


VARIANTS = (
    "v15_tanh_control",
    "v18_rank2_fixed",
    "rank_mixture_complexity",
    "rank_mixture_mild",
    "rank_mixture_uniform",
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


def _rank_mass(names, weights):
    marker = "|target_residual_rank="
    values = {}
    for name, weight in zip(names, weights):
        name = str(name)
        if marker not in name:
            continue
        rank = int(name.rsplit(marker, 1)[1].split("|", 1)[0])
        values[rank] = values.get(rank, 0.0) + float(weight)
    return values


def _cell(rows, variant, scenario, expected_seeds):
    value = _v18_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    priors = [_constraint_prior(row) for row in group]
    residuals = [
        _find_target_residual(dict(row.get("meta_basis") or {}))
        for row in group
    ]
    residuals = [item for item in residuals if item is not None]
    posterior_changes = []
    for prior in priors:
        names = list(prior.get("component_names") or [])
        before = _rank_mass(
            names, list(prior.get("component_prior_weights") or []))
        after = {
            int(rank): float(mass)
            for rank, mass in dict(prior.get(
                "target_residual_rank_posterior_mass") or {}).items()
        }
        ranks = set(before) | set(after)
        if ranks:
            posterior_changes.append(float(sum(
                abs(after.get(rank, 0.0) - before.get(rank, 0.0))
                for rank in ranks)))
    value.update({
        "rank_posterior_active": bool(priors) and all(
            prior.get("target_residual_rank_posterior_active", False)
            for prior in priors),
        "rank_posterior_component_coverage": bool(priors) and all(
            prior.get("component_names")
            and all("|target_residual_rank=" in str(name)
                    for name in prior["component_names"])
            for prior in priors),
        "rank_posterior_structured_masses": sorted(set(round(float(
            prior.get("target_residual_rank_structured_mass", 0.0)), 12)
            for prior in priors)),
        "rank_posterior_selected_ranks": sorted(set(int(
            prior.get("target_residual_rank_selected", -1))
            for prior in priors)),
        "rank_posterior_target_evidence_used": bool(priors) and all(
            prior.get(
                "target_residual_rank_target_labels_used_for_update", False)
            and int(prior.get("target_observation_count", 0)) > 0
            for prior in priors),
        "rank_posterior_oracle_free": bool(priors) and all(
            not prior.get("target_residual_rank_target_oracle_used", True)
            for prior in priors),
        "rank_posterior_maximum_l1_update": (
            None if not posterior_changes else max(posterior_changes)),
        "rank2_outcome_free_coordinate": bool(
            residuals and all(
                item.get("status") == "fit"
                and int(item.get("effective_rank", -1)) == 2
                and not item.get("target_labels_used", True)
                and not item.get("target_oracle_used", True)
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
        structure_contract = all(
            item["rank2_outcome_free_coordinate"]
            and item["rank_posterior_active"]
            and item["rank_posterior_component_coverage"]
            and item["rank_posterior_structured_masses"] == [1.0]
            and item["rank_posterior_target_evidence_used"]
            and item["rank_posterior_oracle_free"]
            and item["rank_posterior_maximum_l1_update"] is not None
            and item["rank_posterior_maximum_l1_update"] > 1e-8
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
            "bayesian_residual_rank_contract": bool(structure_contract),
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
    output = args.out or args.root / "mean_alignment_v19_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
