#!/usr/bin/env python3
"""Analyze the V26 exchangeable target-linear mean-coordinate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .analyze_mean_alignment_v8_gate import _scenario
    from .analyze_mean_alignment_v16_gate import (
        _cell as _v16_cell,
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from .analyze_mean_alignment_v17_gate import _variance_head_exact
    from .analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from .analyze_mean_alignment_v24_gate import _source_scale_contract
except ImportError:  # Direct script execution.
    from analyze_mean_alignment_v8_gate import _scenario
    from analyze_mean_alignment_v16_gate import (
        _cell as _v16_cell,
        _constraint_prior,
        _count_higher_nonworse,
        _count_lower_nonworse,
    )
    from analyze_mean_alignment_v17_gate import _variance_head_exact
    from analyze_mean_alignment_v20_gate import MEAN_SCENARIOS
    from analyze_mean_alignment_v24_gate import _source_scale_contract


VARIANTS = (
    "v15_tanh_control",
    "v23_factorized_none",
    "v25_boundary_s400",
    "exchangeable_none",
    "exchangeable_hierarchical_df4",
    "exchangeable_hierarchical_df16",
)
CHALLENGERS = VARIANTS[3:]
HIERARCHICAL = set(VARIANTS[4:])


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


def _mean_basis(row):
    values = dict(row.get("meta_basis") or {})
    return dict(values.get("1") or values.get(1) or {})


def _constraint_numerics(row):
    values = list(row.get("gpr_numerics") or [])
    return dict(values[1]) if len(values) > 1 else {}


def _oracle_both_count(row):
    audit = dict(row.get("boundary_raw_pool_truth_diagnostics") or {})
    return int(audit.get(
        "boundary_raw_pool_oracle_mean_variance_certified_count", 0) or 0)


def _cell(rows, variant, scenario, expected_seeds):
    value = _v16_cell(rows, variant, scenario, expected_seeds)
    group = [
        row for row in rows
        if row["gate_variant"] == variant and _scenario(row) == scenario
    ]
    bases = [_mean_basis(row) for row in group]
    contracts = [dict(row.get("mean_risk_coordinate_contract") or {})
                 for row in group]
    numerics = [_constraint_numerics(row) for row in group]
    posterior = [dict(item.get("basis_posterior") or {})
                 for item in numerics]
    priors = [_constraint_prior(row) for row in group]
    source_components = [
        component
        for prior in priors
        for component in (prior.get("component_deviation_diagnostics") or [])
        if str(component.get("name", "")).startswith("source:")
    ]
    posterior_spreads = [
        float(item["posterior_channel_block_maximum_distance"])
        for item in posterior
        if item.get("posterior_channel_block_maximum_distance") is not None
    ]
    value.update({
        "descriptor_modes": sorted(set(str(
            row.get("meta_observable_mean_descriptor_mode", "missing"))
            for row in group)),
        "linear_feature_modes": sorted(set(str(
            row.get("meta_observable_mean_feature_mode", "missing"))
            for row in group)),
        "exchangeable_coordinate_contract": bool(group) and all(
            basis.get("coordinate")
            == "exchangeable_equivariant_boundary_linear"
            and basis.get("permutation_equivariant", False)
            and not basis.get("source_role_identity_transferred", True)
            and not basis.get("target_oracle_used", True)
            for basis in bases
        ),
        "exchangeable_joined_contract": bool(group) and all(
            contract.get("exchangeable_channel_role_posterior", False)
            and contract.get(
                "target_channel_roles_learned_from_charged_data", False)
            and contract.get("source_role_identity_transferred") is False
            and contract.get("separate_mean_variance_heads", False)
            and contract.get("shared_observable_exposure_input", False)
            and not contract.get("coordinate_definition_uses_target_labels", True)
            and not contract.get("source_oracle_aided", True)
            for contract in contracts
        ),
        "source_exchangeable_prior_contract": bool(posterior) and all(
            item.get("source_prior_exchangeable", False)
            and float(item.get(
                "source_channel_block_maximum_distance", np.inf)) <= 1e-10
            and not item.get("source_role_identity_transferred", True)
            and not item.get("target_oracle_used", True)
            for item in posterior
        ),
        "target_role_differentiated_count": int(sum(
            item.get("target_channel_roles_differentiated", False)
            for item in posterior)),
        "median_target_channel_block_spread": (
            None if not posterior_spreads
            else float(np.median(posterior_spreads))),
        "source_component_count": int(len(source_components)),
        "source_components_are_exchangeable_hyperpriors": bool(
            source_components) and all(
                item.get("component_kind")
                == "exchangeable_source_hyperprior"
                and item.get("permutation_equivariant", False)
                and not item.get("source_role_identity_transferred", True)
                and not item.get("target_oracle_used", True)
                for item in source_components
            ),
        "adaptation_modes": sorted(set(str(
            prior.get("adaptation_mode", "missing")) for prior in priors)),
        "posterior_target_data_used": bool(priors) and all(
            prior.get("posterior_target_data_used", False)
            and int(prior.get("target_observation_count", -1)) == 20
            and int(prior.get("online_mixture_update_count", -1)) == 10
            and not prior.get("target_oracle_used", True)
            for prior in priors),
        "hierarchical_scale_contract": bool(priors) and all(
            _source_scale_contract(prior) for prior in priors),
        "oracle_both_certified_total": int(sum(
            _oracle_both_count(row) for row in group)),
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
            "target_role_differentiated": int(sum(
                item["target_role_differentiated_count"] for item in values)),
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
    factorized_totals = totals["v23_factorized_none"]
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
        hierarchy = (
            all(
                item["adaptation_modes"]
                == ["sequential_target_evidence_mixture"]
                and item["hierarchical_scale_contract"]
                for item in values
            ) if variant in HIERARCHICAL else all(
                item["adaptation_modes"]
                == ["sequential_target_evidence_mixture"]
                for item in values
            )
        )
        checks[variant] = {
            **global_checks,
            "exchangeable_target_linear_contract": bool(all(
                item["descriptor_modes"] == ["exchangeable_equivariant"]
                and item["linear_feature_modes"] == ["linear"]
                and item["exchangeable_coordinate_contract"]
                and item["exchangeable_joined_contract"]
                and item["source_exchangeable_prior_contract"]
                and item["source_components_are_exchangeable_hyperpriors"]
                and item["posterior_target_data_used"]
                for item in values
            )),
            "source_mean_misspecification_contract": bool(hierarchy),
            "target_roles_differentiated_all_seeds": bool(
                totals[variant]["target_role_differentiated"]
                == len(MEAN_SCENARIOS) * expected_seeds),
            "independent_variance_task_posterior_contract": bool(
                independent_variance),
            "variance_head_exactly_invariant_to_mean_coordinate": bool(
                _variance_head_exact(rows, variant)),
            "true_feasible_matches_v15": bool(
                totals[variant]["true_feasible"]
                >= reference_totals["true_feasible"]),
            "oracle_certifiability_matches_factorized": bool(
                totals[variant]["oracle_both_certificates"]
                >= factorized_totals["oracle_both_certificates"]),
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
    output = args.out or args.root / "mean_alignment_v26_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sequential_gate_eligible": result["sequential_gate_eligible"],
        "totals": result["totals"],
        "variant_checks": result["variant_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
