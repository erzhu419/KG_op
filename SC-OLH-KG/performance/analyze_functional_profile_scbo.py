#!/usr/bin/env python3
"""Frozen attribution analysis for target-only functional-profile SCBO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from performance.analyze_profile_stress_suite import (
    _certified_deployed_success,
    _deployed_regret,
    _initial_contains_true_feasible,
    _initial_regret,
    _paired_deployment_comparison,
    _paired_initial_comparison,
)
from performance.benchmark_functional_profile_scbo import CONTRACT_ID
from performance.statistical_inference import (
    apply_holm_family,
    exact_binomial_interval,
    exact_binomial_upper_bound,
)


def _receipt(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(paths, contract_id):
    rows = []
    failures = []
    seen = set()
    for path in map(Path, paths):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append(f"{path}: unreadable: {error}")
            continue
        if row.get("contract_id") != contract_id or row.get("status") != "ok":
            continue
        key = (
            row["regime"], int(row["target_seed"]),
            int(row["nominal_dimension"]), row.get("arm"),
            row.get("matrix_configuration_id"),
        )
        if key in seen:
            failures.append(f"{path}: duplicate cell {key}")
            continue
        seen.add(key)
        row["_path"] = str(path)
        row["_sha256"] = _receipt(path)
        rows.append(row)
    return rows, failures


def _functional_group_key(row):
    contract = row["functional_coordinate_contract"]
    return (
        row["regime"],
        int(row["nominal_dimension"]),
        int(contract["coefficient_count"]),
        int(row["N"]),
        str(row["matrix_configuration_id"]),
    )


def _summary(group):
    task_count = len(group)
    initial_count = sum(_initial_contains_true_feasible(row) for row in group)
    search_count = sum(bool(row["search_contains_true_feasible"]) for row in group)
    certified_count = sum(_certified_deployed_success(row) for row in group)
    false_count = sum(bool(row["false_certificate"]) for row in group)
    initial_regrets = [
        float(value) for row in group
        if (value := _initial_regret(row)) is not None
    ]
    search_regrets = [
        float(row["search_finite_audit_library_regret"])
        for row in group
        if row["search_finite_audit_library_regret"] is not None
    ]
    deployed_regrets = [
        float(value) for row in group
        if (value := _deployed_regret(row)) is not None
    ]
    return {
        "independent_task_count": int(task_count),
        "initial_design_true_feasible_count": int(initial_count),
        "initial_design_true_feasible_rate": float(initial_count / task_count),
        "initial_design_true_feasible_exact_95ci": exact_binomial_interval(
            initial_count, task_count),
        "search_true_feasible_count": int(search_count),
        "search_true_feasible_rate": float(search_count / task_count),
        "search_true_feasible_exact_95ci": exact_binomial_interval(
            search_count, task_count),
        "certified_true_feasible_deployment_count": int(certified_count),
        "certified_true_feasible_deployment_rate": float(
            certified_count / task_count),
        "certified_true_feasible_deployment_exact_95ci": (
            exact_binomial_interval(certified_count, task_count)),
        "false_certificate_count": int(false_count),
        "false_certificate_rate": float(false_count / task_count),
        "false_certificate_one_sided_95_upper": exact_binomial_upper_bound(
            false_count, task_count),
        "median_initial_design_regret": (
            None if not initial_regrets else float(np.median(initial_regrets))),
        "median_search_regret": (
            None if not search_regrets else float(np.median(search_regrets))),
        "median_certified_deployed_regret": (
            None if not deployed_regrets
            else float(np.median(deployed_regrets))),
        "mean_verification_calls": float(np.mean([
            row["verification_calls"] for row in group])),
        "mean_all_in_calls": float(np.mean([
            row["all_in_calls_unamortized"] for row in group])),
        "median_wall_time_sec": float(np.median([
            row["wall_time_sec"] for row in group])),
    }


def _profile_counterparts(functional_row, profile_rows):
    if int(functional_row["N"]) >= 394:
        allowed_configurations = {"source384-target10"}
    else:
        allowed_configurations = {"primary", None}
    return [
        row for row in profile_rows
        if row.get("arm") == "source_atlas"
        and row["regime"] == functional_row["regime"]
        and int(row["nominal_dimension"])
            == int(functional_row["nominal_dimension"])
        and row.get("matrix_configuration_id") in allowed_configurations
        and row.get("schema_mode") == "declared"
        and row.get("descriptor_mode") == "domain_blind"
    ]


def analyze(functional_paths, profile_paths):
    functional_rows, failures = _load(functional_paths, CONTRACT_ID)
    profile_rows, profile_failures = _load(
        profile_paths, "randomized_ordered_profile_stress_v2")
    failures.extend(profile_failures)
    groups = {}
    for row in functional_rows:
        groups.setdefault(_functional_group_key(row), []).append(row)
    summaries = []
    initial_comparisons = []
    deployment_comparisons = []
    for group_index, (key, group) in enumerate(sorted(groups.items())):
        regime, dimension, coefficient_count, N, configuration = key
        summary = _summary(group)
        summary.update({
            "regime": regime,
            "nominal_dimension": dimension,
            "coefficient_count": coefficient_count,
            "N": N,
            "configuration_id": configuration,
        })
        summaries.append(summary)
        counterparts = _profile_counterparts(group[0], profile_rows)
        if not counterparts:
            continue
        initial = _paired_initial_comparison(
            group,
            counterparts,
            first_name="target_only_dct_space_scbo",
            second_name="source_atlas",
            pair_key=lambda row: int(row["target_seed"]),
            bootstrap_seed=20260808 + group_index,
        )
        initial.update({
            "regime": regime,
            "nominal_dimension": dimension,
            "coefficient_count": coefficient_count,
            "functional_target_search_budget": N,
            "source_atlas_target_search_budget": 10,
            "configuration_id": configuration,
            "inference_family_id": (
                "functional_initial_design_by_registered_regime"),
        })
        initial_comparisons.append(initial)
        deployment = _paired_deployment_comparison(
            group,
            counterparts,
            first_name="target_only_dct_space_scbo",
            second_name="source_atlas",
            pair_key=lambda row: int(row["target_seed"]),
            bootstrap_seed=20261808 + group_index,
        )
        deployment.update({
            "regime": regime,
            "nominal_dimension": dimension,
            "coefficient_count": coefficient_count,
            "functional_target_search_budget": N,
            "source_atlas_target_search_budget": 10,
            "configuration_id": configuration,
            "preverification_cost_relation": (
                "equal" if N == 394 else "functional_has_three_extra_target_calls"),
            "inference_family_id": (
                "functional_deployment_by_registered_regime"),
        })
        deployment_comparisons.append(deployment)
    apply_holm_family(
        initial_comparisons,
        pvalue_field="one_sided_first_better_exact_sign_pvalue",
        family_field="inference_family_id",
    )
    apply_holm_family(
        deployment_comparisons,
        pvalue_field="one_sided_first_better_exact_sign_pvalue",
        family_field="inference_family_id",
    )
    return {
        "schema_version": 1,
        "contract_id": "target_only_functional_profile_scbo_analysis_v1",
        "status": "complete" if not failures else "complete_with_failures",
        "analysis_scope": (
            "registered randomized profile generator only; regime-level task "
            "inference is separated from algorithmic seed repeatability"),
        "primary_endpoint": "certified_true_feasible_deployed_policy",
        "secondary_endpoints": [
            "initial_design_true_feasible_coverage",
            "search_true_feasible_coverage",
            "certified_deployed_finite_audit_library_regret",
            "false_certificate_rate",
            "all_in_calls",
        ],
        "functional_cell_count": int(len(functional_rows)),
        "profile_counterpart_cell_count": int(len(profile_rows)),
        "failures": failures,
        "summaries": summaries,
        "paired_initial_design_comparisons": initial_comparisons,
        "paired_deployment_comparisons": deployment_comparisons,
        "result_receipts": [
            {"path": row["_path"], "sha256": row["_sha256"]}
            for row in functional_rows
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional-root", required=True)
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    functional_paths = sorted(Path(args.functional_root).rglob("cell*.json"))
    profile_paths = sorted(Path(args.profile_root).rglob("cell*.json"))
    payload = analyze(functional_paths, profile_paths)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "status": payload["status"],
        "functional_cell_count": payload["functional_cell_count"],
        "failure_count": len(payload["failures"]),
        "out": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
