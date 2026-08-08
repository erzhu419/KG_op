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


CELL_ERROR_CONTRACT_ID = "target_only_functional_profile_scbo_cell_error_v1"


def _receipt(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(paths, contract_id, *, algorithmic_failure_contracts=()):
    rows = []
    failures = []
    algorithmic_failures = []
    seen = set()
    accepted_failure_contracts = set(algorithmic_failure_contracts)
    for path in map(Path, paths):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append(f"{path}: unreadable: {error}")
            continue
        if row.get("contract_id") in accepted_failure_contracts:
            if row.get("status") != "error" or not isinstance(
                row.get("cell"), dict
            ):
                failures.append(f"{path}: malformed algorithmic failure")
                continue
            failure = dict(row)
            failure["_path"] = str(path)
            failure["_sha256"] = _receipt(path)
            algorithmic_failures.append(failure)
            continue
        if row.get("contract_id") != contract_id or row.get("status") != "ok":
            failures.append(
                f"{path}: unexpected contract/status "
                f"{row.get('contract_id')}/{row.get('status')}"
            )
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
    return rows, failures, algorithmic_failures


def _functional_group_key(row):
    contract = row["functional_coordinate_contract"]
    return (
        row["regime"],
        int(row["nominal_dimension"]),
        int(contract["coefficient_count"]),
        int(row["N"]),
        str(row["matrix_configuration_id"]),
    )


def _summary(group, algorithmic_failures=()):
    successful_task_count = len(group)
    algorithmic_failure_count = len(algorithmic_failures)
    task_count = successful_task_count + algorithmic_failure_count
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
        "successful_task_count": int(successful_task_count),
        "algorithmic_failure_count": int(algorithmic_failure_count),
        "algorithmic_failures_count_as_unsuccessful_primary_outcomes": True,
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
        "mean_verification_calls_successful_runs": float(np.mean([
            row["verification_calls"] for row in group])),
        "mean_verification_calls": float(np.mean([
            row["verification_calls"] for row in group])),
        "mean_all_in_calls_successful_runs": float(np.mean([
            row["all_in_calls_unamortized"] for row in group])),
        "mean_all_in_calls": float(np.mean([
            row["all_in_calls_unamortized"] for row in group])),
        "median_wall_time_sec_successful_runs": float(np.median([
            row["wall_time_sec"] for row in group])),
        "median_wall_time_sec": float(np.median([
            row["wall_time_sec"] for row in group])),
    }


def _failure_group_key(row):
    cell = row["cell"]
    return (
        str(cell["regime"]),
        int(cell["dimension"]),
        int(cell["coefficient_count"]),
        int(cell["N"]),
        str(cell["configuration_id"]),
    )


def _failure_deployment_row(row):
    """Represent an optimizer crash as an unsuccessful deployment outcome."""

    cell = row["cell"]
    return {
        "target_seed": int(cell["target_seed"]),
        "independently_certified": False,
        "false_certificate": False,
        "deployed_truth": None,
        "finite_audit_library_oracle_objective": 0.0,
        "_algorithmic_failure": True,
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
    functional_rows, failures, algorithmic_failures = _load(
        functional_paths,
        CONTRACT_ID,
        algorithmic_failure_contracts=(CELL_ERROR_CONTRACT_ID,),
    )
    profile_rows, profile_failures, profile_algorithmic_failures = _load(
        profile_paths, "randomized_ordered_profile_stress_v2")
    failures.extend(profile_failures)
    if profile_algorithmic_failures:
        failures.append("profile counterpart contains algorithmic failures")
    groups = {}
    for row in functional_rows:
        groups.setdefault(_functional_group_key(row), []).append(row)
    failure_groups = {}
    for row in algorithmic_failures:
        failure_groups.setdefault(_failure_group_key(row), []).append(row)
    all_group_keys = sorted(set(groups) | set(failure_groups))
    summaries = []
    initial_comparisons = []
    deployment_comparisons = []
    for group_index, key in enumerate(all_group_keys):
        group = groups.get(key, [])
        group_failures = failure_groups.get(key, [])
        regime, dimension, coefficient_count, N, configuration = key
        summary = _summary(group, group_failures)
        summary.update({
            "regime": regime,
            "nominal_dimension": dimension,
            "coefficient_count": coefficient_count,
            "N": N,
            "configuration_id": configuration,
        })
        summaries.append(summary)
        reference_row = group[0] if group else {
            "regime": regime,
            "nominal_dimension": dimension,
            "N": N,
        }
        counterparts = _profile_counterparts(reference_row, profile_rows)
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
        deployment_rows = group + [
            _failure_deployment_row(row) for row in group_failures
        ]
        deployment = _paired_deployment_comparison(
            deployment_rows,
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
            "algorithmic_failure_count": len(group_failures),
            "algorithmic_failures_count_as_unsuccessful_primary_outcomes": (
                True),
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
        "status": (
            "complete_with_failures" if failures
            else "complete_with_algorithmic_failures"
            if algorithmic_failures else "complete"
        ),
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
        "observed_functional_cell_count": int(
            len(functional_rows) + len(algorithmic_failures)),
        "algorithmic_failure_count": int(len(algorithmic_failures)),
        "profile_counterpart_cell_count": int(len(profile_rows)),
        "failures": failures,
        "algorithmic_failures": [
            {
                "path": row["_path"],
                "sha256": row["_sha256"],
                "cell": row["cell"],
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
            }
            for row in algorithmic_failures
        ],
        "summaries": summaries,
        "paired_initial_design_comparisons": initial_comparisons,
        "paired_deployment_comparisons": deployment_comparisons,
        "result_receipts": [
            {"path": row["_path"], "sha256": row["_sha256"]}
            for row in functional_rows + algorithmic_failures
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
        "algorithmic_failure_count": payload["algorithmic_failure_count"],
        "failure_count": len(payload["failures"]),
        "out": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
