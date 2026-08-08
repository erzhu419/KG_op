#!/usr/bin/env python3
"""Task-level analysis for the randomized ordered-profile stress suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from performance.statistical_inference import (
    apply_holm_family,
    bootstrap_mean_ci,
    exact_binomial_interval,
    exact_binomial_lower_bound,
    exact_binomial_upper_bound,
)


PRIMARY_ARMS = (
    "source_atlas",
    "generic_dct_maximin",
    "random_low_frequency",
    "natural_blockwise",
    "raw_sobol",
)


CONFIGURATION_DEFAULTS = {
    "active_rank_override": None,
    "alpha": 0.05,
    "safe_mass": 0.08,
    "n0": 10,
    "N": 10,
    "source_task_count": 2,
    "source_profiles_per_task": 64,
    "source_replications_per_profile": 3,
    "atlas_max_frequency": 8,
    "atlas_frequency_penalty": 0.25,
    "atlas_safety_metric_weight": 1.0,
    "atlas_objective_metric_weight": 1.0,
    "atlas_first_center_safety_weight": 0.5,
}


def _configuration(row):
    """Return every registered sensitivity axis as a stable grouping key."""

    return tuple(
        row.get(name, default)
        for name, default in CONFIGURATION_DEFAULTS.items()
    )


def _configuration_payload(configuration):
    return dict(zip(CONFIGURATION_DEFAULTS, configuration))


def _stable_sort_key(value):
    """Order mixed optional configuration values deterministically."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str)


def _one_factor_sensitivity(configuration):
    payload = _configuration_payload(configuration)
    changed = [
        name for name, default in CONFIGURATION_DEFAULTS.items()
        if payload[name] != default
    ]
    if not changed:
        return "baseline", None
    if set(changed) == {"n0", "N"} and payload["N"] == payload["n0"]:
        return "n0", payload["n0"]
    if len(changed) != 1:
        return None, None
    axis = changed[0]
    if axis == "active_rank_override":
        return "active_rank", payload[axis]
    return axis, payload[axis]


def _receipt(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _certified_deployed_success(row):
    deployed = row.get("deployed_truth")
    return bool(
        row["independently_certified"]
        and isinstance(deployed, dict)
        and deployed.get("feasible") is True
        and not row["false_certificate"]
    )


def _deployed_regret(row):
    deployed = row.get("deployed_truth")
    if not _certified_deployed_success(row) or not isinstance(deployed, dict):
        return None
    return max(
        0.0,
        float(deployed["objective"])
        - float(row["finite_audit_library_oracle_objective"]),
    )


def _deployment_outcome_key(row):
    """Lexicographic deployed-policy outcome, larger is better."""

    success = _certified_deployed_success(row)
    regret = _deployed_regret(row)
    return (
        int(success),
        int(not bool(row["false_certificate"])),
        int(regret is not None and regret <= 0.05),
        -float(regret) if regret is not None else 0.0,
    )


def _initial_outcome_key(row):
    """Frontend-only outcome before continuation or verification."""

    return (
        int(_initial_contains_true_feasible(row)),
        -_initial_penalized_loss(row),
    )


def _initial_contains_true_feasible(row):
    return bool(row.get(
        "initial_design_contains_true_feasible",
        row["contains_true_feasible"],
    ))


def _initial_regret(row):
    return row.get(
        "initial_design_finite_audit_library_regret",
        row["finite_library_regret"],
    )


def _initial_penalized_loss(row):
    return float(row.get(
        "initial_design_penalized_loss",
        row["penalized_loss"],
    ))


def _paired_deployment_comparison(
    first_rows,
    second_rows,
    *,
    first_name,
    second_name,
    pair_key=lambda row: int(row["target_seed"]),
    bootstrap_seed=20260808,
):
    first = {pair_key(row): row for row in first_rows}
    second = {pair_key(row): row for row in second_rows}
    common = sorted(set(first) & set(second))
    wins = losses = ties = 0
    success_difference = []
    regret_difference = []
    for seed in common:
        first_key = _deployment_outcome_key(first[seed])
        second_key = _deployment_outcome_key(second[seed])
        if first_key > second_key:
            wins += 1
        elif first_key < second_key:
            losses += 1
        else:
            ties += 1
        success_difference.append(
            int(_certified_deployed_success(first[seed]))
            - int(_certified_deployed_success(second[seed]))
        )
        first_regret = _deployed_regret(first[seed])
        second_regret = _deployed_regret(second[seed])
        if first_regret is not None and second_regret is not None:
            regret_difference.append(first_regret - second_regret)
    non_ties = wins + losses
    return {
        "endpoint": "certified_deployed_policy",
        "first": first_name,
        "second": second_name,
        "paired_task_count": int(len(common)),
        "first_wins": int(wins),
        "first_losses": int(losses),
        "ties": int(ties),
        "one_sided_first_better_exact_sign_pvalue": (
            1.0 if non_ties == 0 else float(binomtest(
                wins, non_ties, p=0.5, alternative="greater").pvalue)
        ),
        "mean_first_minus_second_certified_success": (
            None if not success_difference
            else float(np.mean(success_difference))
        ),
        "mean_first_minus_second_certified_success_bootstrap_95ci": (
            None if not success_difference
            else bootstrap_mean_ci(success_difference, seed=bootstrap_seed)
        ),
        "paired_both_certified_safe_count": len(regret_difference),
        "median_first_minus_second_deployed_regret_when_both_safe": (
            None if not regret_difference
            else float(np.median(regret_difference))
        ),
        "mean_first_minus_second_deployed_regret_when_both_safe": (
            None if not regret_difference
            else float(np.mean(regret_difference))
        ),
    }


def _paired_initial_comparison(
    first_rows,
    second_rows,
    *,
    first_name,
    second_name,
    pair_key=lambda row: int(row["target_seed"]),
    bootstrap_seed=20260808,
):
    first = {pair_key(row): row for row in first_rows}
    second = {pair_key(row): row for row in second_rows}
    common = sorted(set(first) & set(second))
    wins = losses = ties = 0
    loss_difference = []
    coverage_difference = []
    for seed in common:
        first_key = _initial_outcome_key(first[seed])
        second_key = _initial_outcome_key(second[seed])
        if first_key > second_key:
            wins += 1
        elif first_key < second_key:
            losses += 1
        else:
            ties += 1
        coverage_difference.append(
            int(_initial_contains_true_feasible(first[seed]))
            - int(_initial_contains_true_feasible(second[seed]))
        )
        loss_difference.append(
            _initial_penalized_loss(first[seed])
            - _initial_penalized_loss(second[seed])
        )
    non_ties = wins + losses
    return {
        "endpoint": "initial_design_true_feasible_coverage",
        "first": first_name,
        "second": second_name,
        "paired_task_count": int(len(common)),
        "first_wins": int(wins),
        "first_losses": int(losses),
        "ties": int(ties),
        "one_sided_first_better_exact_sign_pvalue": (
            1.0 if non_ties == 0 else float(binomtest(
                wins, non_ties, p=0.5, alternative="greater").pvalue)
        ),
        "mean_first_minus_second_initial_coverage": (
            None if not coverage_difference
            else float(np.mean(coverage_difference))
        ),
        "mean_first_minus_second_initial_coverage_bootstrap_95ci": (
            None if not coverage_difference
            else bootstrap_mean_ci(coverage_difference, seed=bootstrap_seed)
        ),
        "median_first_minus_second_initial_penalized_loss": (
            None if not loss_difference else float(np.median(loss_difference))
        ),
        "mean_first_minus_second_initial_penalized_loss": (
            None if not loss_difference else float(np.mean(loss_difference))
        ),
    }


def _inference_family(schema, descriptor, configuration, *, prefix):
    default_configuration = tuple(CONFIGURATION_DEFAULTS.values())
    if (
        configuration == default_configuration
        and schema == "declared"
        and descriptor == "domain_blind"
    ):
        scope = "primary"
    elif configuration == default_configuration:
        scope = "schema_descriptor"
    else:
        scope = "sensitivity"
    return f"{prefix}:{scope}"


def analyze(paths):
    rows = []
    failures = []
    keys = set()
    for path in map(Path, paths):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path}: unreadable: {exc}")
            continue
        if row.get("contract_id") != "randomized_ordered_profile_stress_v2":
            failures.append(f"{path}: wrong contract_id")
            continue
        if row.get("status") != "ok":
            failures.append(f"{path}: status={row.get('status')}")
            continue
        key = (
            row["regime"], int(row["target_seed"]), row["arm"],
            row["schema_mode"], row["descriptor_mode"],
            int(row["nominal_dimension"]), int(row["effective_rank"]),
            _configuration(row),
        )
        if key in keys:
            failures.append(f"{path}: duplicate cell {key}")
            continue
        keys.add(key)
        row["_path"] = str(path)
        row["_sha256"] = _receipt(path)
        rows.append(row)

    groups = {}
    for row in rows:
        key = (
            row["regime"], row["arm"], row["schema_mode"],
            row["descriptor_mode"], int(row["nominal_dimension"]),
            int(row["effective_rank"]), _configuration(row),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(
        groups.items(), key=lambda item: _stable_sort_key(item[0])):
        regime, arm, schema, descriptor, dimension, rank, configuration = key
        initial_regrets = [
            float(_initial_regret(row))
            for row in group
            if _initial_regret(row) is not None
        ]
        regrets = [
            float(row["finite_library_regret"])
            for row in group if row["finite_library_regret"] is not None
        ]
        initial_losses = [
            _initial_penalized_loss(row)
            for row in group
        ]
        losses = [float(row["penalized_loss"]) for row in group]
        wall_times = [
            float(row["wall_time_sec"])
            for row in group if row.get("wall_time_sec") is not None
        ]
        task_count = int(len(group))
        feasible_count = int(sum(
            bool(row["contains_true_feasible"]) for row in group))
        initial_feasible_count = int(sum(
            _initial_contains_true_feasible(row)
            for row in group
        ))
        certified_count = int(sum(
            bool(row["independently_certified"]) for row in group))
        certified_safe_count = int(sum(
            _certified_deployed_success(row) for row in group))
        false_count = int(sum(
            bool(row["false_certificate"]) for row in group))
        deployed_regrets = [
            regret for row in group
            if (regret := _deployed_regret(row)) is not None
        ]
        summaries.append({
            "regime": regime,
            "arm": arm,
            "schema_mode": schema,
            "descriptor_mode": descriptor,
            "nominal_dimension": dimension,
            "effective_rank": rank,
            **_configuration_payload(configuration),
            "independent_task_count": task_count,
            "initial_design_true_feasible_coverage_count": (
                initial_feasible_count),
            "initial_design_true_feasible_coverage_rate": float(
                initial_feasible_count / task_count),
            "initial_design_true_feasible_coverage_exact_95ci": (
                exact_binomial_interval(initial_feasible_count, task_count)),
            "initial_design_epsilon_optimal_005_count": int(sum(
                bool(
                    _initial_regret(row) is not None
                    and _initial_regret(row) <= 0.05
                )
                for row in group
            )),
            "initial_design_median_feasible_regret": (
                None if not initial_regrets
                else float(np.median(initial_regrets))),
            "initial_design_mean_penalized_loss": float(
                np.mean(initial_losses)),
            "true_feasible_coverage_count": feasible_count,
            "true_feasible_coverage_rate": float(feasible_count / task_count),
            "true_feasible_coverage_exact_95ci": exact_binomial_interval(
                feasible_count, task_count),
            "true_feasible_coverage_one_sided_95_lower": (
                exact_binomial_lower_bound(feasible_count, task_count)),
            "independently_certified_count": certified_count,
            "independently_certified_rate": float(certified_count / task_count),
            "independently_certified_exact_95ci": exact_binomial_interval(
                certified_count, task_count),
            "independently_certified_one_sided_95_lower": (
                exact_binomial_lower_bound(certified_count, task_count)),
            "certified_true_feasible_deployment_count": certified_safe_count,
            "certified_true_feasible_deployment_rate": float(
                certified_safe_count / task_count),
            "certified_true_feasible_deployment_exact_95ci": (
                exact_binomial_interval(certified_safe_count, task_count)),
            "certified_deployed_epsilon_optimal_005_count": int(sum(
                regret <= 0.05 for regret in deployed_regrets)),
            "median_certified_deployed_regret": (
                None if not deployed_regrets
                else float(np.median(deployed_regrets))),
            "false_certificate_count": false_count,
            "false_certificate_rate": float(false_count / task_count),
            "false_certificate_one_sided_95_upper": (
                exact_binomial_upper_bound(false_count, task_count)),
            "coverage_probability_scope": (
                "declared registered task meta-distribution only"),
            "search_contains_epsilon_optimal_005_count": int(sum(
                bool(row["feasible_and_epsilon_optimal_005"]) for row in group)),
            "median_feasible_regret": (
                None if not regrets else float(np.median(regrets))),
            "mean_penalized_loss": float(np.mean(losses)),
            "mean_penalized_loss_bootstrap_95ci": bootstrap_mean_ci(
                losses, seed=20260808 + len(summaries)),
            "mean_verification_calls": float(np.mean([
                row["verification_calls"] for row in group])),
            "mean_all_in_calls_unamortized": float(np.mean([
                row["all_in_calls_unamortized"] for row in group])),
            "mean_all_in_calls_amortized": float(np.mean([
                row["all_in_calls_amortized"] for row in group])),
            "median_wall_time_sec": (
                None if not wall_times else float(np.median(wall_times))),
            "mean_wall_time_sec": (
                None if not wall_times else float(np.mean(wall_times))),
        })

    deployment_comparisons = []
    initial_comparisons = []
    contexts = sorted({
        (
            row["regime"], row["schema_mode"], row["descriptor_mode"],
            int(row["nominal_dimension"]), int(row["effective_rank"]),
            _configuration(row),
        )
        for row in rows
    }, key=_stable_sort_key)
    for context in contexts:
        context_rows = [
            row for row in rows
            if (
                row["regime"], row["schema_mode"], row["descriptor_mode"],
                int(row["nominal_dimension"]), int(row["effective_rank"]),
                _configuration(row),
            ) == context
        ]
        source = [row for row in context_rows if row["arm"] == "source_atlas"]
        for control in (
            "generic_dct_maximin", "random_low_frequency",
            "natural_blockwise", "raw_sobol",
        ):
            control_rows = [row for row in context_rows if row["arm"] == control]
            if source and control_rows:
                comparison = _paired_deployment_comparison(
                    source,
                    control_rows,
                    first_name="source_atlas",
                    second_name=control,
                )
                comparison.update({
                    "regime": context[0],
                    "schema_mode": context[1],
                    "descriptor_mode": context[2],
                    "nominal_dimension": context[3],
                    "effective_rank": context[4],
                    **_configuration_payload(context[5]),
                    "inference_family_id": _inference_family(
                        context[1], context[2], context[5],
                        prefix="task_level_context"),
                })
                deployment_comparisons.append(comparison)
                initial = _paired_initial_comparison(
                    source,
                    control_rows,
                    first_name="source_atlas",
                    second_name=control,
                )
                initial.update({
                    "regime": context[0],
                    "schema_mode": context[1],
                    "descriptor_mode": context[2],
                    "nominal_dimension": context[3],
                    "effective_rank": context[4],
                    **_configuration_payload(context[5]),
                    "inference_family_id": _inference_family(
                        context[1], context[2], context[5],
                        prefix="initial_design_context"),
                })
                initial_comparisons.append(initial)

    apply_holm_family(
        deployment_comparisons,
        pvalue_field="one_sided_first_better_exact_sign_pvalue",
        family_field="inference_family_id",
    )
    apply_holm_family(
        initial_comparisons,
        pvalue_field="one_sided_first_better_exact_sign_pvalue",
        family_field="inference_family_id",
    )

    macro_deployment_comparisons = []
    macro_initial_comparisons = []
    macro_contexts = sorted({
        (
            row["schema_mode"], row["descriptor_mode"],
            int(row["nominal_dimension"]), _configuration(row),
        )
        for row in rows
    }, key=_stable_sort_key)
    for macro_index, context in enumerate(macro_contexts):
        context_rows = [
            row for row in rows
            if (
                row["schema_mode"], row["descriptor_mode"],
                int(row["nominal_dimension"]), _configuration(row),
            ) == context
        ]
        source = [row for row in context_rows if row["arm"] == "source_atlas"]
        for control_index, control in enumerate(PRIMARY_ARMS[1:]):
            control_rows = [
                row for row in context_rows if row["arm"] == control]
            if not source or not control_rows:
                continue
            comparison = _paired_deployment_comparison(
                source,
                control_rows,
                first_name="source_atlas",
                second_name=control,
                pair_key=lambda row: (
                    str(row["regime"]), int(row["target_seed"])),
                bootstrap_seed=(
                    20260808 + 1000 * macro_index + control_index),
            )
            comparison.update({
                "schema_mode": context[0],
                "descriptor_mode": context[1],
                "nominal_dimension": context[2],
                **_configuration_payload(context[3]),
                "registered_regime_count": len({
                    row["regime"] for row in context_rows}),
                "inference_family_id": _inference_family(
                    context[0], context[1], context[3],
                    prefix="registered_generator_macro"),
                "population_claim": (
                    "registered randomized generator only; no unrestricted "
                    "domain-population claim"),
            })
            macro_deployment_comparisons.append(comparison)
            initial = _paired_initial_comparison(
                source,
                control_rows,
                first_name="source_atlas",
                second_name=control,
                pair_key=lambda row: (
                    str(row["regime"]), int(row["target_seed"])),
                bootstrap_seed=(
                    20270808 + 1000 * macro_index + control_index),
            )
            initial.update({
                "schema_mode": context[0],
                "descriptor_mode": context[1],
                "nominal_dimension": context[2],
                **_configuration_payload(context[3]),
                "registered_regime_count": len({
                    row["regime"] for row in context_rows}),
                "inference_family_id": _inference_family(
                    context[0], context[1], context[3],
                    prefix="registered_generator_initial_macro"),
                "population_claim": (
                    "registered randomized generator only; no unrestricted "
                    "domain-population claim"),
            })
            macro_initial_comparisons.append(initial)
    apply_holm_family(
        macro_deployment_comparisons,
        pvalue_field="one_sided_first_better_exact_sign_pvalue",
        family_field="inference_family_id",
    )
    apply_holm_family(
        macro_initial_comparisons,
        pvalue_field="one_sided_first_better_exact_sign_pvalue",
        family_field="inference_family_id",
    )

    configuration_macro = []
    configurations = sorted(
        {_configuration(row) for row in rows}, key=_stable_sort_key)
    for configuration in configurations:
        for arm in PRIMARY_ARMS:
            arm_summaries = [
                row for row in summaries
                if row["arm"] == arm
                and _configuration(row) == configuration
            ]
            if not arm_summaries:
                continue
            configuration_macro.append({
            "arm": arm,
            **_configuration_payload(configuration),
            "group_count": len(arm_summaries),
            "mean_task_feasible_rate": float(np.mean([
                row["true_feasible_coverage_count"]
                / row["independent_task_count"]
                for row in arm_summaries
            ])),
            "mean_task_certificate_rate": float(np.mean([
                row["certified_true_feasible_deployment_count"]
                / row["independent_task_count"]
                for row in arm_summaries
            ])),
            "mean_group_penalized_loss": float(np.mean([
                row["mean_penalized_loss"] for row in arm_summaries
            ])),
            })

    sensitivity_curves = []
    for row in configuration_macro:
        configuration = _configuration(row)
        axis, value = _one_factor_sensitivity(configuration)
        if axis is None:
            continue
        sensitivity_curves.append({
            "sensitivity_axis": axis,
            "sensitivity_value": value,
            **row,
        })

    compact_rows = [{
        "regime": row["regime"],
        "target_seed": int(row["target_seed"]),
        "arm": row["arm"],
        "schema_mode": row["schema_mode"],
        "descriptor_mode": row["descriptor_mode"],
        "nominal_dimension": int(row["nominal_dimension"]),
        "effective_rank": int(row["effective_rank"]),
        **_configuration_payload(_configuration(row)),
        "contains_true_feasible": bool(row["contains_true_feasible"]),
        "initial_design_contains_true_feasible": (
            _initial_contains_true_feasible(row)),
        "independently_certified": bool(row["independently_certified"]),
        "certified_true_feasible_deployment": (
            _certified_deployed_success(row)),
        "false_certificate": bool(row["false_certificate"]),
        "deployed_feasible_regret": _deployed_regret(row),
        "finite_library_regret": row["finite_library_regret"],
        "initial_design_finite_audit_library_regret": _initial_regret(row),
        "penalized_loss": float(row["penalized_loss"]),
        "initial_design_penalized_loss": _initial_penalized_loss(row),
        "source_calls": int(row["source_calls"]),
        "target_search_calls": int(row["target_search_calls"]),
        "verification_calls": int(row["verification_calls"]),
        "all_in_calls_unamortized": int(row["all_in_calls_unamortized"]),
        "wall_time_sec": (
            None if row.get("wall_time_sec") is None
            else float(row["wall_time_sec"])),
        "raw_result": Path(row["_path"]).name,
        "raw_sha256": row["_sha256"],
    } for row in rows]
    return {
        "schema_version": 3,
        "contract_id": "randomized_ordered_profile_stress_analysis_v3",
        "status": "complete" if rows and not failures else "incomplete",
        "inference_unit": "independent_target_task",
        "simulation_seed_role": "within_task_repeatability_only",
        "row_count": len(rows),
        "summaries": summaries,
        "primary_endpoint": "certified_deployed_policy",
        "endpoint_separation": {
            "frontend": "initial-design true-feasible coverage and loss",
            "search": "best true-feasible point encountered during search",
            "deployment": (
                "quality of the independently certified deployed policy"),
            "cross_credit_prohibited": True,
        },
        "paired_task_level_comparisons": deployment_comparisons,
        "registered_generator_macro_comparisons": (
            macro_deployment_comparisons),
        "paired_initial_design_comparisons": initial_comparisons,
        "registered_generator_initial_design_macro_comparisons": (
            macro_initial_comparisons),
        "configuration_macro_summary": configuration_macro,
        "one_factor_sensitivity_curves": sensitivity_curves,
        "compact_rows": compact_rows,
        "failures": failures,
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["status"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv")
    args = parser.parse_args()
    payload = analyze(args.paths)
    _atomic_json(args.out, payload)
    if args.csv:
        _write_csv(args.csv, payload["compact_rows"])
    print(json.dumps({
        "status": payload["status"],
        "out": str(args.out),
        "row_count": payload["row_count"],
        "failure_count": len(payload["failures"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
