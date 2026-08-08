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


PRIMARY_ARMS = (
    "source_atlas",
    "generic_dct_maximin",
    "random_low_frequency",
    "natural_blockwise",
    "raw_sobol",
)


CONFIGURATION_DEFAULTS = {
    "alpha": 0.05,
    "safe_mass": 0.08,
    "n0": 10,
    "source_task_count": 2,
    "source_profiles_per_task": 64,
    "source_replications_per_profile": 3,
    "atlas_max_frequency": 8,
    "atlas_frequency_penalty": 0.25,
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


def _receipt(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_mean(values, *, seed, repetitions=10_000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(int(repetitions), len(values)), replace=True)
    means = np.mean(samples, axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _outcome_key(row):
    """Lexicographic safety-first outcome, larger is better."""

    return (
        int(bool(row["contains_true_feasible"])),
        int(bool(row["independently_certified"])),
        -float(row["penalized_loss"]),
    )


def _paired_comparison(first_rows, second_rows, *, first_name, second_name):
    first = {int(row["target_seed"]): row for row in first_rows}
    second = {int(row["target_seed"]): row for row in second_rows}
    common = sorted(set(first) & set(second))
    wins = losses = ties = 0
    loss_difference = []
    for seed in common:
        first_key = _outcome_key(first[seed])
        second_key = _outcome_key(second[seed])
        if first_key > second_key:
            wins += 1
        elif first_key < second_key:
            losses += 1
        else:
            ties += 1
        loss_difference.append(
            float(first[seed]["penalized_loss"])
            - float(second[seed]["penalized_loss"])
        )
    non_ties = wins + losses
    return {
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
        "median_first_minus_second_penalized_loss": (
            None if not loss_difference else float(np.median(loss_difference))
        ),
    }


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
    for key, group in sorted(groups.items()):
        regime, arm, schema, descriptor, dimension, rank, configuration = key
        regrets = [
            float(row["finite_library_regret"])
            for row in group if row["finite_library_regret"] is not None
        ]
        losses = [float(row["penalized_loss"]) for row in group]
        summaries.append({
            "regime": regime,
            "arm": arm,
            "schema_mode": schema,
            "descriptor_mode": descriptor,
            "nominal_dimension": dimension,
            "effective_rank": rank,
            **_configuration_payload(configuration),
            "independent_task_count": len(group),
            "true_feasible_coverage_count": int(sum(
                bool(row["contains_true_feasible"]) for row in group)),
            "independently_certified_count": int(sum(
                bool(row["independently_certified"]) for row in group)),
            "false_certificate_count": int(sum(
                bool(row["false_certificate"]) for row in group)),
            "epsilon_optimal_005_count": int(sum(
                bool(row["feasible_and_epsilon_optimal_005"]) for row in group)),
            "median_feasible_regret": (
                None if not regrets else float(np.median(regrets))),
            "mean_penalized_loss": float(np.mean(losses)),
            "mean_penalized_loss_bootstrap_95ci": _bootstrap_mean(
                losses, seed=20260808 + len(summaries)),
            "mean_verification_calls": float(np.mean([
                row["verification_calls"] for row in group])),
            "mean_all_in_calls_unamortized": float(np.mean([
                row["all_in_calls_unamortized"] for row in group])),
            "mean_all_in_calls_amortized": float(np.mean([
                row["all_in_calls_amortized"] for row in group])),
        })

    comparisons = []
    contexts = sorted({
        (
            row["regime"], row["schema_mode"], row["descriptor_mode"],
            int(row["nominal_dimension"]), int(row["effective_rank"]),
            _configuration(row),
        )
        for row in rows
    })
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
                comparison = _paired_comparison(
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
                })
                comparisons.append(comparison)

    configuration_macro = []
    configurations = sorted({_configuration(row) for row in rows})
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
                row["independently_certified_count"]
                / row["independent_task_count"]
                for row in arm_summaries
            ])),
            "mean_group_penalized_loss": float(np.mean([
                row["mean_penalized_loss"] for row in arm_summaries
            ])),
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
        "independently_certified": bool(row["independently_certified"]),
        "false_certificate": bool(row["false_certificate"]),
        "finite_library_regret": row["finite_library_regret"],
        "penalized_loss": float(row["penalized_loss"]),
        "source_calls": int(row["source_calls"]),
        "target_search_calls": int(row["target_search_calls"]),
        "verification_calls": int(row["verification_calls"]),
        "all_in_calls_unamortized": int(row["all_in_calls_unamortized"]),
        "raw_result": Path(row["_path"]).name,
        "raw_sha256": row["_sha256"],
    } for row in rows]
    return {
        "schema_version": 2,
        "contract_id": "randomized_ordered_profile_stress_analysis_v2",
        "status": "complete" if rows and not failures else "incomplete",
        "inference_unit": "independent_target_task",
        "simulation_seed_role": "within_task_repeatability_only",
        "row_count": len(rows),
        "summaries": summaries,
        "paired_task_level_comparisons": comparisons,
        "configuration_macro_summary": configuration_macro,
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
