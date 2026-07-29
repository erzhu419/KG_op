#!/usr/bin/env python3
"""Compute preregistered paired statistics from the compact paper audit.

The input is produced by ``paper_result_audit.py``.  No checkpoint, model
weight, simulator trace, or optimizer history is read here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PAIR_KEY = ("domain", "target_dimension", "seed")


def _seed_for(*parts):
    digest = hashlib.sha256(
        "\x1f".join(map(str, parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _exact_two_sided_sign_p(positive, negative):
    """Two-sided exact binomial/sign-test p-value, with ties removed."""

    positive = int(positive)
    negative = int(negative)
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(
        math.comb(n, k) for k in range(tail + 1)
    ) / float(2 ** n)
    return float(min(1.0, 2.0 * probability))


def _paired_bootstrap_interval(
    values,
    *,
    statistic,
    samples=10000,
    seed=20260729,
):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        value = float(statistic(values))
        return value, value
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(
        0, len(values), size=(int(samples), len(values)))
    estimates = np.asarray(
        [statistic(values[index]) for index in indexes],
        dtype=float,
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in original order."""

    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _record_index(records, *, track_id, method_identity):
    selected = [
        row for row in records
        if row["track_id"] == str(track_id)
        and row["method_identity"] == str(method_identity)
        and row["status"] == "ok"
    ]
    index = {}
    for row in selected:
        key = tuple(row[field] for field in PAIR_KEY)
        if key in index:
            raise ValueError(
                f"duplicate result for {track_id}/{method_identity}/{key}")
        index[key] = row
    return index


def _paired_rows(records, specification):
    left = _record_index(
        records,
        track_id=specification["left_track"],
        method_identity=specification["left_method"],
    )
    right = _record_index(
        records,
        track_id=specification["right_track"],
        method_identity=specification["right_method"],
    )
    expected_domains = set(map(
        str, specification.get("domains", ())))
    expected_dimensions = set(map(
        int, specification.get("dimensions", ())))

    def included(key):
        domain, dimension, _seed = key
        return (
            (not expected_domains or domain in expected_domains)
            and (
                not expected_dimensions
                or dimension in expected_dimensions
            )
        )

    left_keys = {key for key in left if included(key)}
    right_keys = {key for key in right if included(key)}
    missing_left = sorted(right_keys - left_keys)
    missing_right = sorted(left_keys - right_keys)
    keys = sorted(left_keys & right_keys)
    pairs = [(left[key], right[key]) for key in keys]
    equality_failures = {}
    for field in specification.get("required_equal_fields", ()):
        bad = [
            tuple(pair[0][key] for key in PAIR_KEY)
            for pair in pairs
            if pair[0].get(field) is None
            or pair[0].get(field) != pair[1].get(field)
        ]
        if bad:
            equality_failures[str(field)] = bad
    return pairs, missing_left, missing_right, equality_failures


def _summarize_pairs(pairs, *, comparison_id, stratum, samples):
    def mean_field(side, field):
        values = [
            pair[side].get(field) for pair in pairs
            if pair[side].get(field) is not None
        ]
        return None if not values else float(np.mean(values))

    def paired_metric(field, *, higher_is_better):
        values = [
            (float(left[field]), float(right[field]))
            for left, right in pairs
            if left.get(field) is not None
            and right.get(field) is not None
        ]
        differences = np.asarray([
            left - right for left, right in values
        ], dtype=float)
        oriented = (
            differences if higher_is_better else -differences
        )
        tolerance = 1e-12
        left_wins = int(np.sum(oriented > tolerance))
        right_wins = int(np.sum(oriented < -tolerance))
        ties = int(len(oriented) - left_wins - right_wins)
        interval = _paired_bootstrap_interval(
            differences,
            statistic=np.median,
            samples=samples,
            seed=_seed_for(comparison_id, stratum, field),
        )
        return {
            f"{field}_pair_count": int(len(values)),
            f"left_mean_{field}": (
                None if not values else float(np.mean([
                    value[0] for value in values
                ]))
            ),
            f"right_mean_{field}": (
                None if not values else float(np.mean([
                    value[1] for value in values
                ]))
            ),
            f"median_paired_{field}_difference_left_minus_right": (
                None
                if len(differences) == 0
                else float(np.median(differences))
            ),
            f"median_paired_{field}_difference_ci_low": interval[0],
            f"median_paired_{field}_difference_ci_high": interval[1],
            f"left_{field}_win_count": left_wins,
            f"right_{field}_win_count": right_wins,
            f"{field}_tie_count": ties,
            f"{field}_exact_sign_p": (
                None
                if not values
                else _exact_two_sided_sign_p(left_wins, right_wins)
            ),
            f"{field}_higher_is_better": bool(higher_is_better),
        }

    left_feasible = np.asarray([
        row[0]["true_feasible"] is True for row in pairs
    ], dtype=float)
    right_feasible = np.asarray([
        row[1]["true_feasible"] is True for row in pairs
    ], dtype=float)
    feasible_difference = left_feasible - right_feasible
    left_rescue = int(np.sum(
        (left_feasible == 1.0) & (right_feasible == 0.0)))
    left_loss = int(np.sum(
        (left_feasible == 0.0) & (right_feasible == 1.0)))
    feasible_ci = _paired_bootstrap_interval(
        feasible_difference,
        statistic=np.mean,
        samples=samples,
        seed=_seed_for(comparison_id, stratum, "feasible"),
    )

    left_certified = np.asarray([
        row[0]["terminal_certified"] is True for row in pairs
    ], dtype=float)
    right_certified = np.asarray([
        row[1]["terminal_certified"] is True for row in pairs
    ], dtype=float)
    certified_gain = int(np.sum(
        (left_certified == 1.0) & (right_certified == 0.0)))
    certified_loss = int(np.sum(
        (left_certified == 0.0) & (right_certified == 1.0)))

    regret_pairs = [
        (float(left["feasible_regret"]), float(right["feasible_regret"]))
        for left, right in pairs
        if left["true_feasible"] is True
        and right["true_feasible"] is True
        and left["feasible_regret"] is not None
        and right["feasible_regret"] is not None
    ]
    regret_differences = np.asarray([
        left - right for left, right in regret_pairs
    ], dtype=float)
    tolerance = 1e-12
    left_regret_wins = int(np.sum(regret_differences < -tolerance))
    right_regret_wins = int(np.sum(regret_differences > tolerance))
    regret_ties = int(
        len(regret_differences) - left_regret_wins - right_regret_wins)
    regret_ci = _paired_bootstrap_interval(
        regret_differences,
        statistic=np.median,
        samples=samples,
        seed=_seed_for(comparison_id, stratum, "regret"),
    )
    non_tied_regret = left_regret_wins + right_regret_wins
    rank_biserial = (
        None
        if non_tied_regret == 0
        else float(
            (left_regret_wins - right_regret_wins)
            / non_tied_regret
        )
    )
    summary = {
        "comparison_id": str(comparison_id),
        "stratum": str(stratum),
        "pair_count": int(len(pairs)),
        "left_mean_source_calls": mean_field(0, "source_calls"),
        "right_mean_source_calls": mean_field(1, "source_calls"),
        "left_mean_target_search_calls": mean_field(
            0, "target_search_calls"),
        "right_mean_target_search_calls": mean_field(
            1, "target_search_calls"),
        "left_mean_target_verification_calls": mean_field(
            0, "target_verification_calls"),
        "right_mean_target_verification_calls": mean_field(
            1, "target_verification_calls"),
        "left_mean_optimization_calls_excluding_verification": mean_field(
            0, "optimization_calls_excluding_verification"),
        "right_mean_optimization_calls_excluding_verification": mean_field(
            1, "optimization_calls_excluding_verification"),
        "left_true_feasible_count": int(np.sum(left_feasible)),
        "right_true_feasible_count": int(np.sum(right_feasible)),
        "left_rescue_count": left_rescue,
        "left_loss_count": left_loss,
        "paired_feasible_rate_difference": (
            None
            if len(feasible_difference) == 0
            else float(np.mean(feasible_difference))
        ),
        "paired_feasible_rate_difference_ci_low": feasible_ci[0],
        "paired_feasible_rate_difference_ci_high": feasible_ci[1],
        "feasibility_mcnemar_exact_p": _exact_two_sided_sign_p(
            left_rescue, left_loss),
        "left_certified_count": int(np.sum(left_certified)),
        "right_certified_count": int(np.sum(right_certified)),
        "left_certificate_gain_count": certified_gain,
        "left_certificate_loss_count": certified_loss,
        "certificate_mcnemar_exact_p": _exact_two_sided_sign_p(
            certified_gain, certified_loss),
        "left_false_certificate_count": int(sum(
            left["false_certificate"] for left, _right in pairs)),
        "right_false_certificate_count": int(sum(
            right["false_certificate"] for _left, right in pairs)),
        "both_feasible_regret_pair_count": int(len(regret_pairs)),
        "median_paired_regret_difference_left_minus_right": (
            None
            if len(regret_differences) == 0
            else float(np.median(regret_differences))
        ),
        "median_paired_regret_difference_ci_low": regret_ci[0],
        "median_paired_regret_difference_ci_high": regret_ci[1],
        "left_regret_win_count": left_regret_wins,
        "right_regret_win_count": right_regret_wins,
        "regret_tie_count": regret_ties,
        "paired_regret_rank_biserial_left_better_positive": rank_biserial,
        "regret_exact_sign_p": _exact_two_sided_sign_p(
            left_regret_wins, right_regret_wins),
    }
    for field, higher_is_better in (
        ("aleatoric_log_variance_rmse", False),
        ("aleatoric_variance_rmse", False),
        ("aleatoric_upper_coverage", True),
        ("aleatoric_variance_shape_correlation", True),
    ):
        summary.update(paired_metric(
            field, higher_is_better=higher_is_better))
    return summary


def analyze(audit, registry, *, bootstrap_samples=10000):
    records = audit["records"]
    rows = []
    comparison_audits = []
    for specification in registry.get("primary_comparisons", ()):
        comparison_id = str(specification["comparison_id"])
        pairs, missing_left, missing_right, equality_failures = (
            _paired_rows(records, specification)
        )
        expected_pairs = specification.get("expected_pairs")
        failure_reasons = []
        if missing_left:
            failure_reasons.append({
                "kind": "missing_left_pairs",
                "count": len(missing_left),
            })
        if missing_right:
            failure_reasons.append({
                "kind": "missing_right_pairs",
                "count": len(missing_right),
            })
        if expected_pairs is not None and len(pairs) != int(expected_pairs):
            failure_reasons.append({
                "kind": "pair_count_mismatch",
                "expected": int(expected_pairs),
                "observed": len(pairs),
            })
        for field, bad in equality_failures.items():
            failure_reasons.append({
                "kind": f"paired_{field}_mismatch",
                "count": len(bad),
            })
        comparison_audits.append({
            "comparison_id": comparison_id,
            "status": "pass" if not failure_reasons else "incomplete",
            "pair_count": len(pairs),
            "failures": failure_reasons,
        })
        if not pairs:
            continue
        domains = sorted({left["domain"] for left, _right in pairs})
        dimensions = sorted({
            left["target_dimension"] for left, _right in pairs
        })
        strata = [("all", pairs)]
        for domain in domains:
            strata.append((
                f"domain={domain}",
                [
                    pair for pair in pairs
                    if pair[0]["domain"] == domain
                ],
            ))
        if len(dimensions) > 1:
            for dimension in dimensions:
                strata.append((
                    f"dimension={dimension}",
                    [
                        pair for pair in pairs
                        if pair[0]["target_dimension"] == dimension
                    ],
                ))
        for stratum, selected in strata:
            row = _summarize_pairs(
                selected,
                comparison_id=comparison_id,
                stratum=stratum,
                samples=int(bootstrap_samples),
            )
            row.update({
                "left_track": specification["left_track"],
                "left_method": specification["left_method"],
                "right_track": specification["right_track"],
                "right_method": specification["right_method"],
            })
            rows.append(row)

    p_fields = (
        "feasibility_mcnemar_exact_p",
        "certificate_mcnemar_exact_p",
        "regret_exact_sign_p",
        "aleatoric_log_variance_rmse_exact_sign_p",
        "aleatoric_variance_rmse_exact_sign_p",
        "aleatoric_upper_coverage_exact_sign_p",
        "aleatoric_variance_shape_correlation_exact_sign_p",
    )
    hypotheses = [
        (row, field)
        for row in rows
        for field in p_fields
        if row.get(field) is not None
    ]
    adjusted = holm_adjust([
        row[field] for row, field in hypotheses
    ])
    for (row, field), value in zip(hypotheses, adjusted):
        row[f"{field}_holm"] = float(value)
    return {
        "schema_version": 1,
        "registry_id": registry.get("registry_id"),
        "audit_status": audit.get("status"),
        "status": (
            "complete"
            if comparison_audits
            and all(
                row["status"] == "pass"
                for row in comparison_audits
            )
            else "incomplete"
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "holm_family": (
            "all preregistered feasibility, certification, and paired "
            "regret hypotheses in this artifact"
        ),
        "comparison_audits": comparison_audits,
        "rows": rows,
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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    result = analyze(
        audit,
        registry,
        bootstrap_samples=args.bootstrap_samples,
    )
    _atomic_json(args.out, result)
    _write_csv(args.csv, result["rows"])
    print(json.dumps({
        "status": result["status"],
        "comparison_count": len(result["comparison_audits"]),
        "row_count": len(result["rows"]),
        "out": str(args.out),
        "csv": str(args.csv),
    }, indent=2))


if __name__ == "__main__":
    main()
