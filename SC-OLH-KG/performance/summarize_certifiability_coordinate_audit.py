"""Summarize the certifiability and coordinate-sufficiency audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _aggregate_metrics(rows):
    keys = (
        "coverage_rate",
        "predicted_safe_count",
        "false_safe_count",
        "safe_recall",
        "spearman",
        "margin_rmse",
        "boundary_mae",
        "normalized_margin_rmse",
        "normalized_boundary_mae",
        "safe_sign_accuracy",
        "boundary_safe_sign_accuracy",
    )
    return {key: _mean([row["metrics"] for row in rows], key) for key in keys}


def _aggregate_certifiability(results):
    groups = defaultdict(list)
    for result in results:
        for pool_name, pool in result["certifiability"].items():
            for budget, metrics in pool["replicate_budgets"].items():
                groups[(result["heldout"], pool_name, int(budget))].append({
                    **metrics,
                    "true_feasible_count": pool["true_feasible_count"],
                    "true_feasible_rate": pool["true_feasible_rate"],
                    "minimum_true_margin": pool["minimum_true_margin"],
                    "median_constraint_sigma": pool[
                        "median_constraint_sigma"],
                })
    output = []
    for (domain, pool, budget), rows in sorted(groups.items()):
        output.append({
            "heldout": domain,
            "pool": pool,
            "replicates": int(budget),
            "n_seeds": int(len(rows)),
            "mean_true_feasible_count": _mean(rows, "true_feasible_count"),
            "mean_true_feasible_rate": _mean(rows, "true_feasible_rate"),
            "mean_minimum_true_margin": _mean(rows, "minimum_true_margin"),
            "mean_median_constraint_sigma": _mean(
                rows, "median_constraint_sigma"),
            "known_variance_feasible_recall": _mean(
                rows, "known_variance_feasible_recall"),
            "unknown_variance_feasible_recall": _mean(
                rows, "unknown_variance_feasible_recall"),
            "known_variance_pool_rate": _mean(
                rows, "known_variance_pool_rate"),
            "unknown_variance_pool_rate": _mean(
                rows, "unknown_variance_pool_rate"),
        })
    return output


def _aggregate_required_replications(results):
    groups = defaultdict(list)
    for result in results:
        for pool_name, pool in result["certifiability"].items():
            row = pool["known_variance_required_replications"]
            groups[(result["heldout"], pool_name)].append(row)
    output = []
    for (domain, pool), rows in sorted(groups.items()):
        output.append({
            "heldout": domain,
            "pool": pool,
            "n_seeds": int(len(rows)),
            "mean_minimum": _mean(rows, "minimum"),
            "mean_q25": _mean(rows, "q25"),
            "mean_median": _mean(rows, "median"),
            "mean_q90": _mean(rows, "q90"),
            "mean_maximum": _mean(rows, "maximum"),
        })
    return output


def _aggregate_aliasing(results):
    groups = defaultdict(list)
    for result in results:
        for coordinate, metrics in result["coordinate_aliasing"].items():
            groups[(result["heldout"], coordinate)].append(metrics)
    keys = (
        "mean_neighbor_margin_discrepancy",
        "boundary_neighbor_margin_discrepancy",
        "normalized_neighbor_discrepancy",
        "normalized_boundary_neighbor_discrepancy",
    )
    return [{
        "heldout": domain,
        "coordinate": coordinate,
        "n_seeds": int(len(rows)),
        **{key: _mean(rows, key) for key in keys},
    } for (domain, coordinate), rows in sorted(groups.items())]


def _aggregate_rows(results):
    groups = defaultdict(list)
    for result in results:
        for row in result["rows"]:
            key = (
                row["heldout"],
                row["coordinate"],
                row["coordinate_stratum"],
                row["fit_stratum"],
                row["model_kind"],
                row["training_policy"],
                int(row["target_train_count"]),
            )
            groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        domain, coordinate, coordinate_stratum, fit_stratum, model, policy, count = key
        output.append({
            "heldout": domain,
            "coordinate": coordinate,
            "coordinate_stratum": coordinate_stratum,
            "fit_stratum": fit_stratum,
            "model_kind": model,
            "training_policy": policy,
            "target_train_count": count,
            "n_seeds": int(len(rows)),
            "coordinate_dimension": int(rows[0]["coordinate_dimension"]),
            "target_oracle_used_for_fit": bool(
                rows[0]["target_oracle_used_for_fit"]),
            "promotion_eligible": bool(rows[0]["promotion_eligible"]),
            "metrics": _aggregate_metrics(rows),
        })
    return output


def _best_coordinate(rows, domain, stratum):
    candidates = [
        row for row in rows
        if row["heldout"] == domain
        and row["coordinate_stratum"] == stratum
        and row["fit_stratum"] == "target_oracle_diagnostic"
        and row["training_policy"] == "oracle_boundary_stratified"
    ]
    if not candidates:
        return None
    largest = max(row["target_train_count"] for row in candidates)
    candidates = [
        row for row in candidates if row["target_train_count"] == largest]
    sufficient = [
        row for row in candidates
        if float(row["metrics"]["spearman"]) >= 0.75
        and float(row["metrics"]["normalized_boundary_mae"]) <= 0.50
    ]
    if sufficient:
        return min(sufficient, key=lambda row: (
            float(row["metrics"]["normalized_boundary_mae"]),
            -float(row["metrics"]["spearman"]),
            str(row["model_kind"]),
            str(row["coordinate"]),
        ))
    return max(candidates, key=lambda row: (
        float(row["metrics"]["spearman"]),
        -float(row["metrics"]["normalized_boundary_mae"]),
        str(row["model_kind"]),
        str(row["coordinate"]),
    ))


def _domain_decisions(results, row_groups, cert_groups):
    domains = sorted({result["heldout"] for result in results})
    decisions = []
    for domain in domains:
        domain_cert = [
            row for row in cert_groups
            if row["heldout"] == domain
            and row["pool"] == "domain_augmented_oracle_pool"
        ]
        max_budget = max((row["replicates"] for row in domain_cert), default=0)
        cert = next((
            row for row in domain_cert if row["replicates"] == max_budget
        ), None)
        source = _best_coordinate(
            row_groups, domain, "source_frozen_observable")
        provider = _best_coordinate(
            row_groups, domain, "domain_tuned_oracle_upper_bound")
        if cert is None or cert["mean_true_feasible_count"] <= 0.0:
            decision = "benchmark_pool_has_no_feasible_support"
        elif cert["known_variance_feasible_recall"] <= 0.0:
            decision = "benchmark_not_certifiable_at_tested_replication"
        elif source is None:
            decision = "source_coordinate_screen_incomplete"
        elif (
            source["metrics"]["spearman"] < 0.75
            or source["metrics"]["normalized_boundary_mae"] > 0.50
        ):
            if (
                provider is not None
                and provider["metrics"]["spearman"] >= 0.75
                and provider["metrics"]["normalized_boundary_mae"] <= 0.50
            ):
                decision = "source_coordinate_insufficient_provider_closes_gap"
            else:
                decision = "source_coordinate_insufficient"
        else:
            decision = "coordinate_sufficient_repair_adaptation_and_certificate"
        decisions.append({
            "heldout": domain,
            "decision": decision,
            "maximum_tested_replicates": int(max_budget),
            "certifiability": cert,
            "best_source_coordinate": source,
            "best_provider_upper_bound": provider,
        })
    return decisions


def summarize(root, *, expected_domains=(), expected_seeds=0):
    files = sorted(Path(root).rglob("result.json"))
    results = []
    invalid = []
    for path in files:
        try:
            result = json.loads(path.read_text())
            contract = result["leakage_contract"]
            if result.get("audit") != (
                "noise_certifiability_and_coordinate_sufficiency"
            ):
                raise ValueError("wrong audit schema")
            if not contract["outer_target_excluded_from_source_model"]:
                raise ValueError("outer target leaked into source model")
            if contract["all_audit_rows_promotion_eligible"]:
                raise ValueError("oracle audit cannot be promotion eligible")
            if any(row.get("promotion_eligible") for row in result["rows"]):
                raise ValueError("a result row is incorrectly promotion eligible")
            results.append(result)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})

    row_groups = _aggregate_rows(results)
    cert_groups = _aggregate_certifiability(results)
    required_groups = _aggregate_required_replications(results)
    aliasing_groups = _aggregate_aliasing(results)
    decisions = _domain_decisions(results, row_groups, cert_groups)
    observed = defaultdict(set)
    for result in results:
        observed[result["heldout"]].add(int(result["target_seed"]))
    expected_domains = tuple(expected_domains)
    completeness = {
        domain: {
            "observed_seeds": sorted(observed.get(domain, set())),
            "complete": bool(
                len(observed.get(domain, set())) >= int(expected_seeds)),
        }
        for domain in expected_domains
    }
    decision_values = {row["decision"] for row in decisions}
    if any("benchmark_" in value for value in decision_values):
        next_action = "repair_benchmark_support_or_replication_design"
    elif any("incomplete" in value for value in decision_values):
        next_action = "audit_incomplete"
    elif any("source_coordinate_insufficient" in value
             for value in decision_values):
        next_action = "rebuild_transferable_observable_coordinate"
    elif decisions:
        next_action = "repair_target_adaptation_and_certification"
    else:
        next_action = "audit_incomplete"
    return {
        "schema_version": 1,
        "audit": "noise_certifiability_and_coordinate_sufficiency_summary",
        "n_files": int(len(files)),
        "n_valid_results": int(len(results)),
        "invalid_files": invalid,
        "completeness": completeness,
        "certifiability_groups": cert_groups,
        "required_replication_groups": required_groups,
        "coordinate_groups": row_groups,
        "coordinate_aliasing_groups": aliasing_groups,
        "domain_decisions": decisions,
        "next_action": next_action,
        "promotion_eligible": False,
        "decision_thresholds": {
            "minimum_oracle_spearman": 0.75,
            "maximum_normalized_boundary_mae": 0.50,
            "provider_gap_ratio": 0.80,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-domains", default="")
    parser.add_argument("--expected-seeds", type=int, default=0)
    args = parser.parse_args()
    domains = tuple(
        value.strip() for value in args.expected_domains.split(",")
        if value.strip())
    result = summarize(
        args.root,
        expected_domains=domains,
        expected_seeds=args.expected_seeds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "n_valid_results": result["n_valid_results"],
        "next_action": result["next_action"],
        "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
