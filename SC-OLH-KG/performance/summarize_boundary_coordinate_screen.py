"""Rank the frozen 96-way source-only boundary-coordinate screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINE_VARIANT_ID = (
    "learned_psi__linear_monotone__frozen__r2"
)


def _variant_id(row):
    variant = row["variant"]
    return "__".join([
        str(variant["coordinate"]),
        str(variant["geometry"]),
        str(variant["adaptation"]),
        f"r{int(variant['rank'])}",
    ])


def _rank_key(row):
    aggregate = row["aggregate"]
    return (
        float(aggregate["worst_false_safe_rate"]),
        bool(aggregate["single_domain_collapse"]),
        float(aggregate["median_rank_loss"]),
        float(aggregate["median_boundary_rmse"]),
        -float(aggregate["minimum_upper_coverage"]),
        _variant_id(row),
    )


def _folds_by_domain(row):
    return {
        str(fold["heldout"]): fold for fold in row.get("folds", [])
    }


def _certificate_metrics(row):
    folds = list(row.get("folds", []))
    if not folds:
        return {
            "mean_certified_safe_rate": 0.0,
            "mean_false_unsafe_rate": 1.0,
            "minimum_dangerous_recall": 0.0,
            "nonvacuous_safe_fold_count": 0,
        }
    certified_safe_rates = []
    false_unsafe_rates = []
    dangerous_recalls = []
    nonvacuous = 0
    for fold in folds:
        safe_prevalence = float(fold["evaluation_feasible_rate"])
        false_safe = float(fold["false_safe_rate"])
        false_unsafe = float(fold["false_unsafe_rate"])
        certified_safe = safe_prevalence - false_unsafe + false_safe
        unsafe_prevalence = max(1.0 - safe_prevalence, 0.0)
        dangerous_recall = (
            1.0
            if unsafe_prevalence <= 1e-12
            else 1.0 - false_safe / unsafe_prevalence
        )
        certified_safe_rates.append(max(certified_safe, 0.0))
        false_unsafe_rates.append(false_unsafe)
        dangerous_recalls.append(dangerous_recall)
        nonvacuous += int(
            safe_prevalence > 0.05 and certified_safe > 1e-12)
    return {
        "mean_certified_safe_rate": float(
            sum(certified_safe_rates) / len(folds)),
        "mean_false_unsafe_rate": float(
            sum(false_unsafe_rates) / len(folds)),
        "minimum_dangerous_recall": float(min(dangerous_recalls)),
        "nonvacuous_safe_fold_count": int(nonvacuous),
    }


def _source_gate(row, baseline):
    aggregate = row["aggregate"]
    baseline_aggregate = baseline["aggregate"]
    row_folds = _folds_by_domain(row)
    baseline_folds = _folds_by_domain(baseline)
    common = sorted(set(row_folds) & set(baseline_folds))
    rank_wins = int(sum(
        float(row_folds[domain]["rank_loss"])
        < float(baseline_folds[domain]["rank_loss"]) - 1e-12
        for domain in common
    ))
    certificate = _certificate_metrics(row)
    baseline_certificate = _certificate_metrics(baseline)
    checks = {
        "complete_five_fold_comparison": len(common) == 5,
        "worst_false_safe_nonworse": bool(
            float(aggregate["worst_false_safe_rate"])
            <= float(baseline_aggregate["worst_false_safe_rate"]) + 1e-12
        ),
        "rank_loss_wins": rank_wins,
        "rank_loss_wins_required": 4,
        "rank_loss_improves_four_of_five": bool(
            len(common) == 5 and rank_wins >= 4),
        "no_single_domain_collapse": not bool(
            aggregate["single_domain_collapse"]),
        "minimum_dangerous_recall": certificate[
            "minimum_dangerous_recall"],
        "baseline_minimum_dangerous_recall": baseline_certificate[
            "minimum_dangerous_recall"],
        "dangerous_recall_nonworse": bool(
            certificate["minimum_dangerous_recall"]
            >= baseline_certificate["minimum_dangerous_recall"] - 1e-12
        ),
        "mean_certified_safe_rate": certificate[
            "mean_certified_safe_rate"],
        "baseline_mean_certified_safe_rate": baseline_certificate[
            "mean_certified_safe_rate"],
        "certified_safe_rate_improved": bool(
            certificate["mean_certified_safe_rate"]
            > baseline_certificate["mean_certified_safe_rate"] + 1e-12
        ),
        "nonvacuous_safe_fold_count": certificate[
            "nonvacuous_safe_fold_count"],
        "nonvacuous_safe_fold_count_required": 2,
        "nonvacuous_safe_folds": bool(
            certificate["nonvacuous_safe_fold_count"] >= 2),
        "adaptation_dimension_admissible": bool(
            aggregate["adaptation_dimension_admissible"]),
        "no_target_evaluation_fit": not bool(
            row["protocol"]["target_evaluation_used_for_fit"]),
        "no_target_oracle": not bool(
            row["protocol"]["target_oracle_used"]),
    }
    checks["passed"] = bool(all(
        value for key, value in checks.items()
        if key not in {
            "rank_loss_wins",
            "rank_loss_wins_required",
            "minimum_dangerous_recall",
            "baseline_minimum_dangerous_recall",
            "mean_certified_safe_rate",
            "baseline_mean_certified_safe_rate",
            "nonvacuous_safe_fold_count",
            "nonvacuous_safe_fold_count_required",
        }
    ))
    return checks


def summarize(root, keep=2):
    root = Path(root)
    rows = []
    failures = []
    for path in sorted(root.rglob("result.json")):
        try:
            row = json.loads(path.read_text())
            row["result_path"] = str(path)
            rows.append(row)
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"path": str(path), "error": repr(exc)})
    admissible = [
        row for row in rows
        if row.get("aggregate", {}).get("all_finite", False)
        and row.get("aggregate", {}).get(
            "adaptation_dimension_admissible", False)
        and not row.get("protocol", {}).get(
            "target_evaluation_used_for_fit", True)
        and not row.get("protocol", {}).get("target_oracle_used", True)
    ]
    by_id = {_variant_id(row): row for row in rows}
    baseline = by_id.get(BASELINE_VARIANT_ID)
    gated = []
    gate_diagnostics = []
    if baseline is not None:
        for row in admissible:
            if _variant_id(row) == BASELINE_VARIANT_ID:
                continue
            gate = _source_gate(row, baseline)
            gate_diagnostics.append({
                "variant_id": _variant_id(row),
                **gate,
            })
            if gate["passed"]:
                gated.append(row)
    ranked = sorted(gated, key=_rank_key)
    selected = ranked[: max(0, int(keep))]
    return {
        "schema_version": 1,
        "screen_root": str(root),
        "expected_configurations": 96,
        "completed_configurations": int(len(rows)),
        "failed_result_files": failures,
        "admissible_configurations": int(len(admissible)),
        "baseline_variant_id": BASELINE_VARIANT_ID,
        "baseline_available": baseline is not None,
        "source_gate_passed_configurations": int(len(gated)),
        "selection_uses_online_sentinel_domains": False,
        "selection_rule": (
            "source_gate(worst_false_safe<=fixed_learned_psi_baseline,"
            "rank_loss_improves>=4/5,no_collapse,adaptation_dim<=0.35*n0,"
            "dangerous_recall_nonworse,certified_safe_rate_improves,"
            "nonvacuous_safe_folds>=2,no_target_leakage); then "
            "lexicographic(worst_false_safe,"
            "median_rank_loss,median_boundary_rmse,-minimum_upper_coverage)"
        ),
        "source_gate_diagnostics": sorted(
            gate_diagnostics, key=lambda item: item["variant_id"]),
        "selected": [
            {
                "variant_id": _variant_id(row),
                "variant": row["variant"],
                "aggregate": row["aggregate"],
                "result_path": row["result_path"],
            }
            for row in selected
        ],
        "ranking": [
            {
                "rank": index + 1,
                "variant_id": _variant_id(row),
                "variant": row["variant"],
                "aggregate": row["aggregate"],
                "result_path": row["result_path"],
            }
            for index, row in enumerate(ranked)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.root, keep=args.keep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "completed": payload["completed_configurations"],
        "admissible": payload["admissible_configurations"],
        "source_gate_passed": payload[
            "source_gate_passed_configurations"],
        "selected": [row["variant_id"] for row in payload["selected"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
