"""Aggregate the multi-family source-only TCB-V2 gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_quality import json_safe  # noqa: E402


CONFIG_FIELDS = (
    "descriptor_mode",
    "coordinate",
    "geometry",
    "rank",
    "ridge",
    "domain_penalty",
    "adaptation_ridge",
    "effect_ridge",
    "rotation_mode",
    "rotation_ridge",
    "target_residual_rank",
    "residual_ridge",
    "upper_alpha",
    "pilot_policy",
)
MODEL_CONFIG_FIELDS = tuple(
    field for field in CONFIG_FIELDS if field != "pilot_policy")


def _key(row):
    return tuple(row[field] for field in CONFIG_FIELDS)


def _model_key(row):
    return tuple(row[field] for field in MODEL_CONFIG_FIELDS)


def _mean(values):
    values = [float(value) for value in values if np.isfinite(float(value))]
    return float(statistics.fmean(values)) if values else None


def _median(values):
    values = [float(value) for value in values if np.isfinite(float(value))]
    return float(statistics.median(values)) if values else None


def load_rows(input_root):
    rows = []
    invalid = []
    for path in sorted(Path(input_root).rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
            for row in payload.get("rows", []):
                rows.append({**row, "source_file": str(path)})
        except (OSError, ValueError, TypeError) as exc:
            invalid.append({"path": str(path), "error": repr(exc)})
    return rows, invalid


def summarize_group(rows, args):
    metrics = [row["metrics"] for row in rows]
    frozen = [row["frozen_metrics"] for row in rows]
    evaluation_count = sum(item["evaluation_count"] for item in metrics)
    coverage_count = sum(item["coverage_count"] for item in metrics)
    predicted_safe_count = sum(item["predicted_safe_count"] for item in metrics)
    false_safe_count = sum(item["false_safe_count"] for item in metrics)
    frozen_predicted_safe = sum(
        item["predicted_safe_count"] for item in frozen)
    frozen_false_safe = sum(item["false_safe_count"] for item in frozen)
    coverage = coverage_count / max(evaluation_count, 1)
    false_safe_conditional = (
        false_safe_count / max(predicted_safe_count, 1))
    frozen_false_safe_conditional = (
        frozen_false_safe / max(frozen_predicted_safe, 1))
    rank_win_rate = float(np.mean([
        row["rank_improved"] for row in rows]))
    nonvacuous_rate = float(np.mean([
        row["nonvacuous_safe_set"] for row in rows]))
    false_safe_nonworse_rate = float(np.mean([
        row["false_safe_nonworse"] for row in rows]))
    target_oracle_used = any(
        bool(row["target_oracle_used_for_fit"]) for row in rows)
    max_adapter_dimension = max(
        int(row["adapter_effective_dimension"]) for row in rows)
    complete_domains = len(set(row["heldout"] for row in rows))
    complete_seeds = len(set(
        (row["heldout"], int(row["target_seed"])) for row in rows))
    expected_rows = int(args.expected_domains) * int(args.expected_seeds)
    checks = {
        "complete": complete_seeds >= expected_rows,
        "no_target_oracle": not target_oracle_used,
        "strict_lodo_descriptor": "provider_" not in str(
            rows[0]["descriptor_mode"]),
        "low_dimensional_adapter": max_adapter_dimension
        <= int(getattr(args, "maximum_adapter_dimension", 4)),
        "coverage": coverage >= float(args.minimum_coverage),
        "false_safe_nonworse": (
            false_safe_conditional
            <= frozen_false_safe_conditional
            + float(args.false_safe_tolerance)
        ),
        "absolute_spearman": _mean(
            item["spearman"] for item in metrics)
        >= float(args.minimum_spearman),
        "nonvacuous": nonvacuous_rate >= float(args.minimum_nonvacuous_rate),
    }
    config = {field: rows[0][field] for field in CONFIG_FIELDS}
    return {
        "config": config,
        "n_rows": int(len(rows)),
        "heldout_domains": sorted(set(row["heldout"] for row in rows)),
        "complete_domain_count": int(complete_domains),
        "complete_seed_count": int(complete_seeds),
        "coverage_rate": float(coverage),
        "predicted_safe_count": int(predicted_safe_count),
        "false_safe_count": int(false_safe_count),
        "false_safe_conditional_rate": float(false_safe_conditional),
        "frozen_predicted_safe_count": int(frozen_predicted_safe),
        "frozen_false_safe_count": int(frozen_false_safe),
        "frozen_false_safe_conditional_rate": float(
            frozen_false_safe_conditional),
        "rank_win_rate": rank_win_rate,
        "nonvacuous_rate": nonvacuous_rate,
        "false_safe_nonworse_rate": false_safe_nonworse_rate,
        "mean_spearman": _mean(item["spearman"] for item in metrics),
        "median_spearman": _median(item["spearman"] for item in metrics),
        "frozen_mean_spearman": _mean(
            item["spearman"] for item in frozen),
        "mean_safe_recall": _mean(item["safe_recall"] for item in metrics),
        "median_boundary_mae": _median(
            item["boundary_mae"] for item in metrics),
        "max_adapter_dimension": int(max_adapter_dimension),
        "target_oracle_used_for_fit": bool(target_oracle_used),
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
    }


def _outer_metrics(rows):
    metrics = [row["metrics"] for row in rows]
    frozen = [row["frozen_metrics"] for row in rows]
    evaluation_count = sum(item["evaluation_count"] for item in metrics)
    coverage_count = sum(item["coverage_count"] for item in metrics)
    predicted_safe_count = sum(
        item["predicted_safe_count"] for item in metrics)
    false_safe_count = sum(item["false_safe_count"] for item in metrics)
    frozen_predicted_safe = sum(
        item["predicted_safe_count"] for item in frozen)
    frozen_false_safe = sum(item["false_safe_count"] for item in frozen)
    return {
        "coverage_rate": float(
            coverage_count / max(evaluation_count, 1)),
        "predicted_safe_count": int(predicted_safe_count),
        "false_safe_count": int(false_safe_count),
        "false_safe_conditional_rate": float(
            false_safe_count / max(predicted_safe_count, 1)),
        "frozen_predicted_safe_count": int(frozen_predicted_safe),
        "frozen_false_safe_count": int(frozen_false_safe),
        "frozen_false_safe_conditional_rate": float(
            frozen_false_safe / max(frozen_predicted_safe, 1)),
        "rank_win_rate": float(np.mean([
            row["rank_improved"] for row in rows])),
        "nonvacuous_rate": float(np.mean([
            row["nonvacuous_safe_set"] for row in rows])),
        "mean_spearman": _mean(item["spearman"] for item in metrics),
        "frozen_mean_spearman": _mean(
            item["spearman"] for item in frozen),
        "mean_safe_recall": _mean(item["safe_recall"] for item in metrics),
        "median_boundary_mae": _median(
            item["boundary_mae"] for item in metrics),
    }


def _source_inner_policy_candidate(rows, args):
    config = {field: rows[0][field] for field in CONFIG_FIELDS}
    inner_rows = [row.get("source_inner_lodo") for row in rows]
    available = bool(inner_rows) and all(
        isinstance(item, dict) for item in inner_rows)
    if not available:
        return {
            "config": config,
            "source_inner_lodo_available": False,
            "source_selection_pass": False,
            "source_selection_checks": {
                "nested_source_lodo_present": False,
            },
            "selection_score": None,
        }
    # Source data and source seed are frozen, so this object is identical over
    # outer target seeds.  Reject accidental drift instead of averaging it.
    canonical = inner_rows[0]
    stable = all(
        item.get("fold_count") == canonical.get("fold_count")
        and np.isclose(
            float(item.get("mean_spearman", np.nan)),
            float(canonical.get("mean_spearman", np.nan)),
            equal_nan=True,
        )
        for item in inner_rows[1:]
    )
    false_safe = float(canonical["false_safe_conditional_rate"])
    frozen_false_safe = float(
        canonical["frozen_false_safe_conditional_rate"])
    expected_inner_folds = max(int(args.expected_domains) - 1, 2)
    checks = {
        "nested_source_lodo_present": True,
        "stable_over_outer_seeds": bool(stable),
        "outer_target_excluded": bool(canonical.get(
            "target_domain_excluded_from_training_and_selection", False)),
        "no_target_oracle": not bool(canonical.get(
            "target_oracle_used", True)),
        "strict_lodo_descriptor": "provider_" not in str(
            config["descriptor_mode"]),
        "complete_inner_folds": int(canonical["fold_count"])
        >= expected_inner_folds,
        "source_coverage": float(canonical["coverage_rate"])
        >= float(args.minimum_coverage),
        "source_false_safe_nonworse": false_safe
        <= frozen_false_safe + float(args.false_safe_tolerance),
        "source_absolute_spearman": float(canonical["mean_spearman"])
        >= float(args.minimum_spearman),
        "source_nonvacuous": float(canonical["nonvacuous_rate"])
        >= float(args.minimum_nonvacuous_rate),
    }
    passed = bool(all(checks.values()))
    score = (
        int(passed),
        float(canonical["mean_spearman"]),
        float(canonical["coverage_rate"]),
        float(canonical["nonvacuous_rate"]),
        -false_safe,
        -int(config["rank"]),
        -float(config["adaptation_ridge"]),
        -float(config["effect_ridge"]),
    )
    return {
        "config": config,
        "source_inner_lodo_available": True,
        "source_inner_lodo": canonical,
        "source_selection_checks": checks,
        "source_selection_pass": passed,
        "selection_score": list(score),
    }


def _source_inner_candidate(rows, args):
    by_policy = defaultdict(list)
    for row in rows:
        by_policy[str(row["pilot_policy"])].append(row)
    policy_candidates = {
        policy: _source_inner_policy_candidate(policy_rows, args)
        for policy, policy_rows in sorted(by_policy.items())
    }
    expected_policies = {"random", "source_boundary"}
    complete_policies = expected_policies.issubset(policy_candidates)
    all_pass = complete_policies and all(
        policy_candidates[policy]["source_selection_pass"]
        for policy in expected_policies
    )
    scores = [
        policy_candidates[policy].get("selection_score")
        for policy in expected_policies
        if policy in policy_candidates
    ]
    finite_scores = [score for score in scores if score is not None]
    if finite_scores:
        robust_score = [
            min(float(score[index]) for score in finite_scores)
            for index in range(len(finite_scores[0]))
        ]
        robust_score[0] = int(all_pass)
    else:
        robust_score = None
    return {
        "config": {
            field: rows[0][field] for field in MODEL_CONFIG_FIELDS
        },
        "pilot_policy_is_hyperparameter": False,
        "required_pilot_policies": sorted(expected_policies),
        "complete_pilot_policies": bool(complete_policies),
        "policy_candidates": policy_candidates,
        "source_selection_pass": bool(all_pass),
        "selection_score": robust_score,
    }


def _selection_sort_key(candidate):
    score = candidate.get("selection_score")
    if score is None:
        return (0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)
    return tuple(float(value) for value in score)


def summarize(rows, args):
    grouped = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)
    groups = [summarize_group(group, args) for group in grouped.values()]
    groups.sort(key=lambda item: (
        not item["gate_pass"],
        -item["rank_win_rate"],
        -item["nonvacuous_rate"],
        item["false_safe_conditional_rate"],
        -(item["mean_spearman"] or -np.inf),
    ))
    passing = [item for item in groups if item["gate_pass"]]
    outer_groups = defaultdict(list)
    for row in rows:
        outer_groups[(str(row["heldout"]), _model_key(row))].append(row)
    candidates_by_heldout = defaultdict(list)
    rows_by_heldout_key = {}
    for (heldout, key), candidate_rows in outer_groups.items():
        candidate = _source_inner_candidate(candidate_rows, args)
        candidate["heldout"] = heldout
        candidate["outer_seed_count"] = len(set(
            int(row["target_seed"]) for row in candidate_rows))
        candidate["outer_metrics"] = _outer_metrics(candidate_rows)
        candidate["outer_metrics_by_pilot_policy"] = {
            policy: _outer_metrics([
                row for row in candidate_rows
                if str(row["pilot_policy"]) == policy
            ])
            for policy in sorted(set(
                str(row["pilot_policy"]) for row in candidate_rows))
        }
        candidates_by_heldout[heldout].append(candidate)
        rows_by_heldout_key[(heldout, key)] = candidate_rows

    selected = {}
    selected_rows = []
    for heldout, candidates in sorted(candidates_by_heldout.items()):
        candidates.sort(key=_selection_sort_key, reverse=True)
        winner = candidates[0]
        selected[heldout] = winner
        winner_key = tuple(
            winner["config"][field] for field in MODEL_CONFIG_FIELDS)
        selected_rows.extend(rows_by_heldout_key[(heldout, winner_key)])

    outer = _outer_metrics(selected_rows) if selected_rows else None
    outer_by_policy = (
        {
            policy: _outer_metrics([
                row for row in selected_rows
                if str(row["pilot_policy"]) == policy
            ])
            for policy in sorted(set(
                str(row["pilot_policy"]) for row in selected_rows))
        }
        if selected_rows else {}
    )
    expected_heldouts = int(args.expected_domains)
    complete_outer = bool(
        len(selected) >= expected_heldouts
        and all(
            winner["outer_seed_count"] >= int(args.expected_seeds)
            for winner in selected.values()
        )
    )
    source_selected = bool(selected) and all(
        winner["source_selection_pass"] for winner in selected.values())
    no_selection_leakage = bool(selected_rows) and all(
        not bool(row.get(
            "target_oracle_used_for_hyperparameter_selection", True))
        for row in selected_rows
    )
    no_fit_leakage = bool(selected_rows) and all(
        not bool(row.get("target_oracle_used_for_fit", True))
        for row in selected_rows
    )
    if outer is None:
        final_checks = {
            "complete_outer_evaluation": False,
            "source_only_hyperparameter_selection": False,
        }
    else:
        final_checks = {
            "complete_outer_evaluation": complete_outer,
            "all_source_selections_pass": source_selected,
            "source_only_hyperparameter_selection": no_selection_leakage,
            "no_target_oracle_in_fit": no_fit_leakage,
            "outer_coverage": outer["coverage_rate"]
            >= float(args.minimum_coverage),
            "outer_false_safe_nonworse": (
                outer["false_safe_conditional_rate"]
                <= outer["frozen_false_safe_conditional_rate"]
                + float(args.false_safe_tolerance)
            ),
            "outer_absolute_spearman": outer["mean_spearman"]
            >= float(args.minimum_spearman),
            "outer_nonvacuous": outer["nonvacuous_rate"]
            >= float(args.minimum_nonvacuous_rate),
            "outer_checks_hold_for_each_pilot_policy": bool(
                {"random", "source_boundary"}.issubset(outer_by_policy)
                and all(
                    metrics["coverage_rate"]
                    >= float(args.minimum_coverage)
                    and metrics["false_safe_conditional_rate"]
                    <= metrics["frozen_false_safe_conditional_rate"]
                    + float(args.false_safe_tolerance)
                    and metrics["mean_spearman"]
                    >= float(args.minimum_spearman)
                    and metrics["nonvacuous_rate"]
                    >= float(args.minimum_nonvacuous_rate)
                    for metrics in outer_by_policy.values()
                )
            ),
        }
    nested_gate_pass = bool(final_checks) and all(final_checks.values())
    return {
        "schema_version": 2,
        "n_input_rows": int(len(rows)),
        "n_configurations": int(len(groups)),
        "n_passing": int(len(passing)),
        "gate_pass": nested_gate_pass,
        "selection_protocol": "nested_source_lodo_per_outer_target",
        "outer_truth_used_for_hyperparameter_selection": False,
        "promoted_candidate": (
            {
                "config_by_heldout": {
                    heldout: winner["config"]
                    for heldout, winner in selected.items()
                },
                "source_selection_by_heldout": selected,
                "outer_evaluation": outer,
                "outer_evaluation_by_pilot_policy": outer_by_policy,
                "checks": final_checks,
            }
            if nested_gate_pass else None
        ),
        "nested_selection_by_heldout": selected,
        "nested_outer_evaluation": outer,
        "nested_outer_evaluation_by_pilot_policy": outer_by_policy,
        "nested_checks": final_checks,
        "best_available": groups[0] if groups else None,
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-domains", type=int, default=5)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--false-safe-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--minimum-spearman", type=float, default=0.35,
        help=(
            "Pre-registered absolute ordering quality. Positive target "
            "location/scale adaptation preserves ranks, so adapted-vs-frozen "
            "rank wins are not a meaningful gate."
        ),
    )
    parser.add_argument("--minimum-nonvacuous-rate", type=float, default=0.50)
    parser.add_argument("--maximum-adapter-dimension", type=int, default=4)
    args = parser.parse_args()
    rows, invalid = load_rows(args.input_root)
    result = summarize(rows, args)
    result["invalid_files"] = invalid
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "gate_pass": result["gate_pass"],
        "n_input_rows": result["n_input_rows"],
        "n_configurations": result["n_configurations"],
        "n_passing": result["n_passing"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
