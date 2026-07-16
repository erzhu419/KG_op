#!/usr/bin/env python3
"""Paired causal analysis for the structural-prior matrix.

Each contrast reuses the same held-out domain, target seed, source archive,
dimension, and target budget.  Feasibility is compared before regret so an
infeasible recommendation can never look good through conditional regret.
Only ``result.json`` files are read through ``aggregate_completed_matrix``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Iterable

from aggregate_completed_matrix import load_rows


COMPONENTS = (
    "low_frequency",
    "orthogonality",
    "sparsity",
    "additivity",
)


def parse_causal_variant(value):
    parts = [part for part in str(value or "").split("/") if part]
    if len(parts) != 4 or parts[0] != "causal_prior_v2":
        return None
    return {
        "causal_mode": parts[1],
        "proposal_mode": parts[2],
        "profile": parts[3],
    }


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _boolean(value):
    return value if isinstance(value, bool) else None


def _median(values):
    finite = [value for value in (_finite(item) for item in values) if value is not None]
    return None if not finite else float(statistics.median(finite))


def _mean(values):
    finite = [value for value in (_finite(item) for item in values) if value is not None]
    return None if not finite else float(statistics.mean(finite))


def _two_sided_sign_p(wins, losses):
    trials = int(wins) + int(losses)
    if trials <= 0:
        return None
    lower = min(int(wins), int(losses))
    probability = sum(
        math.comb(trials, index) for index in range(lower + 1)
    ) / float(2 ** trials)
    return float(min(1.0, 2.0 * probability))


def _paired_boolean(first, second, key):
    first_value = _boolean(first.get(key))
    second_value = _boolean(second.get(key))
    if first_value is None or second_value is None:
        return None
    return int(first_value) - int(second_value)


def _same_nonempty(first, second):
    return first is not None and second is not None and str(first) == str(second)


def build_paired_effects(rows):
    index = {}
    for row in rows:
        variant = parse_causal_variant(row.get("variant"))
        if variant is None or row.get("status") != "ok":
            continue
        key = (
            row.get("run_id"),
            variant["causal_mode"],
            variant["proposal_mode"],
            row.get("domain"),
            row.get("seed"),
            row.get("d"),
            row.get("N"),
            row.get("n0"),
            row.get("source_calls"),
        )
        profile_key = (*key, variant["profile"])
        if profile_key in index:
            raise ValueError(f"duplicate causal cell {profile_key}")
        index[profile_key] = row

    contrasts = [("full_vs_none", "all", "full", "none")]
    for component in COMPONENTS:
        contrasts.extend([
            (
                "single_only_vs_none",
                component,
                f"{component}_only",
                "none",
            ),
            (
                "full_vs_leave_one_out",
                component,
                "full",
                f"leave_out_{component}",
            ),
        ])

    base_keys = sorted({key[:-1] for key in index})
    pairs = []
    for base in base_keys:
        for contrast, component, challenger_profile, reference_profile in contrasts:
            challenger = index.get((*base, challenger_profile))
            reference = index.get((*base, reference_profile))
            if challenger is None or reference is None:
                continue
            pairs.append({
                "run_id": base[0],
                "causal_mode": base[1],
                "proposal_mode": base[2],
                "domain": base[3],
                "seed": base[4],
                "d": base[5],
                "N": base[6],
                "n0": base[7],
                "source_calls": base[8],
                "contrast": contrast,
                "component": component,
                "challenger_profile": challenger_profile,
                "reference_profile": reference_profile,
                "archive_match": _same_nonempty(
                    challenger.get("source_archive_fingerprint"),
                    reference.get("source_archive_fingerprint"),
                ),
                "initial_fingerprint_match": _same_nonempty(
                    challenger.get("initial_design_fingerprint"),
                    reference.get("initial_design_fingerprint"),
                ),
                "initial_feasible_delta": _paired_boolean(
                    challenger, reference, "initial_has_true_feasible"),
                "final_feasible_delta": _paired_boolean(
                    challenger, reference, "true_feasible"),
                "initial_regret_delta": (
                    None
                    if not (
                        challenger.get("initial_has_true_feasible") is True
                        and reference.get("initial_has_true_feasible") is True
                    )
                    else (
                        _finite(challenger.get("initial_best_feasible_regret"))
                        - _finite(reference.get("initial_best_feasible_regret"))
                        if (
                            _finite(challenger.get("initial_best_feasible_regret"))
                            is not None
                            and _finite(reference.get("initial_best_feasible_regret"))
                            is not None
                        )
                        else None
                    )
                ),
                "final_regret_delta": (
                    None
                    if not (
                        challenger.get("true_feasible") is True
                        and reference.get("true_feasible") is True
                    )
                    else (
                        _finite(challenger.get("feasible_regret"))
                        - _finite(reference.get("feasible_regret"))
                        if (
                            _finite(challenger.get("feasible_regret")) is not None
                            and _finite(reference.get("feasible_regret")) is not None
                        )
                        else None
                    )
                ),
            })
    return pairs


def summarize_pairs(pairs):
    groups = {}
    fields = (
        "run_id",
        "causal_mode",
        "proposal_mode",
        "domain",
        "d",
        "N",
        "n0",
        "source_calls",
        "contrast",
        "component",
        "challenger_profile",
        "reference_profile",
    )
    for pair in pairs:
        key = tuple(pair[field] for field in fields)
        groups.setdefault(key, []).append(pair)

    summaries = []
    for key, items in sorted(groups.items()):
        base = dict(zip(fields, key))
        initial_delta = [
            int(item["initial_feasible_delta"])
            for item in items
            if item["initial_feasible_delta"] is not None
        ]
        final_delta = [
            int(item["final_feasible_delta"])
            for item in items
            if item["final_feasible_delta"] is not None
        ]
        final_regret = [
            float(item["final_regret_delta"])
            for item in items
            if item["final_regret_delta"] is not None
        ]
        initial_wins = sum(value > 0 for value in initial_delta)
        initial_losses = sum(value < 0 for value in initial_delta)
        final_wins = sum(value > 0 for value in final_delta)
        final_losses = sum(value < 0 for value in final_delta)
        regret_wins = sum(value < -1e-12 for value in final_regret)
        regret_losses = sum(value > 1e-12 for value in final_regret)
        summaries.append({
            **base,
            "n_pairs": len(items),
            "archive_match_count": sum(item["archive_match"] for item in items),
            "initial_fingerprint_match_count": sum(
                item["initial_fingerprint_match"] for item in items),
            "initial_feasible_win_count": initial_wins,
            "initial_feasible_loss_count": initial_losses,
            "initial_feasible_net": initial_wins - initial_losses,
            "initial_feasible_mcnemar_p": _two_sided_sign_p(
                initial_wins, initial_losses),
            "final_feasible_win_count": final_wins,
            "final_feasible_loss_count": final_losses,
            "final_feasible_net": final_wins - final_losses,
            "final_feasible_mcnemar_p": _two_sided_sign_p(
                final_wins, final_losses),
            "both_final_feasible_count": len(final_regret),
            "median_final_regret_delta": _median(final_regret),
            "mean_final_regret_delta": _mean(final_regret),
            "final_regret_win_count": regret_wins,
            "final_regret_loss_count": regret_losses,
            "final_regret_sign_p": _two_sided_sign_p(
                regret_wins, regret_losses),
            "median_initial_regret_delta": _median(
                item["initial_regret_delta"] for item in items),
        })
    return summaries


def _write_csv(path: Path, rows: list[dict], fields: Iterable[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rows, errors = load_rows(args.roots)
    pairs = build_paired_effects(rows)
    summaries = summarize_pairs(pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "paired_rows.csv", pairs)
    _write_csv(args.out_dir / "paired_effects.csv", summaries)
    audit = {
        "schema_version": 1,
        "roots": [str(root.resolve()) for root in args.roots],
        "parsed_result_count": len(rows),
        "parse_errors": errors,
        "paired_row_count": len(pairs),
        "paired_effect_count": len(summaries),
        "safety_contract": {
            "accepted_filename": "result.json",
            "checkpoint_or_pickle_read": False,
        },
        "effects": summaries,
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "parsed_results": len(rows),
        "paired_rows": len(pairs),
        "paired_effects": len(summaries),
        "parse_errors": len(errors),
    }))


if __name__ == "__main__":
    main()
