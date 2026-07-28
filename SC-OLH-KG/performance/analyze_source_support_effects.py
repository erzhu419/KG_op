#!/usr/bin/env python3
"""Paired comparison of source policy-support designs at fixed target cells."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aggregate_completed_matrix import load_rows
from analyze_causal_prior_effects import (
    _paired_metrics,
    parse_causal_variant,
    summarize_proposal_mode_pairs,
)


def build_source_support_pairs(
        rows,
        *,
        challenger_run,
        reference_run,
        challenger_label,
        reference_label):
    index = {}
    for row in rows:
        variant = parse_causal_variant(row.get("variant"))
        if variant is None or row.get("status") != "ok":
            continue
        key = (
            row.get("run_id"),
            variant["causal_mode"],
            variant["proposal_mode"],
            variant["profile"],
            row.get("domain"),
            row.get("seed"),
            row.get("d"),
            row.get("N"),
            row.get("n0"),
            row.get("source_calls"),
        )
        if key in index:
            raise ValueError(f"duplicate source-support cell {key}")
        index[key] = row

    base_keys = sorted({key[1:] for key in index})
    pairs = []
    for base in base_keys:
        challenger = index.get((challenger_run, *base))
        reference = index.get((reference_run, *base))
        if challenger is None or reference is None:
            continue
        pairs.append({
            "run_id": f"{challenger_run}_vs_{reference_run}",
            "causal_mode": base[0],
            "proposal_mode": base[1],
            "profile": base[2],
            "domain": base[3],
            "seed": base[4],
            "d": base[5],
            "N": base[6],
            "n0": base[7],
            "source_calls": base[8],
            "contrast": "source_policy_support",
            "challenger_proposal_mode": challenger_label,
            "reference_proposal_mode": reference_label,
            "target_budget_match": all(
                challenger.get(key) == reference.get(key)
                for key in ("domain", "seed", "d", "N", "n0", "source_calls")
            ),
            **_paired_metrics(challenger, reference),
        })
    return pairs


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--challenger-run", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--challenger-label", required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rows, errors = load_rows(args.roots)
    pairs = build_source_support_pairs(
        rows,
        challenger_run=args.challenger_run,
        reference_run=args.reference_run,
        challenger_label=args.challenger_label,
        reference_label=args.reference_label,
    )
    summaries = summarize_proposal_mode_pairs(pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "paired_rows.csv", pairs)
    _write_csv(args.out_dir / "paired_effects.csv", summaries)
    audit = {
        "schema_version": 1,
        "challenger_run": args.challenger_run,
        "reference_run": args.reference_run,
        "challenger_label": args.challenger_label,
        "reference_label": args.reference_label,
        "parsed_result_count": len(rows),
        "parse_errors": errors,
        "paired_row_count": len(pairs),
        "paired_effect_count": len(summaries),
        "all_target_budgets_match": all(
            item["target_budget_match"] for item in pairs),
        "source_archive_match_count": sum(
            item["archive_match"] for item in pairs),
        "checkpoint_or_pickle_read": False,
        "effects": summaries,
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "parsed_results": len(rows),
        "paired_rows": len(pairs),
        "paired_effects": len(summaries),
        "parse_errors": len(errors),
        "all_target_budgets_match": audit["all_target_budgets_match"],
    }))


if __name__ == "__main__":
    main()
