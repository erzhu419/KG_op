"""Summarize the eligible-query experiment from its compact staged receipts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from score_eligible_prefix import OUTER_SEEDS, OUTER_SIZES


def cohort_summary(receipts: list[dict]) -> dict:
    primary_key = OUTER_SEEDS[0], OUTER_SIZES[-1]
    by_family = {}
    expected = {(seed, count) for seed in OUTER_SEEDS for count in OUTER_SIZES}
    for family in sorted({row["family"] for row in receipts}):
        rows = [row for row in receipts if row["family"] == family]
        laws = {
            (trial["fantasy_seed"], trial["fantasy_count"]): []
            for trial in rows[0]["laws"]
        }
        for row in rows:
            key = {(trial["fantasy_seed"], trial["fantasy_count"]) for trial in row["laws"]}
            if len(row["laws"]) != len(expected) or key != expected:
                raise ValueError(f"Incomplete or duplicate fantasy rules for {row['job_id']}")
            for trial in row["laws"]:
                laws[(trial["fantasy_seed"], trial["fantasy_count"])].append(trial)
        roster_sizes = Counter()
        for row in rows:
            roster_sizes.update(row["native_roster_sizes"])
        primary = laws[primary_key]
        first_unique = [trial for trial in primary if trial["retained_queries"] == 1]
        gains = [trial["model_value_gain"] for trial in primary]
        stable_numerics = [
            trial["first_query"] == trial["first_beta24_query"]
            and len(set(trial["numerical_winners"])) == 1
            for trial in primary
        ]
        cpu = {
            key: median(row["cpu_seconds_by_component"][key] for row in rows)
            for key in rows[0]["cpu_seconds_by_component"]
        }
        by_family[family] = dict(
            states=len(rows),
            query_positions=sum(row["query_count"] for row in rows),
            retained_query_positions=sum(trial["retained_queries"] for trial in primary),
            queries_refined_across_laws=sum(row["evaluated_by_inner"] for row in rows),
            fantasy_native_updates=sum(row["fantasy_updates"] for row in rows),
            native_roster_sizes=dict(sorted(roster_sizes.items())),
            guarded_fantasies=sum(row["native_guarded_fantasies"] for row in rows),
            newly_eligible_in_slot=[
                sum(row["newly_eligible_in_slot"][slot] for row in rows) for slot in range(3)
            ],
            first_vs_full_disagreements=sum(not trial["full_matches_first"] for trial in primary),
            unique_full_query=len(first_unique),
            unique_full_query_agrees_first=sum(trial["full_matches_first"] for trial in first_unique),
            first_query_excluded=sum(trial["first_excluded"] for trial in primary),
            first_slot_query_vs_cached=sum(trial["first_query"] != row["cached_native_query"]
                                            for row, trial in zip(rows, primary, strict=True)),
            numerical_agreement=sum(stable_numerics),
            numerically_agreeing_disagreements=sum(
                stable and not trial["full_matches_first"]
                for stable, trial in zip(stable_numerics, primary, strict=True)
            ),
            disagreement_status_agrees_outer=sum(
                len({trial["full_matches_first"] for trial in row["laws"]}) == 1 for row in rows
            ),
            first_identity_agrees_outer=sum(
                len({trial["first_query"] for trial in row["laws"]}) == 1 for row in rows
            ),
            full_identity_agrees_outer=sum(
                len({trial["full_query"] for trial in row["laws"]}) == 1 for row in rows
            ),
            median_model_value_gain=median(gains),
            maximum_model_value_gain=max(gains),
            maximum_gain_upper_bound=max(trial["model_gain_bounds"][1] for trial in primary),
            largest_fine_interval_overshoot=max(trial["worst_fine_interval_overshoot"] for trial in primary),
            largest_fine_scramble_difference=max(trial["largest_fine_scramble_difference"] for trial in primary),
            median_cpu_seconds_by_component=cpu,
            median_whole_trial_cpu_seconds=median(row["cpu_seconds"] for row in rows),
            fantasy_law_checks=[
                dict(fantasy_seed=seed, fantasy_count=count,
                     disagreements=sum(not trial["full_matches_first"] for trial in trials),
                     unique_full_query=sum(trial["retained_queries"] == 1 for trial in trials))
                for (seed, count), trials in laws.items()
            ],
        )
    return by_family


def summarize(bundle: dict) -> dict:
    result = dict(primary_fantasy_seed=OUTER_SEEDS[0], primary_fantasy_count=OUTER_SIZES[-1])
    for cohort, stage in bundle["stages"].items():
        rows = stage["state_receipts"]
        if any(r["status"] != "ok" for r in rows) or len({r["job_id"] for r in rows}) != len(rows):
            raise ValueError(f"{cohort} has failed, unfinished, or duplicate receipts")
        result[cohort] = cohort_summary(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = json.loads(args.receipts.read_text(encoding="utf-8"))
    args.output.write_text(json.dumps(summarize(receipts), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
