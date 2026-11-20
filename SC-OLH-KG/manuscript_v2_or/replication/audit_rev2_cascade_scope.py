"""Check Rev2 menu eligibility and reconcile its sealed scientific readouts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def eligibility(records: list, candidates: list, rosters: list, saved_query: int, decimals: int) -> dict:
    def key(policy):
        return np.round(np.asarray(policy, dtype=float), decimals).tobytes()

    observed = {key(record.x_norm) for record in records}
    candidate_keys = [key(policy) for policy in candidates]
    candidate_counts = Counter(candidate_keys)
    checked = []
    for menu in rosters:
        primary, guard, backup = (candidate_keys[i] for i in menu)
        missing = {primary, guard, backup} - observed
        if len(missing) == 1:
            admissible_queries = candidate_counts[next(iter(missing))]
        else:
            admissible_queries = 0 if missing else len(candidates)
        checked.append({
            "roster": menu,
            "slot_observed": [policy in observed for policy in (primary, guard, backup)],
            "primary_backup_distinct": primary != backup,
            "distinct_unobserved_policies": len(missing),
            "admissible_query_count": admissible_queries,
            "admissible_after_saved_query": missing <= {candidate_keys[saved_query]},
        })
    return {"observed_policies": len(observed), "rosters": checked}


def audit_state(work: tuple[Path, Path, Path]) -> bool:
    path, prefix, output = work
    start = time.process_time()
    result = {"job_id": path.stem, "simulator_calls": 0, "source_refits": 0}
    try:
        old = read_json(path)
        with (prefix / f"{path.stem}.pkl").open("rb") as handle:
            state = pickle.load(handle)  # Existing, controlled execution cache.
        candidates = state["candidates"]
        if (state["family"]["name"], state["task_index"], state["saved_query"], len(candidates)) != (
            old["family"], old["target_task_idx"], old["saved_query"], old["candidate_count"]
        ):
            raise ValueError("Frozen prefix and score receipt refer to different candidate pools")
        result.update({
            "family": old["family"],
            "target_task_idx": old["target_task_idx"],
            "candidate_count": len(candidates),
            "by_rounding_places": {
                str(decimals): eligibility(state["records"], candidates, old["rosters"],
                                           old["saved_query"], decimals)
                for decimals in (5, 6)
            },
            "status": "ok",
        })
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    result["worker_cpu_seconds"] = time.process_time() - start
    write_json(output / path.name, result)
    return result["status"] == "ok"


def summarize(roster_rows: list[dict], cascade: dict, source: dict, task_ids: list[str]) -> dict:
    scoring = {row["job_id"]: row for row in cascade["scoring"]["receipts"]}
    refined = {row["job_id"]: row for row in cascade["refinement"]["receipts"]}
    if ({row["job_id"] for row in roster_rows} != set(scoring) or len(roster_rows) != len(scoring)
            or set(refined) != set(scoring) or any(row["status"] != "ok" for row in roster_rows)):
        raise ValueError("Audit receipts do not cover every scored and refined state exactly once")
    roster_summary = {}
    for decimals in (5, 6):
        lists = []
        for row in roster_rows:
            menus = row["by_rounding_places"][str(decimals)]["rosters"]
            if [menu["roster"] for menu in menus] != scoring[row["job_id"]]["rosters"]:
                raise ValueError("Audited and sealed rosters differ")
            lists.extend(menus)
        roster_summary[str(decimals)] = {
            "rosters": len(lists),
            "slot_observed": [sum(menu["slot_observed"][i] for menu in lists) for i in range(3)],
            "primary_backup_distinct": sum(menu["primary_backup_distinct"] for menu in lists),
            "rosters_by_number_unobserved": dict(sorted(Counter(
                menu["distinct_unobserved_policies"] for menu in lists
            ).items())),
            "admissible_for_some_query": sum(menu["admissible_query_count"] > 0 for menu in lists),
            "admissible_after_saved_query": sum(menu["admissible_after_saved_query"] for menu in lists),
        }

    families = {}
    for family, loo in source["stages"]["posterior"].items():
        rows = [row for row in refined.values() if row["family"] == family]
        covariance = {}
        for mode in ("full", "diagonal"):
            results = loo["gap_comparisons"][f"all/obj/{mode}"]
            denom = results["original_strict_pairs"]
            flips = {name: fold["strict_reversals"] for name, fold in results["deletions"].items()}
            covariance[mode] = {
                "pairs": results["pairs"],
                "original_strict_pairs": denom,
                "pairs_reversed_in_at_least_one_fold": results["reversed_under_any_deletion"],
                "reversals_by_fold": flips,
                "mean_fold_reversal_fraction": sum(flips.values()) / (len(flips) * denom),
            }
        comparisons = (
            ("saved_to_refined_primary", "actual_first_slot_top", "first_slot_top"),
            ("refined_primary_to_refined_full", "first_slot_top", "full_top_mean32768"),
            ("saved_to_refined_full", "actual_first_slot_top", "full_top_mean32768"),
        )
        gap_key = "fine_mean_full_value_gap_from_restricted_single"
        largest = max(rows, key=lambda row: row[gap_key])
        summed_gaps = math.fsum(row[gap_key] for row in rows)
        families[family] = {
            "posterior_source_deletion_cost_order": covariance,
            "cost_order_changes_from_offdiagonal_terms": loo["covariance_changes_ranking"]["all/obj"],
            "states": len(rows),
            "choice_disagreements": {name: sum(row[left] != row[right] for row in rows)
                                     for name, left, right in comparisons},
            "states_by_queries_retained": dict(sorted(Counter(row["retained_queries"] for row in rows).items())),
            "states_retaining_at_least_155_queries": sum(row["retained_queries"] >= 155 for row in rows),
            "largest_gain_state": largest["job_id"],
            "largest_full_over_primary_gain": largest[gap_key],
            "sum_full_over_primary_gains": summed_gaps,
            "largest_gain_share": largest[gap_key] / summed_gaps if summed_gaps else None,
        }
    return {
        "scope": "Rev2 frozen rosters; eligibility tested against evaluated policies and one queried policy",
        "scheduler_task_ids": task_ids,
        "simulator_calls": sum(row["simulator_calls"] for row in roster_rows),
        "source_refits": sum(row["source_refits"] for row in roster_rows),
        "worker_cpu_seconds": math.fsum(row["worker_cpu_seconds"] for row in roster_rows),
        "roster_membership_by_rounding_places": roster_summary,
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    scan = modes.add_parser("scan")
    scan.add_argument("--prefix", type=Path, required=True)
    scan.add_argument("--scores", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    scan.add_argument("--start", type=int, required=True)
    scan.add_argument("--end", type=int, required=True)
    scan.add_argument("--workers", type=int, required=True)
    aggregation = modes.add_parser("summarize")
    aggregation.add_argument("--roster-receipts", type=Path, required=True)
    aggregation.add_argument("--cascade-receipts", type=Path, required=True)
    aggregation.add_argument("--source-loo", type=Path, required=True)
    aggregation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "scan":
        args.output.mkdir(parents=True, exist_ok=True)
        entries = sorted(args.scores.glob("*.json"))[args.start:args.end]
        if len(entries) != args.end - args.start:
            raise ValueError("Scored prefix receipt set is incomplete")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(audit_state, [(path, args.prefix, args.output) for path in entries]))
        if not all(results):
            raise SystemExit("Audit shard has failed states; see its receipts")
    else:
        roster = read_json(args.roster_receipts)
        write_json(args.output, summarize(roster["receipts"], read_json(args.cascade_receipts),
                                          read_json(args.source_loo), roster["scheduler_task_ids"]))


if __name__ == "__main__":
    main()
