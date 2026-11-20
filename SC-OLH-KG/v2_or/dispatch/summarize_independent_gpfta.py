"""Produce the paired task readout for the frozen independent GP-FTA panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from math import sqrt
from statistics import mean, stdev

from scipy.stats import t

from v2_or.dispatch.campaign_runner import _holm_adjust


REFERENCE = "GP-FTA-KG"
COMPARATORS = ("GP-FTA-TS", "PublicPsi-TargetGP-TS")


def paired_interval(differences: list[float]) -> dict:
    center = mean(differences)
    degrees = len(differences) - 1
    standard_error = stdev(differences) / sqrt(len(differences))
    radius = float(t.ppf(0.975, degrees)) * standard_error
    p_value = (
        2.0 * float(t.sf(abs(center) / standard_error, degrees))
        if standard_error > 0.0
        else float(center == 0.0)
    )
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    return {
        "n_tasks": len(differences),
        "comparator_minus_kg_mean": center,
        "ci95": [center - radius, center + radius],
        "paired_t_p_two_sided": p_value,
        "kg_wins_losses_ties": [wins, losses, len(differences) - wins - losses],
    }


def readout(campaign_dir: Path, manifest: dict) -> dict:
    csv_path = campaign_dir / "per_run_ledger.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {}
    for row in rows:
        key = (row["family"], int(row["target_task_idx"]), int(row["rep_seed"]), row["method_name"])
        if key in index:
            raise ValueError(f"Duplicate paired cell {key}")
        index[key] = row

    result = {
        "campaign_name": manifest["campaign_name"],
        "primary_measure": "Status-quo loss, where lower is better",
        "interval": "Family-level paired t interval, equal weight per task, individual 95% coverage",
        "families": {},
    }
    for family in manifest["families"]:
        name = family["name"]
        tasks = range(int(family["target_config"]["n_tasks"]))
        seeds = family["target_eval_seeds"]
        online_steps, = family["online_step_budgets"]
        search = (family["n0_init"] + online_steps) * family["batch_reps"]
        groups = defaultdict(list)
        for task in tasks:
            for seed in seeds:
                for arm in (REFERENCE, *COMPARATORS):
                    row = index.pop((name, task, seed, arm))
                    groups[arm].append(row)

        arm_summary = {}
        task_loss = {}
        for arm, arm_rows in groups.items():
            search_obs = [int(r["search_calls_used"]) for r in arm_rows]
            terminal = [int(r["verification_calls_used"]) for r in arm_rows]
            target = [int(r["total_target_calls"]) for r in arm_rows]
            if set(search_obs) != {search} or target != [search + n for n in terminal]:
                raise ValueError(f"Call ledger differs from the protocol for {name}/{arm}")
            if max(terminal) > 380 or min(terminal) < 0:
                raise ValueError(f"Terminal cap violated for {name}/{arm}")
            task_loss[arm] = [
                mean(float(row["status_quo_loss"]) for row in arm_rows if int(row["target_task_idx"]) == i)
                for i in tasks
            ]
            audit_counts = Counter(row["safety_audit_class"] for row in arm_rows)
            audit_probabilities = [
                float(row["deployed_mc_viol_prob"]) for row in arm_rows if row["deployed_mc_viol_prob"]
            ]
            arm_summary[arm] = {
                "runs": len(arm_rows),
                "certified": sum(row["certified"].lower() in {"true", "1"} for row in arm_rows),
                "safety_audit_classes": dict(audit_counts),
                "audited_deployments": len(audit_probabilities),
                "audit_viol_prob_point_estimates_over_threshold": sum(p > 0.05 for p in audit_probabilities),
                "mean_status_quo_loss": mean(task_loss[arm]),
                "mean_status_quo_loss_with_abstention_surcharge": mean(
                    float(row["fixed_reject_loss_10pct"]) for row in arm_rows
                ),
                "search_calls_per_run": search,
                "mean_verification_calls": mean(terminal),
                "sum_target_calls": sum(target),
                "mean_total_target_calls": mean(target),
                "max_total_target_calls": max(target),
                "sum_policy_audit_calls": sum(int(row["audit_calls_used"]) for row in arm_rows),
                "mean_online_wall_seconds": mean(float(row["online_wall_time_sec"]) for row in arm_rows),
            }

        paired = {
            comparator: paired_interval([b - a for a, b in zip(task_loss[REFERENCE], task_loss[comparator])])
            for comparator in COMPARATORS
        }
        result["families"][name] = {
            "independent_tasks": len(tasks),
            "streams_per_task": len(seeds),
            "arm_results": arm_summary,
            "paired_loss": paired,
        }

    if index:
        raise ValueError(f"{len(index)} cells are outside the frozen paired roster")
    tests = {
        f"{family}/{arm}": entry
        for family, group in result["families"].items()
        for arm, entry in group["paired_loss"].items()
    }
    adjusted = _holm_adjust({key: entry["paired_t_p_two_sided"] for key, entry in tests.items()})
    for key, value in adjusted.items():
        tests[key]["paired_t_p_holm_six_tests"] = value
    result["total_cells"] = len(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    completion = args.campaign_dir / "stages/audit_and_summarize/completion.json"
    if not completion.is_file():
        raise RuntimeError("Finish the independent audit stage before reading the paired results")
    result = readout(args.campaign_dir, manifest)
    sealed = json.loads(completion.read_text())
    summary = sealed["campaign_summary"]
    totals = summary["call_accounting"]
    arms = [m for family in result["families"].values() for m in family["arm_results"].values()]
    assert sum(m["sum_target_calls"] for m in arms) == (
        totals["search_calls_used"] + totals["terminal_verification_calls_used"]
    )
    assert sum(m["sum_policy_audit_calls"] for m in arms) == totals["policy_audit_calls_used"]
    result.update(
        execution_code_sha256=sealed["execution_code_sha256"],
        manifest_sha256=sealed["manifest_sha256"],
        call_accounting=totals,
        stage_attempt_fees=summary["stage_attempt_fees"],
    )
    output = args.campaign_dir / "independent_panel_readout.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cells": result["total_cells"], "families": list(result["families"]), "result": str(output)}))


if __name__ == "__main__":
    main()
