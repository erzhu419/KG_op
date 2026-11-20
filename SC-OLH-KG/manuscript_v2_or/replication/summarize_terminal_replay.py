#!/usr/bin/env python3
"""Replay the saved first slot and completed terminal tree without simulator calls."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr

from summarize_operating_policy import read_csv, verify_terminal
from v2_or.dispatch.summarize_gpfta_referee import ARMS, DIAGONAL, FEASIBLE, FULL, WEAK
from v2_or.dispatch.summarize_independent_gpfta import paired_interval

RunKey = tuple[int, int]


@dataclass(frozen=True)
class Replay:
    first_passed: bool
    first_loss: float
    first_calls: int
    full_passed: bool
    full_loss: float
    full_calls: int
    guarded_override: bool
    anchor_loss: float
    release: bool
    log_ratio_weak_over_full: float


def read_run(row: dict, slots: list[dict]) -> Replay:
    slots = sorted(slots, key=lambda cert: int(cert["slot_index"]))
    passing = verify_terminal(row, slots)
    first_passed = 0 in passing
    full_passed = bool(passing)
    anchor_loss = float(row["anchor_mc_obj"])
    first_loss = float(row["primary_slot0_mc_obj"]) if first_passed else anchor_loss
    full_loss = float(row["status_quo_loss"])
    assert full_loss == (float(row["certified_deployed_mc_obj"]) if full_passed else anchor_loss)
    evidence = json.loads(row["mechanism_receipt_json"])["early_release"]
    release = bool(evidence["recommend_weak"])
    log_ratio = float(evidence["log_ratio_weak_over_full"])
    assert release == (log_ratio > evidence["threshold"])
    return Replay(
        first_passed, first_loss, int(slots[0]["n_samples"]),
        full_passed, full_loss, int(row["verification_calls_used"]),
        first_passed and row["selected_slot"] != "0", anchor_loss, release, log_ratio,
    )


def task_means(values: dict[RunKey, float]) -> dict[int, float]:
    by_task = defaultdict(list)
    for (task, _), value in values.items():
        by_task[task].append(value)
    return {task: mean(streams) for task, streams in by_task.items()}


def task_interval(values: dict[RunKey, float]) -> dict:
    result = paired_interval(list(task_means(values).values()))
    return {
        "tasks": result["n_tasks"],
        "mean_loss_difference": result["comparator_minus_kg_mean"],
        "unadjusted_ci95": result["ci95"],
        "unadjusted_p": result["paired_t_p_two_sided"],
    }


def summarize_arm(runs: dict[RunKey, Replay]) -> dict:
    def gate(prefix: str) -> dict:
        return {
            "mean_loss": mean(getattr(r, f"{prefix}_loss") for r in runs.values()),
            "certified": sum(getattr(r, f"{prefix}_passed") for r in runs.values()),
            "mean_terminal_calls": mean(getattr(r, f"{prefix}_calls") for r in runs.values()),
        }

    rescued = [r for r in runs.values() if not r.first_passed and r.full_passed]
    overrides = [r for r in runs.values() if r.guarded_override]
    return {
        "runs": len(runs),
        "first_slot": gate("first"),
        "full_terminal": gate("full"),
        "rescued_after_first_failure": len(rescued),
        "guarded_override_after_first_pass": len(overrides),
        "full_minus_first_contributions": {
            "rescued_after_first_failure": sum(r.full_loss - r.first_loss for r in rescued) / len(runs),
            "guarded_override_after_first_pass": sum(r.full_loss - r.first_loss for r in overrides) / len(runs),
        },
        "full_minus_first": task_interval({key: r.full_loss - r.first_loss for key, r in runs.items()}),
    }


def compare(lhs: dict[RunKey, Replay], rhs: dict[RunKey, Replay]) -> dict:
    first_delta, full_delta = {}, {}
    groups = {name: [] for name in ("both_certified", "left_only_certified", "right_only_certified", "both_abstained")}
    assert lhs.keys() == rhs.keys()
    for key, a in lhs.items():
        b = rhs[key]
        assert a.anchor_loss == b.anchor_loss
        first_delta[key] = a.first_loss - b.first_loss
        full_delta[key] = a.full_loss - b.full_loss
        group = {
            (True, True): "both_certified", (True, False): "left_only_certified",
            (False, True): "right_only_certified", (False, False): "both_abstained",
        }[a.full_passed, b.full_passed]
        groups[group].append(full_delta[key])
    decomposition = {
        key: {
            "runs": len(items),
            "conditional_mean_left_minus_right": mean(items) if items else None,
            "contribution_to_overall_mean": sum(items) / len(full_delta),
        }
        for key, items in groups.items()
    }
    discordant = groups["left_only_certified"] + groups["right_only_certified"]
    discordant_contribution = sum(discordant) / len(full_delta)
    full_interval = task_interval(full_delta)
    full_mean = full_interval["mean_loss_difference"]
    return {
        "first_slot_left_minus_right": task_interval(first_delta),
        "full_terminal_left_minus_right": full_interval,
        "full_minus_first_contrast_change": task_interval({k: d - first_delta[k] for k, d in full_delta.items()}),
        "full_terminal_by_certification": decomposition,
        "discordant_certifications": {
            "runs": len(discordant),
            "contribution_to_overall_mean": discordant_contribution,
            "share_of_observed_difference": discordant_contribution / full_mean if full_mean else None,
        },
    }


def release_summary(weak: dict[RunKey, Replay], full: dict[RunKey, Replay]) -> dict:
    groups = {"release_to_weak": [], "retain_full": []}
    for key, h in full.items():
        w = weak[key]
        assert w.release == h.release and w.log_ratio_weak_over_full == h.log_ratio_weak_over_full
        groups["release_to_weak" if h.release else "retain_full"].append(w.full_loss - h.full_loss)
    status_summary = {
        status: {
            "runs": len(values),
            "weak_lower_loss": sum(x < 0 for x in values),
            "weak_higher_loss": sum(x > 0 for x in values),
            "ties": sum(x == 0 for x in values),
            "mean_weak_minus_full_loss": mean(values) if values else None,
        }
        for status, values in groups.items()
    }
    task_score = task_means({key: h.log_ratio_weak_over_full for key, h in full.items()})
    task_loss_diff = task_means({key: h.full_loss - weak[key].full_loss for key, h in full.items()})
    score = spearmanr(list(task_score.values()), [task_loss_diff[t] for t in task_score]).statistic
    status_summary["descriptive_task_rank_correlation"] = {
        "tasks": len(task_score),
        "score": "Initial weak minus full constraint log predictive density",
        "outcome": "Full minus weak operating loss under complete terminal verification",
        "spearman_rho": float(score),
    }
    return status_summary


def summarize(campaign: Path) -> dict:
    manifest = json.loads((campaign / "sealed_manifest.json").read_text())
    identity = json.loads((campaign / "compact_summary.json").read_text())
    rows = read_csv(campaign / "per_run_ledger.csv")
    slots = read_csv(campaign / "per_slot_certificates.csv")
    by_job = defaultdict(list)
    for cert in slots:
        by_job[cert["job_id"]].append(cert)
    assert len(by_job) == len(rows) == len({r["job_id"] for r in rows})
    expected = {
        (family["name"], task, stream, arm)
        for family in manifest["families"]
        for task in range(family["target_config"]["n_tasks"])
        for stream in family["target_eval_seeds"]
        for arm in ARMS
    }
    cells = {}
    for row in rows:
        assert all(row[key] == identity[key] for key in ("execution_code_sha256", "manifest_sha256"))
        key = (row["family"], int(row["target_task_idx"]), int(row["rep_seed"]), row["method_name"])
        assert key not in cells
        certs = by_job[row["job_id"]]
        assert all(
            all(cert[k] == row[k] for k in ("family", "target_task_idx", "rep_seed", "method_name"))
            for cert in certs
        )
        cells[key] = read_run(row, certs)
    assert cells.keys() == expected
    families = {}
    for family in manifest["families"]:
        name = family["name"]
        arms = {
            arm: {(task, stream): v for (fam, task, stream, method), v in cells.items() if fam == name and method == arm}
            for arm in ARMS
        }
        families[name] = {
            "arms": {arm: summarize_arm(runs) for arm, runs in arms.items()},
            "gate_minus_feasible": compare(arms[FULL], arms[FEASIBLE]),
            "full_minus_diagonal": compare(arms[FULL], arms[DIAGONAL]),
            "full_minus_weak": compare(arms[FULL], arms[WEAK]),
            "early_release": release_summary(arms[WEAK], arms[FULL]),
        }
    return {
        "campaign": manifest["campaign_name"],
        "scope": "Descriptive replay of fixed search, observed first-slot rules, and actual full terminal outcomes.",
        "independent_unit": "Target task, after averaging its two streams. Intervals are ordinary 95% paired t intervals.",
        "per_run_rows_checked": len(rows),
        "slot_rows_checked": len(slots),
        "new_simulator_calls": 0,
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.campaign), indent=2))


if __name__ == "__main__":
    main()
