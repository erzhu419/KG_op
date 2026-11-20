"""Read the matched GP-FTA study without rerunning a simulation."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from v2_or.dispatch.campaign_runner import _audit_anchor_id, _holm_adjust, build_family_tasks
from v2_or.dispatch.summarize_independent_gpfta import paired_interval


WEAK = "TargetWeak-GateKG"
DIAGONAL = "HistoryDiag-GateKG"
FULL = "HistoryFull-GateKG"
FEASIBLE = "HistoryFull-FeasibleKG"
EARLY = "InitialRelease-GateKG"
ARMS = (WEAK, DIAGONAL, FULL, FEASIBLE)
CONTRASTS = ((WEAK, DIAGONAL), (DIAGONAL, FULL), (FEASIBLE, FULL), (FULL, EARLY), (WEAK, EARLY))
SAFE = "confidently_safe"
UNSAFE = "confidently_unsafe"


def _certified(row: dict) -> bool:
    return row["certified"].lower() == "true"


def compact_policy_diagnostics(campaign_dir: Path, manifest: dict, index: dict) -> dict:
    """Retain audit labels and pairwise policy indicators, without publicizing vectors."""
    result = {}
    for family in manifest["families"]:
        name = family["name"]
        tasks = build_family_tasks(family, "target")
        anchors, pairs = {}, {}
        for i, task in enumerate(tasks):
            anchor_name = _audit_anchor_id(name, i)
            receipt = campaign_dir / f"stages/audit_anchors/items/{anchor_name}.json"
            audit = json.loads(receipt.read_text())["anchor_audit"]
            anchors[str(i)] = {
                key: audit[key] for key in ("mc_objective_mean", "mc_violation_prob", "mc_violation_count",
                                           "mc_viol_ci95_lo", "mc_viol_ci95_hi",
                                           "safety_audit_class", "n_audit_samples")
            }
            anchor = task.domain_heuristic_anchor()
            for seed in family["target_eval_seeds"]:
                receipts = [
                    json.loads((campaign_dir / "stages/search_and_verify/items" /
                                (index[name, i, seed, arm]["job_id"] + ".json")).read_text())
                    for arm in (DIAGONAL, FULL)
                ]
                deployed = [
                    r["deployed_policy"] if r["deployed_policy"] is not None else anchor
                    for r in receipts
                ]
                pairs[f"{i}/{seed}"] = {
                    "primary_changed": not np.allclose(
                        receipts[0]["primary_policy"], receipts[1]["primary_policy"], rtol=0, atol=1e-10
                    ),
                    "operating_policy_changed": not np.allclose(*deployed, rtol=0, atol=1e-10),
                }
        result[name] = {"anchor_audits": anchors, "full_vs_diagonal_policy_changes": pairs}
    return result


def summarize(campaign_dir: Path, manifest: dict) -> dict:
    with (campaign_dir / "per_run_ledger.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {
        (r["family"], int(r["target_task_idx"]), int(r["rep_seed"]), r["method_name"]): r
        for r in rows
    }
    if len(rows) != len(index):
        raise ValueError("Duplicate paired cells in the ledger")
    cache = campaign_dir / "operating_policy_diagnostics.json"
    if cache.is_file():
        diagnostics = json.loads(cache.read_text())
    else:
        diagnostics = compact_policy_diagnostics(campaign_dir, manifest, index)
        cache.write_text(json.dumps(diagnostics, indent=2) + "\n")

    families, all_tests = {}, {}
    used = 0
    for fam in manifest["families"]:
        name = fam["name"]
        n = fam["target_config"]["n_tasks"]
        seeds = fam["target_eval_seeds"]
        target_cap = fam["target_call_cap"]
        n_steps, = fam["online_step_budgets"]
        search_cap = (n_steps + fam["n0_init"]) * fam["batch_reps"]
        anchor = diagnostics[name]["anchor_audits"]
        arm_rows = {arm: [] for arm in (*ARMS, EARLY)}
        choices = Counter()
        differential = Counter()
        log_ratios = []
        for i in range(n):
            for seed in seeds:
                paired = {arm: index[name, i, seed, arm] for arm in ARMS}
                used += len(paired)
                for arm, row in paired.items():
                    terminal = int(row["verification_calls_used"])
                    total = int(row["total_target_calls"])
                    if int(row["search_calls_used"]) != search_cap or total != search_cap + terminal or total > target_cap:
                        raise ValueError(f"Target call contract failed for {row['job_id']}")
                    arm_rows[arm].append(row)
                weak = json.loads(paired[WEAK]["mechanism_receipt_json"])["early_release"]
                full = json.loads(paired[FULL]["mechanism_receipt_json"])["early_release"]
                if {k:v for k,v in weak.items() if k != "applied_weak"} != {k:v for k,v in full.items() if k != "applied_weak"}:
                    raise ValueError(f"Initial forecast evidence differs between paired arms: {name}/{i}/{seed}")
                pick = WEAK if full["recommend_weak"] else FULL
                choices[pick] += 1
                log_ratios.append(float(full["log_ratio_weak_over_full"]))
                arm_rows[EARLY].append(paired[pick])

                changed = diagnostics[name]["full_vs_diagonal_policy_changes"][f"{i}/{seed}"]
                differential.update({key: int(value) for key, value in changed.items()})
                first = [json.loads(paired[arm]["mechanism_receipt_json"]).get("first_query_index") for arm in (DIAGONAL,FULL)]
                if None not in first:
                    differential["first_query_changed"] += int(first[0] != first[1])
                diag_row, full_row = paired[DIAGONAL], paired[FULL]
                a = [r["safety_audit_class"] if _certified(r) else anchor[str(i)]["safety_audit_class"]
                     for r in (diag_row, full_row)]
                if a == [UNSAFE, SAFE]:
                    differential["unsafe_to_safe"] += 1
                elif a == [SAFE, UNSAFE]:
                    differential["safe_to_unsafe"] += 1
                elif a == [SAFE, SAFE]:
                    difference = float(diag_row["status_quo_loss"]) - float(full_row["status_quo_loss"])
                    differential["both_safe_full_lower_loss"] += int(difference > 1e-12)
                    differential["both_safe_diagonal_lower_loss"] += int(difference < -1e-12)

        task_losses, summaries = {}, {}
        for arm, selected in arm_rows.items():
            safety = Counter()
            certified_audits = Counter()
            per_task = [[] for _ in range(n)]
            risk, visits = [], []
            for row in selected:
                i = int(row["target_task_idx"])
                per_task[i].append(float(row["status_quo_loss"]))
                if _certified(row):
                    category = row["safety_audit_class"]
                    certified_audits[category] += 1
                    risk.append(float(row["deployed_mc_viol_prob"]))
                else:
                    category = anchor[str(i)]["safety_audit_class"]
                    risk.append(anchor[str(i)]["mc_violation_prob"])
                safety[category] += 1
                visits.append(int(row["n_replication_queries"]))
            task_losses[arm] = [mean(x) for x in per_task]
            mechanisms = [json.loads(row["mechanism_receipt_json"])["variance"] for row in selected]
            variance = {}
            for role in ("primary", "deployed"):
                audited = [r[role]["squared_error_vs_audit"] for r in mechanisms
                           if r.get(role) and "squared_error_vs_audit" in r[role]]
                variance[role] = {
                    "n": len(audited),
                    "mean_squared_errors": {
                        estimator: {head: mean(r[estimator][head] for r in audited)
                                    for head in ("obj_var", "con_var")}
                        for estimator in ("fixed_batch", "pooled")
                    } if audited else {},
                }
            summaries[arm] = {
                "runs": len(selected),
                "certified": sum(certified_audits.values()),
                "certified_audits": dict(certified_audits),
                "operating_audits": dict(safety),
                "mean_operating_violation_rate": mean(risk),
                "mean_operating_loss": mean(task_losses[arm]),
                "mean_search_calls": mean(int(r["search_calls_used"]) for r in selected),
                "mean_terminal_calls": mean(int(r["verification_calls_used"]) for r in selected),
                "mean_total_target_calls": mean(int(r["total_target_calls"]) for r in selected),
                "revisit_count": sum(visits),
                "revisit_share": sum(visits) / (len(selected) * n_steps),
                "variance_diagnostics": variance,
            }

        tests = {}
        for a, b in CONTRASTS:
            key = f"{a} to {b}"
            stats = paired_interval([x-y for x,y in zip(task_losses[a], task_losses[b])])
            tests[key] = dict(improvement=stats["comparator_minus_kg_mean"],
                              ci95=stats["ci95"], latter_wins_losses_ties=stats["kg_wins_losses_ties"],
                              p=stats["paired_t_p_two_sided"])
            all_tests[f"{name}/{key}"] = tests[key]
        families[name] = {
            "tasks": n, "streams_per_task": len(seeds), "arms": summaries, "comparisons": tests,
            "early_choices": dict(choices), "full_vs_diagonal": dict(differential),
            "initial_log_ratio_range": [min(log_ratios), max(log_ratios)],
        }

    if used != len(rows):
        raise ValueError("Ledger contains cells outside the registered roster")
    for key, adj in _holm_adjust({k: v["p"] for k,v in all_tests.items()}).items():
        all_tests[key]["p_holm"] = adj
    completion = json.loads((campaign_dir / "stages/audit_and_summarize/completion.json").read_text())
    accounting = completion["campaign_summary"]
    return {
        "campaign_name": manifest["campaign_name"], "paid_cells": used,
        "exact_early_policy_replays": sum(v["arms"][EARLY]["runs"] for v in families.values()),
        "families": families, "call_accounting": accounting["call_accounting"],
        "stage_attempt_fees": accounting["stage_attempt_fees"],
        "execution_code_sha256": completion["execution_code_sha256"],
        "manifest_sha256": completion["manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = summarize(args.campaign_dir, manifest)
    output = args.campaign_dir / "referee_ablation_readout.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "paid_cells": report["paid_cells"]}))


if __name__ == "__main__":
    main()
