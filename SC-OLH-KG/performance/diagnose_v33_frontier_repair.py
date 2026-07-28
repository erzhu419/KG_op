#!/usr/bin/env python3
"""Truth-only post-hoc diagnosis for the V33 frontier repair matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_quality import json_safe  # noqa: E402
from performance.diagnose_v33_inventory_terminal import (  # noqa: E402
    _truth_evaluator,
)


VARIANTS = (
    "v32",
    "v33_legacy_4",
    "v33_coherent_coverage_4",
    "v33_coherent_coverage_8",
)
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def _read(root, variant, domain, seed):
    path = Path(root) / variant / domain / f"seed{int(seed)}" / "result.json"
    payload = json.loads(path.read_text())
    rows = payload.get("rows") or []
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0], payload.get("config") or {}, path


def _seed_audit(row, truth):
    finalist = row.get("finalist_replication") or {}
    targets = finalist.get("targets") or []
    labels = finalist.get("labels") or []
    target_truth = [truth(target) for target in targets]
    terminal_selected = []
    for terminal in finalist.get("terminal_kg_rows") or []:
        arms = terminal.get("terminal_kg_arms") or []
        selected = int(terminal.get("terminal_kg_selected_index", 0))
        if not arms or not (0 <= selected < len(arms)):
            continue
        terminal_selected.append(truth(arms[selected]))
    recommendation = truth(row["x_recommended"])
    safe_labels = [
        str(label)
        for label, value in zip(labels, target_truth)
        if value["feasible"]
    ]
    target_set_has_safe = any(
        value["feasible"] for value in target_truth)
    terminal_selected_safe = any(
        value["feasible"] for value in terminal_selected)
    return {
        "recommendation": recommendation,
        "posterior_feasible": bool(row.get("posterior_feasible", False)),
        "posterior_theory_chance_margin": row.get(
            "posterior_theory_chance_margin"),
        "target_count": int(len(target_truth)),
        "true_feasible_target_count": int(sum(
            value["feasible"] for value in target_truth)),
        "target_set_has_true_feasible": bool(target_set_has_safe),
        "safe_target_labels": safe_labels,
        "terminal_action_count": int(len(terminal_selected)),
        "true_feasible_terminal_action_count": int(sum(
            value["feasible"] for value in terminal_selected)),
        "terminal_selected_any_true_feasible": bool(
            terminal_selected_safe),
        "support_failure": bool(
            not recommendation["feasible"] and not target_set_has_safe),
        "terminal_misranking": bool(
            not recommendation["feasible"]
            and target_set_has_safe
            and terminal_selected
            and not terminal_selected_safe),
        "final_ranking_failure": bool(
            not recommendation["feasible"] and target_set_has_safe),
        "certificate_vacuity": bool(
            not row.get("posterior_feasible", False)),
        "target_oracle_used": bool(
            finalist.get("target_oracle_used", False)),
    }


def _group_summary(variant, domain, audits):
    margins = [
        float(audit["posterior_theory_chance_margin"])
        for audit in audits
        if audit["posterior_theory_chance_margin"] is not None
    ]
    return {
        "variant": variant,
        "domain": domain,
        "n_seeds": int(len(audits)),
        "true_feasible_recommendations": int(sum(
            audit["recommendation"]["feasible"] for audit in audits)),
        "posterior_certified_recommendations": int(sum(
            audit["posterior_feasible"] for audit in audits)),
        "target_sets_with_true_feasible": int(sum(
            audit["target_set_has_true_feasible"] for audit in audits)),
        "true_feasible_targets": int(sum(
            audit["true_feasible_target_count"] for audit in audits)),
        "terminal_selected_true_feasible": int(sum(
            audit["true_feasible_terminal_action_count"] for audit in audits)),
        "terminal_actions": int(sum(
            audit["terminal_action_count"] for audit in audits)),
        "support_failure_count": int(sum(
            audit["support_failure"] for audit in audits)),
        "terminal_misranking_count": int(sum(
            audit["terminal_misranking"] for audit in audits)),
        "final_ranking_failure_count": int(sum(
            audit["final_ranking_failure"] for audit in audits)),
        "certificate_vacuity_count": int(sum(
            audit["certificate_vacuity"] for audit in audits)),
        "median_posterior_theory_margin": (
            None if not margins else float(np.median(margins))),
        "target_oracle_used_count": int(sum(
            audit["target_oracle_used"] for audit in audits)),
    }


def diagnose(root, variants=VARIANTS, domains=DOMAINS, seeds=range(7)):
    variants = tuple(str(value) for value in variants)
    domains = tuple(str(value) for value in domains)
    seeds = tuple(int(value) for value in seeds)
    per_seed = []
    summaries = []
    for domain in domains:
        _, config, _ = _read(root, variants[0], domain, seeds[0])
        truth = _truth_evaluator(config, domain)
        for variant in variants:
            audits = []
            for seed in seeds:
                row, _, path = _read(root, variant, domain, seed)
                audit = _seed_audit(row, truth)
                audit.update({
                    "variant": variant,
                    "domain": domain,
                    "seed": int(seed),
                    "result_path": str(path),
                })
                audits.append(audit)
                per_seed.append(audit)
            summaries.append(_group_summary(variant, domain, audits))
    return {
        "schema_version": 1,
        "posthoc_truth_audit": True,
        "truth_used_by_optimizer": False,
        "run_root": str(Path(root)),
        "variants": list(variants),
        "domains": list(domains),
        "seeds": list(seeds),
        "summaries": summaries,
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--seeds", default=",".join(map(str, range(7))))
    args = parser.parse_args()
    payload = diagnose(
        args.root,
        variants=args.variants.split(","),
        domains=args.domains.split(","),
        seeds=[int(value) for value in args.seeds.split(",")],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(payload), indent=2) + "\n")
    print(json.dumps({
        "posthoc_truth_audit": True,
        "truth_used_by_optimizer": False,
        "n_summaries": len(payload["summaries"]),
        "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
