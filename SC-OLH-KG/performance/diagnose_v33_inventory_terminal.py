#!/usr/bin/env python3
"""Post-hoc truth audit for V33 terminal policies on Inventory.

This script never feeds truth back into an optimizer.  It reads completed
artifacts and uses the synthetic oracle only to classify failure mechanisms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


DEFAULT_VARIANTS = ("v32", "terminal_kg_1step", "terminal_kg_depth3")


def _finite_summary(values):
    values = np.asarray([
        float(value) for value in values if value is not None
    ], dtype=float)
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _load_row(root, variant, domain, seed):
    path = Path(root) / variant / domain / f"seed{int(seed)}" / "result.json"
    payload = json.loads(path.read_text())
    rows = payload.get("rows") or []
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0], payload.get("config") or {}, path


def _truth_evaluator(config, domain):
    base = make_problem(
        domain,
        d=int(config.get("d", 50)),
        L=int(config.get("L", 100)),
        sigma=float(config.get("sigma", 0.04)),
        alpha=float(config.get("alpha", 0.05)),
    )
    weights = tuple(float(v) for v in str(
        config.get("weights", "0.5,0.5")).split(","))
    problem = ScalarizedProblem(base, weights=weights)
    z_alpha = float(norm.ppf(1.0 - problem.alpha))

    def evaluate(x):
        x = tuple(int(v) for v in x)
        margin = (
            problem.true_constraint_mean(x)
            + z_alpha * float(problem.true_sigma(x)[1])
            - float(problem.tau)
        )
        return {
            "objective": float(problem.true_objective(x)),
            "chance_margin": float(margin),
            "feasible": bool(margin <= 0.0),
        }

    return evaluate


def _variant_seed_audit(row, truth):
    recommendation = truth(row["x_recommended"])
    finalist = row.get("finalist_replication") or {}
    targets = finalist.get("targets") or []
    labels = finalist.get("labels") or []
    statistics = finalist.get("statistics") or []
    replicate_counts = finalist.get("replicate_counts") or []
    target_rows = []
    for index, target in enumerate(targets):
        target_truth = truth(target)
        statistic = statistics[index] if index < len(statistics) else None
        target_rows.append({
            "index": int(index),
            "label": labels[index] if index < len(labels) else None,
            "replicate_count": int(
                replicate_counts[index]
                if index < len(replicate_counts) else 0),
            "truth": target_truth,
            "empirical_upper_chance_margin": (
                None if statistic is None else
                statistic.get("upper_chance_margin")),
            "empirical_constraint_mean": (
                None if statistic is None else
                statistic.get("constraint_mean")),
        })

    terminal_rows = []
    for stage_index, terminal in enumerate(
        finalist.get("terminal_kg_rows") or []
    ):
        selected_index = int(terminal["terminal_kg_selected_index"])
        arms = terminal.get("terminal_kg_arms") or []
        arm_truth = [truth(arm) for arm in arms]
        terminal_rows.append({
            "stage_index": int(stage_index),
            "depth": int(terminal.get("terminal_kg_depth", 0)),
            "selected_index": selected_index,
            "selected_label": (
                labels[selected_index]
                if selected_index < len(labels) else None),
            "selected_gain": float(
                terminal.get("terminal_kg_selected_gain", 0.0)),
            "selected_truth": arm_truth[selected_index],
            "true_feasible_arm_count": int(sum(
                item["feasible"] for item in arm_truth)),
            "arm_count": int(len(arms)),
        })

    pool = row.get("truth_pool_diagnostics") or {}
    target_tuples = {tuple(int(v) for v in target) for target in targets}
    return {
        "recommendation": recommendation,
        "recommendation_in_terminal_arm_set": bool(
            tuple(int(v) for v in row["x_recommended"]) in target_tuples),
        "posterior_feasible": bool(row.get("posterior_feasible", False)),
        "posterior_theory_chance_margin": row.get(
            "posterior_theory_chance_margin"),
        "replicated_finalist_used": bool(
            row.get("replicated_finalist_used", False)),
        "replicated_finalist_reason": row.get("replicated_finalist_reason"),
        "targets": target_rows,
        "terminal_decisions": terminal_rows,
        "pool_has_true_feasible_rate": pool.get(
            "pool_has_true_feasible_rate"),
        "best_true_feasible_posterior_feasible_rate": pool.get(
            "best_true_feasible_posterior_feasible_rate"),
        "mean_best_true_feasible_posterior_margin": pool.get(
            "mean_best_true_feasible_posterior_margin"),
        "missed_true_feasible_rate": pool.get("missed_true_feasible_rate"),
    }


def diagnose(root, domain="InventorySupplyChain", seeds=range(7)):
    seeds = [int(seed) for seed in seeds]
    first_row, config, _ = _load_row(
        root, DEFAULT_VARIANTS[0], domain, seeds[0])
    del first_row
    truth = _truth_evaluator(config, domain)
    per_seed = []
    by_variant = {variant: [] for variant in DEFAULT_VARIANTS}
    for seed in seeds:
        seed_row = {"seed": int(seed), "variants": {}}
        for variant in DEFAULT_VARIANTS:
            row, _, path = _load_row(root, variant, domain, seed)
            audit = _variant_seed_audit(row, truth)
            audit["result_path"] = str(path)
            seed_row["variants"][variant] = audit
            by_variant[variant].append(audit)
        per_seed.append(seed_row)

    aggregates = {}
    for variant, rows in by_variant.items():
        targets = [target for row in rows for target in row["targets"]]
        selected = [
            decision
            for row in rows
            for decision in row["terminal_decisions"]
        ]
        true_feasible_upper = [
            target["empirical_upper_chance_margin"]
            for target in targets
            if target["truth"]["feasible"]
        ]
        minimum_upper_target_is_feasible = []
        for row in rows:
            measured = [
                target for target in row["targets"]
                if target["empirical_upper_chance_margin"] is not None
            ]
            if measured:
                selected_target = min(
                    measured,
                    key=lambda target: float(
                        target["empirical_upper_chance_margin"]),
                )
                minimum_upper_target_is_feasible.append(
                    bool(selected_target["truth"]["feasible"]))
        aggregates[variant] = {
            "true_feasible_recommendations": int(sum(
                row["recommendation"]["feasible"] for row in rows)),
            "posterior_certified_recommendations": int(sum(
                row["posterior_feasible"] for row in rows)),
            "replicated_finalist_used_count": int(sum(
                row["replicated_finalist_used"] for row in rows)),
            "mean_pool_has_true_feasible_rate": float(np.mean([
                row["pool_has_true_feasible_rate"] for row in rows
                if row["pool_has_true_feasible_rate"] is not None
            ])),
            "mean_best_true_feasible_posterior_feasible_rate": float(np.mean([
                row["best_true_feasible_posterior_feasible_rate"]
                for row in rows
                if row["best_true_feasible_posterior_feasible_rate"] is not None
            ])),
            "mean_missed_true_feasible_rate": float(np.mean([
                row["missed_true_feasible_rate"] for row in rows
                if row["missed_true_feasible_rate"] is not None
            ])),
            "true_feasible_terminal_targets": int(sum(
                target["truth"]["feasible"] for target in targets)),
            "terminal_targets": int(len(targets)),
            "terminal_target_sets_with_true_feasible": int(sum(
                any(target["truth"]["feasible"] for target in row["targets"])
                for row in rows
            )),
            "minimum_empirical_upper_target_true_feasible": int(sum(
                minimum_upper_target_is_feasible)),
            "selected_true_feasible_terminal_actions": int(sum(
                decision["selected_truth"]["feasible"]
                for decision in selected)),
            "terminal_actions": int(len(selected)),
            "true_feasible_target_empirical_upper_margin": _finite_summary(
                true_feasible_upper),
        }

    baseline = by_variant["v32"]
    challenger = by_variant["terminal_kg_1step"]
    baseline_rescue_seeds = [
        seeds[index]
        for index, (base, new) in enumerate(zip(baseline, challenger))
        if (
            base["recommendation"]["feasible"]
            and not new["recommendation"]["feasible"]
            and base["replicated_finalist_used"]
        )
    ]
    lost_feasible_seeds = [
        seeds[index]
        for index, (base, new) in enumerate(zip(baseline, challenger))
        if (
            base["recommendation"]["feasible"]
            and not new["recommendation"]["feasible"]
        )
    ]
    return {
        "schema_version": 1,
        "posthoc_truth_audit": True,
        "truth_used_by_optimizer": False,
        "run_root": str(Path(root)),
        "domain": str(domain),
        "seeds": seeds,
        "aggregates": aggregates,
        "paired_failure": {
            "lost_feasible_seeds": lost_feasible_seeds,
            "v32_uncertified_override_rescue_seeds": baseline_rescue_seeds,
        },
        "mechanism": {
            "candidate_pool_support_failure": bool(
                aggregates["terminal_kg_1step"][
                    "mean_pool_has_true_feasible_rate"] < 0.5),
            "terminal_frontier_support_failure": bool(
                aggregates["terminal_kg_1step"][
                    "terminal_target_sets_with_true_feasible"] < len(seeds)),
            "certificate_vacuity": bool(
                aggregates["terminal_kg_1step"][
                    "mean_best_true_feasible_posterior_feasible_rate"] == 0.0),
            "terminal_action_misranking": bool(
                aggregates["terminal_kg_1step"][
                    "selected_true_feasible_terminal_actions"]
                < aggregates["terminal_kg_1step"]["terminal_actions"] / 2.0),
            "uncertified_override_gap": bool(baseline_rescue_seeds),
        },
        "per_seed": per_seed,
    }


def _parse_seeds(value):
    return [int(token) for token in str(value).split(",") if token.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--domain", default="InventorySupplyChain")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = diagnose(
        args.root,
        domain=args.domain,
        seeds=_parse_seeds(args.seeds),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
