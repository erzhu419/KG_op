#!/usr/bin/env python3
"""Analyze the V51 balanced evaluate-or-replicate action-set gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ("balanced4", "balanced8", "new8_only")
SCENARIO_SEEDS = {
    "FactorShockStatePolicyRZDT1": 0,
    "InventorySupplyChain": 0,
    "QueueResourceControl": 3,
}
EXPECTED_NEW = {"balanced4": 4, "balanced8": 8, "new8_only": 8}
EXPECTED_REPLICATION = {
    "balanced4": True, "balanced8": True, "new8_only": False}


def load_rows(root, variants=VARIANTS):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        experiment = str(payload.get("experiment_variant", ""))
        variant = next(
            (name for name in variants if f"/{name}/" in f"/{experiment}/"),
            None,
        )
        if variant is None:
            continue
        for raw in payload.get("rows", []):
            row = dict(raw)
            row["gate_variant"] = variant
            rows.append(row)
    return rows


def _in_scope(row):
    scenario = str(row.get("heldout", ""))
    return bool(
        scenario in SCENARIO_SEEDS
        and int(row.get("seed", -1)) == SCENARIO_SEEDS[scenario]
    )


def _source_contract(row):
    training = dict((row.get("meta_prior") or {}).get("training") or {})
    contract = dict(row.get("source_target_adaptation_contract") or {})
    return bool(
        int(training.get("source_base_domain_count", -1)) == 2
        and int(training.get("source_episode_count_per_base_domain", -1)) == 1
        and int(training.get("source_archive_simulator_calls", -1)) == 384
        and not bool(training.get("target_seed_used_for_source_training", True))
        and not bool(training.get("source_episode_target_oracle_used", True))
        and int(contract.get("source_simulator_calls", -1)) == 384
        and not bool(contract.get("source_oracle_aided", True))
        and not bool(contract.get("target_oracle_used_for_adaptation", True))
    )


def _mode_contract(row, variant):
    decision_contract = dict(row.get("decision_backend_contract") or {})
    new_count = decision_contract.get(
        "evaluate_or_replicate_new_action_count",
        row.get("evaluate_or_replicate_new_action_count"),
    )
    new_policy = decision_contract.get(
        "evaluate_or_replicate_new_action_policy",
        row.get("evaluate_or_replicate_new_action_policy"),
    )
    return bool(
        row.get("decision_backend") == "sobol_exact_joint_voi"
        and row.get("decision_aleatoric_mode") == "posterior_central"
        and row.get("decision_ambiguity_mode") == "posterior_nominal"
        and row.get("decision_violation_loss_mode") == "positive_part"
        and new_count == EXPECTED_NEW[variant]
        and new_policy == "canonical_plus_posterior_risk"
        and bool(row.get("adaptive_replication_voi_enabled", False))
        == EXPECTED_REPLICATION[variant]
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _action_trace_contract(row, variant):
    requested = EXPECTED_NEW[variant]
    allow_replication = EXPECTED_REPLICATION[variant]
    trace = list(row.get("online_action_trace") or [])
    if len(trace) != 10:
        return False
    for action in trace:
        new_count = int(action.get("active_new_action_count") or 0)
        rep_count = int(action.get("active_replication_action_count") or 0)
        scores = action.get("exact_kg_raw_scores_active")
        flags = action.get("exact_kg_active_action_is_replicate")
        if (
            action.get("new_action_policy")
            != "canonical_plus_posterior_risk"
            or new_count != requested
            or (not allow_replication and rep_count != 0)
            or scores is None
            or flags is None
            or len(scores) != new_count + rep_count
            or len(flags) != len(scores)
            or sum(not bool(flag) for flag in flags) != new_count
            or sum(bool(flag) for flag in flags) != rep_count
        ):
            return False
        scores = np.asarray(scores, dtype=float)
        flags = np.asarray(flags, dtype=bool)
        best_new = float(np.max(scores[~flags]))
        logged_new = action.get("exact_kg_best_new_raw")
        if logged_new is None or not np.isclose(best_new, float(logged_new)):
            return False
        if rep_count:
            best_rep = float(np.max(scores[flags]))
            logged_rep = action.get("exact_kg_best_replication_raw")
            logged_delta = action.get("exact_kg_new_minus_replication_raw")
            if (
                logged_rep is None
                or logged_delta is None
                or not np.isclose(best_rep, float(logged_rep))
                or not np.isclose(
                    best_new - best_rep, float(logged_delta))
            ):
                return False
            if (
                best_new > best_rep + 1e-12
                and action.get("action_kind") != "new"
            ) or (
                best_rep > best_new + 1e-12
                and action.get("action_kind") != "replicate"
            ):
                return False
        elif action.get("action_kind") != "new":
            return False
    return True


def _summary(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    regrets = np.asarray([
        np.nan if row.get("feasible_simple_regret") is None
        else float(row["feasible_simple_regret"])
        for row in selected
    ], dtype=float)
    regrets = regrets[np.isfinite(regrets)]
    deltas = []
    new_arm_wins = 0
    replicate_arm_wins = 0
    for row in selected:
        for action in row.get("online_action_trace") or []:
            delta = action.get("exact_kg_new_minus_replication_raw")
            if delta is None:
                continue
            deltas.append(float(delta))
            if float(delta) > 1e-12:
                new_arm_wins += 1
            else:
                replicate_arm_wins += 1
    return {
        "count": len(selected),
        "complete": len(selected) == len(SCENARIO_SEEDS),
        "true_feasible": sum(bool(row.get("true_feasible", False))
                             for row in selected),
        "adaptive_losses": sum(bool(row.get("adaptive_loss", False))
                               for row in selected),
        "adaptive_improvements": sum(bool(row.get(
            "adaptive_improves_initial_best", False)) for row in selected),
        "false_certificates": sum(int((
            row.get("certificate_outcome_audit") or {}
        ).get("false_certificate_count", 0) or 0) for row in selected),
        "selected_replications": sum(int(row.get(
            "adaptive_replication_selected_count", 0) or 0)
            for row in selected),
        "selected_new_points": sum(int(row.get(
            "adaptive_new_point_selected_count", 0) or 0)
            for row in selected),
        "new_arm_wins": int(new_arm_wins),
        "replicate_arm_wins": int(replicate_arm_wins),
        "median_new_minus_replication_raw": (
            float(np.median(deltas)) if deltas else None),
        "median_feasible_regret": (
            float(np.median(regrets)) if len(regrets) else None),
        "by_domain": {
            str(row.get("heldout")): {
                "true_feasible": bool(row.get("true_feasible", False)),
                "adaptive_loss": bool(row.get("adaptive_loss", False)),
                "adaptive_improvement": bool(row.get(
                    "adaptive_improves_initial_best", False)),
                "selected_replications": int(row.get(
                    "adaptive_replication_selected_count", 0) or 0),
                "selected_new_points": int(row.get(
                    "adaptive_new_point_selected_count", 0) or 0),
                "feasible_simple_regret": row.get("feasible_simple_regret"),
            }
            for row in selected
        },
    }


def _load_control(root):
    rows = load_rows(root, variants=("exact_nominal_positive",))
    selected = [row for row in rows if _in_scope(row)]
    for row in selected:
        row["gate_variant"] = "control"
    return _summary(selected, "control")


def summarize(rows, control):
    rows = [row for row in rows if _in_scope(row)]
    summaries = {variant: _summary(rows, variant) for variant in VARIANTS}
    source_contract = {
        variant: summaries[variant]["complete"] and all(
            _source_contract(row) for row in rows
            if row.get("gate_variant") == variant)
        for variant in VARIANTS
    }
    mode_contract = {
        variant: summaries[variant]["complete"] and all(
            _mode_contract(row, variant) for row in rows
            if row.get("gate_variant") == variant)
        for variant in VARIANTS
    }
    action_trace_contract = {
        variant: summaries[variant]["complete"] and all(
            _action_trace_contract(row, variant) for row in rows
            if row.get("gate_variant") == variant)
        for variant in VARIANTS
    }
    strict_candidates = []
    for variant in VARIANTS:
        challenger = summaries[variant]
        no_safety_regression = bool(
            challenger["true_feasible"] >= control["true_feasible"]
            and challenger["adaptive_losses"] <= control["adaptive_losses"]
            and challenger["false_certificates"] == 0
        )
        outcome_improvement = bool(
            challenger["adaptive_improvements"]
            > control["adaptive_improvements"]
            or (
                challenger["median_feasible_regret"] is not None
                and control["median_feasible_regret"] is not None
                and challenger["median_feasible_regret"]
                < control["median_feasible_regret"] - 1e-12
            )
        )
        common_domains = set(challenger["by_domain"]) & set(
            control["by_domain"])
        domainwise_regret_noninferior = bool(common_domains) and all(
            challenger["by_domain"][domain]["feasible_simple_regret"]
            is not None
            and control["by_domain"][domain]["feasible_simple_regret"]
            is not None
            and float(challenger["by_domain"][domain][
                "feasible_simple_regret"])
            <= float(control["by_domain"][domain][
                "feasible_simple_regret"]) + 1e-12
            for domain in common_domains
        )
        if (
            source_contract[variant]
            and mode_contract[variant]
            and action_trace_contract[variant]
            and no_safety_regression
            and outcome_improvement
            and domainwise_regret_noninferior
        ):
            strict_candidates.append(variant)
    return {
        "scope": "balanced_action_set_three_domain_sentinel",
        "scenario_seeds": SCENARIO_SEEDS,
        "fixed_source_call_budget_per_variant": 384,
        "source_contract": source_contract,
        "mode_contract": mode_contract,
        "action_trace_contract": action_trace_contract,
        "v50_exact_nominal_positive_control": control,
        "variant_summaries": summaries,
        "full_gate_warranted": bool(strict_candidates),
        "full_gate_candidates": strict_candidates,
        "promotion_eligible": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(load_rows(args.root), _load_control(args.control_root))
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
