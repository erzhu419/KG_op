#!/usr/bin/env python3
"""Analyze the V49 decision-HVD and terminal-loss factorial sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = (
    "sobol_upper_positive",
    "sobol_central_positive",
    "sobol_upper_probability",
    "sobol_central_probability",
    "exact_upper_positive",
    "exact_central_positive",
    "exact_upper_probability",
    "exact_central_probability",
)
SCENARIO_SEEDS = {
    "FactorShockStatePolicyRZDT1": 0,
    "InventorySupplyChain": 0,
    "QueueResourceControl": 3,
}
EXPECTED_MODES = {
    variant: {
        "backend": (
            "sobol_exact_joint_voi" if variant.startswith("exact_")
            else "sobol_new"),
        "aleatoric": (
            "posterior_central" if "_central_" in variant
            else "certification_upper"),
        "loss": (
            "failure_probability" if variant.endswith("_probability")
            else "positive_part"),
    }
    for variant in VARIANTS
}


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        experiment = str(payload.get("experiment_variant", ""))
        variant = next(
            (name for name in VARIANTS if f"/{name}/" in f"/{experiment}/"),
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
    return (
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
        and contract.get("source_constraint_mean_adaptation_mode")
        == "sequential_aggregate_hyperlaw"
        and not bool(contract.get("source_oracle_aided", True))
        and not bool(contract.get("target_oracle_used_for_adaptation", True))
    )


def _mode_contract(row, variant):
    expected = EXPECTED_MODES[variant]
    audit = dict(row.get("terminal_bayes_pool_audit") or {})
    return bool(
        row.get("decision_backend") == expected["backend"]
        and row.get("decision_aleatoric_mode") == expected["aleatoric"]
        and row.get("decision_violation_loss_mode") == expected["loss"]
        and row.get("decision_backend_terminal_aleatoric_mode")
        == expected["aleatoric"]
        and row.get("decision_backend_terminal_violation_loss_mode")
        == expected["loss"]
        and audit.get("status") == "ranked"
        and audit.get("decision_aleatoric_mode") == expected["aleatoric"]
        and audit.get("violation_loss_mode") == expected["loss"]
        and not bool(audit.get("target_oracle_used_for_ranking", True))
        and not bool(audit.get("target_oracle_used_for_decision", True))
        and not bool(audit.get("truth_admissible_decision_input", True))
        and audit.get("truth_join_timing") == "post_terminal_rank"
    )


def _paired_static_contract(rows):
    static = tuple(v for v in VARIANTS if v.startswith("sobol_"))
    for scenario, seed in SCENARIO_SEEDS.items():
        group = {
            row.get("gate_variant"): row for row in rows
            if row.get("heldout") == scenario
            and int(row.get("seed", -1)) == seed
            and row.get("gate_variant") in static
        }
        if set(group) != set(static):
            return False
        for field in (
            "target_design_fingerprint",
            "online_action_sequence_fingerprint",
        ):
            values = {str(row.get(field, "missing")) for row in group.values()}
            if len(values) != 1 or "missing" in values:
                return False
        traces = [list(row.get("online_action_trace") or [])
                  for row in group.values()]
        reference = traces[0] if traces else []
        if not reference:
            return False
        for trace in traces:
            if len(trace) != len(reference):
                return False
            if any(
                left.get("x_fingerprint") != right.get("x_fingerprint")
                or left.get("observed_response") != right.get("observed_response")
                for left, right in zip(reference, trace)
            ):
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


def summarize(rows):
    rows = [row for row in rows if _in_scope(row)]
    summaries = {variant: _summary(rows, variant) for variant in VARIANTS}
    source_contract = {
        variant: all(_source_contract(row) for row in rows
                     if row.get("gate_variant") == variant)
        and summaries[variant]["complete"]
        for variant in VARIANTS
    }
    mode_contract = {
        variant: all(_mode_contract(row, variant) for row in rows
                     if row.get("gate_variant") == variant)
        and summaries[variant]["complete"]
        for variant in VARIANTS
    }
    control = summaries["exact_upper_positive"]
    challenger = summaries["exact_central_probability"]
    static_control = summaries["sobol_upper_positive"]
    static_challenger = summaries["sobol_central_probability"]
    fewer_repeats = (
        challenger["selected_replications"]
        < control["selected_replications"])
    more_new = (
        challenger["selected_new_points"] > control["selected_new_points"])
    no_safety_regression = bool(
        challenger["true_feasible"] >= control["true_feasible"]
        and challenger["adaptive_losses"] <= control["adaptive_losses"]
        and challenger["false_certificates"] == 0
    )
    terminal_improvement = bool(
        static_challenger["true_feasible"] > static_control["true_feasible"]
        or static_challenger["adaptive_losses"]
        < static_control["adaptive_losses"]
        or static_challenger["adaptive_improvements"]
        > static_control["adaptive_improvements"]
    )
    full_gate_warranted = bool(all((
        all(summary["complete"] for summary in summaries.values()),
        all(source_contract.values()),
        all(mode_contract.values()),
        _paired_static_contract(rows),
        fewer_repeats,
        more_new,
        no_safety_regression,
        terminal_improvement,
    )))
    return {
        "scope": "factorial_three_domain_sentinel",
        "scenario_seeds": SCENARIO_SEEDS,
        "fixed_source_call_budget_per_variant": 384,
        "source_contract": source_contract,
        "mode_contract": mode_contract,
        "paired_static_actions_and_responses": _paired_static_contract(rows),
        "combined_exact_fewer_replications_than_current": bool(fewer_repeats),
        "combined_exact_more_new_points_than_current": bool(more_new),
        "combined_exact_no_safety_regression": bool(no_safety_regression),
        "combined_static_terminal_improvement": bool(terminal_improvement),
        "variant_summaries": summaries,
        "full_gate_warranted": bool(full_gate_warranted),
        "promotion_eligible": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(load_rows(args.root))
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
