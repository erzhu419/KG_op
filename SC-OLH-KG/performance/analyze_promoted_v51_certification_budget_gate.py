#!/usr/bin/env python3
"""Analyze V51 certificate nonvacuity across target and replication budgets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
NEW_ONLY = "new_only"


def _parse_root(value):
    label, separator, raw = str(value).partition("=")
    if not separator or not raw:
        raise ValueError("roots must use N=PATH")
    budget = int(label)
    if budget <= 0:
        raise ValueError("registered budgets must be positive")
    return budget, Path(raw)


def _median(values):
    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite.append(value)
    return None if not finite else float(statistics.median(finite))


def _variant(experiment):
    parts = str(experiment).strip("/").split("/")
    for part in parts:
        if part == NEW_ONLY or part.startswith("joint_cap"):
            return part
    raise ValueError(f"missing registered certification variant in {experiment!r}")


def load_rows(roots):
    rows = []
    errors = []
    root_map = {}
    for budget, root in roots:
        root_map[str(budget)] = str(root)
        for path in sorted(root.rglob("result.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                experiment = payload.get("experiment_variant")
                for raw in payload["rows"]:
                    row = dict(raw)
                    recorded = int(row.get("N", budget))
                    if recorded != budget:
                        raise ValueError(
                            f"registered N={budget}, result records N={recorded}")
                    row["budget"] = budget
                    row["gate_variant"] = _variant(
                        row.get("experiment_variant") or experiment)
                    row["result_path"] = str(path)
                    rows.append(row)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
    return rows, errors, root_map


def _audit(row):
    return dict(row.get("certificate_outcome_audit") or {})


def _source_contract(row):
    training = dict((row.get("meta_prior") or {}).get("training") or {})
    transfer = dict(row.get("source_target_adaptation_contract") or {})
    return bool(
        int(training.get("source_archive_simulator_calls", -1)) == 384
        and int(transfer.get("source_simulator_calls", -1)) == 384
        and not bool(training.get("target_seed_used_for_source_training", True))
        and not bool(training.get("source_episode_target_oracle_used", True))
        and not bool(transfer.get("source_oracle_aided", True))
        and not bool(transfer.get("target_oracle_used_for_adaptation", True))
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _closure_contract(row):
    contract = dict(row.get("decision_backend_contract") or {})
    return bool(
        "observed_actions" in str(contract.get("terminal_value_contract", ""))
        and bool(contract.get("terminal_recommendation_observed_only", False))
        and bool(contract.get(
            "acquisition_and_recommendation_share_terminal_action_universe",
            False,
        ))
        and bool(contract.get(
            "acquisition_and_recommendation_share_risk_penalty", False))
        and bool(contract.get("coherent", False))
        and int(contract.get("forced_sampling_override_count", -1)) == 0
    )


def _initial_contract(rows):
    fingerprints = {}
    archives = {}
    for row in rows:
        design = dict(row.get("task_initial_design") or {})
        if (
            int(design.get("n_unique", -1)) != 10
            or bool(design.get("target_labels_used", True))
            or bool(design.get("target_oracle_used", True))
        ):
            return False
        key = str(row.get("heldout")), int(row.get("seed", -1))
        fingerprints.setdefault(key, set()).add(design.get("fingerprint"))
        archives.setdefault(key, set()).add(
            design.get("source_archive_fingerprint"))
    return bool(
        rows
        and all(None not in values and len(values) == 1
                for values in fingerprints.values())
        and all(None not in values and len(values) == 1
                for values in archives.values())
    )


def _outcome(row):
    feasible = bool(row.get("true_feasible", False))
    regret = row.get("feasible_simple_regret")
    if not feasible or regret is None:
        return feasible, math.inf
    return feasible, float(regret)


def _summarize(rows):
    audits = [_audit(row) for row in rows]
    certified = sum(int(audit.get("posterior_certified_count", 0) or 0)
                    for audit in audits)
    evaluated = sum(int(audit.get("evaluated_point_count", 0) or 0)
                    for audit in audits)
    true_feasible = sum(int(audit.get("true_feasible_count", 0) or 0)
                        for audit in audits)
    certified_true = sum(int(
        audit.get("certified_true_feasible_count", 0) or 0)
                         for audit in audits)
    false = sum(int(audit.get("false_certificate_count", 0) or 0)
                for audit in audits)
    regrets = [
        row.get("feasible_simple_regret") for row in rows
        if bool(row.get("true_feasible", False))
    ]
    return {
        "run_count": len(rows),
        "true_feasible_recommendation_count": sum(
            bool(row.get("true_feasible", False)) for row in rows),
        "median_feasible_regret": _median(regrets),
        "adaptive_improvement_count": sum(bool(row.get(
            "adaptive_improves_initial_best", False)) for row in rows),
        "adaptive_loss_count": sum(bool(row.get("adaptive_loss", False))
                                   for row in rows),
        "evaluated_point_count": evaluated,
        "posterior_certified_count": certified,
        "certified_true_feasible_count": certified_true,
        "true_feasible_evaluated_count": true_feasible,
        "false_certificate_count": false,
        "vacuous_run_count": sum(bool(audit.get(
            "posterior_certificate_vacuous", True)) for audit in audits),
        "certificate_coverage": (
            float(certified / evaluated) if evaluated else None),
        "certificate_precision": (
            float(certified_true / certified) if certified else None),
        "certificate_recall_on_evaluated_feasible": (
            float(certified_true / true_feasible) if true_feasible else None),
        "median_minimum_posterior_margin": _median(
            audit.get("minimum_posterior_margin") for audit in audits),
        "median_minimum_true_margin": _median(
            audit.get("minimum_true_margin") for audit in audits),
        "median_variance_log_rmse": _median(
            row.get("variance_log_rmse") for row in rows),
        "median_variance_upper_coverage": _median(
            row.get("variance_upper_coverage") for row in rows),
        "selected_new_point_count": sum(int(row.get(
            "adaptive_new_point_selected_count", 0) or 0) for row in rows),
        "selected_replication_count": sum(int(row.get(
            "adaptive_replication_selected_count", 0) or 0) for row in rows),
    }


def _paired_effect(rows, budget, challenger):
    selected = [row for row in rows if int(row["budget"]) == int(budget)]
    index = {}
    for row in selected:
        key = str(row.get("heldout")), int(row.get("seed", -1))
        index.setdefault(key, {})[row["gate_variant"]] = row
    pairs = [pair for pair in index.values()
             if NEW_ONLY in pair and challenger in pair]
    wins = losses = ties = 0
    for pair in pairs:
        left = _outcome(pair[NEW_ONLY])
        right = _outcome(pair[challenger])
        if right[0] and not left[0]:
            wins += 1
        elif left[0] and not right[0]:
            losses += 1
        elif right[1] < left[1] - 1e-12:
            wins += 1
        elif left[1] < right[1] - 1e-12:
            losses += 1
        else:
            ties += 1
    return {"pair_count": len(pairs), "wins": wins,
            "losses": losses, "ties": ties}


def analyze(roots, expected_count=None):
    rows, errors, root_map = load_rows(roots)
    budgets = sorted({int(row["budget"]) for row in rows})
    variants = sorted({str(row["gate_variant"]) for row in rows})
    grouped = {}
    for row in rows:
        key = int(row["budget"]), row["gate_variant"], row["heldout"]
        grouped.setdefault(key, []).append(row)
    cells = {
        f"N{budget}/{variant}/{domain}": _summarize(items)
        for (budget, variant, domain), items in sorted(grouped.items())
    }
    overall = {}
    for budget in budgets:
        for variant in variants:
            overall[f"N{budget}/{variant}"] = _summarize([
                row for row in rows
                if row["budget"] == budget
                and row["gate_variant"] == variant
            ])
    effects = {
        f"N{budget}/{variant}": _paired_effect(rows, budget, variant)
        for budget in budgets
        for variant in variants
        if variant != NEW_ONLY
    }
    maximum = max(budgets) if budgets else 0
    survivors = []
    survivor_checks = {}
    for variant in variants:
        if variant == NEW_ONLY:
            continue
        primary = overall.get(f"N{maximum}/{variant}", _summarize([]))
        control = overall.get(f"N{maximum}/{NEW_ONLY}", _summarize([]))
        effect = effects.get(f"N{maximum}/{variant}", {})
        domain_nonvacuous = all(
            cells.get(f"N{maximum}/{variant}/{domain}", {}).get(
                "certified_true_feasible_count", 0) > 0
            for domain in DOMAINS
        )
        certificate_counts = [
            overall.get(f"N{budget}/{variant}", {}).get(
                "posterior_certified_count", 0)
            for budget in budgets
        ]
        monotone = all(
            right >= left
            for left, right in zip(certificate_counts, certificate_counts[1:])
        )
        checks = {
            "complete_primary": bool(
                primary["run_count"]
                and primary["run_count"] == control["run_count"]),
            "positive_useful_coverage_in_every_domain": domain_nonvacuous,
            "zero_false_certificates": primary["false_certificate_count"] == 0,
            "recommendation_feasibility_noninferior": (
                primary["true_feasible_recommendation_count"]
                >= control["true_feasible_recommendation_count"]),
            "adaptive_loss_noninferior": (
                primary["adaptive_loss_count"]
                <= control["adaptive_loss_count"]),
            "paired_outcome_noninferior": (
                effect.get("pair_count", 0) == primary["run_count"]
                and effect.get("wins", 0) >= effect.get("losses", 0)),
            "certificate_count_monotone_in_budget": monotone,
        }
        survivor_checks[variant] = checks
        if all(checks.values()):
            survivors.append(variant)
    complete = bool(
        expected_count is None or len(rows) == int(expected_count))
    source_ok = bool(rows and all(_source_contract(row) for row in rows))
    closure_ok = bool(rows and all(_closure_contract(row) for row in rows))
    initial_ok = _initial_contract(rows)
    return {
        "scope": "promoted_v51_certification_budget_gate",
        "roots": root_map,
        "parsed_count": len(rows),
        "expected_count": expected_count,
        "errors": errors,
        "budgets": budgets,
        "variants": variants,
        "cells": cells,
        "overall": overall,
        "paired_effects": effects,
        "source_contract": source_ok,
        "closure_contract": closure_ok,
        "initial_pairing_contract": initial_ok,
        "survivor_checks": survivor_checks,
        "gate": {
            "complete_expected_matrix": bool(complete and not errors),
            "primary_budget": maximum,
            "survivors": survivors,
            "passes": bool(
                complete and not errors and source_ok and closure_ok
                and initial_ok and survivors),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(
        [_parse_root(value) for value in args.root],
        expected_count=args.expected_count,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
