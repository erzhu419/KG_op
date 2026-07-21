#!/usr/bin/env python3
"""Compare the observed-terminal closure challenger with promoted V51."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
CONTROL = "promoted_v51"
CHALLENGER = "observed_terminal_closure"


def _load_payloads(roots):
    if isinstance(roots, (str, Path)):
        roots = [roots]
    for root in roots:
        for path in sorted(Path(root).rglob("result.json")):
            yield path, json.loads(path.read_text())


def load_rows(control_roots, challenger_roots):
    rows = []
    for variant, roots in (
        (CONTROL, control_roots),
        (CHALLENGER, challenger_roots),
    ):
        for path, payload in _load_payloads(roots):
            experiment = str(payload.get("experiment_variant", ""))
            if variant == CONTROL and "/balanced4/" not in f"/{experiment}/":
                continue
            if variant == CHALLENGER and (
                "/observed_terminal_closure/" not in f"/{experiment}/"
            ):
                continue
            for raw in payload.get("rows", []):
                row = dict(raw)
                row["gate_variant"] = variant
                row["result_path"] = str(path)
                rows.append(row)
    return rows


def _key(row):
    return str(row.get("heldout")), int(row.get("seed", -1))


def _outcome(row):
    feasible = bool(row.get("true_feasible", False))
    regret = row.get("feasible_simple_regret")
    return feasible, (
        float(regret)
        if feasible and regret is not None and np.isfinite(float(regret))
        else np.inf
    )


def _summary(rows, variant, seeds):
    selected = [
        row for row in rows
        if row.get("gate_variant") == variant
        and row.get("heldout") in DOMAINS
        and int(row.get("seed", -1)) in seeds
    ]
    regrets = [
        float(row["feasible_simple_regret"])
        for row in selected
        if row.get("true_feasible", False)
        and row.get("feasible_simple_regret") is not None
    ]
    audits = [dict(row.get("certificate_outcome_audit") or {})
              for row in selected]
    return {
        "count": len(selected),
        "complete": len(selected) == len(DOMAINS) * len(seeds),
        "true_feasible": sum(bool(row.get("true_feasible", False))
                             for row in selected),
        "adaptive_improvements": sum(bool(row.get(
            "adaptive_improves_initial_best", False)) for row in selected),
        "adaptive_losses": sum(bool(row.get("adaptive_loss", False))
                               for row in selected),
        "false_certificates": sum(int(audit.get(
            "false_certificate_count", 0) or 0) for audit in audits),
        "certified_points": sum(int(audit.get(
            "posterior_certified_count", 0) or 0) for audit in audits),
        "vacuous_runs": sum(bool(audit.get(
            "posterior_certificate_vacuous", False)) for audit in audits),
        "selected_new_points": sum(int(row.get(
            "adaptive_new_point_selected_count", 0) or 0)
            for row in selected),
        "selected_replications": sum(int(row.get(
            "adaptive_replication_selected_count", 0) or 0)
            for row in selected),
        "median_feasible_regret": (
            float(np.median(regrets)) if regrets else None),
        "by_domain": {
            domain: {
                "true_feasible": sum(bool(row.get("true_feasible", False))
                                     for row in selected
                                     if row.get("heldout") == domain),
                "adaptive_losses": sum(bool(row.get("adaptive_loss", False))
                                       for row in selected
                                       if row.get("heldout") == domain),
            }
            for domain in DOMAINS
        },
    }


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
    identifier = str(contract.get("terminal_value_contract", ""))
    return bool(
        row.get("decision_backend") == "sobol_exact_joint_voi"
        and "observed_actions" in identifier
        and bool(contract.get("acquisition_terminal_observed_only", False))
        and bool(contract.get(
            "acquisition_and_recommendation_share_terminal_action_universe",
            False,
        ))
        and bool(contract.get(
            "acquisition_and_recommendation_share_risk_penalty", False))
        and bool(contract.get("coherent", False))
        and int(contract.get("forced_sampling_override_count", -1)) == 0
    )


def _paired_initial_contract(control, challenger):
    left = dict(control.get("task_initial_design") or {})
    right = dict(challenger.get("task_initial_design") or {})
    return bool(
        left.get("fingerprint") is not None
        and left.get("fingerprint") == right.get("fingerprint")
        and left.get("source_archive_fingerprint") is not None
        and left.get("source_archive_fingerprint")
        == right.get("source_archive_fingerprint")
        and int(left.get("n_unique", -1)) == 10
        and int(right.get("n_unique", -1)) == 10
        and not bool(right.get("target_labels_used", True))
        and not bool(right.get("target_oracle_used", True))
    )


def summarize(rows, seeds=range(20)):
    seeds = tuple(int(seed) for seed in seeds)
    selected = [
        row for row in rows
        if row.get("heldout") in DOMAINS
        and int(row.get("seed", -1)) in seeds
    ]
    by_variant = {
        variant: {_key(row): row for row in selected
                  if row.get("gate_variant") == variant}
        for variant in (CONTROL, CHALLENGER)
    }
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    paired = expected & set(by_variant[CONTROL]) & set(by_variant[CHALLENGER])
    wins = losses = ties = 0
    initial_contract = True
    for key in paired:
        control = by_variant[CONTROL][key]
        challenger = by_variant[CHALLENGER][key]
        initial_contract &= _paired_initial_contract(control, challenger)
        c_feasible, c_regret = _outcome(challenger)
        b_feasible, b_regret = _outcome(control)
        if c_feasible and not b_feasible:
            wins += 1
        elif b_feasible and not c_feasible:
            losses += 1
        elif c_regret < b_regret - 1e-12:
            wins += 1
        elif b_regret < c_regret - 1e-12:
            losses += 1
        else:
            ties += 1
    summaries = {
        variant: _summary(selected, variant, seeds)
        for variant in (CONTROL, CHALLENGER)
    }
    challenger_rows = list(by_variant[CHALLENGER].values())
    source_ok = bool(
        len(challenger_rows) == len(expected)
        and all(_source_contract(row) for row in challenger_rows))
    closure_ok = bool(
        len(challenger_rows) == len(expected)
        and all(_closure_contract(row) for row in challenger_rows))
    control = summaries[CONTROL]
    challenger = summaries[CHALLENGER]
    domainwise_safe = all(
        challenger["by_domain"][domain]["true_feasible"]
        >= control["by_domain"][domain]["true_feasible"]
        and challenger["by_domain"][domain]["adaptive_losses"]
        <= control["by_domain"][domain]["adaptive_losses"]
        for domain in DOMAINS
    )
    passes = bool(
        len(paired) == len(expected)
        and all(summary["complete"] for summary in summaries.values())
        and source_ok
        and closure_ok
        and initial_contract
        and domainwise_safe
        and challenger["false_certificates"] == 0
        and challenger["true_feasible"] >= control["true_feasible"]
        and challenger["adaptive_losses"] <= control["adaptive_losses"]
        and wins > losses
    )
    return {
        "scope": "promoted_v51_observed_terminal_closure",
        "seeds": list(seeds),
        "paired_count": len(paired),
        "source_contract": source_ok,
        "closure_contract": closure_ok,
        "initial_pairing_contract": bool(initial_contract),
        "summaries": summaries,
        "paired": {"wins": wins, "losses": losses, "ties": ties},
        "domainwise_safety_noninferior": bool(domainwise_safe),
        "gate_passes": passes,
        "promotion_eligible": [CHALLENGER] if passes else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, action="append", required=True)
    parser.add_argument("--challenger-root", type=Path, action="append", required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    result = summarize(
        load_rows(args.control_root, args.challenger_root), seeds=seeds)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()

