#!/usr/bin/env python3
"""Audit V56 independent pilot/confirmation sentinels."""

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
CONTROL = "v51_control"
V56 = ("v56_confirm2048", "v56_confirm4096")
KNOWN = (CONTROL, *V56)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _median(values):
    values = [value for value in (_finite(item) for item in values)
              if value is not None]
    return None if not values else float(statistics.median(values))


def _variant(experiment):
    marker = f"/{str(experiment).strip('/')}/"
    return next((name for name in KNOWN if f"/{name}/" in marker), None)


def load_rows(root):
    rows = []
    errors = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            variant = _variant(payload.get("experiment_variant", ""))
            if variant is None:
                continue
            for raw in payload["rows"]:
                row = dict(raw)
                row["gate_variant"] = variant
                row["result_path"] = str(path)
                rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return rows, errors


def _key(row):
    return str(row.get("heldout")), int(row.get("seed", -1))


def _confirmations(row):
    confirmations = []
    for trace in list(row.get("online_action_trace") or []):
        item = trace.get("policy_improvement_confirmation")
        if item is not None:
            confirmations.append(dict(item))
    return confirmations


def _policy_audits(row):
    return [
        dict(trace.get("policy_improvement_pairwise_audit") or {})
        for trace in list(row.get("online_action_trace") or [])
    ]


def _recommendation(row):
    value = row.get("x_recommended")
    if isinstance(value, list):
        return tuple(value)
    return value


def _paired_performance(control_rows, challenger_rows):
    control = {_key(row): row for row in control_rows}
    challenger = {_key(row): row for row in challenger_rows}
    keys = sorted(set(control) & set(challenger))
    wins = losses = ties = 0
    feasibility_rescues = feasibility_losses = 0
    recommendation_changes = 0
    for key in keys:
        baseline = control[key]
        trial = challenger[key]
        baseline_feasible = bool(baseline.get("true_feasible", False))
        trial_feasible = bool(trial.get("true_feasible", False))
        if trial_feasible and not baseline_feasible:
            feasibility_rescues += 1
            wins += 1
        elif baseline_feasible and not trial_feasible:
            feasibility_losses += 1
            losses += 1
        elif not baseline_feasible and not trial_feasible:
            ties += 1
        else:
            baseline_regret = _finite(baseline.get("feasible_simple_regret"))
            trial_regret = _finite(trial.get("feasible_simple_regret"))
            if baseline_regret is None or trial_regret is None:
                ties += 1
            elif trial_regret < baseline_regret - 1e-12:
                wins += 1
            elif trial_regret > baseline_regret + 1e-12:
                losses += 1
            else:
                ties += 1
        recommendation_changes += int(
            _recommendation(baseline) != _recommendation(trial))
    return {
        "paired_count": len(keys),
        "win_count": wins,
        "loss_count": losses,
        "tie_count": ties,
        "feasibility_rescue_count": feasibility_rescues,
        "feasibility_loss_count": feasibility_losses,
        "recommendation_change_count": recommendation_changes,
        "all_recommendations_identical": bool(
            keys and recommendation_changes == 0),
        "performance_noninferior": bool(keys and losses == 0),
        "strict_gain_detected": bool(wins > 0),
    }


def _contract(row, variant):
    if variant == CONTROL:
        return bool(
            str(row.get("implementation_contract_id"))
            == "promoted_v51_observed_terminal_closure"
            and str(row.get("theory_contract_id"))
            == "v51_statistical_closure_v2"
        )
    contract = dict(row.get("decision_backend_contract") or {})
    confirmations = _confirmations(row)
    expected = 2048 if variant.endswith("2048") else 4096
    return bool(
        str(row.get("implementation_contract_id"))
        == "v56_independent_confirmation_guard"
        and str(row.get("theory_contract_id"))
        == "v56_independent_confirmation_finite_look_v1"
        and str(contract.get("policy_improvement_guard_mode"))
        == "independent_confirmation"
        and str(contract.get("policy_improvement_score_transform"))
        == "bounded_current_gain"
        and int(contract.get(
            "policy_improvement_confirmation_samples", -1)) == expected
        and bool(confirmations)
        and all(item.get("pilot_stream_independent") is True
                for item in confirmations)
        and all(item.get("simulation_stream_independent") is True
                for item in confirmations)
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _summary(rows):
    confirmations = [item for row in rows for item in _confirmations(row)]
    policy_audits = [item for row in rows for item in _policy_audits(row)]
    audits = [dict(row.get("certificate_outcome_audit") or {})
              for row in rows]
    feasible_regret = [
        row.get("feasible_simple_regret") for row in rows
        if bool(row.get("true_feasible", False))
    ]
    return {
        "run_count": len(rows),
        "true_feasible_count": sum(bool(row.get("true_feasible", False))
                                   for row in rows),
        "median_feasible_regret": _median(feasible_regret),
        "adaptive_improvement_count": sum(bool(row.get(
            "adaptive_improves_initial_best", False)) for row in rows),
        "adaptive_loss_count": sum(bool(row.get("adaptive_loss", False))
                                   for row in rows),
        "false_certificate_count": sum(int(item.get(
            "false_certificate_count", 0) or 0) for item in audits),
        "posterior_certified_count": sum(int(item.get(
            "posterior_certified_count", 0) or 0) for item in audits),
        "posterior_certificate_vacuous_count": sum(bool(item.get(
            "posterior_certificate_vacuous", False)) for item in audits),
        "confirmation_decision_count": len(confirmations),
        "joint_confirmation_pass_count": sum(bool(item.get("passed", False))
                                             for item in confirmations),
        "confirmation_switch_count": sum(bool(item.get("switched", False))
                                         for item in policy_audits),
        "confirmation_fallback_count": sum(
            not bool(item.get("passed", False)) for item in confirmations),
        "median_confirmation_samples": _median(
            item.get("sample_count") for item in confirmations),
        "median_risk_first_crossing": _median(
            item.get("risk_first_crossing_sample") for item in confirmations),
        "median_certificate_first_crossing": _median(
            item.get("certificate_first_crossing_sample")
            for item in confirmations),
        "median_risk_gain": _median(
            item.get("risk_sample_mean") for item in confirmations),
        "median_certificate_gain": _median(
            item.get("certificate_sample_mean") for item in confirmations),
        "median_confirmation_time_sec": _median(
            item.get("time_sec") for item in confirmations),
        "median_initialization_time_sec": _median(
            row.get("initialization_time_sec") for row in rows),
        "median_finalization_time_sec": _median(
            row.get("finalization_time_sec") for row in rows),
        "median_algorithm_time_sec": _median(
            row.get("algorithm_time_sec") for row in rows),
    }


def analyze(root, challengers=V56):
    challengers = tuple(challengers)
    if not challengers or any(item not in V56 for item in challengers):
        raise ValueError(f"challengers must be a nonempty subset of {V56}")
    variants = (CONTROL, *challengers)
    rows, errors = load_rows(root)
    rows = [row for row in rows if row["gate_variant"] in variants]
    grouped = {
        variant: {
            domain: [row for row in rows if row["gate_variant"] == variant
                     and str(row.get("heldout")) == domain]
            for domain in DOMAINS
        }
        for variant in variants
    }
    summaries = {
        variant: {
            domain: _summary(domain_rows)
            for domain, domain_rows in domains.items()
        }
        for variant, domains in grouped.items()
    }
    contracts = {
        variant: all(_contract(row, variant)
                     for domain_rows in domains.values()
                     for row in domain_rows)
        for variant, domains in grouped.items()
    }
    expected_keys = {
        variant: {_key(row) for domain_rows in domains.values()
                  for row in domain_rows}
        for variant, domains in grouped.items()
    }
    paired = all(expected_keys[variant] == expected_keys[CONTROL]
                 for variant in challengers)
    complete = all(len(grouped[variant][domain]) == 5
                   for variant in variants for domain in DOMAINS)
    confirmation_nonvacuous = {
        variant: sum(
            summaries[variant][domain]["joint_confirmation_pass_count"]
            for domain in DOMAINS
        ) > 0
        for variant in challengers
    }
    no_false_certificates = all(
        summaries[variant][domain]["false_certificate_count"] == 0
        for variant in variants for domain in DOMAINS
    )
    chance_certificate_nonvacuous = {
        variant: sum(
            summaries[variant][domain]["posterior_certified_count"]
            for domain in DOMAINS
        ) > 0
        for variant in variants
    }
    paired_performance = {
        variant: _paired_performance(
            [row for domain_rows in grouped[CONTROL].values()
             for row in domain_rows],
            [row for domain_rows in grouped[variant].values()
             for row in domain_rows],
        )
        for variant in challengers
    }
    formal_gate_by_variant = {
        variant: bool(
            complete
            and not errors
            and paired
            and contracts[CONTROL]
            and contracts[variant]
            and confirmation_nonvacuous[variant]
            and no_false_certificates
        )
        for variant in challengers
    }
    promotion_gate_by_variant = {
        variant: bool(
            formal_gate_by_variant[variant]
            and paired_performance[variant]["performance_noninferior"]
            and paired_performance[variant]["strict_gain_detected"]
        )
        for variant in challengers
    }
    formal_gate_passed = all(formal_gate_by_variant.values())
    promotion_gate_passed = any(promotion_gate_by_variant.values())
    return {
        "scope": "v56_independent_confirmation_sentinel",
        "challengers": list(challengers),
        "root": str(Path(root)),
        "row_count": len(rows),
        "load_errors": errors,
        "summaries": summaries,
        "contract_valid": contracts,
        "paired_keys_match_control": paired,
        "complete_5_seed_matrix": complete,
        "confirmation_nonvacuous": confirmation_nonvacuous,
        "chance_certificate_nonvacuous": chance_certificate_nonvacuous,
        "no_false_certificates": no_false_certificates,
        "paired_performance": paired_performance,
        "formal_gate_by_variant": formal_gate_by_variant,
        "formal_gate_passed": formal_gate_passed,
        "promotion_gate_by_variant": promotion_gate_by_variant,
        "promotion_gate_passed": promotion_gate_passed,
        "gate_passed": promotion_gate_passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--challengers",
        default=",".join(V56),
        help="Comma-separated subset of V56 variants expected in the matrix",
    )
    args = parser.parse_args()
    challengers = tuple(
        item.strip() for item in args.challengers.split(",") if item.strip())
    report = analyze(args.root, challengers=challengers)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
