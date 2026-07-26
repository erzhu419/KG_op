#!/usr/bin/env python3
"""Audit V61 asymmetric-power shortlist verification against V51."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V56 = _load(
    "v56_analysis",
    "analyze_v56_independent_confirmation_gate.py",
)
V60 = _load(
    "v60_analysis",
    "analyze_v60_shortlist_verification_gate.py",
)

CONTROL = "v51_control"
CHALLENGER = "v61_power_ordered_shortlist_verification"
SEARCH_BUDGET = 13
PRIMARY_BUDGET = 48
FALLBACK_BUDGET = 96
SHORTLIST_SIZE = 2
FAMILYWISE_DELTA = 0.05


def _variant(experiment):
    marker = f"/{str(experiment).strip('/')}/"
    for variant in (CONTROL, CHALLENGER):
        if f"/{variant}/" in marker:
            return variant
    return None


def load_rows(root, expected_variant):
    rows = []
    errors = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            variant = _variant(payload.get("experiment_variant", ""))
            if variant != expected_variant:
                continue
            for raw in payload["rows"]:
                row = dict(raw)
                row["gate_variant"] = variant
                row["result_path"] = str(path)
                rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return rows, errors


def _verification_contract(row):
    verification = dict(row.get("terminal_verification") or {})
    decision = dict(row.get("decision_backend_contract") or {})
    randomness = dict(row.get("simulation_randomness_contract") or {})
    attempts = list(verification.get("attempts") or [])
    expected_budgets = [PRIMARY_BUDGET, FALLBACK_BUDGET]
    actual_calls = sum(
        expected_budgets[index] for index in range(len(attempts))
    )
    certified = bool(verification.get("certified", False))
    selected_rank = verification.get("selected_shortlist_rank")
    attempts_valid = bool(
        attempts
        and len(attempts) <= SHORTLIST_SIZE
        and all(
            int(attempt.get("candidate_index", -1)) == index
            and int(attempt.get("verification_budget", -1))
            == expected_budgets[index]
            and int(attempt.get("sample_count", -1))
            == expected_budgets[index]
            and abs(float(attempt.get("delta", -1.0)) - 0.025) <= 1e-12
            and attempt.get("policy_frozen_before_verification") is True
            and attempt.get("search_samples_reused") is False
            and attempt.get("posterior_updated_from_verification") is False
            and attempt.get("target_oracle_used") is False
            for index, attempt in enumerate(attempts)
        )
        and all(
            not bool(attempt.get("certified", False))
            for attempt in attempts[:-1]
        )
        and bool(attempts[-1].get("certified", False)) == certified
        and (
            selected_rank == len(attempts) if certified
            else selected_rank is None
        )
    )
    return bool(
        str(row.get("implementation_contract_id"))
        == "v61_power_ordered_shortlist_verification"
        and str(row.get("theory_contract_id"))
        == "v61_familywise_asymmetric_shortlist_certificate_v1"
        and int(row.get("N", -1)) == SEARCH_BUDGET
        and int(row.get("n_search_simulations", -1)) == SEARCH_BUDGET
        and int(row.get("n_verification_simulations", -1)) == actual_calls
        and int(row.get("n_simulations", -1))
        == SEARCH_BUDGET + actual_calls
        and str(decision.get("evaluate_or_replicate_new_action_policy"))
        == "canonical_plus_posterior_risk"
        and str(decision.get("policy_improvement_mode")) == "off"
        and str(decision.get("terminal_rule")) == "posterior_bayes_risk"
        and decision.get("coherent") is True
        and decision.get("target_oracle_used") is False
        and str(decision.get("terminal_verification_method"))
        == (
            "ordered_frozen_shortlist_"
            "gaussian_student_t_chi_square"
        )
        and decision.get("terminal_verification_policy_frozen") is True
        and decision.get("terminal_verification_reuses_search_samples")
        is False
        and verification.get("enabled") is True
        and str(verification.get("protocol"))
        == "ordered_frozen_shortlist"
        and verification.get("shortlist_frozen_before_verification") is True
        and int(verification.get("frozen_shortlist_size", -1))
        == SHORTLIST_SIZE
        and len(verification.get("frozen_shortlist") or [])
        == SHORTLIST_SIZE
        and list(verification.get("candidate_verification_budgets") or [])
        == expected_budgets
        and int(verification.get("verification_budget", -1))
        == actual_calls
        and int(verification.get("verification_budget_per_candidate", -1))
        == PRIMARY_BUDGET
        and int(verification.get("fallback_verification_budget", -1))
        == FALLBACK_BUDGET
        and int(verification.get("max_verification_budget", -1))
        == PRIMARY_BUDGET + FALLBACK_BUDGET
        and int(verification.get("search_evaluation_count", -1))
        == SEARCH_BUDGET
        and int(verification.get("total_evaluation_count", -1))
        == SEARCH_BUDGET + actual_calls
        and abs(
            float(verification.get("familywise_delta", -1.0))
            - FAMILYWISE_DELTA
        ) <= 1e-12
        and abs(
            float(verification.get("per_candidate_delta", -1.0))
            - FAMILYWISE_DELTA / SHORTLIST_SIZE
        ) <= 1e-12
        and verification.get("search_samples_reused") is False
        and verification.get("posterior_updated_from_verification") is False
        and verification.get("verification_samples_logged") is False
        and verification.get("target_oracle_used") is False
        and randomness.get("verification_stream_independent") is True
        and str(randomness.get("verification_protocol"))
        == "ordered_frozen_shortlist"
        and int(randomness.get("verification_evaluation_count", -1))
        == actual_calls
        and int(randomness.get("total_evaluation_count", -1))
        == SEARCH_BUDGET + actual_calls
        and not bool(row.get("online_action_trace_target_oracle_used", True))
        and attempts_valid
    )


def _summary(rows):
    summary = V60._summary(rows)
    verification_calls = [
        dict(row.get("terminal_verification") or {}).get(
            "verification_budget")
        for row in rows
    ]
    summary["total_verification_call_count"] = int(sum(
        int(value or 0) for value in verification_calls
    ))
    summary["mean_verification_call_count"] = (
        None if not verification_calls else float(
            summary["total_verification_call_count"]
            / len(verification_calls)
        )
    )
    return summary


def _subset(rows, *, seed_start, seed_stop):
    return [
        row for row in rows
        if seed_start <= int(row.get("seed", -1)) < seed_stop
    ]


def analyze(control_root, challenger_root, expected_seeds=20):
    control_rows, control_errors = load_rows(control_root, CONTROL)
    challenger_rows, challenger_errors = load_rows(
        challenger_root, CHALLENGER)
    errors = control_errors + challenger_errors
    grouped = {
        variant: {
            domain: [
                row for row in rows
                if str(row.get("heldout")) == domain
            ]
            for domain in V56.DOMAINS
        }
        for variant, rows in (
            (CONTROL, control_rows),
            (CHALLENGER, challenger_rows),
        )
    }
    complete = all(
        len(grouped[variant][domain]) == int(expected_seeds)
        for variant in (CONTROL, CHALLENGER)
        for domain in V56.DOMAINS
    )
    keys = {
        CONTROL: {V56._key(row) for row in control_rows},
        CHALLENGER: {V56._key(row) for row in challenger_rows},
    }
    paired = keys[CONTROL] == keys[CHALLENGER]
    contracts = {
        CONTROL: bool(control_rows) and all(
            V56._contract(row, CONTROL) for row in control_rows),
        CHALLENGER: bool(challenger_rows) and all(
            _verification_contract(row) for row in challenger_rows),
    }
    search_identity = V60._search_identity(
        control_rows, challenger_rows)
    performance = V56._paired_performance(
        control_rows, challenger_rows)
    summaries = {
        variant: {
            domain: _summary(rows)
            for domain, rows in domain_rows.items()
        }
        for variant, domain_rows in grouped.items()
    }
    certified_count = sum(
        summaries[CHALLENGER][domain]["terminal_certified_count"]
        for domain in V56.DOMAINS
    )
    false_certificates = sum(
        summaries[CHALLENGER][domain][
            "terminal_false_certificate_count"]
        for domain in V56.DOMAINS
    )
    expected_total = int(expected_seeds) * len(V56.DOMAINS)
    fresh_start = 5
    fresh_control = _subset(
        control_rows, seed_start=fresh_start, seed_stop=expected_seeds)
    fresh_challenger = _subset(
        challenger_rows, seed_start=fresh_start, seed_stop=expected_seeds)
    fresh_performance = V56._paired_performance(
        fresh_control, fresh_challenger)
    fresh_certified = sum(bool(
        dict(row.get("terminal_verification") or {}).get(
            "certified", False)
    ) for row in fresh_challenger)
    fresh_false = sum(
        bool(dict(row.get("terminal_verification") or {}).get(
            "certified", False))
        and not bool(row.get("true_feasible", False))
        for row in fresh_challenger
    )
    formal_gate = bool(
        complete
        and not errors
        and paired
        and all(contracts.values())
        and search_identity[
            "all_search_trajectories_and_primary_actions_identical"]
        and performance["performance_noninferior"]
        and performance["feasibility_loss_count"] == 0
        and false_certificates == 0
    )
    certification_gate = bool(
        formal_gate and certified_count == expected_total)
    fresh_gate = bool(
        expected_seeds > fresh_start
        and fresh_performance["performance_noninferior"]
        and fresh_performance["feasibility_loss_count"] == 0
        and fresh_false == 0
        and fresh_certified == len(fresh_challenger)
    )
    return {
        "scope": "v61_asymmetric_power_shortlist_verification_gate",
        "control_root": str(Path(control_root)),
        "challenger_root": str(Path(challenger_root)),
        "expected_seeds_per_domain": int(expected_seeds),
        "row_count": len(control_rows) + len(challenger_rows),
        "load_errors": errors,
        "complete_matrix": complete,
        "paired_keys_match_control": paired,
        "contract_valid": contracts,
        "search_identity": search_identity,
        "paired_performance": performance,
        "familywise_error_probability": FAMILYWISE_DELTA,
        "terminal_certified_count": certified_count,
        "terminal_false_certificate_count": false_certificates,
        "verification_calls_counted_in_total_budget": True,
        "development_seed_range": [0, 4],
        "fresh_seed_range": [fresh_start, int(expected_seeds) - 1],
        "fresh_seed_performance": fresh_performance,
        "fresh_terminal_certified_count": fresh_certified,
        "fresh_terminal_false_certificate_count": fresh_false,
        "fresh_seed_gate_passed": fresh_gate,
        "formal_gate_passed": formal_gate,
        "certification_gate_passed": certification_gate,
        "gate_passed": bool(certification_gate and fresh_gate),
        "summaries": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control_root", type=Path)
    parser.add_argument("challenger_root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.control_root,
        args.challenger_root,
        expected_seeds=args.expected_seeds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["formal_gate_passed"] else 1)


if __name__ == "__main__":
    main()
