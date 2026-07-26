#!/usr/bin/env python3
"""Audit V60 frozen ordered-shortlist verification against V51."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "v56_analysis",
    HERE / "analyze_v56_independent_confirmation_gate.py",
)
V56 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V56)

CONTROL = "v51_control"
CHALLENGER = "v60_ordered_shortlist_verification"
SEARCH_BUDGET = 13
PER_CANDIDATE_BUDGET = 48
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
    actual_calls = PER_CANDIDATE_BUDGET * len(attempts)
    selected_rank = verification.get("selected_shortlist_rank")
    certified = bool(verification.get("certified", False))
    attempt_contract = bool(
        attempts
        and len(attempts) <= SHORTLIST_SIZE
        and all(
            int(attempt.get("candidate_index", -1)) == index
            and int(attempt.get("verification_budget", -1))
            == PER_CANDIDATE_BUDGET
            and int(attempt.get("sample_count", -1))
            == PER_CANDIDATE_BUDGET
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
        and (
            bool(attempts[-1].get("certified", False)) == certified
        )
        and (
            selected_rank == len(attempts) if certified
            else selected_rank is None
        )
    )
    return bool(
        str(row.get("implementation_contract_id"))
        == "v60_frozen_ordered_shortlist_verification"
        and str(row.get("theory_contract_id"))
        == "v60_familywise_gaussian_shortlist_certificate_v1"
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
        and int(verification.get("verification_budget", -1))
        == actual_calls
        and int(verification.get("verification_budget_per_candidate", -1))
        == PER_CANDIDATE_BUDGET
        and int(verification.get("max_verification_budget", -1))
        == PER_CANDIDATE_BUDGET * SHORTLIST_SIZE
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
        and attempt_contract
    )


def _search_identity(control_rows, challenger_rows):
    control = {V56._key(row): row for row in control_rows}
    challenger = {V56._key(row): row for row in challenger_rows}
    keys = sorted(set(control) & set(challenger))
    mismatches = {
        "target_design_fingerprint": [],
        "online_action_sequence_fingerprint": [],
        "optimization_x_recommended": [],
    }
    for key in keys:
        baseline = control[key]
        trial = challenger[key]
        if (
            baseline.get("target_design_fingerprint")
            != trial.get("target_design_fingerprint")
        ):
            mismatches["target_design_fingerprint"].append(key)
        if (
            baseline.get("online_action_sequence_fingerprint")
            != trial.get("online_action_sequence_fingerprint")
        ):
            mismatches["online_action_sequence_fingerprint"].append(key)
        if (
            baseline.get("x_recommended")
            != trial.get("optimization_x_recommended")
        ):
            mismatches["optimization_x_recommended"].append(key)
    return {
        "paired_count": len(keys),
        "mismatches": mismatches,
        "all_search_trajectories_and_primary_actions_identical": bool(
            keys and all(not values for values in mismatches.values())),
    }


def _summary(rows):
    verifications = [
        dict(row.get("terminal_verification") or {}) for row in rows
    ]
    certified = [
        bool(item.get("certified", False)) for item in verifications
    ]
    feasible = [bool(row.get("true_feasible", False)) for row in rows]
    return {
        "run_count": len(rows),
        "true_feasible_count": sum(feasible),
        "terminal_certified_count": sum(certified),
        "terminal_false_certificate_count": sum(
            cert and not truth
            for cert, truth in zip(certified, feasible)
        ),
        "terminal_switch_count": sum(
            bool(item.get("recommendation_changed", False))
            for item in verifications
        ),
        "rank_1_certificate_count": sum(
            item.get("selected_shortlist_rank") == 1
            for item in verifications
        ),
        "rank_2_certificate_count": sum(
            item.get("selected_shortlist_rank") == 2
            for item in verifications
        ),
        "median_verification_call_count": V56._median(
            item.get("verification_budget") for item in verifications),
        "median_algorithm_time_sec": V56._median(
            row.get("algorithm_time_sec") for row in rows),
        "median_terminal_verification_time_sec": V56._median(
            dict(row.get("finalization_timing_sec") or {}).get(
                "terminal_fixed_policy_verification")
            for row in rows
        ),
    }


def analyze(control_root, challenger_root):
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
        len(grouped[variant][domain]) == 5
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
    search_identity = _search_identity(control_rows, challenger_rows)
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
        formal_gate and certified_count == len(challenger_rows))
    return {
        "scope": "v60_frozen_ordered_shortlist_verification_gate",
        "control_root": str(Path(control_root)),
        "challenger_root": str(Path(challenger_root)),
        "row_count": len(control_rows) + len(challenger_rows),
        "load_errors": errors,
        "complete_5_seed_matrix": complete,
        "paired_keys_match_control": paired,
        "contract_valid": contracts,
        "search_identity": search_identity,
        "paired_performance": performance,
        "familywise_error_probability": FAMILYWISE_DELTA,
        "terminal_certified_count": certified_count,
        "terminal_false_certificate_count": false_certificates,
        "verification_calls_counted_in_total_budget": True,
        "formal_gate_passed": formal_gate,
        "certification_gate_passed": certification_gate,
        "gate_passed": certification_gate,
        "summaries": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control_root", type=Path)
    parser.add_argument("challenger_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.control_root, args.challenger_root)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["formal_gate_passed"] else 1)


if __name__ == "__main__":
    main()
