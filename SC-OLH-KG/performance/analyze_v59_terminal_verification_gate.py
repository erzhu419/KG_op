#!/usr/bin/env python3
"""Audit V59 fixed-policy Gaussian terminal verification against V51."""

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
CHALLENGER = "v59_terminal_gaussian_verification"
VERIFICATION_BUDGET = 48
SEARCH_BUDGET = 13


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
    return bool(
        str(row.get("implementation_contract_id"))
        == "v59_frozen_policy_gaussian_verification"
        and str(row.get("theory_contract_id"))
        == "v59_gaussian_replication_certificate_v1"
        and int(row.get("N", -1)) == SEARCH_BUDGET
        and int(row.get("n_search_simulations", -1)) == SEARCH_BUDGET
        and int(row.get("n_verification_simulations", -1))
        == VERIFICATION_BUDGET
        and int(row.get("n_simulations", -1))
        == SEARCH_BUDGET + VERIFICATION_BUDGET
        and str(decision.get("evaluate_or_replicate_new_action_policy"))
        == "canonical_plus_posterior_risk"
        and str(decision.get("policy_improvement_mode")) == "off"
        and str(decision.get("terminal_rule")) == "posterior_bayes_risk"
        and decision.get("coherent") is True
        and decision.get("target_oracle_used") is False
        and str(decision.get("terminal_verification_method"))
        == "gaussian_student_t_chi_square"
        and decision.get("terminal_verification_policy_frozen") is True
        and decision.get("terminal_verification_reuses_search_samples")
        is False
        and decision.get("terminal_verification_changes_recommendation")
        is False
        and verification.get("enabled") is True
        and str(verification.get("status"))
        in {"certified", "not_certified"}
        and int(verification.get("verification_budget", -1))
        == VERIFICATION_BUDGET
        and int(verification.get("sample_count", -1))
        == VERIFICATION_BUDGET
        and int(verification.get("search_evaluation_count", -1))
        == SEARCH_BUDGET
        and int(verification.get("total_evaluation_count", -1))
        == SEARCH_BUDGET + VERIFICATION_BUDGET
        and abs(float(verification.get("delta", -1.0)) - 0.05) <= 1e-12
        and str(verification.get("noise_model")) == "iid_gaussian"
        and verification.get("policy_frozen_before_verification") is True
        and verification.get("search_samples_reused") is False
        and verification.get("posterior_updated_from_verification") is False
        and verification.get("recommendation_changed") is False
        and verification.get("verification_samples_logged") is False
        and verification.get("target_oracle_used") is False
        and randomness.get("verification_stream_independent") is True
        and int(randomness.get("verification_evaluation_count", -1))
        == VERIFICATION_BUDGET
        and int(randomness.get("total_evaluation_count", -1))
        == SEARCH_BUDGET + VERIFICATION_BUDGET
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _search_identity(control_rows, challenger_rows):
    control = {V56._key(row): row for row in control_rows}
    challenger = {V56._key(row): row for row in challenger_rows}
    keys = sorted(set(control) & set(challenger))
    fields = (
        "target_design_fingerprint",
        "online_action_sequence_fingerprint",
        "x_recommended",
    )
    mismatches = {
        field: [
            key for key in keys
            if control[key].get(field) != challenger[key].get(field)
        ]
        for field in fields
    }
    return {
        "paired_count": len(keys),
        "mismatches": mismatches,
        "all_search_trajectories_identical": bool(
            keys and all(not values for values in mismatches.values())),
    }


def _summary(rows):
    verifications = [
        dict(row.get("terminal_verification") or {}) for row in rows
    ]
    true_feasible = [
        bool(row.get("true_feasible", False)) for row in rows
    ]
    certified = [
        bool(item.get("certified", False)) for item in verifications
    ]
    posterior_audits = [
        dict(row.get("certificate_outcome_audit") or {}) for row in rows
    ]
    return {
        "run_count": len(rows),
        "true_feasible_count": sum(true_feasible),
        "terminal_certified_count": sum(certified),
        "terminal_false_certificate_count": sum(
            is_certified and not feasible
            for is_certified, feasible in zip(certified, true_feasible)
        ),
        "terminal_true_certificate_count": sum(
            is_certified and feasible
            for is_certified, feasible in zip(certified, true_feasible)
        ),
        "median_terminal_upper_margin": V56._median(
            item.get("upper_margin") for item in verifications),
        "median_terminal_sample_count": V56._median(
            item.get("sample_count") for item in verifications),
        "posterior_certified_count": sum(
            int(item.get("posterior_certified_count", 0) or 0)
            for item in posterior_audits
        ),
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
        CONTROL: {
            domain: [
                row for row in control_rows
                if str(row.get("heldout")) == domain
            ]
            for domain in V56.DOMAINS
        },
        CHALLENGER: {
            domain: [
                row for row in challenger_rows
                if str(row.get("heldout")) == domain
            ]
            for domain in V56.DOMAINS
        },
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
    domain_nonvacuous = {
        domain: summaries[CHALLENGER][domain][
            "terminal_certified_count"] > 0
        for domain in V56.DOMAINS
    }
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
        and search_identity["all_search_trajectories_identical"]
        and performance["all_recommendations_identical"]
        and performance["performance_noninferior"]
        and false_certificates == 0
    )
    certification_gate = bool(
        formal_gate and all(domain_nonvacuous.values()))
    return {
        "scope": "v59_fixed_policy_gaussian_verification_gate",
        "control_root": str(Path(control_root)),
        "challenger_root": str(Path(challenger_root)),
        "row_count": len(control_rows) + len(challenger_rows),
        "load_errors": errors,
        "complete_5_seed_matrix": complete,
        "paired_keys_match_control": paired,
        "contract_valid": contracts,
        "search_identity": search_identity,
        "paired_performance": performance,
        "terminal_certificate_nonvacuous_by_domain": domain_nonvacuous,
        "terminal_false_certificate_count": false_certificates,
        "per_policy_error_probability": 0.05,
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
