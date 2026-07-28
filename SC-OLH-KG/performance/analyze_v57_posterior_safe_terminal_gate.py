#!/usr/bin/env python3
"""Audit the V57 confirmation plus posterior-safe terminal gate."""

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

CONTROL = "v56_confirm4096"
CHALLENGER = "v57_posterior_safe_terminal"
KNOWN = (CONTROL, CHALLENGER)
SWITCH_RUN_DELTA = 0.05


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


def _switch_contract(row):
    state = dict(row.get("posterior_dominance") or {})
    history = [dict(item) for item in state.get("history") or []]
    horizon = max(int(row.get("N", 0)) - int(row.get("n0", 0)), 1)
    expected_delta = SWITCH_RUN_DELTA / horizon
    updates = [
        item for item in history
        if item.get("incumbent_before") is not None
    ]
    return bool(
        state.get("enabled") is True
        and bool(row.get("posterior_dominance_terminal_used", False))
        and not bool(state.get("target_oracle_used", True))
        and len(updates) == int(row.get("N", 0)) - int(row.get("n0", 0))
        and all(not bool(item.get("target_oracle_used", True))
                for item in history)
        and all(
            abs(float(item.get("delta_switch", -1.0)) - expected_delta)
            <= 1e-12
            for item in updates
        )
        and horizon * expected_delta <= SWITCH_RUN_DELTA + 1e-12
    )


def _contract(row, variant):
    if variant == CONTROL:
        return V56._contract(row, CONTROL)
    contract = dict(row.get("decision_backend_contract") or {})
    confirmations = V56._confirmations(row)
    return bool(
        str(row.get("implementation_contract_id"))
        == "v57_posterior_safe_terminal_closure"
        and str(row.get("theory_contract_id"))
        == "v57_confirmation_dominance_composition_v1"
        and str(contract.get("policy_improvement_guard_mode"))
        == "independent_confirmation"
        and str(contract.get("policy_improvement_score_transform"))
        == "bounded_current_gain"
        and int(contract.get(
            "policy_improvement_confirmation_samples", -1)) == 4096
        and bool(confirmations)
        and all(V56._confirmation_contract_valid(item)
                for item in confirmations)
        and str(contract.get("terminal_rule")) == "posterior_dominance"
        and bool(contract.get(
            "acquisition_and_recommendation_share_terminal_action_universe",
            False,
        ))
        and not bool(contract.get("target_oracle_used", True))
        and not bool(row.get("online_action_trace_target_oracle_used", True))
        and _switch_contract(row)
    )


def _summary(rows):
    summary = V56._summary(rows)
    histories = [
        dict(row.get("posterior_dominance") or {})
        for row in rows
    ]
    summary.update({
        "posterior_dominance_terminal_count": sum(bool(row.get(
            "posterior_dominance_terminal_used", False)) for row in rows),
        "posterior_switch_count": sum(int(item.get(
            "switch_count", 0) or 0) for item in histories),
        "switch_contract_valid_count": sum(
            _switch_contract(row) for row in rows),
        "median_switch_delta": V56._median(
            item.get("delta_switch") for item in histories),
        "maximum_switch_horizon_bound": max([
            max(int(row.get("N", 0)) - int(row.get("n0", 0)), 1)
            * float(dict(row.get("posterior_dominance") or {}).get(
                "delta_switch", 0.0) or 0.0)
            for row in rows
        ], default=0.0),
    })
    return summary


def analyze(root):
    rows, errors = load_rows(root)
    grouped = {
        variant: {
            domain: [
                row for row in rows
                if row["gate_variant"] == variant
                and str(row.get("heldout")) == domain
            ]
            for domain in V56.DOMAINS
        }
        for variant in KNOWN
    }
    summaries = {
        variant: {
            domain: _summary(domain_rows)
            for domain, domain_rows in domains.items()
        }
        for variant, domains in grouped.items()
    }
    complete = all(
        len(grouped[variant][domain]) == 5
        for variant in KNOWN for domain in V56.DOMAINS
    )
    keys = {
        variant: {
            V56._key(row)
            for domain_rows in grouped[variant].values()
            for row in domain_rows
        }
        for variant in KNOWN
    }
    paired = keys[CONTROL] == keys[CHALLENGER]
    contracts = {
        variant: all(
            _contract(row, variant)
            for domain_rows in grouped[variant].values()
            for row in domain_rows
        )
        for variant in KNOWN
    }
    flat = {
        variant: [
            row
            for domain_rows in grouped[variant].values()
            for row in domain_rows
        ]
        for variant in KNOWN
    }
    performance = V56._paired_performance(
        flat[CONTROL], flat[CHALLENGER])
    audits = [
        dict(row.get("certificate_outcome_audit") or {})
        for row in flat[CHALLENGER]
    ]
    no_false_certificates = all(
        int(item.get("false_certificate_count", 0) or 0) == 0
        for item in audits
    )
    confirmation_nonvacuous = any(
        bool(item.get("passed", False))
        for row in flat[CHALLENGER]
        for item in V56._confirmations(row)
    )
    chance_certificate_nonvacuous = any(
        int(item.get("posterior_certified_count", 0) or 0) > 0
        for item in audits
    )
    formal_gate = bool(
        complete
        and paired
        and all(contracts.values())
        and confirmation_nonvacuous
        and no_false_certificates
    )
    promotion_gate = bool(
        formal_gate
        and performance["performance_noninferior"]
        and performance["strict_gain_detected"]
        and performance["feasibility_loss_count"] == 0
    )
    return {
        "scope": "v57_posterior_safe_terminal_gate",
        "root": str(Path(root)),
        "row_count": len(rows),
        "load_errors": errors,
        "complete_5_seed_matrix": complete,
        "paired_keys_match_control": paired,
        "contract_valid": contracts,
        "confirmation_nonvacuous": confirmation_nonvacuous,
        "chance_certificate_nonvacuous": (
            chance_certificate_nonvacuous),
        "no_false_certificates": no_false_certificates,
        "paired_performance": performance,
        "formal_gate_passed": formal_gate,
        "promotion_gate_passed": promotion_gate,
        "gate_passed": promotion_gate,
        "summaries": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.root)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["formal_gate_passed"] else 1)


if __name__ == "__main__":
    main()
