from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "SC-OLH-KG/performance/analyze_v61_power_shortlist_gate.py"
)
SPEC = importlib.util.spec_from_file_location("v61_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(variant, domain, seed):
    control = variant == MODULE.CONTROL
    primary = [seed, 1]
    fallback = [seed, 2]
    use_fallback = not control and seed >= 4
    row = {
        "heldout": domain,
        "seed": seed,
        "N": MODULE.SEARCH_BUDGET,
        "n0": 10,
        "implementation_contract_id": (
            "promoted_v51_observed_terminal_closure"
            if control
            else "v61_power_ordered_shortlist_verification"
        ),
        "theory_contract_id": (
            "v51_statistical_closure_v2"
            if control
            else "v61_familywise_asymmetric_shortlist_certificate_v1"
        ),
        "decision_backend_contract": {
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk"),
            "policy_improvement_mode": "off",
            "terminal_rule": "posterior_bayes_risk",
            "coherent": True,
            "target_oracle_used": False,
            "terminal_verification_method": (
                "disabled"
                if control else (
                    "ordered_frozen_shortlist_"
                    "gaussian_student_t_chi_square"
                )
            ),
            "terminal_verification_policy_frozen": not control,
            "terminal_verification_reuses_search_samples": False,
            "terminal_verification_changes_recommendation": use_fallback,
        },
        "target_design_fingerprint": f"design-{domain}-{seed}",
        "online_action_sequence_fingerprint": f"online-{domain}-{seed}",
        "x_recommended": fallback if use_fallback else primary,
        "optimization_x_recommended": primary,
        "true_feasible": True,
        "feasible_simple_regret": 0.05 if use_fallback else 0.1,
        "online_action_trace_target_oracle_used": False,
        "algorithm_time_sec": 10.0,
        "finalization_timing_sec": {
            "terminal_fixed_policy_verification": 0.1,
        },
    }
    if control:
        row.update({
            "x_recommended": primary,
            "n_search_simulations": MODULE.SEARCH_BUDGET,
            "n_verification_simulations": 0,
            "n_simulations": MODULE.SEARCH_BUDGET,
            "terminal_verification": {
                "enabled": False,
                "status": "disabled",
            },
            "simulation_randomness_contract": {},
        })
        return row

    attempt_count = 2 if use_fallback else 1
    budgets = [MODULE.PRIMARY_BUDGET, MODULE.FALLBACK_BUDGET]
    attempts = []
    for index in range(attempt_count):
        certified = index == attempt_count - 1
        attempts.append({
            "enabled": True,
            "status": "certified" if certified else "not_certified",
            "certified": certified,
            "candidate_index": index,
            "verification_budget": budgets[index],
            "sample_count": budgets[index],
            "delta": 0.025,
            "policy_frozen_before_verification": True,
            "search_samples_reused": False,
            "posterior_updated_from_verification": False,
            "target_oracle_used": False,
        })
    actual_calls = sum(budgets[:attempt_count])
    row.update({
        "n_search_simulations": MODULE.SEARCH_BUDGET,
        "n_verification_simulations": actual_calls,
        "n_simulations": MODULE.SEARCH_BUDGET + actual_calls,
        "terminal_verification": {
            "enabled": True,
            "status": "certified",
            "protocol": "ordered_frozen_shortlist",
            "shortlist_frozen_before_verification": True,
            "frozen_shortlist_size": MODULE.SHORTLIST_SIZE,
            "frozen_shortlist": [
                {"posterior_rank": 1, "point": primary},
                {"posterior_rank": 2, "point": fallback},
            ],
            "candidate_verification_budgets": budgets,
            "verification_budget": actual_calls,
            "verification_budget_per_candidate": MODULE.PRIMARY_BUDGET,
            "fallback_verification_budget": MODULE.FALLBACK_BUDGET,
            "max_verification_budget": sum(budgets),
            "search_evaluation_count": MODULE.SEARCH_BUDGET,
            "total_evaluation_count": MODULE.SEARCH_BUDGET + actual_calls,
            "familywise_delta": MODULE.FAMILYWISE_DELTA,
            "per_candidate_delta": (
                MODULE.FAMILYWISE_DELTA / MODULE.SHORTLIST_SIZE),
            "search_samples_reused": False,
            "posterior_updated_from_verification": False,
            "verification_samples_logged": False,
            "target_oracle_used": False,
            "attempts": attempts,
            "certified": True,
            "selected_shortlist_rank": attempt_count,
            "recommendation_changed": use_fallback,
        },
        "simulation_randomness_contract": {
            "verification_stream_independent": True,
            "verification_protocol": "ordered_frozen_shortlist",
            "verification_evaluation_count": actual_calls,
            "total_evaluation_count": MODULE.SEARCH_BUDGET + actual_calls,
        },
    })
    return row


def _write_matrix(root, variant, n_seeds=6):
    for domain in MODULE.V56.DOMAINS:
        for seed in range(n_seeds):
            path = root / variant / domain / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "experiment_variant": f"gate/{variant}/{domain}/seed{seed}",
                "rows": [_row(variant, domain, seed)],
            }), encoding="utf-8")


def test_v61_gate_accepts_asymmetric_budget_on_development_and_fresh_seeds(
    tmp_path,
):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    _write_matrix(control_root, MODULE.CONTROL)
    _write_matrix(challenger_root, MODULE.CHALLENGER)
    report = MODULE.analyze(
        control_root, challenger_root, expected_seeds=6)
    assert report["contract_valid"] == {
        MODULE.CONTROL: True,
        MODULE.CHALLENGER: True,
    }
    assert report["search_identity"][
        "all_search_trajectories_and_primary_actions_identical"] is True
    assert report["terminal_certified_count"] == 18
    assert report["terminal_false_certificate_count"] == 0
    assert report["fresh_seed_gate_passed"] is True
    assert report["formal_gate_passed"] is True
    assert report["certification_gate_passed"] is True
    assert report["gate_passed"] is True


def test_v61_gate_rejects_uncertified_fresh_policy(tmp_path):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    _write_matrix(control_root, MODULE.CONTROL)
    _write_matrix(challenger_root, MODULE.CHALLENGER)
    path = (
        challenger_root
        / MODULE.CHALLENGER
        / MODULE.V56.DOMAINS[0]
        / "seed5"
        / "result.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    row["terminal_verification"]["certified"] = False
    row["terminal_verification"]["selected_shortlist_rank"] = None
    row["terminal_verification"]["attempts"][-1]["certified"] = False
    row["terminal_verification"]["attempts"][-1]["status"] = "not_certified"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = MODULE.analyze(
        control_root, challenger_root, expected_seeds=6)
    assert report["fresh_seed_gate_passed"] is False
    assert report["certification_gate_passed"] is False
