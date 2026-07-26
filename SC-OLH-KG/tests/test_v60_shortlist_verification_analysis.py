from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "SC-OLH-KG/performance/analyze_v60_shortlist_verification_gate.py"
)
SPEC = importlib.util.spec_from_file_location("v60_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(variant, domain, seed):
    control = variant == MODULE.CONTROL
    primary = [seed, 1]
    fallback = [seed, 2]
    use_fallback = not control and seed == 4
    row = {
        "heldout": domain,
        "seed": seed,
        "N": MODULE.SEARCH_BUDGET,
        "n0": 10,
        "implementation_contract_id": (
            "promoted_v51_observed_terminal_closure"
            if control
            else "v60_frozen_ordered_shortlist_verification"
        ),
        "theory_contract_id": (
            "v51_statistical_closure_v2"
            if control
            else "v60_familywise_gaussian_shortlist_certificate_v1"
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
    attempts = []
    for index in range(attempt_count):
        certified = index == attempt_count - 1
        attempts.append({
            "enabled": True,
            "status": "certified" if certified else "not_certified",
            "certified": certified,
            "candidate_index": index,
            "verification_budget": MODULE.PER_CANDIDATE_BUDGET,
            "sample_count": MODULE.PER_CANDIDATE_BUDGET,
            "delta": 0.025,
            "policy_frozen_before_verification": True,
            "search_samples_reused": False,
            "posterior_updated_from_verification": False,
            "target_oracle_used": False,
        })
    actual_calls = MODULE.PER_CANDIDATE_BUDGET * attempt_count
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
            "verification_budget": actual_calls,
            "verification_budget_per_candidate": (
                MODULE.PER_CANDIDATE_BUDGET),
            "max_verification_budget": (
                MODULE.PER_CANDIDATE_BUDGET * MODULE.SHORTLIST_SIZE),
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


def _write_matrix(root, variant):
    for domain in MODULE.V56.DOMAINS:
        for seed in range(5):
            path = root / variant / domain / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "experiment_variant": f"gate/{variant}/{domain}/seed{seed}",
                "rows": [_row(variant, domain, seed)],
            }), encoding="utf-8")


def test_v60_gate_accepts_safe_certified_fallbacks(tmp_path):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    _write_matrix(control_root, MODULE.CONTROL)
    _write_matrix(challenger_root, MODULE.CHALLENGER)
    report = MODULE.analyze(control_root, challenger_root)
    assert report["contract_valid"] == {
        MODULE.CONTROL: True,
        MODULE.CHALLENGER: True,
    }
    assert report["search_identity"][
        "all_search_trajectories_and_primary_actions_identical"] is True
    assert report["paired_performance"]["recommendation_change_count"] == 3
    assert report["terminal_certified_count"] == 15
    assert report["terminal_false_certificate_count"] == 0
    assert report["formal_gate_passed"] is True
    assert report["certification_gate_passed"] is True


def test_v60_gate_rejects_false_deployed_certificate(tmp_path):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    _write_matrix(control_root, MODULE.CONTROL)
    _write_matrix(challenger_root, MODULE.CHALLENGER)
    path = next(challenger_root.rglob("result.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["true_feasible"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = MODULE.analyze(control_root, challenger_root)
    assert report["terminal_false_certificate_count"] == 1
    assert report["formal_gate_passed"] is False
