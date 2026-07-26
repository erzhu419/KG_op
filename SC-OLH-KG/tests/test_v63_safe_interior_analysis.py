from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "SC-OLH-KG/performance/analyze_v63_safe_interior_gate.py"
)
SPEC = importlib.util.spec_from_file_location("v63_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(variant, domain, seed):
    control = variant == MODULE.CONTROL
    primary = [seed, 1]
    fallback = [seed, 2]
    use_fallback = not control and seed % 2 == 1
    row = {
        "heldout": domain,
        "seed": seed,
        "N": MODULE.SEARCH_BUDGET,
        "n0": 10,
        "implementation_contract_id": (
            "promoted_v51_observed_terminal_closure"
            if control else
            "v63_cumulative_risk_safe_interior_shortlist_verification"
        ),
        "theory_contract_id": (
            "v51_statistical_closure_v2"
            if control else
            "v63_frozen_selector_familywise_quantile_certificate_v1"
        ),
        "terminal_verification_method": (
            "component_bonferroni"
            if control else "normal_quantile_tolerance"
        ),
        "terminal_verification_shortlist_mode": (
            "posterior_ranked" if control else MODULE.SHORTLIST_MODE
        ),
        "terminal_safe_interior_require_provider": not control,
        "decision_backend_contract": {
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk"),
            "policy_improvement_mode": "off",
            "terminal_rule": "posterior_bayes_risk",
            "coherent": True,
            "target_oracle_used": False,
            "terminal_verification_method": (
                "disabled" if control else MODULE.PROTOCOL_METHOD
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
        budget = budgets[index]
        attempts.append({
            "enabled": True,
            "status": "certified" if certified else "not_certified",
            "certified": certified,
            "candidate_index": index,
            "verification_budget": budget,
            "sample_count": budget,
            "delta": MODULE.PER_CANDIDATE_DELTA,
            "method": MODULE.ATTEMPT_METHOD,
            "method_mode": "normal_quantile_tolerance",
            "degrees_of_freedom": budget - 1,
            "tolerance_factor": 2.0,
            "noncentral_t_quantile": 16.0,
            "noncentrality": 12.0,
            "quantile_upper": -0.1,
            "upper_margin": -0.1,
            "tau": 0.0,
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
            "method": MODULE.PROTOCOL_METHOD,
            "protocol": "ordered_frozen_shortlist",
            "shortlist_mode": MODULE.SHORTLIST_MODE,
            "shortlist_frozen_before_verification": True,
            "frozen_shortlist_size": MODULE.SHORTLIST_SIZE,
            "frozen_shortlist": [
                {
                    "shortlist_role": "posterior_bayes_primary",
                    "point": primary,
                },
                {
                    "shortlist_role":
                    "posterior_safe_interior_diversified",
                    "selection_contract": MODULE.SELECTION_CONTRACT,
                    "candidate_universe": "frozen_initial_atlas",
                    "candidate_universe_size": 10,
                    "eligible_count": 2,
                    "probability_slack": 0.05,
                    "coordinate_source": "cumulative_risk_psi=(A,N)",
                    "coordinate_dimension": 4,
                    "target_labels_used": False,
                    "target_oracle_used": False,
                    "verification_samples_used": False,
                    "point": fallback,
                },
            ],
            "candidate_verification_budgets": budgets,
            "verification_budget": actual_calls,
            "max_verification_budget": sum(budgets),
            "familywise_delta": MODULE.FAMILYWISE_DELTA,
            "per_candidate_delta": MODULE.PER_CANDIDATE_DELTA,
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


def _write_matrix(root, variant, seed_start=40, n_seeds=3):
    for domain in MODULE.V56.DOMAINS:
        for seed in range(seed_start, seed_start + n_seeds):
            path = root / variant / domain / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "experiment_variant": f"gate/{variant}/{domain}/seed{seed}",
                "rows": [_row(variant, domain, seed)],
            }), encoding="utf-8")


def test_v63_gate_accepts_safe_interior_fresh_matrix(tmp_path):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    _write_matrix(control_root, MODULE.CONTROL)
    _write_matrix(challenger_root, MODULE.CHALLENGER)
    report = MODULE.analyze(
        control_root,
        challenger_root,
        seed_start=40,
        expected_seeds=3,
    )
    assert report["contract_valid"] == {
        MODULE.CONTROL: True,
        MODULE.CHALLENGER: True,
    }
    assert report["terminal_certified_count"] == 9
    assert report["terminal_false_certificate_count"] == 0
    assert report["paired_performance"]["strict_gain_detected"] is True
    assert report["formal_gate_passed"] is True


def test_v63_gate_rejects_target_oracle_in_selector(tmp_path):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    _write_matrix(control_root, MODULE.CONTROL)
    _write_matrix(challenger_root, MODULE.CHALLENGER)
    path = (
        challenger_root
        / MODULE.CHALLENGER
        / MODULE.V56.DOMAINS[0]
        / "seed40"
        / "result.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["terminal_verification"]["frozen_shortlist"][1][
        "target_oracle_used"
    ] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = MODULE.analyze(
        control_root,
        challenger_root,
        seed_start=40,
        expected_seeds=3,
    )
    assert report["contract_valid"][MODULE.CHALLENGER] is False
    assert report["formal_gate_passed"] is False
