from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "SC-OLH-KG/performance/analyze_v56_independent_confirmation_gate.py"
)
SPEC = importlib.util.spec_from_file_location("v56_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(variant, domain, seed):
    control = variant == MODULE.CONTROL
    row = {
        "heldout": domain,
        "seed": seed,
        "implementation_contract_id": (
            "promoted_v51_observed_terminal_closure"
            if control else "v56_independent_confirmation_guard"
        ),
        "theory_contract_id": (
            "v51_statistical_closure_v2"
            if control
            else "v56_independent_confirmation_finite_look_v1"
        ),
        "decision_backend_contract": {
            "policy_improvement_guard_mode": (
                "uniform_score" if control else "independent_confirmation"),
            "policy_improvement_score_transform": (
                "identity" if control else "bounded_current_gain"),
            "policy_improvement_confirmation_samples": (
                2048 if variant.endswith("2048") else 4096),
        },
        "online_action_trace_target_oracle_used": False,
        "online_action_trace": [],
        "true_feasible": True,
        "feasible_simple_regret": 0.1 if control else 0.05,
        "x_recommended": [seed] if control else [seed + 10],
        "adaptive_improves_initial_best": True,
        "adaptive_loss": False,
        "certificate_outcome_audit": {
            "false_certificate_count": 0,
            "posterior_certified_count": 1,
            "posterior_certificate_vacuous": False,
        },
        "initialization_time_sec": 2.0,
        "finalization_time_sec": 3.0,
    }
    if not control:
        row["online_action_trace"] = [{
            "policy_improvement_pairwise_audit": {"switched": True},
            "policy_improvement_confirmation": {
                "passed": True,
                "sample_count": 512,
                "risk_first_crossing_sample": 100,
                "certificate_first_crossing_sample": 200,
                "risk_sample_mean": 0.1,
                "certificate_sample_mean": 0.2,
                "time_sec": 4.0,
                "pilot_stream_independent": True,
                "simulation_stream_independent": True,
                "target_oracle_used": False,
            }
        }]
    return row


def test_v56_complete_paired_matrix_passes(tmp_path):
    for variant in MODULE.KNOWN:
        for domain in MODULE.DOMAINS:
            for seed in range(5):
                path = tmp_path / variant / domain / f"seed{seed}/result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "experiment_variant": f"gate/{variant}/{domain}/seed{seed}",
                    "rows": [_row(variant, domain, seed)],
                }), encoding="utf-8")
    report = MODULE.analyze(tmp_path)
    assert report["complete_5_seed_matrix"] is True
    assert report["paired_keys_match_control"] is True
    assert report["contract_valid"] == {
        variant: True for variant in MODULE.KNOWN
    }
    assert report["formal_gate_passed"] is True
    assert report["promotion_gate_passed"] is True
    assert report["gate_passed"] is True
    assert report["paired_performance"]["v56_confirm4096"] == {
        "paired_count": 15,
        "win_count": 15,
        "loss_count": 0,
        "tie_count": 0,
        "feasibility_rescue_count": 0,
        "feasibility_loss_count": 0,
        "recommendation_change_count": 15,
        "all_recommendations_identical": False,
        "performance_noninferior": True,
        "strict_gain_detected": True,
    }
    assert report["summaries"]["v56_confirm4096"][
        "QueueResourceControl"]["joint_confirmation_pass_count"] == 5

    subset = MODULE.analyze(
        tmp_path, challengers=("v56_confirm4096",))
    assert subset["challengers"] == ["v56_confirm4096"]
    assert subset["row_count"] == 30
    assert set(subset["summaries"]) == {
        MODULE.CONTROL, "v56_confirm4096"}


def test_v56_contract_accepts_only_the_declared_zero_sample_skip():
    row = _row(
        "v56_confirm4096",
        MODULE.DOMAINS[0],
        0,
    )
    confirmation = row["online_action_trace"][0][
        "policy_improvement_confirmation"]
    confirmation.clear()
    confirmation.update({
        "passed": False,
        "sample_count": 0,
        "status": "skipped_nonpositive_pilot_joint_score",
        "target_oracle_used": False,
    })
    assert MODULE._contract(row, "v56_confirm4096") is True

    confirmation["status"] = "missing_independent_stream"
    assert MODULE._contract(row, "v56_confirm4096") is False

    confirmation["status"] = "skipped_nonpositive_pilot_joint_score"
    confirmation["sample_count"] = 1
    assert MODULE._contract(row, "v56_confirm4096") is False
