from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "SC-OLH-KG/performance/analyze_v59_terminal_verification_gate.py"
)
SPEC = importlib.util.spec_from_file_location("v59_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(variant, domain, seed):
    control = variant == MODULE.CONTROL
    row = {
        "heldout": domain,
        "seed": seed,
        "N": MODULE.SEARCH_BUDGET,
        "n0": 10,
        "implementation_contract_id": (
            "promoted_v51_observed_terminal_closure"
            if control
            else "v59_frozen_policy_gaussian_verification"
        ),
        "theory_contract_id": (
            "v51_statistical_closure_v2"
            if control
            else "v59_gaussian_replication_certificate_v1"
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
                if control
                else "gaussian_student_t_chi_square"
            ),
            "terminal_verification_policy_frozen": not control,
            "terminal_verification_reuses_search_samples": False,
            "terminal_verification_changes_recommendation": False,
        },
        "target_design_fingerprint": f"design-{domain}-{seed}",
        "online_action_sequence_fingerprint": f"online-{domain}-{seed}",
        "x_recommended": [seed, 1],
        "true_feasible": True,
        "feasible_simple_regret": 0.1,
        "online_action_trace_target_oracle_used": False,
        "certificate_outcome_audit": {
            "posterior_certified_count": 0,
        },
        "algorithm_time_sec": 10.0,
        "finalization_timing_sec": {
            "terminal_fixed_policy_verification": 0.1,
        },
    }
    if control:
        row.update({
            "n_search_simulations": MODULE.SEARCH_BUDGET,
            "n_verification_simulations": 0,
            "n_simulations": MODULE.SEARCH_BUDGET,
            "terminal_verification": {
                "enabled": False,
                "status": "disabled",
            },
            "simulation_randomness_contract": {},
        })
    else:
        certified = seed == 0
        row.update({
            "n_search_simulations": MODULE.SEARCH_BUDGET,
            "n_verification_simulations": MODULE.VERIFICATION_BUDGET,
            "n_simulations": (
                MODULE.SEARCH_BUDGET + MODULE.VERIFICATION_BUDGET),
            "terminal_verification": {
                "enabled": True,
                "status": "certified" if certified else "not_certified",
                "certified": certified,
                "upper_margin": -0.01 if certified else 0.01,
                "verification_budget": MODULE.VERIFICATION_BUDGET,
                "sample_count": MODULE.VERIFICATION_BUDGET,
                "search_evaluation_count": MODULE.SEARCH_BUDGET,
                "total_evaluation_count": (
                    MODULE.SEARCH_BUDGET + MODULE.VERIFICATION_BUDGET),
                "delta": 0.05,
                "noise_model": "iid_gaussian",
                "policy_frozen_before_verification": True,
                "search_samples_reused": False,
                "posterior_updated_from_verification": False,
                "recommendation_changed": False,
                "verification_samples_logged": False,
                "target_oracle_used": False,
            },
            "simulation_randomness_contract": {
                "verification_stream_independent": True,
                "verification_evaluation_count": (
                    MODULE.VERIFICATION_BUDGET),
                "total_evaluation_count": (
                    MODULE.SEARCH_BUDGET + MODULE.VERIFICATION_BUDGET),
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


def test_v59_gate_requires_identical_search_and_nonvacuous_safe_certificate(
    tmp_path,
):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    _write_matrix(control_root, MODULE.CONTROL)
    _write_matrix(challenger_root, MODULE.CHALLENGER)
    report = MODULE.analyze(control_root, challenger_root)
    assert report["complete_5_seed_matrix"] is True
    assert report["contract_valid"] == {
        MODULE.CONTROL: True,
        MODULE.CHALLENGER: True,
    }
    assert report["search_identity"][
        "all_search_trajectories_identical"] is True
    assert report["terminal_false_certificate_count"] == 0
    assert all(
        report["terminal_certificate_nonvacuous_by_domain"].values())
    assert report["formal_gate_passed"] is True
    assert report["certification_gate_passed"] is True


def test_v59_gate_rejects_search_trajectory_change(tmp_path):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    _write_matrix(control_root, MODULE.CONTROL)
    _write_matrix(challenger_root, MODULE.CHALLENGER)
    path = next(challenger_root.rglob("result.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["target_design_fingerprint"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = MODULE.analyze(control_root, challenger_root)
    assert report["search_identity"][
        "all_search_trajectories_identical"] is False
    assert report["formal_gate_passed"] is False
