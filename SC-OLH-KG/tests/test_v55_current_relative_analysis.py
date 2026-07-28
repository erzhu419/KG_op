from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "SC-OLH-KG/performance/analyze_v55_current_relative_gate.py")
SPEC = importlib.util.spec_from_file_location("v55_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _trace(*, mc=128, selected="c", reference=False, unstable=False):
    prefix_risk = [0.8, 0.28, 0.39]
    prefix_certificate = [-0.2, 0.38, 0.34]
    risk = [0.8, 0.3, 0.4]
    certificate = [-0.2, 0.4, 0.35]
    if reference:
        risk = [0.8, 0.31, 0.405]
        certificate = [-0.2, 0.405, 0.355]
    if unstable:
        prefix_risk = [-0.8, -0.3, -0.4]
        prefix_certificate = [0.2, -0.4, -0.35]
    return {
        "x_fingerprint": selected,
        "exact_kg_active_action_fingerprints": ["a", "b", "c"],
        "exact_kg_active_action_is_replicate": [False, False, False],
        "exact_kg_active_action_labels": ["a-label", "b-label", "c-label"],
        "exact_kg_raw_scores_active": [1.0, 0.3, 0.4],
        "exact_kg_policy_scores_active": risk,
        "certificate_deficit_policy_scores_active": certificate,
        "pairwise_prefix_risk_policy_scores_active": prefix_risk,
        "pairwise_prefix_certificate_policy_scores_active": (
            prefix_certificate),
        "policy_improvement_risk_score_scale": 1.0,
        "policy_improvement_certificate_score_scale": 1.0,
    }


def _row(domain, *, mc, reference=False):
    return {
        "heldout": domain,
        "seed": 0,
        "implementation_contract_id": "v55_current_relative_joint_guard",
        "theory_contract_id": (
            "v55_current_relative_joint_improvement_v1"),
        "exact_kg_mc_samples": mc,
        "exact_kg_sampling_mode": "factorized_rqmc_nested",
        "online_action_trace_target_oracle_used": False,
        "decision_backend_contract": {
            "policy_improvement_score_transform": "bounded_current_gain",
            "policy_improvement_guard_mode": "paired_nested_absolute",
            "exact_kg_joint_terminal_reuse": True,
        },
        "task_initial_design": {
            "fingerprint": f"design-{domain}",
            "source_archive_fingerprint": "archive",
            "n_unique": 10,
        },
        "online_action_trace": [
            _trace(mc=mc, reference=reference)
        ],
    }


def _write_result(root, variant, rows):
    path = root / variant / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "experiment_variant": f"gate/{variant}/all",
        "rows": rows,
    }), encoding="utf-8")


def test_audit_reconstructs_current_relative_joint_selector():
    audit = MODULE.audit_trace(_trace(), multiplier=1.25)
    assert audit is not None
    assert audit["admissible_fingerprints"] == ["b", "c"]
    assert audit["expected_selected_fingerprint"] == "c"
    assert audit["selector_matches_trace"]
    assert audit["selected_is_joint_admissible"]
    assert audit["selected_joint_lcb"] == min(
        audit["selected_risk_lcb"], audit["selected_certificate_lcb"])


def test_audit_uses_literal_v51_fallback_when_all_bounds_are_empty():
    audit = MODULE.audit_trace(
        _trace(selected="a", unstable=True), multiplier=1.25)
    assert audit is not None
    assert audit["admissible_fingerprints"] == []
    assert audit["fallback_fingerprint"] == "a"
    assert audit["selector_matches_trace"]


def test_activation_requires_selected_joint_actions_in_every_domain(tmp_path):
    _write_result(
        tmp_path,
        MODULE.LOW_VARIANT,
        [_row(domain, mc=128) for domain in MODULE.DOMAINS],
    )
    result = MODULE.analyze_activation(
        tmp_path, seeds=(0,), minimum_selected_per_domain=1)
    assert result["activation_gate_complete"]
    assert result["activation_gate_pass"]
    assert result["selected_joint_admissible_total"] == 3


def test_triple_fidelity_accepts_covered_current_relative_actions(tmp_path):
    low = tmp_path / "low"
    reference = tmp_path / "reference"
    _write_result(
        low,
        MODULE.LOW_VARIANT,
        [_row(domain, mc=128) for domain in MODULE.DOMAINS],
    )
    _write_result(
        reference,
        MODULE.REFERENCE_VARIANT,
        [
            _row(domain, mc=512, reference=True)
            for domain in MODULE.DOMAINS
        ],
    )
    result = MODULE.analyze_triple_fidelity(
        low, reference, seeds=(0,))
    assert result["triple_fidelity_gate_complete"]
    assert result["triple_fidelity_gate_pass"]
    assert result["selected_joint_action_count"] == 3
    assert result["selected_pair_coverage_rate_against_mc512"] == 1.0
    assert result["selected_reference_regression_count"] == 0
