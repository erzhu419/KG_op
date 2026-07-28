from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "SC-OLH-KG/performance/analyze_v54_paired_difference_gate.py")
SPEC = importlib.util.spec_from_file_location("v54_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _trace(*, paired, mc, reference_risk=0.121):
    prefix_risk = [0.0, 0.01, 0.02, 0.03, 0.10]
    prefix_certificate = [0.0, -0.1, -0.1, -0.1, 0.10]
    risk = (
        [0.0, 0.01, 0.02, 0.03, 0.12]
        if mc == 128
        else [0.0, 0.01, 0.02, 0.03, reference_risk]
    )
    certificate = (
        [0.0, -0.1, -0.1, -0.1, 0.11]
        if mc == 128
        else [0.0, -0.1, -0.1, -0.1, 0.111]
    )
    trace = {
        "exact_kg_active_action_fingerprints": ["a", "b", "c", "d", "e"],
        "exact_kg_active_action_is_replicate": [False] * 5,
        "exact_kg_raw_scores_active": [1.0, 0.9, 0.8, 0.7, 2.0],
        "exact_kg_policy_scores_active": risk,
        "certificate_deficit_policy_scores_active": certificate,
    }
    if paired:
        trace.update({
            "pairwise_prefix_risk_policy_scores_active": prefix_risk,
            "pairwise_prefix_certificate_policy_scores_active": (
                prefix_certificate),
        })
    return trace


def _row(domain, *, paired, mc, reference_risk=0.121):
    return {
        "heldout": domain,
        "seed": 0,
        "implementation_contract_id": (
            "v54_paired_nested_difference_guard"
            if paired
            else "v53_constrained_certificate_deficit_bounded_gain"
        ),
        "theory_contract_id": (
            "v54_paired_difference_guard_v1"
            if paired
            else "v53_constrained_certificate_deficit_v3"
        ),
        "exact_kg_mc_samples": mc,
        "exact_kg_sampling_mode": "factorized_rqmc_nested",
        "online_action_trace_target_oracle_used": False,
        "decision_backend_contract": {
            "policy_improvement_score_transform": "bounded_current_gain",
            "policy_improvement_guard_mode": (
                "paired_nested_difference" if paired else "uniform_score"),
            "exact_kg_joint_terminal_reuse": bool(paired),
        },
        "task_initial_design": {
            "fingerprint": f"design-{domain}",
            "source_archive_fingerprint": "archive",
            "n_unique": 10,
        },
        "online_action_trace": [
            _trace(
                paired=paired,
                mc=mc,
                reference_risk=reference_risk,
            )
        ],
    }


def _write_result(root, variant, rows):
    path = root / variant / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "experiment_variant": f"gate/{variant}/all",
        "rows": rows,
    }), encoding="utf-8")


def test_action_support_requires_joint_supplemental_dominator_in_every_domain(
    tmp_path,
):
    _write_result(
        tmp_path,
        MODULE.ACTION_SUPPORT_VARIANT,
        [
            _row(domain, paired=False, mc=128)
            for domain in MODULE.DOMAINS
        ],
    )
    result = MODULE.analyze_action_support(tmp_path, seeds=(0,))
    assert result["action_support_gate_complete"]
    assert result["action_support_gate_pass"]
    assert result["supplemental_joint_dominator_cells"] == 3


def test_triple_fidelity_accepts_covered_nonregressing_switches(tmp_path):
    _write_result(
        tmp_path,
        MODULE.LOW_VARIANT,
        [
            _row(domain, paired=True, mc=128)
            for domain in MODULE.DOMAINS
        ],
    )
    _write_result(
        tmp_path,
        MODULE.REFERENCE_VARIANT,
        [
            _row(domain, paired=True, mc=512)
            for domain in MODULE.DOMAINS
        ],
    )
    result = MODULE.analyze_triple_fidelity(tmp_path, seeds=(0,))
    assert result["triple_fidelity_gate_complete"]
    assert result["triple_fidelity_gate_pass"]
    assert result["switch_count"] == 3
    assert result["selected_pair_coverage_rate_against_mc512"] == 1.0
    assert result["selected_reference_regression_count"] == 0


def test_triple_fidelity_rejects_reference_risk_flip(tmp_path):
    _write_result(
        tmp_path,
        MODULE.LOW_VARIANT,
        [
            _row(domain, paired=True, mc=128)
            for domain in MODULE.DOMAINS
        ],
    )
    _write_result(
        tmp_path,
        MODULE.REFERENCE_VARIANT,
        [
            _row(domain, paired=True, mc=512, reference_risk=-0.1)
            for domain in MODULE.DOMAINS
        ],
    )
    result = MODULE.analyze_triple_fidelity(tmp_path, seeds=(0,))
    assert result["triple_fidelity_gate_complete"]
    assert not result["triple_fidelity_gate_pass"]
    assert result["selected_reference_regression_count"] == 3
