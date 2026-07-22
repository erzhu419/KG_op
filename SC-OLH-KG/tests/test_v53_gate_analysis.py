from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
FIDELITY_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_v53_mc_fidelity_gate.py")
SENTINEL_PATH = (
    REPO
    / "SC-OLH-KG/performance/analyze_v53_constrained_certificate_gate.py"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FIDELITY = _load("v53_fidelity_analysis", FIDELITY_PATH)
SENTINEL = _load("v53_sentinel_analysis", SENTINEL_PATH)


def _write_result(root, variant, rows):
    path = root / variant / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "experiment_variant": f"gate/{variant}/shock0",
        "rows": rows,
    }), encoding="utf-8")


def _initial_design():
    return {
        "fingerprint": "same-target-design",
        "source_archive_fingerprint": "same-source-archive",
        "n_unique": 10,
        "target_labels_used": False,
        "target_oracle_used": False,
    }


def _fidelity_row(
    domain, seed, mc, offset, sampling_mode="antithetic_nested",
):
    return {
        "heldout": domain,
        "seed": seed,
        "implementation_contract_id": "v53_constrained_certificate_deficit",
        "theory_contract_id": "v53_constrained_certificate_deficit_v1",
        "exact_kg_mc_samples": mc,
        "exact_kg_sampling_mode": sampling_mode,
        "task_initial_design": _initial_design(),
        "decision_backend_contract": {
            "policy_improvement_contract": (
                "v53_constrained_certificate_deficit_v1"),
        },
        "online_action_trace_target_oracle_used": False,
        "online_action_trace": [{
            "exact_kg_active_action_fingerprints": ["a", "b", "c"],
            "exact_kg_selector_plan": {
                "mode": "factorized_rqmc_nested",
                "sample_count": mc,
                "finite_expert_count": 49,
                "selected_expert_count": min(mc, 49),
                "factorized_selector": True,
                "selector_l1_error": 0.2 if mc == 8 else 0.05,
            },
            "exact_kg_raw_scores_active": [
                0.1 + offset, 0.4 + offset, 0.2 + offset],
            "certificate_deficit_raw_scores_active": [
                0.2 + offset, 0.5 + offset, 0.1 + offset],
        }],
    }


def test_v53_fidelity_analyzer_matches_actions_and_calibrates_nonzero_eta(
    tmp_path,
):
    for variant, mc, offset in (
        (FIDELITY.LOW, 8, 0.01),
        (FIDELITY.HIGH, 32, 0.0),
    ):
        rows = [
            _fidelity_row(domain, 100, mc, offset)
            for domain in FIDELITY.DOMAINS
        ]
        _write_result(tmp_path, variant, rows)
    result = FIDELITY.analyze(tmp_path, seeds=[100], multiplier=1.25)
    assert result["paired_count"] == 3
    assert result["fidelity_gate_complete"] is True
    assert result["mc8_stable_enough_for_sentinel"] is True
    assert result["recommended_mc_samples"] == 8
    assert result["recommended_risk_eta"] > 0.0
    assert result["recommended_certificate_eta"] > 0.0
    assert result["bound_status"].startswith("empirical_")



def test_v53_fidelity_analyzer_normalizes_by_current_terminal_scale(
    tmp_path,
):
    sampling_mode = "factorized_rqmc_nested"
    for variant, mc, raw_offset in (
        (FIDELITY.LOW, 8, 1.0),
        (FIDELITY.HIGH, 32, 0.0),
    ):
        rows = []
        for domain in FIDELITY.DOMAINS:
            row = _fidelity_row(
                domain, 0, mc, 0.0, sampling_mode=sampling_mode)
            row["implementation_contract_id"] = (
                "v53_constrained_certificate_deficit_normalized")
            row["theory_contract_id"] = (
                "v53_constrained_certificate_deficit_v2")
            contract = row["decision_backend_contract"]
            contract["policy_improvement_contract"] = (
                "v53_constrained_certificate_deficit_v2")
            contract["policy_improvement_score_normalization"] = (
                "current_terminal")
            trace = row["online_action_trace"][0]
            trace["policy_improvement_score_normalization"] = (
                "current_terminal")
            trace["exact_kg_current_terminal_value"] = [100.0, 40.0]
            trace["certificate_deficit_current_value"] = 2.0
            trace["exact_kg_raw_scores_active"] = [
                10.0 + raw_offset,
                40.0 + raw_offset,
                20.0 + raw_offset,
            ]
            trace["certificate_deficit_raw_scores_active"] = [
                0.4 + 0.02 * raw_offset,
                1.0 + 0.02 * raw_offset,
                0.2 + 0.02 * raw_offset,
            ]
            rows.append(row)
        _write_result(tmp_path, variant, rows)

    result = FIDELITY.analyze(
        tmp_path,
        seeds=[0],
        multiplier=1.25,
        sampling_mode=sampling_mode,
        score_normalization="current_terminal",
    )

    assert result["fidelity_gate_complete"] is True
    assert result["contracts_ok"] is True
    assert result["normalization_scales_ok"] is True
    assert result["risk_score_scale_min"] == 100.0
    assert result["risk_score_scale_max"] == 100.0
    assert result["certificate_score_scale_min"] == 2.0
    assert result["certificate_score_scale_max"] == 2.0
    assert result["recommended_risk_eta"] == pytest.approx(0.0125)
    assert result["recommended_certificate_eta"] == pytest.approx(0.0125)


def test_v53_fidelity_analyzer_accepts_nested_expert_marginalization(
    tmp_path,
):
    sampling_mode = "stratified_expert_nested"
    for variant, mc, offset in (
        (FIDELITY.LOW, 8, 0.01),
        (FIDELITY.HIGH, 32, 0.0),
    ):
        rows = [
            _fidelity_row(
                domain, 0, mc, offset, sampling_mode=sampling_mode)
            for domain in FIDELITY.DOMAINS
        ]
        _write_result(tmp_path, variant, rows)
    result = FIDELITY.analyze(
        tmp_path,
        seeds=[0],
        multiplier=1.25,
        sampling_mode=sampling_mode,
    )
    assert result["fidelity_gate_complete"] is True
    assert result["sampling_mode"] == sampling_mode
    assert result["finite_expert_marginalization"] is True


def test_v53_fidelity_analyzer_accepts_nested_factorized_rqmc(
    tmp_path,
):
    sampling_mode = "factorized_rqmc_nested"
    for variant, mc, offset in (
        (FIDELITY.LOW, 8, 0.01),
        (FIDELITY.HIGH, 32, 0.0),
    ):
        rows = [
            _fidelity_row(
                domain, 0, mc, offset, sampling_mode=sampling_mode)
            for domain in FIDELITY.DOMAINS
        ]
        _write_result(tmp_path, variant, rows)
    result = FIDELITY.analyze(
        tmp_path,
        seeds=[0],
        multiplier=1.25,
        sampling_mode=sampling_mode,
    )
    assert result["fidelity_gate_complete"] is True
    assert result["sampling_mode"] == sampling_mode
    assert result["finite_expert_marginalization"] is False


def _sentinel_contract(variant):
    values = {
        SENTINEL.CONTROL: (
            "promoted_v51_observed_terminal_closure",
            "v51_statistical_closure_v2",
            "disabled_v51_compatible",
        ),
        SENTINEL.V52: (
            "v52_safeguarded_policy_improvement",
            "v52_safeguarded_closure_v1",
            "v52_safeguarded_policy_improvement_v1",
        ),
        SENTINEL.V53: (
            "v53_constrained_certificate_deficit_normalized",
            "v53_constrained_certificate_deficit_v2",
            "v53_constrained_certificate_deficit_v2",
        ),
    }
    return values[variant]


def _sentinel_row(domain, seed, variant):
    implementation, theory, policy = _sentinel_contract(variant)
    v53 = variant == SENTINEL.V53
    control = variant == SENTINEL.CONTROL
    return {
        "heldout": domain,
        "seed": seed,
        "implementation_contract_id": implementation,
        "theory_contract_id": theory,
        "decision_backend": "sobol_exact_joint_voi",
        "exact_kg_sampling_mode": "factorized_rqmc_nested",
        "task_initial_design": _initial_design(),
        "online_action_trace_target_oracle_used": False,
        "decision_backend_contract": {
            "policy_improvement_contract": policy,
            "policy_improvement_score_normalization": (
                "current_terminal" if v53 else "none"),
            "policy_improvement_mc_error_bound": 0.01 if v53 else 0.0,
            "policy_improvement_certificate_mc_error_bound": (
                0.02 if v53 else 0.0),
            "forced_sampling_override_count": 0,
        },
        "true_feasible": True,
        "feasible_simple_regret": 0.09 if v53 else 0.1,
        "adaptive_improves_initial_best": v53,
        "adaptive_loss": False,
        "certificate_outcome_audit": {
            "posterior_certified_count": 0,
            "false_certificate_count": 0,
            "posterior_certificate_vacuous": True,
        },
        "truth_pool_diagnostics": {
            "pool_has_true_safe_good_rate": (
                0.6 if v53 and domain == SENTINEL.DOMAINS[0] else 0.5),
            "mean_pool_min_posterior_margin": 0.2 if v53 else 0.3,
        },
        "variance_diagnostics": {
            "cumulative_statistical_design": {
                "1": {
                    "lean_normalized_excitation_kappa": (
                        0.2 if v53 and domain == SENTINEL.DOMAINS[0] else 0.1),
                },
            },
        },
        "variance_calibration_audit": {
            "certified_log_variance_rmse": 0.9 if v53 else 1.0,
        },
        "decision_backend_diagnostics": {
            "policy_improvement": {
                "one_step_switch_count": 1 if v53 else 0,
                "rollout_switch_count": 0,
            },
        },
    }


def test_v53_sentinel_analyzer_requires_all_preregistered_checks(tmp_path):
    for variant in SENTINEL.VARIANTS:
        rows = [
            _sentinel_row(domain, 0, variant)
            for domain in SENTINEL.DOMAINS
        ]
        _write_result(tmp_path, variant, rows)
    result = SENTINEL.analyze(tmp_path, seeds=[0])
    assert result["sentinel_eligible"] == [SENTINEL.V53]
    assert all(result["v53_checks"].values())
    assert result["promotion_eligible"] == []
