import importlib.util
from pathlib import Path
import sys


ANALYZE_PATH = (
    Path(__file__).resolve().parents[1]
    / "performance/analyze_mean_alignment_v35_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v35_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v35_confidence_split_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v35_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v35",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v35_submitter_separates_predictive_and_confidence_authority(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3 * 5
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            "--source-constraint-mean-misspecification-mode "
            f"{profile['misspecification']}" in spec["cmd"]
            for spec in selected)
        assert all(
            "--posterior-dominance-initialization risk" in spec["cmd"]
            for spec in selected)
        assert all(
            "--certification-head-authority split_gpr_cumulative_hvd"
            in spec["cmd"] for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def test_v35_contract_requires_central_and_robust_covariance_views():
    prior = {
        "source_mean_misspecification_mode": (
            "predictive_scale_sandwich_hc3_confidence"),
        "source_mean_misspecification_applied": True,
        "source_mean_misspecification_scale": 1.5,
        "source_mean_prior_scaled_before_conditioning": True,
        "source_mean_sandwich_applied": True,
        "source_mean_misspecification_application": (
            "source_prior_scale_then_posterior_sandwich"),
        "source_mean_posterior_mean_preserved": True,
        "source_mean_sandwich_covariance_trace": 0.2,
        "source_mean_posterior_covariance_trace_before": 0.8,
        "source_mean_posterior_covariance_trace_after": 1.0,
        "source_mean_sandwich_decision_authority": "confidence_only",
        "decision_covariance_available": True,
        "misspecification_uncertainty_can_only_increase": True,
        "target_oracle_used": False,
        "target_oracle_used_for_misspecification": False,
    }
    assert analyze.base._robust_posterior_contract(
        prior, "predictive_scale_sandwich_hc3_confidence")
    assert not analyze.base._robust_posterior_contract(
        dict(prior, source_mean_sandwich_decision_authority="joint_predictive"),
        "predictive_scale_sandwich_hc3_confidence",
    )


def test_v35_analyzer_uses_only_submitted_optional_arms():
    rows = [
        {"gate_variant": "v29_scale_control"},
        {"gate_variant": "v34_joint_control"},
        {"gate_variant": "v35_confidence"},
    ]
    assert analyze._present_variants(rows) == (
        "v29_scale_control",
        "v34_joint_control",
        "v35_confidence",
    )


def test_v35_online_contract_requires_canonical_paired_crn_trace():
    rows = []
    for heldout, shock in analyze.base.MEAN_SCENARIOS:
        for variant in ("v29_scale_control", "v35_confidence"):
            rows.append({
                "gate_variant": variant,
                "heldout": heldout,
                "target_shared_shock_scale": shock,
                "seed": 4,
                "N": 11,
                "n0": 10,
                "decision_backend": "sobol_new",
                "target_design_fingerprint": f"design:{heldout}",
                "online_action_sequence_fingerprint": f"online:{heldout}",
                "online_action_trace": [{
                    "x_fingerprint": f"x:{heldout}",
                    "candidate_source": "sobol_continuation",
                    "observed_response": [0.25, 0.5],
                }],
                "online_action_trace_target_oracle_used": False,
                "simulation_randomness_contract": {
                    "proposal_rng_independent": True,
                    "common_random_numbers_by_evaluation_index": True,
                    "target_oracle_used": False,
                },
                "proposal_randomness_contract": {
                    "component_streams_independent": True,
                    "simulation_rng_independent": True,
                    "target_oracle_used": False,
                },
            })
    contract = analyze._paired_online_contract(
        rows, ("v29_scale_control", "v35_confidence"), 1)
    assert all(contract.values())

    rows[-1]["online_action_trace"][0]["observed_response"] = [0.3, 0.5]
    broken = analyze._paired_online_contract(
        rows, ("v29_scale_control", "v35_confidence"), 1)
    assert broken["paired_common_random_responses"] is False
