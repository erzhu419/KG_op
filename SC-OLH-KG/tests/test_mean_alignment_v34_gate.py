import importlib.util
from pathlib import Path
import sys


ANALYZE_PATH = (
    Path(__file__).resolve().parents[1]
    / "performance/analyze_mean_alignment_v34_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v34_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v34_robust_eb_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v34_submit", SUBMIT_PATH)
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
        "run_id": "mean-v34",
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


def test_v34_submitter_pairs_mean_adaptation_and_sandwich_uncertainty(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3 * 5
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            "--certification-head-authority split_gpr_cumulative_hvd"
            in spec["cmd"] for spec in selected)
        assert all(
            "--posterior-dominance-enabled" in spec["cmd"]
            for spec in selected)
        assert all(
            "--posterior-dominance-initialization risk" in spec["cmd"]
            for spec in selected)
        assert all(
            "--source-constraint-mean-misspecification-mode "
            f"{profile['misspecification']}" in spec["cmd"]
            for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def test_v34_robust_posterior_contract_requires_both_update_layers():
    valid = {
        "source_mean_misspecification_mode": (
            "predictive_scale_sandwich_hc3"),
        "source_mean_misspecification_applied": True,
        "source_mean_misspecification_scale": 2.0,
        "source_mean_prior_scaled_before_conditioning": True,
        "source_mean_sandwich_applied": True,
        "source_mean_misspecification_application": (
            "source_prior_scale_then_posterior_sandwich"),
        "source_mean_posterior_mean_preserved": True,
        "source_mean_sandwich_covariance_trace": 0.5,
        "source_mean_posterior_covariance_trace_before": 1.0,
        "source_mean_posterior_covariance_trace_after": 1.5,
        "source_mean_sandwich_decision_authority": "joint_predictive",
        "misspecification_uncertainty_can_only_increase": True,
        "target_oracle_used": False,
        "target_oracle_used_for_misspecification": False,
    }
    assert analyze._robust_posterior_contract(
        valid, "predictive_scale_sandwich_hc3")
    assert not analyze._robust_posterior_contract(
        dict(valid, source_mean_sandwich_applied=False),
        "predictive_scale_sandwich_hc3",
    )
    assert not analyze._robust_posterior_contract(
        dict(valid, source_mean_prior_scaled_before_conditioning=False),
        "predictive_scale_sandwich_hc3",
    )
