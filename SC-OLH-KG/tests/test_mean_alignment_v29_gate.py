import importlib.util
from pathlib import Path
import sys


ANALYZE_PATH = (
    Path(__file__).resolve().parents[1]
    / "performance/analyze_mean_alignment_v29_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v29_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v29_calibration_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v29_submit", SUBMIT_PATH)
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
        "run_id": "mean-v29",
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


def test_v29_submitter_wires_calibration_and_dominance(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3 * 5
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            "--certification-head-authority split_gpr_cumulative_hvd"
            in spec["cmd"] for spec in selected)
        dominance_flag = (
            "--posterior-dominance-enabled"
            if profile["posterior_dominance_enabled"]
            else "--no-posterior-dominance-enabled"
        )
        assert all(dominance_flag in spec["cmd"] for spec in selected)
        assert all(
            "--posterior-dominance-delta 0.05" in spec["cmd"]
            for spec in selected)
        misspecification = profile.get("misspecification", "none")
        assert all(
            "--source-constraint-mean-misspecification-mode "
            f"{misspecification}" in spec["cmd"]
            for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def test_v29_dominance_contract_rejects_oracle_and_missing_terminal():
    valid = {
        "posterior_dominance_enabled": True,
        "posterior_dominance_terminal_used": True,
        "posterior_dominance_switch_count": 0,
        "posterior_dominance": {
            "enabled": True,
            "target_oracle_used": False,
            "method": "cantelli_covariance_free",
            "incumbent": [1, 2],
            "delta_switch": 0.05,
            "switch_count": 0,
            "history": [{"status": "initialized", "target_oracle_used": False}],
        },
    }
    assert analyze._dominance_record_contract(valid, True)
    invalid = dict(valid)
    invalid["posterior_dominance"] = dict(
        valid["posterior_dominance"], target_oracle_used=True)
    assert not analyze._dominance_record_contract(invalid, True)
    invalid = dict(valid, posterior_dominance_terminal_used=False)
    assert not analyze._dominance_record_contract(invalid, True)


def test_v29_misspecification_contract_is_monotone_and_oracle_free():
    prior = {
        "source_mean_misspecification_mode": "predictive_scale",
        "source_mean_misspecification_applied": True,
        "source_mean_misspecification_scale": 2.0,
        "source_mean_prior_covariance_trace_before": 1.0,
        "source_mean_prior_covariance_trace_after": 2.0,
        "source_mean_residual_floor_before": 0.1,
        "source_mean_residual_floor_after": 0.2,
        "misspecification_uncertainty_can_only_increase": True,
        "target_oracle_used": False,
        "target_oracle_used_for_misspecification": False,
    }
    assert analyze._misspecification_contract(prior, "predictive_scale")
    assert not analyze._misspecification_contract(
        dict(prior, source_mean_prior_covariance_trace_after=0.5),
        "predictive_scale",
    )
