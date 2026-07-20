import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v39_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v39_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)

SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v39_signed_voi_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v39_submit", SUBMIT_PATH)
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
        "run_id": "mean-v39",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 1,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 12,
        "ram_mb": 8192,
    })()


def test_v39_submitter_separates_clipped_and_signed_estimators(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3
    for variant, samples, signed in (
        ("v38_clipped_mc2", 2, False),
        ("v39_signed_mc2", 2, True),
        ("v39_signed_mc4", 4, True),
    ):
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3
        for spec in selected:
            command = spec["cmd"]
            assert f"--exact-mc-samples {samples}" in command
            assert "--exact-jobs 12" in command
            assert "--parallel-backend process_fork" in command
            if signed:
                assert "--no-exact-clip-negative" in command
            else:
                assert "--exact-clip-negative" in command


def test_v39_signed_contract_requires_unclipped_exact_refits():
    row = {
        "gate_variant": "v39_signed_mc2",
        "decision_backend_contract": {
            "backend": "sobol_exact_joint_voi",
        },
        "adaptive_replication_voi": {
            "exact_refit_action_value": True,
            "target_oracle_used": False,
        },
        "exact_kg_diagnostics": {
            "clip_negative": False,
            "ranking_uses_signed_values": True,
            "mc_samples": 2,
            "n_iterations": 10,
        },
    }
    assert analyze._signed_contract([row], "v39_signed_mc2", 2)
    row["exact_kg_diagnostics"]["clip_negative"] = True
    assert not analyze._signed_contract([row], "v39_signed_mc2", 2)
