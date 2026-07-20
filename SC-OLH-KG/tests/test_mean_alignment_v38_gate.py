import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v38_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v38_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)

SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v38_exact_refit_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v38_submit", SUBMIT_PATH)
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
        "run_id": "mean-v38",
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


def test_v38_submitter_uses_bounded_parallel_exact_refits(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)
    for variant in ("v38_exact_rep4", "v38_exact_rep8"):
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3
        for spec in selected:
            command = spec["cmd"]
            assert "--decision-backend sobol_exact_joint_voi" in command
            assert "--exact-mc-samples 2" in command
            assert "--exact-sampling-mode antithetic" in command
            assert "--exact-jobs 12" in command
            assert "--parallel-backend process_fork" in command
            assert "--adaptive-replication-voi" in command


def test_v38_exact_contract_requires_refit_and_oracle_free_action_value():
    row = {
        "gate_variant": "v38_exact_rep4",
        "decision_backend_contract": {
            "backend": "sobol_exact_joint_voi",
        },
        "adaptive_replication_voi": {
            "exact_refit_action_value": True,
            "unified_exact_voi": True,
            "target_oracle_used": False,
        },
        "exact_kg_diagnostics": {"n_iterations": 10},
    }
    assert analyze._exact_contract([row], "v38_exact_rep4")
    row["adaptive_replication_voi"]["exact_refit_action_value"] = False
    assert not analyze._exact_contract([row], "v38_exact_rep4")
