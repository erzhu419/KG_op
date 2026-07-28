import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v37_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v37_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)

SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v37_cluster_hc3_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v37_submit", SUBMIT_PATH)
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
        "run_id": "mean-v37",
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
        "ram_mb": 4096,
    })()


def test_v37_submitter_isolates_clustered_replication_gate(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)
    for variant, profile in submit.VARIANTS.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3
        flag = (
            "--adaptive-replication-voi"
            if profile["adaptive_replication_voi"]
            else "--no-adaptive-replication-voi"
        )
        assert all(flag in spec["cmd"] for spec in selected)


def test_v37_cluster_contract_requires_actual_replicated_clusters():
    row = {
        "display_variant": "v37_cluster_rep4",
        "gpr_numerics": [{}, {"source_parametric_prior": {
            "source_mean_sandwich_clustered_replicates": True,
            "source_mean_sandwich_replicated_cluster_count": 2,
            "source_mean_sandwich_maximum_cluster_size": 3,
            "target_oracle_used_for_misspecification": False,
        }}],
    }
    assert analyze._cluster_contract([row], "v37_cluster_rep4")
    row["gpr_numerics"][1]["source_parametric_prior"][
        "source_mean_sandwich_replicated_cluster_count"] = 0
    assert not analyze._cluster_contract([row], "v37_cluster_rep4")
