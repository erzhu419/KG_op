import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v18_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v18_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v18_gate import (  # noqa: E402
    MEAN_SCENARIOS,
    REQUESTED_RANK,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v17_gate import _row as _v17_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v18",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 10,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v18_submitter_splits_mean_gate_and_wires_residual_rank(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 5 * 3 * 5
    assert all("shock4" not in spec["signature"] for spec in specs)
    rank2 = next(
        spec for spec in specs
        if "/residual_rank2_quarter/" in spec["signature"])
    assert "--observable-mean-target-residual-rank 2" in rank2["cmd"]
    assert (
        "--observable-mean-target-residual-prior-scale 0.25"
        in rank2["cmd"])
    assert "--task-variance-posterior-mode replication_only" in rank2["cmd"]
    assert set(rank2["allowed_nodes"]) == set(submit.CPU_NODES)


def _row(variant, domain, shock):
    row = _v17_row(variant, domain, shock)
    row["gate_variant"] = variant
    rank = REQUESTED_RANK.get(variant, 0)
    row["meta_basis"]["1"]["target_orthogonal_residual"] = {
        "status": "disabled" if rank == 0 else "fit",
        "requested_rank": rank,
        "effective_rank": rank,
        "maximum_base_cross_moment": 1e-12,
        "target_labels_used": False,
        "target_oracle_used": False,
        "target_labels_used_to_define_coordinate": False,
        "target_oracle_used_to_define_coordinate": False,
    }
    return row


def test_v18_analyzer_requires_orthogonal_outcome_free_improvement():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 5 * 3
    assert set(result["sequential_gate_eligible"]) == set(VARIANTS[1:])
    assert all(
        all(result["variant_checks"][variant].values())
        for variant in VARIANTS[1:]
    )
