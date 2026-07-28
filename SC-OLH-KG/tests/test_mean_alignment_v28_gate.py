import copy
import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v28_authority_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v28_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v28_gate import (  # noqa: E402
    AUTHORITIES,
    CERTIFICATE_SOURCES,
    MEAN_SCENARIOS,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v27_gate import _row as _v27_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v28",
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
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v28_submitter_wires_45_checkpoint_free_authority_shards(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 3 * 5
    for variant, authority in AUTHORITIES.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all("--N 20 --n0 10" in spec["cmd"] for spec in selected)
        assert all(
            f"--certification-head-authority {authority}" in spec["cmd"]
            for spec in selected)
        assert all(
            "--source-constraint-mean-adaptation-mode "
            "sequential_aggregate_hyperlaw" in spec["cmd"]
            for spec in selected)
        assert all("--runtime-checkpoint-dir ''" in spec["cmd"]
                   for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _row(variant, domain, shock):
    row = copy.deepcopy(_v27_row(
        "exchangeable_aggregate_none", domain, shock))
    row["gate_variant"] = variant
    authority = AUTHORITIES[variant]
    row["certification_head_authority"] = authority
    row["posterior_certification_source"] = CERTIFICATE_SOURCES[variant]
    row["certification_margin_decomposition"] = {
        "certification_head_authority": authority,
    }
    row["adaptive_loss"] = False
    audit = row["boundary_raw_pool_truth_diagnostics"]
    audit["boundary_raw_pool_false_certified_count"] = 0
    if variant == "v28_split_cumulative_hvd":
        audit["boundary_raw_pool_true_certified_count"] = 2
        audit["boundary_raw_pool_oracle_mean_variance_certified_count"] = 3
    else:
        audit["boundary_raw_pool_true_certified_count"] = 0
        audit["boundary_raw_pool_oracle_mean_variance_certified_count"] = 1
    return row


def test_v28_analyzer_promotes_only_consistent_improving_authority():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 3 * 3
    assert result["post_gate_baseline_recommendation"] == (
        "v28_split_cumulative_hvd")
    assert result["authority_gate_eligible"] == [
        "v28_split_task_hvd", "v28_split_cumulative_hvd"]
    assert result["promotion_eligible"] == [
        "v28_split_cumulative_hvd"]

    broken = copy.deepcopy(rows)
    target = next(
        row for row in broken
        if row["gate_variant"] == "v28_split_cumulative_hvd")
    target["certification_margin_decomposition"][
        "certification_head_authority"] = "task_joint"
    failed = summarize(broken, expected_seeds=1)
    assert not failed["variant_checks"]["v28_split_cumulative_hvd"][
        "authority_is_explicit_and_consistent"]
    assert failed["post_gate_baseline_recommendation"] is None
