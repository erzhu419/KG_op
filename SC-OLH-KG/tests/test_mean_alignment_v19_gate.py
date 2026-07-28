import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v19_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v19_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v19_gate import (  # noqa: E402
    CHALLENGERS,
    MEAN_SCENARIOS,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v18_gate import _row as _v18_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v19",
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


def test_v19_submitter_wires_bayesian_rank_structure(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 5 * 3 * 5
    challenger = next(
        spec for spec in specs
        if "/rank_mixture_complexity/" in spec["signature"])
    assert "--observable-mean-target-residual-rank 2" in challenger["cmd"]
    assert (
        "--source-constraint-mean-residual-rank-posterior"
        in challenger["cmd"])
    assert (
        "--source-constraint-mean-residual-rank-prior 0.70,0.20,0.10"
        in challenger["cmd"])
    assert set(challenger["allowed_nodes"]) == set(submit.CPU_NODES)


def _row(variant, domain, shock):
    row = _v18_row(variant, domain, shock)
    row["gate_variant"] = variant
    if variant == "v15_tanh_control":
        return row
    row["meta_basis"]["1"]["target_orthogonal_residual"].update({
        "status": "fit",
        "requested_rank": 2,
        "effective_rank": 2,
    })
    if variant not in CHALLENGERS:
        return row
    prior = row["gpr_numerics"][1]["source_parametric_prior"]
    prior.update({
        "component_names": [
            "source:a|target_residual_rank=0",
            "source:a|target_residual_rank=1",
            "source:a|target_residual_rank=2",
        ],
        "component_prior_weights": [0.7, 0.2, 0.1],
        "component_posterior_weights": [0.2, 0.3, 0.5],
        "target_residual_rank_posterior_active": True,
        "target_residual_rank_posterior_mass": {
            "0": 0.2, "1": 0.3, "2": 0.5},
        "target_residual_rank_structured_mass": 1.0,
        "target_residual_rank_selected": 2,
        "target_residual_rank_target_labels_used_for_update": True,
        "target_residual_rank_target_oracle_used": False,
        "target_observation_count": 10,
    })
    return row


def test_v19_analyzer_requires_nonvacuous_oracle_free_rank_adaptation():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 5 * 3
    assert set(result["sequential_gate_eligible"]) == set(CHALLENGERS)
    assert all(
        all(result["variant_checks"][variant].values())
        for variant in CHALLENGERS
    )
