import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v22_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v22_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v22_gate import (  # noqa: E402
    CHALLENGERS,
    MEAN_SCENARIOS,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v21_gate import _row as _v21_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v22",
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


def test_v22_submitter_wires_source_geometry_prior(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 5 * 3 * 5
    for variant in CHALLENGERS:
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            "--observable-mean-role-assignment-prior source_geometry"
            in spec["cmd"] for spec in selected)
    strong = next(
        spec for spec in specs
        if "/geometry_loo_s025_t20/" in spec["signature"])
    weak = next(
        spec for spec in specs
        if "/geometry_loo_s100_t20/" in spec["signature"])
    assert (
        "--observable-mean-role-assignment-prior-temperature-scale 0.25"
        in strong["cmd"])
    assert (
        "--observable-mean-role-assignment-prior-temperature-scale 1.0"
        in weak["cmd"])
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _row(variant, domain, shock):
    if variant == "v15_tanh_control":
        return _v21_row(variant, domain, shock)
    if variant == "v21_uniform_loo_t20":
        row = _v21_row("assignment_loo_t20", domain, shock)
        row["gate_variant"] = variant
        audit = row["boundary_raw_pool_truth_diagnostics"][
            "boundary_raw_pool_role_assignment_oracle_expressivity"]
        audit["best_mae_assignment"] = "2-1"
        audit["best_rank_assignment"] = "2-1"
        return row

    source_variant = (
        "v20_assignment_marginal"
        if "marginal" in variant else "assignment_loo_t20"
    )
    row = _v21_row(source_variant, domain, shock)
    row["gate_variant"] = variant
    role = row["meta_basis"]["1"]["role_assignment_posterior"]
    role.update({
        "prior": "source_geometry",
        "assignment_prior_weights": [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
        "assignment_prior_diagnostics": {
            "status": "fit",
            "mode": "source_geometry",
            "effective_assignment_count": 1.55,
            "hard_assignment": "0-1",
            "maximum_prior_assignment": "0-1",
            "maximum_prior_matches_hard_assignment": True,
            "target_labels_used": False,
            "target_oracle_used": False,
            "permutation_equivariant": True,
        },
    })
    audit = row["boundary_raw_pool_truth_diagnostics"][
        "boundary_raw_pool_role_assignment_oracle_expressivity"]
    audit["best_mae_assignment"] = "0-1"
    audit["best_rank_assignment"] = "0-1"
    return row


def test_v22_analyzer_requires_geometry_prior_and_hvd_isolation():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 5 * 3
    for variant in CHALLENGERS:
        checks = result["variant_checks"][variant]
        assert checks["source_geometry_prior_contract"]
        assert checks["registered_structure_score_contract"]
        assert checks["independent_variance_task_posterior_contract"]
        assert checks["strict_assignment_selection_gain_over_uniform"]
