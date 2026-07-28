import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v21_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v21_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v21_gate import (  # noqa: E402
    CHALLENGERS,
    MEAN_SCENARIOS,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v20_gate import _row as _v20_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v21",
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


def test_v21_submitter_wires_cross_fitted_scores_and_temperatures(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 5 * 3 * 5
    temperatures = {
        "assignment_loo_t05": "0.5",
        "assignment_loo_t10": "1.0",
        "assignment_loo_t20": "2.0",
    }
    for variant, temperature in temperatures.items():
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all(
            "--source-constraint-mean-structure-score-mode loo_predictive"
            in spec["cmd"] for spec in selected)
        assert all(
            "--source-constraint-mean-evidence-temperature " + temperature
            in spec["cmd"] for spec in selected)
        assert all("--runtime-checkpoint-interval 0" in spec["cmd"]
                   for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _row(variant, domain, shock):
    if variant == "v15_tanh_control":
        return _v20_row(variant, domain, shock)
    row = _v20_row("role_assignment_plain", domain, shock)
    row["gate_variant"] = variant
    prior = row["gpr_numerics"][1]["source_parametric_prior"]
    audit = row["boundary_raw_pool_truth_diagnostics"][
        "boundary_raw_pool_role_assignment_oracle_expressivity"]
    if variant == "v20_assignment_marginal":
        prior.update({
            "structure_score_mode": "marginal_likelihood",
            "structure_score_cross_fitted": False,
        })
        audit["best_mae_assignment"] = "2-1"
        audit["best_rank_assignment"] = "2-1"
        return row

    component_count = len(prior["component_names"])
    prior.update({
        "structure_score_mode": "loo_predictive",
        "structure_score_cross_fitted": True,
        "target_oracle_used": False,
        "target_oracle_used_for_structure_score": False,
        "component_loo_predictive_diagnostics": [
            {
                "name": name,
                "loo_count": 10,
                "loo_mean_log_score": -1.0,
                "loo_median_abs_standardized_residual": 0.5,
                "target_oracle_used_for_structure_score": False,
            }
            for name in prior["component_names"]
        ],
    })
    assert len(prior["component_loo_predictive_diagnostics"]) == component_count
    return row


def test_v21_analyzer_requires_cross_fitted_oracle_free_structure_scores():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 5 * 3
    for variant in CHALLENGERS:
        checks = result["variant_checks"][variant]
        assert checks["cross_fitted_predictive_structure_contract"]
        assert checks["finite_role_assignment_posterior_contract"]
        assert checks["independent_variance_task_posterior_contract"]
        assert checks["strict_assignment_selection_gain_over_marginal"]
