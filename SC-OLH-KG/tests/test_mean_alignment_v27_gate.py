import copy
import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v27_sequential_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v27_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v27_gate import (  # noqa: E402
    CHALLENGERS,
    MEAN_SCENARIOS,
    SCALED,
    VARIANTS,
    summarize,
)
from tests.test_mean_alignment_v26_gate import _row as _v26_row  # noqa: E402


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v27",
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


def test_v27_submitter_wires_one_aggregate_hyperlaw(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 6 * 3 * 5
    for variant in CHALLENGERS:
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3 * 5
        assert all("--N 20 --n0 10" in spec["cmd"] for spec in selected)
        assert all(
            "--observable-mean-descriptor-mode exchangeable_equivariant"
            in spec["cmd"] for spec in selected)
        assert all(
            "--source-constraint-mean-adaptation-mode "
            "sequential_aggregate_hyperlaw" in spec["cmd"]
            for spec in selected)
        assert all(
            "--source-constraint-mean-null-weight 0.0" in spec["cmd"]
            for spec in selected)
        assert all(
            "--hvd-source-task-weight-mode independent" in spec["cmd"]
            for spec in selected)
        expected = SCALED.get(variant, "none")
        assert all(
            "--source-constraint-mean-misspecification-mode "
            f"{expected}" in spec["cmd"] for spec in selected)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
               for spec in specs)
    assert all(not spec.get("checkpoint_dir") for spec in specs)


def _row(variant, domain, shock):
    source = {
        "v15_tanh_control": "v15_tanh_control",
        "v23_factorized_none": "v23_factorized_none",
        "v26_sequential_mixture": "exchangeable_none",
    }.get(variant, "exchangeable_none")
    row = copy.deepcopy(_v26_row(source, domain, shock))
    row["gate_variant"] = variant
    if variant not in CHALLENGERS:
        return row

    row["source_constraint_mean_null_weight"] = 0.0
    row["hvd_source_task_weight_mode"] = "independent"
    prior = row["gpr_numerics"][1]["source_parametric_prior"]
    prior.pop("component_deviation_diagnostics", None)
    prior.update({
        "adaptation_mode": "sequential_single_aggregate_hyperlaw",
        "prior_kind": "exchangeable_empirical_bayes_gaussian_hyperlaw",
        "target_task_law": "single_gaussian_draw",
        "source_domain_identity_marginalized": True,
        "source_components_retained_in_target_posterior": False,
        "within_source_covariance_included": True,
        "between_source_covariance_included": True,
        "single_aggregate_hyperlaw": True,
        "single_aggregate_component_count": 1,
        "component_names": ["source:aggregate"],
        "target_null_component_retained": False,
        "posterior_component_count": 1,
        "posterior_projection": "single_gaussian_identity_projection",
        "between_component_covariance_trace": 0.0,
        "posterior_target_data_used": True,
        "target_observation_count": 20,
        "online_mixture_update_count": 10,
        "target_oracle_used": False,
    })
    mode = SCALED.get(variant, "none")
    if variant in SCALED:
        prior.update({
            "source_mean_misspecification_mode": mode,
            "source_mean_misspecification_applied": True,
            "source_mean_misspecification_scale": 1.2,
            "source_mean_prior_covariance_trace_before": 2.0,
            "source_mean_prior_covariance_trace_after": 2.4,
            "source_mean_residual_floor_before": 0.2,
            "source_mean_residual_floor_after": 0.24,
            "misspecification_uncertainty_can_only_increase": True,
            "target_oracle_used_for_misspecification": False,
        })
    else:
        prior.update({
            "source_mean_misspecification_mode": "none",
            "source_mean_misspecification_applied": False,
            "source_mean_misspecification_scale": 1.0,
            "target_oracle_used_for_misspecification": False,
        })
    return row


def test_v27_analyzer_requires_single_hyperlaw_and_monotone_inflation():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in MEAN_SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 6 * 3
    assert result["post_gate_baseline_recommendation"] == (
        "exchangeable_aggregate_none")
    for variant in CHALLENGERS:
        checks = result["variant_checks"][variant]
        assert checks["exchangeable_target_linear_contract"]
        assert checks["single_empirical_bayes_hyperlaw_contract"]
        assert checks["source_mean_misspecification_contract"]
        assert checks["target_roles_differentiated_all_seeds"]
        assert checks["independent_variance_task_posterior_contract"]

    broken = copy.deepcopy(rows)
    target = next(
        row for row in broken
        if row["gate_variant"] == "exchangeable_aggregate_scale")
    target["gpr_numerics"][1]["source_parametric_prior"][
        "source_domain_identity_marginalized"] = False
    failed = summarize(broken, expected_seeds=1)
    assert not failed["variant_checks"]["exchangeable_aggregate_scale"][
        "single_empirical_bayes_hyperlaw_contract"]
