import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v15_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v15_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v15_gate import (  # noqa: E402
    CONTRAST_VARIANTS,
    ISOLATED_VARIANTS,
    SCENARIOS,
    VARIANTS,
    summarize,
)


def _defaults():
    value = submit
    while not hasattr(value, "DEFAULT_SOURCE_RUN_ID"):
        value = value.base
    return value


def _args(tmp_path):
    defaults = _defaults()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v15",
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


def test_v15_submitter_separates_mean_and_variance_task_posteriors(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 7 * 4 * 5
    isolated = next(
        spec for spec in specs
        if "/isolated_linear_contrast/" in spec["signature"])
    assert "--observable-mean-feature-mode linear" in isolated["cmd"]
    assert "--observable-mean-descriptor-mode role_adaptive_ordered" in isolated[
        "cmd"]
    assert "--source-constraint-mean-misspecification-mode source_contrast" in (
        isolated["cmd"])
    assert "--task-variance-posterior-mode replication_only" in isolated["cmd"]
    assert "--hvd-singleton-evidence-mode source_prior" in isolated["cmd"]
    assert set(isolated["allowed_nodes"]) == set(submit.CPU_NODES)


def test_v15_analyzer_requires_product_posterior_head_isolation():
    rows = []
    for variant in VARIANTS:
        isolated = variant in ISOLATED_VARIANTS
        contrast = variant in CONTRAST_VARIANTS
        transformed = "tanh" in variant or "bounded" in variant
        for domain, shock in SCENARIOS:
            factor = domain == "FactorShockStatePolicyRZDT1"
            source_component = {
                "name": "source:a",
                "source_mean_prior_covariance_trace_before": 1.0,
                "source_mean_prior_covariance_trace_after": (
                    2.0 if contrast else 1.0),
                "misspecification_uncertainty_can_only_increase": contrast,
                "source_contrast_rank": 1 if contrast else 0,
                "source_contrast_rank_bound": 1,
                "source_contrast_uses_target_data": False,
                "target_oracle_used_for_misspecification": False,
            }
            rows.append({
                "gate_variant": variant,
                "heldout": domain,
                "target_shared_shock_scale": shock,
                "seed": 0,
                "audit": {"admissible_oracle_free_transfer": True},
                "true_feasible": True,
                "feasible_simple_regret": 0.04,
                "variance_log_rmse": 0.5,
                "certified_variance_log_rmse": 0.6,
                "median_predicted_true_variance_ratio": 1.1,
                "median_certified_true_variance_ratio": 1.4,
                "variance_upper_coverage": 0.95,
                "variance_diagnostics": {
                    "global_var": {"0": 0.01, "1": 0.02},
                    "class_count": {"0": {"0": 5}, "1": {"0": 5}},
                    "cumulative_prior_scale": {"0": 0.01, "1": 0.02},
                    "cumulative_prior_scale_se": {"0": 0.0, "1": 0.0},
                    "cumulative_prior_component_weights": {
                        "0": [0.005, 0.005], "1": [0.01, 0.01]},
                    "cumulative_prior_upper_scale": {"0": 2.0, "1": 2.0},
                },
                "variance_calibration_audit": {
                    "posterior_source": (
                        "replication_variance_task_posterior_hvd_mixture"
                        if isolated else "task_posterior_hvd_mixture"),
                },
                "task_posterior": {
                    "variance_structure_posterior_mode": (
                        "replication_only" if isolated else "shared"),
                    "variance_structure_posterior": ({
                        "status": "frozen_source_prior",
                        "evidence_count": 0,
                        "effective_dof": 0,
                        "target_mean_used": False,
                    } if isolated else {
                        "status": "shared_with_mean_task_posterior",
                        "target_mean_used": True,
                    }),
                },
                "gpr_numerics": [{}, {
                    "max_abs_posterior_mean_seen": 1.0,
                    "source_parametric_prior": {
                        "role_coordinate_selection": {
                            "channel_cardinality_supported": not factor,
                            "selected_coordinate": (
                                "ordered" if factor else "role_aligned"),
                            "selection_uses_target_labels": False,
                            "selection_uses_target_oracle": False,
                            "target_labels_used": False,
                            "target_oracle_used": False,
                        },
                        "component_deviation_diagnostics": [
                            source_component,
                            {
                                "name": "target:null",
                                "null_geometry_mode": "isotropic",
                                "target_labels_used_for_null_geometry": False,
                                "target_oracle_used_for_null_geometry": False,
                            },
                        ],
                    },
                }],
                "meta_basis": {"1": {
                    "selected_coordinate_diagnostics": {
                        "feature_dim": 4,
                        "latent_transform": (
                            "source_tanh" if transformed else "identity"),
                        "latent_transform_diagnostics": {
                            "status": "source_lodo_selected"
                            if transformed else "identity",
                            "selected_temperature": 2.0
                            if transformed else None,
                            "selection_uses_target_data": False,
                            "selection_uses_target_oracle": False,
                        },
                        "alignment": {
                            "alignment_latent_dim": 4,
                            "maximum_abs_source_latent_feature": 1.0,
                        },
                    },
                }},
                "task_initial_design": {
                    "fingerprint": f"{domain}:{shock}:0"},
                "source_target_adaptation_contract": {
                    "source_archive_fingerprint": f"archive:{domain}"},
                "boundary_raw_pool_truth_diagnostics": {
                    "boundary_raw_pool_false_certified_count": (
                        10 if variant == "v4_shared_control"
                        else 5 if variant == "v8_shared_control"
                        else 1 if variant == "bounded_shared_control"
                        else 0),
                    "boundary_raw_pool_constraint_mean_median_abs_error": 0.2,
                    "boundary_raw_pool_constraint_mean_rank_correlation": 0.7,
                    "boundary_raw_pool_chance_margin_rank_correlation": 0.7,
                },
            })
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 7 * 4
    assert set(result["sequential_gate_eligible"]) == ISOLATED_VARIANTS
    assert all(
        all(result["variant_checks"][variant].values())
        for variant in ISOLATED_VARIANTS
    )
