import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v16_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v16_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v16_gate import (  # noqa: E402
    SCENARIOS,
    TRANSPORT_VARIANTS,
    UNCERTAINTY_VARIANTS,
    VARIANTS,
    summarize,
)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v16",
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


def test_v16_submitter_uses_partial_transport_and_product_posterior(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 7 * 4 * 5
    transport = next(
        spec for spec in specs
        if "/transport_uncertainty/" in spec["signature"])
    assert "--observable-mean-descriptor-mode role_transport" in transport["cmd"]
    assert (
        "--source-constraint-mean-role-epistemic-mode matching_uncertainty"
        in transport["cmd"])
    assert "--observable-mean-feature-mode linear" in transport["cmd"]
    assert "--observable-mean-latent-transform source_tanh" in transport["cmd"]
    assert "--task-variance-posterior-mode replication_only" in transport["cmd"]
    assert "--hvd-singleton-evidence-mode source_prior" in transport["cmd"]
    assert set(transport["allowed_nodes"]) == set(submit.CPU_NODES)


def _row(variant, domain, shock):
    transport = variant in TRANSPORT_VARIANTS
    uncertainty = variant in UNCERTAINTY_VARIANTS
    control = variant == "v15_tanh_control"
    alignment = {
        "status": "fit",
        "source_domains": ["source-a", "source-b"],
        "partial_transport": transport,
        "transport_selection": ({
            "status": "source_domain_dropout_selected",
            "selected_temperature": 0.5,
            "target_data_used": False,
            "target_oracle_used": False,
        } if transport else {"status": "disabled"}),
        "target_matches": ({
            "target": {
                "channel_count": 2 if domain.startswith("Factor") else 3,
                "transport_weights": (
                    [[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]]
                    if domain.startswith("Factor")
                    else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]]),
                "target_labels_used": False,
                "target_oracle_used": False,
            },
        } if transport else {}),
        "target_labels_used": False,
        "target_oracle_used": False,
    }
    calibration = {
        "mode": "matching_uncertainty" if uncertainty else "none",
        "source_role_trust": 1.0,
        "epistemic_covariance_scale": 2.0 if uncertainty else 1.0,
        "target_labels_used": False,
        "target_oracle_used": False,
    }
    source_component = {
        "name": "source:a",
        "role_matching_epistemic_covariance_scale": (
            2.0 if uncertainty else 1.0),
        "role_matching_uncertainty_monotone": uncertainty,
        "role_matching_target_labels_used": False,
        "role_matching_target_oracle_used": False,
    }
    return {
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
                "replication_variance_task_posterior_hvd_mixture"),
        },
        "task_posterior": {
            "variance_structure_posterior_mode": "replication_only",
            "variance_structure_posterior": {
                "status": "frozen_source_prior",
                "evidence_count": 0,
                "effective_dof": 0,
                "target_mean_used": False,
            },
        },
        "gpr_numerics": [{}, {
            "max_abs_posterior_mean_seen": 1.0,
            "source_parametric_prior": {
                "source_role_epistemic_calibration": calibration,
                "component_deviation_diagnostics": [
                    source_component,
                    {"name": "target:null"},
                ],
            },
        }],
        "meta_observable_mean_descriptor_mode": (
            "role_transport" if transport else "role_adaptive_ordered"),
        "meta_basis": {"1": {
            "observable_descriptor_mode": (
                "role_transport" if transport else "role_adaptive_ordered"),
            "alignment": {
                "channel_role_alignment": alignment,
                "alignment_latent_dim": 4,
                "maximum_abs_source_latent_feature": 1.0,
            },
            "latent_transform": "source_tanh",
            "latent_transform_diagnostics": {
                "status": "source_lodo_selected",
                "selected_temperature": 2.0,
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            },
        }},
        "task_initial_design": {"fingerprint": f"{domain}:{shock}:0"},
        "source_target_adaptation_contract": {
            "source_archive_fingerprint": f"archive:{domain}"},
        "boundary_raw_pool_truth_diagnostics": {
            "boundary_raw_pool_false_certified_count": 1 if control else 0,
            "boundary_raw_pool_full_certified_count": 2,
            "boundary_raw_pool_true_certified_count": 1 if control else 2,
            "boundary_raw_pool_constraint_mean_median_abs_error": 0.2,
            "boundary_raw_pool_constraint_mean_rank_correlation": 0.7,
            "boundary_raw_pool_chance_margin_rank_correlation": 0.7,
        },
    }


def test_v16_analyzer_requires_transport_calibration_without_vacuity():
    rows = [
        _row(variant, domain, shock)
        for variant in VARIANTS
        for domain, shock in SCENARIOS
    ]
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 7 * 4
    assert set(result["sequential_gate_eligible"]) == set(VARIANTS[1:])
    assert all(
        all(result["variant_checks"][variant].values())
        for variant in VARIANTS[1:]
    )
