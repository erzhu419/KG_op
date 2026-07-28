import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v14_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v14_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v14_gate import (  # noqa: E402
    CHALLENGERS,
    ISOLATED_VARIANTS,
    SCENARIOS,
    VARIANTS,
    summarize,
)


def _args(tmp_path):
    deploy = tmp_path / "deploy"
    defaults = submit.base.base.base.base.base.base.base
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v14",
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


def test_v14_submitter_isolates_singleton_hvd_evidence(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 6 * 4 * 5
    isolated = next(
        spec for spec in specs
        if "/isolated_tanh_mixture/" in spec["signature"])
    assert "--observable-mean-latent-transform source_tanh" in isolated["cmd"]
    assert "--hvd-cumulative-target-evidence-mode replication_only" in isolated[
        "cmd"]
    assert "--hvd-singleton-evidence-mode source_prior" in isolated["cmd"]
    assert "--hvd-source-task-weight-mode independent" in isolated["cmd"]
    assert set(isolated["allowed_nodes"]) == set(submit.CPU_NODES)


def test_v14_analyzer_requires_exact_variance_head_separation():
    rows = []
    for variant in VARIANTS:
        isolated = variant in ISOLATED_VARIANTS
        transformed = "tanh" in variant
        for domain, shock in SCENARIOS:
            factor = domain == "FactorShockStatePolicyRZDT1"
            selection = {
                "channel_cardinality_supported": not factor,
                "selected_coordinate": (
                    "ordered" if factor else "role_aligned"),
                "selection_uses_target_labels": False,
                "selection_uses_target_oracle": False,
                "target_labels_used": False,
                "target_oracle_used": False,
            }
            transform_diagnostics = ({
                "status": "source_lodo_selected",
                "selected_temperature": 2.0,
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            } if transformed else {
                "status": "identity",
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            })
            false_count = (
                10 if variant == "v4_mixture_control"
                else 5 if variant in {
                    "v8_mixture_control", "isolated_v8_mixture"}
                else 1
            )
            variance_diagnostics = {
                "singleton_evidence_mode": (
                    "source_prior" if isolated else "in_sample_residual"),
                "cumulative_target_evidence_mode": (
                    "replication_only" if isolated else "prequential_upper"),
                "source_prior_singleton_count": {
                    "0": 10 if isolated else 0,
                    "1": 10 if isolated else 0,
                },
                "replicated_solution_count": {"0": 0, "1": 0},
                "prequential_upper_solution_count": {"0": 0, "1": 0},
                "residual_square_tail": {
                    "0": {"effective_dof": 0.0 if isolated else 10.0},
                    "1": {"effective_dof": 0.0 if isolated else 10.0},
                },
                "cumulative_prior_target_weight": {"0": 0, "1": 0},
                "global_var": {"0": 0.01, "1": 0.02},
                "class_count": {"0": {"0": 5}, "1": {"0": 5}},
                "cumulative_prior_scale": {"0": 0.01, "1": 0.02},
                "cumulative_prior_scale_se": {"0": 0.0, "1": 0.0},
                "cumulative_prior_component_weights": {
                    "0": [0.005, 0.005], "1": [0.01, 0.01]},
                "cumulative_prior_upper_scale": {"0": 2.0, "1": 2.0},
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
                "hvd_source_task_weight_mode": "independent",
                "variance_diagnostics": variance_diagnostics,
                "gpr_numerics": [{}, {
                    "max_abs_posterior_mean_seen": (
                        1.0 if transformed else 2.0),
                    "source_parametric_prior": {
                        "role_coordinate_selection": selection,
                        "component_deviation_diagnostics": [{
                            "name": "target:null",
                            "null_geometry_mode": "isotropic",
                            "target_labels_used_for_null_geometry": False,
                            "target_oracle_used_for_null_geometry": False,
                        }],
                    },
                }],
                "meta_basis": {"1": {
                    "selected_coordinate_diagnostics": {
                        "feature_dim": 4,
                        "latent_transform": (
                            "source_tanh" if transformed else "identity"),
                        "latent_transform_diagnostics": transform_diagnostics,
                        "alignment": {
                            "alignment_latent_dim": 4,
                            "maximum_abs_source_latent_feature": (
                                1.0 if transformed else 2.0),
                        },
                    },
                }},
                "task_initial_design": {
                    "fingerprint": f"{domain}:{shock}:0"},
                "source_target_adaptation_contract": {
                    "source_archive_fingerprint": f"archive:{domain}"},
                "boundary_raw_pool_truth_diagnostics": {
                    "boundary_raw_pool_false_certified_count": false_count,
                    "boundary_raw_pool_constraint_mean_median_abs_error": 0.2,
                    "boundary_raw_pool_constraint_mean_rank_correlation": 0.7,
                    "boundary_raw_pool_chance_margin_rank_correlation": 0.7,
                },
            })
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 6 * 4
    assert "isolated_tanh_mixture" in result["sequential_gate_eligible"]
    assert all(result["variant_checks"][
        "isolated_tanh_mixture"].values())
