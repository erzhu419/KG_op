import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v13_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v13_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v13_gate import (  # noqa: E402
    CHALLENGERS,
    RESIDUAL_VARIANTS,
    SCENARIOS,
    VARIANTS,
    summarize,
)


def _args(tmp_path):
    deploy = tmp_path / "deploy"
    defaults = submit.base.base.base.base.base.base
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v13",
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


def test_v13_submitter_builds_one_seed_source_support_shards(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 8 * 4 * 5
    residual = next(
        spec for spec in specs
        if "/support_residual_mixture/" in spec["signature"])
    assert (
        "--observable-mean-latent-transform source_support_residual"
        in residual["cmd"])
    assert "--source-constraint-mean-null-geometry isotropic" in residual[
        "cmd"]
    assert "--hvd-source-task-weight-mode independent" in residual["cmd"]
    assert set(residual["allowed_nodes"]) == set(submit.CPU_NODES)


def test_v13_analyzer_accepts_source_only_residual_coordinate():
    rows = []
    for variant in VARIANTS:
        if variant in CHALLENGERS:
            residual = variant in RESIDUAL_VARIANTS
            transform = (
                "source_support_residual" if residual
                else "source_support_clip")
            transform_diagnostics = {
                "status": "source_lodo_selected",
                "selected_quantile": 0.95,
                "support_bounds": [1.0, 1.1, 1.2, 1.3],
                "residual_channel": residual,
                "residual_channel_index": 4 if residual else None,
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            }
        elif variant == "bounded_tanh_control":
            residual = False
            transform = "source_tanh"
            transform_diagnostics = {
                "status": "source_lodo_selected",
                "selected_temperature": 2.0,
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            }
        else:
            residual = False
            transform = "identity"
            transform_diagnostics = {
                "status": "identity",
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            }
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
            false_count = (
                10 if variant == "v4_mixture_control"
                else 5 if variant in {
                    "v8_mixture_control", "v11_aggregate_control"}
                else 1 if variant == "bounded_tanh_control" else 2
            )
            rows.append({
                "gate_variant": variant,
                "heldout": domain,
                "target_shared_shock_scale": shock,
                "seed": 0,
                "audit": {"admissible_oracle_free_transfer": True},
                "true_feasible": True,
                "feasible_simple_regret": 0.04,
                "variance_log_rmse": 0.5,
                "gpr_numerics": [{}, {
                    "max_abs_posterior_mean_seen": (
                        1.0 if variant in CHALLENGERS else 2.0),
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
                        "feature_dim": 5 if residual else 4,
                        "latent_transform": transform,
                        "latent_transform_diagnostics": transform_diagnostics,
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
                    "boundary_raw_pool_false_certified_count": false_count,
                    "boundary_raw_pool_constraint_mean_median_abs_error": 0.2,
                    "boundary_raw_pool_constraint_mean_rank_correlation": 0.7,
                    "boundary_raw_pool_chance_margin_rank_correlation": 0.7,
                },
            })
    result = summarize(rows, expected_seeds=1)
    assert result["row_count"] == 8 * 4
    assert "support_residual_mixture" in result["sequential_gate_eligible"]
    assert all(result["variant_checks"][
        "support_residual_mixture"].values())
