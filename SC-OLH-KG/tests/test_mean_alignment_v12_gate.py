import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v12_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v12_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v12_gate import (  # noqa: E402
    GEOMETRY_VARIANTS,
    SCENARIOS,
    TRANSFORM_VARIANTS,
    VARIANTS,
    summarize,
)


def _args(tmp_path):
    deploy = tmp_path / "deploy"
    defaults = submit.base.base.base.base.base
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v12",
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


def test_v12_submitter_builds_factored_transform_geometry_matrix(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 7 * 4 * 5
    combined = next(
        spec for spec in specs
        if "/bounded_geometry_mixture/" in spec["signature"])
    assert "--observable-mean-latent-transform source_tanh" in combined["cmd"]
    assert "--source-constraint-mean-null-geometry target_pool" in combined[
        "cmd"]
    assert "--hvd-source-task-weight-mode independent" in combined["cmd"]
    assert set(combined["allowed_nodes"]) == set(submit.CPU_NODES)


def test_v12_analyzer_accepts_source_only_bounded_geometry_challenger():
    rows = []
    for variant in VARIANTS:
        transformed = variant in TRANSFORM_VARIANTS
        geometric = variant in GEOMETRY_VARIANTS
        for domain, shock in SCENARIOS:
            factor = domain == "FactorShockStatePolicyRZDT1"
            transform_diagnostics = ({
                "status": "source_lodo_selected",
                "selected_temperature": 1.0,
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            } if transformed else {
                "status": "identity",
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            })
            null = {
                "name": "target:null",
                "null_geometry_mode": (
                    "target_pool" if geometric else "isotropic"),
                "target_labels_used_for_null_geometry": False,
                "target_oracle_used_for_null_geometry": False,
            }
            if geometric:
                null.update({
                    "average_predictive_scale_preserved": True,
                    "minimum_covariance_eigenvalue": 1e-6,
                    "target_geometry_pool_source": (
                        "deterministic_unlabeled_role_matching_pool"),
                })
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
                else 1
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
                        1.0 if transformed else 2.0),
                    "source_parametric_prior": {
                        "role_coordinate_selection": selection,
                        "component_deviation_diagnostics": [null],
                    },
                }],
                "meta_basis": {"1": {
                    "selected_coordinate_diagnostics": {
                        "latent_transform": (
                            "source_tanh" if transformed else "identity"),
                        "latent_transform_diagnostics": transform_diagnostics,
                        "alignment": {
                            "maximum_abs_source_latent_feature": (
                                0.95 if transformed else 2.0),
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
    assert result["row_count"] == 7 * 4
    assert "bounded_geometry_mixture" in result["sequential_gate_eligible"]
    assert all(result["variant_checks"][
        "bounded_geometry_mixture"].values())
