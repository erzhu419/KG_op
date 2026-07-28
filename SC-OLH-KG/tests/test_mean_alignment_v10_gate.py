import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v10_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v10_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v10_gate import (  # noqa: E402
    CHALLENGERS,
    MISSPECIFICATION,
    SCENARIOS,
    VARIANTS,
    summarize,
)


def _args(tmp_path):
    deploy = tmp_path / "deploy"
    defaults = submit.base.base.base
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v10",
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
        "python": defaults.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v10_submitter_builds_aggregate_latent_matrix(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 5 * 4 * 5
    challenger = next(
        spec for spec in specs if "/aggregate_latent/" in spec["signature"])
    assert (
        "--source-constraint-mean-adaptation-mode "
        "sequential_aggregate_mixture" in challenger["cmd"])
    assert "--hvd-source-task-weight-mode independent" in challenger["cmd"]
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)


def test_v10_analyzer_accepts_source_only_aggregate_latent(tmp_path):
    rows = []
    for variant in VARIANTS:
        challenger = variant in CHALLENGERS
        mode = MISSPECIFICATION.get(variant, "none")
        for domain, shock in SCENARIOS:
            factor = domain == "FactorShockStatePolicyRZDT1"
            for seed in range(5):
                selection = ({
                    "channel_cardinality_supported": not factor,
                    "selected_coordinate": (
                        "ordered" if factor else "role_aligned"),
                    "selection_uses_target_labels": False,
                    "selection_uses_target_oracle": False,
                    "target_labels_used": False,
                    "target_oracle_used": False,
                } if variant != "v4_mixture_control" else {})
                component = {
                    "name": "source:aggregate",
                    "aggregate_contains_within_source_uncertainty": True,
                    "aggregate_contains_between_source_disagreement": True,
                    "target_data_used_to_define_aggregate": False,
                    "target_oracle_used_to_define_aggregate": False,
                    "source_mean_misspecification_applied": bool(
                        challenger and mode != "none"),
                    "source_mean_prior_covariance_trace_before": 1.0,
                    "source_mean_prior_covariance_trace_after": 1.5,
                    "source_mean_residual_floor_before": 0.1,
                    "source_mean_residual_floor_after": 0.2,
                    "misspecification_uncertainty_can_only_increase": True,
                }
                prior = {
                    "adaptation_mode": (
                        "sequential_aggregate_target_evidence_mixture"
                        if challenger
                        else "sequential_target_evidence_mixture"),
                    "aggregate_transferability_latent": challenger,
                    "prior_target_data_used": False,
                    "posterior_target_data_used": True,
                    "target_observation_count": 10,
                    "target_oracle_used": False,
                    "source_mean_misspecification_mode": mode,
                    "component_names": (
                        ["source:aggregate", "target:null"]
                        if challenger else ["source:a", "target:null"]),
                    "component_posterior_weights": [0.4, 0.6],
                    "component_deviation_diagnostics": (
                        [component, {"name": "target:null"}]
                        if challenger else []),
                    "role_coordinate_selection": selection,
                }
                rows.append({
                    "gate_variant": variant,
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {"admissible_oracle_free_transfer": True},
                    "true_feasible": True,
                    "feasible_simple_regret": 0.04,
                    "variance_log_rmse": 0.5,
                    "gpr_numerics": [{}, {"source_parametric_prior": prior}],
                    "task_initial_design": {
                        "fingerprint": f"{domain}:{shock}:{seed}"},
                    "source_target_adaptation_contract": {
                        "source_archive_fingerprint": f"archive:{domain}"},
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_false_certified_count": (
                            1 if challenger else 5),
                        "boundary_raw_pool_constraint_mean_median_abs_error":
                            0.2,
                        "boundary_raw_pool_constraint_mean_rank_correlation":
                            0.8,
                        "boundary_raw_pool_chance_margin_rank_correlation":
                            0.8,
                    },
                })
    result = summarize(rows, expected_seeds=5, target_count=10)
    assert result["row_count"] == 100
    assert set(result["sequential_gate_eligible"]) == set(CHALLENGERS)
    assert all(all(result["variant_checks"][variant].values())
               for variant in CHALLENGERS)
