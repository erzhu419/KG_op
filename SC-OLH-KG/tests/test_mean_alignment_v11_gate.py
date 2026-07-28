import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v11_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v11_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v11_gate import (  # noqa: E402
    CHALLENGER,
    SCENARIOS,
    VARIANTS,
    summarize,
)
from performance.analyze_mean_alignment_v11_sequential_gate import (  # noqa: E402
    summarize as summarize_sequential,
)


def _args(tmp_path):
    deploy = tmp_path / "deploy"
    defaults = submit.base.base.base.base
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v11",
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


def test_v11_submitter_builds_support_adaptive_transfer_matrix(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 4 * 5
    challenger = next(
        spec for spec in specs
        if "/support_adaptive_aggregate/" in spec["signature"])
    assert (
        "--source-constraint-mean-adaptation-mode "
        "sequential_support_adaptive_aggregate_mixture" in challenger["cmd"])
    assert "--hvd-source-task-weight-mode independent" in challenger["cmd"]


def test_v11_analyzer_accepts_outcome_free_adaptation_switch(tmp_path):
    rows = []
    for variant in VARIANTS:
        challenger = variant == CHALLENGER
        for domain, shock in SCENARIOS:
            factor = domain == "FactorShockStatePolicyRZDT1"
            for seed in range(5):
                coordinate_selection = ({
                    "channel_cardinality_supported": not factor,
                    "selected_coordinate": (
                        "ordered" if factor else "role_aligned"),
                    "selection_uses_target_labels": False,
                    "selection_uses_target_oracle": False,
                    "target_labels_used": False,
                    "target_oracle_used": False,
                } if variant != "v4_mixture_control" else {})
                support_selection = ({
                    "channel_cardinality_supported": not factor,
                    "selection_uses_target_labels": False,
                    "selection_uses_target_oracle": False,
                    "target_labels_used": False,
                    "target_oracle_used": False,
                } if challenger else {})
                aggregate = bool(challenger and not factor)
                component_names = (
                    ["source:aggregate", "target:null"]
                    if aggregate else ["source:a", "source:b", "target:null"])
                component_rows = ([{
                    "name": "source:aggregate",
                    "aggregate_contains_within_source_uncertainty": True,
                    "aggregate_contains_between_source_disagreement": True,
                    "target_data_used_to_define_aggregate": False,
                    "target_oracle_used_to_define_aggregate": False,
                }, {"name": "target:null"}] if aggregate else [])
                prior = {
                    "adaptation_mode": (
                        (
                            "sequential_aggregate_target_evidence_mixture"
                            if aggregate
                            else "sequential_target_evidence_mixture"
                        ) if challenger
                        else "sequential_target_evidence_mixture"),
                    "aggregate_transferability_latent": aggregate,
                    "support_adaptive_aggregate_requested": challenger,
                    "support_adaptive_aggregate_selection": support_selection,
                    "effective_source_adaptation": (
                        "aggregate_latent" if aggregate else "domain_mixture"),
                    "prior_target_data_used": False,
                    "posterior_target_data_used": True,
                    "target_observation_count": 10,
                    "target_oracle_used": False,
                    "component_names": component_names,
                    "component_posterior_weights": (
                        [0.4, 0.6] if aggregate else [0.2, 0.2, 0.6]),
                    "component_deviation_diagnostics": component_rows,
                    "role_coordinate_selection": coordinate_selection,
                }
                rows.append({
                    "gate_variant": variant,
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {"admissible_oracle_free_transfer": True},
                    "true_feasible": True,
                    "feasible_simple_regret": 0.04,
                    "initial_has_true_feasible": True,
                    "initial_best_feasible_regret": 0.05,
                    "adaptive_rescue": False,
                    "adaptive_loss": False,
                    "adaptive_improves_initial_best": challenger,
                    "adaptive_regret_change": -0.01 if challenger else 0.0,
                    "posterior_certificate_vacuous": True,
                    "posterior_certified_evaluated_count": 0,
                    "false_certificate_count": 0,
                    "variance_log_rmse": 0.5,
                    "gpr_numerics": [{}, {"source_parametric_prior": prior}],
                    "task_initial_design": {
                        "fingerprint": f"{domain}:{shock}:{seed}"},
                    "source_target_adaptation_contract": {
                        "source_archive_fingerprint": f"archive:{domain}"},
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_false_certified_count": (
                            1 if challenger else 5),
                        "boundary_raw_pool_constraint_mean_median_abs_error": (
                            0.1 if challenger and not factor else 0.2),
                        "boundary_raw_pool_constraint_mean_rank_correlation":
                            0.8,
                        "boundary_raw_pool_chance_margin_rank_correlation":
                            0.8,
                    },
                })
    result = summarize(rows, expected_seeds=5, target_count=10)
    assert result["row_count"] == 60
    assert result["sequential_gate_eligible"] == [CHALLENGER]
    assert all(result["checks"].values())
    for row in rows:
        row["gpr_numerics"][1]["source_parametric_prior"][
            "target_observation_count"] = 20
    sequential = summarize_sequential(
        rows, expected_seeds=5, N=20, n0=10)
    assert sequential["promote_support_adaptive_aggregate"]
    assert all(sequential["checks"].values())
