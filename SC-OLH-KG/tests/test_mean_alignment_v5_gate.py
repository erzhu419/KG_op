import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPT = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v5_offline_gate_scheduler.py")
SPEC = importlib.util.spec_from_file_location("mean_v5_submit", SUBMIT_SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v5_offline_gate import (  # noqa: E402
    MISSPEC_VARIANTS,
    ROLE_VARIANTS,
    SCENARIOS,
    VARIANTS,
    load_rows,
    summarize,
)
from performance.analyze_mean_alignment_v5_sequential_gate import (  # noqa: E402
    load_rows as load_sequential_rows,
    summarize as summarize_sequential,
)


def _args(tmp_path, **overrides):
    deploy = tmp_path / "deploy"
    values = {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": submit.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v5",
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
        "python": submit.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
        "sequential": False,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_v5_submitter_builds_checkpoint_free_six_arm_matrix(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 6 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--runtime-checkpoint-dir ''" in spec["cmd"] for spec in specs)
    role = next(spec for spec in specs
                if "/v5_role_directional/" in spec["signature"])
    assert "--observable-mean-descriptor-mode role_aligned" in role["cmd"]
    assert "--observable-mean-feature-mode linear" in role["cmd"]
    assert (
        "--source-constraint-mean-misspecification-mode "
        "predictive_scale_directional"
    ) in role["cmd"]
    assert "--observable-variance-input-mode observable_state_exposure" in (
        role["cmd"])


def test_v5_submitter_builds_selected_sequential_gate(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        variants="v4_latent_control,misspec_scalar",
        N=20,
        n0=10,
        sequential=True,
    ))
    assert len(specs) == 2 * 4 * 5
    assert all("/mean_v5_sequential/" in spec["signature"]
               for spec in specs)
    assert all("--N 20 --n0 10" in spec["cmd"] for spec in specs)


def _write_matrix(root, *, challenger_improves=True):
    index = 0
    for variant in VARIANTS:
        for domain, shock in SCENARIOS:
            for seed in range(5):
                is_role = variant in ROLE_VARIANTS
                is_misspec = variant in MISSPEC_VARIANTS
                false_count = 4 if variant == "v4_latent_control" else (
                    2 if challenger_improves else 4)
                component = {
                    "name": "source:a",
                    "source_mean_misspecification_scale": (
                        2.0 if is_misspec else 1.0),
                    "source_mean_misspecification_directional_mass": 0.1,
                    "misspecification_uncertainty_can_only_increase": True,
                    "source_mean_prior_covariance_trace_before": 1.0,
                    "source_mean_prior_covariance_trace_after": 2.0,
                    "source_mean_residual_floor_before": 0.1,
                    "source_mean_residual_floor_after": 0.2,
                }
                row = {
                    "experiment_variant": (
                        f"mean_v5_offline/{variant}/shock{shock:g}"),
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {"admissible_oracle_free_transfer": True},
                    "mean_risk_coordinate_contract": {
                        "shared_observable_exposure_input": True,
                        "channel_role_alignment_used": is_role,
                        "channel_role_target_matching_uses_labels": False,
                        "channel_role_target_matching_uses_oracle": False,
                    },
                    "variance_log_rmse": 0.50,
                    "variance_upper_coverage": 0.95,
                    "gpr_numerics": [{}, {
                        "source_parametric_prior": {
                            "source_posterior_weight": 0.5,
                            "target_only_posterior_weight": 0.5,
                            "component_deviation_diagnostics": [component],
                        },
                    }],
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_has_true_feasible": True,
                        "boundary_raw_pool_constraint_mean_rank_correlation": 0.75,
                        "boundary_raw_pool_constraint_mean_median_abs_error": 0.20,
                        "boundary_raw_pool_oracle_mean_variance_certified_count": 1,
                        "boundary_raw_pool_full_certified_count": 1,
                        "boundary_raw_pool_false_certified_count": false_count,
                        "boundary_raw_pool_certificate_precision": 0.9,
                        "boundary_raw_pool_best_feasible_epistemic_radius": 0.1,
                        "boundary_raw_pool_best_feasible_true_margin": -0.2,
                    },
                }
                path = root / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1


def test_v5_analyzer_promotes_oracle_free_false_certification_gain(tmp_path):
    _write_matrix(tmp_path)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["row_count"] == 120
    assert result["all_matrix_cells_complete"] is True
    assert result["advance_to_sequential"] is True
    assert result["selected_variant"] in set(VARIANTS) - {
        "v4_latent_control"}


def test_v5_analyzer_rejects_no_false_certification_gain(tmp_path):
    _write_matrix(tmp_path, challenger_improves=False)
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["advance_to_sequential"] is False
    assert all(not candidate["checks"][
        "strict_false_certification_reduction"]
        for candidate in result["candidate_gates"])


def test_v5_sequential_analyzer_promotes_paired_safe_challenger(tmp_path):
    index = 0
    for variant in ("v4_latent_control", "misspec_scalar"):
        challenger = variant == "misspec_scalar"
        for domain, shock in SCENARIOS:
            for seed in range(5):
                row = {
                    "experiment_variant": (
                        f"mean_v5_sequential/{variant}/shock{shock:g}"),
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {"admissible_oracle_free_transfer": True},
                    "true_feasible": True,
                    "feasible_simple_regret": 0.05 if challenger else 0.06,
                    "initial_has_true_feasible": True,
                    "adaptive_rescue": False,
                    "adaptive_loss": False,
                    "adaptive_improves_initial_best": challenger,
                    "adaptive_regret_change": -0.01 if challenger else 0.0,
                    "posterior_feasible": challenger,
                    "posterior_certificate_vacuous": not challenger,
                    "false_certificate_count": 0,
                    "variance_log_rmse": 0.50,
                    "task_initial_design": {
                        "fingerprint": f"{domain}:{seed}"},
                    "source_target_adaptation_contract": {
                        "source_archive_fingerprint": f"archive:{domain}"},
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_false_certified_count": (
                            1 if challenger else 2),
                        "boundary_raw_pool_constraint_mean_median_abs_error": 0.2,
                        "boundary_raw_pool_constraint_mean_rank_correlation": 0.7,
                    },
                }
                path = tmp_path / "sequential" / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1
    result = summarize_sequential(
        load_sequential_rows(tmp_path / "sequential"), expected_seeds=5)
    assert result["row_count"] == 40
    assert result["promote_misspec_scalar"] is True
    assert all(result["checks"].values())
