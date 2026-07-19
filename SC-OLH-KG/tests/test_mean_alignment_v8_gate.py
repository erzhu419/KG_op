import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v8_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v8_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v8_gate import (  # noqa: E402
    ADAPTIVE_VARIANTS,
    CONTRAST_VARIANTS,
    SCENARIOS,
    VARIANTS,
    load_rows,
    summarize,
)
from performance.analyze_mean_alignment_v8_sequential_gate import (  # noqa: E402
    summarize as summarize_sequential,
)


def _args(tmp_path):
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": submit.base.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v8",
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
        "python": submit.base.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v8_submitter_builds_support_adaptive_matrix(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 6 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == len(specs)
    adaptive = next(spec for spec in specs
                    if "/adaptive_role_ordered/" in spec["signature"])
    assert (
        "--observable-mean-descriptor-mode role_adaptive_ordered"
        in adaptive["cmd"])
    invariant = next(spec for spec in specs
                     if "/adaptive_role_set/" in spec["signature"])
    assert (
        "--observable-mean-descriptor-mode role_adaptive_set_invariant"
        in invariant["cmd"])
    contrast = next(spec for spec in specs
                    if "/adaptive_role_set_contrast/" in spec["signature"])
    assert (
        "--source-constraint-mean-misspecification-mode source_contrast"
        in contrast["cmd"])
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--runtime-checkpoint-dir ''" in spec["cmd"] for spec in specs)


def test_v8_submitter_builds_paired_sequential_gate(tmp_path):
    args = _args(tmp_path)
    args.N = 20
    args.variants = "v4_latent_control,adaptive_role_ordered"
    specs = submit.build_specs(args)
    assert len(specs) == 2 * 4 * 5
    assert all("--N 20" in spec["cmd"] for spec in specs)
    assert all("--n0 10" in spec["cmd"] for spec in specs)
    assert all("mean_v8_sequential" in spec["signature"] for spec in specs)


def test_v8_analyzer_accepts_outcome_free_supported_switch(tmp_path):
    index = 0
    for variant in VARIANTS:
        adaptive = variant in ADAPTIVE_VARIANTS
        contrast = variant in CONTRAST_VARIANTS
        for domain, shock in SCENARIOS:
            factor = domain == "FactorShockStatePolicyRZDT1"
            for seed in range(5):
                false_count = {
                    "v4_latent_control": 5,
                    "role_match_raw": 10,
                    "adaptive_role_ordered": 2,
                    "adaptive_role_set": 3,
                    "adaptive_role_ordered_contrast": 1,
                    "adaptive_role_set_contrast": 2,
                }[variant]
                selection = ({
                    "status": (
                        "unsupported_cardinality" if factor else "supported"),
                    "channel_cardinality_supported": not factor,
                    "selected_coordinate": (
                        (
                            "set_invariant"
                            if "_set" in variant else "ordered"
                        ) if factor else "role_aligned"
                    ),
                    "selection_uses_target_labels": False,
                    "selection_uses_target_oracle": False,
                    "target_labels_used": False,
                    "target_oracle_used": False,
                } if adaptive else {})
                source_diagnostics = {
                    "name": "source:a",
                    "source_mean_prior_covariance_trace_before": 1.0,
                    "source_mean_prior_covariance_trace_after": 1.5,
                    "misspecification_uncertainty_can_only_increase": True,
                    "source_contrast_rank": 1,
                    "source_contrast_rank_bound": 1,
                    "source_contrast_uses_target_data": False,
                    "target_oracle_used_for_misspecification": False,
                } if contrast else {"name": "source:a"}
                source_prior = {
                    "role_coordinate_selection": selection,
                    "component_deviation_diagnostics": [source_diagnostics],
                }
                row = {
                    "experiment_variant": f"mean_v8_offline/{variant}/x",
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {"admissible_oracle_free_transfer": True},
                    "true_feasible": True,
                    "feasible_simple_regret": 0.04,
                    "variance_log_rmse": 0.5,
                    "gpr_numerics": [{}, {
                        "source_parametric_prior": source_prior}],
                    "task_initial_design": {
                        "fingerprint": f"{domain}:{shock}:{seed}"},
                    "source_target_adaptation_contract": {
                        "source_archive_fingerprint": f"archive:{domain}"},
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_false_certified_count": false_count,
                        "boundary_raw_pool_constraint_mean_median_abs_error": 0.2,
                    },
                }
                path = tmp_path / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}))
                index += 1
    result = summarize(load_rows(tmp_path), expected_seeds=5)
    assert result["row_count"] == 120
    assert "adaptive_role_set_contrast" in result[
        "sequential_gate_eligible"]
    assert all(result["variant_checks"][
        "adaptive_role_set_contrast"].values())


def test_v8_sequential_analyzer_requires_online_recommendation_gain(tmp_path):
    rows = []
    for variant in ("v4_latent_control", "adaptive_role_ordered"):
        adaptive = variant == "adaptive_role_ordered"
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
                } if adaptive else {})
                rows.append({
                    "gate_variant": variant,
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {"admissible_oracle_free_transfer": True},
                    "initial_has_true_feasible": seed != 0,
                    "initial_true_feasible_count": int(seed != 0),
                    "initial_best_feasible_regret": 0.1,
                    "true_feasible": adaptive or seed != 0,
                    "feasible_simple_regret": 0.04,
                    "adaptive_rescue": adaptive and seed == 0,
                    "adaptive_loss": False,
                    "adaptive_improves_initial_best": adaptive,
                    "adaptive_regret_change": -0.01 if adaptive else 0.0,
                    "posterior_certificate_vacuous": True,
                    "posterior_certified_evaluated_count": 0,
                    "false_certificate_count": 0,
                    "variance_log_rmse": 0.5,
                    "gpr_numerics": [{}, {"source_parametric_prior": {
                        "role_coordinate_selection": selection}}],
                    "task_initial_design": {
                        "fingerprint": f"{domain}:{shock}:{seed}"},
                    "source_target_adaptation_contract": {
                        "source_archive_fingerprint": f"archive:{domain}"},
                    "boundary_raw_pool_truth_diagnostics": {
                        "boundary_raw_pool_false_certified_count": (
                            1 if adaptive else 5),
                        "boundary_raw_pool_constraint_mean_median_abs_error":
                            0.2,
                    },
                })
    result = summarize_sequential(rows, expected_seeds=5, N=20, n0=10)
    assert result["promote_adaptive_role_ordered"]
    assert result["totals"]["adaptive_role_ordered"][
        "adaptive_rescue"] == 4
