import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v7_offline_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v7_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v7_gate import (  # noqa: E402
    SCENARIOS,
    VARIANTS,
    load_rows,
    summarize,
)


def _args(tmp_path):
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": submit.base.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v7",
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


def test_v7_submitter_builds_paired_five_arm_matrix(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 5 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == len(specs)
    role = next(spec for spec in specs
                if "/role_epistemic/" in spec["signature"])
    assert "--observable-mean-descriptor-mode role_aligned" in role["cmd"]
    assert (
        "--source-constraint-mean-role-epistemic-mode matching_loss"
        in role["cmd"])
    contrast = next(spec for spec in specs
                    if "/ordered_source_contrast/" in spec["signature"])
    assert (
        "--source-constraint-mean-misspecification-mode source_contrast"
        in contrast["cmd"])
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--runtime-checkpoint-dir ''" in spec["cmd"] for spec in specs)


def test_v7_analyzer_accepts_supported_low_rank_repair(tmp_path):
    index = 0
    for variant in VARIANTS:
        role = variant in {"role_epistemic", "role_source_contrast"}
        contrast = variant in {
            "ordered_source_contrast", "role_source_contrast"}
        for domain, shock in SCENARIOS:
            for seed in range(5):
                false_count = {
                    "v4_latent_control": 4,
                    "role_match_raw": 10,
                    "role_epistemic": 3,
                    "ordered_source_contrast": 3,
                    "role_source_contrast": 2,
                }[variant]
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
                    "source_role_epistemic_calibration": ({
                        "status": "calibrated",
                        "source_role_trust": 0.2,
                        "target_labels_used": False,
                        "target_oracle_used": False,
                    } if role else {
                        "status": "disabled",
                        "source_role_trust": 1.0,
                        "target_labels_used": False,
                        "target_oracle_used": False,
                    }),
                    "component_deviation_diagnostics": [source_diagnostics],
                }
                row = {
                    "experiment_variant": f"mean_v7_offline/{variant}/x",
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
    assert result["row_count"] == 100
    assert "role_source_contrast" in result["sequential_gate_eligible"]
    assert all(result["variant_checks"][
        "role_source_contrast"].values())
