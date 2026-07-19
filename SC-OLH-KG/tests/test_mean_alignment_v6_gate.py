import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v6_sequential_gate_scheduler.py")
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v6_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.analyze_mean_alignment_v6_sequential_gate import (  # noqa: E402
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
        "source_run_id": submit.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v6",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "python": submit.base.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v6_submitter_builds_paired_three_arm_matrix(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("/mean_v6_sequential/" in spec["signature"]
               for spec in specs)
    dynamic = next(spec for spec in specs
                   if "/misspec_hierarchical/" in spec["signature"])
    assert (
        "--source-constraint-mean-misspecification-mode "
        "hierarchical_predictive_scale"
    ) in dynamic["cmd"]
    assert "--runtime-checkpoint-dir ''" in dynamic["cmd"]


def test_v6_analyzer_promotes_online_oracle_free_improvement(tmp_path):
    index = 0
    for variant in VARIANTS:
        dynamic = variant == "misspec_hierarchical"
        for domain, shock in SCENARIOS:
            for seed in range(5):
                trajectory = [
                    {
                        "target_observation_count": 10 + step,
                        "online_mixture_update_count": step,
                        "component_scales": {
                            "source:a": 1.0 + 0.1 * step,
                            "target:null": 1.0,
                        },
                    }
                    for step in range(11)
                ] if dynamic else []
                source_prior = {
                    "online_mixture_update_count": 10,
                    "target_observation_count": 20,
                    "source_mean_misspecification_online": dynamic,
                    "source_mean_misspecification_scale_trajectory": trajectory,
                    "component_deviation_diagnostics": [
                        {
                            "name": "source:a",
                            "source_mean_misspecification_scale": 2.0,
                        },
                        {
                            "name": "target:null",
                            "source_mean_misspecification_scale": 1.0,
                        },
                    ],
                }
                false_count = {
                    "v4_latent_control": 3,
                    "misspec_scalar_static": 4,
                    "misspec_hierarchical": 2,
                }[variant]
                row = {
                    "experiment_variant": (
                        f"mean_v6_sequential/{variant}/shock{shock:g}"),
                    "heldout": domain,
                    "target_shared_shock_scale": shock,
                    "seed": seed,
                    "audit": {"admissible_oracle_free_transfer": True},
                    "true_feasible": True,
                    "feasible_simple_regret": 0.05 if dynamic else 0.06,
                    "adaptive_loss": False,
                    "adaptive_rescue": dynamic,
                    "posterior_certificate_vacuous": False,
                    "false_certificate_count": 0,
                    "variance_log_rmse": 0.5,
                    "gpr_numerics": [{}, {
                        "source_parametric_prior": source_prior,
                    }],
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
    result = summarize(load_rows(tmp_path), expected_seeds=5, N=20, n0=10)
    assert result["row_count"] == 60
    assert result["promote_misspec_hierarchical"] is True
    assert all(result["checks"].values())
