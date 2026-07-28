import copy
import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SC-OLH-KG"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _load(
    "mean_v47_submit",
    REPO
    / "scripts/submit_scolhkg_mean_alignment_v47_deconvolution_gate_scheduler.py",
)
analyze = _load(
    "mean_v47_analyze",
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v47_gate.py",
)
v46_test = _load(
    "mean_v46_test_fixture",
    REPO / "SC-OLH-KG/tests/test_mean_alignment_v46_gate.py",
)


def _scheduler_args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "python": defaults.REMOTE_PYTHON,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v47",
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seeds": "1,3",
        "scope": "queue_sentinel",
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "confidence_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "cpu": 1,
        "ram_mb": 4096,
    })()


def test_v47_submitter_builds_ten_paired_cpu_sentinels(tmp_path):
    specs = submit.build_specs(_scheduler_args(tmp_path))
    assert len(specs) == 10
    assert all(spec["cpu"] == 1 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES) for spec in specs)
    assert all("jtl110cpu" not in spec["allowed_nodes"] for spec in specs)
    population = [
        spec for spec in specs
        if "/v47_deconvolved_task_bayes/" in spec["signature"]
    ]
    predictive = [
        spec for spec in specs
        if "/v47_deconvolved_task_predictive_bayes/" in spec["signature"]
    ]
    assert len(population) == len(predictive) == 2
    assert all(
        "--source-constraint-mean-hyperlaw-mode grouped_task_deconvolved"
        in spec["cmd"] for spec in population
    )
    assert all(
        "--source-constraint-mean-hyperlaw-mode "
        "grouped_task_deconvolved_predictive" in spec["cmd"]
        for spec in predictive
    )
    assert all(
        "--initial-design-archive-match-mode paired_frozen_control"
        in spec["cmd"] for spec in population + predictive
    )


def _v47_row(variant, seed, challenger=False):
    row = copy.deepcopy(v46_test._analyzer_row(
        "v46_grouped_task_bayes", seed, challenger=challenger))
    predictive = variant.endswith("predictive_bayes")
    mode = (
        "grouped_task_deconvolved_predictive"
        if predictive else "grouped_task_deconvolved"
    )
    row["gate_variant"] = variant
    row["source_constraint_mean_hyperlaw_mode"] = mode
    diagnostics = row["gpr_numerics"][1]["source_parametric_prior"]
    diagnostics.update({
        "configured_hyperlaw_mode": mode,
        "random_effects_deconvolution_selected": True,
        "random_effects_deconvolution": True,
        "source_estimation_covariance_used_for_deconvolution": True,
        "finite_source_predictive_prior_selected": predictive,
        "finite_source_predictive_correction": predictive,
        "target_task_law": (
            "shared_base_mean_plus_deconvolved_predictive_discrepancy"
            if predictive else
            "shared_base_mean_plus_deconvolved_task_discrepancy"
        ),
        "channel_role_observed_covariance_trace": 0.12,
        "channel_role_estimation_noise_trace": 0.02,
        "channel_role_covariance_trace": 0.10,
        "between_base_observed_covariance_trace": 0.15,
        "between_base_estimation_noise_trace": 0.04,
        "between_base_domain_covariance_trace": 0.11,
        "within_base_observed_covariance_trace": 0.22,
        "within_base_estimation_noise_trace": 0.07,
        "within_base_task_covariance_trace": 0.15,
    })
    row["boundary_raw_pool_truth_diagnostics"][
        "boundary_raw_pool_best_feasible_epistemic_radius"
    ] = 0.03 if not predictive else 0.035
    return row


def _gate_rows():
    rows = []
    for seed in analyze.SENTINEL_SEEDS:
        for variant in analyze.VARIANTS:
            if variant in analyze.V47_VARIANTS:
                row = _v47_row(
                    variant,
                    seed,
                    challenger=variant == "v47_deconvolved_task_bayes",
                )
            else:
                row = v46_test._analyzer_row(variant, seed)
                if variant == "v46_grouped_task_bayes":
                    row["boundary_raw_pool_truth_diagnostics"][
                        "boundary_raw_pool_best_feasible_epistemic_radius"
                    ] = 0.05
            rows.append(row)
    return rows


def test_v47_analyzer_promotes_only_strict_deconvolved_population():
    result = analyze.summarize(_gate_rows())
    assert result["paired_initial_design_actions_and_target_responses"]
    assert all(result["source_episode_contract"].values())
    assert all(result["source_hyperlaw_contract"].values())
    assert result["deconvolution_contracts_epistemic_radius"]
    assert result["deconvolved_population_strictly_improves_controls"]
    mechanism = result["mechanism_summaries"][
        "v47_deconvolved_task_bayes"]
    assert mechanism["median_within_base_observed_covariance_trace"] == 0.22
    assert mechanism["median_within_base_estimation_noise_trace"] == 0.07
    assert mechanism["median_within_base_corrected_covariance_trace"] == 0.15
    assert result["promotion_eligible"] == ["v47_deconvolved_task_bayes"]


def test_v47_analyzer_rejects_invalid_noise_deconvolution():
    rows = _gate_rows()
    row = next(
        value for value in rows
        if value["gate_variant"] == "v47_deconvolved_task_bayes"
    )
    diagnostics = row["gpr_numerics"][1]["source_parametric_prior"]
    diagnostics["within_base_task_covariance_trace"] = (
        diagnostics["within_base_observed_covariance_trace"] + 0.01)
    result = analyze.summarize(rows)
    assert not result["source_hyperlaw_contract"][
        "v47_deconvolved_task_bayes"]
    assert result["promotion_eligible"] == []
