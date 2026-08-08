import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_profile_stress_suite import analyze  # noqa: E402


def _row(arm, seed, *, feasible, certified, loss):
    return {
        "contract_id": "randomized_ordered_profile_stress_v2",
        "status": "ok",
        "regime": "aligned_low_frequency",
        "target_seed": seed,
        "arm": arm,
        "schema_mode": "declared",
        "descriptor_mode": "conditioned",
        "nominal_dimension": 1000,
        "effective_rank": 8,
        "contains_true_feasible": feasible,
        "initial_design_contains_true_feasible": feasible,
        "independently_certified": certified,
        "false_certificate": False,
        "feasible_and_epsilon_optimal_005": bool(feasible and loss <= 0.05),
        "finite_library_regret": loss if feasible else None,
        "initial_design_finite_audit_library_regret": (
            loss if feasible else None),
        "penalized_loss": loss,
        "initial_design_penalized_loss": loss,
        "verification_calls": 80,
        "all_in_calls_unamortized": 474,
        "all_in_calls_amortized": 109.2,
        "source_calls": 384 if arm == "source_atlas" else 0,
        "target_search_calls": 10,
    }


def test_analysis_uses_target_tasks_as_paired_units(tmp_path):
    paths = []
    for seed in range(3):
        for arm, feasible, certified, loss in (
            ("source_atlas", True, True, 0.01),
            ("generic_dct_maximin", seed == 0, seed == 0, 1.1),
        ):
            path = tmp_path / f"{arm}_{seed}.json"
            path.write_text(json.dumps(_row(
                arm, seed, feasible=feasible, certified=certified, loss=loss)),
                encoding="utf-8")
            paths.append(path)
    payload = analyze(paths)
    assert payload["status"] == "complete"
    assert payload["inference_unit"] == "independent_target_task"
    comparison = payload["paired_task_level_comparisons"][0]
    assert comparison["paired_task_count"] == 3
    assert comparison["first_wins"] == 3
    assert comparison["holm_family_id"] == "task_level_context:primary"
    assert (
        comparison[
            "one_sided_first_better_exact_sign_pvalue_holm"]
        >= comparison["one_sided_first_better_exact_sign_pvalue"]
    )
    macro = payload["registered_generator_macro_comparisons"][0]
    assert macro["paired_task_count"] == 3
    assert macro["registered_regime_count"] == 1
    assert len(
        macro[
            "mean_first_minus_second_penalized_loss_bootstrap_95ci"]
    ) == 2
    source_summary = next(
        row for row in payload["summaries"]
        if row["arm"] == "source_atlas"
    )
    assert source_summary["true_feasible_coverage_rate"] == 1.0
    assert 0.0 < source_summary[
        "true_feasible_coverage_one_sided_95_lower"] < 1.0
    assert source_summary["false_certificate_one_sided_95_upper"] > 0.0
    assert source_summary["coverage_probability_scope"].startswith(
        "declared registered")


def test_analysis_never_pools_registered_sensitivity_settings(tmp_path):
    paths = []
    for alpha in (0.01, 0.10):
        row = _row(
            "source_atlas", 0, feasible=True, certified=True, loss=alpha)
        row["alpha"] = alpha
        path = tmp_path / f"source_alpha_{alpha}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)

    payload = analyze(paths)
    assert payload["status"] == "complete"
    assert len(payload["summaries"]) == 2
    assert {row["alpha"] for row in payload["summaries"]} == {0.01, 0.10}
    assert {
        row["sensitivity_axis"]
        for row in payload["one_factor_sensitivity_curves"]
    } == {"alpha"}


def test_analysis_reports_cell_wall_time_without_making_it_an_endpoint(tmp_path):
    row = _row("source_atlas", 0, feasible=True, certified=True, loss=0.01)
    row["wall_time_sec"] = 1.25
    path = tmp_path / "source.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    payload = analyze([path])
    assert payload["summaries"][0]["median_wall_time_sec"] == 1.25
    assert payload["compact_rows"][0]["wall_time_sec"] == 1.25


def test_analysis_separates_initial_design_from_full_search(tmp_path):
    row = _row("raw_sobol", 0, feasible=True, certified=True, loss=0.02)
    row.update({
        "initial_design_contains_true_feasible": False,
        "initial_design_finite_audit_library_regret": None,
        "initial_design_penalized_loss": 1.4,
        "target_search_calls": 394,
    })
    path = tmp_path / "target_only_continuation.json"
    path.write_text(json.dumps(row), encoding="utf-8")

    payload = analyze([path])
    summary = payload["summaries"][0]
    assert summary["initial_design_true_feasible_coverage_rate"] == 0.0
    assert summary["true_feasible_coverage_rate"] == 1.0
    assert summary["initial_design_mean_penalized_loss"] == 1.4
    assert summary["mean_penalized_loss"] == 0.02
