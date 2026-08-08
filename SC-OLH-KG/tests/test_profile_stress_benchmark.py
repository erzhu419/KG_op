import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_profile_stress_suite import (  # noqa: E402
    _oracle_library,
    _registered_audit_library,
    run_task,
)
from problems.randomized_profiles import (  # noqa: E402
    RandomizedOrderedProfileProblem,
    generate_structural_profile_library,
)
from performance.run_profile_stress_matrix import (  # noqa: E402
    build_equal_preverification_cost_cells,
    build_primary_cells,
    build_schema_descriptor_cells,
    build_sensitivity_cells,
    derived_design_seed,
    derived_target_seed,
    run_matrix,
)


def test_tiny_profile_stress_task_runs_without_oracle_leakage():
    result = run_task(
        regime="aligned_low_frequency",
        target_seed=3,
        arm="source_atlas",
        dimension=64,
        active_rank=6,
        n0=5,
        source_task_count=2,
        library_size=16,
        source_replications=3,
        verification_budgets=(8, 8, 8),
    )
    assert result["status"] == "ok"
    assert result["source_calls"] == 96
    assert result["target_search_calls"] == 5
    assert result["target_oracle_used_for_design"] is False
    assert result["target_outcomes_used_for_design"] is False
    assert result["atlas_diagnostics"]["target_outcomes_used"] is False
    assert result["all_in_calls_unamortized"] == (
        result["source_calls"]
        + result["target_search_calls"]
        + result["verification_calls"]
    )
    assert result["all_in_budget_cap_unamortized"] == 96 + 5 + 24


def test_oracle_arm_is_explicitly_labeled_upper_bound():
    result = run_task(
        regime="frequency_support_shift",
        target_seed=4,
        arm="oracle_library_upper_bound",
        dimension=64,
        active_rank=6,
        n0=5,
        source_task_count=2,
        library_size=16,
        source_replications=2,
        safe_mass=0.18,
        verification_budgets=(8, 8, 8),
    )
    assert result["target_oracle_used_for_design"] is True
    assert result["target_outcomes_used_for_design"] is True
    assert result["source_calls"] == 0
    assert result["oracle_role"] == "finite_library_upper_bound"


def test_audit_oracle_is_fixed_independently_of_source_library_size():
    target = RandomizedOrderedProfileProblem(
        regime="frequency_support_shift",
        role="target",
        task_seed=17,
        family_seed=1234,
        d=64,
        active_rank=8,
        safe_mass=0.03,
    )
    audit_library = _registered_audit_library(target)
    _, audit_best = _oracle_library(target, audit_library)
    assert target.is_truly_feasible(audit_best["point"])
    assert len(audit_library) == 64

    smaller_source_library = generate_structural_profile_library(
        32, dimension=128, seed=999, maximum_frequency=40)
    larger_source_library = generate_structural_profile_library(
        128, dimension=128, seed=999, maximum_frequency=40)
    assert len(smaller_source_library) != len(larger_source_library)
    assert _registered_audit_library(target) == audit_library


def test_neutral_continuation_changes_only_target_search_budget():
    result = run_task(
        regime="aligned_low_frequency",
        target_seed=5,
        arm="generic_dct_maximin",
        dimension=64,
        active_rank=6,
        n0=5,
        N=8,
        library_size=16,
        verification_budgets=(8, 8, 8),
    )
    assert result["n0"] == 5
    assert result["target_search_calls"] == 8
    assert [row["source"] for row in result["search_records"]] == (
        ["initial_design"] * 5 + ["neutral_sobol_continuation"] * 3
    )
    assert result["initial_design_audit_point_count"] == 5
    assert result["search_audit_point_count"] == 8
    assert (
        result["true_feasible_count_in_search"]
        >= result["true_feasible_count_in_design"]
    )
    if result["initial_design_finite_audit_library_regret"] is not None:
        assert result["finite_audit_library_regret"] is not None
        assert (
            result["finite_audit_library_regret"]
            <= result["initial_design_finite_audit_library_regret"]
        )


def test_confirmatory_matrix_seed_and_sharding_are_deterministic(tmp_path):
    seed = derived_target_seed("bad2d97", "aligned_low_frequency", 0)
    assert seed == derived_target_seed(
        "bad2d97", "aligned_low_frequency", 0)
    cells = build_primary_cells(
        freeze_commit="bad2d97",
        dimensions=(32,),
        task_count=1,
        arms=("raw_sobol",),
        regimes=("aligned_low_frequency",),
    )
    assert cells[0]["target_seed"] == seed
    design_seed = derived_design_seed(
        "bad2d97", "aligned_low_frequency", 0)
    assert cells[0]["design_seed"] == design_seed
    assert design_seed != seed
    summary = run_matrix(
        output_dir=tmp_path,
        freeze_commit="bad2d97",
        start=0,
        end=1,
        workers=1,
        dimensions=(32,),
        task_count=1,
        arms=("raw_sobol",),
        regimes=("aligned_low_frequency",),
        evaluation_commit="evaluation-patch",
    )
    assert summary["status"] == "complete"
    assert summary["completed_count"] == 1
    assert summary["evaluation_implementation_commit"] == "evaluation-patch"
    cell = next(tmp_path.glob("cell*.json"))
    assert json.loads(cell.read_text(encoding="utf-8"))[
        "evaluation_implementation_commit"] == "evaluation-patch"
    payload = json.loads(cell.read_text(encoding="utf-8"))
    assert payload["design_seed"] == design_seed
    contract = payload["target_information_contract"]
    assert contract["exact_semantic_to_raw_index_map_declared"] is True
    assert contract[
        "latent_task_generation_seed_exposed_to_frontend"] is False


def test_registered_sensitivity_and_equal_cost_cells_are_paired():
    sensitivity = build_sensitivity_cells(
        freeze_commit="freeze",
        dimension=1000,
        task_count=1,
        regimes=("aligned_low_frequency",),
    )
    by_configuration = {}
    for cell in sensitivity:
        by_configuration.setdefault(cell["configuration_id"], []).append(cell)
    assert len(by_configuration) == 27
    assert all(
        {cell["arm"] for cell in group}
        == {"source_atlas", "generic_dct_maximin"}
        for group in by_configuration.values()
    )

    equal_cost = build_equal_preverification_cost_cells(
        freeze_commit="freeze",
        dimension=1000,
        task_count=1,
        regimes=("aligned_low_frequency",),
    )
    source = next(cell for cell in equal_cost if cell["arm"] == "source_atlas")
    controls = [cell for cell in equal_cost if cell["arm"] != "source_atlas"]
    assert source["N"] == 10
    assert all(cell["N"] == 394 for cell in controls)

    schema = build_schema_descriptor_cells(
        freeze_commit="freeze",
        dimension=1000,
        task_count=1,
        regimes=("coordinate_permutation",),
    )
    assert len(schema) == 8
    assert {
        (cell["schema_mode"], cell["descriptor_mode"])
        for cell in schema
    } == {
        ("declared", "domain_blind"),
        ("declared", "conditioned"),
        ("schema_blind", "domain_blind"),
        ("schema_blind", "conditioned"),
    }


def test_confirmatory_manifest_cell_counts_match_builders():
    manifest = json.loads((
        ROOT / "performance" / "manifests"
        / "or_review_confirmatory_execution_v1.json"
    ).read_text(encoding="utf-8"))
    commit = manifest["method_freeze_commit"]
    matrices = manifest["matrices"]
    assert len(build_primary_cells(freeze_commit=commit)) == (
        matrices["profile_stress_primary"]["cell_count"])
    assert len(build_sensitivity_cells(freeze_commit=commit)) == (
        matrices["profile_stress_sensitivity"]["cell_count"])
    assert len(build_schema_descriptor_cells(freeze_commit=commit)) == (
        matrices["profile_stress_schema_descriptor"]["cell_count"])
    assert len(build_equal_preverification_cost_cells(
        freeze_commit=commit)) == (
            matrices["profile_stress_equal_preverification_cost"][
                "cell_count"])


def test_evaluation_patch_v2_freezes_full_search_credit():
    patch = json.loads((
        ROOT / "performance" / "manifests"
        / "or_review_profile_evaluation_patch_v2.json"
    ).read_text(encoding="utf-8"))
    assert patch["method_invariance_contract"][
        "atlas_selection_changed"] is False
    assert patch["evaluation_changes"][
        "equal_cost_target394_controls_receive_full_search_credit"] is True
    assert patch["interpretation_contract"][
        "frontend_attribution_uses_initial_design_fields"] is True


def test_evaluation_patch_v3_freezes_inference_labels():
    patch = json.loads((
        ROOT / "performance" / "manifests"
        / "or_review_profile_evaluation_patch_v3.json"
    ).read_text(encoding="utf-8"))
    assert patch["evaluation_implementation_commit"] == (
        "594a88a415bcb93a798530dba6629bd684ea261c")
    assert patch["method_invariance_contract"][
        "atlas_selection_changed"] is False
    assert patch["evaluation_changes"][
        "primary_descriptor_mode"] == "domain_blind"
    assert patch["evaluation_changes"][
        "active_rank_override_recorded"] is True
    assert patch["interpretation_contract"][
        "holm_families_fixed_before_outcome_access"] is True
