import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_profile_stress_suite import run_task  # noqa: E402
from performance.run_profile_stress_matrix import (  # noqa: E402
    build_equal_preverification_cost_cells,
    build_primary_cells,
    build_sensitivity_cells,
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
    )
    assert summary["status"] == "complete"
    assert summary["completed_count"] == 1


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
