import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_profile_stress_suite import run_task  # noqa: E402


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
