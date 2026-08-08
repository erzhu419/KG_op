import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.botorch_adapters import is_botorch_available  # noqa: E402
from performance.analyze_functional_profile_scbo import analyze  # noqa: E402
from performance.benchmark_functional_profile_scbo import (  # noqa: E402
    run_task,
)
from performance.run_functional_profile_scbo_matrix import (  # noqa: E402
    _emit_terminal_status,
    _existing_ok,
    build_equal_preverification_cost_cells,
    build_primary_cells,
    build_rank_sensitivity_cells,
)
from problems.profile_coefficient_space import (  # noqa: E402
    CosineCoefficientProfileProblem,
)
from problems.randomized_profiles import (  # noqa: E402
    RandomizedOrderedProfileProblem,
)


def _target(d=64):
    return RandomizedOrderedProfileProblem(
        regime="aligned_low_frequency",
        role="target",
        task_seed=17,
        family_seed=1234,
        d=d,
        active_rank=6,
        safe_mass=0.18,
    )


def test_cosine_coefficient_map_is_fixed_bounded_and_source_free():
    target = _target()
    problem = CosineCoefficientProfileProblem(
        target, coefficient_count=4, coefficient_scale=0.25)
    center = (50,) * problem.d
    shifted = (50, 100, 50, 50, 50)
    center_profile = problem.semantic_profile(center)
    shifted_profile = problem.semantic_profile(shifted)
    assert len(center_profile) == target.d
    assert np.all((0.0 <= center_profile) & (center_profile <= 1.0))
    assert not np.allclose(center_profile, shifted_profile)
    assert len(problem.raw_point(center)) == target.d
    contract = problem.information_contract()
    assert contract["source_outcomes_used"] is False
    assert contract["target_outcomes_used_to_define_coordinate"] is False
    assert contract["coefficient_dimension"] == 5


def test_cosine_wrapper_delegates_exactly_to_target_simulator():
    target = _target()
    problem = CosineCoefficientProfileProblem(target, coefficient_count=3)
    coefficient_point = (50, 70, 30, 80)
    first_rng = np.random.default_rng(99)
    second_rng = np.random.default_rng(99)
    wrapped = problem.simulate(coefficient_point, first_rng)
    direct = target.simulate(problem.raw_point(coefficient_point), second_rng)
    assert np.allclose(wrapped, direct)


def test_functional_matrix_contracts_have_registered_costs():
    primary = build_primary_cells(
        task_freeze_commit="freeze",
        dimensions=(200, 1000, 10000),
        task_count=2,
        regimes=("aligned_low_frequency",),
    )
    assert len(primary) == 6
    assert {row["N"] for row in primary} == {13}
    assert {row["coefficient_count"] for row in primary} == {8}

    sensitivity = build_rank_sensitivity_cells(
        task_freeze_commit="freeze",
        dimension=1000,
        task_count=2,
        regimes=("aligned_low_frequency",),
    )
    assert len(sensitivity) == 6
    assert {row["coefficient_count"] for row in sensitivity} == {4, 8, 16}

    equal_cost = build_equal_preverification_cost_cells(
        task_freeze_commit="freeze",
        dimension=1000,
        task_count=2,
        regimes=("aligned_low_frequency",),
    )
    assert len(equal_cost) == 2
    assert {row["N"] for row in equal_cost} == {394}

    import json
    manifest = json.loads((
        ROOT / "performance" / "manifests"
        / "or_review_functional_profile_scbo_v1.json"
    ).read_text(encoding="utf-8"))
    assert manifest["matrices"]["primary"]["cell_count"] == len(
        build_primary_cells(
            task_freeze_commit=manifest["task_seed_freeze_commit"]))
    assert manifest["matrices"]["rank_sensitivity"]["cell_count"] == len(
        build_rank_sensitivity_cells(
            task_freeze_commit=manifest["task_seed_freeze_commit"]))
    assert manifest["matrices"]["equal_preverification_cost"][
        "cell_count"] == len(build_equal_preverification_cost_cells(
            task_freeze_commit=manifest["task_seed_freeze_commit"]))

    repaired = json.loads((
        ROOT / "performance" / "manifests"
        / "or_review_functional_profile_scbo_v2.json"
    ).read_text(encoding="utf-8"))
    assert repaired["v1_failure_audit"][
        "performance_outcomes_observed_before_v2_freeze"] is False
    assert repaired["v1_failure_audit"][
        "total_v1_cells_with_target_evaluations"] == 0
    assert repaired["execution_contract"]["cell_error_policy"] == (
        "nonzero shard exit; no DONE marker")
    assert repaired["matrices"] == manifest["matrices"]


def test_functional_matrix_terminal_status_is_fail_closed(capsys):
    assert _emit_terminal_status({"error_count": 2}) is False
    assert "FUNCTIONAL_SCBO_FAILED cell_errors=2" in capsys.readouterr().out
    assert _emit_terminal_status({"error_count": 0}) is True
    assert capsys.readouterr().out.strip() == "DONE"


def test_functional_cell_resume_requires_exact_execution_commit(tmp_path):
    import json

    cell = {"configuration_id": "target13-k8", "design_seed": 19}
    path = tmp_path / "cell.json"
    path.write_text(json.dumps({
        "contract_id": "target_only_functional_profile_scbo_v1",
        "status": "ok",
        "task_freeze_commit": "freeze",
        "functional_method_commit": "method",
        "execution_commit": "execution-a",
        "matrix_configuration_id": "target13-k8",
        "design_seed": 19,
    }), encoding="utf-8")
    assert _existing_ok(
        path,
        task_freeze_commit="freeze",
        functional_method_commit="method",
        execution_commit="execution-a",
        cell=cell,
    )
    assert not _existing_ok(
        path,
        task_freeze_commit="freeze",
        functional_method_commit="method",
        execution_commit="execution-b",
        cell=cell,
    )


def test_functional_analysis_keeps_initial_search_and_deployment_separate(
    tmp_path,
):
    common = {
        "status": "ok",
        "regime": "aligned_low_frequency",
        "target_seed": 17,
        "nominal_dimension": 1000,
        "initial_design_contains_true_feasible": True,
        "initial_design_finite_audit_library_regret": 0.2,
        "initial_design_penalized_loss": 0.2,
        "contains_true_feasible": True,
        "finite_library_regret": 0.1,
        "penalized_loss": 0.1,
        "independently_certified": True,
        "false_certificate": False,
        "deployed_truth": {"feasible": True, "objective": 0.1},
        "finite_audit_library_oracle_objective": 0.0,
        "verification_calls": 80,
        "all_in_calls_unamortized": 93,
        "wall_time_sec": 1.0,
    }
    functional = {
        **common,
        "contract_id": "target_only_functional_profile_scbo_v1",
        "arm": "target_only_dct_space_scbo",
        "matrix_configuration_id": "target13-k8",
        "N": 13,
        "search_contains_true_feasible": True,
        "search_finite_audit_library_regret": 0.1,
        "functional_coordinate_contract": {"coefficient_count": 8},
    }
    profile = {
        **common,
        "contract_id": "randomized_ordered_profile_stress_v2",
        "arm": "source_atlas",
        "matrix_configuration_id": "primary",
        "schema_mode": "declared",
        "descriptor_mode": "domain_blind",
        "N": 10,
    }
    functional_path = tmp_path / "functional.json"
    profile_path = tmp_path / "profile.json"
    import json
    functional_path.write_text(json.dumps(functional), encoding="utf-8")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    payload = analyze([functional_path], [profile_path])
    assert payload["status"] == "complete"
    assert len(payload["paired_initial_design_comparisons"]) == 1
    assert len(payload["paired_deployment_comparisons"]) == 1
    assert payload["primary_endpoint"] == (
        "certified_true_feasible_deployed_policy")


@pytest.mark.skipif(
    not is_botorch_available(), reason="BoTorch is not installed")
def test_tiny_functional_scbo_freezes_before_truth_and_uses_common_verifier():
    result = run_task(
        regime="aligned_low_frequency",
        target_seed=31,
        design_seed=47,
        dimension=32,
        active_rank=4,
        safe_mass=0.18,
        n0=4,
        N=5,
        coefficient_count=3,
        verification_budgets=(8, 8, 8),
        raw_samples=8,
        num_restarts=2,
        maxiter=10,
        batch_candidates=16,
        ts_candidates=32,
    )
    assert result["status"] == "ok"
    assert result["source_calls"] == 0
    assert result["target_search_calls"] == 5
    assert result["backend_contract"]["truth_metrics_evaluated"] is False
    assert result["backend_contract"]["target_oracle_used"] is False
    assert result["functional_coordinate_contract"][
        "target_outcomes_used_to_define_coordinate"] is False
    assert result["all_in_calls_unamortized"] == (
        result["target_search_calls"] + result["verification_calls"])
    assert len(result["shortlist"]) <= 3
    assert all(
        row["coordinate_contract"]
        == "target_only_cosine_coefficient_space_v1"
        for row in result["shortlist"]
    )
