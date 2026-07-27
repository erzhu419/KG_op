from types import SimpleNamespace

import numpy as np

from performance.benchmark_controlled_heteroscedastic_optimum import run_cell
from problems.controlled_heteroscedastic import (
    CONTROLLED_HETERO_SCENARIOS,
    ControlledHeteroscedasticProblem,
)
from problems.rzdt import make_problem


def _policy(problem, values):
    return problem._constant_policy(*values)


def test_controlled_scenarios_have_finite_oracles_and_variances():
    for scenario in CONTROLLED_HETERO_SCENARIOS:
        problem = ControlledHeteroscedasticProblem(
            d=12, scenario=scenario)
        x = _policy(problem, (0.75, 0.70, 0.65))
        sigma = problem.true_sigma(x)
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0.0)
        best_x, best_objective = problem.scalarized_true_best_feasible(
            (0.5, 0.5))
        assert best_x is not None
        assert problem.is_truly_feasible(best_x)
        assert np.isfinite(best_objective)


def test_provider_decomposition_is_exact_except_registered_misspecification():
    for scenario, metadata in CONTROLLED_HETERO_SCENARIOS.items():
        problem = ControlledHeteroscedasticProblem(
            d=15, scenario=scenario)
        x = _policy(problem, (0.31, 0.58, 0.77))
        decomposition = problem.true_cumulative_risk_decomposition(x)
        variance = float(problem.true_sigma(x)[2] ** 2)
        assert np.isclose(decomposition["total"], variance)
        if metadata["provider_exact"]:
            assert np.isclose(decomposition["unmodeled_residual"], 0.0)
            assert np.isclose(
                decomposition["provider_total"], decomposition["total"])
        else:
            assert decomposition["unmodeled_residual"] > 0.0
            assert decomposition["total"] > decomposition["provider_total"]


def test_registry_exposes_controlled_problem_names():
    problem = make_problem(
        "ControlledHeteroSharedFactor",
        d=18,
        sigma=0.04,
    )
    assert problem.scenario == "shared_factor"
    assert problem.cumulative_risk_provider_status()["provider_exact"] is True


def test_compact_benchmark_separates_primary_and_terminal_certificate():
    args = SimpleNamespace(
        scenario="smooth_boundary",
        variance_mode="factor",
        backend="sobol",
        seed=4,
        d=9,
        L=100,
        sigma=0.04,
        alpha=0.05,
        N=4,
        n0=4,
        K1=4,
        posterior_pool_size=16,
        state_candidate_count=2,
        state_inverse_pool_size=24,
        beta_g=2.0,
        risk_penalty=5.0,
        exact_mc_samples=2,
        exact_jobs=1,
        verify=True,
        verification_primary_budget=8,
        verification_support_budget=8,
        verification_delta=0.05,
    )
    result = run_cell(args)
    assert result["status"] == "ok"
    assert result["n_search_simulations"] == 4
    assert result["n_verification_simulations"] in (8, 16)
    assert result["information_contract"]["source_archive_used"] is False
    assert (
        result["posterior_certificate"]["posterior_certificate_vacuous"]
        in (True, False)
    )
    effect = result["paired_deployment_effect"]
    assert effect["primary_true_feasible"] in (True, False)
    assert effect["deployment_true_feasible"] in (True, False)
    assert result["oracle_contract"]["used_for_decision"] is False
