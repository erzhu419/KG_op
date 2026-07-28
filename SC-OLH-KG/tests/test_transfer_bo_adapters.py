import json
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import (  # noqa: E402
    FrozenTransferArchive,
    TransferTaskArchive,
    _archive_fingerprint,
    dimension_equivariant_profile_features,
    resample_normalized_profiles,
)
from baselines.transfer_bo_adapters import (  # noqa: E402
    METHOD_CONTRACTS,
    TRANSFER_METHODS,
    TransferBOConfig,
    TransferConstrainedBO,
    scalar_tasks_from_archive,
)
from baselines.transfer_external_models import external_runtime_report  # noqa: E402
from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from core.designs import (  # noqa: E402
    common_sobol_integer_design,
    integer_design_fingerprint,
)
from problems.rzdt import FactorShockStatePolicyRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from performance.benchmark_transfer_fairness import (  # noqa: E402
    _apply_terminal_verification,
)


def _archive(d=4, n=7, seed=19):
    rng = np.random.default_rng(seed)
    tasks = []
    for task_index in range(2):
        X = rng.random((n, d))
        X_integer = np.rint(100.0 * X).astype(int)
        replicates = []
        for profile in X:
            objective = np.mean(profile) + 0.05 * task_index
            constraint = -0.08 + 0.12 * profile[0]
            scale = 0.01 + 0.03 * profile[-1]
            replicates.append(np.column_stack([
                objective + rng.normal(0.0, 0.02, size=3),
                constraint + rng.normal(0.0, scale, size=3),
            ]))
        means = np.vstack([values.mean(axis=0) for values in replicates])
        variance = np.vstack([
            values.var(axis=0, ddof=1) for values in replicates
        ])
        tasks.append(TransferTaskArchive(
            name=f"source_{task_index}",
            X=X,
            X_integer=X_integer,
            Y_mean=means,
            Y_replicates=tuple(replicates),
            replicate_variance=variance,
            mean_observation_variance=variance / 3.0,
            constraint_sigma=np.sqrt(variance[:, 1]),
            tau=0.0,
            alpha=0.05,
            origins=tuple("universal_low_frequency" for _ in range(n)),
        ))
    tasks = tuple(tasks)
    fingerprint = _archive_fingerprint(tasks, 0, "replicated")
    return FrozenTransferArchive(
        tasks=tasks,
        source_seed=0,
        observation_mode="replicated",
        fingerprint=fingerprint,
    ).validate(expected_dimension=d)


def _problem(d=4):
    return ScalarizedProblem(FactorShockStatePolicyRZDT1(
        d=d, L=100, sigma=0.04, alpha=0.05))


def test_frozen_archive_roundtrip_is_exact_and_validated():
    archive = _archive()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "archive.json"
        archive.save(path)
        loaded = FrozenTransferArchive.load(path)
    assert loaded.fingerprint == archive.fingerprint
    assert loaded.source_domains == archive.source_domains
    assert loaded.simulator_calls == 2 * 7 * 3
    assert loaded.information_contract()["source_oracle_aided"] is False
    for actual, expected in zip(loaded.tasks, archive.tasks):
        np.testing.assert_array_equal(actual.X, expected.X)
        for actual_rep, expected_rep in zip(
            actual.Y_replicates, expected.Y_replicates
        ):
            np.testing.assert_array_equal(actual_rep, expected_rep)


def test_archive_scalar_views_use_observations_not_oracle_sigma():
    archive = _archive()
    objective = scalar_tasks_from_archive(archive, "objective")
    constraint = scalar_tasks_from_archive(archive, "constraint_mean")
    risk = scalar_tasks_from_archive(archive, "log_variance")
    assert len(objective) == len(constraint) == len(risk) == 2
    np.testing.assert_allclose(objective[0].y, archive.tasks[0].Y_mean[:, 0])
    np.testing.assert_allclose(
        constraint[0].y, archive.tasks[0].Y_mean[:, 1])
    np.testing.assert_allclose(
        risk[0].y,
        np.log(archive.tasks[0].constraint_sigma ** 2),
    )


def test_dimension_equivariant_coordinate_matches_across_raw_dimensions():
    source_positions = (np.arange(50, dtype=float) + 0.5) / 50.0
    target_positions = (np.arange(1000, dtype=float) + 0.5) / 1000.0
    source = 0.45 + 0.20 * np.cos(np.pi * source_positions)
    target = 0.45 + 0.20 * np.cos(np.pi * target_positions)
    source_features = dimension_equivariant_profile_features(source)
    target_features = dimension_equivariant_profile_features(target)
    assert source_features.shape == target_features.shape == (1, 65)
    np.testing.assert_allclose(
        source_features, target_features, atol=2e-4, rtol=2e-4)
    assert np.all(np.isfinite(source_features))


def test_cross_dimension_stacked_requires_declared_label_free_adapter():
    archive = _archive(d=4)
    problem = _problem(d=8)
    with pytest.raises(ValueError, match="dimension"):
        TransferConstrainedBO(
            problem,
            archive,
            TransferBOConfig(
                method="stacked_transfer_gp_cbo",
                N=3,
                n0=3,
                seed=13,
                candidate_pool_size=16,
            ),
        )
    runner = TransferConstrainedBO(
        problem,
        archive,
        TransferBOConfig(
            method="stacked_transfer_gp_cbo",
            N=3,
            n0=3,
            seed=13,
            candidate_pool_size=16,
            source_dimension_adapter="ordered_dct_quadratic",
        ),
    )
    result = runner.run()
    source_contract = result["source_information_contract"]
    assert source_contract["source_policy_dimension"] == 4
    assert source_contract["target_policy_dimension"] == 8
    assert source_contract["model_input_dimension"] == 65
    assert source_contract["dimension_adapter_uses_target_labels"] is False
    assert source_contract["dimension_adapter_uses_target_oracle"] is False
    assert all(len(row["x"]) == 8 for row in result["history"])
    lifted = resample_normalized_profiles(archive.tasks[0].X[:1], 8)
    assert lifted.shape == (1, 8)


def test_promoted_and_transfer_paths_use_byte_identical_common_sobol_n0():
    archive = _archive()
    transfer_problem = _problem()
    transfer = TransferConstrainedBO(
        transfer_problem,
        archive,
        TransferBOConfig(
            method="hyperbo_cbo",
            N=4,
            n0=3,
            seed=23,
            candidate_pool_size=16,
        ),
    )
    expected = common_sobol_integer_design(
        transfer_problem, n=3, seed=23)
    assert transfer._common_initial_design() == expected

    promoted = SingleOLHKGAlgorithm(
        _problem(),
        SingleOLHKGConfig(
            N=4,
            n0=3,
            seed=23,
            initial_design="common_sobol",
            use_state_coupling=False,
            use_state_basis=False,
            lambda_coupling=0.0,
        ),
    )
    assert promoted._initial_samples() == expected
    assert promoted._task_initial_design_info[
        "problem_specific_hook_used"
    ] is False
    assert promoted._task_initial_design_info["fingerprint"] == (
        integer_design_fingerprint(transfer._common_initial_design())
    )


def test_transfer_consumes_frozen_source_informed_design_exactly():
    archive = _archive()
    points = (
        (5, 10, 15, 20),
        (25, 30, 35, 40),
        (45, 50, 55, 60),
    )
    config = TransferBOConfig(
        method="hyperbo_cbo",
        N=3,
        n0=3,
        seed=23,
        candidate_pool_size=16,
        initial_design="source_informed",
        initial_points=points,
    )
    result = TransferConstrainedBO(_problem(), archive, config).run()
    assert [tuple(row["x"]) for row in result["history"]] == list(points)
    assert all(
        row["selection_reason"] == "frozen_source_informed_initial_design"
        for row in result["history"]
    )
    contract = result["target_information_contract"]
    assert contract["source_informed_initial_design"] is True
    assert contract["initial_design_fingerprint"] == (
        integer_design_fingerprint(points)
    )


def test_transfer_freezes_method_specific_terminal_shortlist_before_truth():
    archive = _archive()
    runner = TransferConstrainedBO(
        _problem(),
        archive,
        TransferBOConfig(
            method="hyperbo_cbo",
            N=4,
            n0=3,
            seed=29,
            candidate_pool_size=16,
        ),
    )
    result = runner.run(
        freeze_terminal_shortlist=True,
        terminal_probability_slack=0.05,
        terminal_require_provider=True,
    )
    shortlist = result["frozen_terminal_shortlist"]
    assert result["terminal_shortlist_frozen_before_truth_metrics"] is True
    assert len(shortlist) == 2
    assert shortlist[0]["point"] == result["x_recommended"]
    assert shortlist[0]["selector_posterior"] == (
        "transfer_method_specific_delta_chance_margin")
    assert shortlist[1]["coordinate_source"] == (
        "cumulative_risk_psi=(A,N)")
    assert shortlist[1]["target_labels_used"] is False
    assert shortlist[1]["target_oracle_used"] is False
    assert shortlist[1]["verification_samples_used"] is False


def test_transfer_v9_freezes_three_policy_objective_challenger():
    archive = _archive()
    runner = TransferConstrainedBO(
        _problem(),
        archive,
        TransferBOConfig(
            method="hyperbo_cbo",
            N=4,
            n0=3,
            seed=30,
            candidate_pool_size=16,
        ),
    )
    result = runner.run(
        freeze_terminal_shortlist=True,
        terminal_probability_slack=0.05,
        terminal_require_provider=True,
        terminal_shortlist_mode=(
            "posterior_objective_challenger_then_safe"),
        terminal_shortlist_size=3,
        terminal_maximum_violation_probability=0.5,
    )
    shortlist = result["frozen_terminal_shortlist"]
    assert len(shortlist) == 3
    assert len({tuple(row["point"]) for row in shortlist}) == 3
    assert all(row["target_oracle_used"] is False for row in shortlist)
    assert all(
        row["verification_samples_used"] is False for row in shortlist)


def test_transfer_terminal_runner_charges_search_and_verification_separately():
    archive = _archive()
    problem = _problem()
    runner = TransferConstrainedBO(
        problem,
        archive,
        TransferBOConfig(
            method="hyperbo_cbo",
            N=4,
            n0=3,
            seed=31,
            candidate_pool_size=16,
        ),
    )
    result = runner.run(freeze_terminal_shortlist=True)
    args = SimpleNamespace(
        seed=31,
        terminal_verification_primary_budget=8,
        terminal_verification_support_budget=12,
        terminal_verification_delta=0.05,
        terminal_verification_method="normal_quantile_tolerance",
        terminal_verification_mean_delta_fraction=0.5,
    )
    result = _apply_terminal_verification(problem, result, args)
    assert result["n_search_simulations"] == 4
    assert result["n_verification_simulations"] in {8, 20}
    assert result["n_simulations"] == (
        result["n_search_simulations"]
        + result["n_verification_simulations"]
    )
    assert result["terminal_verification_truth_audit"][
        "used_for_selection_or_certification"
    ] is False
    assert result["target_information_contract"][
        "verification_observations_update_posterior"
    ] is False


def test_source_informed_design_rejects_missing_or_duplicate_points():
    with pytest.raises(ValueError, match="requires an explicit"):
        TransferBOConfig(
            method="hyperbo_cbo",
            N=3,
            n0=3,
            initial_design="source_informed",
        )
    with pytest.raises(ValueError, match="must be unique"):
        TransferBOConfig(
            method="hyperbo_cbo",
            N=3,
            n0=3,
            initial_design="source_informed",
            initial_points=((1, 1), (1, 1), (2, 2)),
        )


def test_transfer_progress_separates_source_and_target_phases(capsys):
    archive = _archive()
    config = TransferBOConfig(
        method="hyperbo_cbo",
        N=3,
        n0=3,
        seed=5,
        candidate_pool_size=16,
        source_train_steps=1,
        progress_logging=True,
        progress_label="progress-test",
    )
    runner = TransferConstrainedBO(_problem(), archive, config)
    source_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("SCOLHKG_PROGRESS ")
    ]
    runner.run()
    target_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("SCOLHKG_PROGRESS ")
    ]
    source_payloads = [json.loads(line.split(" ", 1)[1]) for line in source_lines]
    target_payloads = [json.loads(line.split(" ", 1)[1]) for line in target_lines]

    assert source_payloads[0]["kind"] == "source_model_start"
    assert source_payloads[-1]["kind"] == "source_model_done"
    assert source_payloads[-1]["done"] == 3
    assert source_payloads[-1]["total"] == 3
    assert source_payloads[-1]["phase"] == "source_training"
    assert source_payloads[-1]["eta_seconds"] == 0.0
    assert target_payloads[-1]["kind"] == "target_call_done"
    assert target_payloads[-1]["done"] == 3
    assert target_payloads[-1]["total"] == 3
    assert target_payloads[-1]["phase"] == "target_online"
    assert target_payloads[-1]["eta_seconds"] == 0.0


@pytest.mark.parametrize("method", TRANSFER_METHODS)
def test_every_paper_core_method_obeys_the_same_call_contract(method):
    archive = _archive()
    config = TransferBOConfig(
        method=method,
        N=4,
        n0=3,
        seed=5,
        candidate_pool_size=16,
        source_train_steps=2,
        target_finetune_steps=1,
    )
    result = TransferConstrainedBO(_problem(), archive, config).run()
    assert result["n_simulations"] == 4
    assert result["source_archive_fingerprint"] == archive.fingerprint
    assert result["source_information_contract"]["source_simulator_calls"] == 42
    assert result["target_information_contract"]["n0"] == 3
    assert result["target_information_contract"]["target_calls"] == 4
    assert result["target_information_contract"][
        "target_oracle_used_for_selection"
    ] is False
    assert result["target_information_contract"][
        "target_true_sigma_used_for_selection"
    ] is False
    assert result["initial_truth_audit"]["computed_after_recommendation"]
    assert result["initial_truth_audit"]["used_for_selection"] is False
    assert result["initial_truth_audit"]["n"] == 3
    assert result["adaptation_contract"] == METHOD_CONTRACTS[method]
    assert all(row["target_true_sigma_used"] is False
               for row in result["history"])
    assert result["model_diagnostics"]["objective"][
        "adaptation_kind"
    ] == METHOD_CONTRACTS[method]["target_adaptation"]


def test_checkpoint_extension_does_not_repeat_target_calls():
    archive = _archive()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "run.pkl")
        first = TransferConstrainedBO(_problem(), archive, TransferBOConfig(
            method="hyperbo_cbo",
            N=3,
            n0=2,
            seed=17,
            candidate_pool_size=16,
            checkpoint_path=checkpoint,
        )).run()
        second = TransferConstrainedBO(_problem(), archive, TransferBOConfig(
            method="hyperbo_cbo",
            N=5,
            n0=2,
            seed=17,
            candidate_pool_size=16,
            checkpoint_path=checkpoint,
        )).run()
    assert first["n_simulations"] == 3
    assert second["n_simulations"] == 5
    assert second["history"][:3] == first["history"]


def test_official_runtime_report_never_claims_unconfigured_repos():
    report = external_runtime_report()
    assert report["safe_fpacoh_cbo"]["official_adapter_configured"]
    assert report["fsbo_cbo"]["official_adapter_configured"]
    assert report["malibo_cbo"]["official_adapter_configured"]
    assert report["rgpe_cbo"]["official_adapter_configured"]
    assert report["stacked_transfer_gp_cbo"]["official_adapter_configured"]
    assert report["mtgp_cbo"]["official_adapter_configured"]
    assert report["hyperbo_cbo"]["official_adapter_configured"]
    assert report["metabo_cbo"]["official_adapter_configured"]
