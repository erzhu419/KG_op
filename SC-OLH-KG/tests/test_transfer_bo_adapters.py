import tempfile
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import (  # noqa: E402
    FrozenTransferArchive,
    TransferTaskArchive,
    _archive_fingerprint,
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
