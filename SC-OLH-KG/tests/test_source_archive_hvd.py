from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import (  # noqa: E402
    FrozenTransferArchive,
    TransferTaskArchive,
)
from variance.source_archive_hvd import (  # noqa: E402
    DimensionEquivariantRiskProvider,
    FrozenSourceArchiveAleatoricHead,
    SourceArchiveHVDConfig,
)
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class _TargetProblem:
    d = 8
    sigma_level = 0.1
    alpha = 0.05
    tau = 0.0

    @staticmethod
    def normalize(x):
        return np.asarray(x, dtype=float) / 100.0


def _source_task(name, phase):
    provider = DimensionEquivariantRiskProvider()
    positions = np.linspace(0.0, 1.0, 8)
    rows = []
    variances = []
    for index in range(24):
        mean = 0.10 + 0.75 * index / 23.0
        amplitude = 0.03 + 0.18 * ((index + phase) % 7) / 6.0
        profile = np.clip(
            mean + amplitude * np.cos(
                np.pi * (1 + phase) * positions),
            0.0,
            1.0,
        )
        rows.append(profile)
        proxy = np.rint(profile * provider.proxy_scale).astype(int)
        exposure = provider.risk_exposures(proxy)
        variance = (
            0.15
            + 5.0 * exposure.A[0] ** 2
            + 1.5 * exposure.A[1] ** 2
            + 1.2 * exposure.N[0] ** 2
            + 0.8 * exposure.N[1] ** 2
        )
        variances.append(variance)
    X = np.vstack(rows)
    X_integer = np.rint(100.0 * X).astype(int)
    Y_mean = np.zeros((len(X), 2), dtype=float)
    replicate_variance = np.column_stack([
        np.full(len(X), 0.2),
        np.asarray(variances, dtype=float),
    ])
    replicates = tuple(
        np.zeros((3, 2), dtype=float) for _ in range(len(X)))
    return TransferTaskArchive(
        name=name,
        X=X,
        X_integer=X_integer,
        Y_mean=Y_mean,
        Y_replicates=replicates,
        replicate_variance=replicate_variance,
        mean_observation_variance=replicate_variance / 3.0,
        constraint_sigma=np.sqrt(replicate_variance[:, 1]),
        tau=0.0,
        alpha=0.05,
        origins=tuple("test" for _ in range(len(X))),
    )


def _archive():
    return FrozenTransferArchive(
        tasks=(_source_task("source_a", 0), _source_task("source_b", 1)),
        source_seed=0,
        observation_mode="replicated",
        fingerprint="unit-test-fingerprint",
    ).validate()


def _provider_source_task(name, seed):
    problem = ScalarizedProblem(make_problem(
        name,
        d=12,
        L=100,
        sigma=0.04,
        alpha=0.05,
    ))
    rng = np.random.default_rng(seed)
    X_integer = rng.integers(0, 101, size=(32, problem.d))
    X = np.vstack([problem.normalize(point) for point in X_integer])
    variances = np.asarray([
        problem.true_sigma(point) ** 2 for point in X_integer
    ])
    replicates = tuple(
        np.zeros((3, 2), dtype=float) for _ in range(len(X)))
    return TransferTaskArchive(
        name=name,
        X=X,
        X_integer=X_integer,
        Y_mean=np.zeros((len(X), 2), dtype=float),
        Y_replicates=replicates,
        replicate_variance=variances,
        mean_observation_variance=variances / 3.0,
        constraint_sigma=np.sqrt(variances[:, 1]),
        tau=0.0,
        alpha=0.05,
        origins=tuple("provider-test" for _ in range(len(X))),
    )


def _provider_archive():
    return FrozenTransferArchive(
        tasks=(
            _provider_source_task("FactorShockStatePolicyRZDT1", 3),
            _provider_source_task("InventorySupplyChain", 5),
        ),
        source_seed=0,
        observation_mode="replicated",
        fingerprint="provider-unit-test-fingerprint",
    ).validate()


def test_pooled_and_cumulative_heads_share_archive_but_not_shape():
    problem = _TargetProblem()
    archive = _archive()
    pooled = FrozenSourceArchiveAleatoricHead(
        archive=archive,
        target_problem=problem,
        config=SourceArchiveHVDConfig(mode="pooled"),
    )
    factor = FrozenSourceArchiveAleatoricHead(
        archive=archive,
        target_problem=problem,
        config=SourceArchiveHVDConfig(mode="cumulative_factor"),
    )
    points = [
        np.rint(100.0 * np.linspace(0.05, 0.30, 8)).astype(int),
        np.rint(100.0 * np.linspace(0.20, 0.95, 8)).astype(int),
    ]
    pooled_values = np.asarray([
        pooled.predict_variance(point) for point in points])
    factor_values = np.asarray([
        factor.predict_variance(point) for point in points])
    assert np.ptp(pooled_values) <= 1e-12
    assert np.ptp(factor_values) > 1e-6
    assert pooled.archive.fingerprint == factor.archive.fingerprint
    assert pooled.contract_id != factor.contract_id


def test_source_lodo_calibration_is_frozen_and_conservative():
    head = FrozenSourceArchiveAleatoricHead(
        archive=_archive(),
        target_problem=_TargetProblem(),
        config=SourceArchiveHVDConfig(mode="cumulative_factor"),
    )
    point = np.array([5, 10, 20, 35, 55, 70, 85, 95])
    assert (
        head.predict_certification_variance(point)
        >= head.predict_variance(point)
    )
    diagnostics = head.diagnostics()
    assert diagnostics["calibration_multiplier"] >= 1.0
    assert diagnostics["lodo_calibration_row_count"] == 48
    assert diagnostics["target_outcomes_used"] is False
    assert diagnostics["target_oracle_used"] is False
    assert diagnostics["terminal_verifier_labels_used"] is False


def test_dimension_equivariant_coordinate_has_fixed_blocks():
    provider = DimensionEquivariantRiskProvider()
    short = provider.risk_exposures([1000, 3000, 7000, 9000])
    long = provider.risk_exposures(np.linspace(1000, 9000, 1000))
    assert short.A.shape == long.A.shape == (3,)
    assert short.N.shape == long.N.shape == (3,)
    assert np.all(short.A >= 0.0)
    assert np.all(short.N >= 0.0)


def test_provider_cumulative_head_transfers_actual_A_N_shape_without_labels():
    target = ScalarizedProblem(make_problem(
        "QueueResourceControl",
        d=12,
        L=100,
        sigma=0.04,
        alpha=0.05,
    ))
    head = FrozenSourceArchiveAleatoricHead(
        archive=_provider_archive(),
        target_problem=target,
        config=SourceArchiveHVDConfig(
            mode="provider_cumulative_factor"),
    )
    rng = np.random.default_rng(11)
    points = rng.integers(0, 101, size=(128, target.d))
    predicted = np.asarray([
        head.predict_variance(point) for point in points])
    truth = np.asarray([
        target.true_sigma(point)[1] ** 2 for point in points])
    assert np.corrcoef(predicted, truth)[0, 1] > 0.8
    assert np.all(head.predict_certification_variance_many(points) >= predicted)

    decomposition = head.predict_decomposition(points[0])["cumulative"]
    assert np.isclose(
        decomposition["total"],
        decomposition["floor"]
        + decomposition["independent"]
        + decomposition["shared"]
        + decomposition["linear"],
    )
    diagnostics = head.diagnostics()
    shared = np.asarray(diagnostics["hvd"]["parameters"]["B"])
    assert np.min(np.linalg.eigvalsh(shared)) >= -1e-12
    assert diagnostics["coordinate_id"] == "observable_provider_psi_A_N_v1"
    assert diagnostics["provider_schema_is_target_label_free"] is True
    assert diagnostics["target_outcomes_used"] is False
    assert diagnostics["target_oracle_used"] is False
    assert diagnostics["calibration_multiplier"] >= 1.0
    assert diagnostics["lodo_calibration_row_count"] == 2
