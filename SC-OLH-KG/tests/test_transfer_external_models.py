import pytest


torch = pytest.importorskip("torch")

from baselines.transfer_external_models import (  # noqa: E402
    _adaptive_positive_definite_jitter,
    _stable_torch_multivariate_normal,
)


def test_adaptive_jitter_repairs_indefinite_functional_prior_covariance():
    covariance = torch.tensor(
        [[1.0, 1.001], [1.001, 1.0]],
        dtype=torch.double,
    )

    repaired, jitter, retries = _adaptive_positive_definite_jitter(
        covariance,
        initial_jitter=1e-6,
        max_attempts=16,
    )

    torch.linalg.cholesky(repaired)
    assert jitter >= 0.001
    assert retries > 0
    assert torch.allclose(repaired, repaired.T)


def test_spectral_jitter_repairs_beyond_legacy_retry_ceiling():
    covariance = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0]],
        dtype=torch.double,
    )

    repaired, jitter, retries = _adaptive_positive_definite_jitter(
        covariance,
        initial_jitter=1e-6,
        max_attempts=2,
    )

    torch.linalg.cholesky(repaired)
    assert jitter >= 1.0
    assert retries >= 19


def test_adaptive_jitter_is_deterministic():
    covariance = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5]],
        dtype=torch.double,
    )

    first = _adaptive_positive_definite_jitter(
        covariance, initial_jitter=1e-8)
    second = _adaptive_positive_definite_jitter(
        covariance, initial_jitter=1e-8)

    assert torch.equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_stable_torch_mvn_repairs_posterior_covariance():
    mean = torch.zeros(2, dtype=torch.double)
    covariance = torch.tensor(
        [[1.0, 1.0005], [1.0005, 1.0]],
        dtype=torch.double,
    )

    distribution, jitter, retries = _stable_torch_multivariate_normal(
        mean,
        covariance,
        initial_jitter=1e-6,
    )

    assert jitter >= 0.0005
    assert retries > 0
    assert torch.isfinite(distribution.log_prob(mean))
