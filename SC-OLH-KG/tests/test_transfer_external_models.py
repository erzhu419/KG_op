import pytest


torch = pytest.importorskip("torch")

from baselines.transfer_external_models import (  # noqa: E402
    _adaptive_positive_definite_jitter,
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
