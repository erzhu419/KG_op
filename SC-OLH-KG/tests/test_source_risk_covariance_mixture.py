import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import multivariate_normal

from representation.source_risk_covariance_mixture import covariance_mixture_objective, projected_source_blocks
from representation.source_function_prior import FixedFunctionBasis
from representation.transferable_spectral import SourceDomainBatch


def example():
    rng = np.random.default_rng(3610)
    blocks = [(rng.normal(size=(n, 4)), rng.normal(size=n), np.diag(rng.uniform(.1, .3, n))) for n in (5, 7, 6)]
    parameters = np.r_[rng.uniform(.01, .4, 8), .7]
    return rng, blocks, parameters


def test_task_mixture_gradient_matches_numerical_derivative():
    _, blocks, parameters = example()
    numerical = approx_derivative(lambda x: covariance_mixture_objective(x, blocks)[0], parameters).reshape(-1)
    np.testing.assert_allclose(covariance_mixture_objective(parameters, blocks)[1], numerical, rtol=1e-6, atol=1e-8)


def test_class_swap_preserves_likelihood_and_reorders_gradient():
    _, blocks, parameters = example()
    swapped = np.r_[parameters[4:8], parameters[:4], -parameters[-1]]
    value, gradient, posterior = covariance_mixture_objective(parameters, blocks, return_responsibilities=True)
    other, other_gradient, other_posterior = covariance_mixture_objective(swapped, blocks, return_responsibilities=True)
    np.testing.assert_allclose(other, value, atol=1e-12)
    np.testing.assert_allclose(other_gradient, np.r_[gradient[4:8], gradient[:4], -gradient[-1]], atol=1e-12)
    np.testing.assert_allclose(other_posterior, posterior[:, ::-1], atol=1e-12)


def test_identical_classes_equal_independent_gaussian_reference():
    _, blocks, parameters = example()
    parameters[4:8] = parameters[:4]
    value, gradient, posterior = covariance_mixture_objective(parameters, blocks, return_responsibilities=True)
    reference = -sum(multivariate_normal.logpdf(y, mean=np.zeros(len(y)), cov=noise + (Z * parameters[:4]) @ Z.T)
                     for Z, y, noise in blocks) / sum(len(y) for _, y, _ in blocks)
    np.testing.assert_allclose(value, reference, atol=1e-12)
    np.testing.assert_allclose(posterior[:, 0], 1 / (1 + np.exp(-parameters[-1])), atol=1e-12)
    np.testing.assert_allclose(gradient[-1], 0., atol=1e-12)


def test_task_intercept_shifts_cannot_change_covariance_likelihood():
    rng, _, _ = example()
    basis = FixedFunctionBasis(rng.uniform(-2, 2, (96, 8)))
    psi, y = rng.uniform(-2, 2, (12, 8)), rng.normal(size=12)
    noise = rng.uniform(.1, .3, 12)
    one = SourceDomainBatch("one", psi, y)
    shifted = SourceDomainBatch("one", psi, y + 50)
    first, signal = projected_source_blocks([one], [noise], basis)
    second, other_signal = projected_source_blocks([shifted], [noise], basis)
    for left, right in zip(first[0], second[0]):
        np.testing.assert_allclose(left, right, atol=1e-12)
    np.testing.assert_allclose(signal, other_signal, atol=1e-12)
