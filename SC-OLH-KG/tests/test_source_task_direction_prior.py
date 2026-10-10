import numpy as np

from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.source_task_direction_prior import source_coefficient_moments, task_direction_distribution


def test_moment_collapse_preserves_candidate_moments_without_offset_transfer():
    bank = np.random.default_rng(902).normal(size=(8, 2))
    basis = FixedFunctionBasis(bank, [0, 1])
    source = [SourceFunctionPrior(basis, np.array([offset, 0., 0.]), np.diag(v), .02, {})
              for offset, v in [(1., [.25, 1., .01]), (3., [.25, .01, 1.])]]
    components, collapsed = task_direction_distribution(source)
    design = np.column_stack([np.ones(len(bank)), basis.transform(bank)])
    np.testing.assert_allclose(design @ collapsed.mean, np.full(8, 2.))
    mixed = np.mean([np.einsum('ij,jk,ik->i', design, p.covariance, design) + p.discrepancy_variance
                     for p in components], axis=0)
    np.testing.assert_allclose(mixed, np.einsum('ij,jk,ik->i', design, collapsed.covariance, design) + .02)
    assert source[0].mean[0] == 1. and source[1].mean[0] == 3.


def test_task_distribution_retains_joint_sparsity_lost_by_covariance_collapse():
    basis = FixedFunctionBasis(np.random.default_rng(903).normal(size=(8, 2)), [0, 1])
    source = [SourceFunctionPrior(basis, np.zeros(3), np.diag([.25, a, b]), .02, {})
              for a, b in [(1., .01), (.01, 1.)]]
    components, collapsed = task_direction_distribution(source)
    # Independent zero-mean coefficients in each atom have E[b1^2 b2^2]=v1*v2.
    mixed_fourth = np.mean([p.covariance[1, 1] * p.covariance[2, 2] for p in components])
    assert np.isclose(mixed_fourth, .01)
    assert np.isclose(collapsed.covariance[1, 1] * collapsed.covariance[2, 2], .505 ** 2)


def test_source_posterior_matches_independent_precision_form_regression():
    from scipy.stats import multivariate_normal

    design = np.array([[1., -.4], [.3, .8], [-.5, .2]])
    response = np.array([.7, -.2, .1])
    noise = np.array([[.2, .02, 0.], [.02, .3, -.01], [0., -.01, .25]])
    basis = FixedFunctionBasis(np.random.default_rng(904).normal(size=(8, 2)), [0, 1])
    source = [SourceFunctionPrior(basis, np.array([offset, 0., 0.]), np.diag([.25, a, b]), .02, {})
              for offset, a, b in [(1., .7, .1), (3., .2, .9)]]
    result = source_coefficient_moments(source, [.6, .4], design, response, noise)
    known_noise = noise + .02 * np.eye(3)
    noise_precision = np.linalg.inv(known_noise)
    moments, evidence = [], []
    for k, prior in enumerate(source):
        covariance = prior.covariance[1:, 1:]
        posterior_covariance = np.linalg.inv(np.linalg.inv(covariance) + design.T @ noise_precision @ design)
        posterior_mean = posterior_covariance @ design.T @ noise_precision @ response
        np.testing.assert_allclose(result['posterior_mean'][k], posterior_mean, atol=1e-13)
        np.testing.assert_allclose(result['posterior_diagonal_variance'][k], np.diag(posterior_covariance), atol=1e-13)
        moments.append(np.diag(posterior_covariance) + posterior_mean ** 2)
        evidence.append(multivariate_normal.logpdf(response, cov=design @ covariance @ design.T + known_noise))
    unnormalized = np.array([.6, .4]) * np.exp(evidence)
    weights = unnormalized / unnormalized.sum()
    np.testing.assert_allclose(result['class_posterior'], weights, atol=1e-13)
    np.testing.assert_allclose(result['second_moment'], weights @ moments, atol=1e-13)


def test_zero_source_residual_retains_posterior_uncertainty():
    basis = FixedFunctionBasis(np.random.default_rng(905).normal(size=(8, 2)), [0, 1])
    prior = SourceFunctionPrior(basis, np.zeros(3), np.diag([.25, 1., 2.]), .02, {})
    result = source_coefficient_moments([prior], [1.], np.array([[1., 0.]]), np.zeros(1), np.array([[.5]]))
    np.testing.assert_allclose(result['second_moment'], [.52 / 1.52, 2.])
    assert np.all(result['posterior_mean'] == 0.)
