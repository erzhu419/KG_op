"""Deterministic checks of the branch formulas used in the replay."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import ndtr, roots_hermitenorm
from scipy.stats import binom

from score_cascade_prefix import (
    SIZES,
    contribution_coefficients,
    cubature_terms,
    term_bounds,
)


def test_branch_probabilities_for_independent_policy_means():
    mean = np.array([[-1.8, -2.1, -1.5]])
    latent_sd = np.array([0.04, 0.055, 0.03])
    covariance = np.diag(latent_sd**2)
    observation_sd = np.ones(3)
    nodes, weights = roots_hermitenorm(64)
    violation = ndtr(mean + nodes[:, None] * latent_sd)
    p0, p1, p2 = weights @ binom.cdf(1, SIZES, violation) / np.sqrt(2 * np.pi)
    exact = np.array([p0, p0 * p1, (1 - p0) * p1, (1 - p0) * (1 - p1) * p2])

    terms = cubature_terms(mean, covariance, observation_sd, 9)[0]
    lower, upper = term_bounds(mean, covariance, observation_sd)
    assert_allclose(terms, exact, rtol=1e-8, atol=1e-10)
    assert np.all(lower[0] <= exact) and np.all(exact <= upper[0])


def test_guard_correction_includes_latent_objective_covariance():
    fmean = np.array([[1.2, 1.6, 1.9]])
    fcov = np.array([[0.4, 0.2, 0], [0.2, 0.5, 0], [0, 0, 0.3]])
    variance = np.array([0.7, 0.3, 1]) * SIZES
    latent_difference_var = fcov[0, 0] + fcov[1, 1] - 2 * fcov[0, 1]
    nodes, weights = roots_hermitenorm(64)
    latent_difference = fmean[0, 0] - fmean[0, 1] + nodes * np.sqrt(latent_difference_var)
    integral = np.dot(weights, latent_difference * ndtr(latent_difference)) / np.sqrt(2 * np.pi)

    correction = contribution_coefficients(fmean, fcov, variance, 2.0)[0, 1]
    assert_allclose(correction, integral, rtol=1e-12, atol=1e-13)
