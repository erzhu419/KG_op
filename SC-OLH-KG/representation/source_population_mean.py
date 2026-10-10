"""Source-only common coefficient mean, holding task covariance fixed."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from representation.source_function_prior import SourceFunctionPrior


def fit_source_population_mean(prior, batches, observation_variances):
    """Posterior mean of m with beta_t|m ~ N(m,C), m ~ N(m0,C).

    Task-level coefficient uncertainty remains correlated across sites in V_t.
    This attribution uses the resulting m as an empirical-Bayes point estimate
    and preserves the original new-task covariance and discrepancy exactly.
    It is not a simultaneous refit of between-task residual covariance.
    """
    dimension = len(prior.mean)
    precision = cho_solve(cho_factor(prior.covariance, lower=True), np.eye(dimension))
    displacement_rhs = np.zeros(dimension)
    task_count, label_count = 0, 0
    for batch, noise in zip(batches, observation_variances):
        design = np.column_stack([np.ones(len(batch.psi)), prior.basis.transform(batch.psi)])
        variance = design @ prior.covariance @ design.T
        variance += np.diag(np.asarray(noise) + prior.discrepancy_variance)
        factor = cho_factor(variance, lower=True)
        precision += design.T @ cho_solve(factor, design)
        displacement_rhs += design.T @ cho_solve(factor, np.asarray(batch.signals)-design @ prior.mean)
        task_count += 1; label_count += len(design)
    factor = cho_factor(precision, lower=True)
    mean = prior.mean + cho_solve(factor, displacement_rhs)
    mean_covariance = cho_solve(factor, np.eye(dimension))
    info = {**prior.diagnostics, 'coefficient_mean': mean.tolist(),
            'population_mean_estimator': 'Gaussian source-task common-mean posterior point estimate',
            'population_mean_source_task_count': task_count, 'population_mean_label_count': label_count,
            'population_mean_posterior_variance': np.diag(mean_covariance).tolist(),
            'task_covariance_and_discrepancy_refitted': False,
            'new_task_covariance_includes_population_mean_uncertainty': False}
    return SourceFunctionPrior(prior.basis, mean, prior.covariance.copy(), prior.discrepancy_variance, info)
