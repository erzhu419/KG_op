"""An empirical source-function prior retaining task-wise variance directions."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from representation.source_function_prior import SourceFunctionPrior


def source_coefficient_moments(priors, class_weights, design, response, noise):
    """Condition source coefficients in an intercept-projected likelihood.

    Distinct collapsed candidates have independent discrepancy. This returns
    coefficient second moments, not a coefficient mean to transfer to targets.
    """
    means, diagonal_variances, log_evidence = [], [], []
    for prior in priors:
        covariance = prior.covariance[1:, 1:]
        cross = covariance @ design.T
        kernel = design @ cross + noise + prior.discrepancy_variance * np.eye(len(response))
        factor = cho_factor(kernel, lower=True)
        alpha = cho_solve(factor, response)
        mean = cross @ alpha
        variance = np.diag(covariance) - np.einsum("ij,ji->i", cross, cho_solve(factor, cross.T))
        means.append(mean); diagonal_variances.append(variance)
        log_evidence.append(-np.log(np.diag(factor[0])).sum() - .5 * response @ alpha
                            - .5 * len(response) * np.log(2 * np.pi))
    log_posterior = np.log(class_weights) + log_evidence
    weights = np.exp(log_posterior - logsumexp(log_posterior))
    means, diagonal_variances = np.array(means), np.array(diagonal_variances)
    return {"second_moment": weights @ (diagonal_variances + means ** 2),
            "posterior_mean": means, "posterior_diagonal_variance": diagonal_variances,
            "class_posterior": weights, "class_log_evidence": np.array(log_evidence)}


def task_direction_distribution(source_priors):
    """Return equally weighted task covariances and their moment collapse.

    Only covariance directions transfer. A common source intercept removes
    individual archive offsets from target archive-identity inference.
    """
    first = source_priors[0]
    mean = np.zeros_like(first.mean)
    mean[0] = np.mean([p.mean[0] for p in source_priors])
    components = [SourceFunctionPrior(p.basis, mean.copy(), p.covariance.copy(),
                  p.discrepancy_variance, p.diagnostics) for p in source_priors]
    collapsed = SourceFunctionPrior(first.basis, mean.copy(),
        np.mean([p.covariance for p in source_priors], axis=0),
        float(np.mean([p.discrepancy_variance for p in source_priors])), {})
    return components, collapsed
