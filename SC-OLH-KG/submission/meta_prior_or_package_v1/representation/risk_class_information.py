"""Candidate risk predictions and information about the learned task class."""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp


def candidate_risk_predictions(priors, psi, past_ids, labels, noise, future_noise):
    """Predict a future eta batch at each public candidate from paid history."""
    ids = np.asarray(past_ids)
    means, variances = [], []
    for prior in priors:
        design = np.column_stack([np.ones(len(psi)), prior.basis.transform(psi)])
        past = design[ids]
        covariance = past @ prior.covariance @ past.T
        covariance += prior.discrepancy_variance * (ids[:, None] == ids[None, :]) + np.diag(noise)
        cross = design @ prior.covariance @ past.T
        cross += prior.discrepancy_variance * (np.arange(len(psi))[:, None] == ids[None, :])
        factor = cho_factor(covariance, lower=True)
        mean = design @ prior.mean + cross @ cho_solve(factor, labels - past @ prior.mean)
        variance = np.einsum("ij,jk,ik->i", design, prior.covariance, design) + prior.discrepancy_variance
        variance -= np.einsum("ij,ji->i", cross, cho_solve(factor, cross.T))
        means.append(mean); variances.append(variance + future_noise)
    return np.array(means).T, np.array(variances).T


def risk_class_information(means, variances, weights, nodes=49):
    """I(next eta; local/ordinary-meta/high-frequency-meta) in nats.

    The local structural/full atoms form one state with their retained mixture;
    the two learned task covariance atoms each form a separate state.
    """
    weights = np.asarray(weights)
    points, quadrature = hermgauss(nodes)
    quadrature /= np.sqrt(np.pi)
    draws = means[:, :, None] + np.sqrt(2 * variances[:, :, None]) * points
    log_density = -.5 * ((draws[:, :, :, None] - means[:, None, None, :]) ** 2 / variances[:, None, None, :]
                       + np.log(2 * np.pi * variances[:, None, None, :]))
    log_joint = log_density + np.log(weights)
    total = logsumexp(log_joint, axis=-1)
    local = logsumexp(log_joint[:, :, :, :2], axis=-1) - np.log(weights[:2].sum())
    state_density = np.stack([local[:, 0], local[:, 1], log_density[:, 2, :, 2], log_density[:, 3, :, 3]], axis=1)
    information = np.sum((state_density - total) * quadrature[None, None, :] * weights[None, :, None], axis=(1, 2))
    return information
