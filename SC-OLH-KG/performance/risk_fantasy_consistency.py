"""Finite-mixture risk fantasies for A45's fixed-mean diagnostic.

Inputs are paid-history posterior arrays. No simulator or live model is used.
All recommendation candidates remain inside each fantasy's utility maximum.
"""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from core.verification_kg import _pass_probability, objective_quality


def mixture_moments(means, covariances, weights):
    mean = means @ weights
    centered = means - mean[:, None]
    covariance = np.einsum("b,bij->ij", weights, covariances)
    covariance += (centered * weights) @ centered.T
    return mean, covariance


def condition_fantasies(means, covariances, weights, probes, noise, nodes):
    """Condition each Gaussian and Bayes-update its probability for each label.

Label axes are (probe, generating component, quadrature node). The updated
component is the final axis; recommendation is the leading prediction axis.
"""
    means, covariances, weights = map(np.asarray, (means, covariances, weights))
    probes = np.asarray(probes, dtype=int)
    diagonal = np.diagonal(covariances, axis1=1, axis2=2).T
    observed_variance = diagonal[probes] + noise
    points, quadrature = hermgauss(nodes)
    quadrature /= np.sqrt(np.pi)
    labels = (means[probes, :, None]
              + np.sqrt(2 * observed_variance[:, :, None]) * points)
    residual = labels[..., None] - means[probes, None, None, :]
    log_density = -.5 * (residual ** 2 / observed_variance[:, None, None, :]
                         + np.log(2 * np.pi * observed_variance[:, None, None, :]))
    with np.errstate(divide="ignore"):
        posterior = log_density + np.log(weights)
    posterior = np.exp(posterior - logsumexp(posterior, axis=-1, keepdims=True))
    cross = np.moveaxis(covariances[:, :, probes], 0, -1)
    gain = cross / observed_variance[None]
    future_mean = means[:, None, None, None, :] + gain[:, :, None, None, :] * residual[None]
    future_variance = np.maximum(diagonal[:, None, :] - gain * cross, 0.)
    return labels, quadrature, posterior, future_mean, future_variance


def mixture_scores(means, covariances, weights, probes, noise,
                   fixed_mean, fixed_variance, protocol, nodes=9):
    """B: refreshed probabilities; F: fixed probabilities, identical label law."""
    _, quadrature, posterior, future_mean, future_variance = condition_fantasies(
        means, covariances, weights, probes, noise, nodes)
    quality = objective_quality(fixed_mean, fixed_variance)
    probability = _pass_probability(
        fixed_mean[:, 1, None, None, None, None],
        fixed_variance[:, 1, None, None, None, None],
        future_mean, future_variance[:, :, None, None, :], protocol)
    refreshed = np.max(quality[:, None, None, None]
                       * np.sum(probability * posterior[None], axis=-1), axis=0)
    frozen = np.max(quality[:, None, None, None]
                    * np.sum(probability * weights, axis=-1), axis=0)
    diagonal = np.diagonal(covariances, axis1=1, axis2=2).T
    current = quality * (_pass_probability(
        fixed_mean[:, 1, None], fixed_variance[:, 1, None],
        means, diagonal, protocol) @ weights)
    baseline = float(np.max(current))
    refreshed_future = np.einsum("sar,a,r->s", refreshed, weights, quadrature)
    frozen_future = np.einsum("sar,a,r->s", frozen, weights, quadrature)
    return {"current_max": baseline,
            "B_future_max": refreshed_future, "F_future_max": frozen_future,
            "B": refreshed_future - baseline, "F": frozen_future - baseline}


def gaussian_scores(mean, covariance, probes, noise, fixed_mean,
                    fixed_variance, protocol, nodes=9):
    """Moment-collapsed Gaussian KG, with its own current utility maximum."""
    diagonal = np.maximum(np.diag(covariance), 0.)
    points, quadrature = hermgauss(nodes)
    quadrature /= np.sqrt(np.pi)
    slope = covariance[:, probes] / np.sqrt(diagonal[probes] + noise)[None]
    future_mean = mean[:, None, None] + slope[..., None] * (np.sqrt(2) * points)
    future_variance = np.maximum(diagonal[:, None] - slope ** 2, 0.)
    quality = objective_quality(fixed_mean, fixed_variance)
    probability = _pass_probability(
        fixed_mean[:, 1, None, None], fixed_variance[:, 1, None, None],
        future_mean, future_variance[..., None], protocol)
    expected = np.max(quality[:, None, None] * probability, axis=0) @ quadrature
    baseline = float(np.max(quality * _pass_probability(
        fixed_mean[:, 1], fixed_variance[:, 1], mean, diagonal, protocol)))
    return {"current_max": baseline, "future_max": expected, "G": expected - baseline}


def probe_scores(means, covariances, weights, probes, noise,
                 fixed_mean, fixed_variance, protocol, nodes=9):
    mixture = mixture_scores(means, covariances, weights, probes, noise,
                             fixed_mean, fixed_variance, protocol, nodes)
    mean, covariance = mixture_moments(means, covariances, weights)
    gaussian = gaussian_scores(mean, covariance, probes, noise,
                               fixed_mean, fixed_variance, protocol, nodes)
    return {**mixture, "G": gaussian["G"],
            "G_current_max": gaussian["current_max"],
            "G_future_max": gaussian["future_max"]}
