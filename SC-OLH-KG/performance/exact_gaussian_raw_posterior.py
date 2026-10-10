"""Deterministic conditioning on raw f values without Gaussian projections.

The frozen initial f and eta fields are independent Gaussians. Conditional on
the paid eta values, f conditioning is Gaussian and future eta remains Gaussian.
Only the at-most-four-dimensional paid eta integral needs tensor quadrature.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import digamma, logsumexp

from representation.physical_risk_prior import log_sample_variance_density


def quadrature(prior, paid, order):
    k = len(paid); x, w = hermgauss(order)
    indices = np.indices((order,) * k).reshape(k, -1).T
    standard = np.sqrt(2.) * x[indices]
    eta = prior['eta_mean'][:k] + standard @ np.linalg.cholesky(prior['eta_covariance'][:k, :k]).T
    log_weights = np.log(w[indices] / np.sqrt(np.pi)).sum(axis=1)
    residual = np.asarray(paid) - prior['f_mean'][:k]
    for start in range(0, len(eta), 2048):
        batch = eta[start:start + 2048]
        covariance = np.broadcast_to(prior['f_covariance'][:k, :k], (len(batch), k, k)).copy()
        covariance[:, np.arange(k), np.arange(k)] += np.exp(batch)
        factor = np.linalg.cholesky(covariance)
        standardized = np.linalg.solve(factor, np.broadcast_to(residual, (len(batch), k))[..., None])[..., 0]
        log_weights[start:start + len(batch)] += -.5 * (
            k * np.log(2 * np.pi) + 2 * np.log(np.diagonal(factor, axis1=1, axis2=2)).sum(axis=1)
            + np.sum(standardized ** 2, axis=1))
    evidence = float(logsumexp(log_weights)); log_weights -= evidence
    gain_eta = np.linalg.solve(prior['eta_covariance'][:k, :k], prior['eta_covariance'][:k]).T
    eta_conditional = prior['eta_covariance'] - gain_eta @ prior['eta_covariance'][:k]
    assert np.diag(eta_conditional)[4:].min() > 0
    return {'eta': eta, 'log_weights': log_weights, 'paid': np.asarray(paid), 'k': k,
        'evidence': evidence, 'order': order, 'gain_eta': gain_eta,
        'eta_conditional_covariance': .5 * (eta_conditional + eta_conditional.T)}


def conditional_blocks(prior, grid):
    k = grid['k']; cross = prior['f_covariance'][:, :k]
    residual = grid['paid'] - prior['f_mean'][:k]
    for start in range(0, len(grid['eta']), 2048):
        eta = grid['eta'][start:start + 2048]
        covariance = np.broadcast_to(prior['f_covariance'][:k, :k], (len(eta), k, k)).copy()
        covariance[:, np.arange(k), np.arange(k)] += np.exp(eta)
        rhs = np.column_stack([residual, cross.T])
        solved = np.linalg.solve(covariance, np.broadcast_to(rhs, (len(eta), *rhs.shape)))
        displacement = solved[:, :, 0] @ cross.T
        correction = np.einsum('mk,bkn->bmn', cross, solved[:, :, 1:])
        eta_displacement = (eta - prior['eta_mean'][:k]) @ grid['gain_eta'].T
        yield grid['log_weights'][start:start + len(eta)], displacement, eta_displacement, correction


def moments(prior, grid):
    m = len(prior['f_mean']); df, de = np.zeros(m), np.zeros(m)
    ff, ee, fe, correction_mean = [np.zeros((m, m)) for _ in range(4)]
    for log_weights, displacement, eta_displacement, correction in conditional_blocks(prior, grid):
        w = np.exp(log_weights)
        df += w @ displacement; de += w @ eta_displacement
        ff += (displacement.T * w) @ displacement
        ee += (eta_displacement.T * w) @ eta_displacement
        fe += (displacement.T * w) @ eta_displacement
        correction_mean += np.einsum('b,bij->ij', w, correction)
    return {'f_mean': prior['f_mean'] + df, 'eta_mean': prior['eta_mean'] + de,
        'f_covariance': prior['f_covariance'] - correction_mean + ff - np.outer(df, df),
        'eta_covariance': grid['eta_conditional_covariance'] + ee - np.outer(de, de),
        'cross_covariance': fe - np.outer(df, de)}


def future_batch_density(prior, grid, labels, future_order):
    x, w = hermgauss(future_order); log_future_weights = np.log(w / np.sqrt(np.pi))
    sd = np.sqrt(np.diag(grid['eta_conditional_covariance'])[4:])
    log_density = np.full(len(labels), -np.inf)
    sample_variances = np.exp(labels[:, 3] + digamma(1.))
    for log_weights, displacement, eta_displacement, correction in conditional_blocks(prior, grid):
        means = prior['f_mean'][4:] + displacement[:, 4:]
        variances = np.diag(prior['f_covariance'])[4:] - np.diagonal(correction, axis1=1, axis2=2)[:, 4:]
        eta_mean = prior['eta_mean'][4:] + eta_displacement[:, 4:]
        eta = eta_mean[:, :, None] + np.sqrt(2.) * sd[None, :, None] * x
        total = variances[:, :, None] + np.exp(eta) / 3
        assert total.min() > 0
        joint = log_sample_variance_density(eta, sample_variances[None, :, None], 3) - .5 * (
            np.log(2 * np.pi * total) + (labels[None, :, 0, None] - means[:, :, None]) ** 2 / total)
        value = logsumexp(log_weights[:, None, None] + log_future_weights + joint, axis=(0, 2))
        log_density = np.logaddexp(log_density, value)
    return log_density
