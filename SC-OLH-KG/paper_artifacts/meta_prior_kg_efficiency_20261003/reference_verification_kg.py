"""KG for the same finite-budget verification utility used at recommendation."""

from functools import lru_cache
import itertools

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import eigh_tridiagonal
from scipy.special import ndtr, polygamma
from scipy.stats import chi2, norm, t


@lru_cache(maxsize=4)
def verification_quadrature(reps, alpha):
    # Mean and sample variance of an independent Gaussian validation sample.
    df = reps - 1
    eta_nodes, eta_weights = hermgauss(9)
    # Golub-Welsch gives normalized Gamma weights directly. Unnormalized
    # generalized Laguerre weights overflow at the V6 validation budgets.
    alpha_gamma = df / 2 - 1
    j = np.arange(12, dtype=float)
    chi_nodes, eigenvectors = eigh_tridiagonal(
        2 * j + alpha_gamma + 1, np.sqrt(j[1:] * (j[1:] + alpha_gamma)))
    chi_weights = eigenvectors[0] ** 2
    constant = t.ppf(.975, df) / np.sqrt(reps) + norm.ppf(1 - alpha) * np.sqrt(
        df / chi2.ppf(.025, df))
    return (np.sqrt(2) * eta_nodes, eta_weights / np.sqrt(np.pi),
            np.sqrt(2 * chi_nodes / df) * constant, chi_weights / chi_weights.sum())


def predictive_pass_probability(mean, variance, protocol):
    """Posterior predictive power of the unchanged independent verifier.

    Conditional on log variance and chi-square sample variance, integrate the
    uncertain constraint mean and validation sample mean analytically. The two
    remaining one-dimensional integrals use fixed Gaussian quadrature.
    """
    eta_nodes, eta_weights, chi_terms, chi_weights = verification_quadrature(
        protocol["verification_replicates"], protocol["chance_alpha"])
    mean, variance = np.asarray(mean), np.asarray(variance)
    shape = np.broadcast_shapes(mean.shape[:-1], variance.shape[:-1])
    answer = np.zeros(shape)
    for z, weight in zip(eta_nodes, eta_weights):
        sigma = np.exp(.5 * (mean[..., 2] + z * np.sqrt(variance[..., 2])))
        total_sd = np.sqrt(variance[..., 1] + sigma ** 2 / protocol["verification_replicates"])
        argument = -(mean[..., 1, None] + sigma[..., None] * chi_terms) / total_sd[..., None]
        answer += weight * (ndtr(argument) @ chi_weights)
    return answer


def objective_quality(mean, variance):
    """Expected bounded quality, one at f=-5 and zero at f=5.

    Public simulator coefficients bound the true objective inside [-5,5]. A
    failed verification has reward zero. This declared operational utility is
    distinct from the original mathematical chance-constrained optimum.
    """
    sd = np.sqrt(np.maximum(variance[..., 0], 1e-12))
    lo, hi = (-5 - mean[..., 0]) / sd, (5 - mean[..., 0]) / sd
    mass = ndtr(hi) - ndtr(lo)
    return ((5 - mean[..., 0]) * mass + sd * (norm.pdf(hi) - norm.pdf(lo))) / 10 + ndtr(lo)


def verification_utility(mean, variance, protocol):
    probability = predictive_pass_probability(mean, variance, protocol)
    return probability * objective_quality(mean, variance), probability


def posterior_covariances(models, candidates):
    out = []
    for model in models:
        A = model.augmented_feature_matrix(candidates)
        covariance = A @ model.C @ A.T
        unseen = np.array([tuple(x) not in model.sol_to_idx for x in candidates])
        covariance[np.diag_indices(len(candidates))] += model.lambda_i * unseen
        out.append(covariance)
    return np.stack(out, axis=-1)


def verification_kg_scores(models, candidates, protocol):
    """Joint three-head Gaussian fantasies, including covariance downdates.

    The 27-node integration is numerical KG under a Gaussian approximation.
    Mixture weights and learned hyperparameters stay fixed in these fantasies;
    paid observations update the live model subsequently.
    """
    mean = np.column_stack([m.posterior_mean_many(candidates) for m in models])
    variance = np.column_stack([m.posterior_var_many(candidates) for m in models])
    covariance = posterior_covariances(models, candidates)
    reps = protocol["replicates"]
    noise = np.column_stack([np.full(len(candidates), .12 ** 2 / reps),
          np.exp(mean[:, 2] + .5 * variance[:, 2]) / reps,
          np.full(len(candidates), float(polygamma(1, (reps - 1) / 2)))])
    scale = np.sqrt(variance + noise)
    nodes, weights = hermgauss(3)
    triples = list(itertools.product(range(3), repeat=3))
    normals = np.array([[np.sqrt(2) * nodes[j] for j in triple] for triple in triples])
    joint_weights = np.array([np.prod([weights[j] / np.sqrt(np.pi) for j in triple])
                              for triple in triples])
    utility, _ = verification_utility(mean, variance, protocol)
    scores = np.empty(len(candidates))
    # Block sampling actions only; every recommendation and all 27 fantasy
    # nodes remain in each action's expectation. This bounds the dense tensor.
    for start in range(0, len(candidates), 32):
        stop = min(start + 32, len(candidates))
        slope = covariance[:, start:stop, :] / scale[None, start:stop, :]
        future_variance = np.maximum(variance[:, None, :] - slope ** 2, 0.0)
        future_mean = mean[:, None, None, :] + slope[:, :, None, :] * normals[None, None, :, :]
        future_utility, _ = verification_utility(
            future_mean, future_variance[:, :, None, :], protocol)
        scores[start:stop] = np.max(future_utility, axis=0) @ joint_weights - float(np.max(utility))
    return scores


def risk_only_kg_scores(model, candidates, fixed_mean, fixed_variance, protocol):
    """Risk-head KG with an identical, frozen mean context across risk arms."""
    mean, variance = np.array(fixed_mean, copy=True), np.array(fixed_variance, copy=True)
    mean[:, 2] = model.posterior_mean_many(candidates)
    variance[:, 2] = model.posterior_var_many(candidates)
    covariance = posterior_covariances([model], candidates)[..., 0]
    noise = float(polygamma(1, (protocol["replicates"] - 1) / 2))
    slope = covariance / np.sqrt(variance[:, 2] + noise)[None, :]
    nodes, weights = hermgauss(3)
    n = len(candidates)
    future_mean = np.broadcast_to(mean[:, None, None, :], (n, n, 3, 3)).copy()
    future_mean[..., 2] += slope[..., None] * (np.sqrt(2) * nodes)
    future_variance = np.broadcast_to(variance[:, None, :], (n, n, 3)).copy()
    future_variance[..., 2] = np.maximum(variance[:, 2, None] - slope ** 2, 0)
    future_utility, _ = verification_utility(future_mean, future_variance[:, :, None, :], protocol)
    current, _ = verification_utility(mean, variance, protocol)
    return np.max(future_utility, axis=0) @ (weights / np.sqrt(np.pi)) - float(np.max(current))
