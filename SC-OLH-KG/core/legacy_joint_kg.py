"""Single-episode Gaussian-moment KG for the historical Gaussian verifier.

    Only f and g are observed. Eta is latent; its fantasy mean changes through
    Cov(eta,g), with no direct variance-label observation. Actual paid g updates
    use the Normal likelihood in joint_function_risk, not this approximation.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import eigh_tridiagonal
from scipy.special import ndtr
from scipy.stats import nct, norm


@lru_cache(maxsize=4)
def legacy_verifier_quadrature(replicates, alpha, delta):
    df = replicates - 1
    factor = nct.ppf(1 - delta, df, norm.ppf(1 - alpha) * np.sqrt(replicates)) / np.sqrt(replicates)
    j = np.arange(12, dtype=float)
    nodes, vectors = eigh_tridiagonal(2 * j + df / 2,
                                      np.sqrt(j[1:] * (j[1:] + df / 2 - 1)))
    eta, weights = hermgauss(9)
    return np.sqrt(2) * eta, weights / np.sqrt(np.pi), factor * np.sqrt(2 * nodes / df), vectors[0] ** 2


def clipped_objective_quality(f_mean, f_variance, reference):
    """Historical clipped-quality Gaussian integral in the public response units."""
    sd = np.sqrt(np.maximum(f_variance, 1e-12))
    lo, hi = -f_mean / sd, (reference - f_mean) / sd
    return ((reference - f_mean) * (ndtr(hi) - ndtr(lo))
            + sd * (norm.pdf(hi) - norm.pdf(lo))) / reference + ndtr(lo)


def legacy_joint_utility(f_mean, f_variance, mean_risk, point_covariance, protocol):
    """E[clipped objective quality] times primary v69 certificate power.

    Quality is clipped(1-f/reference,0,1), using the public scalarized reference
    point. This is a declared pilot utility, not the old regret metric or the
    full guarded-shortlist deployment utility.
    """
    quality = clipped_objective_quality(f_mean, f_variance, protocol["objective_reference"])
    eta_nodes, eta_weights, chi_terms, chi_weights = legacy_verifier_quadrature(
        protocol["verification_replicates"], protocol["chance_alpha"], protocol["candidate_delta"])
    g, eta = mean_risk[..., 0], mean_risk[..., 1]
    gv, ev, cross = point_covariance[..., 0, 0], point_covariance[..., 1, 1], point_covariance[..., 0, 1]
    eta_sd = np.sqrt(np.maximum(ev, 0.))
    loading = np.divide(cross, eta_sd, out=np.zeros_like(cross), where=eta_sd > 0.)
    conditional = np.maximum(gv - loading ** 2, 0.)
    probability = np.zeros_like(g)
    for z, weight in zip(eta_nodes, eta_weights):
        sigma = np.exp(.5 * (eta + eta_sd * z))
        total_sd = np.sqrt(conditional + sigma ** 2 / protocol["verification_replicates"])
        probability += weight * (ndtr(-(g[..., None] + loading[..., None] * z
            + sigma[..., None] * chi_terms) / total_sd[..., None]) @ chi_weights)
    return quality * probability, probability


def legacy_raw_joint_kg(f_mean, f_covariance, mean_risk, covariance, noise, protocol, *, order=3):
    """Two observed heads per fantasy; source-estimated f noise, E[exp eta] g noise."""
    n = len(f_mean)
    point_cov = covariance[np.arange(n), :, np.arange(n), :]
    utility, _ = legacy_joint_utility(f_mean, np.diag(f_covariance), mean_risk, point_cov, protocol)
    incumbent = int(utility.argmax())
    nodes, weights = hermgauss(order)
    nodes, weights = np.sqrt(2) * nodes, weights / np.sqrt(np.pi)
    scores = np.zeros(n)
    for action in range(n):
        g_slope = covariance[:, :, action, 0] / np.sqrt(point_cov[action, 0, 0] + noise[action, 1])
        future_cov = point_cov - g_slope[:, :, None] * g_slope[:, None, :]
        f_slope = f_covariance[:, action] / np.sqrt(f_covariance[action, action] + noise[action, 0])
        future_f_variance = np.maximum(np.diag(f_covariance) - f_slope ** 2, 0.)
        for zg, wg in zip(nodes, weights):
            future_risk = mean_risk + g_slope * zg
            for zf, wf in zip(nodes, weights):
                values, _ = legacy_joint_utility(f_mean + f_slope * zf, future_f_variance,
                    future_risk, future_cov, protocol)
                scores[action] += wg * wf * (values.max() - values[incumbent])
    return scores
