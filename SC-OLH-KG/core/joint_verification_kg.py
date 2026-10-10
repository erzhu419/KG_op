"""Verification KG with correlated Gaussian moments of (g, log variance).

The objective head is independent. Paid observations still use the finite
particle joint likelihood; only acquisition and recommendation project that
posterior to joint Gaussian moments. Future labels use fixed predictive noise.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr

from core.verification_kg import _objective_quality, verification_quadrature


def correlated_pass_probability(g_mean, g_variance, eta_mean, eta_variance, cross_covariance, protocol):
    """Integrate g conditional on eta, retaining their posterior covariance."""
    eta_nodes, eta_weights, chi_terms, chi_weights = verification_quadrature(
        protocol["verification_replicates"], protocol["chance_alpha"])
    eta_sd = np.sqrt(np.maximum(eta_variance, 0.))
    loading = np.divide(cross_covariance, eta_sd, out=np.zeros_like(eta_sd), where=eta_sd > 0.)
    conditional_variance = np.maximum(g_variance - loading ** 2, 0.)
    answer = np.zeros(np.broadcast_shapes(g_mean.shape, eta_mean.shape, conditional_variance.shape))
    for z, weight in zip(eta_nodes, eta_weights):
        sigma = np.exp(.5 * (eta_mean + z * eta_sd))
        conditional_mean = g_mean + loading * z
        sd = np.sqrt(conditional_variance + sigma ** 2 / protocol["verification_replicates"])
        argument = -(conditional_mean[..., None] + sigma[..., None] * chi_terms) / sd[..., None]
        ndtr(argument, out=argument)
        answer += weight * (argument @ chi_weights)
    return answer


def joint_verification_utility(f_mean, f_variance, mean_risk, point_covariance, protocol):
    """Common recommendation utility, under the joint Gaussian projection."""
    probability = correlated_pass_probability(
        mean_risk[..., 0], point_covariance[..., 0, 0], mean_risk[..., 1],
        point_covariance[..., 1, 1], point_covariance[..., 0, 1], protocol)
    return _objective_quality(f_mean, f_variance) * probability, probability


def joint_verification_kg(f_mean, f_covariance, mean_risk, covariance, measurement_noise, protocol, *, order=3):
    """KG from an independent objective and correlated (g, eta) posterior.

    Shapes: f_mean (K,), f_covariance (K,K), mean_risk (K,2), covariance
    (K,2,K,2), measurement_noise (K,3), with independent batch-label errors.
    Compare each fantasy's best choice to the current incumbent *in that same
    fantasy*. The tower property gives the usual KG expectation; this centered
    quadrature removes baseline-integration drift and is exactly zero for one
    candidate. It does not make Gaussian or finite-node fantasies exact.
    """
    f_mean, f_covariance = np.asarray(f_mean), np.asarray(f_covariance)
    mean_risk, covariance, noise = map(np.asarray, (mean_risk, covariance, measurement_noise))
    n = len(f_mean)
    if f_covariance.shape != (n, n) or mean_risk.shape != (n, 2) or covariance.shape != (n, 2, n, 2) or noise.shape != (n, 3):
        raise ValueError("Joint KG requires full candidate covariance and three batch noise variances")
    indices = np.arange(n)
    point_cov = covariance[indices, :, indices, :]
    f_variance = np.maximum(np.diag(f_covariance), 0.)
    utility, _ = joint_verification_utility(f_mean, f_variance, mean_risk, point_cov, protocol)
    incumbent = int(np.argmax(utility))
    nodes, weights = hermgauss(order)
    nodes, weights = np.sqrt(2) * nodes, weights / np.sqrt(np.pi)
    normals = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    pair_weights = (weights[:, None] * weights[None, :]).reshape(-1)
    scores = np.empty(n)
    for start in range(0, n, 32):
        stop = min(start + 32, n)
        observation_cov = point_cov[start:stop].copy()
        observation_cov[:, 0, 0] += noise[start:stop, 1]
        observation_cov[:, 1, 1] += noise[start:stop, 2]
        cholesky = np.linalg.cholesky(observation_cov)
        cross = covariance[:, :, start:stop, :].transpose(0, 2, 1, 3)
        slope = np.linalg.solve(cholesky[None], cross.swapaxes(-1, -2)).swapaxes(-1, -2)
        future_cov = point_cov[:, None] - slope @ slope.swapaxes(-1, -2)
        future_mean = mean_risk[:, None, :, None] + np.einsum("kajd,sd->kajs", slope, normals)
        probability = correlated_pass_probability(
            future_mean[:, :, 0, :], future_cov[:, :, 0, 0, None],
            future_mean[:, :, 1, :], future_cov[:, :, 1, 1, None], future_cov[:, :, 0, 1, None], protocol)
        f_slope = f_covariance[:, start:stop] / np.sqrt(f_variance[start:stop] + noise[start:stop, 0])[None, :]
        future_f_mean = f_mean[:, None, None] + f_slope[:, :, None] * nodes
        future_f_variance = np.maximum(f_variance[:, None] - f_slope ** 2, 0.)
        quality = _objective_quality(future_f_mean, future_f_variance[..., None])
        score = np.zeros(stop - start)
        for j, weight in enumerate(weights):
            values = probability * quality[:, :, j, None]
            score += weight * ((np.max(values, axis=0) - values[incumbent]) @ pair_weights)
        scores[start:stop] = score
    return scores


def budgeted_joint_kg(f_mean, f_covariance, mean_risk, covariance, measurement_noise,
                     protocol, remaining_calls, *, order=5):
    """One-step net value when a new batch consumes the verification budget.

    A nonpositive maximum means stop under this declared one-step rule. Both
    stopping and continuing spend the same remaining simulator-call budget.
    The comparison uses Gaussian projected utilities and fixed-node KG.
    """
    after_calls = remaining_calls - protocol["replicates"]
    if after_calls < 2:
        raise ValueError("A search batch must leave at least two verification calls")
    before_protocol = {**protocol, "verification_replicates": remaining_calls}
    after_protocol = {**protocol, "verification_replicates": after_calls}
    indices = np.arange(len(f_mean))
    point_covariance = covariance[indices, :, indices, :]
    before, _ = joint_verification_utility(f_mean, np.diag(f_covariance), mean_risk, point_covariance, before_protocol)
    after, _ = joint_verification_utility(f_mean, np.diag(f_covariance), mean_risk, point_covariance, after_protocol)
    information = joint_verification_kg(f_mean, f_covariance, mean_risk, covariance,
        measurement_noise, after_protocol, order=order)
    return information + float(after.max()) - float(before.max())
