"""Public-box price quadrature with conditional Gaussian log-risk filtering.

Price support stays bounded through every likelihood update. Eta is projected
to Gaussian moments conditional on each price node after each paid batch.
Only the already observed sites are stored; other sites use the unchanged
Gaussian conditional field. Acquisition can use the existing moment rule.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import digamma, logsumexp
from scipy.stats import multivariate_normal, norm, truncnorm

from representation.physical_risk_prior import log_sample_variance_density


def box_quadrature(mean, covariance, bounds, order):
    """Sequential conditional truncated-Normal proposal and its log density."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    nodes, weights = leggauss(order)
    u = np.stack(np.meshgrid(*([(nodes + 1) / 2] * len(mean)), indexing="ij"), axis=-1).reshape(-1, len(mean))
    cubature_weights = np.prod(np.stack(np.meshgrid(*([weights / 2] * len(mean)), indexing="ij"), axis=-1), axis=-1).reshape(-1)
    cholesky = np.linalg.cholesky(covariance)
    standardized, prices = np.zeros_like(u), np.zeros_like(u)
    log_proposal = np.zeros(len(u))
    lo, hi = bounds
    for j in range(len(mean)):
        center = mean[j] + standardized[:, :j] @ cholesky[j, :j]
        sd = cholesky[j, j]
        a, b = (lo - center) / sd, (hi - center) / sd
        standardized[:, j] = truncnorm.ppf(u[:, j], a, b)
        prices[:, j] = center + sd * standardized[:, j]
        log_proposal += truncnorm.logpdf(prices[:, j], a, b, loc=center, scale=sd)
    return prices, np.log(cubature_weights) - log_proposal


def price_log_density(prices, covariance, bounds, normalizer_order=9):
    """Normalized restricted Gaussian, or the uniform public-box reference."""
    lo, hi = bounds
    if covariance is None:
        return np.full(len(prices), -prices.shape[1] * np.log(hi - lo))
    center = np.full(prices.shape[1], (lo + hi) / 2)
    nodes, integration = box_quadrature(center, covariance, bounds, normalizer_order)
    log_normalizer = logsumexp(integration + multivariate_normal.logpdf(nodes, center, covariance))
    return multivariate_normal.logpdf(prices, center, covariance) - log_normalizer


def price_nodes(design, labels, noise, covariance, bounds, order, proposal_scale=1.):
    """Observed labels adapt integration placement, without changing the prior."""
    d = design.shape[1]
    center = np.full(d, sum(bounds) / 2)
    # A uniform prior needs a proposal scale; its own box variance supplies it.
    proposal_covariance = (np.eye(d) * (bounds[1] - bounds[0]) ** 2 / 12
                           if covariance is None else covariance)
    precision = np.linalg.inv(proposal_covariance)
    weighted = design / noise[:, None]
    conditional_covariance = np.linalg.inv(precision + design.T @ weighted)
    conditional_mean = conditional_covariance @ (precision @ center + weighted.T @ labels)
    prices, integration = box_quadrature(conditional_mean, proposal_scale * conditional_covariance, bounds, order)
    log_weights = integration + price_log_density(prices, covariance, bounds)
    return prices, log_weights


def conditional_eta_update(mean, variance, g, g_label, eta_label, replicates, order=25):
    """Vectorized one-dimensional integration conditional on fixed prices."""
    k = (replicates - 1) / 2
    sample_variance = np.exp(eta_label + digamma(k) - np.log(k))
    residual = (replicates - 1) * sample_variance + replicates * (g_label - g) ** 2
    mode = np.array(mean, copy=True)
    for _ in range(12):
        tail = .5 * residual * np.exp(-mode)
        step = (-(mode - mean) / variance - replicates / 2 + tail) / (1 / variance + tail)
        mode += step
        if np.max(np.abs(step)) < 1e-10:
            break
    width = 1 / np.sqrt(1 / variance + .5 * residual * np.exp(-mode))
    nodes, weights = hermgauss(order)
    eta = mode[:, None] + width[:, None] * np.sqrt(2) * nodes
    likelihood = norm.logpdf(g_label, loc=g[:, None], scale=np.exp(.5 * eta) / np.sqrt(replicates))
    likelihood += log_sample_variance_density(eta, sample_variance, replicates)
    log_mass = (np.log(weights / np.sqrt(np.pi)) + likelihood
                + norm.logpdf(eta, mean[:, None], np.sqrt(variance[:, None]))
                - norm.logpdf(eta, mode[:, None], width[:, None]))
    evidence = logsumexp(log_mass, axis=1)
    posterior_weights = np.exp(log_mass - evidence[:, None])
    new_mean = np.sum(posterior_weights * eta, axis=1)
    new_variance = np.sum(posterior_weights * (eta - new_mean[:, None]) ** 2, axis=1)
    return new_mean, new_variance, evidence


def gaussian_field(prior, psi):
    design = np.column_stack([np.ones(len(psi)), prior.basis.transform(psi)])
    return design @ prior.mean, design @ prior.covariance @ design.T + prior.discrepancy_variance * np.eye(len(psi))


def bounded_objective(design, offset, observations, covariance, bounds, order):
    indices = np.array([r["index"] for r in observations])
    labels = np.array([r["response"][0] for r in observations]) - np.asarray(offset)[indices]
    noise = np.array([r["noise"][0] for r in observations])
    prices, logs = price_nodes(design[indices], labels, noise, covariance, bounds, order)
    prediction = prices @ design[indices].T
    logs += norm.logpdf(labels, prediction, np.sqrt(noise)).sum(axis=1)
    evidence = float(logsumexp(logs))
    weights = np.exp(logs - evidence)
    price_mean = weights @ prices
    centered = prices - price_mean
    price_cov = (centered.T * weights) @ centered
    return {"mean": offset + design @ price_mean, "covariance": design @ price_cov @ design.T,
            "log_evidence": evidence, "price_mean": price_mean, "price_covariance": price_cov}


def bounded_joint(design, offset, observations, price_covariance, bounds, eta_priors, psi,
                  order=7, eta_order=25):
    indices = np.array([r["index"] for r in observations])
    sites, positions = np.unique(indices, return_inverse=True)
    labels = np.array([r["response"][1] for r in observations]) - np.asarray(offset)[indices]
    proposal_noise = np.array([r["noise"][1] for r in observations])
    # Wider integration placement covers uncertain mean noise. Importance
    # weights divide out this proposal; no prior or likelihood is inflated.
    prices, prior_logs = price_nodes(design[indices], labels, proposal_noise, price_covariance, bounds, order, 2.)
    g_nodes = prices @ design.T + offset
    components = []
    for prior in eta_priors:
        initial_mean, initial_cov = gaussian_field(prior, psi)
        site_cov = initial_cov[np.ix_(sites, sites)]
        gain = np.linalg.solve(site_cov, initial_cov[sites]).T
        conditional_cov = initial_cov - gain @ initial_cov[sites]
        mean = np.broadcast_to(initial_mean[sites], (len(prices), len(sites))).copy()
        covariance = np.broadcast_to(site_cov, (len(prices), len(sites), len(sites))).copy()
        logs = prior_logs.copy() - np.log(len(eta_priors))
        for observation, site in zip(observations, positions):
            local_mean, local_var = mean[:, site], covariance[:, site, site]
            new_mean, new_var, evidence = conditional_eta_update(local_mean, local_var,
                g_nodes[:, observation["index"]], observation["response"][1], observation["response"][2],
                observation["replicates"], eta_order)
            slope = covariance[:, :, site] / local_var[:, None]
            mean += slope * (new_mean - local_mean)[:, None]
            covariance += ((new_var - local_var)[:, None, None] * slope[:, :, None] * slope[:, None, :])
            logs += evidence
        total = float(logsumexp(logs))
        weights = np.exp(logs - total)
        price_mean = weights @ prices
        dp = prices - price_mean
        site_mean = weights @ mean
        de = mean - site_mean
        pcov = (dp.T * weights) @ dp
        ecov = np.einsum("m,mij->ij", weights, covariance) + (de.T * weights) @ de
        cross = (dp.T * weights) @ de
        prediction = np.column_stack([offset + design @ price_mean,
            initial_mean + gain @ (site_mean - initial_mean[sites])])
        field_cov = np.empty((len(psi), 2, len(psi), 2))
        field_cov[:, 0, :, 0] = design @ pcov @ design.T
        field_cov[:, 1, :, 1] = conditional_cov + gain @ ecov @ gain.T
        field_cov[:, 0, :, 1] = design @ cross @ gain.T
        field_cov[:, 1, :, 0] = field_cov[:, 0, :, 1].T
        components.append({"mean": prediction, "covariance": field_cov,
            "log_evidence": total, "price_mean": price_mean, "price_covariance": pcov})
    return mix_moments(components)


def mix_moments(components):
    logs = np.array([r["log_evidence"] for r in components])
    total = float(logsumexp(logs))
    weights = np.exp(logs - total)
    mean = sum(w * c["mean"] for w, c in zip(weights, components))
    shape = components[0]["covariance"].shape
    covariance = np.zeros(shape)
    for weight, component in zip(weights, components):
        difference = (component["mean"] - mean).reshape(-1)
        covariance += weight * (component["covariance"] + np.outer(difference, difference).reshape(shape))
    result = {"mean": mean, "covariance": covariance, "log_evidence": total,
              "component_weights": weights.tolist()}
    if all("price_mean" in c for c in components):
        price_mean = sum(w * c["price_mean"] for w, c in zip(weights, components))
        price_covariance = sum(w * (c["price_covariance"] + np.outer(c["price_mean"] - price_mean,
            c["price_mean"] - price_mean)) for w, c in zip(weights, components))
        result.update(price_mean=price_mean, price_covariance=price_covariance)
    return result
