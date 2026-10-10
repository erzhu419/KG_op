"""Local batch likelihood probe; no trajectory update or acquisition policy."""
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import digamma, gammaln, logsumexp, ndtr

from core.verification_kg import verification_quadrature


def normal_logpdf(value, mean, variance):
    return -.5 * (np.log(2 * np.pi * variance) + (value - mean) ** 2 / variance)


def local_update(g_components, eta_components, response, noise, protocol,
                 label_model, mean_noise_model):
    """Integrate one paid Gaussian batch under supplied independent head priors.

    Components are (weight, mean, variance). Exactness refers to the batch
    likelihood, conditional on these legacy Gaussian component priors.
    The retained eta quadrature and conditional g Gaussian preserve their
    posterior dependence when scoring the independent verifier.
    """
    reps = protocol["replicates"]
    k = (reps - 1) / 2
    bias = float(digamma(k) - np.log(k))
    log_sample_variance = float(response[2]) + bias
    _, _, chi_terms, chi_weights = verification_quadrature(
        protocol["verification_replicates"], protocol["chance_alpha"])
    pairs, max_relative_error = [], 0.0
    for gi, (gw, gm, gv) in enumerate(g_components):
        for ei, (ew, em, ev) in enumerate(eta_components):
            def at_eta(eta):
                sigma2 = np.exp(eta)
                observation_noise = sigma2 / reps if mean_noise_model == "latent_variance" else noise[1]
                total = gv + observation_noise
                g_mean = gm + gv / total * (response[1] - gm)
                g_variance = gv * observation_noise / total
                log_density = normal_logpdf(eta, em, ev)
                if label_model == "log_chi_square":
                    difference = log_sample_variance - eta
                    log_density += k * np.log(k) - gammaln(k) + k * difference - k * np.exp(difference)
                else:
                    log_density += normal_logpdf(response[2], eta, noise[2])
                log_density += normal_logpdf(response[1], gm, total)
                return log_density, g_mean, g_variance, sigma2

            # A finite Gaussian-prior interval avoids overflow during integration.
            # Splitting at the mode resolves narrow posteriors inside that interval.
            sd = np.sqrt(ev)
            lo, hi = em - 12 * sd, em + 12 * sd
            mode = minimize_scalar(lambda eta: -at_eta(eta)[0], bounds=(lo, hi),
                                   method="bounded").x
            shift = at_eta(mode)[0]
            def integrand(eta):
                density, g_mean, g_variance, sigma2 = at_eta(eta)
                power = ndtr(-(g_mean + np.sqrt(sigma2) * chi_terms) /
                             np.sqrt(g_variance + sigma2 / protocol["verification_replicates"])) @ chi_weights
                statistics = np.array([1., g_mean, g_variance + g_mean ** 2,
                                       eta, eta ** 2, g_mean * eta, power])
                return np.exp(density - shift) * statistics
            integral, error = quad_vec(integrand, lo, hi, points=[mode], epsabs=1e-9, epsrel=1e-9)
            max_relative_error = max(max_relative_error, float(error / integral[0]))
            pairs.append((gi, ei, np.log(gw) + np.log(ew) + shift + np.log(integral[0]),
                          integral / integral[0]))
    log_evidence = float(logsumexp([p[2] for p in pairs]))
    weights = np.exp(np.array([p[2] for p in pairs]) - log_evidence)
    total = sum(w * pair[3] for w, pair in zip(weights, pairs))
    g_mean, eta_mean = float(total[1]), float(total[3])
    gv = max(float(total[2] - g_mean ** 2), 0.)
    ev = max(float(total[4] - eta_mean ** 2), 0.)
    covariance = float(total[5] - g_mean * eta_mean)
    return {"g_mean": g_mean, "g_variance": gv, "eta_mean": eta_mean, "eta_variance": ev,
            "g_eta_covariance": covariance, "g_eta_correlation": covariance / np.sqrt(gv * ev),
            "predicted_verifier_power": float(total[6]), "log_batch_evidence": log_evidence,
            "g_component_weights": [float(sum(w for w, p in zip(weights, pairs) if p[0] == i))
                                   for i in range(len(g_components))],
            "eta_component_weights": [float(sum(w for w, p in zip(weights, pairs) if p[1] == i))
                                     for i in range(len(eta_components))],
            "integration_relative_error": max_relative_error}
