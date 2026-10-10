"""Source-calibrated I/C/R prior from saved Gaussian/family-mixture evidence."""
import numpy as np
from scipy.special import logsumexp, roots_legendre


def macro_prior(alpha, q):
    return np.stack(((1. - q) * (1. - alpha), (1. - q) * alpha, q), axis=-1)


def assemble_profile(head_forecasts, risk_forecasts, arm, null_key):
    if [(f["step"], f["candidate"]) for f in head_forecasts] != [
            (f["step"], f["candidate"]) for f in risk_forecasts]:
        raise ValueError("saved head/risk forecast identities differ")
    evidence = np.array([f["head_log_evidence"] for f in head_forecasts])
    densities = np.array([f["eta_component_log_density"] for f in head_forecasts])
    if arm == "fixed_source":
        evidence = np.repeat(evidence[:, :, 1:2], 2, axis=2)
        densities = np.repeat(densities[:, 1:2], 2, axis=1)
    independent = logsumexp(evidence, axis=2).sum(axis=1) - 3. * np.log(2.)
    structural, full = evidence[:, :, 0].sum(axis=1), evidence[:, :, 1].sum(axis=1)
    coordinated = np.logaddexp(structural, full) - np.log(2.)
    score_i = logsumexp(evidence[:, 2] + densities, axis=1) - logsumexp(evidence[:, 2], axis=1)
    score_c = (np.logaddexp(structural + densities[:, 0], full + densities[:, 1])
               - np.logaddexp(structural, full))
    old_i = np.array([f["source"] for f in risk_forecasts])
    error = float(np.max(np.abs(score_i - old_i)))
    if error > 1e-8:
        raise ValueError(f"saved independent-risk density changed: {error}")
    null = np.array([f[null_key] for f in risk_forecasts])
    # Subtract the common I evidence; the R/I ratio comes from A9's 1/2 prior.
    relative_evidence = np.column_stack((np.zeros(len(null)), coordinated - independent,
                                         np.log1p(-null) - np.log(null)))
    scores = np.column_stack((score_i, score_c, [f["adapted"] for f in risk_forecasts]))
    return {"log_evidence": relative_evidence, "log_predictive": scores,
            "identity": [(f["step"], f["candidate"]) for f in head_forecasts],
            "max_independent_control_error": error}


def mixture_scores(log_evidence, log_predictive, masses):
    """Normalize joint numerators by past-data evidence, including zero masses."""
    masses = np.asarray(masses)
    scalar = masses.ndim == 1
    if scalar:
        masses = masses[None, :]
    with np.errstate(divide="ignore"):
        logs = np.log(masses)
    numerator, denominator = None, None
    for k in range(3):
        past = log_evidence[:, k:k + 1] + logs[None, :, k]
        future = past + log_predictive[:, k:k + 1]
        numerator = future if numerator is None else np.logaddexp(numerator, future)
        denominator = past if denominator is None else np.logaddexp(denominator, past)
    scores = numerator - denominator
    return scores[:, 0] if scalar else scores


def posterior_macro_masses(log_evidence, prior):
    with np.errstate(divide="ignore"):
        logs = np.asarray(log_evidence) + np.log(prior)
    return np.exp(logs - logsumexp(logs, axis=1)[:, None])


def fit_source_prior(log_evidence, log_predictive, nodes=64):
    points, weights = roots_legendre(nodes)
    points, weights = (points + 1.) / 2., weights / 2.
    alpha, q = np.repeat(points, nodes), np.tile(points, nodes)
    prior = macro_prior(alpha, q)
    scores = mixture_scores(log_evidence, log_predictive, prior).sum(axis=0)
    log_mass = scores + np.log(np.repeat(weights, nodes) * np.tile(weights, nodes))
    mass = np.exp(log_mass - logsumexp(log_mass))
    a_mean, q_mean = float(mass @ alpha), float(mass @ q)
    a_sd = float(np.sqrt(mass @ ((alpha - a_mean) ** 2)))
    q_sd = float(np.sqrt(mass @ ((q - q_mean) ** 2)))
    covariance = float(mass @ ((alpha - a_mean) * (q - q_mean)))
    return {"alpha_mean": a_mean, "q_mean": q_mean, "alpha_sd": a_sd, "q_sd": q_sd,
            "alpha_q_covariance": covariance, "alpha_q_correlation": covariance / (a_sd * q_sd),
            "macro_prior_mass": (mass @ prior).tolist(), "source_forecasts": len(log_evidence),
            "quadrature_nodes_per_axis": nodes,
            "scope": "Source CV calibration density, not an exact Bayesian posterior or nested validation."}


def concatenate_profiles(profiles):
    return (np.vstack([p["log_evidence"] for p in profiles]),
            np.vstack([p["log_predictive"] for p in profiles]))
