"""Four risk branches and source-seed-held empirical macro calibration."""
import numpy as np
from scipy.special import logsumexp, roots_legendre

from performance.audit_meta_prior_risk_family_prequential import model_log_scores
from performance.risk_family_reweighting import family_predictive_branches
from performance.target_risk_direction_mixture import joint_label_log_scores


def selected_function_forecasts(priors, task, selected, nodes):
    psi, labels, noise = task["psi"], task["response"][:, 2], task["noise"][:, 2]
    forecasts = []
    for t in range(8, 24):
        weights, means, variances, summary = family_predictive_branches(
            priors, psi, np.arange(t), labels[:t], noise[:t], t, {"selected": selected}, nodes)
        score = joint_label_log_scores(means, variances, labels[t:t + 1], noise[t:t + 1])
        null = summary["null_posterior"]
        forecasts.append({"step": t + 1, "candidate": t,
            **model_log_scores(weights, score, len(priors), 1, nodes),
            "relative_log_evidence": float(np.log1p(-null) - np.log(null))})
    return forecasts


def node_grid(nodes):
    x, w = roots_legendre(nodes)
    x, w = (x + 1.) / 2., w / 2.
    u, v, q = [a.ravel() for a in np.meshgrid(x, x, x, indexing="ij")]
    theta = np.column_stack((1. - u, u * (1. - v), u * v))
    masses = np.column_stack(((1. - q[:, None]) * theta, q))
    # Uniform Dirichlet(1,1,1): u=theta_F+theta_G has density 2u.
    weights = np.einsum("i,j,k->ijk", w * 2. * x, w, w).ravel()
    return masses, np.log(weights), theta, q


def predictive_scores(evidence, predictive, masses):
    masses = np.asarray(masses)
    scalar = masses.ndim == 1
    if scalar:
        masses = masses[None, :]
    with np.errstate(divide="ignore"):
        logs = np.log(masses)
    numerator = denominator = None
    for k in range(4):
        past = evidence[:, k:k + 1] + logs[None, :, k]
        future = past + predictive[:, k:k + 1]
        numerator = future if numerator is None else np.logaddexp(numerator, future)
        denominator = past if denominator is None else np.logaddexp(denominator, past)
    scores = numerator - denominator
    return scores[:, 0] if scalar else scores


def outer_priors(seed_scores, grid):
    masses, log_weights, theta, q = grid
    result = {}
    for held in sorted(seed_scores):
        # Explicit exclusion keeps the issued prior independent of held scores.
        scores = sum(seed_scores[s] for s in seed_scores if s != held)
        posterior = np.exp(scores + log_weights - logsumexp(scores + log_weights))
        q_mean = float(posterior @ q)
        theta_mean = posterior @ theta
        result[held] = {"held_seed": held, "training_seeds": sorted(s for s in seed_scores if s != held),
            "macro_prior_mass": (posterior @ masses).tolist(), "theta_mean": theta_mean.tolist(),
            "theta_sd": np.sqrt(posterior @ ((theta - theta_mean) ** 2)).tolist(),
            "q_mean": q_mean, "q_sd": float(np.sqrt(posterior @ ((q - q_mean) ** 2))),
            "mean_risk_source_theta_product": (posterior @ ((1. - q[:, None]) * theta)).tolist()}
    return result
