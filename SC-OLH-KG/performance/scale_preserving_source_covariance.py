"""Source-selected covariance sharing that retains learned within-set scales."""
import numpy as np

from performance.audit_meta_prior_risk_family_prequential import model_log_scores
from performance.risk_family_reweighting import family_predictive_branches
from performance.target_risk_direction_mixture import joint_label_log_scores


def scale_preserving_covariance(prior, indices, rho):
    covariance = prior.covariance.copy()
    source = prior.covariance[1:, 1:]
    extra = np.zeros_like(source)
    positions = np.asarray(indices)
    extra[positions, positions] = np.diag(source)[positions]
    budget = np.trace(source)
    endpoint = budget * (source + extra) / (budget + np.trace(extra))
    covariance[1:, 1:] = (1. - rho) * source + rho * endpoint
    return covariance


def selected_scale_forecasts(priors, task, selected, nodes):
    psi, labels, noise = task["psi"], task["response"][:, 2], task["noise"][:, 2]
    return selected_path_forecasts(priors, psi, np.arange(24), labels, noise, selected, nodes)


def selected_path_forecasts(priors, psi, ids, labels, noise, selected, nodes):
    """Score old query paths, including repeated visits, from earlier eta labels."""
    forecasts = []
    for t in range(8, 24):
        weights, means, variances, summary = family_predictive_branches(
            priors, psi, ids[:t], labels[:t], noise[:t], int(ids[t]),
            {"selected": selected}, nodes, covariance_builder=scale_preserving_covariance)
        score = joint_label_log_scores(means, variances, labels[t:t + 1], noise[t:t + 1])
        null = summary["null_posterior"]
        forecasts.append({"step": t + 1, "candidate": int(ids[t]),
            **model_log_scores(weights, score, len(priors), 1, nodes),
            "relative_log_evidence": float(np.log1p(-null) - np.log(null))})
    return forecasts
