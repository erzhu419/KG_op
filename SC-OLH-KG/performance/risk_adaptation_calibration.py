"""Calibrate source/family predictive mixing using earlier paid risk labels."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from performance.audit_meta_prior_risk_family_prequential import model_log_scores
from performance.risk_family_reweighting import family_predictive_branches
from performance.target_risk_direction_mixture import joint_label_log_scores


def fit_predictive_weight(source_scores, adapted_scores):
    """Maximize a concave sum of log mixture densities, including endpoints."""
    source_scores, adapted_scores = np.asarray(source_scores), np.asarray(adapted_scores)
    scale = np.maximum(source_scores, adapted_scores)
    source, adapted = np.exp(source_scores - scale), np.exp(adapted_scores - scale)
    difference = adapted - source

    def derivative(pi):
        return float(np.sum(difference / ((1. - pi) * source + pi * adapted)))

    if derivative(0.) <= 0.:
        pi = 0.
    elif derivative(1.) >= 0.:
        pi = 1.
    else:
        pi = float(brentq(derivative, 0., 1., xtol=1e-12))
    score = float(np.sum(scale + np.log((1. - pi) * source + pi * adapted)))
    return pi, score


def fit_calibration(priors, psi, ids, labels, noise, families, nodes=32):
    folds = []
    for candidate in np.unique(ids):
        train = ids != candidate
        weights, means, variances, _ = family_predictive_branches(
            priors, psi, ids[train], labels[train], noise[train], int(candidate), families, nodes)
        scores = joint_label_log_scores(means, variances, labels[~train], noise[~train])
        models = model_log_scores(weights, scores, len(priors), len(families), nodes)
        folds.append({"held_candidate": int(candidate), "held_batches": int(np.sum(~train)),
            "training_candidates": np.unique(ids[train]).tolist(),
            "source_log_score": models["source"], "adapted_log_score": models["adapted"]})
    pi, score = fit_predictive_weight([f["source_log_score"] for f in folds],
                                     [f["adapted_log_score"] for f in folds])
    return {"pi": pi, "inner_log_score": score, "folds": folds}


def calibrated_predictive_branches(priors, psi, ids, labels, noise, candidate, families, nodes=32):
    """Issue a predictor using history only, before receiving the next label."""
    fit = fit_calibration(priors, psi, ids, labels, noise, families, nodes)
    weights, means, variances, summary = family_predictive_branches(
        priors, psi, ids, labels, noise, candidate, families, nodes)
    null = np.zeros(len(weights), dtype=bool)
    null[np.arange(len(priors)) * (1 + len(families) * nodes)] = True
    calibrated = np.full_like(weights, -np.inf)
    pi = fit["pi"]
    if pi < 1.:
        calibrated[null] = weights[null] - logsumexp(weights[null]) + np.log1p(-pi)
    if pi > 0.:
        calibrated[~null] = weights[~null] - logsumexp(weights[~null]) + np.log(pi)
    return calibrated, means, variances, weights, {**fit, "a9_null_posterior": summary["null_posterior"]}


def prequential_forecasts(priors, psi, ids, labels, noise, families, nodes=32, steps=range(9, 25)):
    forecasts = []
    for step in steps:
        t = step - 1
        weights, means, variances, original, fit = calibrated_predictive_branches(priors, psi,
            ids[:t], labels[:t], noise[:t], int(ids[t]), families, nodes)
        # Current/future labels and observation noise reach only the scorer.
        scores = joint_label_log_scores(means, variances, labels[t:t + 1], noise[t:t + 1])
        models = model_log_scores(original, scores, len(priors), len(families), nodes)
        mass = np.exp(weights)
        mean = float(mass @ means)
        variance = float(mass @ (variances + (means - mean) ** 2))
        forecasts.append({"step": int(step), "candidate": int(ids[t]),
            "new_candidate": bool(ids[t] not in ids[:t]),
            "pi": fit["pi"], "inner_log_score": fit["inner_log_score"],
            "inner_candidate_groups": len(fit["folds"]), "a9_null_posterior": fit["a9_null_posterior"],
            "issued_eta_mean": mean, "issued_eta_variance": variance,
            "calibrated_log_score": float(logsumexp(weights + scores)), **models})
    return forecasts
