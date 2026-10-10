"""Source-anchored Bayesian averaging of one residual risk direction."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, roots_laguerre

from performance.target_risk_residual import training_matrices


def predictive_branches(priors, psi, ids, labels, noise, candidate, scale_mean,
                        nodes=48, null_probability=.5):
    """Return full training-weighted Gaussian branches at the held candidate."""
    locations, quadrature_weights = roots_laguerre(nodes)
    deltas = scale_mean * locations
    logs, means, variances = [], [], []
    component_log_prior = -np.log(len(priors))
    for prior in priors:
        unique, residual, design, covariance, _ = training_matrices(prior, psi, ids, labels, noise)
        factor = cho_factor(covariance, lower=True)
        alpha = cho_solve(factor, residual)
        features = design[:, 1:]
        solved = cho_solve(factor, features)
        information = np.sum(features * solved, axis=0)
        projected = features.T @ alpha
        feature = np.r_[1., prior.basis.transform(psi[candidate:candidate + 1])[0]]
        cross = design @ prior.covariance @ feature + prior.discrepancy_variance * (unique == candidate)
        mean0 = float(feature @ prior.mean + cross @ alpha)
        var0 = max(float(feature @ prior.covariance @ feature + prior.discrepancy_variance -
                         cross @ cho_solve(factor, cross)), 0.)
        base = float(-.5 * residual @ alpha - np.log(np.diag(factor[0])).sum() -
                     len(residual) * np.log(2 * np.pi) / 2)
        logs.append(np.array([component_log_prior + np.log(null_probability) + base]))
        means.append(np.array([mean0]))
        variances.append(np.array([var0]))
        denominator = 1. + information[:, None] * deltas[None, :]
        ratio = deltas[None, :] / denominator
        innovation = feature[1:] - cross @ solved
        evidence = base - .5 * np.log(denominator) + .5 * projected[:, None] ** 2 * ratio
        prior_log_weight = component_log_prior + np.log1p(-null_probability) - np.log(len(information))
        logs.append((evidence + prior_log_weight + np.log(quadrature_weights)[None, :]).reshape(-1))
        means.append((mean0 + innovation[:, None] * projected[:, None] * ratio).reshape(-1))
        variances.append((var0 + innovation[:, None] ** 2 * ratio).reshape(-1))
    logs = np.concatenate(logs)
    log_evidence = float(logsumexp(logs))
    weights = logs - log_evidence
    # Every component's first branch is the unchanged source model.
    stride = 1 + priors[0].basis.feature_dim * nodes
    null_posterior = float(np.exp(weights[np.arange(len(priors)) * stride]).sum())
    return weights, np.concatenate(means), np.concatenate(variances), log_evidence, null_posterior


def joint_label_log_scores(means, variances, labels, noise):
    """Joint repeated-label density with shared candidate eta uncertainty."""
    precision = 1. / noise
    information = precision.sum()
    center = float(precision @ labels / information)
    within = float(precision @ ((labels - center) ** 2))
    denominator = 1. + variances * information
    return -.5 * (len(labels) * np.log(2 * np.pi) + np.log(noise).sum() +
        np.log(denominator) + within + information * (center - means) ** 2 / denominator)


def grouped_mixture_validation(priors, psi, ids, labels, noise, old_folds, scale_mean,
                               nodes=48, null_probability=.5):
    folds = []
    for old in old_folds:
        candidate = old["held_candidate"]
        train = ids != candidate
        if np.unique(ids[train]).tolist() != old["training_candidates"]:
            raise ValueError("A6 changed the frozen candidate-group holdout")
        weights, means, variances, evidence, null_posterior = predictive_branches(
            priors, psi, ids[train], labels[train], noise[train], candidate,
            scale_mean, nodes, null_probability)
        score = float(logsumexp(weights + joint_label_log_scores(means, variances, labels[~train], noise[~train])))
        folds.append({"held_candidate": int(candidate), "held_batches": int((~train).sum()),
            "training_candidates": old["training_candidates"], "null_posterior": null_posterior,
            "training_log_evidence": evidence, "old_joint_log_score": old["old_joint_log_score"],
            "mixture_joint_log_score": score})
    return {"old_log_score_per_batch": sum(f["old_joint_log_score"] for f in folds) / len(ids),
            "mixture_log_score_per_batch": sum(f["mixture_joint_log_score"] for f in folds) / len(ids),
            "folds": folds}
