"""Oracle-only contrast: add risk covariance versus redistribute it."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.special import logsumexp

from performance.target_risk_direction_mixture import joint_label_log_scores
from performance.target_risk_joint import quadrature
from performance.target_risk_residual import training_matrices
from representation.source_function_prior import SourceFunctionPrior


def support_base(prior, support, reallocate):
    if not reallocate:
        return prior
    covariance = prior.covariance.copy()
    budget = float(np.trace(covariance[1:, 1:]))
    covariance[1:, 1:] = 0.
    covariance[np.asarray(support) + 1, np.asarray(support) + 1] = budget / len(support)
    return SourceFunctionPrior(prior.basis, prior.mean.copy(), covariance,
                               prior.discrepancy_variance, prior.diagnostics)


def base_statistics(prior, psi, ids, labels, noise, candidate):
    unique, residual, design, covariance, _ = training_matrices(prior, psi, ids, labels, noise)
    factor = cho_factor(covariance, lower=True)
    alpha = cho_solve(factor, residual)
    feature = np.r_[1., prior.basis.transform(psi[candidate:candidate + 1])[0]]
    cross = design @ prior.covariance @ feature + prior.discrepancy_variance * (unique == candidate)
    mean = float(feature @ prior.mean + cross @ alpha)
    variance = max(float(feature @ prior.covariance @ feature + prior.discrepancy_variance -
                         cross @ cho_solve(factor, cross)), 0.)
    evidence = float(-.5 * residual @ alpha - np.log(np.diag(factor[0])).sum() -
                     len(residual) * np.log(2 * np.pi) / 2)
    return factor, alpha, design, feature, cross, mean, variance, evidence


def oracle_branches(priors, psi, ids, labels, noise, candidate, support, scale_mean,
                    reallocate, nodes=48):
    locations, qweights = quadrature(nodes)
    deltas = scale_mean * locations
    logs, means, variances = [], [], []
    weight = np.log(.5) - np.log(len(priors))
    for prior in priors:
        original = base_statistics(prior, psi, ids, labels, noise, candidate)
        logs.append(np.array([weight + original[-1]]))
        means.append(np.array([original[-3]]))
        variances.append(np.array([original[-2]]))
        corrected = base_statistics(support_base(prior, support, reallocate), psi, ids, labels, noise, candidate)
        factor, alpha, design, feature, cross, mean0, var0, base = corrected
        features = design[:, np.asarray(support) + 1]
        solved = cho_solve(factor, features)
        gram = features.T @ solved
        values, vectors = eigh((gram + gram.T) / 2)
        values = np.maximum(values, 0.)
        projected = vectors.T @ (features.T @ alpha)
        innovation = vectors.T @ (feature[np.asarray(support) + 1] - cross @ solved)
        denominator = 1. + values[:, None] * deltas[None, :] / len(support)
        ratio = deltas[None, :] / len(support) / denominator
        evidence = base - .5 * np.log(denominator).sum(axis=0) + .5 * (projected[:, None] ** 2 * ratio).sum(axis=0)
        logs.append(weight + np.log(qweights) + evidence)
        means.append(mean0 + (innovation[:, None] * projected[:, None] * ratio).sum(axis=0))
        variances.append(var0 + (innovation[:, None] ** 2 * ratio).sum(axis=0))
    logs = np.concatenate(logs)
    weights = logs - logsumexp(logs)
    null_posterior = float(np.exp(weights[np.arange(len(priors)) * (nodes + 1)]).sum())
    return weights, np.concatenate(means), np.concatenate(variances), null_posterior


def grouped_oracle_validation(priors, psi, ids, labels, noise, old_folds, support, scale_mean, nodes=48):
    folds = []
    for old in old_folds:
        candidate = old["held_candidate"]
        train = ids != candidate
        if np.unique(ids[train]).tolist() != old["training_candidates"]:
            raise ValueError("A8 changed the frozen candidate-group holdout")
        fold = {"held_candidate": int(candidate), "held_batches": int((~train).sum()),
                "training_candidates": old["training_candidates"], "old_joint_log_score": old["old_joint_log_score"]}
        for name, reallocate in (("add", False), ("reallocate", True)):
            weights, means, variances, null = oracle_branches(priors, psi, ids[train], labels[train], noise[train],
                candidate, support, scale_mean, reallocate, nodes)
            fold[name + "_joint_log_score"] = float(logsumexp(weights +
                joint_label_log_scores(means, variances, labels[~train], noise[~train])))
            fold[name + "_null_posterior"] = null
        folds.append(fold)
    return {**{name + "_log_score_per_batch": sum(f[name + "_joint_log_score"] for f in folds) / len(ids)
               for name in ("old", "add", "reallocate")}, "folds": folds}
