"""Average source-anchored risk corrections of rank zero, one and two."""
from functools import lru_cache

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, roots_laguerre

from performance.target_risk_direction_mixture import joint_label_log_scores
from performance.target_risk_residual import training_matrices


@lru_cache(maxsize=4)
def quadrature(nodes):
    return roots_laguerre(nodes)


@lru_cache(maxsize=2)
def function_pairs(size):
    return np.triu_indices(size, 1)


def joint_predictive_branches(priors, psi, ids, labels, noise, candidate, scale_mean, nodes=48):
    """Condition every branch on training labels, keeping its full mixture."""
    locations, quadrature_weights = quadrature(nodes)
    deltas = scale_mean * locations
    logs, means, variances = [], [], []
    component_log_prior = -np.log(len(priors))
    for prior in priors:
        unique, residual, design, covariance, _ = training_matrices(prior, psi, ids, labels, noise)
        factor = cho_factor(covariance, lower=True)
        alpha = cho_solve(factor, residual)
        features = design[:, 1:]
        solved = cho_solve(factor, features)
        gram = features.T @ solved
        information = np.diag(gram)
        projected = features.T @ alpha
        feature = np.r_[1., prior.basis.transform(psi[candidate:candidate + 1])[0]]
        cross = design @ prior.covariance @ feature + prior.discrepancy_variance * (unique == candidate)
        mean0 = float(feature @ prior.mean + cross @ alpha)
        var0 = max(float(feature @ prior.covariance @ feature + prior.discrepancy_variance -
                         cross @ cho_solve(factor, cross)), 0.)
        base = float(-.5 * residual @ alpha - np.log(np.diag(factor[0])).sum() -
                     len(residual) * np.log(2 * np.pi) / 2)
        logs.append(np.array([component_log_prior + np.log(.5) + base]))
        means.append(np.array([mean0]))
        variances.append(np.array([var0]))
        innovation = feature[1:] - cross @ solved
        denominator = 1. + information[:, None] * deltas[None, :]
        ratio = deltas[None, :] / denominator
        evidence = base - .5 * np.log(denominator) + .5 * projected[:, None] ** 2 * ratio
        weight = component_log_prior + np.log(.25) - np.log(len(information))
        logs.append((evidence + weight + np.log(quadrature_weights)[None, :]).reshape(-1))
        means.append((mean0 + innovation[:, None] * projected[:, None] * ratio).reshape(-1))
        variances.append((var0 + innovation[:, None] ** 2 * ratio).reshape(-1))

        j, k = function_pairs(len(information))
        a, b, c = information[j, None], information[k, None], gram[j, k, None]
        t = deltas[None, :] / 2.
        # Expanded determinant preserves its positive unit/linear terms.
        determinant = 1. + t * (a + b) + t ** 2 * np.maximum(a * b - c ** 2, 0.)
        q11, q22, q12 = t * (1. + t * b) / determinant, t * (1. + t * a) / determinant, -t ** 2 * c / determinant
        sj, sk = projected[j, None], projected[k, None]
        dj, dk = innovation[j, None], innovation[k, None]
        evidence = base - .5 * np.log(determinant) + .5 * (sj ** 2 * q11 + 2 * sj * sk * q12 + sk ** 2 * q22)
        weight = component_log_prior + np.log(.25) - np.log(len(j))
        logs.append((evidence + weight + np.log(quadrature_weights)[None, :]).reshape(-1))
        means.append((mean0 + dj * (q11 * sj + q12 * sk) + dk * (q12 * sj + q22 * sk)).reshape(-1))
        variances.append(np.maximum(var0 + dj ** 2 * q11 + 2 * dj * dk * q12 + dk ** 2 * q22, 0.).reshape(-1))
    logs = np.concatenate(logs)
    log_evidence = float(logsumexp(logs))
    weights = logs - log_evidence
    size = priors[0].basis.feature_dim
    single_end = 1 + size * nodes
    stride = single_end + size * (size - 1) // 2 * nodes
    rank_mass = np.zeros(3)
    for start in range(0, len(weights), stride):
        rank_mass += [np.exp(weights[start]), np.exp(logsumexp(weights[start + 1:start + single_end])),
                      np.exp(logsumexp(weights[start + single_end:start + stride]))]
    return weights, np.concatenate(means), np.concatenate(variances), log_evidence, rank_mass


def grouped_joint_validation(priors, psi, ids, labels, noise, old_folds, scale_mean, nodes=48):
    folds = []
    for old in old_folds:
        candidate = old["held_candidate"]
        train = ids != candidate
        if np.unique(ids[train]).tolist() != old["training_candidates"]:
            raise ValueError("A7 changed the frozen candidate-group holdout")
        weights, means, variances, evidence, rank_mass = joint_predictive_branches(
            priors, psi, ids[train], labels[train], noise[train], candidate, scale_mean, nodes)
        score = float(logsumexp(weights + joint_label_log_scores(means, variances, labels[~train], noise[~train])))
        folds.append({"held_candidate": int(candidate), "held_batches": int((~train).sum()),
            "training_candidates": old["training_candidates"], "rank_posterior": rank_mass.tolist(),
            "training_log_evidence": evidence, "old_joint_log_score": old["old_joint_log_score"],
            "joint_model_log_score": score})
    return {"old_log_score_per_batch": sum(f["old_joint_log_score"] for f in folds) / len(ids),
            "joint_model_log_score_per_batch": sum(f["joint_model_log_score"] for f in folds) / len(ids),
            "folds": folds}
