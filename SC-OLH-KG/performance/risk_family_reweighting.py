"""Target-paid risk evidence reweights source variance toward public families."""
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh, solve_triangular
from scipy.special import logsumexp, roots_legendre

from performance.target_risk_direction_mixture import joint_label_log_scores
from performance.target_risk_residual import training_matrices


@lru_cache(maxsize=4)
def rho_quadrature(nodes):
    points, weights = roots_legendre(nodes)
    return (points + 1.) / 2., weights / 2.


def family_covariance(prior, indices, rho):
    covariance = prior.covariance.copy()
    source = prior.covariance[1:, 1:]
    covariance[1:, 1:] = (1. - rho) * source
    positions = np.asarray(indices) + 1
    covariance[positions, positions] += rho * np.trace(source) / len(indices)
    return covariance


def prepare_covariance_paths(prior, psi, ids, labels, noise, candidate, endpoints):
    """Factor the history once for multiple linear covariance paths and orders."""
    unique, residual, design, covariance, _ = training_matrices(prior, psi, ids, labels, noise)
    lower = np.linalg.cholesky(covariance)
    whitened_y = solve_triangular(lower, residual, lower=True)
    feature = np.r_[1., prior.basis.transform(psi[candidate:candidate + 1])[0]]
    cross = design @ prior.covariance @ feature + prior.discrepancy_variance * (unique == candidate)
    whitened_cross = solve_triangular(lower, cross, lower=True)
    mean_prior = float(feature @ prior.mean)
    var_prior = float(feature @ prior.covariance @ feature + prior.discrepancy_variance)
    constant = -np.log(np.diag(lower)).sum() - len(residual) * np.log(2 * np.pi) / 2.
    evidence0 = constant - .5 * whitened_y @ whitened_y
    paths = []
    features = design[:, 1:]
    for endpoint in endpoints:
        change = endpoint[1:, 1:] - prior.covariance[1:, 1:]
        kernel_change = features @ change @ features.T
        whitened = solve_triangular(lower, kernel_change, lower=True)
        whitened = solve_triangular(lower, whitened.T, lower=True).T
        # Signed eigenvalues retain reductions of existing source directions.
        values, vectors = eigh((whitened + whitened.T) / 2.)
        cross_change = features @ change @ feature[1:]
        paths.append({"values": values, "projected_y": vectors.T @ whitened_y,
            "projected_cross": vectors.T @ whitened_cross,
            "projected_change": vectors.T @ solve_triangular(lower, cross_change, lower=True),
            "variance_change": float(feature[1:] @ change @ feature[1:])})
    return {"source": (float(evidence0), float(mean_prior + whitened_cross @ whitened_y),
                       max(var_prior - whitened_cross @ whitened_cross, 0.)),
            "constant": constant, "mean_prior": mean_prior, "var_prior": var_prior, "paths": paths}


def evaluate_covariance_paths(prepared, nodes):
    rho, _ = rho_quadrature(nodes)
    result = []
    for path in prepared["paths"]:
        denominator = 1. + path["values"][:, None] * rho[None, :]
        y = path["projected_y"][:, None]
        projected = path["projected_cross"][:, None] + path["projected_change"][:, None] * rho[None, :]
        evidence = prepared["constant"] - .5 * (np.log(denominator) + y ** 2 / denominator).sum(axis=0)
        mean = prepared["mean_prior"] + (projected * y / denominator).sum(axis=0)
        variance = prepared["var_prior"] + rho * path["variance_change"] - (projected ** 2 / denominator).sum(axis=0)
        result.append((evidence, mean, np.maximum(variance, 0.)))
    return result


def family_predictive_branches(priors, psi, ids, labels, noise, candidate, families, nodes=32,
                               *, covariance_builder=family_covariance):
    rho, quadrature_weights = rho_quadrature(nodes)
    logs, means, variances = [], [], []
    prior_log_mass = np.log(.5) - np.log(len(priors))
    for prior in priors:
        prepared = prepare_covariance_paths(prior, psi, ids, labels, noise, candidate,
            [covariance_builder(prior, indices, 1.) for indices in families.values()])
        evidence0, mean0, variance0 = prepared["source"]
        logs.append(np.array([prior_log_mass + evidence0]))
        means.append(np.array([mean0]))
        variances.append(np.array([variance0]))
        for evidence, mean, variance in evaluate_covariance_paths(prepared, nodes):
            logs.append(prior_log_mass - np.log(len(families)) + np.log(quadrature_weights) + evidence)
            means.append(mean)
            variances.append(variance)
    logs = np.concatenate(logs)
    evidence = float(logsumexp(logs))
    weights = logs - evidence
    stride = 1 + len(families) * nodes
    null_posterior, family_posterior, mean_rho = 0., np.zeros(len(families)), 0.
    for start in range(0, len(weights), stride):
        null_posterior += float(np.exp(weights[start]))
        for group in range(len(families)):
            first = start + 1 + group * nodes
            mass = np.exp(weights[first:first + nodes])
            family_posterior[group] += mass.sum()
            mean_rho += float(mass @ rho)
    summary = {"null_posterior": null_posterior,
               "family_posterior": dict(zip(families, family_posterior.tolist())),
               "mean_rho_posterior": mean_rho, "training_log_evidence": evidence}
    return weights, np.concatenate(means), np.concatenate(variances), summary


def grouped_family_validation(priors, psi, ids, labels, noise, old_folds, families, nodes=32):
    folds = []
    for old in old_folds:
        candidate = old["held_candidate"]
        train = ids != candidate
        if np.unique(ids[train]).tolist() != old["training_candidates"]:
            raise ValueError("A9 changed the frozen candidate-group holdout")
        weights, means, variances, summary = family_predictive_branches(
            priors, psi, ids[train], labels[train], noise[train], candidate, families, nodes)
        score = float(logsumexp(weights + joint_label_log_scores(means, variances, labels[~train], noise[~train])))
        folds.append({"held_candidate": int(candidate), "held_batches": int((~train).sum()),
            "training_candidates": old["training_candidates"], "old_joint_log_score": old["old_joint_log_score"],
            "family_joint_log_score": score, **summary})
    return {"old_log_score_per_batch": sum(f["old_joint_log_score"] for f in folds) / len(ids),
            "family_log_score_per_batch": sum(f["family_joint_log_score"] for f in folds) / len(ids), "folds": folds}
