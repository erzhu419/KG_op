"""Source-anchored eta covariance and grouped predictive validation."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from representation.source_function_prior import SourceFunctionPrior


def adjusted_priors(priors, delta):
    result = []
    for p in priors:
        extra = np.diag(np.r_[0., np.full(p.basis.feature_dim, delta / p.basis.feature_dim)])
        result.append(SourceFunctionPrior(p.basis, p.mean.copy(), p.covariance + extra,
                                         p.discrepancy_variance, p.diagnostics))
    return result


def collapse(ids, labels, noise):
    """Repeated eta labels share one candidate discrepancy, not independent ones."""
    candidates = np.unique(ids)
    means, variances = [], []
    for candidate in candidates:
        same = ids == candidate
        precision = 1. / noise[same]
        means.append(float(precision @ labels[same] / precision.sum()))
        variances.append(float(1. / precision.sum()))
    return candidates, np.asarray(means), np.asarray(variances)


def training_matrices(prior, psi, ids, labels, noise):
    unique, means, variances = collapse(ids, labels, noise)
    features = prior.basis.transform(psi[unique])
    design = np.column_stack([np.ones(len(unique)), features])
    covariance = design @ prior.covariance @ design.T
    covariance += np.diag(variances + prior.discrepancy_variance)
    residual_kernel = features @ features.T / prior.basis.feature_dim
    return unique, means - design @ prior.mean, design, covariance, residual_kernel


def fit_delta(priors, psi, ids, labels, noise):
    """Numerical marginal ML with zero included and a data-derived search bound."""
    spectra, upper = [], 0.
    for prior in priors:
        _, residual, _, base, extra = training_matrices(prior, psi, ids, labels, noise)
        lower = np.linalg.cholesky(base)
        whitened = solve_triangular(lower, extra, lower=True)
        whitened = solve_triangular(lower, whitened.T, lower=True).T
        values, vectors = eigh((whitened + whitened.T) / 2)
        values = np.maximum(values, 0.)
        projected = vectors.T @ solve_triangular(lower, residual, lower=True)
        constant = -np.log(np.diag(lower)).sum() - len(residual) * np.log(2 * np.pi) / 2
        spectra.append((values, projected ** 2, constant))
        positive = values > 0
        # Beyond this bound every component's log-density derivative is <= 0.
        upper = max(upper, float(np.max(np.maximum(projected[positive] ** 2 - 1, 0.) / values[positive])))

    def evidence(delta):
        likelihoods = []
        for values, squared, constant in spectra:
            scale = 1 + delta * values
            likelihoods.append(constant - .5 * np.sum(np.log(scale) + squared / scale))
        return float(logsumexp(likelihoods) - np.log(len(priors)))

    if upper == 0.:
        return 0., evidence(0.)
    # Search in log(1+delta); refine each grid maximum rather than assuming
    # that a two-component marginal likelihood is unimodal.
    grid = np.linspace(0., np.log1p(upper), 65)
    scores = np.array([evidence(np.expm1(t)) for t in grid])
    candidates = [(0., scores[0]), (upper, scores[-1])]
    for i in range(len(grid)):
        left, right = max(i - 1, 0), min(i + 1, len(grid) - 1)
        if scores[i] >= max(scores[left], scores[right]):
            optimum = minimize_scalar(lambda t: -evidence(np.expm1(t)),
                bounds=(grid[left], grid[right]), method="bounded", options={"xatol": 1e-8})
            candidates.append((float(np.expm1(optimum.x)), float(-optimum.fun)))
    best = int(np.argmax(scores))
    candidates.append((float(np.expm1(grid[best])), float(scores[best])))
    return max(candidates, key=lambda item: item[1])


def posterior_components(priors, psi, ids, labels, noise, delta, candidate):
    components, log_weights = [], []
    for prior in adjusted_priors(priors, delta):
        unique, residual, design, covariance, _ = training_matrices(prior, psi, ids, labels, noise)
        factor = cho_factor(covariance, lower=True)
        alpha = cho_solve(factor, residual)
        feature = np.r_[1., prior.basis.transform(psi[candidate:candidate + 1])[0]]
        cross = design @ prior.covariance @ feature
        cross = cross + prior.discrepancy_variance * (unique == candidate)
        mean = float(feature @ prior.mean + cross @ alpha)
        variance = float(feature @ prior.covariance @ feature + prior.discrepancy_variance -
                         cross @ cho_solve(factor, cross))
        components.append((mean, max(variance, 0.)))
        log_weights.append(float(-.5 * residual @ alpha - np.log(np.diag(factor[0])).sum() -
                                 len(residual) * np.log(2 * np.pi) / 2))
    log_weights = np.array(log_weights) - logsumexp(log_weights)
    return log_weights, components


def predictive_group_score(priors, psi, ids, labels, noise, delta, candidate, held_labels, held_noise):
    log_weights, components = posterior_components(priors, psi, ids, labels, noise, delta, candidate)
    scores = []
    for weight, (mean, variance) in zip(log_weights, components):
        covariance = np.full((len(held_labels), len(held_labels)), variance) + np.diag(held_noise)
        factor = cho_factor(covariance, lower=True)
        residual = held_labels - mean
        scores.append(float(weight - .5 * residual @ cho_solve(factor, residual) -
            np.log(np.diag(factor[0])).sum() - len(residual) * np.log(2 * np.pi) / 2))
    return float(logsumexp(scores))


def grouped_validation(priors, psi, ids, labels, noise):
    folds = []
    for candidate in np.unique(ids):
        train = ids != candidate
        held = ~train
        delta, _ = fit_delta(priors, psi, ids[train], labels[train], noise[train])
        scores = [predictive_group_score(priors, psi, ids[train], labels[train], noise[train],
            d, int(candidate), labels[held], noise[held]) for d in (0., delta)]
        folds.append({"held_candidate": int(candidate), "held_batches": int(held.sum()),
                      "training_candidates": np.unique(ids[train]).tolist(), "delta": delta,
                      "old_joint_log_score": scores[0], "residual_joint_log_score": scores[1]})
    return {"old_log_score_per_batch": sum(f["old_joint_log_score"] for f in folds) / len(ids),
            "residual_log_score_per_batch": sum(f["residual_joint_log_score"] for f in folds) / len(ids),
            "folds": folds}
