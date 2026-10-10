"""One paid-label-selected risk direction added to the source covariance."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from performance.target_risk_residual import predictive_group_score, training_matrices
from representation.source_function_prior import SourceFunctionPrior


def direction_priors(priors, direction, delta):
    result = []
    for prior in priors:
        covariance = prior.covariance.copy()
        if direction is not None:
            covariance[direction + 1, direction + 1] += delta
        result.append(SourceFunctionPrior(prior.basis, prior.mean.copy(), covariance,
                                         prior.discrepancy_variance, prior.diagnostics))
    return result


def rank_one_statistics(priors, psi, ids, labels, noise):
    """Factor each original component once for all public directions."""
    bases, information, projected = [], [], []
    for prior in priors:
        _, residual, design, covariance, _ = training_matrices(prior, psi, ids, labels, noise)
        factor = cho_factor(covariance, lower=True)
        features = design[:, 1:]
        alpha = cho_solve(factor, residual)
        bases.append(float(-.5 * residual @ alpha - np.log(np.diag(factor[0])).sum() -
                           len(residual) * np.log(2 * np.pi) / 2))
        information.append(np.sum(features * cho_solve(factor, features), axis=0))
        projected.append((features.T @ alpha) ** 2)
    return np.asarray(bases), np.asarray(information), np.asarray(projected)


def direction_evidence(statistics, direction, delta):
    base, information, squared = statistics
    a, b = information[:, direction], squared[:, direction]
    scale = 1. + delta * a
    return float(logsumexp(base - .5 * np.log(scale) + .5 * delta * b / scale) -
                 np.log(len(base)))


def fit_direction(priors, psi, ids, labels, noise):
    """Numerical marginal ML over one direction and its nonnegative scale."""
    statistics = rank_one_statistics(priors, psi, ids, labels, noise)
    base, information, squared = statistics
    score0 = float(logsumexp(base) - np.log(len(base)))
    best = {"direction": None, "delta": 0., "log_evidence": score0,
            "zero_log_evidence": score0}
    # Each component has its exact rank-one optimum. Their mixture need not
    # share that optimum; the maximum of component densities bounds its peak.
    component_optima = np.zeros_like(information)
    np.divide(np.maximum(squared - information, 0.), information ** 2,
              out=component_optima, where=information > 0.)
    scales = 1. + component_optima * information
    peak = base[:, None] - .5 * np.log(scales) + .5 * component_optima * squared / scales
    bounds = logsumexp(peak, axis=0) - np.log(len(base))
    upper = component_optima.max(axis=0)
    if len(priors) == 1:
        direction = int(np.argmax(bounds))
        if bounds[direction] > score0:
            best.update(direction=direction, delta=float(upper[direction]),
                        log_evidence=float(bounds[direction]))
        return best

    # Batched grid evaluation gives a useful lower bound before refinements.
    grid = np.linspace(0., 1., 65)[None, :] * np.log1p(upper)[:, None]
    deltas = np.expm1(grid)
    scales = 1. + information[:, :, None] * deltas[None, :, :]
    scores = logsumexp(base[:, None, None] - .5 * np.log(scales) +
        .5 * deltas[None, :, :] * squared[:, :, None] / scales, axis=0) - np.log(len(base))
    j, i = np.unravel_index(np.argmax(scores), scores.shape)
    if scores[j, i] > score0:
        best.update(direction=int(j), delta=float(deltas[j, i]), log_evidence=float(scores[j, i]))
    for j in np.argsort(-bounds):
        if bounds[j] <= best["log_evidence"]:
            break  # Even independently optimized components cannot beat best.
        if upper[j] == 0.:
            continue
        for i in range(len(grid[j])):
            left, right = max(i - 1, 0), min(i + 1, len(grid[j]) - 1)
            if scores[j, i] >= max(scores[j, left], scores[j, right]):
                optimum = minimize_scalar(
                    lambda t: -direction_evidence(statistics, j, np.expm1(t)),
                    bounds=(grid[j, left], grid[j, right]), method="bounded", options={"xatol": 1e-8})
                if -optimum.fun > best["log_evidence"]:
                    best.update(direction=int(j), delta=float(np.expm1(optimum.x)),
                                log_evidence=float(-optimum.fun))
    return best


def grouped_direction_validation(priors, psi, ids, labels, noise, old_folds):
    folds = []
    for old in old_folds:
        candidate = old["held_candidate"]
        train = ids != candidate
        if np.unique(ids[train]).tolist() != old["training_candidates"]:
            raise ValueError("A5 changed the frozen candidate-group holdout")
        fitted = fit_direction(priors, psi, ids[train], labels[train], noise[train])
        corrected = direction_priors(priors, fitted["direction"], fitted["delta"])
        score = predictive_group_score(corrected, psi, ids[train], labels[train], noise[train],
                                       0., candidate, labels[~train], noise[~train])
        direction = fitted["direction"]
        folds.append({**fitted, "direction_name": None if direction is None else priors[0].basis.names[direction],
            "held_candidate": int(candidate), "held_batches": int((~train).sum()),
            "training_candidates": old["training_candidates"],
            "old_joint_log_score": old["old_joint_log_score"], "direction_joint_log_score": score})
    return {"old_log_score_per_batch": sum(f["old_joint_log_score"] for f in folds) / len(ids),
            "direction_log_score_per_batch": sum(f["direction_joint_log_score"] for f in folds) / len(ids),
            "folds": folds}
