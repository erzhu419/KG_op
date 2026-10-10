"""Source-learned latent task classes with distinct function covariances."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


def projected_source_blocks(batches, observation_variances, basis):
    """Remove each task intercept and retain its projected measurement noise."""
    blocks, signals = [], []
    for batch, noise in zip(batches, observation_variances):
        y = np.asarray(batch.signals).reshape(-1)
        Z = basis.transform(batch.psi)
        noise = np.asarray(noise)
        n = len(y)
        direction = np.ones(n) / np.sqrt(n)
        direction[0] -= 1.
        H = np.eye(n) - 2 * np.outer(direction, direction) / (direction @ direction)
        Q = H[:, 1:]
        blocks.append((Q.T @ Z, Q.T @ y, Q.T @ (noise[:, None] * Q)))
        signals.append(max(float(np.var(y) - np.mean(noise)), .02))
    return blocks, float(np.mean(signals))


def covariance_mixture_objective(parameters, blocks, *, return_responsibilities=False):
    """Two-class task marginal likelihood and its analytic gradient.

    A task has one latent class for its whole function, rather than a separate
    class for each noisy observation. Parameter order is two variance vectors
    followed by the logit of the first class probability.
    """
    p = blocks[0][0].shape[1]
    tau = np.asarray(parameters[:-1]).reshape(2, p)
    rho = parameters[-1]
    log_pi = np.array([log_expit(rho), log_expit(-rho)])
    value, gradient, responsibilities = 0., np.zeros(2 * p + 1), []
    size = sum(len(y) for _, y, _ in blocks)
    for Z, y, noise in blocks:
        nll, derivatives = [], []
        for variance in tau:
            factor = cho_factor(noise + (Z * variance) @ Z.T, lower=True)
            alpha = cho_solve(factor, y)
            solved = cho_solve(factor, Z)
            nll.append(np.log(np.diag(factor[0])).sum() + .5 * y @ alpha + .5 * len(y) * np.log(2 * np.pi))
            derivatives.append(.5 * ((Z * solved).sum(axis=0) - (Z.T @ alpha) ** 2))
        log_joint = log_pi - nll
        normalizer = logsumexp(log_joint)
        posterior = np.exp(log_joint - normalizer)
        value -= normalizer
        gradient[:-1] += (posterior[:, None] * np.array(derivatives)).reshape(-1)
        gradient[-1] += expit(rho) - posterior[0]
        responsibilities.append(posterior)
    if return_responsibilities:
        return value / size, gradient / size, np.array(responsibilities)
    return value / size, gradient / size


def fit_source_risk_covariance_mixture(batches, observation_variances, public_candidates, initial_tau, *, max_iterations=120):
    """Fit the frozen two-class covariance model once using source data only."""
    basis = FixedFunctionBasis(public_candidates)
    blocks, signal = projected_source_blocks(batches, observation_variances, basis)
    high_frequency = np.array([name.startswith("sin(4*pi*") for name in basis.names])
    multipliers = np.where(high_frequency, .5, 1.5)
    initial_tau = np.asarray(initial_tau)
    initial = np.r_[initial_tau * multipliers, initial_tau * (2 - multipliers), 0.]
    pooled = np.r_[initial_tau, initial_tau, 0.]
    baseline = covariance_mixture_objective(pooled, blocks)[0]
    result = minimize(covariance_mixture_objective, initial, args=(blocks,), jac=True,
                      method="L-BFGS-B", bounds=[(0., None)] * (2 * basis.feature_dim) + [(None, None)],
                      options={"maxiter": max_iterations, "ftol": 1e-9, "gtol": 1e-5})
    value, gradient, responsibilities = covariance_mixture_objective(result.x, blocks, return_responsibilities=True)
    weights = np.array([expit(result.x[-1]), expit(-result.x[-1])])
    variances = result.x[:-1].reshape(2, basis.feature_dim)
    mean = np.r_[np.mean([np.mean(b.signals) for b in batches]), np.zeros(basis.feature_dim)]
    tail = .01 * signal / basis.feature_dim
    components, geometry = [], []
    for k, tau in enumerate(variances):
        covariance = np.diag(np.r_[.25, tau + tail])
        info = {**basis.diagnostics(), "mode": "source_risk_covariance_class", "class_index": k,
                "source_function_variance": tau.tolist(), "tail_variance_budget": .01 * signal,
                "source_label_count": sum(len(b.psi) for b in batches)}
        components.append(SourceFunctionPrior(basis, mean.copy(), covariance, .02, info))
        v = tau + tail
        strongest = np.argsort(-tau)[:4]
        geometry.append({"class_index": k, "source_class_probability": float(weights[k]),
                         "effective_function_dimension": float(v.sum() ** 2 / (v @ v)),
                         "high_frequency_variance_fraction": float(v[high_frequency].sum() / v.sum()),
                         "strongest_function_names": [basis.names[j] for j in strongest]})
    projected_gradient = gradient.copy()
    at_lower_bound = (result.x[:-1] == 0.) & (gradient[:-1] > 0.)
    projected_gradient[:-1][at_lower_bound] = 0.
    diagnostics = {"optimizer_converged": bool(result.success), "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit), "optimizer_iteration_limit": max_iterations,
        "projected_gradient_inf": float(np.max(np.abs(projected_gradient))),
        "source_negative_log_likelihood_per_projected_label": float(value),
        "pooled_initial_negative_log_likelihood_per_projected_label": float(baseline),
        "source_training_likelihood_improvement": float(baseline - value),
        "source_task_count": len(batches), "source_label_count": sum(len(b.psi) for b in batches),
        "task_intercepts": "projected_out", "class_weights": weights.tolist(),
        "task_class_responsibilities": responsibilities.tolist(), "geometry": geometry,
        "likelihood": "task-level Gaussian mixture approximation to bias-corrected log-chi-square labels",
        "nonintercept_prior_mean": "zero", "initialization": "frozen public frequency split of source pooled variance"}
    return components, weights, diagnostics
