"""Noise-aware, multi-source covariance learning for the log-variance head."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


def fit_source_risk_ard(batches, observation_variances, public_candidates):
    """Fit nonnegative function variances by source-only marginal likelihood.

    Each source has its own signed coefficients. A diffuse task intercept is
    projected out before pooling covariance evidence across tasks. The known
    log-sample-variance label noise is retained in every likelihood block.
    """
    basis = FixedFunctionBasis(public_candidates)
    blocks, total_signal = [], []
    for batch, noise in zip(batches, observation_variances):
        y = np.asarray(batch.signals).reshape(-1)
        Z = basis.transform(batch.psi)
        n = len(y)
        direction = np.ones(n) / np.sqrt(n)
        direction[0] -= 1.0
        H = np.eye(n) - 2 * np.outer(direction, direction) / (direction @ direction)
        Q = H[:, 1:]
        blocks.append((Q.T @ Z, Q.T @ y, Q.T @ (np.asarray(noise)[:, None] * Q)))
        total_signal.append(max(float(np.var(y) - np.mean(noise)), 0.02))
    signal = float(np.mean(total_signal))
    size = sum(len(y) for _, y, _ in blocks)

    def likelihood(tau):
        value, gradient = 0.0, np.zeros(basis.feature_dim)
        for Z, y, noise in blocks:
            K = noise + (Z * tau) @ Z.T
            factor = cho_factor(K, lower=True)
            alpha = cho_solve(factor, y)
            solved = cho_solve(factor, Z)
            value += np.log(np.diag(factor[0])).sum() + 0.5 * y @ alpha
            gradient += 0.5 * ((Z * solved).sum(axis=0) - (Z.T @ alpha) ** 2)
        return value / size, gradient / size

    result = minimize(likelihood, np.full(basis.feature_dim, signal / basis.feature_dim),
                      jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * basis.feature_dim,
                      options={"maxiter": 120, "ftol": 1e-9, "gtol": 1e-5})
    tau = np.asarray(result.x)
    # Keep a small total tail so the source fit is not a hard support veto.
    covariance = np.diag([0.25] + (tau + 0.01 * signal / len(tau)).tolist())
    mean = np.zeros(basis.feature_dim + 1)
    mean[0] = np.mean([np.mean(b.signals) for b in batches])
    strongest = np.argsort(-tau)[:4].tolist()
    info = {**basis.diagnostics(), "mode": "source_risk_ard",
            "source_selected_indices": strongest,
            "source_selected_names": [basis.names[j] for j in strongest],
            "coefficient_mean": mean.tolist(), "source_function_variance": tau.tolist(),
            "source_label_count": sum(len(b.psi) for b in batches),
            "optimizer_converged": bool(result.success), "optimizer_iterations": int(result.nit),
            "source_negative_log_likelihood_per_label": float(result.fun),
            "likelihood": "Gaussian approximation to bias-corrected log-chi-square labels",
            "task_intercepts": "projected_out", "tail_variance_budget": 0.01 * signal}
    return SourceFunctionPrior(basis, mean, covariance, 0.02, info)
