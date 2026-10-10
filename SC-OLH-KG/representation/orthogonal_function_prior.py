"""Source-learned sparse group scales in an orthogonal low-order spectral basis.

Orthogonality is in the product of public empirical coordinate marginals.
It does not assert independence under the actual state-trajectory distribution.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from representation.source_function_prior import PublicCoordinateSpectralBasis, SourceFunctionPrior


class OrthogonalLowOrderBasis:
    """Main effects and centered linear pairs; no cross-group whitening.

Each main-effect block diagonalizes its standardized-coordinate derivative
energy after orthonormalization. The dictionary's span is preserved when the
public marginal supports have full rank. Repeated/constant coordinates reduce
that rank, rather than creating artificial functions with enormous scales.
"""

    def __init__(self, reference_psi):
        reference = np.asarray(reference_psi, dtype=float)
        self.dim = reference.shape[1]
        self.dictionary = PublicCoordinateSpectralBasis(
            max_library_size=4 * self.dim + self.dim * (self.dim - 1) // 2)
        raw, raw_names = self._library(reference)
        lookup = {name: i for i, name in enumerate(raw_names)}
        self.axes, self.names, self.groups, frequencies = [], [], [], []
        product_error, centering_error = 0.0, 0.0
        for j in range(self.dim):
            indices, derivative, order = self._axis_spec(reference, j, lookup)
            values = raw[:, indices]
            center = np.where(np.ptp(values, axis=0) == 0, values[0], values.mean(axis=0))
            centered = values - center
            scale = float(centered[:, 0].std())
            coefficients, columns = [], []
            cutoff = 1e-7 * np.sqrt(np.mean(centered ** 2, axis=0).max())
            for k in order:
                coefficient = np.eye(len(indices))[:, k]
                column = centered @ coefficient
                # Reorthogonalize: inventory marginal functions can be nearly
                # dependent on the small public, discrete coordinate support.
                for _ in range(2):
                    for previous, previous_column in zip(coefficients, columns):
                        projection = float(np.mean(previous_column * column))
                        coefficient = coefficient - projection * previous
                        column = centered @ coefficient
                energy = float(np.sqrt(np.mean(column ** 2)))
                if energy <= cutoff:
                    continue
                coefficients.append(coefficient / energy)
                columns.append(column / energy)
            transform = np.column_stack(coefficients) if coefficients else np.empty((len(indices), 0))
            gradients = derivative @ transform
            roughness = gradients.T @ gradients / len(reference)
            eigenvalues, rotation = np.linalg.eigh(roughness)
            transform = transform @ rotation
            # Fix arbitrary eigenvector signs for stable coefficient names.
            for k in range(transform.shape[1]):
                if transform[np.argmax(np.abs(transform[:, k])), k] < 0:
                    transform[:, k] *= -1
            features = centered @ transform
            if len(eigenvalues):
                product_error = max(product_error, float(np.max(np.abs(
                    features.T @ features / len(reference) - np.eye(len(eigenvalues))))))
                centering_error = max(centering_error, float(np.max(np.abs(features.mean(axis=0)))))
            self.axes.append((indices, center, transform, scale))
            self.names.extend(f"main:{j}/mode:{k}" for k in range(len(eigenvalues)))
            self.groups.extend([f"main:{j}"] * len(eigenvalues))
            frequencies.extend(np.maximum(eigenvalues, 0.0).tolist())
        self.pairs = [(i, j) for i in range(self.dim) for j in range(i + 1, self.dim)
                      if self.axes[i][3] > 0 and self.axes[j][3] > 0]
        self.names.extend(f"pair:{i},{j}/linear" for i, j in self.pairs)
        self.groups.extend(f"pair:{i},{j}" for i, j in self.pairs)
        frequencies.extend(self._pair_roughness(reference))
        self.feature_dim = len(self.names)
        if not self.feature_dim:
            raise ValueError("public coordinates have no varying function component")
        self.roughness = np.asarray(frequencies)
        self.group_names = list(dict.fromkeys(self.groups))
        self.group_index = np.array([self.group_names.index(group) for group in self.groups])
        self.product_gram_error = product_error
        self.product_centering_error = centering_error
        joint = self.transform(reference)
        joint_gram = joint.T @ joint / len(reference)
        self.joint_gram_error = float(np.max(np.abs(joint_gram - np.eye(self.feature_dim))))

    def transform(self, psi):
        raw, _ = self._library(psi)
        main, linear = [], []
        for indices, center, transform, scale in self.axes:
            centered = raw[:, indices] - center
            main.append(centered @ transform)
            linear.append(centered[:, 0] / scale if scale > 0 else np.zeros(len(raw)))
        pairs = [linear[i] * linear[j] for i, j in self.pairs]
        return np.column_stack(main + pairs)

    def _library(self, psi):
        return self.dictionary._library(psi)

    def _axis_spec(self, reference, j, lookup):
        indices = [lookup[name] for name in [f"psi{j}", f"psi{j}^2",
            f"sin(pi*psi{j})", f"cos(pi*psi{j})", f"sin(4*pi*psi{j})"]]
        z = np.tanh(0.5 * reference[:, j])
        derivative = np.column_stack([np.ones(len(z)), 2 * z,
            np.pi * np.cos(np.pi * z), -np.pi * np.sin(np.pi * z),
            4 * np.pi * np.cos(4 * np.pi * z)]) * z.std()
        return indices, derivative, list(range(5))

    def _pair_roughness(self, reference):
        return [2.0] * len(self.pairs)

    def diagnostics(self):
        return {"reference_measure": "product_of_public_empirical_coordinate_marginals",
                "orthogonalization": "within_main_effect_derivative_eigenbasis",
                "interaction_order": 2, "active_dim": self.feature_dim,
                "selected_names": self.names, "function_groups": self.groups,
                "roughness_eigenvalues": self.roughness.tolist(),
                "product_gram_max_error": self.product_gram_error,
                "product_centering_max_error": self.product_centering_error,
                "joint_reference_gram_max_error": self.joint_gram_error}


def spectral_weights(basis, exponent):
    """Unit total variance per group; exponent zero admits a flat spectrum."""
    logs = np.log1p(basis.roughness)
    weights = np.exp(-float(exponent) * logs)
    totals = np.bincount(basis.group_index, weights=weights)
    weights /= totals[basis.group_index]
    average = np.bincount(basis.group_index, weights=weights * logs)
    derivative = weights * (average[basis.group_index] - logs)
    return weights, derivative


def fit_orthogonal_function_prior(batches, observation_variances, basis, *, max_iterations=120):
    """Source-only marginal likelihood for group ARD, spectrum and discrepancy.

Every source has its own zero-mean function coefficients and free intercept.
The observed label variances remain in the likelihood. The discrepancy is
latent model error, never additional environmental variance in the eta head.
"""
    blocks, records, signals = [], [], []
    for batch, noise in zip(batches, observation_variances):
        y = np.asarray(batch.signals, dtype=float).reshape(-1)
        Z = basis.transform(batch.psi)
        noise = np.asarray(noise, dtype=float)
        n = len(y)
        direction = np.ones(n) / np.sqrt(n)
        direction[0] -= 1.0
        Q = (np.eye(n) - 2 * np.outer(direction, direction) / (direction @ direction))[:, 1:]
        blocks.append((Q.T @ Z, Q.T @ y, Q.T @ (noise[:, None] * Q)))
        records.append((Z, y, noise))
        signals.append(max(float(np.var(y) - np.mean(noise)), 0.02))
    signal = float(np.mean(signals))
    group_count, size = len(basis.group_names), sum(len(y) for _, y, _ in blocks)
    tail = 0.01 * signal / basis.feature_dim

    def likelihood(parameters):
        group_variances = signal * parameters[:group_count]
        weights, derivative = spectral_weights(basis, parameters[-2])
        tau = group_variances[basis.group_index] * weights + tail
        delta = signal * parameters[-1]
        value, feature_gradient, delta_gradient = 0.0, np.zeros(basis.feature_dim), 0.0
        for Z, y, noise in blocks:
            K = noise + delta * np.eye(len(y)) + (Z * tau) @ Z.T
            factor = cho_factor(K, lower=True)
            alpha = cho_solve(factor, y)
            solved = cho_solve(factor, Z)
            value += np.log(np.diag(factor[0])).sum() + 0.5 * y @ alpha
            feature_gradient += 0.5 * ((Z * solved).sum(axis=0) - (Z.T @ alpha) ** 2)
            delta_gradient += 0.5 * (np.trace(cho_solve(factor, np.eye(len(y)))) - alpha @ alpha)
        gradient = np.r_[signal * np.bincount(basis.group_index,
            weights=feature_gradient * weights, minlength=group_count),
            feature_gradient @ (group_variances[basis.group_index] * derivative),
            signal * delta_gradient]
        return value / size, gradient / size

    initial = np.r_[np.full(group_count, 1.0 / group_count), 1.0, 0.02 / signal]
    result = minimize(likelihood, initial, jac=True, method="L-BFGS-B",
        bounds=[(0.0, None)] * group_count + [(0.0, 4.0), (0.02 / signal, None)],
        options={"maxiter": max_iterations, "ftol": 1e-9, "gtol": 1e-5})
    group_variances = signal * result.x[:group_count]
    exponent, delta = float(result.x[-2]), float(signal * result.x[-1])
    weights, _ = spectral_weights(basis, exponent)
    tau = group_variances[basis.group_index] * weights + tail
    intercepts = []
    for Z, y, noise in records:
        factor = cho_factor(np.diag(noise + delta) + (Z * tau) @ Z.T, lower=True)
        precision_one = cho_solve(factor, np.ones(len(y)))
        intercepts.append(float(precision_one @ y / precision_one.sum()))
    mean = np.zeros(basis.feature_dim + 1)
    mean[0] = np.mean(intercepts)
    intercept_variance = float(np.var(intercepts, ddof=1)) if len(intercepts) > 1 else 0.0
    covariance = np.diag(np.r_[max(intercept_variance, 0.25), tau])
    info = {**basis.diagnostics(), "mode": "source_orthogonal_group_spectral",
        "source_label_count": sum(len(y) for _, y, _ in records),
        "source_task_count": len(records), "source_groups": basis.group_names,
        "source_group_variances": group_variances.tolist(),
        "source_zero_group_count": int(np.count_nonzero(group_variances == 0)),
        "spectral_exponent": exponent, "spectral_exponent_bounds": [0.0, 4.0],
        "tail_variance_budget": 0.01 * signal, "source_signal_scale": signal,
        "discrepancy_variance": delta, "coefficient_mean": mean.tolist(),
        "optimizer_converged": bool(result.success), "optimizer_iterations": int(result.nit),
        "optimizer_max_iterations": int(max_iterations),
        "optimizer_message": str(result.message),
        "source_negative_log_likelihood_per_label": float(result.fun),
        "likelihood": "Gaussian batch labels; log-chi-square approximation for eta"}
    return SourceFunctionPrior(basis, mean, covariance, delta, info)


def same_source_orthogonal_full_prior(prior):
    """Attribution control: same intercept/discrepancy, pooled function scale."""
    covariance = prior.covariance.copy()
    covariance[1:, 1:] = np.eye(prior.basis.feature_dim) * (
        prior.diagnostics["source_signal_scale"] / prior.basis.feature_dim)
    return SourceFunctionPrior(prior.basis, prior.mean.copy(), covariance,
        prior.discrepancy_variance,
        {**prior.basis.diagnostics(), "mode": "same_source_orthogonal_full",
         "source_label_count": prior.diagnostics["source_label_count"],
         "shared_intercept_and_discrepancy": True})
