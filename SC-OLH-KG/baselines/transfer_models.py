"""Paper-core transfer surrogate models under one observable-data contract."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _as_2d(X):
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("model inputs must be a finite matrix")
    return values


def _stable_cholesky(matrix):
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(
        matrix, dtype=float).T)
    eye = np.eye(len(matrix), dtype=float)
    jitter = 1e-10
    for _ in range(9):
        try:
            return np.linalg.cholesky(matrix + jitter * eye), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    repaired = (eigenvectors * np.maximum(eigenvalues, jitter)) @ eigenvectors.T
    return np.linalg.cholesky(repaired + jitter * eye), jitter


def _kernel(X1, X2, *, lengthscale, outputscale, kind="matern52"):
    X1 = _as_2d(X1)
    X2 = _as_2d(X2)
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("kernel input dimensions differ")
    # Mean squared distance keeps a common lengthscale interpretable when d
    # changes; it is an isotropic kernel on x / sqrt(d).
    delta = X1[:, None, :] - X2[None, :, :]
    distance2 = np.mean(delta * delta, axis=2)
    scale = max(float(lengthscale), 1e-8)
    if kind == "rbf":
        base = np.exp(-0.5 * distance2 / (scale * scale))
    elif kind == "matern52":
        radius = np.sqrt(np.maximum(5.0 * distance2, 0.0)) / scale
        base = (1.0 + radius + radius * radius / 3.0) * np.exp(-radius)
    else:
        raise ValueError(f"unknown transfer kernel {kind!r}")
    return max(float(outputscale), 1e-10) * base


@dataclass(frozen=True)
class ScalarTaskData:
    name: str
    X: np.ndarray
    y: np.ndarray
    noise: np.ndarray


class ExactGPSurrogate:
    """Small deterministic exact GP used by the statistical transfer models."""

    def __init__(
        self,
        *,
        kernel="matern52",
        lengthscale=0.25,
        outputscale=1.0,
        noise_floor=1e-6,
        normalization=None,
    ):
        self.kernel = str(kernel)
        self.lengthscale = float(lengthscale)
        self.outputscale = float(outputscale)
        self.noise_floor = max(float(noise_floor), 1e-12)
        self.fixed_normalization = normalization
        self.X = np.empty((0, 0), dtype=float)
        self.y = np.empty(0, dtype=float)
        self.noise = np.empty(0, dtype=float)
        self.y_mean = 0.0
        self.y_scale = 1.0
        self._chol = None
        self._alpha = None
        self._jitter = 0.0

    @staticmethod
    def select_hyperparameters(tasks, kernel="matern52"):
        tasks = list(tasks)
        if not tasks:
            return {"lengthscale": 0.25, "outputscale": 1.0}
        best = None
        for lengthscale in (0.05, 0.10, 0.20, 0.35, 0.60, 1.0):
            for outputscale in (0.25, 1.0, 4.0):
                loss = 0.0
                valid = True
                for task in tasks:
                    X = _as_2d(task.X)
                    y = np.asarray(task.y, dtype=float).reshape(-1)
                    scale = max(float(np.std(y)), 1e-8)
                    normalized = (y - float(np.mean(y))) / scale
                    noise = np.maximum(
                        np.asarray(task.noise, dtype=float).reshape(-1)
                        / (scale * scale),
                        1e-8,
                    )
                    K = _kernel(
                        X, X,
                        lengthscale=lengthscale,
                        outputscale=outputscale,
                        kind=kernel,
                    ) + np.diag(noise)
                    try:
                        chol, _ = _stable_cholesky(K)
                        alpha = np.linalg.solve(
                            chol.T, np.linalg.solve(chol, normalized))
                        loss += 0.5 * float(normalized @ alpha)
                        loss += float(np.sum(np.log(np.diag(chol))))
                        loss += 0.5 * len(y) * math.log(2.0 * math.pi)
                    except (np.linalg.LinAlgError, FloatingPointError):
                        valid = False
                        break
                key = (float(loss), float(lengthscale), float(outputscale))
                if valid and (best is None or key < best[0]):
                    best = (key, {
                        "lengthscale": float(lengthscale),
                        "outputscale": float(outputscale),
                    })
        if best is None:
            raise np.linalg.LinAlgError("no stable source GP hyperparameters")
        return best[1]

    def fit(self, X, y, noise=None):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y):
            raise ValueError("GP input and target lengths differ")
        if len(y) == 0:
            self.X = X.copy()
            self.y = y.copy()
            self.noise = np.empty(0, dtype=float)
            self._chol = None
            self._alpha = None
            return self
        if noise is None:
            noise = np.full(len(y), self.noise_floor, dtype=float)
        noise = np.maximum(
            np.asarray(noise, dtype=float).reshape(-1), self.noise_floor)
        if len(noise) != len(y):
            raise ValueError("GP noise length differs from targets")
        if self.fixed_normalization is None:
            self.y_mean = float(np.mean(y))
            self.y_scale = max(float(np.std(y)), 1e-8)
        else:
            self.y_mean = float(self.fixed_normalization[0])
            self.y_scale = max(float(self.fixed_normalization[1]), 1e-8)
        normalized = (y - self.y_mean) / self.y_scale
        normalized_noise = noise / (self.y_scale * self.y_scale)
        K = _kernel(
            X, X,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            kind=self.kernel,
        ) + np.diag(normalized_noise)
        self._chol, self._jitter = _stable_cholesky(K)
        self._alpha = np.linalg.solve(
            self._chol.T,
            np.linalg.solve(self._chol, normalized),
        )
        self.X = X.copy()
        self.y = y.copy()
        self.noise = noise.copy()
        return self

    def predict(self, X, *, full_cov=False):
        X = _as_2d(X)
        prior_var = np.full(
            len(X), self.outputscale * self.y_scale * self.y_scale,
            dtype=float,
        )
        if self._chol is None or len(self.X) == 0:
            mean = np.full(len(X), self.y_mean, dtype=float)
            if full_cov:
                covariance = _kernel(
                    X, X,
                    lengthscale=self.lengthscale,
                    outputscale=self.outputscale,
                    kind=self.kernel,
                ) * (self.y_scale * self.y_scale)
                return mean, covariance
            return mean, prior_var
        cross = _kernel(
            self.X, X,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            kind=self.kernel,
        )
        mean = self.y_mean + self.y_scale * (cross.T @ self._alpha)
        solved = np.linalg.solve(self._chol, cross)
        if full_cov:
            covariance = _kernel(
                X, X,
                lengthscale=self.lengthscale,
                outputscale=self.outputscale,
                kind=self.kernel,
            ) - solved.T @ solved
            covariance = 0.5 * (covariance + covariance.T)
            covariance *= self.y_scale * self.y_scale
            diagonal = np.maximum(np.diag(covariance), 1e-12)
            covariance[np.diag_indices_from(covariance)] = diagonal
            return np.asarray(mean, dtype=float), covariance
        variance = np.maximum(
            self.outputscale - np.sum(solved * solved, axis=0), 1e-12)
        return np.asarray(mean, dtype=float), (
            variance * self.y_scale * self.y_scale)

    def loo_moments(self):
        if self._chol is None or len(self.X) < 2:
            return self.y.copy(), np.full(len(self.y), np.inf)
        identity = np.eye(len(self.X), dtype=float)
        inverse = np.linalg.solve(
            self._chol.T, np.linalg.solve(self._chol, identity))
        normalized = (self.y - self.y_mean) / self.y_scale
        diagonal = np.maximum(np.diag(inverse), 1e-12)
        loo_mean = normalized - self._alpha / diagonal
        loo_var = 1.0 / diagonal
        return (
            self.y_mean + self.y_scale * loo_mean,
            np.maximum(loo_var * self.y_scale * self.y_scale, 1e-12),
        )

    def diagnostics(self):
        return {
            "kernel": self.kernel,
            "lengthscale": float(self.lengthscale),
            "outputscale": float(self.outputscale),
            "n_train": int(len(self.y)),
            "jitter": float(self._jitter),
        }


def _source_normalization(tasks):
    values = np.concatenate([
        np.asarray(task.y, dtype=float).reshape(-1) for task in tasks
    ])
    return float(np.mean(values)), max(float(np.std(values)), 1e-8)


class HyperBOSurrogate:
    """Pre-train a shared GP prior, then condition without target refitting."""

    adaptation_kind = "posterior_conditioning"
    implementation_family = "hyperbo_pretrained_gp_prior"
    implementation_fidelity = "paper_core_reimplementation"

    def __init__(self, kernel="rbf"):
        self.kernel = kernel
        self.hyperparameters = None
        self.normalization = (0.0, 1.0)
        self.target_model = None
        self.n_target = 0

    def meta_fit(self, tasks):
        tasks = list(tasks)
        self.hyperparameters = ExactGPSurrogate.select_hyperparameters(
            tasks, kernel=self.kernel)
        self.normalization = _source_normalization(tasks)
        self.target_model = ExactGPSurrogate(
            kernel=self.kernel,
            normalization=self.normalization,
            **self.hyperparameters,
        )
        return self

    def adapt(self, X, y, noise):
        self.target_model.fit(X, y, noise)
        self.n_target = int(len(y))

    def predict(self, X, full_cov=False):
        return self.target_model.predict(X, full_cov=full_cov)

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_reference": "google-research/hyperbo@e720fc1",
            "source_prior_frozen_online": True,
            "n_target": self.n_target,
            "model": self.target_model.diagnostics(),
        }


def _ranking_loss(samples, y):
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    truth = y[:, None] < y[None, :]
    predicted = samples[:, :, None] < samples[:, None, :]
    return np.sum(predicted != truth[None, :, :], axis=(1, 2))


class RGPESurrogate:
    """Ranking-weighted source GPs plus an online target GP."""

    adaptation_kind = "expert_reweighting_and_posterior_conditioning"
    implementation_family = "rgpe"
    implementation_fidelity = "audited_independent_reimplementation"

    def __init__(self, *, seed=0, n_weight_samples=128):
        self.rng = np.random.default_rng(seed)
        self.n_weight_samples = max(16, int(n_weight_samples))
        self.source_models = []
        self.source_names = []
        self.target_model = None
        self.weights = np.empty(0, dtype=float)
        self.n_target = 0

    def meta_fit(self, tasks):
        tasks = list(tasks)
        self.source_models = []
        self.source_names = []
        for task in tasks:
            params = ExactGPSurrogate.select_hyperparameters([task])
            model = ExactGPSurrogate(**params).fit(
                task.X, task.y, task.noise)
            self.source_models.append(model)
            self.source_names.append(task.name)
        pooled_params = ExactGPSurrogate.select_hyperparameters(tasks)
        self.target_model = ExactGPSurrogate(**pooled_params)
        self.weights = np.full(
            len(self.source_models) + 1,
            1.0 / max(len(self.source_models), 1),
            dtype=float,
        )
        self.weights[-1] = 0.0
        return self

    def adapt(self, X, y, noise):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_target = int(len(y))
        if self.n_target:
            self.target_model.fit(X, y, noise)
        n_source = len(self.source_models)
        if self.n_target == 0:
            self.weights = np.r_[
                np.full(n_source, 1.0 / max(n_source, 1)), 0.0]
            return
        if self.n_target == 1:
            likelihood = []
            for model in self.source_models:
                mean, variance = model.predict(X)
                likelihood.append(np.exp(-0.5 * (y[0] - mean[0]) ** 2 /
                                         max(variance[0], 1e-12)) /
                                  np.sqrt(max(variance[0], 1e-12)))
            source_weights = np.asarray(likelihood, dtype=float)
            source_weights /= max(float(np.sum(source_weights)), 1e-300)
            self.weights = np.r_[source_weights, 0.0]
            return
        losses = []
        for model in self.source_models:
            mean, covariance = model.predict(X, full_cov=True)
            chol, _ = _stable_cholesky(covariance)
            samples = mean[None, :] + self.rng.normal(
                size=(self.n_weight_samples, self.n_target)) @ chol.T
            losses.append(_ranking_loss(samples, y))
        loo_mean, loo_var = self.target_model.loo_moments()
        target_samples = loo_mean[None, :] + self.rng.normal(
            size=(self.n_weight_samples, self.n_target)
        ) * np.sqrt(np.maximum(loo_var, 1e-12))[None, :]
        target_loss = _ranking_loss(target_samples, y)
        source_loss = np.vstack(losses)
        threshold = float(np.percentile(target_loss, 95.0))
        source_loss[
            np.mean(source_loss, axis=1) > threshold
        ] = self.n_target * self.n_target
        all_loss = np.vstack([source_loss, target_loss[None, :]])
        best = np.argmin(all_loss, axis=0)
        self.weights = np.bincount(
            best, minlength=n_source + 1).astype(float)
        self.weights /= max(float(np.sum(self.weights)), 1.0)

    def predict(self, X, full_cov=False):
        models = self.source_models + [self.target_model]
        active = [
            (model, float(weight))
            for model, weight in zip(models, self.weights)
            if weight > 0.0
        ]
        if not active:
            raise RuntimeError("RGPE has no active experts")
        predictions = [
            model.predict(X, full_cov=full_cov) for model, _ in active
        ]
        mean = sum(
            weight * prediction[0]
            for prediction, (_, weight) in zip(predictions, active)
        )
        variance = sum(
            weight * weight * prediction[1]
            for prediction, (_, weight) in zip(predictions, active)
        )
        return np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)

    def diagnostics(self):
        names = self.source_names + ["target"]
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_reference": (
                "boschresearch/transfergpbo@"
                "4bdf90498c0048d7c80327609961d795d82b6f43"
            ),
            "n_target": int(self.n_target),
            "weights": {
                name: float(weight) for name, weight in zip(names, self.weights)
            },
            "n_weight_samples": int(self.n_weight_samples),
        }


class StackedHierarchicalGPSurrogate:
    """Source GP mixture used as the prior mean for a target residual GP."""

    adaptation_kind = "source_target_discrepancy_posterior"
    implementation_family = "stacked_hierarchical_gp"
    implementation_fidelity = "audited_independent_reimplementation"

    def __init__(self):
        self.source_models = []
        self.source_names = []
        self.source_weights = np.empty(0, dtype=float)
        self.residual_model = None
        self.n_target = 0

    def meta_fit(self, tasks):
        tasks = list(tasks)
        self.source_models = []
        self.source_names = []
        for task in tasks:
            params = ExactGPSurrogate.select_hyperparameters([task])
            self.source_models.append(ExactGPSurrogate(**params).fit(
                task.X, task.y, task.noise))
            self.source_names.append(task.name)
        self.source_weights = np.full(
            len(tasks), 1.0 / max(len(tasks), 1), dtype=float)
        params = ExactGPSurrogate.select_hyperparameters(tasks)
        self.residual_model = ExactGPSurrogate(**params)
        return self

    def _source_predict(self, X, full_cov=False):
        predictions = [
            model.predict(X, full_cov=full_cov)
            for model in self.source_models
        ]
        mean = sum(
            weight * prediction[0]
            for weight, prediction in zip(self.source_weights, predictions)
        )
        variance = sum(
            weight * weight * prediction[1]
            for weight, prediction in zip(self.source_weights, predictions)
        )
        return np.asarray(mean), np.asarray(variance)

    def adapt(self, X, y, noise):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        noise = np.asarray(noise, dtype=float).reshape(-1)
        self.n_target = int(len(y))
        if not self.n_target:
            return
        scores = []
        for model in self.source_models:
            mean, variance = model.predict(X)
            total = np.maximum(variance + noise, 1e-12)
            scores.append(-0.5 * float(np.sum(
                np.log(total) + (y - mean) ** 2 / total)))
        shifted = np.asarray(scores) - max(scores)
        self.source_weights = np.exp(shifted)
        self.source_weights /= max(float(np.sum(self.source_weights)), 1e-300)
        source_mean, _ = self._source_predict(X)
        self.residual_model.fit(X, y - source_mean, noise)

    def predict(self, X, full_cov=False):
        source_mean, source_var = self._source_predict(X, full_cov=full_cov)
        if self.n_target == 0:
            return source_mean, source_var
        residual_mean, residual_var = self.residual_model.predict(
            X, full_cov=full_cov)
        return source_mean + residual_mean, np.maximum(
            source_var + residual_var, 1e-12)

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_reference": (
                "boschresearch/transfergpbo@"
                "4bdf90498c0048d7c80327609961d795d82b6f43"
            ),
            "n_target": int(self.n_target),
            "source_weights": {
                name: float(weight)
                for name, weight in zip(self.source_names, self.source_weights)
            },
            "uncertainty_inheritance": True,
        }


def _psd_projection(matrix, floor=1e-8):
    matrix = 0.5 * (np.asarray(matrix) + np.asarray(matrix).T)
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, floor)) @ vectors.T


class MultiTaskGPSurrogate:
    """Intrinsic-coregionalization GP over source and held-out target tasks."""

    adaptation_kind = "joint_multitask_posterior"
    implementation_family = "multitask_gp_icm"
    implementation_fidelity = "audited_independent_reimplementation"

    def __init__(self):
        self.tasks = []
        self.source_models = []
        self.hyperparameters = None
        self.normalization = (0.0, 1.0)
        self.task_covariance = None
        self._train_X = None
        self._train_task = None
        self._chol = None
        self._alpha = None
        self.n_target = 0

    def meta_fit(self, tasks):
        self.tasks = list(tasks)
        self.hyperparameters = ExactGPSurrogate.select_hyperparameters(
            self.tasks)
        self.normalization = _source_normalization(self.tasks)
        self.source_models = [
            ExactGPSurrogate(**ExactGPSurrogate.select_hyperparameters([task])).fit(
                task.X, task.y, task.noise)
            for task in self.tasks
        ]
        n_source = len(self.tasks)
        source_covariance = np.eye(n_source, dtype=float)
        if n_source > 1:
            lengths = {len(task.y) for task in self.tasks}
            if len(lengths) == 1:
                corr = np.corrcoef(np.vstack([task.y for task in self.tasks]))
                source_covariance = np.nan_to_num(corr, nan=0.0)
                source_covariance = _psd_projection(source_covariance)
                diagonal = np.sqrt(np.maximum(
                    np.diag(source_covariance), 1e-12))
                source_covariance /= diagonal[:, None] * diagonal[None, :]
        covariance = np.eye(n_source + 1, dtype=float)
        covariance[:n_source, :n_source] = source_covariance
        target_cross = np.mean(source_covariance, axis=0)
        covariance[-1, :n_source] = 0.5 * target_cross
        covariance[:n_source, -1] = 0.5 * target_cross
        self.task_covariance = _psd_projection(covariance)
        self._fit_joint(np.empty((0, self.tasks[0].X.shape[1])), [], [])
        return self

    def _fit_joint(self, target_X, target_y, target_noise):
        n_source = len(self.tasks)
        X_rows = [task.X for task in self.tasks]
        task_rows = [
            np.full(len(task.X), index, dtype=int)
            for index, task in enumerate(self.tasks)
        ]
        y_rows = [np.asarray(task.y, dtype=float) for task in self.tasks]
        noise_rows = [np.asarray(task.noise, dtype=float) for task in self.tasks]
        if len(target_y):
            X_rows.append(_as_2d(target_X))
            task_rows.append(np.full(len(target_y), n_source, dtype=int))
            y_rows.append(np.asarray(target_y, dtype=float))
            noise_rows.append(np.asarray(target_noise, dtype=float))
        X = np.vstack(X_rows)
        task_index = np.concatenate(task_rows)
        y = np.concatenate(y_rows)
        noise = np.concatenate(noise_rows)
        mean, scale = self.normalization
        normalized = (y - mean) / scale
        normalized_noise = np.maximum(noise / (scale * scale), 1e-8)
        base = _kernel(
            X, X,
            kind="matern52",
            **self.hyperparameters,
        )
        K = base * self.task_covariance[
            task_index[:, None], task_index[None, :]
        ] + np.diag(normalized_noise)
        self._chol, _ = _stable_cholesky(K)
        self._alpha = np.linalg.solve(
            self._chol.T, np.linalg.solve(self._chol, normalized))
        self._train_X = X
        self._train_task = task_index

    def adapt(self, X, y, noise):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        noise = np.asarray(noise, dtype=float).reshape(-1)
        self.n_target = int(len(y))
        if self.n_target:
            scores = []
            for model in self.source_models:
                mean, variance = model.predict(X)
                total = np.maximum(variance + noise, 1e-12)
                scores.append(-0.5 * float(np.sum(
                    np.log(total) + (y - mean) ** 2 / total)))
            weights = np.exp(np.asarray(scores) - max(scores))
            weights /= max(float(np.sum(weights)), 1e-300)
            covariance = self.task_covariance.copy()
            source_diag = np.sqrt(np.maximum(
                np.diag(covariance)[:-1], 1e-12))
            cross = 0.8 * weights * source_diag
            covariance[-1, :-1] = cross
            covariance[:-1, -1] = cross
            self.task_covariance = _psd_projection(covariance)
            diagonal = np.sqrt(np.maximum(
                np.diag(self.task_covariance), 1e-12))
            self.task_covariance /= diagonal[:, None] * diagonal[None, :]
        self._fit_joint(X, y, noise)

    def predict(self, X, full_cov=False):
        X = _as_2d(X)
        target_task = len(self.tasks)
        cross_base = _kernel(
            self._train_X, X,
            kind="matern52",
            **self.hyperparameters,
        )
        cross = cross_base * self.task_covariance[
            self._train_task[:, None], target_task
        ]
        mean0, scale = self.normalization
        mean = mean0 + scale * (cross.T @ self._alpha)
        solved = np.linalg.solve(self._chol, cross)
        prior = _kernel(
            X, X,
            kind="matern52",
            **self.hyperparameters,
        ) * self.task_covariance[target_task, target_task]
        covariance = 0.5 * (
            prior - solved.T @ solved + (prior - solved.T @ solved).T)
        covariance *= scale * scale
        covariance[np.diag_indices_from(covariance)] = np.maximum(
            np.diag(covariance), 1e-12)
        if full_cov:
            return np.asarray(mean), covariance
        return np.asarray(mean), np.diag(covariance).copy()

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_reference": (
                "boschresearch/transfergpbo@"
                "4bdf90498c0048d7c80327609961d795d82b6f43"
            ),
            "n_target": int(self.n_target),
            "task_covariance": self.task_covariance.tolist(),
            "kernel": dict(self.hyperparameters),
        }
