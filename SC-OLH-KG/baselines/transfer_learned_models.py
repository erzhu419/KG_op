"""Learned transfer models with explicit source/target adaptation semantics.

These compact implementations exercise the paper-level statistical mechanism
under the common SC-OLH-KG archive contract.  They are deliberately labelled
``paper_core_reimplementation``; official external repositories are never
silently substituted by these classes in paper-grade runs.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from baselines.transfer_models import (
    ExactGPSurrogate,
    HyperBOSurrogate,
    _as_2d,
    _stable_cholesky,
)


OFFICIAL_PROVENANCE = {
    "safe_fpacoh": {
        "repository": "jonasrothfuss/f-pacoh-torch",
        "commit": "746ef8155659c0060504874e4118a1a4fddf9f30",
    },
    "fsbo": {
        "repository": "machinelearningnuremberg/FSBO",
        "commit": "b4fbaaeac2fe7a3a2c6c05a222b028371123c3c9",
    },
    "metabo": {
        "repository": "boschresearch/MetaBO",
        "commit": "3f458bd32db340fbe2d5f072a92cfd782072342c",
    },
    "malibo": {
        "repository": "boschresearch/MALIBO",
        "commit": "87f1b3d2ed59441f8197b38ffdf68116bb90c2d8",
    },
}


def _normal_log_score(y, mean, variance):
    variance = np.maximum(np.asarray(variance, dtype=float), 1e-12)
    residual = np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)
    return 0.5 * float(np.sum(
        np.log(2.0 * math.pi * variance) + residual * residual / variance
    ))


class PaperCoreSafeFPACOHSurrogate:
    """Function-prior ensemble followed by target posterior conditioning.

    The ensemble is a finite approximation to a PACOH hyper-posterior.  Its
    source weights are selected by source-task predictive log score and remain
    frozen online; target observations condition every GP particle.
    """

    adaptation_kind = "posterior_conditioning"
    implementation_family = "safe_fpacoh"
    implementation_fidelity = "paper_core_reimplementation"

    def __init__(self, *, temperature=1.0):
        self.temperature = max(float(temperature), 1e-6)
        self.particles = []
        self.weights = np.empty(0, dtype=float)
        self.normalization = (0.0, 1.0)
        self.n_target = 0
        self.source_scores = []

    def meta_fit(self, tasks):
        tasks = list(tasks)
        if not tasks:
            raise ValueError("F-PACOH needs at least one source task")
        values = np.concatenate([np.asarray(task.y) for task in tasks])
        self.normalization = (
            float(np.mean(values)), max(float(np.std(values)), 1e-8))
        candidates = [
            (lengthscale, outputscale)
            for lengthscale in (0.05, 0.10, 0.20, 0.35, 0.60, 1.0)
            for outputscale in (0.25, 1.0, 4.0)
        ]
        scores = []
        for lengthscale, outputscale in candidates:
            score = 0.0
            for task in tasks:
                model = ExactGPSurrogate(
                    kernel="rbf",
                    lengthscale=lengthscale,
                    outputscale=outputscale,
                ).fit(task.X, task.y, task.noise)
                mean, variance = model.loo_moments()
                score += _normal_log_score(
                    task.y, mean, variance + np.asarray(task.noise))
            scores.append(score)
        scores = np.asarray(scores, dtype=float)
        # Retain multiple hypotheses instead of collapsing to MAP.
        order = np.argsort(scores)[: min(8, len(scores))]
        selected = [candidates[int(index)] for index in order]
        selected_scores = scores[order]
        log_weight = -(selected_scores - float(np.min(selected_scores)))
        log_weight /= self.temperature
        weights = np.exp(np.clip(log_weight, -700.0, 0.0))
        self.weights = weights / max(float(np.sum(weights)), 1e-300)
        self.particles = [ExactGPSurrogate(
            kernel="rbf",
            lengthscale=lengthscale,
            outputscale=outputscale,
            normalization=self.normalization,
        ) for lengthscale, outputscale in selected]
        self.source_scores = selected_scores.tolist()
        return self

    def adapt(self, X, y, noise):
        for particle in self.particles:
            particle.fit(X, y, noise)
        self.n_target = int(len(np.asarray(y).reshape(-1)))

    def predict(self, X, full_cov=False):
        predictions = [
            particle.predict(X, full_cov=full_cov)
            for particle in self.particles
        ]
        means = np.vstack([prediction[0] for prediction in predictions])
        mixture_mean = self.weights @ means
        if full_cov:
            covariance = np.zeros_like(predictions[0][1])
            for weight, mean, prediction in zip(
                self.weights, means, predictions
            ):
                delta = mean - mixture_mean
                covariance += weight * (
                    prediction[1] + np.outer(delta, delta))
            return mixture_mean, covariance
        variances = np.vstack([prediction[1] for prediction in predictions])
        mixture_variance = self.weights @ (
            variances + (means - mixture_mean[None, :]) ** 2)
        return mixture_mean, np.maximum(mixture_variance, 1e-12)

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": OFFICIAL_PROVENANCE["safe_fpacoh"],
            "source_prior_frozen_online": True,
            "online_parameters_changed": ["target_gp_posterior"],
            "n_particles": int(len(self.particles)),
            "particle_weights": self.weights.tolist(),
            "source_scores": list(map(float, self.source_scores)),
            "n_target": int(self.n_target),
        }


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by dependency audit
    torch = None
    nn = None


if nn is not None:
    class _DeepKernelFeatureNet(nn.Module):
        def __init__(self, dimension, hidden_dim, latent_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(dimension, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, latent_dim),
                nn.Tanh(),
            )

        def forward(self, X):
            return self.network(X)


class PaperCoreFSBOSurrogate:
    """Source-trained deep kernel with literal target gradient fine-tuning."""

    adaptation_kind = "end_to_end_gradient_finetuning"
    implementation_family = "fsbo"
    implementation_fidelity = "paper_core_reimplementation"

    def __init__(
        self,
        *,
        seed=0,
        hidden_dim=24,
        latent_dim=8,
        source_steps=200,
        target_steps=40,
        learning_rate=3e-3,
        ridge=1e-2,
    ):
        if torch is None:
            raise ImportError("PaperCoreFSBOSurrogate requires PyTorch")
        self.seed = int(seed)
        self.hidden_dim = max(4, int(hidden_dim))
        self.latent_dim = max(2, int(latent_dim))
        self.source_steps = max(1, int(source_steps))
        self.target_steps = max(0, int(target_steps))
        self.learning_rate = float(learning_rate)
        self.ridge = max(float(ridge), 1e-8)
        self.dimension = None
        self.net = None
        self.source_state = None
        self.source_head = None
        self.posterior_mean = None
        self.posterior_covariance = None
        self.residual_variance = 1.0
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.n_target = 0

    @staticmethod
    def _tensor(values):
        return torch.as_tensor(values, dtype=torch.float64)

    def meta_fit(self, tasks):
        tasks = list(tasks)
        if not tasks:
            raise ValueError("FSBO needs at least one source task")
        self.dimension = int(tasks[0].X.shape[1])
        torch.manual_seed(self.seed)
        self.net = _DeepKernelFeatureNet(
            self.dimension, self.hidden_dim, self.latent_dim).double()
        heads = nn.ModuleList([
            nn.Linear(self.latent_dim, 1).double() for _ in tasks
        ])
        values = np.concatenate([np.asarray(task.y) for task in tasks])
        self.y_mean = float(np.mean(values))
        self.y_scale = max(float(np.std(values)), 1e-8)
        parameters = list(self.net.parameters()) + list(heads.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        for _ in range(self.source_steps):
            optimizer.zero_grad()
            loss = torch.zeros((), dtype=torch.float64)
            for head, task in zip(heads, tasks):
                X = self._tensor(task.X)
                y = self._tensor(
                    (np.asarray(task.y) - self.y_mean) / self.y_scale
                ).reshape(-1, 1)
                noise = self._tensor(np.maximum(
                    np.asarray(task.noise) / (self.y_scale ** 2), 1e-6
                )).reshape(-1, 1)
                residual = head(self.net(X)) - y
                loss = loss + torch.mean(residual * residual / noise)
            (loss / len(tasks)).backward()
            optimizer.step()
        self.source_state = copy.deepcopy(self.net.state_dict())
        with torch.no_grad():
            self.source_head = torch.mean(torch.stack([
                torch.cat([head.weight.reshape(-1), head.bias.reshape(-1)])
                for head in heads
            ]), dim=0)
        self.adapt(
            np.empty((0, self.dimension)),
            np.empty(0),
            np.empty(0),
        )
        return self

    def _reset_target(self):
        self.net.load_state_dict(copy.deepcopy(self.source_state))
        head = nn.Linear(self.latent_dim, 1).double()
        with torch.no_grad():
            head.weight.copy_(self.source_head[:-1].reshape(1, -1))
            head.bias.copy_(self.source_head[-1:].reshape(1))
        return head

    def adapt(self, X, y, noise):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        noise = np.asarray(noise, dtype=float).reshape(-1)
        self.n_target = int(len(y))
        head = self._reset_target()
        if self.n_target:
            X_tensor = self._tensor(X)
            y_tensor = self._tensor(
                (y - self.y_mean) / self.y_scale).reshape(-1, 1)
            noise_tensor = self._tensor(np.maximum(
                noise / (self.y_scale ** 2), 1e-6)).reshape(-1, 1)
            optimizer = torch.optim.Adam(
                list(self.net.parameters()) + list(head.parameters()),
                lr=self.learning_rate,
            )
            for _ in range(self.target_steps):
                optimizer.zero_grad()
                residual = head(self.net(X_tensor)) - y_tensor
                loss = torch.mean(residual * residual / noise_tensor)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            if self.n_target:
                features = self.net(self._tensor(X)).numpy()
                target = (y - self.y_mean) / self.y_scale
                design = np.column_stack([np.ones(len(features)), features])
                precision = self.ridge * np.eye(design.shape[1])
                precision += design.T @ design
                covariance = np.linalg.inv(precision)
                mean = covariance @ design.T @ target
                residual = target - design @ mean
                self.residual_variance = max(
                    float(np.mean(residual * residual)),
                    float(np.mean(np.maximum(
                        noise / (self.y_scale ** 2), 1e-8))),
                    1e-6,
                )
            else:
                mean = np.r_[
                    float(head.bias.detach().numpy()[0]),
                    head.weight.detach().numpy().reshape(-1),
                ]
                covariance = np.eye(len(mean), dtype=float) / self.ridge
                self.residual_variance = 1.0
        self.posterior_mean = np.asarray(mean, dtype=float)
        self.posterior_covariance = np.asarray(covariance, dtype=float)

    def predict(self, X, full_cov=False):
        X = _as_2d(X)
        with torch.no_grad():
            features = self.net(self._tensor(X)).numpy()
        design = np.column_stack([np.ones(len(features)), features])
        normalized_mean = design @ self.posterior_mean
        covariance = (
            design @ self.posterior_covariance @ design.T
            * self.residual_variance
        )
        covariance[np.diag_indices_from(covariance)] += self.residual_variance
        covariance *= self.y_scale ** 2
        mean = self.y_mean + self.y_scale * normalized_mean
        if full_cov:
            return mean, covariance
        return mean, np.maximum(np.diag(covariance), 1e-12)

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": OFFICIAL_PROVENANCE["fsbo"],
            "source_prior_frozen_online": False,
            "online_parameters_changed": [
                "deep_feature_extractor",
                "target_bayesian_head",
            ],
            "source_steps": int(self.source_steps),
            "target_gradient_steps": int(self.target_steps),
            "n_target": int(self.n_target),
        }


def _rank_utility(y):
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(y, kind="stable")
    rank = np.empty(len(y), dtype=float)
    rank[order] = np.arange(len(y), dtype=float)
    return 1.0 - rank / max(len(y) - 1, 1)


class PaperCoreMetaBOSurrogate:
    """Frozen source-trained acquisition utility with target GP state."""

    adaptation_kind = "frozen_policy_with_target_posterior_state"
    implementation_family = "metabo"
    implementation_fidelity = "paper_core_reimplementation"

    def __init__(self, *, ridge=1e-2):
        self.ridge = max(float(ridge), 1e-8)
        self.gp = HyperBOSurrogate(kernel="matern52")
        self.policy = np.zeros(5, dtype=float)
        self.n_target = 0

    @staticmethod
    def _features(mean, variance, incumbent, progress):
        scale = max(float(np.std(mean)), 1e-8)
        std = np.sqrt(np.maximum(variance, 1e-12))
        return np.column_stack([
            np.ones(len(mean)),
            -np.asarray(mean) / scale,
            std / scale,
            np.maximum(float(incumbent) - np.asarray(mean), 0.0) / scale,
            np.full(len(mean), float(progress)),
        ])

    def meta_fit(self, tasks):
        tasks = list(tasks)
        self.gp.meta_fit(tasks)
        design = []
        target = []
        for task in tasks:
            model = ExactGPSurrogate(
                **ExactGPSurrogate.select_hyperparameters([task])
            ).fit(task.X, task.y, task.noise)
            mean, variance = model.loo_moments()
            design.append(self._features(
                mean,
                variance,
                float(np.min(task.y)),
                0.5,
            ))
            target.append(_rank_utility(task.y))
        matrix = np.vstack(design)
        values = np.concatenate(target)
        self.policy = np.linalg.solve(
            matrix.T @ matrix + self.ridge * np.eye(matrix.shape[1]),
            matrix.T @ values,
        )
        return self

    def adapt(self, X, y, noise):
        self.gp.adapt(X, y, noise)
        self.n_target = int(len(np.asarray(y).reshape(-1)))

    def predict(self, X, full_cov=False):
        return self.gp.predict(X, full_cov=full_cov)

    def acquisition_scores(self, X, incumbent, progress):
        mean, variance = self.predict(X)
        return self._features(mean, variance, incumbent, progress) @ self.policy

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": OFFICIAL_PROVENANCE["metabo"],
            "source_prior_frozen_online": True,
            "online_parameters_changed": ["target_gp_posterior_state"],
            "policy_weights": self.policy.tolist(),
            "n_target": int(self.n_target),
        }


class PaperCoreMALIBOSurrogate:
    """Meta-learned utility representation with a target Bayesian head."""

    adaptation_kind = "bayesian_utility_head_adaptation"
    implementation_family = "malibo"
    implementation_fidelity = "paper_core_reimplementation"

    def __init__(self, *, seed=0, feature_dim=24, ridge=1.0):
        self.seed = int(seed)
        self.feature_dim = max(4, int(feature_dim))
        self.ridge = max(float(ridge), 1e-8)
        self.projection = None
        self.phase = None
        self.prior_mean = None
        self.prior_precision = None
        self.posterior_mean = None
        self.posterior_covariance = None
        self.n_target = 0

    def _features(self, X):
        X = _as_2d(X)
        projected = X @ self.projection + self.phase[None, :]
        return np.column_stack([
            np.ones(len(X)),
            np.sqrt(2.0 / self.feature_dim) * np.cos(projected),
        ])

    def meta_fit(self, tasks):
        tasks = list(tasks)
        dimension = int(tasks[0].X.shape[1])
        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(
            0.0, 2.0, size=(dimension, self.feature_dim))
        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=self.feature_dim)
        design = []
        labels = []
        weights = []
        for task in tasks:
            utility = _rank_utility(task.y)
            design.append(self._features(task.X))
            labels.append((utility >= 2.0 / 3.0).astype(float))
            weights.append(0.25 + utility)
        matrix = np.vstack(design)
        target = np.concatenate(labels)
        sample_weight = np.concatenate(weights)
        precision = self.ridge * np.eye(matrix.shape[1])
        precision += matrix.T @ (sample_weight[:, None] * matrix)
        mean = np.linalg.solve(
            precision,
            matrix.T @ (sample_weight * target),
        )
        self.prior_mean = mean
        self.prior_precision = precision
        self.posterior_mean = mean.copy()
        self.posterior_covariance = np.linalg.inv(precision)
        return self

    def adapt(self, X, y, noise):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_target = int(len(y))
        if not self.n_target:
            self.posterior_mean = self.prior_mean.copy()
            self.posterior_covariance = np.linalg.inv(self.prior_precision)
            return
        utility = _rank_utility(y)
        labels = (utility >= 2.0 / 3.0).astype(float)
        weights = 0.25 + utility
        matrix = self._features(X)
        precision = self.prior_precision + matrix.T @ (
            weights[:, None] * matrix)
        rhs = self.prior_precision @ self.prior_mean
        rhs += matrix.T @ (weights * labels)
        self.posterior_covariance = np.linalg.inv(precision)
        self.posterior_mean = self.posterior_covariance @ rhs

    def utility_moments(self, X):
        matrix = self._features(X)
        latent = matrix @ self.posterior_mean
        probability = 1.0 / (1.0 + np.exp(-np.clip(latent, -30.0, 30.0)))
        latent_variance = np.einsum(
            "ij,jk,ik->i", matrix, self.posterior_covariance, matrix)
        probability_variance = (
            probability * (1.0 - probability)
        ) ** 2 * latent_variance
        return probability, np.maximum(probability_variance, 1e-12)

    def predict(self, X, full_cov=False):
        probability, variance = self.utility_moments(X)
        mean = -probability
        if full_cov:
            return mean, np.diag(variance)
        return mean, variance

    def acquisition_scores(self, X, incumbent=None, progress=None):
        return self.utility_moments(X)[0]

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": OFFICIAL_PROVENANCE["malibo"],
            "source_prior_frozen_online": True,
            "online_parameters_changed": ["target_bayesian_utility_head"],
            "n_target": int(self.n_target),
            "feature_dim": int(self.feature_dim),
        }
