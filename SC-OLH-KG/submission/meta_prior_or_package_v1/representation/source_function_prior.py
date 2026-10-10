"""Small source-only functional prior for the meta-prior mechanism experiment.

The eight canonical input coordinates are public in this experiment. Learning
selects response functions, not these coordinates or historical good policies.
Each response head gets its own support. Target coefficients start at zero;
source tasks provide their covariance and the intercept distribution.
"""

from dataclasses import dataclass

import numpy as np

from core.gpr import ParametricGPR
from representation.transferable_spectral import (
    SourceDomainBatch, TransferableSpectralBasis,
)


class PublicCoordinateSpectralBasis(TransferableSpectralBasis):
    """Use a declared coordinate system and include a high-frequency control."""

    def _scale_psi(self, psi):
        return np.asarray(psi, dtype=float)

    def _library(self, psi_z):
        library, names = super()._library(psi_z)
        smooth = np.tanh(0.5 * np.atleast_2d(psi_z))
        high = np.sin(4.0 * np.pi * smooth)
        return np.column_stack([library, high]), names + [
            f"sin(4*pi*psi{j})" for j in range(smooth.shape[1])
        ]


class FixedFunctionBasis:
    """Frozen dictionary, optionally restricted to an oracle's support."""

    def __init__(self, reference_psi, indices=None):
        self.dictionary = PublicCoordinateSpectralBasis(max_library_size=64)
        raw, names = self.dictionary._library(reference_psi)
        self.indices = np.arange(raw.shape[1]) if indices is None else np.array(indices)
        raw = raw[:, self.indices]
        self.mean = raw.mean(axis=0)
        self.scale = np.maximum(raw.std(axis=0), 1e-8)
        self.feature_dim = len(self.indices)
        self.names = [names[i] for i in self.indices]

    def transform(self, psi):
        raw, _ = self.dictionary._library(psi)
        return (raw[:, self.indices] - self.mean) / self.scale

    def diagnostics(self):
        return {"selected_names": self.names, "active_dim": self.feature_dim,
                "orthogonalization": "none", "input_coordinates": "public_canonical"}


class IndexedFunctionBasis:
    """Supply the existing integer-policy GPR with learned candidate features."""

    def __init__(self, basis, candidate_psi):
        self.table = basis.transform(candidate_psi)
        self.feature_dim = basis.feature_dim

    def features(self, x):
        return self.table[int(x[0])]

    def features_many(self, X):
        return self.table[np.asarray(X, dtype=int).reshape(-1)]


@dataclass
class SourceFunctionPrior:
    basis: object
    mean: np.ndarray
    covariance: np.ndarray
    discrepancy_variance: float
    diagnostics: dict

    @classmethod
    def fit(cls, batches, observation_variances, public_candidates, mode,
            *, oracle_indices=None, active_dim=4):
        """Fit one head using source labels only; mode is an experimental arm."""
        if mode not in {"target_only", "fixed_source", "learned", "oracle_structure"}:
            raise ValueError(mode)
        if mode == "learned":
            basis = PublicCoordinateSpectralBasis(
                active_dim=active_dim, max_library_size=64,
                pilot_cv_size=8, pilot_cv_repeats=1,
            ).fit(batches)
        else:
            basis = FixedFunctionBasis(public_candidates, oracle_indices)
        return cls.from_basis(batches, observation_variances, basis, mode)

    @classmethod
    def from_basis(cls, batches, observation_variances, basis, mode):
        """Estimate coefficient moments on an already declared source basis."""
        p = basis.feature_dim + 1
        if mode == "target_only":
            covariance = np.diag([4.0] + [1.0 / basis.feature_dim] * basis.feature_dim)
            return cls(basis, np.zeros(p), covariance, 0.02,
                       {"mode": mode, "source_label_count": 0, **basis.diagnostics()})

        coefficients, residuals, response_variances = [], [], []
        for batch, noise in zip(batches, observation_variances):
            y = np.asarray(batch.signals).reshape(-1)
            Z = basis.transform(batch.psi)
            X = np.column_stack([np.ones(len(Z)), Z])
            noise = np.asarray(noise, dtype=float)
            weights = 1.0 / np.maximum(noise, 1e-8)
            penalty = np.eye(p) * 1e-3
            penalty[0, 0] = 1e-8
            beta = np.linalg.solve(X.T @ (weights[:, None] * X) + penalty,
                                   X.T @ (weights * y))
            coefficients.append(beta)
            # A discrepancy in the fitted function is epistemic model error.
            # It is not added to the simulator's environmental variance head.
            residuals.append(max(float(np.mean((y - X @ beta) ** 2 - noise)), 0.0))
            response_variances.append(max(float(np.var(y) - np.mean(noise)), 0.02))
        beta = np.vstack(coefficients)
        mean = np.zeros(p)
        mean[0] = np.mean(beta[:, 0])
        covariance = np.zeros((p, p))
        covariance[0, 0] = max(float(np.var(beta[:, 0], ddof=1)), 0.25)
        if mode == "fixed_source":
            # Same labels, one pooled scale: no learned support or anisotropy.
            covariance[1:, 1:] = np.eye(p - 1) * np.mean(response_variances) / (p - 1)
            mean[0] = np.mean([np.mean(b.signals) for b in batches])
        elif p > 1:
            second_moment = beta[:, 1:].T @ beta[:, 1:] / len(beta)
            floor = max(float(np.trace(second_moment) / (p - 1)) * 0.05, 0.01)
            covariance[1:, 1:] = second_moment + floor * np.eye(p - 1)
        diagnostics = {"mode": mode, "source_label_count": sum(len(b.psi) for b in batches),
                       "source_task_count": len(batches), "coefficient_mean": mean.tolist(),
                       "coefficient_prior_diagonal": np.diag(covariance).tolist(),
                       **basis.diagnostics()}
        diagnostics.pop("fingerprint", None)
        return cls(basis, mean, covariance, max(float(np.mean(residuals)), 0.02), diagnostics)

    def target_model(self, candidate_psi):
        model = ParametricGPR(d=1, basis_map=IndexedFunctionBasis(self.basis, candidate_psi))
        model.set_parametric_prior(self.mean, self.discrepancy_variance, self.covariance)
        return model


def head_batches(source_tasks, head):
    return [SourceDomainBatch(str(i), task["psi"], task["response"][:, head])
            for i, task in enumerate(source_tasks)]
