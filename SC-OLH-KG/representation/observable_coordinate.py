"""Source-learned observable coordinates for chance-constraint means.

The cumulative-risk coordinate ``psi=(A,N)`` describes conditional variance.
This module supplies a separate low-dimensional coordinate ``eta`` for the
constraint mean.  Its input is a fixed multiscale library of normalized policy
statistics; source outcomes learn the projection, while held-out target truth
is never consulted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_PARTITIONS = (3, 4)
DEFAULT_DCT_DIM = 4


def _segment_stats(values, parts):
    means = []
    scales = []
    for segment in np.array_split(values, int(parts)):
        if len(segment) == 0:
            means.append(0.0)
            scales.append(0.0)
        else:
            means.append(float(np.mean(segment)))
            scales.append(float(np.std(segment)))
    return np.asarray(means), np.asarray(scales)


def observable_profile_library(
    profile,
    *,
    partitions=DEFAULT_PARTITIONS,
    dct_dim=DEFAULT_DCT_DIM,
):
    """Return a fixed-dimensional, formula-free ordered policy library."""

    z = np.asarray(profile, dtype=float).reshape(-1)
    if len(z) == 0:
        z = np.zeros(1, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("observable profile must be finite")
    z = np.clip(z, 0.0, 1.0)
    diffs = np.diff(z) if len(z) > 1 else np.zeros(1, dtype=float)
    tail = z[1:] if len(z) > 1 else z
    basic = np.asarray([
        float(np.mean(z)),
        float(np.std(z)),
        float(np.min(z)),
        float(np.max(z)),
        float(z[0]),
        float(z[-1]),
        float(np.mean(tail)),
        float(np.std(tail)),
        float(np.mean(np.abs(diffs))),
        float(np.std(diffs)),
    ], dtype=float)

    multiscale = []
    partition_means = {}
    for parts in tuple(int(value) for value in partitions):
        means, scales = _segment_stats(z, parts)
        partition_means[parts] = means
        multiscale.extend(means.tolist())
        multiscale.extend(scales.tolist())
    multiscale = np.asarray(multiscale, dtype=float)

    centered = z - float(np.mean(z))
    positions = np.arange(len(z), dtype=float) + 0.5
    dct = np.zeros(int(dct_dim), dtype=float)
    normalization = np.sqrt(2.0 / max(len(z), 1))
    for frequency in range(1, int(dct_dim) + 1):
        dct[frequency - 1] = normalization * float(np.sum(
            centered * np.cos(np.pi * frequency * positions / len(z))
        ))

    triadic = partition_means.get(3, np.zeros(3, dtype=float))
    interactions = np.asarray([
        triadic[0] * triadic[1],
        triadic[0] * triadic[2],
        triadic[1] * triadic[2],
    ], dtype=float)
    library = np.concatenate([
        basic,
        basic ** 2,
        multiscale,
        multiscale ** 2,
        dct,
        interactions,
    ])
    if not np.all(np.isfinite(library)):
        raise FloatingPointError("observable policy library is non-finite")
    return library


def observable_policy_library(problem, x):
    return observable_profile_library(problem.normalize(x))


@dataclass
class _SourceBoundaryModel:
    domain: str
    ridge: float
    coefficients: np.ndarray
    residual_scale: float
    validation_loss: float
    n_records: int
    target_scale: float = 1.0
    reliability: float = 1.0


class SourceLearnedObservableCoordinate:
    """Low-dimensional source boundary-score coordinate ``eta(x)``."""

    def __init__(
        self,
        ridge_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        output_mode="latent",
        latent_dim=2,
    ):
        grid = tuple(sorted(set(max(float(value), 1e-10)
                                for value in ridge_grid)))
        if not grid:
            raise ValueError("observable coordinate needs a ridge grid")
        self.ridge_grid = grid
        self.output_mode = str(output_mode).strip().lower()
        if self.output_mode not in {
            "atoms", "aggregate", "latent", "consensus"
        }:
            raise ValueError(
                "observable coordinate mode must be atoms, aggregate, latent, "
                "or consensus"
            )
        self.latent_dim = max(int(latent_dim), 0)
        self.feature_mean = None
        self.feature_scale = None
        self.models = []
        self.atom_mean = None
        self.atom_scale = None
        self.atom_components = None
        self.model_weights = None
        self.feature_dim = 0
        self.fit_status = "unfit"

    @staticmethod
    def _solve(matrix, target, weight, ridge):
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        weight = np.maximum(np.asarray(weight, dtype=float).reshape(-1), 1e-8)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        root = np.sqrt(weight)
        weighted = design * root[:, None]
        response = target * root
        penalty = np.eye(design.shape[1], dtype=float) * float(ridge)
        penalty[0, 0] = 0.0
        return np.linalg.pinv(weighted.T @ weighted + penalty) @ (
            weighted.T @ response)

    def _select_ridge(self, matrix, target, weight):
        n_rows = len(matrix)
        if n_rows < 8:
            return self.ridge_grid[len(self.ridge_grid) // 2], None
        n_folds = min(5, n_rows)
        fold_ids = np.arange(n_rows, dtype=int) % n_folds
        best = None
        for ridge in self.ridge_grid:
            predictions = np.zeros(n_rows, dtype=float)
            for fold in range(n_folds):
                test = fold_ids == fold
                train = ~test
                beta = self._solve(
                    matrix[train], target[train], weight[train], ridge)
                predictions[test] = np.column_stack([
                    np.ones(int(np.sum(test))), matrix[test]
                ]) @ beta
            boundary_weight = 1.0 + np.exp(-np.abs(target))
            false_safe = (predictions <= 0.0) & (target > 0.0)
            loss = float(np.average(
                boundary_weight * (predictions - target) ** 2
                + 4.0 * false_safe.astype(float),
                weights=weight,
            ))
            candidate = (loss, float(ridge))
            if best is None or candidate < best:
                best = candidate
        return best[1], best[0]

    def fit(self, profiles, margins, domains, sample_weight=None):
        profiles = list(profiles)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if not profiles or len(profiles) != len(margins) or len(domains) != len(margins):
            raise ValueError("observable coordinate training rows must align")
        weight = (
            np.ones(len(margins), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if len(weight) != len(margins):
            raise ValueError("observable coordinate weights must align")
        library = np.vstack([
            observable_profile_library(profile) for profile in profiles
        ])
        self.feature_mean = np.mean(library, axis=0)
        self.feature_scale = np.std(library, axis=0)
        self.feature_scale = np.where(
            self.feature_scale < 1e-8, 1.0, self.feature_scale)
        matrix = (library - self.feature_mean) / self.feature_scale
        self.models = []
        for domain in sorted(set(str(value) for value in domains)):
            indices = np.asarray([str(value) == domain for value in domains])
            domain_target = np.asarray(margins[indices], dtype=float)
            domain_weight = np.maximum(
                np.asarray(weight[indices], dtype=float), 1e-8)
            # A consensus atom should encode signed distance to the boundary,
            # not the arbitrary output scale of its source domain.  Scaling
            # without centering preserves the common boundary at zero.
            target_scale = 1.0
            if self.output_mode == "consensus":
                target_scale = max(float(np.sqrt(np.average(
                    domain_target ** 2,
                    weights=domain_weight,
                ))), 1e-6)
            fitted_target = domain_target / target_scale
            ridge, validation_loss = self._select_ridge(
                matrix[indices], fitted_target, domain_weight)
            beta = self._solve(
                matrix[indices], fitted_target, domain_weight, ridge)
            fitted = np.column_stack([
                np.ones(int(np.sum(indices))), matrix[indices]
            ]) @ beta
            residual_scale = max(float(np.sqrt(np.average(
                (fitted - fitted_target) ** 2,
                weights=domain_weight,
            ))), 0.05)
            self.models.append(_SourceBoundaryModel(
                domain=domain,
                ridge=float(ridge),
                coefficients=np.asarray(beta, dtype=float),
                residual_scale=residual_scale,
                validation_loss=(
                    None if validation_loss is None else float(validation_loss)
                ),
                n_records=int(np.sum(indices)),
                target_scale=float(target_scale),
            ))
        losses = np.asarray([
            (
                float(model.validation_loss)
                if model.validation_loss is not None
                else float(model.residual_scale) ** 2
            )
            for model in self.models
        ], dtype=float)
        losses = np.maximum(losses, 1e-8)
        inverse_loss = 1.0 / losses
        inverse_loss /= max(float(np.max(inverse_loss)), 1e-12)
        inverse_loss = np.maximum(inverse_loss, 0.05)
        self.model_weights = inverse_loss / max(
            float(np.sum(inverse_loss)), 1e-12)
        for model, reliability in zip(self.models, self.model_weights):
            model.reliability = float(reliability)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        atoms = np.column_stack([
            design @ model.coefficients for model in self.models
        ])
        atom_limit = 8.0 if self.output_mode == "consensus" else 20.0
        atoms = np.clip(atoms, -atom_limit, atom_limit)
        self.atom_mean = np.mean(atoms, axis=0)
        self.atom_scale = np.std(atoms, axis=0)
        self.atom_scale = np.where(self.atom_scale < 1e-8, 1.0, self.atom_scale)
        standardized_atoms = (atoms - self.atom_mean) / self.atom_scale
        _, _, right_t = np.linalg.svd(
            standardized_atoms, full_matrices=False)
        component_count = min(
            self.latent_dim, len(self.models), right_t.shape[0])
        self.atom_components = np.asarray(
            right_t[:component_count], dtype=float)
        for row in range(len(self.atom_components)):
            pivot = int(np.argmax(np.abs(self.atom_components[row])))
            if self.atom_components[row, pivot] < 0.0:
                self.atom_components[row] *= -1.0
        if self.output_mode == "atoms":
            self.feature_dim = len(self.models) + 4
        elif self.output_mode == "aggregate":
            self.feature_dim = 4
        elif self.output_mode == "consensus":
            self.feature_dim = 2
        else:
            self.feature_dim = 4 + component_count
        self.fit_status = "fit"
        return self

    def _standardized_library(self, profile):
        if self.fit_status != "fit":
            raise RuntimeError("observable coordinate is not fit")
        library = observable_profile_library(profile)
        return (library - self.feature_mean) / self.feature_scale

    def features_profile(self, profile):
        z = self._standardized_library(profile)
        design = np.concatenate([[1.0], z])
        atoms = np.asarray([
            float(design @ model.coefficients)
            for model in self.models
        ], dtype=float)
        atom_limit = 8.0 if self.output_mode == "consensus" else 20.0
        atoms = np.clip(atoms, -atom_limit, atom_limit)
        if self.output_mode == "consensus":
            weights = np.asarray(self.model_weights, dtype=float)
            signed_distance = float(np.sum(weights * atoms))
            disagreement = float(np.sqrt(np.sum(
                weights * (atoms - signed_distance) ** 2
            )))
            return np.asarray([
                np.clip(signed_distance, -atom_limit, atom_limit),
                np.clip(disagreement, 0.0, atom_limit),
            ], dtype=float)
        aggregate = np.asarray([
            float(np.mean(atoms)),
            float(np.std(atoms)),
            float(np.min(atoms)),
            float(np.max(atoms)),
        ], dtype=float)
        if self.output_mode == "atoms":
            return np.concatenate([atoms, aggregate])
        if self.output_mode == "aggregate":
            return aggregate
        latent = (
            (atoms - self.atom_mean) / self.atom_scale
        ) @ self.atom_components.T
        return np.concatenate([aggregate, latent])

    def features(self, problem, x):
        return self.features_profile(problem.normalize(x))

    def features_many(self, problem, points):
        if len(points) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(problem, point) for point in points])

    def diagnostics(self):
        return {
            "status": self.fit_status,
            "feature_dim": int(self.feature_dim),
            "output_mode": self.output_mode,
            "latent_dim": int(
                0 if self.atom_components is None
                else len(self.atom_components)),
            "library_dim": (
                None if self.feature_mean is None else int(len(self.feature_mean))
            ),
            "source_domains": [model.domain for model in self.models],
            "models": [
                {
                    "domain": model.domain,
                    "ridge": float(model.ridge),
                    "residual_scale": float(model.residual_scale),
                    "validation_loss": model.validation_loss,
                    "n_records": int(model.n_records),
                    "target_scale": float(model.target_scale),
                    "reliability": float(model.reliability),
                }
                for model in self.models
            ],
            "boundary_zero_preserved": bool(
                self.output_mode == "consensus"),
            "consensus_effective_sources": (
                None
                if self.model_weights is None
                else float(1.0 / np.sum(self.model_weights ** 2))
            ),
            "target_oracle_used": False,
            "target_labels_used_to_define_coordinate": False,
        }
