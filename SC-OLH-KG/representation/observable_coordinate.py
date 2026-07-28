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
            "atoms", "aggregate", "latent", "consensus", "source_affine",
            "source_rank",
        }:
            raise ValueError(
                "observable coordinate mode must be atoms, aggregate, latent, "
                "consensus, source_affine, or source_rank"
            )
        self.latent_dim = max(int(latent_dim), 0)
        self.feature_mean = None
        self.feature_scale = None
        self.models = []
        self.atom_mean = None
        self.atom_scale = None
        self.atom_components = None
        self.model_weights = None
        self.source_rank_atlases = []
        self.source_rank_diagnostics = {"status": "not_requested"}
        self.source_prior_mean = None
        self.source_prior_covariance = None
        self.source_prior_residual_variance = None
        self.source_prior_domain_coefficients = {}
        self.source_prior_domain_rows = []
        self.source_prior_diagnostics = {"status": "unfit"}
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
            if self.output_mode in {"consensus", "source_affine"}:
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
        if self.output_mode == "source_rank":
            self._fit_source_rank_atlases(
                matrix, margins, domains, weight)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        atoms = np.column_stack([
            design @ model.coefficients for model in self.models
        ])
        atom_limit = (
            8.0
            if self.output_mode in {"consensus", "source_affine"}
            else 20.0
        )
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
        elif self.output_mode == "source_affine":
            self.feature_dim = len(self.models)
        elif self.output_mode == "source_rank":
            self.feature_dim = 2
        elif self.output_mode == "aggregate":
            self.feature_dim = 4
        elif self.output_mode == "consensus":
            self.feature_dim = 2
        else:
            self.feature_dim = 4 + component_count
        self.fit_status = "fit"
        self._fit_source_parametric_prior(
            profiles,
            margins,
            domains,
            weight,
        )
        return self

    @staticmethod
    def _percentile_ranks(values):
        """Stable lower-is-safer percentile ranks within one source domain."""

        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) <= 1:
            return np.zeros(len(values), dtype=float)
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = (
            np.arange(len(values), dtype=float) / float(len(values) - 1))
        return ranks

    def _fit_source_rank_atlases(
        self,
        standardized_library,
        margins,
        domains,
        sample_weight,
    ):
        """Freeze one scale-invariant empirical safety atlas per source.

        Only the ordering of ordinary source chance margins is retained.
        Consequently every strictly increasing rescaling of a source domain's
        margin leaves the learned coordinate unchanged.
        """

        matrix = np.asarray(standardized_library, dtype=float)
        margin = np.asarray(margins, dtype=float).reshape(-1)
        domain_values = np.asarray(domains, dtype=object).reshape(-1)
        weights = np.maximum(
            np.asarray(sample_weight, dtype=float).reshape(-1), 1e-8)
        atlases = []
        for domain in sorted(set(str(value) for value in domain_values)):
            selected = np.asarray([
                str(value) == domain for value in domain_values
            ], dtype=bool)
            atlases.append({
                "domain": domain,
                "features": matrix[selected].copy(),
                "ranks": self._percentile_ranks(margin[selected]),
                "weights": weights[selected].copy(),
            })
        if not atlases:
            raise RuntimeError("source-rank coordinate has no source atlas")
        self.source_rank_atlases = atlases
        self.source_rank_diagnostics = {
            "status": "fit",
            "coordinate": "eta_source_rank",
            "source_domains": [row["domain"] for row in atlases],
            "source_domain_count": int(len(atlases)),
            "source_record_count": int(sum(
                len(row["ranks"]) for row in atlases)),
            "rank_interpolator": "adaptive_rbf_knn",
            "rank_neighbors": 8,
            "rank_range": [0.0, 1.0],
            "strict_monotone_scale_invariant": True,
            "target_data_used": False,
            "target_oracle_used": False,
        }

    def _source_rank_features_standardized(self, standardized_library):
        query = np.asarray(standardized_library, dtype=float)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if not self.source_rank_atlases:
            raise RuntimeError("source-rank coordinate is not fit")
        predictions = []
        for atlas in self.source_rank_atlases:
            source = np.asarray(atlas["features"], dtype=float)
            distance = (
                np.sum(query ** 2, axis=1)[:, None]
                + np.sum(source ** 2, axis=1)[None, :]
                - 2.0 * query @ source.T
            ) / max(query.shape[1], 1)
            distance = np.maximum(distance, 0.0)
            count = min(8, source.shape[0])
            nearest = np.argpartition(
                distance, count - 1, axis=1)[:, :count]
            local_distance = np.take_along_axis(
                distance, nearest, axis=1)
            local_rank = np.asarray(atlas["ranks"], dtype=float)[nearest]
            local_source_weight = np.asarray(
                atlas["weights"], dtype=float)[nearest]
            bandwidth = np.maximum(
                np.median(local_distance, axis=1), 1e-8)
            kernel = (
                np.exp(-local_distance / bandwidth[:, None])
                * local_source_weight
            )
            kernel /= np.maximum(
                np.sum(kernel, axis=1, keepdims=True), 1e-12)
            predictions.append(np.sum(kernel * local_rank, axis=1))
        rank = np.column_stack(predictions)
        mean_rank = np.mean(rank, axis=1)
        worst_rank = np.max(rank, axis=1)
        disagreement = np.std(rank, axis=1)
        consensus = (
            mean_rank + 0.25 * worst_rank + 0.25 * disagreement)
        result = np.column_stack([consensus, disagreement])
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("source-rank coordinate is non-finite")
        return result

    def _source_affine_prior_rows(
        self,
        features,
        target,
        domain_values,
        weights,
    ):
        """Learn a source-only offset/scale law for every boundary atom.

        For source atom ``h_s``, every available source domain ``r`` supplies
        an affine calibration ``a_sr + b_sr h_s``. Their source-only
        distribution is the prior used on a held-out target. Coefficients of
        other atoms are fixed near zero, so a component preserves one complete
        boundary shape instead of relearning an arbitrary cross-atom blend.
        """

        if features.shape[1] != len(self.models):
            raise RuntimeError(
                "source-affine features must contain one atom per source")
        domains = sorted(set(str(value) for value in domain_values))
        coefficient_dim = 1 + features.shape[1]
        rows = []
        for atom_index, source_model in enumerate(self.models):
            calibrations = []
            for target_domain in domains:
                selected = np.asarray([
                    str(value) == target_domain for value in domain_values
                ], dtype=bool)
                matrix = features[selected, atom_index:atom_index + 1]
                domain_target = target[selected]
                domain_weight = weights[selected]
                ridge, validation_loss = self._select_ridge(
                    matrix, domain_target, domain_weight)
                coefficient = self._solve(
                    matrix, domain_target, domain_weight, ridge)
                active_design = np.column_stack([
                    np.ones(int(np.sum(selected))), matrix
                ])
                residual = domain_target - active_design @ coefficient
                residual_variance = max(float(np.average(
                    residual ** 2, weights=domain_weight)), 1e-6)
                penalty = np.diag([0.0, float(ridge)])
                precision = (
                    active_design.T
                    @ (domain_weight[:, None] * active_design)
                    + penalty
                )
                covariance = residual_variance * np.linalg.pinv(precision)
                covariance = 0.5 * (covariance + covariance.T)
                loss = (
                    residual_variance
                    if validation_loss is None
                    else max(float(validation_loss), 1e-8)
                )
                calibrations.append({
                    "target_domain": target_domain,
                    "coefficient": np.asarray(coefficient, dtype=float),
                    "covariance": covariance,
                    "residual_variance": residual_variance,
                    "ridge": float(ridge),
                    "validation_loss": (
                        None if validation_loss is None
                        else float(validation_loss)
                    ),
                    "loss": float(loss),
                    "mass": float(np.sum(domain_weight)),
                    "n_records": int(np.sum(selected)),
                })

            calibration_weight = np.asarray([
                row["mass"] for row in calibrations
            ], dtype=float)
            calibration_weight /= max(
                float(np.sum(calibration_weight)), 1e-12)
            active_mean = np.sum([
                mass * row["coefficient"]
                for mass, row in zip(calibration_weight, calibrations)
            ], axis=0)
            within = np.sum([
                mass * row["covariance"]
                for mass, row in zip(calibration_weight, calibrations)
            ], axis=0)
            between = np.sum([
                mass * np.outer(
                    row["coefficient"] - active_mean,
                    row["coefficient"] - active_mean,
                )
                for mass, row in zip(calibration_weight, calibrations)
            ], axis=0)
            residual_variance = float(np.sum([
                mass * row["residual_variance"]
                for mass, row in zip(calibration_weight, calibrations)
            ]))
            active_floor = max(
                0.05 * residual_variance / max(len(target), 1),
                1e-6,
            )
            active_covariance = (
                within + between + active_floor * np.eye(2, dtype=float))
            coefficient = np.zeros(coefficient_dim, dtype=float)
            active = np.asarray([0, 1 + atom_index], dtype=int)
            coefficient[active] = active_mean
            covariance = 1e-10 * np.eye(coefficient_dim, dtype=float)
            covariance[np.ix_(active, active)] = active_covariance
            validation_loss = float(np.sum([
                mass * row["loss"]
                for mass, row in zip(calibration_weight, calibrations)
            ]))
            rows.append({
                "domain": str(source_model.domain),
                "coefficient": coefficient,
                "covariance": covariance,
                "residual_variance": residual_variance,
                "ridge": float(np.sum([
                    mass * row["ridge"]
                    for mass, row in zip(calibration_weight, calibrations)
                ])),
                "validation_loss": validation_loss,
                "reliability": 1.0 / max(validation_loss, 1e-8),
                "n_records": int(len(target)),
                "active_atom_index": int(atom_index),
                "affine_calibrations": [
                    {
                        "target_domain": row["target_domain"],
                        "offset": float(row["coefficient"][0]),
                        "scale": float(row["coefficient"][1]),
                        "validation_loss": row["validation_loss"],
                        "residual_variance": row["residual_variance"],
                        "n_records": row["n_records"],
                    }
                    for row in calibrations
                ],
            })
        return rows

    def _fit_source_parametric_prior(
        self,
        profiles,
        margins,
        domains,
        sample_weight,
    ):
        """Fit a hierarchical source prior for target constraint coefficients.

        Each source domain supplies a Bayesian-ridge coefficient estimate in
        the frozen ``eta`` coordinate.  Their within-domain uncertainty and
        between-domain disagreement define the prior covariance for a held-out
        target.  No target record is available at this stage.
        """

        features = np.vstack([
            self.features_profile(profile) for profile in profiles
        ])
        target = np.asarray(margins, dtype=float).reshape(-1)
        domain_values = np.asarray(domains, dtype=object).reshape(-1)
        weights = np.maximum(
            np.asarray(sample_weight, dtype=float).reshape(-1), 1e-8)
        design = np.column_stack([np.ones(len(features)), features])
        penalty = np.eye(design.shape[1], dtype=float)
        penalty[0, 0] = 0.0

        if self.output_mode == "source_affine":
            domain_rows = self._source_affine_prior_rows(
                features, target, domain_values, weights)
        else:
            domain_rows = []
            for domain in sorted(set(str(value) for value in domain_values)):
                selected = np.asarray([
                    str(value) == domain for value in domain_values
                ], dtype=bool)
                domain_features = features[selected]
                domain_target = target[selected]
                domain_weight = weights[selected]
                ridge, validation_loss = self._select_ridge(
                    domain_features, domain_target, domain_weight)
                beta = self._solve(
                    domain_features, domain_target, domain_weight, ridge)
                domain_design = design[selected]
                residual = domain_target - domain_design @ beta
                residual_variance = max(float(np.average(
                    residual ** 2,
                    weights=domain_weight,
                )), 1e-6)
                precision = (
                    domain_design.T
                    @ (domain_weight[:, None] * domain_design)
                    + float(ridge) * penalty
                )
                covariance = residual_variance * np.linalg.pinv(precision)
                covariance = 0.5 * (covariance + covariance.T)
                reliability = next(
                    (
                        float(model.reliability)
                        for model in self.models
                        if model.domain == domain
                    ),
                    1.0,
                )
                domain_rows.append({
                    "domain": domain,
                    "coefficient": np.asarray(beta, dtype=float),
                    "covariance": covariance,
                    "residual_variance": residual_variance,
                    "ridge": float(ridge),
                    "validation_loss": (
                        None
                        if validation_loss is None
                        else float(validation_loss)
                    ),
                    "reliability": max(reliability, 1e-8),
                    "n_records": int(np.sum(selected)),
                })

        mixture_weight = np.asarray([
            row["reliability"] for row in domain_rows
        ], dtype=float)
        mixture_weight /= max(float(np.sum(mixture_weight)), 1e-12)
        prior_mean = np.sum([
            weight * row["coefficient"]
            for weight, row in zip(mixture_weight, domain_rows)
        ], axis=0)
        within = np.sum([
            weight * row["covariance"]
            for weight, row in zip(mixture_weight, domain_rows)
        ], axis=0)
        between = np.sum([
            weight * np.outer(
                row["coefficient"] - prior_mean,
                row["coefficient"] - prior_mean,
            )
            for weight, row in zip(mixture_weight, domain_rows)
        ], axis=0)
        residual_variance = float(np.sum([
            weight * row["residual_variance"]
            for weight, row in zip(mixture_weight, domain_rows)
        ]))
        diagonal_scale = np.diag(within)
        finite_diagonal = diagonal_scale[np.isfinite(diagonal_scale)]
        covariance_floor = max(
            (
                0.05 * float(np.median(finite_diagonal))
                if len(finite_diagonal)
                else 0.0
            ),
            0.05 * residual_variance / max(len(target), 1),
            1e-6,
        )
        prior_covariance = (
            within
            + between
            + covariance_floor * np.eye(design.shape[1], dtype=float)
        )
        prior_covariance = 0.5 * (
            prior_covariance + prior_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(prior_covariance)
        prior_covariance = (
            eigenvectors * np.maximum(eigenvalues, covariance_floor)
        ) @ eigenvectors.T

        self.source_prior_mean = np.asarray(prior_mean, dtype=float)
        self.source_prior_covariance = np.asarray(
            prior_covariance, dtype=float)
        self.source_prior_residual_variance = residual_variance
        self.source_prior_domain_coefficients = {
            row["domain"]: row["coefficient"].copy()
            for row in domain_rows
        }
        self.source_prior_domain_rows = [
            {
                "domain": str(row["domain"]),
                "coefficient": np.asarray(
                    row["coefficient"], dtype=float).copy(),
                "covariance": np.asarray(
                    row["covariance"], dtype=float).copy(),
                "residual_variance": float(row["residual_variance"]),
                "ridge": float(row["ridge"]),
                "validation_loss": row["validation_loss"],
                "reliability": float(row["reliability"]),
                "n_records": int(row["n_records"]),
                "active_atom_index": row.get("active_atom_index"),
                "affine_calibrations": [
                    dict(value)
                    for value in row.get("affine_calibrations", [])
                ],
            }
            for row in domain_rows
        ]
        self.source_prior_diagnostics = {
            "status": "fit",
            "coordinate": (
                "eta_source_affine"
                if self.output_mode == "source_affine"
                else (
                    "eta_source_rank"
                    if self.output_mode == "source_rank"
                    else "eta"
                )
            ),
            "coefficient_dim": int(len(prior_mean)),
            "source_domains": [row["domain"] for row in domain_rows],
            "source_domain_count": int(len(domain_rows)),
            "source_record_count": int(len(target)),
            "source_residual_variance": residual_variance,
            "within_trace": float(np.trace(within)),
            "between_trace": float(np.trace(between)),
            "prior_covariance_trace": float(np.trace(prior_covariance)),
            "prior_covariance_min_eigenvalue": float(
                np.min(np.linalg.eigvalsh(prior_covariance))),
            "covariance_floor": float(covariance_floor),
            "domain_fits": [
                {
                    "domain": row["domain"],
                    "ridge": row["ridge"],
                    "validation_loss": row["validation_loss"],
                    "residual_variance": row["residual_variance"],
                    "reliability": row["reliability"],
                    "n_records": row["n_records"],
                    "active_atom_index": row.get("active_atom_index"),
                    "affine_calibrations": row.get(
                        "affine_calibrations", []),
                }
                for row in domain_rows
            ],
            "target_data_used": False,
            "target_oracle_used": False,
        }

    def source_parametric_prior(self, problem):
        """Return the frozen source prior in the target output scale."""

        if self.source_prior_mean is None or self.source_prior_covariance is None:
            raise RuntimeError("source parametric prior is unavailable")
        tau = float(getattr(problem, "tau", 0.0))
        output_scale = max(
            abs(tau),
            float(getattr(problem, "sigma_level", 0.0)),
            1e-6,
        )
        mean = output_scale * self.source_prior_mean.copy()
        mean[0] += tau
        covariance = (
            output_scale ** 2 * self.source_prior_covariance.copy())
        deviation_variance = max(
            output_scale ** 2 * float(self.source_prior_residual_variance),
            1e-8,
        )
        return {
            "mean": mean,
            "covariance": covariance,
            "deviation_variance": deviation_variance,
            "output_scale": output_scale,
            "diagnostics": {
                **self.source_prior_diagnostics,
                "target_output_scale": output_scale,
                "target_tau": tau,
                "target_data_used": False,
                "target_oracle_used": False,
            },
        }

    def source_parametric_prior_components(self, problem):
        """Return source-domain coefficient components in target units.

        The components remain source-only.  Target observations may update
        their model probabilities later, but neither target labels nor target
        oracle values are used to construct the component means/covariances.
        """

        if not self.source_prior_domain_rows:
            raise RuntimeError("source parametric prior components are unavailable")
        tau = float(getattr(problem, "tau", 0.0))
        output_scale = max(
            abs(tau),
            float(getattr(problem, "sigma_level", 0.0)),
            1e-6,
        )
        covariance_floor = max(
            float(self.source_prior_diagnostics.get(
                "covariance_floor", 1e-6)),
            1e-12,
        )
        reliability = np.asarray([
            max(float(row["reliability"]), 1e-12)
            for row in self.source_prior_domain_rows
        ], dtype=float)
        reliability /= max(float(np.sum(reliability)), 1e-12)

        components = []
        for prior_weight, row in zip(
            reliability, self.source_prior_domain_rows
        ):
            mean = output_scale * np.asarray(
                row["coefficient"], dtype=float).copy()
            mean[0] += tau
            covariance = np.asarray(row["covariance"], dtype=float)
            covariance = 0.5 * (covariance + covariance.T)
            component_floor = (
                1e-10
                if self.output_mode == "source_affine"
                else covariance_floor
            )
            if self.output_mode != "source_affine":
                covariance += covariance_floor * np.eye(
                    covariance.shape[0], dtype=float)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            covariance = (
                eigenvectors * np.maximum(eigenvalues, component_floor)
            ) @ eigenvectors.T
            components.append({
                "name": f"source:{row['domain']}",
                "domain": str(row["domain"]),
                "mean": mean,
                "covariance": output_scale ** 2 * covariance,
                "deviation_variance": max(
                    output_scale ** 2
                    * float(row["residual_variance"]),
                    1e-8,
                ),
                "prior_weight": float(prior_weight),
                "diagnostics": {
                    "domain": str(row["domain"]),
                    "ridge": float(row["ridge"]),
                    "validation_loss": row["validation_loss"],
                    "source_residual_variance": float(
                        row["residual_variance"]),
                    "source_reliability": float(row["reliability"]),
                    "source_record_count": int(row["n_records"]),
                    "component_kind": (
                        "source_boundary_affine"
                        if self.output_mode == "source_affine"
                        else (
                            "source_rank_coefficient"
                            if self.output_mode == "source_rank"
                            else "source_coefficient"
                        )
                    ),
                    "active_atom_index": row.get("active_atom_index"),
                    "affine_calibrations": row.get(
                        "affine_calibrations", []),
                    "target_output_scale": output_scale,
                    "target_tau": tau,
                    "target_data_used": False,
                    "target_oracle_used": False,
                },
            })
        return components

    def _standardized_library(self, profile):
        if self.fit_status != "fit":
            raise RuntimeError("observable coordinate is not fit")
        library = observable_profile_library(profile)
        return (library - self.feature_mean) / self.feature_scale

    def features_profile(self, profile):
        z = self._standardized_library(profile)
        if self.output_mode == "source_rank":
            return self._source_rank_features_standardized(z)[0]
        design = np.concatenate([[1.0], z])
        atoms = np.asarray([
            float(design @ model.coefficients)
            for model in self.models
        ], dtype=float)
        atom_limit = (
            8.0
            if self.output_mode in {"consensus", "source_affine"}
            else 20.0
        )
        atoms = np.clip(atoms, -atom_limit, atom_limit)
        if self.output_mode == "source_affine":
            return atoms
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
        if self.output_mode == "source_rank":
            library = np.vstack([
                observable_profile_library(problem.normalize(point))
                for point in points
            ])
            standardized = (
                library - self.feature_mean[None, :]
            ) / self.feature_scale[None, :]
            return self._source_rank_features_standardized(standardized)
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
                self.output_mode in {"consensus", "source_affine"}),
            "consensus_effective_sources": (
                None
                if self.model_weights is None
                else float(1.0 / np.sum(self.model_weights ** 2))
            ),
            "source_parametric_prior": dict(self.source_prior_diagnostics),
            "source_rank": dict(self.source_rank_diagnostics),
            "target_oracle_used": False,
            "target_labels_used_to_define_coordinate": False,
        }
