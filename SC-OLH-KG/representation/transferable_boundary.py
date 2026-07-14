"""Transferable chance-boundary coordinates and low-dimensional geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.special import logsumexp
from scipy.stats import t as student_t


@dataclass(frozen=True)
class TargetBoundaryAdapter:
    shift: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    output_offset: float
    output_scale: float
    output_coordinate_linear: np.ndarray
    calibration_coordinate_count: int
    output_covariance: np.ndarray
    residual_scale: float
    residual_degrees_of_freedom: float
    effective_dimension: int
    mode: str
    diagnostics: dict


@dataclass(frozen=True)
class BoundaryFamilyMixtureAdapter:
    """Target posterior over a frozen finite library of boundary families."""

    family_adapters: tuple[TargetBoundaryAdapter, ...]
    posterior_weights: np.ndarray
    log_predictive_evidence: np.ndarray
    credible_indices: np.ndarray
    credible_mass: float
    effective_dimension: int
    mode: str
    diagnostics: dict


@dataclass(frozen=True)
class BoundaryFamilySynthesisAdapter:
    """Target posterior over a nonnegative source-boundary dictionary."""

    intercept: float
    coefficients: np.ndarray
    output_covariance: np.ndarray
    residual_scale: float
    residual_degrees_of_freedom: float
    effective_dimension: int
    mode: str
    diagnostics: dict


@dataclass(frozen=True)
class BoundaryFamilySemiparametricAdapter:
    """Target posterior for synthesis plus an orthogonal local residual."""

    intercept: float
    coefficients: np.ndarray
    residual_coefficients: np.ndarray
    output_covariance: np.ndarray
    residual_scale: float
    residual_degrees_of_freedom: float
    effective_dimension: int
    mode: str
    diagnostics: dict


class TransferableChanceBoundaryPosterior:
    """Source-fitted chance-boundary model with a tiny target adapter.

    The model consumes only a domain-independent observable descriptor.  Its
    source fit learns both a coordinate chart and a signed chance-margin
    geometry.  A held-out target may use ordinary pilot observations to fit a
    low-dimensional adapter; target evaluation rows are never accepted by the
    fitting API.
    """

    COORDINATES = (
        "explicit_stable",
        "learned_psi",
        "boundary_latent",
        "hybrid_explicit_latent",
    )
    GEOMETRIES = (
        "linear_monotone",
        "diagonal_psd",
        "low_rank_psd",
        "rbf",
    )
    ADAPTATIONS = ("frozen", "shift_scale", "orthogonal_shift")

    def __init__(
        self,
        *,
        coordinate="boundary_latent",
        geometry="low_rank_psd",
        adaptation="frozen",
        rank=2,
        ridge=1e-3,
        domain_penalty=0.5,
        boundary_temperature=1.0,
        adaptation_ridge=5.0,
        upper_alpha=0.01,
        calibration_prior_df=1.0,
        target_residual_rank=1,
    ):
        self.coordinate = str(coordinate)
        self.geometry = str(geometry)
        self.adaptation = str(adaptation)
        self.rank = max(1, int(rank))
        self.ridge = max(float(ridge), 1e-10)
        self.domain_penalty = max(float(domain_penalty), 0.0)
        self.boundary_temperature = max(
            float(boundary_temperature), 1e-8)
        self.adaptation_ridge = max(float(adaptation_ridge), 0.0)
        self.upper_alpha = float(np.clip(upper_alpha, 1e-6, 0.25))
        self.calibration_prior_df = max(float(calibration_prior_df), 1.0)
        self.target_residual_rank = max(int(target_residual_rank), 0)
        if self.coordinate not in self.COORDINATES:
            raise ValueError(f"unknown boundary coordinate {self.coordinate!r}")
        if self.geometry not in self.GEOMETRIES:
            raise ValueError(f"unknown boundary geometry {self.geometry!r}")
        if self.adaptation not in self.ADAPTATIONS:
            raise ValueError(f"unknown target adaptation {self.adaptation!r}")
        self.fit_status = "unfit"
        self.diagnostics_ = {"status": "unfit"}

    @staticmethod
    def _weighted_mean(values, weights):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        return np.sum(np.asarray(values, dtype=float) * weights[:, None], axis=0)

    @staticmethod
    def _margin_bins(margins):
        values = np.asarray(margins, dtype=float).reshape(-1)
        return np.digitize(values, [-1.0, -0.25, 0.25, 1.0])

    def _boundary_weights(self, margins, sample_weight):
        margins = np.asarray(margins, dtype=float).reshape(-1)
        base = np.asarray(sample_weight, dtype=float).reshape(-1)
        boundary = np.exp(
            -0.5 * (margins / self.boundary_temperature) ** 2)
        weights = np.maximum(base, 1e-8) * (1.0 + 2.0 * boundary)
        return weights / max(float(np.mean(weights)), 1e-12)

    @staticmethod
    def _orient_axes(Z, margins, axes):
        axes = np.asarray(axes, dtype=float).copy()
        projected = Z @ axes
        centered_margin = margins - float(np.mean(margins))
        for index in range(axes.shape[1]):
            centered = projected[:, index] - float(np.mean(projected[:, index]))
            if float(centered @ centered_margin) < 0.0:
                axes[:, index] *= -1.0
        return axes

    def _stable_axes(self, Z, margins, domains, rank):
        domains = np.asarray(domains, dtype=object)
        pooled = []
        variability = []
        for feature in range(Z.shape[1]):
            correlations = []
            for domain in sorted(set(domains.tolist())):
                mask = domains == domain
                if int(np.sum(mask)) < 3:
                    continue
                x = Z[mask, feature]
                y = margins[mask]
                denom = float(np.std(x) * np.std(y))
                correlations.append(
                    0.0 if denom <= 1e-12 else float(np.cov(x, y, ddof=0)[0, 1] / denom)
                )
            pooled.append(float(np.mean(np.abs(correlations))) if correlations else 0.0)
            variability.append(float(np.std(correlations)) if correlations else 1.0)
        score = np.asarray(pooled) - 0.5 * np.asarray(variability)
        order = np.argsort(-score, kind="stable")[:rank]
        axes = np.eye(Z.shape[1], dtype=float)[:, order]
        return self._orient_axes(Z, margins, axes), order.tolist(), score[order]

    def _boundary_axes(self, Z, margins, domains, weights, rank):
        weights = np.asarray(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        center = np.sum(Z * weights[:, None], axis=0)
        centered = Z - center
        total = centered.T @ (centered * weights[:, None])
        margin_centered = margins - float(np.sum(weights * margins))
        direction = centered.T @ (weights * margin_centered)
        relevance = np.outer(direction, direction)
        bins = self._margin_bins(margins)
        domains = np.asarray(domains, dtype=object)
        domain_scatter = np.zeros_like(total)
        for bin_index in sorted(set(bins.tolist())):
            bin_mask = bins == bin_index
            if int(np.sum(bin_mask)) < 2:
                continue
            pooled_mean = self._weighted_mean(
                Z[bin_mask], weights[bin_mask])
            for domain in sorted(set(domains.tolist())):
                mask = bin_mask & (domains == domain)
                if int(np.sum(mask)) < 2:
                    continue
                local_mean = self._weighted_mean(Z[mask], weights[mask])
                delta = local_mean - pooled_mean
                domain_scatter += float(np.sum(weights[mask])) * np.outer(
                    delta, delta)
        metric = 0.5 * (total + total.T) + self.ridge * np.eye(Z.shape[1])
        objective = 0.5 * (
            relevance + relevance.T
            - self.domain_penalty * (domain_scatter + domain_scatter.T)
        )
        values, vectors = np.linalg.eigh(metric)
        values = np.maximum(values, self.ridge)
        whitening = (vectors / np.sqrt(values)) @ vectors.T
        whitened = whitening @ objective @ whitening
        eigvals, eigvecs = np.linalg.eigh(0.5 * (whitened + whitened.T))
        order = np.argsort(-eigvals, kind="stable")
        axes = whitening @ eigvecs[:, order[:rank]]
        axes, _ = np.linalg.qr(axes)
        axes = self._orient_axes(Z, margins, axes[:, :rank])
        return axes, eigvals[order[:rank]], domain_scatter

    def _fit_coordinate(self, Z, margins, domains, weights):
        rank = min(self.rank, Z.shape[1])
        stable, selected, stable_score = self._stable_axes(
            Z, margins, domains, rank)
        _, _, vt = np.linalg.svd(Z * np.sqrt(weights[:, None]), full_matrices=False)
        pca = self._orient_axes(Z, margins, vt[:rank].T)
        boundary, eigenvalues, domain_scatter = self._boundary_axes(
            Z, margins, domains, weights, rank)
        if self.coordinate == "explicit_stable":
            axes = stable
        elif self.coordinate == "learned_psi":
            axes = pca
        elif self.coordinate == "boundary_latent":
            axes = boundary
        else:
            stable_count = max(1, rank // 2)
            parts = [stable[:, :stable_count]]
            residual = boundary - parts[0] @ (parts[0].T @ boundary)
            if residual.size:
                residual, _ = np.linalg.qr(residual)
                parts.append(residual[:, : max(0, rank - stable_count)])
            axes = np.column_stack(parts)[:, :rank]
            if axes.shape[1] < rank:
                axes = np.column_stack([axes, pca[:, : rank - axes.shape[1]]])
            axes, _ = np.linalg.qr(axes)
            axes = self._orient_axes(Z, margins, axes[:, :rank])
        coordinate = Z @ axes
        coordinate_mean = np.average(coordinate, axis=0, weights=weights)
        centered = coordinate - coordinate_mean
        coordinate_scale = np.sqrt(np.average(
            centered ** 2, axis=0, weights=weights))
        coordinate_scale = np.where(coordinate_scale < 1e-8, 1.0, coordinate_scale)
        self.coordinate_axes_ = axes
        self.coordinate_mean_ = coordinate_mean
        self.coordinate_scale_ = coordinate_scale
        return centered / coordinate_scale, {
            "rank": int(rank),
            "stable_descriptor_indices": selected,
            "stable_descriptor_scores": stable_score.tolist(),
            "boundary_eigenvalues": eigenvalues.tolist(),
            "conditional_domain_scatter_trace": float(np.trace(domain_scatter)),
        }

    def _weighted_ridge(self, X, y, weights, *, prior=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        weights = np.sqrt(np.maximum(np.asarray(weights, dtype=float), 1e-12))
        Xw = X * weights[:, None]
        yw = y * weights
        penalty = self.ridge * np.eye(X.shape[1])
        penalty[0, 0] = self.ridge * 1e-3
        rhs = Xw.T @ yw
        if prior is not None:
            rhs += penalty @ np.asarray(prior, dtype=float)
        try:
            return np.linalg.solve(Xw.T @ Xw + penalty, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(Xw.T @ Xw + penalty, rhs, rcond=None)[0]

    def _fit_geometry(self, coordinate, margins, weights):
        n, dim = coordinate.shape
        if self.geometry == "linear_monotone":
            centered = coordinate - np.average(coordinate, axis=0, weights=weights)
            y_center = margins - float(np.average(margins, weights=weights))
            sw = np.sqrt(np.maximum(weights, 1e-12))
            augmented = np.vstack([
                centered * sw[:, None],
                np.sqrt(self.ridge) * np.eye(dim),
            ])
            target = np.concatenate([y_center * sw, np.zeros(dim)])
            result = lsq_linear(
                augmented, target, bounds=(0.0, np.inf), method="trf")
            coefficient = result.x
            intercept = float(np.average(
                margins - coordinate @ coefficient, weights=weights))
            self.geometry_state_ = {
                "intercept": intercept,
                "linear": coefficient,
            }
            return
        if self.geometry == "diagonal_psd":
            X = np.column_stack([np.ones(n), coordinate, coordinate ** 2])
            sw = np.sqrt(np.maximum(weights, 1e-12))
            augmented = np.vstack([
                X * sw[:, None],
                np.sqrt(self.ridge) * np.eye(X.shape[1]),
            ])
            target = np.concatenate([margins * sw, np.zeros(X.shape[1])])
            lower = np.concatenate([
                np.full(1 + dim, -np.inf), np.zeros(dim)])
            result = lsq_linear(
                augmented,
                target,
                bounds=(lower, np.full(X.shape[1], np.inf)),
                method="trf",
            )
            self.geometry_state_ = {"coefficient": result.x}
            return
        if self.geometry == "low_rank_psd":
            pairs = [(i, j) for i in range(dim) for j in range(i, dim)]
            quadratic = np.column_stack([
                coordinate[:, i] ** 2
                if i == j
                else 2.0 * coordinate[:, i] * coordinate[:, j]
                for i, j in pairs
            ])
            X = np.column_stack([np.ones(n), coordinate, quadratic])
            beta = self._weighted_ridge(X, margins, weights)
            Q = np.zeros((dim, dim), dtype=float)
            for value, (i, j) in zip(beta[1 + dim:], pairs):
                Q[i, j] = Q[j, i] = float(value)
            values, vectors = np.linalg.eigh(0.5 * (Q + Q.T))
            order = np.argsort(-values, kind="stable")
            keep = order[: min(self.rank, dim)]
            positive = np.maximum(values[keep], 0.0)
            Q = (vectors[:, keep] * positive) @ vectors[:, keep].T
            q_value = np.einsum("ni,ij,nj->n", coordinate, Q, coordinate)
            linear = self._weighted_ridge(
                np.column_stack([np.ones(n), coordinate]),
                margins - q_value,
                weights,
            )
            self.geometry_state_ = {
                "intercept": float(linear[0]),
                "linear": linear[1:],
                "quadratic": Q,
                "quadratic_rank": int(np.sum(positive > 1e-10)),
            }
            return
        center_count = min(max(2 * dim, 4), min(16, n))
        centers = [int(np.argmin(np.abs(margins)))]
        distance = np.linalg.norm(
            coordinate - coordinate[centers[0]][None, :], axis=1)
        while len(centers) < center_count:
            index = int(np.argmax(distance))
            if index in centers:
                break
            centers.append(index)
            distance = np.minimum(
                distance,
                np.linalg.norm(
                    coordinate - coordinate[index][None, :], axis=1),
            )
        centers = coordinate[centers]
        pairwise = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2)
        positive = pairwise[pairwise > 1e-10]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
        radial = np.exp(-0.5 * (
            np.linalg.norm(
                coordinate[:, None, :] - centers[None, :, :], axis=2
            ) / max(bandwidth, 1e-8)
        ) ** 2)
        X = np.column_stack([np.ones(n), coordinate, radial])
        self.geometry_state_ = {
            "coefficient": self._weighted_ridge(X, margins, weights),
            "centers": centers,
            "bandwidth": bandwidth,
        }

    def fit(self, descriptors, margins, domains, sample_weight=None):
        descriptors = np.asarray(descriptors, dtype=float)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if descriptors.ndim != 2 or len(descriptors) != len(margins):
            raise ValueError("descriptors and margins must have matching rows")
        if len(domains) != len(margins) or len(set(domains.tolist())) < 2:
            raise ValueError("source fitting requires at least two domains")
        if sample_weight is None:
            sample_weight = np.ones(len(margins), dtype=float)
        self.descriptor_mean_ = np.mean(descriptors, axis=0)
        self.descriptor_scale_ = np.std(descriptors, axis=0)
        self.descriptor_scale_ = np.where(
            self.descriptor_scale_ < 1e-8, 1.0, self.descriptor_scale_)
        Z = (descriptors - self.descriptor_mean_) / self.descriptor_scale_
        weights = self._boundary_weights(margins, sample_weight)
        coordinate, coordinate_diag = self._fit_coordinate(
            Z, margins, domains, weights)
        self._fit_geometry(coordinate, margins, weights)
        self.source_prototypes_ = self._prototypes(coordinate, margins)
        fitted = self._predict_coordinate(coordinate)
        residual = margins - fitted
        self.residual_guard_ = float(np.quantile(np.abs(residual), 0.90))
        self.residual_scale_ = float(np.sqrt(np.average(
            residual ** 2,
            weights=weights,
        )))
        self.residual_degrees_of_freedom_ = float(max(
            len(margins) - coordinate.shape[1] - 1,
            1,
        ))
        self.fit_status = "fit"
        self.diagnostics_ = {
            "status": "fit",
            "coordinate": self.coordinate,
            "geometry": self.geometry,
            "adaptation": self.adaptation,
            "rank": int(coordinate.shape[1]),
            "n_source_records": int(len(margins)),
            "n_source_domains": int(len(set(domains.tolist()))),
            "source_domains": sorted(set(domains.tolist())),
            "source_boundary_rmse": float(np.sqrt(np.average(
                residual ** 2, weights=weights))),
            "source_false_safe_rate": float(np.mean(
                (fitted <= 0.0) & (margins > 0.0))),
            "source_residual_guard": self.residual_guard_,
            "source_residual_scale": self.residual_scale_,
            "source_residual_degrees_of_freedom": (
                self.residual_degrees_of_freedom_),
            "predictive_upper_alpha": self.upper_alpha,
            "calibration_prior_df": self.calibration_prior_df,
            "target_residual_rank": self.target_residual_rank,
            "coordinate_diagnostics": coordinate_diag,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        return self

    def _coordinate(self, descriptors):
        if self.fit_status != "fit":
            raise RuntimeError("boundary posterior must be fit before transform")
        descriptors = np.asarray(descriptors, dtype=float)
        Z = (descriptors - self.descriptor_mean_) / self.descriptor_scale_
        raw = Z @ self.coordinate_axes_
        return (raw - self.coordinate_mean_) / self.coordinate_scale_

    def _prototypes(self, coordinate, margins):
        bins = self._margin_bins(margins)
        prototypes = {}
        for index in sorted(set(bins.tolist())):
            mask = bins == index
            if np.any(mask):
                prototypes[int(index)] = np.mean(coordinate[mask], axis=0)
        return prototypes

    def _predict_coordinate(self, coordinate):
        coordinate = np.asarray(coordinate, dtype=float)
        state = self.geometry_state_
        if self.geometry == "linear_monotone":
            return state["intercept"] + coordinate @ state["linear"]
        if self.geometry == "diagonal_psd":
            X = np.column_stack([
                np.ones(len(coordinate)), coordinate, coordinate ** 2])
            return X @ state["coefficient"]
        if self.geometry == "low_rank_psd":
            return (
                state["intercept"]
                + coordinate @ state["linear"]
                + np.einsum(
                    "ni,ij,nj->n",
                    coordinate,
                    state["quadratic"],
                    coordinate,
                )
            )
        radial = np.exp(-0.5 * (
            np.linalg.norm(
                coordinate[:, None, :] - state["centers"][None, :, :],
                axis=2,
            ) / max(float(state["bandwidth"]), 1e-8)
        ) ** 2)
        X = np.column_stack([np.ones(len(coordinate)), coordinate, radial])
        return X @ state["coefficient"]

    def _output_calibration(self, predictions, margins, coordinate):
        predictions = np.asarray(predictions, dtype=float).reshape(-1)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        coordinate = np.asarray(coordinate, dtype=float)
        residual_rank = min(
            self.target_residual_rank,
            coordinate.shape[1],
        )
        residual_coordinate = coordinate[:, :residual_rank]
        X = np.column_stack([
            np.ones(len(predictions)),
            predictions,
            residual_coordinate,
        ])
        prior = np.concatenate([
            np.asarray([0.0, 1.0], dtype=float),
            np.zeros(residual_rank, dtype=float),
        ])
        penalty = self.adaptation_ridge * np.eye(X.shape[1])
        penalty[0, 0] *= 1e-3
        precision = X.T @ X + penalty
        try:
            inverse_precision = np.linalg.inv(precision)
            beta = inverse_precision @ (
                X.T @ margins + penalty @ prior)
        except np.linalg.LinAlgError:
            inverse_precision = np.linalg.pinv(precision)
            beta = inverse_precision @ (
                X.T @ margins + penalty @ prior)
        residual = margins - X @ beta
        prior_deviation = beta - prior
        prior_variance = max(self.residual_scale_ ** 2, 1e-10)
        numerator = (
            float(residual @ residual)
            + float(prior_deviation @ penalty @ prior_deviation)
            + self.calibration_prior_df * prior_variance
        )
        degrees_of_freedom = float(
            len(margins) + self.calibration_prior_df)
        residual_variance = max(
            numerator / max(degrees_of_freedom, 1.0),
            1e-10,
        )
        coefficient_covariance = (
            residual_variance * inverse_precision)
        coefficient_covariance = 0.5 * (
            coefficient_covariance + coefficient_covariance.T)
        return (
            float(beta[0]),
            float(beta[1]),
            np.asarray(beta[2:], dtype=float),
            residual_rank,
            coefficient_covariance,
            float(np.sqrt(residual_variance)),
            degrees_of_freedom,
        )

    def fit_target_adapter(self, pilot_descriptors, pilot_margins):
        coordinate = self._coordinate(pilot_descriptors)
        margins = np.asarray(pilot_margins, dtype=float).reshape(-1)
        dim = coordinate.shape[1]
        identity = np.eye(dim, dtype=float)
        if self.adaptation == "frozen" or len(coordinate) == 0:
            return TargetBoundaryAdapter(
                np.zeros(dim),
                np.ones(dim),
                identity,
                0.0,
                1.0,
                np.zeros(0, dtype=float),
                0,
                np.zeros((2, 2), dtype=float),
                self.residual_scale_,
                self.residual_degrees_of_freedom_,
                0,
                "frozen",
                {
                    "status": "frozen",
                    "pilot_records": int(len(coordinate)),
                    "target_oracle_used": False,
                    "predictive_upper_alpha": self.upper_alpha,
                    "posterior_residual_scale": self.residual_scale_,
                },
            )
        shift = np.mean(coordinate, axis=0)
        scale = np.std(coordinate, axis=0)
        scale = np.where(scale < 0.25, 1.0, scale)
        standardized = (coordinate - shift) / scale
        rotation = identity
        status = "shift_scale"
        matched_bins = []
        if self.adaptation == "orthogonal_shift":
            target_prototypes = self._prototypes(standardized, margins)
            matched_bins = sorted(
                set(target_prototypes) & set(self.source_prototypes_))
            if len(matched_bins) >= 2:
                target = np.vstack([
                    target_prototypes[index] for index in matched_bins])
                source = np.vstack([
                    self.source_prototypes_[index] for index in matched_bins])
                target -= np.mean(target, axis=0)
                source -= np.mean(source, axis=0)
                cross = target.T @ source + self.adaptation_ridge * identity
                u, _, vt = np.linalg.svd(cross, full_matrices=False)
                rotation = u @ vt
                status = "orthogonal_shift"
            else:
                status = "fallback_shift_scale"
        adapted = standardized @ rotation
        raw_prediction = self._predict_coordinate(adapted)
        (
            output_offset,
            output_scale,
            output_coordinate_linear,
            calibration_coordinate_count,
            output_covariance,
            residual_scale,
            residual_degrees_of_freedom,
        ) = self._output_calibration(
            raw_prediction,
            margins,
            adapted,
        )
        effective_dimension = 2 + calibration_coordinate_count
        if status == "orthogonal_shift":
            effective_dimension += dim * (dim - 1) // 2
        return TargetBoundaryAdapter(
            shift,
            scale,
            rotation,
            output_offset,
            output_scale,
            output_coordinate_linear,
            calibration_coordinate_count,
            output_covariance,
            residual_scale,
            residual_degrees_of_freedom,
            effective_dimension,
            status,
            {
                "status": status,
                "pilot_records": int(len(coordinate)),
                "matched_boundary_bins": matched_bins,
                "effective_label_adaptation_dimension": int(
                    effective_dimension),
                "rotation_deviation": float(np.linalg.norm(
                    rotation - identity)),
                "output_offset": output_offset,
                "output_scale": output_scale,
                "output_coordinate_linear": (
                    output_coordinate_linear.tolist()),
                "calibration_coordinate_count": int(
                    calibration_coordinate_count),
                "output_covariance_eigenvalues": np.linalg.eigvalsh(
                    output_covariance).tolist(),
                "posterior_residual_scale": residual_scale,
                "posterior_residual_degrees_of_freedom": (
                    residual_degrees_of_freedom),
                "predictive_upper_alpha": self.upper_alpha,
                "target_data_used": True,
                "target_oracle_used": False,
            },
        )

    def predict(self, descriptors, adapter=None):
        coordinate = self._coordinate(descriptors)
        if adapter is None:
            adapted = coordinate
            offset = 0.0
            scale = 1.0
        else:
            adapted = (
                (coordinate - adapter.shift) / adapter.scale
            ) @ adapter.rotation
            offset = float(adapter.output_offset)
            scale = float(adapter.output_scale)
        raw_prediction = self._predict_coordinate(adapted)
        if adapter is None or adapter.calibration_coordinate_count <= 0:
            residual_correction = 0.0
        else:
            residual_correction = (
                adapted[:, :adapter.calibration_coordinate_count]
                @ np.asarray(adapter.output_coordinate_linear, dtype=float)
            )
        return offset + scale * raw_prediction + residual_correction

    def predict_upper(self, descriptors, adapter=None):
        coordinate = self._coordinate(descriptors)
        if adapter is None:
            raw_prediction = self._predict_coordinate(coordinate)
            coefficient_covariance = np.zeros((2, 2), dtype=float)
            residual_scale = self.residual_scale_
            degrees_of_freedom = self.residual_degrees_of_freedom_
            effective_dimension = 0
            mean = raw_prediction
        else:
            adapted = (
                (coordinate - adapter.shift) / adapter.scale
            ) @ adapter.rotation
            raw_prediction = self._predict_coordinate(adapted)
            coefficient_covariance = np.asarray(
                adapter.output_covariance, dtype=float)
            residual_scale = float(adapter.residual_scale)
            degrees_of_freedom = float(
                adapter.residual_degrees_of_freedom)
            effective_dimension = int(adapter.effective_dimension)
            pilot_records = max(int(
                adapter.diagnostics.get("pilot_records", 0)), 1)
            mean = (
                float(adapter.output_offset)
                + float(adapter.output_scale) * raw_prediction
                + adapted[:, :adapter.calibration_coordinate_count]
                @ np.asarray(adapter.output_coordinate_linear, dtype=float)
            )
        design = np.column_stack([
            np.ones(len(raw_prediction), dtype=float),
            raw_prediction,
            (
                np.empty((len(raw_prediction), 0), dtype=float)
                if adapter is None
                else adapted[:, :adapter.calibration_coordinate_count]
            ),
        ])
        parameter_variance = np.einsum(
            "ni,ij,nj->n",
            design,
            coefficient_covariance,
            design,
        )
        coordinate_factor = (
            1.0
            if adapter is None
            else 1.0 + effective_dimension / pilot_records
        )
        predictive_variance = np.maximum(
            residual_scale ** 2 * coordinate_factor + parameter_variance,
            1e-12,
        )
        quantile = float(student_t.ppf(
            1.0 - self.upper_alpha,
            max(degrees_of_freedom, 1.0),
        ))
        return mean + quantile * np.sqrt(predictive_variance)

    def diagnostics(self):
        return dict(self.diagnostics_)


class HierarchicalSignedDistancePosterior(
    TransferableChanceBoundaryPosterior,
):
    """TCB-V2 with shared shape and positive domain-specific scale.

    Source domains identify a canonical signed-distance shape.  Each domain
    receives only a location and positive scale random effect.  A held-out
    target updates those two effects using a replicate-aware Gaussian
    likelihood; its covariance contributes only positive predictive variance.
    """

    def __init__(
        self,
        *,
        coordinate="boundary_latent",
        geometry="low_rank_psd",
        rank=2,
        ridge=1e-3,
        domain_penalty=0.5,
        boundary_temperature=1.0,
        adaptation_ridge=5.0,
        upper_alpha=0.01,
        calibration_prior_df=2.0,
        hierarchy_iterations=5,
        effect_ridge=1.0,
        rotation_mode="none",
        rotation_ridge=5.0,
        target_residual_rank=0,
        residual_ridge=5.0,
        minimum_scale_fraction=0.05,
        maximum_scale_multiple=20.0,
        variance_floor=1e-8,
        allow_single_source_domain=False,
    ):
        super().__init__(
            coordinate=coordinate,
            geometry=geometry,
            adaptation="shift_scale",
            rank=rank,
            ridge=ridge,
            domain_penalty=domain_penalty,
            boundary_temperature=boundary_temperature,
            adaptation_ridge=adaptation_ridge,
            upper_alpha=upper_alpha,
            calibration_prior_df=calibration_prior_df,
            target_residual_rank=target_residual_rank,
        )
        self.hierarchy_iterations = max(1, int(hierarchy_iterations))
        self.effect_ridge = max(float(effect_ridge), 0.0)
        self.rotation_mode = str(rotation_mode).lower()
        if self.rotation_mode not in {"none", "planar"}:
            raise ValueError("hierarchical rotation_mode must be 'none' or 'planar'")
        self.rotation_ridge = max(float(rotation_ridge), 1e-8)
        self.residual_ridge = max(float(residual_ridge), 1e-8)
        self.minimum_scale_fraction = max(
            float(minimum_scale_fraction), 1e-5)
        self.maximum_scale_multiple = max(
            float(maximum_scale_multiple), 1.0)
        self.variance_floor = max(float(variance_floor), 1e-12)
        self.allow_single_source_domain = bool(allow_single_source_domain)

    @staticmethod
    def _weighted_scalar_mean(values, weights):
        values = np.asarray(values, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        total = max(float(np.sum(weights)), 1e-12)
        return float(np.sum(values * weights) / total)

    def _effective_variance(self, variance, replicate_count, size):
        if variance is None:
            variance = np.full(size, self.variance_floor, dtype=float)
        variance = np.maximum(
            np.asarray(variance, dtype=float).reshape(-1),
            self.variance_floor,
        )
        if len(variance) != int(size):
            raise ValueError("margin variance must match margin rows")
        if replicate_count is None:
            replicate_count = np.ones(size, dtype=float)
        replicate_count = np.maximum(
            np.asarray(replicate_count, dtype=float).reshape(-1), 1.0)
        if len(replicate_count) != int(size):
            raise ValueError("replicate count must match margin rows")
        return np.maximum(
            variance / replicate_count,
            self.variance_floor,
        )

    def _shape_from_coordinate(self, coordinate):
        raw = np.asarray(
            self._predict_coordinate(coordinate), dtype=float).reshape(-1)
        return (
            raw - float(self.shared_prediction_mean_)
        ) / max(float(self.shared_prediction_scale_), 1e-12)

    @staticmethod
    def _planar_rotation(dimension, angle):
        rotation = np.eye(int(dimension), dtype=float)
        if int(dimension) >= 2:
            cosine = float(np.cos(angle))
            sine = float(np.sin(angle))
            rotation[:2, :2] = np.asarray([
                [cosine, -sine],
                [sine, cosine],
            ])
        return rotation

    def _rotated_shape(self, coordinate, angle):
        coordinate = np.asarray(coordinate, dtype=float)
        rotation = self._planar_rotation(coordinate.shape[1], angle)
        return self._shape_from_coordinate(coordinate @ rotation)

    def _rotated_shape_derivative(self, coordinate, angle, step=1e-5):
        step = max(float(step), 1e-8)
        return (
            self._rotated_shape(coordinate, angle + step)
            - self._rotated_shape(coordinate, angle - step)
        ) / (2.0 * step)

    def _set_shape_normalization(self, coordinate, weights):
        raw = np.asarray(
            self._predict_coordinate(coordinate), dtype=float).reshape(-1)
        mean = self._weighted_scalar_mean(raw, weights)
        centered = raw - mean
        scale = np.sqrt(max(
            self._weighted_scalar_mean(centered ** 2, weights),
            1e-12,
        ))
        self.shared_prediction_mean_ = float(mean)
        self.shared_prediction_scale_ = float(max(scale, 1e-8))
        return self._shape_from_coordinate(coordinate)

    def _fit_domain_effect(
        self,
        shape,
        margins,
        weights,
        prior_location,
        prior_scale,
        scale_reference,
    ):
        shape = np.asarray(shape, dtype=float).reshape(-1)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        sqrt_weight = np.sqrt(np.maximum(
            np.asarray(weights, dtype=float).reshape(-1), 1e-12))
        design = np.column_stack([
            np.ones(len(shape), dtype=float), shape])
        augmented = design * sqrt_weight[:, None]
        target = margins * sqrt_weight
        if self.effect_ridge > 0.0:
            prior_design = np.sqrt(self.effect_ridge) * np.eye(2)
            augmented = np.vstack([augmented, prior_design])
            target = np.concatenate([
                target,
                np.sqrt(self.effect_ridge) * np.asarray([
                    prior_location, prior_scale], dtype=float),
            ])
        lower_scale = max(
            self.minimum_scale_fraction * scale_reference, 1e-8)
        upper_scale = max(
            self.maximum_scale_multiple * scale_reference,
            lower_scale * 2.0,
        )
        result = lsq_linear(
            augmented,
            target,
            bounds=(
                np.asarray([-np.inf, lower_scale], dtype=float),
                np.asarray([np.inf, upper_scale], dtype=float),
            ),
            method="trf",
        )
        return float(result.x[0]), float(result.x[1])

    def _fit_target_residual_map(self, coordinate, shape, weights):
        coordinate = np.asarray(coordinate, dtype=float)
        shape = np.asarray(shape, dtype=float).reshape(-1)
        rank = min(max(int(self.target_residual_rank), 0), coordinate.shape[1])
        self.target_residual_rank_ = int(rank)
        if rank <= 0:
            self.residual_base_projection_ = np.zeros(
                (2, coordinate.shape[1]), dtype=float)
            self.residual_coordinate_axes_ = np.empty(
                (coordinate.shape[1], 0), dtype=float)
            self.residual_coordinate_scale_ = np.empty(0, dtype=float)
            return np.empty((len(coordinate), 0), dtype=float)
        base = np.column_stack([
            np.ones(len(shape), dtype=float), shape])
        sqrt_weight = np.sqrt(np.maximum(
            np.asarray(weights, dtype=float).reshape(-1), 1e-12))
        weighted_base = base * sqrt_weight[:, None]
        weighted_coordinate = coordinate * sqrt_weight[:, None]
        penalty = self.ridge * np.eye(base.shape[1], dtype=float)
        projection = np.linalg.pinv(
            weighted_base.T @ weighted_base + penalty
        ) @ (weighted_base.T @ weighted_coordinate)
        orthogonal = coordinate - base @ projection
        try:
            _, _, vectors = np.linalg.svd(
                orthogonal * sqrt_weight[:, None], full_matrices=False)
            axes = vectors[:rank].T
        except np.linalg.LinAlgError:
            axes = np.eye(coordinate.shape[1], dtype=float)[:, :rank]
        features = orthogonal @ axes
        scale = np.sqrt(np.average(
            features ** 2, axis=0, weights=np.asarray(weights, dtype=float)))
        scale = np.where(scale < 1e-8, 1.0, scale)
        self.residual_base_projection_ = projection
        self.residual_coordinate_axes_ = axes
        self.residual_coordinate_scale_ = scale
        return features / scale

    def _target_residual_features(self, coordinate, shape):
        coordinate = np.asarray(coordinate, dtype=float)
        rank = int(getattr(self, "target_residual_rank_", 0))
        if rank <= 0:
            return np.empty((len(coordinate), 0), dtype=float)
        base = np.column_stack([
            np.ones(len(coordinate), dtype=float),
            np.asarray(shape, dtype=float).reshape(-1),
        ])
        orthogonal = coordinate - base @ self.residual_base_projection_
        return (
            orthogonal @ self.residual_coordinate_axes_
        ) / self.residual_coordinate_scale_

    def _fit_domain_effect_with_residual(
        self,
        shape,
        residual_features,
        margins,
        weights,
        prior_location,
        prior_scale,
        scale_reference,
    ):
        shape = np.asarray(shape, dtype=float).reshape(-1)
        residual_features = np.asarray(residual_features, dtype=float)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        design = np.column_stack([
            np.ones(len(shape), dtype=float),
            shape,
            residual_features,
        ])
        sqrt_weight = np.sqrt(np.maximum(
            np.asarray(weights, dtype=float).reshape(-1), 1e-12))
        augmented = design * sqrt_weight[:, None]
        target = margins * sqrt_weight
        prior = np.concatenate([
            [prior_location, prior_scale],
            np.zeros(residual_features.shape[1], dtype=float),
        ])
        penalty = np.diag(np.concatenate([
            np.full(2, self.effect_ridge, dtype=float),
            np.full(residual_features.shape[1], self.residual_ridge, dtype=float),
        ]))
        if np.any(np.diag(penalty) > 0.0):
            root = np.sqrt(penalty)
            augmented = np.vstack([augmented, root])
            target = np.concatenate([target, root @ prior])
        lower_scale = max(
            self.minimum_scale_fraction * scale_reference, 1e-8)
        upper_scale = max(
            self.maximum_scale_multiple * scale_reference,
            lower_scale * 2.0,
        )
        lower = np.concatenate([
            [-np.inf, lower_scale],
            np.full(residual_features.shape[1], -np.inf),
        ])
        upper = np.concatenate([
            [np.inf, upper_scale],
            np.full(residual_features.shape[1], np.inf),
        ])
        result = lsq_linear(
            augmented, target, bounds=(lower, upper), method="trf")
        return (
            float(result.x[0]),
            float(result.x[1]),
            np.asarray(result.x[2:], dtype=float),
        )

    def _fit_domain_rotation_effect(
        self,
        coordinate,
        margins,
        weights,
        prior_location,
        prior_scale,
        prior_angle,
        scale_reference,
    ):
        coordinate = np.asarray(coordinate, dtype=float)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        weights = np.maximum(
            np.asarray(weights, dtype=float).reshape(-1), 1e-12)
        lower_scale = max(
            self.minimum_scale_fraction * scale_reference, 1e-8)
        upper_scale = max(
            self.maximum_scale_multiple * scale_reference,
            lower_scale * 2.0,
        )
        prior = np.asarray([
            prior_location,
            np.log(max(prior_scale, 1e-12)),
            prior_angle,
        ], dtype=float)

        def objective(theta):
            location = float(theta[0])
            scale = float(np.exp(theta[1]))
            angle = float(theta[2])
            local_shape = self._rotated_shape(coordinate, angle)
            residual = margins - location - scale * local_shape
            delta = theta - prior
            penalty = (
                self.effect_ridge * (delta[0] ** 2 + delta[1] ** 2)
                + self.rotation_ridge * delta[2] ** 2
            )
            return float(0.5 * np.sum(weights * residual ** 2) + 0.5 * penalty)

        result = minimize(
            objective,
            prior,
            method="L-BFGS-B",
            bounds=[
                (None, None),
                (np.log(lower_scale), np.log(upper_scale)),
                (-np.pi, np.pi),
            ],
        )
        theta = np.asarray(
            result.x if result.success else prior, dtype=float)
        angle = float(theta[2])
        return (
            float(theta[0]),
            float(np.exp(theta[1])),
            angle,
            self._rotated_shape(coordinate, angle),
        )

    def _align_source_coordinate(self, coordinate, domains, angles):
        aligned = np.empty_like(np.asarray(coordinate, dtype=float))
        for domain, angle in angles.items():
            mask = np.asarray(domains, dtype=object) == domain
            rotation = self._planar_rotation(aligned.shape[1], angle)
            aligned[mask] = np.asarray(coordinate, dtype=float)[mask] @ rotation
        return aligned

    def fit(
        self,
        descriptors,
        margins,
        domains,
        sample_weight=None,
        *,
        margin_variance=None,
        replicate_count=None,
    ):
        descriptors = np.asarray(descriptors, dtype=float)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if descriptors.ndim != 2 or len(descriptors) != len(margins):
            raise ValueError("descriptors and margins must have matching rows")
        source_domain_count = len(set(domains.tolist()))
        minimum_domains = 1 if self.allow_single_source_domain else 2
        if len(domains) != len(margins) or source_domain_count < minimum_domains:
            raise ValueError(
                f"hierarchical source fit needs at least {minimum_domains} "
                "source domain(s)")
        if sample_weight is None:
            sample_weight = np.ones(len(margins), dtype=float)
        sample_weight = np.maximum(
            np.asarray(sample_weight, dtype=float).reshape(-1), 1e-12)
        effective_variance = self._effective_variance(
            margin_variance, replicate_count, len(margins))
        precision_weight = sample_weight / effective_variance
        precision_weight /= max(float(np.mean(precision_weight)), 1e-12)
        weights = self._boundary_weights(margins, precision_weight)

        self.descriptor_mean_ = np.mean(descriptors, axis=0)
        self.descriptor_scale_ = np.std(descriptors, axis=0)
        self.descriptor_scale_ = np.where(
            self.descriptor_scale_ < 1e-8, 1.0, self.descriptor_scale_)
        standardized = (
            descriptors - self.descriptor_mean_) / self.descriptor_scale_
        global_location = self._weighted_scalar_mean(margins, weights)
        global_scale = np.sqrt(max(self._weighted_scalar_mean(
            (margins - global_location) ** 2, weights), 1e-12))
        global_scale = max(float(global_scale), 1e-4)
        source_domains = sorted(set(domains.tolist()))
        # Initialize in domain-standardized signed-distance units.  Learning the
        # coordinate from raw margins would let source-specific offsets/scales
        # contaminate the supposedly transferable shape before the hierarchy
        # has a chance to remove them.
        canonical = np.empty(len(margins), dtype=float)
        for domain in source_domains:
            mask = domains == domain
            local_location = self._weighted_scalar_mean(
                margins[mask], weights[mask])
            local_scale = np.sqrt(max(self._weighted_scalar_mean(
                (margins[mask] - local_location) ** 2,
                weights[mask],
            ), 1e-12))
            canonical[mask] = (
                margins[mask] - local_location
            ) / max(float(local_scale), 1e-4)

        effects = {}
        source_rotation_active = bool(
            self.rotation_mode == "planar"
            and self.rank >= 2
            and len(source_domains) >= 2)
        domain_angles = {str(domain): 0.0 for domain in source_domains}
        coordinate_iterations = []
        for iteration in range(self.hierarchy_iterations):
            coordinate, coordinate_diag = self._fit_coordinate(
                standardized, canonical, domains, weights)
            aligned_coordinate = (
                self._align_source_coordinate(
                    coordinate, domains, domain_angles)
                if source_rotation_active else coordinate
            )
            self._fit_geometry(aligned_coordinate, canonical, weights)
            shape = self._set_shape_normalization(
                aligned_coordinate, weights)
            effects = {}
            next_canonical = np.empty(len(margins), dtype=float)
            for domain in source_domains:
                mask = domains == domain
                if source_rotation_active:
                    location, scale, angle, local_shape = (
                        self._fit_domain_rotation_effect(
                            coordinate[mask],
                            margins[mask],
                            weights[mask],
                            global_location,
                            global_scale,
                            domain_angles[str(domain)],
                            global_scale,
                        )
                    )
                    domain_angles[str(domain)] = angle
                else:
                    location, scale = self._fit_domain_effect(
                        shape[mask],
                        margins[mask],
                        weights[mask],
                        global_location,
                        global_scale,
                        global_scale,
                    )
                    angle = 0.0
                    local_shape = shape[mask]
                effects[str(domain)] = {
                    "location": location,
                    "scale": scale,
                    "rotation_angle": angle,
                }
                next_canonical[mask] = (
                    margins[mask] - location) / max(scale, 1e-12)
            canonical = next_canonical
            coordinate_iterations.append({
                "iteration": int(iteration),
                "canonical_mean": float(np.average(
                    canonical, weights=weights)),
                "canonical_scale": float(np.sqrt(np.average(
                    (canonical - np.average(
                        canonical, weights=weights)) ** 2,
                    weights=weights,
                ))),
                "coordinate": coordinate_diag,
                "source_rotation_angles": dict(domain_angles),
            })

        coordinate, coordinate_diag = self._fit_coordinate(
            standardized, canonical, domains, weights)
        aligned_coordinate = (
            self._align_source_coordinate(coordinate, domains, domain_angles)
            if source_rotation_active else coordinate
        )
        self._fit_geometry(aligned_coordinate, canonical, weights)
        self._set_shape_normalization(aligned_coordinate, weights)
        if source_rotation_active:
            for domain in source_domains:
                mask = domains == domain
                _, _, angle, _ = self._fit_domain_rotation_effect(
                    coordinate[mask],
                    margins[mask],
                    weights[mask],
                    global_location,
                    global_scale,
                    domain_angles[str(domain)],
                    global_scale,
                )
                domain_angles[str(domain)] = angle
            aligned_coordinate = self._align_source_coordinate(
                coordinate, domains, domain_angles)
            self._fit_geometry(aligned_coordinate, canonical, weights)
        shape = self._set_shape_normalization(aligned_coordinate, weights)
        residual_features = self._fit_target_residual_map(
            aligned_coordinate, shape, weights)
        fitted = np.empty(len(margins), dtype=float)
        effect_rows = []
        effects = {}
        for domain in source_domains:
            mask = domains == domain
            if self.target_residual_rank_ > 0:
                location, scale, gamma = (
                    self._fit_domain_effect_with_residual(
                        shape[mask],
                        residual_features[mask],
                        margins[mask],
                        weights[mask],
                        global_location,
                        global_scale,
                        global_scale,
                    )
                )
            else:
                location, scale = self._fit_domain_effect(
                    shape[mask],
                    margins[mask],
                    weights[mask],
                    global_location,
                    global_scale,
                    global_scale,
                )
                gamma = np.empty(0, dtype=float)
            fitted[mask] = (
                location
                + scale * shape[mask]
                + residual_features[mask] @ gamma
            )
            effects[str(domain)] = {
                "location": location,
                "scale": scale,
                "residual_coefficient": gamma.tolist(),
                "rotation_angle": float(domain_angles[str(domain)]),
            }
            effect_rows.append([
                location,
                np.log(max(scale, 1e-12)),
                *gamma.tolist(),
            ])

        effect_rows = np.asarray(effect_rows, dtype=float)
        effect_mean = np.mean(effect_rows, axis=0)
        centered_effects = effect_rows - effect_mean[None, :]
        denominator = max(len(effect_rows) - 1, 1)
        effect_covariance = (
            centered_effects.T @ centered_effects / denominator)
        effect_covariance += np.diag(np.concatenate([
            np.asarray([
                max(0.05 * global_scale, 1e-4) ** 2,
                0.05 ** 2,
            ]),
            np.full(self.target_residual_rank_, 0.05 ** 2, dtype=float),
        ]))
        effect_covariance = 0.5 * (
            effect_covariance + effect_covariance.T)
        self.effect_prior_mean_ = np.asarray(effect_mean, dtype=float)
        self.effect_prior_covariance_ = np.asarray(
            effect_covariance, dtype=float)
        self.effect_prior_precision_ = np.linalg.pinv(
            self.effect_prior_covariance_)
        self.source_effects_ = effects
        source_angles = np.asarray([
            domain_angles[str(domain)] for domain in source_domains
        ], dtype=float)
        self.rotation_prior_mean_ = float(np.mean(source_angles))
        self.rotation_prior_variance_ = float(max(
            np.var(source_angles, ddof=1) if len(source_angles) > 1 else 0.0,
            1.0 / self.rotation_ridge,
        ))

        residual = margins - fitted
        intrinsic_variance = max(
            self._weighted_scalar_mean(
                np.maximum(residual ** 2 - effective_variance, 0.0),
                weights,
            ),
            self.variance_floor,
        )
        self.residual_scale_ = float(np.sqrt(intrinsic_variance))
        self.residual_guard_ = float(np.quantile(
            np.abs(residual), 0.90))
        self.residual_degrees_of_freedom_ = float(max(
            len(margins)
            - (2 + self.target_residual_rank_ + int(source_rotation_active))
            * len(source_domains)
            - self.rank,
            1,
        ))
        self.fit_status = "fit"
        self.diagnostics_ = {
            "status": "fit",
            "model_version": "tcb_v2_hierarchical_signed_distance",
            "coordinate": self.coordinate,
            "geometry": self.geometry,
            "rank": int(coordinate.shape[1]),
            "n_source_records": int(len(margins)),
            "n_source_domains": int(len(source_domains)),
            "source_domains": source_domains,
            "source_effects": effects,
            "effect_prior_mean": self.effect_prior_mean_.tolist(),
            "effect_prior_covariance": (
                self.effect_prior_covariance_.tolist()),
            "source_boundary_rmse": float(np.sqrt(
                self._weighted_scalar_mean(residual ** 2, weights))),
            "source_false_safe_rate": float(np.mean(
                (fitted <= 0.0) & (margins > 0.0))),
            "source_residual_guard": self.residual_guard_,
            "source_residual_scale": self.residual_scale_,
            "source_residual_degrees_of_freedom": (
                self.residual_degrees_of_freedom_),
            "hierarchy_iterations": int(self.hierarchy_iterations),
            "effective_target_adaptation_dimension": int(
                2
                + self.target_residual_rank_
                + int(self.rotation_mode == "planar" and coordinate.shape[1] >= 2)),
            "rotation_mode": self.rotation_mode,
            "rotation_ridge": float(self.rotation_ridge),
            "source_rotation_alignment": source_rotation_active,
            "source_rotation_angles": dict(domain_angles),
            "rotation_prior_mean": self.rotation_prior_mean_,
            "rotation_prior_variance": self.rotation_prior_variance_,
            "target_residual_rank": int(self.target_residual_rank_),
            "target_residual_ridge": float(self.residual_ridge),
            "orthogonal_target_residual": bool(
                self.target_residual_rank_ > 0),
            "positive_target_scale": True,
            "replicate_aware_likelihood": True,
            "predictive_upper_alpha": self.upper_alpha,
            "coordinate_diagnostics": coordinate_diag,
            "coordinate_refit_each_hierarchy_iteration": True,
            "coordinate_iteration_diagnostics": coordinate_iterations,
            "target_data_used": False,
            "target_oracle_used": False,
            "single_source_domain_family": bool(
                self.allow_single_source_domain and len(source_domains) == 1),
        }
        return self

    def _shape(self, descriptors, adapter=None):
        coordinate = self._coordinate(descriptors)
        if adapter is None:
            return self._shape_from_coordinate(coordinate)
        return self._shape_from_coordinate(
            coordinate @ np.asarray(adapter.rotation, dtype=float))

    def prior_adapter(self):
        dimension = int(self.coordinate_axes_.shape[1])
        location = float(self.effect_prior_mean_[0])
        scale = float(np.exp(self.effect_prior_mean_[1]))
        planar = self.rotation_mode == "planar" and dimension >= 2
        residual_rank = int(getattr(self, "target_residual_rank_", 0))
        advanced = planar or residual_rank > 0
        if advanced:
            effective_dimension = 2 + residual_rank + int(planar)
            covariance = np.zeros(
                (effective_dimension, effective_dimension), dtype=float)
            covariance[:2 + residual_rank, :2 + residual_rank] = (
                self.effect_prior_covariance_)
            if planar:
                covariance[-1, -1] = float(
                    self.rotation_prior_variance_)
            mode = (
                "hierarchical_planar_residual_prior"
                if planar and residual_rank > 0
                else (
                    "hierarchical_planar_prior"
                    if planar else "hierarchical_residual_prior"
                )
            )
        else:
            transform = np.diag([1.0, scale])
            covariance = (
                transform
                @ self.effect_prior_covariance_
                @ transform.T
            )
            effective_dimension = 2
            mode = "hierarchical_prior"
        return TargetBoundaryAdapter(
            np.zeros(dimension, dtype=float),
            np.ones(dimension, dtype=float),
            self._planar_rotation(
                dimension,
                self.rotation_prior_mean_ if planar else 0.0,
            ),
            location,
            scale,
            np.asarray(
                self.effect_prior_mean_[2:2 + residual_rank], dtype=float),
            residual_rank,
            covariance,
            self.residual_scale_,
            self.residual_degrees_of_freedom_,
            effective_dimension,
            mode,
            {
                "status": mode,
                "pilot_records": 0,
                "effective_label_adaptation_dimension": effective_dimension,
                "positive_output_scale": True,
                "rotation_mode": self.rotation_mode,
                "rotation_angle": float(
                    self.rotation_prior_mean_ if planar else 0.0),
                "target_data_used": False,
                "target_oracle_used": False,
            },
        )

    def fit_target_adapter(
        self,
        pilot_descriptors,
        pilot_margins,
        *,
        pilot_variance=None,
        replicate_count=None,
    ):
        coordinate = self._coordinate(pilot_descriptors)
        planar = self.rotation_mode == "planar" and coordinate.shape[1] >= 2
        shape = self._shape_from_coordinate(coordinate)
        margins = np.asarray(pilot_margins, dtype=float).reshape(-1)
        if len(shape) != len(margins):
            raise ValueError("pilot descriptors and margins must match")
        if len(margins) == 0:
            return self.prior_adapter()
        observation_variance = self._effective_variance(
            pilot_variance, replicate_count, len(margins))
        total_variance = np.maximum(
            observation_variance + self.residual_scale_ ** 2,
            self.variance_floor,
        )
        prior_mean = self.effect_prior_mean_
        prior_precision = self.effect_prior_precision_

        residual_rank = int(getattr(self, "target_residual_rank_", 0))
        advanced = planar or residual_rank > 0
        if advanced:
            advanced_prior_mean = np.asarray(prior_mean, dtype=float)
            advanced_prior_precision = np.asarray(
                prior_precision, dtype=float)
            if planar:
                advanced_prior_mean = np.concatenate([
                    advanced_prior_mean, [self.rotation_prior_mean_]])
                expanded = np.zeros((
                    len(advanced_prior_mean), len(advanced_prior_mean)),
                    dtype=float,
                )
                expanded[:-1, :-1] = advanced_prior_precision
                expanded[-1, -1] = (
                    1.0 / max(self.rotation_prior_variance_, 1e-12))
                advanced_prior_precision = expanded
            gamma_slice = slice(2, 2 + residual_rank)
            angle_index = 2 + residual_rank if planar else None

            def advanced_components(theta):
                location = float(theta[0])
                scale = float(np.exp(theta[1]))
                angle = float(theta[angle_index]) if planar else 0.0
                rotation = self._planar_rotation(
                    coordinate.shape[1], angle)
                adapted_coordinate = coordinate @ rotation
                local_shape = self._shape_from_coordinate(
                    adapted_coordinate)
                local_residual_features = self._target_residual_features(
                    adapted_coordinate, local_shape)
                gamma = np.asarray(theta[gamma_slice], dtype=float)
                mean = (
                    location
                    + scale * local_shape
                    + local_residual_features @ gamma
                )
                return (
                    mean,
                    local_shape,
                    local_residual_features,
                    rotation,
                    angle,
                    gamma,
                )

            def advanced_objective(theta):
                mean = advanced_components(theta)[0]
                residual = margins - mean
                delta = theta - advanced_prior_mean
                return float(
                    0.5 * np.sum(residual ** 2 / total_variance)
                    + 0.5 * delta @ advanced_prior_precision @ delta
                )

            bounds = [(None, None), (-8.0, 8.0)]
            bounds.extend([(None, None)] * residual_rank)
            if planar:
                bounds.append((-np.pi, np.pi))
            result = minimize(
                advanced_objective,
                advanced_prior_mean,
                method="L-BFGS-B",
                bounds=bounds,
            )
            theta = np.asarray(
                result.x if result.success else advanced_prior_mean,
                dtype=float,
            )
            (
                fitted_mean,
                shape,
                residual_features,
                rotation,
                angle,
                gamma,
            ) = advanced_components(theta)
            location = float(theta[0])
            scale = float(np.exp(theta[1]))
            design_columns = [
                np.ones(len(shape), dtype=float),
                scale * shape,
            ]
            design_columns.extend(
                residual_features[:, index]
                for index in range(residual_rank))
            if planar:
                step = 1e-5
                plus = theta.copy()
                minus = theta.copy()
                plus[angle_index] += step
                minus[angle_index] -= step
                angle_derivative = (
                    advanced_components(plus)[0]
                    - advanced_components(minus)[0]
                ) / (2.0 * step)
                design_columns.append(angle_derivative)
            design_theta = np.column_stack(design_columns)
            precision = (
                advanced_prior_precision
                + design_theta.T
                @ (design_theta / total_variance[:, None])
            )
            covariance_output = np.linalg.pinv(precision)
            covariance_output = 0.5 * (
                covariance_output + covariance_output.T)
            residual = margins - fitted_mean
            residual_prior_df = float(self.calibration_prior_df)
            posterior_df = float(residual_prior_df + len(margins))
            posterior_variance = (
                residual_prior_df * self.residual_scale_ ** 2
                + float(np.sum(np.maximum(
                    residual ** 2 - observation_variance, 0.0)))
            ) / max(posterior_df, 1.0)
            dimension = int(self.coordinate_axes_.shape[1])
            effective_dimension = 2 + residual_rank + int(planar)
            status = "hierarchical_orthogonal_residual"
            if planar:
                status += "_planar_rotation"
            return TargetBoundaryAdapter(
                np.zeros(dimension, dtype=float),
                np.ones(dimension, dtype=float),
                rotation,
                location,
                scale,
                gamma,
                residual_rank,
                covariance_output,
                float(np.sqrt(max(
                    posterior_variance, self.variance_floor))),
                posterior_df,
                effective_dimension,
                status,
                {
                    "status": status,
                    "pilot_records": int(len(margins)),
                    "effective_label_adaptation_dimension": (
                        effective_dimension),
                    "output_offset": location,
                    "output_scale": scale,
                    "output_coordinate_linear": gamma.tolist(),
                    "positive_output_scale": True,
                    "rotation_mode": "planar" if planar else "none",
                    "rotation_angle": angle,
                    "rotation_deviation": float(np.linalg.norm(
                        rotation - np.eye(dimension))),
                    "orthogonal_residual_rank": residual_rank,
                    "map_converged": bool(result.success),
                    "map_message": str(result.message),
                    "effect_covariance_eigenvalues": np.linalg.eigvalsh(
                        covariance_output).tolist(),
                    "posterior_residual_scale": float(np.sqrt(max(
                        posterior_variance, self.variance_floor))),
                    "posterior_residual_degrees_of_freedom": posterior_df,
                    "source_residual_prior_degrees_of_freedom": (
                        residual_prior_df),
                    "target_data_used": True,
                    "target_oracle_used": False,
                },
            )

        def objective(theta):
            location = float(theta[0])
            scale = float(np.exp(theta[1]))
            residual = margins - location - scale * shape
            delta = theta - prior_mean
            return float(
                0.5 * np.sum(residual ** 2 / total_variance)
                + 0.5 * delta @ prior_precision @ delta
            )

        def gradient(theta):
            location = float(theta[0])
            scale = float(np.exp(theta[1]))
            residual = margins - location - scale * shape
            weighted = residual / total_variance
            delta = theta - prior_mean
            return np.asarray([
                -float(np.sum(weighted)),
                -float(np.sum(weighted * scale * shape)),
            ]) + prior_precision @ delta

        result = minimize(
            objective,
            np.asarray(prior_mean, dtype=float),
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(None, None), (-8.0, 8.0)],
        )
        theta = np.asarray(
            result.x if result.success else prior_mean, dtype=float)
        location = float(theta[0])
        scale = float(np.exp(theta[1]))
        design_theta = np.column_stack([
            np.ones(len(shape), dtype=float), scale * shape])
        precision = (
            prior_precision
            + design_theta.T
            @ (design_theta / total_variance[:, None])
        )
        covariance_theta = np.linalg.pinv(precision)
        covariance_theta = 0.5 * (
            covariance_theta + covariance_theta.T)
        transform = np.diag([1.0, scale])
        covariance_output = transform @ covariance_theta @ transform.T
        residual = margins - location - scale * shape
        # Source residual degrees of freedom describe how precisely the source
        # residual scale was estimated; they must not become hundreds of
        # pseudo-observations on a new task.  ``calibration_prior_df`` is the
        # declared transferable prior strength for target residual scale.
        residual_prior_df = float(self.calibration_prior_df)
        posterior_df = float(residual_prior_df + len(margins))
        posterior_variance = (
            residual_prior_df * self.residual_scale_ ** 2
            + float(np.sum(np.maximum(
                residual ** 2 - observation_variance, 0.0)))
        ) / max(posterior_df, 1.0)
        dimension = int(self.coordinate_axes_.shape[1])
        return TargetBoundaryAdapter(
            np.zeros(dimension, dtype=float),
            np.ones(dimension, dtype=float),
            np.eye(dimension, dtype=float),
            location,
            scale,
            np.zeros(0, dtype=float),
            0,
            covariance_output,
            float(np.sqrt(max(
                posterior_variance, self.variance_floor))),
            posterior_df,
            2,
            "hierarchical_location_log_scale",
            {
                "status": "hierarchical_location_log_scale",
                "pilot_records": int(len(margins)),
                "effective_label_adaptation_dimension": 2,
                "output_offset": location,
                "output_scale": scale,
                "positive_output_scale": True,
                "map_converged": bool(result.success),
                "map_message": str(result.message),
                "effect_covariance_eigenvalues": np.linalg.eigvalsh(
                    covariance_theta).tolist(),
                "posterior_residual_scale": float(np.sqrt(max(
                    posterior_variance, self.variance_floor))),
                "posterior_residual_degrees_of_freedom": posterior_df,
                "source_residual_prior_degrees_of_freedom": (
                    residual_prior_df),
                "target_data_used": True,
                "target_oracle_used": False,
            },
        )

    def predict(self, descriptors, adapter=None):
        adapter = self.prior_adapter() if adapter is None else adapter
        coordinate = self._coordinate(descriptors)
        adapted_coordinate = (
            coordinate @ np.asarray(adapter.rotation, dtype=float))
        shape = self._shape_from_coordinate(adapted_coordinate)
        residual_features = self._target_residual_features(
            adapted_coordinate, shape)
        gamma = np.asarray(
            adapter.output_coordinate_linear, dtype=float).reshape(-1)
        return (
            float(adapter.output_offset)
            + float(adapter.output_scale) * shape
            + residual_features[:, :len(gamma)] @ gamma
        )

    def canonical_score(self, descriptors):
        """Return the source-normalized signed-distance atom."""

        if self.fit_status != "fit":
            raise RuntimeError("hierarchical boundary model must be fit first")
        return np.asarray(self._shape(descriptors), dtype=float)

    def predict_upper(self, descriptors, adapter=None):
        adapter = self.prior_adapter() if adapter is None else adapter
        coordinate = self._coordinate(descriptors)
        adapted_coordinate = (
            coordinate @ np.asarray(adapter.rotation, dtype=float))
        shape = self._shape_from_coordinate(adapted_coordinate)
        residual_features = self._target_residual_features(
            adapted_coordinate, shape)
        gamma = np.asarray(
            adapter.output_coordinate_linear, dtype=float).reshape(-1)
        mean = (
            float(adapter.output_offset)
            + float(adapter.output_scale) * shape
            + residual_features[:, :len(gamma)] @ gamma
        )
        covariance = np.asarray(adapter.output_covariance, dtype=float)
        advanced_dimension = 2 + len(gamma)
        planar = str(adapter.diagnostics.get(
            "rotation_mode", "none")) == "planar"
        advanced_dimension += int(planar)
        if covariance.shape == (advanced_dimension, advanced_dimension) and (
            len(gamma) > 0 or planar
        ):
            angle = float(adapter.diagnostics.get("rotation_angle", 0.0))
            columns = [
                np.ones(len(shape), dtype=float),
                float(adapter.output_scale) * shape,
            ]
            columns.extend(
                residual_features[:, index]
                for index in range(len(gamma)))
            if planar:
                step = 1e-5

                def rotated_mean(local_angle):
                    local_coordinate = coordinate @ self._planar_rotation(
                        coordinate.shape[1], local_angle)
                    local_shape = self._shape_from_coordinate(
                        local_coordinate)
                    local_residual = self._target_residual_features(
                        local_coordinate, local_shape)
                    return (
                        float(adapter.output_offset)
                        + float(adapter.output_scale) * local_shape
                        + local_residual[:, :len(gamma)] @ gamma
                    )

                derivative = (
                    rotated_mean(angle + step)
                    - rotated_mean(angle - step)
                ) / (2.0 * step)
                columns.append(derivative)
            design = np.column_stack(columns)
        else:
            design = np.column_stack([
                np.ones(len(shape), dtype=float), shape])
        parameter_variance = np.einsum(
            "ni,ij,nj->n", design, covariance, design)
        predictive_variance = np.maximum(
            float(adapter.residual_scale) ** 2 + parameter_variance,
            self.variance_floor,
        )
        quantile = float(student_t.ppf(
            1.0 - self.upper_alpha,
            max(float(adapter.residual_degrees_of_freedom), 1.0),
        ))
        return mean + quantile * np.sqrt(predictive_variance)


class BoundaryFamilyMixturePosterior:
    """Finite source-frozen boundary library with target evidence weighting.

    The library contains one pooled family and, when at least three source
    domains are available, one leave-one-source-domain-out family per source
    domain.  A held-out target never supplies a domain label or evaluation
    oracle.  Its ordinary pilot margins update family weights through a
    leave-one-pilot-out generalized Bayes score.

    Certification does not average incompatible upper margins.  It takes the
    pointwise envelope over the smallest posterior family set carrying at
    least ``1 - family_delta`` mass, then adds a nonnegative between-family
    guard.  Thus a low-weight optimistic family cannot dilute a dangerous
    family that remains in the credible set.
    """

    def __init__(
        self,
        *,
        base_model_kwargs=None,
        family_delta=0.10,
        evidence_temperature=0.50,
        family_guard_scale=0.0,
        variance_floor=1e-10,
        include_pooled_family=True,
        include_leave_one_domain_out=True,
        family_strategy=None,
    ):
        self.base_model_kwargs = dict(base_model_kwargs or {})
        self.family_delta = float(np.clip(family_delta, 0.0, 0.49))
        self.evidence_temperature = max(
            float(evidence_temperature), 0.0)
        self.family_guard_scale = max(float(family_guard_scale), 0.0)
        self.variance_floor = max(float(variance_floor), 1e-12)
        self.include_pooled_family = bool(include_pooled_family)
        self.include_leave_one_domain_out = bool(
            include_leave_one_domain_out)
        if family_strategy is None:
            family_strategy = "pooled_plus_leave_one_source_domain_out"
        self.family_strategy = str(family_strategy).lower()
        aliases = {
            "pooled_loo": "pooled_plus_leave_one_source_domain_out",
            "atoms": "source_domain_atoms",
            "pooled_atoms": "pooled_plus_source_domain_atoms",
        }
        self.family_strategy = aliases.get(
            self.family_strategy, self.family_strategy)
        valid_strategies = {
            "pooled_plus_leave_one_source_domain_out",
            "source_domain_atoms",
            "pooled_plus_source_domain_atoms",
        }
        if self.family_strategy not in valid_strategies:
            raise ValueError(
                f"unknown boundary family strategy {self.family_strategy!r}")
        if not (
            self.include_pooled_family
            or self.include_leave_one_domain_out
        ):
            raise ValueError("boundary family library cannot be empty")
        self.fit_status = "unfit"
        self.diagnostics_ = {"status": "unfit"}

    @staticmethod
    def _optional_rows(values, mask):
        if values is None:
            return None
        return np.asarray(values)[mask]

    @staticmethod
    def _normalize_weights(log_weights):
        log_weights = np.asarray(log_weights, dtype=float).reshape(-1)
        if len(log_weights) == 0:
            raise ValueError("at least one family weight is required")
        normalizer = float(logsumexp(log_weights))
        weights = np.maximum(
            np.exp(log_weights - normalizer), np.finfo(float).tiny)
        return weights / max(float(np.sum(weights)), 1e-12)

    def _credible_indices(self, weights):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        order = np.argsort(-weights, kind="stable")
        threshold = 1.0 - self.family_delta
        selected = []
        mass = 0.0
        for index in order:
            selected.append(int(index))
            mass += float(weights[index])
            if mass + 1e-15 >= threshold:
                break
        if not selected:
            selected = [int(order[0])]
            mass = float(weights[order[0]])
        return np.asarray(selected, dtype=int), float(mass)

    def fit(
        self,
        descriptors,
        margins,
        domains,
        sample_weight=None,
        *,
        margin_variance=None,
        replicate_count=None,
    ):
        descriptors = np.asarray(descriptors, dtype=float)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if descriptors.ndim != 2 or len(descriptors) != len(margins):
            raise ValueError("descriptors and margins must have matching rows")
        if len(domains) != len(margins):
            raise ValueError("domains and margins must have matching rows")
        source_domains = sorted(set(str(value) for value in domains.tolist()))
        if len(source_domains) < 2:
            raise ValueError("boundary family source fit needs two domains")
        if sample_weight is None:
            sample_weight = np.ones(len(margins), dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(sample_weight) != len(margins):
            raise ValueError("sample weights must match margin rows")

        specifications = []
        use_pooled = self.family_strategy in {
            "pooled_plus_leave_one_source_domain_out",
            "pooled_plus_source_domain_atoms",
        }
        if use_pooled:
            specifications.append((
                "pooled",
                np.ones(len(margins), dtype=bool),
                None,
                False,
            ))
        if (
            self.family_strategy
            == "pooled_plus_leave_one_source_domain_out"
            and len(source_domains) >= 3
        ):
            for excluded in source_domains:
                mask = domains != excluded
                if len(set(domains[mask].tolist())) >= 2:
                    specifications.append((
                        f"leave_out:{excluded}", mask, excluded, False))
        if self.family_strategy in {
            "source_domain_atoms",
            "pooled_plus_source_domain_atoms",
        }:
            for source_domain in source_domains:
                specifications.append((
                    f"source_atom:{source_domain}",
                    domains == source_domain,
                    None,
                    True,
                ))

        families = []
        labels = []
        family_rows = []
        for label, mask, excluded, atomic in specifications:
            model_kwargs = dict(self.base_model_kwargs)
            if atomic:
                model_kwargs["allow_single_source_domain"] = True
            model = HierarchicalSignedDistancePosterior(
                **model_kwargs)
            model.fit(
                descriptors[mask],
                margins[mask],
                domains[mask],
                sample_weight=sample_weight[mask],
                margin_variance=self._optional_rows(
                    margin_variance, mask),
                replicate_count=self._optional_rows(
                    replicate_count, mask),
            )
            families.append(model)
            labels.append(label)
            family_rows.append({
                "label": label,
                "excluded_source_domain": excluded,
                "atomic_source_domain": (
                    str(domains[mask][0]) if atomic else None),
                "training_domains": sorted(set(
                    str(value) for value in domains[mask].tolist())),
                "training_records": int(np.sum(mask)),
                "source_boundary_rmse": float(
                    model.diagnostics()["source_boundary_rmse"]),
                "source_false_safe_rate": float(
                    model.diagnostics()["source_false_safe_rate"]),
            })
        if not families:
            raise ValueError("no valid source boundary families were built")

        self.families_ = tuple(families)
        self.family_labels_ = tuple(labels)
        self.source_prior_weights_ = np.full(
            len(families), 1.0 / len(families), dtype=float)
        self.source_domains_ = tuple(source_domains)
        self.fit_status = "fit"
        self.diagnostics_ = {
            "status": "fit",
            "model_version": "tcb_v3_boundary_family_mixture",
            "family_strategy": self.family_strategy,
            "family_count": int(len(families)),
            "family_labels": list(labels),
            "families": family_rows,
            "source_prior_weights": self.source_prior_weights_.tolist(),
            "source_domains": list(source_domains),
            "n_source_records": int(len(margins)),
            "family_delta": float(self.family_delta),
            "evidence_temperature": float(self.evidence_temperature),
            "family_guard_scale": float(self.family_guard_scale),
            "target_label_used": False,
            "target_data_used": False,
            "target_oracle_used": False,
            "family_parameters_frozen_after_source_fit": True,
        }
        return self

    def _require_fit(self):
        if self.fit_status != "fit":
            raise RuntimeError("boundary family posterior must be fit first")

    def prior_adapter(self):
        self._require_fit()
        family_adapters = tuple(
            family.prior_adapter() for family in self.families_)
        weights = np.asarray(self.source_prior_weights_, dtype=float).copy()
        credible, credible_mass = self._credible_indices(weights)
        family_dimension = max(
            adapter.effective_dimension for adapter in family_adapters)
        effective_dimension = family_dimension + max(len(weights) - 1, 0)
        diagnostics = {
            "status": "source_family_prior",
            "pilot_records": 0,
            "family_labels": list(self.family_labels_),
            "posterior_weights": weights.tolist(),
            "log_predictive_evidence": [0.0] * len(weights),
            "credible_family_labels": [
                self.family_labels_[index] for index in credible],
            "credible_family_mass": float(credible_mass),
            "credible_family_count": int(len(credible)),
            "family_posterior_simplex_dimension": max(len(weights) - 1, 0),
            "effective_label_adaptation_dimension": int(
                effective_dimension),
            "evidence_protocol": "none_source_prior",
            "target_label_used": False,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        return BoundaryFamilyMixtureAdapter(
            family_adapters,
            weights,
            np.zeros(len(weights), dtype=float),
            credible,
            credible_mass,
            int(effective_dimension),
            "source_family_prior",
            diagnostics,
        )

    @staticmethod
    def _adapter_quantile(model, adapter):
        degrees = max(float(adapter.residual_degrees_of_freedom), 1.0)
        quantile = float(student_t.ppf(
            1.0 - model.upper_alpha, degrees))
        return max(quantile, 1e-8)

    def _loo_log_predictive_evidence(
        self,
        model,
        descriptors,
        margins,
        pilot_variance,
        replicate_count,
    ):
        count = len(margins)
        if count == 0:
            return 0.0
        observation_variance = np.maximum(
            np.asarray(pilot_variance, dtype=float).reshape(-1)
            / np.maximum(
                np.asarray(replicate_count, dtype=float).reshape(-1), 1.0),
            self.variance_floor,
        )
        log_score = 0.0
        for index in range(count):
            keep = np.arange(count) != index
            adapter = (
                model.fit_target_adapter(
                    descriptors[keep],
                    margins[keep],
                    pilot_variance=np.asarray(pilot_variance)[keep],
                    replicate_count=np.asarray(replicate_count)[keep],
                )
                if np.any(keep) else model.prior_adapter()
            )
            row = descriptors[index:index + 1]
            mean = float(model.predict(row, adapter=adapter)[0])
            upper = float(model.predict_upper(row, adapter=adapter)[0])
            quantile = self._adapter_quantile(model, adapter)
            model_variance = max(
                ((upper - mean) / quantile) ** 2,
                self.variance_floor,
            )
            total_variance = max(
                model_variance + float(observation_variance[index]),
                self.variance_floor,
            )
            residual = float(margins[index]) - mean
            log_score -= 0.5 * (
                np.log(2.0 * np.pi * total_variance)
                + residual ** 2 / total_variance
            )
        return float(log_score)

    def fit_target_adapter(
        self,
        pilot_descriptors,
        pilot_margins,
        *,
        pilot_variance=None,
        replicate_count=None,
    ):
        self._require_fit()
        descriptors = np.asarray(pilot_descriptors, dtype=float)
        margins = np.asarray(pilot_margins, dtype=float).reshape(-1)
        if descriptors.ndim != 2 or len(descriptors) != len(margins):
            raise ValueError("pilot descriptors and margins must match")
        count = len(margins)
        if count == 0:
            return self.prior_adapter()
        if pilot_variance is None:
            pilot_variance = np.full(
                count, self.variance_floor, dtype=float)
        pilot_variance = np.maximum(
            np.asarray(pilot_variance, dtype=float).reshape(-1),
            self.variance_floor,
        )
        if replicate_count is None:
            replicate_count = np.ones(count, dtype=float)
        replicate_count = np.maximum(
            np.asarray(replicate_count, dtype=float).reshape(-1), 1.0)
        if len(pilot_variance) != count or len(replicate_count) != count:
            raise ValueError("pilot variance and replicate counts must match")

        adapters = []
        evidence = []
        for family in self.families_:
            adapters.append(family.fit_target_adapter(
                descriptors,
                margins,
                pilot_variance=pilot_variance,
                replicate_count=replicate_count,
            ))
            evidence.append(self._loo_log_predictive_evidence(
                family,
                descriptors,
                margins,
                pilot_variance,
                replicate_count,
            ))
        evidence = np.asarray(evidence, dtype=float)
        if not np.all(np.isfinite(evidence)):
            raise FloatingPointError(
                "boundary-family predictive evidence must be finite")
        log_prior = np.log(np.maximum(
            self.source_prior_weights_, 1e-300))
        weights = self._normalize_weights(
            log_prior + self.evidence_temperature * evidence)
        credible, credible_mass = self._credible_indices(weights)
        entropy = float(-np.sum(
            weights * np.log(np.maximum(weights, 1e-300))))
        family_dimension = max(
            adapter.effective_dimension for adapter in adapters)
        effective_dimension = family_dimension + max(len(weights) - 1, 0)
        family_adapter_rows = []
        for label, family_adapter in zip(self.family_labels_, adapters):
            covariance = np.asarray(
                family_adapter.output_covariance, dtype=float)
            eigenvalues = np.linalg.eigvalsh(
                0.5 * (covariance + covariance.T))
            family_adapter_rows.append({
                "label": str(label),
                "output_offset": float(family_adapter.output_offset),
                "output_scale": float(family_adapter.output_scale),
                "residual_scale": float(family_adapter.residual_scale),
                "residual_degrees_of_freedom": float(
                    family_adapter.residual_degrees_of_freedom),
                "parameter_covariance_trace": float(np.trace(covariance)),
                "parameter_covariance_max_eigenvalue": float(
                    np.max(eigenvalues) if len(eigenvalues) else 0.0),
                "effective_dimension": int(
                    family_adapter.effective_dimension),
                "orthogonal_residual_rank": int(
                    family_adapter.diagnostics.get(
                        "orthogonal_residual_rank", 0)),
                "map_converged": bool(
                    family_adapter.diagnostics.get("map_converged", True)),
            })
        diagnostics = {
            "status": "generalized_bayes_family_posterior",
            "pilot_records": int(count),
            "family_labels": list(self.family_labels_),
            "posterior_weights": weights.tolist(),
            "log_predictive_evidence": evidence.tolist(),
            "credible_family_labels": [
                self.family_labels_[index] for index in credible],
            "credible_family_mass": float(credible_mass),
            "credible_family_count": int(len(credible)),
            "family_delta": float(self.family_delta),
            "posterior_entropy": entropy,
            "effective_family_count": float(np.exp(entropy)),
            "family_posterior_simplex_dimension": max(len(weights) - 1, 0),
            "effective_label_adaptation_dimension": int(
                effective_dimension),
            "evidence_protocol": "leave_one_pilot_out_generalized_bayes",
            "evidence_temperature": float(self.evidence_temperature),
            "family_target_adapters": family_adapter_rows,
            "family_parameters_frozen": True,
            "target_label_used": False,
            "target_data_used": True,
            "target_oracle_used": False,
        }
        return BoundaryFamilyMixtureAdapter(
            tuple(adapters),
            weights,
            evidence,
            credible,
            credible_mass,
            int(effective_dimension),
            "generalized_bayes_family_posterior",
            diagnostics,
        )

    def predict_components(self, descriptors, adapter=None):
        self._require_fit()
        adapter = self.prior_adapter() if adapter is None else adapter
        descriptors = np.asarray(descriptors, dtype=float)
        means = np.vstack([
            family.predict(descriptors, adapter=family_adapter)
            for family, family_adapter in zip(
                self.families_, adapter.family_adapters)
        ])
        uppers = np.vstack([
            family.predict_upper(descriptors, adapter=family_adapter)
            for family, family_adapter in zip(
                self.families_, adapter.family_adapters)
        ])
        weights = np.asarray(adapter.posterior_weights, dtype=float)
        posterior_mean = weights @ means
        credible = np.asarray(adapter.credible_indices, dtype=int)
        credible_weights = weights[credible]
        credible_weights /= max(float(np.sum(credible_weights)), 1e-12)
        credible_mean = credible_weights @ means[credible]
        between_family_variance = np.sum(
            credible_weights[:, None]
            * (means[credible] - credible_mean[None, :]) ** 2,
            axis=0,
        )
        family_guard = self.family_guard_scale * np.sqrt(np.maximum(
            between_family_variance, 0.0))
        envelope = np.max(uppers[credible], axis=0)
        return {
            "mean": np.asarray(posterior_mean, dtype=float),
            "upper": np.asarray(envelope + family_guard, dtype=float),
            "family_mean": means,
            "family_upper": uppers,
            "credible_indices": credible,
            "credible_mass": float(adapter.credible_mass),
            "between_family_variance": np.asarray(
                between_family_variance, dtype=float),
            "family_selection_guard": np.asarray(family_guard, dtype=float),
        }

    def predict(self, descriptors, adapter=None):
        return self.predict_components(descriptors, adapter=adapter)["mean"]

    def predict_upper(self, descriptors, adapter=None):
        return self.predict_components(descriptors, adapter=adapter)["upper"]

    def diagnostics(self):
        return dict(self.diagnostics_)


class BoundaryFamilySynthesisPosterior:
    """Continuous nonnegative synthesis of source boundary atoms.

    Each source domain contributes one frozen canonical signed-distance atom.
    Source-domain fits define a transferable coefficient prior.  A held-out
    target uses only ordinary pilot margins to update an intercept and one
    nonnegative coefficient per atom.  The unconstrained Laplace covariance
    is retained at active nonnegativity constraints, which is conservative
    relative to truncating those directions.
    """

    def __init__(
        self,
        *,
        base_model_kwargs=None,
        coefficient_ridge=0.1,
        coefficient_prior_strength=1.0,
        coefficient_floor=0.0,
        variance_floor=1e-10,
    ):
        self.base_model_kwargs = dict(base_model_kwargs or {})
        self.coefficient_ridge = max(float(coefficient_ridge), 1e-10)
        self.coefficient_prior_strength = max(
            float(coefficient_prior_strength), 1e-8)
        self.coefficient_floor = max(float(coefficient_floor), 0.0)
        self.variance_floor = max(float(variance_floor), 1e-12)
        self.upper_alpha = float(np.clip(
            self.base_model_kwargs.get("upper_alpha", 0.01), 1e-6, 0.25))
        self.calibration_prior_df = max(float(
            self.base_model_kwargs.get("calibration_prior_df", 2.0)), 1.0)
        self.fit_status = "unfit"
        self.diagnostics_ = {"status": "unfit"}

    def _effective_variance(self, variance, replicate_count, size):
        if variance is None:
            variance = np.full(size, self.variance_floor, dtype=float)
        variance = np.maximum(
            np.asarray(variance, dtype=float).reshape(-1),
            self.variance_floor,
        )
        if replicate_count is None:
            replicate_count = np.ones(size, dtype=float)
        replicate_count = np.maximum(
            np.asarray(replicate_count, dtype=float).reshape(-1), 1.0)
        if len(variance) != size or len(replicate_count) != size:
            raise ValueError("variance and replicate counts must match rows")
        return np.maximum(
            variance / replicate_count, self.variance_floor)

    @staticmethod
    def _precision_root(precision):
        precision = np.asarray(precision, dtype=float)
        precision = 0.5 * (precision + precision.T)
        values, vectors = np.linalg.eigh(precision)
        values = np.maximum(values, 0.0)
        return np.sqrt(values)[:, None] * vectors.T

    def _fit_nonnegative_coefficients(
        self,
        scores,
        margins,
        observation_variance,
        *,
        residual_features=None,
        residual_ridge=None,
        prior_mean=None,
        prior_precision=None,
    ):
        scores = np.asarray(scores, dtype=float)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        observation_variance = np.maximum(
            np.asarray(observation_variance, dtype=float).reshape(-1),
            self.variance_floor,
        )
        if residual_features is None:
            residual_features = np.empty((len(scores), 0), dtype=float)
        residual_features = np.asarray(residual_features, dtype=float)
        if residual_features.ndim != 2 or len(residual_features) != len(scores):
            raise ValueError("residual features must match score rows")
        design = np.column_stack([
            np.ones(len(scores), dtype=float),
            scores,
            residual_features,
        ])
        root_weight = 1.0 / np.sqrt(observation_variance)
        augmented = design * root_weight[:, None]
        target = margins * root_weight
        dimension = design.shape[1]
        if prior_mean is None:
            prior_mean = np.zeros(dimension, dtype=float)
            local_residual_ridge = (
                self.coefficient_ridge
                if residual_ridge is None else max(
                    float(residual_ridge), 1e-10)
            )
            prior_precision = np.diag(np.concatenate([
                [1e-10],
                np.full(scores.shape[1], self.coefficient_ridge),
                np.full(
                    residual_features.shape[1], local_residual_ridge),
            ]))
        else:
            prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
            prior_precision = np.asarray(prior_precision, dtype=float)
        root = self._precision_root(prior_precision)
        augmented = np.vstack([augmented, root])
        target = np.concatenate([target, root @ prior_mean])
        lower = np.concatenate([
            [-np.inf],
            np.full(scores.shape[1], self.coefficient_floor),
            np.full(residual_features.shape[1], -np.inf),
        ])
        upper = np.full(dimension, np.inf, dtype=float)
        result = lsq_linear(
            augmented,
            target,
            bounds=(lower, upper),
            method="trf",
        )
        theta = np.asarray(result.x, dtype=float)
        precision = (
            np.asarray(prior_precision, dtype=float)
            + design.T
            @ (design / observation_variance[:, None])
        )
        covariance = np.linalg.pinv(precision)
        covariance = 0.5 * (covariance + covariance.T)
        return theta, covariance, bool(result.success), str(result.message)

    def _family_scores(self, descriptors):
        descriptors = np.asarray(descriptors, dtype=float)
        return np.column_stack([
            family.canonical_score(descriptors)
            for family in self.families_
        ])

    def fit(
        self,
        descriptors,
        margins,
        domains,
        sample_weight=None,
        *,
        margin_variance=None,
        replicate_count=None,
    ):
        descriptors = np.asarray(descriptors, dtype=float)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if descriptors.ndim != 2 or len(descriptors) != len(margins):
            raise ValueError("descriptors and margins must have matching rows")
        if len(domains) != len(margins):
            raise ValueError("domains and margins must have matching rows")
        source_domains = sorted(set(str(value) for value in domains.tolist()))
        if len(source_domains) < 2:
            raise ValueError("boundary synthesis needs two source domains")
        if sample_weight is None:
            sample_weight = np.ones(len(margins), dtype=float)
        sample_weight = np.maximum(
            np.asarray(sample_weight, dtype=float).reshape(-1), 1e-12)
        observation_variance = self._effective_variance(
            margin_variance, replicate_count, len(margins))
        weighted_variance = observation_variance / sample_weight

        library = BoundaryFamilyMixturePosterior(
            base_model_kwargs=self.base_model_kwargs,
            family_strategy="source_domain_atoms",
            family_delta=0.025,
            evidence_temperature=0.0,
        ).fit(
            descriptors,
            margins,
            domains,
            sample_weight=sample_weight,
            margin_variance=margin_variance,
            replicate_count=replicate_count,
        )
        self.library_ = library
        self.families_ = library.families_
        self.family_labels_ = library.family_labels_
        scores = self._family_scores(descriptors)

        effect_rows = []
        fitted = np.empty(len(margins), dtype=float)
        source_effects = {}
        for domain in source_domains:
            mask = domains == domain
            theta, _, converged, message = (
                self._fit_nonnegative_coefficients(
                    scores[mask],
                    margins[mask],
                    weighted_variance[mask],
                )
            )
            effect_rows.append(theta)
            fitted[mask] = (
                theta[0] + scores[mask] @ theta[1:])
            source_effects[str(domain)] = {
                "intercept": float(theta[0]),
                "coefficients": theta[1:].tolist(),
                "normalized_coefficients": (
                    theta[1:] / max(float(np.sum(theta[1:])), 1e-12)
                ).tolist(),
                "map_converged": converged,
                "map_message": message,
            }

        effect_rows = np.asarray(effect_rows, dtype=float)
        prior_mean = np.mean(effect_rows, axis=0)
        centered = effect_rows - prior_mean[None, :]
        prior_covariance = (
            centered.T @ centered / max(len(effect_rows) - 1, 1))
        global_scale = max(float(np.std(margins)), 1e-4)
        coefficient_scale = max(float(np.mean(
            np.sum(effect_rows[:, 1:], axis=1))), global_scale, 1e-4)
        prior_covariance += np.diag(np.concatenate([
            [max(0.05 * global_scale, 1e-4) ** 2],
            np.full(
                scores.shape[1],
                max(0.05 * coefficient_scale / np.sqrt(scores.shape[1]),
                    1e-4) ** 2,
            ),
        ]))
        prior_covariance = 0.5 * (
            prior_covariance + prior_covariance.T)
        prior_precision = (
            self.coefficient_prior_strength
            * np.linalg.pinv(prior_covariance)
        )

        residual = margins - fitted
        normalized_weight = sample_weight / max(
            float(np.sum(sample_weight)), 1e-12)
        intrinsic_variance = max(float(np.sum(
            normalized_weight
            * np.maximum(residual ** 2 - observation_variance, 0.0)
        )), self.variance_floor)
        self.prior_mean_ = np.asarray(prior_mean, dtype=float)
        self.prior_covariance_ = np.asarray(prior_covariance, dtype=float)
        self.prior_precision_ = np.asarray(prior_precision, dtype=float)
        self.residual_scale_ = float(np.sqrt(intrinsic_variance))
        self.residual_degrees_of_freedom_ = float(max(
            len(margins) - len(source_domains) * (scores.shape[1] + 1),
            1,
        ))
        self.source_domains_ = tuple(source_domains)
        self.fit_status = "fit"
        self.diagnostics_ = {
            "status": "fit",
            "model_version": "tcb_v4_boundary_family_synthesis",
            "family_strategy": "source_domain_atoms_continuous_synthesis",
            "family_count": int(scores.shape[1]),
            "family_labels": list(self.family_labels_),
            "source_domains": list(source_domains),
            "source_effects": source_effects,
            "source_prior_mean": self.prior_mean_.tolist(),
            "source_prior_covariance": self.prior_covariance_.tolist(),
            "source_boundary_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "source_residual_scale": self.residual_scale_,
            "source_residual_degrees_of_freedom": (
                self.residual_degrees_of_freedom_),
            "coefficient_ridge": float(self.coefficient_ridge),
            "coefficient_prior_strength": float(
                self.coefficient_prior_strength),
            "coefficient_floor": float(self.coefficient_floor),
            "nonnegative_family_coefficients": True,
            "source_dictionary_frozen_on_target": True,
            "target_label_used": False,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        return self

    def _require_fit(self):
        if self.fit_status != "fit":
            raise RuntimeError("boundary-family synthesis must be fit first")

    def prior_adapter(self):
        self._require_fit()
        coefficients = np.maximum(
            self.prior_mean_[1:], self.coefficient_floor)
        total = max(float(np.sum(coefficients)), 1e-12)
        diagnostics = {
            "status": "source_synthesis_prior",
            "pilot_records": 0,
            "family_labels": list(self.family_labels_),
            "coefficients": coefficients.tolist(),
            "normalized_coefficients": (coefficients / total).tolist(),
            "parameter_covariance_eigenvalues": np.linalg.eigvalsh(
                self.prior_covariance_).tolist(),
            "source_dictionary_frozen": True,
            "nonnegative_family_coefficients": True,
            "target_label_used": False,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        return BoundaryFamilySynthesisAdapter(
            float(self.prior_mean_[0]),
            coefficients,
            np.asarray(self.prior_covariance_, dtype=float).copy(),
            self.residual_scale_,
            self.residual_degrees_of_freedom_,
            int(len(coefficients) + 1),
            "source_synthesis_prior",
            diagnostics,
        )

    def fit_target_adapter(
        self,
        pilot_descriptors,
        pilot_margins,
        *,
        pilot_variance=None,
        replicate_count=None,
    ):
        self._require_fit()
        margins = np.asarray(pilot_margins, dtype=float).reshape(-1)
        if len(margins) == 0:
            return self.prior_adapter()
        scores = self._family_scores(pilot_descriptors)
        if len(scores) != len(margins):
            raise ValueError("pilot descriptors and margins must match")
        observation_variance = self._effective_variance(
            pilot_variance, replicate_count, len(margins))
        fitting_variance = np.maximum(
            observation_variance + self.residual_scale_ ** 2,
            self.variance_floor,
        )
        theta, covariance, converged, message = (
            self._fit_nonnegative_coefficients(
                scores,
                margins,
                fitting_variance,
                prior_mean=self.prior_mean_,
                prior_precision=self.prior_precision_,
            )
        )
        fitted = theta[0] + scores @ theta[1:]
        residual = margins - fitted
        posterior_df = float(self.calibration_prior_df + len(margins))
        posterior_variance = (
            self.calibration_prior_df * self.residual_scale_ ** 2
            + float(np.sum(np.maximum(
                residual ** 2 - observation_variance, 0.0)))
        ) / max(posterior_df, 1.0)
        coefficients = np.maximum(theta[1:], self.coefficient_floor)
        total = max(float(np.sum(coefficients)), 1e-12)
        normalized = coefficients / total
        entropy = float(-np.sum(
            normalized * np.log(np.maximum(normalized, 1e-300))))
        diagnostics = {
            "status": "nonnegative_boundary_family_synthesis_posterior",
            "pilot_records": int(len(margins)),
            "family_labels": list(self.family_labels_),
            "intercept": float(theta[0]),
            "coefficients": coefficients.tolist(),
            "normalized_coefficients": normalized.tolist(),
            "effective_family_count": float(np.exp(entropy)),
            "parameter_covariance_eigenvalues": np.linalg.eigvalsh(
                covariance).tolist(),
            "parameter_covariance_trace": float(np.trace(covariance)),
            "posterior_residual_scale": float(np.sqrt(max(
                posterior_variance, self.variance_floor))),
            "posterior_residual_degrees_of_freedom": posterior_df,
            "map_converged": converged,
            "map_message": message,
            "source_dictionary_frozen": True,
            "nonnegative_family_coefficients": bool(np.all(
                coefficients >= -1e-12)),
            "target_label_used": False,
            "target_data_used": True,
            "target_oracle_used": False,
        }
        return BoundaryFamilySynthesisAdapter(
            float(theta[0]),
            coefficients,
            covariance,
            float(np.sqrt(max(posterior_variance, self.variance_floor))),
            posterior_df,
            int(len(coefficients) + 1),
            "nonnegative_boundary_family_synthesis_posterior",
            diagnostics,
        )

    def predict(self, descriptors, adapter=None):
        self._require_fit()
        adapter = self.prior_adapter() if adapter is None else adapter
        return np.asarray(
            float(adapter.intercept)
            + self._family_scores(descriptors)
            @ np.asarray(adapter.coefficients, dtype=float),
            dtype=float,
        )

    def predict_upper(self, descriptors, adapter=None):
        self._require_fit()
        adapter = self.prior_adapter() if adapter is None else adapter
        scores = self._family_scores(descriptors)
        design = np.column_stack([
            np.ones(len(scores), dtype=float), scores])
        mean = (
            float(adapter.intercept)
            + scores @ np.asarray(adapter.coefficients, dtype=float)
        )
        covariance = np.asarray(adapter.output_covariance, dtype=float)
        parameter_variance = np.einsum(
            "ni,ij,nj->n", design, covariance, design)
        predictive_variance = np.maximum(
            float(adapter.residual_scale) ** 2 + parameter_variance,
            self.variance_floor,
        )
        quantile = float(student_t.ppf(
            1.0 - self.upper_alpha,
            max(float(adapter.residual_degrees_of_freedom), 1.0),
        ))
        return np.asarray(
            mean + quantile * np.sqrt(predictive_variance), dtype=float)

    def diagnostics(self):
        return dict(self.diagnostics_)


class BoundaryFamilySemiparametricPosterior(
    BoundaryFamilySynthesisPosterior,
):
    """Nonnegative family synthesis plus an orthogonal local RBF residual."""

    def __init__(
        self,
        *,
        residual_feature_count=2,
        residual_ridge=10.0,
        residual_lengthscale_multiplier=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.residual_feature_count = max(int(residual_feature_count), 1)
        self.semiparametric_residual_ridge = max(
            float(residual_ridge), 1e-10)
        self.residual_lengthscale_multiplier = max(
            float(residual_lengthscale_multiplier), 1e-3)

    @staticmethod
    def _farthest_centers(rows, count):
        rows = np.asarray(rows, dtype=float)
        count = min(max(int(count), 1), len(rows))
        selected = [int(np.argmin(np.sum(rows ** 2, axis=1)))]
        while len(selected) < count:
            distance = np.min(np.stack([
                np.sum((rows - rows[index]) ** 2, axis=1)
                for index in selected
            ]), axis=0)
            distance[selected] = -np.inf
            selected.append(int(np.argmax(distance)))
        return rows[selected]

    def _fit_residual_map(self, scores):
        scores = np.asarray(scores, dtype=float)
        self.residual_score_mean_ = np.mean(scores, axis=0)
        self.residual_score_scale_ = np.std(scores, axis=0)
        self.residual_score_scale_ = np.where(
            self.residual_score_scale_ < 1e-8,
            1.0,
            self.residual_score_scale_,
        )
        standardized = (
            scores - self.residual_score_mean_) / self.residual_score_scale_
        center_count = min(
            len(standardized),
            scores.shape[1] + self.residual_feature_count + 3,
        )
        centers = self._farthest_centers(standardized, center_count)
        if len(centers) > 1:
            pairwise = np.sqrt(np.sum(
                (centers[:, None, :] - centers[None, :, :]) ** 2,
                axis=2,
            ))
            positive = pairwise[pairwise > 1e-8]
            base_lengthscale = (
                float(np.median(positive)) if len(positive) else 1.0)
        else:
            base_lengthscale = 1.0
        lengthscale = max(
            base_lengthscale * self.residual_lengthscale_multiplier,
            1e-3,
        )
        squared = np.sum(
            (standardized[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        kernel = np.exp(-0.5 * squared / lengthscale ** 2)
        design = np.column_stack([
            np.ones(len(scores), dtype=float), scores])
        cross = design.T @ kernel
        try:
            _, singular_values, right_t = np.linalg.svd(
                cross, full_matrices=True)
        except np.linalg.LinAlgError:
            singular_values = np.empty(0, dtype=float)
            right_t = np.eye(kernel.shape[1], dtype=float)
        leading = float(singular_values[0]) if len(singular_values) else 0.0
        tolerance = (
            max(cross.shape) * np.finfo(float).eps * max(leading, 1.0))
        cross_rank = int(np.sum(singular_values > tolerance))
        nullspace = np.asarray(right_t[cross_rank:].T, dtype=float)
        if nullspace.shape[1] < 1:
            raise RuntimeError(
                "boundary synthesis dictionary has no orthogonal RBF residual")
        residual_dimension = min(
            self.residual_feature_count, nullspace.shape[1])
        projection = np.asarray(
            nullspace[:, :residual_dimension], dtype=float)
        for column in range(projection.shape[1]):
            pivot = int(np.argmax(np.abs(projection[:, column])))
            if projection[pivot, column] < 0.0:
                projection[:, column] *= -1.0
        residual = kernel @ projection
        residual_scale = np.std(residual, axis=0)
        residual_scale = np.where(residual_scale < 1e-8, 1.0, residual_scale)
        residual /= residual_scale
        cross_after = design.T @ residual
        denominator = max(
            float(np.linalg.norm(design) * np.linalg.norm(residual)),
            1e-12,
        )
        self.residual_centers_ = centers
        self.residual_lengthscale_ = lengthscale
        self.residual_projection_ = projection
        self.residual_feature_scale_ = residual_scale
        self.residual_feature_count_ = int(residual_dimension)
        self.residual_map_diagnostics_ = {
            "parent_kernel_dimension": int(kernel.shape[1]),
            "cross_dictionary_rank": cross_rank,
            "nullspace_dimension": int(nullspace.shape[1]),
            "residual_feature_count": int(residual_dimension),
            "residual_lengthscale": float(lengthscale),
            "orthogonality_fro": float(np.linalg.norm(cross_after)),
            "orthogonality_relative": float(
                np.linalg.norm(cross_after) / denominator),
            "projection_orthonormal_error": float(np.linalg.norm(
                projection.T @ projection
                - np.eye(residual_dimension, dtype=float))),
            "source_rows_used_for_unlabelled_projection": int(len(scores)),
            "target_labels_used_for_residual_dictionary": False,
        }
        return residual

    def _residual_features(self, scores):
        scores = np.asarray(scores, dtype=float)
        standardized = (
            scores - self.residual_score_mean_) / self.residual_score_scale_
        squared = np.sum(
            (
                standardized[:, None, :]
                - self.residual_centers_[None, :, :]
            ) ** 2,
            axis=2,
        )
        kernel = np.exp(
            -0.5 * squared / self.residual_lengthscale_ ** 2)
        return (
            kernel @ self.residual_projection_
        ) / self.residual_feature_scale_

    def fit(
        self,
        descriptors,
        margins,
        domains,
        sample_weight=None,
        *,
        margin_variance=None,
        replicate_count=None,
    ):
        super().fit(
            descriptors,
            margins,
            domains,
            sample_weight=sample_weight,
            margin_variance=margin_variance,
            replicate_count=replicate_count,
        )
        descriptors = np.asarray(descriptors, dtype=float)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if sample_weight is None:
            sample_weight = np.ones(len(margins), dtype=float)
        sample_weight = np.maximum(
            np.asarray(sample_weight, dtype=float).reshape(-1), 1e-12)
        observation_variance = self._effective_variance(
            margin_variance, replicate_count, len(margins))
        weighted_variance = observation_variance / sample_weight
        scores = self._family_scores(descriptors)
        residual_features = self._fit_residual_map(scores)
        family_count = scores.shape[1]

        source_domains = list(self.source_domains_)
        effect_rows = []
        fitted = np.empty(len(margins), dtype=float)
        source_effects = {}
        for domain in source_domains:
            mask = domains == domain
            theta, _, converged, message = (
                self._fit_nonnegative_coefficients(
                    scores[mask],
                    margins[mask],
                    weighted_variance[mask],
                    residual_features=residual_features[mask],
                    residual_ridge=self.semiparametric_residual_ridge,
                )
            )
            effect_rows.append(theta)
            fitted[mask] = (
                theta[0]
                + scores[mask] @ theta[1:1 + family_count]
                + residual_features[mask] @ theta[1 + family_count:]
            )
            source_effects[str(domain)] = {
                "intercept": float(theta[0]),
                "coefficients": theta[1:1 + family_count].tolist(),
                "residual_coefficients": theta[1 + family_count:].tolist(),
                "map_converged": converged,
                "map_message": message,
            }

        effect_rows = np.asarray(effect_rows, dtype=float)
        prior_mean = np.mean(effect_rows, axis=0)
        centered = effect_rows - prior_mean[None, :]
        prior_covariance = (
            centered.T @ centered / max(len(effect_rows) - 1, 1))
        global_scale = max(float(np.std(margins)), 1e-4)
        coefficient_scale = max(float(np.mean(np.sum(
            effect_rows[:, 1:1 + family_count], axis=1))), global_scale, 1e-4)
        prior_covariance += np.diag(np.concatenate([
            [max(0.05 * global_scale, 1e-4) ** 2],
            np.full(
                family_count,
                max(0.05 * coefficient_scale / np.sqrt(family_count),
                    1e-4) ** 2,
            ),
            np.full(
                self.residual_feature_count_,
                max(0.05 * global_scale, 1e-4) ** 2,
            ),
        ]))
        prior_covariance = 0.5 * (
            prior_covariance + prior_covariance.T)
        residual = margins - fitted
        normalized_weight = sample_weight / max(
            float(np.sum(sample_weight)), 1e-12)
        intrinsic_variance = max(float(np.sum(
            normalized_weight
            * np.maximum(residual ** 2 - observation_variance, 0.0)
        )), self.variance_floor)
        self.prior_mean_ = prior_mean
        self.prior_covariance_ = prior_covariance
        self.prior_precision_ = (
            self.coefficient_prior_strength
            * np.linalg.pinv(prior_covariance)
        )
        self.residual_scale_ = float(np.sqrt(intrinsic_variance))
        self.residual_degrees_of_freedom_ = float(max(
            len(margins)
            - len(source_domains) * len(prior_mean),
            1,
        ))
        self.diagnostics_.update({
            "model_version": "tcb_v5_orthogonal_semiparametric_boundary",
            "source_effects": source_effects,
            "source_prior_mean": prior_mean.tolist(),
            "source_prior_covariance": prior_covariance.tolist(),
            "source_boundary_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "source_residual_scale": self.residual_scale_,
            "source_residual_degrees_of_freedom": (
                self.residual_degrees_of_freedom_),
            "semiparametric_residual": True,
            "semiparametric_residual_ridge": float(
                self.semiparametric_residual_ridge),
            **self.residual_map_diagnostics_,
        })
        return self

    def prior_adapter(self):
        self._require_fit()
        family_count = len(self.family_labels_)
        coefficients = np.maximum(
            self.prior_mean_[1:1 + family_count], self.coefficient_floor)
        residual_coefficients = np.asarray(
            self.prior_mean_[1 + family_count:], dtype=float)
        total = max(float(np.sum(coefficients)), 1e-12)
        diagnostics = {
            "status": "source_semiparametric_prior",
            "pilot_records": 0,
            "family_labels": list(self.family_labels_),
            "coefficients": coefficients.tolist(),
            "normalized_coefficients": (coefficients / total).tolist(),
            "residual_coefficients": residual_coefficients.tolist(),
            "residual_feature_count": int(self.residual_feature_count_),
            "parameter_covariance_eigenvalues": np.linalg.eigvalsh(
                self.prior_covariance_).tolist(),
            "source_dictionary_frozen": True,
            "orthogonal_residual_dictionary_frozen": True,
            "nonnegative_family_coefficients": True,
            "target_label_used": False,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        return BoundaryFamilySemiparametricAdapter(
            float(self.prior_mean_[0]),
            coefficients,
            residual_coefficients,
            self.prior_covariance_.copy(),
            self.residual_scale_,
            self.residual_degrees_of_freedom_,
            int(len(self.prior_mean_)),
            "source_semiparametric_prior",
            diagnostics,
        )

    def fit_target_adapter(
        self,
        pilot_descriptors,
        pilot_margins,
        *,
        pilot_variance=None,
        replicate_count=None,
    ):
        self._require_fit()
        margins = np.asarray(pilot_margins, dtype=float).reshape(-1)
        if len(margins) == 0:
            return self.prior_adapter()
        scores = self._family_scores(pilot_descriptors)
        residual_features = self._residual_features(scores)
        observation_variance = self._effective_variance(
            pilot_variance, replicate_count, len(margins))
        fitting_variance = np.maximum(
            observation_variance + self.residual_scale_ ** 2,
            self.variance_floor,
        )
        theta, covariance, converged, message = (
            self._fit_nonnegative_coefficients(
                scores,
                margins,
                fitting_variance,
                residual_features=residual_features,
                residual_ridge=self.semiparametric_residual_ridge,
                prior_mean=self.prior_mean_,
                prior_precision=self.prior_precision_,
            )
        )
        family_count = scores.shape[1]
        fitted = (
            theta[0]
            + scores @ theta[1:1 + family_count]
            + residual_features @ theta[1 + family_count:]
        )
        residual = margins - fitted
        posterior_df = float(self.calibration_prior_df + len(margins))
        posterior_variance = (
            self.calibration_prior_df * self.residual_scale_ ** 2
            + float(np.sum(np.maximum(
                residual ** 2 - observation_variance, 0.0)))
        ) / max(posterior_df, 1.0)
        coefficients = np.maximum(
            theta[1:1 + family_count], self.coefficient_floor)
        residual_coefficients = np.asarray(
            theta[1 + family_count:], dtype=float)
        total = max(float(np.sum(coefficients)), 1e-12)
        normalized = coefficients / total
        entropy = float(-np.sum(
            normalized * np.log(np.maximum(normalized, 1e-300))))
        diagnostics = {
            "status": "orthogonal_semiparametric_boundary_posterior",
            "pilot_records": int(len(margins)),
            "family_labels": list(self.family_labels_),
            "intercept": float(theta[0]),
            "coefficients": coefficients.tolist(),
            "normalized_coefficients": normalized.tolist(),
            "residual_coefficients": residual_coefficients.tolist(),
            "residual_feature_count": int(self.residual_feature_count_),
            "effective_family_count": float(np.exp(entropy)),
            "parameter_covariance_eigenvalues": np.linalg.eigvalsh(
                covariance).tolist(),
            "parameter_covariance_trace": float(np.trace(covariance)),
            "posterior_residual_scale": float(np.sqrt(max(
                posterior_variance, self.variance_floor))),
            "posterior_residual_degrees_of_freedom": posterior_df,
            "map_converged": converged,
            "map_message": message,
            "source_dictionary_frozen": True,
            "orthogonal_residual_dictionary_frozen": True,
            "nonnegative_family_coefficients": bool(np.all(
                coefficients >= -1e-12)),
            "target_label_used": False,
            "target_data_used": True,
            "target_oracle_used": False,
        }
        return BoundaryFamilySemiparametricAdapter(
            float(theta[0]),
            coefficients,
            residual_coefficients,
            covariance,
            float(np.sqrt(max(posterior_variance, self.variance_floor))),
            posterior_df,
            int(len(theta)),
            "orthogonal_semiparametric_boundary_posterior",
            diagnostics,
        )

    def predict(self, descriptors, adapter=None):
        self._require_fit()
        adapter = self.prior_adapter() if adapter is None else adapter
        scores = self._family_scores(descriptors)
        residual_features = self._residual_features(scores)
        return np.asarray(
            float(adapter.intercept)
            + scores @ np.asarray(adapter.coefficients, dtype=float)
            + residual_features
            @ np.asarray(adapter.residual_coefficients, dtype=float),
            dtype=float,
        )

    def predict_upper(self, descriptors, adapter=None):
        self._require_fit()
        adapter = self.prior_adapter() if adapter is None else adapter
        scores = self._family_scores(descriptors)
        residual_features = self._residual_features(scores)
        design = np.column_stack([
            np.ones(len(scores), dtype=float), scores, residual_features])
        mean = self.predict(descriptors, adapter=adapter)
        covariance = np.asarray(adapter.output_covariance, dtype=float)
        parameter_variance = np.einsum(
            "ni,ij,nj->n", design, covariance, design)
        predictive_variance = np.maximum(
            float(adapter.residual_scale) ** 2 + parameter_variance,
            self.variance_floor,
        )
        quantile = float(student_t.ppf(
            1.0 - self.upper_alpha,
            max(float(adapter.residual_degrees_of_freedom), 1.0),
        ))
        return np.asarray(
            mean + quantile * np.sqrt(predictive_variance), dtype=float)
