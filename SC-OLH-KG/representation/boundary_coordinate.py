"""Domain-aligned chance-boundary coordinates and proposal selection.

The cumulative-risk coordinate ``psi=(A,N)`` is reserved for conditional
variance.  This module learns a separate low-dimensional coordinate ``phi``
from source-domain chance-margin strata.  Source constraint means, rather than
chance margins, define the coefficient prior used by the target GPR, so the
mean and aleatoric terms are not silently mixed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from core.cumulative_risk import canonical_risk_descriptor, get_risk_exposure
from representation.observable_coordinate import (
    SourceLearnedObservableCoordinate,
    observable_profile_library,
)
from representation.observable_exposure import (
    canonical_observable_state_descriptor,
    get_observable_state_exposure,
)


DEFAULT_MARGIN_THRESHOLDS = (-1.0, -0.35, 0.0, 0.35, 1.0)
VALID_BOUNDARY_INPUT_MODES = (
    "policy_profile",
    "source_learned_exposure",
    "observable_state_exposure",
    "provider_exposure",
)


def _as_rows(values):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def _stable_ranks(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _rank_correlation(left, right):
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    a = _stable_ranks(left)
    b = _stable_ranks(right)
    a -= float(np.mean(a))
    b -= float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 1e-12 else float(a @ b / denom)


def canonical_exposure_boundary_library(descriptor):
    """Return a stable nonlinear library for one observable exposure.

    ``canonical_risk_descriptor`` already separates local and shared blocks
    and is invariant to coordinate names and permutations.  Signed log
    compression prevents ratios and energies from dominating cross-domain
    alignment; squared terms retain the curvature needed by chance boundaries.
    """

    values = np.asarray(descriptor, dtype=float).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("canonical exposure descriptor must be finite")
    compressed = np.sign(values) * np.log1p(np.abs(values))
    return np.concatenate([compressed, compressed ** 2])


def _domain_scaled(values, domains):
    values = np.asarray(values, dtype=float).reshape(-1)
    domains = np.asarray(domains, dtype=object).reshape(-1)
    result = np.zeros(len(values), dtype=float)
    scales = {}
    for domain in sorted(set(str(value) for value in domains)):
        selected = np.asarray([
            str(value) == domain for value in domains
        ], dtype=bool)
        scale = max(float(np.sqrt(np.mean(values[selected] ** 2))), 1e-6)
        result[selected] = values[selected] / scale
        scales[domain] = scale
    return result, scales


def _scatter(matrix, groups, *, center_by_group=True):
    matrix = _as_rows(matrix)
    groups = np.asarray(groups, dtype=object).reshape(-1)
    result = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    if not center_by_group:
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        return centered.T @ centered
    for group in sorted(set(groups.tolist()), key=str):
        selected = groups == group
        if not np.any(selected):
            continue
        rows = matrix[selected]
        centered = rows - np.mean(rows, axis=0, keepdims=True)
        result += centered.T @ centered
    return result


def _generalized_components(between, nuisance, count, ridge):
    """Return stable leading directions of ``between`` against ``nuisance``."""

    between = np.asarray(between, dtype=float)
    nuisance = np.asarray(nuisance, dtype=float)
    dim = between.shape[0]
    scale = max(float(np.trace(nuisance)) / max(dim, 1), 1.0)
    denominator = 0.5 * (nuisance + nuisance.T)
    denominator += max(float(ridge), 1e-10) * scale * np.eye(dim)
    eigenvalues, eigenvectors = np.linalg.eigh(denominator)
    inv_root = (
        eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10)))
    ) @ eigenvectors.T
    whitened = inv_root @ (0.5 * (between + between.T)) @ inv_root
    values, vectors = np.linalg.eigh(0.5 * (whitened + whitened.T))
    order = np.argsort(values)[::-1]
    count = min(max(int(count), 1), dim)
    components = (inv_root @ vectors[:, order[:count]]).T
    for row in range(len(components)):
        norm = float(np.linalg.norm(components[row]))
        if norm > 1e-12:
            components[row] /= norm
    return components, np.maximum(values[order[:count]], 0.0)


class SourceAlignedBoundaryCoordinate(SourceLearnedObservableCoordinate):
    """Source-only conditional alignment for a target constraint-mean basis.

    ``boundary_margins`` determine the representation through signed
    chance-margin bins. ``coefficient_targets`` independently determine the
    source Gaussian coefficient law.  This distinction prevents source
    aleatoric variance from being baked into the target constraint mean.
    """

    def __init__(
        self,
        ridge_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        latent_dim=2,
        alignment_ridge=0.1,
        domain_penalty=1.0,
        within_bin_weight=0.1,
        margin_thresholds=DEFAULT_MARGIN_THRESHOLDS,
        templates_per_stratum=2,
        input_mode="policy_profile",
        observable_descriptor_mode="ordered",
        feature_mode="linear",
        latent_transform="identity",
        latent_temperature_grid=(0.5, 1.0, 2.0, 4.0),
        latent_support_quantile_grid=(0.8, 0.9, 0.95, 1.0),
        latent_support_min_bound=0.25,
        channel_role_aligner=None,
    ):
        super().__init__(
            ridge_grid=ridge_grid,
            output_mode="latent",
            latent_dim=latent_dim,
        )
        self.output_mode = "boundary_aligned"
        self.alignment_ridge = max(float(alignment_ridge), 1e-10)
        self.domain_penalty = max(float(domain_penalty), 0.0)
        self.within_bin_weight = max(float(within_bin_weight), 0.0)
        self.margin_thresholds = tuple(float(value) for value in margin_thresholds)
        self.templates_per_stratum = max(int(templates_per_stratum), 1)
        self.input_mode = str(input_mode).strip().lower()
        if self.input_mode not in VALID_BOUNDARY_INPUT_MODES:
            raise ValueError(
                "boundary input mode must be policy_profile, "
                "source_learned_exposure, observable_state_exposure, "
                "or provider_exposure")
        self.observable_descriptor_mode = str(
            observable_descriptor_mode or "ordered"
        ).strip().lower().replace("-", "_")
        if self.observable_descriptor_mode not in {
            "ordered", "set_invariant", "role_aligned", "role_transport"
        }:
            raise ValueError(
                "observable descriptor mode must be ordered, set_invariant, "
                "role_aligned, or role_transport")
        self.channel_role_aligner = channel_role_aligner
        if (
            self.observable_descriptor_mode in {
                "role_aligned", "role_transport"
            }
            and self.channel_role_aligner is None
        ):
            raise ValueError(
                "role_aligned observable descriptors require a fitted "
                "channel-role aligner")
        self.feature_mode = str(
            feature_mode or "linear").strip().lower().replace("-", "_")
        if self.feature_mode not in {
            "linear", "diagonal_quadratic", "full_quadratic"
        }:
            raise ValueError(
                "boundary feature mode must be linear, diagonal_quadratic, "
                "or full_quadratic")
        self.latent_transform = str(
            latent_transform or "identity"
        ).strip().lower().replace("-", "_")
        if self.latent_transform not in {
            "identity",
            "source_tanh",
            "source_support_clip",
            "source_support_residual",
        }:
            raise ValueError(
                "boundary latent transform must be identity, source_tanh, "
                "source_support_clip, or source_support_residual")
        self.latent_temperature_grid = tuple(sorted(set(
            max(float(value), 1e-6) for value in latent_temperature_grid
        )))
        if not self.latent_temperature_grid:
            raise ValueError("boundary latent temperature grid cannot be empty")
        self.latent_support_quantile_grid = tuple(sorted(set(
            float(np.clip(value, 0.5, 1.0))
            for value in latent_support_quantile_grid
        )))
        if not self.latent_support_quantile_grid:
            raise ValueError("boundary latent support quantile grid cannot be empty")
        self.latent_support_min_bound = max(
            float(latent_support_min_bound), 1e-6)
        self.latent_transform_temperature = None
        self.latent_support_quantile = None
        self.latent_support_bounds = None
        self.latent_transform_diagnostics = {"status": "unfit"}
        self.alignment_components = None
        self.latent_mean = None
        self.latent_scale = None
        self.boundary_profile_templates = []
        self.alignment_diagnostics = {"status": "unfit"}

    def _select_latent_transform(
        self,
        normalized_latent,
        coefficient_targets,
        domains,
        sample_weight,
    ):
        """Choose a bounded latent scale by source-domain holdout only."""

        normalized_latent = _as_rows(normalized_latent)
        if self.latent_transform == "identity":
            self.latent_transform_temperature = None
            self.latent_transform_diagnostics = {
                "status": "identity",
                "transform": "identity",
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            }
            return normalized_latent.copy()

        target = np.asarray(coefficient_targets, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        weight = np.maximum(
            np.asarray(sample_weight, dtype=float).reshape(-1), 1e-8)
        unique_domains = sorted(set(str(value) for value in domains))
        if self.latent_transform in {
            "source_support_clip", "source_support_residual"
        }:
            residual_channel = bool(
                self.latent_transform == "source_support_residual")
            score_rows = []
            for quantile in self.latent_support_quantile_grid:
                fold_losses = []
                fold_rank_losses = []
                fold_bounds = []
                for heldout in unique_domains:
                    test = np.asarray([
                        str(value) == heldout for value in domains
                    ], dtype=bool)
                    train = ~test
                    if int(np.sum(train)) < 2 or int(np.sum(test)) < 2:
                        continue
                    bounds = self._support_bounds(
                        normalized_latent[train], quantile)
                    transformed_train = self._support_transform(
                        normalized_latent[train], bounds, residual_channel)
                    transformed_test = self._support_transform(
                        normalized_latent[test], bounds, residual_channel)
                    best_prediction = None
                    best_train_loss = None
                    for ridge in self.ridge_grid:
                        coefficient = self._solve(
                            transformed_train,
                            target[train],
                            weight[train],
                            ridge,
                        )
                        train_design = np.column_stack([
                            np.ones(int(np.sum(train))), transformed_train
                        ])
                        train_error = (
                            target[train] - train_design @ coefficient)
                        train_loss = float(np.average(
                            train_error ** 2, weights=weight[train]))
                        if (
                            best_train_loss is None
                            or train_loss < best_train_loss
                        ):
                            best_train_loss = train_loss
                            test_design = np.column_stack([
                                np.ones(int(np.sum(test))), transformed_test
                            ])
                            best_prediction = test_design @ coefficient
                    residual = target[test] - best_prediction
                    target_centered = target[test] - float(np.average(
                        target[test], weights=weight[test]))
                    denominator = max(float(np.average(
                        target_centered ** 2, weights=weight[test])), 1e-8)
                    fold_losses.append(float(np.average(
                        residual ** 2, weights=weight[test])) / denominator)
                    fold_rank_losses.append(
                        1.0 - _rank_correlation(
                            best_prediction, target[test]))
                    fold_bounds.append(bounds)
                prediction_loss = (
                    float(np.mean(fold_losses))
                    if fold_losses else float("inf"))
                rank_loss = (
                    float(np.mean(fold_rank_losses))
                    if fold_rank_losses else float("inf"))
                total = prediction_loss + 0.25 * rank_loss
                score_rows.append({
                    "quantile": float(quantile),
                    "prediction_nmse": float(prediction_loss),
                    "rank_loss": float(rank_loss),
                    "total": float(total),
                    "median_fold_max_bound": (
                        float(np.median([
                            np.max(value) for value in fold_bounds
                        ])) if fold_bounds else None),
                })
            finite_rows = [
                row for row in score_rows if np.isfinite(row["total"])]
            if not finite_rows:
                selected = float(self.latent_support_quantile_grid[-1])
                status = "fallback"
            else:
                selected = min(
                    finite_rows,
                    key=lambda row: (
                        row["total"], -row["quantile"],
                    ),
                )["quantile"]
                status = "source_lodo_selected"
            self.latent_support_quantile = float(selected)
            self.latent_support_bounds = self._support_bounds(
                normalized_latent, selected)
            transformed = self._support_transform(
                normalized_latent,
                self.latent_support_bounds,
                residual_channel,
            )
            self.latent_transform_diagnostics = {
                "status": status,
                "transform": self.latent_transform,
                "selected_quantile": float(selected),
                "support_bounds": self.latent_support_bounds.tolist(),
                "residual_channel": residual_channel,
                "residual_channel_index": (
                    int(transformed.shape[1] - 1)
                    if residual_channel else None),
                "candidate_scores": score_rows,
                "source_domain_count": int(len(unique_domains)),
                "selection_target": "constraint_mean",
                "selection_uses_target_data": False,
                "selection_uses_target_oracle": False,
            }
            return transformed

        score_rows = []
        for temperature in self.latent_temperature_grid:
            transformed = np.tanh(normalized_latent / float(temperature))
            fold_losses = []
            fold_rank_losses = []
            for heldout in unique_domains:
                test = np.asarray([
                    str(value) == heldout for value in domains
                ], dtype=bool)
                train = ~test
                if int(np.sum(train)) < 2 or int(np.sum(test)) < 2:
                    continue
                best_prediction = None
                best_train_loss = None
                for ridge in self.ridge_grid:
                    coefficient = self._solve(
                        transformed[train], target[train], weight[train], ridge)
                    train_design = np.column_stack([
                        np.ones(int(np.sum(train))), transformed[train]
                    ])
                    train_error = target[train] - train_design @ coefficient
                    train_loss = float(np.average(
                        train_error ** 2, weights=weight[train]))
                    if best_train_loss is None or train_loss < best_train_loss:
                        best_train_loss = train_loss
                        test_design = np.column_stack([
                            np.ones(int(np.sum(test))), transformed[test]
                        ])
                        best_prediction = test_design @ coefficient
                residual = target[test] - best_prediction
                target_centered = target[test] - float(np.average(
                    target[test], weights=weight[test]))
                denominator = max(float(np.average(
                    target_centered ** 2, weights=weight[test])), 1e-8)
                fold_losses.append(float(np.average(
                    residual ** 2, weights=weight[test])) / denominator)
                fold_rank_losses.append(
                    1.0 - _rank_correlation(best_prediction, target[test]))
            prediction_loss = (
                float(np.mean(fold_losses)) if fold_losses else float("inf"))
            rank_loss = (
                float(np.mean(fold_rank_losses))
                if fold_rank_losses else float("inf"))
            total = prediction_loss + 0.25 * rank_loss
            score_rows.append({
                "temperature": float(temperature),
                "prediction_nmse": float(prediction_loss),
                "rank_loss": float(rank_loss),
                "total": float(total),
            })
        finite_rows = [
            row for row in score_rows if np.isfinite(row["total"])]
        if not finite_rows:
            selected = float(self.latent_temperature_grid[len(
                self.latent_temperature_grid) // 2])
            status = "fallback"
        else:
            selected = min(
                finite_rows,
                key=lambda row: (
                    row["total"], abs(np.log(row["temperature"])),
                    row["temperature"],
                ),
            )["temperature"]
            status = "source_lodo_selected"
        self.latent_transform_temperature = float(selected)
        self.latent_transform_diagnostics = {
            "status": status,
            "transform": "tanh",
            "selected_temperature": float(selected),
            "candidate_scores": score_rows,
            "source_domain_count": int(len(unique_domains)),
            "selection_target": "constraint_mean",
            "selection_uses_target_data": False,
            "selection_uses_target_oracle": False,
        }
        return np.tanh(normalized_latent / float(selected))

    def _support_bounds(self, normalized_latent, quantile):
        normalized_latent = _as_rows(normalized_latent)
        bounds = np.quantile(
            np.abs(normalized_latent), float(quantile), axis=0)
        return np.maximum(
            np.asarray(bounds, dtype=float), self.latent_support_min_bound)

    @staticmethod
    def _support_transform(normalized_latent, bounds, residual_channel):
        normalized_latent = _as_rows(normalized_latent)
        bounds = np.asarray(bounds, dtype=float).reshape(1, -1)
        clipped = np.clip(normalized_latent, -bounds, bounds)
        if not residual_channel:
            return clipped
        overflow = np.maximum(np.abs(normalized_latent) - bounds, 0.0)
        normalized_overflow = overflow / np.maximum(bounds, 1e-8)
        radial = np.tanh(np.sqrt(np.mean(
            normalized_overflow ** 2, axis=1, keepdims=True)))
        return np.hstack([clipped, radial])

    def _transform_normalized_latent(self, normalized_latent):
        normalized_latent = _as_rows(normalized_latent)
        if self.latent_transform == "identity":
            return normalized_latent.copy()
        if self.latent_transform in {
            "source_support_clip", "source_support_residual"
        }:
            if self.latent_support_bounds is None:
                raise RuntimeError("source-support latent transform is not fit")
            return self._support_transform(
                normalized_latent,
                self.latent_support_bounds,
                self.latent_transform == "source_support_residual",
            )
        if self.latent_transform_temperature is None:
            raise RuntimeError("source-tanh latent transform is not fit")
        return np.tanh(
            normalized_latent / float(self.latent_transform_temperature))

    def _input_library(self, value):
        if self.input_mode != "policy_profile":
            return canonical_exposure_boundary_library(value)
        return observable_profile_library(value)

    def _problem_input(self, problem, x):
        if self.input_mode == "observable_state_exposure":
            exposure = get_observable_state_exposure(problem, x)
            if exposure is None:
                raise ValueError(
                    "observable-state boundary coordinate requires an "
                    "observable_state_exposure adapter")
            if self.observable_descriptor_mode == "role_aligned":
                return self.channel_role_aligner.descriptor(problem, exposure)
            if self.observable_descriptor_mode == "role_transport":
                return self.channel_role_aligner.transport_descriptor(
                    problem, exposure)
            return canonical_observable_state_descriptor(
                exposure, mode=self.observable_descriptor_mode)
        if self.input_mode == "source_learned_exposure":
            if not hasattr(problem, "observable_boundary_exposure"):
                raise ValueError(
                    "source-learned exposure coordinate requires an "
                    "observable_boundary_exposure adapter")
            exposure = problem.observable_boundary_exposure(x)
            return canonical_risk_descriptor(exposure)
        if self.input_mode == "provider_exposure":
            exposure = (
                problem.provider_boundary_exposure(x)
                if hasattr(problem, "provider_boundary_exposure")
                else get_risk_exposure(problem, x, output_index=1)
            )
            if exposure is None:
                raise ValueError(
                    "provider exposure boundary coordinate requires a "
                    "CumulativeRiskFeatureProvider")
            return canonical_risk_descriptor(exposure)
        return problem.normalize(x)

    def _learn_alignment(
        self, matrix, scaled_margin, domains, sample_weight,
    ):
        sample_weight = np.maximum(
            np.asarray(sample_weight, dtype=float).reshape(-1), 1e-8)
        bins = np.digitize(
            scaled_margin,
            np.asarray(self.margin_thresholds, dtype=float),
            right=False,
        )
        overall = np.average(matrix, axis=0, weights=sample_weight)
        between = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
        domain_shift = np.zeros_like(between)
        source_within = np.zeros_like(between)
        global_centers = {}
        for margin_bin in sorted(set(int(value) for value in bins)):
            selected = bins == margin_bin
            selected_weight = sample_weight[selected]
            center = np.average(
                matrix[selected], axis=0, weights=selected_weight)
            global_centers[margin_bin] = center
            delta = center - overall
            between += float(np.sum(selected_weight)) * np.outer(delta, delta)
        joint_groups = np.asarray([
            f"{str(domain)}:{int(margin_bin)}"
            for domain, margin_bin in zip(domains, bins)
        ], dtype=object)
        for group in sorted(set(joint_groups.tolist())):
            selected = joint_groups == group
            margin_bin = int(str(group).rsplit(":", 1)[1])
            selected_weight = sample_weight[selected]
            center = np.average(
                matrix[selected], axis=0, weights=selected_weight)
            centered = matrix[selected] - center
            source_within += (
                centered.T @ (selected_weight[:, None] * centered))
            delta = center - global_centers[margin_bin]
            domain_shift += float(np.sum(selected_weight)) * np.outer(
                delta, delta)
        nuisance = (
            self.domain_penalty * domain_shift
            + self.within_bin_weight * source_within
        )
        components, eigenvalues = _generalized_components(
            between,
            nuisance,
            self.latent_dim,
            self.alignment_ridge,
        )
        projected = matrix @ components.T
        for index in range(projected.shape[1]):
            if float(np.corrcoef(projected[:, index], scaled_margin)[0, 1]) < 0.0:
                components[index] *= -1.0
                projected[:, index] *= -1.0
        return components, projected, bins, eigenvalues, domain_shift, between

    def _freeze_templates(self, profiles, scaled_margin, domains, bins, latent):
        templates = []
        for domain in sorted(set(str(value) for value in domains)):
            for margin_bin in sorted(set(int(value) for value in bins)):
                selected = np.flatnonzero(np.asarray([
                    str(value) == domain for value in domains
                ], dtype=bool) & (bins == margin_bin))
                if len(selected) == 0:
                    continue
                center = np.mean(latent[selected], axis=0)
                order = sorted(
                    selected.tolist(),
                    key=lambda index: (
                        float(np.linalg.norm(latent[index] - center)),
                        abs(float(scaled_margin[index])),
                        int(index),
                    ),
                )
                for index in order[: self.templates_per_stratum]:
                    templates.append({
                        "profile": np.asarray(
                            profiles[index], dtype=float).reshape(-1).copy(),
                        "domain": domain,
                        "margin_bin": int(margin_bin),
                        "scaled_chance_margin": float(scaled_margin[index]),
                        "phi": np.asarray(latent[index], dtype=float).copy(),
                    })
        self.boundary_profile_templates = templates

    def fit(
        self,
        inputs,
        boundary_margins,
        domains,
        sample_weight=None,
        *,
        coefficient_targets=None,
        proposal_profiles=None,
    ):
        inputs = list(inputs)
        boundary_margins = np.asarray(
            boundary_margins, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if (
            not inputs
            or len(inputs) != len(boundary_margins)
            or len(domains) != len(boundary_margins)
        ):
            raise ValueError("aligned boundary training rows must align")
        proposal_profiles = (
            inputs if proposal_profiles is None else list(proposal_profiles))
        if len(proposal_profiles) != len(inputs):
            raise ValueError("boundary proposal profiles must align")
        coefficient_targets = (
            boundary_margins
            if coefficient_targets is None
            else np.asarray(coefficient_targets, dtype=float).reshape(-1)
        )
        if len(coefficient_targets) != len(boundary_margins):
            raise ValueError("coefficient targets must align with boundary rows")
        weight = (
            np.ones(len(boundary_margins), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if len(weight) != len(boundary_margins):
            raise ValueError("aligned boundary weights must align")
        library = np.vstack([
            self._input_library(value) for value in inputs
        ])
        self.feature_mean = np.average(library, axis=0, weights=weight)
        centered = library - self.feature_mean
        self.feature_scale = np.sqrt(np.average(
            centered ** 2, axis=0, weights=weight))
        self.feature_scale = np.where(
            self.feature_scale < 1e-8, 1.0, self.feature_scale)
        matrix = centered / self.feature_scale
        scaled_margin, domain_scales = _domain_scaled(
            boundary_margins, domains)
        (
            self.alignment_components,
            latent,
            bins,
            eigenvalues,
            domain_shift,
            between,
        ) = self._learn_alignment(
            matrix, scaled_margin, domains, weight)
        self.latent_mean = np.average(latent, axis=0, weights=weight)
        latent_centered = latent - self.latent_mean
        self.latent_scale = np.sqrt(np.average(
            latent_centered ** 2, axis=0, weights=weight))
        self.latent_scale = np.where(
            self.latent_scale < 1e-8, 1.0, self.latent_scale)
        normalized_latent = latent_centered / self.latent_scale
        transformed_latent = self._select_latent_transform(
            normalized_latent,
            coefficient_targets,
            domains,
            weight,
        )
        expanded_latent = self._boundary_feature_library_many(
            transformed_latent)
        self.feature_dim = int(expanded_latent.shape[1])
        self.fit_status = "fit"
        self.models = []
        self.model_weights = None
        self.atom_components = None
        self._freeze_templates(
            proposal_profiles,
            scaled_margin,
            domains,
            bins,
            expanded_latent,
        )
        self._fit_source_parametric_prior(
            inputs,
            coefficient_targets,
            domains,
            weight,
        )
        raw_domain_ratio = float(np.trace(domain_shift)) / max(
            float(np.trace(between)), 1e-12)
        projected_groups = np.asarray([
            f"{str(domain)}:{int(margin_bin)}"
            for domain, margin_bin in zip(domains, bins)
        ], dtype=object)
        projected_within = float(np.trace(_scatter(
            transformed_latent, projected_groups)))
        projected_total = float(np.trace(_scatter(
            transformed_latent,
            np.zeros(len(normalized_latent), dtype=int),
        )))
        self.alignment_diagnostics = {
            "status": "fit",
            "coordinate": "phi=source_aligned_chance_boundary",
            "feature_dim": int(self.feature_dim),
            "alignment_latent_dim": int(normalized_latent.shape[1]),
            "boundary_feature_mode": self.feature_mode,
            "library_dim": int(matrix.shape[1]),
            "source_domains": sorted(set(str(value) for value in domains)),
            "source_record_count": int(len(inputs)),
            "margin_thresholds": list(self.margin_thresholds),
            "margin_bin_count": int(len(set(bins.tolist()))),
            "domain_margin_scales": domain_scales,
            "generalized_eigenvalues": eigenvalues.tolist(),
            "raw_domain_to_boundary_scatter_ratio": raw_domain_ratio,
            "projected_within_stratum_fraction": (
                projected_within / max(projected_total, 1e-12)
            ),
            "source_boundary_rank_correlation": _rank_correlation(
                transformed_latent[:, 0], scaled_margin),
            "latent_transform": self.latent_transform,
            "latent_transform_diagnostics": copy.deepcopy(
                self.latent_transform_diagnostics),
            "maximum_abs_source_latent_feature": float(np.max(
                np.abs(transformed_latent))),
            "boundary_profile_template_count": int(
                len(self.boundary_profile_templates)),
            "representation_training_target": "chance_margin_bin",
            "coefficient_prior_training_target": "constraint_mean",
            "input_mode": self.input_mode,
            "observable_descriptor_mode": self.observable_descriptor_mode,
            "channel_role_alignment": (
                None
                if self.channel_role_aligner is None
                else self.channel_role_aligner.diagnostics()
            ),
            "observable_state_exposure": bool(
                self.input_mode == "observable_state_exposure"),
            "source_learned_policy_proxy": bool(
                self.input_mode == "source_learned_exposure"),
            "provider_structural_input": bool(
                self.input_mode == "provider_exposure"),
            "target_data_used": False,
            "target_oracle_used": False,
        }
        self.source_prior_diagnostics.update({
            "coordinate": "phi_source_aligned_boundary",
            "representation_training_target": "chance_margin_bin",
            "coefficient_prior_training_target": "constraint_mean",
        })
        return self

    def _boundary_feature_library_many(self, latent):
        latent = _as_rows(latent)
        if self.feature_mode == "linear":
            return latent.copy()
        if self.feature_mode == "diagonal_quadratic":
            return np.hstack([latent, latent ** 2])
        interactions = np.column_stack([
            latent[:, left] * latent[:, right]
            for left in range(latent.shape[1])
            for right in range(left, latent.shape[1])
        ])
        return np.hstack([latent, interactions])

    def features_profile(self, profile):
        if self.fit_status != "fit" or self.alignment_components is None:
            raise RuntimeError("aligned boundary coordinate is not fit")
        library = self._input_library(profile)
        standardized = (library - self.feature_mean) / self.feature_scale
        latent = standardized @ self.alignment_components.T
        normalized = (latent - self.latent_mean) / self.latent_scale
        transformed = self._transform_normalized_latent(normalized)
        result = self._boundary_feature_library_many(transformed)[0]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("aligned boundary coordinate is non-finite")
        return np.asarray(result, dtype=float)

    def features(self, problem, x):
        return self.features_profile(self._problem_input(problem, x))

    def features_many(self, problem, points):
        if len(points) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(problem, point) for point in points])

    def diagnostics(self):
        alignment = dict(self.alignment_diagnostics)
        if self.channel_role_aligner is not None:
            alignment["channel_role_alignment"] = (
                self.channel_role_aligner.diagnostics())
        return {
            "status": self.fit_status,
            "feature_dim": int(self.feature_dim),
            "output_mode": "boundary_aligned",
            "input_mode": self.input_mode,
            "observable_descriptor_mode": self.observable_descriptor_mode,
            "boundary_feature_mode": self.feature_mode,
            "latent_transform": self.latent_transform,
            "latent_transform_diagnostics": copy.deepcopy(
                self.latent_transform_diagnostics),
            "alignment": alignment,
            "source_parametric_prior": dict(self.source_prior_diagnostics),
            "target_oracle_used": False,
            "target_labels_used_to_define_coordinate": False,
        }


class SourceSupportedRoleBoundaryCoordinate:
    """Select a role coordinate only where its cardinality is identifiable.

    Both candidate coordinates are fitted from the same source archive. The
    held-out target contributes only an unlabeled observable exposure pool.
    When its channel count was absent from every source domain, the learned
    role atlas cannot identify the missing-role semantics and the independent
    fallback coordinate is used instead.
    """

    def __init__(self, role_model, fallback_model, channel_role_aligner,
                 fallback_mode="ordered"):
        if int(role_model.feature_dim) != int(fallback_model.feature_dim):
            raise ValueError("role and fallback coordinates must align")
        self.role_model = role_model
        self.fallback_model = fallback_model
        self.channel_role_aligner = channel_role_aligner
        self.fallback_mode = str(fallback_mode)
        self.feature_dim = int(role_model.feature_dim)
        self.fit_status = "fit"
        self.input_mode = "observable_state_exposure"
        self.observable_descriptor_mode = (
            f"role_adaptive_{self.fallback_mode}")
        self.feature_mode = str(role_model.feature_mode)
        self.boundary_profile_templates = list(
            role_model.boundary_profile_templates)

    def selection(self, problem):
        support = self.channel_role_aligner.target_support_diagnostics(problem)
        use_role = bool(support["channel_cardinality_supported"])
        return {
            **support,
            "selected_coordinate": (
                "role_aligned" if use_role else self.fallback_mode),
            "fallback_mode": self.fallback_mode,
            "target_labels_used": False,
            "target_oracle_used": False,
        }

    def _model(self, problem):
        return (
            self.role_model
            if self.selection(problem)["channel_cardinality_supported"]
            else self.fallback_model
        )

    def features(self, problem, x):
        return self._model(problem).features(problem, x)

    def features_many(self, problem, points):
        return self._model(problem).features_many(problem, points)

    def source_parametric_prior(self, problem):
        prior = copy.deepcopy(
            self._model(problem).source_parametric_prior(problem))
        prior.setdefault("diagnostics", {})[
            "role_coordinate_selection"] = self.selection(problem)
        return prior

    def source_parametric_prior_components(self, problem):
        selection = self.selection(problem)
        components = copy.deepcopy(
            self._model(problem).source_parametric_prior_components(problem))
        for component in components:
            component.setdefault("diagnostics", {})[
                "role_coordinate_selection"] = copy.deepcopy(selection)
        return components

    def diagnostics(self):
        return {
            "status": "fit",
            "feature_dim": self.feature_dim,
            "output_mode": "boundary_aligned",
            "input_mode": self.input_mode,
            "observable_descriptor_mode": self.observable_descriptor_mode,
            "boundary_feature_mode": self.feature_mode,
            "role_coordinate": self.role_model.diagnostics(),
            "fallback_coordinate": self.fallback_model.diagnostics(),
            "fallback_mode": self.fallback_mode,
            "target_oracle_used": False,
            "target_labels_used_to_define_coordinate": False,
        }

    def diagnostics_for_problem(self, problem):
        return {
            **self.diagnostics(),
            "role_coordinate_selection": self.selection(problem),
            "selected_coordinate_diagnostics": self._model(
                problem).diagnostics(),
        }


@dataclass(frozen=True)
class BoundaryProposalSelection:
    indices: tuple[int, ...]
    roles: tuple[str, ...]
    diagnostics: dict


def select_boundary_coordinate_candidates(
    features,
    observed_features,
    posterior_mean,
    posterior_variance,
    chance_margin,
    *,
    count,
    safe_fraction=0.30,
    boundary_fraction=0.40,
    coverage_fraction=0.30,
):
    """Select safe, boundary, and under-covered points in one ``phi`` space."""

    features = _as_rows(features)
    observed_features = np.asarray(observed_features, dtype=float)
    if observed_features.size == 0:
        observed_features = np.empty((0, features.shape[1]), dtype=float)
    observed_features = _as_rows(observed_features) if len(observed_features) else observed_features
    posterior_mean = np.asarray(posterior_mean, dtype=float).reshape(-1)
    posterior_variance = np.maximum(
        np.asarray(posterior_variance, dtype=float).reshape(-1), 0.0)
    chance_margin = np.asarray(chance_margin, dtype=float).reshape(-1)
    n_rows = len(features)
    if not (
        len(posterior_mean) == n_rows
        and len(posterior_variance) == n_rows
        and len(chance_margin) == n_rows
    ):
        raise ValueError("boundary proposal arrays must align")
    count = min(max(int(count), 0), n_rows)
    if count == 0:
        return BoundaryProposalSelection((), (), {
            "status": "empty",
            "requested": int(count),
            "target_oracle_used": False,
        })

    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (features - center) / scale
    if len(observed_features):
        observed_normalized = (observed_features - center) / scale
        distance = np.sqrt(np.maximum(
            np.sum(normalized ** 2, axis=1)[:, None]
            + np.sum(observed_normalized ** 2, axis=1)[None, :]
            - 2.0 * normalized @ observed_normalized.T,
            0.0,
        ))
        coverage = np.min(distance, axis=1)
    else:
        coverage = np.linalg.norm(normalized, axis=1)
    uncertainty = np.sqrt(np.maximum(posterior_variance, 1e-12))
    boundary_score = np.abs(chance_margin) / np.maximum(uncertainty, 1e-6)
    safe_score = chance_margin

    fractions = np.maximum(np.asarray([
        safe_fraction, boundary_fraction, coverage_fraction
    ], dtype=float), 0.0)
    if float(np.sum(fractions)) <= 0.0:
        fractions[:] = 1.0
    fractions /= float(np.sum(fractions))
    raw_quota = fractions * count
    quota = np.floor(raw_quota).astype(int)
    for position in np.argsort(-(raw_quota - quota)):
        if int(np.sum(quota)) >= count:
            break
        quota[int(position)] += 1

    chosen = []
    roles = []

    def add_ranked(order, limit, role):
        candidates = [int(index) for index in order if int(index) not in chosen]
        if limit <= 0 or not candidates:
            return
        shortlist = candidates[: min(len(candidates), max(4 * limit, limit))]
        while len([value for value in roles if value == role]) < limit and shortlist:
            if not chosen:
                selected = shortlist[0]
            else:
                chosen_rows = normalized[np.asarray(chosen, dtype=int)]
                separation = np.asarray([
                    float(np.min(np.linalg.norm(
                        chosen_rows - normalized[index][None, :], axis=1)))
                    for index in shortlist
                ])
                separation /= max(float(np.max(separation)), 1e-12)
                rank_quality = np.linspace(1.0, 0.0, len(shortlist), endpoint=False)
                selected = shortlist[int(np.argmax(
                    rank_quality + 0.20 * separation))]
            chosen.append(int(selected))
            roles.append(str(role))
            shortlist.remove(int(selected))

    add_ranked(np.argsort(safe_score, kind="stable"), quota[0], "safe")
    add_ranked(np.argsort(boundary_score, kind="stable"), quota[1], "boundary")
    add_ranked(np.argsort(-coverage, kind="stable"), quota[2], "coverage")
    hybrid = boundary_score - 0.25 * coverage + 0.10 * np.maximum(chance_margin, 0.0)
    add_ranked(np.argsort(hybrid, kind="stable"), count - len(chosen), "fill")

    selected = np.asarray(chosen, dtype=int)
    diagnostics = {
        "status": "selected",
        "requested": int(count),
        "selected": int(len(selected)),
        "feature_dim": int(features.shape[1]),
        "pool_size": int(n_rows),
        "role_counts": {
            role: int(sum(value == role for value in roles))
            for role in sorted(set(roles))
        },
        "minimum_pool_chance_margin": float(np.min(chance_margin)),
        "minimum_selected_chance_margin": float(np.min(chance_margin[selected])),
        "median_selected_boundary_score": float(np.median(
            boundary_score[selected])),
        "median_selected_coverage_distance": float(np.median(
            coverage[selected])),
        "posterior_target_data_used": True,
        "coordinate_source_only": True,
        "target_oracle_used": False,
    }
    return BoundaryProposalSelection(
        tuple(int(value) for value in selected),
        tuple(roles),
        diagnostics,
    )
