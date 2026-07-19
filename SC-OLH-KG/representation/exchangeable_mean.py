"""Permutation-equivariant linear posterior for target constraint means.

The source archive learns an exchangeable coefficient hyperprior.  It does
not decide that target channel ``j`` has the semantics of a particular source
channel.  Charged target observations learn the channel-specific signs and
magnitudes while the prior remains invariant to a simultaneous relabeling of
channels and coefficients.
"""

from __future__ import annotations

import copy

import numpy as np

from representation.observable_exposure import (
    MAX_OBSERVABLE_CHANNELS,
    as_observable_state_exposure,
    get_observable_state_exposure,
)


def _nearest_psd(matrix, floor):
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, float(floor))) @ vectors.T


class ExchangeableBoundaryMeanCoordinate:
    """Low-dimensional target-linear mean head with an exchangeable prior.

    Each observable channel contributes ``(mean, mean^2, scale^2)``.  One
    global dispersion term captures variation between and within channels.
    Source-domain fits are reduced to a distribution over channel blocks;
    the same block law is assigned to every target channel.  Thus source data
    learn shrinkage and effect scales, while target data learn channel roles.
    """

    channel_block_dim = 3
    global_dim = 1

    def __init__(
        self,
        ridge_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        covariance_floor=1e-6,
    ):
        grid = tuple(sorted(set(
            max(float(value), 1e-10) for value in ridge_grid
        )))
        if not grid:
            raise ValueError("exchangeable mean coordinate needs a ridge grid")
        self.ridge_grid = grid
        self.covariance_floor = max(float(covariance_floor), 1e-12)
        self.feature_dim = (
            MAX_OBSERVABLE_CHANNELS * self.channel_block_dim
            + self.global_dim
        )
        self.fit_status = "unfit"
        self.input_mode = "observable_state_exposure"
        self.observable_descriptor_mode = "exchangeable_equivariant"
        self.feature_mode = "linear"
        self.latent_transform = "identity"
        self.boundary_profile_templates = []
        self.block_mean = None
        self.block_scale = None
        self.global_mean = None
        self.global_scale = None
        self.domain_rows = []
        self.fit_diagnostics = {"status": "unfit"}

    @property
    def _global_index(self):
        return MAX_OBSERVABLE_CHANNELS * self.channel_block_dim

    @staticmethod
    def _raw_channel_blocks(exposure):
        exposure = as_observable_state_exposure(exposure)
        if exposure is None:
            raise ValueError("exchangeable mean coordinate needs an exposure")
        means = np.asarray(exposure.channel_means, dtype=float)
        scales = np.asarray(exposure.channel_scales, dtype=float)
        if len(means) <= 0 or len(means) > MAX_OBSERVABLE_CHANNELS:
            raise ValueError("observable channel count is outside the atlas")
        blocks = np.column_stack([means, means ** 2, scales ** 2])
        profile_scale = (
            float(exposure.dynamics[1])
            if len(exposure.dynamics) > 1
            else float(np.sqrt(np.mean(scales ** 2)))
        )
        return blocks, np.asarray([profile_scale ** 2], dtype=float)

    @staticmethod
    def _solve(features, target, weight, ridge):
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        weight = np.maximum(
            np.asarray(weight, dtype=float).reshape(-1), 1e-10)
        design = np.column_stack([np.ones(len(features)), features])
        root = np.sqrt(weight)
        weighted = design * root[:, None]
        penalty = float(ridge) * np.eye(design.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        precision = weighted.T @ weighted + penalty
        coefficient = np.linalg.pinv(precision) @ (
            weighted.T @ (target * root))
        return coefficient, precision

    def _select_ridge(self, features, target, weight):
        n_rows = int(len(features))
        if n_rows < 8:
            return self.ridge_grid[len(self.ridge_grid) // 2], None
        n_folds = min(5, n_rows)
        fold = np.arange(n_rows, dtype=int) % n_folds
        best = None
        for ridge in self.ridge_grid:
            prediction = np.zeros(n_rows, dtype=float)
            for index in range(n_folds):
                test = fold == index
                train = ~test
                coefficient, _ = self._solve(
                    features[train], target[train], weight[train], ridge)
                prediction[test] = np.column_stack([
                    np.ones(int(np.sum(test))), features[test]
                ]) @ coefficient
            boundary_weight = 1.0 + np.exp(-np.abs(target))
            false_safe = (prediction <= 0.0) & (target > 0.0)
            loss = float(np.average(
                boundary_weight * (prediction - target) ** 2
                + 4.0 * false_safe.astype(float),
                weights=weight,
            ))
            candidate = (loss, float(ridge))
            if best is None or candidate < best:
                best = candidate
        return best[1], best[0]

    def _standardized_features_from_exposure(self, exposure):
        if self.fit_status not in {"fitting", "fit"}:
            raise RuntimeError("exchangeable mean coordinate is not fit")
        blocks, global_values = self._raw_channel_blocks(exposure)
        standardized = (
            blocks - self.block_mean[None, :]
        ) / self.block_scale[None, :]
        result = np.zeros(self.feature_dim, dtype=float)
        for channel, row in enumerate(standardized):
            start = channel * self.channel_block_dim
            result[start:start + self.channel_block_dim] = row
        result[self._global_index:] = (
            global_values - self.global_mean
        ) / self.global_scale
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "exchangeable boundary features are non-finite")
        return result

    def fit(self, exposures, targets, domains, sample_weight=None):
        exposures = [as_observable_state_exposure(value) for value in exposures]
        target = np.asarray(targets, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if (
            not exposures
            or any(value is None for value in exposures)
            or len(exposures) != len(target)
            or len(domains) != len(target)
        ):
            raise ValueError("exchangeable source rows must align")
        weight = (
            np.ones(len(target), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if len(weight) != len(target):
            raise ValueError("exchangeable source weights must align")
        weight = np.maximum(weight, 1e-10)

        raw = [self._raw_channel_blocks(value) for value in exposures]
        block_rows = np.vstack([value[0] for value in raw])
        block_weights = np.concatenate([
            np.full(len(value[0]), float(row_weight), dtype=float)
            for value, row_weight in zip(raw, weight)
        ])
        block_weights /= float(np.sum(block_weights))
        self.block_mean = np.sum(
            block_weights[:, None] * block_rows, axis=0)
        self.block_scale = np.sqrt(np.sum(
            block_weights[:, None]
            * (block_rows - self.block_mean[None, :]) ** 2,
            axis=0,
        ))
        self.block_scale = np.where(
            self.block_scale > 1e-8, self.block_scale, 1.0)
        global_rows = np.vstack([value[1] for value in raw])
        normalized_weight = weight / float(np.sum(weight))
        self.global_mean = np.sum(
            normalized_weight[:, None] * global_rows, axis=0)
        self.global_scale = np.sqrt(np.sum(
            normalized_weight[:, None]
            * (global_rows - self.global_mean[None, :]) ** 2,
            axis=0,
        ))
        self.global_scale = np.where(
            self.global_scale > 1e-8, self.global_scale, 1.0)

        self.fit_status = "fitting"
        features = np.vstack([
            self._standardized_features_from_exposure(value)
            for value in exposures
        ])
        domain_rows = []
        for domain in sorted(set(str(value) for value in domains)):
            selected = np.asarray([
                str(value) == domain for value in domains
            ], dtype=bool)
            counts = sorted(set(
                len(exposures[index].channel_means)
                for index in np.flatnonzero(selected)
            ))
            if len(counts) != 1:
                raise ValueError("source domain channel count must be stable")
            channel_count = int(counts[0])
            active_feature = []
            for channel in range(channel_count):
                start = channel * self.channel_block_dim
                active_feature.extend(range(
                    start, start + self.channel_block_dim))
            active_feature.append(self._global_index)
            active_feature = np.asarray(active_feature, dtype=int)
            domain_features = features[selected][:, active_feature]
            domain_target = target[selected]
            domain_weight = weight[selected]
            ridge, validation_loss = self._select_ridge(
                domain_features, domain_target, domain_weight)
            coefficient, precision = self._solve(
                domain_features, domain_target, domain_weight, ridge)
            design = np.column_stack([
                np.ones(len(domain_features)), domain_features
            ])
            residual = domain_target - design @ coefficient
            residual_variance = max(float(np.average(
                residual ** 2, weights=domain_weight)), 1e-6)
            covariance = residual_variance * np.linalg.pinv(precision)
            covariance = _nearest_psd(covariance, self.covariance_floor)

            full_coefficient = np.zeros(1 + self.feature_dim, dtype=float)
            full_coefficient[0] = coefficient[0]
            full_indices = np.concatenate([
                np.asarray([0], dtype=int), 1 + active_feature
            ])
            full_coefficient[1 + active_feature] = coefficient[1:]
            full_covariance = self.covariance_floor * np.eye(
                1 + self.feature_dim, dtype=float)
            full_covariance[np.ix_(full_indices, full_indices)] = covariance
            block_coefficients = np.vstack([
                full_coefficient[
                    1 + channel * self.channel_block_dim:
                    1 + (channel + 1) * self.channel_block_dim
                ]
                for channel in range(channel_count)
            ])
            block_covariances = [
                full_covariance[np.ix_(
                    np.arange(
                        1 + channel * self.channel_block_dim,
                        1 + (channel + 1) * self.channel_block_dim,
                    ),
                    np.arange(
                        1 + channel * self.channel_block_dim,
                        1 + (channel + 1) * self.channel_block_dim,
                    ),
                )]
                for channel in range(channel_count)
            ]
            block_mean = np.mean(block_coefficients, axis=0)
            block_between = np.mean([
                np.outer(value - block_mean, value - block_mean)
                for value in block_coefficients
            ], axis=0)
            block_covariance = (
                np.mean(block_covariances, axis=0) + block_between)
            block_covariance = _nearest_psd(
                block_covariance, self.covariance_floor)
            common_indices = np.asarray(
                [0, 1 + self._global_index], dtype=int)
            domain_rows.append({
                "domain": domain,
                "channel_count": channel_count,
                "common_mean": full_coefficient[common_indices],
                "common_covariance": _nearest_psd(
                    full_covariance[np.ix_(common_indices, common_indices)],
                    self.covariance_floor,
                ),
                "block_mean": block_mean,
                "block_covariance": block_covariance,
                "residual_variance": residual_variance,
                "ridge": float(ridge),
                "validation_loss": (
                    None if validation_loss is None
                    else float(validation_loss)),
                "reliability": 1.0 / max(
                    float(validation_loss)
                    if validation_loss is not None
                    else residual_variance,
                    1e-8,
                ),
                "n_records": int(np.sum(selected)),
            })
        if not domain_rows:
            raise RuntimeError("exchangeable source fit produced no domain")
        self.domain_rows = domain_rows
        self.fit_status = "fit"
        self.fit_diagnostics = {
            "status": "fit",
            "coordinate": "exchangeable_equivariant_boundary_linear",
            "feature_dim": int(self.feature_dim),
            "channel_block_dim": int(self.channel_block_dim),
            "maximum_channels": int(MAX_OBSERVABLE_CHANNELS),
            "channel_features": ["mean", "mean_square", "scale_square"],
            "global_features": ["profile_scale_square"],
            "source_domains": [row["domain"] for row in domain_rows],
            "source_domain_count": int(len(domain_rows)),
            "source_record_count": int(len(target)),
            "source_channel_counts": {
                row["domain"]: int(row["channel_count"])
                for row in domain_rows
            },
            "role_semantics": "target_posterior_coefficients",
            "source_role_identity_transferred": False,
            "source_hyperparameters_transferred": [
                "channel_block_mean",
                "channel_block_covariance",
                "ridge_scale",
                "residual_variance",
            ],
            "permutation_equivariant": True,
            "coefficient_prior_training_target": "constraint_mean",
            "representation_training_target": "observable_channel_exposure",
            "target_data_used": False,
            "target_oracle_used": False,
            "domain_fits": [
                {
                    key: row[key]
                    for key in (
                        "domain", "channel_count", "ridge",
                        "validation_loss", "residual_variance", "reliability",
                        "n_records",
                    )
                }
                for row in domain_rows
            ],
        }
        return self

    def features_profile(self, exposure):
        return self._standardized_features_from_exposure(exposure)

    def features(self, problem, x):
        exposure = get_observable_state_exposure(problem, x)
        if exposure is None:
            raise ValueError("target problem has no observable state exposure")
        return self.features_profile(exposure)

    def features_many(self, problem, points):
        if len(points) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(problem, point) for point in points])

    @staticmethod
    def _target_channel_count(problem):
        lo, _ = problem.int_bounds()
        exposure = get_observable_state_exposure(
            problem, tuple(int(value) for value in lo))
        if exposure is None:
            raise ValueError("target problem has no observable state exposure")
        count = int(len(exposure.channel_means))
        if count <= 0 or count > MAX_OBSERVABLE_CHANNELS:
            raise ValueError("target channel count is outside the atlas")
        return count

    def _target_component(self, row, problem, prior_weight):
        channel_count = self._target_channel_count(problem)
        coefficient = np.zeros(1 + self.feature_dim, dtype=float)
        coefficient[0] = float(row["common_mean"][0])
        for channel in range(channel_count):
            start = 1 + channel * self.channel_block_dim
            coefficient[start:start + self.channel_block_dim] = row[
                "block_mean"]
        coefficient[1 + self._global_index] = float(row["common_mean"][1])

        covariance = self.covariance_floor * np.eye(
            1 + self.feature_dim, dtype=float)
        common_indices = np.asarray(
            [0, 1 + self._global_index], dtype=int)
        covariance[np.ix_(common_indices, common_indices)] = row[
            "common_covariance"]
        for channel in range(channel_count):
            indices = np.arange(
                1 + channel * self.channel_block_dim,
                1 + (channel + 1) * self.channel_block_dim,
            )
            covariance[np.ix_(indices, indices)] = row["block_covariance"]
        covariance = _nearest_psd(covariance, self.covariance_floor)

        tau = float(getattr(problem, "tau", 0.0))
        output_scale = max(
            abs(tau),
            float(getattr(problem, "sigma_level", 0.0)),
            1e-6,
        )
        scaled_mean = output_scale * coefficient
        scaled_mean[0] += tau
        return {
            "name": f"source:{row['domain']}",
            "domain": str(row["domain"]),
            "mean": scaled_mean,
            "covariance": output_scale ** 2 * covariance,
            "deviation_variance": max(
                output_scale ** 2 * float(row["residual_variance"]),
                1e-8,
            ),
            "prior_weight": float(prior_weight),
            "diagnostics": {
                "component_kind": "exchangeable_source_hyperprior",
                "domain": str(row["domain"]),
                "source_channel_count": int(row["channel_count"]),
                "target_channel_count": int(channel_count),
                "ridge": float(row["ridge"]),
                "validation_loss": row["validation_loss"],
                "source_residual_variance": float(
                    row["residual_variance"]),
                "permutation_equivariant": True,
                "source_role_identity_transferred": False,
                "target_channel_roles_learned_from_charged_data": True,
                "target_output_scale": float(output_scale),
                "target_tau": float(tau),
                "target_data_used": False,
                "target_oracle_used": False,
            },
        }

    def source_parametric_prior_components(self, problem):
        if self.fit_status != "fit":
            raise RuntimeError("exchangeable source prior is unavailable")
        reliability = np.asarray([
            max(float(row["reliability"]), 1e-12)
            for row in self.domain_rows
        ], dtype=float)
        reliability /= float(np.sum(reliability))
        return [
            self._target_component(row, problem, mass)
            for row, mass in zip(self.domain_rows, reliability)
        ]

    def source_parametric_prior(self, problem):
        components = self.source_parametric_prior_components(problem)
        weights = np.asarray([
            float(component["prior_weight"]) for component in components
        ], dtype=float)
        weights /= float(np.sum(weights))
        means = np.vstack([component["mean"] for component in components])
        mean = np.sum(weights[:, None] * means, axis=0)
        covariance = np.zeros((len(mean), len(mean)), dtype=float)
        for weight, component, component_mean in zip(
            weights, components, means
        ):
            delta = component_mean - mean
            covariance += float(weight) * (
                component["covariance"] + np.outer(delta, delta))
        covariance = _nearest_psd(
            covariance,
            self.covariance_floor
            * max(float(getattr(problem, "sigma_level", 0.0)) ** 2, 1e-12),
        )
        deviation = float(np.sum(weights * np.asarray([
            component["deviation_variance"] for component in components
        ], dtype=float)))
        return {
            "mean": mean,
            "covariance": covariance,
            "deviation_variance": max(deviation, 1e-8),
            "prior_weight": 1.0,
            "diagnostics": {
                **copy.deepcopy(self.fit_diagnostics),
                "prior_kind": (
                    "exchangeable_empirical_bayes_gaussian_hyperlaw"),
                "target_task_law": "single_gaussian_draw",
                "source_domain_identity_marginalized": True,
                "source_components_retained_in_target_posterior": False,
                "within_source_covariance_included": True,
                "between_source_covariance_included": True,
                "target_channel_count": self._target_channel_count(problem),
                "source_component_count": int(len(components)),
                "prior_covariance_trace": float(np.trace(covariance)),
                "prior_covariance_min_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(covariance))),
                "target_data_used": False,
                "target_oracle_used": False,
            },
        }

    def diagnostics_for_problem(self, problem):
        return {
            **self.diagnostics(),
            "target_channel_count": self._target_channel_count(problem),
        }

    def diagnostics(self):
        return copy.deepcopy(self.fit_diagnostics)
