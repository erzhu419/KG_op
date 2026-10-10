"""Adaptive source-to-target coefficient sparsity.

The source domains provide only a hyper-prior over inclusion probabilities and
coefficient scales.  A held-out target updates those probabilities from its own
ordinary observations with a variational spike-and-slab posterior.  The
dictionary remains fixed, so posterior refits do not invalidate candidate
features or caches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip_probability(value, lower, upper):
    return np.clip(np.asarray(value, dtype=float), float(lower), float(upper))


def _resize_vector(value, size, fill):
    src = np.asarray(value, dtype=float).reshape(-1)
    out = np.full(int(size), float(fill), dtype=float)
    count = min(len(src), len(out))
    if count:
        out[:count] = src[:count]
    return out


@dataclass
class AdaptiveSparsePosteriorResult:
    """Posterior in the raw GPR basis, including visited-point deviations."""

    mean: np.ndarray
    covariance: np.ndarray
    sampled_set: list[tuple[int, ...]]
    sol_to_idx: dict[tuple[int, ...], int]
    posterior_pip: np.ndarray
    diagnostics: dict


class AdaptiveSpikeSlabPosterior:
    """Variational spike-and-slab posterior over a frozen feature dictionary.

    The conditional Gaussian posterior is recomputed from all target
    observations whenever a new evaluation arrives.  This avoids recursively
    shrinking an already-updated covariance, which would count old target data
    more than once.  Visited-solution deviations are included in the same
    linear system as the sparse parametric coefficients.
    """

    def __init__(
        self,
        source_pip,
        source_slab_scale,
        *,
        min_pip=0.05,
        max_pip=0.95,
        spike_ratio=0.05,
        damping=0.5,
        max_iter=40,
        tolerance=1e-5,
        residual_floor_scale=0.05,
        intercept_variance=10.0,
        multiplicity_correction=1.0,
        max_effective_fraction=0.35,
        always_active_count=0,
        allowed_mask=None,
        shared_shrinkage_groups=None,
    ):
        if not 0.0 < float(min_pip) < float(max_pip) < 1.0:
            raise ValueError("adaptive PIP bounds must satisfy 0 < min < max < 1")
        if not 0.0 < float(spike_ratio) < 1.0:
            raise ValueError("spike_ratio must be in (0, 1)")
        self.source_pip = np.asarray(source_pip, dtype=float).reshape(-1)
        self.source_slab_scale = np.asarray(
            source_slab_scale, dtype=float).reshape(-1)
        self.min_pip = float(min_pip)
        self.max_pip = float(max_pip)
        self.spike_ratio = float(spike_ratio)
        self.damping = float(np.clip(damping, 0.0, 0.99))
        self.max_iter = max(1, int(max_iter))
        self.tolerance = max(float(tolerance), 1e-12)
        self.residual_floor_scale = max(float(residual_floor_scale), 0.0)
        self.intercept_variance = max(float(intercept_variance), 1e-8)
        self.multiplicity_correction = max(
            float(multiplicity_correction), 0.0)
        self.max_effective_fraction = float(np.clip(
            max_effective_fraction, 0.05, 1.0))
        self.always_active_count = max(0, int(always_active_count))
        self.allowed_mask = (
            None
            if allowed_mask is None
            else np.asarray(allowed_mask, dtype=bool).reshape(-1)
        )
        self.shared_shrinkage_groups = (
            None
            if shared_shrinkage_groups is None
            else np.asarray(
                shared_shrinkage_groups, dtype=int).reshape(-1)
        )

        self.result_: AdaptiveSparsePosteriorResult | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.response_mean_: float = 0.0
        self.response_scale_: float = 1.0
        self.slab_variance_: np.ndarray | None = None
        self.spike_variance_: np.ndarray | None = None
        self.posterior_pip_: np.ndarray | None = None
        self.standardized_mean_: np.ndarray | None = None
        self.standardized_covariance_: np.ndarray | None = None

    def fit(
        self,
        features,
        response,
        noise_variance,
        sample_keys,
        *,
        deviation_variance,
    ):
        X = np.asarray(features, dtype=float)
        y = np.asarray(response, dtype=float).reshape(-1)
        noise = np.asarray(noise_variance, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("features must be 2-D with one row per response")
        if len(y) == 0:
            raise ValueError("adaptive sparsity needs at least one observation")
        if len(noise) == 1 and len(y) > 1:
            noise = np.full(len(y), float(noise[0]), dtype=float)
        if len(noise) != len(y) or len(sample_keys) != len(y):
            raise ValueError("noise and sample_keys must align with responses")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("adaptive sparsity inputs must be finite")

        n, m = X.shape
        self.feature_mean_ = np.mean(X, axis=0)
        raw_feature_scale = np.std(X, axis=0)
        self.feature_scale_ = raw_feature_scale.copy()
        self.feature_scale_ = np.where(
            self.feature_scale_ < 1e-8, 1.0, self.feature_scale_)
        group_ids = np.full(m, -1, dtype=int)
        if self.shared_shrinkage_groups is not None:
            count = min(len(self.shared_shrinkage_groups), m)
            group_ids[:count] = self.shared_shrinkage_groups[:count]
            for group in sorted(set(group_ids[group_ids >= 0].tolist())):
                members = np.flatnonzero(group_ids == group)
                common_scale = float(np.sqrt(np.mean(
                    raw_feature_scale[members] ** 2)))
                self.feature_scale_[members] = (
                    common_scale if common_scale >= 1e-8 else 1.0
                )
        Z = (X - self.feature_mean_) / self.feature_scale_

        self.response_mean_ = float(np.mean(y))
        empirical_scale = float(np.std(y))
        noise_scale = float(np.sqrt(max(float(np.median(noise)), 1e-12)))
        self.response_scale_ = max(empirical_scale, noise_scale, 1e-6)
        target = (y - self.response_mean_) / self.response_scale_
        noise_std = np.maximum(noise / self.response_scale_ ** 2, 1e-8)

        sampled_set = []
        sol_to_idx = {}
        for key in sample_keys:
            normalized_key = tuple(int(v) for v in key)
            if normalized_key not in sol_to_idx:
                sol_to_idx[normalized_key] = len(sampled_set)
                sampled_set.append(normalized_key)
        E = np.zeros((n, len(sampled_set)), dtype=float)
        for row, key in enumerate(sample_keys):
            E[row, sol_to_idx[tuple(int(v) for v in key)]] = 1.0
        design = np.column_stack([np.ones(n, dtype=float), Z, E])

        source_pip = _resize_vector(self.source_pip, m, self.min_pip)
        source_pip = _clip_probability(
            source_pip, self.min_pip, self.max_pip)
        slab_scale = _resize_vector(self.source_slab_scale, m, 0.5)
        slab_scale = np.clip(slab_scale, 0.10, 5.0)
        for group in sorted(set(group_ids[group_ids >= 0].tolist())):
            members = np.flatnonzero(group_ids == group)
            shared_pip = float(np.mean(source_pip[members]))
            shared_scale = float(np.sqrt(np.mean(
                slab_scale[members] ** 2)))
            source_pip[members] = shared_pip
            slab_scale[members] = shared_scale
        slab_var = np.maximum(slab_scale ** 2, 1e-6)
        spike_var = np.maximum(
            self.spike_ratio ** 2 * slab_var,
            1e-8,
        )
        self.slab_variance_ = slab_var
        self.spike_variance_ = spike_var

        deviation_var_std = max(
            float(deviation_variance) / self.response_scale_ ** 2,
            1e-6,
        )
        likelihood_precision = 1.0 / noise_std
        weighted_design = design * likelihood_precision[:, None]
        data_precision = design.T @ weighted_design
        data_rhs = design.T @ (likelihood_precision * target)

        fixed_count = min(self.always_active_count, m)
        adaptive_count = m - fixed_count
        allowed = np.ones(m, dtype=bool)
        if self.allowed_mask is not None:
            count = min(len(self.allowed_mask), m)
            allowed[:] = False
            allowed[:count] = self.allowed_mask[:count]
        if fixed_count:
            allowed[:fixed_count] = True
        adaptive_allowed = allowed[fixed_count:]
        disallowed_indices = np.arange(fixed_count, m)[~adaptive_allowed]
        allowed_count = int(np.sum(adaptive_allowed))
        disallowed_count = adaptive_count - allowed_count
        if adaptive_count:
            adaptive_floor = self.min_pip * float(adaptive_count)
            total_budget = max(
                float(fixed_count),
                max(1.0, self.max_effective_fraction * float(n)),
            )
            optional_budget = max(total_budget - float(fixed_count), 0.0)
            allowed_budget = (
                max(
                    self.min_pip * float(allowed_count),
                    min(
                        float(allowed_count),
                        optional_budget,
                    ),
                )
                if allowed_count else 0.0
            )
            max_adaptive_dim = (
                self.min_pip * float(disallowed_count) + allowed_budget
            )
            max_adaptive_dim = max(adaptive_floor, max_adaptive_dim)
        else:
            max_adaptive_dim = 0.0
        max_effective_dim = float(fixed_count) + max_adaptive_dim
        shared_groups = {}
        for group in sorted(set(group_ids[group_ids >= 0].tolist())):
            members = np.flatnonzero(
                (group_ids == group)
                & allowed
                & (np.arange(m) >= fixed_count)
            )
            if len(members):
                shared_groups[int(group)] = members

        def cardinality_project(logits):
            logits = np.nan_to_num(
                np.asarray(logits, dtype=float),
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            logits = np.clip(logits, -1e6, 1e6)
            probability = _clip_probability(
                1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0))),
                self.min_pip,
                self.max_pip,
            )
            if fixed_count:
                probability[:fixed_count] = 1.0
            probability[disallowed_indices] = self.min_pip
            if float(np.sum(probability[fixed_count:])) <= max_adaptive_dim:
                return probability
            adaptive_logits = logits[fixed_count:]
            high = max(
                50.0,
                float(np.max(adaptive_logits)) + 50.0,
            )
            low = 0.0
            for _ in range(80):
                shift = 0.5 * (low + high)
                candidate_adaptive = _clip_probability(
                    1.0 / (
                        1.0 + np.exp(-np.clip(
                            logits[fixed_count:] - shift,
                            -35.0,
                            35.0,
                        ))
                    ),
                    self.min_pip,
                    self.max_pip,
                )
                candidate_adaptive[~adaptive_allowed] = self.min_pip
                if float(np.sum(candidate_adaptive)) > max_adaptive_dim:
                    low = shift
                else:
                    high = shift
            projected = probability.copy()
            projected[fixed_count:] = _clip_probability(
                1.0 / (
                    1.0 + np.exp(-np.clip(
                        logits[fixed_count:] - high,
                        -35.0,
                        35.0,
                    ))
                ),
                self.min_pip,
                self.max_pip,
            )
            projected[disallowed_indices] = self.min_pip
            if fixed_count:
                projected[:fixed_count] = 1.0
            return projected

        logit_source = np.log(source_pip) - np.log1p(-source_pip)
        if shared_groups:
            grouped = set(np.concatenate(
                list(shared_groups.values())).tolist())
            ungrouped = [
                index
                for index in range(fixed_count, m)
                if index not in grouped and allowed[index]
            ]
            selection_units = len(shared_groups) + len(ungrouped)
            correction = self.multiplicity_correction * np.log(
                max(selection_units, 2))
            logit_source[fixed_count:] -= correction
        else:
            logit_source[fixed_count:] -= (
                self.multiplicity_correction * np.log(max(adaptive_count, 2))
            )
        if fixed_count:
            logit_source[:fixed_count] = 35.0
        logit_source[~allowed] = -35.0
        pip = cardinality_project(logit_source)
        converged = False
        posterior_mean = np.zeros(design.shape[1], dtype=float)
        posterior_cov = np.eye(design.shape[1], dtype=float)
        log_bayes_factor = np.zeros(m, dtype=float)
        group_log_bayes_factor = {}

        def conditional_posterior(current_pip):
            coefficient_precision = (
                current_pip / slab_var
                + (1.0 - current_pip) / spike_var
            )
            if fixed_count:
                coefficient_precision[:fixed_count] = (
                    1.0 / slab_var[:fixed_count]
                )
            prior_precision = np.concatenate([
                [1.0 / self.intercept_variance],
                coefficient_precision,
                np.full(len(sampled_set), 1.0 / deviation_var_std),
            ])
            precision = data_precision + np.diag(prior_precision)
            try:
                covariance = np.linalg.inv(precision)
            except np.linalg.LinAlgError:
                covariance = np.linalg.pinv(precision)
            covariance = 0.5 * (covariance + covariance.T)
            mean = covariance @ data_rhs
            return mean, covariance

        for iteration in range(self.max_iter):
            posterior_mean, posterior_cov = conditional_posterior(pip)
            coefficient_mean = posterior_mean[1:1 + m]
            global_prediction = (
                posterior_mean[0] + Z @ coefficient_mean
            )
            selection_precision = 1.0 / np.maximum(
                noise_std + deviation_var_std,
                1e-8,
            )
            feature_energy = np.sum(
                selection_precision[:, None] * Z ** 2,
                axis=0,
            )
            partial_residual = (
                target[:, None]
                - global_prediction[:, None]
                + Z * coefficient_mean[None, :]
            )
            feature_score = np.sum(
                selection_precision[:, None] * Z * partial_residual,
                axis=0,
            )
            slab_denom = 1.0 + slab_var * feature_energy
            spike_denom = 1.0 + spike_var * feature_energy
            log_bayes_factor = (
                -0.5 * np.log(slab_denom / spike_denom)
                + 0.5 * feature_score ** 2 * (
                    slab_var / slab_denom
                    - spike_var / spike_denom
                )
            )
            selection_log_bayes_factor = log_bayes_factor.copy()
            group_log_bayes_factor = {}
            for group, members in shared_groups.items():
                group_design = Z[:, members]
                group_mean = coefficient_mean[members]
                group_residual = (
                    target
                    - global_prediction
                    + group_design @ group_mean
                )
                gram = group_design.T @ (
                    selection_precision[:, None] * group_design)
                rhs = group_design.T @ (
                    selection_precision * group_residual)

                def log_marginal(prior_variance):
                    matrix = (
                        np.eye(len(members), dtype=float)
                        + float(prior_variance) * gram
                    )
                    sign, logdet = np.linalg.slogdet(matrix)
                    if sign <= 0.0:
                        return -np.inf
                    try:
                        solved = np.linalg.solve(matrix, rhs)
                    except np.linalg.LinAlgError:
                        solved = np.linalg.pinv(matrix) @ rhs
                    return (
                        -0.5 * float(logdet)
                        + 0.5 * float(prior_variance)
                        * float(rhs @ solved)
                    )

                shared_slab = float(slab_var[members[0]])
                shared_spike = float(spike_var[members[0]])
                group_bf = (
                    log_marginal(shared_slab)
                    - log_marginal(shared_spike)
                )
                group_log_bayes_factor[str(group)] = float(group_bf)
                selection_log_bayes_factor[members] = group_bf
            proposed = cardinality_project(
                logit_source + selection_log_bayes_factor)
            updated = self.damping * pip + (1.0 - self.damping) * proposed
            delta = float(np.max(np.abs(updated - pip)))
            pip = updated
            if delta <= self.tolerance:
                converged = True
                break
        posterior_mean, posterior_cov = conditional_posterior(pip)
        self.posterior_pip_ = pip.copy()
        self.standardized_mean_ = posterior_mean.copy()
        self.standardized_covariance_ = posterior_cov.copy()

        total_dim = 1 + m + len(sampled_set)
        transform = np.zeros((total_dim, total_dim), dtype=float)
        slope_scale = self.response_scale_ / self.feature_scale_
        transform[0, 0] = self.response_scale_
        transform[0, 1:1 + m] = -self.feature_mean_ * slope_scale
        transform[1:1 + m, 1:1 + m] = np.diag(slope_scale)
        if sampled_set:
            transform[1 + m:, 1 + m:] = (
                self.response_scale_ * np.eye(len(sampled_set), dtype=float)
            )
        offset = np.zeros(total_dim, dtype=float)
        offset[0] = self.response_mean_
        mean_raw = offset + transform @ posterior_mean
        covariance_raw = transform @ posterior_cov @ transform.T
        covariance_raw = 0.5 * (covariance_raw + covariance_raw.T)

        train_floor = self.mask_uncertainty(X)
        diagnostics = {
            "status": "fit",
            "method": "variational_spike_slab_bma",
            "inclusion_update": "collapsed_gaussian_bayes_factor",
            "n_observations": int(n),
            "n_unique_solutions": int(len(sampled_set)),
            "dictionary_dim": int(m),
            "iterations": int(iteration + 1),
            "converged": bool(converged),
            "source_pip": source_pip.tolist(),
            "posterior_pip": pip.tolist(),
            "effective_dimension": float(np.sum(pip)),
            "active_count_0_5": int(np.sum(pip >= 0.5)),
            "always_active_count": int(fixed_count),
            "allowed_adaptive_count": int(allowed_count),
            "frozen_out_adaptive_count": int(disallowed_count),
            "adaptive_effective_dimension": float(
                np.sum(pip[fixed_count:])),
            "adaptive_active_count_0_5": int(
                np.sum(pip[fixed_count:] >= 0.5)),
            "escaped_source_spike_count": int(np.sum(
                (source_pip <= 0.25) & (pip >= 0.5))),
            "source_supported_count": int(np.sum(source_pip >= 0.5)),
            "max_log_bayes_factor": float(np.max(log_bayes_factor)),
            "median_log_bayes_factor": float(np.median(log_bayes_factor)),
            "shared_shrinkage_active": bool(shared_groups),
            "shared_shrinkage_groups": {
                str(group): {
                    "indices": members.tolist(),
                    "source_pip": float(source_pip[members[0]]),
                    "posterior_pip": float(pip[members[0]]),
                    "slab_scale": float(slab_scale[members[0]]),
                    "log_bayes_factor": group_log_bayes_factor.get(
                        str(group)),
                    "effective_dimension": float(
                        len(members) * pip[members[0]]),
                }
                for group, members in shared_groups.items()
            },
            "spike_ratio": float(self.spike_ratio),
            "multiplicity_correction": float(self.multiplicity_correction),
            "max_effective_dimension": float(max_effective_dim),
            "effective_dimension_budget_slack": float(
                max_effective_dim - np.sum(pip)),
            "effective_dimension_budget_respected": bool(
                np.sum(pip) <= max_effective_dim + 1e-10),
            "residual_floor_scale": float(self.residual_floor_scale),
            "r_perp_train_mean": float(np.mean(train_floor)),
            "r_perp_train_max": float(np.max(train_floor)),
            "response_scale": float(self.response_scale_),
        }
        self.result_ = AdaptiveSparsePosteriorResult(
            mean=mean_raw,
            covariance=covariance_raw,
            sampled_set=sampled_set,
            sol_to_idx=sol_to_idx,
            posterior_pip=pip.copy(),
            diagnostics=diagnostics,
        )
        return self

    def predict_parametric_mean(self, features):
        if self.result_ is None:
            raise RuntimeError("adaptive sparsity posterior is not fit")
        X = np.asarray(features, dtype=float)
        one = X.ndim == 1
        if one:
            X = X[None, :]
        p = X.shape[1]
        beta = self.result_.mean[:1 + p]
        out = np.column_stack([np.ones(len(X)), X]) @ beta
        return float(out[0]) if one else out

    def mask_uncertainty(self, features):
        """Between-mask uncertainty retained as the orthogonal residual R_perp."""

        if self.posterior_pip_ is None:
            X = np.asarray(features, dtype=float)
            return 0.0 if X.ndim == 1 else np.zeros(len(X), dtype=float)
        X = np.asarray(features, dtype=float)
        one = X.ndim == 1
        if one:
            X = X[None, :]
        Z = (X - self.feature_mean_) / self.feature_scale_
        mask_variance = (
            self.posterior_pip_
            * (1.0 - self.posterior_pip_)
            * np.maximum(self.slab_variance_ - self.spike_variance_, 0.0)
        )
        out = (
            self.residual_floor_scale
            * self.response_scale_ ** 2
            * np.sum(Z ** 2 * mask_variance[None, :], axis=1)
        )
        out = np.maximum(out, 0.0)
        return float(out[0]) if one else out

    def diagnostics(self):
        if self.result_ is None:
            return {"status": "unfit", "method": "variational_spike_slab_bma"}
        return dict(self.result_.diagnostics)


class AdaptiveGroupRidgePosterior:
    """Nested-LOO empirical Bayes over semantic group ridge penalties.

    Every coefficient direction remains continuous.  Target observations
    choose only one isotropic precision per declared semantic group.  Full
    nested refits, rather than an interpolation-region hat-matrix shortcut,
    score the finite penalty grid.  Exact-KG fantasy clones run the same update,
    so complexity-parameter value of information is retained.
    """

    def __init__(
        self,
        group_ids,
        *,
        penalty_grid=(1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0),
        initial_feature_penalty=None,
        coordinate_passes=2,
        safety_weight=2.0,
        residual_floor_scale=0.05,
        intercept_variance=10.0,
    ):
        self.group_ids = np.asarray(group_ids, dtype=int).reshape(-1)
        grid = sorted({max(float(value), 1e-8) for value in penalty_grid})
        if not grid:
            raise ValueError("group ridge penalty grid must be nonempty")
        self.penalty_grid = np.asarray(grid, dtype=float)
        self.initial_feature_penalty = (
            None
            if initial_feature_penalty is None
            else np.asarray(initial_feature_penalty, dtype=float).reshape(-1)
        )
        self.coordinate_passes = max(1, int(coordinate_passes))
        self.safety_weight = max(float(safety_weight), 0.0)
        self.residual_floor_scale = max(float(residual_floor_scale), 0.0)
        self.intercept_variance = max(float(intercept_variance), 1e-8)

        self.result_: AdaptiveSparsePosteriorResult | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.response_mean_: float = 0.0
        self.response_scale_: float = 1.0
        self.selected_group_penalties_: dict[int, float] = {}
        self.selected_feature_penalty_: np.ndarray | None = None
        self.loo_loss_: float | None = None
        self.loo_residual_variance_: float = 0.0
        self.n_observations_: int = 0

    def _resolved_groups(self, feature_dim):
        groups = np.full(int(feature_dim), -1, dtype=int)
        count = min(len(self.group_ids), int(feature_dim))
        if count:
            groups[:count] = self.group_ids[:count]
        next_group = int(np.max(groups)) + 1 if np.any(groups >= 0) else 0
        for index in np.flatnonzero(groups < 0):
            groups[index] = next_group
            next_group += 1
        return groups

    @staticmethod
    def _group_standardization(features, groups):
        X = np.asarray(features, dtype=float)
        mean = np.mean(X, axis=0)
        raw_scale = np.std(X, axis=0)
        scale = raw_scale.copy()
        for group in sorted(set(groups.tolist())):
            members = np.flatnonzero(groups == group)
            common = float(np.sqrt(np.mean(raw_scale[members] ** 2)))
            scale[members] = common if common >= 1e-8 else 1.0
        return mean, scale

    @staticmethod
    def _solve(precision, rhs):
        try:
            return np.linalg.solve(precision, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(precision) @ rhs

    def _feature_penalty(self, groups, group_penalties):
        return np.asarray([
            float(group_penalties[int(group)]) for group in groups
        ], dtype=float)

    def _ridge_predict_fold(
        self,
        train_x,
        train_y,
        train_noise,
        test_x,
        groups,
        group_penalties,
    ):
        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float).reshape(-1)
        train_noise = np.maximum(
            np.asarray(train_noise, dtype=float).reshape(-1), 1e-12)
        test_x = np.asarray(test_x, dtype=float)
        mean, scale = self._group_standardization(train_x, groups)
        Z = (train_x - mean) / scale
        T = (test_x - mean) / scale
        response_mean = float(np.mean(train_y))
        response_scale = max(
            float(np.std(train_y)),
            float(np.sqrt(np.median(train_noise))),
            1e-6,
        )
        target = (train_y - response_mean) / response_scale
        noise = np.maximum(train_noise / response_scale ** 2, 1e-8)
        design = np.column_stack([np.ones(len(Z)), Z])
        weights = 1.0 / noise
        penalty = self._feature_penalty(groups, group_penalties)
        prior_precision = np.concatenate([
            [1.0 / self.intercept_variance], penalty,
        ])
        precision = design.T @ (weights[:, None] * design)
        precision += np.diag(prior_precision)
        rhs = design.T @ (weights * target)
        beta = self._solve(precision, rhs)
        standardized = np.column_stack([np.ones(len(T)), T]) @ beta
        return response_mean + response_scale * standardized

    def _nested_loo(self, X, y, noise, groups, group_penalties):
        predictions = np.empty(len(y), dtype=float)
        for heldout in range(len(y)):
            train = np.arange(len(y)) != heldout
            predictions[heldout] = self._ridge_predict_fold(
                X[train], y[train], noise[train],
                X[heldout:heldout + 1], groups, group_penalties,
            )[0]
        scale = max(float(np.std(y)), 1e-6)
        residual = (y - predictions) / scale
        loss = float(np.mean(
            residual ** 2
            + self.safety_weight * np.maximum(residual, 0.0) ** 2
        ))
        return loss, predictions

    def _effective_df(self, X, y, noise, groups, group_penalties):
        mean, scale = self._group_standardization(X, groups)
        Z = (X - mean) / scale
        response_scale = max(
            float(np.std(y)),
            float(np.sqrt(max(float(np.median(noise)), 1e-12))),
            1e-6,
        )
        standardized_noise = np.asarray(noise, dtype=float) / response_scale ** 2
        weights = 1.0 / np.maximum(standardized_noise, 1e-8)
        gram = Z.T @ (weights[:, None] * Z)
        penalty = self._feature_penalty(groups, group_penalties)
        return float(np.trace(
            gram @ np.linalg.pinv(gram + np.diag(penalty))))

    def _initial_penalties(self, groups):
        selected = {}
        for group in sorted(set(groups.tolist())):
            members = np.flatnonzero(groups == group)
            if self.initial_feature_penalty is None:
                value = 1.0
            else:
                source = _resize_vector(
                    self.initial_feature_penalty, len(groups), 1.0)
                value = float(np.exp(np.mean(np.log(np.maximum(
                    source[members], 1e-8)))))
            selected[int(group)] = float(self.penalty_grid[
                np.argmin(np.abs(np.log(self.penalty_grid) - np.log(value)))
            ])
        return selected

    def _select_penalties(self, X, y, noise, groups):
        starts = [self._initial_penalties(groups)]
        if self.selected_group_penalties_:
            starts.append({
                group: float(self.selected_group_penalties_.get(group, 1.0))
                for group in sorted(set(groups.tolist()))
            })
        starts.append({
            int(group): 1.0 for group in sorted(set(groups.tolist()))
        })
        cache = {}

        def evaluate(candidate):
            key = tuple(
                float(candidate[group])
                for group in sorted(candidate)
            )
            if key not in cache:
                loss, predictions = self._nested_loo(
                    X, y, noise, groups, candidate)
                effective_df = self._effective_df(
                    X, y, noise, groups, candidate)
                cache[key] = (loss, effective_df, predictions)
            return cache[key]

        solutions = []
        for initial in starts:
            selected = dict(initial)
            for _ in range(self.coordinate_passes):
                changed = False
                for group in sorted(selected):
                    candidates = []
                    for value in self.penalty_grid:
                        proposal = dict(selected)
                        proposal[group] = float(value)
                        loss, effective_df, _ = evaluate(proposal)
                        candidates.append((
                            float(loss),
                            float(effective_df),
                            float(value),
                            proposal,
                        ))
                    best = min(candidates, key=lambda row: row[:3])
                    if best[2] != selected[group]:
                        changed = True
                    selected = best[3]
                if not changed:
                    break
            loss, effective_df, predictions = evaluate(selected)
            solutions.append((
                float(loss), float(effective_df),
                tuple(selected[group] for group in sorted(selected)),
                selected, predictions,
            ))
        best = min(solutions, key=lambda row: row[:3])
        return best[3], best[0], best[1], best[4], len(cache)

    def fit(
        self,
        features,
        response,
        noise_variance,
        sample_keys,
        *,
        deviation_variance,
    ):
        X = np.asarray(features, dtype=float)
        y = np.asarray(response, dtype=float).reshape(-1)
        noise = np.asarray(noise_variance, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("features must be 2-D with one row per response")
        if len(y) < 3:
            raise ValueError("group ridge needs at least three observations")
        if len(noise) == 1 and len(y) > 1:
            noise = np.full(len(y), float(noise[0]), dtype=float)
        if len(noise) != len(y) or len(sample_keys) != len(y):
            raise ValueError("noise and sample_keys must align with responses")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("group ridge inputs must be finite")

        n, m = X.shape
        groups = self._resolved_groups(m)
        selected, loo_loss, effective_df, loo_prediction, models_tested = (
            self._select_penalties(X, y, noise, groups)
        )
        self.selected_group_penalties_ = dict(selected)
        self.selected_feature_penalty_ = self._feature_penalty(
            groups, selected)
        self.loo_loss_ = float(loo_loss)
        self.loo_residual_variance_ = float(np.mean(
            (y - loo_prediction) ** 2))
        self.n_observations_ = int(n)

        self.feature_mean_, self.feature_scale_ = (
            self._group_standardization(X, groups))
        Z = (X - self.feature_mean_) / self.feature_scale_
        self.response_mean_ = float(np.mean(y))
        self.response_scale_ = max(
            float(np.std(y)),
            float(np.sqrt(max(float(np.median(noise)), 1e-12))),
            1e-6,
        )
        target = (y - self.response_mean_) / self.response_scale_
        noise_std = np.maximum(noise / self.response_scale_ ** 2, 1e-8)

        sampled_set = []
        sol_to_idx = {}
        for key in sample_keys:
            normalized = tuple(int(value) for value in key)
            if normalized not in sol_to_idx:
                sol_to_idx[normalized] = len(sampled_set)
                sampled_set.append(normalized)
        E = np.zeros((n, len(sampled_set)), dtype=float)
        for row, key in enumerate(sample_keys):
            E[row, sol_to_idx[tuple(int(value) for value in key)]] = 1.0
        design = np.column_stack([np.ones(n), Z, E])
        deviation_var_std = max(
            float(deviation_variance) / self.response_scale_ ** 2, 1e-6)
        prior_precision = np.concatenate([
            [1.0 / self.intercept_variance],
            self.selected_feature_penalty_,
            np.full(len(sampled_set), 1.0 / deviation_var_std),
        ])
        likelihood_precision = 1.0 / noise_std
        precision = design.T @ (likelihood_precision[:, None] * design)
        precision += np.diag(prior_precision)
        rhs = design.T @ (likelihood_precision * target)
        covariance = np.linalg.pinv(precision)
        covariance = 0.5 * (covariance + covariance.T)
        mean = covariance @ rhs

        total_dim = 1 + m + len(sampled_set)
        transform = np.zeros((total_dim, total_dim), dtype=float)
        slope_scale = self.response_scale_ / self.feature_scale_
        transform[0, 0] = self.response_scale_
        transform[0, 1:1 + m] = -self.feature_mean_ * slope_scale
        transform[1:1 + m, 1:1 + m] = np.diag(slope_scale)
        if sampled_set:
            transform[1 + m:, 1 + m:] = (
                self.response_scale_ * np.eye(len(sampled_set)))
        offset = np.zeros(total_dim, dtype=float)
        offset[0] = self.response_mean_
        mean_raw = offset + transform @ mean
        covariance_raw = transform @ covariance @ transform.T
        covariance_raw = 0.5 * (covariance_raw + covariance_raw.T)

        group_diagnostics = {
            str(group): {
                "indices": np.flatnonzero(groups == group).tolist(),
                "selected_penalty": float(selected[int(group)]),
            }
            for group in sorted(set(groups.tolist()))
        }
        train_floor = self.mask_uncertainty(X)
        diagnostics = {
            "status": "fit",
            "method": "nested_loo_group_ridge",
            "selection_data": "charged_target_observations",
            "oracle_used": False,
            "n_observations": int(n),
            "n_unique_solutions": int(len(sampled_set)),
            "dictionary_dim": int(m),
            "group_count": int(len(group_diagnostics)),
            "groups": group_diagnostics,
            "penalty_grid": self.penalty_grid.tolist(),
            "coordinate_passes": int(self.coordinate_passes),
            "models_tested": int(models_tested),
            "nested_loo_loss": float(loo_loss),
            "nested_loo_residual_variance": float(
                self.loo_residual_variance_),
            "effective_dimension": float(effective_df),
            "max_effective_dimension": None,
            "complexity_selection_valid": bool(
                np.isfinite(loo_loss)
                and 0.0 <= effective_df <= float(m) + 1e-8
            ),
            "residual_floor_scale": float(self.residual_floor_scale),
            "r_perp_train_mean": float(np.mean(train_floor)),
            "r_perp_train_max": float(np.max(train_floor)),
            "response_scale": float(self.response_scale_),
        }
        self.result_ = AdaptiveSparsePosteriorResult(
            mean=mean_raw,
            covariance=covariance_raw,
            sampled_set=sampled_set,
            sol_to_idx=sol_to_idx,
            posterior_pip=np.empty(0, dtype=float),
            diagnostics=diagnostics,
        )
        return self

    def predict_parametric_mean(self, features):
        if self.result_ is None:
            raise RuntimeError("group ridge posterior is not fit")
        X = np.asarray(features, dtype=float)
        one = X.ndim == 1
        if one:
            X = X[None, :]
        p = X.shape[1]
        beta = self.result_.mean[:1 + p]
        out = np.column_stack([np.ones(len(X)), X]) @ beta
        return float(out[0]) if one else out

    def mask_uncertainty(self, features):
        X = np.asarray(features, dtype=float)
        one = X.ndim == 1
        if one:
            X = X[None, :]
        if self.feature_mean_ is None:
            out = np.zeros(len(X), dtype=float)
        else:
            Z = (X - self.feature_mean_) / self.feature_scale_
            novelty = np.sum(Z ** 2, axis=1) / max(
                self.n_observations_, 1)
            out = (
                self.residual_floor_scale
                * self.loo_residual_variance_
                * (1.0 + novelty)
            )
        out = np.maximum(out, 0.0)
        return float(out[0]) if one else out

    def diagnostics(self):
        if self.result_ is None:
            return {"status": "unfit", "method": "nested_loo_group_ridge"}
        return dict(self.result_.diagnostics)
