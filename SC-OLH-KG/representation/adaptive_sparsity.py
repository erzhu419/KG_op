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
        self.feature_scale_ = np.std(X, axis=0)
        self.feature_scale_ = np.where(
            self.feature_scale_ < 1e-8, 1.0, self.feature_scale_)
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
            allowed_budget = (
                max(
                    self.min_pip * float(allowed_count),
                    min(
                        float(allowed_count),
                        max(1.0, self.max_effective_fraction * float(n)),
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

        def cardinality_project(logits):
            logits = np.asarray(logits, dtype=float)
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
            low, high = 0.0, 50.0
            for _ in range(60):
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
            proposed = cardinality_project(logit_source + log_bayes_factor)
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
            "spike_ratio": float(self.spike_ratio),
            "multiplicity_correction": float(self.multiplicity_correction),
            "max_effective_dimension": float(max_effective_dim),
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
