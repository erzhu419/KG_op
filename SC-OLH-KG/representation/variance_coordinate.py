"""Source-aligned cumulative-variance coordinates from observable exposure.

The constraint-mean and variance heads share the same target-observable input
``e(x)`` but are fitted independently.  This module learns the variance head
from ordinary replicated source simulations.  Its output is a nonnegative
``RiskExposure(A, N)`` consumed by cumulative factor-HVD.
"""

from __future__ import annotations

import numpy as np

from core.cumulative_risk import RiskExposure
from representation.boundary_coordinate import (
    SourceAlignedBoundaryCoordinate,
    _rank_correlation,
)
from representation.observable_exposure import (
    canonical_observable_state_descriptor,
    get_observable_state_exposure,
)


def _domain_standardized_log_variance(variance, domains, sample_weight):
    variance = np.maximum(np.asarray(variance, dtype=float).reshape(-1), 1e-12)
    domains = np.asarray(domains, dtype=object).reshape(-1)
    weight = np.maximum(
        np.asarray(sample_weight, dtype=float).reshape(-1), 1e-12)
    log_variance = np.log(variance)
    standardized = np.zeros(len(log_variance), dtype=float)
    centers = {}
    scales = {}
    for domain in sorted(set(str(value) for value in domains)):
        selected = np.asarray([
            str(value) == domain for value in domains
        ], dtype=bool)
        values = log_variance[selected]
        weights = weight[selected]
        center = float(np.average(values, weights=weights))
        scale = float(np.sqrt(np.average(
            (values - center) ** 2,
            weights=weights,
        )))
        scale = max(scale, 1e-6)
        standardized[selected] = (values - center) / scale
        centers[domain] = center
        scales[domain] = scale
    return standardized, log_variance, centers, scales


def _weighted_kmeans(values, count, weight, temperature, iterations=25):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("variance-coordinate latent rows must be nonempty")
    count = max(int(count), 1)
    weight = np.maximum(np.asarray(weight, dtype=float).reshape(-1), 1e-12)
    order = np.argsort(values[:, 0], kind="stable")
    positions = np.linspace(0, len(order) - 1, count)
    centers = values[order[np.rint(positions).astype(int)]].copy()
    for _ in range(max(int(iterations), 1)):
        d2 = np.sum(
            (values[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(d2, axis=1)
        updated = centers.copy()
        for index in range(count):
            selected = labels == index
            if np.any(selected):
                updated[index] = np.average(
                    values[selected],
                    axis=0,
                    weights=weight[selected],
                )
        if float(np.linalg.norm(updated - centers)) <= 1e-8:
            centers = updated
            break
        centers = updated
    # Keep regime labels stable from low to high source variance.
    centers = centers[np.argsort(centers[:, 0], kind="stable")]
    return centers, max(float(temperature), 1e-6)


class SourceAlignedVarianceRiskCoordinate:
    """Independent ``h_v(e)`` head producing cumulative HVD coordinates."""

    def __init__(
        self,
        local_dim=3,
        shared_dim=3,
        ridge_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        alignment_ridge=0.1,
        domain_penalty=1.0,
        within_bin_weight=0.1,
        soft_temperature=0.75,
    ):
        self.local_dim = max(int(local_dim), 1)
        self.shared_dim = max(int(shared_dim), 1)
        self.soft_temperature = max(float(soft_temperature), 1e-6)
        self.coordinate = SourceAlignedBoundaryCoordinate(
            ridge_grid=ridge_grid,
            latent_dim=self.local_dim,
            alignment_ridge=alignment_ridge,
            domain_penalty=domain_penalty,
            within_bin_weight=within_bin_weight,
            input_mode="observable_state_exposure",
        )
        self.cluster_centers = None
        self.fit_status = "unfit"
        self.training_diagnostics = {"status": "unfit"}

    def fit(self, inputs, variance_targets, domains, sample_weight=None):
        inputs = list(inputs)
        variance_targets = np.asarray(
            variance_targets, dtype=float).reshape(-1)
        domains = np.asarray(domains, dtype=object).reshape(-1)
        if not inputs or len(inputs) != len(variance_targets):
            raise ValueError("observable variance rows must align")
        if len(domains) != len(inputs):
            raise ValueError("observable variance domains must align")
        weight = (
            np.ones(len(inputs), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if len(weight) != len(inputs):
            raise ValueError("observable variance weights must align")
        standardized, log_variance, centers, scales = (
            _domain_standardized_log_variance(
                variance_targets,
                domains,
                weight,
            )
        )
        # Reuse the source-domain alignment machinery with variance strata as
        # supervision.  This object is independent of the chance-mean head.
        self.coordinate.fit(
            inputs,
            standardized,
            domains,
            sample_weight=weight,
            coefficient_targets=log_variance,
            proposal_profiles=inputs,
        )
        latent = np.vstack([
            self.coordinate.features_profile(value) for value in inputs
        ])
        self.cluster_centers, self.soft_temperature = _weighted_kmeans(
            latent,
            self.shared_dim,
            weight,
            self.soft_temperature,
        )
        self.fit_status = "fit"
        self.training_diagnostics = {
            "status": "fit",
            "coordinate": "psi_v=h_v(observable_state_exposure)",
            "input_mode": "observable_state_exposure",
            "representation_training_target": (
                "source_replicated_log_constraint_variance_strata"),
            "source_record_count": int(len(inputs)),
            "source_domains": sorted(set(str(value) for value in domains)),
            "local_dim": int(self.local_dim),
            "shared_dim": int(self.shared_dim),
            "domain_log_variance_centers": centers,
            "domain_log_variance_scales": scales,
            "source_variance_rank_correlation": _rank_correlation(
                latent[:, 0],
                standardized,
            ),
            "mean_head_parameters_shared": False,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        return self

    def latent_from_descriptor(self, descriptor):
        if self.fit_status != "fit" or self.cluster_centers is None:
            raise RuntimeError("observable variance coordinate is not fit")
        return np.asarray(
            self.coordinate.features_profile(descriptor), dtype=float
        ).reshape(-1)

    def risk_exposure_from_descriptor(self, descriptor):
        latent = self.latent_from_descriptor(descriptor)
        # Softplus keeps local cumulative exposures nonnegative while retaining
        # the source-learned ordering of each variance direction.
        A = np.logaddexp(0.0, latent) / np.log(2.0)
        d2 = np.sum(
            (self.cluster_centers - latent[None, :]) ** 2,
            axis=1,
        )
        logits = -d2 / self.soft_temperature
        logits -= float(np.max(logits))
        N = np.exp(np.clip(logits, -50.0, 0.0))
        N /= max(float(np.sum(N)), 1e-12)
        return RiskExposure(
            A,
            N,
            local_names=tuple(
                f"observable_variance_local_{index}"
                for index in range(self.local_dim)
            ),
            shared_names=tuple(
                f"observable_variance_regime_{index}"
                for index in range(self.shared_dim)
            ),
            meta={
                "provider": "SourceAlignedVarianceRiskCoordinate",
                "coordinate": "psi_v=h_v(e)",
                "input_mode": "observable_state_exposure",
                "source_only": True,
                "target_data_used": False,
                "target_oracle_used": False,
            },
        )

    def risk_exposure(self, problem, x):
        exposure = get_observable_state_exposure(problem, x)
        if exposure is None:
            raise ValueError(
                "observable variance coordinate requires an "
                "observable_state_exposure adapter")
        return self.risk_exposure_from_descriptor(
            canonical_observable_state_descriptor(exposure))

    def diagnostics(self):
        return {
            **dict(self.training_diagnostics),
            "head": "variance_hvd",
            "fit_status": self.fit_status,
            "mean_head_parameters_shared": False,
        }
