"""Source-only aleatoric heads for controlled HVD backend comparisons.

Both heads consume the same ordinary replicated source archive.  The pooled
head transfers one variance scale.  The cumulative-factor head transfers a
dimension-equivariant risk shape in the common ``psi=(A,N)`` coordinate.
Neither head reads held-out target outcomes, target oracle values, or terminal
verification labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.stats import chi2

from core.cumulative_risk import (
    CumulativeRiskFeatureProvider,
    RiskExposure,
)
from variance.orthogonal_hvd import OrthogonalHVD


SOURCE_HVD_COORDINATE_ID = "dimension_equivariant_profile_psi_v1"
SOURCE_HVD_CALIBRATION_ID = "source_task_lodo_chi_square_upper_v1"
VALID_SOURCE_HVD_MODES = {"pooled", "cumulative_factor"}


class DimensionEquivariantRiskProvider(CumulativeRiskFeatureProvider):
    """Label-free profile coordinate with fixed dimension for every raw D."""

    def __init__(self, proxy_scale=10000):
        self.proxy_scale = max(100, int(proxy_scale))
        self.d = 1

    def _profile(self, x):
        values = np.asarray(x, dtype=float).reshape(-1)
        if len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("risk coordinate requires a finite policy profile")
        return np.clip(values / float(self.proxy_scale), 0.0, 1.0)

    @staticmethod
    def _cosine_amplitude(values, frequency):
        positions = (
            np.arange(len(values), dtype=float) + 0.5
        ) / float(len(values))
        centered = values - float(np.mean(values))
        return float(abs(2.0 * np.mean(
            centered * np.cos(np.pi * int(frequency) * positions)
        )))

    def risk_exposures(self, x, output_index=1):
        del output_index
        profile = self._profile(x)
        differences = np.diff(profile)
        local_roughness = (
            float(np.sqrt(np.mean(differences ** 2)))
            if len(differences) else 0.0
        )
        local_dispersion = float(np.std(profile))
        local_interior = float(np.sqrt(np.mean(
            np.maximum(profile * (1.0 - profile), 0.0)
        )))
        shared_mean = float(np.mean(profile))
        shared_low_1 = self._cosine_amplitude(profile, 1)
        shared_low_2 = self._cosine_amplitude(profile, 2)
        return RiskExposure(
            A=np.asarray([
                local_roughness,
                local_dispersion,
                local_interior,
            ]),
            N=np.asarray([
                shared_mean,
                shared_low_1,
                shared_low_2,
            ]),
            local_names=(
                "local_roughness",
                "local_dispersion",
                "local_interior",
            ),
            shared_names=(
                "global_mean",
                "low_frequency_1",
                "low_frequency_2",
            ),
            meta={
                "coordinate_id": SOURCE_HVD_COORDINATE_ID,
                "target_labels_used": False,
                "target_oracle_used": False,
            },
        )

    def _reference_risk_exposure(self):
        return self.risk_exposures(
            np.full(8, self.proxy_scale // 2, dtype=int))

    def normalize(self, x):
        return self._profile(x)

    def risk_class(self, x):
        mean = float(np.mean(self._profile(x)))
        if mean < 1.0 / 3.0:
            return 0
        if mean < 2.0 / 3.0:
            return 1
        return 2


@dataclass(frozen=True)
class SourceArchiveHVDConfig:
    mode: str
    output_index: int = 1
    proxy_scale: int = 10000
    calibration_delta: float = 0.05
    calibration_quantile: float = 0.95
    ridge_alpha: float = 1e-2
    certification_kappa: float = 1.0

    def __post_init__(self):
        if self.mode not in VALID_SOURCE_HVD_MODES:
            raise ValueError(f"unknown source HVD mode {self.mode!r}")
        if not 0.0 < float(self.calibration_delta) < 1.0:
            raise ValueError("calibration_delta must lie in (0, 1)")
        if not 0.0 < float(self.calibration_quantile) <= 1.0:
            raise ValueError("calibration_quantile must lie in (0, 1]")


class FrozenSourceArchiveAleatoricHead:
    """Frozen source-trained variance head consumed by a BO backend."""

    def __init__(
        self,
        *,
        archive,
        target_problem,
        config: SourceArchiveHVDConfig,
    ):
        archive.validate()
        self.archive = archive
        self.target_problem = target_problem
        self.config = config
        self.provider = DimensionEquivariantRiskProvider(
            proxy_scale=config.proxy_scale)
        self.target_variance_scale = max(
            float(getattr(target_problem, "sigma_level", 0.04)) ** 2,
            1e-12,
        )
        self._task_scales = {
            task.name: self._task_variance_scale(task)
            for task in archive.tasks
        }
        self._lodo_rows = self._source_task_lodo_rows()
        ratios = np.asarray([
            row["upper_variance_ratio"] for row in self._lodo_rows
            if np.isfinite(row["upper_variance_ratio"])
        ], dtype=float)
        if len(ratios) == 0:
            raise RuntimeError("source HVD calibration produced no finite rows")
        self.calibration_multiplier = float(max(
            1.0,
            np.quantile(
                ratios,
                float(config.calibration_quantile),
                method="higher",
            ),
        ))
        self.model = self._fit_tasks(archive.tasks)
        contract_payload = {
            "coordinate": SOURCE_HVD_COORDINATE_ID,
            "calibration": SOURCE_HVD_CALIBRATION_ID,
            "archive": str(archive.fingerprint),
            "mode": str(config.mode),
            "output_index": int(config.output_index),
            "delta": float(config.calibration_delta),
            "quantile": float(config.calibration_quantile),
        }
        self.contract_id = (
            f"source_archive_{config.mode}_"
            + hashlib.sha256(json.dumps(
                contract_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()[:16]
        )

    @staticmethod
    def _task_variance_scale(task):
        values = np.asarray(
            task.replicate_variance[:, 1], dtype=float).reshape(-1)
        finite = values[np.isfinite(values) & (values >= 0.0)]
        if len(finite) == 0:
            raise ValueError(f"source task {task.name!r} has no variances")
        return float(max(np.mean(finite), 1e-12))

    def _proxy_many(self, normalized_profiles):
        values = np.asarray(normalized_profiles, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return np.rint(np.clip(values, 0.0, 1.0)
                       * float(self.config.proxy_scale)).astype(int)

    def _target_proxy(self, x):
        normalized = np.asarray(
            self.target_problem.normalize(x), dtype=float).reshape(1, -1)
        return self._proxy_many(normalized)[0]

    def _fit_tasks(self, tasks):
        mode = (
            "pooled" if self.config.mode == "pooled" else "factor")
        model = OrthogonalHVD(
            mode=mode,
            n_outputs=2,
            ridge_alpha=float(self.config.ridge_alpha),
            activation_min_records=8,
            certification_kappa=float(self.config.certification_kappa),
            use_cumulative_provider=True,
        )
        profiles = []
        relative_variances = []
        replicate_counts = []
        for task in tasks:
            scale = self._task_scales[task.name]
            profiles.extend(self._proxy_many(task.X))
            relative_variances.extend(
                np.maximum(
                    np.asarray(task.replicate_variance[:, 1], dtype=float),
                    0.0,
                ) / scale
            )
            replicate_counts.extend(
                len(row) for row in task.Y_replicates)
        model.fit_from_variances(
            profiles,
            relative_variances,
            output_index=int(self.config.output_index),
            problem=self.provider,
            replicate_counts=replicate_counts,
            replace=True,
        )
        return model

    def _source_task_lodo_rows(self):
        rows = []
        tasks = tuple(self.archive.tasks)
        for heldout in tasks:
            training = tuple(task for task in tasks if task.name != heldout.name)
            if not training:
                training = tasks
            model = self._fit_tasks(training)
            points = self._proxy_many(heldout.X)
            predicted = model.predict_variance_many(
                int(self.config.output_index),
                points,
                problem=self.provider,
            )
            scale = self._task_scales[heldout.name]
            observed = np.maximum(
                np.asarray(
                    heldout.replicate_variance[:, 1], dtype=float),
                0.0,
            ) / scale
            for index, (sample, prediction, replicates) in enumerate(zip(
                observed,
                predicted,
                heldout.Y_replicates,
            )):
                dof = max(int(len(replicates)) - 1, 1)
                denominator = max(
                    float(chi2.ppf(
                        float(self.config.calibration_delta), dof)),
                    1e-12,
                )
                upper = float(dof * sample / denominator)
                rows.append({
                    "heldout_source_task": str(heldout.name),
                    "profile_index": int(index),
                    "replication_dof": int(dof),
                    "observed_relative_variance": float(sample),
                    "predicted_relative_variance": float(max(
                        prediction, 1e-12)),
                    "chi_square_upper_relative_variance": upper,
                    "upper_variance_ratio": float(
                        upper / max(float(prediction), 1e-12)),
                })
        return rows

    def predict_variance(self, x):
        relative = self.model.predict_variance(
            int(self.config.output_index),
            self._target_proxy(x),
            problem=self.provider,
        )
        return float(max(
            self.target_variance_scale * relative,
            1e-12,
        ))

    def predict_certification_variance(self, x):
        return float(max(
            self.predict_variance(x) * self.calibration_multiplier,
            1e-12,
        ))

    def predict_certification_variance_many(self, points):
        relative = self.model.predict_variance_many(
            int(self.config.output_index),
            [self._target_proxy(point) for point in points],
            problem=self.provider,
        )
        return np.maximum(
            relative
            * self.target_variance_scale
            * self.calibration_multiplier,
            1e-12,
        )

    def predict_decomposition(self, x):
        payload = self.model.predict_decomposition(
            int(self.config.output_index),
            self._target_proxy(x),
            problem=self.provider,
        )
        cumulative = payload.get("cumulative")
        if isinstance(cumulative, dict):
            for key in (
                "fitted_variance",
                "certification_variance",
                "independent",
                "shared",
                "linear",
                "floor",
                "total",
            ):
                if cumulative.get(key) is not None:
                    cumulative[key] = float(
                        cumulative[key] * self.target_variance_scale)
        return payload

    def diagnostics(self):
        ratios = np.asarray([
            row["upper_variance_ratio"] for row in self._lodo_rows
        ], dtype=float)
        return {
            "status": "frozen",
            "contract_id": self.contract_id,
            "mode": str(self.config.mode),
            "coordinate_id": SOURCE_HVD_COORDINATE_ID,
            "calibration_id": SOURCE_HVD_CALIBRATION_ID,
            "source_archive_fingerprint": str(self.archive.fingerprint),
            "source_domains": list(self.archive.source_domains),
            "source_simulator_calls": int(self.archive.simulator_calls),
            "target_outcomes_used": False,
            "target_oracle_used": False,
            "terminal_verifier_labels_used": False,
            "target_variance_scale": float(self.target_variance_scale),
            "source_task_relative_scales": dict(self._task_scales),
            "lodo_calibration_row_count": int(len(self._lodo_rows)),
            "lodo_upper_ratio_median": float(np.median(ratios)),
            "lodo_upper_ratio_quantile": float(np.quantile(
                ratios,
                float(self.config.calibration_quantile),
                method="higher",
            )),
            "calibration_multiplier": float(
                self.calibration_multiplier),
            "hvd": self.model.diagnostics(),
        }

