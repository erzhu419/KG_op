"""Source-only aleatoric heads for controlled HVD backend comparisons.

Every head consumes the same ordinary replicated source archive.  The pooled
head transfers one variance scale.  The legacy cumulative-factor head uses a
raw-profile proxy.  The provider-cumulative head fits the actual observable
``psi=(A,N)`` blocks supplied by each task schema.  No head reads held-out
target outcomes, target oracle values, or terminal verification labels.
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
    cumulative_feature_names,
    cumulative_feature_vector,
    cumulative_layout,
    decompose_cumulative_risk,
    get_risk_exposure,
    project_cumulative_beta,
)
from variance.orthogonal_hvd import OrthogonalHVD


SOURCE_HVD_COORDINATE_ID = "dimension_equivariant_profile_psi_v1"
SOURCE_HVD_CALIBRATION_ID = "source_task_lodo_chi_square_upper_v1"
PROVIDER_SOURCE_HVD_COORDINATE_ID = "observable_provider_psi_A_N_v1"
PROVIDER_SOURCE_HVD_CALIBRATION_ID = (
    "source_task_lodo_aggregate_chi_square_upper_v1")
VALID_SOURCE_HVD_MODES = {
    "pooled",
    "cumulative_factor",
    "provider_cumulative_factor",
}


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
    provider_ridge_per_source_row: float = 1.0
    certification_kappa: float = 1.0

    def __post_init__(self):
        if self.mode not in VALID_SOURCE_HVD_MODES:
            raise ValueError(f"unknown source HVD mode {self.mode!r}")
        if not 0.0 < float(self.calibration_delta) < 1.0:
            raise ValueError("calibration_delta must lie in (0, 1)")
        if not 0.0 < float(self.calibration_quantile) <= 1.0:
            raise ValueError("calibration_quantile must lie in (0, 1]")
        if float(self.provider_ridge_per_source_row) <= 0.0:
            raise ValueError(
                "provider_ridge_per_source_row must be positive")


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
        self._provider_mode = (
            config.mode == "provider_cumulative_factor")
        self.provider = (
            target_problem
            if self._provider_mode
            else DimensionEquivariantRiskProvider(
                proxy_scale=config.proxy_scale)
        )
        self.target_variance_scale = max(
            float(getattr(target_problem, "sigma_level", 0.04)) ** 2,
            1e-12,
        )
        self._task_scales = {
            task.name: self._task_variance_scale(task)
            for task in archive.tasks
        }
        self._provider_task_cache = {}
        self._provider_target_exposure = None
        if self._provider_mode:
            self._provider_target_exposure = self._target_reference_exposure()
            self._lodo_rows = self._provider_source_task_lodo_rows()
        else:
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
        if self._provider_mode:
            # The provider gate uses the maximum simultaneous source-task
            # scale upper bound.  The per-row quantile above is retained only
            # for the legacy profile heads.
            self.calibration_multiplier = float(max(
                1.0,
                np.max(ratios),
            ))
            self.model = None
            self._provider_fit = self._fit_provider_tasks(archive.tasks)
        else:
            self.model = self._fit_tasks(archive.tasks)
            self._provider_fit = None
        coordinate_id = (
            PROVIDER_SOURCE_HVD_COORDINATE_ID
            if self._provider_mode
            else SOURCE_HVD_COORDINATE_ID
        )
        calibration_id = (
            PROVIDER_SOURCE_HVD_CALIBRATION_ID
            if self._provider_mode
            else SOURCE_HVD_CALIBRATION_ID
        )
        contract_payload = {
            "coordinate": coordinate_id,
            "calibration": calibration_id,
            "archive": str(archive.fingerprint),
            "mode": str(config.mode),
            "output_index": int(config.output_index),
            "delta": float(config.calibration_delta),
            "quantile": float(config.calibration_quantile),
            "provider_ridge_per_source_row": float(
                config.provider_ridge_per_source_row),
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

    def _target_reference_exposure(self):
        if not hasattr(self.target_problem, "int_bounds"):
            raise ValueError(
                "provider cumulative HVD requires integer target bounds")
        lower, upper = self.target_problem.int_bounds()
        midpoint = np.rint(
            0.5 * (
                np.asarray(lower, dtype=float)
                + np.asarray(upper, dtype=float)
            )
        ).astype(int)
        exposure = get_risk_exposure(
            self.target_problem,
            midpoint,
            output_index=int(self.config.output_index),
        )
        if exposure is None:
            raise ValueError(
                "provider cumulative HVD requires target risk_exposures")
        if exposure.n_local < 1 or exposure.n_shared < 1:
            raise ValueError(
                "provider cumulative HVD requires nonempty A and N blocks")
        return exposure

    def _source_problem(self, task):
        cached = self._provider_task_cache.get(task.name)
        if cached is not None:
            return cached
        # Import lazily to keep the variance module independent of the
        # benchmark registry during ordinary algorithm imports.
        from problems.rzdt import make_problem

        dimension = int(task.X_integer.shape[1])
        problem = make_problem(
            task.name,
            d=dimension,
            L=int(getattr(self.target_problem, "L", 100)),
            sigma=float(np.sqrt(self.target_variance_scale)),
            alpha=float(task.alpha),
        )
        self._provider_task_cache[task.name] = problem
        return problem

    def _provider_task_design(self, task):
        cache_key = f"design:{task.name}"
        cached = self._provider_task_cache.get(cache_key)
        if cached is not None:
            return cached
        problem = self._source_problem(task)
        exposures = []
        features = []
        for point in np.asarray(task.X_integer, dtype=int):
            exposure = get_risk_exposure(
                problem,
                point,
                output_index=int(self.config.output_index),
            )
            if exposure is None:
                raise ValueError(
                    f"source task {task.name!r} has no risk_exposures")
            if (
                exposure.n_local
                != self._provider_target_exposure.n_local
                or exposure.n_shared
                != self._provider_target_exposure.n_shared
            ):
                raise ValueError(
                    "source and target provider A/N block dimensions differ")
            exposures.append(exposure)
            features.append(cumulative_feature_vector(exposure))
        values = np.maximum(
            np.asarray(
                task.replicate_variance[:, self.config.output_index],
                dtype=float,
            ),
            1e-12,
        )
        dof = np.asarray([
            max(int(len(row)) - 1, 1)
            for row in task.Y_replicates
        ], dtype=float)
        payload = {
            "problem": problem,
            "exposures": tuple(exposures),
            "features": np.vstack(features),
            "variances": values,
            "dof": dof,
        }
        self._provider_task_cache[cache_key] = payload
        return payload

    @staticmethod
    def _positive_spectrum_condition(matrix):
        eigenvalues = np.linalg.eigvalsh(
            0.5 * (matrix + matrix.T))
        positive = eigenvalues[eigenvalues > 1e-12]
        if len(positive) == 0:
            return None
        return float(np.max(positive) / np.min(positive))

    def _fit_provider_tasks(self, tasks):
        tasks = tuple(tasks)
        if not tasks:
            raise ValueError("provider cumulative HVD needs source tasks")
        designs = [self._provider_task_design(task) for task in tasks]
        features = np.vstack([row["features"] for row in designs])
        variances = np.concatenate([row["variances"] for row in designs])
        dof = np.concatenate([row["dof"] for row in designs])
        weights = dof / max(float(np.mean(dof)), 1.0)
        weight_total = max(float(np.sum(weights)), 1.0)
        column_rms = np.sqrt(np.sum(
            weights[:, None] * features ** 2,
            axis=0,
        ) / weight_total)
        column_rms = np.maximum(column_rms, 1e-8)
        column_rms[0] = 1.0
        normalized = features / column_rms[None, :]
        weighted_normalized = np.sqrt(weights)[:, None] * normalized
        weighted_variance = np.sqrt(weights) * variances
        ridge_alpha = float(
            self.config.provider_ridge_per_source_row * len(features))
        regularizer = ridge_alpha * np.eye(features.shape[1])
        regularizer[0, 0] = 0.0
        gram = weighted_normalized.T @ weighted_normalized
        solve_gram = gram + regularizer
        rhs = weighted_normalized.T @ weighted_variance
        try:
            normalized_beta = np.linalg.solve(solve_gram, rhs)
        except np.linalg.LinAlgError:
            normalized_beta = np.linalg.lstsq(
                solve_gram, rhs, rcond=None)[0]
        unconstrained_beta = normalized_beta / column_rms
        beta, params = project_cumulative_beta(
            unconstrained_beta,
            self._provider_target_exposure,
        )
        beta = np.asarray(beta, dtype=float)
        predictions = np.maximum(features @ beta, 1e-12)
        log_rmse = float(np.sqrt(np.mean(
            (np.log(predictions) - np.log(variances)) ** 2
        )))
        effective_df = float(np.trace(
            gram @ np.linalg.pinv(solve_gram)))
        task_rows = []
        offset = 0
        for task, design in zip(tasks, designs):
            count = len(design["features"])
            task_prediction = predictions[offset: offset + count]
            task_variance = design["variances"]
            task_rows.append({
                "domain": str(task.name),
                "source_rows": int(count),
                "replication_dof": float(np.sum(design["dof"])),
                "log_variance_rmse": float(np.sqrt(np.mean(
                    (
                        np.log(task_prediction)
                        - np.log(task_variance)
                    ) ** 2
                ))),
                "mean_predicted_variance": float(np.mean(task_prediction)),
                "mean_observed_variance": float(np.mean(task_variance)),
            })
            offset += count
        return {
            "beta": beta,
            "params": params,
            "feature_names": cumulative_feature_names(
                self._provider_target_exposure),
            "source_rows": int(len(features)),
            "effective_replication_dof": float(np.sum(dof)),
            "ridge_alpha": ridge_alpha,
            "ridge_per_source_row": float(
                self.config.provider_ridge_per_source_row),
            "column_rms": column_rms.tolist(),
            "normalized_gram_condition": (
                self._positive_spectrum_condition(gram)),
            "regularized_gram_condition": (
                self._positive_spectrum_condition(solve_gram)),
            "effective_parameter_df": effective_df,
            "fit_log_variance_rmse": log_rmse,
            "task_fit": task_rows,
            "fit_method": "weighted_ridge_then_cumulative_cone_projection",
            "scale_contract": "common_source_target_output_variance_units",
            "source_provider_reconstruction": (
                "task_family_schema_without_outcome_queries"),
        }

    def _provider_source_task_lodo_rows(self):
        rows = []
        tasks = tuple(self.archive.tasks)
        # Bonferroni makes all source-task aggregate scale bounds simultaneous.
        per_task_delta = float(
            self.config.calibration_delta / max(len(tasks), 1))
        for heldout in tasks:
            training = tuple(
                task for task in tasks if task.name != heldout.name)
            if not training:
                training = tasks
            fitted = self._fit_provider_tasks(training)
            design = self._provider_task_design(heldout)
            prediction = np.maximum(
                design["features"] @ fitted["beta"],
                1e-12,
            )
            observed = design["variances"]
            dof = design["dof"]
            total_dof = max(float(np.sum(dof)), 1.0)
            denominator = max(float(chi2.ppf(
                per_task_delta,
                total_dof,
            )), 1e-12)
            upper_scale = float(np.sum(
                dof * observed / prediction
            ) / denominator)
            centered_prediction = prediction - float(np.mean(prediction))
            centered_observed = observed - float(np.mean(observed))
            correlation_denominator = float(np.sqrt(
                np.sum(centered_prediction ** 2)
                * np.sum(centered_observed ** 2)
            ))
            rows.append({
                "heldout_source_task": str(heldout.name),
                "source_task_row_count": int(len(observed)),
                "replication_dof": int(total_dof),
                "per_task_delta": per_task_delta,
                "observed_to_predicted_ratio_mean": float(np.average(
                    observed / prediction,
                    weights=dof,
                )),
                "observed_to_predicted_ratio_q95": float(np.quantile(
                    observed / prediction,
                    0.95,
                    method="higher",
                )),
                "variance_shape_correlation": (
                    0.0
                    if correlation_denominator <= 1e-14
                    else float(np.sum(
                        centered_prediction * centered_observed
                    ) / correlation_denominator)
                ),
                "upper_variance_ratio": upper_scale,
                "calibration_contract": (
                    "aggregate_proportional_scale_chi_square_upper"),
            })
        return rows

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
        if self._provider_mode:
            exposure = get_risk_exposure(
                self.target_problem,
                x,
                output_index=int(self.config.output_index),
            )
            if exposure is None:
                raise ValueError("target risk exposure became unavailable")
            feature = cumulative_feature_vector(exposure)
            return float(max(
                feature @ self._provider_fit["beta"],
                1e-12,
            ))
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
        if self._provider_mode:
            return np.asarray([
                self.predict_certification_variance(point)
                for point in points
            ], dtype=float)
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
        if self._provider_mode:
            exposure = get_risk_exposure(
                self.target_problem,
                x,
                output_index=int(self.config.output_index),
            )
            if exposure is None:
                raise ValueError("target risk exposure became unavailable")
            cumulative = decompose_cumulative_risk(
                exposure,
                self._provider_fit["params"],
            )
            cumulative.update({
                "fitted_variance": float(cumulative["total"]),
                "certification_variance": float(
                    cumulative["total"]
                    * self.calibration_multiplier),
                "tail_guard": float(
                    cumulative["total"]
                    * (self.calibration_multiplier - 1.0)),
                "v_C_plus": float(
                    cumulative["total"]
                    * self.calibration_multiplier),
                "coordinate_id": PROVIDER_SOURCE_HVD_COORDINATE_ID,
            })
            return {
                "mode": "factor",
                "provider_active": True,
                "cumulative": cumulative,
            }
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
        coordinate_id = (
            PROVIDER_SOURCE_HVD_COORDINATE_ID
            if self._provider_mode
            else SOURCE_HVD_COORDINATE_ID
        )
        calibration_id = (
            PROVIDER_SOURCE_HVD_CALIBRATION_ID
            if self._provider_mode
            else SOURCE_HVD_CALIBRATION_ID
        )
        diagnostics = {
            "status": "frozen",
            "contract_id": self.contract_id,
            "mode": str(self.config.mode),
            "coordinate_id": coordinate_id,
            "calibration_id": calibration_id,
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
            "provider_schema_is_target_label_free": bool(
                self._provider_mode),
            "provider_schema_uses_task_family_identifier": bool(
                self._provider_mode),
            "provider_scale_model": (
                "common_source_target_output_variance_units"
                if self._provider_mode
                else "source_task_relative_then_target_nominal"
            ),
            "lodo_rows": list(self._lodo_rows),
        }
        diagnostics["hvd"] = (
            {
                key: value
                for key, value in self._provider_fit.items()
                if key not in {"beta", "params"}
            }
            if self._provider_mode
            else self.model.diagnostics()
        )
        if self._provider_mode:
            diagnostics["hvd"]["beta"] = self._provider_fit[
                "beta"].tolist()
            params = self._provider_fit["params"]
            diagnostics["hvd"]["parameters"] = {
                "floor": float(params.floor),
                "Lambda": params.Lambda.tolist(),
                "B": params.B.tolist(),
                "omega": params.omega.tolist(),
            }
        return diagnostics
