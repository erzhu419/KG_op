"""Orthogonal latent heteroscedastic decomposition (OLH/HVD).

The model deliberately exposes a compact API compatible with the old VEPM
role while supporting several decomposition levels:

* pooled: one variance per output.
* oracle: use `problem.true_sigma` for diagnostics.
* class: low-dimensional risk-regime variances.
* orthogonal: ridge model for log variance on orthogonal polynomial features.
* factor: orthogonal model plus factor diagnostics for shared-shock studies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from core.cumulative_risk import (
    decompose_cumulative_risk,
    get_risk_exposure,
    project_cumulative_beta,
)


@dataclass
class HVDConfig:
    mode: str = "class"
    n_outputs: int = 2
    ridge_alpha: float = 1e-3
    floor: float = 1e-8
    shrinkage_kappa: float = 2.0
    n_factors: int = 3
    activation_min_records: int = 20
    certification_kappa: float = 1.0
    residual_tail_delta: float = 0.05
    use_cumulative_provider: bool = True
    cumulative_irls_steps: int = 4
    cumulative_projected_steps: int = 256
    cumulative_weight_clip: float = 20.0
    cumulative_transfer_mode: str = "scalar"
    cumulative_transfer_upper_z: float = 1.6448536269514722
    cumulative_source_task_weight_mode: str = "independent"
    cumulative_target_evidence_mode: str = "replication_only"
    singleton_evidence_mode: str = "in_sample_residual"


def gaussian_square_subexp_params(sigma2):
    """Conservative sub-exponential constants for centered Gaussian squares.

    If a residual is modeled as sub-Gaussian with proxy variance `sigma2`, then
    its centered square is sub-exponential under the common Bernstein-style
    constants `(nu, b) = (2 sqrt(2) sigma2, 4 sigma2)`.
    """
    sigma2 = max(float(sigma2), 1e-12)
    return float(2.0 * np.sqrt(2.0) * sigma2), float(4.0 * sigma2)


def sub_exponential_residual_square_radius(nu, b, delta):
    """Radius r with `2 exp(-min(r^2/(2nu^2), r/(2b))) <= delta`.

    This is the manuscript-level default tail radius paired with the Lean
    `ResidualSquareTail` interface.  It is intentionally standalone so changing
    the certification policy is an explicit experiment rather than an accidental
    side effect of HVD fitting.
    """
    nu = max(float(nu), 1e-12)
    b = max(float(b), 1e-12)
    delta = float(delta)
    if not 0.0 < delta < 2.0:
        raise ValueError("delta must lie in (0, 2)")
    log_term = float(np.log(2.0 / delta))
    return float(max(np.sqrt(2.0 * nu * nu * log_term), 2.0 * b * log_term))


def sub_exponential_sample_mean_radius(nu, b, delta, n):
    """Bernstein radius for the mean of ``n`` centered residual squares.

    For ``2 exp(-n min(r^2/(2 nu^2), r/(2 b))) <= delta``, the quadratic
    branch contracts as ``n^-1/2`` while the linear branch contracts as
    ``n^-1``.  Applying ``n^-1/2`` to the already inverted single-observation
    radius is unnecessarily conservative in the linear branch.
    """

    nu = max(float(nu), 1e-12)
    b = max(float(b), 1e-12)
    delta = float(delta)
    n = int(n)
    if not 0.0 < delta < 2.0:
        raise ValueError("delta must lie in (0, 2)")
    if n <= 0:
        raise ValueError("n must be positive")
    log_term = float(np.log(2.0 / delta))
    return float(max(
        np.sqrt(2.0 * nu * nu * log_term / float(n)),
        2.0 * b * log_term / float(n),
    ))


class OrthogonalHVD:
    """Variance decomposition model for simulation residuals."""

    VALID_MODES = {"pooled", "oracle", "class", "orthogonal", "factor"}

    def __init__(self, mode="class", n_outputs=2, **kwargs):
        self.config = HVDConfig(mode=mode, n_outputs=n_outputs, **kwargs)
        if self.config.mode not in self.VALID_MODES:
            raise ValueError(f"unknown HVD mode {mode!r}")
        self.mode = self.config.mode
        if self.config.cumulative_transfer_mode not in {"scalar", "source_mixture"}:
            raise ValueError(
                "cumulative_transfer_mode must be 'scalar' or 'source_mixture'")
        if self.config.cumulative_source_task_weight_mode not in {
            "independent", "constraint_mean",
        }:
            raise ValueError(
                "cumulative_source_task_weight_mode must be 'independent' "
                "or 'constraint_mean'")
        if self.config.cumulative_target_evidence_mode not in {
            "replication_only", "prequential_upper",
        }:
            raise ValueError(
                "cumulative_target_evidence_mode must be "
                "'replication_only' or 'prequential_upper'")
        if self.config.singleton_evidence_mode not in {
            "in_sample_residual", "source_prior",
        }:
            raise ValueError(
                "singleton_evidence_mode must be in_sample_residual or "
                "source_prior")
        self.n_outputs = int(self.config.n_outputs)
        self.records = {i: [] for i in range(self.n_outputs)}
        self.global_var = {i: 0.01 for i in range(self.n_outputs)}
        self.class_var = {i: {} for i in range(self.n_outputs)}
        self.class_count = {i: {} for i in range(self.n_outputs)}
        self.beta = {i: None for i in range(self.n_outputs)}
        self.cumulative_beta = {i: None for i in range(self.n_outputs)}
        self.cumulative_params = {i: None for i in range(self.n_outputs)}
        self.cumulative_provider_active = {i: False for i in range(self.n_outputs)}
        self.cumulative_fit_rmse = {i: None for i in range(self.n_outputs)}
        self.cumulative_prior_used = {i: False for i in range(self.n_outputs)}
        self.cumulative_prior_precision = {i: None for i in range(self.n_outputs)}
        self.cumulative_prior_scale = {i: None for i in range(self.n_outputs)}
        self.cumulative_prior_scale_se = {i: None for i in range(self.n_outputs)}
        self.cumulative_prior_target_weight = {
            i: 0 for i in range(self.n_outputs)
        }
        self.cumulative_prior_scale_source = {
            i: "none" for i in range(self.n_outputs)
        }
        self.cumulative_prior_upper_scale = {
            i: 1.0 for i in range(self.n_outputs)
        }
        self.cumulative_prior_components = {
            i: None for i in range(self.n_outputs)
        }
        self.cumulative_prior_component_domains = {
            i: [] for i in range(self.n_outputs)
        }
        self.cumulative_prior_component_weights = {
            i: None for i in range(self.n_outputs)
        }
        self.cumulative_prior_component_covariance = {
            i: None for i in range(self.n_outputs)
        }
        self.cumulative_prior_shape_target_dof = {
            i: 0.0 for i in range(self.n_outputs)
        }
        self.cumulative_source_task_posterior = {
            i: None for i in range(self.n_outputs)
        }
        self.cumulative_activation_records = {
            i: int(self.config.activation_min_records)
            for i in range(self.n_outputs)
        }
        self.replicated_keys = {i: set() for i in range(self.n_outputs)}
        self.source_prior_pseudo_keys = {
            i: set() for i in range(self.n_outputs)
        }
        self.prequential_upper_records = {
            i: {} for i in range(self.n_outputs)
        }
        self.replication_dof = {i: {} for i in range(self.n_outputs)}
        self.cumulative_fit_method = {i: "inactive" for i in range(self.n_outputs)}
        self.cumulative_fit_effective_dof = {
            i: 0.0 for i in range(self.n_outputs)
        }
        self.cumulative_fit_weight_range = {
            i: None for i in range(self.n_outputs)
        }
        self.cumulative_prior_replication_only = {
            i: False for i in range(self.n_outputs)
        }
        self.factor_energy = {i: [] for i in range(self.n_outputs)}
        self._last_problem = None

    def __getstate__(self):
        """Drop non-picklable problem/simulator handles during exact-KG clones."""
        state = self.__dict__.copy()
        # Traffic/SUMO providers can own imported simulator modules and live
        # handles. Exact posterior-update KG always passes the current problem
        # explicitly after cloning, so retaining this cache is unnecessary and
        # can make ``copy.deepcopy`` fail with "cannot pickle module object".
        state["_last_problem"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self.config, "cumulative_target_evidence_mode"):
            self.config.cumulative_target_evidence_mode = "replication_only"
        if not hasattr(self.config, "singleton_evidence_mode"):
            self.config.singleton_evidence_mode = "in_sample_residual"
        if "_last_problem" not in self.__dict__:
            self._last_problem = None
        if "prequential_upper_records" not in self.__dict__:
            self.prequential_upper_records = {
                i: {} for i in range(self.n_outputs)
            }
        if "replication_dof" not in self.__dict__:
            self.replication_dof = {
                i: {} for i in range(self.n_outputs)
            }
        if "source_prior_pseudo_keys" not in self.__dict__:
            self.source_prior_pseudo_keys = {
                i: set() for i in range(self.n_outputs)
            }
        if "cumulative_fit_method" not in self.__dict__:
            self.cumulative_fit_method = {
                i: "legacy_projection" for i in range(self.n_outputs)
            }
        if "cumulative_fit_effective_dof" not in self.__dict__:
            self.cumulative_fit_effective_dof = {
                i: float(len(self.records.get(i, [])))
                for i in range(self.n_outputs)
            }
        if "cumulative_fit_weight_range" not in self.__dict__:
            self.cumulative_fit_weight_range = {
                i: None for i in range(self.n_outputs)
            }
        for name, default in (
            ("cumulative_prior_components", None),
            ("cumulative_prior_component_domains", []),
            ("cumulative_prior_component_weights", None),
            ("cumulative_prior_component_covariance", None),
            ("cumulative_prior_shape_target_dof", 0.0),
            ("cumulative_source_task_posterior", None),
        ):
            if name not in self.__dict__:
                setattr(self, name, {
                    i: (list(default) if isinstance(default, list) else default)
                    for i in range(self.n_outputs)
                })

    def set_source_task_posterior(
        self,
        output_index,
        component_names,
        posterior_weights,
    ):
        """Share the target-pilot task law with cumulative HVD transfer.

        Mean and variance transfer then condition on the same latent source
        domain.  A ``target:null`` component is retained explicitly instead
        of silently redistributing no-transfer mass over source HVD shapes.
        """

        i = int(output_index)
        names = [str(name) for name in component_names]
        weights = np.maximum(
            np.asarray(posterior_weights, dtype=float).reshape(-1), 0.0)
        if len(names) != len(weights) or not names:
            raise ValueError("source task names and weights must agree")
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("source task posterior must have positive mass")
        weights /= total
        self.cumulative_source_task_posterior[i] = {
            "component_names": names,
            "posterior_weights": weights.tolist(),
            "target_null_weight": float(sum(
                weight for name, weight in zip(names, weights)
                if name == "target:null"
            )),
            "target_data_used": True,
            "target_oracle_used": False,
        }
        return dict(self.cumulative_source_task_posterior[i])

    @property
    def floor(self):
        return max(float(self.config.floor), 1e-12)

    def risk_class(self, x, problem=None):
        problem = problem or self._last_problem
        if problem is not None and hasattr(problem, "risk_class"):
            return int(problem.risk_class(x))
        z = self._normalize(x, problem)
        u = float(z[0]) if len(z) else 0.0
        if u < 1.0 / 3.0:
            return 0
        if u < 2.0 / 3.0:
            return 1
        return 2

    def risk_classes_many(self, X, problem=None):
        """Vectorized risk class labels for a candidate list."""
        problem = problem or self._last_problem
        if problem is not None and hasattr(problem, "risk_class"):
            return np.array([int(problem.risk_class(x)) for x in X], dtype=int)
        Z = np.vstack([self._normalize(x, problem) for x in X])
        u = Z[:, 0] if Z.size else np.zeros(len(X), dtype=float)
        cls = np.zeros(len(X), dtype=int)
        cls[u >= 1.0 / 3.0] = 1
        cls[u >= 2.0 / 3.0] = 2
        return cls

    def _normalize(self, x, problem=None):
        problem = problem or self._last_problem
        if problem is not None and hasattr(problem, "normalize"):
            return np.asarray(problem.normalize(x), dtype=float)
        x = np.asarray(x, dtype=float)
        scale = max(float(np.max(np.abs(x))) if len(x) else 1.0, 1.0)
        return np.clip(x / scale, 0.0, 1.0)

    def _features(self, x, problem=None):
        """Near-orthogonal polynomial/log-variance features on [0, 1]."""
        problem = problem or self._last_problem
        if problem is not None and hasattr(problem, "hvd_features"):
            try:
                return np.asarray(problem.hvd_features(x), dtype=float)
            except AttributeError:
                pass
        z = self._normalize(x, problem)
        if len(z) == 0:
            z = np.array([0.0])
        # Shifted Legendre basis terms.
        p1 = np.sqrt(3.0) * (2.0 * z - 1.0)
        p2 = np.sqrt(5.0) * (6.0 * z ** 2 - 6.0 * z + 1.0)
        stats = np.array([
            float(np.mean(z)),
            float(np.std(z)),
            float(np.min(z)),
            float(np.max(z)),
            float(np.linalg.norm(z - 0.5) / np.sqrt(len(z))),
        ])
        return np.concatenate([[1.0], p1, p2, stats])

    def _feature_matrix(self, X, problem=None):
        rows = []
        for x in X:
            rows.append(self._features(x, problem))
        return np.vstack(rows) if rows else np.empty((0, 1), dtype=float)

    def _cumulative_features(self, x, problem=None, output_index=0):
        """Linear variance features for trajectory/meta cumulative risk."""
        if not bool(self.config.use_cumulative_provider):
            return None
        problem = problem or self._last_problem
        feat = None
        if problem is not None and hasattr(problem, "cumulative_risk_features"):
            try:
                feat = problem.cumulative_risk_features(
                    x,
                    output_index=int(output_index),
                )
            except TypeError:
                feat = problem.cumulative_risk_features(x)
            if feat is not None:
                feat = np.asarray(feat, dtype=float)
                if feat.ndim != 1 or len(feat) == 0 or not np.all(np.isfinite(feat)):
                    feat = None

        manifold = self._manifold_cumulative_features(x, problem)
        if feat is None and manifold is None:
            return None
        if feat is None:
            return np.concatenate([[1.0], manifold])
        if manifold is None:
            return feat
        return np.concatenate([feat, manifold])

    def _base_cumulative_feature_names(self, problem=None, output_index=0):
        problem = problem or self._last_problem
        if problem is None or not hasattr(problem, "cumulative_risk_feature_names"):
            return None
        try:
            names = problem.cumulative_risk_feature_names(output_index=int(output_index))
        except TypeError:
            names = problem.cumulative_risk_feature_names()
        if names is None:
            return None
        return [str(name) for name in names]

    def _cumulative_feature_names(self, x, problem=None, output_index=0, feat_len=None):
        problem = problem or self._last_problem
        names = self._base_cumulative_feature_names(problem, output_index)
        manifold = self._manifold_cumulative_features(x, problem)
        if names is None:
            if manifold is None:
                return None
            names = ["floor"] + [
                f"manifold_rho{j}_sq" for j in range(len(manifold) // 2)
            ] + [
                f"manifold_rho{j}_abs" for j in range(len(manifold) // 2)
            ]
        elif manifold is not None:
            k = len(manifold) // 2
            names = list(names) + [
                f"manifold_rho{j}_sq" for j in range(k)
            ] + [
                f"manifold_rho{j}_abs" for j in range(k)
            ]
        if feat_len is not None and len(names) != int(feat_len):
            # Keep diagnostics honest when a problem exposes an older feature
            # name list while representation features are appended.
            names = list(names[: int(feat_len)])
            while len(names) < int(feat_len):
                names.append(f"feature_{len(names)}")
        return names

    def _manifold_cumulative_features(self, x, problem=None):
        """Positive low-dimensional latent blocks for cumulative variance.

        These do not replace the problem's trajectory/factor features.  They
        append tangent/meta information when a representation encoder is active,
        and provide a fallback cumulative feature block for purely synthetic
        representation ablations.
        """
        problem = problem or self._last_problem
        if problem is None or not getattr(problem, "_scolhkg_use_manifold_hvd", False):
            return None
        encoder = getattr(problem, "_scolhkg_representation_encoder", None)
        if encoder is None or not hasattr(encoder, "occupancy"):
            return None
        try:
            rho = np.asarray(encoder.occupancy(x), dtype=float)
        except Exception:
            return None
        if rho.ndim != 1 or len(rho) == 0 or not np.all(np.isfinite(rho)):
            return None
        k = min(8, len(rho))
        rho = np.clip(rho[:k], -5.0, 5.0)
        return np.concatenate([rho ** 2, np.abs(rho)])

    def _cumulative_feature_matrix(self, X, problem=None, output_index=0):
        rows = []
        expected_dim = None
        for x in X:
            feat = self._cumulative_features(x, problem, output_index)
            if feat is None:
                return None
            if expected_dim is None:
                expected_dim = len(feat)
            elif len(feat) != expected_dim:
                return None
            rows.append(feat)
        return np.vstack(rows) if rows else np.empty((0, 1), dtype=float)

    def _cumulative_active(self, i):
        if self.mode != "factor":
            return False
        if self.cumulative_beta.get(int(i)) is None:
            return False
        threshold = int(self.cumulative_activation_records.get(
            int(i), self.config.activation_min_records))
        return len(self.records.get(int(i), [])) >= threshold

    def _variance_model_active(self, i):
        if self.mode == "factor" and self._cumulative_active(i):
            return True
        return self._orthogonal_active(i)

    def _residual_variance_cap(self, i, problem=None):
        problem = problem or self._last_problem
        if problem is None or not hasattr(problem, "hvd_residual_variance_cap"):
            return None
        cap = problem.hvd_residual_variance_cap(output_index=int(i))
        if cap is None:
            return None
        cap = float(cap)
        if not np.isfinite(cap) or cap <= 0.0:
            return None
        return max(cap, self.floor)

    def _safe_residual_square(self, residual, output_index=0, problem=None):
        """Square a residual with the same cap used by the HVD fit.

        High-dimensional basis experiments can occasionally produce enormous
        transient posterior means before the surrogate stabilizes.  The HVD
        model already caps residual variances during fitting; applying that cap
        before squaring keeps the numerical path identical in intent but avoids
        Python float overflow on the raw subtraction.
        """
        i = int(output_index)
        cap = self._residual_variance_cap(i, problem)
        if cap is not None:
            max_abs = float(np.sqrt(max(cap, self.floor)))
        else:
            max_abs = float(np.sqrt(np.finfo(float).max / 16.0))
        try:
            r_abs = abs(float(residual))
        except (OverflowError, ValueError):
            r_abs = max_abs
        if not np.isfinite(r_abs):
            r_abs = max_abs
        r_abs = min(r_abs, max_abs)
        resid2 = r_abs * r_abs
        if not np.isfinite(resid2):
            resid2 = cap if cap is not None else max(self.global_var.get(i, 0.01), self.floor)
        return float(max(resid2, self.floor))

    def _record_replication_dof(self, i, x):
        """Degrees of freedom behind one stored variance observation."""

        return float(max(
            self.replication_dof.get(int(i), {}).get(tuple(x), 1.0),
            1.0,
        ))

    def _effective_variance_dof(self, i):
        """Independent chi-square degrees of freedom used by the tail guard.

        A raw squared innovation contributes one degree of freedom.  A sample
        variance from ``r`` independent simulator replications contributes
        ``r - 1``.  The old implementation counted both as one observation,
        so additional replications reduced the fitted noise but not its
        certification uncertainty.
        """

        return float(sum(
            self._record_replication_dof(i, x)
            for x, _ in self.records.get(int(i), [])
            if tuple(x) not in self.source_prior_pseudo_keys.get(int(i), set())
        ))

    def _residual_square_tail_radius(self, i):
        tail_delta = float(self.config.residual_tail_delta)
        nu, b = gaussian_square_subexp_params(self.global_var.get(int(i), 0.01))
        return float(sub_exponential_residual_square_radius(nu, b, tail_delta))

    def _residual_tail_uncertainty(self, i):
        effective_dof = max(self._effective_variance_dof(i), 0.0)
        nu, b = gaussian_square_subexp_params(self.global_var.get(int(i), 0.01))
        return float(sub_exponential_sample_mean_radius(
            nu,
            b,
            self.config.residual_tail_delta,
            max(int(np.floor(effective_dof)) + 1, 1),
        ))

    def _high_frequency_residual_floor(self, i, x, problem=None):
        """Residual floor from pruned/low-pass representation components."""
        i = int(i)
        problem = problem or self._last_problem
        floor = 0.0
        if problem is not None and hasattr(problem, "hvd_high_frequency_floor"):
            try:
                floor = max(
                    floor,
                    float(problem.hvd_high_frequency_floor(x, output_index=int(i))),
                )
            except TypeError:
                floor = max(floor, float(problem.hvd_high_frequency_floor(x)))
            except (ValueError, AttributeError):
                pass
        encoder = getattr(problem, "_scolhkg_representation_encoder", None)
        if encoder is not None and hasattr(encoder, "residual_floor"):
            try:
                floor = max(floor, float(encoder.residual_floor(x, output_index=int(i))))
            except TypeError:
                floor = max(floor, float(encoder.residual_floor(x)))
            except (ValueError, AttributeError):
                pass
        if not np.isfinite(floor):
            return 0.0
        return float(max(floor, 0.0))

    def _high_frequency_residual_floor_many(self, i, X, problem=None):
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        return np.asarray([
            self._high_frequency_residual_floor(i, x, problem)
            for x in X
        ], dtype=float)

    def _orthogonal_active(self, i):
        """Whether smooth orthogonal variance is allowed for this output.

        Orthogonal log-variance regression is intentionally delayed until
        enough residual evidence exists.  Before activation, orthogonal/factor
        modes use the class-HVD estimate, which is more stable and cheaper in
        the small-budget regime.
        """
        if self.mode not in ("orthogonal", "factor"):
            return False
        if self.beta.get(int(i)) is None:
            return False
        return len(self.records.get(int(i), [])) >= int(self.config.activation_min_records)

    def fit_from_residuals(self, X, residuals, output_index=0, problem=None):
        """Direct fit helper used by tests and by `initialize`."""
        self._last_problem = problem or self._last_problem
        i = int(output_index)
        for x, r in zip(X, residuals):
            x_tuple = tuple(int(v) for v in x)
            self.records[i].append((
                x_tuple,
                self._safe_residual_square(r, i, problem),
            ))
            self.replication_dof[i][x_tuple] = 1.0
        self._fit_output(i, problem)

    def fit_from_variances(
        self,
        X,
        variances,
        output_index=0,
        problem=None,
        replicate_counts=None,
        *,
        replace=False,
    ):
        """Fit directly from independent within-policy sample variances.

        This is the source-archive entry point for cumulative HVD.  Unlike
        ``fit_from_residuals``, every row is known to come from repeated
        simulator evaluations, so its chi-square degrees of freedom are
        retained for the certification tail guard.
        """

        self._last_problem = problem or self._last_problem
        i = int(output_index)
        rows = list(X)
        values = np.asarray(variances, dtype=float).reshape(-1)
        if len(rows) != len(values):
            raise ValueError("X and variances must have the same length")
        if replicate_counts is None:
            counts = np.full(len(rows), 2, dtype=int)
        else:
            counts = np.asarray(replicate_counts, dtype=int).reshape(-1)
            if len(counts) != len(rows):
                raise ValueError(
                    "replicate_counts and variances must have the same length")
            if np.any(counts < 2):
                raise ValueError("sample variances require at least 2 replicates")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("variances must be finite and nonnegative")
        if replace:
            self.records[i] = []
            self.replicated_keys[i] = set()
            self.source_prior_pseudo_keys[i] = set()
            self.prequential_upper_records[i] = {}
            self.replication_dof[i] = {}
        for x, variance, count in zip(rows, values, counts):
            x_tuple = tuple(int(v) for v in np.asarray(x).reshape(-1))
            self.records[i].append((
                x_tuple,
                max(float(variance), self.floor),
            ))
            self.replicated_keys[i].add(x_tuple)
            self.replication_dof[i][x_tuple] = float(int(count) - 1)
        self._fit_output(i, problem)
        return self.diagnostics()

    def _source_prior_singleton_variance(self, i, x, problem=None):
        """Predict singleton variance without reading its response residual.

        Source HVD coefficients come from ordinary replicated source
        simulations. Their held-out target prediction supplies an
        outcome-free placeholder until the same target policy is replicated.
        """

        problem_ref = problem or self._last_problem
        i = int(i)
        x_tuple = tuple(int(value) for value in np.asarray(x, dtype=int))
        if problem_ref is None:
            return float(max(self.global_var.get(i, 0.01), self.floor))
        feature = self._cumulative_features(
            x_tuple, problem_ref, output_index=i)
        if feature is None:
            sigma = max(float(getattr(problem_ref, "sigma_level", 0.0)), 0.0)
            return float(max(sigma ** 2, self.floor))
        feature = np.asarray(feature, dtype=float).reshape(-1)
        scale = 1.0
        if hasattr(problem_ref, "cumulative_hvd_prior_scale_mean"):
            value = problem_ref.cumulative_hvd_prior_scale_mean(
                output_index=i)
            if value is not None:
                scale = max(float(value), self.floor)

        coefficients = None
        domains = []
        if (
            str(self.config.cumulative_transfer_mode) == "source_mixture"
            and hasattr(problem_ref, "cumulative_hvd_prior_components")
        ):
            try:
                payload = problem_ref.cumulative_hvd_prior_components(
                    output_index=i, feature_dim=len(feature))
            except TypeError:
                payload = problem_ref.cumulative_hvd_prior_components(i)
            if isinstance(payload, dict):
                domains = list(payload.get("domains", []))
                payload = payload.get("coefficients")
            if payload is not None:
                matrix = np.asarray(payload, dtype=float)
                if (
                    matrix.ndim == 2
                    and matrix.shape[1] == len(feature)
                    and len(matrix) > 0
                    and np.all(np.isfinite(matrix))
                ):
                    weights = np.full(
                        len(matrix), 1.0 / float(len(matrix)), dtype=float)
                    task = self.cumulative_source_task_posterior.get(i)
                    if (
                        self.config.cumulative_source_task_weight_mode
                        == "constraint_mean"
                        and task is not None
                        and domains
                    ):
                        task_weight = dict(zip(
                            task["component_names"],
                            task["posterior_weights"],
                        ))
                        candidate = np.asarray([
                            task_weight.get(
                                domain,
                                task_weight.get(f"source:{domain}", 0.0),
                            )
                            for domain in domains
                        ], dtype=float)
                        if float(np.sum(candidate)) > 0.0:
                            weights = candidate / float(np.sum(candidate))
                    coefficients = weights @ matrix
        if coefficients is None and hasattr(
            problem_ref, "cumulative_hvd_prior_beta"
        ):
            try:
                coefficients = problem_ref.cumulative_hvd_prior_beta(
                    output_index=i, feature_dim=len(feature))
            except TypeError:
                coefficients = problem_ref.cumulative_hvd_prior_beta(i)
        if coefficients is None:
            sigma = max(float(getattr(problem_ref, "sigma_level", 0.0)), 0.0)
            return float(max(sigma ** 2, self.floor))
        coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        if len(coefficients) != len(feature):
            raise RuntimeError("source HVD singleton prior changed dimension")
        variance = scale * max(float(feature @ coefficients), self.floor)
        cap = self._residual_variance_cap(i, problem_ref)
        if cap is not None:
            variance = min(variance, float(cap))
        return float(max(variance, self.floor))

    def initialize(self, samples, observations, gpr_models, problem=None):
        """Initialize from pre-sample residuals.

        `observations` maps `x_tuple -> list[np.ndarray]`, and each observation
        contains all output channels.
        """
        self._last_problem = problem or self._last_problem
        if self.config.singleton_evidence_mode == "source_prior":
            for x_tuple in samples:
                key = tuple(int(value) for value in x_tuple)
                obs_list = list(observations.get(key, []))
                if not obs_list:
                    continue
                for i in range(self.n_outputs):
                    values = np.asarray([
                        float(observation[i]) for observation in obs_list
                    ], dtype=float)
                    if len(values) >= 2:
                        variance = max(float(np.var(values, ddof=1)), self.floor)
                        self.replicated_keys[i].add(key)
                        self.replication_dof[i][key] = float(len(values) - 1)
                    else:
                        variance = self._source_prior_singleton_variance(
                            i, key, problem)
                        self.source_prior_pseudo_keys[i].add(key)
                        self.replication_dof[i][key] = 0.0
                    self.records[i].append((key, variance))
            for i in range(self.n_outputs):
                self._fit_output(i, problem)
            return
        for x_tuple in samples:
            obs_list = observations.get(tuple(x_tuple), [])
            if not obs_list:
                continue
            x_arr = np.asarray(x_tuple, dtype=int)
            for y_vec in obs_list:
                for i in range(self.n_outputs):
                    mu = float(gpr_models[i].posterior_mean(x_arr))
                    resid2 = self._safe_residual_square(
                        float(y_vec[i]) - mu, i, problem)
                    self.records[i].append((tuple(x_tuple), resid2))
                    self.replication_dof[i][tuple(x_tuple)] = 1.0
        for i in range(self.n_outputs):
            self._fit_output(i, problem)

    def update(
        self,
        i,
        x,
        y,
        mu,
        gpr_model=None,
        problem=None,
        epistemic_var=None,
        replicate_variance=None,
        replicate_count=None,
    ):
        """Add one residual and refit lightweight summaries."""
        del gpr_model
        self._last_problem = problem or self._last_problem
        i = int(i)
        x_tuple = tuple(int(v) for v in np.asarray(x, dtype=int))
        raw_resid2 = self._safe_residual_square(float(y) - float(mu), i, problem)
        epistemic_correction = max(float(epistemic_var or 0.0), 0.0)
        if replicate_variance is None:
            if self.config.singleton_evidence_mode == "source_prior":
                resid2 = self._source_prior_singleton_variance(
                    i, x_tuple, problem)
                variance_source = "source_prior_singleton"
                if not any(
                    tuple(record[0]) == x_tuple for record in self.records[i]
                ):
                    self.records[i].append((x_tuple, resid2))
                self.source_prior_pseudo_keys[i].add(x_tuple)
                self.replication_dof[i][x_tuple] = 0.0
            else:
                resid2 = max(raw_resid2 - epistemic_correction, self.floor)
                variance_source = "innovation_minus_epistemic"
                self.records[i].append((x_tuple, resid2))
                # This prediction was formed before observing y. Its raw
                # squared innovation has expectation v(x) plus squared mean
                # error, so it is conservative variance-shape evidence.
                self.prequential_upper_records[i][x_tuple] = float(raw_resid2)
                self.replication_dof[i][x_tuple] = 1.0
        else:
            resid2 = max(float(replicate_variance), self.floor)
            variance_source = "within_solution_replication"
            self.records[i] = [
                record for record in self.records[i]
                if tuple(record[0]) != x_tuple
            ]
            self.records[i].append((x_tuple, resid2))
            self.replicated_keys[i].add(x_tuple)
            self.source_prior_pseudo_keys[i].discard(x_tuple)
            count = 2 if replicate_count is None else max(
                int(replicate_count), 2)
            self.replication_dof[i][x_tuple] = float(count - 1)
        old = self.predict_variance(i, x_tuple, problem)
        self._fit_output(i, problem)
        new = self.predict_variance(i, x_tuple, problem)
        return {
            "mode": self.mode,
            "output_index": i,
            "x": list(x_tuple),
            "raw_innovation2": float(raw_resid2),
            "epistemic_correction": float(epistemic_correction),
            "resid2": float(resid2),
            "variance_source": variance_source,
            "replicate_count": (
                None if replicate_variance is None
                else int(2 if replicate_count is None else max(
                    int(replicate_count), 2))
            ),
            "replication_dof": float(
                self._record_replication_dof(i, x_tuple)),
            "old_variance": float(old),
            "new_variance": float(new),
            "risk_class": int(self.risk_class(x_tuple, problem)),
        }

    def _fit_source_shape_mixture(
        self,
        component_design,
        target_variance,
        dof,
        *,
        source_scale,
        prior_precision,
        upper_scale,
        prior_weights=None,
    ):
        """Fit a nonnegative low-rank source-shape posterior.

        Each column of ``component_design`` is one source-domain PSD variance
        shape evaluated at replicated target policies.  The nonnegative
        coefficient vector preserves PSD cumulative risk by construction.
        A Gaussian source prior supplies the precision; replicated sample
        variances contribute their scaled-chi-square Fisher information.
        """

        design = np.asarray(component_design, dtype=float)
        target = np.asarray(target_variance, dtype=float).reshape(-1)
        dof = np.maximum(np.asarray(dof, dtype=float).reshape(-1), 1.0)
        n_components = int(design.shape[1])
        source_scale = max(float(source_scale), self.floor)
        prior_precision = max(float(prior_precision), 1e-6)
        if prior_weights is None:
            normalized_prior = np.full(
                n_components, 1.0 / float(n_components), dtype=float)
        else:
            normalized_prior = np.maximum(
                np.asarray(prior_weights, dtype=float).reshape(-1), 0.0)
            if len(normalized_prior) != n_components:
                raise ValueError("source-shape prior weight dimension mismatch")
            total_prior = float(np.sum(normalized_prior))
            if not np.isfinite(total_prior) or total_prior <= 0.0:
                raise ValueError("source-shape prior weights must have positive mass")
            normalized_prior /= total_prior
        theta0 = source_scale * normalized_prior
        relative_radius = max(float(upper_scale) - 1.0, 0.10)
        upper_z = max(float(self.config.cumulative_transfer_upper_z), 1e-6)
        total_scale_sd = relative_radius * source_scale / upper_z
        component_variance = np.maximum(
            total_scale_sd ** 2 * normalized_prior,
            self.floor ** 2,
        )
        prior_information = np.maximum(
            prior_precision / component_variance,
            1e-8,
        )
        prior_matrix = np.diag(prior_information)
        theta = theta0.copy()
        information = prior_matrix.copy()
        weight_range = None
        if len(target):
            for _ in range(max(int(self.config.cumulative_irls_steps), 1)):
                prediction = np.maximum(design @ theta, self.floor)
                weights = dof / np.maximum(
                    2.0 * prediction ** 2,
                    self.floor ** 2,
                )
                finite = weights[np.isfinite(weights) & (weights > 0.0)]
                if len(finite):
                    median = max(float(np.median(finite)), 1e-12)
                    clip = max(float(self.config.cumulative_weight_clip), 1.0)
                    weights = np.clip(weights, median / clip, median * clip)
                    weight_range = (
                        float(np.min(weights)),
                        float(np.max(weights)),
                    )
                else:
                    weights = np.ones(len(target), dtype=float)
                    weight_range = (1.0, 1.0)
                information = (
                    prior_matrix
                    + design.T @ (weights[:, None] * design)
                )
                rhs = prior_matrix @ theta0 + design.T @ (weights * target)
                active = np.ones(n_components, dtype=bool)
                candidate = np.zeros(n_components, dtype=float)
                while np.any(active):
                    active_information = information[np.ix_(active, active)]
                    active_rhs = rhs[active]
                    try:
                        active_solution = np.linalg.solve(
                            active_information, active_rhs)
                    except np.linalg.LinAlgError:
                        active_solution = np.linalg.lstsq(
                            active_information, active_rhs, rcond=None)[0]
                    if np.all(active_solution >= 0.0):
                        candidate[active] = active_solution
                        break
                    active_indices = np.flatnonzero(active)
                    active[active_indices[int(np.argmin(active_solution))]] = False
                if not np.any(candidate > 0.0):
                    candidate = theta0.copy()
                if np.allclose(candidate, theta, rtol=1e-8, atol=1e-12):
                    theta = candidate
                    break
                theta = candidate
        try:
            covariance = np.linalg.inv(information)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(information)
        covariance = 0.5 * (covariance + covariance.T)
        return theta, covariance, {
            "target_dof": float(np.sum(dof) if len(target) else 0.0),
            "weight_range": weight_range,
            "prior_component_variance": component_variance.tolist(),
            "prior_information": prior_information.tolist(),
            "prior_weights": normalized_prior.tolist(),
        }

    def _fit_constrained_cumulative_ridge(
        self,
        X,
        y,
        dof,
        exposure,
        reg,
        prior_center,
        initial_beta,
    ):
        """Replication-aware IRLS on the cumulative-risk parameter cone.

        Sample variances have variance proportional to ``v(x)^2 / dof``.
        IRLS therefore weights each row by ``dof / v(x)^2``.  Every gradient
        step is projected back to nonnegative local/linear components and a
        PSD shared-shock matrix, instead of projecting an unconstrained ridge
        solution only once.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        dof = np.maximum(np.asarray(dof, dtype=float).reshape(-1), 1.0)
        reg = np.asarray(reg, dtype=float)
        prior_center = np.asarray(prior_center, dtype=float).reshape(-1)
        beta, params = project_cumulative_beta(initial_beta, exposure)
        beta = np.asarray(beta, dtype=float)
        weight_range = (1.0, 1.0)
        n_outer = max(int(self.config.cumulative_irls_steps), 1)
        n_steps = max(int(self.config.cumulative_projected_steps), 1)
        weight_clip = max(float(self.config.cumulative_weight_clip), 1.0)

        for _ in range(n_outer):
            predicted = np.maximum(X @ beta, self.floor)
            weights = dof / np.maximum(predicted ** 2, self.floor ** 2)
            finite = weights[np.isfinite(weights) & (weights > 0.0)]
            scale = float(np.median(finite)) if len(finite) else 1.0
            weights = np.where(np.isfinite(weights), weights / max(scale, 1e-12), 1.0)
            weights = np.clip(weights, 1.0 / weight_clip, weight_clip)
            weight_range = (float(np.min(weights)), float(np.max(weights)))
            weighted_X = np.sqrt(weights)[:, None] * X
            lipschitz = float(np.linalg.norm(weighted_X, ord=2) ** 2)
            try:
                lipschitz += float(np.linalg.norm(reg, ord=2))
            except np.linalg.LinAlgError:
                lipschitz += float(np.max(np.abs(reg)))
            step = 1.0 / max(lipschitz, 1e-12)

            def objective(value):
                residual = X @ value - y
                centered = value - prior_center
                return float(
                    0.5 * np.sum(weights * residual ** 2)
                    + 0.5 * centered @ reg @ centered
                )

            current = objective(beta)
            for _ in range(n_steps):
                gradient = (
                    X.T @ (weights * (X @ beta - y))
                    + reg @ (beta - prior_center)
                )
                trial_step = step
                accepted = False
                candidate = beta
                candidate_params = params
                for _ in range(12):
                    candidate, candidate_params = project_cumulative_beta(
                        beta - trial_step * gradient,
                        exposure,
                    )
                    candidate = np.asarray(candidate, dtype=float)
                    candidate_value = objective(candidate)
                    if candidate_value <= current + 1e-18:
                        accepted = True
                        break
                    trial_step *= 0.5
                if not accepted:
                    break
                change = float(np.linalg.norm(candidate - beta))
                beta = candidate
                params = candidate_params
                current = candidate_value
                if change <= 1e-10 * (1.0 + float(np.linalg.norm(beta))):
                    break

        return beta, params, {
            "method": "replication_aware_projected_irls",
            "effective_dof": float(np.sum(dof)),
            "weight_range": weight_range,
        }

    def _fit_output(self, i, problem=None):
        recs = self.records[i]
        if not recs:
            self.global_var[i] = max(self.global_var.get(i, 0.01), self.floor)
            return
        raw_vals = np.array([max(v, self.floor) for _, v in recs], dtype=float)
        cap = self._residual_variance_cap(i, problem)
        vals = np.minimum(raw_vals, cap) if cap is not None else raw_vals
        self.global_var[i] = float(max(np.mean(vals), self.floor))

        by_class = defaultdict(list)
        for (x, _), v in zip(recs, vals):
            by_class[self.risk_class(x, problem)].append(max(v, self.floor))
        self.class_var[i] = {}
        self.class_count[i] = {}
        for c, c_vals in by_class.items():
            n_c = len(c_vals)
            raw = float(np.mean(c_vals))
            kappa = max(float(self.config.shrinkage_kappa), 0.0)
            weight = n_c / (n_c + kappa) if kappa > 0 else 1.0
            shrunk = weight * raw + (1.0 - weight) * self.global_var[i]
            self.class_var[i][int(c)] = float(max(shrunk, self.floor))
            self.class_count[i][int(c)] = int(n_c)

        self.cumulative_beta[i] = None
        self.cumulative_params[i] = None
        self.cumulative_provider_active[i] = False
        self.cumulative_fit_rmse[i] = None
        self.cumulative_prior_used[i] = False
        self.cumulative_prior_precision[i] = None
        self.cumulative_prior_scale[i] = None
        self.cumulative_prior_scale_se[i] = None
        self.cumulative_prior_target_weight[i] = 0
        self.cumulative_prior_scale_source[i] = "none"
        self.cumulative_prior_upper_scale[i] = 1.0
        self.cumulative_prior_components[i] = None
        self.cumulative_prior_component_domains[i] = []
        self.cumulative_prior_component_weights[i] = None
        self.cumulative_prior_component_covariance[i] = None
        self.cumulative_prior_shape_target_dof[i] = 0.0
        self.cumulative_fit_method[i] = "inactive"
        self.cumulative_fit_effective_dof[i] = 0.0
        self.cumulative_fit_weight_range[i] = None
        activation_records = int(self.config.activation_min_records)
        prior_replication_only = False
        problem_ref = problem or self._last_problem
        prior_beta_probe = None
        prior_component_probe = None
        prior_component_domains = []
        if self.mode == "factor" and problem_ref is not None and recs:
            feature_probe = self._cumulative_features(
                recs[0][0], problem, output_index=i)
            if (
                feature_probe is not None
                and hasattr(problem_ref, "cumulative_hvd_prior_beta")
            ):
                try:
                    prior_beta_probe = problem_ref.cumulative_hvd_prior_beta(
                        output_index=i,
                        feature_dim=len(feature_probe),
                    )
                except TypeError:
                    prior_beta_probe = problem_ref.cumulative_hvd_prior_beta(i)
                if prior_beta_probe is not None:
                    prior_beta_probe = np.asarray(prior_beta_probe, dtype=float)
                    if prior_beta_probe.shape != (len(feature_probe),):
                        prior_beta_probe = None
            if (
                str(self.config.cumulative_transfer_mode) == "source_mixture"
                and feature_probe is not None
                and hasattr(problem_ref, "cumulative_hvd_prior_components")
            ):
                try:
                    component_payload = problem_ref.cumulative_hvd_prior_components(
                        output_index=i,
                        feature_dim=len(feature_probe),
                    )
                except TypeError:
                    component_payload = problem_ref.cumulative_hvd_prior_components(i)
                if isinstance(component_payload, dict):
                    prior_component_domains = list(
                        component_payload.get("domains", []))
                    component_payload = component_payload.get("coefficients")
                if component_payload is not None:
                    component_payload = np.asarray(component_payload, dtype=float)
                    if (
                        component_payload.ndim == 2
                        and component_payload.shape[0] > 0
                        and component_payload.shape[1] == len(feature_probe)
                        and np.all(np.isfinite(component_payload))
                    ):
                        prior_component_probe = component_payload
            if (
                prior_beta_probe is not None
                and hasattr(problem_ref, "cumulative_hvd_prior_min_records")
            ):
                prior_min = problem_ref.cumulative_hvd_prior_min_records()
                if prior_min is not None:
                    prior_replication_only = True
                    activation_records = min(
                        activation_records,
                        max(1, int(prior_min)),
                    )
        self.cumulative_prior_replication_only[i] = bool(
            prior_replication_only)
        self.cumulative_activation_records[i] = int(activation_records)
        if self.mode == "factor" and len(recs) >= activation_records:
            X_c = []
            y_c = []
            dof_c = []
            exposure_layout_ref = None
            for (x, _), v in zip(recs, vals):
                feat = self._cumulative_features(x, problem, output_index=i)
                if feat is None:
                    X_c = []
                    break
                if exposure_layout_ref is None:
                    exposure_layout_ref = get_risk_exposure(
                        problem or self._last_problem,
                        x,
                        output_index=i,
                    )
                X_c.append(feat)
                y_c.append(max(float(v), self.floor))
                dof_c.append(self._record_replication_dof(i, x))
            if X_c:
                X_c = np.vstack(X_c)
                y_c = np.asarray(y_c, dtype=float)
                dof_c = np.asarray(dof_c, dtype=float)
                ridge_alpha = float(self.config.ridge_alpha)
                reg = ridge_alpha * np.eye(X_c.shape[1])
                prior_beta = prior_beta_probe
                prior_center = np.zeros(X_c.shape[1], dtype=float)
                scalar_calibration = False
                source_mixture_calibration = False
                source_mixture_diagnostics = None
                if prior_beta is None:
                    X_fit = X_c
                    y_fit = y_c
                    dof_fit = dof_c
                    reg[0, 0] = 0.0
                    rhs = X_fit.T @ y_fit
                    solve_gram = X_fit.T @ X_fit + reg
                else:
                    prior_precision = ridge_alpha
                    if hasattr(problem_ref, "cumulative_hvd_prior_precision"):
                        learned_precision = problem_ref.cumulative_hvd_prior_precision(
                            output_index=i)
                        if learned_precision is not None:
                            prior_precision = max(
                                float(learned_precision),
                                ridge_alpha,
                            )
                    if hasattr(problem_ref, "cumulative_hvd_prior_upper_scale"):
                        upper_scale = problem_ref.cumulative_hvd_prior_upper_scale(
                            output_index=i)
                        if upper_scale is not None:
                            self.cumulative_prior_upper_scale[i] = float(max(
                                float(upper_scale),
                                1.0,
                            ))
                    if prior_replication_only:
                        replicate_mask = np.asarray([
                            tuple(x) in self.replicated_keys.get(i, set())
                            for x, _ in recs
                        ], dtype=bool)
                        prequential_mask = np.asarray([
                            (
                                tuple(x) in self.prequential_upper_records.get(
                                    i, {})
                                and not replicate_mask[index]
                            )
                            for index, (x, _) in enumerate(recs)
                        ], dtype=bool)
                        if (
                            self.config.cumulative_target_evidence_mode
                            != "prequential_upper"
                        ):
                            prequential_mask[:] = False
                        target_evidence_mask = (
                            replicate_mask | prequential_mask)
                        X_fit = X_c[target_evidence_mask]
                        y_fit = y_c[target_evidence_mask].copy()
                        selected_records = np.flatnonzero(
                            target_evidence_mask)
                        for local_index, record_index in enumerate(
                            selected_records
                        ):
                            if prequential_mask[record_index]:
                                key = tuple(recs[record_index][0])
                                y_fit[local_index] = max(float(
                                    self.prequential_upper_records[i][key]
                                ), self.floor)
                        dof_fit = dof_c[target_evidence_mask]
                    else:
                        replicate_mask = np.zeros(len(recs), dtype=bool)
                        prequential_mask = np.zeros(len(recs), dtype=bool)
                        target_evidence_mask = np.ones(len(recs), dtype=bool)
                        X_fit = X_c
                        y_fit = y_c
                        dof_fit = dof_c
                    source_scale_mean = None
                    if hasattr(problem_ref, "cumulative_hvd_prior_scale_mean"):
                        source_scale_mean = problem_ref.cumulative_hvd_prior_scale_mean(
                            output_index=i)
                    normalized_shape = source_scale_mean is not None
                    prior_shape_beta = np.asarray(prior_beta, dtype=float)
                    if normalized_shape and exposure_layout_ref is not None:
                        try:
                            prior_shape_beta, _ = project_cumulative_beta(
                                prior_shape_beta,
                                exposure_layout_ref,
                            )
                            prior_shape_beta = np.asarray(
                                prior_shape_beta, dtype=float)
                        except (ValueError, IndexError):
                            prior_shape_beta = np.asarray(
                                prior_beta, dtype=float)
                    source_prediction = np.maximum(
                        X_c @ prior_shape_beta,
                        self.floor,
                    )
                    informative = y_c > 10.0 * self.floor
                    if prior_replication_only:
                        if len(y_fit):
                            target_values = y_fit
                            target_source_prediction = source_prediction[
                                target_evidence_mask]
                            if np.any(prequential_mask) and np.any(
                                replicate_mask
                            ):
                                scale_source = (
                                    "replication_and_prequential_upper")
                            elif np.any(prequential_mask):
                                scale_source = "prequential_upper"
                            else:
                                scale_source = "within_solution_replication"
                        else:
                            target_values = np.zeros(0, dtype=float)
                            target_source_prediction = np.zeros(
                                0, dtype=float)
                            scale_source = "source_prior_fallback"
                    else:
                        identifiable = int(np.sum(informative)) >= max(
                            2,
                            int(np.ceil(0.5 * len(y_c))),
                        )
                        target_values = (
                            y_c if identifiable else np.zeros(0, dtype=float)
                        )
                        target_source_prediction = (
                            source_prediction
                            if identifiable
                            else np.zeros(0, dtype=float)
                        )
                        scale_source = (
                            "cross_sectional_moment"
                            if identifiable
                            else "source_prior_fallback"
                        )
                    ratios = np.asarray(
                        target_values / target_source_prediction,
                        dtype=float,
                    ) if len(target_values) else np.zeros(0, dtype=float)
                    finite_ratio = np.isfinite(ratios)
                    ratios = ratios[finite_ratio]
                    ratio_dof = (
                        np.maximum(np.asarray(dof_fit, dtype=float), 1.0)[
                            finite_ratio]
                        if len(target_values)
                        else np.zeros(0, dtype=float)
                    )
                    scale_prior_mean = (
                        max(float(source_scale_mean), self.floor)
                        if normalized_shape
                        else 1.0
                    )
                    scale_lower = self.floor if normalized_shape else 0.05
                    scale_upper = (
                        20.0 * max(
                            scale_prior_mean,
                            float(np.mean(y_c)),
                            self.floor,
                        )
                        if normalized_shape
                        else 20.0
                    )
                    ratios = np.clip(ratios, scale_lower, scale_upper)
                    if len(ratios):
                        if normalized_shape:
                            # With a frozen normalized source shape, sample
                            # variances identify one positive target scale.
                            # The chi-square likelihood weights each ratio by
                            # its replication degrees of freedom.
                            target_weight = float(np.sum(ratio_dof))
                            moment_ratio = float(np.sum(
                                ratio_dof * ratios
                            ) / max(target_weight, 1.0))
                        else:
                            target_weight = float(len(target_values))
                            moment_ratio = float(
                                np.mean(target_values) / max(
                                    float(np.mean(target_source_prediction)),
                                    self.floor,
                                )
                            )
                        moment_ratio = float(np.clip(
                            moment_ratio, scale_lower, scale_upper))
                        prior_scale = float(
                            (
                                target_weight * moment_ratio
                                + prior_precision * scale_prior_mean
                            )
                            / (target_weight + prior_precision)
                        )
                        prior_scale = float(np.clip(
                            prior_scale, scale_lower, scale_upper))
                        if normalized_shape:
                            prior_scale_se = float(
                                np.sqrt(2.0)
                                * prior_scale
                                / np.sqrt(max(
                                    target_weight + prior_precision, 1.0))
                            )
                        else:
                            prior_scale_se = float(
                                np.std(ratios, ddof=1) / np.sqrt(len(ratios))
                                if len(ratios) > 1
                                else 0.0
                            )
                    else:
                        prior_scale = scale_prior_mean
                        prior_scale_se = 0.0
                    use_source_mixture = bool(
                        prior_replication_only
                        and normalized_shape
                        and prior_component_probe is not None
                        and len(prior_component_probe) > 1
                    )
                    if use_source_mixture:
                        source_task_weights = None
                        source_task = self.cumulative_source_task_posterior.get(i)
                        if (
                            self.config.cumulative_source_task_weight_mode
                            == "constraint_mean"
                            and source_task is not None
                        ):
                            task_weight = dict(zip(
                                source_task["component_names"],
                                source_task["posterior_weights"],
                            ))
                            source_task_weights = np.asarray([
                                task_weight.get(
                                    domain,
                                    task_weight.get(f"source:{domain}", 0.0),
                                )
                                for domain in prior_component_domains
                            ], dtype=float)
                            null_weight = max(float(
                                source_task.get("target_null_weight", 0.0)
                            ), 0.0)
                            if null_weight > 0.0:
                                # The null task does not borrow a source-domain
                                # shape.  Its PSD shape is the target pilot's
                                # pooled variance, expressed in the same
                                # normalized coefficient units as the source
                                # components.
                                null_component = np.zeros(
                                    prior_component_probe.shape[1],
                                    dtype=float,
                                )
                                null_component[0] = max(
                                    self.global_var[i] / max(
                                        scale_prior_mean, self.floor),
                                    self.floor,
                                )
                                prior_component_probe = np.vstack([
                                    prior_component_probe,
                                    null_component,
                                ])
                                prior_component_domains = [
                                    *prior_component_domains,
                                    "target:null",
                                ]
                                source_task_weights = np.concatenate([
                                    source_task_weights,
                                    [null_weight],
                                ])
                            if float(np.sum(source_task_weights)) <= 0.0:
                                source_task_weights = None
                        component_design = np.maximum(
                            X_fit @ prior_component_probe.T,
                            self.floor,
                        )
                        component_weights, component_covariance, (
                            source_mixture_diagnostics
                        ) = self._fit_source_shape_mixture(
                            component_design,
                            y_fit,
                            dof_fit,
                            source_scale=scale_prior_mean,
                            prior_precision=prior_precision,
                            upper_scale=self.cumulative_prior_upper_scale[i],
                            prior_weights=source_task_weights,
                        )
                        centered_prior = (
                            prior_component_probe.T @ component_weights)
                        prior_center = centered_prior
                        prior_scale = float(np.sum(component_weights))
                        prior_scale_se = float(np.sqrt(max(
                            np.ones(len(component_weights), dtype=float)
                            @ component_covariance
                            @ np.ones(len(component_weights), dtype=float),
                            0.0,
                        )))
                        self.cumulative_prior_components[i] = (
                            prior_component_probe.copy())
                        self.cumulative_prior_component_domains[i] = list(
                            prior_component_domains)
                        self.cumulative_prior_component_weights[i] = (
                            component_weights.copy())
                        self.cumulative_prior_component_covariance[i] = (
                            component_covariance.copy())
                        self.cumulative_prior_shape_target_dof[i] = float(
                            source_mixture_diagnostics["target_dof"])
                        self.cumulative_prior_scale[i] = float(prior_scale)
                        self.cumulative_prior_scale_se[i] = float(prior_scale_se)
                        source_mixture_calibration = True
                    else:
                        centered_prior = prior_scale * prior_shape_beta
                        prior_center = centered_prior
                        reg = prior_precision * np.eye(X_c.shape[1])
                        rhs = X_fit.T @ y_fit + reg @ centered_prior
                        solve_gram = X_fit.T @ X_fit + reg
                        scalar_calibration = bool(
                            prior_replication_only and normalized_shape)
                    self.cumulative_prior_used[i] = True
                    self.cumulative_prior_precision[i] = float(prior_precision)
                    self.cumulative_prior_scale[i] = float(prior_scale)
                    self.cumulative_prior_scale_se[i] = float(prior_scale_se)
                    self.cumulative_prior_target_weight[i] = int(
                        len(target_values))
                    self.cumulative_prior_scale_source[i] = scale_source
                if scalar_calibration or source_mixture_calibration:
                    # Four or five replicated target policies cannot identify
                    # a full cumulative-risk coefficient vector. Preserve the
                    # source-learned cone shape and update only its target
                    # scale; the unrestricted projected IRLS fit remains the
                    # no-source/control path.
                    beta_c = np.asarray(prior_center, dtype=float)
                else:
                    try:
                        beta_c = np.linalg.solve(solve_gram, rhs)
                    except np.linalg.LinAlgError:
                        beta_c = np.linalg.lstsq(
                            solve_gram,
                            rhs,
                            rcond=None,
                        )[0]
                beta_c = np.asarray(beta_c, dtype=float)
                params = None
                provider_active = False
                if (
                    exposure_layout_ref is not None
                    and len(beta_c) == len(self._cumulative_features(
                        recs[0][0],
                        problem,
                        output_index=i,
                    ))
                ):
                    try:
                        if source_mixture_calibration:
                            projected, params = project_cumulative_beta(
                                beta_c,
                                exposure_layout_ref,
                            )
                            self.cumulative_fit_method[i] = (
                                {
                                    "prequential_upper": (
                                        "prequential_upper_source_shape_mixture"
                                    ),
                                    "replication_and_prequential_upper": (
                                        "hybrid_source_shape_mixture"
                                    ),
                                }.get(
                                    scale_source,
                                    "replication_source_shape_mixture",
                                )
                                if len(X_fit) else
                                "prior_source_shape_mixture"
                            )
                            self.cumulative_fit_effective_dof[i] = float(
                                source_mixture_diagnostics["target_dof"])
                            self.cumulative_fit_weight_range[i] = (
                                source_mixture_diagnostics["weight_range"])
                        elif scalar_calibration and len(X_fit):
                            projected, params = project_cumulative_beta(
                                beta_c,
                                exposure_layout_ref,
                            )
                            self.cumulative_fit_method[i] = (
                                "replication_scalar_calibration")
                            self.cumulative_fit_effective_dof[i] = float(
                                np.sum(dof_fit))
                            self.cumulative_fit_weight_range[i] = (1.0, 1.0)
                        elif scalar_calibration:
                            projected, params = project_cumulative_beta(
                                beta_c,
                                exposure_layout_ref,
                            )
                            self.cumulative_fit_method[i] = "prior_projection"
                        elif len(X_fit):
                            projected, params, fit_diagnostics = (
                                self._fit_constrained_cumulative_ridge(
                                    X_fit,
                                    y_fit,
                                    dof_fit,
                                    exposure_layout_ref,
                                    reg,
                                    prior_center,
                                    beta_c,
                                )
                            )
                            self.cumulative_fit_method[i] = str(
                                fit_diagnostics["method"])
                            self.cumulative_fit_effective_dof[i] = float(
                                fit_diagnostics["effective_dof"])
                            self.cumulative_fit_weight_range[i] = tuple(
                                float(value)
                                for value in fit_diagnostics["weight_range"]
                            )
                        else:
                            projected, params = project_cumulative_beta(
                                beta_c,
                                exposure_layout_ref,
                            )
                            self.cumulative_fit_method[i] = "prior_projection"
                        if len(projected) == len(beta_c):
                            beta_c = projected
                            provider_active = True
                        else:
                            params = None
                    except (ValueError, IndexError):
                        params = None
                        provider_active = False
                if not provider_active:
                    params = None
                    beta_c = np.maximum(beta_c, 0.0)
                    beta_c[0] = max(float(beta_c[0]), 0.1 * self.floor)
                    self.cumulative_fit_method[i] = "nonnegative_projection"
                    self.cumulative_fit_effective_dof[i] = float(
                        np.sum(dof_fit))
                self.cumulative_beta[i] = beta_c
                self.cumulative_params[i] = params
                self.cumulative_provider_active[i] = bool(provider_active)
                if len(X_fit):
                    pred_c = np.maximum(X_fit @ beta_c, self.floor)
                    self.cumulative_fit_rmse[i] = float(np.sqrt(
                        np.mean((pred_c - y_fit) ** 2)))
                energy = beta_c[1:] ** 2
                total = float(np.sum(energy))
                if total > 0:
                    self.factor_energy[i] = (energy / total).tolist()

        if self.mode in ("orthogonal", "factor") and len(recs) >= int(self.config.activation_min_records):
            X = np.vstack([self._features(x, problem) for x, _ in recs])
            y = np.log(vals)
            reg = float(self.config.ridge_alpha) * np.eye(X.shape[1])
            reg[0, 0] = 0.0
            try:
                beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X.T @ X + reg, X.T @ y, rcond=None)[0]
            self.beta[i] = beta
            if self.mode == "factor" and self.cumulative_beta.get(i) is None:
                coefs = np.asarray(beta[1:], dtype=float)
                energy = coefs ** 2
                total = float(np.sum(energy))
                if total > 0:
                    self.factor_energy[i] = (energy / total).tolist()
                else:
                    self.factor_energy[i] = [0.0 for _ in energy]
        elif self.mode in ("orthogonal", "factor"):
            self.beta[i] = None
            self.factor_energy[i] = []

    def _class_variance(self, i, x, problem=None):
        c = self.risk_class(x, problem)
        return float(self.class_var.get(i, {}).get(c, self.global_var.get(i, 0.01)))

    def _class_variance_many(self, i, X, problem=None):
        classes = self.risk_classes_many(X, problem)
        global_v = float(self.global_var.get(int(i), 0.01))
        c_map = self.class_var.get(int(i), {})
        return np.array([
            float(max(c_map.get(int(c), global_v), self.floor))
            for c in classes
        ], dtype=float)

    def _predict_cumulative_variance(self, i, x, problem=None):
        i = int(i)
        if not self._cumulative_active(i):
            return None
        feat = self._cumulative_features(x, problem, output_index=i)
        beta = self.cumulative_beta.get(i)
        if feat is None or beta is None:
            return None
        pred = float(np.maximum(float(feat @ beta), self.floor))
        cap = self._residual_variance_cap(i, problem)
        if cap is not None:
            pred = min(pred, cap)
        return float(max(pred, self.floor))

    def _predict_cumulative_variance_many(self, i, X, problem=None):
        i = int(i)
        if len(X) == 0 or not self._cumulative_active(i):
            return None
        F = self._cumulative_feature_matrix(X, problem, output_index=i)
        beta = self.cumulative_beta.get(i)
        if F is None or beta is None:
            return None
        pred = np.maximum(F @ beta, self.floor)
        cap = self._residual_variance_cap(i, problem)
        if cap is not None:
            pred = np.minimum(pred, cap)
        return np.maximum(pred, self.floor)

    def predict_cumulative_variance(self, i, x, problem=None):
        """Public cumulative-risk variance prediction when available."""
        pred = self._predict_cumulative_variance(i, x, problem)
        if pred is None:
            pred = self.predict_variance(i, x, problem)
        return float(max(pred, self.floor))

    def information_reduction_many(
        self,
        i,
        action_points,
        reference_points,
        problem=None,
        *,
        action_reliability=None,
        reference_weights=None,
    ):
        """Approximate HVD parameter-uncertainty reduction per unit-cost action.

        For a frozen normalized source shape ``v(x) = s h(x)``, this uses the
        scaled-chi-square Fisher precision of the target scale ``s``. This is
        a covariance in physical variance units and can therefore enter the
        chance-margin delta method.

        The unrestricted cumulative variance model is linear in its risk
        features. Let ``P`` be the same replication-aware IRLS information
        matrix used by the fit and ``Q`` the boundary-weighted second moment
        of a reference policy pool. Adding a variance observation with feature
        ``phi`` and IRLS reliability weight ``w`` reduces integrated
        linear-predictor variance by

        ``w phi' P^-1 Q P^-1 phi / (1 + w phi' P^-1 phi)``.

        This quantity uses no target truth. A caller can downweight a fresh
        residual-square observation while assigning unit reliability to a
        clean within-policy replication. For pooled HVD the same expression
        reduces to scalar variance-parameter information.
        """

        i = int(i)
        actions = [tuple(int(v) for v in x) for x in action_points]
        references = [tuple(int(v) for v in x) for x in reference_points]
        if not actions:
            return np.zeros(0, dtype=float)
        if not references:
            references = list(actions)

        use_cumulative = bool(
            self.mode == "factor"
            and self.config.use_cumulative_provider
            and self._cumulative_features(actions[0], problem, i) is not None
        )
        replication_only = bool(
            use_cumulative
            and self.cumulative_prior_replication_only.get(i, False)
        )
        source_mixture_calibration = bool(
            replication_only
            and self.cumulative_prior_component_covariance.get(i) is not None
            and self.cumulative_prior_components.get(i) is not None
        )
        scalar_calibration = bool(
            replication_only
            and not source_mixture_calibration
            and self.cumulative_prior_used.get(i, False)
            and self.cumulative_prior_scale.get(i) is not None
            and self.cumulative_beta.get(i) is not None
        )
        if use_cumulative:
            raw_action_features = self._cumulative_feature_matrix(
                actions, problem, output_index=i)
            raw_reference_features = self._cumulative_feature_matrix(
                references, problem, output_index=i)
            if source_mixture_calibration:
                components = np.asarray(
                    self.cumulative_prior_components[i], dtype=float)
                action_features = np.maximum(
                    raw_action_features @ components.T, self.floor)
                reference_features = np.maximum(
                    raw_reference_features @ components.T, self.floor)

                def feature(point):
                    raw = self._cumulative_features(point, problem, i)
                    return np.maximum(raw @ components.T, self.floor)
            elif scalar_calibration:
                scale = max(
                    float(self.cumulative_prior_scale[i]), self.floor)
                shape_beta = np.asarray(
                    self.cumulative_beta[i], dtype=float) / scale
                action_shape = np.maximum(
                    raw_action_features @ shape_beta, self.floor)
                reference_shape = np.maximum(
                    raw_reference_features @ shape_beta, self.floor)
                action_unclipped = scale * action_shape
                reference_unclipped = scale * reference_shape
                action_active = action_unclipped > self.floor
                reference_active = reference_unclipped > self.floor
                cap = self._residual_variance_cap(i, problem)
                if cap is not None:
                    action_active &= action_unclipped < cap
                    reference_active &= reference_unclipped < cap
                # Predictions use clip(s h(x), floor, cap), so the scale
                # derivative is h(x) only in the unsaturated region. Using
                # the raw source shape past the cap creates fictitious VOI.
                action_features = np.where(
                    action_active, action_shape, 0.0)[:, None]
                reference_features = np.where(
                    reference_active, reference_shape, 0.0)[:, None]

                def feature(point):
                    raw = self._cumulative_features(point, problem, i)
                    shape = max(float(raw @ shape_beta), self.floor)
                    unclipped = scale * shape
                    active = unclipped > self.floor
                    if cap is not None:
                        active = active and unclipped < cap
                    return np.asarray([shape if active else 0.0], dtype=float)
            else:
                action_features = raw_action_features
                reference_features = raw_reference_features

                def feature(point):
                    return self._cumulative_features(point, problem, i)
        elif self.mode in ("orthogonal", "factor"):
            action_features = np.vstack([
                self._features(point, problem) for point in actions
            ])
            reference_features = np.vstack([
                self._features(point, problem) for point in references
            ])

            def feature(point):
                return self._features(point, problem)
        else:
            action_features = np.ones((len(actions), 1), dtype=float)
            reference_features = np.ones((len(references), 1), dtype=float)

            def feature(point):
                del point
                return np.ones(1, dtype=float)

        if source_mixture_calibration:
            if reference_weights is None:
                ref_weight = np.ones(len(references), dtype=float)
            else:
                ref_weight = np.maximum(
                    np.asarray(reference_weights, dtype=float).reshape(-1),
                    0.0,
                )
                if len(ref_weight) != len(references):
                    raise ValueError("reference_weights length mismatch")
            ref_total = float(np.sum(ref_weight))
            if not np.isfinite(ref_total) or ref_total <= 0.0:
                ref_weight = np.ones(len(references), dtype=float)
                ref_total = float(len(references))
            ref_weight /= ref_total
            if action_reliability is None:
                reliability = np.ones(len(actions), dtype=float)
            else:
                reliability = np.clip(
                    np.asarray(action_reliability, dtype=float).reshape(-1),
                    0.0,
                    1.0,
                )
                if len(reliability) != len(actions):
                    raise ValueError("action_reliability length mismatch")
            covariance = np.asarray(
                self.cumulative_prior_component_covariance[i], dtype=float)
            reference_second_moment = (
                reference_features.T
                @ (ref_weight[:, None] * reference_features)
            )
            predicted = np.maximum(
                self.predict_variance_many(i, actions, problem), self.floor)
            reductions = np.zeros(len(actions), dtype=float)
            for index, row in enumerate(action_features):
                if reliability[index] <= 0.0:
                    continue
                covariance_row = covariance @ row
                latent_variance = max(float(row @ covariance_row), 0.0)
                observation_variance = (
                    2.0 * predicted[index] ** 2
                    / max(float(reliability[index]), 1e-8)
                )
                denominator = max(
                    observation_variance + latent_variance,
                    self.floor ** 2,
                )
                reductions[index] = float(
                    covariance_row
                    @ reference_second_moment
                    @ covariance_row
                    / denominator
                )
            return np.maximum(reductions, 0.0)

        if scalar_calibration:
            if reference_weights is None:
                ref_weight = np.ones(len(references), dtype=float)
            else:
                ref_weight = np.maximum(
                    np.asarray(reference_weights, dtype=float).reshape(-1),
                    0.0,
                )
                if len(ref_weight) != len(references):
                    raise ValueError("reference_weights length mismatch")
            ref_total = float(np.sum(ref_weight))
            if not np.isfinite(ref_total) or ref_total <= 0.0:
                ref_weight = np.ones(len(references), dtype=float)
                ref_total = float(len(references))
            ref_weight /= ref_total

            if action_reliability is None:
                reliability = np.ones(len(actions), dtype=float)
            else:
                reliability = np.clip(
                    np.asarray(action_reliability, dtype=float).reshape(-1),
                    0.0,
                    1.0,
                )
                if len(reliability) != len(actions):
                    raise ValueError("action_reliability length mismatch")
            reliability *= np.asarray(
                action_features[:, 0] > 0.0, dtype=float)

            # If v(x) = s h(x), then a sample variance divided by h(x)
            # has scaled-chi-square variance 2 s^2 / dof. Consequently the
            # Fisher precision for s is dof / (2 s^2), in physical variance
            # units. This is the covariance needed by chance-margin VOI; the
            # median-normalized IRLS matrix below is only a numerical fitting
            # geometry and cannot be added to GPR variance reduction.
            scale = max(float(self.cumulative_prior_scale[i]), self.floor)
            source_pseudo_dof = max(float(
                self.cumulative_prior_precision.get(i) or 0.0), 1e-8)
            target_dof = float(sum(
                self._record_replication_dof(i, point)
                for point in self.replicated_keys.get(i, set())
            ))
            current_precision = (
                source_pseudo_dof + target_dof
            ) / max(2.0 * scale ** 2, self.floor ** 2)
            increment = reliability / max(
                2.0 * scale ** 2, self.floor ** 2)
            scale_variance_reduction = increment / np.maximum(
                current_precision * (current_precision + increment),
                self.floor,
            )
            integrated_shape_square = float(np.sum(
                ref_weight * np.asarray(
                    reference_features[:, 0], dtype=float) ** 2
            ))
            return np.maximum(
                scale_variance_reduction * integrated_shape_square,
                0.0,
            )

        dimension = int(action_features.shape[1])
        precision = max(float(self.config.ridge_alpha), 1e-10)
        if use_cumulative and self.cumulative_prior_used.get(i, False):
            learned = self.cumulative_prior_precision.get(i)
            if learned is not None:
                precision = max(precision, float(learned))
        information = precision * np.eye(dimension, dtype=float)

        record_rows = []
        raw_record_weights = []
        for point, _ in self.records.get(i, []):
            point = tuple(point)
            if replication_only and point not in self.replicated_keys.get(i, set()):
                continue
            row = np.asarray(feature(point), dtype=float)
            if row.shape != (dimension,) or not np.all(np.isfinite(row)):
                continue
            predicted = max(
                float(self.predict_variance(i, point, problem)), self.floor)
            # This is the same relative information geometry as the
            # replication-aware projected IRLS fit: Var(S^2) is proportional
            # to v(x)^2 / dof.
            weight = self._record_replication_dof(i, point) / max(
                predicted ** 2, self.floor ** 2)
            record_rows.append(row)
            raw_record_weights.append(weight)
        finite_weights = np.asarray([
            weight for weight in raw_record_weights
            if np.isfinite(weight) and weight > 0.0
        ], dtype=float)
        weight_scale = (
            float(np.median(finite_weights)) if len(finite_weights) else 1.0
        )
        weight_clip = max(float(self.config.cumulative_weight_clip), 1.0)
        for row, raw_weight in zip(record_rows, raw_record_weights):
            weight = float(np.clip(
                raw_weight / max(weight_scale, 1e-12),
                1.0 / weight_clip,
                weight_clip,
            ))
            information += weight * np.outer(row, row)

        inverse = np.linalg.pinv(
            0.5 * (information + information.T), hermitian=True)
        if reference_weights is None:
            ref_weight = np.ones(len(references), dtype=float)
        else:
            ref_weight = np.maximum(
                np.asarray(reference_weights, dtype=float).reshape(-1), 0.0)
            if len(ref_weight) != len(references):
                raise ValueError("reference_weights length mismatch")
        ref_total = float(np.sum(ref_weight))
        if not np.isfinite(ref_total) or ref_total <= 0.0:
            ref_weight = np.ones(len(references), dtype=float)
            ref_total = float(len(references))
        ref_weight /= ref_total
        reference_moment = reference_features.T @ (
            ref_weight[:, None] * reference_features)

        if action_reliability is None:
            reliability = np.ones(len(actions), dtype=float)
        else:
            reliability = np.clip(
                np.asarray(action_reliability, dtype=float).reshape(-1),
                0.0,
                1.0,
            )
            if len(reliability) != len(actions):
                raise ValueError("action_reliability length mismatch")
        action_variance = np.asarray([
            max(float(self.predict_variance(i, point, problem)), self.floor)
            for point in actions
        ], dtype=float)
        update_weight = reliability / np.maximum(
            action_variance ** 2, self.floor ** 2)
        update_weight = np.clip(
            update_weight / max(weight_scale, 1e-12),
            0.0,
            weight_clip,
        )
        projected = action_features @ inverse
        leverage = np.einsum("ij,ij->i", projected, action_features)
        integrated = np.einsum(
            "ij,jk,ik->i", projected, reference_moment, projected)
        gain = update_weight * np.maximum(integrated, 0.0) / np.maximum(
            1.0 + update_weight * np.maximum(leverage, 0.0), 1e-12)
        return np.maximum(np.asarray(gain, dtype=float), 0.0)

    def certification_margin_information_reduction_many(
        self,
        i,
        action_points,
        reference_points,
        problem=None,
        *,
        action_reliability=None,
        reference_weights=None,
        z_alpha=1.0,
    ):
        """Expected reduction of the HVD certification radius.

        The hierarchical source-shape posterior contributes

        ``z_h * sqrt(phi(x)' Cov(theta) phi(x))``

        to ``v_C_plus``. A prospective variance observation has a deterministic
        rank-one covariance update under the current scaled-chi-square
        information model. This method maps that update through
        ``z_alpha * sqrt(v_C_plus)`` before integrating over reference policies,
        so its output has the same response units as a GPR confidence-radius
        reduction. It intentionally returns zero when this covariance bridge is
        unavailable instead of mixing an arbitrary HVD score into the margin.
        """

        i = int(i)
        actions = [tuple(int(v) for v in x) for x in action_points]
        references = [tuple(int(v) for v in x) for x in reference_points]
        if not actions:
            return np.zeros(0, dtype=float)
        if not references:
            references = list(actions)
        components = self.cumulative_prior_components.get(i)
        covariance = self.cumulative_prior_component_covariance.get(i)
        source_mixture_active = bool(
            self.mode == "factor"
            and self.config.use_cumulative_provider
            and self.cumulative_prior_replication_only.get(i, False)
            and components is not None
            and covariance is not None
        )
        if not source_mixture_active:
            return np.zeros(len(actions), dtype=float)

        action_raw = self._cumulative_feature_matrix(
            actions, problem, output_index=i)
        reference_raw = self._cumulative_feature_matrix(
            references, problem, output_index=i)
        if action_raw is None or reference_raw is None:
            return np.zeros(len(actions), dtype=float)
        components = np.asarray(components, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        action_features = np.maximum(
            action_raw @ components.T, self.floor)
        reference_features = np.maximum(
            reference_raw @ components.T, self.floor)

        if action_reliability is None:
            reliability = np.ones(len(actions), dtype=float)
        else:
            reliability = np.clip(
                np.asarray(action_reliability, dtype=float).reshape(-1),
                0.0,
                1.0,
            )
            if len(reliability) != len(actions):
                raise ValueError("action_reliability length mismatch")
        if reference_weights is None:
            weights = np.ones(len(references), dtype=float)
        else:
            weights = np.maximum(
                np.asarray(reference_weights, dtype=float).reshape(-1), 0.0)
            if len(weights) != len(references):
                raise ValueError("reference_weights length mismatch")
        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            weights = np.ones(len(references), dtype=float)
            total_weight = float(len(references))
        weights /= max(total_weight, self.floor)

        predicted_action = np.maximum(
            self.predict_variance_many(i, actions, problem), self.floor)
        certification_variance = np.maximum(
            self.predict_certification_variance_many(
                i, references, problem), self.floor)
        covariance_reference = reference_features @ covariance
        current_shape_variance = np.maximum(np.einsum(
            "ij,ij->i", covariance_reference, reference_features), 0.0)
        transfer_z = max(
            float(self.config.cumulative_transfer_upper_z), 0.0)
        current_guard = transfer_z * np.sqrt(current_shape_variance)
        z_alpha = max(float(z_alpha), 0.0)
        gains = np.zeros(len(actions), dtype=float)
        for index, row in enumerate(action_features):
            if reliability[index] <= 0.0:
                continue
            covariance_row = covariance @ row
            latent_variance = max(float(row @ covariance_row), 0.0)
            observation_variance = (
                2.0 * predicted_action[index] ** 2
                / max(float(reliability[index]), 1e-8)
            )
            denominator = max(
                observation_variance + latent_variance,
                self.floor ** 2,
            )
            cross = reference_features @ covariance_row
            shape_reduction = np.minimum(
                np.maximum(cross ** 2 / denominator, 0.0),
                current_shape_variance,
            )
            updated_guard = transfer_z * np.sqrt(np.maximum(
                current_shape_variance - shape_reduction, 0.0))
            guard_reduction = np.minimum(
                np.maximum(current_guard - updated_guard, 0.0),
                certification_variance - self.floor,
            )
            updated_certification = np.maximum(
                certification_variance - guard_reduction, self.floor)
            margin_reduction = z_alpha * (
                np.sqrt(certification_variance)
                - np.sqrt(updated_certification)
            )
            gains[index] = float(weights @ np.maximum(
                margin_reduction, 0.0))
        return np.maximum(gains, 0.0)

    def predict_variance(self, i, x, problem=None):
        """Predict observation-noise variance for one output channel."""
        i = int(i)
        if self.mode == "oracle":
            if problem is None:
                problem = self._last_problem
            if problem is not None and hasattr(problem, "true_sigma"):
                return float(max(problem.true_sigma(x)[i] ** 2, self.floor))
        if self.mode == "pooled":
            return float(max(self.global_var.get(i, 0.01), self.floor))
        if self.mode == "class":
            return float(max(self._class_variance(i, x, problem), self.floor))
        if self.mode == "factor":
            cumulative = self._predict_cumulative_variance(i, x, problem)
            if cumulative is not None:
                return cumulative
        beta = self.beta.get(i)
        if beta is None or not self._orthogonal_active(i):
            return float(max(self._class_variance(i, x, problem), self.floor))
        feat = self._features(x, problem)
        pred = float(np.exp(float(feat @ beta)))
        # Sparse-data guard: keep the smooth model near regime/global scale.
        class_v = self._class_variance(i, x, problem)
        pred = float(np.clip(pred, 0.25 * class_v, 4.0 * class_v))
        return float(max(pred, self.floor))

    def predict_variance_many(self, i, X, problem=None):
        """Batch variance prediction for candidate pools."""
        i = int(i)
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        if self.mode == "oracle":
            if problem is None:
                problem = self._last_problem
            if problem is not None and hasattr(problem, "true_sigma"):
                return np.array([
                    max(float(problem.true_sigma(x)[i] ** 2), self.floor)
                    for x in X
                ], dtype=float)
        if self.mode == "pooled":
            return np.full(len(X), max(float(self.global_var.get(i, 0.01)), self.floor))
        class_v = self._class_variance_many(i, X, problem)
        if self.mode == "factor":
            cumulative = self._predict_cumulative_variance_many(i, X, problem)
            if cumulative is not None:
                return cumulative
        beta = self.beta.get(i)
        if beta is None or not self._orthogonal_active(i):
            return class_v
        F = self._feature_matrix(X, problem)
        pred = np.exp(F @ beta)
        pred = np.clip(pred, 0.25 * class_v, 4.0 * class_v)
        return np.maximum(pred, self.floor)

    def model_uncertainty(self, i, x, problem=None):
        """Conservative variance-estimation uncertainty for certification.

        This term is not used as simulation-noise variance in the Kalman update.
        It only guards chance feasibility decisions against sparse class cells
        and not-yet-active smooth variance fits.
        """
        i = int(i)
        if self.mode not in ("orthogonal", "factor"):
            return 0.0
        base = self.predict_variance(i, x, problem)
        c = self.risk_class(x, problem)
        n_c = int(self.class_count.get(i, {}).get(c, 0))
        n_total = int(len(self.records.get(i, [])))
        if n_total <= 0:
            return float(base)
        class_unc = 1.0 / np.sqrt(n_c + 1.0)
        total_unc = 1.0 / np.sqrt(n_total + 1.0)
        activation_penalty = 1.0 if not self._variance_model_active(i) else 0.25
        return float(
            max(float(self.config.certification_kappa), 0.0)
            * base
            * max(class_unc, total_unc)
            * activation_penalty
        )

    def model_uncertainty_many(self, i, X, problem=None, base_variance=None):
        i = int(i)
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        if self.mode not in ("orthogonal", "factor"):
            return np.zeros(len(X), dtype=float)
        base = (
            np.asarray(base_variance, dtype=float)
            if base_variance is not None
            else self.predict_variance_many(i, X, problem)
        )
        classes = self.risk_classes_many(X, problem)
        counts = self.class_count.get(i, {})
        n_c = np.array([int(counts.get(int(c), 0)) for c in classes], dtype=float)
        n_total = float(len(self.records.get(i, [])))
        if n_total <= 0:
            return base.copy()
        class_unc = 1.0 / np.sqrt(n_c + 1.0)
        total_unc = 1.0 / np.sqrt(n_total + 1.0)
        activation_penalty = 1.0 if not self._variance_model_active(i) else 0.25
        scale = (
            max(float(self.config.certification_kappa), 0.0)
            * np.maximum(class_unc, total_unc)
            * activation_penalty
        )
        return np.maximum(base * scale, 0.0)

    def _source_mixture_guard_many(self, i, X, problem=None):
        """Pointwise upper radius for the transferred variance shape."""
        i = int(i)
        components = self.cumulative_prior_components.get(i)
        covariance = self.cumulative_prior_component_covariance.get(i)
        if components is None or covariance is None or len(X) == 0:
            return np.zeros(len(X), dtype=float)
        features = self._cumulative_feature_matrix(
            X, problem, output_index=i)
        if features is None:
            return np.zeros(len(X), dtype=float)
        component_values = np.maximum(
            features @ np.asarray(components, dtype=float).T,
            self.floor,
        )
        covariance = np.asarray(covariance, dtype=float)
        prediction_variance = np.einsum(
            "ni,ij,nj->n",
            component_values,
            covariance,
            component_values,
        )
        upper_z = max(float(self.config.cumulative_transfer_upper_z), 0.0)
        return upper_z * np.sqrt(np.maximum(prediction_variance, 0.0))

    def _source_mixture_guard(self, i, x, problem=None):
        return float(self._source_mixture_guard_many(
            i, [x], problem)[0])

    def predict_certification_variance(self, i, x, problem=None):
        """Variance used inside conservative chance feasibility checks."""
        cert = (
            self.predict_variance(i, x, problem)
            + self.model_uncertainty(i, x, problem)
        )
        if self.mode == "factor" and self._cumulative_active(i):
            cert += self._residual_tail_uncertainty(i)
        cert += self._high_frequency_residual_floor(i, x, problem)
        source_prior_active = bool(
            self.mode == "factor"
            and self._cumulative_active(i)
            and self.cumulative_prior_used.get(int(i), False)
            and self.cumulative_prior_replication_only.get(int(i), False)
        )
        if source_prior_active:
            if self.cumulative_prior_component_covariance.get(int(i)) is not None:
                cert += self._source_mixture_guard(i, x, problem)
            else:
                cert += (
                    max(float(self.cumulative_prior_upper_scale.get(i, 1.0)), 1.0)
                    - 1.0
                ) * self.predict_variance(i, x, problem)
        if self.mode in ("orthogonal", "factor") and not source_prior_active:
            # Smooth log-variance fits are allowed to guide learning, but
            # feasibility certification should not be more optimistic than the
            # coarse regime evidence in small-budget runs.
            cert = max(cert, self._class_variance(i, x, problem))
        return float(max(cert, self.floor))

    def predict_certification_variance_many(self, i, X, problem=None):
        base = self.predict_variance_many(i, X, problem)
        cert = base + self.model_uncertainty_many(i, X, problem, base)
        if self.mode == "factor" and self._cumulative_active(i):
            cert = cert + self._residual_tail_uncertainty(i)
        cert = cert + self._high_frequency_residual_floor_many(i, X, problem)
        source_prior_active = bool(
            self.mode == "factor"
            and self._cumulative_active(i)
            and self.cumulative_prior_used.get(int(i), False)
            and self.cumulative_prior_replication_only.get(int(i), False)
        )
        if source_prior_active:
            if self.cumulative_prior_component_covariance.get(int(i)) is not None:
                cert = cert + self._source_mixture_guard_many(i, X, problem)
            else:
                cert = cert + (
                    max(float(self.cumulative_prior_upper_scale.get(i, 1.0)), 1.0)
                    - 1.0
                ) * base
        if self.mode in ("orthogonal", "factor") and not source_prior_active:
            cert = np.maximum(cert, self._class_variance_many(i, X, problem))
        return np.maximum(cert, self.floor)

    def predict_decomposition(self, i, x, problem=None):
        """Return interpretable decomposition diagnostics."""
        i = int(i)
        c = self.risk_class(x, problem)
        feat = self._features(x, problem)
        beta = self.beta.get(i)
        factor_contrib = None
        if beta is not None:
            factor_contrib = (feat[1:] * beta[1:]).tolist()
        cumulative = None
        c_feat = self._cumulative_features(x, problem, output_index=i)
        c_beta = self.cumulative_beta.get(i)
        if c_feat is not None:
            problem_ref = problem or self._last_problem
            names = self._cumulative_feature_names(
                x,
                problem_ref,
                output_index=i,
                feat_len=len(c_feat),
            )
            fitted_contrib = None
            fitted_variance = None
            fitted_by_name = None
            fitted_blocks = None
            manifold_blocks = None
            if c_beta is not None:
                contrib = np.asarray(c_feat * c_beta, dtype=float)
                fitted_contrib = contrib.tolist()
                fitted_variance = float(max(float(c_feat @ c_beta), self.floor))
                if names is not None and len(names) == len(contrib):
                    fitted_by_name = {
                        str(name): float(value)
                        for name, value in zip(names, contrib)
                    }
                if len(contrib) >= 9:
                    fitted_blocks = {
                        "floor": float(contrib[0]),
                        "independent": float(np.sum(contrib[1:4])),
                        "shared": float(np.sum(contrib[4:7])),
                        "linear": float(np.sum(contrib[7:9])),
                        "total": float(max(np.sum(contrib), self.floor)),
                    }
                params = self.cumulative_params.get(i)
                exposure = get_risk_exposure(problem_ref, x, output_index=i)
                if params is not None and exposure is not None:
                    try:
                        fitted_blocks = decompose_cumulative_risk(exposure, params)
                        fitted_variance = float(max(fitted_blocks["total"], self.floor))
                    except (ValueError, TypeError):
                        pass
                decomposer = getattr(problem_ref, "_scolhkg_manifold_decomposer", None)
                if decomposer is not None:
                    manifold_blocks = decomposer.decompose(
                        x,
                        total_variance=fitted_variance,
                    )
            oracle = None
            oracle_blocks = None
            if problem_ref is not None and hasattr(problem_ref, "true_cumulative_risk_decomposition"):
                try:
                    oracle = problem_ref.true_cumulative_risk_decomposition(
                        x,
                        output_index=i,
                    )
                except TypeError:
                    oracle = problem_ref.true_cumulative_risk_decomposition(x)
                if oracle is not None:
                    oracle_blocks = {
                        key: float(oracle[key])
                        for key in ("floor", "independent", "shared", "linear", "total")
                        if key in oracle
                    }
            cumulative = {
                "active": bool(self._cumulative_active(i)),
                "feature_names": names,
                "features": c_feat.tolist(),
                "fitted_contrib": fitted_contrib,
                "fitted_by_name": fitted_by_name,
                "fitted_blocks": fitted_blocks,
                "manifold_blocks": manifold_blocks,
                "fitted_variance": fitted_variance,
                "fit_rmse": self.cumulative_fit_rmse.get(i),
                "provider_active": bool(self.cumulative_provider_active.get(i, False)),
                "v_C_plus": float(self.predict_certification_variance(i, x, problem)),
                "tail_guard": float(self._residual_tail_uncertainty(i))
                if self._cumulative_active(i) else 0.0,
                "source_shape_guard": float(
                    self._source_mixture_guard(i, x, problem))
                if self.cumulative_prior_component_covariance.get(i) is not None
                else float(
                    max(self.cumulative_prior_upper_scale.get(i, 1.0) - 1.0, 0.0)
                    * self.predict_variance(i, x, problem)
                ) if self.cumulative_prior_used.get(i, False) else 0.0,
                "high_frequency_floor": float(
                    self._high_frequency_residual_floor(i, x, problem)),
                "oracle": oracle,
                "oracle_blocks": oracle_blocks,
            }
        return {
            "mode": self.mode,
            "output_index": i,
            "variance": float(self.predict_variance(i, x, problem)),
            "pooled_variance": float(max(self.global_var.get(i, 0.01), self.floor)),
            "class_id": int(c),
            "class_variance": float(max(self._class_variance(i, x, problem), self.floor)),
            "class_count": int(self.class_count.get(i, {}).get(c, 0)),
            "factor_contrib": factor_contrib,
            "factor_energy": self.factor_energy.get(i, []),
            "cumulative": cumulative,
            "model_uncertainty": float(self.model_uncertainty(i, x, problem)),
            "residual_tail_uncertainty": float(self._residual_tail_uncertainty(i)),
            "high_frequency_floor": float(
                self._high_frequency_residual_floor(i, x, problem)),
            "certification_variance": float(
                self.predict_certification_variance(i, x, problem)),
        }

    def variance_information(self, i, x, problem=None):
        """Heuristic VOI for learning the variance decomposition."""
        i = int(i)
        pred = self.predict_variance(i, x, problem)
        c = self.risk_class(x, problem)
        n_c = self.class_count.get(i, {}).get(c, 0)
        class_unc = 1.0 / np.sqrt(n_c + 1.0)
        recs = self.records.get(i, [])
        if not recs:
            novelty = 1.0
        else:
            fx = self._features(x, problem)
            F = np.vstack([self._features(r[0], problem) for r in recs])
            dist = np.linalg.norm(F - fx[None, :], axis=1)
            novelty = float(np.min(dist) / (1.0 + np.min(dist)))
        return float(max(pred, self.floor) * (class_unc + novelty))

    def variance_information_many(self, i, X, problem=None):
        """Batch VOI for learning the variance decomposition."""
        i = int(i)
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        pred = self.predict_variance_many(i, X, problem)
        classes = self.risk_classes_many(X, problem)
        counts = self.class_count.get(i, {})
        n_c = np.array([int(counts.get(int(c), 0)) for c in classes], dtype=float)
        class_unc = 1.0 / np.sqrt(n_c + 1.0)

        recs = self.records.get(i, [])
        if not recs:
            novelty = np.ones(len(X), dtype=float)
        else:
            F_x = self._feature_matrix(X, problem)
            F_r = self._feature_matrix([r[0] for r in recs], problem)
            # Small matrices here; broadcasting avoids repeated Python calls.
            dist = np.linalg.norm(F_x[:, None, :] - F_r[None, :, :], axis=2)
            min_dist = np.min(dist, axis=1)
            novelty = min_dist / (1.0 + min_dist)
        return np.maximum(pred, self.floor) * (class_unc + novelty)

    def _cumulative_statistical_design_diagnostics(self, i):
        """Return the active HVD excitation audited by the Lean rate theorem.

        Target data usually calibrate a frozen source shape or a small source
        mixture, rather than refitting every raw cumulative coefficient.  The
        active Gram matrix must therefore be computed after that projection.
        This method is diagnostic only and never changes fitting or decisions.
        """
        i = int(i)
        problem = self._last_problem
        method = str(self.cumulative_fit_method.get(i, "inactive"))
        replicated = set(self.replicated_keys.get(i, set()))
        evidence = set(replicated)
        if str(self.config.cumulative_target_evidence_mode) == "prequential_upper":
            evidence.update(self.prequential_upper_records.get(i, {}))
        points = sorted(evidence)

        raw_dimension = 0
        active_dimension = 0
        projection = "inactive"
        raw_design = np.empty((0, 0), dtype=float)
        active_design = np.empty((0, 0), dtype=float)
        if problem is not None and points:
            try:
                candidate_design = self._cumulative_feature_matrix(
                    points,
                    problem,
                    output_index=i,
                )
                if candidate_design is not None:
                    raw_design = np.asarray(candidate_design, dtype=float)
            except (TypeError, ValueError, FloatingPointError):
                raw_design = np.empty((0, 0), dtype=float)
        if raw_design.ndim == 2 and raw_design.size:
            raw_dimension = int(raw_design.shape[1])
            components = self.cumulative_prior_components.get(i)
            beta = self.cumulative_beta.get(i)
            if (
                components is not None
                and "source_shape_mixture" in method
            ):
                component_matrix = np.asarray(components, dtype=float)
                if (
                    component_matrix.ndim == 2
                    and component_matrix.shape[1] == raw_dimension
                ):
                    active_design = raw_design @ component_matrix.T
                    projection = "source_shape_mixture"
            if active_design.size == 0 and (
                beta is not None
                and self.cumulative_prior_used.get(i, False)
            ):
                beta_vector = np.asarray(beta, dtype=float).reshape(-1)
                if len(beta_vector) == raw_dimension:
                    active_design = (raw_design @ beta_vector)[:, None]
                    projection = "frozen_source_shape_scalar"
            if active_design.size == 0:
                active_design = raw_design.copy()
                projection = "full_cumulative_feature"
            active_dimension = int(active_design.shape[1])

        def geometry(matrix):
            if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
                return {
                    "rank": 0,
                    "minimum_eigenvalue": 0.0,
                    "minimum_positive_eigenvalue": None,
                    "maximum_eigenvalue": 0.0,
                    "condition_number_positive_spectrum": None,
                }
            gram = (matrix.T @ matrix) / float(matrix.shape[0])
            gram = 0.5 * (gram + gram.T)
            eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
            maximum = float(np.max(eigenvalues))
            tolerance = max(
                np.finfo(float).eps * max(matrix.shape) * max(maximum, 1.0),
                1e-14,
            )
            positive = eigenvalues[eigenvalues > tolerance]
            rank = int(len(positive))
            minimum_positive = (
                None if len(positive) == 0 else float(np.min(positive))
            )
            return {
                "rank": rank,
                "minimum_eigenvalue": float(np.min(eigenvalues)),
                "minimum_positive_eigenvalue": minimum_positive,
                "maximum_eigenvalue": maximum,
                "condition_number_positive_spectrum": (
                    None
                    if minimum_positive is None or minimum_positive <= 0.0
                    else float(maximum / minimum_positive)
                ),
            }

        raw_geometry = geometry(raw_design)
        active_geometry = geometry(active_design)

        active_column_rms = np.empty(0, dtype=float)
        normalized_active_design = np.empty((0, 0), dtype=float)
        normalized_feature_radius = 0.0
        if (
            active_design.ndim == 2
            and active_design.shape[0] > 0
            and active_design.shape[1] > 0
        ):
            active_column_rms = np.sqrt(np.mean(
                np.asarray(active_design, dtype=float) ** 2,
                axis=0,
            ))
            scale_tolerance = max(
                np.finfo(float).eps
                * max(active_design.shape)
                * max(float(np.max(active_column_rms)), 1.0),
                1e-14,
            )
            safe_scale = np.where(
                active_column_rms > scale_tolerance,
                active_column_rms,
                1.0,
            )
            normalized_active_design = active_design / safe_scale[None, :]
            normalized_feature_radius = float(np.max(np.linalg.norm(
                normalized_active_design,
                axis=1,
            )))
        normalized_active_geometry = geometry(normalized_active_design)
        positive_column_rms = active_column_rms[
            active_column_rms > 0.0]
        column_scale_condition = (
            None
            if len(positive_column_rms) == 0
            else float(
                np.max(positive_column_rms)
                / np.min(positive_column_rms)
            )
        )
        return {
            "theory_contract": "v51_statistical_closure_v2",
            "target_evidence_mode": str(
                self.config.cumulative_target_evidence_mode),
            "fit_method": method,
            "projection": projection,
            "replicated_solution_count": int(len(replicated)),
            "target_evidence_solution_count": int(len(points)),
            "effective_replication_dof": float(
                self._effective_variance_dof(i)),
            "raw_feature_dimension": int(raw_dimension),
            "active_calibration_dimension": int(active_dimension),
            "raw_geometry": raw_geometry,
            "active_geometry": active_geometry,
            "normalized_active_geometry": normalized_active_geometry,
            "active_column_rms": active_column_rms.tolist(),
            "active_column_scale_condition": column_scale_condition,
            "normalized_feature_radius": normalized_feature_radius,
            "lean_excitation_kappa": float(
                len(points) * active_geometry["minimum_eigenvalue"]),
            "lean_normalized_excitation_kappa": float(
                len(points)
                * normalized_active_geometry["minimum_eigenvalue"]),
            "gram_normalization": "X_transpose_X_div_target_evidence_count",
            "normalized_gram_contract": (
                "active_columns_divided_by_target_evidence_rms"
            ),
            "active_identifiable": bool(
                active_dimension > 0
                and len(points) >= active_dimension
                and active_geometry["rank"] == active_dimension
                and active_geometry["minimum_eigenvalue"] > 0.0
            ),
            "normalized_active_identifiable": bool(
                active_dimension > 0
                and len(points) >= active_dimension
                and normalized_active_geometry["rank"] == active_dimension
                and normalized_active_geometry["minimum_eigenvalue"] > 0.0
            ),
        }

    def diagnostics(self):
        tail_delta = float(self.config.residual_tail_delta)
        tail_radius = {}
        for i in range(self.n_outputs):
            nu, b = gaussian_square_subexp_params(self.global_var.get(i, 0.01))
            tail_radius[str(i)] = {
                "delta": tail_delta,
                "nu": float(nu),
                "b": float(b),
                "effective_dof": float(self._effective_variance_dof(i)),
                "radius": float(sub_exponential_residual_square_radius(
                    nu, b, tail_delta)),
                "uncertainty": float(self._residual_tail_uncertainty(i)),
            }
        return {
            "mode": self.mode,
            "n_outputs": self.n_outputs,
            "global_var": {str(k): float(v) for k, v in self.global_var.items()},
            "class_count": {
                str(i): {str(c): int(n) for c, n in counts.items()}
                for i, counts in self.class_count.items()
            },
            "n_records": {
                str(i): int(len(self.records.get(i, [])))
                for i in range(self.n_outputs)
            },
            "has_beta": {
                str(i): self.beta.get(i) is not None
                for i in range(self.n_outputs)
            },
            "has_cumulative_beta": {
                str(i): self.cumulative_beta.get(i) is not None
                for i in range(self.n_outputs)
            },
            "cumulative_active": {
                str(i): bool(self._cumulative_active(i))
                for i in range(self.n_outputs)
            },
            "cumulative_provider_active": {
                str(i): bool(self.cumulative_provider_active.get(i, False))
                for i in range(self.n_outputs)
            },
            "cumulative_fit_rmse": {
                str(i): (
                    None if self.cumulative_fit_rmse.get(i) is None
                    else float(self.cumulative_fit_rmse[i])
                )
                for i in range(self.n_outputs)
            },
            "cumulative_fit_method": {
                str(i): str(self.cumulative_fit_method.get(i, "inactive"))
                for i in range(self.n_outputs)
            },
            "cumulative_information_geometry": {
                str(i): (
                    "scaled_chi_square_source_shape_mixture"
                    if self.cumulative_prior_component_covariance.get(i) is not None
                    else "scaled_chi_square_scalar"
                    if (
                        self.cumulative_prior_replication_only.get(i, False)
                        and self.cumulative_prior_used.get(i, False)
                        and self.cumulative_prior_scale.get(i) is not None
                        and self.cumulative_beta.get(i) is not None
                    )
                    else (
                        "irls_linear_predictor"
                        if self._cumulative_active(i)
                        else "inactive"
                    )
                )
                for i in range(self.n_outputs)
            },
            "cumulative_fit_effective_dof": {
                str(i): float(self.cumulative_fit_effective_dof.get(i, 0.0))
                for i in range(self.n_outputs)
            },
            "cumulative_fit_weight_range": {
                str(i): (
                    None
                    if self.cumulative_fit_weight_range.get(i) is None
                    else [
                        float(value)
                        for value in self.cumulative_fit_weight_range[i]
                    ]
                )
                for i in range(self.n_outputs)
            },
            "cumulative_statistical_design": {
                str(i): self._cumulative_statistical_design_diagnostics(i)
                for i in range(self.n_outputs)
            },
            "cumulative_prior_used": {
                str(i): bool(self.cumulative_prior_used.get(i, False))
                for i in range(self.n_outputs)
            },
            "cumulative_prior_precision": {
                str(i): (
                    None
                    if self.cumulative_prior_precision.get(i) is None
                    else float(self.cumulative_prior_precision[i])
                )
                for i in range(self.n_outputs)
            },
            "cumulative_prior_scale": {
                str(i): (
                    None
                    if self.cumulative_prior_scale.get(i) is None
                    else float(self.cumulative_prior_scale[i])
                )
                for i in range(self.n_outputs)
            },
            "cumulative_prior_scale_se": {
                str(i): (
                    None
                    if self.cumulative_prior_scale_se.get(i) is None
                    else float(self.cumulative_prior_scale_se[i])
                )
                for i in range(self.n_outputs)
            },
            "cumulative_prior_target_weight": {
                str(i): int(self.cumulative_prior_target_weight.get(i, 0))
                for i in range(self.n_outputs)
            },
            "cumulative_prior_scale_source": {
                str(i): str(self.cumulative_prior_scale_source.get(i, "none"))
                for i in range(self.n_outputs)
            },
            "cumulative_prior_upper_scale": {
                str(i): float(self.cumulative_prior_upper_scale.get(i, 1.0))
                for i in range(self.n_outputs)
            },
            "cumulative_prior_replication_only": {
                str(i): bool(self.cumulative_prior_replication_only.get(i, False))
                for i in range(self.n_outputs)
            },
            "cumulative_transfer_mode": str(
                self.config.cumulative_transfer_mode),
            "cumulative_source_task_weight_mode": str(
                self.config.cumulative_source_task_weight_mode),
            "cumulative_target_evidence_mode": str(
                self.config.cumulative_target_evidence_mode),
            "singleton_evidence_mode": str(
                self.config.singleton_evidence_mode),
            "cumulative_source_task_posterior": {
                str(i): (
                    None
                    if self.cumulative_source_task_posterior.get(i) is None
                    else dict(self.cumulative_source_task_posterior[i])
                )
                for i in range(self.n_outputs)
            },
            "cumulative_prior_component_count": {
                str(i): int(
                    0
                    if self.cumulative_prior_components.get(i) is None
                    else len(self.cumulative_prior_components[i])
                )
                for i in range(self.n_outputs)
            },
            "cumulative_prior_component_domains": {
                str(i): list(self.cumulative_prior_component_domains.get(i, []))
                for i in range(self.n_outputs)
            },
            "cumulative_prior_component_weights": {
                str(i): (
                    None
                    if self.cumulative_prior_component_weights.get(i) is None
                    else np.asarray(
                        self.cumulative_prior_component_weights[i],
                        dtype=float,
                    ).tolist()
                )
                for i in range(self.n_outputs)
            },
            "cumulative_prior_shape_target_dof": {
                str(i): float(self.cumulative_prior_shape_target_dof.get(i, 0.0))
                for i in range(self.n_outputs)
            },
            "replicated_solution_count": {
                str(i): int(len(self.replicated_keys.get(i, set())))
                for i in range(self.n_outputs)
            },
            "source_prior_singleton_count": {
                str(i): int(len(self.source_prior_pseudo_keys.get(i, set())))
                for i in range(self.n_outputs)
            },
            "prequential_upper_solution_count": {
                str(i): int(len(
                    self.prequential_upper_records.get(i, {})))
                for i in range(self.n_outputs)
            },
            "cumulative_activation_records": {
                str(i): int(self.cumulative_activation_records.get(
                    i, self.config.activation_min_records))
                for i in range(self.n_outputs)
            },
            "orthogonal_active": {
                str(i): bool(self._orthogonal_active(i))
                for i in range(self.n_outputs)
            },
            "activation_min_records": int(self.config.activation_min_records),
            "use_cumulative_provider": bool(
                self.config.use_cumulative_provider),
            "certification_kappa": float(self.config.certification_kappa),
            "certification_uses_class_floor": bool(self.mode in ("orthogonal", "factor")),
            "uses_manifold_hvd_features": bool(getattr(
                self._last_problem,
                "_scolhkg_use_manifold_hvd",
                False,
            )),
            "residual_square_tail": tail_radius,
            "residual_variance_cap": {
                str(i): self._residual_variance_cap(i)
                for i in range(self.n_outputs)
            },
            "uses_high_frequency_floor": bool(
                getattr(
                    getattr(self._last_problem, "_scolhkg_representation_encoder", None),
                    "residual_floor",
                    None,
                )
            ),
        }
