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


class OrthogonalHVD:
    """Variance decomposition model for simulation residuals."""

    VALID_MODES = {"pooled", "oracle", "class", "orthogonal", "factor"}

    def __init__(self, mode="class", n_outputs=2, **kwargs):
        self.config = HVDConfig(mode=mode, n_outputs=n_outputs, **kwargs)
        if self.config.mode not in self.VALID_MODES:
            raise ValueError(f"unknown HVD mode {mode!r}")
        self.mode = self.config.mode
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
        if "_last_problem" not in self.__dict__:
            self._last_problem = None

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
        return len(self.records.get(int(i), [])) >= int(self.config.activation_min_records)

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

    def _residual_square_tail_radius(self, i):
        tail_delta = float(self.config.residual_tail_delta)
        nu, b = gaussian_square_subexp_params(self.global_var.get(int(i), 0.01))
        return float(sub_exponential_residual_square_radius(nu, b, tail_delta))

    def _residual_tail_uncertainty(self, i):
        n = max(int(len(self.records.get(int(i), []))), 0)
        return float(self._residual_square_tail_radius(i) / np.sqrt(n + 1.0))

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
            self.records[i].append((
                tuple(int(v) for v in x),
                self._safe_residual_square(r, i, problem),
            ))
        self._fit_output(i, problem)

    def initialize(self, samples, observations, gpr_models, problem=None):
        """Initialize from pre-sample residuals.

        `observations` maps `x_tuple -> list[np.ndarray]`, and each observation
        contains all output channels.
        """
        self._last_problem = problem or self._last_problem
        for x_tuple in samples:
            obs_list = observations.get(tuple(x_tuple), [])
            if not obs_list:
                continue
            x_arr = np.asarray(x_tuple, dtype=int)
            for y_vec in obs_list:
                for i in range(self.n_outputs):
                    mu = float(gpr_models[i].posterior_mean(x_arr))
                    resid2 = self._safe_residual_square(float(y_vec[i]) - mu, i, problem)
                    self.records[i].append((tuple(x_tuple), resid2))
        for i in range(self.n_outputs):
            self._fit_output(i, problem)

    def update(self, i, x, y, mu, gpr_model=None, problem=None):
        """Add one residual and refit lightweight summaries."""
        del gpr_model
        self._last_problem = problem or self._last_problem
        i = int(i)
        x_tuple = tuple(int(v) for v in np.asarray(x, dtype=int))
        resid2 = self._safe_residual_square(float(y) - float(mu), i, problem)
        self.records[i].append((x_tuple, resid2))
        old = self.predict_variance(i, x_tuple, problem)
        self._fit_output(i, problem)
        new = self.predict_variance(i, x_tuple, problem)
        return {
            "mode": self.mode,
            "output_index": i,
            "x": list(x_tuple),
            "resid2": float(resid2),
            "old_variance": float(old),
            "new_variance": float(new),
            "risk_class": int(self.risk_class(x_tuple, problem)),
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
        if self.mode == "factor" and len(recs) >= int(self.config.activation_min_records):
            X_c = []
            y_c = []
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
            if X_c:
                X_c = np.vstack(X_c)
                y_c = np.asarray(y_c, dtype=float)
                ridge_alpha = float(self.config.ridge_alpha)
                reg = ridge_alpha * np.eye(X_c.shape[1])
                prior_beta = None
                problem_ref = problem or self._last_problem
                if (
                    problem_ref is not None
                    and hasattr(problem_ref, "cumulative_hvd_prior_beta")
                ):
                    try:
                        prior_beta = problem_ref.cumulative_hvd_prior_beta(
                            output_index=i,
                            feature_dim=X_c.shape[1],
                        )
                    except TypeError:
                        prior_beta = problem_ref.cumulative_hvd_prior_beta(i)
                    if prior_beta is not None:
                        prior_beta = np.asarray(prior_beta, dtype=float)
                        if prior_beta.shape != (X_c.shape[1],):
                            prior_beta = None
                if prior_beta is None:
                    reg[0, 0] = 0.0
                    rhs = X_c.T @ y_c
                else:
                    rhs = X_c.T @ y_c + reg @ prior_beta
                try:
                    beta_c = np.linalg.solve(X_c.T @ X_c + reg, rhs)
                except np.linalg.LinAlgError:
                    beta_c = np.linalg.lstsq(
                        X_c.T @ X_c + reg,
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
                        projected, params = project_cumulative_beta(
                            beta_c,
                            exposure_layout_ref,
                        )
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
                self.cumulative_beta[i] = beta_c
                self.cumulative_params[i] = params
                self.cumulative_provider_active[i] = bool(provider_active)
                pred_c = np.maximum(X_c @ beta_c, self.floor)
                self.cumulative_fit_rmse[i] = float(np.sqrt(np.mean((pred_c - y_c) ** 2)))
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
        activation_penalty = 1.0 if (
            self.mode in ("orthogonal", "factor") and not self._orthogonal_active(i)
        ) else 0.25
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
        activation_penalty = 1.0 if (
            self.mode in ("orthogonal", "factor") and not self._orthogonal_active(i)
        ) else 0.25
        scale = (
            max(float(self.config.certification_kappa), 0.0)
            * np.maximum(class_unc, total_unc)
            * activation_penalty
        )
        return np.maximum(base * scale, 0.0)

    def predict_certification_variance(self, i, x, problem=None):
        """Variance used inside conservative chance feasibility checks."""
        cert = (
            self.predict_variance(i, x, problem)
            + self.model_uncertainty(i, x, problem)
        )
        if self.mode == "factor" and self._cumulative_active(i):
            cert += self._residual_tail_uncertainty(i)
        if self.mode in ("orthogonal", "factor"):
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
        if self.mode in ("orthogonal", "factor"):
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

    def diagnostics(self):
        tail_delta = float(self.config.residual_tail_delta)
        tail_radius = {}
        for i in range(self.n_outputs):
            nu, b = gaussian_square_subexp_params(self.global_var.get(i, 0.01))
            tail_radius[str(i)] = {
                "delta": tail_delta,
                "nu": float(nu),
                "b": float(b),
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
            "orthogonal_active": {
                str(i): bool(self._orthogonal_active(i))
                for i in range(self.n_outputs)
            },
            "activation_min_records": int(self.config.activation_min_records),
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
        }
