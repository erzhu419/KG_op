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


@dataclass
class HVDConfig:
    mode: str = "class"
    n_outputs: int = 2
    ridge_alpha: float = 1e-3
    floor: float = 1e-8
    shrinkage_kappa: float = 2.0
    n_factors: int = 3


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
        self.factor_energy = {i: [] for i in range(self.n_outputs)}
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

    def _normalize(self, x, problem=None):
        problem = problem or self._last_problem
        if problem is not None and hasattr(problem, "normalize"):
            return np.asarray(problem.normalize(x), dtype=float)
        x = np.asarray(x, dtype=float)
        scale = max(float(np.max(np.abs(x))) if len(x) else 1.0, 1.0)
        return np.clip(x / scale, 0.0, 1.0)

    def _features(self, x, problem=None):
        """Near-orthogonal polynomial/log-variance features on [0, 1]."""
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

    def fit_from_residuals(self, X, residuals, output_index=0, problem=None):
        """Direct fit helper used by tests and by `initialize`."""
        self._last_problem = problem or self._last_problem
        i = int(output_index)
        for x, r in zip(X, residuals):
            self.records[i].append((tuple(int(v) for v in x), float(r) ** 2))
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
                    resid2 = (float(y_vec[i]) - mu) ** 2
                    self.records[i].append((tuple(x_tuple), resid2))
        for i in range(self.n_outputs):
            self._fit_output(i, problem)

    def update(self, i, x, y, mu, gpr_model=None, problem=None):
        """Add one residual and refit lightweight summaries."""
        del gpr_model
        self._last_problem = problem or self._last_problem
        i = int(i)
        x_tuple = tuple(int(v) for v in np.asarray(x, dtype=int))
        resid2 = (float(y) - float(mu)) ** 2
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
        vals = np.array([max(v, self.floor) for _, v in recs], dtype=float)
        self.global_var[i] = float(max(np.mean(vals), self.floor))

        by_class = defaultdict(list)
        for x, v in recs:
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

        if self.mode in ("orthogonal", "factor"):
            X = np.vstack([self._features(x, problem) for x, _ in recs])
            y = np.log(vals)
            reg = float(self.config.ridge_alpha) * np.eye(X.shape[1])
            reg[0, 0] = 0.0
            try:
                beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X.T @ X + reg, X.T @ y, rcond=None)[0]
            self.beta[i] = beta
            if self.mode == "factor":
                coefs = np.asarray(beta[1:], dtype=float)
                energy = coefs ** 2
                total = float(np.sum(energy))
                if total > 0:
                    self.factor_energy[i] = (energy / total).tolist()
                else:
                    self.factor_energy[i] = [0.0 for _ in energy]

    def _class_variance(self, i, x, problem=None):
        c = self.risk_class(x, problem)
        return float(self.class_var.get(i, {}).get(c, self.global_var.get(i, 0.01)))

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
        beta = self.beta.get(i)
        if beta is None:
            return float(max(self._class_variance(i, x, problem), self.floor))
        feat = self._features(x, problem)
        pred = float(np.exp(float(feat @ beta)))
        # Sparse-data guard: keep the smooth model near regime/global scale.
        class_v = self._class_variance(i, x, problem)
        pred = float(np.clip(pred, 0.25 * class_v, 4.0 * class_v))
        return float(max(pred, self.floor))

    def predict_decomposition(self, i, x, problem=None):
        """Return interpretable decomposition diagnostics."""
        i = int(i)
        c = self.risk_class(x, problem)
        feat = self._features(x, problem)
        beta = self.beta.get(i)
        factor_contrib = None
        if beta is not None:
            factor_contrib = (feat[1:] * beta[1:]).tolist()
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

    def diagnostics(self):
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
        }
