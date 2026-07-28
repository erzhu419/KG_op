"""Lightweight SOTA-inspired baselines for chance-constrained benchmarks.

These are deliberately dependency-light.  They are not exact BoTorch TuRBO or
SCBO implementations; they provide reproducible local trust-region and Sobol
baselines that can run in this repo without GPyTorch/BoTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.stats import norm, qmc

from core.candidates import boundary_solutions, unique_candidates


@dataclass
class BaselineConfig:
    N: int = 30
    n0: int = 8
    seed: int = 123
    method: str = "sobol"
    batch_candidates: int = 64
    tr_radius_init: float = 0.35
    tr_radius_min: float = 0.04
    tr_radius_max: float = 0.8
    tr_success_tolerance: int = 3
    tr_failure_tolerance: int = 5
    nominal_sigma_scale: float = 1.0
    ridge: float = 1e-6
    risk_aversion: float = 0.5
    safe_beta: float = 2.0
    embedding_dim: int = 8
    embedding_dim_max: int = 32
    use_problem_initial_samples: bool = True
    use_boundary_initial_samples: bool = True
    progress_logging: bool = False
    progress_label: str = ""


class SequentialBaseline:
    """Sequential derivative-free baseline with shared evaluation semantics."""

    VALID_METHODS = {
        "random",
        "sobol",
        "turbo_lite",
        "scbo_lite",
        "hetgp_lite",
        "rahbo_lite",
        "safeopt_lite",
        "legacy_vepm_lite",
        "rembo_lite",
        "baxus_lite",
    }

    def __init__(self, problem, config: BaselineConfig):
        if config.method not in self.VALID_METHODS:
            raise ValueError(f"unknown baseline method {config.method!r}")
        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.history: list[tuple[tuple[int, ...], np.ndarray]] = []
        self._sobol_index = 0
        self._sobol = qmc.Sobol(
            d=int(problem.d),
            scramble=True,
            seed=int(self.rng.integers(1, 2**31 - 1)),
        )
        self._tr_radius = float(config.tr_radius_init)
        self._last_score = None
        self._embedding_dim = max(1, min(int(config.embedding_dim), int(problem.d)))
        self._embedding_dim_max = max(
            self._embedding_dim,
            min(int(config.embedding_dim_max), int(problem.d)),
        )
        self._embedding_matrix = self._make_embedding(self._embedding_dim)
        self._embedding_failures = 0

    def _make_embedding(self, k):
        mat = self.rng.normal(size=(int(self.problem.d), int(k)))
        scale = np.linalg.norm(mat, axis=0, keepdims=True)
        mat = mat / np.maximum(scale, 1e-12)
        return mat

    def _nominal_margin(self, y):
        sigma = float(getattr(self.problem, "sigma_level", 0.04))
        z = norm.ppf(1 - self.problem.alpha)
        return float(y[1] + z * self.config.nominal_sigma_scale * sigma - self.problem.tau)

    def _score_observation(self, y):
        margin = self._nominal_margin(y)
        if margin <= 0.0:
            return (0, float(y[0]), margin)
        return (1, margin, float(y[0]))

    def _initial_samples(self):
        rows = []
        if self.config.use_problem_initial_samples and hasattr(self.problem, "initial_samples"):
            rows.extend(self.problem.initial_samples(n=self.config.n0, rng=self.rng))
            rows = unique_candidates(rows)
        if self.config.use_boundary_initial_samples and len(rows) < self.config.n0:
            for x in boundary_solutions(self.problem):
                if len(rows) >= self.config.n0:
                    break
                rows.append(tuple(x))
                rows = unique_candidates(rows)
        while len(rows) < self.config.n0:
            rows.append(self.problem.sample_random(self.rng))
            rows = unique_candidates(rows)
        return rows[: self.config.n0]

    def _simulate(self, x):
        x_tuple = tuple(int(v) for v in x)
        y = self.problem.simulate(x_tuple, self.rng)
        self.history.append((x_tuple, y))
        return y

    def _observed_best(self, feasible_first=True):
        if not self.history:
            return self.problem.sample_random(self.rng)
        if feasible_first:
            key = lambda item: self._score_observation(item[1])
        else:
            key = lambda item: (float(item[1][0]), self._nominal_margin(item[1]))
        return min(self.history, key=key)[0]

    def _sobol_candidate(self):
        row = self._sobol.random(1)[0]
        self._sobol_index += 1
        return self.problem.continuous_to_int(row)

    def _trust_region_candidate(self, feasible_first=True):
        center = np.asarray(
            self.problem.normalize(self._observed_best(feasible_first=feasible_first)),
            dtype=float,
        )
        rows = []
        for _ in range(max(1, int(self.config.batch_candidates))):
            z = center + self.rng.uniform(
                -self._tr_radius,
                self._tr_radius,
                size=int(self.problem.d),
            )
            rows.append(self.problem.continuous_to_int(np.clip(z, 0.0, 1.0)))
        rows = unique_candidates(rows)
        if feasible_first:
            return rows[int(self.rng.integers(0, len(rows)))]
        best_seen = set(x for x, _ in self.history)
        for row in rows:
            if row not in best_seen:
                return row
        return rows[int(self.rng.integers(0, len(rows)))]

    def _next_candidate(self):
        method = self.config.method
        if method == "random":
            return self.problem.sample_random(self.rng)
        if method == "sobol":
            return self._sobol_candidate()
        if method == "turbo_lite":
            return self._trust_region_candidate(feasible_first=False)
        if method == "scbo_lite":
            return self._trust_region_candidate(feasible_first=True)
        if method in ("rembo_lite", "baxus_lite"):
            return self._embedding_candidate(adaptive=(method == "baxus_lite"))
        if method in ("hetgp_lite", "rahbo_lite", "safeopt_lite", "legacy_vepm_lite"):
            return self._surrogate_candidate(method)
        raise AssertionError(method)

    def _update_tr_radius(self, y):
        if self.config.method not in (
            "turbo_lite",
            "scbo_lite",
            "hetgp_lite",
            "rahbo_lite",
            "safeopt_lite",
            "legacy_vepm_lite",
            "rembo_lite",
            "baxus_lite",
        ):
            return
        score = self._score_observation(y)
        improved = self._last_score is None or score < self._last_score
        if improved:
            self._tr_radius = min(
                float(self.config.tr_radius_max),
                1.25 * self._tr_radius,
            )
            self._last_score = score
        else:
            self._tr_radius = max(
                float(self.config.tr_radius_min),
                0.85 * self._tr_radius,
            )
            if self.config.method == "baxus_lite":
                self._embedding_failures += 1
                if (
                    self._embedding_failures >= max(2, self.config.tr_failure_tolerance)
                    and self._embedding_dim < self._embedding_dim_max
                ):
                    self._expand_embedding()
                    self._embedding_failures = 0

    def _expand_embedding(self):
        old = self._embedding_matrix
        new_k = min(self._embedding_dim_max, max(self._embedding_dim + 1, 2 * self._embedding_dim))
        if new_k <= self._embedding_dim:
            return
        extra = self._make_embedding(new_k - self._embedding_dim)
        self._embedding_matrix = np.hstack([old, extra])
        self._embedding_dim = new_k
        self._tr_radius = min(float(self.config.tr_radius_max), 1.25 * self._tr_radius)

    def _basis(self, x):
        z = np.asarray(self.problem.normalize(x), dtype=float)
        return np.concatenate([[1.0], z, z ** 2])

    def _fit_ridge(self, y):
        X = np.vstack([self._basis(x) for x, _ in self.history])
        y = np.asarray(y, dtype=float)
        ridge = max(float(self.config.ridge), 0.0)
        reg = ridge * np.eye(X.shape[1], dtype=float)
        reg[0, 0] = 0.0
        try:
            beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X.T @ X + reg, X.T @ y, rcond=None)[0]
        resid = y - X @ beta
        return beta, resid

    def _surrogate_pool(self):
        rows = []
        if self.config.use_boundary_initial_samples:
            rows.extend(boundary_solutions(self.problem))
        if self.config.use_problem_initial_samples and hasattr(self.problem, "structured_candidates"):
            rows.extend(self.problem.structured_candidates(
                n=max(8, int(self.config.batch_candidates) // 2),
                rng=self.rng,
            ))
        rows.extend(self._trust_region_pool(feasible_first=True))
        for _ in range(max(8, int(self.config.batch_candidates))):
            rows.append(self.problem.sample_random(self.rng))
        return unique_candidates(rows)

    def _trust_region_pool(self, feasible_first=True):
        center = np.asarray(
            self.problem.normalize(self._observed_best(feasible_first=feasible_first)),
            dtype=float,
        )
        rows = []
        for _ in range(max(8, int(self.config.batch_candidates))):
            z = center + self.rng.uniform(
                -self._tr_radius,
                self._tr_radius,
                size=int(self.problem.d),
            )
            rows.append(self.problem.continuous_to_int(np.clip(z, 0.0, 1.0)))
        return unique_candidates(rows)

    def _latent_from_x(self, x):
        z = np.asarray(self.problem.normalize(x), dtype=float) - 0.5
        try:
            latent, *_ = np.linalg.lstsq(self._embedding_matrix, z, rcond=None)
        except np.linalg.LinAlgError:
            latent = self.rng.uniform(-1.0, 1.0, size=self._embedding_dim)
        return np.clip(latent, -1.0, 1.0)

    def _embedding_to_x(self, latent):
        z = 0.5 + self._embedding_matrix @ np.asarray(latent, dtype=float)
        return self.problem.continuous_to_int(np.clip(z, 0.0, 1.0))

    def _embedding_candidate(self, adaptive=False):
        seen = {x for x, _ in self.history}
        center = self._latent_from_x(self._observed_best(feasible_first=True))
        rows = []
        for _ in range(max(8, int(self.config.batch_candidates))):
            if adaptive:
                noise = self.rng.normal(scale=self._tr_radius, size=self._embedding_dim)
            else:
                noise = self.rng.uniform(
                    -self._tr_radius,
                    self._tr_radius,
                    size=self._embedding_dim,
                )
            latent = np.clip(center + noise, -1.0, 1.0)
            rows.append(self._embedding_to_x(latent))
        rows = [row for row in unique_candidates(rows) if row not in seen]
        if rows:
            return rows[int(self.rng.integers(0, len(rows)))]
        return self.problem.sample_random(self.rng)

    def _class_noise_estimates(self, residuals):
        global_var = max(float(np.var(residuals)), 1e-8)
        grouped = {}
        for (x, _), resid in zip(self.history, residuals):
            cls = self.problem.risk_class(x) if hasattr(self.problem, "risk_class") else 0
            grouped.setdefault(cls, []).append(float(resid) ** 2)
        return {
            cls: max(float(np.mean(vals)), 1e-8)
            for cls, vals in grouped.items()
        }, global_var

    def _nearest_uncertainty(self, pool):
        if not self.history:
            return np.ones(len(pool), dtype=float)
        X_obs = np.vstack([
            np.asarray(self.problem.normalize(x), dtype=float)
            for x, _ in self.history
        ])
        X_pool = np.vstack([
            np.asarray(self.problem.normalize(x), dtype=float)
            for x in pool
        ])
        dist = np.sqrt(np.sum((X_pool[:, None, :] - X_obs[None, :, :]) ** 2, axis=2))
        nearest = np.min(dist, axis=1)
        hi = max(float(np.max(nearest)), 1e-12)
        return nearest / hi

    def _surrogate_candidate(self, method):
        if len(self.history) < max(3, self.config.n0):
            return self.problem.sample_random(self.rng)
        y_obj = [float(y[0]) for _, y in self.history]
        y_con = [float(y[1]) for _, y in self.history]
        beta_obj, resid_obj = self._fit_ridge(y_obj)
        beta_con, resid_con = self._fit_ridge(y_con)
        var_by_class, global_var = self._class_noise_estimates(resid_con)
        obj_var_by_class, obj_global_var = self._class_noise_estimates(resid_obj)
        pool = self._surrogate_pool()
        Phi = np.vstack([self._basis(x) for x in pool])
        mu_obj = Phi @ beta_obj
        mu_con = Phi @ beta_con
        uncert = self._nearest_uncertainty(pool)
        classes = [
            self.problem.risk_class(x) if hasattr(self.problem, "risk_class") else 0
            for x in pool
        ]
        v_con = np.asarray([
            var_by_class.get(cls, global_var) for cls in classes
        ], dtype=float)
        v_obj = np.asarray([
            obj_var_by_class.get(cls, obj_global_var) for cls in classes
        ], dtype=float)
        sigma_floor = float(getattr(self.problem, "sigma_level", 0.04))
        if method == "legacy_vepm_lite":
            v_con = np.maximum(v_con, sigma_floor ** 2)
        if method == "hetgp_lite":
            margin = (
                mu_con
                + norm.ppf(1 - self.problem.alpha) * np.sqrt(np.maximum(v_con, 1e-12))
                - self.problem.tau
            )
            score = mu_obj + 5.0 * np.maximum(margin, 0.0) - 0.05 * uncert
        elif method == "rahbo_lite":
            margin = (
                mu_con
                + norm.ppf(1 - self.problem.alpha) * np.sqrt(np.maximum(v_con, 1e-12))
                - self.problem.tau
            )
            score = (
                mu_obj
                + self.config.risk_aversion * np.sqrt(np.maximum(v_obj, 1e-12))
                + 5.0 * np.maximum(margin, 0.0)
            )
        elif method == "safeopt_lite":
            margin = (
                mu_con
                + np.sqrt(max(float(self.config.safe_beta), 0.0)) * uncert
                + norm.ppf(1 - self.problem.alpha) * np.sqrt(np.maximum(v_con, 1e-12))
                - self.problem.tau
            )
            safe = margin <= 0.0
            if np.any(safe):
                score = np.where(safe, mu_obj - 0.05 * uncert, np.inf)
            else:
                score = margin
        else:
            margin = (
                mu_con
                + norm.ppf(1 - self.problem.alpha) * np.sqrt(np.maximum(v_con, 1e-12))
                - self.problem.tau
            )
            score = mu_obj + 6.0 * np.maximum(margin, 0.0) - 0.08 * uncert
        seen = {x for x, _ in self.history}
        order = np.argsort(np.nan_to_num(score, nan=np.inf, posinf=np.inf))
        for idx in order:
            row = tuple(pool[int(idx)])
            if row not in seen:
                return row
        return tuple(pool[int(order[0])])

    def _recommendation(self):
        x_best = self._observed_best(feasible_first=True)
        y_best = None
        for x, y in self.history:
            if tuple(x) == tuple(x_best):
                y_best = y
                break
        return x_best, y_best

    def _evaluate_recommendation(self, x_best, y_best):
        del y_best
        true_obj = self.problem.true_objective(x_best)
        true_con = self.problem.true_constraint_mean(x_best)
        true_sig = self.problem.true_sigma(x_best)
        true_vector = None
        if hasattr(self.problem, "true_vector_objectives"):
            true_vector = [
                float(v)
                for v in self.problem.true_vector_objectives(x_best)
            ]
        true_margin = (
            true_con
            + norm.ppf(1 - self.problem.alpha) * true_sig[1]
            - self.problem.tau
        )
        true_best_x, true_best_obj = self.problem.true_best_feasible()
        true_best_vector = None
        if true_best_x is not None and hasattr(self.problem, "true_vector_objectives"):
            true_best_vector = [
                float(v)
                for v in self.problem.true_vector_objectives(true_best_x)
            ]
        regret = true_obj - true_best_obj if np.isfinite(true_best_obj) else np.nan
        out = {
            "x_recommended": list(map(int, x_best)),
            "true_objective": float(true_obj),
            "true_constraint_mean": float(true_con),
            "true_constraint_sigma": float(true_sig[1]),
            "true_chance_margin": float(true_margin),
            "true_feasible": bool(true_margin <= 0.0),
            "true_best_x": None if true_best_x is None else list(map(int, true_best_x)),
            "true_best_objective": float(true_best_obj),
            "simple_regret": float(regret),
        }
        if true_vector is not None:
            out["true_vector_objectives"] = true_vector
            if len(true_vector) >= 2:
                out["true_f1"] = float(true_vector[0])
                out["true_f2"] = float(true_vector[1])
        if true_best_vector is not None:
            out["true_best_vector_objectives"] = true_best_vector
            if len(true_best_vector) >= 2:
                out["true_best_f1"] = float(true_best_vector[0])
                out["true_best_f2"] = float(true_best_vector[1])
        return out

    def _progress_enabled(self):
        return bool(getattr(self.config, "progress_logging", False))

    def _progress_label(self):
        label = str(getattr(self.config, "progress_label", "") or "").strip()
        return label or f"{self.config.method}:seed={int(self.config.seed)}"

    def _emit_progress(self, started_at):
        if not self._progress_enabled():
            return
        total = max(1, int(self.config.N))
        current = max(0, min(total, len(self.history)))
        elapsed = max(0.0, time.time() - float(started_at))
        done = max(1, current)
        eta_sec = (elapsed / float(done)) * float(max(0, total - current))
        print(
            f"Iter {current}/{total} [kg-inner] "
            f"kind=baseline label={self._progress_label()} "
            f"Time: {elapsed:.1f}s ETA {eta_sec:.1f}s",
            flush=True,
        )

    def run(self):
        t_start = time.time()
        for x in self._initial_samples():
            y = self._simulate(x)
            self._update_tr_radius(y)
            self._emit_progress(t_start)
        while len(self.history) < int(self.config.N):
            x = self._next_candidate()
            y = self._simulate(x)
            self._update_tr_radius(y)
            self._emit_progress(t_start)
        x_best, y_best = self._recommendation()
        result = self._evaluate_recommendation(x_best, y_best)
        result.update({
            "method": self.config.method,
            "posterior_feasible": bool(self._nominal_margin(y_best) <= 0.0),
            "posterior_chance_margin": float(self._nominal_margin(y_best)),
            "total_time_sec": float(time.time() - t_start),
            "n_simulations": int(len(self.history)),
            "n_distinct_solutions": int(len(set(x for x, _ in self.history))),
            "tr_radius_final": float(self._tr_radius),
            "embedding_dim_final": int(self._embedding_dim),
        })
        return result
