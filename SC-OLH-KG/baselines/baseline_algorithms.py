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
    nominal_sigma_scale: float = 1.0


class SequentialBaseline:
    """Sequential derivative-free baseline with shared evaluation semantics."""

    VALID_METHODS = {"random", "sobol", "turbo_lite", "scbo_lite"}

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
        if hasattr(self.problem, "initial_samples"):
            rows.extend(self.problem.initial_samples(n=self.config.n0, rng=self.rng))
            rows = unique_candidates(rows)
        if len(rows) < self.config.n0:
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
        raise AssertionError(method)

    def _update_tr_radius(self, y):
        if self.config.method not in ("turbo_lite", "scbo_lite"):
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

    def run(self):
        t_start = time.time()
        for x in self._initial_samples():
            y = self._simulate(x)
            self._update_tr_radius(y)
        while len(self.history) < int(self.config.N):
            x = self._next_candidate()
            y = self._simulate(x)
            self._update_tr_radius(y)
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
        })
        return result
