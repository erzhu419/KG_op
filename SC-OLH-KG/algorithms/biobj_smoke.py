"""Bi-objective OLH-KG smoke test.

This intentionally keeps the current paper's bi-objective shape but replaces
VEPM with `OrthogonalHVD`.  It is a smoke test for variance decomposition, not
the main paper algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import time

import numpy as np

from core.candidates import boundary_solutions, latin_hypercube_candidates, unique_candidates
from core.gpr import ParametricGPR
from core.kg import compute_kg_vectorized
from core.metrics import (
    compute_hypervolume_2d,
    crowding_distance_select,
    pareto_filter,
    summarize_stage_times,
)
from variance.orthogonal_hvd import OrthogonalHVD


@dataclass
class BiObjSmokeConfig:
    N: int = 20
    n0: int = 6
    K1: int = 20
    lambda_i: float = 0.1
    prior_var: float = 10.0
    variance_mode: str = "class"
    seed: int = 123


class BiObjectiveOLHKGSmoke:
    def __init__(self, problem, config: BiObjSmokeConfig | None = None):
        self.problem = problem
        self.config = config or BiObjSmokeConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.gpr = [
            ParametricGPR(
                problem.d,
                self.config.lambda_i,
                self.config.prior_var,
                normalize_func=problem.normalize,
            )
            for _ in range(3)
        ]
        self.variance_model = OrthogonalHVD(
            mode=self.config.variance_mode,
            n_outputs=3,
            floor=1e-8,
        )
        self.observations = {}
        self.history = []
        self.iteration_log = []
        self.final_log = None

    def _simulate_and_store(self, x):
        y = self.problem.simulate(x, self.rng)
        x_tuple = tuple(int(v) for v in x)
        self.observations.setdefault(x_tuple, []).append(y)
        self.history.append((x_tuple, y))
        return y

    def _initial_samples(self):
        samples = []
        for x in boundary_solutions(self.problem):
            if len(samples) >= self.config.n0:
                break
            samples.append(tuple(x))
        while len(set(samples)) < self.config.n0:
            samples.append(self.problem.sample_random(self.rng))
            samples = unique_candidates(samples)
        return samples[: self.config.n0]

    def _fit_initial_belief(self, samples):
        for x in samples:
            self._simulate_and_store(x)
        Phi = self.gpr[0].basis_matrix(samples)
        for i in range(3):
            y_i = np.array([self.observations[x][0][i] for x in samples], dtype=float)
            beta = np.linalg.lstsq(Phi, y_i, rcond=None)[0]
            resid = y_i - Phi @ beta
            self.gpr[i].set_parametric_prior(
                beta,
                max(float(np.var(resid)), 1e-6),
                max(float(np.var(beta)), 1e-6),
            )
        for x in samples:
            for model in self.gpr:
                model.dimension_augment(x)
        self.variance_model.initialize(samples, self.observations, self.gpr, self.problem)

    def _posterior_pareto_set(self):
        pool = set(x for x, _ in self.history)
        for _ in range(300):
            pool.add(self.problem.sample_random(self.rng))
        pool = list(pool)
        objs = self.gpr[0].posterior_mean_many(pool)
        objs2 = self.gpr[1].posterior_mean_many(pool)
        vals = np.column_stack([objs, objs2])
        _, idx = pareto_filter(vals, return_indices=True)
        return [pool[i] for i in idx]

    def run(self, verbose=False):
        t_start = time.time()
        samples = self._initial_samples()
        self._fit_initial_belief(samples)
        for n in range(self.config.n0, self.config.N):
            iteration = n - self.config.n0
            row = {"iteration": iteration, "stage": n}
            t_iter = time.time()

            t0 = time.time()
            pareto_set = self._posterior_pareto_set()
            row["t_posterior_solve"] = time.time() - t0
            row["posterior_pareto_size"] = len(pareto_set)

            t0 = time.time()
            candidates = unique_candidates(latin_hypercube_candidates(
                self.problem, self.config.K1, self.rng) + pareto_set)
            row["t_candidate_gen"] = time.time() - t0
            row["n_candidates"] = len(candidates)

            t0 = time.time()
            kg_cols = []
            for i in range(2):
                sig2 = np.array([
                    self.variance_model.predict_variance(i, x, self.problem)
                    for x in candidates
                ])
                kg_cols.append(compute_kg_vectorized(self.gpr[i], candidates, sig2))
            kg_pairs = np.column_stack(kg_cols)
            _, nd_idx = pareto_filter(-kg_pairs, return_indices=True)
            if len(nd_idx) == 0:
                selected_idx = int(np.argmax(np.sum(kg_pairs, axis=1)))
            else:
                selected_idx = int(nd_idx[crowding_distance_select(kg_pairs[nd_idx])])
            x_selected = candidates[selected_idx]
            row["t_kg_compute"] = time.time() - t0
            row["x_selected"] = list(map(int, x_selected))
            row["kg_pair_selected"] = kg_pairs[selected_idx].tolist()

            x_arr = np.asarray(x_selected, dtype=int)
            mu_before = [self.gpr[i].posterior_mean(x_arr) for i in range(3)]
            sig2_before = [
                self.variance_model.predict_variance(i, x_arr, self.problem)
                for i in range(3)
            ]
            t0 = time.time()
            y = self._simulate_and_store(x_selected)
            row["t_simulate"] = time.time() - t0
            row["Y_observed"] = y.tolist()

            t0 = time.time()
            for i in range(3):
                self.gpr[i].update(x_arr, y[i], sig2_before[i])
            for i in range(3):
                self.variance_model.update(i, x_arr, y[i], mu_before[i], self.gpr[i], self.problem)
            row["t_update"] = time.time() - t0

            t0 = time.time()
            if iteration % 5 == 0 or n == self.config.N - 1:
                pf_solutions = self._posterior_pareto_set()
                true_objs = np.array([
                    self.problem.true_objectives(x)[:2]
                    for x in pf_solutions
                ], dtype=float)
                row["hv"] = compute_hypervolume_2d(true_objs, self.problem.ref_point)
                row["pareto_size"] = len(true_objs)
            row["t_eval"] = time.time() - t0
            row["t_total"] = time.time() - t_iter
            self.iteration_log.append(row)
            if verbose:
                print(f"iter={iteration:03d} x={x_selected}")

        pf_solutions = self._posterior_pareto_set()
        true_objs = np.array([
            self.problem.true_objectives(x)[:2]
            for x in pf_solutions
        ], dtype=float)
        self.final_log = {
            "hv_final": float(compute_hypervolume_2d(true_objs, self.problem.ref_point)),
            "pareto_size": int(len(true_objs)),
            "pareto_solutions": [list(map(int, x)) for x in pf_solutions],
            "total_time_sec": float(time.time() - t_start),
            "n_simulations": int(len(self.history)),
            "stage_times": summarize_stage_times(self.iteration_log),
            "variance": self.variance_model.diagnostics(),
            "config": asdict(self.config),
        }
        return self.final_log
