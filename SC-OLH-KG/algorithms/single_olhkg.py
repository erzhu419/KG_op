"""Single-objective chance-constrained OLH-KG / SC-OLH-KG algorithm."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import time

import numpy as np
from scipy.stats import norm

from acquisition.olhkg import OLHKGAcquisition
from core.candidates import (
    axis_candidates,
    boundary_solutions,
    latin_hypercube_candidates,
    posterior_sample_candidates,
    random_candidates,
    structured_candidates,
    unique_candidates,
)
from core.gpr import ParametricGPR
from core.metrics import summarize_stage_times
from encoders.policy_state_encoder import (
    StateCoupledFeatureMap,
    SyntheticPolicyStateEncoder,
)
from variance.orthogonal_hvd import OrthogonalHVD


@dataclass
class SingleOLHKGConfig:
    N: int = 30
    n0: int = 8
    K1: int = 25
    K2: int = 0
    posterior_pool_size: int = 300
    posterior_keep: int = 15
    axis_candidate_count: int = -1
    structured_candidate_count: int = 0
    state_candidate_count: int = -1
    state_inverse_pool_size: int = 500
    state_inverse_neighbors: int = 2
    n_thr: int = 5
    lambda_i: float = 0.1
    prior_var: float = 10.0
    variance_mode: str = "class"
    lambda_feas: float = 0.25
    lambda_var: float = 0.25
    lambda_coupling: float = 0.0
    coupling_safety_z: float = 0.5
    coupling_gate_temperature: float = 0.25
    recommendation_safety_z: float = 0.5
    recommendation_noise_floor_scale: float = 1.0
    use_state_coupling: bool = False
    use_state_basis: bool = False
    eval_pool_size: int = 500
    seed: int = 123


class SingleOLHKGAlgorithm:
    """Minimal but complete single-objective OLH-KG implementation."""

    def __init__(self, problem, config: SingleOLHKGConfig | None = None):
        self.problem = problem
        self.config = config or SingleOLHKGConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.encoder = (
            SyntheticPolicyStateEncoder(problem)
            if (
                self.config.use_state_coupling
                or self.config.use_state_basis
                or self.config.lambda_coupling > 0
            )
            else None
        )
        basis_map = (
            StateCoupledFeatureMap(problem, self.encoder)
            if self.config.use_state_basis else None
        )
        self.gpr = [
            ParametricGPR(
                problem.d,
                self.config.lambda_i,
                self.config.prior_var,
                normalize_func=problem.normalize,
                basis_map=basis_map,
            )
            for _ in range(2)
        ]
        self.variance_model = OrthogonalHVD(
            mode=self.config.variance_mode,
            n_outputs=2,
            floor=1e-8,
        )
        self.acquisition = OLHKGAcquisition(
            lambda_feas=self.config.lambda_feas,
            lambda_var=self.config.lambda_var,
            lambda_coupling=self.config.lambda_coupling,
            coupling_safety_z=self.config.coupling_safety_z,
            coupling_gate_temperature=self.config.coupling_gate_temperature,
            encoder=self.encoder,
        )

        self.observations: dict[tuple[int, ...], list[np.ndarray]] = {}
        self.history: list[tuple[tuple[int, ...], np.ndarray]] = []
        self.iteration_log: list[dict] = []
        self.pre_sampling_log: dict | None = None
        self.final_log: dict | None = None

    def _initial_samples(self):
        samples = []
        if hasattr(self.problem, "initial_samples"):
            samples.extend(self.problem.initial_samples(
                n=self.config.n0,
                rng=self.rng,
            ))
            samples = unique_candidates(samples)
        for x in boundary_solutions(self.problem):
            if len(samples) >= self.config.n0:
                break
            samples.append(tuple(x))
        while len(set(samples)) < self.config.n0:
            samples.append(self.problem.sample_random(self.rng))
            samples = unique_candidates(samples)
        return unique_candidates(samples)[: self.config.n0]

    def _simulate_and_store(self, x):
        y = self.problem.simulate(x, self.rng)
        x_tuple = tuple(int(v) for v in x)
        self.observations.setdefault(x_tuple, []).append(y)
        self.history.append((x_tuple, y))
        return y

    def _fit_initial_belief(self, samples):
        for x in samples:
            self._simulate_and_store(x)

        Phi = self.gpr[0].basis_matrix(samples)
        for i in range(2):
            y_i = np.array([self.observations[x][0][i] for x in samples], dtype=float)
            try:
                beta = np.linalg.lstsq(Phi, y_i, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = np.zeros(Phi.shape[1], dtype=float)
            resid = y_i - Phi @ beta
            lambda_data = max(float(np.var(resid)), 1e-6)
            prior_var = max(float(np.var(beta)), 1e-6)
            self.gpr[i].set_parametric_prior(beta, lambda_data, prior_var)

        for x in samples:
            for model in self.gpr:
                model.dimension_augment(x)

        self.variance_model.initialize(
            samples, self.observations, self.gpr, self.problem)

    def _variance_lookup(self, i, x):
        return self.variance_model.predict_variance(i, x, self.problem)

    def _generate_candidates(self, iteration):
        candidates = []
        sources = {}

        def add(rows, source):
            for row in rows:
                x_tuple = tuple(int(v) for v in row)
                if x_tuple not in sources:
                    candidates.append(x_tuple)
                    sources[x_tuple] = source

        add(latin_hypercube_candidates(
            self.problem, self.config.K1, self.rng), "lhs")
        n_axis = int(self.config.axis_candidate_count)
        if n_axis < 0:
            n_axis = 5
        add(axis_candidates(self.problem, n_axis, self.rng), "axis")
        if hasattr(self.problem, "structured_candidates"):
            n_structured = int(self.config.structured_candidate_count)
            if n_structured < 0:
                n_structured = max(5, self.config.K1 // 2)
            add(structured_candidates(
                self.problem, n_structured, self.rng), "structured")
        if self.config.use_state_coupling and self.encoder is not None:
            n_state = int(self.config.state_candidate_count)
            if n_state < 0:
                n_state = max(5, self.config.K1)
            add(self.encoder.state_space_candidates(
                n_anchors=n_state,
                inverse_pool_size=self.config.state_inverse_pool_size,
                inverse_neighbors=self.config.state_inverse_neighbors,
                rng=self.rng,
                observed=[x for x, _ in self.history],
            ), "state")
        add(random_candidates(
            self.problem, max(5, self.config.K1 // 5), self.rng), "random")
        use_constraint = iteration > self.config.n_thr
        add(posterior_sample_candidates(
            self.problem,
            self.gpr,
            n_batches=self.config.K2,
            pool_size=self.config.posterior_pool_size,
            keep_per_batch=self.config.posterior_keep,
            rng=self.rng,
            use_constraint=use_constraint,
            variance_lookup=self._variance_lookup,
            tau=self.problem.tau,
            alpha_z=norm.ppf(1 - self.problem.alpha),
        ), "posterior")
        if not candidates:
            add([self.problem.sample_random(self.rng)], "random")
        return candidates, sources

    def _recommendation_pool(self):
        pool = set(x for x, _ in self.history)
        for x in random_candidates(self.problem, self.config.eval_pool_size, self.rng):
            pool.add(tuple(x))
        if hasattr(self.problem, "all_axis_solutions"):
            for x in self.problem.all_axis_solutions():
                pool.add(tuple(x))
        return list(pool)

    def _observed_nominal_incumbent(self):
        z = norm.ppf(1 - self.problem.alpha)
        sigma_floor = float(getattr(self.problem, "sigma_level", 0.0))
        margin_limit = -0.5 * sigma_floor
        best = None
        for x, ys in self.observations.items():
            y_bar = np.mean(np.asarray(ys, dtype=float), axis=0)
            margin = float(y_bar[1] + z * sigma_floor - self.problem.tau)
            if margin > margin_limit:
                continue
            item = (float(y_bar[0]), margin, tuple(int(v) for v in x))
            if best is None or item < best:
                best = item
        if best is None:
            return None
        obj, margin, x = best
        return {
            "x": x,
            "empirical_objective": float(obj),
            "empirical_chance_margin": float(margin),
        }

    def _solve_posterior_recommendation(self):
        pool = self._recommendation_pool()
        mu_obj = self.gpr[0].posterior_mean_many(pool)
        mu_con = self.gpr[1].posterior_mean_many(pool)
        z = norm.ppf(1 - self.problem.alpha)
        v_con = self.variance_model.predict_certification_variance_many(
            1, pool, self.problem)
        margins = mu_con + z * np.sqrt(np.maximum(v_con, 1e-12)) - self.problem.tau
        sig_con = np.sqrt(np.maximum(v_con, 1e-12))
        nominal_floor = (
            self.config.recommendation_noise_floor_scale
            * float(getattr(self.problem, "sigma_level", 0.0))
        )
        safety_buffer = np.maximum(
            self.config.recommendation_safety_z * sig_con,
            nominal_floor,
        )
        robust_margins = margins + safety_buffer
        feasible = robust_margins <= 0.0
        if np.any(feasible):
            local = int(np.argmin(np.where(feasible, mu_obj, np.inf)))
        else:
            scaled_obj = mu_obj - float(np.min(mu_obj))
            obj_span = float(np.max(scaled_obj))
            if obj_span > 1e-12:
                scaled_obj = scaled_obj / obj_span
            scaled_margin = np.maximum(robust_margins, 0.0)
            margin_span = float(np.max(scaled_margin))
            if margin_span > 1e-12:
                scaled_margin = scaled_margin / margin_span
            local = int(np.argmin(scaled_obj + 2.0 * scaled_margin))
        used_observed_incumbent = False
        observed_incumbent = self._observed_nominal_incumbent()
        if (
            observed_incumbent is not None
            and observed_incumbent["empirical_objective"] <= float(mu_obj[local])
        ):
            try:
                local = pool.index(observed_incumbent["x"])
                used_observed_incumbent = True
            except ValueError:
                pass
        x_best = tuple(int(v) for v in pool[local])
        return x_best, {
            "posterior_mu_obj": float(mu_obj[local]),
            "posterior_mu_con": float(mu_con[local]),
            "posterior_variance_con": float(v_con[local]),
            "posterior_chance_margin": float(margins[local]),
            "posterior_robust_chance_margin": float(robust_margins[local]),
            "recommendation_safety_z": float(self.config.recommendation_safety_z),
            "recommendation_noise_floor_scale": float(
                self.config.recommendation_noise_floor_scale),
            "observed_incumbent_used": bool(used_observed_incumbent),
            "observed_incumbent_objective": (
                None if observed_incumbent is None
                else float(observed_incumbent["empirical_objective"])
            ),
            "observed_incumbent_chance_margin": (
                None if observed_incumbent is None
                else float(observed_incumbent["empirical_chance_margin"])
            ),
            "posterior_feasible": bool(feasible[local]),
            "n_pool": int(len(pool)),
            "n_posterior_feasible": int(np.sum(feasible)),
        }

    def _evaluate_recommendation(self, x_best):
        true_obj = self.problem.true_objective(x_best)
        true_con = self.problem.true_constraint_mean(x_best)
        true_sig = self.problem.true_sigma(x_best)
        true_margin = (
            true_con
            + norm.ppf(1 - self.problem.alpha) * true_sig[1]
            - self.problem.tau
        )
        true_best_x, true_best_obj = self.problem.true_best_feasible()
        regret = true_obj - true_best_obj if np.isfinite(true_best_obj) else np.nan
        return {
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

    def run(self, verbose=False):
        t_start = time.time()
        samples = self._initial_samples()
        t0 = time.time()
        self._fit_initial_belief(samples)
        self.pre_sampling_log = {
            "n0": self.config.n0,
            "samples": [list(map(int, x)) for x in samples],
            "time_sec": float(time.time() - t0),
            "variance": self.variance_model.diagnostics(),
        }

        for n in range(self.config.n0, self.config.N):
            iteration = n - self.config.n0
            row = {"iteration": iteration, "stage": n}
            t_iter = time.time()

            t0 = time.time()
            rec_x, rec_details = self._solve_posterior_recommendation()
            row["t_posterior_solve"] = time.time() - t0
            row["recommendation_before"] = list(map(int, rec_x))
            row.update({f"rec_{k}": v for k, v in rec_details.items()})

            t0 = time.time()
            candidates, candidate_sources = self._generate_candidates(iteration)
            row["t_candidate_gen"] = time.time() - t0
            row["n_candidates"] = len(candidates)

            t0 = time.time()
            score = self.acquisition.score(
                candidates,
                self.gpr[0],
                self.gpr[1],
                self.variance_model,
                self.problem,
                observed=self.history,
            )
            selected_idx = int(np.argmax(score["total"]))
            x_selected = candidates[selected_idx]
            row["t_kg_compute"] = time.time() - t0
            row["x_selected"] = list(map(int, x_selected))
            row["candidate_source_selected"] = candidate_sources.get(
                tuple(x_selected), "unknown")
            row["score_selected"] = float(score["total"][selected_idx])
            row["kg_obj_selected"] = float(score["kg_obj"][selected_idx])
            row["kg_obj_scaled_selected"] = float(score["kg_obj_scaled"][selected_idx])
            row["kg_feas_selected"] = float(score["kg_feas"][selected_idx])
            row["kg_var_selected"] = float(score["kg_var"][selected_idx])
            row["kg_coupling_selected"] = float(score["kg_coupling"][selected_idx])
            row["kg_coupling_raw_selected"] = float(
                score["kg_coupling_raw"][selected_idx])
            row["kg_coupling_gate_selected"] = float(
                score["kg_coupling_gate"][selected_idx])

            x_arr = np.asarray(x_selected, dtype=int)
            mu_before = [self.gpr[i].posterior_mean(x_arr) for i in range(2)]
            sigma2_before = [
                self.variance_model.predict_variance(i, x_arr, self.problem)
                for i in range(2)
            ]
            row["mu_before"] = [float(v) for v in mu_before]
            row["sigma2_before"] = [float(v) for v in sigma2_before]

            t0 = time.time()
            y = self._simulate_and_store(x_selected)
            row["t_simulate"] = time.time() - t0
            row["Y_observed"] = [float(v) for v in y]

            t0 = time.time()
            for i in range(2):
                self.gpr[i].update(x_arr, y[i], sigma2_before[i])
            hvd_details = []
            for i in range(2):
                hvd_details.append(self.variance_model.update(
                    i, x_arr, y[i], mu_before[i], self.gpr[i], self.problem))
            row["t_update"] = time.time() - t0
            row["hvd_update"] = hvd_details
            row["n_visited"] = len(self.gpr[0].sampled_set)

            t0 = time.time()
            if iteration % 5 == 0 or n == self.config.N - 1:
                rec_x_after, rec_after = self._solve_posterior_recommendation()
                eval_after = self._evaluate_recommendation(rec_x_after)
                row["recommendation_after"] = list(map(int, rec_x_after))
                row["eval"] = {**rec_after, **eval_after}
            row["t_eval"] = time.time() - t0
            row["t_total"] = time.time() - t_iter
            self.iteration_log.append(row)
            if verbose:
                print(
                    f"iter={iteration:03d} x={x_selected} "
                    f"score={row['score_selected']:.4g}"
                )

        final_x, final_post = self._solve_posterior_recommendation()
        final_eval = self._evaluate_recommendation(final_x)
        self.final_log = {
            **final_post,
            **final_eval,
            "total_time_sec": float(time.time() - t_start),
            "n_simulations": int(len(self.history)),
            "n_distinct_solutions": int(len(self.gpr[0].sampled_set)),
            "stage_times": summarize_stage_times(self.iteration_log),
            "variance": self.variance_model.diagnostics(),
            "config": asdict(self.config),
        }
        return self.final_log
