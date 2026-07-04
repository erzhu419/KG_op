"""BoTorch-backed TuRBO, SCBO, and SAASBO baselines.

The lightweight baselines in :mod:`baseline_algorithms` are useful for fast
regression tests, but they are not the actual BoTorch stack.  This module keeps
the same evaluation semantics while using BoTorch models, acquisition
functions, and `optimize_acqf` for budget-matched SOTA comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math
import signal
import time
import warnings

import numpy as np
from scipy.stats import norm

from core.candidates import boundary_solutions, unique_candidates


try:  # pragma: no cover - exercised in environments with BoTorch installed.
    import torch
    from botorch.acquisition.analytic import (
        LogConstrainedExpectedImprovement,
        LogExpectedImprovement,
        LogProbabilityOfFeasibility,
    )
    from botorch.acquisition.logei import qLogExpectedImprovement, qLogProbabilityOfFeasibility
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.exceptions.warnings import BotorchWarning
    from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

    BOTORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional dependency.
    torch = None
    LogConstrainedExpectedImprovement = None
    LogExpectedImprovement = None
    LogProbabilityOfFeasibility = None
    qLogExpectedImprovement = None
    qLogProbabilityOfFeasibility = None
    GenericMCObjective = None
    BotorchWarning = Warning
    fit_fully_bayesian_model_nuts = None
    fit_gpytorch_mll = None
    ModelListGP = None
    SingleTaskGP = None
    SaasFullyBayesianSingleTaskGP = None
    optimize_acqf = None
    SobolQMCNormalSampler = None
    ExactMarginalLogLikelihood = None
    SumMarginalLogLikelihood = None
    BOTORCH_IMPORT_ERROR = exc


def is_botorch_available():
    """Return whether the real BoTorch backend can be used."""

    return BOTORCH_IMPORT_ERROR is None


class _CandidateTimeout(TimeoutError):
    pass


@contextmanager
def _wall_time_limit(seconds):
    """Raise during long BoTorch candidate generation on POSIX platforms."""
    if seconds is None or float(seconds) <= 0.0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _raise_timeout(signum, frame):
        del signum, frame
        raise _CandidateTimeout(f"BoTorch candidate timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


@dataclass
class BoTorchBaselineConfig:
    N: int = 30
    n0: int = 8
    seed: int = 123
    method: str = "botorch_scbo"
    batch_candidates: int = 128
    tr_radius_init: float = 0.35
    tr_radius_min: float = 0.04
    tr_radius_max: float = 0.8
    tr_success_tolerance: int = 3
    tr_failure_tolerance: int = 5
    raw_samples: int = 64
    num_restarts: int = 5
    maxiter: int = 50
    timeout_sec: float | None = None
    nominal_sigma_scale: float = 1.0
    saas_warmup_steps: int = 16
    saas_num_samples: int = 16
    saas_thinning: int = 1
    saas_max_tree_depth: int = 4
    saas_mc_samples: int = 64
    saas_constrained: bool = True
    max_candidate_failures: int = 8
    saas_fallback_after_failures: bool = True


@dataclass
class _TrustRegion:
    length: float
    length_min: float
    length_max: float
    success_tolerance: int
    failure_tolerance: int
    success_counter: int = 0
    failure_counter: int = 0
    best_score: tuple | None = None

    def update(self, score):
        improved = self.best_score is None or score < self.best_score
        if improved:
            self.best_score = score
            self.success_counter += 1
            self.failure_counter = 0
            if self.success_counter >= self.success_tolerance:
                self.length = min(self.length_max, 2.0 * self.length)
                self.success_counter = 0
            return
        self.failure_counter += 1
        self.success_counter = 0
        if self.failure_counter >= self.failure_tolerance:
            self.length = max(self.length_min, 0.5 * self.length)
            self.failure_counter = 0


class BoTorchBaseline:
    """Sequential BoTorch baseline with shared chance-constraint semantics."""

    VALID_METHODS = {"botorch_turbo", "botorch_scbo", "botorch_saasbo"}

    def __init__(self, problem, config: BoTorchBaselineConfig):
        method = _normalize_method(config.method)
        if method not in self.VALID_METHODS:
            raise ValueError(f"unknown BoTorch baseline method {config.method!r}")
        if not is_botorch_available():
            raise ImportError(
                "BoTorch baseline requested, but BoTorch/GPyTorch is unavailable"
            ) from BOTORCH_IMPORT_ERROR
        self.problem = problem
        self.config = config
        self.config.method = method
        self.rng = np.random.default_rng(config.seed)
        self.history: list[tuple[tuple[int, ...], np.ndarray]] = []
        self._tr = _TrustRegion(
            length=float(config.tr_radius_init),
            length_min=float(config.tr_radius_min),
            length_max=float(config.tr_radius_max),
            success_tolerance=max(1, int(config.tr_success_tolerance)),
            failure_tolerance=max(1, int(config.tr_failure_tolerance)),
        )
        self._last_fit_failures = 0
        self._candidate_failures = 0
        self._timeout_fallback_active = False

    def _nominal_margin(self, y):
        sigma = float(getattr(self.problem, "sigma_level", 0.04))
        z = norm.ppf(1 - self.problem.alpha)
        return float(
            y[1] + z * self.config.nominal_sigma_scale * sigma - self.problem.tau
        )

    def _score_observation(self, y):
        margin = self._nominal_margin(y)
        if margin <= 0.0:
            return (0, float(y[0]), margin)
        return (1, margin, float(y[0]))

    def _objective_score(self, y):
        return (float(y[0]), self._nominal_margin(y))

    def _initial_samples(self):
        rows = []
        if hasattr(self.problem, "initial_samples"):
            rows.extend(self.problem.initial_samples(n=self.config.n0, rng=self.rng))
            rows = unique_candidates(rows)
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
            key = lambda item: self._objective_score(item[1])
        return min(self.history, key=key)[0]

    def _update_tr_radius(self, y):
        if self.config.method == "botorch_turbo":
            self._tr.update(self._objective_score(y))
        else:
            self._tr.update(self._score_observation(y))

    def _training_tensors(self):
        X = np.asarray([self.problem.normalize(x) for x, _ in self.history], dtype=float)
        Y = np.asarray([y for _, y in self.history], dtype=float)
        obj = -Y[:, [0]]
        con = np.asarray([[self._nominal_margin(y)] for y in Y], dtype=float)
        return (
            torch.as_tensor(X, dtype=torch.double),
            torch.as_tensor(obj, dtype=torch.double),
            torch.as_tensor(con, dtype=torch.double),
        )

    def _fit_single_task(self, train_X, train_Y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BotorchWarning)
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={"options": {"maxiter": int(self.config.maxiter)}},
            )
        model.eval()
        return model

    def _fit_model_list(self, train_X, train_obj, train_con):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BotorchWarning)
            obj_model = SingleTaskGP(train_X, train_obj)
            con_model = SingleTaskGP(train_X, train_con)
            model = ModelListGP(obj_model, con_model)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={"options": {"maxiter": int(self.config.maxiter)}},
            )
        model.eval()
        return model

    def _fit_saas_single(self, train_X, train_Y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BotorchWarning)
            model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
            fit_fully_bayesian_model_nuts(
                model,
                warmup_steps=int(self.config.saas_warmup_steps),
                num_samples=int(self.config.saas_num_samples),
                thinning=int(self.config.saas_thinning),
                max_tree_depth=int(self.config.saas_max_tree_depth),
                disable_progbar=True,
            )
        model.eval()
        return model

    def _trust_region_bounds(self, feasible_first=True):
        center = np.asarray(
            self.problem.normalize(self._observed_best(feasible_first=feasible_first)),
            dtype=float,
        )
        half = 0.5 * float(self._tr.length)
        lower = np.clip(center - half, 0.0, 1.0)
        upper = np.clip(center + half, 0.0, 1.0)
        too_small = upper - lower < 1e-6
        upper[too_small] = np.minimum(1.0, lower[too_small] + 1e-3)
        return torch.as_tensor(np.vstack([lower, upper]), dtype=torch.double)

    def _global_bounds(self):
        return torch.stack([
            torch.zeros(int(self.problem.d), dtype=torch.double),
            torch.ones(int(self.problem.d), dtype=torch.double),
        ])

    def _dedupe_or_fallback(self, x, bounds, acqf=None):
        seen = {row for row, _ in self.history}
        if tuple(x) not in seen:
            return tuple(x)
        lower = bounds[0].detach().cpu().numpy()
        upper = bounds[1].detach().cpu().numpy()
        rows = []
        for _ in range(max(8, int(self.config.batch_candidates))):
            z = lower + self.rng.random(int(self.problem.d)) * (upper - lower)
            rows.append(self.problem.continuous_to_int(z))
        rows = [row for row in unique_candidates(rows) if row not in seen]
        if not rows:
            return self.problem.sample_random(self.rng)
        if acqf is None:
            return rows[int(self.rng.integers(0, len(rows)))]
        try:
            X = torch.as_tensor(
                np.asarray([self.problem.normalize(row) for row in rows], dtype=float),
                dtype=torch.double,
            ).unsqueeze(1)
            with torch.no_grad():
                vals = acqf(X).detach().cpu().numpy().reshape(-1)
            order = np.argsort(-np.nan_to_num(vals, nan=-np.inf))
            return rows[int(order[0])]
        except Exception:
            return rows[int(self.rng.integers(0, len(rows)))]

    def _cheap_candidate(self, feasible_first=True):
        bounds = self._trust_region_bounds(feasible_first=feasible_first)
        return self._dedupe_or_fallback(self.problem.sample_random(self.rng), bounds)

    def _failure_budget_exhausted(self):
        budget = int(getattr(self.config, "max_candidate_failures", 0))
        return budget > 0 and self._candidate_failures >= budget

    def _optimize_acqf(self, acqf, bounds):
        options = {
            "maxiter": int(self.config.maxiter),
            "batch_limit": 5,
            "init_batch_limit": 64,
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BotorchWarning)
                cand, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=1,
                    num_restarts=int(self.config.num_restarts),
                    raw_samples=int(self.config.raw_samples),
                    options=options,
                    timeout_sec=self.config.timeout_sec,
                )
            x = self.problem.continuous_to_int(cand.detach().cpu().numpy().reshape(-1))
            return self._dedupe_or_fallback(x, bounds, acqf=acqf)
        except Exception:
            self._candidate_failures += 1
            return self._dedupe_or_fallback(self.problem.sample_random(self.rng), bounds)

    def _turbo_candidate(self):
        train_X, train_obj, _ = self._training_tensors()
        try:
            model = self._fit_single_task(train_X, train_obj)
            best_f = train_obj.max()
            acqf = LogExpectedImprovement(model=model, best_f=best_f)
            return self._optimize_acqf(acqf, self._trust_region_bounds(feasible_first=False))
        except Exception:
            self._last_fit_failures += 1
            return self._dedupe_or_fallback(
                self.problem.sample_random(self.rng),
                self._trust_region_bounds(feasible_first=False),
            )

    def _scbo_candidate(self):
        train_X, train_obj, train_con = self._training_tensors()
        try:
            model = self._fit_model_list(train_X, train_obj, train_con)
            feasible = train_con.squeeze(-1) <= 0.0
            if bool(feasible.any()):
                best_f = train_obj.squeeze(-1)[feasible].max()
                acqf = LogConstrainedExpectedImprovement(
                    model=model,
                    best_f=best_f,
                    objective_index=0,
                    constraints={1: (None, 0.0)},
                )
            else:
                acqf = LogProbabilityOfFeasibility(
                    model=model,
                    constraints={1: (None, 0.0)},
                )
            return self._optimize_acqf(acqf, self._trust_region_bounds(feasible_first=True))
        except Exception:
            self._last_fit_failures += 1
            return self._dedupe_or_fallback(
                self.problem.sample_random(self.rng),
                self._trust_region_bounds(feasible_first=True),
            )

    def _saas_candidate(self):
        if self._timeout_fallback_active:
            return self._cheap_candidate(feasible_first=True)
        try:
            with _wall_time_limit(self.config.timeout_sec):
                train_X, train_obj, train_con = self._training_tensors()
                sampler = SobolQMCNormalSampler(
                    sample_shape=torch.Size([max(8, int(self.config.saas_mc_samples))]),
                    seed=int(self.rng.integers(1, 2**31 - 1)),
                )
                if self.config.saas_constrained:
                    obj_model = self._fit_saas_single(train_X, train_obj)
                    con_model = self._fit_saas_single(train_X, train_con)
                    model = ModelListGP(obj_model, con_model)
                    feasible = train_con.squeeze(-1) <= 0.0
                    constraints = [lambda samples: samples[..., 1]]
                    if bool(feasible.any()):
                        best_f = train_obj.squeeze(-1)[feasible].max()
                        objective = GenericMCObjective(
                            lambda samples, X=None: samples[..., 0])
                        acqf = qLogExpectedImprovement(
                            model=model,
                            best_f=best_f,
                            sampler=sampler,
                            objective=objective,
                            constraints=constraints,
                        )
                    else:
                        acqf = qLogProbabilityOfFeasibility(
                            model=model,
                            constraints=constraints,
                            sampler=sampler,
                        )
                else:
                    model = self._fit_saas_single(train_X, train_obj)
                    acqf = qLogExpectedImprovement(
                        model=model,
                        best_f=train_obj.max(),
                        sampler=sampler,
                    )
                return self._optimize_acqf(acqf, self._global_bounds())
        except Exception:
            self._last_fit_failures += 1
            self._candidate_failures += 1
            if (
                self.config.saas_fallback_after_failures
                and self._failure_budget_exhausted()
            ):
                self._timeout_fallback_active = True
            bounds = self._global_bounds()
            return self._dedupe_or_fallback(self.problem.sample_random(self.rng), bounds)

    def _next_candidate(self):
        method = self.config.method
        if method == "botorch_turbo":
            return self._turbo_candidate()
        if method == "botorch_scbo":
            return self._scbo_candidate()
        if method == "botorch_saasbo":
            if (
                self.config.saas_fallback_after_failures
                and self._failure_budget_exhausted()
            ):
                self._timeout_fallback_active = True
            return self._saas_candidate()
        raise AssertionError(method)

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
        regret = true_obj - true_best_obj if math.isfinite(true_best_obj) else np.nan
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
            "backend": "botorch",
            "posterior_feasible": bool(self._nominal_margin(y_best) <= 0.0),
            "posterior_chance_margin": float(self._nominal_margin(y_best)),
            "total_time_sec": float(time.time() - t_start),
            "n_simulations": int(len(self.history)),
            "n_distinct_solutions": int(len(set(x for x, _ in self.history))),
            "tr_radius_final": float(self._tr.length),
            "botorch_fit_failures": int(self._last_fit_failures),
            "botorch_candidate_failures": int(self._candidate_failures),
            "botorch_timeout_fallback": bool(self._timeout_fallback_active),
            "saas_constrained": bool(self.config.saas_constrained),
        })
        return result


def _normalize_method(method):
    aliases = {
        "turbo": "botorch_turbo",
        "scbo": "botorch_scbo",
        "saasbo": "botorch_saasbo",
    }
    return aliases.get(str(method), str(method))


def fallback_method(method):
    """Return the dependency-light fallback for a BoTorch method."""

    method = _normalize_method(method)
    if method == "botorch_scbo":
        return "scbo_lite"
    if method in ("botorch_turbo", "botorch_saasbo"):
        return "turbo_lite"
    return method
