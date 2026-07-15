"""Canonical BoTorch baselines for constrained, noisy optimization.

The adapters in this module follow the state and candidate-generation logic in
the BoTorch TuRBO-1, SCBO, and SAASBO tutorials.  The surrounding benchmark is
integer-valued and chance constrained, so continuous candidates are rounded by
the problem adapter and the observed constraint is shifted by a declared
nominal aleatoric margin.  No analytic problem truth enters model fitting,
candidate generation, posterior certification, or recommendation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
import math
from pathlib import Path
import pickle
import signal
import time
import warnings

import numpy as np
from scipy.stats import norm

from core.candidates import boundary_solutions, unique_candidates


try:  # pragma: no cover - exercised only when the optional stack is installed.
    import torch
    from torch.quasirandom import SobolEngine

    from botorch.acquisition.logei import (
        qLogExpectedImprovement,
        qLogProbabilityOfFeasibility,
    )
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.exceptions.warnings import BotorchWarning
    from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
    from botorch.generation.sampling import (
        ConstrainedMaxPosteriorSampling,
        MaxPosteriorSampling,
    )
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood

    BOTORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional dependency.
    torch = None
    SobolEngine = None
    qLogExpectedImprovement = None
    qLogProbabilityOfFeasibility = None
    GenericMCObjective = None
    BotorchWarning = Warning
    fit_fully_bayesian_model_nuts = None
    fit_gpytorch_mll = None
    ConstrainedMaxPosteriorSampling = None
    MaxPosteriorSampling = None
    ModelListGP = None
    SingleTaskGP = None
    SaasFullyBayesianSingleTaskGP = None
    Standardize = None
    optimize_acqf = None
    SobolQMCNormalSampler = None
    Interval = None
    MaternKernel = None
    ScaleKernel = None
    GaussianLikelihood = None
    ExactMarginalLogLikelihood = None
    BOTORCH_IMPORT_ERROR = exc


CANONICAL_IMPLEMENTATION_ID = "botorch-tutorial-canonical-v1"


def is_botorch_available():
    """Return whether the real BoTorch/GPyTorch stack is importable."""

    return BOTORCH_IMPORT_ERROR is None


def botorch_runtime_fingerprint():
    """Return package versions required to reproduce a baseline row."""

    versions = {}
    for package in ("torch", "botorch", "gpytorch", "pyro-ppl", "scipy", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "implementation_id": CANONICAL_IMPLEMENTATION_ID,
        "versions": versions,
        "torch_device": "cpu",
    }


def canonical_failure_tolerance(dim, batch_size=1):
    """Failure tolerance used by the official TuRBO/SCBO tutorials."""

    batch_size = max(1, int(batch_size))
    return int(math.ceil(max(4.0 / batch_size, float(dim) / batch_size)))


def canonical_ts_candidate_count(dim):
    """Default Thompson-sampling candidate count from the TuRBO tutorial."""

    return int(min(5000, max(2000, 200 * int(dim))))


def canonical_turbo_bounds(center, length, lengthscales):
    """Lengthscale-shaped TuRBO trust-region bounds on the unit cube."""

    center = torch.as_tensor(center, dtype=torch.double).reshape(-1)
    lengthscales = torch.as_tensor(
        lengthscales, dtype=torch.double, device=center.device).reshape(-1)
    if len(lengthscales) != len(center):
        raise ValueError("lengthscale dimension does not match trust-region center")
    lengthscales = torch.clamp(lengthscales, min=1e-12)
    weights = lengthscales / torch.mean(lengthscales)
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    half = 0.5 * float(length)
    return (
        torch.clamp(center - weights * half, 0.0, 1.0),
        torch.clamp(center + weights * half, 0.0, 1.0),
    )


def canonical_scbo_bounds(center, length):
    """Axis-aligned SCBO trust-region bounds on the unit cube."""

    center = torch.as_tensor(center, dtype=torch.double).reshape(-1)
    half = 0.5 * float(length)
    return (
        torch.clamp(center - half, 0.0, 1.0),
        torch.clamp(center + half, 0.0, 1.0),
    )


class _CandidateTimeout(TimeoutError):
    pass


@contextmanager
def _wall_time_limit(seconds):
    """Raise during long candidate generation on POSIX platforms."""

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
    tr_radius_init: float = 0.8
    tr_radius_min: float = 0.5 ** 7
    tr_radius_max: float = 1.6
    tr_success_tolerance: int = 10
    tr_failure_tolerance: int = 0
    raw_samples: int = 1024
    num_restarts: int = 10
    maxiter: int = 100
    timeout_sec: float | None = None
    nominal_sigma_scale: float = 1.0
    certification_beta: float = 2.0
    ts_candidates: int = 0
    gp_noise_lower: float = 1e-6
    gp_noise_upper: float = 1.0
    saas_warmup_steps: int = 256
    saas_num_samples: int = 128
    saas_thinning: int = 16
    saas_max_tree_depth: int = 6
    saas_mc_samples: int = 256
    saas_constrained: bool = True
    max_candidate_failures: int = 1
    saas_fallback_after_failures: bool = False
    strict_failures: bool = True
    use_problem_initial_samples: bool = False
    use_boundary_initial_samples: bool = False
    initial_design: str = "sobol"
    initial_points: tuple | list | None = None
    recommendation_rule: str = "posterior_certified"
    checkpoint_path: str = ""
    checkpoint_resume: bool = False
    checkpoint_interval: int = 1
    progress_logging: bool = False
    progress_label: str = ""


@dataclass
class _TrustRegionState:
    dim: int
    constrained: bool
    length: float
    length_min: float
    length_max: float
    success_tolerance: int
    failure_tolerance: int
    success_counter: int = 0
    failure_counter: int = 0
    best_value: float = -float("inf")
    best_constraint: float = float("inf")
    restart_triggered: bool = False

    def update_length(self):
        if self.success_counter >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0
        if self.length < self.length_min:
            self.restart_triggered = True

    def update_turbo(self, objective):
        threshold = self.best_value + 1e-3 * abs(self.best_value)
        if not math.isfinite(self.best_value) or float(objective) > threshold:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        self.best_value = max(self.best_value, float(objective))
        self.update_length()

    def update_scbo(self, objective, constraint):
        objective = float(objective)
        constraint = float(constraint)
        if constraint <= 0.0:
            threshold = self.best_value + 1e-3 * abs(self.best_value)
            success = objective > threshold or self.best_constraint > 0.0
        else:
            success = max(constraint, 0.0) < max(self.best_constraint, 0.0)
        if success:
            self.success_counter += 1
            self.failure_counter = 0
            self.best_value = objective
            self.best_constraint = constraint
        else:
            self.success_counter = 0
            self.failure_counter += 1
        self.update_length()


class BoTorchBaseline:
    """Sequential canonical BoTorch baseline with chance-margin adaptation."""

    VALID_METHODS = {"botorch_turbo", "botorch_scbo", "botorch_saasbo"}

    def __init__(self, problem, config: BoTorchBaselineConfig):
        method = _normalize_method(config.method)
        if method not in self.VALID_METHODS:
            raise ValueError(f"unknown BoTorch baseline method {config.method!r}")
        if not is_botorch_available():
            raise ImportError(
                "BoTorch baseline requested, but BoTorch/GPyTorch is unavailable"
            ) from BOTORCH_IMPORT_ERROR
        if int(config.n0) < 2 or int(config.N) < int(config.n0):
            raise ValueError("BoTorch baseline requires 2 <= n0 <= N")
        self.problem = problem
        self.config = config
        self.config.method = method
        self.rng = np.random.default_rng(config.seed)
        self.history: list[tuple[tuple[int, ...], np.ndarray]] = []
        self._model_start_index = 0
        self._fit_failures = 0
        self._candidate_failures = 0
        self._timeout_fallback_active = False
        self._restart_count = 0
        self._restart_design_sizes: list[int] = []
        self._initial_design_source = "uninitialized"
        self._last_models = (None, None)
        self._tr = self._new_tr_state(constrained=method == "botorch_scbo")
        self._tr_initialized = False
        self._resumed_from_checkpoint = False
        if self.config.checkpoint_resume and self._checkpoint_path() is not None:
            self._load_checkpoint()

    def _checkpoint_path(self):
        value = str(self.config.checkpoint_path or "").strip()
        return None if not value else Path(value)

    def _checkpoint_signature(self):
        return {
            "method": str(self.config.method),
            "seed": int(self.config.seed),
            "dimension": int(self.problem.d),
            "n0": int(self.config.n0),
        }

    def _save_checkpoint(self, *, force=False):
        path = self._checkpoint_path()
        if path is None:
            return
        interval = max(1, int(self.config.checkpoint_interval))
        if not force and len(self.history) % interval != 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "signature": self._checkpoint_signature(),
            "history": [
                (tuple(map(int, x)), np.asarray(y, dtype=float))
                for x, y in self.history
            ],
            "rng_state": self.rng.bit_generator.state,
            "model_start_index": int(self._model_start_index),
            "trust_region": dict(vars(self._tr)),
            "trust_region_initialized": bool(self._tr_initialized),
            "fit_failures": int(self._fit_failures),
            "candidate_failures": int(self._candidate_failures),
            "timeout_fallback_active": bool(self._timeout_fallback_active),
            "restart_count": int(self._restart_count),
            "restart_design_sizes": list(self._restart_design_sizes),
            "initial_design_source": str(self._initial_design_source),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        if path is None or not path.exists():
            return
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported BoTorch checkpoint schema")
        if payload.get("signature") != self._checkpoint_signature():
            raise ValueError("BoTorch checkpoint signature does not match run")
        history = payload.get("history", [])
        if len(history) > int(self.config.N):
            raise ValueError("checkpoint history exceeds requested budget")
        self.history = [
            (tuple(map(int, x)), np.asarray(y, dtype=float))
            for x, y in history
        ]
        self.rng.bit_generator.state = payload["rng_state"]
        self._model_start_index = int(payload.get("model_start_index", 0))
        self._tr = _TrustRegionState(**payload["trust_region"])
        self._tr_initialized = bool(
            payload.get("trust_region_initialized", False))
        self._fit_failures = int(payload.get("fit_failures", 0))
        self._candidate_failures = int(payload.get("candidate_failures", 0))
        self._timeout_fallback_active = bool(
            payload.get("timeout_fallback_active", False))
        self._restart_count = int(payload.get("restart_count", 0))
        self._restart_design_sizes = list(
            payload.get("restart_design_sizes", []))
        self._initial_design_source = str(
            payload.get("initial_design_source", "checkpoint"))
        self._resumed_from_checkpoint = True

    def _new_tr_state(self, constrained):
        failure_tolerance = int(self.config.tr_failure_tolerance)
        if failure_tolerance <= 0:
            failure_tolerance = canonical_failure_tolerance(self.problem.d, 1)
        return _TrustRegionState(
            dim=int(self.problem.d),
            constrained=bool(constrained),
            length=float(self.config.tr_radius_init),
            length_min=float(self.config.tr_radius_min),
            length_max=float(self.config.tr_radius_max),
            success_tolerance=max(1, int(self.config.tr_success_tolerance)),
            failure_tolerance=max(1, failure_tolerance),
        )

    def _observed_chance_margin(self, y):
        sigma = float(getattr(self.problem, "sigma_level", 0.04))
        z = norm.ppf(1.0 - float(self.problem.alpha))
        return float(
            y[1]
            + z * float(self.config.nominal_sigma_scale) * sigma
            - float(self.problem.tau)
        )

    def _score_observation(self, y):
        margin = self._observed_chance_margin(y)
        if margin <= 0.0:
            return (0, float(y[0]), margin)
        return (1, margin, float(y[0]))

    def _sobol_candidates(self, n, *, seed_offset=0):
        n = max(0, int(n))
        if n == 0:
            return []
        engine = SobolEngine(
            dimension=int(self.problem.d),
            scramble=True,
            seed=int(self.config.seed) + int(seed_offset),
        )
        seen = {x for x, _ in self.history}
        rows = []
        draw_count = max(n, 8)
        while len(rows) < n:
            points = engine.draw(draw_count).to(dtype=torch.double)
            for point in points:
                x = tuple(self.problem.continuous_to_int(
                    point.detach().cpu().numpy()))
                if x in seen:
                    continue
                seen.add(x)
                rows.append(x)
                if len(rows) >= n:
                    break
            draw_count = max(8, n - len(rows))
        return rows

    def _initial_samples(self):
        rows = []
        supplied = self.config.initial_points
        if supplied:
            rows.extend(tuple(int(v) for v in x) for x in supplied)
            rows = unique_candidates(rows)
            self._initial_design_source = "shared_external"
        if (
            len(rows) < int(self.config.n0)
            and self.config.use_problem_initial_samples
            and hasattr(self.problem, "initial_samples")
        ):
            rows.extend(self.problem.initial_samples(
                n=int(self.config.n0) - len(rows), rng=self.rng))
            rows = unique_candidates(rows)
            self._initial_design_source = "problem_hook"
        if len(rows) < int(self.config.n0) and self.config.use_boundary_initial_samples:
            for x in boundary_solutions(self.problem):
                if len(rows) >= int(self.config.n0):
                    break
                rows.append(tuple(x))
                rows = unique_candidates(rows)
            self._initial_design_source = "boundary_hook"
        remaining = int(self.config.n0) - len(rows)
        if remaining > 0:
            mode = str(self.config.initial_design or "sobol").strip().lower()
            if mode != "sobol":
                raise ValueError("canonical BoTorch baselines require Sobol initialization")
            rows.extend(self._sobol_candidates(remaining, seed_offset=0))
            rows = unique_candidates(rows)
            if self._initial_design_source == "uninitialized":
                self._initial_design_source = "sobol"
            else:
                self._initial_design_source += "+sobol"
        return rows[: int(self.config.n0)]

    def _simulate(self, x):
        x = tuple(int(v) for v in x)
        y = np.asarray(self.problem.simulate(x, self.rng), dtype=float)
        self.history.append((x, y))
        return y

    def _training_tensors(self, *, active_only=True):
        rows = self.history[self._model_start_index:] if active_only else self.history
        X = np.asarray([self.problem.normalize(x) for x, _ in rows], dtype=float)
        Y = np.asarray([y for _, y in rows], dtype=float)
        obj = -Y[:, [0]]
        con = np.asarray([
            [self._observed_chance_margin(y)] for y in Y
        ], dtype=float)
        return (
            torch.as_tensor(X, dtype=torch.double),
            torch.as_tensor(obj, dtype=torch.double),
            torch.as_tensor(con, dtype=torch.double),
        )

    def _single_task_model(self, train_X, train_Y):
        dim = int(train_X.shape[-1])
        likelihood = GaussianLikelihood(noise_constraint=Interval(
            float(self.config.gp_noise_lower),
            float(self.config.gp_noise_upper),
        ))
        covar_module = ScaleKernel(MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_constraint=Interval(0.005, 4.0),
        ))
        model = SingleTaskGP(
            train_X,
            train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BotorchWarning)
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={"options": {"maxiter": int(self.config.maxiter)}},
            )
        model.eval()
        return model

    def _fit_standard_models(self, train_X, train_obj, train_con):
        try:
            obj_model = self._single_task_model(train_X, train_obj)
            con_model = self._single_task_model(train_X, train_con)
        except Exception:
            self._fit_failures += 1
            raise
        self._last_models = (obj_model, con_model)
        return obj_model, con_model

    def _fit_objective_model(self, train_X, train_obj):
        try:
            obj_model = self._single_task_model(train_X, train_obj)
        except Exception:
            self._fit_failures += 1
            raise
        self._last_models = (obj_model, self._last_models[1])
        return obj_model

    def _fit_saas_single(self, train_X, train_Y):
        model = SaasFullyBayesianSingleTaskGP(
            train_X,
            train_Y,
            outcome_transform=Standardize(m=1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BotorchWarning)
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

    def _fit_saas_models(self, train_X, train_obj, train_con):
        try:
            obj_model = self._fit_saas_single(train_X, train_obj)
            con_model = self._fit_saas_single(train_X, train_con)
        except Exception:
            self._fit_failures += 1
            raise
        self._last_models = (obj_model, con_model)
        return obj_model, con_model

    @staticmethod
    def _model_lengthscales(model):
        kernel = model.covar_module
        if hasattr(kernel, "base_kernel"):
            kernel = kernel.base_kernel
        lengthscale = kernel.lengthscale.squeeze().detach()
        if lengthscale.ndim == 0:
            lengthscale = lengthscale.expand(model.train_inputs[0].shape[-1])
        return lengthscale.reshape(-1)

    @staticmethod
    def _best_scbo_index(Y, C):
        feasible = C.squeeze(-1) <= 0.0
        if bool(feasible.any()):
            score = Y.squeeze(-1).clone()
            score[~feasible] = -float("inf")
            return int(torch.argmax(score).item())
        return int(torch.argmin(torch.clamp(C.squeeze(-1), min=0.0)).item())

    def _initialize_tr_state(self):
        train_X, train_obj, train_con = self._training_tensors(active_only=True)
        constrained = self.config.method == "botorch_scbo"
        self._tr = self._new_tr_state(constrained=constrained)
        if constrained:
            index = self._best_scbo_index(train_obj, train_con)
            self._tr.best_value = float(train_obj[index].item())
            self._tr.best_constraint = float(train_con[index].item())
        else:
            self._tr.best_value = float(torch.max(train_obj).item())
        self._tr_initialized = True

    def _ts_candidate_pool(self, center, lower, upper, n_candidates, seed):
        engine = SobolEngine(
            dimension=int(self.problem.d), scramble=True, seed=int(seed))
        pert = engine.draw(int(n_candidates)).to(dtype=torch.double)
        pert = lower + (upper - lower) * pert
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 104729)
        probability = min(20.0 / float(self.problem.d), 1.0)
        mask = torch.rand(
            int(n_candidates), int(self.problem.d),
            dtype=torch.double, generator=generator,
        ) <= probability
        empty = torch.where(mask.sum(dim=1) == 0)[0]
        if len(empty):
            columns = torch.randint(
                0, int(self.problem.d), (len(empty),), generator=generator)
            mask[empty, columns] = True
        candidates = center.expand(int(n_candidates), int(self.problem.d)).clone()
        candidates[mask] = pert[mask]
        return candidates

    def _unseen_integer_candidate(self, continuous_rows):
        seen = {x for x, _ in self.history}
        rows = continuous_rows.detach().cpu().numpy().reshape(-1, int(self.problem.d))
        for row in rows:
            x = tuple(self.problem.continuous_to_int(row))
            if x not in seen:
                return x
        return None

    def _canonical_ts_candidate(self, obj_model, con_model=None):
        train_X, train_obj, train_con = self._training_tensors(active_only=True)
        if con_model is None:
            center_index = int(torch.argmax(train_obj.squeeze(-1)).item())
            center = train_X[center_index].clone()
            lower, upper = canonical_turbo_bounds(
                center, self._tr.length, self._model_lengthscales(obj_model))
        else:
            center_index = self._best_scbo_index(train_obj, train_con)
            center = train_X[center_index].clone()
            lower, upper = canonical_scbo_bounds(center, self._tr.length)
        n_candidates = int(self.config.ts_candidates)
        if n_candidates <= 0:
            n_candidates = canonical_ts_candidate_count(self.problem.d)
        for attempt in range(3):
            seed = int(self.rng.integers(1, 2**31 - 1))
            X_cand = self._ts_candidate_pool(
                center, lower, upper, n_candidates, seed)
            if con_model is None:
                sampler = MaxPosteriorSampling(
                    model=obj_model, replacement=False)
            else:
                sampler = ConstrainedMaxPosteriorSampling(
                    model=obj_model,
                    constraint_model=con_model,
                    replacement=False,
                )
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(seed + 2097593)
                candidate = sampler(X_cand, num_samples=1)
            x = self._unseen_integer_candidate(candidate)
            if x is not None:
                return x
        self._candidate_failures += 1
        raise RuntimeError("canonical Thompson sampling repeatedly returned duplicates")

    def _turbo_candidate(self):
        train_X, train_obj, _ = self._training_tensors(active_only=True)
        obj_model = self._fit_objective_model(train_X, train_obj)
        return self._canonical_ts_candidate(obj_model)

    def _scbo_candidate(self):
        train_X, train_obj, train_con = self._training_tensors(active_only=True)
        obj_model, con_model = self._fit_standard_models(
            train_X, train_obj, train_con)
        return self._canonical_ts_candidate(obj_model, con_model)

    def _global_bounds(self):
        return torch.stack([
            torch.zeros(int(self.problem.d), dtype=torch.double),
            torch.ones(int(self.problem.d), dtype=torch.double),
        ])

    def _optimize_saas_acquisition(self, acqf):
        try:
            candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=self._global_bounds(),
                q=1,
                num_restarts=int(self.config.num_restarts),
                raw_samples=int(self.config.raw_samples),
                options={
                    "maxiter": int(self.config.maxiter),
                    "batch_limit": 5,
                    "init_batch_limit": 64,
                },
                timeout_sec=self.config.timeout_sec,
            )
        except Exception:
            self._candidate_failures += 1
            raise
        x = self._unseen_integer_candidate(candidate)
        if x is None:
            self._candidate_failures += 1
            raise RuntimeError("SAASBO acquisition returned an evaluated integer point")
        return x

    def _saas_candidate(self):
        with _wall_time_limit(self.config.timeout_sec):
            train_X, train_obj, train_con = self._training_tensors(active_only=False)
            obj_model, con_model = self._fit_saas_models(
                train_X, train_obj, train_con)
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([
                    max(8, int(self.config.saas_mc_samples))]),
                seed=int(self.rng.integers(1, 2**31 - 1)),
            )
            if self.config.saas_constrained:
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
                acqf = qLogExpectedImprovement(
                    model=obj_model,
                    best_f=train_obj.max(),
                    sampler=sampler,
                )
            return self._optimize_saas_acquisition(acqf)

    def _handle_candidate_error(self, exc):
        if self.config.strict_failures:
            raise exc
        self._candidate_failures += 1
        if (
            self.config.method == "botorch_saasbo"
            and self.config.saas_fallback_after_failures
            and self._candidate_failures >= max(
                1, int(self.config.max_candidate_failures))
        ):
            self._timeout_fallback_active = True
        rows = self._sobol_candidates(1, seed_offset=7919 + len(self.history))
        return rows[0]

    def _next_candidate(self):
        try:
            if self.config.method == "botorch_turbo":
                return self._turbo_candidate()
            if self.config.method == "botorch_scbo":
                return self._scbo_candidate()
            if self.config.method == "botorch_saasbo":
                return self._saas_candidate()
        except Exception as exc:
            return self._handle_candidate_error(exc)
        raise AssertionError(self.config.method)

    def _update_tr_state(self, y):
        if self.config.method == "botorch_turbo":
            self._tr.update_turbo(-float(y[0]))
        elif self.config.method == "botorch_scbo":
            self._tr.update_scbo(
                -float(y[0]), self._observed_chance_margin(y))

    def _restart_if_needed(self, remaining_budget):
        if self.config.method == "botorch_saasbo" or not self._tr.restart_triggered:
            return []
        self._restart_count += 1
        self._model_start_index = len(self.history)
        self._tr = self._new_tr_state(
            constrained=self.config.method == "botorch_scbo")
        self._tr_initialized = False
        restart_size = min(int(self.config.n0), max(0, int(remaining_budget)))
        rows = self._sobol_candidates(
            restart_size, seed_offset=1000003 * self._restart_count)
        self._restart_design_sizes.append(len(rows))
        return rows

    def _posterior_recommendation(self):
        train_X, train_obj, train_con = self._training_tensors(active_only=False)
        if self.config.method == "botorch_saasbo":
            obj_model, con_model = self._fit_saas_models(
                train_X, train_obj, train_con)
        else:
            obj_model, con_model = self._fit_standard_models(
                train_X, train_obj, train_con)
        unique = []
        seen = set()
        for x, _ in self.history:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        X = torch.as_tensor(np.asarray([
            self.problem.normalize(x) for x in unique
        ], dtype=float), dtype=torch.double)
        with torch.no_grad():
            obj_posterior = obj_model.posterior(X)
            con_posterior = con_model.posterior(X)
            obj_mean = getattr(
                obj_posterior, "mixture_mean", obj_posterior.mean)
            con_mean = getattr(
                con_posterior, "mixture_mean", con_posterior.mean)
            con_variance = getattr(
                con_posterior, "mixture_variance", con_posterior.variance)
            objective = obj_mean.reshape(-1).detach().cpu().numpy()
            margin_mean = con_mean.reshape(-1).detach().cpu().numpy()
            margin_std = np.sqrt(np.maximum(
                con_variance.reshape(-1).detach().cpu().numpy(),
                1e-14,
            ))
        margin = margin_mean + math.sqrt(max(
            float(self.config.certification_beta), 0.0)) * margin_std
        feasible = margin <= 0.0
        rule = str(self.config.recommendation_rule).strip().lower()
        if rule != "posterior_certified":
            raise ValueError(f"unknown recommendation rule {rule!r}")
        if np.any(feasible):
            index = int(np.argmax(np.where(feasible, objective, -np.inf)))
        else:
            minimum = float(np.min(margin))
            near = margin <= minimum + 1e-12
            index = int(np.argmax(np.where(near, objective, -np.inf)))
        return unique[index], {
            "posterior_feasible": bool(feasible[index]),
            "posterior_chance_margin": float(margin[index]),
            "posterior_chance_margin_mean": float(margin_mean[index]),
            "posterior_chance_margin_std": float(margin_std[index]),
            "posterior_beta_g": float(self.config.certification_beta),
            "n_posterior_feasible": int(np.sum(feasible)),
            "posterior_certificate_kind": (
                "gp_latent_ucb_plus_nominal_aleatoric_shift"),
            "recommendation_rule": rule,
        }

    def _evaluate_recommendation(self, x_best):
        true_obj = self.problem.true_objective(x_best)
        true_con = self.problem.true_constraint_mean(x_best)
        true_sig = self.problem.true_sigma(x_best)
        true_vector = None
        if hasattr(self.problem, "true_vector_objectives"):
            true_vector = [
                float(v) for v in self.problem.true_vector_objectives(x_best)]
        z = norm.ppf(1.0 - float(self.problem.alpha))
        true_margin = true_con + z * true_sig[1] - self.problem.tau
        true_best_x, true_best_obj = self.problem.true_best_feasible()
        true_best_vector = None
        if true_best_x is not None and hasattr(self.problem, "true_vector_objectives"):
            true_best_vector = [
                float(v) for v in self.problem.true_vector_objectives(true_best_x)]
        regret = true_obj - true_best_obj if math.isfinite(true_best_obj) else np.nan
        out = {
            "x_recommended": list(map(int, x_best)),
            "true_objective": float(true_obj),
            "true_constraint_mean": float(true_con),
            "true_constraint_sigma": float(true_sig[1]),
            "true_chance_margin": float(true_margin),
            "true_feasible": bool(true_margin <= 0.0),
            "true_best_x": (
                None if true_best_x is None else list(map(int, true_best_x))),
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
            f"Iter {current}/{total} [botorch-canonical] "
            f"label={self._progress_label()} Time: {elapsed:.1f}s ETA {eta_sec:.1f}s",
            flush=True,
        )

    def run(self):
        t_start = time.time()
        if len(self.history) < int(self.config.n0):
            for x in self._initial_samples():
                if len(self.history) >= int(self.config.n0):
                    break
                if x in {seen for seen, _ in self.history}:
                    continue
                self._simulate(x)
                self._save_checkpoint()
                self._emit_progress(t_start)
        if (
            self.config.method != "botorch_saasbo"
            and not self._tr_initialized
        ):
            self._initialize_tr_state()
            self._save_checkpoint(force=True)
        while len(self.history) < int(self.config.N):
            restart_rows = self._restart_if_needed(
                int(self.config.N) - len(self.history))
            if restart_rows:
                for x in restart_rows:
                    self._simulate(x)
                    self._emit_progress(t_start)
                if len(self.history) < int(self.config.N):
                    self._initialize_tr_state()
                self._save_checkpoint(force=True)
            else:
                x = self._next_candidate()
                y = self._simulate(x)
                self._update_tr_state(y)
                self._save_checkpoint()
            self._emit_progress(t_start)
        x_best, posterior = self._posterior_recommendation()
        result = self._evaluate_recommendation(x_best)
        runtime = botorch_runtime_fingerprint()
        runtime["torch_device"] = str(
            self._last_models[0].train_inputs[0].device
            if self._last_models[0] is not None else "cpu")
        result.update({
            "method": self.config.method,
            "backend": "botorch",
            **posterior,
            "total_time_sec": float(time.time() - t_start),
            "n_simulations": int(len(self.history)),
            "n_distinct_solutions": int(len(set(x for x, _ in self.history))),
            "tr_radius_final": float(self._tr.length),
            "tr_failure_tolerance": int(self._tr.failure_tolerance),
            "tr_success_tolerance": int(self._tr.success_tolerance),
            "tr_restart_count": int(self._restart_count),
            "tr_restart_design_sizes": list(self._restart_design_sizes),
            "botorch_fit_failures": int(self._fit_failures),
            "botorch_candidate_failures": int(self._candidate_failures),
            "botorch_timeout_fallback": bool(self._timeout_fallback_active),
            "botorch_strict_failures": bool(self.config.strict_failures),
            "checkpoint_path": (
                None if self._checkpoint_path() is None
                else str(self._checkpoint_path())),
            "checkpoint_resumed": bool(self._resumed_from_checkpoint),
            "saas_constrained": bool(self.config.saas_constrained),
            "initial_design": str(self._initial_design_source),
            "ts_candidates": int(
                self.config.ts_candidates
                if int(self.config.ts_candidates) > 0
                else canonical_ts_candidate_count(self.problem.d)
            ),
            "algorithm_fidelity": (
                "canonical_turbo1_ts"
                if self.config.method == "botorch_turbo"
                else (
                    "canonical_scbo_constrained_ts"
                    if self.config.method == "botorch_scbo"
                    else "saas_fully_bayesian_nuts_constrained_qlogei"
                )
            ),
            "saas_nuts_schedule": {
                "warmup_steps": int(self.config.saas_warmup_steps),
                "num_samples": int(self.config.saas_num_samples),
                "thinning": int(self.config.saas_thinning),
                "max_tree_depth": int(self.config.saas_max_tree_depth),
                "mc_samples": int(self.config.saas_mc_samples),
                "formal_budget": bool(
                    int(self.config.saas_warmup_steps) >= 256
                    and int(self.config.saas_num_samples) >= 128
                ),
            },
            "runtime_fingerprint": runtime,
            "history": [
                {"x": list(map(int, x)), "y": np.asarray(y, dtype=float).tolist()}
                for x, y in self.history
            ],
        })
        self._save_checkpoint(force=True)
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
