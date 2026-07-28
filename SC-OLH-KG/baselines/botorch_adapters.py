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
import gc
import hashlib
from importlib import metadata
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import signal
import time
import warnings

import numpy as np
from scipy.stats import norm

from core.candidates import boundary_solutions, unique_candidates
from core.designs import integer_design_fingerprint
from core.terminal_verification import (
    build_verification_aware_shortlist,
    select_posterior_safe_interior,
)


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


def _fit_saas_cpu_worker(payload):
    """Fit one independent SAAS output in an isolated CPU process."""

    threads = max(1, int(payload["threads"]))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed = int(payload["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import pyro

        pyro.set_rng_seed(seed)
    except ImportError:
        pass
    train_X = torch.as_tensor(payload["train_X"], dtype=torch.double)
    train_Y = torch.as_tensor(payload["train_Y"], dtype=torch.double)
    model = SaasFullyBayesianSingleTaskGP(
        train_X,
        train_Y,
        outcome_transform=Standardize(m=1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BotorchWarning)
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=int(payload["warmup_steps"]),
            num_samples=int(payload["num_samples"]),
            thinning=int(payload["thinning"]),
            max_tree_depth=int(payload["max_tree_depth"]),
            disable_progbar=True,
        )
    model.eval()
    return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)


def is_botorch_available():
    """Return whether the real BoTorch/GPyTorch stack is importable."""

    return BOTORCH_IMPORT_ERROR is None


def botorch_runtime_fingerprint(torch_device="cpu"):
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
        "torch_device": str(torch_device),
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
    torch_device: str = "cpu"
    saas_parallel_models: bool = True
    saas_parallel_min_total_steps: int = 64
    saas_parallel_threads_per_model: int = 0
    saas_parallel_start_method: str = "spawn"
    saas_parallel_fallback: bool = True
    saas_refit_schedule: str = "every_iteration"
    saas_refit_interval: int = 16
    saas_refit_growth_factor: float = 2.0
    saas_refit_max_history: int = 0


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
        requested_device = str(config.torch_device or "cpu").strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"torch_device={requested_device!r} requested but CUDA is unavailable"
            )
        self._torch_device = torch.device(requested_device)
        self.rng = np.random.default_rng(config.seed)
        self.history: list[tuple[tuple[int, ...], np.ndarray]] = []
        self._model_start_index = 0
        self._fit_failures = 0
        self._candidate_failures = 0
        self._saas_parallel_fit_count = 0
        self._saas_parallel_failures = 0
        self._saas_parallel_last_error = ""
        self._saas_cached_models = (None, None)
        self._saas_cached_history_size = 0
        self._saas_last_refit_history_size = 0
        self._saas_full_refit_count = 0
        self._saas_condition_count = 0
        self._saas_condition_failures = 0
        self._saas_condition_last_error = ""
        self._saas_resume_rebuild_history_size = 0
        self._saas_discrete_candidate_fallback_count = 0
        self._timeout_fallback_active = False
        self._restart_count = 0
        self._restart_design_sizes: list[int] = []
        self._initial_design_source = "uninitialized"
        self._last_models = (None, None)
        self._tr = self._new_tr_state(constrained=method == "botorch_scbo")
        self._tr_initialized = False
        self._resumed_from_checkpoint = False
        self._normalized_saas_refit_schedule()
        if self.config.checkpoint_resume and self._checkpoint_path() is not None:
            self._load_checkpoint()

    def _stage_seed(self, stage, *, history_size=None):
        """Stable per-iteration seed, independent of process restart timing."""

        completed = len(self.history) if history_size is None else int(history_size)
        material = (
            f"scolhkg-botorch-v1|{int(self.config.seed)}|{completed}|"
            f"{int(self._model_start_index)}|{stage}"
        ).encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        return 1 + int.from_bytes(digest, "big") % (2**31 - 2)

    @contextmanager
    def _deterministic_torch_stage(self, stage):
        """Isolate Torch/Pyro/Python RNGs for restart-stable model stages."""

        seed = self._stage_seed(stage)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = []
        if self._torch_device.type == "cuda":
            cuda_devices = [
                self._torch_device.index
                if self._torch_device.index is not None
                else torch.cuda.current_device()
            ]
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(seed)
                try:
                    import pyro

                    pyro.set_rng_seed(seed)
                except ImportError:
                    pass
                yield seed
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

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
            "schema_version": 2,
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
            "saas_discrete_candidate_fallback_count": int(
                self._saas_discrete_candidate_fallback_count),
            "timeout_fallback_active": bool(self._timeout_fallback_active),
            "restart_count": int(self._restart_count),
            "restart_design_sizes": list(self._restart_design_sizes),
            "initial_design_source": str(self._initial_design_source),
            "stochastic_schedule": {
                "kind": "per_iteration_stage_seed_v1",
                "base_seed": int(self.config.seed),
                "completed_evaluations": int(len(self.history)),
            },
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
        if int(payload.get("schema_version", 0)) not in (1, 2):
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
        self._saas_discrete_candidate_fallback_count = int(payload.get(
            "saas_discrete_candidate_fallback_count", 0))
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
            points = engine.draw(draw_count).to(
                dtype=torch.double, device=self._torch_device)
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
            torch.as_tensor(X, dtype=torch.double, device=self._torch_device),
            torch.as_tensor(obj, dtype=torch.double, device=self._torch_device),
            torch.as_tensor(con, dtype=torch.double, device=self._torch_device),
        )

    def _single_task_model(self, train_X, train_Y, *, role):
        with self._deterministic_torch_stage(f"standard_fit:{role}"):
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
            obj_model = self._single_task_model(
                train_X, train_obj, role="objective")
            con_model = self._single_task_model(
                train_X, train_con, role="constraint")
        except Exception:
            self._fit_failures += 1
            raise
        self._last_models = (obj_model, con_model)
        return obj_model, con_model

    def _fit_objective_model(self, train_X, train_obj):
        try:
            obj_model = self._single_task_model(
                train_X, train_obj, role="objective")
        except Exception:
            self._fit_failures += 1
            raise
        self._last_models = (obj_model, self._last_models[1])
        return obj_model

    def _fit_saas_single(self, train_X, train_Y, *, role):
        with self._deterministic_torch_stage(f"saas_nuts:{role}"):
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

    def _use_parallel_saas_models(self):
        total_steps = (
            int(self.config.saas_warmup_steps)
            + int(self.config.saas_num_samples)
        )
        return bool(
            self.config.saas_parallel_models
            and self._torch_device.type == "cpu"
            and total_steps >= int(self.config.saas_parallel_min_total_steps)
        )

    def _saas_parallel_threads(self):
        configured = int(self.config.saas_parallel_threads_per_model)
        if configured > 0:
            return configured
        try:
            total_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
        except ValueError:
            total_threads = 2
        return max(1, total_threads // 2)

    def _fit_saas_models_parallel(self, train_X, train_obj, train_con):
        common = {
            "train_X": train_X.detach().cpu().numpy(),
            "warmup_steps": int(self.config.saas_warmup_steps),
            "num_samples": int(self.config.saas_num_samples),
            "thinning": int(self.config.saas_thinning),
            "max_tree_depth": int(self.config.saas_max_tree_depth),
            "threads": int(self._saas_parallel_threads()),
        }
        payloads = [
            {
                **common,
                "train_Y": train_obj.detach().cpu().numpy(),
                "seed": self._stage_seed("saas_nuts:objective"),
            },
            {
                **common,
                "train_Y": train_con.detach().cpu().numpy(),
                "seed": self._stage_seed("saas_nuts:constraint"),
            },
        ]
        context = mp.get_context(str(self.config.saas_parallel_start_method))
        pool = context.Pool(processes=2, maxtasksperchild=1)
        try:
            serialized_models = pool.map(_fit_saas_cpu_worker, payloads)
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
        self._saas_parallel_fit_count += 1
        return tuple(pickle.loads(payload) for payload in serialized_models)

    def _fit_saas_models(self, train_X, train_obj, train_con):
        try:
            if self._use_parallel_saas_models():
                try:
                    obj_model, con_model = self._fit_saas_models_parallel(
                        train_X, train_obj, train_con)
                except Exception as exc:
                    self._saas_parallel_failures += 1
                    self._saas_parallel_last_error = (
                        f"{type(exc).__name__}: {exc}")[:300]
                    if not self.config.saas_parallel_fallback:
                        raise
                    obj_model = self._fit_saas_single(
                        train_X, train_obj, role="objective")
                    con_model = self._fit_saas_single(
                        train_X, train_con, role="constraint")
            else:
                obj_model = self._fit_saas_single(
                    train_X, train_obj, role="objective")
                con_model = self._fit_saas_single(
                    train_X, train_con, role="constraint")
        except Exception:
            self._fit_failures += 1
            raise
        self._last_models = (obj_model, con_model)
        return obj_model, con_model

    def _normalized_saas_refit_schedule(self):
        schedule = str(
            getattr(self.config, "saas_refit_schedule", "every_iteration")
            or "every_iteration"
        ).strip().lower()
        aliases = {
            "canonical": "every_iteration",
            "every": "every_iteration",
            "periodic": "interval",
            "geometric": "doubling",
        }
        schedule = aliases.get(schedule, schedule)
        if schedule not in {"every_iteration", "interval", "doubling"}:
            raise ValueError(
                "saas_refit_schedule must be every_iteration, interval, or "
                f"doubling, got {schedule!r}"
            )
        return schedule

    def _saas_should_refit(self, history_size):
        if self._saas_cached_models[0] is None:
            return True
        schedule = self._normalized_saas_refit_schedule()
        if schedule == "every_iteration":
            return True
        last = int(self._saas_last_refit_history_size)
        max_history = max(
            0, int(getattr(self.config, "saas_refit_max_history", 0)))
        effective_history = int(history_size)
        if max_history > 0:
            effective_history = min(effective_history, max_history)
            if last >= max_history:
                return False
        if schedule == "interval":
            interval = max(1, int(self.config.saas_refit_interval))
            threshold = last + interval
            if max_history > 0:
                threshold = min(threshold, max_history)
            return effective_history >= threshold
        growth = max(1.01, float(self.config.saas_refit_growth_factor))
        threshold = max(last + 1, int(math.ceil(last * growth)))
        if max_history > 0:
            threshold = min(threshold, max_history)
        return effective_history >= threshold

    def _condition_saas_models_to_history(self):
        history_size = len(self.history)
        start = int(self._saas_cached_history_size)
        if start >= history_size:
            return self._saas_cached_models
        train_X, train_obj, train_con = self._training_tensors(
            active_only=False)
        X_new = train_X[start:history_size].detach()
        obj_new = train_obj[start:history_size].detach()
        con_new = train_con[start:history_size].detach()
        obj_model, con_model = self._saas_cached_models
        # Conditioning is an exact posterior update, not a differentiable
        # training step. Retaining its fantasy graph across hundreds of
        # sequential updates otherwise grows CUDA memory linearly.
        with torch.no_grad():
            for model in (obj_model, con_model):
                if (
                    hasattr(model, "prediction_strategy")
                    and model.prediction_strategy is None
                ):
                    model.posterior(X_new[:1])
            obj_model = obj_model.condition_on_observations(
                X=X_new,
                Y=obj_new,
            )
            con_model = con_model.condition_on_observations(
                X=X_new,
                Y=con_new,
            )
        obj_model.eval()
        con_model.eval()
        self._saas_cached_models = (obj_model, con_model)
        self._saas_cached_history_size = int(history_size)
        self._saas_condition_count += int(history_size - start)
        self._last_models = self._saas_cached_models
        return self._saas_cached_models

    def _saas_models_for_history(self):
        history_size = len(self.history)
        schedule = self._normalized_saas_refit_schedule()
        max_history = max(
            0, int(getattr(self.config, "saas_refit_max_history", 0)))
        if (
            self._saas_cached_models[0] is None
            and schedule != "every_iteration"
            and max_history > 0
            and history_size > max_history
        ):
            train_X, train_obj, train_con = self._training_tensors(
                active_only=False)
            models = self._fit_saas_models(
                train_X[:max_history],
                train_obj[:max_history],
                train_con[:max_history],
            )
            self._saas_cached_models = models
            self._saas_cached_history_size = int(max_history)
            self._saas_last_refit_history_size = int(max_history)
            self._saas_resume_rebuild_history_size = int(max_history)
            self._saas_full_refit_count += 1
            return self._condition_saas_models_to_history()
        if self._saas_should_refit(history_size):
            train_X, train_obj, train_con = self._training_tensors(
                active_only=False)
            models = self._fit_saas_models(train_X, train_obj, train_con)
            self._saas_cached_models = models
            self._saas_cached_history_size = int(history_size)
            self._saas_last_refit_history_size = int(history_size)
            self._saas_full_refit_count += 1
            return models
        try:
            return self._condition_saas_models_to_history()
        except Exception as exc:
            self._saas_condition_failures += 1
            self._saas_condition_last_error = (
                f"{type(exc).__name__}: {exc}")[:300]
            train_X, train_obj, train_con = self._training_tensors(
                active_only=False)
            models = self._fit_saas_models(train_X, train_obj, train_con)
            self._saas_cached_models = models
            self._saas_cached_history_size = int(history_size)
            self._saas_last_refit_history_size = int(history_size)
            self._saas_full_refit_count += 1
            return models

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
        pert = engine.draw(int(n_candidates)).to(
            dtype=torch.double, device=center.device)
        pert = lower + (upper - lower) * pert
        generator = torch.Generator(device=center.device)
        generator.manual_seed(int(seed) + 104729)
        probability = min(20.0 / float(self.problem.d), 1.0)
        mask = torch.rand(
            int(n_candidates), int(self.problem.d),
            dtype=torch.double, device=center.device, generator=generator,
        ) <= probability
        empty = torch.where(mask.sum(dim=1) == 0)[0]
        if len(empty):
            columns = torch.randint(
                0, int(self.problem.d), (len(empty),),
                device=center.device, generator=generator)
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
            fork_devices = []
            if self._torch_device.type == "cuda":
                fork_devices = [
                    self._torch_device.index
                    if self._torch_device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
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
            torch.zeros(
                int(self.problem.d), dtype=torch.double,
                device=self._torch_device),
            torch.ones(
                int(self.problem.d), dtype=torch.double,
                device=self._torch_device),
        ])

    def _optimize_saas_acquisition(self, acqf):
        try:
            with self._deterministic_torch_stage("saas_acquisition_optimize"):
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
            x = self._discrete_saas_acquisition_fallback(acqf)
            self._saas_discrete_candidate_fallback_count += 1
        return x

    def _discrete_saas_acquisition_fallback(self, acqf):
        """Maximize the same acquisition over unseen integer Sobol policies.

        Continuous acquisition maximization can legitimately round to an
        already evaluated integer policy. Failing the whole benchmark in that
        case is an adapter artifact, not a property of SAASBO. This fallback
        keeps the fitted model and acquisition unchanged and only replaces
        continuous optimization by an audited finite integer candidate set.
        """

        seed = self._stage_seed("saas_discrete_candidate_fallback")
        pool_size = max(128, min(512, 4 * int(self.config.raw_samples)))
        engine = SobolEngine(
            dimension=int(self.problem.d),
            scramble=True,
            seed=int(seed),
        )
        continuous = engine.draw(pool_size).to(
            dtype=torch.double, device=self._torch_device)
        seen = {x for x, _ in self.history}
        lo, hi = self.problem.int_bounds()
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        scale = np.maximum(hi - lo, 1.0)
        integer_rows = []
        normalized_rows = []
        accepted = set()
        for row in continuous.detach().cpu().numpy():
            point = tuple(self.problem.continuous_to_int(row))
            if point in seen or point in accepted:
                continue
            accepted.add(point)
            integer_rows.append(point)
            normalized_rows.append(
                (np.asarray(point, dtype=float) - lo) / scale)
        if not integer_rows:
            self._candidate_failures += 1
            raise RuntimeError(
                "SAASBO discrete acquisition fallback found no unseen policy")
        values = []
        rows = torch.as_tensor(
            np.asarray(normalized_rows),
            dtype=torch.double,
            device=self._torch_device,
        )
        with self._deterministic_torch_stage(
            "saas_discrete_candidate_fallback_score"
        ), torch.no_grad():
            for chunk in torch.split(rows, 8, dim=0):
                value = acqf(chunk.unsqueeze(-2))
                values.append(value.reshape(-1).detach().cpu())
        scores = torch.cat(values).numpy()
        finite = np.isfinite(scores)
        if not np.any(finite):
            self._candidate_failures += 1
            raise RuntimeError(
                "SAASBO discrete acquisition fallback returned no finite score")
        best = int(np.argmax(np.where(finite, scores, -np.inf)))
        return integer_rows[best]

    def _saas_candidate(self):
        with _wall_time_limit(self.config.timeout_sec):
            train_X, train_obj, train_con = self._training_tensors(
                active_only=False)
            obj_model, con_model = self._saas_models_for_history()
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
            candidate = self._optimize_saas_acquisition(acqf)
            del acqf, sampler
            if self.config.saas_constrained:
                del model
            if (
                self._torch_device.type == "cuda"
                and (len(self.history) + 1) % 16 == 0
            ):
                gc.collect()
                torch.cuda.empty_cache()
            return candidate

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
            obj_model, con_model = self._saas_models_for_history()
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
        ], dtype=float), dtype=torch.double, device=self._torch_device)
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

    def _constraint_posterior_for_points(self, points):
        """Return this baseline's latent chance-margin posterior."""

        con_model = self._last_models[1]
        if con_model is None:
            raise RuntimeError(
                "terminal shortlist requires a fitted constraint model")
        X = torch.as_tensor(np.asarray([
            self.problem.normalize(point) for point in points
        ], dtype=float), dtype=torch.double, device=self._torch_device)
        with torch.no_grad():
            posterior = con_model.posterior(X)
            mean = getattr(
                posterior, "mixture_mean", posterior.mean
            ).reshape(-1).detach().cpu().numpy()
            variance = getattr(
                posterior, "mixture_variance", posterior.variance
            ).reshape(-1).detach().cpu().numpy()
        variance = np.maximum(np.asarray(variance, dtype=float), 1e-14)
        mean = np.asarray(mean, dtype=float)
        return {
            "chance_margin_mean": mean,
            "chance_margin_epistemic_variance": variance,
            "probability_violation": norm.cdf(
                mean / np.sqrt(variance)),
        }

    def _terminal_posterior_for_points(self, points):
        """Return objective minimization and chance-margin posteriors."""

        obj_model = self._last_models[0]
        if obj_model is None:
            raise RuntimeError(
                "terminal shortlist requires a fitted objective model")
        X = torch.as_tensor(np.asarray([
            self.problem.normalize(point) for point in points
        ], dtype=float), dtype=torch.double, device=self._torch_device)
        with torch.no_grad():
            posterior = obj_model.posterior(X)
            maximization_mean = getattr(
                posterior, "mixture_mean", posterior.mean
            ).reshape(-1).detach().cpu().numpy()
        constraint = self._constraint_posterior_for_points(points)
        return {
            "objective_mean": -np.asarray(
                maximization_mean, dtype=float),
            **constraint,
        }

    def terminal_verification_shortlist(
        self,
        primary,
        *,
        probability_slack=0.05,
        require_provider=True,
        shortlist_mode="posterior_primary_safe_interior",
        shortlist_size=2,
        maximum_violation_probability=0.5,
    ):
        """Freeze a BoTorch-posterior shortlist without target truth."""

        primary = tuple(int(value) for value in primary)
        normalized_mode = str(
            shortlist_mode
        ).strip().lower().replace("-", "_")
        if normalized_mode == "posterior_objective_challenger_then_safe":
            observed = []
            seen = set()
            for point, _ in self.history:
                point = tuple(int(value) for value in point)
                if point not in seen:
                    seen.add(point)
                    observed.append(point)
            if primary not in seen:
                observed.append(primary)
            posterior = self._terminal_posterior_for_points(observed)
            shortlist, _ = build_verification_aware_shortlist(
                self.problem,
                primary,
                observed,
                posterior["objective_mean"],
                posterior["probability_violation"],
                shortlist_size=int(shortlist_size),
                maximum_violation_probability=float(
                    maximum_violation_probability),
                probability_slack=float(probability_slack),
                support_selection_mode="diverse",
                require_provider=require_provider,
                selector_posterior=(
                    "botorch_latent_chance_margin_posterior"),
                candidate_universe=(
                    "frozen_observed_history_plus_search_recommendation"),
            )
            return shortlist
        if normalized_mode != "posterior_primary_safe_interior":
            raise ValueError(
                "unknown BoTorch terminal shortlist mode")

        initial = []
        seen = set()
        for point, _ in self.history[: int(self.config.n0)]:
            point = tuple(int(value) for value in point)
            if point not in seen:
                seen.add(point)
                initial.append(point)
        posterior = self._constraint_posterior_for_points(initial)
        support = select_posterior_safe_interior(
            self.problem,
            primary,
            initial,
            posterior["probability_violation"],
            probability_slack=probability_slack,
            require_provider=require_provider,
        )
        return [
            {
                "shortlist_position": 1,
                "shortlist_role": "posterior_bayes_primary",
                "posterior_rank": 1,
                "point": list(primary),
                "point_fingerprint": integer_design_fingerprint([primary]),
                "selector_posterior": (
                    "botorch_latent_chance_margin_posterior"),
                "target_labels_used": False,
                "target_oracle_used": False,
                "verification_samples_used": False,
            },
            {
                "shortlist_position": 2,
                "shortlist_role": "posterior_safe_interior_diversified",
                "posterior_rank": None,
                "point": list(map(int, support["point"])),
                "point_fingerprint": integer_design_fingerprint([
                    support["point"]]),
                "selector_posterior": (
                    "botorch_latent_chance_margin_posterior"),
                **{
                    key: value
                    for key, value in support.items()
                    if key != "point"
                },
            },
        ]

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
            "feasible_regret": (
                max(0.0, float(regret)) if true_margin <= 0.0 else None
            ),
            "constraint_violation": max(0.0, float(true_margin)),
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
        self._progress_timing.append((current, elapsed))
        completed_here = max(1, current - int(self._progress_start_unit))
        eta_sec = (elapsed / float(completed_here)) * float(
            max(0, total - current))
        eta_model = "current_run_average"
        saas_schedule = self._normalized_saas_refit_schedule()
        saas_cap = max(
            0, int(getattr(self.config, "saas_refit_max_history", 0)))
        capped_conditioning_phase = (
            self.config.method == "botorch_saasbo"
            and saas_schedule != "every_iteration"
            and saas_cap > 0
            and current >= saas_cap
        )
        if (
            capped_conditioning_phase
            and int(self._progress_start_unit) >= saas_cap
            and len(self._progress_timing) < 2
        ):
            # A resumed run first rebuilds the capped hyperposterior. That
            # one-off cost is not a per-iteration rate and must not be
            # multiplied by every remaining conditioned update.
            return
        if capped_conditioning_phase and len(self._progress_timing) >= 2:
            recent_points = self._progress_timing[-13:]
            rates = []
            for left, right in zip(recent_points, recent_points[1:]):
                delta_units = int(right[0]) - int(left[0])
                delta_sec = float(right[1]) - float(left[1])
                if delta_units > 0 and delta_sec > 0:
                    rates.append(delta_sec / float(delta_units))
            if rates:
                ordered_rates = sorted(rates)
                rolling_rate = ordered_rates[len(ordered_rates) // 2]
                eta_sec = rolling_rate * float(max(0, total - current))
                eta_model = "rolling_condition_cost"
        if (
            self.config.method == "botorch_saasbo"
            and saas_schedule == "every_iteration"
            and len(self._progress_timing) >= 12
        ):
            points = [
                point for point in self._progress_timing
                if point[0] >= 0.25 * current
            ]
            if len(points) < 12:
                points = self._progress_timing
            stride = max(2, len(points) // 24)
            rates = []
            for start in range(0, len(points) - stride, stride):
                left = points[start]
                right = points[min(len(points) - 1, start + stride)]
                delta_units = right[0] - left[0]
                delta_sec = right[1] - left[1]
                if delta_units > 0 and delta_sec > 0:
                    rates.append((
                        (left[0] + right[0]) / 2.0,
                        delta_sec / float(delta_units),
                    ))
            if len(rates) >= 4:
                mean_x = sum(x for x, _ in rates) / len(rates)
                mean_y = sum(y for _, y in rates) / len(rates)
                variance_x = sum((x - mean_x) ** 2 for x, _ in rates)
                if variance_x > 0:
                    slope = max(0.0, sum(
                        (x - mean_x) * (y - mean_y) for x, y in rates
                    ) / variance_x)
                    intercept = mean_y - slope * mean_x
                    recent = sorted(y for _, y in rates[-min(5, len(rates)):])
                    recent_rate = recent[len(recent) // 2]
                    current_rate = max(
                        1e-9, recent_rate, intercept + slope * current)
                    slope = min(
                        slope,
                        2.0 * current_rate / max(1.0, float(current)),
                    )
                    remaining = float(max(0, total - current))
                    eta_sec = (
                        current_rate * remaining
                        + 0.5 * slope * remaining * remaining
                    )
                    eta_model = "growing_iter_cost"
        fidelity_label = (
            "botorch-canonical"
            if (
                self.config.method != "botorch_saasbo"
                or self._normalized_saas_refit_schedule()
                == "every_iteration"
            )
            else "botorch-periodic-hyperposterior"
        )
        print(
            f"Iter {current}/{total} [{fidelity_label}] "
            f"label={self._progress_label()} Time: {elapsed:.1f}s "
            f"ETA {eta_sec:.1f}s eta_model={eta_model}",
            flush=True,
        )

    def run(
        self,
        *,
        freeze_terminal_shortlist=False,
        terminal_probability_slack=0.05,
        terminal_require_provider=True,
        terminal_shortlist_mode="posterior_primary_safe_interior",
        terminal_shortlist_size=2,
        terminal_maximum_violation_probability=0.5,
    ):
        t_start = time.time()
        self._progress_start_unit = len(self.history)
        self._progress_timing = []
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
        frozen_terminal_shortlist = None
        if freeze_terminal_shortlist:
            frozen_terminal_shortlist = self.terminal_verification_shortlist(
                x_best,
                probability_slack=terminal_probability_slack,
                require_provider=terminal_require_provider,
                shortlist_mode=terminal_shortlist_mode,
                shortlist_size=terminal_shortlist_size,
                maximum_violation_probability=(
                    terminal_maximum_violation_probability),
            )
        result = self._evaluate_recommendation(x_best)
        runtime = botorch_runtime_fingerprint(self._torch_device)
        runtime["torch_device"] = str(
            self._last_models[0].train_inputs[0].device
            if self._last_models[0] is not None else "cpu")
        result.update({
            "method": self.config.method,
            "backend": "botorch",
            **posterior,
            "frozen_terminal_shortlist": frozen_terminal_shortlist,
            "terminal_shortlist_frozen_before_truth_metrics": bool(
                freeze_terminal_shortlist),
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
            "stochastic_schedule": {
                "kind": "per_iteration_stage_seed_v1",
                "base_seed": int(self.config.seed),
                "resume_replays_inflight_stage": True,
            },
            "saas_constrained": bool(self.config.saas_constrained),
            "saas_parallel_models": bool(self._use_parallel_saas_models()),
            "saas_parallel_fit_count": int(self._saas_parallel_fit_count),
            "saas_parallel_failures": int(self._saas_parallel_failures),
            "saas_parallel_last_error": str(self._saas_parallel_last_error),
            "saas_parallel_threads_per_model": int(
                self._saas_parallel_threads()),
            "saas_refit_schedule": self._normalized_saas_refit_schedule(),
            "saas_refit_interval": int(self.config.saas_refit_interval),
            "saas_refit_growth_factor": float(
                self.config.saas_refit_growth_factor),
            "saas_refit_max_history": int(
                getattr(self.config, "saas_refit_max_history", 0)),
            "saas_full_refit_count": int(self._saas_full_refit_count),
            "saas_condition_count": int(self._saas_condition_count),
            "saas_condition_failures": int(self._saas_condition_failures),
            "saas_condition_last_error": str(
                self._saas_condition_last_error),
            "saas_discrete_candidate_fallback_count": int(
                self._saas_discrete_candidate_fallback_count),
            "saas_resume_rebuild_history_size": int(
                self._saas_resume_rebuild_history_size),
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
                    else (
                        "saas_fully_bayesian_nuts_constrained_qlogei"
                        if self._normalized_saas_refit_schedule()
                        == "every_iteration"
                        else (
                            "saas_fully_bayesian_periodic_capped_"
                            "hyperposterior_constrained_qlogei"
                            if int(getattr(
                                self.config,
                                "saas_refit_max_history",
                                0,
                            )) > 0
                            else (
                                "saas_fully_bayesian_periodic_"
                                "hyperposterior_constrained_qlogei"
                            )
                        )
                    )
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
                "hyperposterior_refit_schedule": (
                    self._normalized_saas_refit_schedule()),
                "hyperposterior_refit_interval": int(
                    self.config.saas_refit_interval),
                "hyperposterior_refit_growth_factor": float(
                    self.config.saas_refit_growth_factor),
                "hyperposterior_refit_max_history": int(
                    getattr(self.config, "saas_refit_max_history", 0)),
                "posterior_conditions_on_every_observation": True,
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
