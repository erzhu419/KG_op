"""Chance-constrained transfer BO under one frozen-archive contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.stats import norm, qmc

from baselines.transfer_archive import FrozenTransferArchive
from core.designs import (
    common_sobol_integer_design,
    integer_design_fingerprint,
)
from baselines.transfer_learned_models import (
    PaperCoreFSBOSurrogate,
    PaperCoreMALIBOSurrogate,
    PaperCoreMetaBOSurrogate,
    PaperCoreSafeFPACOHSurrogate,
)
from baselines.transfer_models import (
    HyperBOSurrogate,
    MultiTaskGPSurrogate,
    RGPESurrogate,
    ScalarTaskData,
    StackedHierarchicalGPSurrogate,
)


TRANSFER_METHODS = (
    "safe_fpacoh_cbo",
    "rgpe_cbo",
    "stacked_transfer_gp_cbo",
    "mtgp_cbo",
    "fsbo_cbo",
    "hyperbo_cbo",
    "metabo_cbo",
    "malibo_cbo",
)


METHOD_CONTRACTS = {
    "safe_fpacoh_cbo": {
        "source_mechanism": "function_space_hyperposterior",
        "target_adaptation": "posterior_conditioning",
    },
    "rgpe_cbo": {
        "source_mechanism": "source_gp_experts",
        "target_adaptation": "expert_reweighting_and_posterior_conditioning",
    },
    "stacked_transfer_gp_cbo": {
        "source_mechanism": "hierarchical_source_prior_mean",
        "target_adaptation": "source_target_discrepancy_posterior",
    },
    "mtgp_cbo": {
        "source_mechanism": "intrinsic_coregionalization",
        "target_adaptation": "joint_multitask_posterior",
    },
    "fsbo_cbo": {
        "source_mechanism": "deep_kernel_meta_training",
        "target_adaptation": "end_to_end_gradient_finetuning",
    },
    "hyperbo_cbo": {
        "source_mechanism": "pretrained_gp_prior",
        "target_adaptation": "posterior_conditioning",
    },
    "metabo_cbo": {
        "source_mechanism": "source_trained_acquisition_policy",
        "target_adaptation": "frozen_policy_with_target_posterior_state",
    },
    "malibo_cbo": {
        "source_mechanism": "meta_learned_utility_representation",
        "target_adaptation": "bayesian_utility_head_adaptation",
    },
}


@dataclass
class TransferBOConfig:
    method: str
    N: int = 20
    n0: int = 10
    seed: int = 0
    candidate_pool_size: int = 1024
    beta_g: float = 2.0
    beta_risk: float = 2.0
    initial_design: str = "common_sobol"
    implementation: str = "paper_core"
    source_train_steps: int = 200
    target_finetune_steps: int = 40
    checkpoint_path: str | None = None
    checkpoint_resume: bool = True
    progress_logging: bool = False
    progress_label: str = ""

    def __post_init__(self):
        self.method = str(self.method).lower()
        if self.method not in TRANSFER_METHODS:
            raise ValueError(f"unknown transfer BO method {self.method!r}")
        if self.implementation not in {"paper_core", "official"}:
            raise ValueError("implementation must be paper_core or official")
        if self.N < 1 or self.n0 < 1 or self.n0 > self.N:
            raise ValueError("transfer BO requires 1 <= n0 <= N")
        if self.initial_design != "common_sobol":
            raise ValueError("paper comparison requires common_sobol initial design")


def scalar_tasks_from_archive(archive, output):
    """Expose exactly the same scalar source rows to every method."""

    if output not in {"objective", "constraint_mean", "log_variance"}:
        raise ValueError(f"unknown source output {output!r}")
    rows = []
    for task in archive.tasks:
        if output == "objective":
            y = task.Y_mean[:, 0]
            noise = task.mean_observation_variance[:, 0]
        elif output == "constraint_mean":
            y = task.Y_mean[:, 1] - float(task.tau)
            noise = task.mean_observation_variance[:, 1]
        else:
            y = np.log(np.maximum(task.constraint_sigma ** 2, 1e-12))
            replicate_count = np.asarray([
                len(values) for values in task.Y_replicates
            ], dtype=float)
            # Delta-method variance of log(sample variance).  This uses only
            # the disclosed replicate count, never analytic source sigma.
            noise = 2.0 / np.maximum(replicate_count - 1.0, 1.0)
        rows.append(ScalarTaskData(
            name=task.name,
            X=np.asarray(task.X, dtype=float).copy(),
            y=np.asarray(y, dtype=float).copy(),
            noise=np.maximum(np.asarray(noise, dtype=float), 1e-8),
        ))
    return rows


def _paper_core_model(method, *, role, config, seed):
    if method == "safe_fpacoh_cbo":
        return PaperCoreSafeFPACOHSurrogate()
    if method == "rgpe_cbo":
        return RGPESurrogate(seed=seed, n_weight_samples=128)
    if method == "stacked_transfer_gp_cbo":
        return StackedHierarchicalGPSurrogate()
    if method == "mtgp_cbo":
        return MultiTaskGPSurrogate()
    if method == "fsbo_cbo":
        return PaperCoreFSBOSurrogate(
            seed=seed,
            source_steps=config.source_train_steps,
            target_steps=config.target_finetune_steps,
        )
    if method == "hyperbo_cbo":
        return HyperBOSurrogate(kernel="rbf")
    if method == "metabo_cbo" and role == "objective":
        return PaperCoreMetaBOSurrogate()
    if method == "malibo_cbo" and role == "objective":
        return PaperCoreMALIBOSurrogate(seed=seed)
    # MetaBO and MALIBO do not define a safe heteroscedastic constraint
    # model.  Their CBO extensions use the same frozen HyperBO constraint and
    # risk prior, which is reported explicitly in diagnostics.
    return HyperBOSurrogate(kernel="rbf")


def _official_model(method, *, role, config, seed):
    try:
        from baselines.transfer_external_models import official_model
    except ImportError as exc:  # pragma: no cover - dependency audit path
        raise RuntimeError(
            "official transfer adapters are unavailable; paper-grade runs "
            "must not silently fall back to paper_core"
        ) from exc
    return official_model(method, role=role, config=config, seed=seed)


def make_transfer_model(method, *, role, config, seed):
    if config.implementation == "official":
        return _official_model(
            method, role=role, config=config, seed=seed)
    return _paper_core_model(
        method, role=role, config=config, seed=seed)


def _expected_improvement(mean, variance, incumbent):
    mean = np.asarray(mean, dtype=float)
    std = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12))
    improvement = float(incumbent) - mean
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def _atomic_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


class TransferConstrainedBO:
    """One-call-at-a-time transfer CBO with a shared heteroscedastic bound."""

    def __init__(self, problem, archive: FrozenTransferArchive, config):
        self.problem = problem
        self.archive = archive.validate(expected_dimension=problem.d)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.objective_model = make_transfer_model(
            config.method, role="objective", config=config, seed=config.seed)
        self.constraint_model = make_transfer_model(
            config.method,
            role="constraint_mean",
            config=config,
            seed=config.seed + 1009,
        )
        self.risk_model = make_transfer_model(
            config.method,
            role="log_variance",
            config=config,
            seed=config.seed + 2017,
        )
        self.history = []
        self._fit_source_archive()
        self.candidate_points, self.candidate_origins = self._candidate_pool()

    def _fit_source_archive(self):
        self.objective_model.meta_fit(scalar_tasks_from_archive(
            self.archive, "objective"))
        self.constraint_model.meta_fit(scalar_tasks_from_archive(
            self.archive, "constraint_mean"))
        self.risk_model.meta_fit(scalar_tasks_from_archive(
            self.archive, "log_variance"))

    def _candidate_pool(self):
        requested = max(int(self.config.candidate_pool_size), self.config.N)
        exponent = int(math.ceil(math.log2(max(requested, 2))))
        sobol = qmc.Sobol(
            d=self.problem.d,
            scramble=True,
            seed=self.config.seed + 7919,
        ).random_base2(exponent)
        points = []
        point_origins = []
        seen = set()

        def append_point(point, origin):
            point = tuple(map(int, point))
            if point in seen:
                return
            seen.add(point)
            points.append(point)
            point_origins.append(origin)

        initial = common_sobol_integer_design(
            self.problem,
            self.config.n0,
            self.config.seed,
        )
        for point in initial:
            append_point(point, "common_sobol_pool")

        source = np.vstack([task.X for task in self.archive.tasks])
        for profile in source:
            append_point(
                self.problem.continuous_to_int(profile),
                "source_archive_profile",
            )
        for profile in sobol:
            append_point(
                self.problem.continuous_to_int(profile),
                "common_sobol_pool",
            )
        if len(points) < self.config.N:
            raise RuntimeError("common candidate pool collapsed below N")
        return points, point_origins

    def _common_initial_design(self):
        return [
            point for point, origin in zip(
                self.candidate_points, self.candidate_origins
            )
            if origin == "common_sobol_pool"
        ][:self.config.n0]

    def _arrays(self):
        if not self.history:
            empty = np.empty((0, self.problem.d), dtype=float)
            return empty, np.empty(0), np.empty(0), np.empty(0)
        X = np.vstack([row["x_normalized"] for row in self.history])
        objective = np.asarray([
            row["observation"][0] for row in self.history], dtype=float)
        constraint = np.asarray([
            row["constraint_centered"] for row in self.history], dtype=float)
        log_variance = np.asarray([
            row["prequential_log_variance_target"]
            for row in self.history
        ], dtype=float)
        return X, objective, constraint, log_variance

    def _adapt_models(self):
        X, objective, constraint, log_variance = self._arrays()
        nominal = max(float(self.problem.sigma_level) ** 2, 1e-8)
        observation_noise = np.full(len(objective), nominal, dtype=float)
        risk_noise = np.full(len(objective), 2.0, dtype=float)
        self.objective_model.adapt(X, objective, observation_noise)
        self.constraint_model.adapt(X, constraint, observation_noise)
        self.risk_model.adapt(X, log_variance, risk_noise)

    def _posterior(self, profiles):
        objective_mean, objective_variance = self.objective_model.predict(
            profiles)
        constraint_mean, constraint_variance = self.constraint_model.predict(
            profiles)
        log_variance_mean, log_variance_variance = self.risk_model.predict(
            profiles)
        v_plus = np.exp(np.clip(
            log_variance_mean
            + np.sqrt(max(float(self.config.beta_risk), 0.0))
            * np.sqrt(np.maximum(log_variance_variance, 0.0)),
            -30.0,
            20.0,
        ))
        chance_bound = (
            constraint_mean
            + np.sqrt(max(float(self.config.beta_g), 0.0))
            * np.sqrt(np.maximum(constraint_variance, 0.0))
            + norm.ppf(1.0 - float(self.problem.alpha)) * np.sqrt(v_plus)
        )
        return {
            "objective_mean": np.asarray(objective_mean),
            "objective_variance": np.asarray(objective_variance),
            "constraint_mean": np.asarray(constraint_mean),
            "constraint_variance": np.asarray(constraint_variance),
            "log_variance_mean": np.asarray(log_variance_mean),
            "log_variance_variance": np.asarray(log_variance_variance),
            "v_c_plus": np.asarray(v_plus),
            "chance_bound": np.asarray(chance_bound),
        }

    def _available_pool(self, include_observed=False):
        observed = {tuple(row["x"]) for row in self.history}
        indices = [
            index for index, point in enumerate(self.candidate_points)
            if include_observed or point not in observed
        ]
        points = [self.candidate_points[index] for index in indices]
        profiles = np.vstack([
            np.asarray(self.problem.normalize(point), dtype=float)
            for point in points
        ])
        origins = [self.candidate_origins[index] for index in indices]
        return indices, points, profiles, origins

    def _select_candidate(self):
        _, points, profiles, origins = self._available_pool()
        posterior = self._posterior(profiles)
        certified = posterior["chance_bound"] <= 0.0
        incumbent = min(
            (float(row["observation"][0]) for row in self.history),
            default=float(np.min(posterior["objective_mean"])),
        )
        if hasattr(self.objective_model, "acquisition_scores"):
            scores = np.asarray(self.objective_model.acquisition_scores(
                profiles,
                incumbent,
                len(self.history) / max(self.config.N, 1),
            ), dtype=float)
        else:
            scores = _expected_improvement(
                posterior["objective_mean"],
                posterior["objective_variance"],
                incumbent,
            )
        if np.any(certified):
            eligible = np.flatnonzero(certified)
            local = int(eligible[np.argmax(scores[eligible])])
            reason = "certified_acquisition"
        else:
            # Safety-margin lexicographic fallback is common to all methods.
            order = np.lexsort((
                posterior["objective_mean"],
                posterior["chance_bound"],
            ))
            local = int(order[0])
            reason = "minimum_posterior_chance_bound"
        audit = {
            key: float(np.asarray(values)[local])
            for key, values in posterior.items()
        }
        audit.update({
            "acquisition_score": float(scores[local]),
            "selection_reason": reason,
            "candidate_origin": origins[local],
            "certified_pool_count": int(np.sum(certified)),
            "candidate_pool_count": int(len(points)),
        })
        return points[local], audit

    def _observe(self, point, selection):
        profile = np.asarray(self.problem.normalize(point), dtype=float)
        constraint_mean_before, constraint_variance_before = (
            self.constraint_model.predict(profile.reshape(1, -1)))
        observation = np.asarray(
            self.problem.simulate(point, self.rng), dtype=float)
        centered = float(observation[1] - self.problem.tau)
        residual_variance = max(
            (centered - float(constraint_mean_before[0])) ** 2
            - float(constraint_variance_before[0]),
            (0.10 * float(self.problem.sigma_level)) ** 2,
            1e-12,
        )
        row = {
            "iteration": int(len(self.history)),
            "x": list(map(int, point)),
            "x_normalized": profile.tolist(),
            "observation": observation.tolist(),
            "constraint_centered": centered,
            "prequential_constraint_mean": float(constraint_mean_before[0]),
            "prequential_constraint_variance": float(
                constraint_variance_before[0]),
            "prequential_log_variance_target": float(np.log(
                residual_variance)),
            "target_true_sigma_used": False,
            **selection,
        }
        self.history.append(row)
        self._adapt_models()

    def _checkpoint_state(self):
        return {
            "schema_version": 1,
            "method": self.config.method,
            "archive_fingerprint": self.archive.fingerprint,
            "N": int(self.config.N),
            "n0": int(self.config.n0),
            "history": self.history,
            "rng_state": self.rng.bit_generator.state,
        }

    def _save_checkpoint(self):
        if self.config.checkpoint_path:
            _atomic_pickle(self.config.checkpoint_path, self._checkpoint_state())

    def _resume(self):
        path = Path(self.config.checkpoint_path or "")
        if not self.config.checkpoint_resume or not path.is_file():
            return
        with path.open("rb") as stream:
            state = pickle.load(stream)
        if state.get("archive_fingerprint") != self.archive.fingerprint:
            raise ValueError("checkpoint source archive fingerprint changed")
        if state.get("method") != self.config.method:
            raise ValueError("checkpoint transfer method changed")
        self.history = list(state.get("history", []))
        if len(self.history) > self.config.N:
            raise ValueError("checkpoint exceeds requested target budget")
        self.rng.bit_generator.state = state["rng_state"]
        self._adapt_models()

    def _emit_progress(self, event):
        if not self.config.progress_logging:
            return
        payload = {
            "kind": event,
            "label": self.config.progress_label,
            "done": int(len(self.history)),
            "total": int(self.config.N),
            "method": self.config.method,
        }
        print("SCOLHKG_PROGRESS " + json.dumps(payload), flush=True)

    def _recommend(self):
        _, points, profiles, _ = self._available_pool(include_observed=True)
        posterior = self._posterior(profiles)
        certified = posterior["chance_bound"] <= 0.0
        if np.any(certified):
            eligible = np.flatnonzero(certified)
            selected = int(eligible[np.argmin(
                posterior["objective_mean"][eligible])])
            reason = "minimum_posterior_mean_among_certified"
        else:
            selected = int(np.lexsort((
                posterior["objective_mean"],
                posterior["chance_bound"],
            ))[0])
            reason = "minimum_posterior_chance_bound"
        point = points[selected]
        return point, {
            key: float(np.asarray(value)[selected])
            for key, value in posterior.items()
        } | {"recommendation_reason": reason}

    def run(self):
        started = time.time()
        self._resume()
        initial = self._common_initial_design()
        while len(self.history) < self.config.N:
            if len(self.history) < self.config.n0:
                point = initial[len(self.history)]
                selection = {
                    "selection_reason": "common_sobol_initial_design",
                    "candidate_origin": "common_sobol_pool",
                    "certified_pool_count": None,
                    "candidate_pool_count": len(self.candidate_points),
                }
            else:
                point, selection = self._select_candidate()
            self._observe(point, selection)
            self._save_checkpoint()
            self._emit_progress("target_call_done")
        recommended, posterior = self._recommend()
        true_objective = float(self.problem.true_objective(recommended))
        true_constraint_mean = float(
            self.problem.true_constraint_mean(recommended))
        true_sigma = float(self.problem.true_sigma(recommended)[1])
        true_margin = (
            true_constraint_mean
            + norm.ppf(1.0 - float(self.problem.alpha)) * true_sigma
            - float(self.problem.tau)
        )
        _, optimum = self.problem.true_best_feasible()
        diagnostics = {
            "objective": self.objective_model.diagnostics(),
            "constraint_mean": self.constraint_model.diagnostics(),
            "log_variance": self.risk_model.diagnostics(),
        }
        return {
            "method": self.config.method,
            "implementation": self.config.implementation,
            "implementation_fidelity": diagnostics["objective"].get(
                "implementation_fidelity", "paper_core_reimplementation"),
            "source_archive_fingerprint": self.archive.fingerprint,
            "source_information_contract": self.archive.information_contract(),
            "target_information_contract": {
                "dimension": int(self.problem.d),
                "n0": int(self.config.n0),
                "target_calls": int(self.config.N),
                "one_simulator_call_per_observation": True,
                "initial_design": self.config.initial_design,
                "initial_design_fingerprint": integer_design_fingerprint(
                    initial),
                "target_oracle_used_for_selection": False,
                "target_true_sigma_used_for_selection": False,
                "post_run_truth_used_for_metrics_only": True,
            },
            "adaptation_contract": METHOD_CONTRACTS[self.config.method],
            "model_diagnostics": diagnostics,
            "x_recommended": list(map(int, recommended)),
            "posterior": posterior,
            "true_objective": true_objective,
            "true_constraint_mean": true_constraint_mean,
            "true_constraint_sigma": true_sigma,
            "true_chance_margin": float(true_margin),
            "true_feasible": bool(true_margin <= 0.0),
            "feasible_regret": (
                max(0.0, true_objective - float(optimum))
                if true_margin <= 0.0 else None
            ),
            "constraint_violation": max(0.0, float(true_margin)),
            "n_simulations": int(len(self.history)),
            "history": self.history,
            "wall_time_sec": float(time.time() - started),
        }
