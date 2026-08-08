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

from baselines.transfer_archive import (
    FrozenTransferArchive,
    dimension_equivariant_profile_features,
    resample_normalized_profiles,
)
from core.designs import (
    common_sobol_integer_design,
    integer_design_fingerprint,
)
from core.terminal_verification import (
    build_verification_aware_shortlist,
    select_posterior_safe_interior,
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
    initial_points: tuple[tuple[int, ...], ...] | None = None
    implementation: str = "paper_core"
    source_train_steps: int = 200
    target_finetune_steps: int = 40
    checkpoint_path: str | None = None
    checkpoint_resume: bool = True
    progress_logging: bool = False
    progress_label: str = ""
    source_dimension_adapter: str = "none"
    source_coordinate_max_frequency: int = 8
    source_coordinate_frequency_penalty: float = 0.10

    def __post_init__(self):
        self.method = str(self.method).lower()
        if self.method not in TRANSFER_METHODS:
            raise ValueError(f"unknown transfer BO method {self.method!r}")
        if self.implementation not in {"paper_core", "official"}:
            raise ValueError("implementation must be paper_core or official")
        if self.N < 1 or self.n0 < 1 or self.n0 > self.N:
            raise ValueError("transfer BO requires 1 <= n0 <= N")
        if self.initial_design not in {
            "common_sobol",
            "source_informed",
            "native_source_sequential",
        }:
            raise ValueError(
                "initial_design must be common_sobol, source_informed, or "
                "native_source_sequential")
        if self.initial_points is not None:
            self.initial_points = tuple(
                tuple(map(int, point)) for point in self.initial_points)
        if self.initial_design in {
            "common_sobol", "native_source_sequential",
        }:
            if self.initial_points is not None:
                raise ValueError(
                    f"{self.initial_design} must not receive explicit points")
        elif self.initial_points is None:
            raise ValueError(
                "source_informed requires an explicit frozen initial design")
        elif len(self.initial_points) != self.n0:
            raise ValueError("source_informed design must contain exactly n0 points")
        elif len(set(self.initial_points)) != self.n0:
            raise ValueError("source_informed design points must be unique")
        if self.source_dimension_adapter not in {
            "none",
            "ordered_dct_quadratic",
        }:
            raise ValueError(
                "source_dimension_adapter must be none or "
                "ordered_dct_quadratic"
            )
        if self.source_coordinate_max_frequency < 0:
            raise ValueError(
                "source_coordinate_max_frequency cannot be negative")
        if self.source_coordinate_frequency_penalty < 0.0:
            raise ValueError(
                "source_coordinate_frequency_penalty cannot be negative")


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
        self._progress_started_at = time.time()
        self._progress_phase_started_at = self._progress_started_at
        self._progress_phase_start_done = 0
        self.problem = problem
        self.config = config
        expected_dimension = (
            problem.d
            if config.source_dimension_adapter == "none"
            else None
        )
        self.archive = archive.validate(
            expected_dimension=expected_dimension)
        self.source_dimension = int(self.archive.tasks[0].X.shape[1])
        self.model_input_dimension = int(self._model_profiles(
            self.archive.tasks[0].X[:1]).shape[1])
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
        self._validate_initial_points()
        self._fit_source_archive()
        self.candidate_points, self.candidate_origins = self._candidate_pool()

    def _validate_initial_points(self):
        points = self.config.initial_points
        if points is None:
            return
        for point in points:
            if len(point) != int(self.problem.d):
                raise ValueError(
                    "source_informed point dimension does not match target")
            normalized = np.asarray(
                self.problem.normalize(point), dtype=float)
            if normalized.shape != (int(self.problem.d),):
                raise ValueError("target returned an invalid normalized point")
            if not np.all(np.isfinite(normalized)):
                raise ValueError("source_informed point contains nonfinite values")
            roundtrip = tuple(map(
                int, self.problem.continuous_to_int(normalized)))
            if roundtrip != tuple(point):
                raise ValueError("source_informed point is outside target bounds")

    def _fit_source_archive(self):
        self._progress_phase_started_at = time.time()
        roles = (
            ("objective", self.objective_model),
            ("constraint_mean", self.constraint_model),
            ("log_variance", self.risk_model),
        )
        for index, (role, model) in enumerate(roles):
            self._emit_progress(
                "source_model_start",
                phase="source_training",
                done=index,
                total=len(roles),
                role=role,
            )
            tasks = scalar_tasks_from_archive(self.archive, role)
            tasks = [
                ScalarTaskData(
                    name=task.name,
                    X=self._model_profiles(task.X),
                    y=task.y,
                    noise=task.noise,
                )
                for task in tasks
            ]
            model.meta_fit(tasks)
            self._emit_progress(
                "source_model_done",
                phase="source_training",
                done=index + 1,
                total=len(roles),
                role=role,
            )

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

        if self.config.initial_points is not None:
            for point in self.config.initial_points:
                append_point(point, "source_informed_pool")

        sobol_initial = common_sobol_integer_design(
            self.problem, self.config.n0, self.config.seed)
        for point in sobol_initial:
            append_point(point, "common_sobol_pool")

        source = np.vstack([task.X for task in self.archive.tasks])
        if source.shape[1] != int(self.problem.d):
            source = resample_normalized_profiles(
                source, int(self.problem.d))
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
        return common_sobol_integer_design(
            self.problem, self.config.n0, self.config.seed)

    def _initial_design(self):
        if self.config.initial_design == "source_informed":
            return list(self.config.initial_points)
        if self.config.initial_design == "native_source_sequential":
            return []
        return self._common_initial_design()

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

    def _model_profiles(self, profiles):
        values = np.asarray(profiles, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if self.config.source_dimension_adapter == "none":
            return values
        return dimension_equivariant_profile_features(
            values,
            max_frequency=self.config.source_coordinate_max_frequency,
            frequency_penalty=(
                self.config.source_coordinate_frequency_penalty),
        )

    def _adapt_models(self):
        X, objective, constraint, log_variance = self._arrays()
        nominal = max(float(self.problem.sigma_level) ** 2, 1e-8)
        observation_noise = np.full(len(objective), nominal, dtype=float)
        risk_noise = np.full(len(objective), 2.0, dtype=float)
        model_X = self._model_profiles(X)
        self.objective_model.adapt(model_X, objective, observation_noise)
        self.constraint_model.adapt(
            model_X, constraint, observation_noise)
        self.risk_model.adapt(model_X, log_variance, risk_noise)

    def _posterior(self, profiles):
        profiles = self._model_profiles(profiles)
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
        z_alpha = float(norm.ppf(1.0 - float(self.problem.alpha)))
        central_variance = np.exp(np.clip(
            log_variance_mean,
            -30.0,
            20.0,
        ))
        nominal_chance_margin = (
            constraint_mean + z_alpha * np.sqrt(central_variance)
        )
        # Delta-method posterior uncertainty of
        # m_g + z_alpha * exp(log(v_C) / 2). The source mean and risk heads
        # are fitted independently, so the covariance term is zero under the
        # declared transfer-model contract.
        log_variance_derivative = (
            0.5 * z_alpha * np.sqrt(central_variance)
        )
        chance_margin_epistemic_variance = np.maximum(
            constraint_variance
            + log_variance_derivative ** 2
            * np.maximum(log_variance_variance, 0.0),
            1e-12,
        )
        probability_violation = norm.cdf(
            nominal_chance_margin
            / np.sqrt(chance_margin_epistemic_variance)
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
            "nominal_chance_margin": np.asarray(nominal_chance_margin),
            "chance_margin_epistemic_variance": np.asarray(
                chance_margin_epistemic_variance),
            "probability_violation": np.asarray(probability_violation),
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
            model_profiles = self._model_profiles(profiles)
            scores = np.asarray(self.objective_model.acquisition_scores(
                model_profiles,
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
        model_profile = self._model_profiles(profile.reshape(1, -1))
        constraint_mean_before, constraint_variance_before = (
            self.constraint_model.predict(model_profile))
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
            "source_dimension_adapter": str(
                self.config.source_dimension_adapter),
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
        if state.get("source_dimension_adapter", "none") != (
            self.config.source_dimension_adapter
        ):
            raise ValueError("checkpoint source dimension adapter changed")
        self.history = list(state.get("history", []))
        if len(self.history) > self.config.N:
            raise ValueError("checkpoint exceeds requested target budget")
        self.rng.bit_generator.state = state["rng_state"]
        self._adapt_models()

    def _emit_progress(self, event, *, phase="target_online", done=None,
                       total=None, role=None):
        if not self.config.progress_logging:
            return
        now = time.time()
        done = int(len(self.history) if done is None else done)
        total = int(self.config.N if total is None else total)
        phase_elapsed = max(0.0, now - self._progress_phase_started_at)
        phase_done = max(0, done - int(self._progress_phase_start_done))
        eta_seconds = None
        if phase_done > 0 and total > done:
            eta_seconds = (
                phase_elapsed / float(phase_done) * float(total - done))
        elif done >= total:
            eta_seconds = 0.0
        payload = {
            "kind": event,
            "label": self.config.progress_label,
            "done": done,
            "total": total,
            "method": self.config.method,
            "phase": phase,
            "elapsed_s": max(0.0, now - self._progress_started_at),
            "phase_elapsed_s": phase_elapsed,
        }
        if eta_seconds is not None:
            payload["eta_seconds"] = float(eta_seconds)
        if role is not None:
            payload["role"] = str(role)
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
        """Freeze a method-specific posterior shortlist without target truth."""

        primary = tuple(int(value) for value in primary)
        normalized_mode = str(
            shortlist_mode
        ).strip().lower().replace("-", "_")
        if normalized_mode == "posterior_objective_challenger_then_safe":
            observed = []
            seen = set()
            for row in self.history:
                point = tuple(int(value) for value in row["x"])
                if point not in seen:
                    seen.add(point)
                    observed.append(point)
            if primary not in seen:
                observed.append(primary)
            profiles = np.vstack([
                np.asarray(self.problem.normalize(point), dtype=float)
                for point in observed
            ])
            posterior = self._posterior(profiles)
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
                    "transfer_method_specific_delta_chance_margin"),
                candidate_universe=(
                    "frozen_observed_history_plus_search_recommendation"),
            )
            return shortlist
        if normalized_mode != "posterior_primary_safe_interior":
            raise ValueError(
                "unknown transfer terminal shortlist mode")

        initial = []
        seen = set()
        for row in self.history[: int(self.config.n0)]:
            point = tuple(int(value) for value in row["x"])
            if point not in seen:
                seen.add(point)
                initial.append(point)
        profiles = np.vstack([
            np.asarray(self.problem.normalize(point), dtype=float)
            for point in initial
        ])
        posterior = self._posterior(profiles)
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
                    "transfer_method_specific_delta_chance_margin"),
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
                    "transfer_method_specific_delta_chance_margin"),
                **{
                    key: value
                    for key, value in support.items()
                    if key != "point"
                },
            },
        ]

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
        started = time.time()
        self._resume()
        self._progress_phase_started_at = time.time()
        self._progress_phase_start_done = len(self.history)
        initial = self._initial_design()
        initial_origin = (
            "source_informed_pool"
            if self.config.initial_design == "source_informed"
            else "common_sobol_pool"
        )
        initial_reason = (
            "frozen_source_informed_initial_design"
            if self.config.initial_design == "source_informed"
            else "common_sobol_initial_design"
        )
        native_initialization = bool(
            self.config.initial_design == "native_source_sequential")
        if native_initialization and not self.history:
            self._adapt_models()
        while len(self.history) < self.config.N:
            if native_initialization and len(self.history) < self.config.n0:
                point, selection = self._select_candidate()
                selection = {
                    **selection,
                    "selection_reason": (
                        "native_source_sequential__"
                        + str(selection["selection_reason"])
                    ),
                    "native_source_initialization": True,
                    "source_scored_atlas_used": False,
                }
            elif len(self.history) < self.config.n0:
                point = initial[len(self.history)]
                selection = {
                    "selection_reason": initial_reason,
                    "candidate_origin": initial_origin,
                    "certified_pool_count": None,
                    "candidate_pool_count": len(self.candidate_points),
                }
            else:
                point, selection = self._select_candidate()
            self._observe(point, selection)
            self._save_checkpoint()
            self._emit_progress("target_call_done")
        if native_initialization:
            initial = [
                tuple(map(int, row["x"]))
                for row in self.history[: int(self.config.n0)]
            ]
        recommended, posterior = self._recommend()
        frozen_terminal_shortlist = None
        if freeze_terminal_shortlist:
            frozen_terminal_shortlist = self.terminal_verification_shortlist(
                recommended,
                probability_slack=terminal_probability_slack,
                require_provider=terminal_require_provider,
                shortlist_mode=terminal_shortlist_mode,
                shortlist_size=terminal_shortlist_size,
                maximum_violation_probability=(
                    terminal_maximum_violation_probability),
            )
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
        z_alpha = float(norm.ppf(1.0 - float(self.problem.alpha)))
        initial_truth_rows = []
        for point in initial:
            initial_objective = float(self.problem.true_objective(point))
            initial_mean = float(self.problem.true_constraint_mean(point))
            initial_sigma = float(self.problem.true_sigma(point)[1])
            initial_margin = (
                initial_mean + z_alpha * initial_sigma
                - float(self.problem.tau)
            )
            initial_truth_rows.append({
                "objective": initial_objective,
                "chance_margin": float(initial_margin),
                "true_feasible": bool(initial_margin <= 0.0),
                "feasible_regret": (
                    max(0.0, initial_objective - float(optimum))
                    if initial_margin <= 0.0 else None
                ),
            })
        initial_feasible_regrets = [
            float(row["feasible_regret"])
            for row in initial_truth_rows
            if row["feasible_regret"] is not None
        ]
        initial_truth_audit = {
            "computed_after_recommendation": True,
            "used_for_selection": False,
            "n": int(len(initial_truth_rows)),
            "true_feasible_count": int(sum(
                row["true_feasible"] for row in initial_truth_rows)),
            "true_feasible_rate": float(np.mean([
                row["true_feasible"] for row in initial_truth_rows
            ])),
            "true_min_chance_margin": float(min(
                row["chance_margin"] for row in initial_truth_rows)),
            "best_true_feasible_regret": (
                float(min(initial_feasible_regrets))
                if initial_feasible_regrets else None
            ),
        }
        final_feasible_regret = (
            max(0.0, true_objective - float(optimum))
            if true_margin <= 0.0 else None
        )
        initial_truth_audit["final_improves_initial_best"] = bool(
            final_feasible_regret is not None
            and initial_feasible_regrets
            and final_feasible_regret < min(initial_feasible_regrets) - 1e-12
        )
        diagnostics = {
            "objective": self.objective_model.diagnostics(),
            "constraint_mean": self.constraint_model.diagnostics(),
            "log_variance": self.risk_model.diagnostics(),
        }
        source_contract = self.archive.information_contract()
        source_contract.update({
            "source_policy_dimension": int(self.source_dimension),
            "target_policy_dimension": int(self.problem.d),
            "source_dimension_adapter": str(
                self.config.source_dimension_adapter),
            "model_input_dimension": int(self.model_input_dimension),
            "dimension_adapter_uses_target_labels": False,
            "dimension_adapter_uses_target_oracle": False,
        })
        return {
            "method": self.config.method,
            "implementation": self.config.implementation,
            "implementation_fidelity": diagnostics["objective"].get(
                "implementation_fidelity", "paper_core_reimplementation"),
            "source_archive_fingerprint": self.archive.fingerprint,
            "source_information_contract": source_contract,
            "target_information_contract": {
                "dimension": int(self.problem.d),
                "n0": int(self.config.n0),
                "target_calls": int(self.config.N),
                "one_simulator_call_per_observation": True,
                "initial_design": self.config.initial_design,
                "initial_design_fingerprint": integer_design_fingerprint(
                    initial),
                "initial_points": [list(map(int, point)) for point in initial],
                "source_informed_initial_design": bool(
                    self.config.initial_design == "source_informed"),
                "source_scored_atlas_initial_design": bool(
                    self.config.initial_design == "source_informed"),
                "native_source_sequential_initialization": (
                    native_initialization),
                "source_dimension_adapter": str(
                    self.config.source_dimension_adapter),
                "model_input_dimension": int(self.model_input_dimension),
                "target_oracle_used_for_selection": False,
                "target_true_sigma_used_for_selection": False,
                "post_run_truth_used_for_metrics_only": True,
            },
            "initial_truth_audit": initial_truth_audit,
            "adaptation_contract": METHOD_CONTRACTS[self.config.method],
            "model_diagnostics": diagnostics,
            "x_recommended": list(map(int, recommended)),
            "posterior": posterior,
            "frozen_terminal_shortlist": frozen_terminal_shortlist,
            "terminal_shortlist_frozen_before_truth_metrics": bool(
                freeze_terminal_shortlist),
            "true_objective": true_objective,
            "true_constraint_mean": true_constraint_mean,
            "true_constraint_sigma": true_sigma,
            "true_chance_margin": float(true_margin),
            "true_feasible": bool(true_margin <= 0.0),
            "feasible_regret": (
                final_feasible_regret
            ),
            "constraint_violation": max(0.0, float(true_margin)),
            "n_simulations": int(len(self.history)),
            "history": self.history,
            "wall_time_sec": float(time.time() - started),
        }
