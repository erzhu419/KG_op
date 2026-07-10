"""Single-objective chance-constrained OLH-KG / SC-OLH-KG algorithm."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.stats import norm

from acquisition.olhkg import OLHKGAcquisition
from core.certification import conservative_chance_margin
from core.candidates import (
    axis_candidates,
    axis_landmark_candidates,
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
    ContrastivePolicyEncoder,
    GraphLaplacianEncoder,
    HybridSSLPolicyEncoder,
    KernelManifoldEncoder,
    LowFrequencyOrthogonalSparsePolicyEncoder,
    MaskedTrajectoryEncoder,
    NextRiskEncoder,
    PCAManifoldEncoder,
    SelfSupervisedPolicyStateEncoder,
    SmallTransformerEncoder,
    StateCoupledFeatureMap,
    SyntheticPolicyStateEncoder,
)
from representation.llm_structural_prior import LLMStructuralPriorAdvisor
from representation.manifold import ManifoldRiskDecomposer
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
    variance_mode: str = "factor"
    lambda_feas: float = 0.25
    lambda_var: float = 0.25
    lambda_mean: float = 0.10
    lambda_constraint_epistemic: float = 0.0
    lambda_coupling: float = 0.05
    beta_g: float = 2.0
    certification_mode: str = "theory"
    coupling_safety_z: float = 0.5
    coupling_gate_temperature: float = 0.25
    recommendation_safety_z: float = 0.5
    recommendation_noise_floor_scale: float = 1.0
    recommendation_infeasible_penalty: float = 5.0
    recommendation_infeasible_strategy: str = "penalty"
    recommendation_observed_fallback: bool = False
    observed_incumbent_margin_scale: float = -0.5
    recommendation_calibration: bool = True
    recommendation_calibration_scope: str = "refinement"
    recommendation_calibration_ridge: float = 1e-6
    recommendation_calibration_min_obs: int = 8
    recommendation_calibration_max_leverage: float = 0.0
    recommendation_calibration_max_theory_margin: float = 0.0
    certification_calibration: bool = False
    certification_calibration_min_obs: int = 8
    certification_calibration_ridge: float = 1e-6
    certification_calibration_noise_floor_scale: float = 0.5
    certification_calibration_beta: float = 2.0
    certification_calibration_policy: str = "guarded"
    certification_calibration_max_leverage: float = 10.0
    certification_calibration_max_theory_margin: float = 0.25
    certification_calibration_raise_delta: float = 0.10
    calibration_standardize_features: bool = False
    recommend_observed_only: bool = False
    recommendation_axis_oracle: bool = True
    use_problem_initial_samples: bool = True
    use_boundary_initial_samples: bool = True
    use_recommendation_refinement: bool = True
    recommendation_axis_candidate_count: int = -1
    use_state_coupling: bool = True
    use_state_basis: bool = True
    state_basis_mode: str = "raw+state"
    constraint_state_basis_mode: str = ""
    raw_basis_dim: int = -1
    raw_projection_seed: int = 314159
    numeric_backend: str = "numpy"
    numeric_backend_device: str = "auto"
    torch_dtype: str = "float64"
    torch_min_rows: int = 128
    use_manifold_hvd_features: bool = True
    encoder_kind: str = "synthetic"
    encoder_latent_dim: int = 8
    encoder_fit_pool_size: int = 512
    lf_os_max_library_size: int = 30
    lf_os_low_frequency_components: int = 8
    lf_os_max_active: int = 8
    lf_os_graph_neighbors: int = 12
    lf_os_residual_floor_scale: float = 0.05
    lf_os_use_problem_state_anchor: bool = True
    acquisition_mode: str = "exact_mc"
    exact_kg_mc_samples: int = 8
    exact_kg_jobs: int = 1
    exact_kg_use_score: bool = False
    exact_kg_blend: float = 0.0
    constraint_uncertain_candidate_count: int = 0
    constraint_uncertain_pool_size: int = 300
    constraint_uncertain_state_pool_fraction: float = 0.25
    constraint_uncertain_use_calibration: bool = False
    constraint_epistemic_margin_softening: float = 3.0
    replication_candidate_count: int = 0
    replication_max_per_solution: int = 5
    replication_margin_softening: float = 3.0
    safe_interior_candidate_count: int = 0
    safe_interior_pool_size: int = 300
    safe_interior_margin: float = 0.0
    observed_neighbor_candidate_count: int = 0
    observed_neighbor_radius: float = 0.08
    observed_neighbor_safe_margin_scale: float = 1.0
    recommendation_slack_initial: float = 0.0
    recommendation_slack_decay: str = "sqrt"
    use_source_recommendation_slack: bool = False
    source_mean_prior_fallback: bool = False
    source_mean_prior_z: float = 1.0
    source_mean_prior_margin_tol: float = 0.0
    truth_pool_diagnostics: bool = False
    truth_pool_good_regret: float = 0.05
    truth_pool_max_candidates: int = 0
    llm_prior_enabled: bool = False
    llm_prior_base_url: str = "https://ruoli.dev"
    llm_prior_model: str = "gpt-5.4-mini"
    llm_prior_api_key_env: str = "SCOLHKG_LLM_API_KEY"
    llm_prior_candidate_count: int = 0
    llm_prior_inverse_pool_size: int = 1024
    llm_prior_interval: int = 5
    llm_prior_min_obs: int = 8
    llm_prior_timeout_sec: float = 30.0
    llm_prior_gate_floor: float = 0.05
    llm_prior_max_observations: int = 24
    checkpoint_dir: str = ""
    checkpoint_resume: bool = False
    checkpoint_interval: int = 1
    checkpoint_keep_last: int = 3
    progress_logging: bool = False
    progress_label: str = ""
    progress_units_per_iteration: int = 100
    progress_exact_updates: int = 10
    eval_pool_size: int = 500
    evaluate_interval: int = 5
    seed: int = 123


class SingleOLHKGAlgorithm:
    """Minimal but complete single-objective OLH-KG implementation."""

    def __init__(self, problem, config: SingleOLHKGConfig | None = None):
        self.problem = problem
        self.config = config or SingleOLHKGConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.rec_rng = np.random.default_rng(int(self.config.seed) + 1_000_003)

        self.encoder = self._build_encoder()
        provider_available = False
        if hasattr(problem, "cumulative_risk_provider_status"):
            try:
                provider_available = (
                    problem.cumulative_risk_provider_status().get("status")
                    == "available"
                )
            except AttributeError:
                provider_available = False
        explicit_representation_basis = str(
            self.config.state_basis_mode or ""
        ).lower() in {
            "manifold",
            "raw+manifold",
            "raw_plus_manifold",
        }
        basis_map = None
        constraint_basis_map = None
        direct_meta_basis = bool(
            getattr(problem, "prefer_direct_gpr_basis", False)
            and hasattr(problem, "gpr_basis_map")
        )
        if direct_meta_basis:
            basis_map = problem.gpr_basis_map(output_index=0)
            constraint_basis_map = problem.gpr_basis_map(output_index=1)
        elif self.config.use_state_basis and (
            provider_available or explicit_representation_basis
        ):
            basis_map = StateCoupledFeatureMap(
                problem,
                self.encoder,
                mode=self.config.state_basis_mode,
                raw_basis_dim=self.config.raw_basis_dim,
                raw_projection_seed=self.config.raw_projection_seed,
            )
        elif hasattr(problem, "gpr_basis_map"):
            basis_map = problem.gpr_basis_map()
        elif self.config.use_state_basis:
            basis_map = StateCoupledFeatureMap(
                problem,
                self.encoder,
                mode=self.config.state_basis_mode,
                raw_basis_dim=self.config.raw_basis_dim,
                raw_projection_seed=self.config.raw_projection_seed,
            )
        if constraint_basis_map is None:
            constraint_basis_map = basis_map
            constraint_basis_mode = str(
                self.config.constraint_state_basis_mode or "").strip()
            if constraint_basis_mode:
                constraint_basis_map = StateCoupledFeatureMap(
                    problem,
                    self.encoder,
                    mode=constraint_basis_mode,
                    raw_basis_dim=self.config.raw_basis_dim,
                    raw_projection_seed=self.config.raw_projection_seed,
                )
        self._attach_representation_to_problem()
        self.gpr = [
            ParametricGPR(
                problem.d,
                self.config.lambda_i,
                self.config.prior_var,
                normalize_func=problem.normalize,
                basis_map=basis_map if output_index == 0 else constraint_basis_map,
                numeric_backend=self.config.numeric_backend,
                numeric_backend_device=self.config.numeric_backend_device,
                torch_dtype=self.config.torch_dtype,
                torch_min_rows=self.config.torch_min_rows,
            )
            for output_index in range(2)
        ]
        self.variance_model = OrthogonalHVD(
            mode=self.config.variance_mode,
            n_outputs=2,
            floor=1e-8,
        )
        self.acquisition = OLHKGAcquisition(
            lambda_feas=self.config.lambda_feas,
            lambda_var=self.config.lambda_var,
            lambda_mean=self.config.lambda_mean,
            lambda_constraint_epistemic=self.config.lambda_constraint_epistemic,
            lambda_coupling=self.config.lambda_coupling,
            constraint_epistemic_margin_softening=(
                self.config.constraint_epistemic_margin_softening),
            beta_g=self.config.beta_g,
            certification_mode=self.config.certification_mode,
            coupling_safety_z=self.config.coupling_safety_z,
            coupling_gate_temperature=self.config.coupling_gate_temperature,
            encoder=self.encoder,
        )
        self.llm_prior = None
        if self.config.llm_prior_enabled:
            self.llm_prior = LLMStructuralPriorAdvisor(
                base_url=self.config.llm_prior_base_url,
                model=self.config.llm_prior_model,
                api_key_env=self.config.llm_prior_api_key_env,
                timeout_sec=self.config.llm_prior_timeout_sec,
                min_obs=self.config.llm_prior_min_obs,
                gate_floor=self.config.llm_prior_gate_floor,
                max_observations=self.config.llm_prior_max_observations,
            )
        self._last_llm_prior_info = {
            "status": "disabled" if self.llm_prior is None else "not_called",
            "gate": 0.0,
            "n_regions": 0,
        }

        self.observations: dict[tuple[int, ...], list[np.ndarray]] = {}
        self.history: list[tuple[tuple[int, ...], np.ndarray]] = []
        self.iteration_log: list[dict] = []
        self.pre_sampling_log: dict | None = None
        self.final_log: dict | None = None
        self._true_best_feasible_cache = None

    def _build_encoder(self):
        needs_encoder = (
            self.config.use_state_coupling
            or self.config.use_state_basis
            or self.config.lambda_coupling > 0
        )
        if not needs_encoder:
            return None
        kind = str(self.config.encoder_kind or "synthetic").lower()
        trajectory_records = getattr(self.problem, "_scolhkg_trajectory_records", None)
        if kind in ("self_supervised", "masked", "contrastive"):
            return SelfSupervisedPolicyStateEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                mode="masked",
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind in ("transformer", "attention"):
            return SelfSupervisedPolicyStateEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                mode="transformer",
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind == "pca_manifold":
            return PCAManifoldEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind == "kernel_manifold":
            return KernelManifoldEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind in ("graph_laplacian", "diffusion_manifold", "graph_manifold"):
            return GraphLaplacianEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind in ("ssl_masked", "masked_trajectory"):
            return MaskedTrajectoryEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in ("ssl_contrastive", "contrastive_policy"):
            return ContrastivePolicyEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in ("ssl_next_risk", "next_risk"):
            return NextRiskEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in ("ssl_transformer", "small_transformer"):
            return SmallTransformerEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in ("ssl_hybrid", "hybrid_ssl", "contextual_manifold"):
            return HybridSSLPolicyEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in (
            "lf_os",
            "lf_orthogonal_sparse",
            "low_frequency_orthogonal_sparse",
            "orthogonal_sparse",
        ):
            return LowFrequencyOrthogonalSparsePolicyEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                max_library_size=self.config.lf_os_max_library_size,
                low_frequency_components=self.config.lf_os_low_frequency_components,
                max_active=self.config.lf_os_max_active,
                n_neighbors=self.config.lf_os_graph_neighbors,
                residual_floor_scale=self.config.lf_os_residual_floor_scale,
                use_problem_state_anchor=self.config.lf_os_use_problem_state_anchor,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        return SyntheticPolicyStateEncoder(self.problem)

    def _attach_representation_to_problem(self):
        if self.encoder is None:
            return
        kind = str(self.config.encoder_kind or "synthetic").lower()
        setattr(self.problem, "_scolhkg_representation_encoder", self.encoder)
        if (
            self.config.use_manifold_hvd_features
            and kind in {
                "pca_manifold",
                "kernel_manifold",
                "graph_laplacian",
                "diffusion_manifold",
                "graph_manifold",
                "ssl_masked",
                "ssl_contrastive",
                "ssl_next_risk",
                "ssl_transformer",
                "ssl_hybrid",
                "hybrid_ssl",
                "contextual_manifold",
                "small_transformer",
                "lf_os",
                "lf_orthogonal_sparse",
                "low_frequency_orthogonal_sparse",
                "orthogonal_sparse",
            }
        ):
            setattr(self.problem, "_scolhkg_use_manifold_hvd", True)
            setattr(
                self.problem,
                "_scolhkg_manifold_decomposer",
                ManifoldRiskDecomposer(self.encoder),
            )

    def _initial_samples(self):
        samples = []
        if self.config.use_problem_initial_samples and hasattr(
            self.problem, "initial_samples"
        ):
            samples.extend(self.problem.initial_samples(
                n=self.config.n0,
                rng=self.rng,
            ))
            samples = unique_candidates(samples)
        if len(samples) < self.config.n0 and self.config.use_boundary_initial_samples:
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

        for i in range(2):
            basis_map = getattr(self.gpr[i], "basis_map", None)
            if basis_map is not None and hasattr(
                basis_map, "fit_from_observations"
            ):
                basis_map.fit_from_observations(
                    self.observations,
                    output_index=i,
                )
            self._rebuild_gpr_from_history(i, replay_sequential=False)

        self.variance_model.initialize(
            samples, self.observations, self.gpr, self.problem)

    def _rebuild_gpr_from_history(self, output_index, replay_sequential=True):
        """Rebuild one GPR after its feature semantics change.

        The first ``n0`` records reproduce the original empirical-prior fit.
        Every later rank-one update is replayed with the observation variance
        recorded when that sample was selected.  Coefficients from an old
        basis are therefore never interpreted in a new basis.
        """

        output_index = int(output_index)
        model = self.gpr[output_index]
        n_initial = int(self.config.n0)
        if len(self.history) < n_initial:
            raise RuntimeError("cannot rebuild GPR before the initial history exists")
        initial_history = self.history[:n_initial]
        samples = [tuple(int(v) for v in x) for x, _ in initial_history]
        y_i = np.asarray([
            float(np.asarray(y, dtype=float)[output_index])
            for _, y in initial_history
        ])
        Phi = model.basis_matrix(samples)
        basis_map = getattr(model, "basis_map", None)
        try:
            if basis_map is not None and hasattr(
                basis_map, "initial_parametric_coefficients"
            ):
                beta = basis_map.initial_parametric_coefficients(Phi, y_i)
            else:
                beta = np.linalg.lstsq(Phi, y_i, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(Phi.shape[1], dtype=float)
        resid = y_i - Phi @ beta
        lambda_data = max(float(np.var(resid)), 1e-6)
        prior_var = max(float(np.var(beta)), 1e-6)
        adaptive_spec = None
        if basis_map is not None and hasattr(
            basis_map, "adaptive_sparsity_spec"
        ):
            adaptive_spec = basis_map.adaptive_sparsity_spec(
                self.observations)
        if adaptive_spec is not None:
            known_sigma = max(
                float(getattr(self.problem, "sigma_level", 0.0)),
                1e-3,
            )
            initial_noise = max(lambda_data, known_sigma ** 2)
            model.enable_adaptive_sparsity(
                adaptive_spec,
                samples,
                y_i,
                np.full(len(samples), initial_noise, dtype=float),
                deviation_variance=max(lambda_data, initial_noise),
            )
        else:
            if basis_map is not None and hasattr(
                basis_map, "apply_coefficient_prior"
            ):
                beta, prior_var = basis_map.apply_coefficient_prior(
                    beta, prior_var)
            model.set_parametric_prior(beta, lambda_data, prior_var)
            for x in samples:
                model.dimension_augment(x)

        replayed = 0
        if replay_sequential:
            sequential_history = self.history[n_initial:]
            if len(sequential_history) != len(self.iteration_log):
                raise RuntimeError(
                    "sequential history and iteration log disagree during GPR replay"
                )
            for (history_x, history_y), row in zip(
                sequential_history, self.iteration_log
            ):
                row_x = tuple(int(v) for v in row["x_selected"])
                history_x = tuple(int(v) for v in history_x)
                if row_x != history_x:
                    raise RuntimeError(
                        "iteration log candidate does not match GPR history"
                    )
                sigma2 = float(row["sigma2_before"][output_index])
                observed = float(np.asarray(history_y, dtype=float)[output_index])
                model.update(history_x, observed, sigma2)
                replayed += 1
        return {
            "initial_records": int(len(initial_history)),
            "replayed_updates": int(replayed),
            "basis_dim": int(model.p),
        }

    def _checkpoint_root(self):
        root = str(self.config.checkpoint_dir or "").strip()
        if not root:
            return None
        return Path(root)

    def _gpr_checkpoint_state(self, model):
        basis_map = getattr(model, "basis_map", None)
        return {
            "d": int(model.d),
            "p": int(model.p),
            "lambda_i": float(model.lambda_i),
            "a": np.asarray(model.a, dtype=float),
            "C": np.asarray(model.C, dtype=float),
            "sampled_set": [tuple(int(v) for v in x) for x in model.sampled_set],
            "sol_to_idx": {
                tuple(int(v) for v in key): int(value)
                for key, value in model.sol_to_idx.items()
            },
            "state_version": int(getattr(model, "_state_version", 0)),
            "adaptive_sparsity": copy.deepcopy(
                getattr(model, "_adaptive_sparsity", None)),
            "adaptive_records": copy.deepcopy(
                getattr(model, "_adaptive_records", [])),
            "adaptive_spec": copy.deepcopy(
                getattr(model, "_adaptive_spec", None)),
            "basis_runtime_state": (
                copy.deepcopy(basis_map.runtime_state())
                if basis_map is not None
                and hasattr(basis_map, "runtime_state")
                else None
            ),
        }

    def _restore_gpr_checkpoint_state(self, model, state):
        if int(state.get("d", model.d)) != int(model.d):
            raise ValueError("checkpoint GPR dimension does not match current problem")
        if int(state.get("p", model.p)) != int(model.p):
            raise ValueError("checkpoint basis dimension does not match current config")
        model.lambda_i = float(state["lambda_i"])
        model.a = np.asarray(state["a"], dtype=float).copy()
        model.C = np.asarray(state["C"], dtype=float).copy()
        model.sampled_set = [
            tuple(int(v) for v in row)
            for row in state.get("sampled_set", [])
        ]
        model.sol_to_idx = {
            tuple(int(v) for v in key): int(value)
            for key, value in state.get("sol_to_idx", {}).items()
        }
        model._adaptive_sparsity = copy.deepcopy(
            state.get("adaptive_sparsity"))
        model._adaptive_records = copy.deepcopy(
            state.get("adaptive_records", []))
        model._adaptive_spec = copy.deepcopy(state.get("adaptive_spec"))
        basis_state = state.get("basis_runtime_state")
        if (
            basis_state is not None
            and model.basis_map is not None
            and hasattr(model.basis_map, "load_runtime_state")
        ):
            model.basis_map.load_runtime_state(copy.deepcopy(basis_state))
        model._state_version = int(state.get("state_version", 0)) + 1
        model._torch_cache = {}

    def _clone_gpr_for_exact_kg(self, model):
        """Clone mutable GPR state without deep-copying simulator/provider handles."""
        clone = object.__new__(model.__class__)
        clone.__dict__ = model.__dict__.copy()
        clone.a = np.asarray(model.a, dtype=float).copy()
        clone.C = np.asarray(model.C, dtype=float).copy()
        clone.sampled_set = [
            tuple(int(v) for v in row)
            for row in getattr(model, "sampled_set", [])
        ]
        clone.sol_to_idx = {
            tuple(int(v) for v in key): int(value)
            for key, value in getattr(model, "sol_to_idx", {}).items()
        }
        clone._adaptive_sparsity = copy.deepcopy(
            getattr(model, "_adaptive_sparsity", None))
        clone._adaptive_records = copy.deepcopy(
            getattr(model, "_adaptive_records", []))
        clone._adaptive_spec = copy.deepcopy(
            getattr(model, "_adaptive_spec", None))
        if clone._adaptive_sparsity is not None and model.basis_map is not None:
            basis_clone = object.__new__(model.basis_map.__class__)
            basis_clone.__dict__ = model.basis_map.__dict__.copy()
            if hasattr(model.basis_map, "_adaptive_sparsity_diagnostics"):
                basis_clone._adaptive_sparsity_diagnostics = copy.deepcopy(
                    model.basis_map._adaptive_sparsity_diagnostics)
            clone.basis_map = basis_clone
        clone._torch_cache = {}
        return clone

    def _clone_variance_model_for_exact_kg(self):
        state = copy.deepcopy(self.variance_model.__getstate__())
        clone = object.__new__(self.variance_model.__class__)
        clone.__setstate__(state)
        clone._last_problem = self.problem
        return clone

    def _runtime_checkpoint_payload(self, next_stage_n, reason):
        return {
            "schema_version": 1,
            "reason": str(reason),
            "next_stage_n": int(next_stage_n),
            "saved_at": float(time.time()),
            "config": asdict(self.config),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "rec_rng_state": copy.deepcopy(self.rec_rng.bit_generator.state),
            "observations": copy.deepcopy(self.observations),
            "history": copy.deepcopy(self.history),
            "iteration_log": copy.deepcopy(self.iteration_log),
            "pre_sampling_log": copy.deepcopy(self.pre_sampling_log),
            "final_log": copy.deepcopy(self.final_log),
            "gpr": [self._gpr_checkpoint_state(model) for model in self.gpr],
            "variance_model": copy.deepcopy(self.variance_model.__getstate__()),
        }

    def _write_pickle_atomic(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _prune_checkpoints(self, root):
        keep = int(self.config.checkpoint_keep_last)
        if keep <= 0:
            return
        stage_files = sorted(root.glob("checkpoint_stage_*.pkl"))
        for path in stage_files[:-keep]:
            try:
                path.unlink()
            except OSError:
                pass

    def _save_checkpoint(self, next_stage_n, reason="iteration", force=False):
        root = self._checkpoint_root()
        if root is None:
            return None
        interval = max(1, int(self.config.checkpoint_interval))
        should_stage = (
            force
            or reason in {"initial", "final"}
            or int(next_stage_n) >= int(self.config.N)
            or int(next_stage_n) % interval == 0
        )
        payload = self._runtime_checkpoint_payload(next_stage_n, reason)
        latest_path = root / "checkpoint_latest.pkl"
        self._write_pickle_atomic(latest_path, payload)
        if should_stage:
            stage_path = root / f"checkpoint_stage_{int(next_stage_n):05d}.pkl"
            self._write_pickle_atomic(stage_path, payload)
            self._prune_checkpoints(root)
        return latest_path

    def _reattach_runtime_handles_after_checkpoint(self):
        if self.encoder is not None:
            if hasattr(self.encoder, "problem"):
                self.encoder.problem = self.problem
            if hasattr(self.encoder, "rng"):
                self.encoder.rng = self.rng
            raw_encoder = getattr(self.encoder, "raw_encoder", None)
            if raw_encoder is not None:
                if hasattr(raw_encoder, "problem"):
                    raw_encoder.problem = self.problem
                if hasattr(raw_encoder, "rng"):
                    raw_encoder.rng = self.rng
        self._attach_representation_to_problem()
        self.variance_model._last_problem = self.problem
        self.acquisition.encoder = self.encoder

    def _load_checkpoint_payload(self, payload):
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported SC-OLH-KG checkpoint schema")
        self.rng.bit_generator.state = payload["rng_state"]
        self.rec_rng.bit_generator.state = payload["rec_rng_state"]
        self.observations = copy.deepcopy(payload.get("observations", {}))
        self.history = copy.deepcopy(payload.get("history", []))
        self.iteration_log = copy.deepcopy(payload.get("iteration_log", []))
        self.pre_sampling_log = copy.deepcopy(payload.get("pre_sampling_log"))
        self.final_log = copy.deepcopy(payload.get("final_log"))
        for output_index, (model, state) in enumerate(
            zip(self.gpr, payload.get("gpr", []))
        ):
            basis_map = getattr(model, "basis_map", None)
            if basis_map is not None and hasattr(
                basis_map, "fit_from_observations"
            ):
                basis_map.fit_from_observations(
                    self.observations,
                    output_index=output_index,
                )
            self._restore_gpr_checkpoint_state(model, state)
        variance_state = copy.deepcopy(payload.get("variance_model"))
        if variance_state is None:
            raise ValueError("checkpoint is missing variance model state")
        self.variance_model.__setstate__(variance_state)
        self._reattach_runtime_handles_after_checkpoint()
        return int(payload.get("next_stage_n", self.config.n0))

    def _try_resume_from_checkpoint(self, verbose=False):
        root = self._checkpoint_root()
        if root is None or not self.config.checkpoint_resume:
            return None
        path = root if root.is_file() else root / "checkpoint_latest.pkl"
        if not path.exists():
            return None
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        next_stage_n = self._load_checkpoint_payload(payload)
        if verbose:
            print(
                f"resumed checkpoint={path} "
                f"next_stage_n={next_stage_n} "
                f"history={len(self.history)}"
            )
        return next_stage_n

    def _initialize_or_resume(self, verbose=False):
        resumed = self._try_resume_from_checkpoint(verbose=verbose)
        if resumed is not None:
            return resumed
        samples = self._initial_samples()
        t0 = time.time()
        self._fit_initial_belief(samples)
        self.pre_sampling_log = {
            "n0": self.config.n0,
            "samples": [list(map(int, x)) for x in samples],
            "time_sec": float(time.time() - t0),
            "variance": self.variance_model.diagnostics(),
            "meta_basis": (
                self.problem.meta_basis_diagnostics()
                if hasattr(self.problem, "meta_basis_diagnostics")
                else None
            ),
        }
        self._save_checkpoint(self.config.n0, reason="initial", force=True)
        return int(self.config.n0)

    def _refresh_sequential_basis(self):
        refresh = []
        for output_index, model in enumerate(self.gpr):
            basis_map = getattr(model, "basis_map", None)
            if (
                basis_map is None
                or not hasattr(basis_map, "should_refit_from_observations")
                or not basis_map.should_refit_from_observations(
                    self.observations)
            ):
                continue
            snapshot = self._gpr_checkpoint_state(model)
            before = (
                basis_map.runtime_state()
                if hasattr(basis_map, "runtime_state")
                else {"selected_basis": getattr(
                    basis_map, "selected_basis", None)}
            )
            try:
                basis_map.fit_from_observations(
                    self.observations,
                    output_index=output_index,
                )
                after = (
                    basis_map.runtime_state()
                    if hasattr(basis_map, "runtime_state")
                    else {"selected_basis": getattr(
                        basis_map, "selected_basis", None)}
                )
                changed = bool(
                    self._basis_semantic_signature(before)
                    != self._basis_semantic_signature(after)
                )
                rebuild = (
                    self._rebuild_gpr_from_history(
                        output_index, replay_sequential=True)
                    if changed
                    else {
                        "initial_records": 0,
                        "replayed_updates": 0,
                        "basis_dim": int(model.p),
                    }
                )
            except Exception:
                self._restore_gpr_checkpoint_state(model, snapshot)
                raise
            refresh.append({
                "output_index": int(output_index),
                "before_basis": before.get("selected_basis"),
                "after_basis": after.get("selected_basis"),
                "before_groups": before.get("selected_additive_groups", []),
                "after_groups": after.get("selected_additive_groups", []),
                "changed": changed,
                "gpr_rebuilt": changed,
                "replayed_updates": int(rebuild["replayed_updates"]),
                "rebuild_initial_records": int(rebuild["initial_records"]),
                "n_observations": int(len(self.observations)),
            })
        return refresh

    @staticmethod
    def _basis_semantic_signature(state):
        selected = str(state.get("selected_basis", ""))
        alignment = state.get("target_risk_alignment") or {}
        matrix_signature = b""
        if selected in {"risk_aligned_coordinate", "risk_aligned_spectral"}:
            matrix = alignment.get("matrix")
            if matrix is not None:
                matrix_signature = np.asarray(
                    matrix, dtype=np.float64).tobytes()
        return (
            selected,
            float(state.get("selected_parametric_ridge", 0.0)),
            tuple(int(value) for value in state.get(
                "selected_additive_groups", [])),
            str(state.get("additive_base_basis", "")),
            str(state.get("additive_bank_kind", "")),
            matrix_signature,
        )

    def _progress_enabled(self):
        return bool(getattr(self.config, "progress_logging", False))

    def _progress_label(self):
        label = str(getattr(self.config, "progress_label", "") or "").strip()
        if label:
            return label
        return f"seed={int(self.config.seed)}"

    def _progress_units_per_iteration(self):
        return max(1, int(getattr(self.config, "progress_units_per_iteration", 100)))

    def _progress_emit(
        self,
        *,
        n,
        frac,
        kind,
        started_at,
        run_started_at,
        extra="",
    ):
        if not self._progress_enabled():
            return
        units = self._progress_units_per_iteration()
        total_units = max(1, int(self.config.N) * units)
        current_units = int(round((float(n) + float(np.clip(frac, 0.0, 1.0))) * units))
        current_units = max(0, min(total_units, current_units))
        elapsed = max(0.0, time.time() - float(run_started_at))
        now = time.perf_counter()
        elapsed = max(0.0, now - float(run_started_at))
        step_elapsed = max(0.0, now - float(started_at))
        start_units = max(0, int(self.config.n0) * units)
        done_units = max(1, current_units - start_units)
        remaining_units = max(0, total_units - current_units)
        eta_sec = (elapsed / float(done_units)) * float(remaining_units)
        msg = (
            f"Step {current_units}/{total_units} [kg-inner] "
            f"kind={kind} label={self._progress_label()} "
            f"stage={int(n)}/{int(self.config.N)} "
            f"elapsed={elapsed:.1f}s step_elapsed={step_elapsed:.1f}s "
            f"ETA {eta_sec:.1f}s"
        )
        if extra:
            msg += f" {extra}"
        print(msg, flush=True)

    def _variance_lookup(self, i, x):
        return self.variance_model.predict_variance(i, x, self.problem)

    def _constraint_epistemic_lookup(self, x):
        return self.gpr[1].posterior_var(x)

    def _true_best_feasible_cached(self):
        if self._true_best_feasible_cache is None:
            self._true_best_feasible_cache = self.problem.true_best_feasible()
        return self._true_best_feasible_cache

    def _pilot_constraint_guard(self):
        if not hasattr(self.problem, "pilot_constraint_guard"):
            return 0.0
        try:
            return max(float(self.problem.pilot_constraint_guard()), 0.0)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _certification_result(self, mu_con, candidates, v_con=None):
        if v_con is None:
            v_con = self.variance_model.predict_certification_variance_many(
                1, candidates, self.problem)
        epistemic = self.gpr[1].posterior_var_many(candidates)
        return conservative_chance_margin(
            np.asarray(mu_con, dtype=float) + self._pilot_constraint_guard(),
            epistemic,
            v_con,
            tau=self.problem.tau,
            alpha=self.problem.alpha,
            beta_g=self.config.beta_g,
            mode=self.config.certification_mode,
        )

    def _axis_candidate_count(self):
        n_axis = int(self.config.axis_candidate_count)
        if n_axis >= 0:
            return n_axis
        return max(5, min(21, max(1, self.config.K1 // 2)))

    def _recommendation_axis_candidate_count(self):
        n_axis = int(self.config.recommendation_axis_candidate_count)
        if n_axis >= 0:
            return n_axis
        return self._axis_candidate_count()

    def _recommendation_refinement_candidates(self):
        if not self.config.use_recommendation_refinement:
            return []
        if not hasattr(self.problem, "recommendation_refinement_candidates"):
            return []
        return unique_candidates(self.problem.recommendation_refinement_candidates())

    def _recommendation_random_pool_size(self):
        if hasattr(self.problem, "recommendation_random_pool_size"):
            try:
                return max(0, int(self.problem.recommendation_random_pool_size()))
            except AttributeError:
                pass
        return max(0, int(self.config.eval_pool_size))

    def _recommendation_slack(self):
        slack = max(float(self.config.recommendation_slack_initial), 0.0)
        if self.config.use_source_recommendation_slack and hasattr(
            self.problem, "source_calibrated_recommendation_slack"
        ):
            try:
                slack = max(
                    slack,
                    float(self.problem.source_calibrated_recommendation_slack()),
                )
            except (AttributeError, TypeError, ValueError):
                pass
        if slack <= 0.0:
            return 0.0
        n_eff = max(1, len(self.history))
        decay = str(self.config.recommendation_slack_decay or "none").lower()
        if decay in ("sqrt", "sqrt_n"):
            return float(slack / np.sqrt(n_eff))
        if decay in ("linear", "n"):
            return float(slack / n_eff)
        return float(slack)

    def _safe_interior_candidates(self):
        n_safe = int(self.config.safe_interior_candidate_count)
        if n_safe <= 0:
            return []
        pool_size = max(n_safe, int(self.config.safe_interior_pool_size))
        pool = unique_candidates(random_candidates(self.problem, pool_size, self.rng))
        if not pool:
            return []
        try:
            mu_con = self.gpr[1].posterior_mean_many(pool)
            mu_obj = self.gpr[0].posterior_mean_many(pool)
            v_con = self.variance_model.predict_certification_variance_many(
                1, pool, self.problem)
            cert = self._certification_result(mu_con, pool, v_con)
        except Exception:
            return []
        margins = np.asarray(cert.margin, dtype=float) + self._recommendation_slack()
        eta = max(float(self.config.safe_interior_margin), 0.0)
        feasible = np.where(margins <= -eta)[0]
        if len(feasible):
            order = feasible[np.argsort(mu_obj[feasible])]
        else:
            order = np.lexsort((mu_obj, margins))
        return [pool[int(i)] for i in order[:n_safe]]

    def _observed_neighbor_candidates(self, n=None, rng=None):
        """Local, target-feedback-only candidates around empirical safe points.

        This is deliberately domain-generic: it uses only observed outputs and
        box-normalized policy perturbations.  It gives LODO runs a way to
        recover target-specific safe basins without calling hidden structural
        hooks such as target anchors or refinement grids.
        """
        n_target = (
            int(self.config.observed_neighbor_candidate_count)
            if n is None else int(n)
        )
        if n_target <= 0 or not self.observations:
            return []
        rng = rng or self.rng
        z_alpha = float(norm.ppf(1 - self.problem.alpha))
        sigma_floor = max(float(getattr(self.problem, "sigma_level", 0.0)), 1e-8)
        margin_limit = (
            float(self.config.observed_neighbor_safe_margin_scale)
            * sigma_floor
        )
        scored = []
        for x, ys in self.observations.items():
            y_bar = np.mean(np.asarray(ys, dtype=float), axis=0)
            margin = float(y_bar[1] + z_alpha * sigma_floor - self.problem.tau)
            obj = float(y_bar[0])
            # Prefer empirically safe points, then near-boundary points, then
            # low objective.  The margin gate is intentionally soft; in noisy
            # few-shot runs a slightly positive empirical margin can still be
            # useful for boundary-local exploration.
            safe_rank = 0 if margin <= margin_limit else 1
            scored.append((safe_rank, max(margin, 0.0), obj, tuple(int(v) for v in x)))
        if not scored:
            return []
        scored.sort()
        seeds = [row[3] for row in scored[: max(1, min(len(scored), 8))]]
        radius = max(float(self.config.observed_neighbor_radius), 0.0)
        if radius <= 0.0:
            return unique_candidates(seeds)[:n_target]
        rows = list(seeds)
        d = max(1, int(getattr(self.problem, "d", len(seeds[0]))))

        def add_from_z(z):
            try:
                rows.append(tuple(int(v) for v in self.problem.continuous_to_int(z)))
            except Exception:
                L = int(getattr(self.problem, "L", 100))
                rows.append(tuple(
                    int(np.clip(round(float(v) * L), 0, L))
                    for v in np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
                ))

        patterns = ("global", "tail", "third", "sparse")
        attempts = 0
        while len(unique_candidates(rows)) < n_target and attempts < 20 * n_target:
            attempts += 1
            seed = seeds[int(rng.integers(0, len(seeds)))]
            z = np.asarray(self.problem.normalize(seed), dtype=float).reshape(-1)
            if len(z) != d:
                d = len(z)
            mode = patterns[attempts % len(patterns)]
            cand = z.copy()
            if mode == "global":
                cand += rng.normal(0.0, radius, size=len(cand))
            elif mode == "tail" and len(cand) > 1:
                cand[1:] += rng.normal(0.0, radius)
                cand[0] += rng.normal(0.0, 0.5 * radius)
            elif mode == "third" and len(cand) > 2:
                thirds = np.array_split(np.arange(len(cand)), 3)
                block = thirds[int(rng.integers(0, len(thirds)))]
                cand[block] += rng.normal(0.0, radius)
            else:
                k = max(1, min(len(cand), int(np.ceil(np.sqrt(len(cand))))))
                idx = rng.choice(len(cand), size=k, replace=False)
                cand[idx] += rng.normal(0.0, radius, size=k)
            add_from_z(np.clip(cand, 0.0, 1.0))
        return unique_candidates(rows)[:n_target]

    def _constraint_uncertain_candidates(self):
        n_uncertain = int(self.config.constraint_uncertain_candidate_count)
        if n_uncertain <= 0:
            return []
        pool_size = max(n_uncertain, int(self.config.constraint_uncertain_pool_size))
        pool = []
        state_fraction = float(np.clip(
            self.config.constraint_uncertain_state_pool_fraction, 0.0, 1.0))
        if (
            state_fraction > 0.0
            and self.config.use_state_coupling
            and self.encoder is not None
        ):
            n_state_pool = int(round(pool_size * state_fraction))
            if n_state_pool > 0:
                try:
                    pool.extend(self.encoder.state_space_candidates(
                        n_anchors=max(1, n_state_pool),
                        inverse_pool_size=max(
                            self.config.state_inverse_pool_size,
                            n_state_pool,
                        ),
                        inverse_neighbors=max(
                            1,
                            int(self.config.state_inverse_neighbors),
                        ),
                        rng=self.rng,
                        observed=[x for x, _ in self.history],
                    ))
                except Exception:
                    pass
        n_random_pool = max(pool_size - len(pool), n_uncertain)
        pool.extend(random_candidates(self.problem, n_random_pool, self.rng))
        pool = unique_candidates(pool)
        if not pool:
            return []
        try:
            mu_con = self.gpr[1].posterior_mean_many(pool)
            epistemic = self.gpr[1].posterior_var_many(pool)
            v_con = self.variance_model.predict_certification_variance_many(
                1, pool, self.problem)
            cert = self._certification_result(mu_con, pool, v_con)
        except Exception:
            return []
        calibration = (
            self._calibrated_certification_result(pool, v_con)
            if self.config.constraint_uncertain_use_calibration
            else None
        )
        total_var = (
            np.maximum(cert.aleatoric_var, 1e-12)
            + max(float(cert.beta_g), 0.0) * np.maximum(cert.epistemic_var, 0.0)
        )
        margins = np.asarray(cert.margin, dtype=float)
        if calibration is not None:
            cal_epistemic = np.asarray(calibration["epistemic_var"], dtype=float)
            epistemic = np.maximum(np.asarray(epistemic, dtype=float), cal_epistemic)
            total_var = np.maximum(
                total_var,
                np.maximum(calibration["aleatoric_var"], 1e-12)
                + max(float(calibration["beta_g"]), 0.0) * np.maximum(cal_epistemic, 0.0),
            )
            cal_margins = np.asarray(calibration["margin"], dtype=float)
            margins = np.where(np.abs(cal_margins) < np.abs(margins), cal_margins, margins)
        sig = np.sqrt(np.maximum(total_var, 1e-12))
        softened = (
            max(float(self.config.constraint_epistemic_margin_softening), 1e-8)
            * sig
        )
        boundary_weight = np.exp(
            -0.5
            * (margins / np.maximum(softened, 1e-8)) ** 2
        )
        score = np.sqrt(np.maximum(epistemic, 0.0)) * (0.25 + boundary_weight)
        if not np.any(np.isfinite(score)):
            return []
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf))
        return [pool[int(i)] for i in order[:n_uncertain]]

    def _replication_candidates(self):
        n_replicate = max(0, int(self.config.replication_candidate_count))
        if n_replicate == 0 or not self.observations:
            return []
        max_per_solution = max(
            1, int(self.config.replication_max_per_solution))
        pool = [
            tuple(int(v) for v in x)
            for x, values in self.observations.items()
            if len(values) < max_per_solution
        ]
        if not pool:
            return []
        mu_con = self.gpr[1].posterior_mean_many(pool)
        epistemic = self.gpr[1].posterior_var_many(pool)
        v_con = self.variance_model.predict_certification_variance_many(
            1, pool, self.problem)
        cert = self._certification_result(mu_con, pool, v_con)
        total_var = (
            np.maximum(cert.aleatoric_var, 1e-12)
            + max(float(cert.beta_g), 0.0)
            * np.maximum(cert.epistemic_var, 0.0)
        )
        sig = np.sqrt(np.maximum(total_var, 1e-12))
        softened = max(
            float(self.config.replication_margin_softening), 1e-8) * sig
        boundary = np.exp(
            -0.5 * (np.asarray(cert.margin) / softened) ** 2)
        observation_noise = np.maximum(cert.aleatoric_var, 1e-12)
        epistemic = np.maximum(np.asarray(epistemic, dtype=float), 0.0)
        expected_reduction = epistemic ** 2 / np.maximum(
            epistemic + observation_noise, 1e-12)
        nominal_scale = np.sqrt(np.maximum(observation_noise, 1e-12))
        nominal_safe = 1.0 / (1.0 + np.exp(np.clip(
            np.asarray(mu_con, dtype=float) / nominal_scale,
            -60.0,
            60.0,
        )))
        score = expected_reduction * (0.25 + boundary) * (0.5 + nominal_safe)
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf))
        return [pool[int(position)] for position in order[:n_replicate]]

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
        n_axis = self._axis_candidate_count()
        add(axis_landmark_candidates(self.problem, n_axis, self.rng), "axis_landmark")
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
        add(self._constraint_uncertain_candidates(), "constraint_uncertain")
        add(self._replication_candidates(), "replication")
        add(self._safe_interior_candidates(), "safe_interior")
        add(self._observed_neighbor_candidates(), "observed_neighbor")
        self._last_llm_prior_info = {
            "status": "disabled" if self.llm_prior is None else "skipped",
            "gate": 0.0,
            "n_regions": 0,
        }
        if self.llm_prior is not None:
            interval = max(1, int(self.config.llm_prior_interval))
            n_llm = int(self.config.llm_prior_candidate_count)
            if n_llm <= 0:
                n_llm = max(5, self.config.K1 // 2)
            if iteration % interval == 0:
                regions, info = self.llm_prior.propose(
                    self.problem,
                    self.observations,
                    iteration=iteration,
                    budget_remaining=max(0, int(self.config.N) - len(self.history)),
                )
                self._last_llm_prior_info = info
                add(self.llm_prior.inverse_candidates(
                    self.problem,
                    regions,
                    n=n_llm,
                    rng=self.rng,
                    pool_size=self.config.llm_prior_inverse_pool_size,
                    gate=info.get("gate", 0.0),
                ), "llm_prior")
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
            epistemic_lookup=self._constraint_epistemic_lookup,
            tau=self.problem.tau,
            alpha_z=norm.ppf(1 - self.problem.alpha),
            beta_g=self.config.beta_g,
            certification_mode=self.config.certification_mode,
        ), "posterior")
        if not candidates:
            add([self.problem.sample_random(self.rng)], "random")
        return candidates, sources

    def _recommendation_pool(self):
        pool = set(x for x, _ in self.history)
        if self.config.recommend_observed_only:
            return list(pool)
        for x in random_candidates(
            self.problem,
            self._recommendation_random_pool_size(),
            self.rec_rng,
        ):
            pool.add(tuple(x))
        if self.config.recommendation_axis_oracle and hasattr(
            self.problem, "all_axis_solutions"
        ):
            for x in self.problem.all_axis_solutions():
                pool.add(tuple(x))
        elif not self.config.recommendation_axis_oracle:
            n_axis = self._recommendation_axis_candidate_count()
            for x in axis_landmark_candidates(self.problem, n_axis, self.rec_rng):
                pool.add(tuple(x))
            for x in axis_candidates(self.problem, n_axis, self.rec_rng):
                pool.add(tuple(x))
        for x in self._recommendation_refinement_candidates():
            pool.add(tuple(x))
        for x in self._observed_neighbor_candidates(
            n=max(0, int(self.config.observed_neighbor_candidate_count)),
            rng=self.rec_rng,
        ):
            pool.add(tuple(x))
        return list(pool)

    def _observed_nominal_incumbent(self):
        z = norm.ppf(1 - self.problem.alpha)
        sigma_floor = float(getattr(self.problem, "sigma_level", 0.0))
        margin_limit = float(self.config.observed_incumbent_margin_scale) * sigma_floor
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

    def _surrogate_feature_matrix(
        self,
        basis,
        xs,
        *,
        feature_mean=None,
        feature_scale=None,
    ):
        if hasattr(basis, "features_many"):
            raw = np.asarray(basis.features_many(xs), dtype=float)
        else:
            raw = np.vstack([
                np.asarray(basis.features(x), dtype=float).reshape(-1)
                for x in xs
            ])
        if feature_mean is None or feature_scale is None:
            if self.config.calibration_standardize_features:
                feature_mean = np.mean(raw, axis=0)
                feature_scale = np.std(raw, axis=0)
                feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
            else:
                feature_mean = np.zeros(raw.shape[1], dtype=float)
                feature_scale = np.ones(raw.shape[1], dtype=float)
        scaled = (raw - feature_mean) / feature_scale
        Phi = np.column_stack([np.ones(len(scaled), dtype=float), scaled])
        return Phi, np.asarray(feature_mean, dtype=float), np.asarray(feature_scale, dtype=float)

    def _constraint_calibration_fit(self):
        if not self.config.certification_calibration:
            return None
        if not hasattr(self.problem, "surrogate_basis_map"):
            return None
        basis = self.problem.surrogate_basis_map()
        if basis is None:
            return None
        if len(self.observations) < int(self.config.certification_calibration_min_obs):
            return None

        train_x = []
        train_con = []
        for x, ys in self.observations.items():
            train_x.append(tuple(int(v) for v in x))
            y_bar = np.mean(np.asarray(ys, dtype=float), axis=0)
            train_con.append(float(y_bar[1]))
        Phi, feature_mean, feature_scale = self._surrogate_feature_matrix(
            basis, train_x)
        y_con = np.asarray(train_con, dtype=float)
        ridge = max(float(self.config.certification_calibration_ridge), 0.0)
        penalty = ridge * np.eye(Phi.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = Phi.T @ Phi + penalty
        rhs = Phi.T @ y_con
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        try:
            inv_lhs = np.linalg.pinv(lhs)
        except np.linalg.LinAlgError:
            inv_lhs = np.eye(lhs.shape[0], dtype=float) / max(ridge, 1e-8)
        resid = y_con - Phi @ beta
        resid_sigma = (
            float(np.sqrt(np.mean(resid ** 2)))
            if len(resid)
            else 0.0
        )
        nominal_floor = (
            float(self.config.certification_calibration_noise_floor_scale)
            * float(getattr(self.problem, "sigma_level", 0.0))
        )
        sigma_cal = max(resid_sigma, nominal_floor, 1e-8)
        return {
            "basis": basis,
            "beta": beta,
            "inv_lhs": inv_lhs,
            "sigma": float(sigma_cal),
            "resid_sigma": float(resid_sigma),
            "n_train": int(len(train_x)),
            "feature_dim": int(Phi.shape[1]),
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        }

    def _calibrated_certification_result(self, pool, v_con=None):
        fit = self._constraint_calibration_fit()
        if fit is None or not pool:
            return None
        Phi_cand, _, _ = self._surrogate_feature_matrix(
            fit["basis"],
            pool,
            feature_mean=fit["feature_mean"],
            feature_scale=fit["feature_scale"],
        )
        mu = Phi_cand @ fit["beta"]
        leverage = np.sum((Phi_cand @ fit["inv_lhs"]) * Phi_cand, axis=1)
        leverage = np.maximum(leverage, 0.0)
        epistemic = (float(fit["sigma"]) ** 2) * leverage
        aleatoric = np.full(
            len(pool),
            max(float(fit["sigma"]) ** 2, 1e-12),
            dtype=float,
        )
        # Certification calibration is allowed to add structure, but not to
        # erase cumulative-risk uncertainty.  Sparse LODO runs otherwise turn
        # theory-unsafe points into calibrated-safe recommendations.
        if v_con is not None:
            hvd = np.maximum(np.asarray(v_con, dtype=float), 1e-12)
            aleatoric = np.maximum(hvd, aleatoric)
        beta = max(float(self.config.certification_calibration_beta), 0.0)
        cert = conservative_chance_margin(
            mu + self._pilot_constraint_guard(),
            epistemic,
            aleatoric,
            tau=self.problem.tau,
            alpha=self.problem.alpha,
            beta_g=beta,
            mode="theory",
        )
        return {
            "margin": cert.margin,
            "mu": cert.mu,
            "epistemic_var": cert.epistemic_var,
            "aleatoric_var": cert.aleatoric_var,
            "beta_g": float(cert.beta_g),
            "z_alpha": float(cert.z_alpha),
            "sigma": float(fit["sigma"]),
            "resid_sigma": float(fit["resid_sigma"]),
            "n_train": int(fit["n_train"]),
            "feature_dim": int(fit["feature_dim"]),
            "n_feasible": int(np.sum(cert.margin <= 0.0)),
            "leverage": leverage,
        }

    def _recommendation_calibration_fit(self):
        if not self.config.recommendation_calibration:
            return None
        if not hasattr(self.problem, "surrogate_basis_map"):
            return None
        basis = self.problem.surrogate_basis_map()
        if basis is None:
            return None
        if len(self.observations) < int(self.config.recommendation_calibration_min_obs):
            return None

        train_x = []
        train_obj = []
        train_con = []
        for x, ys in self.observations.items():
            train_x.append(tuple(int(v) for v in x))
            y_bar = np.mean(np.asarray(ys, dtype=float), axis=0)
            train_obj.append(float(y_bar[0]))
            train_con.append(float(y_bar[1]))
        Phi, feature_mean, feature_scale = self._surrogate_feature_matrix(
            basis, train_x)
        y_obj = np.asarray(train_obj, dtype=float)
        y_con = np.asarray(train_con, dtype=float)
        ridge = max(float(self.config.recommendation_calibration_ridge), 0.0)
        penalty = ridge * np.eye(Phi.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = Phi.T @ Phi + penalty
        rhs_obj = Phi.T @ y_obj
        rhs_con = Phi.T @ y_con
        try:
            beta_obj = np.linalg.solve(lhs, rhs_obj)
            beta_con = np.linalg.solve(lhs, rhs_con)
        except np.linalg.LinAlgError:
            beta_obj = np.linalg.lstsq(lhs, rhs_obj, rcond=None)[0]
            beta_con = np.linalg.lstsq(lhs, rhs_con, rcond=None)[0]
        try:
            lhs_inv = np.linalg.pinv(lhs)
        except np.linalg.LinAlgError:
            lhs_inv = None
        resid_con = y_con - Phi @ beta_con
        resid_sigma = float(np.sqrt(np.mean(resid_con ** 2))) if len(resid_con) else 0.0
        nominal_floor = (
            float(self.config.recommendation_noise_floor_scale)
            * 0.35
            * float(getattr(self.problem, "sigma_level", 0.0))
        )
        sigma_cal = max(resid_sigma, nominal_floor, 1e-8)
        return {
            "basis": basis,
            "Phi_train": Phi,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "beta_obj": beta_obj,
            "beta_con": beta_con,
            "lhs_inv": lhs_inv,
            "sigma": float(sigma_cal),
            "resid_sigma": float(resid_sigma),
            "n_train": int(len(train_x)),
            "feature_dim": int(Phi.shape[1]),
        }

    def _recommendation_calibration_pool_features(self, fit, pool):
        if fit is None or not pool:
            return None
        Phi_pool, _, _ = self._surrogate_feature_matrix(
            fit["basis"],
            pool,
            feature_mean=fit["feature_mean"],
            feature_scale=fit["feature_scale"],
        )
        return Phi_pool

    def _recommendation_calibration_audit_scores(
        self,
        pool,
        *,
        fit=None,
        phi_pool=None,
    ):
        """Evaluate the empirical recommendation surrogate on a fixed pool.

        This is diagnostics-only.  It mirrors the fallback model used by
        `_calibrated_recommendation_index`, but it does not choose a point.
        The audit lets us see whether a true-feasible pool member was missed
        because the theory certificate, the calibrated surrogate, or the source
        prior was too conservative.
        """
        if not pool:
            return None
        fit = fit or self._recommendation_calibration_fit()
        if fit is None:
            return None
        Phi_pool = (
            phi_pool
            if phi_pool is not None and len(phi_pool) == len(pool)
            else self._recommendation_calibration_pool_features(fit, pool)
        )
        pred_obj = Phi_pool @ fit["beta_obj"]
        pred_con = Phi_pool @ fit["beta_con"]
        z_alpha = float(norm.ppf(1 - self.problem.alpha))
        margin = pred_con + z_alpha * float(fit["sigma"]) - float(self.problem.tau)
        if fit["lhs_inv"] is None:
            leverage = np.full(len(pool), np.nan, dtype=float)
        else:
            leverage = np.sum((Phi_pool @ fit["lhs_inv"]) * Phi_pool, axis=1)
            leverage = np.maximum(np.asarray(leverage, dtype=float), 0.0)
        return {
            "pred_obj": np.asarray(pred_obj, dtype=float),
            "pred_con": np.asarray(pred_con, dtype=float),
            "margin": np.asarray(margin, dtype=float),
            "leverage": np.asarray(leverage, dtype=float),
            "sigma": float(fit["sigma"]),
            "resid_sigma": float(fit["resid_sigma"]),
            "n_train": int(fit["n_train"]),
            "feature_dim": int(fit["feature_dim"]),
            "n_feasible": int(np.sum(margin <= 0.0)),
        }

    def _calibrated_recommendation_index(
        self,
        pool,
        robust_margins,
        source_margins=None,
        guard_margins=None,
        fit=None,
        phi_pool=None,
    ):
        fit = fit or self._recommendation_calibration_fit()
        if fit is None:
            return None, {}
        scope = str(self.config.recommendation_calibration_scope or "refinement").lower()
        refinement = self._recommendation_refinement_candidates()
        if scope not in ("pool", "full", "recommendation_pool") and not refinement:
            return None, {}

        pool_index = {tuple(int(v) for v in x): i for i, x in enumerate(pool)}
        if scope in ("pool", "full", "recommendation_pool"):
            candidate_indices = list(range(len(pool)))
            scope_label = "pool"
        else:
            candidate_indices = [
                pool_index[x]
                for x in refinement
                if x in pool_index
            ]
            scope_label = "refinement"
        if not candidate_indices:
            return None, {}
        if phi_pool is not None and len(phi_pool) == len(pool):
            Phi_cand = np.asarray(phi_pool, dtype=float)[candidate_indices]
        else:
            Phi_cand, _, _ = self._surrogate_feature_matrix(
                fit["basis"],
                [pool[i] for i in candidate_indices],
                feature_mean=fit["feature_mean"],
                feature_scale=fit["feature_scale"],
            )
        pred_obj = Phi_cand @ fit["beta_obj"]
        if fit["lhs_inv"] is None:
            leverage = np.full(len(candidate_indices), np.nan, dtype=float)
        else:
            leverage = np.sum((Phi_cand @ fit["lhs_inv"]) * Phi_cand, axis=1)
            leverage = np.maximum(np.asarray(leverage, dtype=float), 0.0)
        decision_margin = np.asarray([
            robust_margins[i]
            for i in candidate_indices
        ], dtype=float)
        if guard_margins is None:
            theory_margin = decision_margin
        else:
            guard_margins = np.asarray(guard_margins, dtype=float)
            if len(guard_margins) == len(pool):
                theory_margin = np.asarray([
                    guard_margins[i]
                    for i in candidate_indices
                ], dtype=float)
            else:
                theory_margin = decision_margin
        source_margin_local = None
        if source_margins is not None:
            source_margins = np.asarray(source_margins, dtype=float)
            if len(source_margins) == len(pool):
                source_margin_local = np.asarray([
                    source_margins[i]
                    for i in candidate_indices
                ], dtype=float)
        max_leverage = float(
            self.config.recommendation_calibration_max_leverage)
        max_theory_margin = float(
            self.config.recommendation_calibration_max_theory_margin)
        calibration_guard = np.ones(len(candidate_indices), dtype=bool)
        if max_leverage > 0.0:
            calibration_guard &= np.isfinite(leverage)
            calibration_guard &= leverage <= max_leverage
        if max_theory_margin > 0.0:
            calibration_guard &= np.isfinite(theory_margin)
            calibration_guard &= theory_margin <= max_theory_margin

        def choose_position(local_positions):
            positions = np.asarray(local_positions, dtype=int)
            details = {
                "source_mean_prior_guard_used": False,
                "source_mean_prior_ranker_used": False,
                "source_mean_prior_guard_n_feasible": None,
                "source_mean_prior_guard_selected_margin": None,
            }
            if len(positions) == 0:
                return None, details
            if source_margin_local is not None:
                margins = np.asarray(source_margin_local[positions], dtype=float)
                finite = np.isfinite(margins)
                tol = float(self.config.source_mean_prior_margin_tol)
                source_safe = finite & (margins <= tol)
                details["source_mean_prior_guard_n_feasible"] = int(np.sum(source_safe))
                if np.any(source_safe):
                    guarded = positions[source_safe]
                    chosen = int(guarded[int(np.argmin(pred_obj[guarded]))])
                    details["source_mean_prior_guard_used"] = True
                    details["source_mean_prior_guard_selected_margin"] = float(
                        source_margin_local[chosen])
                    return chosen, details
                if str(self.config.recommendation_infeasible_strategy).lower() in (
                    "source_prior",
                    "source_prior_margin",
                ) and np.any(finite):
                    finite_positions = positions[finite]
                    order = np.lexsort((
                        pred_obj[finite_positions],
                        source_margin_local[finite_positions],
                    ))
                    chosen = int(finite_positions[int(order[0])])
                    details["source_mean_prior_ranker_used"] = True
                    details["source_mean_prior_guard_selected_margin"] = float(
                        source_margin_local[chosen])
                    return chosen, details
            chosen = int(positions[int(np.argmin(pred_obj[positions]))])
            if source_margin_local is not None and np.isfinite(source_margin_local[chosen]):
                details["source_mean_prior_guard_selected_margin"] = float(
                    source_margin_local[chosen])
            return chosen, details

        certified = decision_margin <= 0.0
        certified_guarded = certified & calibration_guard
        if np.any(certified_guarded):
            local_cert = np.where(certified_guarded)[0]
            chosen_pos, source_details = choose_position(local_cert)
            if chosen_pos is None:
                return None, {}
            return int(candidate_indices[chosen_pos]), {
                "calibrated_recommendation_reason": (
                    f"guarded_certified_{scope_label}_objective"),
                "calibrated_objective": float(pred_obj[chosen_pos]),
                "calibrated_constraint_margin": None,
                "calibrated_constraint_feasible": None,
                "calibrated_constraint_sigma": None,
                "calibrated_recommendation_scope": scope_label,
                "n_calibration_refinement": int(len(candidate_indices)),
                "n_calibration_candidates": int(len(candidate_indices)),
                "n_calibration_certified": int(np.sum(certified)),
                "n_calibration_certified_guarded": int(np.sum(certified_guarded)),
                "n_calibration_guarded": int(np.sum(calibration_guard)),
                "calibration_max_leverage": (
                    None if max_leverage <= 0.0 else float(max_leverage)),
                "calibration_max_theory_margin": (
                    None if max_theory_margin <= 0.0
                    else float(max_theory_margin)),
                "calibration_selected_leverage": (
                    None if not np.isfinite(leverage[chosen_pos])
                    else float(leverage[chosen_pos])),
                "calibration_selected_theory_margin": (
                    None if not np.isfinite(theory_margin[chosen_pos])
                    else float(theory_margin[chosen_pos])),
                **source_details,
            }

        # If the theory bound is too conservative everywhere, fit a local
        # low-dimensional constraint surrogate on observed data and use it only
        # as a fallback.  The returned posterior_feasible flag remains false;
        # this path is an empirical recommendation rescue, not a certification
        # claim.
        pred_con = Phi_cand @ fit["beta_con"]
        sigma_cal = float(fit["sigma"])
        z_alpha = float(norm.ppf(1 - self.problem.alpha))
        calibrated_margin = (
            pred_con
            + z_alpha * sigma_cal
            - float(self.problem.tau)
        )
        feasible_raw = calibrated_margin <= 0.0
        feasible = feasible_raw & calibration_guard
        if not np.any(feasible):
            source_rescue = None
            source_rescue_details = {}
            if (
                source_margin_local is not None
                and str(self.config.recommendation_infeasible_strategy).lower() in (
                    "source_prior",
                    "source_prior_margin",
                )
            ):
                tol = float(self.config.source_mean_prior_margin_tol)
                source_safe = (
                    np.isfinite(source_margin_local)
                    & (source_margin_local <= tol)
                    & calibration_guard
                )
                if np.any(source_safe):
                    source_rescue, source_rescue_details = choose_position(
                        np.where(source_safe)[0])
                    if source_rescue is not None:
                        return int(candidate_indices[source_rescue]), {
                            "calibrated_recommendation_reason": (
                                "source_safe_calibrated_margin_rescue"),
                            "calibrated_objective": float(pred_obj[source_rescue]),
                            "calibrated_constraint_margin": float(
                                calibrated_margin[source_rescue]),
                            "calibrated_guarded_constraint_margin": float(
                                calibrated_margin[source_rescue]),
                            "calibrated_constraint_feasible": False,
                            "calibrated_constraint_sigma": float(sigma_cal),
                            "calibrated_recommendation_scope": scope_label,
                            "n_calibration_refinement": int(len(candidate_indices)),
                            "n_calibration_candidates": int(len(candidate_indices)),
                            "n_calibration_certified": 0,
                            "n_calibration_certified_guarded": 0,
                            "n_calibration_feasible": int(np.sum(feasible)),
                            "n_calibration_raw_feasible": int(np.sum(feasible_raw)),
                            "n_calibration_guarded": int(np.sum(calibration_guard)),
                            "n_calibration_source_safe_guarded": int(np.sum(source_safe)),
                            "calibration_max_leverage": (
                                None if max_leverage <= 0.0 else float(max_leverage)),
                            "calibration_max_theory_margin": (
                                None if max_theory_margin <= 0.0
                                else float(max_theory_margin)),
                            "calibration_selected_leverage": (
                                None if not np.isfinite(leverage[source_rescue])
                                else float(leverage[source_rescue])),
                            "calibration_selected_theory_margin": (
                                None if not np.isfinite(theory_margin[source_rescue])
                                else float(theory_margin[source_rescue])),
                            "calibration_selected_source_margin": float(
                                source_margin_local[source_rescue]),
                            **source_rescue_details,
                        }
            if np.any(calibration_guard):
                guarded_min = float(np.min(calibrated_margin[calibration_guard]))
            else:
                guarded_min = None
            return None, {
                "calibrated_recommendation_reason": (
                    "no_calibrated_feasible_after_guard"
                    if np.any(feasible_raw)
                    else "no_calibrated_feasible"),
                "calibrated_objective": None,
                "calibrated_constraint_margin": float(np.min(calibrated_margin)),
                "calibrated_guarded_constraint_margin": guarded_min,
                "calibrated_constraint_feasible": False,
                "calibrated_constraint_sigma": float(sigma_cal),
                "calibrated_recommendation_scope": scope_label,
                "n_calibration_refinement": int(len(candidate_indices)),
                "n_calibration_candidates": int(len(candidate_indices)),
                "n_calibration_certified": 0,
                "n_calibration_certified_guarded": 0,
                "n_calibration_feasible": int(np.sum(feasible)),
                "n_calibration_raw_feasible": int(np.sum(feasible_raw)),
                "n_calibration_guarded": int(np.sum(calibration_guard)),
                "calibration_max_leverage": (
                    None if max_leverage <= 0.0 else float(max_leverage)),
                "calibration_max_theory_margin": (
                    None if max_theory_margin <= 0.0
                    else float(max_theory_margin)),
                "calibration_min_leverage": (
                    None if not np.any(np.isfinite(leverage))
                    else float(np.nanmin(leverage))),
                "calibration_median_leverage": (
                    None if not np.any(np.isfinite(leverage))
                    else float(np.nanmedian(leverage))),
                "calibration_min_theory_margin": (
                    None if not np.any(np.isfinite(theory_margin))
                    else float(np.nanmin(theory_margin))),
                "source_mean_prior_guard_n_feasible": (
                    None
                    if source_margin_local is None
                    else int(np.sum(
                        np.isfinite(source_margin_local)
                        & (source_margin_local <= float(
                            self.config.source_mean_prior_margin_tol))
                        & calibration_guard
                    ))
                ),
            }
        local_feas = np.where(feasible)[0]
        chosen_pos, source_details = choose_position(local_feas)
        if chosen_pos is None:
            return None, {}
        return int(candidate_indices[chosen_pos]), {
            "calibrated_recommendation_reason": "calibrated_constraint_fallback",
            "calibrated_objective": float(pred_obj[chosen_pos]),
            "calibrated_constraint_margin": float(calibrated_margin[chosen_pos]),
            "calibrated_guarded_constraint_margin": float(
                calibrated_margin[chosen_pos]),
            "calibrated_constraint_feasible": True,
            "calibrated_constraint_sigma": float(sigma_cal),
            "calibrated_recommendation_scope": scope_label,
            "n_calibration_refinement": int(len(candidate_indices)),
            "n_calibration_candidates": int(len(candidate_indices)),
            "n_calibration_certified": 0,
            "n_calibration_certified_guarded": 0,
            "n_calibration_feasible": int(np.sum(feasible)),
            "n_calibration_raw_feasible": int(np.sum(feasible_raw)),
            "n_calibration_guarded": int(np.sum(calibration_guard)),
            "calibration_max_leverage": (
                None if max_leverage <= 0.0 else float(max_leverage)),
            "calibration_max_theory_margin": (
                None if max_theory_margin <= 0.0 else float(max_theory_margin)),
            "calibration_selected_leverage": (
                None if not np.isfinite(leverage[chosen_pos])
                else float(leverage[chosen_pos])),
            "calibration_selected_theory_margin": (
                None if not np.isfinite(theory_margin[chosen_pos])
                else float(theory_margin[chosen_pos])),
            "calibration_min_leverage": (
                None if not np.any(np.isfinite(leverage))
                else float(np.nanmin(leverage))),
            "calibration_median_leverage": (
                None if not np.any(np.isfinite(leverage))
                else float(np.nanmedian(leverage))),
            "calibration_min_theory_margin": (
                None if not np.any(np.isfinite(theory_margin))
                else float(np.nanmin(theory_margin))),
            **source_details,
        }

    def _solve_posterior_recommendation(self):
        pool = self._recommendation_pool()
        mu_obj = self.gpr[0].posterior_mean_many(pool)
        mu_con = self.gpr[1].posterior_mean_many(pool)
        v_con = self.variance_model.predict_certification_variance_many(
            1, pool, self.problem)
        cert = self._certification_result(mu_con, pool, v_con)
        margins = cert.margin
        sig_con = np.sqrt(np.maximum(cert.aleatoric_var, 1e-12))
        if cert.mode == "theory":
            safety_buffer = np.zeros_like(margins)
        else:
            nominal_floor = (
                self.config.recommendation_noise_floor_scale
                * float(getattr(self.problem, "sigma_level", 0.0))
            )
            safety_buffer = np.maximum(
                self.config.recommendation_safety_z * sig_con,
                nominal_floor,
            )
        recommendation_slack = self._recommendation_slack()
        theory_margins = np.asarray(margins + safety_buffer, dtype=float)
        robust_margins = theory_margins + recommendation_slack
        effective_mu_con = np.asarray(mu_con, dtype=float)
        effective_epistemic = np.asarray(cert.epistemic_var, dtype=float)
        effective_aleatoric = np.asarray(cert.aleatoric_var, dtype=float)
        certification_sources = np.full(len(pool), "theory", dtype=object)
        calibrated_cert = self._calibrated_certification_result(pool, v_con)
        calibration_policy = str(
            self.config.certification_calibration_policy or "conservative"
        ).lower()
        if calibrated_cert is not None:
            calibrated_margins = (
                np.asarray(calibrated_cert["margin"], dtype=float)
                + recommendation_slack
            )
            if calibration_policy in ("replace", "optimistic", "legacy"):
                use_calibrated = calibrated_margins < robust_margins
                robust_margins = np.where(
                    use_calibrated,
                    calibrated_margins,
                    robust_margins,
                )
            elif calibration_policy in ("guarded", "supported"):
                lower_calibrated = calibrated_margins < robust_margins
                max_leverage = float(
                    self.config.certification_calibration_max_leverage)
                if max_leverage > 0.0:
                    leverage = np.asarray(
                        calibrated_cert.get("leverage", np.inf),
                        dtype=float,
                    )
                    lower_calibrated &= np.isfinite(leverage)
                    lower_calibrated &= leverage <= max_leverage
                max_theory_margin = float(
                    self.config.certification_calibration_max_theory_margin)
                if max_theory_margin > 0.0:
                    lower_calibrated &= np.isfinite(robust_margins)
                    lower_calibrated &= robust_margins <= max_theory_margin
                raise_delta = max(
                    float(self.config.certification_calibration_raise_delta),
                    0.0,
                )
                raise_calibrated = calibrated_margins > robust_margins + raise_delta
                use_calibrated = lower_calibrated | raise_calibrated
                robust_margins = np.where(
                    use_calibrated,
                    calibrated_margins,
                    robust_margins,
                )
            elif calibration_policy in ("off", "disabled", "none"):
                use_calibrated = np.zeros(len(pool), dtype=bool)
            else:
                use_calibrated = calibrated_margins > robust_margins
                robust_margins = np.maximum(robust_margins, calibrated_margins)
            effective_mu_con = np.where(
                use_calibrated,
                calibrated_cert["mu"],
                effective_mu_con,
            )
            effective_epistemic = np.where(
                use_calibrated,
                calibrated_cert["epistemic_var"],
                effective_epistemic,
            )
            effective_aleatoric = np.where(
                use_calibrated,
                calibrated_cert["aleatoric_var"],
                effective_aleatoric,
            )
            certification_sources = np.where(
                use_calibrated,
                "calibrated",
                certification_sources,
            )
        else:
            calibrated_margins = None
        recommendation_calibration_fit = self._recommendation_calibration_fit()
        recommendation_calibration_phi_pool = (
            self._recommendation_calibration_pool_features(
                recommendation_calibration_fit,
                pool,
            )
            if recommendation_calibration_fit is not None
            else None
        )
        recommendation_calibration_audit = (
            self._recommendation_calibration_audit_scores(
                pool,
                fit=recommendation_calibration_fit,
                phi_pool=recommendation_calibration_phi_pool,
            )
        )
        feasible = robust_margins <= 0.0
        if np.any(feasible):
            local = int(np.argmin(np.where(feasible, mu_obj, np.inf)))
        elif str(self.config.recommendation_infeasible_strategy).lower() in (
            "min_margin",
            "lexicographic",
        ):
            min_margin = float(np.min(robust_margins))
            near_min_margin = robust_margins <= min_margin + 1e-12
            local = int(np.argmin(np.where(near_min_margin, mu_obj, np.inf)))
        else:
            scaled_obj = mu_obj - float(np.min(mu_obj))
            obj_span = float(np.max(scaled_obj))
            if obj_span > 1e-12:
                scaled_obj = scaled_obj / obj_span
            scaled_margin = np.maximum(robust_margins, 0.0)
            margin_span = float(np.max(scaled_margin))
            if margin_span > 1e-12:
                scaled_margin = scaled_margin / margin_span
            local = int(np.argmin(
                scaled_obj
                + self.config.recommendation_infeasible_penalty * scaled_margin
            ))
        used_observed_incumbent = False
        observed_incumbent_rejected = False
        observed_incumbent_reason = None
        observed_idx = None
        observed_incumbent = self._observed_nominal_incumbent()
        if observed_incumbent is not None:
            try:
                observed_idx = pool.index(observed_incumbent["x"])
                observed_is_robust = bool(robust_margins[observed_idx] <= 0.0)
                if (
                    observed_is_robust
                    and observed_incumbent["empirical_objective"] <= float(mu_obj[local])
                ):
                    local = observed_idx
                    used_observed_incumbent = True
                    observed_incumbent_reason = "robust_observed_incumbent"
                elif (
                    self.config.recommendation_observed_fallback
                    and not np.any(feasible)
                ):
                    local = observed_idx
                    used_observed_incumbent = True
                    observed_incumbent_reason = "empirical_observed_fallback"
                else:
                    observed_incumbent_rejected = True
            except ValueError:
                pass
        source_prior_recommendation_used = False
        source_margins = None
        source_prior_details = {
            "source_mean_prior_fallback": bool(self.config.source_mean_prior_fallback),
            "source_mean_prior_used": False,
            "source_mean_prior_available": False,
            "source_mean_prior_n_feasible": 0,
            "source_mean_prior_min_margin": None,
            "source_mean_prior_selected_margin": None,
            "source_mean_prior_guard_used": False,
            "source_mean_prior_ranker_used": False,
            "source_mean_prior_guard_n_feasible": None,
        }
        if (
            self.config.source_mean_prior_fallback
            and hasattr(self.problem, "source_mean_prior_predict_many")
        ):
            try:
                source_mu = self.problem.source_mean_prior_predict_many(pool, output_index=1)
                if source_mu is not None:
                    source_mu = np.asarray(source_mu, dtype=float)
                    try:
                        source_sigma = float(
                            self.problem.source_mean_prior_sigma(output_index=1))
                    except Exception:
                        source_sigma = float(getattr(self.problem, "sigma_level", 0.0))
                    source_margin = (
                        source_mu
                        + float(self.config.source_mean_prior_z) * max(source_sigma, 1e-8)
                        - float(self.problem.tau)
                    )
                    source_margins = np.asarray(source_margin, dtype=float)
                    source_prior_details.update({
                        "source_mean_prior_available": True,
                        "source_mean_prior_sigma": float(source_sigma),
                        "source_mean_prior_min_margin": float(np.min(source_margin)),
                        "source_mean_prior_n_feasible": int(
                            np.sum(source_margin <= float(
                                self.config.source_mean_prior_margin_tol))),
                    })
            except Exception:
                source_margins = None
        calibrated_recommendation_used = False
        calibrated_details = {}
        calibrated_idx, calibrated_details = self._calibrated_recommendation_index(
            pool,
            robust_margins,
            source_margins=source_margins,
            guard_margins=theory_margins + recommendation_slack,
            fit=recommendation_calibration_fit,
            phi_pool=recommendation_calibration_phi_pool,
        )
        if calibrated_idx is not None:
            keep_observed = False
            if used_observed_incumbent and observed_idx is not None:
                calibrated_is_robust = bool(robust_margins[calibrated_idx] <= 0.0)
                calibrated_better = (
                    float(mu_obj[calibrated_idx])
                    < float(observed_incumbent["empirical_objective"]) - 1e-12
                )
                keep_observed = not (calibrated_is_robust and calibrated_better)
                if keep_observed:
                    calibrated_details = {
                        **calibrated_details,
                        "calibrated_recommendation_rejected_by_observed": True,
                        "calibrated_recommendation_rejected_observed_reason": (
                            observed_incumbent_reason),
                        "calibrated_recommendation_rejected_candidate_margin": float(
                            robust_margins[calibrated_idx]),
                        "calibrated_recommendation_rejected_candidate_mu_obj": float(
                            mu_obj[calibrated_idx]),
                    }
            if not keep_observed:
                local = calibrated_idx
                calibrated_recommendation_used = True
                if source_margins is not None:
                    guard_used = bool(calibrated_details.get(
                        "source_mean_prior_guard_used", False))
                    ranker_used = bool(calibrated_details.get(
                        "source_mean_prior_ranker_used", False))
                    if guard_used or ranker_used:
                        source_prior_details.update({
                            "source_mean_prior_used": True,
                            "source_mean_prior_guard_used": guard_used,
                            "source_mean_prior_ranker_used": ranker_used,
                            "source_mean_prior_guard_n_feasible": calibrated_details.get(
                                "source_mean_prior_guard_n_feasible"),
                            "source_mean_prior_selected_margin": float(
                                source_margins[local]),
                        })
        if (
            self.config.source_mean_prior_fallback
            and not np.any(feasible)
            and not calibrated_recommendation_used
            and not used_observed_incumbent
            and source_margins is not None
        ):
            source_feasible = source_margins <= float(
                self.config.source_mean_prior_margin_tol)
            if np.any(source_feasible):
                local = int(np.argmin(np.where(source_feasible, mu_obj, np.inf)))
                source_prior_recommendation_used = True
            elif str(self.config.recommendation_infeasible_strategy).lower() in (
                "source_prior",
                "source_prior_margin",
            ):
                local = int(np.lexsort((mu_obj, source_margins))[0])
                source_prior_recommendation_used = True
                source_prior_details["source_mean_prior_ranker_used"] = True
            if source_prior_recommendation_used:
                source_prior_details.update({
                    "source_mean_prior_used": True,
                    "source_mean_prior_selected_margin": float(source_margins[local]),
                })
        x_best = tuple(int(v) for v in pool[local])
        calibration_details = {}
        if calibrated_cert is not None:
            calibration_details = {
                "certification_calibration_used": True,
                "certification_calibration_sigma": float(calibrated_cert["sigma"]),
                "certification_calibration_resid_sigma": float(
                    calibrated_cert["resid_sigma"]),
                "certification_calibration_n_train": int(calibrated_cert["n_train"]),
                "certification_calibration_feature_dim": int(
                    calibrated_cert["feature_dim"]),
                "certification_calibration_beta_g": float(
                    calibrated_cert["beta_g"]),
                "certification_calibration_n_feasible": int(
                    calibrated_cert["n_feasible"]),
                "certification_calibration_policy": calibration_policy,
                "certification_calibration_n_used": int(np.sum(use_calibrated)),
                "certification_calibration_max_leverage": float(
                    self.config.certification_calibration_max_leverage),
                "certification_calibration_max_theory_margin": float(
                    self.config.certification_calibration_max_theory_margin),
                "certification_calibration_raise_delta": float(
                    self.config.certification_calibration_raise_delta),
                "posterior_calibrated_chance_margin": float(
                    calibrated_margins[local]),
            }
        else:
            calibration_details = {
                "certification_calibration_used": False,
                "certification_calibration_policy": calibration_policy,
                "posterior_calibrated_chance_margin": None,
            }
        recommendation_calibration_details = {}
        if recommendation_calibration_audit is not None:
            cal_margin = recommendation_calibration_audit["margin"]
            cal_pred_obj = recommendation_calibration_audit["pred_obj"]
            cal_leverage = recommendation_calibration_audit["leverage"]
            recommendation_calibration_details = {
                "recommendation_calibration_audit_available": True,
                "recommendation_calibration_sigma": float(
                    recommendation_calibration_audit["sigma"]),
                "recommendation_calibration_resid_sigma": float(
                    recommendation_calibration_audit["resid_sigma"]),
                "recommendation_calibration_n_train": int(
                    recommendation_calibration_audit["n_train"]),
                "recommendation_calibration_feature_dim": int(
                    recommendation_calibration_audit["feature_dim"]),
                "recommendation_calibration_n_feasible": int(
                    recommendation_calibration_audit["n_feasible"]),
                "recommendation_selected_calibrated_rec_margin": float(
                    cal_margin[local]),
                "recommendation_selected_calibrated_rec_objective": float(
                    cal_pred_obj[local]),
                "recommendation_selected_calibrated_rec_leverage": (
                    None
                    if not np.isfinite(cal_leverage[local])
                    else float(cal_leverage[local])
                ),
            }
        else:
            recommendation_calibration_details = {
                "recommendation_calibration_audit_available": False,
            }
        return x_best, {
            "posterior_mu_obj": float(mu_obj[local]),
            "posterior_mu_con": float(effective_mu_con[local]),
            "posterior_gpr_mu_con": float(mu_con[local]),
            "posterior_epistemic_variance_con": float(effective_epistemic[local]),
            "posterior_gpr_epistemic_variance_con": float(cert.epistemic_var[local]),
            "posterior_variance_con": float(effective_aleatoric[local]),
            "posterior_hvd_variance_con": float(cert.aleatoric_var[local]),
            "posterior_beta_g": float(cert.beta_g),
            "certification_mode": cert.mode,
            "posterior_chance_margin": float(robust_margins[local]),
            "posterior_theory_chance_margin": float(theory_margins[local]),
            "posterior_robust_chance_margin": float(robust_margins[local]),
            "posterior_certification_source": str(certification_sources[local]),
            "recommendation_safety_z": float(self.config.recommendation_safety_z),
            "recommendation_noise_floor_scale": float(
                self.config.recommendation_noise_floor_scale),
            "recommendation_slack": float(recommendation_slack),
            "recommendation_infeasible_penalty": float(
                self.config.recommendation_infeasible_penalty),
            "recommendation_observed_fallback": bool(
                self.config.recommendation_observed_fallback),
            "recommendation_calibration": bool(self.config.recommendation_calibration),
            "calibrated_recommendation_used": bool(calibrated_recommendation_used),
            "source_prior_recommendation_used": bool(source_prior_recommendation_used),
            **calibration_details,
            **recommendation_calibration_details,
            **calibrated_details,
            **source_prior_details,
            "observed_incumbent_used": bool(used_observed_incumbent),
            "observed_incumbent_rejected": bool(observed_incumbent_rejected),
            "observed_incumbent_reason": observed_incumbent_reason,
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
            "n_theory_posterior_feasible": int(np.sum(theory_margins <= 0.0)),
            **self._truth_pool_diagnostics(
                pool,
                selected=x_best,
                prefix="recommendation",
            ),
            **self._truth_pool_decision_margin_audit(
                pool,
                robust_margins,
                selected=x_best,
                mu_con=effective_mu_con,
                epistemic_var=effective_epistemic,
                aleatoric_var=effective_aleatoric,
                theory_margins=theory_margins,
                calibrated_margins=calibrated_margins,
                recommendation_calibrated_margins=(
                    None
                    if recommendation_calibration_audit is None
                    else recommendation_calibration_audit["margin"]
                ),
                recommendation_calibrated_objectives=(
                    None
                    if recommendation_calibration_audit is None
                    else recommendation_calibration_audit["pred_obj"]
                ),
                recommendation_calibrated_leverage=(
                    None
                    if recommendation_calibration_audit is None
                    else recommendation_calibration_audit["leverage"]
                ),
                source_margins=source_margins,
                certification_sources=certification_sources,
                prefix="recommendation",
            ),
        }

    def _terminal_value_from_models(self, gpr_models, variance_model, pool):
        """Terminal certified value used by the optional exact-update KG.

        Lower is better.  If the fixed terminal pool has no robust-feasible
        point, use the same normalized infeasibility penalty shape as
        `_solve_posterior_recommendation` so the value remains comparable.
        """
        if len(pool) == 0:
            return 0.0
        mu_obj = gpr_models[0].posterior_mean_many(pool)
        mu_con = gpr_models[1].posterior_mean_many(pool)
        v_con = variance_model.predict_certification_variance_many(
            1, pool, self.problem)
        epistemic = gpr_models[1].posterior_var_many(pool)
        cert = conservative_chance_margin(
            np.asarray(mu_con, dtype=float) + self._pilot_constraint_guard(),
            epistemic,
            v_con,
            tau=self.problem.tau,
            alpha=self.problem.alpha,
            beta_g=self.config.beta_g,
            mode=self.config.certification_mode,
        )
        margins = cert.margin
        sig_con = np.sqrt(np.maximum(cert.aleatoric_var, 1e-12))
        if cert.mode == "theory":
            safety_buffer = np.zeros_like(margins)
        else:
            nominal_floor = (
                self.config.recommendation_noise_floor_scale
                * float(getattr(self.problem, "sigma_level", 0.0))
            )
            safety_buffer = np.maximum(
                self.config.recommendation_safety_z * sig_con,
                nominal_floor,
            )
        robust_margins = margins + safety_buffer + self._recommendation_slack()
        feasible = robust_margins <= 0.0
        if np.any(feasible):
            return float(np.min(np.where(feasible, mu_obj, np.inf)))
        if str(self.config.recommendation_infeasible_strategy).lower() in (
            "min_margin",
            "lexicographic",
        ):
            min_margin = float(np.min(robust_margins))
            near_min_margin = robust_margins <= min_margin + 1e-12
            return float(np.min(np.where(near_min_margin, mu_obj, np.inf)))
        scaled_obj = mu_obj - float(np.min(mu_obj))
        obj_span = float(np.max(scaled_obj))
        if obj_span > 1e-12:
            scaled_obj = scaled_obj / obj_span
        scaled_margin = np.maximum(robust_margins, 0.0)
        margin_span = float(np.max(scaled_margin))
        if margin_span > 1e-12:
            scaled_margin = scaled_margin / margin_span
        penalized = (
            scaled_obj
            + self.config.recommendation_infeasible_penalty * scaled_margin
        )
        return float(np.min(penalized))

    def _effective_exact_kg_mc_samples(self):
        mode = str(self.config.acquisition_mode or "additive").lower()
        if (
            mode == "additive"
            and not self.config.exact_kg_use_score
            and float(self.config.exact_kg_blend) <= 0.0
        ):
            return 0
        if mode in ("exact_mc", "blend") and int(self.config.exact_kg_mc_samples) <= 0:
            return 8
        return int(self.config.exact_kg_mc_samples)

    def _exact_posterior_update_score_one(
        self,
        x,
        common_z,
        terminal_pool,
        current_value,
    ):
        x_arr = np.asarray(x, dtype=int)
        mu_before = [self.gpr[i].posterior_mean(x_arr) for i in range(2)]
        sigma2_before = [
            self.variance_model.predict_variance(i, x_arr, self.problem)
            for i in range(2)
        ]
        pred_sd = [
            np.sqrt(max(
                float(sigma2_before[i]) + self.gpr[i].posterior_var(x_arr),
                1e-12,
            ))
            for i in range(2)
        ]
        gains = []
        existing_observations = list(self.observations.get(
            tuple(int(v) for v in x_arr), []))
        for z_vec in common_z:
            gpr_clone = [
                self._clone_gpr_for_exact_kg(model)
                for model in self.gpr
            ]
            var_clone = self._clone_variance_model_for_exact_kg()
            y = [
                float(mu_before[i] + pred_sd[i] * z_vec[i])
                for i in range(2)
            ]
            for i in range(2):
                gpr_clone[i].update(x_arr, y[i], sigma2_before[i])
            for i in range(2):
                replicate_values = [
                    float(np.asarray(observed, dtype=float)[i])
                    for observed in existing_observations
                ] + [float(y[i])]
                replicate_variance = (
                    float(np.var(replicate_values, ddof=1))
                    if len(replicate_values) >= 2
                    else None
                )
                var_clone.update(
                    i,
                    x_arr,
                    y[i],
                    mu_before[i],
                    gpr_clone[i],
                    self.problem,
                    replicate_variance=replicate_variance,
                )
            future_value = self._terminal_value_from_models(
                gpr_clone, var_clone, terminal_pool)
            gains.append(current_value - future_value)
        return max(float(np.mean(gains)), 0.0)

    def _exact_posterior_update_scores(self, candidates, terminal_pool):
        """Monte Carlo exact posterior-update KG over a fixed terminal pool.

        This is intentionally optional and small-budget friendly.  It samples
        predictive observations, applies the same GPR update and HVD residual
        update as the main loop, and measures current terminal value minus
        updated terminal value.  Candidate-level work is embarrassingly
        parallel; thread workers avoid process pickling of SUMO/problem handles.
        """
        mc = self._effective_exact_kg_mc_samples()
        if mc <= 0 or len(candidates) == 0:
            return np.zeros(len(candidates), dtype=float)
        current_value = self._terminal_value_from_models(
            self.gpr, self.variance_model, terminal_pool)
        self._last_exact_kg_current_value = float(current_value)
        common_z = self.rng.standard_normal((mc, 2))
        out = np.zeros(len(candidates), dtype=float)
        jobs = max(1, int(self.config.exact_kg_jobs))
        jobs = min(jobs, len(candidates))
        stage_n = int(getattr(self, "_progress_stage_n", len(self.history)))
        step_started_at = float(
            getattr(self, "_progress_step_started_at", time.perf_counter()))
        run_started_at = float(
            getattr(self, "_progress_run_started_at", step_started_at))
        emit_updates = max(1, int(getattr(self.config, "progress_exact_updates", 10)))
        emit_every = max(1, int(np.ceil(len(candidates) / float(emit_updates))))
        self._progress_emit(
            n=stage_n,
            frac=0.35,
            kind="exact_kg_start",
            started_at=step_started_at,
            run_started_at=run_started_at,
            extra=f"candidates={len(candidates)} mc={int(mc)} jobs={int(jobs)}",
        )
        if jobs <= 1:
            for j, x in enumerate(candidates):
                out[j] = self._exact_posterior_update_score_one(
                    x, common_z, terminal_pool, current_value)
                done = j + 1
                if done == len(candidates) or done % emit_every == 0:
                    frac = 0.35 + 0.55 * (float(done) / float(len(candidates)))
                    self._progress_emit(
                        n=stage_n,
                        frac=frac,
                        kind="exact_kg_candidates",
                        started_at=step_started_at,
                        run_started_at=run_started_at,
                        extra=f"candidates_done={done}/{len(candidates)}",
                    )
            return out
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    self._exact_posterior_update_score_one,
                    x,
                    common_z,
                    terminal_pool,
                    current_value,
                ): j
                for j, x in enumerate(candidates)
            }
            done = 0
            for future in as_completed(futures):
                out[futures[future]] = future.result()
                done += 1
                if done == len(candidates) or done % emit_every == 0:
                    frac = 0.35 + 0.55 * (float(done) / float(len(candidates)))
                    self._progress_emit(
                        n=stage_n,
                        frac=frac,
                        kind="exact_kg_candidates",
                        started_at=step_started_at,
                        run_started_at=run_started_at,
                        extra=f"candidates_done={done}/{len(candidates)}",
                    )
        return out

    def _evaluate_recommendation(self, x_best):
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
        true_best_x, true_best_obj = self._true_best_feasible_cached()
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

    def _true_chance_margin(self, x):
        sig = self.problem.true_sigma(x)
        return float(
            self.problem.true_constraint_mean(x)
            + norm.ppf(1 - self.problem.alpha) * float(sig[1])
            - self.problem.tau
        )

    def _truth_pool_diagnostics(self, pool, selected=None, prefix="candidate"):
        if not self.config.truth_pool_diagnostics or not pool:
            return {}
        pool = [tuple(int(v) for v in x) for x in unique_candidates(pool)]
        cap = int(self.config.truth_pool_max_candidates)
        if cap > 0 and len(pool) > cap:
            # Deterministic subsample for diagnostics only; never affects decisions.
            idx = np.linspace(0, len(pool) - 1, cap)
            pool = [pool[int(round(i))] for i in idx]
        try:
            _, true_best_obj = self._true_best_feasible_cached()
        except Exception:
            true_best_obj = np.inf
        margins = []
        regrets = []
        for x in pool:
            try:
                margin = self._true_chance_margin(x)
                obj = float(self.problem.true_objective(x))
            except Exception:
                continue
            margins.append(margin)
            regrets.append(obj - true_best_obj if np.isfinite(true_best_obj) else np.nan)
        if not margins:
            return {f"{prefix}_truth_diagnostics_available": False}
        margins = np.asarray(margins, dtype=float)
        regrets = np.asarray(regrets, dtype=float)
        feasible = margins <= 0.0
        good_eps = float(self.config.truth_pool_good_regret)
        good = feasible & np.isfinite(regrets) & (regrets <= good_eps)
        out = {
            f"{prefix}_truth_diagnostics_available": True,
            f"{prefix}_truth_n": int(len(margins)),
            f"{prefix}_true_feasible_count": int(np.sum(feasible)),
            f"{prefix}_true_feasible_rate": float(np.mean(feasible)),
            f"{prefix}_has_true_feasible": bool(np.any(feasible)),
            f"{prefix}_true_safe_good_count": int(np.sum(good)),
            f"{prefix}_has_true_safe_good": bool(np.any(good)),
            f"{prefix}_true_min_margin": float(np.min(margins)),
            f"{prefix}_true_median_margin": float(np.median(margins)),
            f"{prefix}_true_best_regret": (
                float(np.nanmin(regrets)) if np.any(np.isfinite(regrets)) else None
            ),
            f"{prefix}_true_best_feasible_regret": (
                float(np.nanmin(np.where(feasible, regrets, np.nan)))
                if np.any(feasible & np.isfinite(regrets))
                else None
            ),
        }
        try:
            mu_con = self.gpr[1].posterior_mean_many(pool)
            cert = self._certification_result(mu_con, pool)
            posterior_margins = np.asarray(cert.margin, dtype=float)
            out[f"{prefix}_posterior_feasible_count"] = int(
                np.sum(posterior_margins <= 0.0))
            out[f"{prefix}_posterior_min_margin"] = float(np.min(posterior_margins))
            if np.any(feasible & np.isfinite(regrets)):
                feasible_idx = np.where(feasible & np.isfinite(regrets))[0]
                best_pos = int(feasible_idx[int(np.nanargmin(regrets[feasible_idx]))])
                out[f"{prefix}_best_true_feasible_posterior_margin"] = float(
                    posterior_margins[best_pos])
                out[f"{prefix}_best_true_feasible_posterior_feasible"] = bool(
                    posterior_margins[best_pos] <= 0.0)
                out[f"{prefix}_best_true_feasible_regret_audit"] = float(
                    regrets[best_pos])
        except Exception:
            out[f"{prefix}_posterior_audit_available"] = False
        if selected is not None:
            selected = tuple(int(v) for v in selected)
            try:
                sel_margin = self._true_chance_margin(selected)
                sel_regret = (
                    float(self.problem.true_objective(selected) - true_best_obj)
                    if np.isfinite(true_best_obj)
                    else np.nan
                )
                sel_feasible = sel_margin <= 0.0
                out.update({
                    f"{prefix}_selected_true_margin": float(sel_margin),
                    f"{prefix}_selected_true_regret": (
                        float(sel_regret) if np.isfinite(sel_regret) else None
                    ),
                    f"{prefix}_selected_true_feasible": bool(sel_feasible),
                    f"{prefix}_missed_true_feasible": bool(np.any(feasible) and not sel_feasible),
                    f"{prefix}_missed_true_safe_good": bool(np.any(good) and not (
                        sel_feasible
                        and np.isfinite(sel_regret)
                        and sel_regret <= good_eps
                    )),
                })
            except Exception:
                out[f"{prefix}_selected_truth_available"] = False
        return out

    def _truth_pool_decision_margin_audit(
        self,
        pool,
        decision_margins,
        selected=None,
        mu_con=None,
        epistemic_var=None,
        aleatoric_var=None,
        theory_margins=None,
        calibrated_margins=None,
        recommendation_calibrated_margins=None,
        recommendation_calibrated_objectives=None,
        recommendation_calibrated_leverage=None,
        source_margins=None,
        certification_sources=None,
        prefix="recommendation",
    ):
        if not self.config.truth_pool_diagnostics or not pool:
            return {}
        pool = [tuple(int(v) for v in x) for x in pool]
        decision_margins = np.asarray(decision_margins, dtype=float)
        if len(pool) != len(decision_margins):
            return {}
        try:
            _, true_best_obj = self._true_best_feasible_cached()
        except Exception:
            true_best_obj = np.inf
        feasible = []
        regrets = []
        for x in pool:
            try:
                margin = self._true_chance_margin(x)
                obj = float(self.problem.true_objective(x))
            except Exception:
                margin = np.nan
                obj = np.nan
            feasible.append(bool(np.isfinite(margin) and margin <= 0.0))
            regrets.append(obj - true_best_obj if np.isfinite(true_best_obj) else np.nan)
        feasible = np.asarray(feasible, dtype=bool)
        regrets = np.asarray(regrets, dtype=float)
        mu_con = None if mu_con is None else np.asarray(mu_con, dtype=float)
        epistemic_var = (
            None if epistemic_var is None else np.asarray(epistemic_var, dtype=float)
        )
        aleatoric_var = (
            None if aleatoric_var is None else np.asarray(aleatoric_var, dtype=float)
        )
        theory_margins = (
            None if theory_margins is None else np.asarray(theory_margins, dtype=float)
        )
        calibrated_margins = (
            None
            if calibrated_margins is None
            else np.asarray(calibrated_margins, dtype=float)
        )
        recommendation_calibrated_margins = (
            None
            if recommendation_calibrated_margins is None
            else np.asarray(recommendation_calibrated_margins, dtype=float)
        )
        recommendation_calibrated_objectives = (
            None
            if recommendation_calibrated_objectives is None
            else np.asarray(recommendation_calibrated_objectives, dtype=float)
        )
        recommendation_calibrated_leverage = (
            None
            if recommendation_calibrated_leverage is None
            else np.asarray(recommendation_calibrated_leverage, dtype=float)
        )
        source_margins = (
            None
            if source_margins is None
            else np.asarray(source_margins, dtype=float)
        )
        certification_sources = (
            None
            if certification_sources is None
            else np.asarray(certification_sources, dtype=object)
        )
        out = {}
        idx = np.where(feasible & np.isfinite(regrets))[0]
        if len(idx):
            best_pos = int(idx[int(np.nanargmin(regrets[idx]))])
            out[f"{prefix}_best_true_feasible_decision_margin"] = float(
                decision_margins[best_pos])
            out[f"{prefix}_best_true_feasible_decision_feasible"] = bool(
                decision_margins[best_pos] <= 0.0)
            out[f"{prefix}_best_true_feasible_decision_regret"] = float(
                regrets[best_pos])
            if mu_con is not None and len(mu_con) == len(pool):
                out[f"{prefix}_best_true_feasible_mu_con"] = float(mu_con[best_pos])
            if epistemic_var is not None and len(epistemic_var) == len(pool):
                out[f"{prefix}_best_true_feasible_epistemic_var"] = float(
                    epistemic_var[best_pos])
            if aleatoric_var is not None and len(aleatoric_var) == len(pool):
                out[f"{prefix}_best_true_feasible_aleatoric_var"] = float(
                    aleatoric_var[best_pos])
            if theory_margins is not None and len(theory_margins) == len(pool):
                out[f"{prefix}_best_true_feasible_theory_margin"] = float(
                    theory_margins[best_pos])
            if calibrated_margins is not None and len(calibrated_margins) == len(pool):
                out[f"{prefix}_best_true_feasible_calibrated_margin"] = float(
                    calibrated_margins[best_pos])
            if (
                recommendation_calibrated_margins is not None
                and len(recommendation_calibrated_margins) == len(pool)
            ):
                out[f"{prefix}_best_true_feasible_calibrated_rec_margin"] = float(
                    recommendation_calibrated_margins[best_pos])
            if (
                recommendation_calibrated_objectives is not None
                and len(recommendation_calibrated_objectives) == len(pool)
            ):
                out[f"{prefix}_best_true_feasible_calibrated_rec_objective"] = float(
                    recommendation_calibrated_objectives[best_pos])
            if (
                recommendation_calibrated_leverage is not None
                and len(recommendation_calibrated_leverage) == len(pool)
            ):
                lev = recommendation_calibrated_leverage[best_pos]
                out[f"{prefix}_best_true_feasible_calibrated_rec_leverage"] = (
                    None if not np.isfinite(lev) else float(lev)
                )
            if source_margins is not None and len(source_margins) == len(pool):
                out[f"{prefix}_best_true_feasible_source_margin"] = float(
                    source_margins[best_pos])
            if certification_sources is not None and len(certification_sources) == len(pool):
                out[f"{prefix}_best_true_feasible_certification_source"] = str(
                    certification_sources[best_pos])
        if selected is not None:
            selected = tuple(int(v) for v in selected)
            for i, x in enumerate(pool):
                if tuple(int(v) for v in x) == selected:
                    out[f"{prefix}_selected_decision_margin"] = float(
                        decision_margins[i])
                    out[f"{prefix}_selected_decision_feasible"] = bool(
                        decision_margins[i] <= 0.0)
                    if mu_con is not None and len(mu_con) == len(pool):
                        out[f"{prefix}_selected_mu_con"] = float(mu_con[i])
                    if epistemic_var is not None and len(epistemic_var) == len(pool):
                        out[f"{prefix}_selected_epistemic_var"] = float(
                            epistemic_var[i])
                    if aleatoric_var is not None and len(aleatoric_var) == len(pool):
                        out[f"{prefix}_selected_aleatoric_var"] = float(
                            aleatoric_var[i])
                    if theory_margins is not None and len(theory_margins) == len(pool):
                        out[f"{prefix}_selected_theory_margin"] = float(
                            theory_margins[i])
                    if calibrated_margins is not None and len(calibrated_margins) == len(pool):
                        out[f"{prefix}_selected_calibrated_margin"] = float(
                            calibrated_margins[i])
                    if (
                        recommendation_calibrated_margins is not None
                        and len(recommendation_calibrated_margins) == len(pool)
                    ):
                        out[f"{prefix}_selected_calibrated_rec_margin"] = float(
                            recommendation_calibrated_margins[i])
                    if (
                        recommendation_calibrated_objectives is not None
                        and len(recommendation_calibrated_objectives) == len(pool)
                    ):
                        out[f"{prefix}_selected_calibrated_rec_objective"] = float(
                            recommendation_calibrated_objectives[i])
                    if (
                        recommendation_calibrated_leverage is not None
                        and len(recommendation_calibrated_leverage) == len(pool)
                    ):
                        lev = recommendation_calibrated_leverage[i]
                        out[f"{prefix}_selected_calibrated_rec_leverage"] = (
                            None if not np.isfinite(lev) else float(lev)
                        )
                    if source_margins is not None and len(source_margins) == len(pool):
                        out[f"{prefix}_selected_source_margin"] = float(
                            source_margins[i])
                    if certification_sources is not None and len(certification_sources) == len(pool):
                        out[f"{prefix}_selected_certification_source"] = str(
                            certification_sources[i])
                    break
        return out

    def _summarize_truth_pool_diagnostics(self):
        if not self.iteration_log:
            return {}
        rows = [
            row for row in self.iteration_log
            if row.get("candidate_truth_diagnostics_available")
        ]
        if not rows:
            return {"enabled": bool(self.config.truth_pool_diagnostics), "n_logged": 0}
        def mean_bool(key):
            vals = [bool(row.get(key, False)) for row in rows]
            return float(sum(vals) / len(vals)) if vals else 0.0
        def mean_float(key):
            vals = []
            for row in rows:
                val = row.get(key)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(val):
                    vals.append(val)
            return float(np.mean(vals)) if vals else None
        return {
            "enabled": True,
            "n_logged": int(len(rows)),
            "pool_has_true_feasible_rate": mean_bool("candidate_has_true_feasible"),
            "pool_has_true_safe_good_rate": mean_bool("candidate_has_true_safe_good"),
            "selected_true_feasible_rate": mean_bool("candidate_selected_true_feasible"),
            "missed_true_feasible_rate": mean_bool("candidate_missed_true_feasible"),
            "missed_true_safe_good_rate": mean_bool("candidate_missed_true_safe_good"),
            "mean_pool_true_feasible_rate": mean_float("candidate_true_feasible_rate"),
            "mean_pool_best_feasible_regret": mean_float(
                "candidate_true_best_feasible_regret"),
            "mean_selected_true_regret": mean_float("candidate_selected_true_regret"),
            "mean_selected_true_margin": mean_float("candidate_selected_true_margin"),
            "mean_best_true_feasible_posterior_margin": mean_float(
                "candidate_best_true_feasible_posterior_margin"),
            "best_true_feasible_posterior_feasible_rate": mean_bool(
                "candidate_best_true_feasible_posterior_feasible"),
            "recommendation_has_true_feasible_rate": mean_bool(
                "rec_recommendation_has_true_feasible"),
        }

    def run(self, verbose=False):
        t_start = time.time()
        t_progress_start = time.perf_counter()
        start_n = self._initialize_or_resume(verbose=verbose)
        if self.final_log is not None and int(start_n) >= int(self.config.N):
            return self.final_log

        for n in range(int(start_n), self.config.N):
            iteration = n - self.config.n0
            row = {"iteration": iteration, "stage": n}
            t_iter = time.time()
            t_iter_progress = time.perf_counter()
            self._progress_stage_n = int(n)
            self._progress_step_started_at = float(t_iter_progress)
            self._progress_run_started_at = float(t_progress_start)

            row["sequential_basis_refresh"] = self._refresh_sequential_basis()

            t0 = time.time()
            rec_x, rec_details = self._solve_posterior_recommendation()
            row["t_posterior_solve"] = time.time() - t0
            row["recommendation_before"] = list(map(int, rec_x))
            row.update({f"rec_{k}": v for k, v in rec_details.items()})

            t0 = time.time()
            candidates, candidate_sources = self._generate_candidates(iteration)
            row["t_candidate_gen"] = time.time() - t0
            row["n_candidates"] = len(candidates)
            row["llm_prior"] = dict(self._last_llm_prior_info)

            t0 = time.time()
            score = self.acquisition.score(
                candidates,
                self.gpr[0],
                self.gpr[1],
                self.variance_model,
                self.problem,
                observed=self.history,
            )
            exact_mc_samples = self._effective_exact_kg_mc_samples()
            acquisition_mode = str(self.config.acquisition_mode or "additive").lower()
            if exact_mc_samples > 0:
                terminal_pool = self._recommendation_pool()
                exact_kg = self._exact_posterior_update_scores(
                    candidates, terminal_pool)
                score["exact_kg"] = exact_kg
                row["exact_kg_mc_samples"] = int(exact_mc_samples)
                row["exact_kg_jobs"] = int(max(1, self.config.exact_kg_jobs))
                row["exact_kg_parallel_backend"] = (
                    "thread"
                    if int(self.config.exact_kg_jobs) > 1 and len(candidates) > 1
                    else "serial"
                )
                row["acquisition_mode"] = acquisition_mode
                row["certified_terminal_value_before"] = float(getattr(
                    self,
                    "_last_exact_kg_current_value",
                    np.nan,
                ))
                blend = 0.0
                if acquisition_mode == "exact_mc" or self.config.exact_kg_use_score:
                    score["total"] = exact_kg
                else:
                    blend = float(np.clip(self.config.exact_kg_blend, 0.0, 1.0))
                    if acquisition_mode == "blend" and blend <= 0.0:
                        blend = 0.5
                if acquisition_mode == "blend" or (
                    acquisition_mode == "additive" and float(self.config.exact_kg_blend) > 0.0
                ):
                    score["total"] = (
                        (1.0 - blend) * score["total"]
                        + blend * exact_kg
                    )
            selected_idx = int(np.argmax(score["total"]))
            x_selected = candidates[selected_idx]
            row["t_kg_compute"] = time.time() - t0
            row["x_selected"] = list(map(int, x_selected))
            row["candidate_source_selected"] = candidate_sources.get(
                tuple(x_selected), "unknown")
            row["score_selected"] = float(score["total"][selected_idx])
            row.update(self._truth_pool_diagnostics(
                candidates,
                selected=x_selected,
                prefix="candidate",
            ))
            row["v_C_plus_selected"] = float(
                self.variance_model.predict_certification_variance(
                    1,
                    x_selected,
                    self.problem,
                )
            )
            selected_decomp = self.variance_model.predict_decomposition(
                1,
                x_selected,
                self.problem,
            )
            selected_cumulative = selected_decomp.get("cumulative") or {}
            row["v_C_plus_source"] = (
                "provider_cumulative"
                if selected_cumulative.get("provider_active")
                else "fallback_hvd"
            )
            row["selected_cumulative_blocks"] = selected_cumulative.get("fitted_blocks")
            row["kg_obj_selected"] = float(score["kg_obj"][selected_idx])
            row["kg_obj_scaled_selected"] = float(score["kg_obj_scaled"][selected_idx])
            row["kg_feas_selected"] = float(score["kg_feas"][selected_idx])
            row["kg_var_selected"] = float(score["kg_var"][selected_idx])
            row["kg_constraint_epistemic_selected"] = float(
                score["kg_constraint_epistemic"][selected_idx])
            row["kg_coupling_selected"] = float(score["kg_coupling"][selected_idx])
            row["kg_coupling_raw_selected"] = float(
                score["kg_coupling_raw"][selected_idx])
            row["kg_coupling_gate_selected"] = float(
                score["kg_coupling_gate"][selected_idx])
            if "exact_kg" in score:
                row["exact_kg_selected"] = float(score["exact_kg"][selected_idx])

            x_arr = np.asarray(x_selected, dtype=int)
            mu_before = [self.gpr[i].posterior_mean(x_arr) for i in range(2)]
            epistemic_before = [
                self.gpr[i].posterior_var(x_arr) for i in range(2)
            ]
            sigma2_before = [
                self.variance_model.predict_variance(i, x_arr, self.problem)
                for i in range(2)
            ]
            row["mu_before"] = [float(v) for v in mu_before]
            row["epistemic_before"] = [float(v) for v in epistemic_before]
            row["sigma2_before"] = [float(v) for v in sigma2_before]

            t0 = time.time()
            y = self._simulate_and_store(x_selected)
            row["t_simulate"] = time.time() - t0
            row["Y_observed"] = [float(v) for v in y]

            t0 = time.time()
            for i in range(2):
                self.gpr[i].update(x_arr, y[i], sigma2_before[i])
            row["adaptive_sparsity"] = [
                model.adaptive_sparsity_diagnostics()
                for model in self.gpr
            ]
            hvd_details = []
            for i in range(2):
                replicate_values = [
                    float(np.asarray(observed, dtype=float)[i])
                    for observed in self.observations.get(
                        tuple(int(v) for v in x_selected), [])
                ]
                replicate_variance = (
                    float(np.var(replicate_values, ddof=1))
                    if len(replicate_values) >= 2
                    else None
                )
                hvd_details.append(self.variance_model.update(
                    i,
                    x_arr,
                    y[i],
                    mu_before[i],
                    self.gpr[i],
                    self.problem,
                    epistemic_var=epistemic_before[i],
                    replicate_variance=replicate_variance,
                ))
            row["t_update"] = time.time() - t0
            row["hvd_update"] = hvd_details
            row["n_visited"] = len(self.gpr[0].sampled_set)

            t0 = time.time()
            eval_interval = int(self.config.evaluate_interval)
            if eval_interval > 0 and (
                iteration % eval_interval == 0 or n == self.config.N - 1
            ):
                rec_x_after, rec_after = self._solve_posterior_recommendation()
                eval_after = self._evaluate_recommendation(rec_x_after)
                row["recommendation_after"] = list(map(int, rec_x_after))
                row["eval"] = {**rec_after, **eval_after}
            row["t_eval"] = time.time() - t0
            row["t_total"] = time.time() - t_iter
            self.iteration_log.append(row)
            self._save_checkpoint(n + 1, reason="iteration")
            self._progress_emit(
                n=n,
                frac=1.0,
                kind="iteration_done",
                started_at=t_iter_progress,
                run_started_at=t_progress_start,
                extra=(
                    f"kg={row['t_kg_compute']:.3f}s "
                    f"cand={row['t_candidate_gen']:.3f}s "
                    f"sim={row['t_simulate']:.3f}s "
                    f"update={row['t_update']:.3f}s "
                    f"eval={row['t_eval']:.3f}s"
                ),
            )
            if verbose:
                print(
                    f"iter={iteration:03d} x={x_selected} "
                    f"score={row['score_selected']:.4g}"
                )

        candidate_source_counts = {}
        llm_status_counts = {}
        llm_gates = []
        for row in self.iteration_log:
            source = str(row.get("candidate_source_selected", "unknown"))
            candidate_source_counts[source] = candidate_source_counts.get(source, 0) + 1
            info = row.get("llm_prior") or {}
            status = str(info.get("status", "missing"))
            llm_status_counts[status] = llm_status_counts.get(status, 0) + 1
            if "gate" in info:
                try:
                    llm_gates.append(float(info.get("gate", 0.0)))
                except (TypeError, ValueError):
                    pass
        llm_prior_summary = {
            "enabled": bool(self.llm_prior is not None),
            "last": dict(self._last_llm_prior_info),
            "status_counts": llm_status_counts,
            "called_count": int(sum(
                count for status, count in llm_status_counts.items()
                if status not in {"disabled", "skipped", "not_called", "missing"}
            )),
            "ok_count": int(llm_status_counts.get("ok", 0)),
            "selected_count": int(candidate_source_counts.get("llm_prior", 0)),
            "gate_mean": float(np.mean(llm_gates)) if llm_gates else 0.0,
            "gate_max": float(np.max(llm_gates)) if llm_gates else 0.0,
        }

        final_x, final_post = self._solve_posterior_recommendation()
        final_eval = self._evaluate_recommendation(final_x)
        self.final_log = {
            **final_post,
            **final_eval,
            "total_time_sec": float(time.time() - t_start),
            "n_simulations": int(len(self.history)),
            "n_distinct_solutions": int(len(self.gpr[0].sampled_set)),
            "stage_times": summarize_stage_times(self.iteration_log),
            "candidate_source_counts": candidate_source_counts,
            "llm_prior": llm_prior_summary,
            "truth_pool_diagnostics": self._summarize_truth_pool_diagnostics(),
            "variance": self.variance_model.diagnostics(),
            "adaptive_sparsity": [
                model.adaptive_sparsity_diagnostics()
                for model in self.gpr
            ],
            "meta_basis": (
                self.problem.meta_basis_diagnostics()
                if hasattr(self.problem, "meta_basis_diagnostics")
                else None
            ),
            "cumulative_risk_provider": (
                self.problem.cumulative_risk_provider_status()
                if hasattr(self.problem, "cumulative_risk_provider_status")
                else {"status": "missing_provider"}
            ),
            "mainline_high_dependence": bool(
                self.config.variance_mode == "factor"
                and self.config.certification_mode == "theory"
                and self.config.use_state_coupling
                and self.config.use_state_basis
                and str(self.config.acquisition_mode).lower() == "exact_mc"
            ),
            "numeric_backend": self.gpr[0].backend_status(),
            "config": asdict(self.config),
        }
        self._save_checkpoint(self.config.N, reason="final", force=True)
        return self.final_log
