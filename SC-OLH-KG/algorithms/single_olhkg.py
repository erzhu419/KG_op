"""Single-objective chance-constrained OLH-KG / SC-OLH-KG algorithm."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import multiprocessing
import os
from pathlib import Path
import pickle
import time
import zlib

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import score_decision_backend
from acquisition.olhkg import OLHKGAcquisition
from core.certification import CertificationResult, conservative_chance_margin
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
from core.designs import (
    COMMON_SOBOL_SEED_OFFSET,
    common_sobol_integer_design,
    integer_design_fingerprint,
    next_sobol_integer_candidate,
    sobol_integer_sequence,
)
from core.gpr import (
    ParametricGPR,
    normalize_mixture_weights,
    posterior_mixture_weights,
)
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
from representation.boundary_coordinate import (
    select_boundary_coordinate_candidates,
)
from representation.task_posterior import (
    FiniteTaskModelEnsemble,
    FiniteTaskPosterior,
    FiniteTaskSensitivityPosterior,
    TaskExpertState,
)
from variance.orthogonal_hvd import OrthogonalHVD


_FORK_EXACT_KG_CONTEXT = None
_FORK_TERMINAL_ROLLOUT_CONTEXT = None
_FORK_TERMINAL_DEPTH3_CONTEXT = None
SIMULATION_STREAM_TAG = 0x53434F4C
PROPOSAL_STREAM_TAG = 0x50524F50
EXACT_KG_STREAM_TAG = 0x45584B47


def _fork_exact_kg_candidate(x):
    """Evaluate one exact-KG candidate from a fork-inherited context."""
    if _FORK_EXACT_KG_CONTEXT is None:
        raise RuntimeError("fork exact-KG worker has no inherited context")
    (
        algorithm,
        common_z,
        terminal_pool,
        current_value,
        expert_uniform,
        sample_weights,
    ) = _FORK_EXACT_KG_CONTEXT
    return algorithm._exact_posterior_update_score_one(
        x,
        common_z,
        terminal_pool,
        current_value,
        expert_uniform,
        sample_weights,
        return_diagnostics=True,
    )


def _fork_exact_kg_candidate_chunk(payload):
    """Evaluate one candidate/sample chunk from fork-inherited state."""

    if _FORK_EXACT_KG_CONTEXT is None:
        raise RuntimeError("fork exact-KG worker has no inherited context")
    candidate, sample_indices = payload
    (
        algorithm,
        common_z,
        terminal_pool,
        current_value,
        expert_uniform,
        sample_weights,
    ) = _FORK_EXACT_KG_CONTEXT
    indices = np.asarray(sample_indices, dtype=int)
    chunk_weights = np.asarray(sample_weights, dtype=float)[indices]
    mass = float(np.sum(chunk_weights))
    result = algorithm._exact_posterior_update_score_one(
        candidate,
        np.asarray(common_z, dtype=float)[indices],
        terminal_pool,
        current_value,
        np.asarray(expert_uniform, dtype=float)[indices],
        chunk_weights,
        return_diagnostics=True,
    )
    return mass, result


def _fork_terminal_rollout_action(action_index):
    """Evaluate one root Bellman action from a fork-inherited state."""
    if _FORK_TERMINAL_ROLLOUT_CONTEXT is None:
        raise RuntimeError("fork terminal-rollout worker has no inherited context")
    (
        algorithm,
        state,
        arms,
        terminal_pool,
        depth,
        node_code,
        common_z,
        common_uniform,
        sample_weights,
    ) = _FORK_TERMINAL_ROLLOUT_CONTEXT
    return algorithm._terminal_rollout_expected_value_for_action(
        state,
        arms[int(action_index)],
        arms,
        terminal_pool,
        depth,
        node_code,
        int(action_index),
        common_z,
        common_uniform,
        sample_weights,
    )


def _fork_terminal_depth3_prefix(payload):
    """Evaluate one depth-three (root sample, second action) prefix."""
    if _FORK_TERMINAL_DEPTH3_CONTEXT is None:
        raise RuntimeError("fork depth-three rollout has no inherited context")
    (
        algorithm,
        state,
        arms,
        terminal_pool,
        node_code,
        common_z,
        common_uniform,
    ) = _FORK_TERMINAL_DEPTH3_CONTEXT
    root_action_index, root_sample_index, second_action_index = payload
    return algorithm._terminal_rollout_depth3_prefix_value(
        state,
        arms,
        terminal_pool,
        node_code,
        common_z,
        common_uniform,
        int(root_action_index),
        int(root_sample_index),
        int(second_action_index),
    )


@dataclass
class SingleOLHKGConfig:
    implementation_contract_id: str = "unversioned"
    theory_contract_id: str = "unversioned"
    N: int = 30
    n0: int = 8

    def __setstate__(self, state):
        """Keep pre-contract checkpoints readable without relabeling them."""
        self.__dict__.update(state)
        if "implementation_contract_id" not in self.__dict__:
            self.implementation_contract_id = "unversioned"
        if "theory_contract_id" not in self.__dict__:
            self.theory_contract_id = "unversioned"

    initial_design: str = "auto"
    initial_design_points: tuple[tuple[int, ...], ...] = ()
    initial_design_fingerprint: str = ""
    initial_design_source_archive_fingerprint: str = ""
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
    hvd_use_cumulative_provider: bool = True
    hvd_cumulative_transfer_mode: str = "scalar"
    hvd_source_task_weight_mode: str = "independent"
    hvd_cumulative_target_evidence_mode: str = "replication_only"
    hvd_singleton_evidence_mode: str = "in_sample_residual"
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
    recommendation_calibration_max_effective_fraction: float = 0.35
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
    decision_backend: str = "legacy"
    decision_risk_penalty: float = 5.0
    decision_aleatoric_mode: str = "certification_upper"
    decision_violation_loss_mode: str = "positive_part"
    decision_ambiguity_mode: str = "kl_robust"
    decision_source_utility_weight: float = 1.0
    decision_backend_seed_offset: int = 470_003
    decision_recommend_observed_only: bool = True
    exact_kg_mc_samples: int = 8
    exact_kg_jobs: int = 1
    exact_kg_parallel_backend: str = "thread"
    exact_kg_sampling_mode: str = "iid"
    exact_kg_clip_negative: bool = True
    exact_kg_use_score: bool = False
    exact_kg_blend: float = 0.0
    exact_kg_terminal_mode: str = "hard_certified"
    decision_contract_mode: str = "legacy"
    terminal_bayes_violation_penalty: float = 5.0
    terminal_frontier_candidate_count: int = 0
    tcb_v2_mode: str = "off"
    tcb_v2_frontier_count: int = 1
    finalist_terminal_value_mode: str = "model_default"
    task_posterior_mode: str = "off"
    source_discrepancy_update: bool = True
    task_posterior_initial_design: bool = True
    task_posterior_boundary_bracket_fraction: float = 0.0
    task_posterior_mandatory_universal_count: int = 0
    task_posterior_pilot_count: int = -1
    task_posterior_temperature: float = 0.5
    task_posterior_temperature_decay: float = 0.5
    task_posterior_boundary_score_weight: float = 0.25
    task_posterior_objective_score_weight: float = 0.25
    task_posterior_constraint_score_weight: float = 1.0
    task_posterior_safe_generalized: bool = False
    task_posterior_safe_boundary_score_weight: float = 1.0
    task_posterior_safe_pairwise_score_weight: float = 1.0
    task_posterior_safe_pairwise_max_history: int = 16
    task_posterior_safe_pairwise_probability_floor: float = 1e-6
    task_posterior_kl_radius_numerator: float = 0.5
    task_posterior_confidence_delta: float = 0.05
    task_posterior_max_kl_radius: float = 4.0
    task_posterior_robust_certificate_mode: str = "separable"
    certification_head_authority: str = "task_joint"
    task_posterior_prior_protection_numerator: float = 0.0
    task_posterior_prior_protection_max: float = 0.5
    task_posterior_local_kernel_expert: bool = False
    task_posterior_candidate_count: int = 0
    task_posterior_recommendation_count: int = 0
    task_posterior_proposal_pool_size: int = 1024
    task_posterior_proposal_exploration: float = 0.10
    task_posterior_proposal_min_per_expert: int = 2
    task_posterior_sensitivity_mode: str = "off"
    task_variance_posterior_mode: str = "shared"
    task_latent_inference_mode: str = "shadow"
    task_latent_calibration_mode: str = "source_profiles"
    constraint_uncertain_candidate_count: int = 0
    constraint_uncertain_pool_size: int = 300
    constraint_uncertain_state_pool_fraction: float = 0.25
    constraint_uncertain_use_calibration: bool = False
    constraint_epistemic_margin_softening: float = 3.0
    replication_candidate_count: int = 0
    replication_max_per_solution: int = 5
    replication_margin_softening: float = 3.0
    adaptive_replication_voi: bool = False
    evaluate_or_replicate_new_action_count: int = 1
    evaluate_or_replicate_new_action_policy: str = "canonical_sobol"
    evaluate_or_replicate_baseline_new_action_count: int = 0
    policy_improvement_mode: str = "off"
    policy_improvement_mc_error_bound: float = 0.0
    policy_improvement_rollout_depth: int = 1
    policy_improvement_rollout_max_arms: int = 4
    policy_improvement_rollout_mc_samples: int = 2
    policy_improvement_rollout_mc_error_bound: float = 0.0
    posterior_dominance_enabled: bool = False
    posterior_dominance_delta: float = 0.05
    posterior_dominance_min_mean_gain: float = 0.0
    posterior_dominance_initialization: str = "risk"
    certification_recheck_top_k: int = 0
    certification_recheck_min_replicates: int = 3
    certification_recheck_soft_margin_scale: float = 2.0
    certification_recheck_variance_prior_df: float = 2.0
    finalist_replication_budget: int = 0
    finalist_replication_count: int = 2
    finalist_observed_safety_count: int = 1
    finalist_replication_min_replicates: int = 2
    finalist_replication_delta: float = 0.05
    finalist_replication_variance_prior_df: float = 2.0
    finalist_replication_expert_stratified: bool = False
    finalist_replication_adaptive_race: bool = False
    finalist_replication_fixed_universe: bool = False
    finalist_replication_policy: str = "legacy"
    finalist_empirical_override: str = "legacy"
    finalist_frontier_policy: str = "legacy"
    finalist_terminal_max_arms: int = 4
    finalist_terminal_mc_samples: int = 2
    observed_incumbent_use_replicate_variance: bool = False
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
    source_constraint_mean_coefficient_prior: bool = False
    source_constraint_mean_hyperlaw_mode: str = "single_gaussian_draw"
    source_constraint_mean_adaptation_mode: str = "frozen"
    source_constraint_mean_deviation_mode: str = "raw_independent"
    source_constraint_mean_misspecification_mode: str = "none"
    source_constraint_mean_misspecification_prior_df: float = 4.0
    source_constraint_mean_misspecification_ridge: float = 1.0
    source_constraint_mean_misspecification_max_scale: float = 100.0
    source_constraint_mean_misspecification_delta: float = 0.05
    source_constraint_mean_confidence_mode: str = "model"
    source_constraint_mean_confidence_delta: float = 0.05
    source_constraint_mean_contrast_scale: float = 1.0
    source_constraint_mean_role_epistemic_mode: str = "none"
    source_constraint_mean_null_weight: float = 0.5
    source_constraint_mean_null_geometry: str = "isotropic"
    source_constraint_mean_null_geometry_ridge: float = 1e-3
    source_constraint_mean_evidence_temperature: float = 1.0
    source_constraint_mean_structure_score_mode: str = "marginal_likelihood"
    source_constraint_mean_residual_rank_posterior: bool = False
    source_constraint_mean_residual_rank_prior: str = "0.70,0.20,0.10"
    source_constraint_mean_residual_rank_inactive_variance: float = 1e-12
    boundary_coordinate_candidate_count: int = 0
    boundary_coordinate_pool_size: int = 512
    boundary_coordinate_safe_fraction: float = 0.30
    boundary_coordinate_boundary_fraction: float = 0.40
    boundary_coordinate_coverage_fraction: float = 0.30
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
        self._canonical_sobol_sequence = None
        self._last_canonical_sobol_candidate = None

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
            use_cumulative_provider=self.config.hvd_use_cumulative_provider,
            cumulative_transfer_mode=(
                self.config.hvd_cumulative_transfer_mode),
            cumulative_source_task_weight_mode=(
                self.config.hvd_source_task_weight_mode),
            cumulative_target_evidence_mode=(
                self.config.hvd_cumulative_target_evidence_mode),
            singleton_evidence_mode=(
                self.config.hvd_singleton_evidence_mode),
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
        self._last_task_proposal_info = {
            "status": "not_called",
            "requested": 0,
        }
        self._last_boundary_coordinate_proposal_info = {
            "status": "not_called",
            "requested": 0,
        }
        self._boundary_raw_pool_audit_cache = None
        self._task_initial_design_info = {
            "status": "not_called",
            "requested": 0,
        }
        self._certification_recheck_targets: list[tuple[int, ...]] = []
        self._finalist_replication_initialized = False
        self._finalist_replication_targets: list[tuple[int, ...]] = []
        self._finalist_replication_labels: list[str] = []
        self._finalist_replication_frozen_stage: int | None = None
        self._finalist_replication_active_target: tuple[int, ...] | None = None
        self._finalist_replication_active_label: str | None = None
        self._finalist_replication_refresh_history: list[dict] = []
        self._finalist_replication_pool: list[tuple[int, ...]] = []
        self._last_terminal_pool: list[tuple[int, ...]] = []
        self._posterior_dominance_incumbent: tuple[int, ...] | None = None
        self._posterior_dominance_history: list[dict] = []

        self.observations: dict[tuple[int, ...], list[np.ndarray]] = {}
        self.history: list[tuple[tuple[int, ...], np.ndarray]] = []
        self.iteration_log: list[dict] = []
        self.pre_sampling_log: dict | None = None
        self.final_log: dict | None = None
        self._true_best_feasible_cache = None
        self.task_ensemble: FiniteTaskModelEnsemble | None = None

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

    def _task_prior_initial_samples(self, n):
        n = max(0, int(n))
        required = (
            "task_posterior_expert_specs",
            "task_expert_proposal_candidates",
        )
        if (
            n == 0
            or not self._task_posterior_requested()
            or not self.config.task_posterior_initial_design
            or any(not hasattr(self.problem, name) for name in required)
        ):
            self._task_initial_design_info = {
                "status": "disabled",
                "requested": int(n),
            }
            return []
        try:
            specs = list(self.problem.task_posterior_expert_specs(
                include_local_kernel=bool(
                    self.config.task_posterior_local_kernel_expert),
            ))
        except TypeError:
            specs = list(self.problem.task_posterior_expert_specs())
        prior = FiniteTaskPosterior(
            [str(spec["name"]) for spec in specs],
            [float(spec.get("prior_weight", 1.0)) for spec in specs],
            temperature=0.0,
        )
        mandatory_requested = min(
            n,
            max(0, int(
                self.config.task_posterior_mandatory_universal_count)),
        )
        mandatory_rows = []
        if (
            mandatory_requested > 0
            and "universal_coordinate" in prior.expert_names
        ):
            initial_universal = getattr(
                self.problem,
                "task_initial_universal_candidates",
                None,
            )
            if initial_universal is None:
                initial_universal = lambda **kwargs: (
                    self.problem.task_expert_proposal_candidates(
                        "universal_coordinate", **kwargs))
            mandatory_rows = unique_candidates(
                initial_universal(
                    n=mandatory_requested,
                    rng=self.rng,
                    pool_size=max(
                        mandatory_requested,
                        int(self.config.task_posterior_proposal_pool_size),
                    ),
                )
            )[:mandatory_requested]
        bracket_fraction = float(np.clip(
            self.config.task_posterior_boundary_bracket_fraction,
            0.0,
            1.0,
        ))
        bracket_desired = int(round(bracket_fraction * n))
        bracket_requested = min(
            bracket_desired,
            max(0, n - len(mandatory_rows)),
        )
        bracket_rows = []
        if bracket_requested > 0 and hasattr(
            self.problem, "task_boundary_bracket_candidates"
        ):
            bracket_rows = unique_candidates(
                self.problem.task_boundary_bracket_candidates(
                    n=bracket_requested,
                    rng=self.rng,
                    pool_size=max(
                        int(self.config.task_posterior_proposal_pool_size),
                        16 * bracket_requested,
                    ),
                )
            )[:bracket_requested]
        rows = unique_candidates(mandatory_rows + bracket_rows)
        expert_budget = max(0, n - len(rows))
        residual_names = [
            name for name in prior.expert_names
            if not (mandatory_rows and name == "universal_coordinate")
        ]
        spec_by_name = {
            str(spec["name"]): spec for spec in specs
        }
        if expert_budget > 0 and residual_names:
            residual_prior = FiniteTaskPosterior(
                residual_names,
                [
                    float(spec_by_name[name].get("prior_weight", 1.0))
                    for name in residual_names
                ],
                temperature=0.0,
            )
            residual_allocation = residual_prior.proposal_allocation(
                expert_budget,
                exploration=1.0,
                minimum_per_expert=(
                    self.config.task_posterior_proposal_min_per_expert),
            )
        else:
            residual_allocation = {name: 0 for name in residual_names}
        allocation = {
            name: int(residual_allocation.get(name, 0))
            for name in prior.expert_names
        }
        generated = {
            "universal_coordinate": int(len(mandatory_rows))
        } if mandatory_rows else {}
        for name in residual_names:
            count = int(allocation.get(name, 0))
            expert_rows = self.problem.task_expert_proposal_candidates(
                name,
                n=count,
                rng=self.rng,
                pool_size=max(
                    count,
                    int(self.config.task_posterior_proposal_pool_size),
                ),
            ) if count > 0 else []
            expert_rows = unique_candidates(expert_rows)[:count]
            generated[name] = int(generated.get(name, 0) + len(expert_rows))
            rows.extend(expert_rows)
        rows = unique_candidates(rows)
        self._task_initial_design_info = {
            "status": "generated",
            "requested": int(n),
            "mandatory_universal_requested": int(mandatory_requested),
            "mandatory_universal_generated": int(len(mandatory_rows)),
            "boundary_bracket_fraction": float(bracket_fraction),
            "boundary_bracket_desired": int(bracket_desired),
            "boundary_bracket_requested": int(bracket_requested),
            "boundary_bracket_generated": int(len(bracket_rows)),
            "boundary_bracket_diagnostics": (
                self.problem.task_boundary_bracket_diagnostics()
                if hasattr(self.problem, "task_boundary_bracket_diagnostics")
                else {"status": "unavailable"}
            ),
            "allocation": allocation,
            "generated": generated,
            "n_unique": int(len(rows)),
            "source_only": True,
            "target_oracle_used": False,
        }
        return rows[:n]

    def _initial_samples(self):
        initial_design = str(self.config.initial_design or "auto").lower()
        if initial_design == "source_informed":
            raw = np.asarray(self.config.initial_design_points, dtype=float)
            expected_shape = (int(self.config.n0), int(self.problem.d))
            if raw.shape != expected_shape:
                raise ValueError(
                    "source-informed initial design must have shape "
                    f"{expected_shape}, got {raw.shape}"
                )
            if not np.all(np.isfinite(raw)) or not np.all(raw == np.rint(raw)):
                raise ValueError(
                    "source-informed initial design must contain finite integers"
                )
            samples = [tuple(map(int, row)) for row in raw]
            if len(set(samples)) != int(self.config.n0):
                raise ValueError(
                    "source-informed initial design points must be unique")
            lo, hi = self.problem.int_bounds()
            integer = np.asarray(samples, dtype=int)
            if np.any(integer < np.asarray(lo)) or np.any(
                integer > np.asarray(hi)
            ):
                raise ValueError(
                    "source-informed initial design contains out-of-bounds points"
                )
            fingerprint = integer_design_fingerprint(samples)
            expected_fingerprint = str(
                self.config.initial_design_fingerprint or "")
            if not expected_fingerprint:
                raise ValueError(
                    "source-informed initial design requires a frozen fingerprint"
                )
            if fingerprint != expected_fingerprint:
                raise ValueError(
                    "source-informed initial design fingerprint mismatch")
            source_fingerprint = str(
                self.config.initial_design_source_archive_fingerprint or "")
            if not source_fingerprint:
                raise ValueError(
                    "source-informed initial design requires an archive fingerprint"
                )
            self._task_initial_design_info = {
                "status": "loaded",
                "design_kind": "frozen_source_informed_rank_spanning",
                "requested": int(self.config.n0),
                "n_unique": int(len(samples)),
                "seed": int(self.config.seed),
                "fingerprint": fingerprint,
                "source_archive_fingerprint": source_fingerprint,
                "source_prior_used": True,
                "source_only": True,
                "target_labels_used": False,
                "problem_specific_hook_used": False,
                "target_oracle_used": False,
            }
            return samples
        if initial_design == "common_sobol":
            samples = common_sobol_integer_design(
                self.problem,
                self.config.n0,
                self.config.seed,
            )
            self._task_initial_design_info = {
                "status": "generated",
                "design_kind": "common_sobol",
                "requested": int(self.config.n0),
                "n_unique": int(len(samples)),
                "seed": int(self.config.seed),
                "seed_offset": int(COMMON_SOBOL_SEED_OFFSET),
                "fingerprint": integer_design_fingerprint(samples),
                "source_prior_used": False,
                "problem_specific_hook_used": False,
                "target_oracle_used": False,
            }
            return samples
        if initial_design != "auto":
            raise ValueError(
                "initial_design must be 'auto', 'common_sobol', or "
                "'source_informed'"
            )
        samples = []
        if self.config.use_problem_initial_samples and hasattr(
            self.problem, "initial_samples"
        ):
            samples.extend(self._task_prior_initial_samples(self.config.n0))
            if len(samples) < self.config.n0:
                samples.extend(self.problem.initial_samples(
                    n=self.config.n0 - len(samples),
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
        evaluation_index = int(len(self.history))
        simulation_rng = np.random.default_rng(np.random.SeedSequence([
            int(self.config.seed),
            evaluation_index,
            SIMULATION_STREAM_TAG,
        ]))
        y = self.problem.simulate(x, simulation_rng)
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

        self._configure_hvd_source_task_posterior(
            self.variance_model, self.gpr)
        self.variance_model.initialize(
            samples, self.observations, self.gpr, self.problem)
        self._initialize_task_ensemble(samples)
        self._initialize_certification_recheck_targets(samples)

    def _task_posterior_requested(self):
        mode = str(self.config.task_posterior_mode or "off").lower()
        return mode not in ("", "off", "none", "disabled")

    def _fit_initial_task_expert_gpr(self, model, output_index, samples):
        y = np.asarray([
            float(np.asarray(self.observations[tuple(x)][0], dtype=float)[
                int(output_index)])
            for x in samples
        ], dtype=float)
        phi = model.basis_matrix(samples)
        basis_map = getattr(model, "basis_map", None)
        try:
            if basis_map is not None and hasattr(
                basis_map, "initial_parametric_coefficients"
            ):
                beta = basis_map.initial_parametric_coefficients(phi, y)
            else:
                beta = np.linalg.lstsq(phi, y, rcond=1e-3)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(phi.shape[1], dtype=float)
        residual = y - phi @ beta
        nominal_noise = max(
            float(getattr(self.problem, "sigma_level", 0.0)) ** 2,
            1e-6,
        )
        source_posterior = self._fit_source_constraint_coefficient_posterior(
            model,
            output_index,
            samples,
            y,
            nominal_noise,
        )
        if source_posterior is not None:
            return
        lambda_data = max(float(np.var(residual)), nominal_noise)
        prior_var = max(float(np.var(beta)), 1e-6)
        adaptive_spec = None
        if basis_map is not None and hasattr(
            basis_map, "adaptive_sparsity_spec"
        ):
            adaptive_spec = basis_map.adaptive_sparsity_spec(
                self.observations)
        if adaptive_spec is not None:
            model.enable_adaptive_sparsity(
                adaptive_spec,
                samples,
                y,
                np.full(len(samples), lambda_data, dtype=float),
                deviation_variance=lambda_data,
            )
            return
        model.set_parametric_prior(beta, lambda_data, prior_var)
        # The pilot labels define the empirical prior mean, but the GPR state
        # still has to be conditioned on them so solution-specific deviations
        # and covariance reflect actually observed points.
        for x, target in zip(samples, y):
            model.update(x, float(target), lambda_data)

    def _source_constraint_coefficient_prior(self, model, output_index):
        """Read a frozen source coefficient law without consulting target truth."""

        if (
            int(output_index) != 1
            or not self.config.source_constraint_mean_coefficient_prior
        ):
            return None
        basis_map = getattr(model, "basis_map", None)
        if basis_map is None or not hasattr(
            basis_map, "source_parametric_prior"
        ):
            return None
        prior = copy.deepcopy(basis_map.source_parametric_prior())
        hyperlaw_mode = str(
            self.config.source_constraint_mean_hyperlaw_mode
            or "single_gaussian_draw"
        ).strip().lower()
        if hyperlaw_mode not in {
            "single_gaussian_draw",
            "shared_low_rank_discrepancy",
            "shared_low_rank_predictive",
            "grouped_task_discrepancy",
            "grouped_task_predictive",
            "grouped_task_deconvolved",
            "grouped_task_deconvolved_predictive",
        }:
            raise ValueError(
                "source constraint mean hyperlaw mode must be "
                "single_gaussian_draw, shared_low_rank_discrepancy, or "
                "shared_low_rank_predictive, grouped_task_discrepancy, or "
                "grouped_task_predictive, grouped_task_deconvolved, or "
                "grouped_task_deconvolved_predictive"
            )
        if hyperlaw_mode != "single_gaussian_draw":
            key = {
                "shared_low_rank_discrepancy": "shared_low_rank_prior",
                "shared_low_rank_predictive": (
                    "shared_low_rank_predictive_prior"),
                "grouped_task_discrepancy": "grouped_task_prior",
                "grouped_task_predictive": (
                    "grouped_task_predictive_prior"),
                "grouped_task_deconvolved": (
                    "grouped_task_deconvolved_prior"),
                "grouped_task_deconvolved_predictive": (
                    "grouped_task_deconvolved_predictive_prior"),
            }[hyperlaw_mode]
            selected = prior.get(key)
            if selected is None:
                raise RuntimeError(
                    f"source basis does not expose {hyperlaw_mode}"
                )
            prior = copy.deepcopy(selected)
        diagnostics = dict(prior.get("diagnostics", {}))
        diagnostics.update({
            "configured_hyperlaw_mode": hyperlaw_mode,
            "shared_low_rank_prior_selected": bool(
                hyperlaw_mode != "single_gaussian_draw"),
            "finite_source_predictive_prior_selected": bool(
                hyperlaw_mode in {
                    "shared_low_rank_predictive",
                    "grouped_task_predictive",
                    "grouped_task_deconvolved_predictive",
                }),
            "grouped_task_prior_selected": bool(
                hyperlaw_mode.startswith("grouped_task_")),
            "random_effects_deconvolution_selected": bool(
                hyperlaw_mode.startswith("grouped_task_deconvolved")),
            "target_oracle_used": False,
        })
        prior["diagnostics"] = diagnostics
        mean = np.asarray(prior.get("mean"), dtype=float).reshape(-1)
        covariance = np.asarray(prior.get("covariance"), dtype=float)
        if len(mean) != model.p or covariance.shape != (model.p, model.p):
            raise RuntimeError(
                "source constraint coefficient prior does not match GPR basis"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise FloatingPointError("source constraint coefficient prior is non-finite")
        prior["mean"] = mean
        prior["covariance"] = covariance
        prior["deviation_variance"] = max(
            float(prior.get("deviation_variance", 1e-6)), 1e-12)
        return prior

    def _calibrate_source_constraint_deviation(self, model, component):
        """Place transferable source discrepancy in the latent coefficient law.

        A source residual is not necessarily an independent deviation attached
        to every raw policy.  Under ``latent_shared`` it is split into a
        coefficient-space discrepancy shared by policies with similar mean
        features and a finite residual floor.  At reference feature energy
        ``E[||phi||^2] = p`` the two pieces preserve the original source
        residual variance, while charged target observations can contract the
        shared coefficient uncertainty.
        """

        calibrated = copy.deepcopy(component)
        mode = str(
            self.config.source_constraint_mean_deviation_mode
            or "raw_independent"
        ).strip().lower()
        if mode not in {"raw_independent", "latent_shared"}:
            raise ValueError(
                "source constraint mean deviation mode must be "
                "raw_independent or latent_shared"
            )
        original = max(
            float(calibrated.get("deviation_variance", 1e-6)), 1e-12)
        diagnostics = dict(calibrated.get("diagnostics", {}))
        n_records = max(int(diagnostics.get(
            "source_record_count",
            diagnostics.get("n_records", 1),
        )), 1)
        if mode == "raw_independent":
            diagnostics.update({
                "source_deviation_mode": mode,
                "source_original_deviation_variance": float(original),
                "source_latent_discrepancy_trace": 0.0,
                "source_residual_floor": float(original),
                "source_reference_predictive_variance": float(original),
            })
            calibrated["diagnostics"] = diagnostics
            return calibrated

        residual_floor = original / float(n_records)
        latent_mass = max(original - residual_floor, 0.0)
        active_coefficients = diagnostics.get(
            "role_assignment_active_coefficients")
        if active_coefficients is None:
            active = np.arange(int(model.p), dtype=int)
        else:
            active = np.asarray(active_coefficients, dtype=int).reshape(-1)
            active = np.unique(active[
                (active >= 0) & (active < int(model.p))
            ])
            if len(active) == 0:
                raise RuntimeError(
                    "role-assignment component has no active coefficient")
        feature_dim = max(int(len(active)), 1)
        covariance = np.asarray(
            calibrated["covariance"], dtype=float).copy()
        covariance[np.ix_(active, active)] += (
            latent_mass / float(feature_dim)
        ) * np.eye(feature_dim, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (
            eigenvectors * np.maximum(eigenvalues, 1e-12)
        ) @ eigenvectors.T
        calibrated["covariance"] = covariance
        calibrated["deviation_variance"] = max(residual_floor, 1e-12)
        diagnostics.update({
            "source_deviation_mode": mode,
            "source_original_deviation_variance": float(original),
            "source_latent_discrepancy_trace": float(latent_mass),
            "source_residual_floor": float(residual_floor),
            "source_reference_feature_energy": int(feature_dim),
            "source_latent_discrepancy_active_coefficients": active.tolist(),
            "source_reference_predictive_variance": float(
                latent_mass + residual_floor),
            "source_record_count_for_deviation_split": int(n_records),
            "target_data_used_for_deviation_split": False,
            "target_oracle_used_for_deviation_split": False,
        })
        calibrated["diagnostics"] = diagnostics
        return calibrated

    def _calibrate_source_constraint_misspecification(
        self,
        model,
        component,
        samples,
        targets,
        observation_variance,
    ):
        """Inflate a source mean law when charged target residuals reject it.

        This is a conservative empirical-Bayes scale posterior.  It never
        contracts a source covariance or residual floor.  The directional
        variant adds a PSD rank-one term along the coefficient direction that
        best explains the target residual, allowing ordinary posterior
        conditioning to correct a biased source expert without treating that
        expert as certain.  Target truth outside the charged observations is
        never consulted.
        """

        calibrated = copy.deepcopy(component)
        mode = str(
            self.config.source_constraint_mean_misspecification_mode or "none"
        ).strip().lower()
        if mode not in {
            "none",
            "predictive_scale",
            "predictive_scale_directional",
            "predictive_scale_upper_target",
            "predictive_scale_upper",
            "predictive_sandwich_hc3",
            "predictive_sandwich_hc3_task",
            "predictive_scale_sandwich_hc3",
            "predictive_scale_sandwich_hc3_task",
            "predictive_scale_sandwich_hc3_confidence",
            "predictive_scale_sandwich_hc3_task_confidence",
            "hierarchical_predictive_scale",
            "source_contrast",
        }:
            raise ValueError(
                "source constraint mean misspecification mode must be none, "
                "predictive_scale, predictive_scale_directional, "
                "predictive_scale_upper_target, predictive_scale_upper, "
                "predictive_sandwich_hc3, "
                "predictive_sandwich_hc3_task, "
                "predictive_scale_sandwich_hc3, "
                "predictive_scale_sandwich_hc3_task, "
                "predictive_scale_sandwich_hc3_confidence, "
                "predictive_scale_sandwich_hc3_task_confidence, or "
                "hierarchical_predictive_scale/source_contrast"
            )
        diagnostics = dict(calibrated.get("diagnostics", {}))
        component_name = str(calibrated.get("name", "source:aggregate"))
        is_source = not component_name.startswith("target:")
        if mode == "none" or not is_source or len(samples) == 0:
            diagnostics.update({
                "source_mean_misspecification_mode": mode,
                "source_mean_misspecification_applied": False,
                "source_mean_misspecification_scale": 1.0,
                "source_mean_misspecification_directional_mass": 0.0,
                "target_observations_used_for_misspecification": int(
                    len(samples) if is_source and mode != "none" else 0),
                "target_oracle_used_for_misspecification": False,
            })
            calibrated["diagnostics"] = diagnostics
            return calibrated

        if mode == "source_contrast" and diagnostics.get(
            "source_mean_misspecification_applied", False
        ):
            calibrated["diagnostics"] = diagnostics
            return calibrated

        if mode in {
            "hierarchical_predictive_scale",
            "predictive_scale_upper_target",
            "predictive_scale_upper",
            "predictive_sandwich_hc3",
            "predictive_sandwich_hc3_task",
            "predictive_scale_sandwich_hc3",
            "predictive_scale_sandwich_hc3_task",
            "predictive_scale_sandwich_hc3_confidence",
            "predictive_scale_sandwich_hc3_task_confidence",
            "source_contrast",
        }:
            diagnostics.update({
                "source_mean_misspecification_mode": mode,
                "source_mean_misspecification_applied": False,
                "source_mean_misspecification_deferred_to_online_mixture": bool(
                    mode in {
                        "hierarchical_predictive_scale",
                        "predictive_scale_upper_target",
                        "predictive_scale_upper",
                        "predictive_sandwich_hc3",
                        "predictive_sandwich_hc3_task",
                        "predictive_scale_sandwich_hc3",
                        "predictive_scale_sandwich_hc3_task",
                        "predictive_scale_sandwich_hc3_confidence",
                        "predictive_scale_sandwich_hc3_task_confidence",
                    }),
                "source_mean_misspecification_deferred_to_component_bank": bool(
                    mode == "source_contrast"),
                "source_mean_misspecification_scale": 1.0,
                "source_mean_misspecification_directional_mass": 0.0,
                "target_observations_used_for_misspecification": 0,
                "target_oracle_used_for_misspecification": False,
            })
            calibrated["diagnostics"] = diagnostics
            return calibrated

        phi = np.asarray(model.basis_matrix(samples), dtype=float)
        target = np.asarray(targets, dtype=float).reshape(-1)
        mean = np.asarray(calibrated["mean"], dtype=float).reshape(-1)
        covariance = np.asarray(
            calibrated["covariance"], dtype=float).copy()
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (
            eigenvectors * np.maximum(eigenvalues, 1e-12)
        ) @ eigenvectors.T
        deviation = max(
            float(calibrated.get("deviation_variance", 1e-6)), 1e-12)
        observation_variance = max(float(observation_variance), 1e-12)
        residual = target - phi @ mean
        predictive = phi @ covariance @ phi.T
        predictive = 0.5 * (predictive + predictive.T)
        predictive += (
            deviation + observation_variance
        ) * np.eye(len(phi), dtype=float)
        jitter = max(
            1e-12,
            1e-10 * float(np.trace(predictive)) / max(len(phi), 1),
        )
        solved = None
        for _ in range(8):
            try:
                chol = np.linalg.cholesky(
                    predictive + jitter * np.eye(len(phi), dtype=float))
                solved = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if solved is None:
            solved = np.linalg.pinv(predictive) @ residual
        mahalanobis = max(float(residual @ solved), 0.0)
        prior_df = max(
            float(
                self.config.source_constraint_mean_misspecification_prior_df
            ),
            1e-8,
        )
        max_scale = max(
            float(
                self.config.source_constraint_mean_misspecification_max_scale
            ),
            1.0,
        )
        posterior_scale = float(np.clip(
            (prior_df + mahalanobis) / (prior_df + len(phi)),
            1.0,
            max_scale,
        ))
        inflated_covariance = posterior_scale * covariance
        inflated_deviation = posterior_scale * deviation
        directional_mass = 0.0
        directional_energy = 0.0
        if mode == "predictive_scale_directional":
            ridge = max(
                float(
                    self.config.source_constraint_mean_misspecification_ridge
                ),
                1e-10,
            )
            gram = phi.T @ phi + ridge * np.eye(phi.shape[1], dtype=float)
            raw_direction = np.linalg.solve(gram, phi.T @ residual)
            direction_norm = float(np.linalg.norm(raw_direction))
            if direction_norm > 1e-12:
                direction = raw_direction / direction_norm
                directional_energy = float(np.mean((phi @ direction) ** 2))
                reference_variance = float(np.mean(np.diag(predictive)))
                empirical_error = float(np.mean(residual ** 2))
                excess = max(empirical_error - reference_variance, 0.0)
                directional_mass = min(
                    excess,
                    max(max_scale - 1.0, 0.0)
                    * max(reference_variance, 1e-12),
                )
                if directional_energy > 1e-12 and directional_mass > 0.0:
                    inflated_covariance += (
                        directional_mass / directional_energy
                    ) * np.outer(direction, direction)
        inflated_covariance = 0.5 * (
            inflated_covariance + inflated_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(inflated_covariance)
        inflated_covariance = (
            eigenvectors * np.maximum(eigenvalues, 1e-12)
        ) @ eigenvectors.T
        calibrated["covariance"] = inflated_covariance
        calibrated["deviation_variance"] = max(
            inflated_deviation, deviation)
        diagnostics.update({
            "source_mean_misspecification_mode": mode,
            "source_mean_misspecification_applied": True,
            "source_mean_misspecification_scale": posterior_scale,
            "source_mean_misspecification_mahalanobis": mahalanobis,
            "source_mean_misspecification_prior_df": prior_df,
            "source_mean_misspecification_directional_mass": float(
                directional_mass),
            "source_mean_misspecification_directional_energy": float(
                directional_energy),
            "source_mean_prior_covariance_trace_before": float(
                np.trace(covariance)),
            "source_mean_prior_covariance_trace_after": float(
                np.trace(inflated_covariance)),
            "source_mean_residual_floor_before": float(deviation),
            "source_mean_residual_floor_after": float(
                calibrated["deviation_variance"]),
            "source_mean_bias_adaptation": (
                "ordinary_target_posterior_conditioning_under_inflated_law"
            ),
            "target_observations_used_for_misspecification": int(len(samples)),
            "target_oracle_used_for_misspecification": False,
            "misspecification_uncertainty_can_only_increase": True,
        })
        calibrated["diagnostics"] = diagnostics
        return calibrated

    def _calibrate_source_contrast_posterior(self, components):
        """Add a source-only low-rank discrepancy law to every expert.

        Differences between source-domain coefficient means identify the only
        directions in which source evidence supports a transferable mean
        correction.  Their weighted covariance is PSD and has rank at most
        ``n_source - 1``.  Target observations subsequently update the mean in
        that frozen subspace through ordinary Bayesian conditioning; target
        truth is never used to construct the subspace or its scale.
        """

        rows = [copy.deepcopy(component) for component in components]
        mode = str(
            self.config.source_constraint_mean_misspecification_mode or "none"
        ).strip().lower()
        if mode != "source_contrast" or not rows:
            return rows
        scale = max(
            float(self.config.source_constraint_mean_contrast_scale), 0.0)
        marker = "|role_assignment="
        groups = {}
        for index, component in enumerate(rows):
            name = str(component.get("name", f"source:{index}"))
            group = (
                name.rsplit(marker, 1)[1].split("|", 1)[0]
                if marker in name else "all"
            )
            groups.setdefault(group, []).append(index)
        for group, indices in groups.items():
            means = np.vstack([
                np.asarray(rows[index]["mean"], dtype=float).reshape(-1)
                for index in indices
            ])
            weights = np.asarray([
                max(float(rows[index].get("prior_weight", 0.0)), 0.0)
                for index in indices
            ], dtype=float)
            if float(np.sum(weights)) <= 0.0:
                weights = np.ones(len(indices), dtype=float)
            weights /= float(np.sum(weights))
            center = np.sum(weights[:, None] * means, axis=0)
            centered = means - center
            contrast = scale * np.einsum(
                "i,ij,ik->jk", weights, centered, centered)
            contrast = 0.5 * (contrast + contrast.T)
            eigenvalues, eigenvectors = np.linalg.eigh(contrast)
            tolerance = max(
                1e-12,
                1e-10 * float(np.max(np.abs(eigenvalues)))
                if len(eigenvalues) else 1e-12,
            )
            positive = np.maximum(eigenvalues, 0.0)
            contrast = (eigenvectors * positive) @ eigenvectors.T
            contrast = 0.5 * (contrast + contrast.T)
            rank = int(np.sum(positive > tolerance))
            for index in indices:
                component = rows[index]
                covariance = np.asarray(
                    component["covariance"], dtype=float)
                covariance = 0.5 * (covariance + covariance.T)
                calibrated = covariance + contrast
                calibrated = 0.5 * (calibrated + calibrated.T)
                component["covariance"] = calibrated
                diagnostics = dict(component.get("diagnostics", {}))
                diagnostics.update({
                    "source_mean_misspecification_mode": "source_contrast",
                    "source_mean_misspecification_applied": True,
                    "source_mean_misspecification_scale": 1.0,
                    "source_mean_misspecification_directional_mass": float(
                        np.trace(contrast)),
                    "source_contrast_scale": float(scale),
                    "source_contrast_rank": int(rank),
                    "source_contrast_rank_bound": max(len(indices) - 1, 0),
                    "source_contrast_covariance_trace": float(
                        np.trace(contrast)),
                    "source_contrast_assignment_group": str(group),
                    "source_contrast_assignment_conditional": bool(
                        marker in str(component.get("name", ""))),
                    "source_contrast_group_component_count": int(
                        len(indices)),
                    "source_mean_prior_covariance_trace_before": float(
                        np.trace(covariance)),
                    "source_mean_prior_covariance_trace_after": float(
                        np.trace(calibrated)),
                    "source_mean_residual_floor_before": float(
                        component["deviation_variance"]),
                    "source_mean_residual_floor_after": float(
                        component["deviation_variance"]),
                    "source_mean_bias_adaptation": (
                        "target_posterior_conditioning_in_frozen_"
                        "assignment_conditional_source_contrast_subspace"),
                    "source_contrast_uses_target_data": False,
                    "target_observations_used_for_misspecification": 0,
                    "target_oracle_used_for_misspecification": False,
                    "misspecification_uncertainty_can_only_increase": True,
                })
                component["diagnostics"] = diagnostics
        return rows

    def _configure_hvd_source_task_posterior(
        self,
        variance_model,
        gpr_models,
    ):
        mode = str(
            self.config.hvd_source_task_weight_mode or "independent"
        ).strip().lower()
        if mode == "independent":
            return None
        if mode != "constraint_mean":
            raise ValueError(
                "hvd_source_task_weight_mode must be independent or "
                "constraint_mean"
            )
        diagnostics = getattr(
            gpr_models[1], "source_parametric_prior_diagnostics", None)
        if not diagnostics or diagnostics.get(
            "adaptation_mode"
        ) not in {
            "target_evidence_mixture",
            "sequential_target_evidence_mixture",
        }:
            raise RuntimeError(
                "constraint-mean HVD task weighting requires an evidence-"
                "mixture source constraint posterior"
            )
        return variance_model.set_source_task_posterior(
            1,
            diagnostics["component_names"],
            diagnostics["component_posterior_weights"],
        )

    def _source_constraint_coefficient_components(self, model, output_index):
        if (
            int(output_index) != 1
            or not self.config.source_constraint_mean_coefficient_prior
        ):
            return []
        basis_map = getattr(model, "basis_map", None)
        if basis_map is None or not hasattr(
            basis_map, "source_parametric_prior_components"
        ):
            return []
        components = []
        for raw in basis_map.source_parametric_prior_components():
            component = dict(raw)
            mean = np.asarray(component.get("mean"), dtype=float).reshape(-1)
            covariance = np.asarray(
                component.get("covariance"), dtype=float)
            if len(mean) != model.p or covariance.shape != (model.p, model.p):
                raise RuntimeError(
                    "source constraint coefficient component does not match "
                    "GPR basis"
                )
            if not np.all(np.isfinite(mean)) or not np.all(
                np.isfinite(covariance)
            ):
                raise FloatingPointError(
                    "source constraint coefficient component is non-finite"
                )
            component["mean"] = mean
            component["covariance"] = covariance
            component["deviation_variance"] = max(
                float(component.get("deviation_variance", 1e-6)), 1e-12)
            component["prior_weight"] = max(
                float(component.get("prior_weight", 0.0)), 0.0)
            components.append(component)
        return components

    def _source_constraint_role_epistemic_calibration(self, model):
        """Return the outcome-free source mass retained by role alignment."""

        mode = str(
            self.config.source_constraint_mean_role_epistemic_mode or "none"
        ).strip().lower()
        if mode == "none":
            return {
                "status": "disabled",
                "mode": mode,
                "source_role_trust": 1.0,
                "target_labels_used": False,
                "target_oracle_used": False,
            }
        if mode not in {"matching_loss", "matching_uncertainty"}:
            raise ValueError(
                "source constraint mean role epistemic mode must be none or "
                "matching_loss/matching_uncertainty")
        basis_map = getattr(model, "basis_map", None)
        if basis_map is None or not hasattr(
            basis_map, "source_target_epistemic_calibration"
        ):
            raise RuntimeError(
                "matching-loss role calibration requires an observable "
                "constraint mean basis")
        diagnostics = dict(
            basis_map.source_target_epistemic_calibration())
        diagnostics["mode"] = mode
        diagnostics["source_role_trust"] = (
            1.0
            if mode == "matching_uncertainty"
            else float(np.clip(
                diagnostics.get("source_role_trust", 1.0), 0.0, 1.0))
        )
        diagnostics["epistemic_covariance_scale"] = max(
            float(diagnostics.get("epistemic_covariance_scale", 1.0)),
            1.0,
        )
        if diagnostics.get("target_labels_used") or diagnostics.get(
            "target_oracle_used"
        ):
            raise RuntimeError(
                "role epistemic calibration must be outcome-free")
        return diagnostics

    def _expand_source_constraint_residual_rank_posterior(
        self,
        model,
        components,
    ):
        if not bool(
            self.config.source_constraint_mean_residual_rank_posterior
        ):
            return list(components)
        basis_map = getattr(model, "basis_map", None)
        if basis_map is None or not hasattr(
            basis_map, "expand_target_residual_rank_components"
        ):
            raise RuntimeError(
                "residual-rank posterior requires an observable constraint "
                "mean basis with nested target residual features")
        raw = self.config.source_constraint_mean_residual_rank_prior
        if isinstance(raw, str):
            rank_prior = [
                float(value.strip())
                for value in raw.split(",") if value.strip()
            ]
        else:
            rank_prior = [float(value) for value in raw]
        return basis_map.expand_target_residual_rank_components(
            components,
            rank_prior,
            inactive_variance=(
                self.config
                .source_constraint_mean_residual_rank_inactive_variance),
        )

    @staticmethod
    def _source_component_log_evidence(
        model,
        samples,
        targets,
        component,
        observation_variance,
    ):
        phi = model.basis_matrix(samples)
        mean = np.asarray(component["mean"], dtype=float)
        covariance = np.asarray(component["covariance"], dtype=float)
        target = np.asarray(targets, dtype=float).reshape(-1)
        noise = max(
            float(component["deviation_variance"])
            + float(observation_variance),
            1e-12,
        )
        predictive_covariance = phi @ covariance @ phi.T
        predictive_covariance = 0.5 * (
            predictive_covariance + predictive_covariance.T)
        predictive_covariance += noise * np.eye(len(phi), dtype=float)
        residual = target - phi @ mean
        jitter = max(1e-12, 1e-10 * float(np.trace(
            predictive_covariance)) / max(len(phi), 1))
        for _ in range(8):
            try:
                chol = np.linalg.cholesky(
                    predictive_covariance
                    + jitter * np.eye(len(phi), dtype=float)
                )
                solved = np.linalg.solve(chol, residual)
                log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
                return float(-0.5 * (
                    solved @ solved
                    + log_det
                    + len(phi) * np.log(2.0 * np.pi)
                ))
            except np.linalg.LinAlgError:
                jitter *= 10.0
        raise np.linalg.LinAlgError(
            "source component predictive covariance is not positive definite"
        )

    def _target_null_constraint_component(self, model, aggregate_prior):
        covariance = np.asarray(
            aggregate_prior["covariance"], dtype=float)
        diagonal = np.diag(covariance)
        finite = diagonal[np.isfinite(diagonal) & (diagonal > 0.0)]
        output_scale = max(
            float(aggregate_prior.get("output_scale", 0.0)),
            float(getattr(self.problem, "sigma_level", 0.0)),
            1e-6,
        )
        null_variance = max(
            float(np.median(finite)) if len(finite) else 0.0,
            output_scale ** 2,
            1e-6,
        )
        geometry_mode = str(
            self.config.source_constraint_mean_null_geometry or "isotropic"
        ).strip().lower().replace("-", "_")
        if geometry_mode not in {"isotropic", "target_pool"}:
            raise ValueError(
                "source constraint mean null geometry must be isotropic or "
                "target_pool")
        null_covariance = null_variance * np.eye(model.p, dtype=float)
        geometry_diagnostics = {
            "null_geometry_mode": geometry_mode,
            "target_labels_used_for_null_geometry": False,
            "target_oracle_used_for_null_geometry": False,
        }
        if geometry_mode == "target_pool":
            basis_map = getattr(model, "basis_map", None)
            bridge = (
                basis_map.target_null_feature_geometry()
                if basis_map is not None
                and hasattr(basis_map, "target_null_feature_geometry")
                else {"status": "unavailable"}
            )
            if bridge.get("status") != "available":
                raise RuntimeError(
                    "target-pool null geometry requires an unlabeled feature "
                    "geometry bridge")
            if (
                bridge.get("target_labels_used", True)
                or bridge.get("target_oracle_used", True)
            ):
                raise RuntimeError(
                    "target-pool null geometry must be outcome-free")
            design = np.asarray(bridge["basis_matrix"], dtype=float)
            if (
                design.ndim != 2
                or design.shape[1] != model.p
                or len(design) < model.p
                or not np.all(np.isfinite(design))
            ):
                raise RuntimeError(
                    "target-pool null geometry has an invalid design")
            gram = design.T @ design / float(len(design))
            gram = 0.5 * (gram + gram.T)
            average_scale = max(
                float(np.trace(gram)) / max(model.p, 1), 1e-12)
            relative_ridge = max(float(
                self.config.source_constraint_mean_null_geometry_ridge), 1e-10)
            regularized = (
                gram + relative_ridge * average_scale
                * np.eye(model.p, dtype=float))
            inverse = np.linalg.pinv(regularized)
            inverse = 0.5 * (inverse + inverse.T)
            isotropic_average = max(
                float(null_variance * np.trace(gram)), 1e-12)
            raw_average = max(float(np.trace(gram @ inverse)), 1e-12)
            null_covariance = (isotropic_average / raw_average) * inverse
            null_covariance = 0.5 * (
                null_covariance + null_covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(null_covariance)
            null_covariance = (
                eigenvectors * np.maximum(eigenvalues, 1e-12)
            ) @ eigenvectors.T
            geometric_average = float(np.trace(gram @ null_covariance))
            geometry_diagnostics.update({
                "target_geometry_pool_size": int(len(design)),
                "target_geometry_basis_dim": int(design.shape[1]),
                "target_geometry_pool_source": str(
                    bridge.get("pool_source", "unknown")),
                "target_geometry_relative_ridge": float(relative_ridge),
                "target_geometry_gram_condition": float(np.linalg.cond(
                    regularized)),
                "isotropic_average_predictive_variance": float(
                    isotropic_average),
                "geometric_average_predictive_variance": float(
                    geometric_average),
                "average_predictive_variance_ratio": float(
                    geometric_average / isotropic_average),
                "average_predictive_scale_preserved": bool(
                    abs(geometric_average - isotropic_average)
                    <= 1e-8 * max(isotropic_average, 1.0)),
                "minimum_covariance_eigenvalue": float(np.min(
                    np.linalg.eigvalsh(null_covariance))),
            })
        mean = np.zeros(model.p, dtype=float)
        mean[0] = float(getattr(self.problem, "tau", 0.0))
        return {
            "name": "target:null",
            "domain": None,
            "mean": mean,
            "covariance": null_covariance,
            "deviation_variance": max(output_scale ** 2, 1e-8),
            "prior_weight": max(
                float(self.config.source_constraint_mean_null_weight),
                0.0,
            ),
            "diagnostics": {
                "component_kind": "nontransfer_null",
                "null_variance": float(null_variance),
                **geometry_diagnostics,
                "target_data_used": False,
                "target_oracle_used": False,
            },
        }

    def _fit_source_constraint_coefficient_posterior(
        self,
        model,
        output_index,
        samples,
        targets,
        observation_variance,
    ):
        """Condition the frozen source coefficient law on charged target data."""

        basis_map = getattr(model, "basis_map", None)
        if (
            int(output_index) == 1
            and basis_map is not None
            and hasattr(
                basis_map,
                "calibrate_role_assignment_boundary_posterior",
            )
        ):
            basis_map.calibrate_role_assignment_boundary_posterior(
                samples,
                targets,
                np.full(
                    len(samples), observation_variance, dtype=float),
            )
        raw_aggregate_prior = self._source_constraint_coefficient_prior(
            model, output_index)
        if raw_aggregate_prior is None:
            return None
        aggregate_prior = self._calibrate_source_constraint_deviation(
            model, raw_aggregate_prior)
        mode = str(
            self.config.source_constraint_mean_adaptation_mode
            or "frozen"
        ).strip().lower()
        if mode == "sequential_aggregate_hyperlaw":
            misspecification_mode = str(
                self.config.source_constraint_mean_misspecification_mode
                or "none"
            ).strip().lower()
            if misspecification_mode not in {
                "none",
                "predictive_scale",
                "predictive_scale_directional",
                "predictive_scale_upper_target",
                "predictive_scale_upper",
                "predictive_sandwich_hc3",
                "predictive_sandwich_hc3_task",
                "predictive_scale_sandwich_hc3",
                "predictive_scale_sandwich_hc3_task",
                "predictive_scale_sandwich_hc3_confidence",
                "predictive_scale_sandwich_hc3_task_confidence",
                "hierarchical_predictive_scale",
            }:
                raise ValueError(
                    "sequential aggregate hyperlaw requires none, "
                    "predictive_scale, predictive_scale_directional, "
                    "predictive_scale_upper_target, predictive_scale_upper, "
                    "predictive_sandwich_hc3, "
                    "predictive_sandwich_hc3_task, "
                    "predictive_scale_sandwich_hc3, "
                    "predictive_scale_sandwich_hc3_task, "
                    "predictive_scale_sandwich_hc3_confidence, "
                    "predictive_scale_sandwich_hc3_task_confidence, or "
                    "hierarchical_predictive_scale misspecification")
            aggregate_component = copy.deepcopy(aggregate_prior)
            aggregate_component["name"] = "source:aggregate"
            aggregate_component["prior_weight"] = 1.0
            component_diagnostics = dict(
                aggregate_component.get("diagnostics", {}))
            component_diagnostics.update({
                "component_kind": (
                    "exchangeable_empirical_bayes_gaussian_hyperlaw"),
                "single_aggregate_hyperlaw": True,
                "source_domain_identity_marginalized": True,
                "source_components_retained_in_target_posterior": False,
                "target_null_component_retained": False,
                "target_data_used_to_define_aggregate": False,
                "target_oracle_used_to_define_aggregate": False,
            })
            aggregate_component["diagnostics"] = component_diagnostics
            component_model = ParametricGPR(
                model.d,
                lambda_i=model.lambda_i,
                prior_var=1.0,
                normalize_func=model.normalize_func,
                basis_map=model.basis_map,
                basis_config=copy.copy(model.basis_config),
                numeric_backend=model.numeric_backend,
                numeric_backend_device=model.numeric_backend_device,
                torch_dtype=model.torch_dtype,
                torch_min_rows=model.torch_min_rows,
            )
            diagnostics = {
                **dict(aggregate_prior.get("diagnostics", {})),
                "adaptation_mode": "sequential_single_aggregate_hyperlaw",
                "single_aggregate_hyperlaw": True,
                "single_aggregate_component_count": 1,
                "source_domain_identity_marginalized": True,
                "source_components_retained_in_target_posterior": False,
                "target_null_component_retained": False,
                "prior_target_data_used": False,
                "posterior_target_data_used": bool(len(samples)),
                "target_observation_count": int(len(samples)),
                "online_mixture_update_count": 0,
                "target_oracle_used": False,
                "null_prior_weight": 0.0,
                "requested_null_prior_weight": 0.0,
                "source_prior_mass_after_role_calibration": 1.0,
                "source_mean_misspecification_mode": misspecification_mode,
            }
            model.set_hierarchical_misspecification_posterior(
                [component_model],
                [aggregate_component],
                np.asarray([1.0], dtype=float),
                samples,
                targets,
                np.full(
                    len(samples), observation_variance, dtype=float),
                diagnostics=diagnostics,
                prior_df=(
                    self.config
                    .source_constraint_mean_misspecification_prior_df),
                max_scale=(
                    self.config
                    .source_constraint_mean_misspecification_max_scale),
                misspecification_mode=misspecification_mode,
                misspecification_ridge=(
                    self.config
                    .source_constraint_mean_misspecification_ridge),
                misspecification_delta=(
                    self.config
                    .source_constraint_mean_misspecification_delta),
            )
            return dict(model.source_parametric_prior_diagnostics)

        aggregate_prior = self._calibrate_source_constraint_misspecification(
            model,
            aggregate_prior,
            samples,
            targets,
            observation_variance,
        )
        if mode in ("frozen", "aggregate", "none"):
            if str(
                self.config.source_constraint_mean_misspecification_mode
            ).strip().lower() == "source_contrast":
                raise ValueError(
                    "source_contrast misspecification requires an evidence "
                    "mixture with source-domain components")
            model.set_parametric_prior(
                aggregate_prior["mean"],
                aggregate_prior["deviation_variance"],
                aggregate_prior["covariance"],
            )
            diagnostics = dict(aggregate_prior.get("diagnostics", {}))
            diagnostics.update({
                "adaptation_mode": "frozen_aggregate",
                "prior_target_data_used": False,
                "posterior_target_data_used": True,
                "target_observation_count": int(len(samples)),
                "target_oracle_used": False,
            })
            model.source_parametric_prior_diagnostics = diagnostics
            for x, target in zip(samples, targets):
                model.update(x, float(target), observation_variance)
            return diagnostics
        if mode not in (
            "evidence_mixture", "mixture", "adaptive_mixture",
            "sequential_evidence_mixture", "sequential_mixture",
            "aggregate_mixture", "sequential_aggregate_mixture",
            "sequential_aggregate_hyperlaw",
            "support_adaptive_aggregate_mixture",
            "sequential_support_adaptive_aggregate_mixture",
        ):
            raise ValueError(
                "source constraint mean adaptation mode must be frozen or "
                "evidence_mixture/sequential_evidence_mixture/"
                "aggregate_mixture/sequential_aggregate_mixture/"
                "sequential_aggregate_hyperlaw/"
                "support_adaptive_aggregate_mixture/"
                "sequential_support_adaptive_aggregate_mixture"
            )
        sequential_updates = mode in {
            "sequential_evidence_mixture", "sequential_mixture",
            "sequential_aggregate_mixture",
            "sequential_support_adaptive_aggregate_mixture",
        }
        support_adaptive_aggregate = mode in {
            "support_adaptive_aggregate_mixture",
            "sequential_support_adaptive_aggregate_mixture",
        }
        support_selection = None
        if support_adaptive_aggregate:
            basis_map = getattr(model, "basis_map", None)
            if basis_map is None or not hasattr(
                basis_map, "source_target_coordinate_selection"
            ):
                raise RuntimeError(
                    "support-adaptive aggregate transfer requires a "
                    "coordinate-selection bridge")
            support_selection = dict(
                basis_map.source_target_coordinate_selection())
        aggregate_mixture = mode in {
            "aggregate_mixture", "sequential_aggregate_mixture",
        } or bool(
            support_adaptive_aggregate
            and support_selection["channel_cardinality_supported"])

        if aggregate_mixture:
            aggregate_component = copy.deepcopy(aggregate_prior)
            aggregate_component["name"] = "source:aggregate"
            aggregate_component["prior_weight"] = 1.0
            aggregate_diagnostics = dict(
                aggregate_component.get("diagnostics", {}))
            aggregate_diagnostics.update({
                "component_kind": "hierarchical_source_aggregate",
                "aggregate_contains_within_source_uncertainty": True,
                "aggregate_contains_between_source_disagreement": True,
                "target_data_used_to_define_aggregate": False,
                "target_oracle_used_to_define_aggregate": False,
            })
            aggregate_component["diagnostics"] = aggregate_diagnostics
            source_components = [aggregate_component]
        else:
            source_components = [
                self._calibrate_source_constraint_deviation(model, component)
                for component in self._source_constraint_coefficient_components(
                    model, output_index)
            ]
            source_components = self._calibrate_source_contrast_posterior(
                source_components)
            calibrated_components = []
            for component in source_components:
                component = self._calibrate_source_constraint_misspecification(
                    model,
                    component,
                    samples,
                    targets,
                    observation_variance,
                )
                calibrated_components.append(component)
            source_components = calibrated_components
        if not source_components:
            raise RuntimeError(
                "evidence-mixture source mean adaptation requires source "
                "coefficient components"
            )
        requested_null_weight = float(np.clip(
            self.config.source_constraint_mean_null_weight, 0.0, 1.0))
        role_epistemic = self._source_constraint_role_epistemic_calibration(
            model)
        if role_epistemic["mode"] == "matching_uncertainty":
            covariance_scale = min(
                float(role_epistemic["epistemic_covariance_scale"]),
                max(float(
                    self.config
                    .source_constraint_mean_misspecification_max_scale
                ), 1.0),
            )
            for component in source_components:
                covariance = np.asarray(
                    component["covariance"], dtype=float)
                component["covariance"] = (
                    covariance_scale * covariance)
                component_diagnostics = dict(
                    component.get("diagnostics", {}))
                component_diagnostics.update({
                    "role_matching_epistemic_covariance_scale": float(
                        covariance_scale),
                    "role_matching_uncertainty_monotone": True,
                    "role_matching_target_labels_used": False,
                    "role_matching_target_oracle_used": False,
                })
                component["diagnostics"] = component_diagnostics
        source_mass_before_role_calibration = 1.0 - requested_null_weight
        source_mass = (
            source_mass_before_role_calibration
            * float(role_epistemic["source_role_trust"])
        )
        null_weight = 1.0 - source_mass
        raw_source_weight = np.asarray([
            component["prior_weight"] for component in source_components
        ], dtype=float)
        if float(np.sum(raw_source_weight)) <= 0.0:
            raw_source_weight = np.ones(len(source_components), dtype=float)
        raw_source_weight /= float(np.sum(raw_source_weight))
        for mass, component in zip(raw_source_weight, source_components):
            component["prior_weight"] = float(source_mass * mass)
        components = source_components + [
            self._target_null_constraint_component(model, raw_aggregate_prior)
        ]
        components[-1]["prior_weight"] = null_weight
        basis_map = getattr(model, "basis_map", None)
        if basis_map is not None and hasattr(
            basis_map, "expand_target_role_assignment_components"
        ):
            components = basis_map.expand_target_role_assignment_components(
                components)
        components = self._expand_source_constraint_residual_rank_posterior(
            model, components)
        prior_weight = normalize_mixture_weights([
            max(float(component["prior_weight"]), 0.0)
            for component in components
        ])
        misspecification_mode = str(
            self.config.source_constraint_mean_misspecification_mode or "none"
        ).strip().lower()
        structure_score_mode = str(
            self.config.source_constraint_mean_structure_score_mode
            or "marginal_likelihood"
        ).strip().lower().replace("-", "_")
        if structure_score_mode not in {
            "marginal_likelihood", "loo_predictive", "geometry_conditional"
        }:
            raise ValueError(
                "source constraint mean structure score mode must be "
                "marginal_likelihood, loo_predictive, or "
                "geometry_conditional")
        if (
            structure_score_mode == "loo_predictive"
            and misspecification_mode == "hierarchical_predictive_scale"
        ):
            raise ValueError(
                "LOO structure scoring and hierarchical predictive scaling "
                "must be evaluated as separate calibration mechanisms")
        if structure_score_mode == "geometry_conditional":
            basis_map = getattr(model, "basis_map", None)
            role_diagnostics = (
                basis_map.diagnostics().get("role_assignment_posterior", {})
                if basis_map is not None and hasattr(basis_map, "diagnostics")
                else {}
            )
            if role_diagnostics.get("prior") not in {
                "source_geometry", "source_geometry_boundary"
            }:
                raise ValueError(
                    "geometry-conditional adaptation requires the "
                    "source_geometry/source_geometry_boundary "
                    "role-assignment prior")
            assignment_prior_target_labels_used = bool(
                role_diagnostics.get(
                    "target_labels_used_to_define_assignments", False))
            marker = "|role_assignment="
            group_labels = []
            for component in components:
                name = str(component["name"])
                if marker not in name:
                    raise RuntimeError(
                        "geometry-conditional component has no assignment")
                group_labels.append(
                    name.rsplit(marker, 1)[1].split("|", 1)[0])
            group_masses = {}
            for label, mass in zip(group_labels, prior_weight):
                group_masses[label] = (
                    group_masses.get(label, 0.0) + float(mass))
            names = [str(component["name"]) for component in components]
            if misspecification_mode == "hierarchical_predictive_scale":
                posterior_models = [
                    ParametricGPR(
                        model.d,
                        lambda_i=model.lambda_i,
                        prior_var=1.0,
                        normalize_func=model.normalize_func,
                        basis_map=model.basis_map,
                        basis_config=copy.copy(model.basis_config),
                        numeric_backend=model.numeric_backend,
                        numeric_backend_device=model.numeric_backend_device,
                        torch_dtype=model.torch_dtype,
                        torch_min_rows=model.torch_min_rows,
                    )
                    for _component in components
                ]
                diagnostics = {
                    **dict(aggregate_prior.get("diagnostics", {})),
                    "adaptation_mode": (
                        "sequential_assignment_prior_conditional_"
                        "hierarchical_expert_mixture"),
                    "structure_score_mode": "geometry_conditional",
                    "structure_score_cross_fitted": False,
                    "prior_target_data_used": False,
                    "posterior_target_data_used": True,
                    "target_observation_count": int(len(samples)),
                    "target_oracle_used": False,
                    "evidence_temperature": float(max(
                        self.config
                        .source_constraint_mean_evidence_temperature,
                        1e-6,
                    )),
                    "null_prior_weight": float(null_weight),
                    "requested_null_prior_weight": float(
                        requested_null_weight),
                    "source_prior_mass_before_role_calibration": float(
                        source_mass_before_role_calibration),
                    "source_prior_mass_after_role_calibration": float(
                        source_mass),
                    "source_role_epistemic_calibration": dict(
                        role_epistemic),
                    "component_names": names,
                    "component_prior_weights": prior_weight.tolist(),
                    "online_mixture_update_count": 0,
                    "source_deviation_mode": str(
                        self.config.source_constraint_mean_deviation_mode),
                    "source_mean_misspecification_mode": (
                        misspecification_mode),
                    "assignment_group_masses_fixed": True,
                    "assignment_group_masses": dict(group_masses),
                    "target_labels_used_for_group_masses": bool(
                        assignment_prior_target_labels_used),
                    "target_oracle_used_for_group_masses": False,
                    "component_deviation_diagnostics": [
                        {
                            "name": str(component["name"]),
                            **dict(component.get("diagnostics", {})),
                        }
                        for component in components
                    ],
                }
                model.set_hierarchical_misspecification_posterior(
                    posterior_models,
                    components,
                    prior_weight,
                    samples,
                    targets,
                    np.full(
                        len(samples), observation_variance, dtype=float),
                    diagnostics=diagnostics,
                    prior_df=(
                        self.config
                        .source_constraint_mean_misspecification_prior_df),
                    max_scale=(
                        self.config
                        .source_constraint_mean_misspecification_max_scale),
                    misspecification_mode=misspecification_mode,
                    misspecification_ridge=(
                        self.config
                        .source_constraint_mean_misspecification_ridge),
                    misspecification_delta=(
                        self.config
                        .source_constraint_mean_misspecification_delta),
                    group_labels=group_labels,
                    group_masses=group_masses,
                )
                return dict(model.source_parametric_prior_diagnostics)
            log_evidence = np.asarray([
                self._source_component_log_evidence(
                    model,
                    samples,
                    targets,
                    component,
                    observation_variance,
                )
                for component in components
            ], dtype=float)
            posterior_weight, group_masses = (
                ParametricGPR.group_mass_preserving_weights(
                    prior_weight,
                    log_evidence,
                    group_labels,
                    group_masses,
                    self.config.source_constraint_mean_evidence_temperature,
                )
            )
            posterior_models = []
            for component in components:
                component_model = ParametricGPR(
                    model.d,
                    lambda_i=model.lambda_i,
                    prior_var=1.0,
                    normalize_func=model.normalize_func,
                    basis_map=model.basis_map,
                    basis_config=copy.copy(model.basis_config),
                    numeric_backend=model.numeric_backend,
                    numeric_backend_device=model.numeric_backend_device,
                    torch_dtype=model.torch_dtype,
                    torch_min_rows=model.torch_min_rows,
                )
                component_model.set_parametric_prior(
                    component["mean"],
                    component["deviation_variance"],
                    component["covariance"],
                )
                for x, target in zip(samples, targets):
                    component_model.update(
                        x, float(target), observation_variance)
                posterior_models.append(component_model)
            diagnostics = {
                **dict(aggregate_prior.get("diagnostics", {})),
                "adaptation_mode": (
                    "sequential_assignment_prior_conditional_expert_mixture"),
                "structure_score_mode": "geometry_conditional",
                "structure_score_cross_fitted": False,
                "prior_target_data_used": False,
                "posterior_target_data_used": True,
                "target_observation_count": int(len(samples)),
                "target_oracle_used": False,
                "evidence_temperature": float(max(
                    self.config.source_constraint_mean_evidence_temperature,
                    1e-6,
                )),
                "null_prior_weight": float(null_weight),
                "requested_null_prior_weight": float(requested_null_weight),
                "source_prior_mass_before_role_calibration": float(
                    source_mass_before_role_calibration),
                "source_prior_mass_after_role_calibration": float(
                    source_mass),
                "source_role_epistemic_calibration": dict(role_epistemic),
                "component_names": names,
                "component_prior_weights": prior_weight.tolist(),
                "component_log_evidence": log_evidence.tolist(),
                "component_posterior_weights": posterior_weight.tolist(),
                "selected_component": str(names[int(
                    np.argmax(posterior_weight))]),
                "target_only_posterior_weight": float(sum(
                    mass for name, mass in zip(names, posterior_weight)
                    if str(name).startswith("target:null"))),
                "source_posterior_weight": float(sum(
                    mass for name, mass in zip(names, posterior_weight)
                    if not str(name).startswith("target:null"))),
                "online_mixture_update_count": 0,
                "source_deviation_mode": str(
                    self.config.source_constraint_mean_deviation_mode),
                "source_mean_misspecification_mode": misspecification_mode,
                "assignment_group_masses_fixed": True,
                "assignment_group_masses": dict(group_masses),
                "target_labels_used_for_group_masses": bool(
                    assignment_prior_target_labels_used),
                "target_oracle_used_for_group_masses": False,
                "component_deviation_diagnostics": [
                    {
                        "name": str(component["name"]),
                        **dict(component.get("diagnostics", {})),
                    }
                    for component in components
                ],
            }
            model.set_group_mass_preserving_posterior(
                posterior_models,
                posterior_weight,
                group_labels,
                group_masses,
                diagnostics=diagnostics,
            )
            return dict(model.source_parametric_prior_diagnostics)
        if structure_score_mode == "loo_predictive":
            component_models = [
                ParametricGPR(
                    model.d,
                    lambda_i=model.lambda_i,
                    prior_var=1.0,
                    normalize_func=model.normalize_func,
                    basis_map=model.basis_map,
                    basis_config=copy.copy(model.basis_config),
                    numeric_backend=model.numeric_backend,
                    numeric_backend_device=model.numeric_backend_device,
                    torch_dtype=model.torch_dtype,
                    torch_min_rows=model.torch_min_rows,
                )
                for _component in components
            ]
            diagnostics = {
                **dict(aggregate_prior.get("diagnostics", {})),
                "adaptation_mode": (
                    "sequential_cross_validated_target_evidence_mixture"),
                "structure_score_mode": "loo_predictive",
                "structure_score_cross_fitted": True,
                "prior_target_data_used": False,
                "posterior_target_data_used": True,
                "target_observation_count": int(len(samples)),
                "target_oracle_used": False,
                "evidence_temperature": float(max(
                    self.config.source_constraint_mean_evidence_temperature,
                    1e-6,
                )),
                "null_prior_weight": float(null_weight),
                "requested_null_prior_weight": float(requested_null_weight),
                "source_prior_mass_before_role_calibration": float(
                    source_mass_before_role_calibration),
                "source_prior_mass_after_role_calibration": float(
                    source_mass),
                "source_role_epistemic_calibration": dict(role_epistemic),
                "component_names": [
                    str(component["name"]) for component in components
                ],
                "component_prior_weights": prior_weight.tolist(),
                "online_mixture_update_count": 0,
                "source_deviation_mode": str(
                    self.config.source_constraint_mean_deviation_mode),
                "source_mean_misspecification_mode": misspecification_mode,
                "component_deviation_diagnostics": [
                    {
                        "name": str(component["name"]),
                        **dict(component.get("diagnostics", {})),
                    }
                    for component in components
                ],
            }
            model.set_cross_validated_structure_posterior(
                component_models,
                components,
                prior_weight,
                samples,
                targets,
                np.full(
                    len(samples), observation_variance, dtype=float),
                diagnostics=diagnostics,
            )
            return dict(model.source_parametric_prior_diagnostics)
        if misspecification_mode == "hierarchical_predictive_scale":
            component_models = []
            for component in components:
                component_models.append(ParametricGPR(
                    model.d,
                    lambda_i=model.lambda_i,
                    prior_var=1.0,
                    normalize_func=model.normalize_func,
                    basis_map=model.basis_map,
                    basis_config=copy.copy(model.basis_config),
                    numeric_backend=model.numeric_backend,
                    numeric_backend_device=model.numeric_backend_device,
                    torch_dtype=model.torch_dtype,
                    torch_min_rows=model.torch_min_rows,
                ))
            diagnostics = {
                **dict(aggregate_prior.get("diagnostics", {})),
                "adaptation_mode": "sequential_target_evidence_mixture",
                "prior_target_data_used": False,
                "posterior_target_data_used": True,
                "target_observation_count": int(len(samples)),
                "target_oracle_used": False,
                "evidence_temperature": float(max(
                    self.config.source_constraint_mean_evidence_temperature,
                    1e-6,
                )),
                "null_prior_weight": float(null_weight),
                "requested_null_prior_weight": float(requested_null_weight),
                "source_prior_mass_before_role_calibration": float(
                    source_mass_before_role_calibration),
                "source_prior_mass_after_role_calibration": float(source_mass),
                "source_role_epistemic_calibration": dict(role_epistemic),
                "component_names": [
                    str(component["name"]) for component in components
                ],
                "component_prior_weights": prior_weight.tolist(),
                "online_mixture_update_count": 0,
                "source_deviation_mode": str(
                    self.config.source_constraint_mean_deviation_mode),
                "source_mean_misspecification_mode": misspecification_mode,
                "component_deviation_diagnostics": [
                    {
                        "name": str(component["name"]),
                        **dict(component.get("diagnostics", {})),
                    }
                    for component in components
                ],
            }
            model.set_hierarchical_misspecification_posterior(
                component_models,
                components,
                prior_weight,
                samples,
                targets,
                np.full(
                    len(samples), observation_variance, dtype=float),
                diagnostics=diagnostics,
                prior_df=(
                    self.config.source_constraint_mean_misspecification_prior_df
                ),
                max_scale=(
                    self.config.source_constraint_mean_misspecification_max_scale
                ),
                misspecification_mode=misspecification_mode,
                misspecification_ridge=(
                    self.config.source_constraint_mean_misspecification_ridge
                ),
                misspecification_delta=(
                    self.config.source_constraint_mean_misspecification_delta
                ),
            )
            return dict(model.source_parametric_prior_diagnostics)

        log_evidence = np.asarray([
            self._source_component_log_evidence(
                model,
                samples,
                targets,
                component,
                observation_variance,
            )
            for component in components
        ], dtype=float)
        temperature = max(
            float(self.config.source_constraint_mean_evidence_temperature),
            1e-6,
        )
        prior_weight, posterior_weight = posterior_mixture_weights(
            prior_weight, log_evidence, temperature)

        posterior_models = []
        for component in components:
            component_model = ParametricGPR(
                model.d,
                lambda_i=model.lambda_i,
                prior_var=1.0,
                normalize_func=model.normalize_func,
                basis_map=model.basis_map,
                basis_config=copy.copy(model.basis_config),
                numeric_backend=model.numeric_backend,
                numeric_backend_device=model.numeric_backend_device,
                torch_dtype=model.torch_dtype,
                torch_min_rows=model.torch_min_rows,
            )
            component_model.set_parametric_prior(
                component["mean"],
                component["deviation_variance"],
                component["covariance"],
            )
            for x, target in zip(samples, targets):
                component_model.update(
                    x, float(target), observation_variance)
            posterior_models.append(component_model)

        diagnostics = {
            **dict(aggregate_prior.get("diagnostics", {})),
            "adaptation_mode": (
                (
                    "sequential_aggregate_target_evidence_mixture"
                    if sequential_updates
                    else "aggregate_target_evidence_mixture"
                )
                if aggregate_mixture
                else (
                    "sequential_target_evidence_mixture"
                    if sequential_updates
                    else "target_evidence_mixture"
                )
            ),
            "prior_target_data_used": False,
            "posterior_target_data_used": True,
            "target_observation_count": int(len(samples)),
            "target_oracle_used": False,
            "evidence_temperature": float(temperature),
            "null_prior_weight": float(null_weight),
            "requested_null_prior_weight": float(requested_null_weight),
            "source_prior_mass_before_role_calibration": float(
                source_mass_before_role_calibration),
            "source_prior_mass_after_role_calibration": float(source_mass),
            "source_role_epistemic_calibration": dict(role_epistemic),
            "component_names": [
                str(component["name"]) for component in components
            ],
            "component_prior_weights": prior_weight.tolist(),
            "component_log_evidence": log_evidence.tolist(),
            "component_posterior_weights": posterior_weight.tolist(),
            "selected_component": str(components[int(
                np.argmax(posterior_weight))]["name"]),
            "target_only_posterior_weight": float(sum(
                mass for component, mass in zip(
                    components, posterior_weight)
                if str(component["name"]).startswith("target:null"))),
            "source_posterior_weight": float(sum(
                mass for component, mass in zip(
                    components, posterior_weight)
                if not str(component["name"]).startswith("target:null"))),
            "online_mixture_update_count": 0,
            "source_deviation_mode": str(
                self.config.source_constraint_mean_deviation_mode),
            "source_mean_misspecification_mode": str(
                self.config.source_constraint_mean_misspecification_mode),
            "structure_score_mode": "marginal_likelihood",
            "structure_score_cross_fitted": False,
            "aggregate_transferability_latent": bool(aggregate_mixture),
            "support_adaptive_aggregate_requested": bool(
                support_adaptive_aggregate),
            "support_adaptive_aggregate_selection": copy.deepcopy(
                support_selection),
            "effective_source_adaptation": (
                "aggregate_latent" if aggregate_mixture
                else "domain_mixture"),
            "component_deviation_diagnostics": [
                {
                    "name": str(component["name"]),
                    **dict(component.get("diagnostics", {})),
                }
                for component in components
            ],
        }
        model.set_moment_matched_posterior(
            posterior_models,
            posterior_weight,
            diagnostics=diagnostics,
            sequential_updates=sequential_updates,
        )
        return diagnostics

    def _initialize_task_ensemble(self, samples):
        if not self._task_posterior_requested():
            self.task_ensemble = None
            return None
        mode = str(self.config.task_posterior_mode).lower()
        if mode not in ("finite", "finite_expert", "finite-expert"):
            raise ValueError(f"unknown task posterior mode {mode!r}")
        required = (
            "task_posterior_expert_specs",
            "task_expert_basis_map",
            "task_expert_problem_view",
        )
        missing = [name for name in required if not hasattr(self.problem, name)]
        if missing:
            raise RuntimeError(
                "finite task posterior requires a source-trained admissible "
                f"meta-prior provider; missing {missing}"
            )
        samples = [tuple(int(v) for v in x) for x in samples]
        if not samples:
            raise RuntimeError("finite task posterior requires pilot observations")
        configured_pilot = int(self.config.task_posterior_pilot_count)
        if configured_pilot < 0:
            pilot_count = min(4, len(samples))
            if len(samples) > 1:
                pilot_count = min(pilot_count, len(samples) - 1)
        else:
            pilot_count = int(np.clip(
                configured_pilot,
                1,
                len(samples),
            ))
        pilot_samples = samples[:pilot_count]
        try:
            specs = list(self.problem.task_posterior_expert_specs(
                include_local_kernel=bool(
                    self.config.task_posterior_local_kernel_expert),
            ))
        except TypeError:
            specs = list(self.problem.task_posterior_expert_specs())
        states = []
        for spec in specs:
            name = str(spec["name"])
            expert_problem = self.problem.task_expert_problem_view(name)
            expert_gpr = []
            for output_index in range(2):
                basis_map = self.problem.task_expert_basis_map(
                    name, output_index=output_index)
                model = ParametricGPR(
                    self.problem.d,
                    self.config.lambda_i,
                    self.config.prior_var,
                    normalize_func=self.problem.normalize,
                    basis_map=basis_map,
                    numeric_backend=self.config.numeric_backend,
                    numeric_backend_device=self.config.numeric_backend_device,
                    torch_dtype=self.config.torch_dtype,
                    torch_min_rows=self.config.torch_min_rows,
                )
                self._fit_initial_task_expert_gpr(
                    model, output_index, pilot_samples)
                expert_gpr.append(model)
            variance_model = OrthogonalHVD(
                mode=str(spec.get("variance_mode", "factor")),
                n_outputs=2,
                floor=1e-8,
                use_cumulative_provider=(
                    self.config.hvd_use_cumulative_provider),
                cumulative_transfer_mode=(
                    self.config.hvd_cumulative_transfer_mode),
                cumulative_source_task_weight_mode=(
                    self.config.hvd_source_task_weight_mode),
                cumulative_target_evidence_mode=(
                    self.config.hvd_cumulative_target_evidence_mode),
                singleton_evidence_mode=(
                    self.config.hvd_singleton_evidence_mode),
            )
            self._configure_hvd_source_task_posterior(
                variance_model, expert_gpr)
            variance_model.initialize(
                pilot_samples,
                self.observations,
                expert_gpr,
                expert_problem,
            )
            states.append(TaskExpertState(
                name=name,
                gpr_models=expert_gpr,
                variance_model=variance_model,
                problem=expert_problem,
            ))
        posterior = FiniteTaskPosterior(
            [state.name for state in states],
            [float(spec.get("prior_weight", 1.0)) for spec in specs],
            temperature=self.config.task_posterior_temperature,
            temperature_decay=self.config.task_posterior_temperature_decay,
            output_score_weights=[
                self.config.task_posterior_objective_score_weight,
                self.config.task_posterior_constraint_score_weight,
            ],
            boundary_score_weight=(
                self.config.task_posterior_boundary_score_weight),
            decision_prior_protection_numerator=(
                self.config.task_posterior_prior_protection_numerator),
            decision_prior_protection_max=(
                self.config.task_posterior_prior_protection_max),
            safe_generalized=(
                self.config.task_posterior_safe_generalized),
            safe_boundary_score_weight=(
                self.config.task_posterior_safe_boundary_score_weight),
            safe_pairwise_score_weight=(
                self.config.task_posterior_safe_pairwise_score_weight),
        )
        sensitivity_posterior = None
        sensitivity_mode = str(
            self.config.task_posterior_sensitivity_mode or "off"
        ).lower()
        if sensitivity_mode not in ("off", "none", "disabled"):
            if sensitivity_mode not in (
                "finite",
                "latent",
                "finite_latent",
                "fixed",
                "fixed_balanced",
                "no_latent",
            ):
                raise ValueError(
                    "unknown task sensitivity posterior mode "
                    f"{sensitivity_mode!r}"
                )
            fixed = sensitivity_mode in (
                "fixed", "fixed_balanced", "no_latent")
            prior = ({
                "class_names": ("fixed",),
                "scales": (1.0,),
                "biases": (0.0,),
                "bias_coefficients": None,
                "bias_feature_names": None,
                "decision_penalties": (5.0,),
                "empirical_trust": (0.25,),
                "prior_weights": (1.0,),
            } if fixed else (
                self.problem.task_sensitivity_prior()
                if hasattr(self.problem, "task_sensitivity_prior")
                else {}
            ))
            sensitivity_posterior = FiniteTaskSensitivityPosterior(
                class_names=prior.get(
                    "class_names", ("stable", "balanced", "sensitive")),
                scales=prior.get("scales", (0.5, 1.0, 2.0)),
                biases=prior.get("biases"),
                bias_coefficients=prior.get("bias_coefficients"),
                bias_feature_names=prior.get("bias_feature_names"),
                decision_penalties=prior.get(
                    "decision_penalties", (2.0, 5.0, 20.0)),
                empirical_trust=prior.get(
                    "empirical_trust", (1.0, 0.25, 0.0)),
                prior_weights=prior.get("prior_weights"),
                temperature=(
                    0.0 if fixed else self.config.task_posterior_temperature),
                temperature_decay=(
                    0.0
                    if fixed
                    else self.config.task_posterior_temperature_decay
                ),
                boundary_score_weight=(
                    0.0
                    if fixed
                    else self.config.task_posterior_boundary_score_weight
                ),
            )
        self.task_ensemble = FiniteTaskModelEnsemble(
            states,
            posterior,
            kl_radius_numerator=(
                self.config.task_posterior_kl_radius_numerator),
            confidence_delta=(
                self.config.task_posterior_confidence_delta),
            maximum_kl_radius=self.config.task_posterior_max_kl_radius,
            pilot_count=pilot_count,
            sensitivity_posterior=sensitivity_posterior,
            safe_pairwise_max_history=(
                self.config.task_posterior_safe_pairwise_max_history),
            safe_pairwise_probability_floor=(
                self.config.task_posterior_safe_pairwise_probability_floor),
            task_latent_inference_mode=(
                self.config.task_latent_inference_mode),
            task_latent_calibration_mode=(
                self.config.task_latent_calibration_mode),
            source_discrepancy_update=(
                self.config.source_discrepancy_update),
            variance_structure_posterior_mode=(
                self.config.task_variance_posterior_mode),
        )
        # Score the remaining initial observations prequentially: each label
        # updates Q before it is inserted into any expert GPR/HVD.
        for x in samples[pilot_count:]:
            y = np.asarray(self.observations[x][0], dtype=float)
            self.task_ensemble.update(
                x,
                y,
                existing_observations=[],
                tau=self.problem.tau,
            )
        return self.task_ensemble

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
        nominal_noise = max(
            float(getattr(self.problem, "sigma_level", 0.0)) ** 2,
            1e-6,
        )
        source_posterior = self._fit_source_constraint_coefficient_posterior(
            model,
            output_index,
            samples,
            y_i,
            nominal_noise,
        )
        if source_posterior is not None:
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
                    observed = float(
                        np.asarray(history_y, dtype=float)[output_index])
                    model.update(history_x, observed, sigma2)
                    replayed += 1
            return {
                "initial_records": int(len(initial_history)),
                "replayed_updates": int(replayed),
                "basis_dim": int(model.p),
                "source_constraint_coefficient_prior": True,
                "source_constraint_mean_adaptation_mode": str(
                    source_posterior.get("adaptation_mode", "unknown")),
            }
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
            "source_parametric_prior_diagnostics": copy.deepcopy(
                getattr(model, "source_parametric_prior_diagnostics", None)),
            "decision_covariance": (
                None
                if getattr(model, "_decision_covariance", None) is None
                else np.asarray(
                    model._decision_covariance, dtype=float).copy()
            ),
            "decision_lambda_i": getattr(model, "_decision_lambda_i", None),
            "source_conditioned_confidence": copy.deepcopy(getattr(
                model, "_source_conditioned_confidence", None)),
            "finite_mixture_sequential": bool(getattr(
                model, "_finite_mixture_sequential", False)),
            "finite_mixture_weights": (
                None
                if getattr(model, "_finite_mixture_weights", None) is None
                else np.asarray(
                    model._finite_mixture_weights, dtype=float).copy()
            ),
            "finite_mixture_component_names": list(getattr(
                model, "_finite_mixture_component_names", [])),
            "finite_mixture_update_count": int(getattr(
                model, "_finite_mixture_update_count", 0)),
            "finite_mixture_hierarchical_misspecification": bool(getattr(
                model,
                "_finite_mixture_hierarchical_misspecification",
                False,
            )),
            "finite_mixture_component_priors": copy.deepcopy(getattr(
                model, "_finite_mixture_component_priors", [])),
            "finite_mixture_prior_weights": (
                None
                if getattr(model, "_finite_mixture_prior_weights", None) is None
                else np.asarray(
                    model._finite_mixture_prior_weights, dtype=float).copy()
            ),
            "finite_mixture_target_history": copy.deepcopy(getattr(
                model, "_finite_mixture_target_history", [])),
            "finite_mixture_misspecification_prior_df": float(getattr(
                model, "_finite_mixture_misspecification_prior_df", 4.0)),
            "finite_mixture_misspecification_max_scale": float(getattr(
                model, "_finite_mixture_misspecification_max_scale", 100.0)),
            "finite_mixture_misspecification_mode": str(getattr(
                model,
                "_finite_mixture_misspecification_mode",
                "hierarchical_predictive_scale",
            )),
            "finite_mixture_misspecification_ridge": float(getattr(
                model, "_finite_mixture_misspecification_ridge", 1.0)),
            "finite_mixture_misspecification_delta": float(getattr(
                model, "_finite_mixture_misspecification_delta", 0.05)),
            "finite_mixture_components": [
                self._gpr_checkpoint_state(component)
                for component in getattr(
                    model, "_finite_mixture_components", [])
            ],
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
        model.source_parametric_prior_diagnostics = copy.deepcopy(
            state.get("source_parametric_prior_diagnostics"))
        saved_decision_covariance = state.get("decision_covariance")
        model._decision_covariance = (
            None
            if saved_decision_covariance is None
            else np.asarray(saved_decision_covariance, dtype=float).copy()
        )
        model._decision_lambda_i = state.get("decision_lambda_i")
        model._source_conditioned_confidence = copy.deepcopy(
            state.get("source_conditioned_confidence"))
        saved_components = list(state.get("finite_mixture_components", []))
        current_components = list(getattr(
            model, "_finite_mixture_components", []))
        if saved_components:
            if len(saved_components) != len(current_components):
                raise ValueError(
                    "checkpoint source-mixture component count does not match config"
                )
            for component, component_state in zip(
                current_components, saved_components
            ):
                self._restore_gpr_checkpoint_state(
                    component, component_state)
        model._finite_mixture_components = current_components
        saved_weight = state.get("finite_mixture_weights")
        model._finite_mixture_weights = (
            None
            if saved_weight is None
            else np.asarray(saved_weight, dtype=float).copy()
        )
        model._finite_mixture_component_names = [
            str(value) for value in state.get(
                "finite_mixture_component_names", [])
        ]
        model._finite_mixture_sequential = bool(state.get(
            "finite_mixture_sequential", False))
        model._finite_mixture_update_count = int(state.get(
            "finite_mixture_update_count", 0))
        model._finite_mixture_hierarchical_misspecification = bool(state.get(
            "finite_mixture_hierarchical_misspecification", False))
        model._finite_mixture_component_priors = copy.deepcopy(state.get(
            "finite_mixture_component_priors", []))
        saved_prior_weight = state.get("finite_mixture_prior_weights")
        model._finite_mixture_prior_weights = (
            None
            if saved_prior_weight is None
            else np.asarray(saved_prior_weight, dtype=float).copy()
        )
        model._finite_mixture_target_history = copy.deepcopy(state.get(
            "finite_mixture_target_history", []))
        model._finite_mixture_misspecification_prior_df = float(state.get(
            "finite_mixture_misspecification_prior_df", 4.0))
        model._finite_mixture_misspecification_max_scale = float(state.get(
            "finite_mixture_misspecification_max_scale", 100.0))
        model._finite_mixture_misspecification_mode = str(state.get(
            "finite_mixture_misspecification_mode",
            "hierarchical_predictive_scale",
        ))
        model._finite_mixture_misspecification_ridge = float(state.get(
            "finite_mixture_misspecification_ridge", 1.0))
        model._finite_mixture_misspecification_delta = float(state.get(
            "finite_mixture_misspecification_delta", 0.05))
        basis_state = state.get("basis_runtime_state")
        if (
            basis_state is not None
            and model.basis_map is not None
            and hasattr(model.basis_map, "load_runtime_state")
        ):
            model.basis_map.load_runtime_state(copy.deepcopy(basis_state))
        model._state_version = int(state.get("state_version", 0)) + 1
        model._torch_cache = {}

    def _task_ensemble_checkpoint_state(self):
        if self.task_ensemble is None:
            return None
        return {
            "posterior": copy.deepcopy(self.task_ensemble.posterior),
            "sensitivity_posterior": copy.deepcopy(
                self.task_ensemble.sensitivity_posterior),
            "task_latent_posterior": copy.deepcopy(
                self.task_ensemble._task_latent()),
            "task_latent_inference_mode": str(
                self.task_ensemble.task_latent_inference_mode),
            "task_latent_calibration_mode": str(
                self.task_ensemble.task_latent_calibration_mode),
            "variance_structure_posterior_mode": str(
                self.task_ensemble.variance_structure_posterior_mode),
            "variance_structure_posterior": copy.deepcopy(
                self.task_ensemble.variance_structure_posterior),
            "last_update": copy.deepcopy(self.task_ensemble.last_update),
            "pilot_count": int(self.task_ensemble.pilot_count),
            "safe_history": copy.deepcopy(
                self.task_ensemble.safe_history),
            "states": [
                {
                    "name": state.name,
                    "gpr": [
                        self._gpr_checkpoint_state(model)
                        for model in state.gpr_models
                    ],
                    "variance_model": copy.deepcopy(
                        state.variance_model.__getstate__()),
                }
                for state in self.task_ensemble.states
            ],
        }

    def _restore_task_ensemble_checkpoint_state(self, state):
        if state is None:
            self.task_ensemble = None
            return
        samples = [
            tuple(int(v) for v in x)
            for x, _ in self.history[: int(self.config.n0)]
        ]
        self._initialize_task_ensemble(samples)
        if self.task_ensemble is None:
            raise RuntimeError("checkpoint contains a task posterior but config disables it")
        saved_states = list(state.get("states", []))
        saved_names = [str(item.get("name")) for item in saved_states]
        current_names = [item.name for item in self.task_ensemble.states]
        if saved_names != current_names:
            raise ValueError("checkpoint task experts do not match current source prior")
        for expert, saved in zip(self.task_ensemble.states, saved_states):
            for model, model_state in zip(
                expert.gpr_models, saved.get("gpr", [])
            ):
                self._restore_gpr_checkpoint_state(model, model_state)
            expert.variance_model.__setstate__(copy.deepcopy(
                saved["variance_model"]))
            expert.variance_model._last_problem = expert.problem
        self.task_ensemble.posterior = copy.deepcopy(state["posterior"])
        if "sensitivity_posterior" in state:
            self.task_ensemble.sensitivity_posterior = copy.deepcopy(
                state["sensitivity_posterior"])
        if "task_latent_posterior" in state:
            self.task_ensemble.task_latent_posterior = copy.deepcopy(
                state["task_latent_posterior"])
        self.task_ensemble.task_latent_inference_mode = str(state.get(
            "task_latent_inference_mode",
            self.task_ensemble.task_latent_inference_mode,
        ))
        self.task_ensemble.task_latent_calibration_mode = str(state.get(
            "task_latent_calibration_mode",
            self.task_ensemble.task_latent_calibration_mode,
        ))
        self.task_ensemble.variance_structure_posterior_mode = str(state.get(
            "variance_structure_posterior_mode",
            self.task_ensemble.variance_structure_posterior_mode,
        ))
        if "variance_structure_posterior" in state:
            self.task_ensemble.variance_structure_posterior = copy.deepcopy(
                state["variance_structure_posterior"])
        self.task_ensemble.pilot_count = int(state.get(
            "pilot_count", self.task_ensemble.pilot_count))
        self.task_ensemble.safe_history = copy.deepcopy(state.get(
            "safe_history", self.task_ensemble.safe_history))
        self.task_ensemble.last_update = copy.deepcopy(
            state.get("last_update", {"status": "restored"}))

    def _clone_gpr_for_exact_kg(self, model):
        """Clone mutable GPR state without deep-copying simulator/provider handles."""
        clone = object.__new__(model.__class__)
        clone.__dict__ = model.__dict__.copy()
        clone.a = np.asarray(model.a, dtype=float).copy()
        clone.C = np.asarray(model.C, dtype=float).copy()
        clone._decision_covariance = (
            None
            if getattr(model, "_decision_covariance", None) is None
            else np.asarray(
                model._decision_covariance, dtype=float).copy()
        )
        clone._decision_lambda_i = (
            None
            if getattr(model, "_decision_lambda_i", None) is None
            else float(model._decision_lambda_i)
        )
        clone._source_conditioned_confidence = copy.deepcopy(
            getattr(model, "_source_conditioned_confidence", None))
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
        clone.source_parametric_prior_diagnostics = copy.deepcopy(
            getattr(model, "source_parametric_prior_diagnostics", None))
        clone._finite_mixture_components = [
            self._clone_gpr_for_exact_kg(component)
            for component in getattr(
                model, "_finite_mixture_components", [])
        ]
        clone._finite_mixture_weights = (
            None
            if getattr(model, "_finite_mixture_weights", None) is None
            else np.asarray(
                model._finite_mixture_weights, dtype=float).copy()
        )
        clone._finite_mixture_component_names = list(getattr(
            model, "_finite_mixture_component_names", []))
        clone._finite_mixture_sequential = bool(getattr(
            model, "_finite_mixture_sequential", False))
        clone._finite_mixture_update_count = int(getattr(
            model, "_finite_mixture_update_count", 0))
        clone._finite_mixture_hierarchical_misspecification = bool(getattr(
            model,
            "_finite_mixture_hierarchical_misspecification",
            False,
        ))
        clone._finite_mixture_component_priors = copy.deepcopy(getattr(
            model, "_finite_mixture_component_priors", []))
        clone._finite_mixture_prior_weights = (
            None
            if getattr(model, "_finite_mixture_prior_weights", None) is None
            else np.asarray(
                model._finite_mixture_prior_weights, dtype=float).copy()
        )
        clone._finite_mixture_target_history = copy.deepcopy(getattr(
            model, "_finite_mixture_target_history", []))
        clone._finite_mixture_misspecification_prior_df = float(getattr(
            model, "_finite_mixture_misspecification_prior_df", 4.0))
        clone._finite_mixture_misspecification_max_scale = float(getattr(
            model, "_finite_mixture_misspecification_max_scale", 100.0))
        clone._finite_mixture_misspecification_mode = str(getattr(
            model,
            "_finite_mixture_misspecification_mode",
            "hierarchical_predictive_scale",
        ))
        clone._finite_mixture_misspecification_ridge = float(getattr(
            model, "_finite_mixture_misspecification_ridge", 1.0))
        clone._finite_mixture_misspecification_delta = float(getattr(
            model, "_finite_mixture_misspecification_delta", 0.05))
        if clone._adaptive_sparsity is not None and model.basis_map is not None:
            basis_clone = object.__new__(model.basis_map.__class__)
            basis_clone.__dict__ = model.basis_map.__dict__.copy()
            if hasattr(model.basis_map, "_adaptive_sparsity_diagnostics"):
                basis_clone._adaptive_sparsity_diagnostics = copy.deepcopy(
                    model.basis_map._adaptive_sparsity_diagnostics)
            clone.basis_map = basis_clone
        clone._torch_cache = {}
        return clone

    def _clone_variance_model_for_exact_kg(self, model=None, problem=None):
        model = self.variance_model if model is None else model
        state = copy.deepcopy(model.__getstate__())
        clone = object.__new__(model.__class__)
        clone.__setstate__(state)
        clone._last_problem = self.problem if problem is None else problem
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
            "task_initial_design": copy.deepcopy(
                self._task_initial_design_info),
            "certification_recheck_targets": [
                list(map(int, x))
                for x in self._certification_recheck_targets
            ],
            "finalist_replication": {
                "initialized": bool(self._finalist_replication_initialized),
                "targets": [
                    list(map(int, x))
                    for x in self._finalist_replication_targets
                ],
                "labels": list(self._finalist_replication_labels),
                "frozen_stage": self._finalist_replication_frozen_stage,
                "active_target": (
                    None
                    if self._finalist_replication_active_target is None
                    else list(map(
                        int, self._finalist_replication_active_target))
                ),
                "active_label": self._finalist_replication_active_label,
                "refresh_history": copy.deepcopy(
                    self._finalist_replication_refresh_history),
                "fixed_universe_pool": [
                    list(map(int, x))
                    for x in self._finalist_replication_pool
                ],
            },
            "last_terminal_pool": [
                list(map(int, x)) for x in self._last_terminal_pool
            ],
            "posterior_dominance": {
                "incumbent": (
                    None
                    if self._posterior_dominance_incumbent is None
                    else list(map(
                        int, self._posterior_dominance_incumbent))
                ),
                "history": copy.deepcopy(
                    self._posterior_dominance_history),
            },
            "gpr": [self._gpr_checkpoint_state(model) for model in self.gpr],
            "variance_model": copy.deepcopy(self.variance_model.__getstate__()),
            "task_ensemble": self._task_ensemble_checkpoint_state(),
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
        interval = int(self.config.checkpoint_interval)
        if interval <= 0:
            return None
        should_save = (
            force
            or reason in {"initial", "final"}
            or int(next_stage_n) >= int(self.config.N)
            or int(next_stage_n) % interval == 0
        )
        if not should_save:
            return None
        payload = self._runtime_checkpoint_payload(next_stage_n, reason)
        latest_path = root / "checkpoint_latest.pkl"
        self._write_pickle_atomic(latest_path, payload)
        stage_path = root / f"checkpoint_stage_{int(next_stage_n):05d}.pkl"
        stage_tmp = stage_path.with_name(stage_path.name + ".tmp")
        try:
            stage_tmp.unlink(missing_ok=True)
            os.link(latest_path, stage_tmp)
            os.replace(stage_tmp, stage_path)
        except OSError:
            stage_tmp.unlink(missing_ok=True)
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
        if self.task_ensemble is not None:
            for state in self.task_ensemble.states:
                state.variance_model._last_problem = state.problem

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
        self._task_initial_design_info = copy.deepcopy(payload.get(
            "task_initial_design",
            {"status": "checkpoint_legacy", "requested": 0},
        ))
        saved_recheck = payload.get("certification_recheck_targets")
        if saved_recheck is None:
            initial_samples = [
                tuple(int(v) for v in x)
                for x, _ in self.history[: int(self.config.n0)]
            ]
            self._initialize_certification_recheck_targets(initial_samples)
        else:
            self._certification_recheck_targets = [
                tuple(int(v) for v in x) for x in saved_recheck
            ]
        saved_finalists = payload.get("finalist_replication") or {}
        self._finalist_replication_initialized = bool(
            saved_finalists.get("initialized", False))
        self._finalist_replication_targets = [
            tuple(int(v) for v in x)
            for x in saved_finalists.get("targets", [])
        ]
        self._finalist_replication_labels = [
            str(value) for value in saved_finalists.get("labels", [])
        ]
        frozen_stage = saved_finalists.get("frozen_stage")
        self._finalist_replication_frozen_stage = (
            None if frozen_stage is None else int(frozen_stage)
        )
        active_target = saved_finalists.get("active_target")
        self._finalist_replication_active_target = (
            None
            if active_target is None
            else tuple(int(v) for v in active_target)
        )
        active_label = saved_finalists.get("active_label")
        self._finalist_replication_active_label = (
            None if active_label is None else str(active_label)
        )
        self._finalist_replication_refresh_history = copy.deepcopy(
            saved_finalists.get("refresh_history", []))
        self._finalist_replication_pool = [
            tuple(int(v) for v in x)
            for x in saved_finalists.get("fixed_universe_pool", [])
        ]
        self._last_terminal_pool = [
            tuple(int(v) for v in x)
            for x in payload.get("last_terminal_pool", [])
        ]
        dominance_state = payload.get("posterior_dominance") or {}
        dominance_incumbent = dominance_state.get("incumbent")
        self._posterior_dominance_incumbent = (
            None
            if dominance_incumbent is None
            else tuple(int(v) for v in dominance_incumbent)
        )
        self._posterior_dominance_history = copy.deepcopy(
            dominance_state.get("history", []))
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
        task_state = copy.deepcopy(payload.get("task_ensemble"))
        if task_state is not None:
            self._restore_task_ensemble_checkpoint_state(task_state)
        elif self._task_posterior_requested():
            samples = [
                tuple(int(v) for v in x)
                for x, _ in self.history[: int(self.config.n0)]
            ]
            self._initialize_task_ensemble(samples)
        self._reattach_runtime_handles_after_checkpoint()
        if (
            self._posterior_dominance_active()
            and self._posterior_dominance_incumbent is None
        ):
            samples = [
                tuple(int(v) for v in x)
                for x, _ in self.history[: int(self.config.n0)]
            ]
            self._initialize_posterior_dominance_incumbent(samples)
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
        dominance_initial = self._initialize_posterior_dominance_incumbent(
            samples)
        initial_truth_audit = self._truth_pool_diagnostics(
            samples,
            prefix="initial_design",
        )
        if initial_truth_audit:
            self._task_initial_design_info["truth_audit"] = copy.deepcopy(
                initial_truth_audit)
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
            "mean_risk_coordinate_contract": (
                self.problem.mean_risk_coordinate_contract()
                if hasattr(self.problem, "mean_risk_coordinate_contract")
                else None
            ),
            "task_posterior": (
                None
                if self.task_ensemble is None
                else self.task_ensemble.diagnostics()
            ),
            "initial_design_truth_audit": initial_truth_audit,
            "posterior_dominance": dominance_initial,
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

    def _task_robust_certificate_mode(self):
        mode = str(
            self.config.task_posterior_robust_certificate_mode
            or "separable"
        ).strip().lower().replace("-", "_")
        aliases = {
            "legacy": "separable",
            "joint": "joint_tangent",
            "joint_kl": "joint_tangent",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"separable", "joint_tangent"}:
            raise ValueError(
                "task posterior robust certificate mode must be "
                "separable or joint_tangent"
            )
        return mode

    def _certification_head_authority(self):
        mode = str(
            self.config.certification_head_authority or "task_joint"
        ).strip().lower().replace("-", "_")
        aliases = {
            "legacy": "task_joint",
            "joint": "task_joint",
            "split_task": "split_gpr_task_hvd",
            "split_cumulative": "split_gpr_cumulative_hvd",
        }
        mode = aliases.get(mode, mode)
        allowed = {
            "task_joint",
            "split_gpr_task_hvd",
            "split_gpr_cumulative_hvd",
        }
        if mode not in allowed:
            raise ValueError(
                "certification head authority must be task_joint, "
                "split_gpr_task_hvd, or split_gpr_cumulative_hvd"
            )
        return mode

    def _certification_source(self, task_ensemble_active=None):
        authority = self._certification_head_authority()
        active = (
            self.task_ensemble is not None
            if task_ensemble_active is None
            else bool(task_ensemble_active)
        )
        if not active:
            return "theory_hvd"
        return {
            "task_joint": (
                "task_joint_kl_hvd"
                if self._task_robust_certificate_mode() == "joint_tangent"
                else "task_separable_kl_hvd"
            ),
            "split_gpr_task_hvd": "split_aggregate_gpr_task_hvd",
            "split_gpr_cumulative_hvd": (
                "split_aggregate_gpr_cumulative_hvd"
            ),
        }[authority]

    def _pilot_constraint_guard(self):
        if not hasattr(self.problem, "pilot_constraint_guard"):
            return 0.0
        try:
            return max(float(self.problem.pilot_constraint_guard()), 0.0)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _objective_posterior_mean_many(self, candidates):
        if self.task_ensemble is None:
            return self.gpr[0].posterior_mean_many(candidates)
        return self.task_ensemble.mixture_moments_many(
            0, candidates, certification=False).mean

    def _certification_result(
        self,
        mu_con,
        candidates,
        v_con=None,
        epistemic=None,
    ):
        task_robust = None
        if self.task_ensemble is not None:
            authority = self._certification_head_authority()
            if authority != "task_joint":
                # The aggregate GPR is the sole mean/epistemic authority.
                # Inputs from callers are intentionally ignored so an old
                # joint task posterior cannot silently retake either head.
                mu_con = self.gpr[1].posterior_mean_many(candidates)
                epistemic = self._constraint_certification_epistemic_many(
                    self.gpr[1], candidates)
                if authority == "split_gpr_task_hvd":
                    task_robust = self.task_ensemble.robust_moments_many(
                        1, candidates, certification=True)
                    v_con = task_robust.aleatoric_upper
                else:
                    v_con = (
                        self.variance_model
                        .predict_certification_variance_many(
                            1, candidates, self.problem)
                    )
            elif mu_con is None or epistemic is None or v_con is None:
                task_robust = self.task_ensemble.robust_moments_many(
                    1, candidates, certification=True)
                mu_con = task_robust.mean_upper
                epistemic = task_robust.epistemic_upper
                v_con = task_robust.aleatoric_upper
        else:
            if v_con is None:
                v_con = self.variance_model.predict_certification_variance_many(
                    1, candidates, self.problem)
            if epistemic is None:
                epistemic = self._constraint_certification_epistemic_many(
                    self.gpr[1], candidates)
        guard = (
            0.0
            if self.task_ensemble is not None
            else self._pilot_constraint_guard()
        )
        cert = conservative_chance_margin(
            np.asarray(mu_con, dtype=float) + guard,
            epistemic,
            v_con,
            tau=self.problem.tau,
            alpha=self.problem.alpha,
            beta_g=self.config.beta_g,
            mode=self.config.certification_mode,
        )
        if (
            task_robust is None
            or self._certification_head_authority() != "task_joint"
            or self._task_robust_certificate_mode() != "joint_tangent"
        ):
            return cert
        joint = self.task_ensemble.robust_chance_margin_many(
            candidates,
            beta_g=cert.beta_g,
            z_alpha=cert.z_alpha,
            tau=self.problem.tau,
            certification=True,
        )
        return CertificationResult(
            margin=np.asarray(joint.upper, dtype=float),
            mu=cert.mu,
            epistemic_var=cert.epistemic_var,
            aleatoric_var=cert.aleatoric_var,
            beta_g=cert.beta_g,
            z_alpha=cert.z_alpha,
            tau=cert.tau,
            mode=cert.mode,
        )

    def _tcb_v2_mode(self):
        mode = str(self.config.tcb_v2_mode or "off").lower()
        aliases = {
            "disabled": "off",
            "none": "off",
            "nomination": "frontier",
            "terminal": "frontier",
            "main": "certified",
            "authoritative": "certified",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"off", "shadow", "frontier", "certified"}:
            raise ValueError(f"unknown TCB-V2 mode {mode!r}")
        return mode

    def _tcb_v2_source_model(self):
        if self._tcb_v2_mode() == "off":
            return None
        if not hasattr(self.problem, "hierarchical_boundary_model"):
            return None
        model = self.problem.hierarchical_boundary_model()
        if model is None or getattr(model, "fit_status", "unfit") != "fit":
            return None
        return model

    def _tcb_v2_margin_many(
        self,
        candidates,
        *,
        task_ensemble=None,
        observations=None,
    ):
        """Fit the two-dimensional target effect and return mean/upper margin."""
        model = self._tcb_v2_source_model()
        candidates = [tuple(int(v) for v in x) for x in candidates]
        if model is None or not candidates:
            return None
        task_ensemble = (
            self.task_ensemble if task_ensemble is None else task_ensemble)
        observations = self.observations if observations is None else observations
        if task_ensemble is None:
            return None
        pilot_points = []
        pilot_margin = []
        pilot_variance = []
        pilot_replicates = []
        for key, values in observations.items():
            point = tuple(int(v) for v in key)
            values = np.asarray(values, dtype=float)
            if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2:
                continue
            robust = task_ensemble.robust_moments_many(
                1, [point], certification=True)
            aleatoric = max(float(robust.aleatoric_upper[0]), 1e-12)
            pilot_points.append(point)
            pilot_margin.append(
                float(np.mean(values[:, 1]))
                + float(norm.ppf(1.0 - self.problem.alpha))
                * np.sqrt(aleatoric)
                - float(self.problem.tau)
            )
            pilot_variance.append(aleatoric)
            pilot_replicates.append(int(len(values)))
        descriptors = np.vstack([
            np.asarray(
                self.problem.hierarchical_boundary_descriptor(x),
                dtype=float,
            )
            for x in candidates
        ])
        if pilot_points:
            pilot_descriptors = np.vstack([
                np.asarray(
                    self.problem.hierarchical_boundary_descriptor(x),
                    dtype=float,
                )
                for x in pilot_points
            ])
            adapter = model.fit_target_adapter(
                pilot_descriptors,
                np.asarray(pilot_margin, dtype=float),
                pilot_variance=np.asarray(pilot_variance, dtype=float),
                replicate_count=np.asarray(pilot_replicates, dtype=float),
            )
        else:
            adapter = model.prior_adapter()
        mean = np.asarray(
            model.predict(descriptors, adapter=adapter), dtype=float)
        upper = np.asarray(
            model.predict_upper(descriptors, adapter=adapter), dtype=float)
        return {
            "mean": mean,
            "upper": upper,
            "adapter": adapter,
            "adapter_diagnostics": copy.deepcopy(adapter.diagnostics),
            "pilot_points": int(len(pilot_points)),
            "target_oracle_used": False,
        }

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

    def _safe_interior_candidates(self, rng=None):
        n_safe = int(self.config.safe_interior_candidate_count)
        if n_safe <= 0:
            return []
        rng = self.rng if rng is None else rng
        pool_size = max(n_safe, int(self.config.safe_interior_pool_size))
        pool = unique_candidates(random_candidates(self.problem, pool_size, rng))
        if not pool:
            return []
        try:
            mu_con = self.gpr[1].posterior_mean_many(pool)
            mu_obj = self._objective_posterior_mean_many(pool)
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

    def _constraint_uncertain_candidates(self, rng=None):
        n_uncertain = int(self.config.constraint_uncertain_candidate_count)
        if n_uncertain <= 0:
            return []
        rng = self.rng if rng is None else rng
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
                        rng=rng,
                        observed=[x for x, _ in self.history],
                    ))
                except Exception:
                    pass
        n_random_pool = max(pool_size - len(pool), n_uncertain)
        pool.extend(random_candidates(self.problem, n_random_pool, rng))
        pool = unique_candidates(pool)
        if not pool:
            return []
        try:
            mu_con = self.gpr[1].posterior_mean_many(pool)
            epistemic = self._constraint_certification_epistemic_many(
                self.gpr[1], pool)
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

    def _replication_candidate_budget(self):
        configured = max(0, int(self.config.replication_candidate_count))
        if self.config.adaptive_replication_voi and configured == 0:
            return max(1, min(4, int(self.config.n0)))
        return configured

    def _replication_candidates(self):
        n_replicate = self._replication_candidate_budget()
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
        epistemic = self._constraint_decision_epistemic_many(
            self.gpr[1], pool)
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

    def _task_expert_proposal_batches(
        self,
        n,
        rng,
        *,
        record=False,
        iteration=None,
    ):
        """Draw candidates from the posterior mixture of frozen experts."""
        n = max(0, int(n))
        if (
            n == 0
            or self.task_ensemble is None
            or not hasattr(self.problem, "task_expert_proposal_candidates")
        ):
            if record:
                self._last_task_proposal_info = {
                    "status": "disabled",
                    "requested": int(n),
                }
            return []
        exploration = float(np.clip(
            self.config.task_posterior_proposal_exploration,
            0.0,
            1.0,
        ))
        allocation = self.task_ensemble.structure_proposal_allocation(
            n,
            exploration=exploration,
            minimum_per_expert=(
                self.config.task_posterior_proposal_min_per_expert),
        )
        batches = []
        generated = {}
        for name in self.task_ensemble.posterior.expert_names:
            count = int(allocation.get(name, 0))
            if count <= 0:
                generated[name] = 0
                continue
            expert_rng = (
                rng
                if iteration is None
                else self._proposal_rng(
                    iteration, f"task_expert:{name}")
            )
            rows = self.problem.task_expert_proposal_candidates(
                name,
                n=count,
                rng=expert_rng,
                pool_size=max(
                    count,
                    int(self.config.task_posterior_proposal_pool_size),
                ),
            )
            rows = unique_candidates(rows)[:count]
            generated[name] = int(len(rows))
            if rows:
                batches.append((name, rows))
        if record:
            proposal_weights = (
                self.task_ensemble.structure_proposal_weights(
                    exploration=exploration))
            self._last_task_proposal_info = {
                "status": "generated",
                "requested": int(n),
                "exploration": float(exploration),
                "proposal_weights": {
                    name: float(weight)
                    for name, weight in zip(
                        self.task_ensemble.posterior.expert_names,
                        proposal_weights,
                    )
                },
                "allocation": allocation,
                "generated": generated,
                "source_only": True,
                "target_oracle_used": False,
            }
        return batches

    def _proposal_rng(self, iteration, namespace):
        namespace_tag = int(zlib.crc32(
            str(namespace).encode("utf-8")) & 0xFFFFFFFF)
        return np.random.default_rng(np.random.SeedSequence([
            int(self.config.seed),
            int(iteration),
            PROPOSAL_STREAM_TAG,
            namespace_tag,
        ]))

    def _canonical_sobol_continuation_candidate(self):
        seed = (
            int(self.config.seed)
            + int(self.config.decision_backend_seed_offset)
        )
        if self._canonical_sobol_sequence is None:
            self._canonical_sobol_sequence = sobol_integer_sequence(
                self.problem,
                max(64, int(self.config.N) + int(self.config.n0) + 32),
                seed,
            )
        observed = set(tuple(int(value) for value in point)
                       for point in self.observations)
        for point in self._canonical_sobol_sequence:
            if point not in observed:
                return point
        return next_sobol_integer_candidate(
            self.problem,
            seed,
            observed=observed,
        )

    def _boundary_coordinate_proposal_batches(self, rng, *, record=False):
        """Rank a generic/source-frozen pool with the target-calibrated phi."""

        requested = max(
            0, int(self.config.boundary_coordinate_candidate_count))
        basis_map = getattr(self.gpr[1], "basis_map", None)
        diagnostics = (
            basis_map.diagnostics()
            if basis_map is not None and hasattr(basis_map, "diagnostics")
            else {}
        )
        if (
            requested == 0
            or basis_map is None
            or not getattr(basis_map, "constraint_mean_coordinate", False)
            or str(diagnostics.get("output_mode", "")).lower()
            != "boundary_aligned"
            or not hasattr(self.problem, "boundary_excitation_candidates")
        ):
            if record:
                self._last_boundary_coordinate_proposal_info = {
                    "status": "disabled_or_missing_phi",
                    "requested": int(requested),
                    "coordinate_output_mode": diagnostics.get("output_mode"),
                    "target_oracle_used": False,
                }
            return []

        pool_size = max(
            requested,
            int(self.config.boundary_coordinate_pool_size),
        )
        pool = self.problem.boundary_excitation_candidates(
            n=pool_size,
            rng=rng,
            pool_size=pool_size,
        )
        observed = set(tuple(int(v) for v in x) for x in self.observations)
        pool = [
            tuple(int(v) for v in x)
            for x in unique_candidates(pool)
            if tuple(int(v) for v in x) not in observed
        ]
        if not pool:
            if record:
                self._last_boundary_coordinate_proposal_info = {
                    "status": "empty_pool",
                    "requested": int(requested),
                    "target_oracle_used": False,
                }
            return []

        phi = basis_map.features_many(pool)
        observed_points = list(self.observations)
        observed_phi = (
            basis_map.features_many(observed_points)
            if observed_points
            else np.empty((0, phi.shape[1]), dtype=float)
        )
        if self.task_ensemble is None:
            mu = self.gpr[1].posterior_mean_many(pool)
            epistemic = self._constraint_certification_epistemic_many(
                self.gpr[1], pool)
            cert = self._certification_result(
                mu, pool, epistemic=epistemic)
        else:
            cert = self._certification_result(None, pool)
            mu = np.asarray(cert.mu, dtype=float)
            epistemic = np.asarray(cert.epistemic_var, dtype=float)
        selection = select_boundary_coordinate_candidates(
            phi,
            observed_phi,
            mu,
            epistemic,
            cert.margin,
            count=requested,
            safe_fraction=self.config.boundary_coordinate_safe_fraction,
            boundary_fraction=(
                self.config.boundary_coordinate_boundary_fraction),
            coverage_fraction=(
                self.config.boundary_coordinate_coverage_fraction),
        )
        batches = []
        for role in sorted(set(selection.roles)):
            rows = [
                pool[index]
                for index, selected_role in zip(
                    selection.indices, selection.roles)
                if selected_role == role
            ]
            if rows:
                batches.append((role, rows))
        if record:
            self._last_boundary_coordinate_proposal_info = {
                **selection.diagnostics,
                "coordinate": "phi=source_aligned_chance_boundary",
                "mean_model": "target_conditioned_constraint_gpr",
                "variance_model": "frozen_current_cumulative_hvd",
                "source_pool_contract": copy.deepcopy(getattr(
                    getattr(basis_map, "meta_prior", None),
                    "boundary_excitation_diagnostics",
                    {},
                )),
                "target_observation_count": int(len(self.history)),
                "target_oracle_used": False,
            }
        return batches

    def _generate_candidates(self, iteration):
        candidates = []
        sources = {}

        def add(rows, source):
            for row in rows:
                x_tuple = tuple(int(v) for v in row)
                if x_tuple not in sources:
                    candidates.append(x_tuple)
                    sources[x_tuple] = source

        decision_backend = str(
            self.config.decision_backend or "legacy"
        ).strip().lower().replace("-", "_")
        if decision_backend in {
            "sobol_new", "sobol_new_only",
            "sobol_hvd_voi", "hvd_voi_sobol",
            "sobol_joint_voi", "joint_voi_sobol",
            "sobol_exact_joint_voi", "exact_joint_voi_sobol",
        }:
            self._last_canonical_sobol_candidate = (
                self._canonical_sobol_continuation_candidate())
            add([
                self._last_canonical_sobol_candidate
            ], "sobol_continuation")
        else:
            self._last_canonical_sobol_candidate = None

        add(latin_hypercube_candidates(
            self.problem,
            self.config.K1,
            self._proposal_rng(iteration, "lhs"),
        ), "lhs")
        n_axis = self._axis_candidate_count()
        add(axis_landmark_candidates(
            self.problem,
            n_axis,
            self._proposal_rng(iteration, "axis_landmark"),
        ), "axis_landmark")
        add(axis_candidates(
            self.problem,
            n_axis,
            self._proposal_rng(iteration, "axis"),
        ), "axis")
        if hasattr(self.problem, "structured_candidates"):
            n_structured = int(self.config.structured_candidate_count)
            if n_structured < 0:
                n_structured = max(5, self.config.K1 // 2)
            add(structured_candidates(
                self.problem,
                n_structured,
                self._proposal_rng(iteration, "structured"),
            ), "structured")
        if self.config.use_state_coupling and self.encoder is not None:
            n_state = int(self.config.state_candidate_count)
            if n_state < 0:
                n_state = max(5, self.config.K1)
            add(self.encoder.state_space_candidates(
                n_anchors=n_state,
                inverse_pool_size=self.config.state_inverse_pool_size,
                inverse_neighbors=self.config.state_inverse_neighbors,
                rng=self._proposal_rng(iteration, "state"),
                observed=[x for x, _ in self.history],
            ), "state")
        for expert_name, rows in self._task_expert_proposal_batches(
            self.config.task_posterior_candidate_count,
            self._proposal_rng(iteration, "task_experts"),
            record=True,
            iteration=iteration,
        ):
            add(rows, f"task_expert:{expert_name}")
        for role, rows in self._boundary_coordinate_proposal_batches(
            self._proposal_rng(iteration, "boundary_coordinate"),
            record=True,
        ):
            add(rows, f"boundary_phi:{role}")
        if hasattr(self.problem, "frozen_source_consensus_candidates"):
            add(
                self.problem.frozen_source_consensus_candidates(),
                "source_consensus_frozen",
            )
        add(self._constraint_uncertain_candidates(
            self._proposal_rng(iteration, "constraint_uncertain")
        ), "constraint_uncertain")
        add(self._replication_candidates(), "replication")
        add(self._safe_interior_candidates(
            self._proposal_rng(iteration, "safe_interior")
        ), "safe_interior")
        add(self._observed_neighbor_candidates(
            rng=self._proposal_rng(iteration, "observed_neighbor")
        ), "observed_neighbor")
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
                    rng=self._proposal_rng(iteration, "llm_inverse"),
                    pool_size=self.config.llm_prior_inverse_pool_size,
                    gate=info.get("gate", 0.0),
                ), "llm_prior")
        add(random_candidates(
            self.problem,
            max(5, self.config.K1 // 5),
            self._proposal_rng(iteration, "random"),
        ), "random")
        use_constraint = iteration > self.config.n_thr
        add(posterior_sample_candidates(
            self.problem,
            self.gpr,
            n_batches=self.config.K2,
            pool_size=self.config.posterior_pool_size,
            keep_per_batch=self.config.posterior_keep,
            rng=self._proposal_rng(iteration, "posterior"),
            use_constraint=use_constraint,
            variance_lookup=self._variance_lookup,
            epistemic_lookup=self._constraint_epistemic_lookup,
            tau=self.problem.tau,
            alpha_z=norm.ppf(1 - self.problem.alpha),
            beta_g=self.config.beta_g,
            certification_mode=self.config.certification_mode,
        ), "posterior")
        if not candidates:
            add([self.problem.sample_random(
                self._proposal_rng(iteration, "fallback_random")
            )], "random")
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
        for _, rows in self._task_expert_proposal_batches(
            self.config.task_posterior_recommendation_count,
            self.rec_rng,
            record=False,
        ):
            for x in rows:
                pool.add(tuple(x))
        if hasattr(self.problem, "frozen_source_consensus_candidates"):
            for x in self.problem.frozen_source_consensus_candidates():
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
            values = np.asarray(ys, dtype=float)
            y_bar = np.mean(values, axis=0)
            sigma_used = sigma_floor
            sigma_source = "global_floor"
            if (
                self.config.observed_incumbent_use_replicate_variance
                and len(values)
                >= max(2, int(
                    self.config.certification_recheck_min_replicates))
            ):
                sample_var = float(np.var(values[:, 1], ddof=1))
                prior_df = max(float(
                    self.config.certification_recheck_variance_prior_df), 0.0)
                numerator = (
                    (len(values) - 1) * max(sample_var, 0.0)
                    + prior_df * sigma_floor ** 2
                )
                denominator = max(len(values) - 1 + prior_df, 1.0)
                sigma_used = float(np.sqrt(max(
                    numerator / denominator,
                    1e-12,
                )))
                sigma_source = "replicate_shrinkage"
            margin = float(y_bar[1] + z * sigma_used - self.problem.tau)
            if margin > margin_limit:
                continue
            item = (
                float(y_bar[0]),
                margin,
                tuple(int(v) for v in x),
                float(sigma_used),
                str(sigma_source),
                int(len(values)),
            )
            if best is None or item < best:
                best = item
        if best is None:
            return None
        obj, margin, x, sigma_used, sigma_source, replicate_count = best
        return {
            "x": x,
            "empirical_objective": float(obj),
            "empirical_chance_margin": float(margin),
            "empirical_sigma": float(sigma_used),
            "empirical_sigma_source": str(sigma_source),
            "replicate_count": int(replicate_count),
        }

    def _observed_safety_challengers(self, limit=None):
        """Rank charged candidates by their empirical chance margin."""

        z_alpha = float(norm.ppf(1.0 - self.problem.alpha))
        sigma_floor = max(
            float(getattr(self.problem, "sigma_level", 0.0)), 1e-8)
        protected = set()
        if hasattr(self.problem, "frozen_source_coverage_candidates"):
            requested = max(
                0,
                int(self.config.task_posterior_mandatory_universal_count),
            )
            protected = {
                tuple(int(v) for v in x)
                for x in self.problem.frozen_source_coverage_candidates(
                    n=requested)
            }
        protected_observed = protected.intersection(self.observations)
        eligible = protected_observed if protected_observed else None
        rows = []
        for x, ys in self.observations.items():
            x = tuple(int(v) for v in x)
            if eligible is not None and x not in eligible:
                continue
            values = np.asarray(ys, dtype=float)
            if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2:
                continue
            mean = np.mean(values, axis=0)
            sigma = sigma_floor
            sigma_source = "global_floor"
            if (
                self.config.observed_incumbent_use_replicate_variance
                and len(values) >= 2
            ):
                sample_variance = float(np.var(values[:, 1], ddof=1))
                prior_df = max(float(
                    self.config.certification_recheck_variance_prior_df), 0.0)
                sigma = float(np.sqrt(max(
                    (
                        (len(values) - 1) * max(sample_variance, 0.0)
                        + prior_df * sigma_floor ** 2
                    ) / max(len(values) - 1 + prior_df, 1.0),
                    1e-12,
                )))
                sigma_source = "replicate_shrinkage"
            margin = float(
                mean[1] + z_alpha * sigma - float(self.problem.tau))
            rows.append((
                margin,
                float(mean[0]),
                x,
                float(sigma),
                str(sigma_source),
                int(len(values)),
            ))
        rows.sort()
        if limit is not None:
            rows = rows[:max(0, int(limit))]
        selection_scope = (
            "frozen_source_coverage"
            if eligible is not None else "all_charged_observations"
        )
        return [
            {
                "x": x,
                "empirical_objective": float(objective),
                "empirical_chance_margin": float(margin),
                "empirical_sigma": float(sigma),
                "empirical_sigma_source": str(sigma_source),
                "replicate_count": int(replicate_count),
                "selection_rank": int(rank),
                "selection_scope": selection_scope,
                "protected_candidate_count": int(len(protected_observed)),
                "target_oracle_used": False,
            }
            for rank, (
                margin,
                objective,
                x,
                sigma,
                sigma_source,
                replicate_count,
            ) in enumerate(rows, start=1)
        ]

    def _observed_safety_challenger(self):
        """Return the first ranked charged safety challenger, if present."""

        challengers = self._observed_safety_challengers(limit=1)
        return challengers[0] if challengers else None

    def _initialize_certification_recheck_targets(self, samples):
        top_k = max(0, int(self.config.certification_recheck_top_k))
        self._certification_recheck_targets = []
        if top_k == 0:
            return []
        z_alpha = float(norm.ppf(1 - self.problem.alpha))
        sigma_floor = max(
            float(getattr(self.problem, "sigma_level", 0.0)), 1e-8)
        soft_limit = (
            float(self.config.certification_recheck_soft_margin_scale)
            * sigma_floor
        )
        scored = []
        for x in unique_candidates(samples):
            key = tuple(int(v) for v in x)
            values = self.observations.get(key, [])
            if not values:
                continue
            y_bar = np.mean(np.asarray(values, dtype=float), axis=0)
            proxy_margin = float(
                y_bar[1] + z_alpha * sigma_floor - self.problem.tau)
            if proxy_margin > soft_limit:
                continue
            scored.append((
                abs(proxy_margin),
                float(y_bar[0]),
                proxy_margin,
                key,
            ))
        scored.sort()
        self._certification_recheck_targets = [
            row[3] for row in scored[:top_k]
        ]
        self._task_initial_design_info["certification_recheck"] = {
            "status": (
                "initialized"
                if self._certification_recheck_targets
                else "no_soft_boundary_target"
            ),
            "top_k": int(top_k),
            "min_replicates": int(max(
                1, self.config.certification_recheck_min_replicates)),
            "soft_margin_scale": float(
                self.config.certification_recheck_soft_margin_scale),
            "n_targets": int(len(self._certification_recheck_targets)),
            "targets": [
                list(map(int, x)) for x in self._certification_recheck_targets
            ],
            "ranking": "absolute_empirical_chance_margin_then_objective",
            "selection_data": "budgeted_initial_observations",
            "target_oracle_used": False,
        }
        return list(self._certification_recheck_targets)

    def _certification_recheck_candidate(self):
        min_replicates = max(
            1, int(self.config.certification_recheck_min_replicates))
        candidates = []
        for x in self._certification_recheck_targets:
            values = self.observations.get(tuple(x), [])
            if not values or len(values) >= min_replicates:
                continue
            y_bar = np.mean(np.asarray(values, dtype=float), axis=0)
            candidates.append((
                int(len(values)),
                float(y_bar[1]),
                float(y_bar[0]),
                tuple(x),
            ))
        if not candidates:
            return None, {"status": "not_due"}
        candidates.sort()
        replicate_count, mean_constraint, mean_objective, x = candidates[0]
        return x, {
            "status": "forced_recheck",
            "replicate_count_before": int(replicate_count),
            "target_replicates": int(min_replicates),
            "empirical_constraint_mean": float(mean_constraint),
            "empirical_objective_mean": float(mean_objective),
            "target_oracle_used": False,
        }

    def _finalist_replication_start_stage(self):
        budget = max(0, int(self.config.finalist_replication_budget))
        return max(int(self.config.n0), int(self.config.N) - budget)

    def _finalist_replication_policy(self):
        policy = str(
            self.config.finalist_replication_policy or "legacy"
        ).strip().lower()
        aliases = {
            "v32": "legacy",
            "commit": "commit_before_switch",
            "terminal_kg": "terminal_kg_1step",
            "terminal-kg-1step": "terminal_kg_1step",
            "terminal-kg-depth3": "terminal_kg_depth3",
        }
        policy = aliases.get(policy, policy)
        valid = {
            "legacy",
            "commit_before_switch",
            "terminal_kg_1step",
            "terminal_kg_depth3",
        }
        if policy not in valid:
            raise ValueError(
                f"unknown finalist replication policy {policy!r}")
        return policy

    def _decision_contract_mode(self):
        mode = str(
            self.config.decision_contract_mode or "legacy"
        ).strip().lower()
        aliases = {
            "off": "legacy",
            "default": "legacy",
            "coherent": "certified_lexicographic",
            "three_layer": "certified_lexicographic",
            "three-layer": "certified_lexicographic",
            "lexicographic": "certified_lexicographic",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"legacy", "certified_lexicographic"}:
            raise ValueError(f"unknown decision contract mode {mode!r}")
        return mode

    def _coherent_certificate_contract(self):
        return self._decision_contract_mode() == "certified_lexicographic"

    def _finalist_frontier_policy(self):
        policy = str(
            self.config.finalist_frontier_policy or "legacy"
        ).strip().lower()
        aliases = {
            "default": "legacy",
            "coverage": "coverage_reserved",
            "reserved": "coverage_reserved",
            "coverage-reserved": "coverage_reserved",
            "observed-safety": "observed_safety_reserved",
            "observed_safety": "observed_safety_reserved",
        }
        policy = aliases.get(policy, policy)
        if policy not in {
            "legacy", "coverage_reserved", "observed_safety_reserved"
        }:
            raise ValueError(f"unknown finalist frontier policy {policy!r}")
        return policy

    def _finalist_empirical_override_policy(self):
        policy = str(
            self.config.finalist_empirical_override or "legacy"
        ).strip().lower()
        aliases = {
            "none": "off",
            "disabled": "off",
            "certificate": "certified_only",
            "certified": "certified_only",
        }
        policy = aliases.get(policy, policy)
        if policy not in {"legacy", "certified_only", "off"}:
            raise ValueError(
                f"unknown finalist empirical override policy {policy!r}")
        return policy

    def _terminal_replication_policy_active(self):
        return self._finalist_replication_policy() in {
            "terminal_kg_1step",
            "terminal_kg_depth3",
        }

    def _finalist_replication_active(self, stage):
        return bool(
            int(self.config.finalist_replication_budget) > 0
            and int(stage) >= self._finalist_replication_start_stage()
        )

    def _finalist_expert_safety_nominations(self, candidates):
        candidates = [tuple(int(v) for v in x) for x in candidates]
        if self.task_ensemble is None or not candidates:
            return []
        latent = (
            self.task_ensemble._task_latent()
            if hasattr(self.task_ensemble, "_task_latent")
            else None
        )
        if (
            bool(getattr(latent, "adaptive_bias_enabled", False))
            and hasattr(
                self.task_ensemble,
                "expert_calibrated_constraint_moments_many",
            )
        ):
            con_mu, con_epistemic, con_aleatoric = (
                self.task_ensemble.expert_calibrated_constraint_moments_many(
                    candidates, certification=True)
            )
        else:
            con_mu, con_epistemic, con_aleatoric = (
                self.task_ensemble.expert_moments_many(
                    1, candidates, certification=True)
            )
        z_alpha = float(norm.ppf(1.0 - self.problem.alpha))
        expert_margin = (
            np.asarray(con_mu, dtype=float)
            + z_alpha * np.sqrt(np.maximum(con_aleatoric, 0.0))
            - float(self.problem.tau)
        )
        expert_violation = self._normal_positive_part(
            expert_margin,
            np.maximum(con_epistemic, 0.0),
        )
        nominations = []
        for expert, name in enumerate(
            self.task_ensemble.posterior.expert_names
        ):
            values = np.asarray(expert_violation[expert], dtype=float)
            if not np.any(np.isfinite(values)):
                continue
            index = int(np.nanargmin(values))
            nominations.append((
                float(values[index]),
                int(expert),
                str(name),
                int(index),
            ))
        nominations.sort()
        return nominations

    def _refresh_finalist_replication_targets(self, stage, pool):
        source_pool = pool
        if (
            bool(self.config.finalist_replication_fixed_universe)
            and self._finalist_replication_pool
        ):
            source_pool = self._finalist_replication_pool
        candidates = [tuple(int(v) for v in x) for x in source_pool]
        nominations = self._finalist_expert_safety_nominations(candidates)
        chosen = None
        for value, expert, name, index in nominations:
            target = candidates[int(index)]
            # Keep a distinct safety challenger when the Bayes action also
            # happens to be an expert's first nomination.
            if (
                len(self._finalist_replication_targets) > 1
                and target == self._finalist_replication_targets[0]
            ):
                continue
            chosen = (value, expert, name, index, target)
            break
        if chosen is None and nominations:
            value, expert, name, index = nominations[0]
            chosen = (
                value, expert, name, index, candidates[int(index)])
        if chosen is None:
            return {
                "status": "adaptive_refresh_no_nomination",
                "frozen_stage": self._finalist_replication_frozen_stage,
                "target_oracle_used": False,
            }

        value, expert, name, index, target = chosen
        label = f"expert_safety_nomination:{name}"
        is_new = target not in self._finalist_replication_targets
        if is_new:
            self._finalist_replication_targets.append(target)
            self._finalist_replication_labels.append(label)
        self._finalist_replication_active_target = target
        self._finalist_replication_active_label = label
        event = {
            "stage": int(stage),
            "history_size_before_observation": int(len(self.history)),
            "label": str(label),
            "expert_index": int(expert),
            "candidate_index": int(index),
            "predicted_positive_violation": float(value),
            "target": list(map(int, target)),
            "new_archive_target": bool(is_new),
            "replicate_count_before": int(len(
                self.observations.get(target, []))),
            "target_oracle_used": False,
        }
        self._finalist_replication_refresh_history.append(event)
        return {
            "status": "adaptive_refreshed",
            "frozen_stage": self._finalist_replication_frozen_stage,
            "active_target": list(map(int, target)),
            "active_label": str(label),
            "archive_target_count": int(len(
                self._finalist_replication_targets)),
            "refresh_event": copy.deepcopy(event),
            "target_oracle_used": False,
        }

    def _initialize_finalist_replication_targets(self, stage, pool):
        policy = self._finalist_replication_policy()
        frontier_policy = self._finalist_frontier_policy()
        if self._finalist_replication_initialized:
            if (
                bool(self.config.finalist_replication_adaptive_race)
                and self._finalist_replication_active(stage)
                and policy not in {
                    "terminal_kg_1step",
                    "terminal_kg_depth3",
                }
            ):
                if (
                    policy == "commit_before_switch"
                    and self._finalist_replication_active_target is not None
                ):
                    active = tuple(self._finalist_replication_active_target)
                    replicate_count = int(len(
                        self.observations.get(active, [])))
                    minimum = max(1, int(
                        self.config.finalist_replication_min_replicates))
                    if replicate_count < minimum:
                        return {
                            "status": "active_target_commit_incomplete",
                            "frozen_stage": (
                                self._finalist_replication_frozen_stage),
                            "active_target": list(map(int, active)),
                            "active_label": (
                                self._finalist_replication_active_label),
                            "replicate_count_before": replicate_count,
                            "minimum_replicates": minimum,
                            "target_oracle_used": False,
                        }
                if (
                    policy == "commit_before_switch"
                    and frontier_policy == "observed_safety_reserved"
                ):
                    minimum = max(1, int(
                        self.config.finalist_replication_min_replicates))
                    for label, target in zip(
                        self._finalist_replication_labels,
                        self._finalist_replication_targets,
                    ):
                        if "observed_safety_rank_" not in str(label):
                            continue
                        target = tuple(target)
                        replicate_count = int(len(
                            self.observations.get(target, [])))
                        if replicate_count >= minimum:
                            continue
                        changed = (
                            target != self._finalist_replication_active_target
                            or str(label)
                            != self._finalist_replication_active_label
                        )
                        self._finalist_replication_active_target = target
                        self._finalist_replication_active_label = str(label)
                        if changed:
                            self._finalist_replication_refresh_history.append({
                                "stage": int(stage),
                                "history_size_before_observation": int(
                                    len(self.history)),
                                "label": str(label),
                                "target": list(map(int, target)),
                                "new_archive_target": False,
                                "replicate_count_before": replicate_count,
                                "reserved_safety_handoff": True,
                                "target_oracle_used": False,
                            })
                        return {
                            "status": "reserved_safety_commit_pending",
                            "frozen_stage": (
                                self._finalist_replication_frozen_stage),
                            "active_target": list(map(int, target)),
                            "active_label": str(label),
                            "replicate_count_before": replicate_count,
                            "minimum_replicates": minimum,
                            "target_oracle_used": False,
                        }
                return self._refresh_finalist_replication_targets(
                    stage, pool)
            return {
                "status": "already_frozen",
                "frozen_stage": self._finalist_replication_frozen_stage,
                "targets": [
                    list(map(int, x))
                    for x in self._finalist_replication_targets
                ],
                "labels": list(self._finalist_replication_labels),
                "frontier_policy": frontier_policy,
                "target_oracle_used": False,
            }
        if not self._finalist_replication_active(stage):
            return {"status": "not_due", "target_oracle_used": False}
        self._finalist_replication_initialized = True
        self._finalist_replication_frozen_stage = int(stage)
        count = max(0, int(self.config.finalist_replication_count))
        safety_count = max(
            0, int(self.config.finalist_observed_safety_count))
        if (
            count > 0
            and frontier_policy == "observed_safety_reserved"
            and safety_count > 0
        ):
            count = max(count, 1 + safety_count)
        if policy in {"terminal_kg_1step", "terminal_kg_depth3"}:
            count = max(
                count,
                max(1, int(self.config.finalist_terminal_max_arms)),
            )
        candidates = [tuple(int(v) for v in x) for x in pool]
        if (
            bool(self.config.finalist_replication_fixed_universe)
            and (
                bool(self.config.finalist_replication_adaptive_race)
                or policy in {"terminal_kg_1step", "terminal_kg_depth3"}
            )
        ):
            self._finalist_replication_pool = list(candidates)
        if count <= 0 or not candidates:
            return {
                "status": "no_targets_requested",
                "frozen_stage": int(stage),
                "target_oracle_used": False,
            }
        components = self._terminal_bayes_risk_components(
            self.gpr,
            self.variance_model,
            candidates,
            task_ensemble=self.task_ensemble,
        )
        certificate = None
        if frontier_policy == "coverage_reserved":
            certificate = self._terminal_certificate_components(
                self.gpr,
                self.variance_model,
                candidates,
                task_ensemble=self.task_ensemble,
                observations=self.observations,
            )
        targets = []
        labels = []
        frozen_metrics = []
        skipped_duplicate_criteria = []

        def add_target(label, index, value, source):
            target = candidates[int(index)]
            if target in targets or len(targets) >= count:
                return False
            targets.append(target)
            labels.append(str(label))
            frozen_metrics.append({
                "label": str(label),
                "index": int(index),
                "value": float(value),
                "source": str(source),
                "replicate_count_at_freeze": int(len(
                    self.observations.get(target, []))),
            })
            return True

        def add_target_or_alias(label, index, value, source):
            target = candidates[int(index)]
            if target not in targets:
                return add_target(label, index, value, source)
            target_index = targets.index(target)
            existing_labels = str(labels[target_index]).split("+")
            if str(label) not in existing_labels:
                labels[target_index] = f"{labels[target_index]}+{label}"
            frozen_metrics.append({
                "label": str(label),
                "index": int(index),
                "value": float(value),
                "source": str(source),
                "alias_of_target_index": int(target_index),
                "replicate_count_at_freeze": int(len(
                    self.observations.get(target, []))),
            })
            return True

        def add_ordered_target(label, values, source):
            values = np.asarray(values, dtype=float)
            if (
                values.shape != (len(candidates),)
                or not np.any(np.isfinite(values))
            ):
                return False
            for index in np.argsort(
                np.where(np.isfinite(values), values, np.inf),
                kind="stable",
            ):
                if add_target(label, index, values[index], source):
                    return True
                skipped_duplicate_criteria.append({
                    "label": str(label),
                    "index": int(index),
                    "reason": "duplicate_or_frontier_full",
                })
                if len(targets) >= count:
                    break
            return False

        risk = np.asarray(components["risk"], dtype=float)
        if frontier_policy == "coverage_reserved":
            reserved = [
                ("minimum_bayes_risk", risk, "mixture_reserved"),
                (
                    "minimum_certificate_margin",
                    certificate["margin"],
                    f"{certificate['source']}_reserved",
                ),
                (
                    "minimum_robust_expected_violation",
                    components["expected_violation"],
                    "mixture_reserved",
                ),
                (
                    "minimum_nominal_expected_violation",
                    components["nominal_expected_violation"],
                    "mixture_reserved",
                ),
            ]
            for label, values, source in reserved:
                if len(targets) >= count:
                    break
                add_ordered_target(label, values, source)
        elif frontier_policy == "observed_safety_reserved":
            if np.any(np.isfinite(risk)):
                add_ordered_target(
                    "minimum_bayes_risk", risk, "mixture_reserved")
            challengers = self._observed_safety_challengers(
                limit=safety_count)
            for rank, challenger in enumerate(challengers, start=1):
                challenger_x = tuple(challenger["x"])
                if challenger_x in candidates:
                    add_target_or_alias(
                        f"observed_safety_rank_{rank}",
                        candidates.index(challenger_x),
                        challenger["empirical_chance_margin"],
                        (
                            "charged_source_coverage_observation_reserved"
                            if challenger.get("selection_scope")
                            == "frozen_source_coverage"
                            else "charged_observation_reserved"
                        ),
                    )
        elif np.any(np.isfinite(risk)):
            index = int(np.nanargmin(risk))
            add_target(
                "minimum_bayes_risk", index, risk[index], "mixture")

        if (
            bool(self.config.finalist_replication_expert_stratified)
            and self.task_ensemble is not None
            and len(targets) < count
        ):
            nominations = self._finalist_expert_safety_nominations(
                candidates)
            for value, _expert, name, index in nominations:
                add_target(
                    f"expert_safety_nomination:{name}",
                    index,
                    value,
                    "expert_stratified",
                )
                if len(targets) >= count:
                    break

        criteria = []
        if frontier_policy == "legacy":
            criteria.extend([
                (
                    "minimum_nominal_expected_violation",
                    components["nominal_expected_violation"],
                ),
                (
                    "minimum_robust_expected_violation",
                    components["expected_violation"],
                ),
            ])
        criteria.append((
            "maximum_model_disagreement",
            -np.asarray(components["model_disagreement"], dtype=float),
        ))
        for label, values in criteria:
            if len(targets) >= count:
                break
            add_ordered_target(label, values, "mixture_fallback")

        if frontier_policy == "coverage_reserved" and len(targets) < count:
            signature = np.column_stack([
                np.asarray(components["objective"], dtype=float),
                risk,
                np.asarray(certificate["margin"], dtype=float),
                np.asarray(components["expected_violation"], dtype=float),
                np.asarray(
                    components["nominal_expected_violation"], dtype=float),
                np.asarray(components["model_disagreement"], dtype=float),
            ])
            finite = np.all(np.isfinite(signature), axis=1)
            if np.any(finite):
                center = np.nanmedian(signature[finite], axis=0)
                scale = np.nanquantile(
                    signature[finite], 0.75, axis=0
                ) - np.nanquantile(signature[finite], 0.25, axis=0)
                scale = np.where(scale > 1e-12, scale, 1.0)
                normalized = (signature - center) / scale
                selected_indices = [
                    index
                    for index in (
                        candidates.index(target) for target in targets
                    )
                    if finite[index]
                ]
                while len(targets) < count:
                    available = finite.copy()
                    if selected_indices:
                        available[selected_indices] = False
                    if not np.any(available):
                        break
                    if selected_indices:
                        distances = np.min(
                            np.linalg.norm(
                                normalized[:, None, :]
                                - normalized[selected_indices][None, :, :],
                                axis=2,
                            ),
                            axis=1,
                        )
                    else:
                        distances = np.linalg.norm(normalized, axis=1)
                    index = int(np.argmax(np.where(
                        available, distances, -np.inf)))
                    if not add_target(
                        "posterior_signature_diversity",
                        index,
                        -float(distances[index]),
                        "coverage_diversity",
                    ):
                        break
                    selected_indices.append(index)

        if len(targets) < count:
            while len(targets) < count and add_ordered_target(
                "bayes_risk_frontier_fill", risk, "mixture_fill"
            ):
                pass
        self._finalist_replication_targets = targets
        self._finalist_replication_labels = labels
        active_index = next((
            index for index, label in enumerate(labels)
            if "observed_safety_rank_" in str(label)
        ), None)
        if active_index is None:
            active_index = next((
                index for index, label in enumerate(labels)
                if str(label).startswith("expert_safety_nomination:")
            ), None)
        if active_index is None and targets:
            active_index = len(targets) - 1
        if active_index is not None:
            self._finalist_replication_active_target = targets[active_index]
            self._finalist_replication_active_label = labels[active_index]
        if bool(self.config.finalist_replication_adaptive_race) and targets:
            self._finalist_replication_refresh_history.append({
                "stage": int(stage),
                "history_size_before_observation": int(len(self.history)),
                "label": str(self._finalist_replication_active_label),
                "target": list(map(
                    int, self._finalist_replication_active_target)),
                "new_archive_target": True,
                "replicate_count_before": int(len(self.observations.get(
                    self._finalist_replication_active_target, []))),
                "initial_refresh": True,
                "target_oracle_used": False,
            })
        return {
            "status": "frozen" if targets else "no_finite_target",
            "frozen_stage": int(stage),
            "selection_data": "charged_posterior_before_new_label",
            "frontier_policy": frontier_policy,
            "decision_contract_mode": self._decision_contract_mode(),
            "certificate_source": (
                None if certificate is None else certificate["source"]),
            "reserved_coverage_labels": [
                label for label in labels
                if (
                    label in {
                        "minimum_bayes_risk",
                        "minimum_certificate_margin",
                        "minimum_robust_expected_violation",
                        "minimum_nominal_expected_violation",
                    }
                    or "observed_safety_rank_" in str(label)
                )
            ],
            "skipped_duplicate_criteria": skipped_duplicate_criteria,
            "expert_stratified": bool(
                self.config.finalist_replication_expert_stratified),
            "adaptive_race": bool(
                self.config.finalist_replication_adaptive_race),
            "fixed_universe": bool(
                self.config.finalist_replication_fixed_universe),
            "fixed_universe_size": int(len(
                self._finalist_replication_pool)),
            "active_target": (
                None
                if self._finalist_replication_active_target is None
                else list(map(
                    int, self._finalist_replication_active_target))
            ),
            "active_label": self._finalist_replication_active_label,
            "targets": [list(map(int, x)) for x in targets],
            "labels": list(labels),
            "frozen_metrics": frozen_metrics,
            "target_oracle_used": False,
        }

    def _finalist_replication_candidate(self, stage, pool):
        initialization = self._initialize_finalist_replication_targets(
            stage, pool)
        if not self._finalist_replication_active(stage):
            return None, initialization
        policy = self._finalist_replication_policy()
        if policy in {"terminal_kg_1step", "terminal_kg_depth3"}:
            max_arms = max(1, int(self.config.finalist_terminal_max_arms))
            arms = list(self._finalist_replication_targets[:max_arms])
            if not arms:
                return None, {
                    **initialization,
                    "status": "terminal_kg_no_arms",
                    "policy": policy,
                    "target_oracle_used": False,
                }
            remaining = max(1, int(self.config.N) - int(stage))
            depth = 1 if policy == "terminal_kg_1step" else min(3, remaining)
            selected, terminal_info = self._terminal_replication_kg_candidate(
                arms,
                pool,
                depth=depth,
                stage=stage,
            )
            return selected, {
                **initialization,
                **terminal_info,
                "status": "forced_terminal_replication_kg",
                "policy": policy,
                "target_oracle_used": False,
            }
        minimum = max(
            1, int(self.config.finalist_replication_min_replicates))
        if (
            bool(self.config.finalist_replication_adaptive_race)
            and self._finalist_replication_active_target is not None
        ):
            active = tuple(self._finalist_replication_active_target)
            replicate_count = int(len(self.observations.get(active, [])))
            if replicate_count < minimum:
                return active, {
                    **initialization,
                    "status": "forced_adaptive_finalist_replication",
                    "label": str(self._finalist_replication_active_label),
                    "replicate_count_before": int(replicate_count),
                    "minimum_replicates": int(minimum),
                    "reserved_budget": int(max(
                        0, self.config.finalist_replication_budget)),
                    "expert_stratified": bool(
                        self.config.finalist_replication_expert_stratified),
                    "adaptive_race": True,
                    "archive_target_count": int(len(
                        self._finalist_replication_targets)),
                    "target_oracle_used": False,
                }
        pending = []
        for order, (label, x) in enumerate(zip(
            self._finalist_replication_labels,
            self._finalist_replication_targets,
        )):
            replicate_count = int(len(self.observations.get(tuple(x), [])))
            if replicate_count >= minimum:
                continue
            pending.append((replicate_count, order, str(label), tuple(x)))
        if not pending:
            return None, {
                **initialization,
                "status": "targets_sufficiently_replicated",
                "minimum_replicates": int(minimum),
                "replicate_counts": [
                    int(len(self.observations.get(tuple(x), [])))
                    for x in self._finalist_replication_targets
                ],
                "target_oracle_used": False,
            }
        pending.sort()
        replicate_count, _, label, x = pending[0]
        return x, {
            **initialization,
            "status": "forced_finalist_replication",
            "label": str(label),
            "replicate_count_before": int(replicate_count),
            "minimum_replicates": int(minimum),
            "reserved_budget": int(max(
                0, self.config.finalist_replication_budget)),
            "expert_stratified": bool(
                self.config.finalist_replication_expert_stratified),
            "adaptive_race": bool(
                self.config.finalist_replication_adaptive_race),
            "fixed_universe": bool(
                self.config.finalist_replication_fixed_universe),
            "target_oracle_used": False,
        }

    def _replicated_finalist_statistics(self, x):
        key = tuple(int(v) for v in x)
        values = np.asarray(self.observations.get(key, []), dtype=float)
        if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2:
            return None
        replicate_count = int(len(values))
        if self.task_ensemble is None:
            prior_variance = float(
                self.variance_model.predict_certification_variance(
                    1, key, self.problem))
        else:
            robust = self.task_ensemble.robust_moments_many(
                1, [key], certification=True)
            prior_variance = float(robust.aleatoric_upper[0])
        prior_variance = max(prior_variance, 1e-12)
        sample_variance = (
            float(np.var(values[:, 1], ddof=1))
            if replicate_count >= 2
            else prior_variance
        )
        prior_df = max(
            float(self.config.finalist_replication_variance_prior_df), 0.0)
        denominator = max(replicate_count - 1 + prior_df, 1.0)
        shrunk_variance = (
            (replicate_count - 1) * max(sample_variance, 0.0)
            + prior_df * prior_variance
        ) / denominator
        shrunk_variance = max(float(shrunk_variance), 1e-12)
        nominal_delta = float(np.clip(
            self.config.finalist_replication_delta, 1e-12, 0.5))
        familywise_multiplicity = 1
        if bool(self.config.finalist_replication_adaptive_race):
            configured = (
                int(self.config.finalist_replication_count)
                + int(self.config.finalist_replication_budget)
            )
            if self._finalist_replication_policy() == "legacy":
                familywise_multiplicity = max(1, configured)
            else:
                familywise_multiplicity = max(
                    1,
                    configured,
                    len(self._finalist_replication_targets)
                    + int(self.config.finalist_replication_budget),
                )
        delta = float(np.clip(
            nominal_delta / familywise_multiplicity, 1e-12, 0.5))
        z_delta = float(norm.ppf(1.0 - delta))
        z_alpha = float(norm.ppf(1.0 - self.problem.alpha))
        sigma = float(np.sqrt(shrunk_variance))
        mean_radius = float(z_delta * sigma / np.sqrt(replicate_count))
        objective_mean = float(np.mean(values[:, 0]))
        constraint_mean = float(np.mean(values[:, 1]))
        upper_margin = float(
            constraint_mean
            + z_alpha * sigma
            + mean_radius
            - float(self.problem.tau)
        )
        return {
            "x": list(map(int, key)),
            "replicate_count": int(replicate_count),
            "objective_mean": objective_mean,
            "constraint_mean": constraint_mean,
            "prior_variance": float(prior_variance),
            "sample_variance": float(sample_variance),
            "shrunk_variance": float(shrunk_variance),
            "mean_confidence_radius": mean_radius,
            "upper_chance_margin": upper_margin,
            "delta": delta,
            "nominal_delta": nominal_delta,
            "familywise_multiplicity": int(familywise_multiplicity),
            "target_oracle_used": False,
        }

    def _replicated_finalist_recommendation_index(self, pool):
        override_policy = self._finalist_empirical_override_policy()
        if override_policy == "off":
            return None, {
                "replicated_finalist_used": False,
                "replicated_finalist_reason": "empirical_override_disabled",
                "replicated_finalist_override_policy": override_policy,
            }
        if (
            not self._finalist_replication_initialized
            or not self._finalist_replication_targets
        ):
            return None, {
                "replicated_finalist_used": False,
                "replicated_finalist_override_policy": override_policy,
            }
        minimum = max(
            1, int(self.config.finalist_replication_min_replicates))
        rows = []
        incomplete = []
        for order, x in enumerate(self._finalist_replication_targets):
            stats = self._replicated_finalist_statistics(x)
            if stats is None or int(stats["replicate_count"]) < minimum:
                if bool(self.config.finalist_replication_adaptive_race):
                    incomplete.append({
                        "order": int(order),
                        "x": list(map(int, x)),
                        "replicate_count": (
                            0 if stats is None
                            else int(stats["replicate_count"])
                        ),
                    })
                    continue
                return None, {
                    "replicated_finalist_used": False,
                    "replicated_finalist_reason": "incomplete_replication",
                }
            rows.append((order, tuple(x), stats))
        if not rows:
            return None, {
                "replicated_finalist_used": False,
                "replicated_finalist_reason": "no_completed_finalist",
                "replicated_finalist_incomplete_rows": incomplete,
            }
        feasible = [row for row in rows if row[2]["upper_chance_margin"] <= 0.0]
        if feasible:
            chosen = min(feasible, key=lambda row: (
                row[2]["objective_mean"],
                row[2]["upper_chance_margin"],
                row[0],
            ))
            reason = "replicated_upper_bound_feasible"
        elif override_policy == "certified_only":
            return None, {
                "replicated_finalist_used": False,
                "replicated_finalist_reason": (
                    "no_empirically_certified_finalist"),
                "replicated_finalist_override_policy": override_policy,
                "replicated_finalist_rows": [
                    copy.deepcopy(row[2]) for row in rows
                ],
                "replicated_finalist_incomplete_rows": incomplete,
                "replicated_finalist_adaptive_race": bool(
                    self.config.finalist_replication_adaptive_race),
                "replicated_finalist_target_oracle_used": False,
            }
        else:
            chosen = min(rows, key=lambda row: (
                row[2]["upper_chance_margin"],
                row[2]["objective_mean"],
                row[0],
            ))
            reason = "minimum_replicated_upper_margin"
        pool_lookup = {
            tuple(int(v) for v in x): index for index, x in enumerate(pool)
        }
        if chosen[1] not in pool_lookup:
            return None, {
                "replicated_finalist_used": False,
                "replicated_finalist_reason": "target_missing_from_pool",
            }
        return int(pool_lookup[chosen[1]]), {
            "replicated_finalist_used": True,
            "replicated_finalist_reason": str(reason),
            "replicated_finalist_empirical_certificate": bool(
                chosen[2]["upper_chance_margin"] <= 0.0),
            "replicated_finalist_selected": copy.deepcopy(chosen[2]),
            "replicated_finalist_rows": [
                copy.deepcopy(row[2]) for row in rows
            ],
            "replicated_finalist_incomplete_rows": incomplete,
            "replicated_finalist_adaptive_race": bool(
                self.config.finalist_replication_adaptive_race),
            "replicated_finalist_override_policy": override_policy,
            "replicated_finalist_target_oracle_used": False,
        }

    def _two_stage_decision_contract_summary(
            self, recommendation, finalist_summary):
        """Audit the two-stage theory contract without changing decisions."""
        total_budget = max(0, int(self.config.N))
        verification_budget = min(
            total_budget,
            max(0, int(self.config.finalist_replication_budget)),
        )
        search_budget = total_budget - verification_budget
        initial_design_budget = min(
            search_budget, max(0, int(self.config.n0)))
        adaptive_search_budget = search_budget - initial_design_budget
        posterior_certified = bool(
            recommendation.get("posterior_feasible", False))
        replicated_used = bool(
            recommendation.get("replicated_finalist_used", False))
        replicated_certified = bool(
            recommendation.get(
                "replicated_finalist_empirical_certificate", False))
        replicated_reason = str(
            recommendation.get("replicated_finalist_reason", ""))

        if posterior_certified:
            terminal_status = "posterior_certified"
            certificate_scope = "gp_hvd_simultaneous_coverage_event"
        elif replicated_used and replicated_certified:
            terminal_status = "replicated_event_certified"
            certificate_scope = "replicated_mean_and_variance_joint_event"
        elif replicated_used and replicated_reason == (
                "minimum_replicated_upper_margin"):
            terminal_status = "uncertified_least_risk_fallback"
            certificate_scope = "none"
        else:
            terminal_status = "unaccounted_legacy_fallback"
            certificate_scope = "none"

        frozen_stage = finalist_summary.get("frozen_stage")
        freeze_precedes_labels = bool(
            finalist_summary.get("initialized", False)
            and frozen_stage is not None
            and int(frozen_stage) == search_budget)
        frozen_pool = {
            tuple(int(v) for v in x)
            for x in self._finalist_replication_pool
        }
        verification_targets = {
            tuple(int(v) for v in x)
            for x in self._finalist_replication_targets
        }
        targets_in_frozen_universe = bool(
            verification_targets.issubset(frozen_pool))
        forced_evaluations = int(
            finalist_summary.get("forced_evaluations", 0))
        charged_inside_budget = bool(
            forced_evaluations <= verification_budget)
        verification_budget_fully_charged = bool(
            forced_evaluations == verification_budget)
        terminal_rule_accounted = bool(
            posterior_certified or replicated_used)
        implementation_contract_closed = bool(
            verification_budget > 0
            and search_budget + verification_budget == total_budget
            and finalist_summary.get("fixed_universe", False)
            and freeze_precedes_labels
            and bool(verification_targets)
            and targets_in_frozen_universe
            and charged_inside_budget
            and verification_budget_fully_charged
            and terminal_rule_accounted
            and not finalist_summary.get("target_oracle_used", False)
        )
        return {
            "architecture": "two_stage_search_then_verification",
            "search_policy": (
                "source_informed_initial_design_then_state_coupled_exact_kg"),
            "verification_policy": (
                "heteroscedastic_fixed_universe_ranking_and_selection"),
            "total_budget": total_budget,
            "search_budget": search_budget,
            "initial_design_budget": initial_design_budget,
            "adaptive_search_budget": adaptive_search_budget,
            "verification_budget": verification_budget,
            "budget_partition_valid": bool(
                search_budget + verification_budget == total_budget),
            "freeze_precedes_verification_labels": freeze_precedes_labels,
            "verification_targets_in_frozen_universe": (
                targets_in_frozen_universe),
            "verification_calls_inside_total_budget": charged_inside_budget,
            "verification_budget_fully_charged": (
                verification_budget_fully_charged),
            "terminal_status": terminal_status,
            "certificate_scope": certificate_scope,
            "fallback_claims_certification": False,
            "terminal_rule_accounted": terminal_rule_accounted,
            "implementation_contract_closed": implementation_contract_closed,
            "global_exact_kg_claim": False,
            "adaptive_search_acquisition_configured_as_exact_kg": bool(
                str(self.config.acquisition_mode).lower() == "exact_mc"),
            "legacy_single_value_contract_closed": bool(
                finalist_summary.get("mathematically_closed", False)),
            "regret_terms": [
                "search_error",
                "proposal_coverage_error",
                "verification_error",
            ],
        }

    def _surrogate_feature_matrix(
        self,
        basis,
        xs,
        *,
        feature_mean=None,
        feature_scale=None,
        force_standardize=False,
    ):
        if hasattr(basis, "features_many"):
            raw = np.asarray(basis.features_many(xs), dtype=float)
        else:
            raw = np.vstack([
                np.asarray(basis.features(x), dtype=float).reshape(-1)
                for x in xs
            ])
        if feature_mean is None or feature_scale is None:
            if self.config.calibration_standardize_features or force_standardize:
                feature_mean = np.mean(raw, axis=0)
                feature_scale = np.std(raw, axis=0)
                feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
            else:
                feature_mean = np.zeros(raw.shape[1], dtype=float)
                feature_scale = np.ones(raw.shape[1], dtype=float)
        scaled = (raw - feature_mean) / feature_scale
        Phi = np.column_stack([np.ones(len(scaled), dtype=float), scaled])
        return Phi, np.asarray(feature_mean, dtype=float), np.asarray(feature_scale, dtype=float)

    @staticmethod
    def _calibration_standardize(raw_train, raw_test=None):
        raw_train = np.asarray(raw_train, dtype=float)
        feature_mean = np.mean(raw_train, axis=0)
        feature_scale = np.std(raw_train, axis=0)
        feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
        train_scaled = (raw_train - feature_mean) / feature_scale
        Phi_train = np.column_stack([
            np.ones(len(train_scaled), dtype=float),
            train_scaled,
        ])
        Phi_test = None
        if raw_test is not None:
            raw_test = np.asarray(raw_test, dtype=float)
            test_scaled = (raw_test - feature_mean) / feature_scale
            Phi_test = np.column_stack([
                np.ones(len(test_scaled), dtype=float),
                test_scaled,
            ])
        return Phi_train, Phi_test, feature_mean, feature_scale

    @staticmethod
    def _calibration_ridge_fit(Phi, target, ridge):
        penalty = float(ridge) * np.eye(Phi.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = Phi.T @ Phi + penalty
        lhs_inv = np.linalg.pinv(lhs)
        beta = lhs_inv @ (Phi.T @ np.asarray(target, dtype=float))
        hat_diag = np.sum((Phi @ lhs_inv) * Phi, axis=1)
        effective_rank = float(np.sum(np.clip(hat_diag, 0.0, 1.0)))
        return beta, lhs_inv, effective_rank

    def _nested_calibration_ridge_candidate(
        self,
        raw,
        y_obj,
        y_con,
        ridge,
        max_effective_fraction,
    ):
        """Refit preprocessing and ridge inside every held-out fold."""
        raw = np.asarray(raw, dtype=float)
        y_obj = np.asarray(y_obj, dtype=float)
        y_con = np.asarray(y_con, dtype=float)
        n = len(raw)
        pred_obj = np.empty(n, dtype=float)
        pred_con = np.empty(n, dtype=float)
        fold_ranks = []
        admissible = True
        for heldout in range(n):
            train = np.asarray([i for i in range(n) if i != heldout], dtype=int)
            Phi_train, Phi_test, _, _ = self._calibration_standardize(
                raw[train], raw[[heldout]])
            beta_obj, _, rank_obj = self._calibration_ridge_fit(
                Phi_train, y_obj[train], ridge)
            beta_con, _, rank_con = self._calibration_ridge_fit(
                Phi_train, y_con[train], ridge)
            fold_rank = max(rank_obj, rank_con)
            fold_ranks.append(float(fold_rank))
            rank_cap = max(
                2,
                int(np.floor(max_effective_fraction * len(train))),
            )
            admissible &= bool(fold_rank <= rank_cap + 1e-8)
            pred_obj[heldout] = float((Phi_test @ beta_obj)[0])
            pred_con[heldout] = float((Phi_test @ beta_con)[0])

        Phi, _, feature_mean, feature_scale = self._calibration_standardize(raw)
        beta_obj, lhs_inv_obj, rank_obj = self._calibration_ridge_fit(
            Phi, y_obj, ridge)
        beta_con, lhs_inv_con, rank_con = self._calibration_ridge_fit(
            Phi, y_con, ridge)
        effective_rank = max(rank_obj, rank_con)
        full_rank_cap = max(
            2,
            int(np.floor(max_effective_fraction * n)),
        )
        admissible &= bool(effective_rank <= full_rank_cap + 1e-8)

        con_scale = max(float(np.var(y_con)), 1e-8)
        obj_scale = max(float(np.var(y_obj)), 1e-8)
        con_mse = float(np.mean((y_con - pred_con) ** 2)) / con_scale
        obj_mse = float(np.mean((y_obj - pred_obj) ** 2)) / obj_scale
        dangerous = float(np.mean(np.maximum(y_con - pred_con, 0.0) ** 2))
        dangerous /= con_scale
        upper = np.triu_indices(n, k=1)
        true_diff = (y_con[:, None] - y_con[None, :])[upper]
        pred_diff = (pred_con[:, None] - pred_con[None, :])[upper]
        informative = np.abs(true_diff) > 1e-10
        rank_loss = (
            float(np.mean(true_diff[informative] * pred_diff[informative] < 0.0))
            if np.any(informative)
            else 0.0
        )
        score = con_mse + 0.10 * obj_mse + 0.50 * dangerous + 0.25 * rank_loss
        return {
            "ridge": float(ridge),
            "score": float(score),
            "con_mse": float(con_mse),
            "obj_mse": float(obj_mse),
            "dangerous_underprediction": float(dangerous),
            "rank_loss": float(rank_loss),
            "effective_rank": float(effective_rank),
            "fold_max_effective_rank": float(max(fold_ranks, default=0.0)),
            "rank_cap": int(full_rank_cap),
            "admissible": bool(admissible),
            "Phi": Phi,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "beta_obj": beta_obj,
            "beta_con": beta_con,
            "lhs_inv": lhs_inv_con,
            "nested_pred_obj": pred_obj,
            "nested_pred_con": pred_con,
        }

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
        strategy = str(
            self.config.recommendation_infeasible_strategy).lower()
        task_adaptive = bool(
            self.task_ensemble is not None
            and self.task_ensemble.sensitivity_posterior is not None
            and strategy
            in ("task_adaptive", "task-adaptive", "sensitivity_posterior")
        )
        y_obj = np.asarray(train_obj, dtype=float)
        y_con = np.asarray(train_con, dtype=float)
        base_ridge = max(
            float(self.config.recommendation_calibration_ridge), 0.0)
        if task_adaptive:
            if hasattr(basis, "features_many"):
                raw = np.asarray(basis.features_many(train_x), dtype=float)
            else:
                raw = np.vstack([
                    np.asarray(basis.features(x), dtype=float).reshape(-1)
                    for x in train_x
                ])
            ridge_grid = sorted(set([
                1e-4, 1e-2, 1.0, 10.0, 100.0, 1e3, 1e4, base_ridge,
            ]))
            max_effective_fraction = float(np.clip(
                self.config.recommendation_calibration_max_effective_fraction,
                0.05,
                0.95,
            ))
            ridge_candidates = [
                self._nested_calibration_ridge_candidate(
                    raw,
                    y_obj,
                    y_con,
                    candidate_ridge,
                    max_effective_fraction,
                )
                for candidate_ridge in ridge_grid
            ]
            admissible = [
                item for item in ridge_candidates if item["admissible"]
            ]
            selection_pool = admissible or ridge_candidates
            chosen_fit = min(
                selection_pool,
                key=lambda item: (
                    item["score"] if item["admissible"] else float("inf"),
                    item["effective_rank"],
                    -item["ridge"],
                ),
            )
        else:
            # Preserve the original calibration path for non-task-adaptive
            # baselines.  The nested/rank-constrained fit is an explicit
            # challenger, not a silent change to every historical result.
            Phi, feature_mean, feature_scale = self._surrogate_feature_matrix(
                basis, train_x)
            penalty = base_ridge * np.eye(Phi.shape[1], dtype=float)
            penalty[0, 0] = 0.0
            lhs = Phi.T @ Phi + penalty
            try:
                beta_obj = np.linalg.solve(lhs, Phi.T @ y_obj)
                beta_con = np.linalg.solve(lhs, Phi.T @ y_con)
            except np.linalg.LinAlgError:
                beta_obj = np.linalg.lstsq(lhs, Phi.T @ y_obj, rcond=None)[0]
                beta_con = np.linalg.lstsq(lhs, Phi.T @ y_con, rcond=None)[0]
            try:
                lhs_inv = np.linalg.pinv(lhs)
            except np.linalg.LinAlgError:
                lhs_inv = None
            effective_rank = (
                float(np.sum(np.clip(
                    np.sum((Phi @ lhs_inv) * Phi, axis=1), 0.0, 1.0)))
                if lhs_inv is not None
                else float(Phi.shape[1])
            )
            chosen_fit = {
                "ridge": float(base_ridge),
                "score": float("nan"),
                "con_mse": float("nan"),
                "obj_mse": float("nan"),
                "dangerous_underprediction": float("nan"),
                "rank_loss": float("nan"),
                "effective_rank": float(effective_rank),
                "fold_max_effective_rank": float("nan"),
                "rank_cap": None,
                "admissible": None,
                "Phi": Phi,
                "feature_mean": feature_mean,
                "feature_scale": feature_scale,
                "beta_obj": beta_obj,
                "beta_con": beta_con,
                "lhs_inv": lhs_inv,
            }
            ridge_candidates = [chosen_fit]
        ridge = float(chosen_fit["ridge"])
        Phi = chosen_fit["Phi"]
        feature_mean = chosen_fit["feature_mean"]
        feature_scale = chosen_fit["feature_scale"]
        lhs_inv = chosen_fit["lhs_inv"]
        beta_obj = chosen_fit["beta_obj"]
        beta_con = chosen_fit["beta_con"]
        resid_con = y_con - Phi @ beta_con
        resid_sigma = float(np.sqrt(np.mean(resid_con ** 2))) if len(resid_con) else 0.0
        if task_adaptive:
            loo_residual = y_con - np.asarray(
                chosen_fit["nested_pred_con"], dtype=float)
            loo_sigma = float(np.sqrt(np.mean(loo_residual ** 2)))
            ordered_abs = np.sort(np.abs(loo_residual))
            conformal_index = min(
                len(ordered_abs) - 1,
                max(
                    0,
                    int(np.ceil(
                        (1.0 - float(self.problem.alpha))
                        * (len(ordered_abs) + 1)
                    )) - 1,
                ),
            )
            conformal_radius = float(ordered_abs[conformal_index])
            z_alpha = max(float(norm.ppf(1 - self.problem.alpha)), 1e-8)
            conformal_sigma = conformal_radius / z_alpha
        elif lhs_inv is None or not len(resid_con):
            loo_sigma = float(resid_sigma)
            conformal_sigma = float(resid_sigma)
        else:
            hat_diag = np.sum((Phi @ lhs_inv) * Phi, axis=1)
            loo_denom = np.maximum(1.0 - np.clip(hat_diag, 0.0, 0.95), 0.05)
            loo_residual = resid_con / loo_denom
            loo_sigma = float(np.sqrt(np.mean(loo_residual ** 2)))
            ordered_abs = np.sort(np.abs(loo_residual))
            conformal_index = min(
                len(ordered_abs) - 1,
                max(
                    0,
                    int(np.ceil(
                        (1.0 - float(self.problem.alpha))
                        * (len(ordered_abs) + 1)
                    )) - 1,
                ),
            )
            conformal_radius = float(ordered_abs[conformal_index])
            z_alpha = max(float(norm.ppf(1 - self.problem.alpha)), 1e-8)
            conformal_sigma = conformal_radius / z_alpha
        nominal_floor = (
            float(self.config.recommendation_noise_floor_scale)
            * 0.35
            * float(getattr(self.problem, "sigma_level", 0.0))
        )
        sigma_cal = max(resid_sigma, nominal_floor, 1e-8)
        prequential_sigma = max(
            sigma_cal,
            loo_sigma,
            conformal_sigma,
        )
        return {
            "basis": basis,
            "Phi_train": Phi,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "beta_obj": beta_obj,
            "beta_con": beta_con,
            "lhs_inv": lhs_inv,
            "sigma": float(sigma_cal),
            "prequential_sigma": float(prequential_sigma),
            "loo_sigma": float(loo_sigma),
            "conformal_sigma": float(conformal_sigma),
            "resid_sigma": float(resid_sigma),
            "n_train": int(len(train_x)),
            "feature_dim": int(Phi.shape[1]),
            "selected_ridge": float(ridge),
            "effective_rank": float(chosen_fit["effective_rank"]),
            "effective_rank_cap": (
                None
                if chosen_fit["rank_cap"] is None
                else int(chosen_fit["rank_cap"])
            ),
            "rank_cap_satisfied": (
                None
                if chosen_fit["admissible"] is None
                else bool(chosen_fit["admissible"])
            ),
            "nested_refit": bool(task_adaptive),
            "ridge_scores": [
                {
                    "ridge": float(item["ridge"]),
                    "score": float(item["score"]),
                    "effective_rank": float(item["effective_rank"]),
                    "fold_max_effective_rank": float(
                        item["fold_max_effective_rank"]),
                    "rank_cap": (
                        None
                        if item["rank_cap"] is None
                        else int(item["rank_cap"])
                    ),
                    "admissible": (
                        None
                        if item["admissible"] is None
                        else bool(item["admissible"])
                    ),
                    "con_mse": float(item["con_mse"]),
                    "dangerous_underprediction": float(
                        item["dangerous_underprediction"]),
                    "rank_loss": float(item["rank_loss"]),
                }
                for item in ridge_candidates
            ],
            "features_standardized": bool(
                self.config.calibration_standardize_features or task_adaptive),
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
            "selected_ridge": float(fit.get(
                "selected_ridge",
                self.config.recommendation_calibration_ridge,
            )),
            "effective_rank": float(fit.get(
                "effective_rank", fit["feature_dim"])),
            "effective_rank_cap": fit.get("effective_rank_cap"),
            "rank_cap_satisfied": fit.get("rank_cap_satisfied"),
            "nested_refit": bool(fit.get("nested_refit", False)),
            "ridge_scores": copy.deepcopy(fit.get("ridge_scores", [])),
            "features_standardized": bool(fit.get(
                "features_standardized",
                self.config.calibration_standardize_features,
            )),
            "n_feasible": int(np.sum(margin <= 0.0)),
        }

    def _task_adaptive_recommendation_index(
        self,
        pool,
        robust_margins,
        *,
        guard_margins=None,
        model_objective=None,
        model_aleatoric=None,
        fit=None,
        phi_pool=None,
    ):
        """Minimize posterior expected violation loss on an empirical boundary.

        This fallback is used only when the theory certificate has no feasible
        point.  The latent task class changes calibration-error scale and the
        cost of a violation, but never changes the certificate itself.
        """
        sensitivity = (
            None
            if self.task_ensemble is None
            else self.task_ensemble.sensitivity_posterior
        )
        fit = fit or self._recommendation_calibration_fit()
        if sensitivity is None or fit is None or not pool:
            return None, {
                "calibrated_recommendation_reason": (
                    "task_adaptive_empirical_model_unavailable")
            }
        Phi_pool = (
            np.asarray(phi_pool, dtype=float)
            if phi_pool is not None and len(phi_pool) == len(pool)
            else self._recommendation_calibration_pool_features(fit, pool)
        )
        pred_obj = np.asarray(Phi_pool @ fit["beta_obj"], dtype=float)
        pred_con = np.asarray(Phi_pool @ fit["beta_con"], dtype=float)
        if fit["lhs_inv"] is None:
            return None, {
                "calibrated_recommendation_reason": (
                    "task_adaptive_leverage_unavailable")
            }
        leverage = np.sum((Phi_pool @ fit["lhs_inv"]) * Phi_pool, axis=1)
        leverage = np.maximum(np.asarray(leverage, dtype=float), 0.0)
        theory_margin = np.asarray(
            robust_margins if guard_margins is None else guard_margins,
            dtype=float,
        )
        guard = np.isfinite(leverage) & np.isfinite(theory_margin)
        max_leverage = float(
            self.config.recommendation_calibration_max_leverage)
        max_theory_margin = float(
            self.config.recommendation_calibration_max_theory_margin)
        if max_leverage > 0.0:
            guard &= leverage <= max_leverage
        if max_theory_margin > 0.0:
            guard &= theory_margin <= max_theory_margin
        if not np.any(guard):
            return None, {
                "calibrated_recommendation_reason": (
                    "no_task_adaptive_candidate_after_guard"),
                "n_calibration_candidates": int(len(pool)),
                "n_calibration_guarded": 0,
            }

        risk = sensitivity.posterior_violation_decision_risk(
            pred_con,
            fit.get("prequential_sigma", fit["sigma"]),
            leverage,
            tau=self.problem.tau,
            aleatoric_variance=model_aleatoric,
        )
        violation_probability = np.asarray(
            risk["posterior_violation_probability"], dtype=float)
        violation_loss = np.asarray(
            risk["posterior_expected_decision_risk"], dtype=float)
        empirical_objective_loss = pred_obj - float(np.min(pred_obj[guard]))
        objective_span = float(np.max(empirical_objective_loss[guard]))
        if objective_span > 1e-12:
            empirical_objective_loss /= objective_span
        else:
            empirical_objective_loss[:] = 0.0
        robust_objective = np.asarray(
            pred_obj if model_objective is None else model_objective,
            dtype=float,
        )
        robust_objective_loss = robust_objective - float(
            np.min(robust_objective[guard]))
        robust_objective_span = float(np.max(robust_objective_loss[guard]))
        if robust_objective_span > 1e-12:
            robust_objective_loss /= robust_objective_span
        else:
            robust_objective_loss[:] = 0.0
        robust_margin_loss = np.maximum(
            np.asarray(robust_margins, dtype=float), 0.0)
        robust_margin_span = float(np.max(robust_margin_loss[guard]))
        if robust_margin_span > 1e-12:
            robust_margin_loss /= robust_margin_span
        else:
            robust_margin_loss[:] = 0.0

        class_weights = sensitivity.posterior_weights()
        class_penalties = sensitivity.decision_penalties
        class_trust = sensitivity.empirical_trust
        class_violation_probability = np.asarray(
            risk["class_violation_probability"], dtype=float)
        robust_class_loss = (
            robust_objective_loss[:, None]
            + robust_margin_loss[:, None] * class_penalties[None, :]
        )
        empirical_class_loss = (
            empirical_objective_loss[:, None]
            + class_violation_probability * class_penalties[None, :]
        )
        robust_component = np.sum(
            robust_class_loss
            * (class_weights * (1.0 - class_trust))[None, :],
            axis=1,
        )
        empirical_component = np.sum(
            empirical_class_loss
            * (class_weights * class_trust)[None, :],
            axis=1,
        )
        total_loss = robust_component + empirical_component
        chosen = int(np.argmin(np.where(guard, total_loss, np.inf)))
        z_alpha = float(norm.ppf(1 - self.problem.alpha))
        empirical_sigma = float(fit.get("prequential_sigma", fit["sigma"]))
        calibrated_margin = (
            pred_con + z_alpha * empirical_sigma - float(self.problem.tau)
        )
        chance_feasible = violation_probability <= float(self.problem.alpha)
        return chosen, {
            "calibrated_recommendation_reason": (
                "task_posterior_expected_violation_loss"),
            "calibrated_objective": float(pred_obj[chosen]),
            "calibrated_constraint_margin": float(calibrated_margin[chosen]),
            "calibrated_guarded_constraint_margin": float(
                calibrated_margin[chosen]),
            "calibrated_constraint_feasible": bool(chance_feasible[chosen]),
            "calibrated_constraint_sigma": empirical_sigma,
            "calibrated_recommendation_scope": "pool",
            "n_calibration_candidates": int(len(pool)),
            "n_calibration_certified": 0,
            "n_calibration_certified_guarded": 0,
            "n_calibration_guarded": int(np.sum(guard)),
            "n_calibration_feasible": int(np.sum(chance_feasible & guard)),
            "n_calibration_raw_feasible": int(np.sum(chance_feasible)),
            "calibration_max_leverage": (
                None if max_leverage <= 0.0 else max_leverage),
            "calibration_max_theory_margin": (
                None if max_theory_margin <= 0.0 else max_theory_margin),
            "calibration_selected_leverage": float(leverage[chosen]),
            "calibration_selected_theory_margin": float(
                theory_margin[chosen]),
            "calibration_min_leverage": float(np.min(leverage[guard])),
            "calibration_median_leverage": float(np.median(leverage[guard])),
            "calibration_min_theory_margin": float(
                np.min(theory_margin[guard])),
            "task_adaptive_violation_probability": float(
                violation_probability[chosen]),
            "task_adaptive_expected_violation_loss": float(
                violation_loss[chosen]),
            "task_adaptive_objective_loss": float(
                empirical_objective_loss[chosen]),
            "task_adaptive_robust_component": float(
                robust_component[chosen]),
            "task_adaptive_empirical_component": float(
                empirical_component[chosen]),
            "task_adaptive_total_loss": float(total_loss[chosen]),
            "task_adaptive_class_weights": np.asarray(
                risk["posterior_weights"], dtype=float).tolist(),
            "task_adaptive_expected_empirical_trust": float(
                sensitivity.expected_empirical_trust()),
            "task_adaptive_prequential_sigma": empirical_sigma,
            "task_adaptive_loo_sigma": float(fit.get("loo_sigma", fit["sigma"])),
            "task_adaptive_conformal_sigma": float(
                fit.get("conformal_sigma", fit["sigma"])),
            "task_adaptive_empirical_hvd_variance": (
                None
                if model_aleatoric is None
                else float(np.asarray(model_aleatoric, dtype=float)[chosen])
            ),
            "task_adaptive_affects_theory_certificate": False,
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

    def _solve_posterior_recommendation(
        self,
        pool=None,
        terminal_frontier_count=0,
    ):
        pool = (
            self._recommendation_pool()
            if pool is None
            else [tuple(int(v) for v in x) for x in pool]
        )
        coherent_contract = self._coherent_certificate_contract()
        cached_expert_moments = bool(
            self.task_ensemble is not None
            and getattr(
                self.task_ensemble,
                "supports_precomputed_expert_moments",
                False,
            )
        )
        objective_expert_moments = None
        constraint_expert_moments = None
        if cached_expert_moments:
            objective_expert_moments = self.task_ensemble.expert_moments_many(
                0, pool, certification=False)
            mu_obj = self.task_ensemble.mixture_moments_many(
                0,
                pool,
                certification=False,
                expert_moments=objective_expert_moments,
            ).mean
        else:
            mu_obj = self._objective_posterior_mean_many(pool)
        empirical_aleatoric = None
        task_observation_nominal = None
        task_certification_nominal = None
        task_robust = None
        task_joint = None
        if self.task_ensemble is None:
            mu_con = self.gpr[1].posterior_mean_many(pool)
            v_con = self.variance_model.predict_certification_variance_many(
                1, pool, self.problem)
            cert = self._certification_result(mu_con, pool, v_con)
        else:
            if cached_expert_moments:
                observation_expert_moments = (
                    self.task_ensemble.expert_moments_many(
                        1, pool, certification=False)
                )
                constraint_expert_moments = (
                    self.task_ensemble.expert_moments_many(
                        1, pool, certification=True)
                )
                task_observation_nominal = (
                    self.task_ensemble.mixture_moments_many(
                        1,
                        pool,
                        certification=False,
                        expert_moments=observation_expert_moments,
                    )
                )
                task_certification_nominal = (
                    self.task_ensemble.mixture_moments_many(
                        1,
                        pool,
                        certification=True,
                        expert_moments=constraint_expert_moments,
                    )
                )
                task_robust = self.task_ensemble.robust_moments_many(
                    1,
                    pool,
                    certification=True,
                    expert_moments=constraint_expert_moments,
                )
            else:
                task_observation_nominal = (
                    self.task_ensemble.mixture_moments_many(
                        1, pool, certification=False)
                )
                task_certification_nominal = (
                    self.task_ensemble.mixture_moments_many(
                        1, pool, certification=True)
                )
                task_robust = self.task_ensemble.robust_moments_many(
                    1, pool, certification=True)
            empirical_aleatoric = np.asarray(
                task_certification_nominal.aleatoric, dtype=float)
            mu_con = task_robust.mean_upper
            v_con = task_robust.aleatoric_upper
            cert = self._certification_result(
                mu_con,
                pool,
                v_con,
                epistemic=task_robust.epistemic_upper,
            )
            # Downstream ranking and diagnostics must use the same heads that
            # produced the certificate.  In split mode these are the
            # aggregate GPR mean/epistemic and exactly one HVD variance head.
            mu_con = np.asarray(cert.mu, dtype=float)
            v_con = np.asarray(cert.aleatoric_var, dtype=float)
            joint_kwargs = {}
            if cached_expert_moments:
                joint_kwargs["expert_moments"] = constraint_expert_moments
            task_joint = self.task_ensemble.robust_chance_margin_many(
                pool,
                beta_g=cert.beta_g,
                z_alpha=cert.z_alpha,
                tau=self.problem.tau,
                certification=True,
                **joint_kwargs,
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
        recommendation_slack = self._recommendation_slack()
        theory_margins = np.asarray(margins + safety_buffer, dtype=float)
        robust_margins = theory_margins + recommendation_slack
        effective_mu_con = np.asarray(mu_con, dtype=float)
        effective_epistemic = np.asarray(cert.epistemic_var, dtype=float)
        effective_aleatoric = np.asarray(cert.aleatoric_var, dtype=float)
        certification_sources = np.full(
            len(pool),
            self._certification_source(),
            dtype=object,
        )
        tcb_v2 = self._tcb_v2_margin_many(pool)
        tcb_v2_mode = self._tcb_v2_mode()
        if tcb_v2_mode == "certified" and tcb_v2 is None:
            raise RuntimeError(
                "TCB-V2 certified recommendation requires a fitted "
                "hierarchical boundary provider")
        tcb_v2_authoritative = bool(
            tcb_v2_mode == "certified" and tcb_v2 is not None)
        if tcb_v2_authoritative:
            # TCB-V2 predicts the signed chance margin directly.  Its upper
            # posterior bound is therefore the complete certificate; mixing
            # it with a second calibration or empirical override would create
            # a different decision rule at recommendation time.
            theory_margins = np.asarray(tcb_v2["upper"], dtype=float)
            robust_margins = theory_margins + recommendation_slack
            certification_sources = np.full(
                len(pool), "tcb_v2_hierarchical", dtype=object)
        calibrated_cert = (
            None
            if (
                coherent_contract
                or self.task_ensemble is not None
                or tcb_v2_authoritative
            )
            else self._calibrated_certification_result(pool, v_con)
        )
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
        infeasible_strategy = str(
            self.config.recommendation_infeasible_strategy).lower()
        task_adaptive_empirical = bool(
            self.task_ensemble is not None
            and self.task_ensemble.sensitivity_posterior is not None
            and infeasible_strategy
            in ("task_adaptive", "task-adaptive", "sensitivity_posterior")
        )
        recommendation_calibration_fit = (
            self._recommendation_calibration_fit()
            if (
                not coherent_contract
                and
                not tcb_v2_authoritative
                and (self.task_ensemble is None or task_adaptive_empirical)
            )
            else None
        )
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
        effective_infeasible_penalty = float(
            self.config.recommendation_infeasible_penalty)
        if (
            infeasible_strategy
            in ("task_adaptive", "task-adaptive", "sensitivity_posterior")
            and self.task_ensemble is not None
        ):
            effective_infeasible_penalty = (
                self.task_ensemble.adaptive_infeasible_penalty(
                    fallback=effective_infeasible_penalty)
            )
        feasible = robust_margins <= 0.0
        bayes_risk_details = {
            "posterior_bayes_risk_used": False,
            "posterior_bayes_risk": None,
            "posterior_bayes_objective": None,
            "posterior_bayes_expected_violation": None,
            "posterior_bayes_kl_radius": None,
        }
        bayes_components = None
        if (coherent_contract or tcb_v2_authoritative) and np.any(feasible):
            local = int(np.argmin(np.where(feasible, mu_obj, np.inf)))
        elif coherent_contract or tcb_v2_authoritative:
            min_margin = float(np.min(robust_margins))
            near_min_margin = robust_margins <= min_margin + 1e-12
            local = int(np.argmin(np.where(near_min_margin, mu_obj, np.inf)))
        elif np.any(feasible):
            local = int(np.argmin(np.where(feasible, mu_obj, np.inf)))
        elif infeasible_strategy in (
            "bayes_risk",
            "bayes-risk",
            "posterior_bayes_risk",
        ):
            components = self._terminal_bayes_risk_components(
                self.gpr,
                self.variance_model,
                pool,
                task_ensemble=self.task_ensemble,
                objective_expert_moments=objective_expert_moments,
                constraint_expert_moments=constraint_expert_moments,
            )
            bayes_components = components
            local = int(np.argmin(components["risk"]))
            bayes_risk_details = {
                "posterior_bayes_risk_used": True,
                "posterior_bayes_risk": float(components["risk"][local]),
                "posterior_bayes_objective": float(
                    components["objective"][local]),
                "posterior_bayes_expected_violation": float(
                    components["expected_violation"][local]),
                "posterior_bayes_kl_radius": float(
                    components["kl_radius"]),
            }
        elif infeasible_strategy in (
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
                + effective_infeasible_penalty * scaled_margin
            ))
        used_observed_incumbent = False
        observed_incumbent_rejected = False
        observed_incumbent_reason = None
        observed_idx = None
        observed_incumbent = (
            None
            if coherent_contract or tcb_v2_authoritative
            else self._observed_nominal_incumbent()
        )
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
                    and not tcb_v2_authoritative
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
            not coherent_contract
            and
            self.task_ensemble is None
            and
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
        if coherent_contract:
            calibrated_idx = None
            calibrated_details = {
                "calibrated_recommendation_reason": (
                    "disabled_by_coherent_certificate_contract")
            }
        elif self.task_ensemble is None:
            calibrated_idx, calibrated_details = (
                self._calibrated_recommendation_index(
                    pool,
                    robust_margins,
                    source_margins=source_margins,
                    guard_margins=theory_margins + recommendation_slack,
                    fit=recommendation_calibration_fit,
                    phi_pool=recommendation_calibration_phi_pool,
                )
            )
        elif task_adaptive_empirical and not np.any(feasible):
            calibrated_idx, calibrated_details = (
                self._task_adaptive_recommendation_index(
                    pool,
                    robust_margins,
                    guard_margins=theory_margins + recommendation_slack,
                    model_objective=mu_obj,
                    model_aleatoric=empirical_aleatoric,
                    fit=recommendation_calibration_fit,
                    phi_pool=recommendation_calibration_phi_pool,
                )
            )
        else:
            calibrated_idx = None
            calibrated_details = {
                "calibrated_recommendation_reason": (
                    "disabled_for_task_posterior_robust_certificate")
            }
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
            not coherent_contract
            and
            self.task_ensemble is None
            and
            self.config.source_mean_prior_fallback
            and not tcb_v2_authoritative
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
        replicated_finalist_details = {
            "replicated_finalist_used": False,
        }
        if (
            not coherent_contract
            and not np.any(feasible)
            and not tcb_v2_authoritative
        ):
            replicated_idx, replicated_finalist_details = (
                self._replicated_finalist_recommendation_index(pool)
            )
            if replicated_idx is not None:
                local = int(replicated_idx)
                used_observed_incumbent = False
                calibrated_recommendation_used = False
                source_prior_recommendation_used = False
                if bayes_components is not None:
                    bayes_risk_details.update({
                        "posterior_bayes_risk": float(
                            bayes_components["risk"][local]),
                        "posterior_bayes_objective": float(
                            bayes_components["objective"][local]),
                        "posterior_bayes_expected_violation": float(
                            bayes_components["expected_violation"][local]),
                    })
        frontier_budget = int(terminal_frontier_count)
        if tcb_v2 is not None and tcb_v2_mode in ("frontier", "certified"):
            frontier_budget = max(
                frontier_budget,
                1 + max(int(self.config.tcb_v2_frontier_count), 0),
            )
        frontier_indices, frontier_labels = self._terminal_frontier_indices(
            mu_obj,
            robust_margins,
            local,
            frontier_budget,
            bayes_components=bayes_components,
            tcb_upper=(
                None if tcb_v2 is None else tcb_v2["upper"]),
            tcb_count=(
                0
                if tcb_v2_mode not in ("frontier", "certified")
                else self.config.tcb_v2_frontier_count
            ),
            chosen_label=(
                "tcb_certified_action"
                if tcb_v2_authoritative else "bayes_action"
            ),
        )
        frontier_candidates = [
            tuple(int(v) for v in pool[index])
            for index in frontier_indices
        ]
        x_best = tuple(int(v) for v in pool[local])

        def margin_components(
            mean,
            epistemic,
            aleatoric,
            index,
            extra=0.0,
            margin_override=None,
        ):
            mean_term = float(np.asarray(mean, dtype=float)[index]) - float(
                self.problem.tau)
            epistemic_term = float(
                np.sqrt(max(float(cert.beta_g), 0.0))
                * np.sqrt(max(float(np.asarray(epistemic, dtype=float)[index]), 0.0))
            )
            aleatoric_term = float(
                cert.z_alpha
                * np.sqrt(max(float(np.asarray(aleatoric, dtype=float)[index]), 0.0))
            )
            extra_array = np.asarray(extra, dtype=float)
            extra_term = float(
                extra_array if extra_array.ndim == 0 else extra_array[index])
            component_margin = float(
                mean_term + epistemic_term + aleatoric_term + extra_term)
            margin_value = (
                component_margin
                if margin_override is None
                else float(np.asarray(margin_override, dtype=float)[index])
            )
            return {
                "mean_minus_tau": mean_term,
                "epistemic_radius": epistemic_term,
                "aleatoric_radius": aleatoric_term,
                "extra_guard": extra_term,
                "component_sum_margin": component_margin,
                "joint_coupling_correction": float(
                    margin_value - component_margin),
                "margin": margin_value,
                "epistemic_variance": float(
                    np.asarray(epistemic, dtype=float)[index]),
                "aleatoric_variance": float(
                    np.asarray(aleatoric, dtype=float)[index]),
            }

        def margin_audit_at(index):
            row = {
                "pool_index": int(index),
                "x": [int(v) for v in pool[index]],
                "final_certificate": margin_components(
                    effective_mu_con,
                    effective_epistemic,
                    effective_aleatoric,
                    index,
                    safety_buffer + recommendation_slack,
                    margin_override=robust_margins,
                ),
                "theory_certificate": margin_components(
                    effective_mu_con,
                    effective_epistemic,
                    effective_aleatoric,
                    index,
                    safety_buffer,
                    margin_override=theory_margins,
                ),
            }
            if task_observation_nominal is None:
                return row
            observation = margin_components(
                task_observation_nominal.mean,
                task_observation_nominal.epistemic,
                task_observation_nominal.aleatoric,
                index,
            )
            expert_certified = margin_components(
                task_certification_nominal.mean,
                task_certification_nominal.epistemic,
                task_certification_nominal.aleatoric,
                index,
            )
            separable_robust = margin_components(
                task_robust.mean_upper,
                task_robust.epistemic_upper,
                task_robust.aleatoric_upper,
                index,
            )
            joint_robust = margin_components(
                task_robust.mean_upper,
                task_robust.epistemic_upper,
                task_robust.aleatoric_upper,
                index,
                margin_override=task_joint.upper,
            )
            robust = (
                joint_robust
                if self._task_robust_certificate_mode() == "joint_tangent"
                else separable_robust
            )
            observation_aleatoric = max(
                float(observation["aleatoric_variance"]), 1e-12)
            expert_aleatoric = max(
                float(expert_certified["aleatoric_variance"]), 1e-12)
            observation_epistemic = max(
                float(observation["epistemic_variance"]), 1e-12)
            expert_epistemic = max(
                float(expert_certified["epistemic_variance"]), 1e-12)
            row.update({
                "observation_nominal": observation,
                "expert_certified": expert_certified,
                "task_robust": robust,
                "task_separable_robust": separable_robust,
                "task_joint_robust": joint_robust,
                "task_joint_tightening": float(
                    separable_robust["margin"] - joint_robust["margin"]),
                "task_joint_used_separable_upper": bool(
                    task_joint.used_separable_upper[index]),
                "task_joint_epistemic_tangent_scale": float(
                    task_joint.tangent_epistemic_scale[index]),
                "task_joint_aleatoric_tangent_scale": float(
                    task_joint.tangent_aleatoric_scale[index]),
                "expert_to_observation_aleatoric_ratio": float(
                    expert_aleatoric / observation_aleatoric),
                "robust_to_expert_aleatoric_ratio": float(
                    max(float(robust["aleatoric_variance"]), 1e-12)
                    / expert_aleatoric),
                "expert_to_observation_epistemic_ratio": float(
                    expert_epistemic / observation_epistemic),
                "robust_to_expert_epistemic_ratio": float(
                    max(float(robust["epistemic_variance"]), 1e-12)
                    / expert_epistemic),
                "robust_mean_inflation": float(
                    robust["mean_minus_tau"]
                    - expert_certified["mean_minus_tau"]),
            })
            return row

        minimum_margin_index = int(np.argmin(robust_margins))
        certification_margin_decomposition = {
            "schema_version": 1,
            "task_posterior_active": bool(self.task_ensemble is not None),
            "certification_head_authority": (
                self._certification_head_authority()),
            "task_robust_certificate_mode": (
                self._task_robust_certificate_mode()
                if self.task_ensemble is not None else None
            ),
            "selected": margin_audit_at(local),
            "minimum_margin": margin_audit_at(minimum_margin_index),
            "selected_is_minimum_margin": bool(local == minimum_margin_index),
            "n_pool": int(len(pool)),
            "n_certified": int(np.sum(robust_margins <= 0.0)),
        }
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
                "recommendation_calibration_selected_ridge": float(
                    recommendation_calibration_audit["selected_ridge"]),
                "recommendation_calibration_effective_rank": float(
                    recommendation_calibration_audit["effective_rank"]),
                "recommendation_calibration_effective_rank_cap": (
                    recommendation_calibration_audit["effective_rank_cap"]),
                "recommendation_calibration_rank_cap_satisfied": (
                    recommendation_calibration_audit["rank_cap_satisfied"]),
                "recommendation_calibration_nested_refit": bool(
                    recommendation_calibration_audit["nested_refit"]),
                "recommendation_calibration_ridge_scores": copy.deepcopy(
                    recommendation_calibration_audit["ridge_scores"]),
                "recommendation_calibration_features_standardized": bool(
                    recommendation_calibration_audit[
                        "features_standardized"]),
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
            "decision_contract_mode": self._decision_contract_mode(),
            "decision_contract_coherent": bool(coherent_contract),
            "posterior_chance_margin": float(robust_margins[local]),
            "posterior_theory_chance_margin": float(theory_margins[local]),
            "posterior_robust_chance_margin": float(robust_margins[local]),
            "posterior_certification_source": str(certification_sources[local]),
            "certification_margin_decomposition": (
                certification_margin_decomposition),
            "tcb_v2_mode": tcb_v2_mode,
            "tcb_v2_available": bool(tcb_v2 is not None),
            "tcb_v2_authoritative": bool(tcb_v2_authoritative),
            "tcb_v2_target_oracle_used": (
                None if tcb_v2 is None
                else bool(tcb_v2["target_oracle_used"])
            ),
            "tcb_v2_pilot_points": (
                None if tcb_v2 is None else int(tcb_v2["pilot_points"])
            ),
            "tcb_v2_margin_mean": (
                None if tcb_v2 is None
                else float(tcb_v2["mean"][local])
            ),
            "tcb_v2_margin_upper": (
                None if tcb_v2 is None
                else float(tcb_v2["upper"][local])
            ),
            "tcb_v2_adapter_diagnostics": (
                None if tcb_v2 is None
                else copy.deepcopy(tcb_v2["adapter_diagnostics"])
            ),
            "recommendation_safety_z": float(self.config.recommendation_safety_z),
            "recommendation_noise_floor_scale": float(
                self.config.recommendation_noise_floor_scale),
            "recommendation_slack": float(recommendation_slack),
            "recommendation_infeasible_penalty": float(
                self.config.recommendation_infeasible_penalty),
            "recommendation_effective_infeasible_penalty": float(
                effective_infeasible_penalty),
            "recommendation_infeasible_strategy": str(
                self.config.recommendation_infeasible_strategy),
            "recommendation_observed_fallback": bool(
                self.config.recommendation_observed_fallback),
            "recommendation_calibration": bool(self.config.recommendation_calibration),
            "task_posterior_mode": str(self.config.task_posterior_mode),
            "task_posterior_active": bool(self.task_ensemble is not None),
            "certification_head_authority": (
                self._certification_head_authority()),
            "task_posterior_weights": (
                None
                if self.task_ensemble is None
                else self.task_ensemble.posterior.posterior_weights().tolist()
            ),
            "task_posterior_experts": (
                None
                if self.task_ensemble is None
                else list(self.task_ensemble.posterior.expert_names)
            ),
            "task_posterior_entropy": (
                None
                if self.task_ensemble is None
                else self.task_ensemble.posterior.entropy()
            ),
            "task_posterior_kl_radius": (
                None
                if self.task_ensemble is None
                else self.task_ensemble.effective_kl_radius()
            ),
            "calibrated_recommendation_used": bool(calibrated_recommendation_used),
            "source_prior_recommendation_used": bool(source_prior_recommendation_used),
            **bayes_risk_details,
            **calibration_details,
            **recommendation_calibration_details,
            **calibrated_details,
            **source_prior_details,
            **replicated_finalist_details,
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
            "observed_incumbent_sigma": (
                None if observed_incumbent is None
                else float(observed_incumbent["empirical_sigma"])
            ),
            "observed_incumbent_sigma_source": (
                None if observed_incumbent is None
                else str(observed_incumbent["empirical_sigma_source"])
            ),
            "observed_incumbent_replicate_count": (
                None if observed_incumbent is None
                else int(observed_incumbent["replicate_count"])
            ),
            "posterior_feasible": bool(feasible[local]),
            "n_pool": int(len(pool)),
            "n_posterior_feasible": int(np.sum(feasible)),
            "n_theory_posterior_feasible": int(np.sum(theory_margins <= 0.0)),
            "terminal_frontier_candidate_count": int(
                len(frontier_candidates)),
            "terminal_frontier_labels": list(frontier_labels),
            **(
                {"_terminal_frontier_candidates": frontier_candidates}
                if int(frontier_budget) > 0
                else {}
            ),
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

    @staticmethod
    def _normal_positive_part(mean, variance):
        """Expected positive part of a Gaussian latent chance margin."""
        mean = np.asarray(mean, dtype=float)
        variance = np.maximum(np.asarray(variance, dtype=float), 0.0)
        sd = np.sqrt(variance)
        safe_sd = np.maximum(sd, 1e-12)
        standardized = mean / safe_sd
        value = (
            safe_sd * norm.pdf(standardized)
            + mean * norm.cdf(standardized)
        )
        return np.where(sd > 1e-12, value, np.maximum(mean, 0.0))

    @staticmethod
    def _normal_positive_part_variance(mean, variance):
        """Variance of the positive part of a Gaussian random variable."""
        mean = np.asarray(mean, dtype=float)
        variance = np.maximum(np.asarray(variance, dtype=float), 0.0)
        sd = np.sqrt(variance)
        safe_sd = np.maximum(sd, 1e-12)
        standardized = mean / safe_sd
        first = SingleOLHKGAlgorithm._normal_positive_part(mean, variance)
        second = (
            (variance + mean ** 2) * norm.cdf(standardized)
            + mean * safe_sd * norm.pdf(standardized)
        )
        second = np.where(
            sd > 1e-12,
            second,
            np.maximum(mean, 0.0) ** 2,
        )
        return np.maximum(second - first ** 2, 0.0)

    def _constraint_certification_epistemic_many(self, model, pool):
        """Return the single epistemic law authorized for certification."""

        mode = str(
            self.config.source_constraint_mean_confidence_mode or "model"
        ).strip().lower()
        if mode == "model":
            values = np.maximum(np.asarray(
                model.posterior_var_many(pool), dtype=float), 0.0)
            diagnostics = {
                "mode": "model",
                "status": "active",
                "target_oracle_used": False,
            }
        elif mode in {"source_bayes", "source_self_normalized"}:
            if not hasattr(model, "source_conditioned_certification_var_many"):
                raise RuntimeError(
                    "source-conditioned confidence requires ParametricGPR support"
                )
            values, diagnostics = (
                model.source_conditioned_certification_var_many(
                    pool,
                    beta_g=self.config.beta_g,
                    mode=mode,
                    delta=(
                        self.config
                        .source_constraint_mean_confidence_delta),
                )
            )
            values = np.maximum(np.asarray(values, dtype=float), 0.0)
        else:
            raise ValueError(
                "source constraint mean confidence mode must be model, "
                "source_bayes, or source_self_normalized"
            )
        if model is self.gpr[1]:
            self._last_source_conditioned_confidence_diagnostics = (
                copy.deepcopy(diagnostics))
        return values

    def _constraint_decision_epistemic_many(self, model, pool):
        mode = str(
            self.config.source_constraint_mean_misspecification_mode or "none"
        ).strip().lower()
        confidence_only = mode in {
            "predictive_scale_sandwich_hc3_confidence",
            "predictive_scale_sandwich_hc3_task_confidence",
        }
        if confidence_only and hasattr(model, "decision_posterior_var_many"):
            return np.maximum(np.asarray(
                model.decision_posterior_var_many(pool), dtype=float), 0.0)
        return np.maximum(np.asarray(
            model.posterior_var_many(pool), dtype=float), 0.0)

    def _terminal_bayes_risk_components(
        self,
        gpr_models,
        variance_model,
        pool,
        task_ensemble=None,
        risk_penalty=None,
        objective_expert_moments=None,
        constraint_expert_moments=None,
    ):
        """Fixed posterior Bayes risk used by smooth constrained KG.

        The decision and certification variance views are deliberately
        separate. ``posterior_central`` uses the cumulative-HVD posterior mean
        in Bayes actions, while ``certification_upper`` retains the historical
        conservative upper variance. Certification itself always uses the
        upper view elsewhere. The violation loss is either expected positive
        margin severity or posterior chance-failure probability. Likewise,
        ``posterior_nominal`` uses the actual task-posterior mixture in a Bayes
        action while certification retains its KL-robust upper envelope.
        """
        aleatoric_mode = str(
            self.config.decision_aleatoric_mode
            or "certification_upper"
        ).strip().lower().replace("-", "_")
        if aleatoric_mode not in {
            "certification_upper", "posterior_central",
        }:
            raise ValueError(
                "decision aleatoric mode must be certification_upper or "
                "posterior_central"
            )
        violation_loss_mode = str(
            self.config.decision_violation_loss_mode
            or "positive_part"
        ).strip().lower().replace("-", "_")
        if violation_loss_mode not in {
            "positive_part", "failure_probability",
        }:
            raise ValueError(
                "decision violation loss mode must be positive_part or "
                "failure_probability"
            )
        ambiguity_mode = str(
            self.config.decision_ambiguity_mode
            or "kl_robust"
        ).strip().lower().replace("-", "_")
        if ambiguity_mode not in {"kl_robust", "posterior_nominal"}:
            raise ValueError(
                "decision ambiguity mode must be kl_robust or "
                "posterior_nominal"
            )
        if aleatoric_mode == "posterior_central":
            # Callers historically cached certification=True expert moments.
            # A central Bayes action must refetch the semantically distinct
            # posterior view instead of reinterpreting that cache.
            constraint_expert_moments = None
        if len(pool) == 0:
            empty = np.asarray([], dtype=float)
            return {
                "objective": empty,
                "objective_variance": empty,
                "expected_violation": empty,
                "nominal_expected_violation": empty,
                "probability_violation": empty,
                "nominal_probability_violation": empty,
                "violation_loss": empty,
                "nominal_violation_loss": empty,
                "violation_variance": empty,
                "risk": empty,
                "risk_variance": empty,
                "model_disagreement": empty,
                "kl_radius": 0.0,
                "certification_head_authority": (
                    self._certification_head_authority()),
                "constraint_posterior_source": "empty",
                "decision_aleatoric_mode": aleatoric_mode,
                "violation_loss_mode": violation_loss_mode,
                "decision_ambiguity_mode": ambiguity_mode,
            }
        penalty = max(float(
            self.config.terminal_bayes_violation_penalty
            if risk_penalty is None
            else risk_penalty
        ), 0.0)
        robust_expected_violation = None
        robust_probability_violation = None
        if (
            task_ensemble is not None
            and self._certification_head_authority() == "task_joint"
            and bool(getattr(
                task_ensemble, "task_latent_authoritative", False))
            and aleatoric_mode == "certification_upper"
            and violation_loss_mode == "positive_part"
            and ambiguity_mode == "kl_robust"
        ):
            authoritative = task_ensemble.joint_terminal_risk_many(
                pool,
                tau=self.problem.tau,
                alpha=self.problem.alpha,
            )
            disagreement = np.asarray(
                authoritative.get("model_disagreement", 0.0), dtype=float)
            authoritative.setdefault(
                "objective_variance", np.zeros(len(pool), dtype=float))
            authoritative.setdefault(
                "violation_variance", disagreement ** 2)
            authoritative.setdefault(
                "risk_variance", np.maximum(disagreement ** 2, 1e-12))
            expected = np.asarray(
                authoritative.get("expected_violation", 0.0), dtype=float)
            authoritative.setdefault("violation_loss", expected)
            authoritative.setdefault("nominal_violation_loss", expected)
            authoritative.setdefault(
                "probability_violation",
                np.full(len(pool), np.nan, dtype=float),
            )
            authoritative.setdefault(
                "nominal_probability_violation",
                np.full(len(pool), np.nan, dtype=float),
            )
            authoritative.setdefault(
                "decision_aleatoric_mode", aleatoric_mode)
            authoritative.setdefault(
                "violation_loss_mode", violation_loss_mode)
            authoritative.setdefault(
                "decision_ambiguity_mode", ambiguity_mode)
            authoritative.setdefault(
                "robust_expected_violation", expected)
            authoritative.setdefault(
                "nominal_expected_violation", expected)
            authoritative.setdefault(
                "robust_probability_violation",
                authoritative["probability_violation"],
            )
            return authoritative
        z_alpha = float(norm.ppf(1 - self.problem.alpha))
        if task_ensemble is None:
            objective = np.asarray(
                gpr_models[0].posterior_mean_many(pool), dtype=float)
            objective_variance = np.maximum(np.asarray(
                gpr_models[0].posterior_var_many(pool), dtype=float), 0.0)
            mu_con = np.asarray(
                gpr_models[1].posterior_mean_many(pool), dtype=float)
            if hasattr(self.problem, "pilot_constraint_guard"):
                mu_con = mu_con + max(
                    float(self.problem.pilot_constraint_guard()), 0.0)
            robust_epistemic = np.maximum(np.asarray(
                gpr_models[1].posterior_var_many(pool), dtype=float), 0.0)
            decision_epistemic = self._constraint_decision_epistemic_many(
                gpr_models[1], pool)
            variance_method = (
                variance_model.predict_certification_variance_many
                if aleatoric_mode == "certification_upper"
                else variance_model.predict_variance_many
            )
            aleatoric = np.maximum(np.asarray(
                variance_method(1, pool, self.problem), dtype=float), 0.0)
            margin_mean = (
                mu_con + z_alpha * np.sqrt(aleatoric) - self.problem.tau)
            expected_violation = self._normal_positive_part(
                margin_mean, decision_epistemic)
            positive_part_variance = self._normal_positive_part_variance(
                margin_mean, robust_epistemic)
            probability_violation = norm.cdf(
                margin_mean / np.sqrt(np.maximum(
                    decision_epistemic, 1e-12)))
            probability_variance = np.maximum(
                probability_violation * (1.0 - probability_violation), 0.0)
            nominal_violation = np.asarray(
                expected_violation, dtype=float)
            nominal_probability_violation = np.asarray(
                probability_violation, dtype=float)
            if violation_loss_mode == "failure_probability":
                violation_loss = probability_violation
                nominal_violation_loss = nominal_probability_violation
                violation_variance = probability_variance
            else:
                violation_loss = expected_violation
                nominal_violation_loss = nominal_violation
                violation_variance = positive_part_variance
            model_disagreement = np.sqrt(np.maximum(
                violation_variance, 0.0))
            kl_radius = 0.0
        else:
            if objective_expert_moments is None:
                objective_expert_moments = (
                    task_ensemble.expert_moments_many(
                        0, pool, certification=False)
                )
            obj_mu, obj_epistemic, _ = objective_expert_moments
            decision_weights = task_ensemble.posterior.decision_weights()
            objective_weights = (
                task_ensemble.posterior.posterior_weights()
                if task_ensemble.posterior.safe_generalized
                else decision_weights
            )
            objective = np.asarray(objective_weights @ obj_mu, dtype=float)
            objective_variance = np.maximum(
                objective_weights @ np.maximum(obj_epistemic, 0.0)
                + objective_weights @ (
                    obj_mu - objective[None, :]
                ) ** 2,
                0.0,
            )
            authority = self._certification_head_authority()
            if authority == "task_joint":
                if constraint_expert_moments is None:
                    constraint_expert_moments = (
                        task_ensemble.expert_moments_many(
                            1,
                            pool,
                            certification=(
                                aleatoric_mode == "certification_upper"),
                        )
                    )
                con_mu, con_epistemic, con_aleatoric = (
                    constraint_expert_moments)
                expert_margin_mean = (
                    np.asarray(con_mu, dtype=float)
                    + z_alpha * np.sqrt(np.maximum(con_aleatoric, 0.0))
                    - self.problem.tau
                )
                expert_violation = self._normal_positive_part(
                    expert_margin_mean,
                    np.maximum(con_epistemic, 0.0),
                )
                expert_violation_variance = (
                    self._normal_positive_part_variance(
                        expert_margin_mean,
                        np.maximum(con_epistemic, 0.0),
                    )
                )
                expert_probability_violation = norm.cdf(
                    expert_margin_mean / np.sqrt(np.maximum(
                        con_epistemic, 1e-12)))
                expert_probability_variance = np.maximum(
                    expert_probability_violation
                    * (1.0 - expert_probability_violation),
                    0.0,
                )
                kl_radius = float(task_ensemble.effective_kl_radius())
                robust_expected_violation = np.asarray(
                    task_ensemble.posterior.kl_robust_expectation(
                        expert_violation,
                        kl_radius,
                    ),
                    dtype=float,
                )
                nominal_violation = np.asarray(
                    decision_weights @ expert_violation,
                    dtype=float,
                )
                robust_probability_violation = np.asarray(
                    task_ensemble.posterior.kl_robust_expectation(
                        expert_probability_violation,
                        kl_radius,
                    ),
                    dtype=float,
                )
                nominal_probability_violation = np.asarray(
                    decision_weights @ expert_probability_violation,
                    dtype=float,
                )
                if ambiguity_mode == "posterior_nominal":
                    expected_violation = nominal_violation
                    probability_violation = nominal_probability_violation
                else:
                    expected_violation = robust_expected_violation
                    probability_violation = robust_probability_violation
                if violation_loss_mode == "failure_probability":
                    expert_loss = expert_probability_violation
                    expert_loss_variance = expert_probability_variance
                    violation_loss = probability_violation
                    nominal_violation_loss = nominal_probability_violation
                else:
                    expert_loss = expert_violation
                    expert_loss_variance = expert_violation_variance
                    violation_loss = expected_violation
                    nominal_violation_loss = nominal_violation
                model_disagreement = np.sqrt(np.maximum(
                    decision_weights @ (
                        expert_loss
                        - nominal_violation_loss[None, :]
                    ) ** 2,
                    0.0,
                ))
                violation_variance = np.maximum(
                    decision_weights @ expert_loss_variance
                    + decision_weights @ (
                        expert_loss - nominal_violation_loss[None, :]
                    ) ** 2,
                    0.0,
                )
            else:
                mu_con = np.asarray(
                    gpr_models[1].posterior_mean_many(pool), dtype=float)
                robust_epistemic = np.maximum(np.asarray(
                    gpr_models[1].posterior_var_many(pool), dtype=float), 0.0)
                decision_epistemic = self._constraint_decision_epistemic_many(
                    gpr_models[1], pool)
                if authority == "split_gpr_task_hvd":
                    if ambiguity_mode == "posterior_nominal":
                        nominal = task_ensemble.mixture_moments_many(
                            1,
                            pool,
                            certification=(
                                aleatoric_mode == "certification_upper"),
                        )
                        aleatoric = np.maximum(np.asarray(
                            nominal.aleatoric, dtype=float), 0.0)
                    else:
                        robust = task_ensemble.robust_moments_many(
                            1,
                            pool,
                            certification=(
                                aleatoric_mode == "certification_upper"),
                        )
                        aleatoric = np.maximum(np.asarray(
                            robust.aleatoric_upper, dtype=float), 0.0)
                else:
                    variance_method = (
                        variance_model.predict_certification_variance_many
                        if aleatoric_mode == "certification_upper"
                        else variance_model.predict_variance_many
                    )
                    aleatoric = np.maximum(np.asarray(
                        variance_method(1, pool, self.problem), dtype=float),
                        0.0)
                margin_mean = (
                    mu_con + z_alpha * np.sqrt(aleatoric)
                    - self.problem.tau
                )
                expected_violation = self._normal_positive_part(
                    margin_mean, decision_epistemic)
                nominal_violation = np.asarray(
                    expected_violation, dtype=float)
                positive_part_variance = self._normal_positive_part_variance(
                    margin_mean, robust_epistemic)
                probability_violation = norm.cdf(
                    margin_mean / np.sqrt(np.maximum(
                        decision_epistemic, 1e-12)))
                nominal_probability_violation = np.asarray(
                    probability_violation, dtype=float)
                probability_variance = np.maximum(
                    probability_violation
                    * (1.0 - probability_violation), 0.0)
                if violation_loss_mode == "failure_probability":
                    violation_loss = probability_violation
                    nominal_violation_loss = nominal_probability_violation
                    violation_variance = probability_variance
                else:
                    violation_loss = expected_violation
                    nominal_violation_loss = nominal_violation
                    violation_variance = positive_part_variance
                model_disagreement = np.sqrt(np.maximum(
                    violation_variance, 0.0))
                kl_radius = 0.0
        if robust_expected_violation is None:
            robust_expected_violation = np.asarray(
                expected_violation, dtype=float)
        if robust_probability_violation is None:
            robust_probability_violation = np.asarray(
                probability_violation, dtype=float)
        risk = objective + penalty * violation_loss
        # Cauchy-Schwarz gives a covariance-free upper bound for the loss
        # variance. This is intentionally conservative because it feeds the
        # one-sided Cantelli switching certificate below.
        risk_variance = (
            np.sqrt(np.maximum(objective_variance, 0.0))
            + penalty * np.sqrt(np.maximum(violation_variance, 0.0))
        ) ** 2
        if task_ensemble is not None and ambiguity_mode == "kl_robust":
            nominal_risk = objective + penalty * nominal_violation_loss
            risk_variance = (
                np.sqrt(np.maximum(risk_variance, 0.0))
                + np.abs(risk - nominal_risk)
            ) ** 2
        return {
            "objective": np.asarray(objective, dtype=float),
            "objective_variance": np.asarray(
                objective_variance, dtype=float),
            "expected_violation": np.asarray(
                expected_violation, dtype=float),
            "nominal_expected_violation": np.asarray(
                nominal_violation, dtype=float),
            "robust_expected_violation": np.asarray(
                robust_expected_violation, dtype=float),
            "probability_violation": np.asarray(
                probability_violation, dtype=float),
            "nominal_probability_violation": np.asarray(
                nominal_probability_violation, dtype=float),
            "robust_probability_violation": np.asarray(
                robust_probability_violation, dtype=float),
            "violation_loss": np.asarray(violation_loss, dtype=float),
            "nominal_violation_loss": np.asarray(
                nominal_violation_loss, dtype=float),
            "violation_variance": np.asarray(
                violation_variance, dtype=float),
            "risk": np.asarray(risk, dtype=float),
            "risk_variance": np.asarray(risk_variance, dtype=float),
            "model_disagreement": np.asarray(
                model_disagreement, dtype=float),
            "kl_radius": float(kl_radius),
            "certification_head_authority": (
                self._certification_head_authority()),
            "constraint_posterior_source": self._certification_source(
                task_ensemble_active=task_ensemble is not None),
            "decision_aleatoric_mode": aleatoric_mode,
            "violation_loss_mode": violation_loss_mode,
            "decision_ambiguity_mode": ambiguity_mode,
        }

    def _posterior_dominance_active(self):
        mode = str(
            self.config.exact_kg_terminal_mode or ""
        ).lower().replace("-", "_")
        return bool(
            self.config.posterior_dominance_enabled
            or mode in {
                "bayes_risk_dominance",
                "posterior_bayes_risk_dominance",
            }
        )

    @staticmethod
    def _cantelli_dominance_lower_bound(
        incumbent_mean,
        challenger_mean,
        incumbent_variance,
        challenger_variance,
        min_mean_gain=0.0,
    ):
        """Lower-bound posterior improvement probability via Cantelli.

        The variance of a difference is bounded without a covariance
        assumption by ``(sqrt(v_inc) + sqrt(v_chal))^2``.  Cantelli's
        one-sided inequality then gives a distribution-free posterior lower
        bound for ``P(L_challenger < L_incumbent)``.
        """
        mean_gain = float(incumbent_mean) - float(challenger_mean)
        variance_upper = (
            np.sqrt(max(float(incumbent_variance), 0.0))
            + np.sqrt(max(float(challenger_variance), 0.0))
        ) ** 2
        if mean_gain <= max(float(min_mean_gain), 0.0):
            lower_bound = 0.0
        elif variance_upper <= 1e-18:
            lower_bound = 1.0
        else:
            lower_bound = mean_gain ** 2 / (
                mean_gain ** 2 + variance_upper)
        return {
            "posterior_mean_gain": float(mean_gain),
            "posterior_difference_variance_upper": float(variance_upper),
            "posterior_dominance_lower_bound": float(np.clip(
                lower_bound, 0.0, 1.0)),
        }

    def _posterior_dominance_decision_from_models(
        self,
        gpr_models,
        variance_model,
        pool,
        incumbent,
        *,
        task_ensemble=None,
        return_diagnostics=False,
    ):
        candidates = unique_candidates(pool)
        if incumbent is not None:
            incumbent = tuple(int(v) for v in incumbent)
            if incumbent not in candidates:
                candidates.append(incumbent)
        if not candidates:
            return None, np.inf, {
                "status": "empty_pool",
                "posterior_dominance_used": True,
            }
        components = self._terminal_bayes_risk_components(
            gpr_models,
            variance_model,
            candidates,
            task_ensemble=task_ensemble,
        )
        risks = np.asarray(components["risk"], dtype=float)
        risk_variance = np.maximum(np.asarray(
            components.get("risk_variance", np.zeros(len(candidates))),
            dtype=float,
        ), 0.0)
        if incumbent is None:
            initialization = str(
                self.config.posterior_dominance_initialization or "risk"
            ).strip().lower().replace("-", "_")
            if initialization not in {
                "risk", "certificate_lexicographic", "certified_only",
            }:
                raise ValueError(
                    "posterior dominance initialization must be risk, "
                    "certificate_lexicographic, or certified_only")
            certificate_margins = None
            certified_count = 0
            if initialization in {
                "certificate_lexicographic", "certified_only",
            }:
                certificate = self._terminal_certificate_components(
                    gpr_models,
                    variance_model,
                    candidates,
                    task_ensemble=task_ensemble,
                    observations=self.observations,
                )
                certificate_margins = np.asarray(
                    certificate["margin"], dtype=float)
                certified = np.flatnonzero(certificate_margins <= 0.0)
                certified_count = int(len(certified))
                if certified_count:
                    chosen_index = int(min(
                        certified, key=lambda index: (risks[index], index)))
                    initialization_status = "initialized_certified"
                elif initialization == "certified_only":
                    return None, np.inf, {
                        "status": "uninitialized_no_certificate",
                        "posterior_dominance_used": True,
                        "posterior_dominance_initialization": initialization,
                        "initial_certified_count": 0,
                        "selected_certificate_margin": None,
                        "incumbent_before": None,
                        "incumbent_after": None,
                        "selected_risk": None,
                        "selected_risk_variance": None,
                        "switch_accepted": False,
                        "terminal_fallback_required": True,
                        "target_oracle_used": False,
                    }
                else:
                    expected_violation = np.asarray(
                        components["expected_violation"], dtype=float)
                    order = np.lexsort((
                        np.arange(len(candidates), dtype=int),
                        risks,
                        expected_violation,
                        certificate_margins,
                    ))
                    chosen_index = int(order[0])
                    initialization_status = "initialized_safety_first"
            else:
                chosen_index = int(np.argmin(risks))
                initialization_status = "initialized"
            selected = tuple(int(v) for v in candidates[chosen_index])
            details = {
                "status": initialization_status,
                "posterior_dominance_used": True,
                "posterior_dominance_initialization": initialization,
                "initial_certified_count": int(certified_count),
                "selected_certificate_margin": (
                    None
                    if certificate_margins is None
                    else float(certificate_margins[chosen_index])
                ),
                "incumbent_before": None,
                "incumbent_after": list(selected),
                "selected_risk": float(risks[chosen_index]),
                "selected_risk_variance": float(
                    risk_variance[chosen_index]),
                "switch_accepted": True,
                "target_oracle_used": False,
            }
            return selected, float(risks[chosen_index]), details

        incumbent_index = candidates.index(incumbent)
        delta = float(np.clip(
            self.config.posterior_dominance_delta, 1e-12, 1.0 - 1e-12))
        threshold = 1.0 - delta
        comparisons = []
        accepted = []
        for index, candidate in enumerate(candidates):
            if index == incumbent_index:
                continue
            comparison = self._cantelli_dominance_lower_bound(
                risks[incumbent_index],
                risks[index],
                risk_variance[incumbent_index],
                risk_variance[index],
                min_mean_gain=(
                    self.config.posterior_dominance_min_mean_gain),
            )
            comparison.update({
                "candidate": list(map(int, candidate)),
                "candidate_risk": float(risks[index]),
                "candidate_risk_variance": float(risk_variance[index]),
                "accepted": bool(
                    comparison["posterior_dominance_lower_bound"]
                    >= threshold
                ),
            })
            comparisons.append(comparison)
            if comparison["accepted"]:
                accepted.append(index)
        chosen_index = (
            min(accepted, key=lambda index: (risks[index], index))
            if accepted else incumbent_index
        )
        selected = tuple(int(v) for v in candidates[chosen_index])
        selected_comparison = next((
            row for row in comparisons
            if tuple(row["candidate"]) == selected
        ), None)
        details = {
            "status": (
                "switched" if chosen_index != incumbent_index else "retained"
            ),
            "posterior_dominance_used": True,
            "method": "cantelli_covariance_free",
            "delta_switch": float(delta),
            "required_lower_bound": float(threshold),
            "incumbent_before": list(map(int, incumbent)),
            "incumbent_after": list(map(int, selected)),
            "incumbent_risk": float(risks[incumbent_index]),
            "incumbent_risk_variance": float(
                risk_variance[incumbent_index]),
            "selected_risk": float(risks[chosen_index]),
            "selected_risk_variance": float(risk_variance[chosen_index]),
            "switch_accepted": bool(chosen_index != incumbent_index),
            "accepted_challenger_count": int(len(accepted)),
            "selected_comparison": selected_comparison,
            "false_switch_posterior_bound": (
                float(delta) if chosen_index != incumbent_index else 0.0
            ),
            "target_oracle_used": False,
        }
        if return_diagnostics:
            details["comparisons"] = comparisons
        return selected, float(risks[chosen_index]), details

    def _initialize_posterior_dominance_incumbent(
        self, samples, *, stage=None, reason="initial_design"
    ):
        if not self._posterior_dominance_active():
            return {"status": "disabled"}
        selected, _, details = self._posterior_dominance_decision_from_models(
            self.gpr,
            self.variance_model,
            samples,
            None,
            task_ensemble=self.task_ensemble,
            return_diagnostics=True,
        )
        self._posterior_dominance_incumbent = selected
        record = {
            **details,
            "stage": int(self.config.n0 if stage is None else stage),
            "reason": str(reason),
        }
        if self._posterior_dominance_history:
            self._posterior_dominance_history.append(record)
        else:
            self._posterior_dominance_history = [record]
        return copy.deepcopy(record)

    def _update_posterior_dominance_incumbent(self, *, stage, reason):
        if not self._posterior_dominance_active():
            return {"status": "disabled"}
        evaluated = unique_candidates([x for x, _ in self.history])
        if self._posterior_dominance_incumbent is None:
            return self._initialize_posterior_dominance_incumbent(
                evaluated, stage=stage, reason=reason)
        selected, _, details = self._posterior_dominance_decision_from_models(
            self.gpr,
            self.variance_model,
            evaluated,
            self._posterior_dominance_incumbent,
            task_ensemble=self.task_ensemble,
            return_diagnostics=True,
        )
        self._posterior_dominance_incumbent = selected
        record = {
            **details,
            "stage": int(stage),
            "reason": str(reason),
        }
        self._posterior_dominance_history.append(record)
        return copy.deepcopy(record)

    @staticmethod
    def _terminal_frontier_indices(
        mu_obj,
        robust_margins,
        chosen,
        count,
        bayes_components=None,
        tcb_upper=None,
        tcb_count=0,
        chosen_label="bayes_action",
    ):
        """Select posterior-only terminal actions worth discriminating.

        The returned actions cover the current Bayes decision, the safest
        model action, the smallest predicted violation, and an uncertain
        low-risk action.  Target truth is deliberately absent from this API.
        """
        count = max(0, int(count))
        n = len(mu_obj)
        if count <= 0 or n == 0:
            return [], []
        mu_obj = np.asarray(mu_obj, dtype=float)
        robust_margins = np.asarray(robust_margins, dtype=float)
        indices = []
        labels = []

        def add(index, label):
            index = int(index)
            if 0 <= index < n and index not in indices:
                indices.append(index)
                labels.append(str(label))

        add(chosen, chosen_label)
        if tcb_upper is not None:
            tcb_upper = np.asarray(tcb_upper, dtype=float)
            if tcb_upper.shape != (n,):
                raise ValueError("TCB upper margins must match terminal pool")
            tcb_count = max(0, int(tcb_count))
            tcb_order = []
            if tcb_count > 0:
                tcb_order.append((int(np.argmin(tcb_upper)), "minimum_tcb_upper"))
            if tcb_count > 1:
                for index in np.argsort(np.abs(tcb_upper), kind="stable"):
                    tcb_order.append((
                        int(index),
                        "closest_tcb_boundary",
                    ))
            if tcb_count > 2:
                for index in np.argsort(tcb_upper, kind="stable"):
                    tcb_order.append((int(index), "tcb_frontier_fill"))
            added = 0
            for index, label in tcb_order:
                before = len(indices)
                if len(indices) < count:
                    add(index, label)
                if len(indices) > before:
                    added += 1
                if added >= tcb_count or len(indices) >= count:
                    break
        if len(indices) < count:
            add(np.argmin(robust_margins), "minimum_theory_margin")
        if bayes_components is not None and len(indices) < count:
            expected = np.asarray(
                bayes_components["expected_violation"], dtype=float)
            add(np.argmin(expected), "minimum_expected_violation")
        if bayes_components is not None and len(indices) < count:
            risk = np.asarray(bayes_components["risk"], dtype=float)
            disagreement = np.asarray(
                bayes_components["model_disagreement"], dtype=float)
            finite = np.isfinite(risk) & np.isfinite(disagreement)
            if np.any(finite):
                cutoff = float(np.quantile(risk[finite], 0.5))
                supported = finite & (risk <= cutoff)
                add(
                    np.argmax(np.where(supported, disagreement, -np.inf)),
                    "maximum_supported_disagreement",
                )
        if bayes_components is not None:
            order = np.argsort(
                np.asarray(bayes_components["risk"], dtype=float),
                kind="stable",
            )
        else:
            order = np.lexsort((mu_obj, robust_margins))
        for index in order:
            if len(indices) >= count:
                break
            add(index, "risk_frontier_fill")
        return indices[:count], labels[:count]

    def _terminal_certificate_components(
        self,
        gpr_models,
        variance_model,
        pool,
        *,
        task_ensemble=None,
        observations=None,
    ):
        """Canonical certificate used by all coherent decision layers.

        TCB-V2 is authoritative only in ``certified`` mode.  Shadow/frontier
        modes may nominate candidates, but they cannot silently change the
        terminal value while the final recommendation still uses the theory
        HVD certificate.
        """
        pool = [tuple(int(v) for v in x) for x in pool]
        if not pool:
            empty = np.asarray([], dtype=float)
            return {
                "objective": empty,
                "margin": empty,
                "source": "empty",
                "tcb_v2": None,
                "certification_head_authority": (
                    self._certification_head_authority()),
            }
        if task_ensemble is None:
            objective = np.asarray(
                gpr_models[0].posterior_mean_many(pool), dtype=float)
        else:
            objective = np.asarray(
                task_ensemble.mixture_moments_many(
                    0, pool, certification=False).mean,
                dtype=float,
            )

        tcb_v2 = None
        if self._tcb_v2_mode() == "certified":
            tcb_v2 = self._tcb_v2_margin_many(
                pool,
                task_ensemble=task_ensemble,
                observations=(
                    self.observations
                    if observations is None else observations
                ),
            )
            if tcb_v2 is None:
                raise RuntimeError(
                    "TCB-V2 certified decision contract requires a fitted "
                    "hierarchical boundary provider")
            margin = (
                np.asarray(tcb_v2["upper"], dtype=float)
                + self._recommendation_slack()
            )
            return {
                "objective": objective,
                "margin": margin,
                "source": "tcb_v2_hierarchical",
                "tcb_v2": tcb_v2,
                "certification_head_authority": (
                    self._certification_head_authority()),
            }

        authority = self._certification_head_authority()
        if task_ensemble is None:
            mu_con = np.asarray(
                gpr_models[1].posterior_mean_many(pool), dtype=float)
            epistemic = np.asarray(
                self._constraint_certification_epistemic_many(
                    gpr_models[1], pool), dtype=float)
            aleatoric = np.asarray(
                variance_model.predict_certification_variance_many(
                    1, pool, self.problem),
                dtype=float,
            )
            guard = self._pilot_constraint_guard()
        elif authority == "task_joint":
            robust = task_ensemble.robust_moments_many(
                1, pool, certification=True)
            mu_con = np.asarray(robust.mean_upper, dtype=float)
            epistemic = np.asarray(robust.epistemic_upper, dtype=float)
            aleatoric = np.asarray(robust.aleatoric_upper, dtype=float)
            guard = 0.0
        else:
            mu_con = np.asarray(
                gpr_models[1].posterior_mean_many(pool), dtype=float)
            epistemic = np.asarray(
                self._constraint_certification_epistemic_many(
                    gpr_models[1], pool), dtype=float)
            if authority == "split_gpr_task_hvd":
                robust = task_ensemble.robust_moments_many(
                    1, pool, certification=True)
                aleatoric = np.asarray(
                    robust.aleatoric_upper, dtype=float)
            else:
                aleatoric = np.asarray(
                    variance_model.predict_certification_variance_many(
                        1, pool, self.problem),
                    dtype=float,
                )
            guard = 0.0
        cert = conservative_chance_margin(
            mu_con + guard,
            epistemic,
            aleatoric,
            tau=self.problem.tau,
            alpha=self.problem.alpha,
            beta_g=self.config.beta_g,
            mode=self.config.certification_mode,
        )
        joint = None
        if (
            task_ensemble is not None
            and authority == "task_joint"
            and self._task_robust_certificate_mode() == "joint_tangent"
        ):
            joint = task_ensemble.robust_chance_margin_many(
                pool,
                beta_g=cert.beta_g,
                z_alpha=cert.z_alpha,
                tau=self.problem.tau,
                certification=True,
            )
            margin = np.asarray(joint.upper, dtype=float)
        else:
            margin = np.asarray(cert.margin, dtype=float)
        if cert.mode != "theory":
            sigma = np.sqrt(np.maximum(cert.aleatoric_var, 1e-12))
            nominal_floor = (
                self.config.recommendation_noise_floor_scale
                * float(getattr(self.problem, "sigma_level", 0.0))
            )
            margin = margin + np.maximum(
                self.config.recommendation_safety_z * sigma,
                nominal_floor,
            )
        margin = margin + self._recommendation_slack()
        return {
            "objective": objective,
            "margin": np.asarray(margin, dtype=float),
            "source": (
                self._certification_source(
                    task_ensemble_active=task_ensemble is not None)
            ),
            "certification_head_authority": authority,
            "tcb_v2": None,
            "mu_con": mu_con,
            "epistemic": np.asarray(cert.epistemic_var, dtype=float),
            "aleatoric": np.asarray(cert.aleatoric_var, dtype=float),
            "separable_margin": np.asarray(cert.margin, dtype=float),
            "joint_margin": (
                None
                if joint is None
                else np.asarray(joint.upper, dtype=float)
            ),
        }

    def _effective_exact_terminal_mode(self):
        if self._coherent_certificate_contract():
            return "tcb_certified_lexicographic"
        if self._posterior_dominance_active():
            return "bayes_risk_dominance"
        return str(
            self.config.exact_kg_terminal_mode or "hard_certified"
        ).lower()

    def _decision_backend_observed_terminal_active(self):
        """Whether online VOI and the final Bayes action share observed arms."""

        backend = str(
            self.config.decision_backend or "legacy"
        ).strip().lower().replace("-", "_")
        return bool(
            self.config.decision_recommend_observed_only
            and backend not in {"legacy", "legacy_kg", "exact_kg", "additive"}
        )

    def _terminal_action_pool(self, pool, observations=None):
        """Return the action universe used by both fantasy and final decisions.

        The promoted evaluate-or-replicate backend recommends only evaluated
        policies. A fantasy evaluation makes its policy eligible immediately;
        a replication leaves the eligible set unchanged. Keeping this logic in
        one helper prevents acquisition from optimizing over unobserved terminal
        actions that the final decision is forbidden to return.
        """

        pool = unique_candidates(pool)
        if not self._decision_backend_observed_terminal_active():
            return pool
        effective = self.observations if observations is None else observations
        if not effective:
            # This branch is only useful for model-level unit calls before n0.
            # Every sequential promoted run has a nonempty charged pilot.
            return pool
        observed = unique_candidates(effective.keys())
        observed_set = set(observed)
        ordered = [x for x in pool if x in observed_set]
        ordered.extend(x for x in observed if x not in set(ordered))
        return unique_candidates(ordered)

    def _terminal_value_contract_id(self):
        mode = self._effective_exact_terminal_mode().replace("-", "_")
        aleatoric = str(
            self.config.decision_aleatoric_mode or "certification_upper"
        ).strip().lower().replace("-", "_")
        ambiguity = str(
            self.config.decision_ambiguity_mode or "kl_robust"
        ).strip().lower().replace("-", "_")
        violation = str(
            self.config.decision_violation_loss_mode or "positive_part"
        ).strip().lower().replace("-", "_")
        universe = (
            "observed_actions"
            if self._decision_backend_observed_terminal_active()
            else "fixed_terminal_pool"
        )
        return f"{mode}:{aleatoric}:{ambiguity}:{violation}:{universe}:v1"

    def _terminal_value_from_models(
        self,
        gpr_models,
        variance_model,
        pool,
        task_ensemble=None,
        observations=None,
    ):
        """Terminal certified value used by the optional exact-update KG.

        Lower is better.  If the fixed terminal pool has no robust-feasible
        point, use the same normalized infeasibility penalty shape as
        `_solve_posterior_recommendation` so the value remains comparable.
        """
        pool = self._terminal_action_pool(pool, observations=observations)
        if len(pool) == 0:
            return 0.0
        terminal_mode = self._effective_exact_terminal_mode()
        if terminal_mode in (
            "tcb_certified_lexicographic",
            "tcb-certified-lexicographic",
            "certified_lexicographic",
        ):
            return self._terminal_certified_lexicographic_value(
                gpr_models,
                variance_model,
                pool,
                task_ensemble=task_ensemble,
                observations=(
                    self.observations if observations is None else observations
                ),
            )
        if terminal_mode in (
            "bayes_risk_dominance",
            "bayes-risk-dominance",
            "posterior_bayes_risk_dominance",
        ):
            effective_observations = (
                self.observations if observations is None else observations
            )
            dominance_pool = pool
            if (
                self.config.decision_recommend_observed_only
                and effective_observations
            ):
                dominance_pool = unique_candidates(
                    effective_observations.keys())
            incumbent = self._posterior_dominance_incumbent
            if incumbent is None:
                components = self._terminal_bayes_risk_components(
                    gpr_models,
                    variance_model,
                    dominance_pool,
                    task_ensemble=task_ensemble,
                )
                return float(np.min(components["risk"]))
            _, selected_value, _ = (
                self._posterior_dominance_decision_from_models(
                    gpr_models,
                    variance_model,
                    dominance_pool,
                    incumbent,
                    task_ensemble=task_ensemble,
                )
            )
            return float(selected_value)
        if terminal_mode in (
            "bayes_risk",
            "bayes-risk",
            "posterior_bayes_risk",
        ):
            components = self._terminal_bayes_risk_components(
                gpr_models,
                variance_model,
                pool,
                task_ensemble=task_ensemble,
                risk_penalty=(
                    self.config.decision_risk_penalty
                    if self._decision_backend_observed_terminal_active()
                    else None
                ),
            )
            return float(np.min(components["risk"]))
        if terminal_mode not in (
            "hard_certified",
            "hard-certified",
            "certified",
            "legacy",
        ):
            raise ValueError(
                f"unknown exact KG terminal mode {terminal_mode!r}")
        certificate = self._terminal_certificate_components(
            gpr_models,
            variance_model,
            pool,
            task_ensemble=task_ensemble,
            observations=observations,
        )
        mu_obj = certificate["objective"]
        robust_margins = certificate["margin"]
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
        infeasible_penalty = float(
            self.config.recommendation_infeasible_penalty)
        if (
            str(self.config.recommendation_infeasible_strategy).lower()
            in ("task_adaptive", "task-adaptive", "sensitivity_posterior")
            and task_ensemble is not None
        ):
            infeasible_penalty = task_ensemble.adaptive_infeasible_penalty(
                fallback=infeasible_penalty)
        penalized = (
            scaled_obj
            + infeasible_penalty * scaled_margin
        )
        return float(np.min(penalized))

    def _finalist_terminal_value_mode(self):
        if self._coherent_certificate_contract():
            return "certified_lexicographic"
        mode = str(
            self.config.finalist_terminal_value_mode or "model_default"
        ).lower()
        aliases = {
            "default": "model_default",
            "legacy": "model_default",
            "lexicographic": "certified_lexicographic",
            "tcb_v2": "certified_lexicographic",
            "tcb_v2_lexicographic": "certified_lexicographic",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"model_default", "certified_lexicographic"}:
            raise ValueError(
                f"unknown finalist terminal value mode {mode!r}")
        return mode

    def _terminal_certified_lexicographic_value(
        self,
        gpr_models,
        variance_model,
        pool,
        *,
        task_ensemble=None,
        observations=None,
    ):
        """Return `(uncertified, positive margin, objective)`.

        The tuple is deliberately not scalarized.  Bellman comparisons first
        minimize the posterior probability of ending without a certificate,
        then the expected positive certificate distance, and only then the
        objective.  This is the exact terminal decision contract used by the
        TCB-V2 finalist policy.
        """
        _, value, _ = self._terminal_certified_lexicographic_decision(
            gpr_models,
            variance_model,
            pool,
            task_ensemble=task_ensemble,
            observations=observations,
        )
        return value

    def _terminal_certified_lexicographic_decision(
        self,
        gpr_models,
        variance_model,
        pool,
        *,
        task_ensemble=None,
        observations=None,
    ):
        """Return the Bayes action and its robust lexicographic loss."""
        if len(pool) == 0:
            value = np.asarray([1.0, np.inf, np.inf], dtype=float)
            return None, value, {
                "status": "empty_pool",
                "certified_count": 0,
            }
        pool = [tuple(int(v) for v in x) for x in pool]
        certificate = self._terminal_certificate_components(
            gpr_models,
            variance_model,
            pool,
            task_ensemble=task_ensemble,
            observations=observations,
        )
        mu_obj = np.asarray(certificate["objective"], dtype=float)
        robust_margins = np.asarray(certificate["margin"], dtype=float)
        feasible = robust_margins <= 0.0
        if np.any(feasible):
            local = int(np.argmin(np.where(feasible, mu_obj, np.inf)))
            value = np.asarray(
                [0.0, 0.0, float(mu_obj[local])], dtype=float)
            status = "certified_objective"
        else:
            min_margin = float(np.min(robust_margins))
            near_min = robust_margins <= min_margin + 1e-12
            local = int(np.argmin(np.where(near_min, mu_obj, np.inf)))
            value = np.asarray([
                1.0,
                max(min_margin, 0.0),
                float(mu_obj[local]),
            ], dtype=float)
            status = "minimum_robust_margin"
        selected = tuple(int(v) for v in pool[local])
        return selected, value, {
            "status": status,
            "certified_count": int(np.sum(feasible)),
            "selected_margin": float(robust_margins[local]),
            "selected_objective": float(mu_obj[local]),
            "certificate_source": certificate.get("source"),
            "certification_head_authority": certificate.get(
                "certification_head_authority"),
            "target_oracle_used": False,
        }

    @staticmethod
    def _terminal_value_index(values):
        """Index of the minimum scalar or lexicographic terminal value."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return int(np.argmin(values))
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("terminal action values must be 1D or 2D")
        keys = tuple(
            values[:, index]
            for index in reversed(range(values.shape[1]))
        )
        return int(np.lexsort(keys)[0])

    @classmethod
    def _terminal_best_value(cls, values):
        values = np.asarray(values, dtype=float)
        selected = cls._terminal_value_index(values)
        if values.ndim == 1:
            return float(values[selected])
        return np.asarray(values[selected], dtype=float)

    @staticmethod
    def _terminal_diagnostic_gain(current, future):
        """First decisive component gain, for legacy scalar diagnostics only."""
        delta = np.asarray(current, dtype=float) - np.asarray(
            future, dtype=float)
        if delta.ndim == 0:
            return float(delta)
        for value in delta.reshape(-1):
            if abs(float(value)) > 1e-12:
                return float(value)
        return 0.0

    def _effective_exact_kg_mc_samples(self):
        decision_backend = str(
            self.config.decision_backend or "legacy"
        ).strip().lower().replace("-", "_")
        if decision_backend in {
            "sobol_exact_joint_voi", "exact_joint_voi_sobol",
        }:
            return max(1, int(self.config.exact_kg_mc_samples))
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

    def _policy_improvement_mode(self):
        mode = str(
            self.config.policy_improvement_mode or "off"
        ).strip().lower().replace("-", "_")
        aliases = {
            "none": "off",
            "disabled": "off",
            "superset": "action_superset",
            "rollout": "guarded_rollout",
            "both": "joint",
        }
        mode = aliases.get(mode, mode)
        if mode not in {
            "off", "action_superset", "guarded_rollout", "joint",
        }:
            raise ValueError(f"unknown policy improvement mode {mode!r}")
        return mode

    def _guarded_one_step_policy_improvement(
        self,
        candidates,
        exact_scores,
        backend_score,
    ):
        """Select an action-superset challenger with a V51 fallback."""

        scores = np.asarray(exact_scores, dtype=float).reshape(-1)
        active = backend_score.get("evaluate_or_replicate_active_indices")
        baseline = backend_score.get("evaluate_or_replicate_baseline_indices")
        if active is None:
            active = np.flatnonzero(np.isfinite(scores))
        else:
            active = np.asarray(active, dtype=int).reshape(-1)
        if baseline is None:
            baseline = active
        else:
            baseline = np.asarray(baseline, dtype=int).reshape(-1)
        baseline = np.asarray([
            index for index in baseline if index in set(active.tolist())
        ], dtype=int)
        if len(active) == 0 or len(baseline) == 0:
            raise RuntimeError(
                "policy improvement requires nonempty active and baseline sets")

        raw_scores = np.asarray(getattr(
            self, "_last_exact_kg_raw_scores", scores), dtype=float)
        finite_raw = np.where(np.isfinite(raw_scores), raw_scores, -np.inf)
        baseline_index = int(baseline[
            int(np.argmax(finite_raw[baseline]))
        ])
        union_index = int(active[int(np.argmax(finite_raw[active]))])
        estimated_advantage = float(
            finite_raw[union_index] - finite_raw[baseline_index])
        eta = max(
            float(self.config.policy_improvement_mc_error_bound), 0.0)
        threshold = 2.0 * eta
        mode = self._policy_improvement_mode()
        superset_enabled = mode in {"action_superset", "joint"}
        switched = bool(
            superset_enabled
            and union_index != baseline_index
            and estimated_advantage > threshold
        )
        selected = union_index if switched else baseline_index
        return selected, {
            "status": (
                "superset_switched" if switched
                else (
                    "baseline_best_in_union"
                    if union_index == baseline_index
                    else "baseline_mc_guard"
                )
            ),
            "mode": mode,
            "baseline_index": baseline_index,
            "union_index": union_index,
            "selected_index": int(selected),
            "baseline_action": list(map(int, candidates[baseline_index])),
            "union_action": list(map(int, candidates[union_index])),
            "estimated_advantage": estimated_advantage,
            "mc_uniform_error_bound": eta,
            "switch_threshold": threshold,
            "switched": switched,
            "baseline_action_count": int(len(baseline)),
            "union_action_count": int(len(active)),
            "conditional_noninferiority_contract": (
                "uniform_mc_error_implies_exact_one_step_noninferiority"
            ),
            "target_oracle_used": False,
        }

    def _guarded_rollout_policy_improvement(
        self,
        candidates,
        terminal_pool,
        exact_scores,
        active_indices,
        fallback_index,
        *,
        stage,
    ):
        """Apply finite-horizon rollout only beyond a uniform-error guard."""

        mode = self._policy_improvement_mode()
        depth = max(1, int(self.config.policy_improvement_rollout_depth))
        if mode not in {"guarded_rollout", "joint"} or depth <= 1:
            return int(fallback_index), {
                "status": "disabled",
                "mode": mode,
                "depth": depth,
                "switched": False,
                "target_oracle_used": False,
            }
        active = np.asarray(active_indices, dtype=int).reshape(-1)
        raw_scores = np.asarray(getattr(
            self, "_last_exact_kg_raw_scores", exact_scores), dtype=float)
        finite_raw = np.where(np.isfinite(raw_scores), raw_scores, -np.inf)
        ordered = sorted(
            (int(index) for index in active),
            key=lambda index: (-float(finite_raw[index]), index),
        )
        max_arms = max(1, int(
            self.config.policy_improvement_rollout_max_arms))
        arm_indices = [int(fallback_index)]
        arm_indices.extend(
            index for index in ordered if index != int(fallback_index))
        arm_indices = list(dict.fromkeys(arm_indices))[:max_arms]
        if len(arm_indices) <= 1:
            return int(fallback_index), {
                "status": "single_arm",
                "mode": mode,
                "depth": depth,
                "switched": False,
                "target_oracle_used": False,
            }
        arms = [candidates[index] for index in arm_indices]
        rollout_action, rollout = self._terminal_replication_kg_candidate(
            arms,
            terminal_pool,
            depth=depth,
            stage=stage,
        )
        expected = np.asarray(
            rollout["terminal_kg_expected_values"], dtype=float)
        if expected.ndim != 1:
            return int(fallback_index), {
                **rollout,
                "status": "vector_terminal_fallback",
                "mode": mode,
                "switched": False,
                "target_oracle_used": False,
            }
        rollout_local = int(rollout["terminal_kg_selected_index"])
        fallback_local = 0
        estimated_advantage = float(
            expected[fallback_local] - expected[rollout_local])
        eta = max(float(
            self.config.policy_improvement_rollout_mc_error_bound), 0.0)
        threshold = 2.0 * eta
        switched = bool(
            rollout_local != fallback_local
            and estimated_advantage > threshold
        )
        selected_index = (
            arm_indices[rollout_local] if switched else int(fallback_index))
        return int(selected_index), {
            **rollout,
            "status": (
                "rollout_switched" if switched
                else (
                    "fallback_best_in_rollout"
                    if rollout_local == fallback_local
                    else "fallback_mc_guard"
                )
            ),
            "mode": mode,
            "fallback_candidate_index": int(fallback_index),
            "rollout_candidate_index": int(arm_indices[rollout_local]),
            "selected_candidate_index": int(selected_index),
            "estimated_advantage_over_fallback": estimated_advantage,
            "mc_uniform_error_bound": eta,
            "switch_threshold": threshold,
            "switched": switched,
            "conditional_noninferiority_contract": (
                "uniform_rollout_error_implies_posterior_value_noninferiority"
            ),
            "target_oracle_used": False,
        }

    def _terminal_rollout_root_state(self):
        observations = {
            tuple(int(v) for v in key): [
                np.asarray(value, dtype=float).copy() for value in values
            ]
            for key, values in self.observations.items()
        }
        if self.task_ensemble is not None:
            return {
                "task_ensemble": self.task_ensemble,
                "gpr_models": None,
                "variance_model": None,
                "observations": observations,
            }
        return {
            "task_ensemble": None,
            "gpr_models": self.gpr,
            "variance_model": self.variance_model,
            "observations": observations,
        }

    def _terminal_rollout_value(self, state, terminal_pool):
        ensemble = state["task_ensemble"]
        if self._finalist_terminal_value_mode() == "certified_lexicographic":
            return self._terminal_certified_lexicographic_value(
                state["gpr_models"],
                state["variance_model"],
                terminal_pool,
                task_ensemble=ensemble,
                observations=state["observations"],
            )
        if ensemble is not None:
            return self._terminal_value_from_models(
                None,
                None,
                terminal_pool,
                task_ensemble=ensemble,
                observations=state["observations"],
            )
        return self._terminal_value_from_models(
            state["gpr_models"],
            state["variance_model"],
            terminal_pool,
            observations=state["observations"],
        )

    def _terminal_rollout_samples(self, depth, node_code):
        if (
            self._policy_improvement_mode()
            in {"guarded_rollout", "joint"}
            and int(self.config.policy_improvement_rollout_depth) > 1
        ):
            mc = int(self.config.policy_improvement_rollout_mc_samples)
        else:
            mc = int(self.config.finalist_terminal_mc_samples)
        if mc <= 0:
            mc = max(1, self._effective_exact_kg_mc_samples())
        seed_sequence = np.random.SeedSequence([
            int(self.config.seed) & 0xFFFFFFFF,
            int(len(self.history)) & 0xFFFFFFFF,
            int(depth) & 0xFFFFFFFF,
            int(node_code) & 0xFFFFFFFF,
        ])
        rng = np.random.default_rng(seed_sequence)
        mode = str(self.config.exact_kg_sampling_mode or "iid").lower()
        if mode in ("antithetic", "paired", "antithetic_pairs"):
            pairs = mc // 2
            base = rng.standard_normal((pairs, 2))
            rows = []
            uniforms = []
            for z_vec, expert_uniform in zip(base, rng.random(pairs)):
                rows.extend([z_vec, -z_vec])
                uniforms.extend([expert_uniform, 1.0 - expert_uniform])
            if mc % 2:
                rows.append(np.zeros(2, dtype=float))
                uniforms.append(0.5)
            z_rows = np.asarray(rows, dtype=float).reshape(mc, 2)
            expert_uniforms = np.asarray(uniforms, dtype=float)
        else:
            # A uniform expert selector remains exact for finite mixtures.
            # Stratification is deliberately not reused recursively because
            # posterior expert weights change after every fantasy update.
            z_rows = rng.standard_normal((mc, 2))
            expert_uniforms = rng.random(mc)
        weights = np.full(mc, 1.0 / float(mc), dtype=float)
        return z_rows, expert_uniforms, weights

    def _terminal_rollout_update_state(
        self,
        state,
        x,
        z_vec,
        expert_uniform,
    ):
        key = tuple(int(v) for v in np.asarray(x, dtype=int))
        existing = list(state["observations"].get(key, []))
        ensemble = state["task_ensemble"]
        if ensemble is not None:
            y, _ = ensemble.predictive_sample(
                np.asarray(key, dtype=int), z_vec, expert_uniform)
            ensemble_clone = ensemble.clone(
                gpr_cloner=self._clone_gpr_for_exact_kg,
                variance_cloner=lambda model: (
                    self._clone_variance_model_for_exact_kg(model)
                ),
            )
            ensemble_clone.update(
                np.asarray(key, dtype=int),
                y,
                existing_observations=existing,
                tau=self.problem.tau,
            )
            next_observations = dict(state["observations"])
            next_observations[key] = existing + [
                np.asarray(y, dtype=float).copy()]
            return {
                "task_ensemble": ensemble_clone,
                "gpr_models": None,
                "variance_model": None,
                "observations": next_observations,
            }

        gpr_models = state["gpr_models"]
        variance_model = state["variance_model"]
        mu_before = [
            float(gpr_models[i].posterior_mean(key)) for i in range(2)
        ]
        sigma2_before = [
            float(variance_model.predict_variance(i, key, self.problem))
            for i in range(2)
        ]
        y = np.asarray([
            mu_before[i] + np.sqrt(max(
                sigma2_before[i] + gpr_models[i].posterior_var(key),
                1e-12,
            )) * float(z_vec[i])
            for i in range(2)
        ], dtype=float)
        gpr_clone = [
            self._clone_gpr_for_exact_kg(model) for model in gpr_models
        ]
        variance_clone = self._clone_variance_model_for_exact_kg(
            variance_model)
        for output_index in range(2):
            gpr_clone[output_index].update(
                key, y[output_index], sigma2_before[output_index])
        for output_index in range(2):
            replicate_values = [
                float(np.asarray(value, dtype=float)[output_index])
                for value in existing
            ] + [float(y[output_index])]
            replicate_variance = (
                float(np.var(replicate_values, ddof=1))
                if len(replicate_values) >= 2
                else None
            )
            variance_clone.update(
                output_index,
                key,
                float(y[output_index]),
                mu_before[output_index],
                gpr_clone[output_index],
                self.problem,
                replicate_variance=replicate_variance,
                replicate_count=len(replicate_values),
            )
        next_observations = dict(state["observations"])
        next_observations[key] = existing + [y.copy()]
        return {
            "task_ensemble": None,
            "gpr_models": gpr_clone,
            "variance_model": variance_clone,
            "observations": next_observations,
        }

    @staticmethod
    def _terminal_rollout_child_code(
        node_code,
        action_index,
        sample_index,
        depth,
    ):
        return (
            int(node_code) * 1009
            + (int(action_index) + 1) * 97
            + (int(sample_index) + 1) * 17
            + int(depth) * 31
        ) & 0xFFFFFFFF

    def _terminal_rollout_expected_value_for_action(
        self,
        state,
        action,
        arms,
        terminal_pool,
        depth,
        node_code,
        action_index,
        common_z,
        common_uniform,
        sample_weights,
    ):
        branch_values = []
        for sample_index, (z_vec, expert_uniform) in enumerate(zip(
            common_z, common_uniform
        )):
            next_state = self._terminal_rollout_update_state(
                state, action, z_vec, expert_uniform)
            if int(depth) <= 1:
                value = self._terminal_rollout_value(
                    next_state, terminal_pool)
            else:
                child_code = self._terminal_rollout_child_code(
                    node_code,
                    action_index,
                    sample_index,
                    depth,
                )
                child_values = self._terminal_rollout_action_values(
                    next_state,
                    arms,
                    terminal_pool,
                    depth=int(depth) - 1,
                    node_code=child_code,
                )
                value = self._terminal_best_value(child_values)
            branch_values.append(value)
        expected = np.tensordot(
            np.asarray(sample_weights, dtype=float),
            np.asarray(branch_values, dtype=float),
            axes=(0, 0),
        )
        if np.asarray(expected).ndim == 0:
            return float(expected)
        return np.asarray(expected, dtype=float)

    def _terminal_rollout_action_values(
        self,
        state,
        arms,
        terminal_pool,
        *,
        depth,
        node_code,
    ):
        common_z, common_uniform, sample_weights = (
            self._terminal_rollout_samples(depth, node_code))
        return np.asarray([
            self._terminal_rollout_expected_value_for_action(
                state,
                action,
                arms,
                terminal_pool,
                depth,
                node_code,
                action_index,
                common_z,
                common_uniform,
                sample_weights,
            )
            for action_index, action in enumerate(arms)
        ], dtype=float)

    def _terminal_rollout_depth3_prefix_value(
        self,
        state,
        arms,
        terminal_pool,
        node_code,
        common_z,
        common_uniform,
        root_action_index,
        root_sample_index,
        second_action_index,
    ):
        """Return one second-stage action value in the depth-three tree."""
        state_after_root = self._terminal_rollout_update_state(
            state,
            arms[int(root_action_index)],
            common_z[int(root_sample_index)],
            common_uniform[int(root_sample_index)],
        )
        second_node_code = self._terminal_rollout_child_code(
            node_code,
            root_action_index,
            root_sample_index,
            3,
        )
        second_z, second_uniform, second_weights = (
            self._terminal_rollout_samples(2, second_node_code))
        branch_values = []
        for second_sample_index, (
            z_vec,
            expert_uniform,
        ) in enumerate(zip(second_z, second_uniform)):
            state_after_second = self._terminal_rollout_update_state(
                state_after_root,
                arms[int(second_action_index)],
                z_vec,
                expert_uniform,
            )
            third_node_code = self._terminal_rollout_child_code(
                second_node_code,
                second_action_index,
                second_sample_index,
                2,
            )
            third_values = self._terminal_rollout_action_values(
                state_after_second,
                arms,
                terminal_pool,
                depth=1,
                node_code=third_node_code,
            )
            branch_values.append(self._terminal_best_value(third_values))
        expected = np.tensordot(
            np.asarray(second_weights, dtype=float),
            np.asarray(branch_values, dtype=float),
            axes=(0, 0),
        )
        if np.asarray(expected).ndim == 0:
            return float(expected)
        return np.asarray(expected, dtype=float)

    def _terminal_rollout_depth3_parallel_values(
        self,
        state,
        arms,
        terminal_pool,
        node_code,
        common_z,
        common_uniform,
        sample_weights,
        jobs,
        backend,
    ):
        """Flatten depth-three prefixes so all requested cores do real work."""
        payloads = [
            (root_action_index, root_sample_index, second_action_index)
            for root_action_index in range(len(arms))
            for root_sample_index in range(len(common_z))
            for second_action_index in range(len(arms))
        ]
        jobs = min(max(1, int(jobs)), len(payloads))
        vector_value = (
            self._finalist_terminal_value_mode()
            == "certified_lexicographic"
        )
        second_shape = (len(arms), len(common_z), len(arms))
        if vector_value:
            second_shape = second_shape + (3,)
        second_values = np.empty(second_shape, dtype=float)

        def evaluate(payload):
            return self._terminal_rollout_depth3_prefix_value(
                state,
                arms,
                terminal_pool,
                node_code,
                common_z,
                common_uniform,
                *payload,
            )

        global _FORK_TERMINAL_DEPTH3_CONTEXT
        if backend in ("process_fork", "fork", "process"):
            if "fork" not in multiprocessing.get_all_start_methods():
                raise RuntimeError(
                    "terminal rollout process backend requires Linux fork")
            _FORK_TERMINAL_DEPTH3_CONTEXT = (
                self,
                state,
                arms,
                terminal_pool,
                node_code,
                common_z,
                common_uniform,
            )
            executor = ProcessPoolExecutor(
                max_workers=jobs,
                mp_context=multiprocessing.get_context("fork"),
            )
            submit = lambda pool, payload: pool.submit(
                _fork_terminal_depth3_prefix, payload)
        elif backend in ("thread", "threads"):
            executor = ThreadPoolExecutor(max_workers=jobs)
            submit = lambda pool, payload: pool.submit(evaluate, payload)
        else:
            raise ValueError(
                f"unknown terminal rollout backend {backend!r}")
        try:
            with executor as pool:
                futures = {
                    submit(pool, payload): payload for payload in payloads
                }
                for future in as_completed(futures):
                    payload = futures[future]
                    second_values[payload] = future.result()
        finally:
            if backend in ("process_fork", "fork", "process"):
                _FORK_TERMINAL_DEPTH3_CONTEXT = None

        if vector_value:
            root_branch_values = np.empty(
                (len(arms), len(common_z), 3), dtype=float)
            for root_index in range(len(arms)):
                for sample_index in range(len(common_z)):
                    root_branch_values[root_index, sample_index] = (
                        self._terminal_best_value(
                            second_values[root_index, sample_index])
                    )
            expected = np.tensordot(
                root_branch_values,
                np.asarray(sample_weights, dtype=float),
                axes=(1, 0),
            )
        else:
            root_branch_values = np.min(second_values, axis=2)
            expected = root_branch_values @ np.asarray(
                sample_weights, dtype=float)
        return np.asarray(expected, dtype=float), int(jobs)

    def _terminal_replication_kg_candidate(
        self,
        arms,
        terminal_pool,
        *,
        depth,
        stage,
    ):
        """Solve the finite suffix Bellman problem on a frozen arm set."""
        arms = [tuple(int(v) for v in x) for x in arms]
        terminal_pool = [tuple(int(v) for v in x) for x in terminal_pool]
        state = self._terminal_rollout_root_state()
        current_value = self._terminal_rollout_value(state, terminal_pool)
        depth = max(1, int(depth))
        node_code = (int(stage) + 1) * 104729
        common_z, common_uniform, sample_weights = (
            self._terminal_rollout_samples(depth, node_code))
        vector_value = np.asarray(current_value).ndim > 0
        expected = np.empty(
            (len(arms), 3) if vector_value else len(arms),
            dtype=float,
        )
        requested_jobs = max(1, int(self.config.exact_kg_jobs))
        jobs = min(
            requested_jobs,
            max(1, len(arms)),
        )
        backend = str(
            self.config.exact_kg_parallel_backend or "thread").lower()
        started = time.perf_counter()

        def evaluate(index):
            return self._terminal_rollout_expected_value_for_action(
                state,
                arms[index],
                arms,
                terminal_pool,
                depth,
                node_code,
                index,
                common_z,
                common_uniform,
                sample_weights,
            )

        if int(depth) == 3 and requested_jobs > len(arms):
            expected, jobs = self._terminal_rollout_depth3_parallel_values(
                state,
                arms,
                terminal_pool,
                node_code,
                common_z,
                common_uniform,
                sample_weights,
                requested_jobs,
                backend,
            )
        elif jobs <= 1:
            for index in range(len(arms)):
                expected[index] = evaluate(index)
        else:
            global _FORK_TERMINAL_ROLLOUT_CONTEXT
            if backend in ("process_fork", "fork", "process"):
                if "fork" not in multiprocessing.get_all_start_methods():
                    raise RuntimeError(
                        "terminal rollout process backend requires Linux fork")
                _FORK_TERMINAL_ROLLOUT_CONTEXT = (
                    self,
                    state,
                    arms,
                    terminal_pool,
                    depth,
                    node_code,
                    common_z,
                    common_uniform,
                    sample_weights,
                )
                executor = ProcessPoolExecutor(
                    max_workers=jobs,
                    mp_context=multiprocessing.get_context("fork"),
                )
                submit = lambda pool, index: pool.submit(
                    _fork_terminal_rollout_action, index)
            elif backend in ("thread", "threads"):
                executor = ThreadPoolExecutor(max_workers=jobs)
                submit = lambda pool, index: pool.submit(evaluate, index)
            else:
                raise ValueError(
                    f"unknown terminal rollout backend {backend!r}")
            try:
                with executor as pool:
                    futures = {
                        submit(pool, index): index
                        for index in range(len(arms))
                    }
                    for future in as_completed(futures):
                        expected[futures[future]] = future.result()
            finally:
                if backend in ("process_fork", "fork", "process"):
                    _FORK_TERMINAL_ROLLOUT_CONTEXT = None

        selected_index = self._terminal_value_index(expected)
        if vector_value:
            component_gain = (
                np.asarray(current_value, dtype=float)[None, :] - expected)
            raw_gain = np.asarray([
                self._terminal_diagnostic_gain(current_value, future)
                for future in expected
            ], dtype=float)
        else:
            component_gain = None
            raw_gain = float(current_value) - expected
        clipped_gain = np.maximum(raw_gain, 0.0)
        serialized_current = (
            np.asarray(current_value, dtype=float).tolist()
            if vector_value else float(current_value)
        )
        return arms[selected_index], {
            "terminal_kg_depth": int(depth),
            "terminal_kg_mc_samples": int(len(common_z)),
            "terminal_kg_jobs": int(jobs),
            "terminal_kg_backend": (
                backend if jobs > 1 else "serial"),
            "terminal_kg_current_value": serialized_current,
            "terminal_kg_expected_values": expected.tolist(),
            "terminal_kg_raw_gains": raw_gain.tolist(),
            "terminal_kg_clipped_gains": clipped_gain.tolist(),
            "terminal_kg_selected_index": selected_index,
            "terminal_kg_selected_gain": float(raw_gain[selected_index]),
            "terminal_kg_arm_count": int(len(arms)),
            "terminal_kg_arms": [list(map(int, x)) for x in arms],
            "terminal_kg_elapsed_sec": float(
                time.perf_counter() - started),
            "terminal_kg_frozen_universe": bool(
                self.config.finalist_replication_fixed_universe),
            "terminal_kg_target_oracle_used": False,
            "terminal_kg_value_mode": self._finalist_terminal_value_mode(),
            "terminal_kg_component_names": (
                ["uncertified_probability", "positive_upper_margin", "objective"]
                if vector_value else None
            ),
            "terminal_kg_component_gains": (
                None if component_gain is None else component_gain.tolist()
            ),
            "terminal_kg_tcb_v2_mode": self._tcb_v2_mode(),
        }

    def _exact_primary_fantasy_update(
        self,
        x_arr,
        y,
        existing_observations,
        mu_before,
        sigma2_before,
        epistemic_before,
    ):
        """Clone and update the primary GPR/HVD posterior for one fantasy."""

        gpr_clone = [
            self._clone_gpr_for_exact_kg(model)
            for model in self.gpr
        ]
        variance_clone = self._clone_variance_model_for_exact_kg()
        for output_index in range(2):
            gpr_clone[output_index].update(
                x_arr, y[output_index], sigma2_before[output_index])
        self._configure_hvd_source_task_posterior(
            variance_clone, gpr_clone)
        for output_index in range(2):
            replicate_values = [
                float(np.asarray(observed, dtype=float)[output_index])
                for observed in existing_observations
            ] + [float(y[output_index])]
            replicate_variance = (
                float(np.var(replicate_values, ddof=1))
                if len(replicate_values) >= 2
                else None
            )
            variance_clone.update(
                output_index,
                x_arr,
                y[output_index],
                mu_before[output_index],
                gpr_clone[output_index],
                self.problem,
                epistemic_var=epistemic_before[output_index],
                replicate_variance=replicate_variance,
                replicate_count=len(replicate_values),
            )
        return gpr_clone, variance_clone

    def _exact_posterior_update_score_one(
        self,
        x,
        common_z,
        terminal_pool,
        current_value,
        common_expert_uniform=None,
        common_sample_weights=None,
        return_diagnostics=False,
    ):
        x_arr = np.asarray(x, dtype=int)
        sample_weights = (
            np.asarray(common_sample_weights, dtype=float).reshape(-1)
            if common_sample_weights is not None
            else np.ones(len(common_z), dtype=float)
        )
        if len(sample_weights) != len(common_z):
            raise ValueError("exact KG sample weights must match common samples")
        weight_total = float(np.sum(sample_weights))
        if (
            weight_total <= 0.0
            or not np.all(np.isfinite(sample_weights))
            or np.any(sample_weights < 0.0)
        ):
            raise ValueError(
                "exact KG sample weights must be finite and nonnegative")
        sample_weights = sample_weights / weight_total
        existing_observations = list(self.observations.get(
            tuple(int(v) for v in x_arr), []))
        mu_before = [self.gpr[i].posterior_mean(x_arr) for i in range(2)]
        sigma2_before = [
            self.variance_model.predict_variance(i, x_arr, self.problem)
            for i in range(2)
        ]
        epistemic_before = [
            self.gpr[i].posterior_var(x_arr) for i in range(2)
        ]
        if self.task_ensemble is not None:
            uniforms = (
                np.asarray(common_expert_uniform, dtype=float)
                if common_expert_uniform is not None
                else np.full(len(common_z), 0.5, dtype=float)
            )
            future_values = []
            entropy_gains = []
            weight_movements = []
            timing = {
                "clone": 0.0,
                "predictive_sample": 0.0,
                "joint_update": 0.0,
                "robust_terminal": 0.0,
            }
            entropy_before = self.task_ensemble.inference_entropy()
            weights_before = self.task_ensemble.inference_weights()
            for z_vec, expert_uniform in zip(common_z, uniforms):
                started = time.perf_counter()
                ensemble_clone = self.task_ensemble.clone(
                    gpr_cloner=self._clone_gpr_for_exact_kg,
                    variance_cloner=lambda model: (
                        self._clone_variance_model_for_exact_kg(model)
                    ),
                )
                timing["clone"] += time.perf_counter() - started
                started = time.perf_counter()
                y, _ = self.task_ensemble.predictive_sample(
                    x_arr, z_vec, expert_uniform)
                timing["predictive_sample"] += time.perf_counter() - started
                started = time.perf_counter()
                ensemble_clone.update(
                    x_arr,
                    y,
                    existing_observations=existing_observations,
                    tau=self.problem.tau,
                )
                gpr_clone, variance_clone = (
                    self._exact_primary_fantasy_update(
                        x_arr,
                        y,
                        existing_observations,
                        mu_before,
                        sigma2_before,
                        epistemic_before,
                    )
                )
                future_observations = {
                    tuple(int(v) for v in key): [
                        np.asarray(value, dtype=float).copy()
                        for value in values
                    ]
                    for key, values in self.observations.items()
                }
                future_observations.setdefault(
                    tuple(int(v) for v in x_arr), []).append(
                        np.asarray(y, dtype=float))
                timing["joint_update"] += time.perf_counter() - started
                started = time.perf_counter()
                future_value = self._terminal_value_from_models(
                    gpr_clone,
                    variance_clone,
                    terminal_pool,
                    task_ensemble=ensemble_clone,
                    observations=future_observations,
                )
                timing["robust_terminal"] += time.perf_counter() - started
                future_values.append(future_value)
                entropy_gains.append(
                    entropy_before - ensemble_clone.inference_entropy())
                weight_movements.append(float(np.sum(np.abs(
                    ensemble_clone.inference_weights()
                    - weights_before
                ))))
            expected_value = np.tensordot(
                sample_weights,
                np.asarray(future_values, dtype=float),
                axes=(0, 0),
            )
            component_gain = (
                np.asarray(current_value, dtype=float)
                - np.asarray(expected_value, dtype=float)
            )
            raw_score = self._terminal_diagnostic_gain(
                current_value, expected_value)
            result = {
                "score": (
                    max(raw_score, 0.0)
                    if self.config.exact_kg_clip_negative
                    else raw_score
                ),
                "raw_score": raw_score,
                "expected_terminal_value": (
                    np.asarray(expected_value, dtype=float).tolist()
                    if np.asarray(expected_value).ndim > 0
                    else float(expected_value)
                ),
                "component_gain": (
                    np.asarray(component_gain, dtype=float).tolist()
                    if np.asarray(component_gain).ndim > 0
                    else float(component_gain)
                ),
                "task_entropy_gain": float(np.dot(
                    sample_weights, entropy_gains)),
                "task_weight_movement": float(np.dot(
                    sample_weights, weight_movements)),
                **{
                    f"time_{name}": float(value / max(len(common_z), 1))
                    for name, value in timing.items()
                },
            }
            return result if return_diagnostics else result["score"]

        pred_sd = [
            np.sqrt(max(
                float(sigma2_before[i]) + epistemic_before[i],
                1e-12,
            ))
            for i in range(2)
        ]
        future_values = []
        for z_vec in common_z:
            y = [
                float(mu_before[i] + pred_sd[i] * z_vec[i])
                for i in range(2)
            ]
            gpr_clone, var_clone = self._exact_primary_fantasy_update(
                x_arr,
                y,
                existing_observations,
                mu_before,
                sigma2_before,
                epistemic_before,
            )
            future_value = self._terminal_value_from_models(
                gpr_clone,
                var_clone,
                terminal_pool,
                observations={
                    **{
                        tuple(int(v) for v in key): [
                            np.asarray(value, dtype=float).copy()
                            for value in values
                        ]
                        for key, values in self.observations.items()
                    },
                    tuple(int(v) for v in x_arr): (
                        [
                            np.asarray(value, dtype=float).copy()
                            for value in self.observations.get(
                                tuple(int(v) for v in x_arr), [])
                        ]
                        + [np.asarray(y, dtype=float)]
                    ),
                },
            )
            future_values.append(future_value)
        expected_value = np.tensordot(
            sample_weights,
            np.asarray(future_values, dtype=float),
            axes=(0, 0),
        )
        component_gain = (
            np.asarray(current_value, dtype=float)
            - np.asarray(expected_value, dtype=float)
        )
        raw_score = self._terminal_diagnostic_gain(
            current_value, expected_value)
        result = {
            "score": (
                max(raw_score, 0.0)
                if self.config.exact_kg_clip_negative
                else raw_score
            ),
            "raw_score": raw_score,
            "expected_terminal_value": (
                np.asarray(expected_value, dtype=float).tolist()
                if np.asarray(expected_value).ndim > 0
                else float(expected_value)
            ),
            "component_gain": (
                np.asarray(component_gain, dtype=float).tolist()
                if np.asarray(component_gain).ndim > 0
                else float(component_gain)
            ),
            "task_entropy_gain": 0.0,
            "task_weight_movement": 0.0,
            "time_clone": 0.0,
            "time_predictive_sample": 0.0,
            "time_joint_update": 0.0,
            "time_robust_terminal": 0.0,
        }
        return result if return_diagnostics else result["score"]

    def _exact_kg_common_samples(self, mc):
        """Draw shared predictive innovations for every design candidate."""
        mc = max(0, int(mc))
        mode = str(self.config.exact_kg_sampling_mode or "iid").lower()
        if mode in ("iid", "random"):
            return (
                self.rng.standard_normal((mc, 2)),
                self.rng.random(mc),
            )
        if mode in (
            "antithetic_nested",
            "nested_antithetic",
            "paired_nested",
        ):
            # Pair-indexed streams make the 2-sample plan an exact prefix of
            # the 8- and 32-sample plans at the same charged posterior state.
            # This mode is used by the numerical-fidelity gate so MC error is
            # not confounded by unrelated random draws.
            z_rows = []
            uniforms = []
            stage = int(len(self.history))
            for pair_index in range(mc // 2):
                pair_rng = np.random.default_rng(np.random.SeedSequence([
                    int(self.config.seed),
                    stage,
                    int(EXACT_KG_STREAM_TAG),
                    int(pair_index),
                ]))
                z_vec = pair_rng.standard_normal(2)
                expert_uniform = float(pair_rng.random())
                z_rows.extend([z_vec, -z_vec])
                uniforms.extend([expert_uniform, 1.0 - expert_uniform])
            if mc % 2:
                z_rows.append(np.zeros(2, dtype=float))
                uniforms.append(0.5)
            return (
                np.asarray(z_rows, dtype=float).reshape(mc, 2),
                np.asarray(uniforms, dtype=float),
            )
        if mode in ("antithetic", "paired", "antithetic_pairs"):
            n_pairs = mc // 2
            base_z = self.rng.standard_normal((n_pairs, 2))
            base_u = self.rng.random(n_pairs)
            z_rows = []
            uniforms = []
            for z_vec, expert_uniform in zip(base_z, base_u):
                z_rows.extend([z_vec, -z_vec])
                uniforms.extend([expert_uniform, 1.0 - expert_uniform])
            if mc % 2:
                z_rows.append(np.zeros(2, dtype=float))
                uniforms.append(0.5)
            return (
                np.asarray(z_rows, dtype=float).reshape(mc, 2),
                np.asarray(uniforms, dtype=float),
            )
        raise ValueError(
            f"unknown exact KG sampling mode {self.config.exact_kg_sampling_mode!r}")

    def _exact_kg_sample_plan(self, mc):
        """Return predictive innovations, expert selectors, and quadrature weights.

        Ordinary IID and antithetic modes use equal Monte Carlo weights.  The
        stratified-expert mode enumerates every finite task expert exactly and
        uses common antithetic Gaussian innovations within each expert.  This
        Rao-Blackwellizes the categorical task identity without changing the
        posterior predictive distribution being integrated.
        """

        mc = max(0, int(mc))
        mode = str(self.config.exact_kg_sampling_mode or "iid").lower()
        stratified_modes = {
            "stratified_expert",
            "expert_stratified",
            "stratified_expert_antithetic",
        }
        if mode not in stratified_modes:
            z_rows, uniforms = self._exact_kg_common_samples(mc)
            weights = np.full(mc, 1.0 / max(mc, 1), dtype=float)
            return z_rows, uniforms, weights
        if self.task_ensemble is None:
            raise ValueError(
                "stratified-expert exact KG requires a finite task ensemble")
        if mc <= 0:
            return (
                np.empty((0, 2), dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
            )

        n_pairs = mc // 2
        base = self.rng.standard_normal((n_pairs, 2))
        gaussian_rows = []
        for z_vec in base:
            gaussian_rows.extend([z_vec, -z_vec])
        if mc % 2:
            gaussian_rows.append(np.zeros(2, dtype=float))
        gaussian_rows = np.asarray(gaussian_rows, dtype=float).reshape(mc, 2)

        if hasattr(self.task_ensemble, "predictive_selector_weights"):
            selector_weights = (
                self.task_ensemble.predictive_selector_weights())
        elif bool(getattr(
            self.task_ensemble, "task_latent_authoritative", False
        )):
            selector_weights = self.task_ensemble._task_latent(
            ).posterior_weights(safe=True).reshape(-1)
        elif hasattr(self.task_ensemble, "structure_weights"):
            selector_weights = self.task_ensemble.structure_weights(
                objective=False)
        else:
            selector_weights = (
                self.task_ensemble.posterior.decision_weights())
        expert_weights = np.asarray(selector_weights, dtype=float)
        expert_weights = np.clip(expert_weights, 0.0, np.inf)
        expert_weights /= max(float(np.sum(expert_weights)), 1e-15)
        edges = np.concatenate([[0.0], np.cumsum(expert_weights)])
        z_blocks = []
        uniform_blocks = []
        weight_blocks = []
        for expert_index, expert_weight in enumerate(expert_weights):
            if expert_weight <= 0.0:
                continue
            midpoint = 0.5 * (edges[expert_index] + edges[expert_index + 1])
            z_blocks.append(gaussian_rows)
            uniform_blocks.append(np.full(mc, midpoint, dtype=float))
            weight_blocks.append(np.full(
                mc, expert_weight / float(mc), dtype=float))
        return (
            np.vstack(z_blocks),
            np.concatenate(uniform_blocks),
            np.concatenate(weight_blocks),
        )

    def _exact_posterior_update_scores(self, candidates, terminal_pool):
        """Monte Carlo exact posterior-update KG over a fixed terminal pool.

        This is intentionally optional and small-budget friendly.  It samples
        predictive observations, applies the same GPR update and HVD residual
        update as the main loop, and measures current terminal value minus
        updated terminal value.  Candidate-level work is embarrassingly
        parallel. Threads remain the portable/live-simulator option; synthetic
        workloads can explicitly use a Linux fork pool to bypass the GIL
        without pickling simulator/provider handles.
        """
        mc = self._effective_exact_kg_mc_samples()
        if mc <= 0 or len(candidates) == 0:
            self._last_exact_kg_raw_scores = np.zeros(
                len(candidates), dtype=float)
            return np.zeros(len(candidates), dtype=float)
        current_value = self._terminal_value_from_models(
            self.gpr,
            self.variance_model,
            terminal_pool,
            task_ensemble=self.task_ensemble,
            observations=self.observations,
        )
        self._last_exact_kg_terminal_value_contract = (
            self._terminal_value_contract_id())
        self._last_exact_kg_current_terminal_action_pool_size = int(len(
            self._terminal_action_pool(
                terminal_pool, observations=self.observations)))
        self._last_exact_kg_current_value = (
            np.asarray(current_value, dtype=float).tolist()
            if np.asarray(current_value).ndim > 0
            else float(current_value)
        )
        self._last_exact_kg_certification_head_authority = (
            self._certification_head_authority())
        self._last_exact_kg_constraint_posterior_source = (
            self._certification_source(
                task_ensemble_active=self.task_ensemble is not None))
        (
            common_z,
            common_expert_uniform,
            common_sample_weights,
        ) = self._exact_kg_sample_plan(mc)
        out = np.zeros(len(candidates), dtype=float)
        raw_out = np.zeros(len(candidates), dtype=float)
        vector_value = np.asarray(current_value).ndim > 0
        expected_values = np.empty(
            (len(candidates), len(np.asarray(current_value).reshape(-1)))
            if vector_value else len(candidates),
            dtype=float,
        )
        component_gains = np.empty_like(expected_values)
        entropy_gain = np.zeros(len(candidates), dtype=float)
        weight_movement = np.zeros(len(candidates), dtype=float)
        task_timing = {
            name: np.zeros(len(candidates), dtype=float)
            for name in (
                "clone",
                "predictive_sample",
                "joint_update",
                "robust_terminal",
            )
        }

        def record_result(index, result):
            out[index] = result["score"]
            raw_out[index] = result["raw_score"]
            expected_values[index] = np.asarray(
                result["expected_terminal_value"], dtype=float)
            component_gains[index] = np.asarray(
                result["component_gain"], dtype=float)
            entropy_gain[index] = result["task_entropy_gain"]
            weight_movement[index] = result["task_weight_movement"]
            for name, values in task_timing.items():
                values[index] = result[f"time_{name}"]

        def combine_chunk_results(chunks):
            total_mass = float(sum(mass for mass, _ in chunks))
            if total_mass <= 0.0:
                raise ValueError("exact KG chunk weights have zero mass")
            expected = sum(
                mass * np.asarray(
                    result["expected_terminal_value"], dtype=float)
                for mass, result in chunks
            ) / total_mass
            component_gain = (
                np.asarray(current_value, dtype=float) - expected)
            raw_score = self._terminal_diagnostic_gain(
                current_value, expected)
            combined = {
                "score": (
                    max(raw_score, 0.0)
                    if self.config.exact_kg_clip_negative
                    else raw_score
                ),
                "raw_score": float(raw_score),
                "expected_terminal_value": (
                    expected.tolist()
                    if np.asarray(expected).ndim > 0
                    else float(expected)
                ),
                "component_gain": (
                    component_gain.tolist()
                    if np.asarray(component_gain).ndim > 0
                    else float(component_gain)
                ),
            }
            for field in ("task_entropy_gain", "task_weight_movement"):
                combined[field] = float(sum(
                    mass * float(result[field])
                    for mass, result in chunks
                ) / total_mass)
            for name in task_timing:
                combined[f"time_{name}"] = float(sum(
                    mass * float(result[f"time_{name}"])
                    for mass, result in chunks
                ) / total_mass)
            return combined
        parallel_backend = str(
            self.config.exact_kg_parallel_backend or "thread").lower()
        requested_jobs = max(1, int(self.config.exact_kg_jobs))
        process_backend = parallel_backend in (
            "process_fork", "fork", "process")
        maximum_parallel_units = (
            len(candidates) * max(len(common_z), 1)
            if process_backend else len(candidates)
        )
        jobs = min(requested_jobs, maximum_parallel_units)
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
            extra=(
                f"candidates={len(candidates)} mc={int(mc)} jobs={int(jobs)} "
                f"effective_samples={len(common_z)} backend={parallel_backend}"
            ),
        )
        if jobs <= 1:
            for j, x in enumerate(candidates):
                result = self._exact_posterior_update_score_one(
                    x,
                    common_z,
                    terminal_pool,
                    current_value,
                    common_expert_uniform,
                    common_sample_weights,
                    return_diagnostics=True,
                )
                record_result(j, result)
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
            if vector_value:
                order = np.lexsort(tuple(
                    expected_values[:, index]
                    for index in reversed(range(expected_values.shape[1]))
                ))
                out[:] = 0.0
                out[order] = np.arange(
                    len(candidates), 0, -1, dtype=float)
            self._last_exact_kg_expected_values = expected_values.copy()
            self._last_exact_kg_component_gains = component_gains.copy()
            self._last_exact_kg_task_entropy_gain = entropy_gain
            self._last_exact_kg_task_weight_movement = weight_movement
            self._last_exact_kg_task_timing = task_timing
            self._last_exact_kg_raw_scores = raw_out
            return out
        global _FORK_EXACT_KG_CONTEXT
        if parallel_backend in ("process_fork", "fork", "process"):
            if "fork" not in multiprocessing.get_all_start_methods():
                raise RuntimeError("exact_kg process_fork backend requires Linux fork")
            _FORK_EXACT_KG_CONTEXT = (
                self,
                common_z,
                terminal_pool,
                current_value,
                common_expert_uniform,
                common_sample_weights,
            )
            executor = ProcessPoolExecutor(
                max_workers=jobs,
                mp_context=multiprocessing.get_context("fork"),
            )
            chunks_per_candidate = min(
                max(len(common_z), 1),
                max(1, int(np.ceil(jobs / float(len(candidates))))),
            )
            chunked_process = chunks_per_candidate > 1
            submit = lambda pool, x: pool.submit(
                _fork_exact_kg_candidate, x)
        elif parallel_backend in ("thread", "threads"):
            executor = ThreadPoolExecutor(max_workers=jobs)
            submit = lambda pool, x: pool.submit(
                self._exact_posterior_update_score_one,
                x,
                common_z,
                terminal_pool,
                current_value,
                common_expert_uniform,
                common_sample_weights,
                True,
            )
            chunked_process = False
        else:
            raise ValueError(
                f"unknown exact KG parallel backend {parallel_backend!r}")
        self._last_exact_kg_parallel_workers = int(jobs)
        self._last_exact_kg_chunks_per_candidate = int(
            chunks_per_candidate if process_backend else 1)
        try:
            with executor as pool:
                if chunked_process:
                    chunk_rows = []
                    for candidate_index, candidate in enumerate(candidates):
                        for sample_indices in np.array_split(
                            np.arange(len(common_z), dtype=int),
                            chunks_per_candidate,
                        ):
                            if len(sample_indices):
                                chunk_rows.append((
                                    candidate_index,
                                    candidate,
                                    sample_indices,
                                ))
                    futures = {
                        pool.submit(
                            _fork_exact_kg_candidate_chunk,
                            (candidate, sample_indices),
                        ): candidate_index
                        for candidate_index, candidate, sample_indices
                        in chunk_rows
                    }
                    candidate_chunks = {
                        index: [] for index in range(len(candidates))
                    }
                    done = 0
                    for future in as_completed(futures):
                        index = futures[future]
                        candidate_chunks[index].append(future.result())
                        done += 1
                        if done == len(futures) or done % max(
                            1, int(np.ceil(len(futures) / emit_updates))
                        ) == 0:
                            frac = 0.35 + 0.55 * (
                                float(done) / float(len(futures)))
                            self._progress_emit(
                                n=stage_n,
                                frac=frac,
                                kind="exact_kg_chunks",
                                started_at=step_started_at,
                                run_started_at=run_started_at,
                                extra=f"chunks_done={done}/{len(futures)}",
                            )
                    for index in range(len(candidates)):
                        record_result(
                            index,
                            combine_chunk_results(candidate_chunks[index]),
                        )
                else:
                    futures = {
                        submit(pool, x): j
                        for j, x in enumerate(candidates)
                    }
                    done = 0
                    for future in as_completed(futures):
                        index = futures[future]
                        result = future.result()
                        record_result(index, result)
                        done += 1
                        if done == len(candidates) or done % emit_every == 0:
                            frac = 0.35 + 0.55 * (
                                float(done) / float(len(candidates)))
                            self._progress_emit(
                                n=stage_n,
                                frac=frac,
                                kind="exact_kg_candidates",
                                started_at=step_started_at,
                                run_started_at=run_started_at,
                                extra=(
                                    f"candidates_done={done}/"
                                    f"{len(candidates)}"),
                            )
        finally:
            if parallel_backend in ("process_fork", "fork", "process"):
                _FORK_EXACT_KG_CONTEXT = None
        self._last_exact_kg_task_entropy_gain = entropy_gain
        self._last_exact_kg_task_weight_movement = weight_movement
        self._last_exact_kg_task_timing = task_timing
        if vector_value:
            order = np.lexsort(tuple(
                expected_values[:, index]
                for index in reversed(range(expected_values.shape[1]))
            ))
            out[:] = 0.0
            out[order] = np.arange(len(candidates), 0, -1, dtype=float)
        self._last_exact_kg_expected_values = expected_values.copy()
        self._last_exact_kg_component_gains = component_gains.copy()
        self._last_exact_kg_raw_scores = raw_out
        return out

    def _exact_posterior_update_scores_for_actions(
        self,
        candidates,
        terminal_pool,
        active_indices,
    ):
        """Run exact fantasy refits only for the declared active actions."""

        active = []
        for raw_index in np.asarray(active_indices, dtype=int).reshape(-1):
            index = int(raw_index)
            if index < 0 or index >= len(candidates):
                raise IndexError("exact VOI active candidate index out of range")
            if index not in active:
                active.append(index)
        if not active:
            raise ValueError("exact VOI requires at least one active action")
        active_candidates = [candidates[index] for index in active]
        active_scores = self._exact_posterior_update_scores(
            active_candidates, terminal_pool)

        def expand(values, fill=np.nan):
            values = np.asarray(values, dtype=float)
            shape = (len(candidates),) + values.shape[1:]
            expanded = np.full(shape, fill, dtype=float)
            expanded[np.asarray(active, dtype=int)] = values
            return expanded

        full_scores = np.full(len(candidates), -1e300, dtype=float)
        full_scores[np.asarray(active, dtype=int)] = np.asarray(
            active_scores, dtype=float)
        self._last_exact_kg_raw_scores = expand(
            self._last_exact_kg_raw_scores)
        self._last_exact_kg_expected_values = expand(
            self._last_exact_kg_expected_values)
        self._last_exact_kg_component_gains = expand(
            self._last_exact_kg_component_gains)
        self._last_exact_kg_task_entropy_gain = expand(
            self._last_exact_kg_task_entropy_gain)
        self._last_exact_kg_task_weight_movement = expand(
            self._last_exact_kg_task_weight_movement)
        self._last_exact_kg_task_timing = {
            name: expand(values)
            for name, values in self._last_exact_kg_task_timing.items()
        }
        active_mask = np.zeros(len(candidates), dtype=bool)
        active_mask[np.asarray(active, dtype=int)] = True
        self._last_exact_kg_active_indices = np.asarray(active, dtype=int)
        self._last_exact_kg_active_mask = active_mask
        return full_scores

    def _decision_backend_terminal_recommendation(self, pool):
        """Return the Bayes action paired with a non-legacy online backend.

        Random, Sobol and Thompson rules randomize data collection, not the
        final Bayesian decision.  Their recommendation therefore minimizes
        the same posterior risk used by the risk-aware backends.  Restricting
        this action to evaluated points provides incumbent retention without
        consulting target truth or adding an empirical fallback.
        """

        backend = str(
            self.config.decision_backend or "legacy"
        ).strip().lower().replace("-", "_")
        if backend in {"legacy", "legacy_kg", "exact_kg", "additive"}:
            return None
        evaluated = unique_candidates([x for x, _ in self.history])
        if backend in {"n0_best", "frozen_incumbent"}:
            action_pool = evaluated[: int(self.config.n0)]
            pool_source = "initial_design_only"
        elif self._decision_backend_observed_terminal_active():
            action_pool = self._terminal_action_pool(
                pool, observations=self.observations)
            pool_source = "all_budgeted_target_evaluations"
        else:
            action_pool = unique_candidates(list(pool) + evaluated)
            pool_source = "terminal_pool_plus_evaluations"
        if not action_pool:
            return None
        terminal_mode = self._effective_exact_terminal_mode()
        if (
            backend in {"sobol_exact_joint_voi", "exact_joint_voi_sobol"}
            and terminal_mode in {
                "tcb_certified_lexicographic",
                "tcb-certified-lexicographic",
                "certified_lexicographic",
            }
        ):
            selected, value, terminal = (
                self._terminal_certified_lexicographic_decision(
                    self.gpr,
                    self.variance_model,
                    action_pool,
                    task_ensemble=self.task_ensemble,
                    observations=self.observations,
                )
            )
            if selected is None:
                return None
            _, details = self._solve_posterior_recommendation(pool=[selected])
            details.update({
                "decision_backend_terminal_used": True,
                "decision_backend_terminal_rule": (
                    "robust_certified_lexicographic"),
                "decision_backend_terminal_name": backend,
                "decision_backend_terminal_pool_source": pool_source,
                "decision_backend_terminal_pool_size": int(len(action_pool)),
                "decision_backend_terminal_observed_only": bool(
                    self.config.decision_recommend_observed_only),
                "decision_backend_terminal_value": np.asarray(
                    value, dtype=float).tolist(),
                "decision_backend_terminal_status": terminal["status"],
                "decision_backend_terminal_certified_count": int(
                    terminal["certified_count"]),
                "decision_backend_terminal_margin": float(
                    terminal["selected_margin"]),
                "decision_backend_terminal_objective": float(
                    terminal["selected_objective"]),
                "decision_backend_terminal_target_oracle_used": False,
            })
            return selected, details
        components = self._terminal_bayes_risk_components(
            self.gpr,
            self.variance_model,
            action_pool,
            task_ensemble=self.task_ensemble,
            risk_penalty=self.config.decision_risk_penalty,
        )
        local = int(np.argmin(components["risk"]))
        selected = tuple(int(v) for v in action_pool[local])
        _, details = self._solve_posterior_recommendation(pool=[selected])
        details.update({
            "decision_backend_terminal_used": True,
            "decision_backend_terminal_rule": "posterior_bayes_risk",
            "decision_backend_terminal_name": backend,
            "decision_backend_terminal_pool_source": pool_source,
            "decision_backend_terminal_pool_size": int(len(action_pool)),
            "decision_backend_terminal_observed_only": bool(
                self.config.decision_recommend_observed_only),
            "decision_backend_terminal_risk": float(
                components["risk"][local]),
            "decision_backend_terminal_objective": float(
                components["objective"][local]),
            "decision_backend_terminal_expected_violation": float(
                components["expected_violation"][local]),
            "decision_backend_terminal_probability_violation": float(
                components["probability_violation"][local]),
            "decision_backend_terminal_violation_loss": float(
                components["violation_loss"][local]),
            "decision_backend_terminal_violation_loss_mode": str(
                components["violation_loss_mode"]),
            "decision_backend_terminal_aleatoric_mode": str(
                components["decision_aleatoric_mode"]),
            "decision_backend_terminal_ambiguity_mode": str(
                components["decision_ambiguity_mode"]),
            "decision_backend_terminal_nominal_expected_violation": float(
                components["nominal_expected_violation"][local]),
            "decision_backend_terminal_robust_expected_violation": float(
                components["robust_expected_violation"][local]),
            "decision_backend_terminal_nominal_probability_violation": float(
                components["nominal_probability_violation"][local]),
            "decision_backend_terminal_robust_probability_violation": float(
                components["robust_probability_violation"][local]),
            "decision_backend_terminal_model_disagreement": float(
                components["model_disagreement"][local]),
            "decision_backend_terminal_kl_radius": float(
                components["kl_radius"]),
            "decision_backend_terminal_target_oracle_used": False,
            "terminal_value_contract": self._terminal_value_contract_id(),
            "terminal_bayes_pool_audit": self._terminal_bayes_pool_audit(
                action_pool, components, selected),
        })
        return selected, details

    def _terminal_bayes_pool_audit(self, pool, components, selected):
        """Freeze posterior-risk ranks before joining post-decision truth."""

        candidates = [tuple(int(v) for v in x) for x in pool]
        risks = np.asarray(components["risk"], dtype=float).reshape(-1)
        if not candidates or len(candidates) != len(risks):
            return {
                "status": "unavailable",
                "target_oracle_used_for_ranking": False,
            }
        selected = tuple(int(v) for v in selected)
        if selected not in candidates:
            return {
                "status": "selected_outside_pool",
                "target_oracle_used_for_ranking": False,
            }
        order = np.argsort(risks, kind="stable")
        selected_index = candidates.index(selected)
        selected_rank = int(np.flatnonzero(order == selected_index)[0]) + 1
        best_index = int(order[0])
        def component(name, index):
            values = components.get(name)
            if values is None:
                return None
            value = float(np.asarray(values, dtype=float).reshape(-1)[index])
            return value if np.isfinite(value) else None

        ranked = []
        for rank, index in enumerate(order, start=1):
            index = int(index)
            ranked.append({
                "posterior_rank": int(rank),
                "point_fingerprint": integer_design_fingerprint([
                    candidates[index]]),
                "risk": component("risk", index),
                "objective": component("objective", index),
                "expected_violation": component(
                    "expected_violation", index),
                "probability_violation": component(
                    "probability_violation", index),
                "violation_loss": component("violation_loss", index),
                "nominal_expected_violation": component(
                    "nominal_expected_violation", index),
                "robust_expected_violation": component(
                    "robust_expected_violation", index),
                "nominal_probability_violation": component(
                    "nominal_probability_violation", index),
                "robust_probability_violation": component(
                    "robust_probability_violation", index),
            })
        frozen = {
            "status": "ranked",
            "pool_size": int(len(candidates)),
            "selected_fingerprint": integer_design_fingerprint([selected]),
            "selected_risk": float(risks[selected_index]),
            "selected_risk_rank": int(selected_rank),
            "counterfactual_bayes_fingerprint": integer_design_fingerprint([
                candidates[best_index]]),
            "counterfactual_bayes_risk": float(risks[best_index]),
            "selected_matches_counterfactual_bayes": bool(
                selected_index == best_index),
            "decision_aleatoric_mode": str(components.get(
                "decision_aleatoric_mode", "unknown")),
            "violation_loss_mode": str(components.get(
                "violation_loss_mode", "unknown")),
            "decision_ambiguity_mode": str(components.get(
                "decision_ambiguity_mode", "unknown")),
            "posterior_ranked_candidates": ranked,
            "target_oracle_used_for_ranking": False,
        }
        if not self.config.truth_pool_diagnostics:
            return frozen
        # The ranking and both indices above are immutable before this truth
        # join. These fields audit the decision and never feed another action.
        try:
            true_margins = np.asarray([
                float(self._true_chance_margin(candidate))
                for candidate in candidates
            ], dtype=float)
        except Exception:
            frozen["truth_after_rank_freeze_available"] = False
            return frozen
        for record in ranked:
            index = int(order[int(record["posterior_rank"]) - 1])
            record["true_margin_post_rank"] = float(true_margins[index])
            record["true_feasible_post_rank"] = bool(
                true_margins[index] <= 0.0)
        feasible_indices = np.flatnonzero(true_margins <= 0.0)
        best_true_feasible = None
        if len(feasible_indices):
            best_true_index = int(feasible_indices[
                np.argmin(risks[feasible_indices])])
            best_true_rank = int(
                np.flatnonzero(order == best_true_index)[0]) + 1
            best_true_feasible = {
                "point_fingerprint": integer_design_fingerprint([
                    candidates[best_true_index]]),
                "posterior_rank": best_true_rank,
                "risk": float(risks[best_true_index]),
                "true_margin": float(true_margins[best_true_index]),
            }
        frozen.update({
            "truth_after_rank_freeze_available": True,
            "truth_join_timing": "post_terminal_rank",
            "truth_admissible_decision_input": False,
            "selected_true_margin": float(true_margins[selected_index]),
            "selected_true_feasible": bool(
                true_margins[selected_index] <= 0.0),
            "counterfactual_bayes_true_margin": float(
                true_margins[best_index]),
            "counterfactual_bayes_true_feasible": bool(
                true_margins[best_index] <= 0.0),
            "true_feasible_count_post_rank": int(len(feasible_indices)),
            "best_true_feasible_post_rank": best_true_feasible,
            "target_oracle_used_for_decision": False,
        })
        return frozen

    def _posterior_dominance_terminal_recommendation(self):
        """Return the sequentially maintained posterior-safe incumbent."""
        if (
            not self._posterior_dominance_active()
            or self._posterior_dominance_incumbent is None
        ):
            return None
        selected = tuple(int(v) for v in self._posterior_dominance_incumbent)
        action_pool = unique_candidates([x for x, _ in self.history])
        if selected not in action_pool:
            action_pool.append(selected)
        components = self._terminal_bayes_risk_components(
            self.gpr,
            self.variance_model,
            action_pool,
            task_ensemble=self.task_ensemble,
        )
        selected_index = action_pool.index(selected)
        _, details = self._solve_posterior_recommendation(pool=[selected])
        last_update = (
            copy.deepcopy(self._posterior_dominance_history[-1])
            if self._posterior_dominance_history else None
        )
        details.update({
            "posterior_dominance_terminal_used": True,
            "posterior_dominance_terminal_rule": (
                "sequential_cantelli_safe_incumbent"
            ),
            "posterior_dominance_incumbent": list(map(int, selected)),
            "posterior_dominance_terminal_risk": float(
                components["risk"][selected_index]),
            "posterior_dominance_terminal_risk_variance": float(
                components["risk_variance"][selected_index]),
            "posterior_dominance_delta": float(
                self.config.posterior_dominance_delta),
            "posterior_dominance_last_update": last_update,
            "posterior_dominance_target_oracle_used": False,
            "terminal_bayes_pool_audit": self._terminal_bayes_pool_audit(
                action_pool, components, selected),
        })
        return selected, details

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

    def _adaptive_outcome_audit(self, final_evaluation):
        """Post-run n0-to-final truth audit that never affects decisions."""

        initial_points = unique_candidates([
            x for x, _ in self.history[: int(self.config.n0)]
        ])
        try:
            _, true_best_objective = self._true_best_feasible_cached()
        except Exception:
            true_best_objective = np.inf
        feasible_regrets = []
        margins = []
        for x in initial_points:
            try:
                margin = self._true_chance_margin(x)
                objective = float(self.problem.true_objective(x))
            except Exception:
                continue
            margins.append(float(margin))
            if margin <= 0.0 and np.isfinite(true_best_objective):
                feasible_regrets.append(objective - true_best_objective)
        initial_has_feasible = bool(feasible_regrets)
        final_feasible = bool(final_evaluation.get("true_feasible", False))
        final_regret = (
            float(final_evaluation["simple_regret"])
            if final_feasible
            and np.isfinite(float(final_evaluation.get("simple_regret", np.nan)))
            else None
        )
        initial_best = (
            float(min(feasible_regrets)) if feasible_regrets else None)
        return {
            "initial_design_size": int(len(initial_points)),
            "initial_true_feasible_count": int(len(feasible_regrets)),
            "initial_has_true_feasible": initial_has_feasible,
            "initial_best_feasible_regret": initial_best,
            "initial_true_min_margin": (
                float(min(margins)) if margins else None),
            "final_true_feasible": final_feasible,
            "final_feasible_regret": final_regret,
            "adaptive_rescue": bool(
                not initial_has_feasible and final_feasible),
            "adaptive_loss": bool(
                initial_has_feasible and not final_feasible),
            "adaptive_preservation": bool(
                initial_has_feasible and final_feasible),
            "adaptive_improves_initial_best": bool(
                initial_best is not None
                and final_regret is not None
                and final_regret < initial_best - 1e-12),
            "adaptive_regret_change": (
                None
                if initial_best is None or final_regret is None
                else float(final_regret - initial_best)
            ),
            "audit_timing": "post_run_only",
            "target_oracle_used_for_decision": False,
        }

    def _certificate_outcome_audit(self):
        """Post-run certificate coverage and false-certificate audit."""

        points = unique_candidates([x for x, _ in self.history])
        if not points:
            return {
                "status": "empty",
                "target_oracle_used_for_decision": False,
            }
        if self.task_ensemble is None:
            mu_con = self.gpr[1].posterior_mean_many(points)
            aleatoric = self.variance_model.predict_certification_variance_many(
                1, points, self.problem)
            cert = self._certification_result(mu_con, points, aleatoric)
        else:
            robust = self.task_ensemble.robust_moments_many(
                1, points, certification=True)
            cert = self._certification_result(
                robust.mean_upper,
                points,
                robust.aleatoric_upper,
                epistemic=robust.epistemic_upper,
            )
        posterior_feasible = np.asarray(cert.margin, dtype=float) <= 0.0
        true_margins = np.asarray([
            self._true_chance_margin(x) for x in points
        ], dtype=float)
        true_feasible = true_margins <= 0.0
        true_positive = posterior_feasible & true_feasible
        false_positive = posterior_feasible & ~true_feasible
        return {
            "status": "audited",
            "evaluated_point_count": int(len(points)),
            "posterior_certified_count": int(np.sum(posterior_feasible)),
            "posterior_certificate_vacuous": bool(
                not np.any(posterior_feasible)),
            "true_feasible_count": int(np.sum(true_feasible)),
            "certified_true_feasible_count": int(np.sum(true_positive)),
            "false_certificate_count": int(np.sum(false_positive)),
            "certificate_precision": (
                float(np.mean(true_feasible[posterior_feasible]))
                if np.any(posterior_feasible) else None
            ),
            "certificate_recall_on_evaluated_feasible": (
                float(np.mean(posterior_feasible[true_feasible]))
                if np.any(true_feasible) else None
            ),
            "minimum_posterior_margin": float(np.min(cert.margin)),
            "minimum_true_margin": float(np.min(true_margins)),
            "certification_mode": str(cert.mode),
            "audit_timing": "post_run_only",
            "target_oracle_used_for_decision": False,
        }

    def _true_chance_margin(self, x):
        sig = self.problem.true_sigma(x)
        return float(
            self.problem.true_constraint_mean(x)
            + norm.ppf(1 - self.problem.alpha) * float(sig[1])
            - self.problem.tau
        )

    def _truth_pool_diagnostics(
        self,
        pool,
        selected=None,
        prefix="candidate",
        sources=None,
    ):
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
        true_means = []
        true_variances = []
        retained_pool = []
        for x in pool:
            try:
                margin = self._true_chance_margin(x)
                obj = float(self.problem.true_objective(x))
                true_mean = float(self.problem.true_constraint_mean(x))
                true_variance = float(self.problem.true_sigma(x)[1]) ** 2
            except Exception:
                continue
            retained_pool.append(x)
            margins.append(margin)
            regrets.append(obj - true_best_obj if np.isfinite(true_best_obj) else np.nan)
            true_means.append(true_mean)
            true_variances.append(true_variance)
        if not margins:
            return {f"{prefix}_truth_diagnostics_available": False}
        pool = retained_pool
        margins = np.asarray(margins, dtype=float)
        regrets = np.asarray(regrets, dtype=float)
        true_means = np.asarray(true_means, dtype=float)
        true_variances = np.maximum(
            np.asarray(true_variances, dtype=float), 1e-12)
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
            predicted_mean = np.asarray(cert.mu, dtype=float)
            predicted_variance = np.maximum(
                np.asarray(cert.aleatoric_var, dtype=float), 1e-12)
            epistemic_radius = np.sqrt(
                max(float(cert.beta_g), 0.0)
                * np.maximum(np.asarray(
                    cert.epistemic_var, dtype=float), 0.0)
            )
            oracle_variance_margin = (
                predicted_mean
                + epistemic_radius
                + float(cert.z_alpha) * np.sqrt(true_variances)
                - float(cert.tau)
            )
            oracle_mean_margin = (
                true_means
                + epistemic_radius
                + float(cert.z_alpha) * np.sqrt(predicted_variance)
                - float(cert.tau)
            )
            oracle_both_margin = (
                true_means
                + epistemic_radius
                + float(cert.z_alpha) * np.sqrt(true_variances)
                - float(cert.tau)
            )

            def rank_correlation(left, right):
                left = np.asarray(left, dtype=float)
                right = np.asarray(right, dtype=float)
                if len(left) < 2:
                    return None
                left_order = np.argsort(left, kind="stable")
                right_order = np.argsort(right, kind="stable")
                left_rank = np.empty(len(left), dtype=float)
                right_rank = np.empty(len(right), dtype=float)
                left_rank[left_order] = np.arange(len(left), dtype=float)
                right_rank[right_order] = np.arange(len(right), dtype=float)
                left_rank -= float(np.mean(left_rank))
                right_rank -= float(np.mean(right_rank))
                denominator = float(
                    np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
                return (
                    None if denominator <= 1e-12
                    else float(left_rank @ right_rank / denominator)
                )

            full_count = int(np.sum(posterior_margins <= 0.0))
            posterior_certified = posterior_margins <= 0.0
            true_certified_count = int(np.sum(
                posterior_certified & feasible))
            false_certified_count = int(np.sum(
                posterior_certified & ~feasible))
            certificate_precision = (
                None
                if full_count == 0
                else float(true_certified_count / full_count)
            )
            oracle_variance_count = int(np.sum(
                oracle_variance_margin <= 0.0))
            oracle_mean_count = int(np.sum(oracle_mean_margin <= 0.0))
            oracle_both_count = int(np.sum(oracle_both_margin <= 0.0))
            if not np.any(feasible):
                failure_layer = "candidate_support"
            elif oracle_both_count == 0:
                failure_layer = "epistemic_or_safety_depth"
            elif full_count > 0:
                failure_layer = "closed"
            elif (
                oracle_mean_count - full_count
                >= oracle_variance_count - full_count
            ):
                failure_layer = "constraint_mean"
            else:
                failure_layer = "cumulative_variance"
            out.update({
                f"{prefix}_failure_decomposition_available": True,
                f"{prefix}_failure_layer": failure_layer,
                f"{prefix}_constraint_mean_rank_correlation": rank_correlation(
                    predicted_mean, true_means),
                f"{prefix}_chance_margin_rank_correlation": rank_correlation(
                    posterior_margins, margins),
                f"{prefix}_constraint_mean_median_abs_error": float(
                    np.median(np.abs(predicted_mean - true_means))),
                f"{prefix}_variance_median_abs_log_error": float(np.median(
                    np.abs(np.log(predicted_variance)
                           - np.log(true_variances)))),
                f"{prefix}_full_certified_count": full_count,
                f"{prefix}_true_certified_count": true_certified_count,
                f"{prefix}_false_certified_count": false_certified_count,
                f"{prefix}_certificate_precision": certificate_precision,
                f"{prefix}_oracle_variance_certified_count": (
                    oracle_variance_count),
                f"{prefix}_oracle_mean_certified_count": oracle_mean_count,
                f"{prefix}_oracle_mean_variance_certified_count": (
                    oracle_both_count),
                f"{prefix}_median_oracle_variance_margin": float(
                    np.median(oracle_variance_margin)),
                f"{prefix}_median_oracle_mean_margin": float(
                    np.median(oracle_mean_margin)),
                f"{prefix}_median_oracle_mean_variance_margin": float(
                    np.median(oracle_both_margin)),
                f"{prefix}_minimum_oracle_mean_variance_margin": float(
                    np.min(oracle_both_margin)),
                f"{prefix}_minimum_epistemic_radius": float(
                    np.min(epistemic_radius)),
                f"{prefix}_audit_target_oracle_used_for_decision": False,
            })
            basis_map = getattr(self.gpr[1], "basis_map", None)
            if basis_map is not None and hasattr(
                basis_map, "role_assignment_oracle_expressivity_audit"
            ):
                out[f"{prefix}_role_assignment_oracle_expressivity"] = (
                    basis_map.role_assignment_oracle_expressivity_audit(
                        pool, true_means)
                )
            if np.any(feasible):
                feasible_indices = np.flatnonzero(feasible)
                safest = int(feasible_indices[int(np.argmin(
                    oracle_both_margin[feasible_indices]))])
                out.update({
                    f"{prefix}_best_feasible_oracle_mean_variance_margin": (
                        float(oracle_both_margin[safest])
                    ),
                    f"{prefix}_best_feasible_epistemic_radius": float(
                        epistemic_radius[safest]),
                    f"{prefix}_best_feasible_true_margin": float(
                        margins[safest]),
                })
        except Exception:
            out[f"{prefix}_posterior_audit_available"] = False
        if sources:
            source_rows = {}
            for index, x in enumerate(pool):
                source = str(sources.get(tuple(x), "unknown"))
                row = source_rows.setdefault(source, {
                    "count": 0,
                    "true_feasible_count": 0,
                    "true_min_margin": np.inf,
                })
                row["count"] += 1
                row["true_feasible_count"] += int(bool(feasible[index]))
                row["true_min_margin"] = min(
                    float(row["true_min_margin"]), float(margins[index]))
            out[f"{prefix}_source_truth_support"] = {
                source: {
                    "count": int(row["count"]),
                    "true_feasible_count": int(row["true_feasible_count"]),
                    "has_true_feasible": bool(row["true_feasible_count"] > 0),
                    "true_min_margin": float(row["true_min_margin"]),
                    "target_oracle_used_for_decision": False,
                }
                for source, row in source_rows.items()
            }
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
            out[f"{prefix}_best_true_feasible_x"] = list(
                map(int, pool[best_pos]))
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

    def _truth_acquisition_score_audit(self, pool, scores, selected_index):
        """Post-decision score audit; synthetic truth never changes selection."""
        if not self.config.truth_pool_diagnostics or not pool:
            return {}
        scores = np.asarray(scores, dtype=float)
        if len(pool) != len(scores):
            return {}
        feasible_indices = []
        objectives = np.full(len(pool), np.inf, dtype=float)
        for index, x in enumerate(pool):
            try:
                if self._true_chance_margin(x) <= 0.0:
                    feasible_indices.append(index)
                    objectives[index] = float(self.problem.true_objective(x))
            except Exception:
                continue
        if not feasible_indices:
            return {
                "acquisition_truth_score_audit_available": True,
                "acquisition_true_feasible_candidate_count": 0,
            }
        feasible_indices = np.asarray(feasible_indices, dtype=int)
        best_objective_index = int(feasible_indices[
            np.argmin(objectives[feasible_indices])
        ])
        best_score_index = int(feasible_indices[
            np.argmax(scores[feasible_indices])
        ])
        descending = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=int)
        ranks[descending] = np.arange(1, len(scores) + 1)
        selected_index = int(selected_index)
        return {
            "acquisition_truth_score_audit_available": True,
            "acquisition_true_feasible_candidate_count": int(
                len(feasible_indices)),
            "acquisition_best_true_feasible_score": float(
                scores[best_objective_index]),
            "acquisition_best_true_feasible_score_rank": int(
                ranks[best_objective_index]),
            "acquisition_highest_score_true_feasible": float(
                scores[best_score_index]),
            "acquisition_highest_score_true_feasible_rank": int(
                ranks[best_score_index]),
            "acquisition_selected_minus_highest_feasible_score": float(
                scores[selected_index] - scores[best_score_index]),
        }

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
        failure_layers = {}
        phi_support_iterations = 0
        phi_feasible_iterations = 0
        for row in rows:
            layer = row.get("candidate_failure_layer")
            if layer is not None:
                failure_layers[str(layer)] = int(
                    failure_layers.get(str(layer), 0) + 1)
            source_support = row.get("candidate_source_truth_support") or {}
            phi_rows = [
                value for source, value in source_support.items()
                if str(source).startswith("boundary_phi:")
            ]
            if phi_rows:
                phi_support_iterations += 1
                phi_feasible_iterations += int(any(
                    bool(value.get("has_true_feasible", False))
                    for value in phi_rows
                ))
        return {
            "enabled": True,
            "n_logged": int(len(rows)),
            "pool_has_true_feasible_rate": mean_bool("candidate_has_true_feasible"),
            "pool_has_true_safe_good_rate": mean_bool("candidate_has_true_safe_good"),
            "selected_true_feasible_rate": mean_bool("candidate_selected_true_feasible"),
            "missed_true_feasible_rate": mean_bool("candidate_missed_true_feasible"),
            "missed_true_safe_good_rate": mean_bool("candidate_missed_true_safe_good"),
            "mean_pool_true_feasible_rate": mean_float("candidate_true_feasible_rate"),
            "mean_pool_min_true_margin": mean_float("candidate_true_min_margin"),
            "mean_pool_min_posterior_margin": mean_float(
                "candidate_posterior_min_margin"),
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
            "mean_best_true_feasible_score_rank": mean_float(
                "acquisition_best_true_feasible_score_rank"),
            "mean_highest_score_true_feasible_rank": mean_float(
                "acquisition_highest_score_true_feasible_rank"),
            "mean_selected_minus_highest_feasible_score": mean_float(
                "acquisition_selected_minus_highest_feasible_score"),
            "terminal_frontier_selected_rate": mean_bool(
                "terminal_frontier_selected"),
            "failure_layer_counts": failure_layers,
            "mean_constraint_mean_rank_correlation": mean_float(
                "candidate_constraint_mean_rank_correlation"),
            "mean_chance_margin_rank_correlation": mean_float(
                "candidate_chance_margin_rank_correlation"),
            "mean_constraint_mean_median_abs_error": mean_float(
                "candidate_constraint_mean_median_abs_error"),
            "mean_variance_median_abs_log_error": mean_float(
                "candidate_variance_median_abs_log_error"),
            "mean_oracle_variance_certified_count": mean_float(
                "candidate_oracle_variance_certified_count"),
            "mean_oracle_mean_certified_count": mean_float(
                "candidate_oracle_mean_certified_count"),
            "mean_oracle_mean_variance_certified_count": mean_float(
                "candidate_oracle_mean_variance_certified_count"),
            "phi_candidate_iteration_count": int(phi_support_iterations),
            "phi_candidate_true_feasible_iteration_rate": (
                float(phi_feasible_iterations / phi_support_iterations)
                if phi_support_iterations else None
            ),
            "target_oracle_used_for_decision": False,
        }

    def _summarize_boundary_coordinate_proposals(self):
        rows = [
            row.get("boundary_coordinate_proposal") or {}
            for row in self.iteration_log
        ]
        generated = [row for row in rows if row.get("status") == "selected"]
        role_counts = {}
        for row in generated:
            for role, count in (row.get("role_counts") or {}).items():
                role_counts[str(role)] = int(
                    role_counts.get(str(role), 0) + int(count))
        return {
            "enabled": bool(
                self.config.boundary_coordinate_candidate_count > 0),
            "generated_iteration_count": int(len(generated)),
            "selected_candidate_count": int(sum(
                int(row.get("selected", 0)) for row in generated)),
            "role_counts": role_counts,
            "last": copy.deepcopy(
                rows[-1] if rows
                else self._last_boundary_coordinate_proposal_info),
            "coordinate": "phi=source_aligned_chance_boundary",
            "target_observations_used_for_calibration": True,
            "target_oracle_used": False,
        }

    def _boundary_coordinate_raw_pool_audit(self):
        """Audit one shared unlabeled pool after all decisions are frozen.

        Source-stratum templates are deliberately disabled here. This makes
        the raw policy pool identical across latent, profile-phi, learned
        exposure-phi, and provider upper-bound variants for a fixed seed.
        Target truth is read only by ``_truth_pool_diagnostics`` after the
        recommendation has already been selected.
        """

        if self._boundary_raw_pool_audit_cache is not None:
            return copy.deepcopy(self._boundary_raw_pool_audit_cache)
        if (
            not self.config.truth_pool_diagnostics
            or not hasattr(self.problem, "boundary_excitation_candidates")
        ):
            result = {
                "status": "disabled",
                "post_run_only": True,
                "target_oracle_used_for_decision": False,
            }
            self._boundary_raw_pool_audit_cache = result
            return copy.deepcopy(result)

        pool_size = max(1, int(self.config.boundary_coordinate_pool_size))
        audit_rng = np.random.default_rng(
            int(self.config.seed) + 8_300_003)
        try:
            pool = self.problem.boundary_excitation_candidates(
                n=pool_size,
                rng=audit_rng,
                pool_size=pool_size,
                include_source_templates=False,
            )
        except TypeError:
            pool = self.problem.boundary_excitation_candidates(
                n=pool_size,
                rng=audit_rng,
                pool_size=pool_size,
            )
        pool = [tuple(int(v) for v in x) for x in unique_candidates(pool)]
        diagnostics = self._truth_pool_diagnostics(
            pool,
            prefix="boundary_raw_pool",
        )
        basis_map = getattr(self.gpr[1], "basis_map", None)
        basis_diagnostics = (
            basis_map.diagnostics()
            if basis_map is not None and hasattr(basis_map, "diagnostics")
            else {}
        )
        result = {
            "status": (
                "audited"
                if diagnostics.get(
                    "boundary_raw_pool_truth_diagnostics_available", False)
                else "truth_unavailable"
            ),
            "pool_size": int(len(pool)),
            "pool_contract": "universal_low_frequency_no_source_templates",
            "coordinate_output_mode": basis_diagnostics.get("output_mode"),
            "coordinate_input_mode": (
                basis_diagnostics.get("input_mode")
                or basis_diagnostics.get("mean_coordinate_input")
            ),
            "separate_mean_variance_heads": bool(
                basis_diagnostics.get("separate_mean_variance_heads", False)),
            "post_run_only": True,
            "target_truth_used_for_audit": True,
            "target_oracle_used_for_decision": False,
            **diagnostics,
        }
        self._boundary_raw_pool_audit_cache = copy.deepcopy(result)
        return result

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

            row["task_posterior_before"] = (
                None
                if self.task_ensemble is None
                else self.task_ensemble.posterior.diagnostics()
            )

            row["sequential_basis_refresh"] = self._refresh_sequential_basis()

            t0 = time.time()
            recheck_x, recheck_info = self._certification_recheck_candidate()
            candidates, candidate_sources = self._generate_candidates(iteration)
            if recheck_x is not None:
                recheck_x = tuple(int(v) for v in recheck_x)
                if recheck_x not in candidates:
                    candidates.append(recheck_x)
                candidate_sources[recheck_x] = "certification_recheck"
            base_candidate_time = time.time() - t0

            decision_backend_name = str(
                self.config.decision_backend or "legacy"
            ).strip().lower().replace("-", "_")
            exact_refit_backends = {
                "sobol_exact_joint_voi", "exact_joint_voi_sobol",
            }
            exact_kg_online = bool(
                decision_backend_name in {
                    "legacy", "legacy_kg", "exact_kg", "additive",
                } | exact_refit_backends
                and self._effective_exact_kg_mc_samples() > 0
            )
            terminal_pool = list(dict.fromkeys([
                tuple(int(v) for v in x)
                for x in self._recommendation_pool()
            ] + [
                tuple(int(v) for v in x) for x in candidates
            ] + [
                tuple(int(v) for v in x)
                for x in self._finalist_replication_targets
            ]))
            self._last_terminal_pool = list(terminal_pool)

            t0 = time.time()
            if (
                int(self.config.terminal_frontier_candidate_count) > 0
                or exact_kg_online
            ):
                rec_x, rec_details = self._solve_posterior_recommendation(
                    pool=terminal_pool,
                    terminal_frontier_count=(
                        self.config.terminal_frontier_candidate_count),
                )
                online_terminal_solve_skipped = False
            else:
                rec_x = None
                rec_details = {
                    "status": "skipped_no_terminal_frontier_action",
                    "terminal_frontier_labels": [],
                    "_terminal_frontier_candidates": [],
                    "target_oracle_used": False,
                }
                online_terminal_solve_skipped = True
            frontier_candidates = rec_details.pop(
                "_terminal_frontier_candidates", [])
            frontier_labels = list(rec_details.get(
                "terminal_frontier_labels", []))
            row["t_posterior_solve"] = time.time() - t0
            row["recommendation_before"] = (
                None if rec_x is None else list(map(int, rec_x)))
            row["online_terminal_solve_skipped"] = bool(
                online_terminal_solve_skipped)
            row["online_terminal_pool_deferred"] = False
            row.update({f"rec_{k}": v for k, v in rec_details.items()})

            terminal_frontier = {}
            for label, x in zip(frontier_labels, frontier_candidates):
                x = tuple(int(v) for v in x)
                terminal_frontier[x] = str(label)
                if x not in candidate_sources:
                    candidates.append(x)
                    candidate_sources[x] = (
                        "terminal_frontier_replication"
                        if x in self.observations
                        else "terminal_frontier"
                    )
            finalist_started = time.perf_counter()
            finalist_x, finalist_info = self._finalist_replication_candidate(
                n, terminal_pool)
            finalist_policy_time = time.perf_counter() - finalist_started
            for target in self._finalist_replication_targets:
                target = tuple(int(v) for v in target)
                if target not in terminal_pool:
                    terminal_pool.append(target)
            self._last_terminal_pool = list(terminal_pool)
            if finalist_x is not None:
                finalist_x = tuple(int(v) for v in finalist_x)
                if finalist_x not in candidates:
                    candidates.append(finalist_x)
                candidate_sources[finalist_x] = (
                    "terminal_replication_kg"
                    if finalist_info.get("status")
                    == "forced_terminal_replication_kg"
                    else "finalist_replication"
                )
            row["t_candidate_gen"] = float(base_candidate_time)
            row["n_candidates"] = len(candidates)
            row["terminal_pool_shared"] = True
            row["terminal_pool_size"] = int(len(terminal_pool))
            row["terminal_frontier_candidate_count"] = int(
                len(terminal_frontier))
            row["terminal_frontier_candidates_in_action_set"] = int(sum(
                x in candidate_sources for x in terminal_frontier
            ))
            row["certification_recheck"] = copy.deepcopy(recheck_info)
            row["finalist_replication"] = copy.deepcopy(finalist_info)
            row["t_finalist_replication_policy"] = float(
                finalist_policy_time)
            row["llm_prior"] = dict(self._last_llm_prior_info)
            row["task_expert_proposals"] = copy.deepcopy(
                self._last_task_proposal_info)
            row["boundary_coordinate_proposal"] = copy.deepcopy(
                self._last_boundary_coordinate_proposal_info)

            t0 = time.time()
            score = self.acquisition.score(
                candidates,
                self.gpr[0],
                self.gpr[1],
                self.variance_model,
                self.problem,
                observed=self.history,
            )
            decision_backend = str(
                self.config.decision_backend or "legacy"
            ).strip().lower().replace("-", "_")
            legacy_backends = {
                "legacy", "legacy_kg", "exact_kg", "additive",
            }
            backend_score = None
            if decision_backend not in legacy_backends:
                backend_score = score_decision_backend(
                    decision_backend,
                    candidates,
                    self.gpr[0],
                    self.gpr[1],
                    self.variance_model,
                    self.problem,
                    observed=self.history,
                    task_ensemble=self.task_ensemble,
                    rng=self.rng,
                    iteration=iteration,
                    seed=(
                        int(self.config.seed)
                        + int(self.config.decision_backend_seed_offset)
                    ),
                    risk_penalty=self.config.decision_risk_penalty,
                    decision_aleatoric_mode=(
                        self.config.decision_aleatoric_mode),
                    violation_loss_mode=(
                        self.config.decision_violation_loss_mode),
                    decision_ambiguity_mode=(
                        self.config.decision_ambiguity_mode),
                    source_utility_weight=(
                        self.config.decision_source_utility_weight),
                    replication_max_per_solution=(
                        self.config.replication_max_per_solution),
                    certification_beta_g=self.config.beta_g,
                    robust_certificate_mode=(
                        self.config.task_posterior_robust_certificate_mode),
                    canonical_sobol_candidate=(
                        self._last_canonical_sobol_candidate),
                    allow_replication_actions=bool(
                        self.config.adaptive_replication_voi),
                    evaluate_or_replicate_new_action_count=(
                        self.config.evaluate_or_replicate_new_action_count),
                    evaluate_or_replicate_new_action_policy=(
                        self.config.evaluate_or_replicate_new_action_policy),
                    evaluate_or_replicate_baseline_new_action_count=(
                        None
                        if int(self.config.
                            evaluate_or_replicate_baseline_new_action_count)
                        <= 0
                        else int(self.config.
                            evaluate_or_replicate_baseline_new_action_count)
                    ),
                )
                score["total"] = backend_score["total"]
            row["decision_backend"] = decision_backend
            row["decision_backend_forced_override"] = bool(
                recheck_x is not None or finalist_x is not None)
            exact_mc_samples = (
                self._effective_exact_kg_mc_samples()
                if decision_backend in {
                    "legacy", "legacy_kg", "exact_kg",
                } | exact_refit_backends
                else 0
            )
            acquisition_mode = str(self.config.acquisition_mode or "additive").lower()
            forced_selection = (
                recheck_x if recheck_x is not None else finalist_x)
            policy_improvement_selected_idx = None
            if exact_mc_samples > 0 and forced_selection is None:
                if decision_backend in exact_refit_backends:
                    active_indices = backend_score.get(
                        "evaluate_or_replicate_active_indices")
                    if active_indices is None:
                        raise RuntimeError(
                            "exact evaluate-or-replicate backend did not "
                            "declare its active action set")
                    exact_kg = self._exact_posterior_update_scores_for_actions(
                        candidates, terminal_pool, active_indices)
                    row["exact_kg_action_scope"] = (
                        str(self.config.
                            evaluate_or_replicate_new_action_policy)
                        + "_new_plus_eligible_replicates"
                    )
                    row["exact_kg_active_action_count"] = int(
                        len(np.asarray(active_indices).reshape(-1)))
                    row["exact_kg_full_posterior_refit"] = True
                    row["exact_kg_refits_gpr_hc3_hvd"] = True
                else:
                    exact_kg = self._exact_posterior_update_scores(
                        candidates, terminal_pool)
                    row["exact_kg_action_scope"] = "all_candidates"
                    row["exact_kg_active_action_count"] = int(len(candidates))
                    row["exact_kg_full_posterior_refit"] = True
                    row["exact_kg_refits_gpr_hc3_hvd"] = True
                score["exact_kg"] = exact_kg
                row["exact_kg_mc_samples"] = int(exact_mc_samples)
                row["exact_kg_jobs"] = int(max(1, self.config.exact_kg_jobs))
                row["exact_kg_parallel_backend"] = (
                    str(self.config.exact_kg_parallel_backend)
                    if int(self.config.exact_kg_jobs) > 1 and len(candidates) > 1
                    else "serial"
                )
                row["exact_kg_parallel_workers_effective"] = int(getattr(
                    self, "_last_exact_kg_parallel_workers", 1))
                row["exact_kg_chunks_per_candidate"] = int(getattr(
                    self, "_last_exact_kg_chunks_per_candidate", 1))
                row["acquisition_mode"] = acquisition_mode
                row["exact_kg_sampling_mode"] = str(
                    self.config.exact_kg_sampling_mode)
                row["exact_kg_clip_negative"] = bool(
                    self.config.exact_kg_clip_negative)
                row["exact_kg_terminal_mode"] = str(
                    self.config.exact_kg_terminal_mode)
                row["exact_kg_terminal_mode_effective"] = str(
                    self._effective_exact_terminal_mode())
                row["exact_kg_terminal_value_contract"] = str(getattr(
                    self,
                    "_last_exact_kg_terminal_value_contract",
                    self._terminal_value_contract_id(),
                ))
                row["exact_kg_current_terminal_action_pool_size"] = int(
                    getattr(
                        self,
                        "_last_exact_kg_current_terminal_action_pool_size",
                        0,
                    ))
                row["exact_kg_terminal_observed_only"] = bool(
                    self._decision_backend_observed_terminal_active())
                row["exact_kg_certification_head_authority"] = str(
                    self._last_exact_kg_certification_head_authority)
                row["exact_kg_constraint_posterior_source"] = str(
                    self._last_exact_kg_constraint_posterior_source)
                row["decision_contract_mode"] = str(
                    self._decision_contract_mode())
                raw_exact_kg = np.asarray(getattr(
                    self,
                    "_last_exact_kg_raw_scores",
                    exact_kg,
                ), dtype=float)
                active_exact = np.isfinite(raw_exact_kg)
                raw_active = raw_exact_kg[active_exact]
                score_active = np.asarray(exact_kg, dtype=float)[active_exact]
                row["exact_kg_raw_min"] = float(np.min(raw_active))
                row["exact_kg_raw_max"] = float(np.max(raw_active))
                row["exact_kg_raw_negative_fraction"] = float(np.mean(
                    raw_active < 0.0))
                row["exact_kg_zero_fraction"] = float(np.mean(
                    score_active == 0.0))
                if decision_backend in exact_refit_backends:
                    exact_active_indices = np.asarray(
                        self._last_exact_kg_active_indices, dtype=int)
                    row["exact_kg_raw_scores_active"] = (
                        raw_exact_kg[exact_active_indices].tolist())
                    row["exact_kg_active_action_fingerprints"] = [
                        integer_design_fingerprint([
                            tuple(int(value) for value in candidates[index])
                        ])
                        for index in exact_active_indices
                    ]
                    action_is_replicate = np.asarray(
                        backend_score.get("hvd_action_is_replicate"),
                        dtype=bool,
                    )
                    active_is_replicate = action_is_replicate[
                        exact_active_indices]
                    active_raw_scores = raw_exact_kg[exact_active_indices]
                    row["exact_kg_active_action_is_replicate"] = (
                        active_is_replicate.tolist())
                    new_raw = active_raw_scores[~active_is_replicate]
                    replicate_raw = active_raw_scores[active_is_replicate]
                    row["exact_kg_best_new_raw"] = (
                        None if len(new_raw) == 0
                        else float(np.max(new_raw)))
                    row["exact_kg_best_replication_raw"] = (
                        None if len(replicate_raw) == 0
                        else float(np.max(replicate_raw)))
                    row["exact_kg_new_minus_replication_raw"] = (
                        None
                        if len(new_raw) == 0 or len(replicate_raw) == 0
                        else float(np.max(new_raw) - np.max(replicate_raw))
                    )
                row["certified_terminal_value_before"] = copy.deepcopy(
                    getattr(
                        self,
                        "_last_exact_kg_current_value",
                        np.nan,
                    ))
                if hasattr(self, "_last_exact_kg_expected_values"):
                    expected_values = np.asarray(
                        self._last_exact_kg_expected_values, dtype=float)
                    if decision_backend in exact_refit_backends:
                        active_indices = np.asarray(
                            self._last_exact_kg_active_indices, dtype=int)
                        row["exact_kg_active_indices"] = (
                            active_indices.tolist())
                        row["exact_kg_expected_terminal_values_active"] = (
                            expected_values[active_indices].tolist())
                    else:
                        row["exact_kg_expected_terminal_values"] = (
                            expected_values.tolist())
                if hasattr(self, "_last_exact_kg_component_gains"):
                    component_gains = np.asarray(
                        self._last_exact_kg_component_gains, dtype=float)
                    if decision_backend in exact_refit_backends:
                        row["exact_kg_component_gains_active"] = (
                            component_gains[
                                self._last_exact_kg_active_indices
                            ].tolist())
                    else:
                        row["exact_kg_component_gains"] = (
                            component_gains.tolist())
                blend = 0.0
                if (
                    decision_backend in exact_refit_backends
                    or acquisition_mode == "exact_mc"
                    or self.config.exact_kg_use_score
                ):
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
                if (
                    decision_backend in exact_refit_backends
                    and self._policy_improvement_mode() != "off"
                ):
                    one_step_index, one_step_info = (
                        self._guarded_one_step_policy_improvement(
                            candidates,
                            exact_kg,
                            backend_score,
                        )
                    )
                    rollout_index, rollout_info = (
                        self._guarded_rollout_policy_improvement(
                            candidates,
                            terminal_pool,
                            exact_kg,
                            backend_score[
                                "evaluate_or_replicate_active_indices"],
                            one_step_index,
                            stage=n,
                        )
                    )
                    policy_improvement_selected_idx = int(rollout_index)
                    row["policy_improvement_one_step"] = one_step_info
                    row["policy_improvement_rollout"] = rollout_info
                    row["policy_improvement_selected_index"] = int(
                        policy_improvement_selected_idx)
                    row["policy_improvement_contract_id"] = (
                        "v52_safeguarded_policy_improvement_v1"
                    )
            elif exact_mc_samples > 0:
                row["exact_kg_skipped_reason"] = (
                    "forced_certification_recheck"
                    if recheck_x is not None
                    else (
                        "terminal_replication_kg_already_computed"
                        if finalist_info.get("status")
                        == "forced_terminal_replication_kg"
                        else "forced_finalist_replication"
                    )
                )
            if recheck_x is None:
                if finalist_x is None:
                    if policy_improvement_selected_idx is None:
                        selected_idx = int(np.argmax(score["total"]))
                        row["selection_policy"] = "acquisition"
                    else:
                        selected_idx = int(policy_improvement_selected_idx)
                        row["selection_policy"] = (
                            "safeguarded_policy_improvement"
                        )
                else:
                    selected_idx = candidates.index(finalist_x)
                    row["selection_policy"] = (
                        "terminal_replication_kg"
                        if finalist_info.get("status")
                        == "forced_terminal_replication_kg"
                        else "finalist_replication"
                    )
            else:
                selected_idx = candidates.index(recheck_x)
                row["selection_policy"] = "certification_recheck"
            x_selected = candidates[selected_idx]
            row["t_kg_compute"] = (
                time.time() - t0 + float(finalist_policy_time))
            row["x_selected"] = list(map(int, x_selected))
            row["candidate_source_selected"] = candidate_sources.get(
                tuple(x_selected), "unknown")
            row["terminal_frontier_selected"] = bool(
                tuple(x_selected) in terminal_frontier)
            row["terminal_frontier_selected_label"] = terminal_frontier.get(
                tuple(x_selected))
            row["score_selected"] = float(score["total"][selected_idx])
            if backend_score is not None:
                backend_fields = (
                    "objective_mean",
                    "objective_epistemic",
                    "constraint_mean",
                    "constraint_epistemic",
                    "constraint_aleatoric",
                    "constraint_between_expert",
                    "stochastic_margin_mean",
                    "expected_violation",
                    "probability_violation",
                    "violation_loss",
                    "probability_feasible",
                    "theory_margin",
                    "bayes_risk",
                    "bayes_risk_ei",
                    "constrained_ei",
                    "transfer_utility",
                    "hvd_information_reduction",
                    "constraint_epistemic_information_reduction",
                    "hvd_margin_information_reduction",
                    "joint_information_reduction",
                    "hvd_action_reliability",
                    "hvd_action_is_replicate",
                    "risk_coordinate_coverage",
                )
                for field in backend_fields:
                    values = backend_score.get(field)
                    if values is not None:
                        row[f"decision_{field}_selected"] = float(
                            np.asarray(values, dtype=float)[selected_idx])
                row["decision_posterior_source"] = backend_score.get(
                    "posterior_source")
                row["decision_incumbent_bayes_risk"] = float(
                    backend_score["incumbent_bayes_risk"])
                row["decision_sampled_expert"] = backend_score.get(
                    "sampled_expert")
                row["decision_transfer_utility_status"] = backend_score.get(
                    "transfer_utility_status")
                row["decision_joint_information_unit"] = backend_score.get(
                    "joint_information_unit")
                row["decision_joint_information_contract"] = backend_score.get(
                    "joint_information_contract")
                row["decision_hvd_sobol_new_index"] = backend_score.get(
                    "hvd_sobol_new_index")
                row["decision_canonical_sobol_index"] = backend_score.get(
                    "canonical_sobol_index")
                row["decision_canonical_sobol_injected"] = bool(
                    backend_score.get("canonical_sobol_injected", False))
                row["decision_evaluate_or_replicate_active_count"] = (
                    backend_score.get(
                        "evaluate_or_replicate_active_count"))
                row["decision_evaluate_or_replicate_new_action_count"] = (
                    backend_score.get(
                        "evaluate_or_replicate_new_action_count"))
                row[
                    "decision_evaluate_or_replicate_replication_action_count"
                ] = backend_score.get(
                    "evaluate_or_replicate_replication_action_count")
                row["decision_evaluate_or_replicate_new_action_policy"] = (
                    backend_score.get(
                        "evaluate_or_replicate_new_action_policy"))
                row["decision_evaluate_or_replicate_new_action_indices"] = (
                    None
                    if backend_score.get(
                        "evaluate_or_replicate_new_action_indices") is None
                    else np.asarray(backend_score[
                        "evaluate_or_replicate_new_action_indices"
                    ], dtype=int).tolist()
                )
                for field in (
                    "evaluate_or_replicate_baseline_indices",
                    "evaluate_or_replicate_supplemental_indices",
                ):
                    values = backend_score.get(field)
                    row[f"decision_{field}"] = (
                        None
                        if values is None
                        else np.asarray(values, dtype=int).tolist()
                    )
                row[
                    "decision_evaluate_or_replicate_supplemental_labels"
                ] = list(backend_score.get(
                    "evaluate_or_replicate_supplemental_labels", []))
                row["decision_risk_coordinate_coverage_source"] = (
                    backend_score.get("risk_coordinate_coverage_source"))
                row["decision_evaluate_or_replicate_exact_refit_required"] = (
                    bool(backend_score.get(
                        "evaluate_or_replicate_exact_refit_required", False))
                )
            if finalist_info.get("terminal_kg_selected_gain") is not None:
                row["terminal_kg_selected_gain"] = float(
                    finalist_info["terminal_kg_selected_gain"])
            if "exact_kg" in score:
                raw_exact_kg = np.asarray(
                    self._last_exact_kg_raw_scores, dtype=float)
                row["exact_kg_raw_selected"] = float(
                    raw_exact_kg[selected_idx])
            row.update(self._truth_pool_diagnostics(
                candidates,
                selected=x_selected,
                prefix="candidate",
                sources=candidate_sources,
            ))
            row.update(self._truth_acquisition_score_audit(
                candidates,
                score["total"],
                selected_idx,
            ))
            selected_authority = self._certification_head_authority()
            selected_robust = None
            if (
                self.task_ensemble is None
                or selected_authority == "split_gpr_cumulative_hvd"
            ):
                row["v_C_plus_selected"] = float(
                    self.variance_model.predict_certification_variance(
                        1,
                        x_selected,
                        self.problem,
                    )
                )
            else:
                selected_robust = self.task_ensemble.robust_moments_many(
                    1, [x_selected], certification=True)
                row["v_C_plus_selected"] = float(
                    selected_robust.aleatoric_upper[0])
                row["task_between_mean_variance_selected"] = float(
                    selected_robust.nominal.between_mean[0])
                row["task_robust_epistemic_selected"] = float(
                    selected_robust.epistemic_upper[0])
            selected_decomp = self.variance_model.predict_decomposition(
                1,
                x_selected,
                self.problem,
            )
            selected_cumulative = selected_decomp.get("cumulative") or {}
            row["v_C_plus_source"] = (
                "task_posterior_robust_cumulative"
                if (
                    self.task_ensemble is not None
                    and selected_authority in {
                        "task_joint", "split_gpr_task_hvd"
                    }
                )
                else (
                    "provider_cumulative"
                    if selected_cumulative.get("provider_active")
                    else "fallback_hvd"
                )
            )
            row["certification_head_authority"] = selected_authority
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
                task_entropy_gain = getattr(
                    self, "_last_exact_kg_task_entropy_gain", None)
                task_weight_movement = getattr(
                    self, "_last_exact_kg_task_weight_movement", None)
                if task_entropy_gain is not None:
                    row["exact_kg_task_entropy_gain_selected"] = float(
                        task_entropy_gain[selected_idx])
                    row["exact_kg_task_weight_movement_selected"] = float(
                        task_weight_movement[selected_idx])
                task_timing = getattr(
                    self, "_last_exact_kg_task_timing", None)
                if task_timing is not None:
                    for name, values in task_timing.items():
                        row[f"exact_kg_time_{name}_selected"] = float(
                            values[selected_idx])
                        row[f"exact_kg_time_{name}_mean"] = float(
                            np.mean(values))

            x_arr = np.asarray(x_selected, dtype=int)
            replicate_count_before = int(len(self.observations.get(
                tuple(int(v) for v in x_selected), [])))
            row["action_kind"] = (
                "replicate" if replicate_count_before > 0 else "new"
            )
            if "exact_kg" in score:
                selected_raw_gain = float(
                    self._last_exact_kg_raw_scores[selected_idx])
                row["exact_kg_selected_is_replicate"] = bool(
                    row["action_kind"] == "replicate")
                row["exact_kg_selected_nonpositive"] = bool(
                    selected_raw_gain <= 0.0)
                row["exact_kg_selected_clipped_to_zero"] = bool(
                    self.config.exact_kg_clip_negative
                    and selected_raw_gain < 0.0
                )
            row["replicate_count_before"] = replicate_count_before
            row["adaptive_replication_voi_enabled"] = bool(
                self.config.adaptive_replication_voi)
            row["evaluation_cost"] = 1.0
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
            if str(
                self.config.source_constraint_mean_adaptation_mode
            ).strip().lower() in {
                "sequential_evidence_mixture", "sequential_mixture",
            }:
                self._configure_hvd_source_task_posterior(
                    self.variance_model, self.gpr)
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
                    replicate_count=len(replicate_values),
                ))
            task_update = None
            if self.task_ensemble is not None:
                stored = self.observations.get(
                    tuple(int(v) for v in x_selected), [])
                task_update = self.task_ensemble.update(
                    x_arr,
                    y,
                    existing_observations=stored[:-1],
                    tau=self.problem.tau,
                )
            row["t_update"] = time.time() - t0
            row["hvd_update"] = hvd_details
            row["task_posterior_update"] = task_update
            row["task_posterior_after"] = (
                None
                if self.task_ensemble is None
                else self.task_ensemble.posterior.diagnostics()
            )
            row["posterior_dominance_update"] = (
                self._update_posterior_dominance_incumbent(
                    stage=n + 1,
                    reason="budgeted_target_observation",
                )
            )
            row["n_visited"] = len(self.gpr[0].sampled_set)

            t0 = time.time()
            eval_interval = int(self.config.evaluate_interval)
            if eval_interval > 0 and (
                iteration % eval_interval == 0 or n == self.config.N - 1
            ):
                rec_x_after, rec_after = self._solve_posterior_recommendation(
                    pool=terminal_pool)
                eval_after = self._evaluate_recommendation(rec_x_after)
                row["recommendation_after"] = list(map(int, rec_x_after))
                row["eval"] = {**rec_after, **eval_after}
            row["t_eval"] = time.time() - t0
            row["t_total_pre_checkpoint"] = time.time() - t_iter
            attributed = sum(float(row.get(name, 0.0) or 0.0) for name in (
                "t_candidate_gen",
                "t_kg_compute",
                "t_simulate",
                "t_update",
                "t_eval",
            ))
            row["t_unattributed_pre_checkpoint"] = max(
                0.0, row["t_total_pre_checkpoint"] - attributed)
            row["t_checkpoint"] = None
            self.iteration_log.append(row)
            checkpoint_started = time.time()
            checkpoint_path = self._save_checkpoint(
                n + 1, reason="iteration")
            row["t_checkpoint"] = time.time() - checkpoint_started
            row["checkpoint_saved"] = checkpoint_path is not None
            row["t_total"] = time.time() - t_iter
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
                    f"eval={row['t_eval']:.3f}s "
                    f"other={row['t_unattributed_pre_checkpoint']:.3f}s "
                    f"ckpt={row['t_checkpoint']:.3f}s"
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
        exact_rows = [
            row for row in self.iteration_log
            if row.get("exact_kg_mc_samples") is not None
        ]
        exact_kg_summary = {
            "sampling_mode": str(self.config.exact_kg_sampling_mode),
            "nested_common_random_numbers": bool(
                str(self.config.exact_kg_sampling_mode).lower() in {
                    "antithetic_nested",
                    "nested_antithetic",
                    "paired_nested",
                }
            ),
            "mc_samples": int(self._effective_exact_kg_mc_samples()),
            "clip_negative": bool(self.config.exact_kg_clip_negative),
            "ranking_uses_signed_values": bool(
                not self.config.exact_kg_clip_negative),
            "n_iterations": int(len(exact_rows)),
            "mean_raw_negative_fraction": (
                float(np.mean([
                    row["exact_kg_raw_negative_fraction"]
                    for row in exact_rows
                ]))
                if exact_rows
                else None
            ),
            "mean_zero_fraction": (
                float(np.mean([
                    row["exact_kg_zero_fraction"] for row in exact_rows
                ]))
                if exact_rows
                else None
            ),
            "mean_raw_selected": (
                float(np.mean([
                    row["exact_kg_raw_selected"] for row in exact_rows
                ]))
                if exact_rows
                else None
            ),
            "selected_nonpositive_count": int(sum(bool(
                row.get("exact_kg_selected_nonpositive", False)
            ) for row in exact_rows)),
            "selected_nonpositive_replication_count": int(sum(bool(
                row.get("exact_kg_selected_nonpositive", False)
                and row.get("exact_kg_selected_is_replicate", False)
            ) for row in exact_rows)),
            "selected_clipped_replication_count": int(sum(bool(
                row.get("exact_kg_selected_clipped_to_zero", False)
                and row.get("exact_kg_selected_is_replicate", False)
            ) for row in exact_rows)),
        }
        decision_backend_rows = [
            row for row in self.iteration_log
            if row.get("decision_backend") is not None
        ]
        decision_backend_summary = {
            "configured": str(self.config.decision_backend),
            "joint_information_units": sorted({
                str(row["decision_joint_information_unit"])
                for row in decision_backend_rows
                if row.get("decision_joint_information_unit") is not None
            }),
            "joint_information_contracts": sorted({
                str(row["decision_joint_information_contract"])
                for row in decision_backend_rows
                if row.get("decision_joint_information_contract") is not None
            }),
            "effective_counts": {
                name: int(sum(
                    row.get("decision_backend") == name
                    for row in decision_backend_rows
                ))
                for name in sorted({
                    str(row.get("decision_backend"))
                    for row in decision_backend_rows
                })
            },
            "forced_override_count": int(sum(
                bool(row.get("decision_backend_forced_override", False))
                for row in decision_backend_rows
            )),
            "mean_selected_expected_violation": (
                float(np.mean([
                    row["decision_expected_violation_selected"]
                    for row in decision_backend_rows
                    if row.get("decision_expected_violation_selected") is not None
                ]))
                if any(
                    row.get("decision_expected_violation_selected") is not None
                    for row in decision_backend_rows
                )
                else None
            ),
            "mean_selected_constraint_epistemic_information_reduction": (
                float(np.mean([
                    row[
                        "decision_constraint_epistemic_information_reduction_selected"
                    ]
                    for row in decision_backend_rows
                    if row.get(
                        "decision_constraint_epistemic_information_reduction_selected"
                    ) is not None
                ]))
                if any(
                    row.get(
                        "decision_constraint_epistemic_information_reduction_selected"
                    ) is not None
                    for row in decision_backend_rows
                )
                else None
            ),
            "mean_selected_hvd_margin_information_reduction": (
                float(np.mean([
                    row["decision_hvd_margin_information_reduction_selected"]
                    for row in decision_backend_rows
                    if row.get(
                        "decision_hvd_margin_information_reduction_selected"
                    ) is not None
                ]))
                if any(
                    row.get(
                        "decision_hvd_margin_information_reduction_selected"
                    ) is not None
                    for row in decision_backend_rows
                )
                else None
            ),
            "mean_selected_joint_information_reduction": (
                float(np.mean([
                    row["decision_joint_information_reduction_selected"]
                    for row in decision_backend_rows
                    if row.get(
                        "decision_joint_information_reduction_selected"
                    ) is not None
                ]))
                if any(
                    row.get(
                        "decision_joint_information_reduction_selected"
                    ) is not None
                    for row in decision_backend_rows
                )
                else None
            ),
            "target_oracle_used": False,
        }
        policy_rows = [
            row for row in decision_backend_rows
            if row.get("policy_improvement_one_step") is not None
        ]
        one_step_rows = [
            dict(row.get("policy_improvement_one_step") or {})
            for row in policy_rows
        ]
        rollout_rows = [
            dict(row.get("policy_improvement_rollout") or {})
            for row in policy_rows
        ]
        decision_backend_summary["policy_improvement"] = {
            "mode": self._policy_improvement_mode(),
            "contract_id": (
                "v52_safeguarded_policy_improvement_v1"
                if self._policy_improvement_mode() != "off"
                else "disabled_v51_compatible"
            ),
            "iteration_count": int(len(policy_rows)),
            "one_step_switch_count": int(sum(bool(
                row.get("switched", False)) for row in one_step_rows)),
            "one_step_guard_fallback_count": int(sum(
                row.get("status") == "baseline_mc_guard"
                for row in one_step_rows
            )),
            "rollout_switch_count": int(sum(bool(
                row.get("switched", False)) for row in rollout_rows)),
            "rollout_guard_fallback_count": int(sum(
                row.get("status") == "fallback_mc_guard"
                for row in rollout_rows
            )),
            "mean_one_step_estimated_advantage": (
                float(np.mean([
                    float(row["estimated_advantage"])
                    for row in one_step_rows
                    if row.get("estimated_advantage") is not None
                ]))
                if any(row.get("estimated_advantage") is not None
                       for row in one_step_rows)
                else None
            ),
            "mean_rollout_estimated_advantage": (
                float(np.mean([
                    float(row["estimated_advantage_over_fallback"])
                    for row in rollout_rows
                    if row.get("estimated_advantage_over_fallback") is not None
                ]))
                if any(row.get("estimated_advantage_over_fallback") is not None
                       for row in rollout_rows)
                else None
            ),
            "conditional_posterior_noninferiority_only": True,
            "target_oracle_used": False,
        }
        adaptive_replication_summary = {
            "enabled": bool(self.config.adaptive_replication_voi),
            "candidate_budget": int(self._replication_candidate_budget()),
            "maximum_replicates_per_solution": int(max(
                1, self.config.replication_max_per_solution)),
            "action_space": "new_and_observed_points",
            "value": (
                "exact_refit_posterior_bayes_risk_reduction_per_unit_cost"
                if str(self.config.decision_backend).lower().replace(
                    "-", "_") in {
                        "sobol_exact_joint_voi", "exact_joint_voi_sobol",
                    }
                else
                "chance_margin_joint_epistemic_hvd_reduction_per_unit_cost"
                if str(self.config.decision_backend).lower().replace(
                    "-", "_") in {"sobol_joint_voi", "joint_voi_sobol"}
                else "boundary_weighted_hvd_information_reduction_per_unit_cost"
                if str(self.config.decision_backend).lower().replace(
                    "-", "_") in {"sobol_hvd_voi", "hvd_voi_sobol"}
                else "exact_posterior_bayes_risk_reduction_per_unit_cost"
            ),
            "unit_evaluation_cost": 1.0,
            "selected_replication_count": int(sum(
                row.get("action_kind") == "replicate"
                for row in self.iteration_log
            )),
            "selected_new_point_count": int(sum(
                row.get("action_kind") == "new"
                for row in self.iteration_log
            )),
            "forced_recheck_count": int(sum(
                row.get("selection_policy") == "certification_recheck"
                for row in self.iteration_log
            )),
            "unified_exact_voi": bool(
                self.config.adaptive_replication_voi
                and (
                    str(self.config.decision_backend).lower().replace(
                        "-", "_") in {
                            "sobol_exact_joint_voi",
                            "exact_joint_voi_sobol",
                        }
                    or (
                        str(self.config.decision_backend).lower()
                        in {"legacy", "legacy_kg", "exact_kg"}
                        and str(self.config.acquisition_mode).lower()
                        == "exact_mc"
                    )
                )
            ),
            "unified_hvd_voi": bool(
                self.config.adaptive_replication_voi
                and str(self.config.decision_backend).lower().replace(
                    "-", "_") in {
                        "sobol_hvd_voi", "hvd_voi_sobol",
                        "sobol_joint_voi", "joint_voi_sobol",
                        "sobol_exact_joint_voi", "exact_joint_voi_sobol",
                    }
            ),
            "unified_joint_margin_voi": bool(
                self.config.adaptive_replication_voi
                and str(self.config.decision_backend).lower().replace(
                    "-", "_") in {
                        "sobol_joint_voi", "joint_voi_sobol",
                        "sobol_exact_joint_voi", "exact_joint_voi_sobol",
                    }
            ),
            "exact_refit_action_value": bool(
                self.config.adaptive_replication_voi
                and str(self.config.decision_backend).lower().replace(
                    "-", "_") in {
                        "sobol_exact_joint_voi", "exact_joint_voi_sobol",
                    }
            ),
            "target_oracle_used": False,
        }
        finalist_replication_summary = {
            "enabled": bool(self.config.finalist_replication_budget > 0),
            "policy": self._finalist_replication_policy(),
            "frontier_policy": self._finalist_frontier_policy(),
            "decision_contract_mode": self._decision_contract_mode(),
            "coherent_three_layer_contract": bool(
                self._coherent_certificate_contract()),
            "effective_exact_terminal_mode": (
                self._effective_exact_terminal_mode()),
            "effective_finalist_terminal_value_mode": (
                self._finalist_terminal_value_mode()),
            "terminal_max_arms": int(max(
                1, self.config.finalist_terminal_max_arms)),
            "empirical_override_policy": (
                self._finalist_empirical_override_policy()),
            "reserved_budget": int(max(
                0, self.config.finalist_replication_budget)),
            "initialized": bool(self._finalist_replication_initialized),
            "frozen_stage": self._finalist_replication_frozen_stage,
            "adaptive_race": bool(
                self.config.finalist_replication_adaptive_race),
            "fixed_universe": bool(
                self.config.finalist_replication_fixed_universe),
            "fixed_universe_size": int(len(
                self._finalist_replication_pool)),
            "active_target": (
                None
                if self._finalist_replication_active_target is None
                else list(map(
                    int, self._finalist_replication_active_target))
            ),
            "active_label": self._finalist_replication_active_label,
            "refresh_history": copy.deepcopy(
                self._finalist_replication_refresh_history),
            "minimum_replicates": int(max(
                1, self.config.finalist_replication_min_replicates)),
            "targets": [
                list(map(int, x))
                for x in self._finalist_replication_targets
            ],
            "labels": list(self._finalist_replication_labels),
            "replicate_counts": [
                int(len(self.observations.get(tuple(x), [])))
                for x in self._finalist_replication_targets
            ],
            "completed_target_count": int(sum(
                len(self.observations.get(tuple(x), [])) >= max(
                    1, self.config.finalist_replication_min_replicates)
                for x in self._finalist_replication_targets
            )),
            "statistics": [
                self._replicated_finalist_statistics(x)
                for x in self._finalist_replication_targets
            ],
            "forced_evaluations": int(sum(
                row.get("selection_policy") in {
                    "finalist_replication",
                    "terminal_replication_kg",
                }
                for row in self.iteration_log
            )),
            "terminal_kg_evaluations": int(sum(
                row.get("selection_policy") == "terminal_replication_kg"
                for row in self.iteration_log
            )),
            "terminal_kg_rows": [
                copy.deepcopy(row.get("finalist_replication"))
                for row in self.iteration_log
                if row.get("selection_policy") == "terminal_replication_kg"
            ],
            "target_oracle_used": False,
        }
        no_forced_terminal_stage = int(
            self.config.finalist_replication_budget) <= 0
        terminal_stage_uses_same_value = bool(
            no_forced_terminal_stage
            or self._terminal_replication_policy_active())
        empirical_override_disabled = bool(
            self._finalist_empirical_override_policy() == "off")
        finalist_replication_summary.update({
            "sampling_terminal_contract_closed": bool(
                terminal_stage_uses_same_value),
            "recommendation_override_closed": bool(
                empirical_override_disabled),
            "mathematically_closed": bool(
                self._coherent_certificate_contract()
                and terminal_stage_uses_same_value
                and empirical_override_disabled
                and self._effective_exact_terminal_mode()
                == "tcb_certified_lexicographic"
                and self._finalist_terminal_value_mode()
                == "certified_lexicographic"
            ),
            "closure_definition": (
                "exact acquisition, optional terminal replication, and final "
                "recommendation share the certified lexicographic value; no "
                "empirical recommendation override"
            ),
        })

        final_pool = (
            list(self._last_terminal_pool)
            if self._last_terminal_pool
            else self._recommendation_pool()
        )
        final_x, final_post = self._solve_posterior_recommendation(
            pool=final_pool)
        backend_terminal = self._decision_backend_terminal_recommendation(
            final_pool)
        if backend_terminal is not None:
            final_x, final_post = backend_terminal
        dominance_terminal = (
            self._posterior_dominance_terminal_recommendation())
        if dominance_terminal is not None:
            final_x, final_post = dominance_terminal
        task_meta_coherence = (
            None
            if self.task_ensemble is None
            else self.task_ensemble.meta_coherence_diagnostics(
                final_pool,
                tau=self.problem.tau,
                alpha=self.problem.alpha,
                beta_g=self.config.beta_g,
                algorithm_selected_x=final_x,
                robust_certificate_mode=(
                    self._task_robust_certificate_mode()),
            )
        )
        final_post["terminal_pool_shared"] = bool(
            self._last_terminal_pool)
        final_post["terminal_pool_size"] = int(len(final_pool))
        two_stage_decision = self._two_stage_decision_contract_summary(
            final_post, finalist_replication_summary)
        final_eval = self._evaluate_recommendation(final_x)
        adaptive_outcome = self._adaptive_outcome_audit(final_eval)
        certificate_outcome = self._certificate_outcome_audit()
        backend_name = str(
            self.config.decision_backend or "legacy"
        ).strip().lower().replace("-", "_")
        backend_forced_overrides = int(
            decision_backend_summary["forced_override_count"])
        decision_backend_contract = {
            "backend": backend_name,
            "decision_aleatoric_mode": str(
                self.config.decision_aleatoric_mode),
            "decision_violation_loss_mode": str(
                self.config.decision_violation_loss_mode),
            "decision_ambiguity_mode": str(
                self.config.decision_ambiguity_mode),
            "evaluate_or_replicate_new_action_count": int(
                self.config.evaluate_or_replicate_new_action_count),
            "evaluate_or_replicate_new_action_policy": str(
                self.config.evaluate_or_replicate_new_action_policy),
            "evaluate_or_replicate_baseline_new_action_count": int(
                self.config.evaluate_or_replicate_baseline_new_action_count),
            "policy_improvement_mode": self._policy_improvement_mode(),
            "policy_improvement_mc_error_bound": float(
                self.config.policy_improvement_mc_error_bound),
            "policy_improvement_rollout_depth": int(
                self.config.policy_improvement_rollout_depth),
            "policy_improvement_rollout_max_arms": int(
                self.config.policy_improvement_rollout_max_arms),
            "policy_improvement_rollout_mc_samples": int(
                self.config.policy_improvement_rollout_mc_samples),
            "policy_improvement_rollout_mc_error_bound": float(
                self.config.policy_improvement_rollout_mc_error_bound),
            "policy_improvement_contract": (
                "v52_safeguarded_policy_improvement_v1"
                if self._policy_improvement_mode() != "off"
                else "disabled_v51_compatible"
            ),
            "source_proposal_frozen_before_target": bool(
                self.config.initial_design == "source_informed"),
            "online_updates_use_budgeted_target_observations_only": True,
            "source_discrepancy_update": bool(
                self.config.source_discrepancy_update),
            "terminal_rule": (
                "posterior_dominance"
                if final_post.get(
                    "posterior_dominance_terminal_used", False)
                else (
                    "posterior_bayes_risk"
                    if final_post.get(
                        "decision_backend_terminal_used", False)
                    else str(self.config.exact_kg_terminal_mode)
                )
            ),
            "terminal_recommendation_observed_only": bool(
                self.config.decision_recommend_observed_only),
            "terminal_value_contract": self._terminal_value_contract_id(),
            "acquisition_terminal_observed_only": bool(
                self._decision_backend_observed_terminal_active()),
            "acquisition_and_recommendation_share_terminal_action_universe": (
                bool(self._decision_backend_observed_terminal_active())
                == bool(self.config.decision_recommend_observed_only)
            ),
            "acquisition_and_recommendation_share_risk_penalty": bool(
                self._decision_backend_observed_terminal_active()
                or abs(
                    float(self.config.terminal_bayes_violation_penalty)
                    - float(self.config.decision_risk_penalty)
                ) <= 1e-12
            ),
            "forced_sampling_override_count": backend_forced_overrides,
            "coherent": bool(
                (
                    final_post.get(
                        "posterior_dominance_terminal_used", False)
                    or final_post.get(
                        "decision_backend_terminal_used", False)
                )
                and backend_forced_overrides == 0
            ) if (
                self._posterior_dominance_active()
                or backend_name not in {
                    "legacy", "legacy_kg", "exact_kg", "additive"
                }
            ) else bool(finalist_replication_summary.get(
                "mathematically_closed", False)),
            "target_oracle_used": False,
        }
        # Join synthetic/oracle truth only after every charged decision is
        # frozen.  These fields support paper convergence plots and cannot
        # affect acquisition, posterior updates, or the terminal Bayes action.
        post_run_truth_available = bool(self.config.truth_pool_diagnostics)
        try:
            _, trace_true_best_objective = self._true_best_feasible_cached()
            trace_true_best_objective = float(trace_true_best_objective)
        except Exception:
            trace_true_best_objective = np.nan
            post_run_truth_available = False
        trace_incumbent_regret = None
        trace_initial_points = unique_candidates([
            x for x, _ in self.history[: int(self.config.n0)]
        ])
        if post_run_truth_available and np.isfinite(trace_true_best_objective):
            for initial_point in trace_initial_points:
                try:
                    initial_margin = float(
                        self._true_chance_margin(initial_point))
                    if initial_margin <= 0.0:
                        initial_regret = float(
                            self.problem.true_objective(initial_point)
                            - trace_true_best_objective
                        )
                        trace_incumbent_regret = (
                            initial_regret
                            if trace_incumbent_regret is None
                            else min(trace_incumbent_regret, initial_regret)
                        )
                except Exception:
                    post_run_truth_available = False
                    trace_incumbent_regret = None
                    break

        online_action_trace = []
        for trace_index, iteration_row in enumerate(self.iteration_log):
            selected = iteration_row.get("x_selected")
            if selected is None:
                continue
            point = tuple(int(value) for value in selected)
            observed = iteration_row.get("Y_observed")
            expert_proposals = dict(
                iteration_row.get("task_expert_proposals") or {})
            true_objective = None
            true_margin = None
            true_feasible = None
            feasible_regret = None
            if post_run_truth_available:
                try:
                    true_objective = float(self.problem.true_objective(point))
                    true_margin = float(self._true_chance_margin(point))
                    true_feasible = bool(true_margin <= 0.0)
                    if (
                        true_feasible
                        and np.isfinite(trace_true_best_objective)
                    ):
                        feasible_regret = float(
                            true_objective - trace_true_best_objective)
                        trace_incumbent_regret = (
                            feasible_regret
                            if trace_incumbent_regret is None
                            else min(trace_incumbent_regret, feasible_regret)
                        )
                except Exception:
                    post_run_truth_available = False
                    true_objective = None
                    true_margin = None
                    true_feasible = None
                    feasible_regret = None
                    trace_incumbent_regret = None
            online_action_trace.append({
                "iteration": int(iteration_row.get("iteration", -1)),
                "target_call": int(self.config.n0) + trace_index + 1,
                "x_fingerprint": integer_design_fingerprint([point]),
                "x_first_coordinate": int(point[0]) if point else None,
                "x_coordinate_mean": (
                    float(np.mean(point)) if point else None
                ),
                "candidate_source": str(iteration_row.get(
                    "candidate_source_selected", "missing")),
                "action_kind": str(iteration_row.get(
                    "action_kind", "missing")),
                "n_candidates": int(iteration_row.get("n_candidates", 0)),
                "selected_score": float(iteration_row.get(
                    "score_selected", np.nan)),
                "decision_bayes_risk": float(iteration_row.get(
                    "decision_bayes_risk_selected", np.nan)),
                "decision_theory_margin": float(iteration_row.get(
                    "decision_theory_margin_selected", np.nan)),
                "decision_constraint_epistemic": float(iteration_row.get(
                    "decision_constraint_epistemic_selected", np.nan)),
                "active_new_action_count": iteration_row.get(
                    "decision_evaluate_or_replicate_new_action_count"),
                "active_replication_action_count": iteration_row.get(
                    "decision_evaluate_or_replicate_replication_action_count"),
                "new_action_policy": iteration_row.get(
                    "decision_evaluate_or_replicate_new_action_policy"),
                "exact_kg_raw_scores_active": copy.deepcopy(
                    iteration_row.get("exact_kg_raw_scores_active")),
                "exact_kg_active_action_fingerprints": copy.deepcopy(
                    iteration_row.get(
                        "exact_kg_active_action_fingerprints")),
                "exact_kg_active_action_is_replicate": copy.deepcopy(
                    iteration_row.get(
                        "exact_kg_active_action_is_replicate")),
                "exact_kg_best_new_raw": iteration_row.get(
                    "exact_kg_best_new_raw"),
                "exact_kg_best_replication_raw": iteration_row.get(
                    "exact_kg_best_replication_raw"),
                "exact_kg_new_minus_replication_raw": iteration_row.get(
                    "exact_kg_new_minus_replication_raw"),
                "task_expert_allocation": copy.deepcopy(
                    expert_proposals.get("allocation", {})),
                "task_expert_proposal_weights": copy.deepcopy(
                    expert_proposals.get("proposal_weights", {})),
                "observed_response": (
                    None
                    if observed is None
                    else [float(value) for value in observed]
                ),
                "true_objective_post_run": true_objective,
                "true_chance_margin_post_run": true_margin,
                "true_feasible_post_run": true_feasible,
                "feasible_regret_post_run": feasible_regret,
                "incumbent_feasible_regret_post_run": (
                    None
                    if trace_incumbent_regret is None
                    else float(trace_incumbent_regret)
                ),
                "truth_join_timing": "post_run_after_all_decisions_frozen",
                "truth_admissible_decision_input": False,
                "target_oracle_used_for_decision": False,
            })
        history_points = [tuple(int(value) for value in x) for x, _ in self.history]
        self.final_log = {
            **final_post,
            **final_eval,
            "implementation_contract_id": str(
                getattr(self.config, "implementation_contract_id", "unversioned")),
            "theory_contract_id": str(getattr(
                self.config, "theory_contract_id", "unversioned")),
            "theory_contract_timing": "declared_before_target_evaluation",
            "total_time_sec": float(time.time() - t_start),
            "n_simulations": int(len(self.history)),
            "n_distinct_solutions": int(len(self.gpr[0].sampled_set)),
            "stage_times": summarize_stage_times(self.iteration_log),
            "candidate_source_counts": candidate_source_counts,
            "target_design_fingerprint": (
                integer_design_fingerprint(history_points)
                if history_points else None
            ),
            "online_action_trace": online_action_trace,
            "online_action_trace_truth_available": bool(
                post_run_truth_available),
            "online_action_trace_initial_best_feasible_regret": (
                adaptive_outcome.get("initial_best_feasible_regret")
            ),
            "online_action_sequence_fingerprint": (
                integer_design_fingerprint([
                    tuple(int(value) for value in row["x_selected"])
                    for row in self.iteration_log
                    if row.get("x_selected") is not None
                ])
                if online_action_trace else None
            ),
            "online_action_trace_target_oracle_used": False,
            "simulation_randomness_contract": {
                "mode": "evaluation_indexed_seed_sequence",
                "base_seed": int(self.config.seed),
                "stream_tag": int(SIMULATION_STREAM_TAG),
                "evaluation_count": int(len(self.history)),
                "proposal_rng_independent": True,
                "common_random_numbers_by_evaluation_index": True,
                "target_oracle_used": False,
            },
            "proposal_randomness_contract": {
                "mode": "iteration_and_namespace_seed_sequence",
                "base_seed": int(self.config.seed),
                "stream_tag": int(PROPOSAL_STREAM_TAG),
                "component_streams_independent": True,
                "simulation_rng_independent": True,
                "target_oracle_used": False,
            },
            "decision_backend_diagnostics": decision_backend_summary,
            "decision_backend_contract": decision_backend_contract,
            "adaptive_replication_voi": adaptive_replication_summary,
            "posterior_dominance": {
                "enabled": bool(self._posterior_dominance_active()),
                "method": "cantelli_covariance_free",
                "delta_switch": float(
                    self.config.posterior_dominance_delta),
                "incumbent": (
                    None
                    if self._posterior_dominance_incumbent is None
                    else list(map(
                        int, self._posterior_dominance_incumbent))
                ),
                "switch_count": int(sum(
                    bool(row.get("switch_accepted", False))
                    for row in self._posterior_dominance_history
                    if row.get("reason") != "initial_design"
                )),
                "history": copy.deepcopy(
                    self._posterior_dominance_history),
                "target_oracle_used": False,
            },
            "adaptive_outcome_audit": adaptive_outcome,
            "certificate_outcome_audit": certificate_outcome,
            "exact_kg_diagnostics": exact_kg_summary,
            "finalist_replication": finalist_replication_summary,
            "two_stage_decision": two_stage_decision,
            "task_initial_design": copy.deepcopy(
                self._task_initial_design_info),
            "llm_prior": llm_prior_summary,
            "truth_pool_diagnostics": self._summarize_truth_pool_diagnostics(),
            "boundary_raw_pool_truth_diagnostics": (
                self._boundary_coordinate_raw_pool_audit()),
            "boundary_coordinate_proposal": (
                self._summarize_boundary_coordinate_proposals()),
            "variance": self.variance_model.diagnostics(),
            "adaptive_sparsity": [
                model.adaptive_sparsity_diagnostics()
                for model in self.gpr
            ],
            "gpr_numerics": [model.numerical_diagnostics() for model in self.gpr],
            "source_conditioned_confidence": copy.deepcopy(getattr(
                self,
                "_last_source_conditioned_confidence_diagnostics",
                {"mode": self.config.source_constraint_mean_confidence_mode,
                 "status": "not_queried", "target_oracle_used": False},
            )),
            "meta_basis": (
                self.problem.meta_basis_diagnostics()
                if hasattr(self.problem, "meta_basis_diagnostics")
                else None
            ),
            "mean_risk_coordinate_contract": (
                self.problem.mean_risk_coordinate_contract()
                if hasattr(self.problem, "mean_risk_coordinate_contract")
                else None
            ),
            "task_posterior": (
                None
                if self.task_ensemble is None
                else self.task_ensemble.diagnostics()
            ),
            "task_meta_coherence": task_meta_coherence,
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
                and (
                    not self._task_posterior_requested()
                    or self.task_ensemble is not None
                )
            ),
            "structural_mainline_acquisition_agnostic": bool(
                self.config.variance_mode == "factor"
                and self.config.certification_mode == "theory"
                and self.config.use_state_coupling
                and self.config.use_state_basis
                and (
                    not self._task_posterior_requested()
                    or self.task_ensemble is not None
                )
            ),
            "numeric_backend": self.gpr[0].backend_status(),
            "config": asdict(self.config),
        }
        self._save_checkpoint(self.config.N, reason="final", force=True)
        return self.final_log
