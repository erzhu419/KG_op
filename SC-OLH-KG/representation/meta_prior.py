"""Learned admissible meta-prior for LODO SC-OLH-KG experiments.

This module intentionally uses a small dependency-free model.  The goal is to
replace target-specific anchors/refinement/risk coordinates with a frozen
source-trained structural prior:

* a dimension-invariant policy descriptor,
* whitened continuous local exposures A,
* soft shared-shock regime exposures N,
* a source-fitted cumulative-HVD beta prior,
* and a meta-anchor proposal distribution in psi=(A,N) space.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.stats import norm

from core.admissibility import (
    domain_tuned_audit,
    lodo_meta_prior_audit,
    strict_universal_audit,
)
from core.candidates import unique_candidates
from core.cumulative_risk import (
    RiskExposure,
    canonical_risk_descriptor,
    cumulative_feature_names,
    cumulative_feature_vector,
    get_risk_exposure,
)
from representation.transferable_spectral import (
    SourceDomainBatch,
    TransferableSpectralBasis,
)
from representation.adaptive_sparsity import AdaptiveSpikeSlabPosterior
from representation.additive_groups import TransferableAdditiveGroupBank
from representation.risk_aligned_subspace import (
    BoundaryAlignedRiskSubspaces,
    TargetRiskAlignment,
)
from representation.source_boundary_episodes import (
    SourceBoundaryEpisodePrior,
)
from representation.observable_coordinate import (
    SourceLearnedObservableCoordinate,
)
from representation.observable_exposure import (
    canonical_observable_state_descriptor,
    canonical_set_invariant_observable_state_descriptor,
    get_observable_state_exposure,
    role_aligned_observable_state_descriptor,
)
from representation.channel_role_alignment import (
    EquivariantChannelRoleAligner,
)
from representation.boundary_coordinate import (
    SourceAlignedBoundaryCoordinate,
    SourceSupportedRoleBoundaryCoordinate,
)
from representation.exchangeable_mean import (
    ExchangeableBoundaryMeanCoordinate,
)
from representation.variance_coordinate import (
    SourceAlignedVarianceRiskCoordinate,
)
from representation.transferable_boundary import (
    HierarchicalSignedDistancePosterior,
)


HIDDEN_TARGET_STRUCTURAL_METHODS = {
    "all_axis_solutions",
    "cumulative_risk_features",
    "cumulative_risk_feature_names",
    "cumulative_risk_parameters",
    "cumulative_risk_provider_status",
    "gpr_basis_map",
    "hvd_features",
    "initial_samples",
    "inverse_state_anchor",
    "recommendation_random_pool_size",
    "recommendation_refinement_candidates",
    "risk_class",
    "risk_exposures",
    "state_anchor_points",
    "structured_candidates",
    "surrogate_basis_map",
    "true_cumulative_risk_decomposition",
}


def _as_tuple(x):
    return tuple(int(v) for v in x)


def _softmax_negdist(d2, temperature):
    d2 = np.asarray(d2, dtype=float)
    temp = max(float(temperature), 1e-8)
    logits = -d2 / temp
    logits -= float(np.max(logits))
    w = np.exp(logits)
    total = float(np.sum(w))
    if total <= 1e-12:
        return np.full(len(d2), 1.0 / max(len(d2), 1), dtype=float)
    return w / total


def _project_psd_features(beta, n_local, n_shared):
    """Project cumulative beta coefficients onto the admissible nonnegative cone."""

    beta = np.asarray(beta, dtype=float).reshape(-1).copy()
    expected = 1 + n_local + n_shared * (n_shared + 1) // 2 + n_shared
    if len(beta) != expected:
        return np.maximum(beta, 0.0)
    beta[0] = max(float(beta[0]), 1e-10)
    beta[1:1 + n_local] = np.maximum(beta[1:1 + n_local], 0.0)
    start = 1 + n_local
    end = start + n_shared * (n_shared + 1) // 2
    B = np.zeros((n_shared, n_shared), dtype=float)
    pos = start
    for i in range(n_shared):
        for j in range(i, n_shared):
            B[i, j] = B[j, i] = float(beta[pos])
            pos += 1
    try:
        vals, vecs = np.linalg.eigh(0.5 * (B + B.T))
        vals = np.maximum(vals, 0.0)
        B = (vecs * vals) @ vecs.T
    except np.linalg.LinAlgError:
        B = np.maximum(B, 0.0)
    pos = start
    for i in range(n_shared):
        for j in range(i, n_shared):
            beta[pos] = float(B[i, j])
            pos += 1
    beta[end:] = np.maximum(beta[end:], 0.0)
    return beta


@dataclass
class SourceRecord:
    domain: str
    x: tuple[int, ...]
    y: np.ndarray
    descriptor: np.ndarray
    profile: np.ndarray | None
    tau: float
    alpha: float
    sigma_level: float
    constraint_sigma: float | None = None
    observable_state_descriptor: np.ndarray | None = None
    observable_state_invariant_descriptor: np.ndarray | None = None
    observable_state_exposure: object | None = None
    provider_risk_descriptor: np.ndarray | None = None
    provider_risk_coordinate: np.ndarray | None = None
    origin: str = "random"
    sample_weight: float = 1.0
    replicate_count: int = 1
    replicates: np.ndarray | None = None


class LearnedMetaPrior:
    """Frozen source-trained prior used by held-out target adapters."""

    VALID_COMPONENT_STAGES = {
        "legacy_all",
        "coordinate",
        "spectral",
        "spectral_hvd",
    }
    VALID_BOUNDARY_DESCRIPTOR_MODES = {
        "raw",
        "learned_coordinate",
        "raw+learned_coordinate",
        "learned_risk",
        "raw+learned_risk",
        "provider_coordinate",
        "raw+provider_coordinate",
        "provider_risk",
        "raw+provider_risk",
    }

    def __init__(
        self,
        local_dim=3,
        shared_dim=3,
        anchor_count=24,
        kmeans_iters=25,
        soft_temperature=0.75,
        ridge=1e-4,
        boundary_weight=1.0,
        boundary_temperature=1.0,
        variance_weight=0.5,
        feasible_penalty=6.0,
        feasible_bonus=0.15,
        elite_fraction=0.40,
        boundary_fraction=0.35,
        teacher_records_per_domain=0,
        teacher_weight=3.0,
        teacher_pool_size=2048,
        teacher_elite_fraction=0.50,
        teacher_boundary_fraction=0.35,
        anchor_sampling_temperature=0.0,
        hvd_noise_floor_scale=0.0,
        universal_shape_count=0,
        component_stage="legacy_all",
        spectral_active_dim=6,
        spectral_max_library_size=64,
        spectral_low_frequency_components=8,
        spectral_low_frequency_prior=True,
        spectral_graph_neighbors=10,
        spectral_orthogonalization="symmetric",
        spectral_relevance_floor=0.05,
        spectral_gate_boundary_weight=2.0,
        spectral_gate_dangerous_weight=3.0,
        spectral_gate_selection_tolerance=0.02,
        spectral_gate_calibration_quantile=0.90,
        spectral_frequency_adaptation=False,
        spectral_frequency_cutoffs=(3, 5, 8, 12),
        spectral_frequency_ridges=(1e-4, 1e-2, 1.0),
        spectral_frequency_source_penalty=0.05,
        spectral_frequency_temperature=0.5,
        spectral_frequency_refit_interval=5,
        spectral_risk_alignment=False,
        spectral_alignment_active_dim=4,
        spectral_alignment_subspace_dim=2,
        spectral_alignment_domain_penalty=0.5,
        spectral_alignment_source_procrustes=False,
        spectral_alignment_target_ridge=5.0,
        spectral_alignment_target_min_gain=0.02,
        spectral_alignment_target_min_bins=3,
        spectral_alignment_refit_interval=5,
        spectral_alignment_source_episodes=0,
        spectral_alignment_admission=True,
        spectral_alignment_latent_proposals=False,
        spectral_alignment_inverse_pool_size=1024,
        spectral_alignment_episode_pilot_size=10,
        spectral_alignment_episode_evaluation_size=24,
        spectral_alignment_episode_ridge=0.1,
        spectral_additive_adaptation=False,
        spectral_additive_max_groups=8,
        spectral_additive_target_max_groups=2,
        spectral_additive_source_penalty=0.05,
        spectral_additive_complexity_penalty=0.05,
        spectral_additive_temperature=0.5,
        spectral_additive_refit_interval=5,
        spectral_additive_max_saturation_fraction=0.20,
        spectral_coefficient_shrinkage=False,
        spectral_shrinkage_strength=1.0,
        spectral_shrinkage_floor=0.05,
        spectral_adaptive_sparsity=False,
        spectral_adaptive_min_pip=0.05,
        spectral_adaptive_max_pip=0.95,
        spectral_adaptive_spike_ratio=0.05,
        spectral_adaptive_damping=0.5,
        spectral_adaptive_max_iter=40,
        spectral_adaptive_tolerance=1e-5,
        spectral_adaptive_residual_floor_scale=0.05,
        spectral_adaptive_gate_tolerance=0.05,
        spectral_adaptive_multiplicity_correction=1.0,
        spectral_adaptive_max_effective_fraction=0.35,
        spectral_adaptive_saturation_fraction=0.90,
        ordered_cumulative_exposure=False,
        ordered_exposure_max_frequency=8,
        ordered_exposure_active_dim=2,
        ordered_exposure_frequency_penalty=0.10,
        ordered_exposure_basis_mode="full_quadratic",
        ordered_exposure_orthogonal_coordinates=True,
        ordered_exposure_adaptive_sparsity=False,
        ordered_exposure_replace_local_kernel=False,
        ordered_exposure_semiparametric_residual=False,
        ordered_exposure_latent_structure_selection=False,
        ordered_exposure_group_shared_shrinkage=False,
        ordered_exposure_group_ridge_learning=False,
        coordinate_mode="pca",
        coordinate_relevance_floor=0.05,
        observable_mean_coordinate=False,
        observable_mean_ridges=(0.01, 0.1, 1.0, 10.0, 100.0),
        observable_mean_mode="latent",
        observable_mean_latent_dim=2,
        observable_mean_training_target="constraint_mean",
        observable_mean_input_mode="policy_profile",
        observable_mean_descriptor_mode="ordered",
        observable_mean_feature_mode="linear",
        observable_mean_latent_transform="identity",
        observable_mean_target_residual_rank=0,
        observable_mean_target_residual_prior_scale=1.0,
        observable_mean_target_residual_pool_size=128,
        observable_mean_target_residual_rcond=1e-8,
        observable_mean_role_assignment_posterior=False,
        observable_mean_role_assignment_prior="uniform",
        observable_mean_role_assignment_prior_temperature_scale=1.0,
        observable_mean_role_assignment_inactive_variance=1e-12,
        observable_variance_input_mode="legacy_policy_proxy",
        source_observation_mode="analytic",
        source_observation_replicates=1,
        source_design_mode="random",
        source_universal_fraction=0.75,
        source_consensus_template_count=0,
        seed=123,
    ):
        self.local_dim = int(local_dim)
        self.shared_dim = int(shared_dim)
        self.anchor_count = int(anchor_count)
        self.kmeans_iters = int(kmeans_iters)
        self.soft_temperature = float(soft_temperature)
        self.ridge = float(ridge)
        self.boundary_weight = float(boundary_weight)
        self.boundary_temperature = float(boundary_temperature)
        self.variance_weight = float(variance_weight)
        self.feasible_penalty = float(feasible_penalty)
        self.feasible_bonus = float(feasible_bonus)
        self.elite_fraction = float(elite_fraction)
        self.boundary_fraction = float(boundary_fraction)
        self.teacher_records_per_domain = int(teacher_records_per_domain)
        self.teacher_weight = float(teacher_weight)
        self.teacher_pool_size = int(teacher_pool_size)
        self.teacher_elite_fraction = float(teacher_elite_fraction)
        self.teacher_boundary_fraction = float(teacher_boundary_fraction)
        self.anchor_sampling_temperature = float(anchor_sampling_temperature)
        self.hvd_noise_floor_scale = float(hvd_noise_floor_scale)
        self.universal_shape_count = int(universal_shape_count)
        self.component_stage = str(component_stage)
        if self.component_stage not in self.VALID_COMPONENT_STAGES:
            raise ValueError(
                f"component_stage must be one of {sorted(self.VALID_COMPONENT_STAGES)}"
            )
        self.spectral_active_dim = int(spectral_active_dim)
        self.spectral_max_library_size = int(spectral_max_library_size)
        self.spectral_low_frequency_components = int(
            spectral_low_frequency_components)
        self.spectral_low_frequency_prior = bool(
            spectral_low_frequency_prior)
        self.spectral_graph_neighbors = int(spectral_graph_neighbors)
        self.spectral_orthogonalization = str(spectral_orthogonalization)
        if self.spectral_orthogonalization not in {
            "symmetric", "ordered_cholesky", "none"
        }:
            raise ValueError(
                "spectral orthogonalization must be symmetric, "
                "ordered_cholesky, or none")
        self.spectral_relevance_floor = float(spectral_relevance_floor)
        self.spectral_gate_boundary_weight = float(spectral_gate_boundary_weight)
        self.spectral_gate_dangerous_weight = float(spectral_gate_dangerous_weight)
        self.spectral_gate_selection_tolerance = float(
            spectral_gate_selection_tolerance)
        self.spectral_gate_calibration_quantile = float(
            spectral_gate_calibration_quantile)
        self.spectral_frequency_adaptation = bool(
            spectral_frequency_adaptation)
        self.spectral_frequency_cutoffs = self._positive_grid(
            spectral_frequency_cutoffs,
            int,
            self.spectral_low_frequency_components,
        )
        self.spectral_frequency_ridges = self._positive_grid(
            spectral_frequency_ridges,
            float,
            1.0,
        )
        self.spectral_frequency_source_penalty = max(
            float(spectral_frequency_source_penalty), 0.0)
        self.spectral_frequency_temperature = max(
            float(spectral_frequency_temperature), 1e-8)
        self.spectral_frequency_refit_interval = max(
            1, int(spectral_frequency_refit_interval))
        self.spectral_risk_alignment = bool(spectral_risk_alignment)
        self.spectral_alignment_active_dim = max(
            1, int(spectral_alignment_active_dim))
        self.spectral_alignment_subspace_dim = max(
            1, int(spectral_alignment_subspace_dim))
        self.spectral_alignment_domain_penalty = max(
            float(spectral_alignment_domain_penalty), 0.0)
        self.spectral_alignment_source_procrustes = bool(
            spectral_alignment_source_procrustes)
        self.spectral_alignment_target_ridge = max(
            float(spectral_alignment_target_ridge), 0.0)
        self.spectral_alignment_target_min_gain = max(
            float(spectral_alignment_target_min_gain), 0.0)
        self.spectral_alignment_target_min_bins = max(
            2, int(spectral_alignment_target_min_bins))
        self.spectral_alignment_refit_interval = max(
            1, int(spectral_alignment_refit_interval))
        self.spectral_alignment_source_episodes = max(
            0, int(spectral_alignment_source_episodes))
        self.spectral_alignment_admission = bool(
            spectral_alignment_admission)
        self.spectral_alignment_latent_proposals = bool(
            spectral_alignment_latent_proposals)
        self.spectral_alignment_inverse_pool_size = max(
            64, int(spectral_alignment_inverse_pool_size))
        self.spectral_alignment_episode_pilot_size = max(
            6, int(spectral_alignment_episode_pilot_size))
        self.spectral_alignment_episode_evaluation_size = max(
            8, int(spectral_alignment_episode_evaluation_size))
        self.spectral_alignment_episode_ridge = max(
            float(spectral_alignment_episode_ridge), 1e-10)
        self.spectral_additive_adaptation = bool(
            spectral_additive_adaptation)
        self.spectral_additive_max_groups = max(
            1, int(spectral_additive_max_groups))
        self.spectral_additive_target_max_groups = max(
            1, int(spectral_additive_target_max_groups))
        self.spectral_additive_source_penalty = max(
            float(spectral_additive_source_penalty), 0.0)
        self.spectral_additive_complexity_penalty = max(
            float(spectral_additive_complexity_penalty), 0.0)
        self.spectral_additive_temperature = max(
            float(spectral_additive_temperature), 1e-8)
        self.spectral_additive_refit_interval = max(
            1, int(spectral_additive_refit_interval))
        self.spectral_additive_max_saturation_fraction = float(np.clip(
            spectral_additive_max_saturation_fraction, 0.0, 1.0))
        self.spectral_coefficient_shrinkage = bool(
            spectral_coefficient_shrinkage)
        self.spectral_shrinkage_strength = max(
            float(spectral_shrinkage_strength), 0.0)
        self.spectral_shrinkage_floor = float(np.clip(
            spectral_shrinkage_floor, 1e-6, 1.0))
        self.spectral_adaptive_sparsity = bool(spectral_adaptive_sparsity)
        if self.spectral_adaptive_sparsity and (
            self.spectral_frequency_adaptation
            or self.spectral_additive_adaptation
        ):
            raise ValueError(
                "adaptive spike-and-slab is a challenger to the composed "
                "frequency/additive hierarchy"
            )
        self.spectral_adaptive_min_pip = float(spectral_adaptive_min_pip)
        self.spectral_adaptive_max_pip = float(spectral_adaptive_max_pip)
        if not (
            0.0 < self.spectral_adaptive_min_pip
            < self.spectral_adaptive_max_pip
            < 1.0
        ):
            raise ValueError("adaptive PIP bounds must satisfy 0 < min < max < 1")
        self.spectral_adaptive_spike_ratio = float(
            spectral_adaptive_spike_ratio)
        self.spectral_adaptive_damping = float(spectral_adaptive_damping)
        self.spectral_adaptive_max_iter = max(
            1, int(spectral_adaptive_max_iter))
        self.spectral_adaptive_tolerance = max(
            float(spectral_adaptive_tolerance), 1e-12)
        self.spectral_adaptive_residual_floor_scale = max(
            float(spectral_adaptive_residual_floor_scale), 0.0)
        self.spectral_adaptive_gate_tolerance = max(
            float(spectral_adaptive_gate_tolerance), 0.0)
        self.spectral_adaptive_multiplicity_correction = max(
            float(spectral_adaptive_multiplicity_correction), 0.0)
        self.spectral_adaptive_max_effective_fraction = float(np.clip(
            spectral_adaptive_max_effective_fraction, 0.05, 1.0))
        self.spectral_adaptive_saturation_fraction = float(np.clip(
            spectral_adaptive_saturation_fraction, 0.5, 1.0))
        self.ordered_cumulative_exposure = bool(ordered_cumulative_exposure)
        self.ordered_exposure_max_frequency = max(
            1, int(ordered_exposure_max_frequency))
        self.ordered_exposure_active_dim = max(
            1, min(
                int(ordered_exposure_active_dim),
                self.ordered_exposure_max_frequency,
            ),
        )
        self.ordered_exposure_frequency_penalty = max(
            float(ordered_exposure_frequency_penalty), 0.0)
        self.ordered_exposure_basis_mode = str(ordered_exposure_basis_mode)
        if self.ordered_exposure_basis_mode not in {
            "full_quadratic", "diagonal_quadratic",
        }:
            raise ValueError(
                "ordered exposure basis mode must be full_quadratic or "
                "diagonal_quadratic")
        self.ordered_exposure_orthogonal_coordinates = bool(
            ordered_exposure_orthogonal_coordinates)
        self.ordered_exposure_adaptive_sparsity = bool(
            ordered_exposure_adaptive_sparsity)
        self.ordered_exposure_replace_local_kernel = bool(
            ordered_exposure_replace_local_kernel)
        self.ordered_exposure_semiparametric_residual = bool(
            ordered_exposure_semiparametric_residual)
        self.ordered_exposure_latent_structure_selection = bool(
            ordered_exposure_latent_structure_selection)
        self.ordered_exposure_group_shared_shrinkage = bool(
            ordered_exposure_group_shared_shrinkage)
        self.ordered_exposure_group_ridge_learning = bool(
            ordered_exposure_group_ridge_learning)
        if (
            self.ordered_exposure_latent_structure_selection
            and self.ordered_exposure_semiparametric_residual
        ):
            raise ValueError(
                "latent ordered/local structure selection is mutually "
                "exclusive with a semiparametric direct-sum residual"
            )
        if (
            self.ordered_exposure_group_shared_shrinkage
            and self.ordered_exposure_semiparametric_residual
        ):
            raise ValueError(
                "ordered group shared shrinkage is mutually exclusive with "
                "a semiparametric direct-sum residual"
            )
        if (
            self.ordered_exposure_group_ridge_learning
            and self.ordered_exposure_semiparametric_residual
        ):
            raise ValueError(
                "ordered group ridge learning is mutually exclusive with "
                "a semiparametric direct-sum residual"
            )
        if (
            self.ordered_exposure_group_ridge_learning
            and self.ordered_exposure_group_shared_shrinkage
        ):
            raise ValueError(
                "ordered group ridge learning and group spike-slab "
                "shrinkage are mutually exclusive"
            )
        if (
            self.ordered_exposure_group_ridge_learning
            and not self.ordered_exposure_adaptive_sparsity
        ):
            raise ValueError(
                "ordered group ridge learning requires adaptive sparsity "
                "posterior wiring"
            )
        self.coordinate_mode = str(coordinate_mode)
        if self.coordinate_mode not in {"pca", "stable_supervised"}:
            raise ValueError("coordinate_mode must be 'pca' or 'stable_supervised'")
        self.coordinate_relevance_floor = float(coordinate_relevance_floor)
        self.observable_mean_coordinate = bool(observable_mean_coordinate)
        self.observable_mean_ridges = self._positive_grid(
            observable_mean_ridges, float, 1.0)
        self.observable_mean_mode = str(
            observable_mean_mode).strip().lower()
        if self.observable_mean_mode not in {
            "atoms", "aggregate", "latent", "consensus", "source_affine",
            "source_rank", "boundary_aligned",
        }:
            raise ValueError(
                "observable_mean_mode must be atoms, aggregate, latent, or "
                "consensus/source_affine/source_rank/boundary_aligned"
            )
        self.observable_mean_latent_dim = max(
            int(observable_mean_latent_dim), 0)
        self.observable_mean_training_target = str(
            observable_mean_training_target).strip().lower()
        if self.observable_mean_training_target not in {
            "constraint_mean", "chance_margin"
        }:
            raise ValueError(
                "observable_mean_training_target must be constraint_mean "
                "or chance_margin"
            )
        if (
            self.observable_mean_mode == "boundary_aligned"
            and self.observable_mean_training_target != "chance_margin"
        ):
            raise ValueError(
                "boundary_aligned representation must be trained from "
                "source chance-margin strata"
            )
        self.observable_mean_input_mode = str(
            observable_mean_input_mode).strip().lower()
        if self.observable_mean_input_mode not in {
            "policy_profile",
            "source_learned_exposure",
            "observable_state_exposure",
            "provider_exposure",
        }:
            raise ValueError(
                "observable_mean_input_mode must be policy_profile, "
                "source_learned_exposure, observable_state_exposure, "
                "or provider_exposure"
            )
        if (
            self.observable_mean_input_mode != "policy_profile"
            and self.observable_mean_mode != "boundary_aligned"
        ):
            raise ValueError(
                "exposure input currently requires the separate "
                "boundary_aligned mean head"
            )
        self.observable_mean_descriptor_mode = str(
            observable_mean_descriptor_mode or "ordered"
        ).strip().lower().replace("-", "_")
        if self.observable_mean_descriptor_mode not in {
            "ordered", "set_invariant", "role_aligned",
            "role_transport", "role_intervention_transport",
            "role_adaptive_ordered", "role_adaptive_set_invariant",
            "exchangeable_equivariant",
        }:
            raise ValueError(
                "observable mean descriptor mode must be ordered, "
                "set_invariant, role_aligned, role_transport, "
                "role_intervention_transport, "
                "role_adaptive_ordered, role_adaptive_set_invariant, or "
                "exchangeable_equivariant")
        self.observable_mean_feature_mode = str(
            observable_mean_feature_mode or "linear"
        ).strip().lower().replace("-", "_")
        if self.observable_mean_feature_mode not in {
            "linear", "diagonal_quadratic", "full_quadratic"
        }:
            raise ValueError(
                "observable mean feature mode must be linear, "
                "diagonal_quadratic, or full_quadratic")
        self.observable_mean_latent_transform = str(
            observable_mean_latent_transform or "identity"
        ).strip().lower().replace("-", "_")
        if self.observable_mean_latent_transform not in {
            "identity",
            "source_tanh",
            "source_support_clip",
            "source_support_residual",
        }:
            raise ValueError(
                "observable mean latent transform must be identity, "
                "source_tanh, source_support_clip, or "
                "source_support_residual")
        self.observable_mean_target_residual_rank = max(
            0, min(int(observable_mean_target_residual_rank), 8))
        self.observable_mean_target_residual_prior_scale = max(
            float(observable_mean_target_residual_prior_scale), 1e-8)
        self.observable_mean_target_residual_pool_size = max(
            int(observable_mean_target_residual_pool_size), 32)
        self.observable_mean_target_residual_rcond = max(
            float(observable_mean_target_residual_rcond), 1e-12)
        self.observable_mean_role_assignment_posterior = bool(
            observable_mean_role_assignment_posterior)
        self.observable_mean_role_assignment_prior = str(
            observable_mean_role_assignment_prior or "uniform"
        ).strip().lower().replace("-", "_")
        if self.observable_mean_role_assignment_prior not in {
            "uniform", "source_geometry", "source_geometry_boundary"
        }:
            raise ValueError(
                "observable mean role-assignment prior must be uniform or "
                "source_geometry/source_geometry_boundary")
        self.observable_mean_role_assignment_prior_temperature_scale = max(
            float(observable_mean_role_assignment_prior_temperature_scale),
            1e-8,
        )
        self.observable_mean_role_assignment_inactive_variance = max(
            float(observable_mean_role_assignment_inactive_variance), 1e-12)
        if (
            self.observable_mean_role_assignment_posterior
            and self.observable_mean_input_mode != "observable_state_exposure"
        ):
            raise ValueError(
                "role-assignment posterior requires observable-state exposure")
        if (
            self.observable_mean_role_assignment_posterior
            and self.observable_mean_target_residual_rank > 0
        ):
            raise ValueError(
                "role-assignment and residual-rank posteriors are separate "
                "registered challengers")
        if (
            self.observable_mean_role_assignment_posterior
            and self.observable_mean_descriptor_mode
            == "exchangeable_equivariant"
        ):
            raise ValueError(
                "exchangeable coefficients replace the discrete source-role "
                "assignment posterior")
        self.observable_variance_input_mode = str(
            observable_variance_input_mode).strip().lower()
        if self.observable_variance_input_mode not in {
            "legacy_policy_proxy",
            "observable_state_exposure",
        }:
            raise ValueError(
                "observable_variance_input_mode must be "
                "legacy_policy_proxy or observable_state_exposure"
            )
        self.source_observation_mode = str(
            source_observation_mode).strip().lower()
        if self.source_observation_mode not in {
            "analytic", "nominal", "replicated"
        }:
            raise ValueError(
                "source_observation_mode must be analytic, nominal, or replicated"
            )
        self.source_observation_replicates = max(
            1, int(source_observation_replicates))
        self.source_design_mode = str(source_design_mode).strip().lower()
        if self.source_design_mode not in {
            "random",
            "universal_mixture",
            "shared_uniform",
        }:
            raise ValueError(
                "source_design_mode must be random, universal_mixture, or "
                "shared_uniform"
            )
        self.source_universal_fraction = float(np.clip(
            source_universal_fraction, 0.0, 1.0))
        self.source_consensus_template_count = max(
            0, int(source_consensus_template_count))
        if (
            self.teacher_records_per_domain > 0
            and self.source_observation_mode != "analytic"
        ):
            raise ValueError(
                "analytic teacher records cannot enter an oracle-free source path"
            )
        self.seed = int(seed)
        self.feature_mean = None
        self.feature_scale = None
        self.pca_components = None
        self.cluster_centers = None
        self.anchor_psi = np.empty((0, self.local_dim + self.shared_dim))
        self.anchor_scores = np.empty(0, dtype=float)
        self.anchor_meta = []
        self.profile_templates = []
        self.source_consensus_templates = []
        self.dimension_equivariant_proposal_diagnostics = {
            "status": "not_materialized",
            "source_only": True,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        self.risk_objective_proposal_diagnostics = {
            "status": "not_materialized",
            "source_only": True,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        self.boundary_excitation_diagnostics = {
            "status": "not_materialized",
            "source_only": True,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        self.source_consensus_diagnostics = {"status": "not_requested"}
        self.beta_prior = {}
        self.beta_prior_precision = {}
        self.beta_prior_reference_mean = {}
        self.beta_prior_upper_scale = {}
        self.beta_prior_components = {}
        self.beta_prior_component_domains = {}
        self.unaligned_beta_prior = {}
        self.unaligned_beta_prior_precision = {}
        self.unaligned_beta_prior_reference_mean = {}
        self.unaligned_beta_prior_upper_scale = {}
        self.unaligned_beta_prior_components = {}
        self.unaligned_beta_prior_component_domains = {}
        self.task_sensitivity_prior_ = {
            "status": "unfit",
            "class_names": ["stable", "balanced", "sensitive"],
            "scales": [0.5, 1.0, 2.0],
            "biases": [0.0, 0.0, 0.0],
            "decision_penalties": [2.0, 5.0, 20.0],
            "empirical_trust": [1.0, 0.25, 0.0],
            "prior_weights": [1.0 / 3.0] * 3,
            "target_data_used": False,
        }
        bias_feature_dim = 1 + self.local_dim + max(
            self.shared_dim - 1, 0)
        self.task_bias_profiles_ = np.zeros(
            (1, bias_feature_dim), dtype=float)
        self.task_bias_profile_names_ = ["null_bias_profile"]
        self.task_bias_profile_prior_ = np.ones(1, dtype=float)
        self.task_bias_profile_diagnostics_ = {"status": "unfit"}
        self.task_adaptive_bias_prior_ = {
            "status": "unfit",
            "mean": np.zeros(bias_feature_dim, dtype=float),
            "precision": np.eye(bias_feature_dim, dtype=float),
            "feature_names": self.task_bias_feature_names(),
            "target_data_used": False,
            "target_oracle_used": False,
        }
        self.mean_prior = {}
        self.mean_prior_sigma = {}
        self.spectral_basis: TransferableSpectralBasis | None = None
        self.stage1_spectral_basis: TransferableSpectralBasis | None = None
        self.spectral_feature_dim = 0
        self.spectral_always_active_count = 0
        self.spectral_frequency_bank = []
        self.spectral_frequency_diagnostics = {"status": "not_requested"}
        self.risk_subspace_alignment: BoundaryAlignedRiskSubspaces | None = None
        self.risk_aligned_spectral_basis: TransferableSpectralBasis | None = None
        self.risk_aligned_frequency_bank = []
        self.risk_aligned_additive_bank: TransferableAdditiveGroupBank | None = None
        self.risk_alignment_diagnostics = {"status": "not_requested"}
        self.alignment_episode_prior: SourceBoundaryEpisodePrior | None = None
        self.alignment_episode_diagnostics = {"status": "not_requested"}
        self.alignment_profile_templates = []
        self.source_boundary_bracket_model = {"status": "unfit"}
        self.spectral_additive_bank: TransferableAdditiveGroupBank | None = None
        self.spectral_additive_diagnostics = {"status": "not_requested"}
        self.spectral_coefficient_prior = {}
        self.spectral_adaptive_calibration = {"status": "not_requested"}
        self.ordered_exposure_selected_frequencies = np.empty(0, dtype=int)
        self.ordered_exposure_scale = np.ones(2, dtype=float)
        self.ordered_exposure_diagnostics = {"status": "not_requested"}
        self.ordered_coefficient_prior = {}
        self.hierarchical_boundary_posterior = None
        self.hierarchical_boundary_descriptor_mode = "raw"
        self.hierarchical_boundary_diagnostics = {
            "status": "not_requested",
        }
        self.observable_mean_model = None
        self.observable_channel_role_aligner = None
        self.observable_variance_model = None
        self.source_domains = []
        self.n_records = 0
        self.source_records_ = []
        self.fit_status = "unfit"
        self.training_diagnostics = {}
        self.coordinate_diagnostics = {"mode": self.coordinate_mode}

    @staticmethod
    def _positive_grid(values, cast, required):
        if isinstance(values, str):
            values = [item.strip() for item in values.split(",") if item.strip()]
        parsed = [cast(value) for value in values]
        parsed.append(cast(required))
        parsed = sorted({value for value in parsed if float(value) > 0.0})
        if not parsed:
            raise ValueError("frequency adaptation grids must be positive")
        return tuple(parsed)

    def component_enabled(self, name):
        name = str(name)
        if name == "coordinate":
            return True
        if self.component_stage == "legacy_all":
            return name in {"hvd", "mean", "proposal"}
        if self.component_stage in {"spectral", "spectral_hvd"}:
            if name == "hvd":
                return self.component_stage == "spectral_hvd"
            return name == "spectral"
        return False

    @staticmethod
    def descriptor(problem, x):
        """Dimension-invariant policy descriptor from observable bounds only."""

        z = np.asarray(problem.normalize(x), dtype=float).reshape(-1)
        if len(z) == 0:
            z = np.zeros(1, dtype=float)
        qs = np.quantile(z, [0.10, 0.25, 0.50, 0.75, 0.90])
        center_norm = float(np.linalg.norm(z - 0.5) / np.sqrt(max(len(z), 1)))
        diffs = np.diff(z) if len(z) > 1 else np.array([0.0])
        segs = np.array_split(z, 4)
        seg_stats = []
        for seg in segs:
            if len(seg) == 0:
                seg_stats.extend([0.0, 0.0])
            else:
                seg_stats.extend([float(np.mean(seg)), float(np.std(seg))])
        hist, _ = np.histogram(z, bins=np.linspace(0.0, 1.0, 6))
        hist = hist.astype(float) / max(float(len(z)), 1.0)
        out = np.concatenate([
            np.array([
                float(np.mean(z)),
                float(np.std(z)),
                float(np.min(z)),
                float(np.max(z)),
                center_norm,
                float(np.mean(np.abs(diffs))),
                float(z[0]),
                float(z[-1]),
                float(np.mean(z[1:])) if len(z) > 1 else float(z[0]),
                float(np.std(z[1:])) if len(z) > 1 else 0.0,
            ]),
            qs,
            np.asarray(seg_stats, dtype=float),
            hist,
        ])
        return np.asarray(out, dtype=float)

    @classmethod
    def provider_risk_descriptor(cls, problem, x):
        """Describe a declared structural provider without reading outcomes."""

        exposure = get_risk_exposure(problem, x, output_index=1)
        if exposure is None:
            return None
        return canonical_risk_descriptor(exposure)

    @classmethod
    def provider_risk_coordinate(cls, problem, x):
        """Return the declared provider's exact ``[A,N]`` coordinate."""

        exposure = get_risk_exposure(problem, x, output_index=1)
        if exposure is None:
            return None
        return np.concatenate([exposure.A, exposure.N]).astype(float)

    @staticmethod
    def observable_state_descriptor(problem, x, mode="ordered"):
        """Describe a declared target-observable state/trajectory record."""

        exposure = get_observable_state_exposure(problem, x)
        if exposure is None:
            return None
        mode = str(mode or "ordered").strip().lower().replace("-", "_")
        if mode == "set_invariant":
            return canonical_set_invariant_observable_state_descriptor(exposure)
        return canonical_observable_state_descriptor(exposure)

    def boundary_descriptor_from_raw(
        self,
        raw_descriptor,
        *,
        mode=None,
        provider_risk_descriptor=None,
        provider_risk_coordinate=None,
        domain=None,
    ):
        """Build the exact descriptor consumed by the hierarchical boundary.

        ``learned_risk`` is source-frozen and target-agnostic.  ``provider_risk``
        is a separately audited structure-aware input and is never silently
        substituted when a provider is absent.
        """

        mode = str(
            self.hierarchical_boundary_descriptor_mode if mode is None else mode
        ).lower()
        if mode not in self.VALID_BOUNDARY_DESCRIPTOR_MODES:
            raise ValueError(f"unknown hierarchical boundary descriptor mode {mode!r}")
        raw = np.asarray(raw_descriptor, dtype=float).reshape(-1)
        if "learned_" in mode:
            exposure = self.cumulative_risk_exposure_from_descriptor(
                raw, domain=domain)
            risk = (
                np.concatenate([exposure.A, exposure.N]).astype(float)
                if "learned_coordinate" in mode
                else canonical_risk_descriptor(exposure)
            )
        elif "provider_" in mode:
            provider = (
                provider_risk_coordinate
                if "provider_coordinate" in mode
                else provider_risk_descriptor
            )
            if provider is None:
                raise ValueError(
                    "provider boundary coordinate requested for a problem without "
                    "CumulativeRiskFeatureProvider")
            risk = np.asarray(provider, dtype=float).reshape(-1)
        else:
            risk = None
        if mode == "raw":
            return raw
        if not mode.startswith("raw+"):
            return risk
        return np.concatenate([raw, risk])

    def boundary_descriptor(self, problem, x, *, mode=None):
        mode = str(
            self.hierarchical_boundary_descriptor_mode if mode is None else mode
        ).lower()
        raw = self.descriptor(problem, x)
        provider = None
        provider_coordinate = None
        if "provider_" in mode:
            provider = self.provider_risk_descriptor(problem, x)
            provider_coordinate = self.provider_risk_coordinate(problem, x)
        return self.boundary_descriptor_from_raw(
            raw,
            mode=mode,
            provider_risk_descriptor=provider,
            provider_risk_coordinate=provider_coordinate,
            domain=None,
        )

    def _scaled_descriptor(self, descriptor):
        desc = np.asarray(descriptor, dtype=float)
        return (desc - self.feature_mean) / self.feature_scale

    def _fit_scaler_pca(self, descriptors, records=None):
        X = np.vstack(descriptors)
        self.feature_mean = np.mean(X, axis=0)
        self.feature_scale = np.std(X, axis=0)
        self.feature_scale = np.where(self.feature_scale < 1e-8, 1.0, self.feature_scale)
        Z = (X - self.feature_mean) / self.feature_scale
        if self.coordinate_mode == "stable_supervised":
            if records is None or len(records) != len(Z):
                raise ValueError(
                    "stable_supervised coordinates require aligned source records")
            return self._fit_stable_descriptor_projection(Z, records)
        try:
            _, _, vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = np.eye(Z.shape[1], dtype=float)
        k = min(self.local_dim, vt.shape[0])
        comp = np.zeros((self.local_dim, Z.shape[1]), dtype=float)
        comp[:k, :] = vt[:k, :]
        if k < self.local_dim:
            comp[k:, :self.local_dim - k] = np.eye(self.local_dim - k)
        self.pca_components = comp
        A = Z @ self.pca_components.T
        a_scale = np.std(A, axis=0)
        a_scale = np.where(a_scale < 1e-8, 1.0, a_scale)
        self.pca_components = self.pca_components / a_scale[:, None]
        self.coordinate_diagnostics = {
            "mode": "pca",
            "selected_names": [],
        }
        return Z

    @staticmethod
    def _descriptor_names():
        return [
            "mean", "std", "min", "max", "center_norm", "mean_abs_diff",
            "first", "last", "tail_mean", "tail_std",
            "q10", "q25", "q50", "q75", "q90",
            "segment0_mean", "segment0_std", "segment1_mean", "segment1_std",
            "segment2_mean", "segment2_std", "segment3_mean", "segment3_std",
            "hist0", "hist1", "hist2", "hist3", "hist4",
        ]

    def _ordered_profile_library(self, profile):
        """Dimension-invariant global and ordered cosine policy moments."""

        z = np.asarray(profile, dtype=float).reshape(-1)
        if len(z) == 0:
            z = np.zeros(1, dtype=float)
        centered = z - float(np.mean(z))
        positions = (np.arange(len(z), dtype=float) + 0.5) / float(len(z))
        coefficients = [
            float(2.0 * np.mean(
                centered * np.cos(np.pi * frequency * positions)
            ))
            for frequency in range(1, self.ordered_exposure_max_frequency + 1)
        ]
        return np.asarray([
            float(np.mean(z)),
            float(np.std(z)),
            *coefficients,
        ], dtype=float)

    @staticmethod
    def _weighted_abs_correlation(x, y, weights):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        weights = np.clip(
            np.asarray(weights, dtype=float).reshape(-1), 0.0, np.inf)
        total = float(np.sum(weights))
        if len(x) < 3 or total <= 0.0:
            return 0.0, 0.0
        weights = weights / total
        x_centered = x - float(np.sum(weights * x))
        y_centered = y - float(np.sum(weights * y))
        denom = np.sqrt(
            float(np.sum(weights * x_centered ** 2))
            * float(np.sum(weights * y_centered ** 2))
        )
        if denom <= 1e-12:
            return 0.0, 0.0
        correlation = float(np.sum(
            weights * x_centered * y_centered) / denom)
        return abs(correlation), float(np.sign(correlation))

    def _fit_ordered_exposure(self, records):
        if not self.ordered_cumulative_exposure:
            self.ordered_exposure_diagnostics = {"status": "not_requested"}
            return
        usable = [rec for rec in records if rec.profile is not None]
        if not usable:
            raise ValueError("ordered cumulative exposure needs source profiles")
        library = np.vstack([
            self._ordered_profile_library(rec.profile) for rec in usable
        ])
        by_domain = {}
        for index, rec in enumerate(usable):
            by_domain.setdefault(str(rec.domain), []).append(index)
        domain_relevance = []
        domain_sign = []
        for indices in by_domain.values():
            idx = np.asarray(indices, dtype=int)
            margins = np.asarray([
                self._source_margin(usable[index])
                / self._source_margin_scale(usable[index])
                for index in idx
            ], dtype=float)
            weights = np.asarray([
                self._boundary_sample_weight(usable[index])
                * float(usable[index].sample_weight)
                for index in idx
            ], dtype=float)
            relevance = []
            signs = []
            for frequency_index in range(self.ordered_exposure_max_frequency):
                value, sign = self._weighted_abs_correlation(
                    library[idx, 2 + frequency_index], margins, weights)
                relevance.append(value)
                signs.append(sign)
            domain_relevance.append(relevance)
            domain_sign.append(signs)
        relevance = np.asarray(domain_relevance, dtype=float)
        signs = np.asarray(domain_sign, dtype=float)
        relevance_mean = np.mean(relevance, axis=0)
        relevance_std = np.std(relevance, axis=0)
        prevalence = np.mean(relevance >= 0.05, axis=0)
        sign_consistency = np.maximum(
            np.mean(signs >= 0.0, axis=0),
            np.mean(signs <= 0.0, axis=0),
        )
        frequencies = np.arange(
            1, self.ordered_exposure_max_frequency + 1, dtype=int)
        low_frequency_prior = 1.0 / (
            1.0 + self.ordered_exposure_frequency_penalty * (frequencies - 1)
        )
        score = (
            relevance_mean
            / (1.0 + relevance_std)
            * (0.5 + 0.5 * prevalence)
            * (0.75 + 0.25 * sign_consistency)
            * low_frequency_prior
        )
        order = sorted(
            range(len(score)), key=lambda index: (-score[index], index))
        selected = np.asarray([
            frequencies[index]
            for index in order[: self.ordered_exposure_active_dim]
        ], dtype=int)
        selected_columns = [0, 1] + [int(frequency) + 1 for frequency in selected]
        selected_library = library[:, selected_columns]
        record_weights = np.asarray([
            self._boundary_sample_weight(rec) * float(rec.sample_weight)
            for rec in usable
        ], dtype=float)
        record_weights = np.clip(record_weights, 1e-8, np.inf)
        record_weights /= float(np.sum(record_weights))
        scale = np.sqrt(np.sum(
            record_weights[:, None] * selected_library ** 2, axis=0))
        self.ordered_exposure_selected_frequencies = selected
        self.ordered_exposure_scale = np.where(scale < 1e-6, 1.0, scale)
        self.ordered_exposure_diagnostics = {
            "status": "fit",
            "source_only": True,
            "target_data_used": False,
            "semiparametric_residual": bool(
                self.ordered_exposure_semiparametric_residual),
            "latent_structure_selection": bool(
                self.ordered_exposure_latent_structure_selection),
            "group_shared_shrinkage": bool(
                self.ordered_exposure_group_shared_shrinkage),
            "group_ridge_learning": bool(
                self.ordered_exposure_group_ridge_learning),
            "orthogonal_coordinates": bool(
                self.ordered_exposure_orthogonal_coordinates),
            "max_frequency": int(self.ordered_exposure_max_frequency),
            "active_dim": int(len(selected)),
            "selected_frequencies": selected.tolist(),
            "selected_scores": [
                float(score[int(frequency) - 1]) for frequency in selected
            ],
            "selected_prevalence": [
                float(prevalence[int(frequency) - 1]) for frequency in selected
            ],
            "selected_sign_consistency": [
                float(sign_consistency[int(frequency) - 1])
                for frequency in selected
            ],
            "scale": self.ordered_exposure_scale.tolist(),
            "n_source_domains": int(len(by_domain)),
            "n_source_records": int(len(usable)),
        }

    def _ordered_exposure_from_record(self, record):
        library = self._ordered_profile_library(record.profile)
        columns = [0, 1] + [
            int(frequency) + 1
            for frequency in self.ordered_exposure_selected_frequencies
        ]
        A = self._ordered_coordinate_transform(np.asarray(
            library[columns] / self.ordered_exposure_scale, dtype=float))
        if (
            self.component_stage == "spectral_hvd"
            and self.risk_subspace_alignment is not None
        ):
            base = self.aligned_cumulative_risk_exposure_from_descriptor(
                record.descriptor,
                domain=record.domain,
            )
        else:
            base = self.risk_exposure_from_descriptor(record.descriptor)
        return RiskExposure(A, base.N)

    def _fit_ordered_coefficient_prior(self, records):
        if not (
            self.ordered_cumulative_exposure
            and self.ordered_exposure_adaptive_sparsity
        ):
            self.ordered_coefficient_prior = {}
            return
        usable = [rec for rec in records if rec.profile is not None]
        by_domain = {}
        for rec in usable:
            by_domain.setdefault(str(rec.domain), []).append(rec)
        features_by_domain = {
            domain: np.vstack([
                self._ordered_basis_from_exposure(
                    self._ordered_exposure_from_record(rec))
                for rec in domain_records
            ])
            for domain, domain_records in by_domain.items()
        }
        priors = {}
        for output_index in (0, 1):
            relevance_rows = []
            coefficient_rows = []
            for domain, domain_records in by_domain.items():
                X = features_by_domain[domain]
                if output_index == 1:
                    y = np.asarray([
                        self._source_margin(rec)
                        / self._source_margin_scale(rec)
                        for rec in domain_records
                    ], dtype=float)
                else:
                    y = np.asarray([
                        float(rec.y[output_index]) for rec in domain_records
                    ], dtype=float)
                weights = np.asarray([
                    self._boundary_sample_weight(rec)
                    * float(rec.sample_weight)
                    for rec in domain_records
                ], dtype=float)
                relevance_rows.append([
                    self._weighted_abs_correlation(
                        X[:, feature], y, weights)[0]
                    for feature in range(X.shape[1])
                ])
                x_mean = np.average(X, axis=0, weights=weights)
                x_scale = np.sqrt(np.average(
                    (X - x_mean) ** 2, axis=0, weights=weights))
                x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
                y_mean = float(np.average(y, weights=weights))
                y_scale = max(float(np.sqrt(np.average(
                    (y - y_mean) ** 2, weights=weights))), 1e-8)
                Z = (X - x_mean) / x_scale
                target = (y - y_mean) / y_scale
                sqrt_w = np.sqrt(np.clip(weights, 1e-8, np.inf))
                Zw = Z * sqrt_w[:, None]
                yw = target * sqrt_w
                ridge = max(float(self.ridge), 1e-3)
                beta = np.linalg.pinv(
                    Zw.T @ Zw + ridge * np.eye(Z.shape[1])) @ Zw.T @ yw
                coefficient_rows.append(beta)
            relevance = np.asarray(relevance_rows, dtype=float)
            coefficients = np.asarray(coefficient_rows, dtype=float)
            relevance_mean = np.mean(relevance, axis=0)
            relevance_std = np.std(relevance, axis=0)
            prevalence = np.mean(relevance >= 0.05, axis=0)
            score = relevance_mean / (1.0 + relevance_std) * (
                0.5 + 0.5 * prevalence)
            normalized_score = score / max(float(np.max(score)), 1e-12)
            source_pip = (
                self.spectral_adaptive_min_pip
                + (
                    self.spectral_adaptive_max_pip
                    - self.spectral_adaptive_min_pip
                ) * normalized_score
            )
            n_local = 2 + len(self.ordered_exposure_selected_frequencies)
            source_pip[:n_local] = self.spectral_adaptive_max_pip
            slab_scale = np.clip(
                np.sqrt(np.mean(coefficients ** 2, axis=0)) + 0.10,
                0.10,
                5.0,
            )
            priors[output_index] = {
                "source_pip": source_pip,
                "source_slab_scale": slab_scale,
                "allowed_mask": np.ones(len(source_pip), dtype=bool),
                "always_active_count": int(n_local),
                "feature_dim": int(len(source_pip)),
                "source_domains": sorted(by_domain),
                "source_relevance": relevance_mean,
                "source_prevalence": prevalence,
            }
        self.ordered_coefficient_prior = priors
        self.ordered_exposure_diagnostics.update({
            "basis_mode": self.ordered_exposure_basis_mode,
            "adaptive_sparsity": True,
            "semiparametric_residual": bool(
                self.ordered_exposure_semiparametric_residual),
            "latent_structure_selection": bool(
                self.ordered_exposure_latent_structure_selection),
            "group_shared_shrinkage": bool(
                self.ordered_exposure_group_shared_shrinkage),
            "group_ridge_learning": bool(
                self.ordered_exposure_group_ridge_learning),
            "coefficient_prior_feature_dim": int(
                len(priors.get(1, {}).get("source_pip", []))),
            "coefficient_prior_source_only": True,
        })

    def _fit_stable_descriptor_projection(self, Z, records):
        by_domain = {}
        for index, rec in enumerate(records):
            by_domain.setdefault(rec.domain, []).append(index)
        relevance_rows = []
        sign_rows = []
        for indices in by_domain.values():
            idx = np.asarray(indices, dtype=int)
            Xd = Z[idx]
            signals = np.column_stack([
                np.asarray([float(records[i].y[0]) for i in idx], dtype=float),
                np.asarray([
                    (float(records[i].y[1]) - float(records[i].tau))
                    / self._source_margin_scale(records[i])
                    for i in idx
                ], dtype=float),
            ])
            Xd = Xd - np.mean(Xd, axis=0, keepdims=True)
            x_scale = np.sqrt(np.mean(Xd ** 2, axis=0))
            x_scale = np.where(x_scale < 1e-10, 1.0, x_scale)
            Xd = Xd / x_scale
            signals = signals - np.mean(signals, axis=0, keepdims=True)
            y_scale = np.sqrt(np.mean(signals ** 2, axis=0))
            y_scale = np.where(y_scale < 1e-10, 1.0, y_scale)
            signals = signals / y_scale
            corr = Xd.T @ signals / max(float(len(Xd)), 1.0)
            relevance_rows.append(np.max(np.abs(corr), axis=1))
            sign_rows.append(np.sign(corr))
        relevance = np.vstack(relevance_rows)
        signs = np.stack(sign_rows, axis=0)
        relevance_mean = np.mean(relevance, axis=0)
        relevance_std = np.std(relevance, axis=0)
        stability = relevance_mean / (relevance_mean + relevance_std + 1e-12)
        prevalence = np.mean(relevance >= self.coordinate_relevance_floor, axis=0)
        sign_consistency = np.empty(Z.shape[1], dtype=float)
        for feature in range(Z.shape[1]):
            agreements = []
            for signal in range(signs.shape[2]):
                values = signs[:, feature, signal]
                values = values[values != 0.0]
                positive = float(np.mean(values > 0.0)) if len(values) else 0.5
                agreements.append(max(positive, 1.0 - positive))
            sign_consistency[feature] = max(agreements)
        score = relevance_mean * (
            0.5 + 0.5 * stability
        ) * (
            0.5 + 0.5 * prevalence
        ) * (
            0.75 + 0.25 * sign_consistency
        )
        order = np.argsort(-score, kind="stable")
        selected = []
        corr_z = np.corrcoef(Z, rowvar=False)
        corr_z = np.nan_to_num(corr_z, nan=0.0)
        for feature in order:
            if any(abs(float(corr_z[int(feature), old])) > 0.97 for old in selected):
                continue
            selected.append(int(feature))
            if len(selected) >= self.local_dim:
                break
        for feature in order:
            if len(selected) >= self.local_dim:
                break
            if int(feature) not in selected:
                selected.append(int(feature))
        components = np.zeros((self.local_dim, Z.shape[1]), dtype=float)
        for row, feature in enumerate(selected[: self.local_dim]):
            components[row, feature] = 1.0
        self.pca_components = components
        names = self._descriptor_names()
        self.coordinate_diagnostics = {
            "mode": "stable_supervised",
            "selected_indices": selected[: self.local_dim],
            "selected_names": [
                names[index] if index < len(names) else f"descriptor{index}"
                for index in selected[: self.local_dim]
            ],
            "selected_scores": [float(score[index]) for index in selected[: self.local_dim]],
            "selected_prevalence": [
                float(prevalence[index]) for index in selected[: self.local_dim]
            ],
            "selected_sign_consistency": [
                float(sign_consistency[index]) for index in selected[: self.local_dim]
            ],
        }
        return Z

    def _fit_kmeans(self, Z):
        rng = np.random.default_rng(self.seed + 17)
        n = len(Z)
        k = max(1, min(self.shared_dim, n))
        if n == 0:
            self.cluster_centers = np.zeros((self.shared_dim, self.local_dim), dtype=float)
            return
        # Cluster in the local-exposure coordinates so A and N share a geometry
        # without using target-specific labels.
        A = Z @ self.pca_components.T
        first = int(rng.integers(0, n))
        centers = [A[first]]
        while len(centers) < k:
            C = np.vstack(centers)
            d2 = np.min(np.sum((A[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1)
            if float(np.sum(d2)) <= 1e-12:
                centers.append(A[int(rng.integers(0, n))])
            else:
                probs = d2 / float(np.sum(d2))
                centers.append(A[int(rng.choice(n, p=probs))])
        centers = np.vstack(centers)
        for _ in range(max(1, self.kmeans_iters)):
            d2 = np.sum((A[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            new_centers = centers.copy()
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = np.mean(A[mask], axis=0)
            if np.linalg.norm(new_centers - centers) <= 1e-8:
                break
            centers = new_centers
        if k < self.shared_dim:
            pad = np.repeat(centers[-1:], self.shared_dim - k, axis=0)
            centers = np.vstack([centers, pad])
        self.cluster_centers = centers[: self.shared_dim]

    def exposure_from_descriptor(self, descriptor):
        if (
            self.feature_mean is None
            or self.feature_scale is None
            or self.pca_components is None
            or self.cluster_centers is None
        ):
            raise RuntimeError("LearnedMetaPrior must be fit before use")
        z = self._scaled_descriptor(descriptor)
        A = z @ self.pca_components.T
        d2 = np.sum((A[None, :] - self.cluster_centers) ** 2, axis=1)
        N = _softmax_negdist(d2, self.soft_temperature)
        return np.asarray(A, dtype=float), np.asarray(N, dtype=float)

    def risk_exposure_from_descriptor(self, descriptor):
        A, N = self.exposure_from_descriptor(descriptor)
        return RiskExposure(
            A,
            N,
            local_names=tuple(f"meta_A{j}" for j in range(self.local_dim)),
            shared_names=tuple(f"meta_N{j}" for j in range(self.shared_dim)),
            meta={"provider": "LearnedMetaPrior", "source_domains": list(self.source_domains)},
        )

    def aligned_cumulative_risk_exposure_from_descriptor(
        self,
        descriptor,
        *,
        domain=None,
    ):
        """Nonnegative cumulative-risk blocks in the frozen aligned space.

        The four compact source-alignment statistics are semantic rather than
        named latent axes: boundary score, score magnitude, source-expert
        disagreement, and heteroscedastic risk correction.  Local exposures
        encode magnitudes; shared exposures are a soft regime occupancy.  The
        mapping is source-frozen and never reads held-out target labels.
        """

        if self.risk_subspace_alignment is None:
            return self.risk_exposure_from_descriptor(descriptor)
        compact = np.asarray(
            self.risk_subspace_alignment.transform_compact(
                np.asarray(descriptor, dtype=float),
                domain=domain,
                adapter=None,
            ),
            dtype=float,
        ).reshape(-1)
        compact = np.pad(compact, (0, max(0, 4 - len(compact))))[:4]
        score, score_magnitude, disagreement, correction = compact
        local_semantics = np.asarray([
            np.clip(score_magnitude / 2.0, 0.0, 1.0),
            np.clip(disagreement / 2.0, 0.0, 1.0),
            np.clip(abs(correction) / 2.0, 0.0, 1.0),
            np.clip(abs(score - correction) / 4.0, 0.0, 1.0),
        ], dtype=float)
        A = np.zeros(self.local_dim, dtype=float)
        A[: min(len(A), len(local_semantics))] = local_semantics[: len(A)]

        shared_logits = np.asarray([
            score,
            -score,
            correction,
            -correction,
            0.0,
        ], dtype=float)
        logits = np.zeros(self.shared_dim, dtype=float)
        logits[: min(len(logits), len(shared_logits))] = shared_logits[: len(logits)]
        logits -= float(np.max(logits)) if len(logits) else 0.0
        N = np.exp(np.clip(logits, -50.0, 0.0))
        N /= max(float(np.sum(N)), 1e-12)
        return RiskExposure(
            A,
            N,
            local_names=tuple(
                f"aligned_local_risk_{index}" for index in range(self.local_dim)
            ),
            shared_names=tuple(
                f"aligned_regime_{index}" for index in range(self.shared_dim)
            ),
            meta={
                "provider": "LearnedMetaPrior",
                "coordinate": "frozen_source_boundary_alignment",
                "source_domains": list(self.source_domains),
                "source_domain": None if domain is None else str(domain),
                "target_data_used": False,
            },
        )

    def cumulative_risk_exposure_from_descriptor(self, descriptor, *, domain=None):
        if (
            self.component_stage == "spectral_hvd"
            and self.risk_subspace_alignment is not None
        ):
            return self.aligned_cumulative_risk_exposure_from_descriptor(
                descriptor,
                domain=domain,
            )
        return self.risk_exposure_from_descriptor(descriptor)

    def ordered_local_exposure(self, problem, x):
        if self.ordered_exposure_diagnostics.get("status") != "fit":
            raise RuntimeError("ordered cumulative exposure is not fit")
        profile = np.asarray(problem.normalize(x), dtype=float)
        library = self._ordered_profile_library(profile)
        columns = [0, 1] + [
            int(frequency) + 1
            for frequency in self.ordered_exposure_selected_frequencies
        ]
        return self._ordered_coordinate_transform(np.asarray(
            library[columns] / self.ordered_exposure_scale, dtype=float))

    def _ordered_coordinate_transform(self, values):
        """Return the declared ordered coordinate system.

        The control uses a fixed invertible triangular mixing of the same DCT
        span.  It removes coordinate orthogonality without changing which
        source-learned functions are available, making the ablation narrower
        than replacing the representation family.
        """

        values = np.asarray(values, dtype=float).reshape(-1)
        if self.ordered_exposure_orthogonal_coordinates or len(values) <= 1:
            return values
        indices = np.arange(len(values))
        distance = indices[:, None] - indices[None, :]
        mixing = np.where(distance >= 0, 0.75 ** distance, 0.0)
        return np.asarray(mixing @ values, dtype=float)

    def ordered_cumulative_risk_exposure(self, problem, x, output_index=1):
        if self.observable_variance_input_mode == "observable_state_exposure":
            return self.observable_variance_risk_exposure(
                problem, x, output_index=output_index)
        del output_index
        descriptor = self.descriptor(problem, x)
        if (
            self.component_stage == "spectral_hvd"
            and self.risk_subspace_alignment is not None
        ):
            base = self.aligned_cumulative_risk_exposure_from_descriptor(
                descriptor,
                domain=None,
            )
        else:
            base = self.risk_exposure_from_descriptor(descriptor)
        A = self.ordered_local_exposure(problem, x)
        frequencies = self.ordered_exposure_selected_frequencies.tolist()
        return RiskExposure(
            A,
            base.N,
            local_names=("ordered_global_mean", "ordered_global_std") + tuple(
                f"ordered_dct_{frequency}" for frequency in frequencies
            ),
            shared_names=base.shared_names,
            meta={
                **dict(base.meta),
                "provider": "LearnedMetaPriorOrderedCumulative",
                "selected_frequencies": frequencies,
                "orthogonal_coordinates": bool(
                    self.ordered_exposure_orthogonal_coordinates),
                "source_only": True,
                "target_data_used": False,
            },
        )

    def ordered_coordinate_basis_features(self, problem, x):
        exposure = self.ordered_cumulative_risk_exposure(problem, x)
        return self._ordered_basis_from_exposure(exposure)

    def _ordered_basis_from_exposure(self, exposure):
        A = np.asarray(exposure.A, dtype=float)
        if self.ordered_exposure_basis_mode == "diagonal_quadratic":
            interactions = A ** 2
        else:
            interactions = np.asarray([
                A[i] * A[j]
                for i in range(len(A))
                for j in range(i, len(A))
            ], dtype=float)
        return np.concatenate([
            A,
            interactions,
            np.asarray(exposure.N, dtype=float),
        ])

    def risk_coordinate_from_descriptor(self, descriptor):
        exposure = self.risk_exposure_from_descriptor(descriptor)
        return np.concatenate([exposure.A, exposure.N]).astype(float)

    def observable_variance_risk_exposure_from_descriptor(self, descriptor):
        if self.observable_variance_model is None:
            raise RuntimeError("observable variance coordinate is unavailable")
        return self.observable_variance_model.risk_exposure_from_descriptor(
            descriptor)

    def observable_variance_risk_exposure(
        self, problem, x, output_index=1,
    ):
        del output_index
        if self.observable_variance_model is None:
            raise RuntimeError("observable variance coordinate is unavailable")
        descriptor = self.observable_state_descriptor(problem, x)
        if descriptor is None:
            raise ValueError(
                "observable variance coordinate requires a declared "
                "observable state/trajectory exposure"
            )
        return self.observable_variance_risk_exposure_from_descriptor(
            descriptor)

    def _cumulative_risk_exposure_from_record(self, record, *, aligned=True):
        if self.observable_variance_input_mode == "observable_state_exposure":
            if record.observable_state_descriptor is None:
                raise ValueError(
                    "observable variance coordinate requires source exposure "
                    "coverage for every record"
                )
            return self.observable_variance_risk_exposure_from_descriptor(
                record.observable_state_descriptor)
        if aligned:
            return self.cumulative_risk_exposure_from_descriptor(
                record.descriptor,
                domain=record.domain,
            )
        return self.risk_exposure_from_descriptor(record.descriptor)

    def risk_exposure(self, problem, x, output_index=1):
        if self.observable_variance_input_mode == "observable_state_exposure":
            return self.observable_variance_risk_exposure(
                problem, x, output_index=output_index)
        del output_index
        return self.risk_exposure_from_descriptor(self.descriptor(problem, x))

    def cumulative_risk_exposure(self, problem, x, output_index=1):
        if self.observable_variance_input_mode == "observable_state_exposure":
            return self.observable_variance_risk_exposure(
                problem, x, output_index=output_index)
        del output_index
        return self.cumulative_risk_exposure_from_descriptor(
            self.descriptor(problem, x))

    def risk_coordinate(self, problem, x):
        exposure = self.risk_exposure(problem, x)
        return np.concatenate([exposure.A, exposure.N]).astype(float)

    def cumulative_features(self, problem, x, output_index=1):
        return cumulative_feature_vector(self.cumulative_risk_exposure(
            problem, x, output_index=output_index))

    def cumulative_feature_names(self):
        if self.observable_variance_input_mode == "observable_state_exposure":
            prefix_local = "observable_variance_local_"
            prefix_shared = "observable_variance_regime_"
            return cumulative_feature_names(RiskExposure(
                np.zeros(self.local_dim),
                np.zeros(self.shared_dim),
                local_names=tuple(
                    prefix_local + str(index)
                    for index in range(self.local_dim)
                ),
                shared_names=tuple(
                    prefix_shared + str(index)
                    for index in range(self.shared_dim)
                ),
            ))
        aligned = (
            self.component_stage == "spectral_hvd"
            and self.risk_subspace_alignment is not None
        )
        return cumulative_feature_names(
            RiskExposure(
                np.zeros(self.local_dim),
                np.zeros(self.shared_dim),
                local_names=tuple(
                    ("aligned_local_risk_" if aligned else "meta_A") + str(j)
                    for j in range(self.local_dim)
                ),
                shared_names=tuple(
                    ("aligned_regime_" if aligned else "meta_N") + str(j)
                    for j in range(self.shared_dim)
                ),
            )
        )

    def hvd_features(self, problem, x):
        desc = self._scaled_descriptor(self.descriptor(problem, x))
        exposure = self.risk_exposure(problem, x)
        A = exposure.A
        N = exposure.N
        return np.concatenate([
            np.array([1.0], dtype=float),
            desc,
            A,
            N,
            A ** 2,
            N ** 2,
        ])

    def risk_class(self, problem, x):
        # Called after risk_exposure in most hot paths; recompute deliberately
        # for simple statelessness.
        return int(np.argmax(self.risk_exposure(problem, x).N))

    def _source_observation(self, problem, x, rng):
        mode = self.source_observation_mode
        replicates = (
            self.source_observation_replicates
            if mode == "replicated" else 1
        )
        observations = np.vstack([
            np.asarray(problem.simulate(x, rng), dtype=float)
            for _ in range(replicates)
        ])
        mean = np.mean(observations, axis=0)
        nominal = max(float(getattr(problem, "sigma_level", 0.04)), 1e-8)
        if mode == "analytic":
            sigma = (
                float(problem.true_sigma(x)[1])
                if hasattr(problem, "true_sigma") else nominal
            )
        elif mode == "replicated" and replicates > 1:
            centered = observations[:, 1] - float(mean[1])
            prior_df = 2.0
            variance = (
                float(centered @ centered) + prior_df * nominal ** 2
            ) / (float(replicates - 1) + prior_df)
            sigma = float(np.sqrt(max(variance, 1e-12)))
        else:
            sigma = nominal
        return mean, sigma, int(replicates), observations.copy()

    def _source_universal_design(self, problem, n, rng):
        """Select formula-free low-frequency profiles by geometric coverage."""

        n = max(0, int(n))
        if n == 0:
            return []
        library = self.universal_shape_candidates(
            problem,
            n=10000,
            rng=rng,
            force=True,
        )
        if len(library) <= n:
            return list(library)
        profiles = np.vstack([
            np.asarray(problem.normalize(x), dtype=float).reshape(-1)
            for x in library
        ])
        constant = [
            int(index) for index in np.flatnonzero(
                np.std(profiles, axis=1) <= 1e-10)
        ]
        head_tail = []
        ramps = []
        piecewise = []
        for index, profile in enumerate(profiles):
            if index in constant:
                continue
            if len(profile) > 1 and float(np.std(profile[1:])) <= 1e-10:
                head_tail.append(index)
            elif len(np.unique(np.round(profile, 10))) > 3:
                ramps.append(index)
            else:
                piecewise.append(index)

        def maximin(group, count):
            group = [int(index) for index in group]
            count = min(max(0, int(count)), len(group))
            if count == 0:
                return []
            chosen = [group[0]]
            distance = np.mean(
                (profiles[group] - profiles[chosen[0]][None, :]) ** 2,
                axis=1,
            )
            while len(chosen) < count:
                local = int(np.argmax(distance))
                chosen.append(group[local])
                distance = np.minimum(distance, np.mean(
                    (profiles[group] - profiles[group[local]][None, :]) ** 2,
                    axis=1,
                ))
                distance[[group.index(index) for index in chosen]] = -np.inf
            return chosen

        # Cover each generic low-frequency family before filling globally.
        selected = maximin(constant, min(len(constant), n))
        remaining = max(0, n - len(selected))
        ramp_count = min(len(ramps), max(1, int(round(0.15 * remaining))))
        selected.extend(maximin(ramps, ramp_count))
        remaining = max(0, n - len(selected))
        head_count = min(
            len(head_tail), max(1, int(round(0.45 * remaining))))
        selected.extend(maximin(head_tail, head_count))
        remaining = max(0, n - len(selected))
        selected.extend(maximin(piecewise, remaining))

        selected = list(dict.fromkeys(selected))
        minimum_distance = np.full(len(library), np.inf, dtype=float)
        for index in selected:
            minimum_distance = np.minimum(minimum_distance, np.mean(
                (profiles - profiles[index][None, :]) ** 2,
                axis=1,
            ))
        minimum_distance[selected] = -np.inf
        while len(selected) < n:
            index = int(np.argmax(minimum_distance))
            if not np.isfinite(minimum_distance[index]):
                break
            selected.append(index)
            minimum_distance = np.minimum(minimum_distance, np.mean(
                (profiles - profiles[index][None, :]) ** 2,
                axis=1,
            ))
            minimum_distance[selected] = -np.inf
        return [library[index] for index in selected[:n]]

    def _source_design_candidates(self, problem, n, rng):
        n = max(1, int(n))
        if self.source_design_mode == "random":
            # Keep sampling lazy so source-x draws and simulator-noise draws
            # retain the historical RNG interleaving exactly.
            return (
                (_as_tuple(problem.sample_random(rng)), "random")
                for _ in range(n)
            )
        if self.source_design_mode == "shared_uniform":
            # The same unrestricted normalized profiles are evaluated in
            # every source domain. This provides cross-domain paired ranks
            # without baking a low-frequency policy family into the archive.
            shared_rng = np.random.default_rng(self.seed + 104729)
            rows = []
            seen = set()
            while len(rows) < n:
                profile = shared_rng.uniform(
                    0.0, 1.0, size=int(getattr(problem, "d", 1)))
                x = _as_tuple(problem.continuous_to_int(profile))
                if x in seen:
                    continue
                seen.add(x)
                rows.append((x, "universal_shared_uniform"))
            return rows
        n_universal = min(
            n,
            max(1, int(round(n * self.source_universal_fraction))),
        )
        rows = [
            (_as_tuple(x), "universal_low_frequency")
            for x in self._source_universal_design(
                problem, n_universal, rng)
        ]
        seen = {x for x, _ in rows}
        while len(rows) < n:
            x = _as_tuple(problem.sample_random(rng))
            if x in seen:
                continue
            seen.add(x)
            rows.append((x, "random"))
        return rows[:n]

    @staticmethod
    def _canonical_profile(profile, size=64):
        """Map dimension-varying normalized policies to a common source grid."""

        profile = np.clip(
            np.asarray(profile, dtype=float).reshape(-1), 0.0, 1.0)
        size = max(2, int(size))
        if len(profile) == 0:
            return np.zeros(size, dtype=float)
        if len(profile) == size:
            return profile.copy()
        if len(profile) == 1:
            return np.full(size, float(profile[0]), dtype=float)
        return np.interp(
            np.linspace(0.0, 1.0, size),
            np.linspace(0.0, 1.0, len(profile)),
            profile,
        )

    @staticmethod
    def _percentile_ranks(values):
        """Return stable lower-is-better empirical percentile ranks."""

        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) <= 1:
            return np.zeros(len(values), dtype=float)
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float) / float(len(values) - 1)
        return ranks

    def _fit_source_consensus_templates(self, records):
        """Rank shared universal profiles using source observations only.

        Chance margins are first converted to within-domain percentile ranks,
        so a domain's arbitrary constraint scale cannot dominate another.  A
        profile is retained only when it was evaluated in every source domain.
        No held-out target values, target hooks, or analytic source truth enter
        this fit.
        """

        self.source_consensus_templates = []
        requested = int(self.source_consensus_template_count)
        if requested <= 0:
            self.source_consensus_diagnostics = {
                "status": "not_requested",
                "requested": requested,
                "target_data_used": False,
                "target_oracle_used": False,
            }
            return
        consensus_origin = (
            "universal_shared_uniform"
            if self.source_design_mode == "shared_uniform"
            else "universal_low_frequency"
        )
        usable = [
            rec for rec in records
            if rec.origin == consensus_origin and rec.profile is not None
        ]
        domains = sorted({str(rec.domain) for rec in usable})
        if len(domains) < 2:
            self.source_consensus_diagnostics = {
                "status": "insufficient_source_domains",
                "requested": requested,
                "n_source_domains": int(len(domains)),
                "target_data_used": False,
                "target_oracle_used": False,
            }
            return

        rank_by_record = {}
        objective_rank_by_record = {}
        for domain in domains:
            indices = [
                index for index, rec in enumerate(usable)
                if str(rec.domain) == domain
            ]
            ranks = self._percentile_ranks([
                self._source_margin(usable[index]) for index in indices
            ])
            for index, rank in zip(indices, ranks):
                rank_by_record[index] = float(rank)
            objective_ranks = self._percentile_ranks([
                float(usable[index].y[0]) for index in indices
            ])
            for index, rank in zip(indices, objective_ranks):
                objective_rank_by_record[index] = float(rank)

        grouped = {}
        for index, rec in enumerate(usable):
            canonical = self._canonical_profile(rec.profile)
            key = tuple(np.round(canonical, 6))
            grouped.setdefault(key, []).append((index, rec, canonical))

        rows = []
        for entries in grouped.values():
            by_domain = {}
            for index, rec, canonical in entries:
                domain = str(rec.domain)
                candidate = (
                    float(rank_by_record[index]),
                    abs(float(self._source_margin(rec))),
                    index,
                    rec,
                    canonical,
                )
                if domain not in by_domain or candidate[:3] < by_domain[domain][:3]:
                    by_domain[domain] = candidate
            if any(domain not in by_domain for domain in domains):
                continue
            ranks = np.asarray([
                by_domain[domain][0] for domain in domains
            ], dtype=float)
            objective_ranks = np.asarray([
                objective_rank_by_record[by_domain[domain][2]]
                for domain in domains
            ], dtype=float)
            margins = np.asarray([
                self._source_margin(by_domain[domain][3]) for domain in domains
            ], dtype=float)
            mean_rank = float(np.mean(ranks))
            worst_rank = float(np.max(ranks))
            disagreement = float(np.std(ranks))
            objective_mean_rank = float(np.mean(objective_ranks))
            objective_worst_rank = float(np.max(objective_ranks))
            objective_disagreement = float(np.std(objective_ranks))
            objective_score = (
                objective_mean_rank
                + 0.25 * objective_worst_rank
                + 0.25 * objective_disagreement
            )
            # Mean rank carries consensus; the remaining terms reject a
            # template that is excellent in only one source domain.
            score = mean_rank + 0.25 * worst_rank + 0.25 * disagreement
            rows.append({
                "profile": by_domain[domains[0]][4],
                "score": float(score),
                "mean_margin_rank": mean_rank,
                "worst_margin_rank": worst_rank,
                "rank_disagreement": disagreement,
                "objective_score": float(objective_score),
                "mean_objective_rank": objective_mean_rank,
                "worst_objective_rank": objective_worst_rank,
                "objective_rank_disagreement": objective_disagreement,
                "feasible_source_count": int(np.sum(margins <= 0.0)),
                "source_domain_count": int(len(domains)),
                "domain_margin_ranks": {
                    domain: float(by_domain[domain][0]) for domain in domains
                },
                "domain_objective_ranks": {
                    domain: float(objective_rank_by_record[
                        by_domain[domain][2]])
                    for domain in domains
                },
                "source_only": True,
                "target_data_used": False,
                "target_oracle_used": False,
            })
        rows.sort(key=lambda row: (
            float(row["score"]),
            float(row["mean_margin_rank"]),
            float(row["worst_margin_rank"]),
            tuple(np.round(row["profile"], 6)),
        ))
        self.source_consensus_templates = rows[:requested]
        self.source_consensus_diagnostics = {
            "status": "fit" if self.source_consensus_templates else "no_shared_profiles",
            "requested": requested,
            "n_source_domains": int(len(domains)),
            "source_domains": domains,
            "n_universal_records": int(len(usable)),
            "consensus_origin": consensus_origin,
            "n_shared_profiles": int(len(rows)),
            "n_selected_templates": int(len(self.source_consensus_templates)),
            "ranking_target": "observed_source_chance_margin_percentile",
            "objective_ranking_target": (
                "observed_source_objective_percentile"),
            "selected": [
                {
                    "score": float(row["score"]),
                    "mean_margin_rank": float(row["mean_margin_rank"]),
                    "worst_margin_rank": float(row["worst_margin_rank"]),
                    "rank_disagreement": float(row["rank_disagreement"]),
                    "objective_score": float(row["objective_score"]),
                    "mean_objective_rank": float(row["mean_objective_rank"]),
                    "worst_objective_rank": float(row["worst_objective_rank"]),
                    "objective_rank_disagreement": float(
                        row["objective_rank_disagreement"]),
                    "feasible_source_count": int(row["feasible_source_count"]),
                    "profile": np.round(row["profile"], 6).tolist(),
                }
                for row in self.source_consensus_templates
            ],
            "source_observation_mode": self.source_observation_mode,
            "source_only": True,
            "target_data_used": False,
            "target_oracle_used": False,
        }

    def _record_source_data(self, source_problems, n_records_per_domain, rng):
        records = []
        for domain_name, problem in source_problems:
            for x, origin in self._source_design_candidates(
                problem, n_records_per_domain, rng):
                y, sigma, replicate_count, replicate_values = self._source_observation(
                    problem, x, rng)
                records.append(SourceRecord(
                    domain=str(domain_name),
                    x=x,
                    y=y,
                    descriptor=self.descriptor(problem, x),
                    profile=np.asarray(problem.normalize(x), dtype=float),
                    tau=float(getattr(problem, "tau", 0.0)),
                    alpha=float(getattr(problem, "alpha", 0.05)),
                    sigma_level=float(getattr(problem, "sigma_level", 0.04)),
                    constraint_sigma=sigma,
                    observable_state_descriptor=(
                        self.observable_state_descriptor(problem, x)),
                    observable_state_invariant_descriptor=(
                        self.observable_state_descriptor(
                            problem, x, mode="set_invariant")),
                    observable_state_exposure=(
                        get_observable_state_exposure(problem, x)),
                    provider_risk_descriptor=self.provider_risk_descriptor(
                        problem, x),
                    provider_risk_coordinate=self.provider_risk_coordinate(
                        problem, x),
                    origin=origin,
                    sample_weight=1.0,
                    replicate_count=replicate_count,
                    replicates=replicate_values,
                ))
            records.extend(self._record_teacher_source_data(
                str(domain_name),
                problem,
                rng,
            ))
        return records

    def _candidate_pool_from_teacher_hooks(self, problem, rng):
        rows = []
        n_pool = max(1, int(self.teacher_pool_size))
        try:
            rows.extend(problem.initial_samples(n=min(64, n_pool), rng=rng))
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            rows.extend(problem.structured_candidates(n=min(128, n_pool), rng=rng))
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            for anchor in problem.state_anchor_points(n=min(64, n_pool), rng=rng):
                rows.extend(problem.inverse_state_anchor(anchor, rng=rng, n=2))
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            rows.extend(problem.recommendation_refinement_candidates())
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            rows.extend(problem.all_axis_solutions())
        except (AttributeError, TypeError, ValueError):
            pass
        rows = unique_candidates(rows)
        if len(rows) > max(n_pool * 4, 64):
            order = rng.permutation(len(rows))[: max(n_pool * 4, 64)]
            rows = [rows[int(i)] for i in order]
        return rows

    def _record_teacher_source_data(self, domain_name, problem, rng):
        n_teacher = max(0, int(self.teacher_records_per_domain))
        if n_teacher <= 0:
            return []
        rows = self._candidate_pool_from_teacher_hooks(problem, rng)
        if not rows:
            return []
        scored = []
        z_alpha = norm.ppf(1.0 - float(getattr(problem, "alpha", 0.05)))
        for x in rows:
            x = _as_tuple(x)
            if hasattr(problem, "true_outputs"):
                y = np.asarray(problem.true_outputs(x), dtype=float)
            else:
                y = np.asarray(problem.simulate(x, rng), dtype=float)
            if hasattr(problem, "true_sigma"):
                sigma_con = float(problem.true_sigma(x)[1])
            else:
                sigma_con = float(getattr(problem, "sigma_level", 0.04))
            tau = float(getattr(problem, "tau", 0.0))
            margin = float(y[1]) + z_alpha * sigma_con - tau
            scale = max(abs(tau), sigma_con, 1e-6)
            scaled = margin / scale
            feasible = margin <= 0.0
            score = (
                float(y[0])
                + self.feasible_penalty * max(scaled, 0.0)
                + self.boundary_weight * abs(scaled)
                - self.feasible_bonus * float(feasible)
            )
            scored.append({
                "x": x,
                "y": y,
                "sigma_con": sigma_con,
                "margin": margin,
                "scaled": scaled,
                "feasible": feasible,
                "score": float(score),
            })
        selected = []
        seen = set()
        n_keep = min(n_teacher, len(scored))
        n_elite = int(np.ceil(
            n_keep * np.clip(self.teacher_elite_fraction, 0.0, 1.0)))
        n_boundary = int(np.ceil(
            n_keep * np.clip(self.teacher_boundary_fraction, 0.0, 1.0)))

        def add(items, limit):
            for row in items:
                if len(selected) >= n_keep or limit <= 0:
                    break
                if row["x"] in seen:
                    continue
                seen.add(row["x"])
                selected.append(row)
                limit -= 1

        feasible_rows = [row for row in scored if row["feasible"]]
        feasible_rows.sort(key=lambda row: (row["y"][0], abs(row["scaled"])))
        add(feasible_rows, n_elite)
        boundary_rows = sorted(
            scored,
            key=lambda row: (abs(row["scaled"]), max(row["scaled"], 0.0), row["y"][0]),
        )
        add(boundary_rows, n_boundary)
        add(sorted(scored, key=lambda row: row["score"]), n_keep)

        if len(selected) < n_keep:
            chosen_desc = [
                self.descriptor(problem, row["x"])
                for row in selected
            ]
            while len(selected) < n_keep:
                if not chosen_desc:
                    add(sorted(scored, key=lambda row: row["score"]), 1)
                    chosen_desc = [
                        self.descriptor(problem, row["x"])
                        for row in selected
                    ]
                    continue
                D = np.vstack(chosen_desc)
                diverse = []
                for row in scored:
                    if row["x"] in seen:
                        continue
                    desc = self.descriptor(problem, row["x"])
                    dist = float(np.min(np.linalg.norm(D - desc[None, :], axis=1)))
                    diverse.append((dist, row))
                if not diverse:
                    break
                diverse.sort(key=lambda item: (-item[0], item[1]["score"]))
                add([diverse[0][1]], 1)
                chosen_desc.append(self.descriptor(problem, diverse[0][1]["x"]))

        records = []
        for row in selected:
            records.append(SourceRecord(
                domain=domain_name,
                x=row["x"],
                y=np.asarray(row["y"], dtype=float),
                descriptor=self.descriptor(problem, row["x"]),
                profile=np.asarray(problem.normalize(row["x"]), dtype=float),
                tau=float(getattr(problem, "tau", 0.0)),
                alpha=float(getattr(problem, "alpha", 0.05)),
                sigma_level=float(getattr(problem, "sigma_level", 0.04)),
                constraint_sigma=float(row["sigma_con"]),
                observable_state_descriptor=(
                    self.observable_state_descriptor(problem, row["x"])),
                observable_state_invariant_descriptor=(
                    self.observable_state_descriptor(
                        problem, row["x"], mode="set_invariant")),
                observable_state_exposure=(
                    get_observable_state_exposure(problem, row["x"])),
                provider_risk_descriptor=self.provider_risk_descriptor(
                    problem, row["x"]),
                provider_risk_coordinate=self.provider_risk_coordinate(
                    problem, row["x"]),
                origin="source_domain_tuned_teacher",
                sample_weight=max(float(self.teacher_weight), 1e-8),
                replicate_count=1,
            ))
        return records

    @staticmethod
    def _source_margin(rec):
        z = norm.ppf(1.0 - float(rec.alpha))
        sigma = (
            float(rec.constraint_sigma)
            if rec.constraint_sigma is not None
            else float(rec.sigma_level)
        )
        return float(rec.y[1]) + z * sigma - float(rec.tau)

    @staticmethod
    def _source_margin_scale(rec):
        sigma = (
            float(rec.constraint_sigma)
            if rec.constraint_sigma is not None
            else float(rec.sigma_level)
        )
        return max(abs(float(rec.tau)), sigma, 1e-6)

    def _boundary_sample_weight(self, rec):
        margin = self._source_margin(rec)
        scaled = margin / self._source_margin_scale(rec)
        temp = max(float(self.boundary_temperature), 1e-6)
        boundary = np.exp(-0.5 * (scaled / temp) ** 2)
        violation = max(float(scaled), 0.0)
        base = 1.0 + self.boundary_weight * boundary + 0.25 * violation
        return float(base * max(float(rec.sample_weight), 1e-8))

    def _record_training_diagnostics(self, records, weights):
        margins = np.asarray([self._source_margin(rec) for rec in records], dtype=float)
        scaled = np.asarray([
            self._source_margin(rec) / self._source_margin_scale(rec)
            for rec in records
        ], dtype=float)
        weights = np.asarray(weights, dtype=float)
        feasible = margins <= 0.0
        positive = np.maximum(margins, 0.0)
        near_boundary = np.abs(scaled) <= 1.0
        if np.any(near_boundary):
            slack_source = float(np.quantile(positive[near_boundary], 0.75))
        else:
            slack_source = float(np.quantile(positive, 0.75)) if len(positive) else 0.0
        self.training_diagnostics = {
            "source_feasible_rate": float(np.mean(feasible)) if len(feasible) else None,
            "source_margin_mean": float(np.mean(margins)) if len(margins) else None,
            "source_margin_median": float(np.median(margins)) if len(margins) else None,
            "source_scaled_margin_median_abs": (
                float(np.median(np.abs(scaled))) if len(scaled) else None
            ),
            "source_recommendation_slack": max(slack_source, 0.0),
            "boundary_weight_mean": float(np.mean(weights)) if len(weights) else None,
            "boundary_weight_max": float(np.max(weights)) if len(weights) else None,
            "boundary_temperature": float(self.boundary_temperature),
            "boundary_weight": float(self.boundary_weight),
            "variance_weight": float(self.variance_weight),
            "feasible_penalty": float(self.feasible_penalty),
            "feasible_bonus": float(self.feasible_bonus),
            "teacher_records_per_domain": int(self.teacher_records_per_domain),
            "teacher_weight": float(self.teacher_weight),
            "teacher_pool_size": int(self.teacher_pool_size),
            "hvd_noise_floor_scale": float(self.hvd_noise_floor_scale),
            "teacher_record_count": int(sum(
                1 for rec in records
                if rec.origin == "source_domain_tuned_teacher"
            )),
            "record_origins": {
                origin: int(sum(1 for rec in records if rec.origin == origin))
                for origin in sorted({rec.origin for rec in records})
            },
            "source_observation_mode": self.source_observation_mode,
            "source_observation_replicates": int(
                self.source_observation_replicates),
            "source_design_mode": self.source_design_mode,
            "source_universal_fraction": float(
                self.source_universal_fraction),
            "source_universal_record_count": int(sum(
                rec.origin in {
                    "universal_low_frequency",
                    "universal_shared_uniform",
                }
                for rec in records
            )),
            "source_analytic_sigma_used": bool(
                self.source_observation_mode == "analytic"),
            "source_analytic_teacher_used": bool(
                self.teacher_records_per_domain > 0),
            "source_simulator_calls": int(sum(
                max(1, int(rec.replicate_count)) for rec in records
                if rec.origin in {
                    "random",
                    "universal_low_frequency",
                    "universal_shared_uniform",
                }
            )),
        }

    def fit_from_source_problems(
        self,
        source_problems,
        n_records_per_domain=128,
        rng=None,
        hierarchical_boundary_config=None,
    ):
        rng = rng or np.random.default_rng(self.seed)
        source_problems = list(source_problems)
        if not source_problems:
            raise ValueError("at least one source domain is required")
        records = self._record_source_data(source_problems, n_records_per_domain, rng)
        if not records:
            raise ValueError("source training produced no records")
        # Preserve the exact ordinary observations consumed by source training.
        # Transfer baselines use this frozen archive rather than regenerating a
        # statistically similar but non-identical source dataset.
        self.source_records_ = copy.deepcopy(records)
        self.source_domains = sorted({rec.domain for rec in records})
        self.n_records = int(len(records))
        descriptors = [rec.descriptor for rec in records]
        Z = self._fit_scaler_pca(descriptors, records=records)
        self._fit_kmeans(Z)
        self.fit_status = "fitting"
        weights = [self._boundary_sample_weight(rec) for rec in records]
        self._record_training_diagnostics(records, weights)
        self._fit_source_consensus_templates(records)
        self.training_diagnostics["source_consensus_templates"] = copy.deepcopy(
            self.source_consensus_diagnostics)
        if self.observable_mean_coordinate:
            role_descriptor_requested = self.observable_mean_descriptor_mode in {
                "role_aligned",
                "role_transport",
                "role_intervention_transport",
                "role_adaptive_ordered",
                "role_adaptive_set_invariant",
            }
            usable = [
                rec for rec in records
                if rec.profile is not None
                and (
                    (
                        self.observable_mean_input_mode
                        != "provider_exposure"
                        or rec.provider_risk_descriptor is not None
                    )
                    and (
                        self.observable_mean_input_mode
                        != "observable_state_exposure"
                        or (
                            rec.observable_state_exposure is not None
                            if role_descriptor_requested
                            else (
                                rec.observable_state_invariant_descriptor is not None
                                if self.observable_mean_descriptor_mode
                                == "set_invariant"
                                else rec.observable_state_descriptor is not None
                            )
                        )
                    )
                )
            ]
            if (
                self.observable_mean_input_mode == "provider_exposure"
                and len(usable) != len(records)
            ):
                raise ValueError(
                    "provider exposure mean coordinate requires source "
                    "provider coverage for every record"
                )
            if (
                self.observable_mean_input_mode == "observable_state_exposure"
                and len(usable) != len(records)
            ):
                raise ValueError(
                    "observable-state mean coordinate requires a declared "
                    "observable exposure for every source record"
                )
            profiles = [rec.profile for rec in usable]
            domains = [rec.domain for rec in usable]
            observable_weight = [
                self._boundary_sample_weight(rec) for rec in usable
            ]
            if (
                self.observable_mean_input_mode
                == "observable_state_exposure"
                and role_descriptor_requested
            ):
                self.observable_channel_role_aligner = (
                    EquivariantChannelRoleAligner(
                        seed=self.seed,
                        partial_transport=(
                            self.observable_mean_descriptor_mode in {
                                "role_transport",
                                "role_intervention_transport",
                            }),
                        signature_mode=(
                            "intervention_response"
                            if self.observable_mean_descriptor_mode
                            == "role_intervention_transport"
                            else "distribution"
                        ),
                        barycentric_transport=(
                            self.observable_mean_descriptor_mode
                            == "role_intervention_transport"),
                    ).fit(
                        [rec.observable_state_exposure for rec in usable],
                        domains,
                        boundary_margins=[
                            self._source_margin(rec)
                            / self._source_margin_scale(rec)
                            for rec in usable
                        ],
                        profiles=[rec.profile for rec in usable],
                        source_problems=source_problems,
                    )
                )
            if self.observable_mean_mode == "boundary_aligned":
                # Representation supervision and coefficient supervision are
                # deliberately different. Chance-margin strata define phi;
                # source constraint means define the target GPR prior.
                if self.observable_mean_input_mode == "provider_exposure":
                    coordinate_inputs = [
                        rec.provider_risk_descriptor for rec in usable]
                elif (
                    self.observable_mean_input_mode
                    == "observable_state_exposure"
                ):
                    if role_descriptor_requested:
                        coordinate_inputs = [
                            (
                                self.observable_channel_role_aligner
                                .source_transport_descriptor(
                                    rec.domain,
                                    rec.observable_state_exposure,
                                )
                                if self.observable_mean_descriptor_mode in {
                                    "role_transport",
                                    "role_intervention_transport",
                                }
                                else self.observable_channel_role_aligner
                                .source_descriptor(
                                    rec.domain,
                                    rec.observable_state_exposure,
                                )
                            )
                            for rec in usable
                        ]
                    else:
                        coordinate_inputs = [
                            (
                                rec.observable_state_invariant_descriptor
                                if self.observable_mean_descriptor_mode
                                == "set_invariant"
                                else rec.observable_state_descriptor
                            )
                            for rec in usable
                        ]
                elif (
                    self.observable_mean_input_mode
                    == "source_learned_exposure"
                ):
                    coordinate_inputs = [
                        canonical_risk_descriptor(
                            self.risk_exposure_from_descriptor(rec.descriptor))
                        for rec in usable
                    ]
                else:
                    coordinate_inputs = profiles
                primary_descriptor_mode = (
                    (
                        "role_transport"
                        if self.observable_mean_descriptor_mode in {
                            "role_transport",
                            "role_intervention_transport",
                        }
                        else "role_aligned"
                    ) if role_descriptor_requested
                    else self.observable_mean_descriptor_mode)
                if (
                    self.observable_mean_descriptor_mode
                    == "exchangeable_equivariant"
                ):
                    primary_model = ExchangeableBoundaryMeanCoordinate(
                        ridge_grid=self.observable_mean_ridges,
                    ).fit(
                        [
                            rec.observable_state_exposure
                            for rec in usable
                        ],
                        [
                            (float(rec.y[1]) - float(rec.tau))
                            / self._source_margin_scale(rec)
                            for rec in usable
                        ],
                        domains,
                        sample_weight=observable_weight,
                    )
                else:
                    primary_model = SourceAlignedBoundaryCoordinate(
                        ridge_grid=self.observable_mean_ridges,
                        latent_dim=self.observable_mean_latent_dim,
                        input_mode=self.observable_mean_input_mode,
                        observable_descriptor_mode=primary_descriptor_mode,
                        feature_mode=self.observable_mean_feature_mode,
                        latent_transform=self.observable_mean_latent_transform,
                        channel_role_aligner=(
                            self.observable_channel_role_aligner),
                    ).fit(
                        coordinate_inputs,
                        [
                            self._source_margin(rec)
                            / self._source_margin_scale(rec)
                            for rec in usable
                        ],
                        domains,
                        sample_weight=observable_weight,
                        coefficient_targets=[
                            (float(rec.y[1]) - float(rec.tau))
                            / self._source_margin_scale(rec)
                            for rec in usable
                        ],
                        proposal_profiles=profiles,
                    )
                if self.observable_mean_descriptor_mode in {
                    "role_adaptive_ordered",
                    "role_adaptive_set_invariant",
                }:
                    fallback_mode = (
                        "set_invariant"
                        if self.observable_mean_descriptor_mode.endswith(
                            "set_invariant")
                        else "ordered"
                    )
                    fallback_inputs = [
                        (
                            rec.observable_state_invariant_descriptor
                            if fallback_mode == "set_invariant"
                            else rec.observable_state_descriptor
                        )
                        for rec in usable
                    ]
                    fallback_model = SourceAlignedBoundaryCoordinate(
                        ridge_grid=self.observable_mean_ridges,
                        latent_dim=self.observable_mean_latent_dim,
                        input_mode=self.observable_mean_input_mode,
                        observable_descriptor_mode=fallback_mode,
                        feature_mode=self.observable_mean_feature_mode,
                        latent_transform=(
                            self.observable_mean_latent_transform),
                    ).fit(
                        fallback_inputs,
                        [
                            self._source_margin(rec)
                            / self._source_margin_scale(rec)
                            for rec in usable
                        ],
                        domains,
                        sample_weight=observable_weight,
                        coefficient_targets=[
                            (float(rec.y[1]) - float(rec.tau))
                            / self._source_margin_scale(rec)
                            for rec in usable
                        ],
                        proposal_profiles=profiles,
                    )
                    self.observable_mean_model = (
                        SourceSupportedRoleBoundaryCoordinate(
                            primary_model,
                            fallback_model,
                            self.observable_channel_role_aligner,
                            fallback_mode=fallback_mode,
                        )
                    )
                else:
                    self.observable_mean_model = primary_model
            else:
                self.observable_mean_model = SourceLearnedObservableCoordinate(
                    ridge_grid=self.observable_mean_ridges,
                    output_mode=self.observable_mean_mode,
                    latent_dim=self.observable_mean_latent_dim,
                ).fit(
                    profiles,
                    [
                        (
                            (float(rec.y[1]) - float(rec.tau))
                            if self.observable_mean_training_target
                            == "constraint_mean"
                            else self._source_margin(rec)
                        ) / self._source_margin_scale(rec)
                        for rec in usable
                    ],
                    domains,
                    sample_weight=observable_weight,
                )
            self.training_diagnostics["observable_mean_coordinate"] = (
                self.observable_mean_model.diagnostics())
            self.training_diagnostics["observable_mean_coordinate"].update({
                "training_target": self.observable_mean_training_target,
                "coefficient_prior_training_target": (
                    "constraint_mean"
                    if self.observable_mean_mode == "boundary_aligned"
                    else self.observable_mean_training_target
                ),
                "chance_boundary_weighted": True,
                "input_mode": self.observable_mean_input_mode,
                "observable_descriptor_mode": (
                    self.observable_mean_descriptor_mode),
                "boundary_feature_mode": self.observable_mean_feature_mode,
                "latent_transform": self.observable_mean_latent_transform,
                "provider_structural_input": bool(
                    self.observable_mean_input_mode == "provider_exposure"),
                "observable_state_input": bool(
                    self.observable_mean_input_mode
                    == "observable_state_exposure"),
                "channel_role_alignment": (
                    None
                    if self.observable_channel_role_aligner is None
                    else self.observable_channel_role_aligner.diagnostics()
                ),
            })
        if self.observable_variance_input_mode == "observable_state_exposure":
            usable_variance = [
                rec for rec in records
                if rec.observable_state_descriptor is not None
            ]
            if len(usable_variance) != len(records):
                raise ValueError(
                    "observable-state variance coordinate requires a declared "
                    "observable exposure for every source record"
                )
            self.observable_variance_model = (
                SourceAlignedVarianceRiskCoordinate(
                    local_dim=self.local_dim,
                    shared_dim=self.shared_dim,
                    ridge_grid=self.observable_mean_ridges,
                    alignment_ridge=max(self.ridge, 0.1),
                    domain_penalty=1.0,
                    within_bin_weight=0.1,
                    soft_temperature=self.soft_temperature,
                ).fit(
                    [
                        rec.observable_state_descriptor
                        for rec in usable_variance
                    ],
                    [
                        max(float(
                            rec.constraint_sigma
                            if rec.constraint_sigma is not None
                            else rec.sigma_level
                        ) ** 2, 1e-12)
                        for rec in usable_variance
                    ],
                    [rec.domain for rec in usable_variance],
                    sample_weight=[
                        self._boundary_sample_weight(rec)
                        for rec in usable_variance
                    ],
                )
            )
            self.training_diagnostics["observable_variance_coordinate"] = {
                **self.observable_variance_model.diagnostics(),
                "variance_label_source": (
                    "source_replicated_sample_variance"
                    if self.source_observation_mode == "replicated"
                    else f"source_{self.source_observation_mode}_variance"
                ),
                "shared_observable_input_with_mean": bool(
                    self.observable_mean_model is not None
                    and self.observable_mean_input_mode
                    == "observable_state_exposure"
                ),
                "independent_head_parameters": True,
            }
        if hierarchical_boundary_config is not None:
            self._fit_hierarchical_boundary(
                records,
                dict(hierarchical_boundary_config),
            )
        self._fit_ordered_exposure(records)
        if self.component_enabled("spectral"):
            self._fit_spectral_basis(records)
            if (
                self.spectral_coefficient_shrinkage
                or self.spectral_adaptive_sparsity
            ):
                self._fit_spectral_coefficient_prior(records)
        self._fit_ordered_coefficient_prior(records)
        if self.component_enabled("hvd") or self.component_enabled("mean"):
            self._fit_hvd_beta_priors(records)
        if self.component_enabled("proposal"):
            self._fit_anchor_distribution(records)
        self.training_diagnostics["component_stage"] = self.component_stage
        self.training_diagnostics["enabled_components"] = [
            name for name in ("coordinate", "spectral", "hvd", "mean", "proposal")
            if self.component_enabled(name)
        ]
        self.fit_status = "fit"
        return self

    def _fit_hierarchical_boundary(self, records, config):
        descriptor_mode = str(config.get(
            "descriptor_mode", "learned_risk")).lower()
        if descriptor_mode not in self.VALID_BOUNDARY_DESCRIPTOR_MODES:
            raise ValueError(
                f"unknown hierarchical boundary descriptor mode {descriptor_mode!r}")
        model = HierarchicalSignedDistancePosterior(
            coordinate=str(config.get(
                "coordinate", "boundary_latent")),
            geometry=str(config.get("geometry", "low_rank_psd")),
            rank=int(config.get("rank", 2)),
            ridge=float(config.get("ridge", max(self.ridge, 1e-3))),
            domain_penalty=float(config.get("domain_penalty", 0.5)),
            boundary_temperature=float(config.get(
                "boundary_temperature", self.boundary_temperature)),
            adaptation_ridge=float(config.get("adaptation_ridge", 5.0)),
            upper_alpha=float(config.get("upper_alpha", 0.01)),
            calibration_prior_df=float(config.get(
                "calibration_prior_df", 2.0)),
            hierarchy_iterations=int(config.get(
                "hierarchy_iterations", 5)),
            effect_ridge=float(config.get("effect_ridge", 1.0)),
            rotation_mode=str(config.get("rotation_mode", "none")),
            rotation_ridge=float(config.get("rotation_ridge", 5.0)),
            target_residual_rank=int(config.get("target_residual_rank", 0)),
            residual_ridge=float(config.get("residual_ridge", 5.0)),
        )
        descriptors = np.vstack([
            self.boundary_descriptor_from_raw(
                rec.descriptor,
                mode=descriptor_mode,
                provider_risk_descriptor=rec.provider_risk_descriptor,
                provider_risk_coordinate=rec.provider_risk_coordinate,
                domain=rec.domain,
            )
            for rec in records
        ])
        margins = np.asarray([
            self._source_margin(rec) for rec in records
        ], dtype=float)
        domains = np.asarray([
            str(rec.domain) for rec in records
        ], dtype=object)
        sample_weight = np.asarray([
            self._boundary_sample_weight(rec) for rec in records
        ], dtype=float)
        margin_variance = np.asarray([
            max(float(
                rec.constraint_sigma
                if rec.constraint_sigma is not None
                else rec.sigma_level
            ) ** 2, 1e-10)
            for rec in records
        ], dtype=float)
        replicate_count = np.asarray([
            max(int(rec.replicate_count), 1) for rec in records
        ], dtype=float)
        model.fit(
            descriptors,
            margins,
            domains,
            sample_weight=sample_weight,
            margin_variance=margin_variance,
            replicate_count=replicate_count,
        )
        self.hierarchical_boundary_posterior = model
        self.hierarchical_boundary_descriptor_mode = descriptor_mode
        self.hierarchical_boundary_diagnostics = model.diagnostics()
        self.hierarchical_boundary_diagnostics.update({
            "descriptor_mode": descriptor_mode,
            "descriptor_dimension": int(descriptors.shape[1]),
            "provider_structural_input": bool(
                "provider_" in descriptor_mode),
            "source_provider_coverage": float(np.mean([
                rec.provider_risk_descriptor is not None for rec in records
            ])),
        })
        self.training_diagnostics["hierarchical_boundary"] = copy.deepcopy(
            self.hierarchical_boundary_diagnostics)

    def _fit_spectral_basis(self, records):
        by_domain = {}
        for rec in records:
            by_domain.setdefault(rec.domain, []).append(rec)
        batches = []
        alignment_batches = []
        for domain, domain_records in sorted(by_domain.items()):
            psi = np.vstack([
                self.risk_coordinate_from_descriptor(rec.descriptor)
                for rec in domain_records
            ])
            objective = np.asarray([
                float(rec.y[0]) for rec in domain_records
            ], dtype=float)
            constraint = np.asarray([
                self._source_margin(rec) / self._source_margin_scale(rec)
                for rec in domain_records
            ], dtype=float)
            observable_constraint = np.asarray([
                (
                    float(rec.y[1])
                    + norm.ppf(1.0 - float(rec.alpha))
                    * float(rec.sigma_level)
                    - float(rec.tau)
                ) / max(
                    abs(float(rec.tau)),
                    float(rec.sigma_level),
                    1e-6,
                )
                for rec in domain_records
            ], dtype=float)
            risk_correction = np.asarray([
                norm.ppf(1.0 - float(rec.alpha))
                * (
                    float(
                        rec.constraint_sigma
                        if rec.constraint_sigma is not None
                        else rec.sigma_level
                    )
                    - float(rec.sigma_level)
                ) / max(
                    abs(float(rec.tau)),
                    float(rec.sigma_level),
                    1e-6,
                )
                for rec in domain_records
            ], dtype=float)
            transferable_constraint = observable_constraint + risk_correction
            objective_scale = max(float(np.std(objective)), 1e-8)
            objective_z = (objective - float(np.median(objective))) / objective_scale
            temperature = max(float(self.boundary_temperature), 1e-6)
            boundary_sign = np.tanh(constraint / temperature)
            decision = (
                objective_z
                + self.feasible_penalty * np.maximum(constraint, 0.0)
                + 0.25 * np.abs(constraint)
            )
            signals = np.column_stack([
                objective,
                constraint,
                boundary_sign,
                decision,
            ])
            transferable_boundary_sign = np.tanh(
                transferable_constraint / temperature)
            transferable_decision = (
                objective_z
                + self.feasible_penalty
                * np.maximum(transferable_constraint, 0.0)
                + 0.25 * np.abs(transferable_constraint)
            )
            alignment_signals = np.column_stack([
                objective,
                transferable_constraint,
                transferable_boundary_sign,
                transferable_decision,
                risk_correction,
            ])
            sample_weight = np.asarray([
                self._boundary_sample_weight(rec) for rec in domain_records
            ], dtype=float)
            alignment_sample_weight = np.asarray([
                (
                    1.0
                    + self.boundary_weight * np.exp(
                        -0.5
                        * (value / max(self.boundary_temperature, 1e-6)) ** 2
                    )
                    + 0.25 * max(float(value), 0.0)
                ) * max(float(rec.sample_weight), 1e-8)
                for rec, value in zip(domain_records, transferable_constraint)
            ], dtype=float)
            batches.append(SourceDomainBatch(
                domain=str(domain),
                psi=psi,
                signals=signals,
                sample_weight=sample_weight,
                signal_weight=np.asarray([0.35, 1.0, 1.25, 0.75]),
            ))
            alignment_batches.append(SourceDomainBatch(
                domain=str(domain),
                psi=np.vstack([rec.descriptor for rec in domain_records]),
                signals=alignment_signals,
                sample_weight=alignment_sample_weight,
                signal_weight=np.asarray([0.35, 1.0, 1.25, 0.75, 0.50]),
            ))
        if (
            self.spectral_frequency_adaptation
            or self.spectral_additive_adaptation
        ):
            # Adaptation must challenge the same full Stage-1 model used by
            # the no-adaptation control.  Otherwise merely enabling a module
            # changes the fallback before the target gate makes a decision.
            self.spectral_basis = TransferableSpectralBasis(
                active_dim=max(1, self.spectral_active_dim),
                max_library_size=self.spectral_max_library_size,
                low_frequency_components=self.spectral_low_frequency_components,
                n_neighbors=self.spectral_graph_neighbors,
                relevance_floor=self.spectral_relevance_floor,
                ridge=self.ridge,
                orthogonalization=self.spectral_orthogonalization,
                use_low_frequency_score=self.spectral_low_frequency_prior,
            ).fit(batches)
            self.stage1_spectral_basis = self.spectral_basis
            self.spectral_feature_dim = int(self.spectral_basis.feature_dim)
            self.spectral_always_active_count = 0
        elif self.spectral_adaptive_sparsity:
            self.stage1_spectral_basis = TransferableSpectralBasis(
                active_dim=min(2, max(1, self.spectral_active_dim)),
                max_library_size=self.spectral_max_library_size,
                low_frequency_components=4,
                n_neighbors=self.spectral_graph_neighbors,
                relevance_floor=self.spectral_relevance_floor,
                ridge=self.ridge,
                orthogonalization=self.spectral_orthogonalization,
                use_low_frequency_score=self.spectral_low_frequency_prior,
            ).fit(batches)
            residual_dim = max(
                self.spectral_active_dim
                - int(self.stage1_spectral_basis.feature_dim),
                0,
            )
            if residual_dim:
                self.spectral_basis = TransferableSpectralBasis(
                    active_dim=residual_dim,
                    max_library_size=self.spectral_max_library_size,
                    low_frequency_components=self.spectral_low_frequency_components,
                    n_neighbors=self.spectral_graph_neighbors,
                    relevance_floor=self.spectral_relevance_floor,
                    ridge=self.ridge,
                    orthogonalization=(
                        "none"
                        if self.spectral_orthogonalization == "none"
                        else "ordered_cholesky"
                    ),
                    use_low_frequency_score=self.spectral_low_frequency_prior,
                ).fit(batches)
            else:
                self.spectral_basis = self.stage1_spectral_basis
            self.spectral_feature_dim = int(
                self.stage1_spectral_basis.feature_dim
            ) + (
                0
                if self.spectral_basis is self.stage1_spectral_basis
                else int(self.spectral_basis.feature_dim)
            )
            self.spectral_always_active_count = int(
                self.stage1_spectral_basis.feature_dim)
        else:
            self.spectral_basis = TransferableSpectralBasis(
                active_dim=self.spectral_active_dim,
                max_library_size=self.spectral_max_library_size,
                low_frequency_components=self.spectral_low_frequency_components,
                n_neighbors=self.spectral_graph_neighbors,
                relevance_floor=self.spectral_relevance_floor,
                ridge=self.ridge,
                orthogonalization=self.spectral_orthogonalization,
                use_low_frequency_score=self.spectral_low_frequency_prior,
            ).fit(batches)
            self.stage1_spectral_basis = self.spectral_basis
            self.spectral_feature_dim = int(self.spectral_basis.feature_dim)
            self.spectral_always_active_count = 0
        if self.spectral_risk_alignment:
            self._fit_risk_aligned_spectral_basis(alignment_batches)
        if self.spectral_frequency_adaptation:
            self._fit_spectral_frequency_bank(batches)
        if self.spectral_additive_adaptation:
            self._fit_spectral_additive_bank(batches)
        if (
            self.spectral_risk_alignment
            and self.spectral_alignment_source_episodes > 0
        ):
            self._fit_alignment_episode_prior(records)

    def _fit_risk_aligned_spectral_basis(self, batches):
        psi_dim = int(batches[0].psi.shape[1])
        self.risk_subspace_alignment = BoundaryAlignedRiskSubspaces(
            active_dim=min(self.spectral_alignment_active_dim, psi_dim),
            subspace_dim=self.spectral_alignment_subspace_dim,
            boundary_signal_index=1,
            boundary_temperature=self.boundary_temperature,
            domain_penalty=self.spectral_alignment_domain_penalty,
            ridge=self.ridge,
            apply_source_procrustes=self.spectral_alignment_source_procrustes,
            target_adapter_ridge=self.spectral_alignment_target_ridge,
            target_min_gain=self.spectral_alignment_target_min_gain,
            target_min_bins=self.spectral_alignment_target_min_bins,
        ).fit(batches)
        aligned_batches = self.risk_subspace_alignment.transform_batches(batches)
        aligned_active_dim = min(
            max(1, int(self.stage1_spectral_basis.feature_dim)),
            int(self.risk_subspace_alignment.feature_dim),
        )
        self.risk_aligned_spectral_basis = TransferableSpectralBasis(
            active_dim=aligned_active_dim,
            max_library_size=self.spectral_max_library_size,
            low_frequency_components=self.spectral_low_frequency_components,
            n_neighbors=self.spectral_graph_neighbors,
            relevance_floor=self.spectral_relevance_floor,
            ridge=self.ridge,
            orthogonalization=self.spectral_orthogonalization,
            use_low_frequency_score=self.spectral_low_frequency_prior,
        ).fit(aligned_batches)
        if self.spectral_frequency_adaptation:
            self._fit_risk_aligned_frequency_bank(aligned_batches)
        if self.spectral_additive_adaptation:
            self.risk_aligned_additive_bank = TransferableAdditiveGroupBank(
                max_groups=self.spectral_additive_max_groups,
                max_library_size=self.spectral_max_library_size,
                low_frequency_components=self.spectral_low_frequency_components,
                n_neighbors=self.spectral_graph_neighbors,
                relevance_floor=self.spectral_relevance_floor,
                temperature=self.spectral_additive_temperature,
                ridge=self.ridge,
                base_basis=self.risk_aligned_spectral_basis,
                strong_heredity=True,
                max_interactions=min(2, self.spectral_additive_target_max_groups),
            ).fit(aligned_batches)
        self.risk_alignment_diagnostics = {
            "status": "fit",
            "target_data_used": False,
            "input_space": "universal_observable_descriptor",
            "alignment": self.risk_subspace_alignment.diagnostics(),
            "spectral_basis": self.risk_aligned_spectral_basis.diagnostics(),
            "frequency_entries": [
                {
                    key: value
                    for key, value in entry.items()
                    if key != "basis"
                }
                for entry in self.risk_aligned_frequency_bank
            ],
            "additive": (
                None
                if self.risk_aligned_additive_bank is None
                else self.risk_aligned_additive_bank.diagnostics()
            ),
        }

    def _fit_alignment_episode_prior(self, records):
        # Keep one source history row per policy.  If both a random record and
        # a source historical-teacher record exist, retain the latter because
        # it carries the deliberately acquired boundary observation.
        unique = {}
        for rec in records:
            key = (str(rec.domain), tuple(rec.x))
            previous = unique.get(key)
            if previous is None or float(rec.sample_weight) > float(
                previous.sample_weight
            ):
                unique[key] = rec
        episode_records = list(unique.values())
        domains = np.asarray([
            str(rec.domain) for rec in episode_records], dtype=str)
        margins = np.asarray([
            self._source_margin(rec) / self._source_margin_scale(rec)
            for rec in episode_records
        ], dtype=float)
        baseline = np.vstack([
            self.stage1_spectral_features_from_descriptor(rec.descriptor)
            for rec in episode_records
        ])
        # `domain=...` applies leave-that-domain-out expert weights.  Thus the
        # pseudo-target source domain cannot replay its own fitted expert.
        aligned = np.vstack([
            self.risk_subspace_alignment.transform_compact(
                rec.descriptor, domain=str(rec.domain))
            for rec in episode_records
        ])
        self.alignment_episode_prior = SourceBoundaryEpisodePrior(
            pilot_size=self.spectral_alignment_episode_pilot_size,
            pilot_sizes=(
                self.spectral_alignment_episode_pilot_size,
                2 * self.spectral_alignment_episode_pilot_size,
                4 * self.spectral_alignment_episode_pilot_size,
            ),
            evaluation_size=self.spectral_alignment_episode_evaluation_size,
            episodes_per_domain=self.spectral_alignment_source_episodes,
            ridge=self.spectral_alignment_episode_ridge,
            seed=self.seed + 17041,
        ).fit(domains, margins, baseline, aligned)
        self.alignment_episode_diagnostics = (
            self.alignment_episode_prior.diagnostics())
        self.risk_alignment_diagnostics[
            "source_boundary_episode_admission"
        ] = dict(self.alignment_episode_diagnostics)
        self._fit_alignment_profile_templates(episode_records)

    def _fit_alignment_profile_templates(self, records):
        templates = []
        seen = set()
        by_domain = {}
        for rec in records:
            if rec.profile is not None and len(rec.profile):
                by_domain.setdefault(str(rec.domain), []).append(rec)
        for domain, domain_records in sorted(by_domain.items()):
            rows = sorted(
                domain_records,
                key=lambda rec: (
                    0 if self._source_margin(rec) <= 0.0 else 1,
                    abs(self._source_margin(rec) / self._source_margin_scale(rec)),
                    float(rec.y[0]),
                ),
            )
            feasible = [rec for rec in rows if self._source_margin(rec) <= 0.0]
            infeasible = [rec for rec in rows if self._source_margin(rec) > 0.0]
            selected = feasible[:8] + infeasible[:8]
            for rec in selected:
                profile = np.clip(
                    np.asarray(rec.profile, dtype=float).reshape(-1), 0.0, 1.0)
                key = tuple(np.round(profile, 4))
                if key in seen:
                    continue
                seen.add(key)
                templates.append({
                    "profile": profile,
                    "domain": domain,
                    "feasible": bool(self._source_margin(rec) <= 0.0),
                    "scaled_margin": float(
                        self._source_margin(rec) / self._source_margin_scale(rec)),
                    "origin": str(rec.origin),
                    "aligned_coordinate": np.asarray(
                        self.risk_subspace_alignment.transform_compact(
                            rec.descriptor,
                            domain=domain,
                        ),
                        dtype=float,
                    ),
                })
        self.alignment_profile_templates = templates
        self._fit_source_boundary_bracket_model()
        self.alignment_episode_diagnostics["profile_template_count"] = int(
            len(templates))
        self.alignment_episode_diagnostics[
            "profile_template_source_only"
        ] = True
        self.risk_alignment_diagnostics[
            "source_boundary_episode_admission"
        ] = dict(self.alignment_episode_diagnostics)

    @staticmethod
    def _source_boundary_ridge_fit(train_x, train_y, ridge, test_x=None):
        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float)
        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        Phi = np.column_stack([
            np.ones(len(train_x), dtype=float),
            (train_x - mean) / scale,
        ])
        penalty = float(ridge) * np.eye(Phi.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = Phi.T @ Phi + penalty
        lhs_inv = np.linalg.pinv(lhs)
        beta = lhs_inv @ (Phi.T @ train_y)
        prediction = None
        if test_x is not None:
            test_x = np.asarray(test_x, dtype=float)
            Phi_test = np.column_stack([
                np.ones(len(test_x), dtype=float),
                (test_x - mean) / scale,
            ])
            prediction = Phi_test @ beta
        leverage = np.sum((Phi @ lhs_inv) * Phi, axis=1)
        effective_rank = float(np.sum(np.clip(leverage, 0.0, 1.0)))
        return beta, mean, scale, prediction, effective_rank

    def _fit_source_boundary_bracket_model(self):
        """Learn a source-only ordering of the chance boundary coordinate.

        The fitted score is deliberately used only for quantile stratification
        on an unlabeled held-out pool.  Its absolute zero is never interpreted
        as the target task's feasibility threshold.
        """
        rows = [
            row for row in self.alignment_profile_templates
            if row.get("aligned_coordinate") is not None
            and np.all(np.isfinite(row["aligned_coordinate"]))
        ]
        domains = np.asarray([str(row["domain"]) for row in rows], dtype=object)
        unique_domains = sorted(set(domains.tolist()))
        if len(rows) < 6 or len(unique_domains) < 2:
            self.source_boundary_bracket_model = {
                "status": "insufficient_source_support",
                "n_records": int(len(rows)),
                "n_domains": int(len(unique_domains)),
                "target_data_used": False,
                "target_oracle_used": False,
            }
            return
        coordinates = np.vstack([
            np.asarray(row["aligned_coordinate"], dtype=float) for row in rows
        ])
        target = np.clip(np.asarray([
            float(row["scaled_margin"]) for row in rows
        ], dtype=float), -4.0, 4.0)
        target_scale = max(float(np.var(target)), 1e-8)
        ridge_grid = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
        candidates = []

        def add_candidate(prediction, *, kind, ridge):
            prediction = np.asarray(prediction, dtype=float)
            valid = np.isfinite(prediction)
            if not np.any(valid):
                return
            nmse = float(np.mean(
                (target[valid] - prediction[valid]) ** 2)) / target_scale
            false_safe = float(np.mean(
                (target[valid] > 0.0) & (prediction[valid] <= 0.0)))
            pair_losses = []
            for domain in unique_domains:
                index = np.where((domains == domain) & valid)[0]
                if len(index) < 2:
                    continue
                upper = np.triu_indices(len(index), k=1)
                true_diff = (
                    target[index, None] - target[None, index])[upper]
                pred_diff = (
                    prediction[index, None] - prediction[None, index])[upper]
                informative = np.abs(true_diff) > 1e-10
                if np.any(informative):
                    pair_losses.extend(
                        (true_diff[informative] * pred_diff[informative] < 0.0)
                        .astype(float).tolist()
                    )
            rank_loss = float(np.mean(pair_losses)) if pair_losses else 0.5
            # This model is used to order an unlabeled target pool, not to
            # transfer an absolute target margin.  Ranking therefore dominates
            # scale-sensitive NMSE during source LODO selection.
            candidates.append({
                "kind": str(kind),
                "ridge": float(ridge),
                "score": float(
                    rank_loss + 0.25 * false_safe + 0.10 * nmse),
                "lodo_nmse": float(nmse),
                "lodo_rank_loss": float(rank_loss),
                "lodo_false_safe_rate": float(false_safe),
            })

        # The first compact alignment coordinate is already a source-trained,
        # leave-domain-out ensemble boundary score.  Keep it as a candidate so
        # a second ridge layer cannot erase useful ordering by choosing a
        # nearly constant high-penalty fit.
        add_candidate(
            coordinates[:, 0],
            kind="direct_aligned_boundary_score",
            ridge=0.0,
        )
        for ridge in ridge_grid:
            prediction = np.full(len(rows), np.nan, dtype=float)
            for heldout_domain in unique_domains:
                test = domains == heldout_domain
                train = ~test
                if int(np.sum(train)) < 2:
                    continue
                _, _, _, fold_prediction, _ = self._source_boundary_ridge_fit(
                    coordinates[train], target[train], ridge, coordinates[test])
                prediction[test] = fold_prediction
            add_candidate(
                prediction,
                kind="nested_ridge",
                ridge=ridge,
            )
        if not candidates:
            self.source_boundary_bracket_model = {
                "status": "fit_failed",
                "target_data_used": False,
                "target_oracle_used": False,
            }
            return
        selected = min(candidates, key=lambda row: (
            row["score"], row["lodo_rank_loss"], row["ridge"]))
        if selected["kind"] == "direct_aligned_boundary_score":
            beta = np.zeros(coordinates.shape[1] + 1, dtype=float)
            beta[1] = 1.0
            mean = np.zeros(coordinates.shape[1], dtype=float)
            scale = np.ones(coordinates.shape[1], dtype=float)
            fitted = coordinates[:, 0]
            effective_rank = 1.0
        else:
            beta, mean, scale, fitted, effective_rank = (
                self._source_boundary_ridge_fit(
                    coordinates,
                    target,
                    selected["ridge"],
                    coordinates,
                )
            )
        self.source_boundary_bracket_model = {
            "status": "fit",
            "method": "source_domain_lodo_boundary_ordering",
            "beta": beta,
            "feature_mean": mean,
            "feature_scale": scale,
            "score_kind": str(selected["kind"]),
            "ridge": float(selected["ridge"]),
            "effective_rank": float(effective_rank),
            "n_records": int(len(rows)),
            "n_domains": int(len(unique_domains)),
            "source_domains": list(unique_domains),
            "score": float(selected["score"]),
            "lodo_nmse": float(selected["lodo_nmse"]),
            "lodo_rank_loss": float(selected["lodo_rank_loss"]),
            "lodo_false_safe_rate": float(selected["lodo_false_safe_rate"]),
            "full_fit_rank_correlation_proxy": float(
                1.0 - np.mean(
                    ((target[:, None] - target[None, :])
                     * (fitted[:, None] - fitted[None, :])) < 0.0
                )
            ),
            "ridge_candidates": candidates,
            "target_data_used": False,
            "target_oracle_used": False,
        }

    def source_boundary_bracket_candidates(
        self, problem, n=5, rng=None, pool_size=None,
    ):
        """Cover source-predicted risk strata on an unlabeled target pool."""
        n = max(0, int(n))
        model = self.source_boundary_bracket_model
        if n == 0 or model.get("status") != "fit":
            return []
        rng = rng or np.random.default_rng(self.seed)
        pool_size = max(
            64,
            int(pool_size or self.spectral_alignment_inverse_pool_size),
            16 * n,
        )
        pool = self._alignment_inverse_pool(problem, rng, pool_size)
        if not pool:
            return []
        coordinates = np.vstack([
            self.frozen_risk_aligned_coordinate(problem, x) for x in pool
        ])
        scaled = (
            coordinates - np.asarray(model["feature_mean"], dtype=float)
        ) / np.asarray(model["feature_scale"], dtype=float)
        scores = np.column_stack([
            np.ones(len(scaled), dtype=float), scaled
        ]) @ np.asarray(model["beta"], dtype=float)
        score_scale = max(float(np.std(scores)), 1e-8)
        coordinate_scale = np.std(coordinates, axis=0)
        coordinate_scale = np.where(coordinate_scale < 1e-8, 1.0, coordinate_scale)
        quantile_cycle = (0.05, 0.25, 0.50, 0.75, 0.95)
        selected = []
        selected_index = []
        for step in range(min(n, len(pool))):
            quantile = quantile_cycle[step % len(quantile_cycle)]
            target_score = float(np.quantile(scores, quantile))
            merit = np.abs(scores - target_score) / score_scale
            if selected_index:
                previous = coordinates[np.asarray(selected_index, dtype=int)]
                distance = np.linalg.norm(
                    (coordinates[:, None, :] - previous[None, :, :])
                    / coordinate_scale[None, None, :],
                    axis=2,
                )
                merit -= 0.15 * np.minimum(np.min(distance, axis=1), 4.0)
            if selected_index:
                merit[np.asarray(selected_index, dtype=int)] = np.inf
            position = int(np.argmin(merit))
            selected_index.append(position)
            selected.append(tuple(pool[position]))
        self.alignment_episode_diagnostics["last_boundary_bracket"] = {
            "status": "ok",
            "method": "source_score_target_unlabeled_quantile_bracket",
            "n_candidates": int(len(selected)),
            "pool_size": int(len(pool)),
            "score_min": float(np.min(scores)),
            "score_median": float(np.median(scores)),
            "score_max": float(np.max(scores)),
            "quantile_cycle": list(quantile_cycle),
            "source_score_kind": str(model.get("score_kind", "nested_ridge")),
            "source_model_ridge": float(model["ridge"]),
            "target_labels_used": False,
            "target_oracle_used": False,
        }
        return unique_candidates(selected)[:n]

    def alignment_profile_candidates(self, problem, n=16, rng=None):
        """Replay frozen source boundary profiles without target-side hooks."""
        n = max(0, int(n))
        if n == 0 or not self.alignment_profile_templates:
            return []
        rng = rng or np.random.default_rng(self.seed)
        templates = list(self.alignment_profile_templates)
        # Balance source domains and source feasibility labels before thinning.
        buckets = {}
        for row in templates:
            buckets.setdefault(
                (str(row["domain"]), bool(row["feasible"])), []).append(row)
        for rows in buckets.values():
            rows.sort(key=lambda row: abs(float(row["scaled_margin"])))
        ordered = []
        keys = sorted(buckets)
        while any(buckets[key] for key in keys):
            for key in keys:
                if buckets[key]:
                    ordered.append(buckets[key].pop(0))
        if len(ordered) > n:
            head_count = max(1, min(n, n // 2))
            head = ordered[:head_count]
            tail = ordered[head_count:]
            need = n - len(head)
            if need > 0 and tail:
                pick = rng.permutation(len(tail))[:need]
                head.extend(tail[int(index)] for index in pick)
            ordered = head
        d = max(1, int(getattr(problem, "d", 1)))
        candidates = []
        for row in ordered[:n]:
            profile = np.asarray(row["profile"], dtype=float).reshape(-1)
            if len(profile) != d:
                profile = np.interp(
                    np.linspace(0.0, 1.0, d),
                    np.linspace(0.0, 1.0, len(profile)),
                    profile,
                )
            candidates.append(self._continuous_to_tuple(
                problem, np.clip(profile, 0.0, 1.0)))
        return unique_candidates(candidates)[:n]

    def _alignment_inverse_pool(self, problem, rng, pool_size):
        d = max(1, int(getattr(problem, "d", 1)))
        pool_size = max(64, int(pool_size))
        rows = self.alignment_profile_candidates(
            problem,
            n=min(len(self.alignment_profile_templates), pool_size),
            rng=rng,
        )
        rows = unique_candidates(rows)
        seen = {tuple(row) for row in rows}

        def add(row):
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                rows.append(key)

        templates = list(self.alignment_profile_templates)
        universal_budget = min(32, max(8, pool_size // 8))
        for row in self.universal_shape_candidates(
            problem,
            n=universal_budget,
            rng=rng,
            force=True,
        ):
            add(row)
        for level in (0.0, 0.05, 0.95, 1.0):
            add(self._continuous_to_tuple(
                problem,
                np.full(d, level, dtype=float),
            ))
        add(self._continuous_to_tuple(
            problem, np.linspace(0.0, 1.0, d)))
        add(self._continuous_to_tuple(
            problem, np.linspace(1.0, 0.0, d)))
        n_random = max(8, pool_size // 4)
        for _ in range(n_random):
            add(problem.sample_random(rng))
        grid = np.linspace(0.0, 1.0, d)
        knot_grid = np.linspace(0.0, 1.0, 5)
        attempts = 0
        while len(rows) < pool_size and attempts < 8 * pool_size:
            index = len(rows)
            if templates and index % 2 == 0:
                template = templates[index % len(templates)]
                profile = np.asarray(
                    template["profile"], dtype=float).reshape(-1)
                base = np.interp(
                    grid,
                    np.linspace(0.0, 1.0, len(profile)),
                    profile,
                )
                perturbation = np.interp(
                    grid,
                    knot_grid,
                    rng.normal(0.0, 0.10, size=len(knot_grid)),
                )
                z = base + perturbation
            else:
                z = np.interp(
                    grid,
                    knot_grid,
                    rng.uniform(0.05, 0.95, size=len(knot_grid)),
                )
            if index % 5 == 0:
                z[0] = rng.uniform(0.0, 1.0)
            add(self._continuous_to_tuple(
                problem, np.clip(z, 0.0, 1.0)))
            attempts += 1
        return rows[:pool_size]

    def alignment_latent_candidates(
        self, problem, n=16, rng=None, pool_size=None,
    ):
        """Invert frozen source boundary coordinates without target labels."""

        n = max(0, int(n))
        if (
            n == 0
            or self.risk_subspace_alignment is None
            or not self.alignment_profile_templates
        ):
            return []
        rng = rng or np.random.default_rng(self.seed)
        pool_size = max(
            64,
            int(pool_size or self.spectral_alignment_inverse_pool_size),
        )
        pool = self._alignment_inverse_pool(problem, rng, pool_size)
        coordinates = np.vstack([
            self.frozen_risk_aligned_coordinate(problem, x) for x in pool
        ])
        targets = [
            row for row in self.alignment_profile_templates
            if row.get("aligned_coordinate") is not None
            and np.all(np.isfinite(row["aligned_coordinate"]))
        ]
        targets.sort(key=lambda row: (
            not bool(row["feasible"]),
            abs(float(row["scaled_margin"])),
            str(row["domain"]),
        ))
        if not targets:
            return []
        target_matrix = np.vstack([
            np.asarray(row["aligned_coordinate"], dtype=float)
            for row in targets
        ])
        scale = np.std(target_matrix, axis=0)
        scale = np.where(scale < 0.10, 0.10, scale)
        selected = []
        used = set()
        cursor = 0
        while len(selected) < min(n, len(pool)):
            target = target_matrix[cursor % len(target_matrix)]
            distance = np.linalg.norm(
                (coordinates - target[None, :]) / scale[None, :],
                axis=1,
            )
            for position in np.argsort(distance):
                key = tuple(pool[int(position)])
                if key not in used:
                    used.add(key)
                    selected.append(key)
                    break
            cursor += 1
            if cursor > len(target_matrix) + len(pool):
                break
        self.alignment_episode_diagnostics["last_latent_inverse"] = {
            "status": "ok",
            "n_candidates": int(len(selected)),
            "pool_size": int(len(pool)),
            "n_source_targets": int(len(targets)),
            "n_source_feasible_targets": int(sum(
                bool(row["feasible"]) for row in targets)),
            "target_labels_used": False,
            "target_oracle_used": False,
        }
        return selected[:n]

    def alignment_latent_proposal_supported(self):
        prior = self.alignment_episode_prior
        diagnostics = (
            prior.diagnostics() if prior is not None else {})
        threshold = (
            float(prior.min_global_win_rate)
            if prior is not None
            else (2.0 / 3.0)
        )
        win_rate = diagnostics.get("evaluation_win_rate")
        accepted = bool(
            diagnostics.get("status") == "fit"
            and win_rate is not None
            and float(win_rate) >= threshold
        )
        self.alignment_episode_diagnostics[
            "latent_proposal_source_gate"
        ] = {
            "status": "accepted" if accepted else "fallback_direct",
            "source_evaluation_win_rate": (
                None if win_rate is None else float(win_rate)),
            "minimum_source_win_rate": float(threshold),
            "target_data_used": False,
            "target_oracle_used": False,
        }
        return accepted

    def _fit_risk_aligned_frequency_bank(self, aligned_batches):
        active_dim = int(self.risk_aligned_spectral_basis.feature_dim)
        basis_by_cutoff = {}
        entries = []
        for cutoff in self.spectral_frequency_cutoffs:
            if int(cutoff) == int(self.spectral_low_frequency_components):
                basis = self.risk_aligned_spectral_basis
            else:
                basis = TransferableSpectralBasis(
                    active_dim=active_dim,
                    max_library_size=self.spectral_max_library_size,
                    low_frequency_components=int(cutoff),
                    n_neighbors=self.spectral_graph_neighbors,
                    relevance_floor=self.spectral_relevance_floor,
                    ridge=self.ridge,
                    orthogonalization=self.spectral_orthogonalization,
                    use_low_frequency_score=self.spectral_low_frequency_prior,
                ).fit(aligned_batches)
            basis_by_cutoff[int(cutoff)] = basis
            diag = basis.diagnostics()
            cv_loss = np.asarray(
                diag.get("selected_pilot_cv_loss", []), dtype=float)
            source_score = (
                float(np.mean(cv_loss)) if len(cv_loss) else float("inf"))
            for ridge in self.spectral_frequency_ridges:
                entries.append({
                    "cutoff": int(cutoff),
                    "ridge": float(ridge),
                    "basis": basis_by_cutoff[int(cutoff)],
                    "source_score": source_score,
                    "is_stage1_baseline": bool(
                        int(cutoff) == int(self.spectral_low_frequency_components)
                        and np.isclose(float(ridge), 1.0)
                    ),
                    "base_variant": "risk_aligned_spectral",
                })
        scores = np.asarray([entry["source_score"] for entry in entries], dtype=float)
        finite = np.isfinite(scores)
        reference = float(np.min(scores[finite])) if np.any(finite) else 0.0
        logits = np.where(
            finite,
            -(scores - reference) / self.spectral_frequency_temperature,
            -50.0,
        )
        logits -= float(np.max(logits))
        weights = np.exp(np.clip(logits, -50.0, 0.0))
        weights /= max(float(np.sum(weights)), 1e-12)
        for index, (entry, weight) in enumerate(zip(entries, weights)):
            entry["index"] = int(index)
            entry["variant"] = f"aligned_frequency_band_{index:02d}"
            entry["source_weight"] = float(weight)
        if sum(bool(entry["is_stage1_baseline"]) for entry in entries) != 1:
            raise RuntimeError(
                "aligned frequency bank must contain exactly one fallback")
        self.risk_aligned_frequency_bank = entries

    def risk_aligned_coordinate_from_descriptor(self, descriptor, adapter=None):
        if self.risk_subspace_alignment is None:
            raise RuntimeError("risk-aligned subspace representation is not enabled")
        return self.risk_subspace_alignment.transform_compact(
            np.asarray(descriptor, dtype=float), adapter=adapter)

    def frozen_risk_aligned_coordinate_from_descriptor(self, descriptor):
        if self.risk_subspace_alignment is None:
            raise RuntimeError("risk-aligned subspace representation is not enabled")
        return self.risk_subspace_alignment.transform_compact(
            np.asarray(descriptor, dtype=float), domain=None, adapter=None)

    def frozen_risk_aligned_coordinate(self, problem, x):
        return self.frozen_risk_aligned_coordinate_from_descriptor(
            self.descriptor(problem, x))

    def risk_aligned_full_coordinate_from_descriptor(
        self, descriptor, adapter=None,
    ):
        if self.risk_subspace_alignment is None:
            raise RuntimeError("risk-aligned subspace representation is not enabled")
        return self.risk_subspace_alignment.transform(
            np.asarray(descriptor, dtype=float), adapter=adapter)

    def risk_aligned_coordinate(self, problem, x, adapter=None):
        return self.risk_aligned_coordinate_from_descriptor(
            self.descriptor(problem, x), adapter=adapter)

    def risk_aligned_spectral_features(self, problem, x, adapter=None):
        if self.risk_aligned_spectral_basis is None:
            raise RuntimeError("risk-aligned spectral basis is not enabled")
        coordinate = self.risk_aligned_full_coordinate_from_descriptor(
            self.descriptor(problem, x), adapter=adapter)
        return self.risk_aligned_spectral_basis.transform(coordinate)

    def risk_aligned_frequency_features(self, problem, x, index, adapter=None):
        coordinate = self.risk_aligned_full_coordinate_from_descriptor(
            self.descriptor(problem, x), adapter=adapter)
        entry = self.risk_aligned_frequency_bank[int(index)]
        return entry["basis"].transform(coordinate)

    def risk_aligned_additive_features(
        self, problem, x, indices, adapter=None,
    ):
        if self.risk_aligned_additive_bank is None:
            raise RuntimeError("risk-aligned additive bank is not enabled")
        coordinate = self.risk_aligned_full_coordinate_from_descriptor(
            self.descriptor(problem, x), adapter=adapter)
        return self.risk_aligned_additive_bank.transform_groups(
            coordinate, indices)

    def fit_target_risk_alignment(self, problem, observations):
        if self.risk_subspace_alignment is None:
            return None
        xs = list(observations)
        psi = np.vstack([self.descriptor(problem, x) for x in xs])
        observed = np.vstack([
            np.mean(np.asarray(observations[x], dtype=float), axis=0)
            for x in xs
        ])
        tau = float(getattr(problem, "tau", 0.0))
        alpha = float(getattr(problem, "alpha", 0.05))
        sigma = max(float(getattr(problem, "sigma_level", 0.0)), 1e-8)
        chance_shift = norm.ppf(1.0 - alpha) * sigma
        scale = max(abs(tau), sigma, 1e-6)
        margins = (observed[:, 1] + chance_shift - tau) / scale
        return self.risk_subspace_alignment.fit_target_adapter(psi, margins)

    def _fit_spectral_frequency_bank(self, batches):
        active_dim = int(self.stage1_spectral_basis.feature_dim)
        entries = []
        basis_by_cutoff = {}
        for cutoff in self.spectral_frequency_cutoffs:
            if (
                int(cutoff) == int(self.spectral_low_frequency_components)
                and int(self.stage1_spectral_basis.feature_dim) == active_dim
            ):
                basis_by_cutoff[int(cutoff)] = self.stage1_spectral_basis
            else:
                basis_by_cutoff[int(cutoff)] = TransferableSpectralBasis(
                    active_dim=active_dim,
                    max_library_size=self.spectral_max_library_size,
                    low_frequency_components=int(cutoff),
                    n_neighbors=self.spectral_graph_neighbors,
                    relevance_floor=self.spectral_relevance_floor,
                    ridge=self.ridge,
                    orthogonalization=self.spectral_orthogonalization,
                    use_low_frequency_score=self.spectral_low_frequency_prior,
                ).fit(batches)
            for ridge in self.spectral_frequency_ridges:
                is_baseline = bool(
                    int(cutoff) == int(self.spectral_low_frequency_components)
                    and np.isclose(float(ridge), 1.0)
                    and int(self.stage1_spectral_basis.feature_dim) == active_dim
                )
                basis = basis_by_cutoff[int(cutoff)]
                diag = basis.diagnostics()
                cv_loss = np.asarray(
                    diag.get("selected_pilot_cv_loss", []), dtype=float)
                source_score = float(
                    np.mean(cv_loss)) if len(cv_loss) else float("inf")
                entries.append({
                    "cutoff": int(cutoff),
                    "ridge": float(ridge),
                    "basis": basis,
                    "source_score": source_score,
                    "is_stage1_baseline": is_baseline,
                    "base_variant": "source_spectral",
                })
        finite_scores = np.asarray([
            entry["source_score"] for entry in entries
        ], dtype=float)
        finite = np.isfinite(finite_scores)
        reference = float(np.min(finite_scores[finite])) if np.any(finite) else 0.0
        logits = np.where(
            finite,
            -(finite_scores - reference) / self.spectral_frequency_temperature,
            -50.0,
        )
        logits -= float(np.max(logits))
        weights = np.exp(np.clip(logits, -50.0, 0.0))
        weights /= max(float(np.sum(weights)), 1e-12)
        for index, (entry, weight) in enumerate(zip(entries, weights)):
            entry["index"] = int(index)
            entry["variant"] = f"frequency_band_{index:02d}"
            entry["source_weight"] = float(weight)
        baseline_count = sum(
            bool(entry["is_stage1_baseline"]) for entry in entries
        )
        if baseline_count != 1:
            raise RuntimeError(
                "frequency bank must contain exactly one Stage-1 fallback"
            )
        self.spectral_frequency_bank = entries
        self.spectral_frequency_diagnostics = {
            "status": "fit",
            "method": "source_cv_finite_band_hyperprior",
            "target_data_used": False,
            "active_dim": int(active_dim),
            "source_penalty": float(self.spectral_frequency_source_penalty),
            "temperature": float(self.spectral_frequency_temperature),
            "entries": [
                {
                    "index": entry["index"],
                    "variant": entry["variant"],
                    "cutoff": entry["cutoff"],
                    "ridge": entry["ridge"],
                    "source_score": entry["source_score"],
                    "source_weight": entry["source_weight"],
                    "is_stage1_baseline": entry["is_stage1_baseline"],
                    "fingerprint": entry["basis"].fingerprint(),
                    "selected_names": entry["basis"].diagnostics().get(
                        "selected_names", []),
                }
                for entry in entries
            ],
        }

    def spectral_frequency_features(self, problem, x, index):
        entry = self.spectral_frequency_bank[int(index)]
        psi = self.risk_coordinate(problem, x)
        return entry["basis"].transform(psi)

    def _fit_spectral_additive_bank(self, batches):
        self.spectral_additive_bank = TransferableAdditiveGroupBank(
            max_groups=self.spectral_additive_max_groups,
            max_library_size=self.spectral_max_library_size,
            low_frequency_components=self.spectral_low_frequency_components,
            n_neighbors=self.spectral_graph_neighbors,
            relevance_floor=self.spectral_relevance_floor,
            temperature=self.spectral_additive_temperature,
            ridge=self.ridge,
            base_basis=self.stage1_spectral_basis,
            strong_heredity=True,
            max_interactions=min(2, self.spectral_additive_target_max_groups),
        ).fit(batches)
        self.spectral_additive_diagnostics = (
            self.spectral_additive_bank.diagnostics())

    def spectral_additive_features(self, problem, x, indices):
        if self.spectral_additive_bank is None:
            raise RuntimeError("source additive group bank is not enabled")
        psi = self.risk_coordinate(problem, x)
        return self.spectral_additive_bank.transform_groups(psi, indices)

    def spectral_features_from_descriptor(self, descriptor):
        if self.spectral_basis is None:
            raise RuntimeError("source-invariant spectral basis is not enabled")
        psi = self.risk_coordinate_from_descriptor(descriptor)
        residual = self.spectral_basis.transform(psi)
        if (
            self.spectral_adaptive_sparsity
            and self.stage1_spectral_basis is not self.spectral_basis
        ):
            prefix = self.stage1_spectral_basis.transform(psi)
            return np.concatenate([prefix, residual])
        return residual

    def spectral_features(self, problem, x):
        return self.spectral_features_from_descriptor(self.descriptor(problem, x))

    def stage1_spectral_features_from_descriptor(self, descriptor):
        if self.stage1_spectral_basis is None:
            raise RuntimeError("Stage-1 spectral baseline is not enabled")
        if self.stage1_spectral_basis is self.spectral_basis:
            return self.spectral_features_from_descriptor(descriptor)
        psi = self.risk_coordinate_from_descriptor(descriptor)
        return self.stage1_spectral_basis.transform(psi)

    def stage1_spectral_features(self, problem, x):
        if self.stage1_spectral_basis is self.spectral_basis:
            return self.spectral_features(problem, x)
        return self.stage1_spectral_features_from_descriptor(
            self.descriptor(problem, x))

    def _fit_spectral_coefficient_prior(self, records):
        """Estimate source-only coefficient inclusion evidence.

        Coefficients are fit separately within each source domain after
        standardizing both the frozen spectral features and the response.  The
        resulting z evidence is dimensionless, so it can transfer across
        domains without importing target-specific objective or constraint
        scales.
        """

        by_domain = {}
        for rec in records:
            by_domain.setdefault(rec.domain, []).append(rec)
        domain_stats = {0: [], 1: []}
        for domain, domain_records in sorted(by_domain.items()):
            features = np.vstack([
                self.spectral_features_from_descriptor(rec.descriptor)
                for rec in domain_records
            ])
            feature_mean = np.mean(features, axis=0)
            feature_scale = np.std(features, axis=0)
            feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
            X = (features - feature_mean) / feature_scale
            design = np.column_stack([np.ones(len(X), dtype=float), X])
            weights = np.asarray([
                max(float(rec.sample_weight), 1e-8)
                * self._boundary_sample_weight(rec)
                for rec in domain_records
            ], dtype=float)
            root_w = np.sqrt(weights / max(float(np.mean(weights)), 1e-8))
            weighted_design = design * root_w[:, None]
            penalty = self.ridge * np.eye(design.shape[1], dtype=float)
            penalty[0, 0] = 0.0
            lhs = weighted_design.T @ weighted_design + penalty
            try:
                lhs_inv = np.linalg.pinv(lhs)
            except np.linalg.LinAlgError:
                lhs_inv = np.eye(lhs.shape[0], dtype=float)
            for output_index in (0, 1):
                if output_index == 0:
                    target = np.asarray([
                        float(rec.y[0]) for rec in domain_records
                    ], dtype=float)
                else:
                    target = np.asarray([
                        self._source_margin(rec) / self._source_margin_scale(rec)
                        for rec in domain_records
                    ], dtype=float)
                target_scale = max(float(np.std(target)), 1e-8)
                target_z = (target - float(np.mean(target))) / target_scale
                weighted_target = target_z * root_w
                beta = lhs_inv @ weighted_design.T @ weighted_target
                residual = target_z - design @ beta
                dof = max(len(target_z) - design.shape[1], 1)
                sigma2 = max(float(np.sum(weights * residual ** 2) / dof), 1e-10)
                se = np.sqrt(np.maximum(sigma2 * np.diag(lhs_inv), 1e-12))
                z_score = np.abs(beta[1:]) / np.maximum(se[1:], 1e-8)
                domain_stats[output_index].append({
                    "domain": str(domain),
                    "coefficient": np.asarray(beta[1:], dtype=float),
                    "z_score": np.asarray(z_score, dtype=float),
                })

        self.spectral_coefficient_prior = {
            output_index: self._aggregate_spectral_coefficient_prior(rows)
            for output_index, rows in domain_stats.items()
            if rows
        }
        if self.spectral_adaptive_sparsity:
            self._calibrate_adaptive_sparsity(records, domain_stats[1])

    def _aggregate_spectral_coefficient_prior(self, rows):
        coefficients = np.vstack([row["coefficient"] for row in rows])
        z_scores = np.vstack([row["z_score"] for row in rows])
        evidence = 1.0 - np.exp(-0.5 * z_scores ** 2)
        nonzero = np.abs(coefficients) > 1e-10
        positive = np.mean(coefficients > 0.0, axis=0)
        negative = np.mean(coefficients < 0.0, axis=0)
        sign_consistency = np.maximum(positive, negative)
        sign_consistency = np.where(
            np.any(nonzero, axis=0), sign_consistency, 0.0)
        adjusted_evidence = evidence * (0.5 + 0.5 * sign_consistency)
        pip = np.clip(np.mean(adjusted_evidence, axis=0), 0.0, 1.0)
        beta_alpha = 1.0 + np.sum(adjusted_evidence, axis=0)
        beta_beta = 1.0 + len(rows) - np.sum(adjusted_evidence, axis=0)
        prior_pip = np.clip(
            beta_alpha / np.maximum(beta_alpha + beta_beta, 1e-12),
            self.spectral_adaptive_min_pip,
            self.spectral_adaptive_max_pip,
        )
        median_abs = np.median(np.abs(coefficients), axis=0)
        positive_scale = median_abs[median_abs > 1e-8]
        reference_scale = (
            float(np.median(positive_scale)) if len(positive_scale) else 1.0
        )
        slab_scale = np.clip(
            median_abs / max(reference_scale, 1e-8),
            0.10,
            5.0,
        )
        weight = self.spectral_shrinkage_floor + (
            1.0 - self.spectral_shrinkage_floor
        ) * pip ** self.spectral_shrinkage_strength
        max_weight = max(float(np.max(weight)), self.spectral_shrinkage_floor)
        weight = np.clip(
            weight / max_weight,
            self.spectral_shrinkage_floor,
            1.0,
        )
        return {
            "pip": pip,
            "prior_pip": prior_pip,
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "weight": np.asarray(weight, dtype=float),
            "slab_scale": np.asarray(slab_scale, dtype=float),
            "sign_consistency": np.asarray(sign_consistency, dtype=float),
            "median_abs_coefficient": median_abs,
            "median_z_score": np.median(z_scores, axis=0),
            "source_domains": [row["domain"] for row in rows],
            "normalization": "max_one",
        }

    def _calibrate_adaptive_sparsity(self, records, constraint_domain_stats):
        """Nested source-LODO calibration; the held-out target is never read."""

        base_domain = lambda name: str(name).split("#aug", 1)[0]
        domains = sorted({
            base_domain(row["domain"]) for row in constraint_domain_stats
        })
        by_domain = {}
        for rec in records:
            by_domain.setdefault(base_domain(rec.domain), []).append(rec)
        if len(domains) < 2:
            self.spectral_adaptive_calibration = {
                "status": "insufficient_source_domains",
                "selected_spike_ratio": float(
                    self.spectral_adaptive_spike_ratio),
            }
            return
        base = float(np.clip(self.spectral_adaptive_spike_ratio, 0.01, 0.30))
        candidates = sorted({
            float(np.clip(0.5 * base, 0.01, 0.30)),
            base,
            float(np.clip(2.0 * base, 0.01, 0.30)),
        })
        candidate_scores = {}
        fold_rows = []
        for spike_ratio in candidates:
            scores = []
            for heldout in domains:
                source_rows = [
                    row for row in constraint_domain_stats
                    if base_domain(row["domain"]) != heldout
                ]
                target_records = by_domain.get(heldout, [])
                if not source_rows or len(target_records) < 12:
                    continue
                hyper = self._aggregate_spectral_coefficient_prior(source_rows)
                features = np.vstack([
                    self.spectral_features_from_descriptor(rec.descriptor)
                    for rec in target_records
                ])
                target = np.asarray([
                    self._source_margin(rec) / self._source_margin_scale(rec)
                    for rec in target_records
                ])
                seed = sum(ord(ch) for ch in heldout) + self.seed
                order = np.random.default_rng(seed).permutation(len(target_records))
                pilot_count = min(10, max(6, len(order) // 3))
                train = order[:pilot_count]
                test = order[pilot_count:]
                if len(test) < 2:
                    continue
                noise = max(0.05 * float(np.var(target[train])), 1e-4)
                posterior = AdaptiveSpikeSlabPosterior(
                    hyper["prior_pip"],
                    hyper["slab_scale"],
                    min_pip=self.spectral_adaptive_min_pip,
                    max_pip=self.spectral_adaptive_max_pip,
                    spike_ratio=spike_ratio,
                    damping=self.spectral_adaptive_damping,
                    max_iter=self.spectral_adaptive_max_iter,
                    tolerance=self.spectral_adaptive_tolerance,
                    residual_floor_scale=(
                        self.spectral_adaptive_residual_floor_scale),
                    multiplicity_correction=(
                        self.spectral_adaptive_multiplicity_correction),
                    max_effective_fraction=(
                        self.spectral_adaptive_max_effective_fraction),
                    always_active_count=self.spectral_always_active_count,
                    allowed_mask=np.concatenate([
                        np.ones(
                            self.spectral_always_active_count,
                            dtype=bool,
                        ),
                        np.asarray(hyper["prior_pip"], dtype=float)[
                            self.spectral_always_active_count:
                        ] >= 0.5,
                    ]),
                ).fit(
                    features[train],
                    target[train],
                    np.full(len(train), noise),
                    [target_records[index].x for index in train],
                    deviation_variance=max(0.05 * noise, 1e-6),
                )
                prediction = np.asarray(
                    posterior.predict_parametric_mean(features[test]))
                truth = target[test]
                scale = max(float(np.var(truth)), 1e-8)
                nmse = float(np.mean((truth - prediction) ** 2) / scale)
                infeasible = truth > 0.0
                false_feasible = (
                    float(np.mean(prediction[infeasible] <= 0.0))
                    if np.any(infeasible) else 0.0
                )
                feasible = ~infeasible
                false_infeasible = (
                    float(np.mean(prediction[feasible] > 0.0))
                    if np.any(feasible) else 0.0
                )
                score = nmse + 3.0 * false_feasible + 0.25 * false_infeasible
                scores.append(float(score))
                fold_rows.append({
                    "heldout_source": heldout,
                    "spike_ratio": float(spike_ratio),
                    "score": float(score),
                    "false_feasible_rate": float(false_feasible),
                })
            candidate_scores[str(spike_ratio)] = (
                float(np.mean(scores)) if scores else float("inf"))
        selected = min(
            candidates,
            key=lambda value: (candidate_scores[str(value)], abs(value - base)),
        )
        self.spectral_adaptive_spike_ratio = float(selected)
        self.spectral_adaptive_calibration = {
            "status": "fit",
            "method": "nested_source_lodo_decision_loss",
            "candidate_scores": candidate_scores,
            "selected_spike_ratio": float(selected),
            "configured_spike_ratio": float(base),
            "folds": fold_rows,
            "target_data_used": False,
        }

    def spectral_shrinkage_weights(self, output_index=1):
        if not self.spectral_coefficient_shrinkage:
            return None
        prior = self.spectral_coefficient_prior.get(int(output_index))
        if prior is None:
            return None
        return np.asarray(prior["weight"], dtype=float).copy()

    def coordinate_basis_features(self, problem, x):
        desc = self._scaled_descriptor(self.descriptor(problem, x))
        psi = self.risk_coordinate(problem, x)
        exposure = self.risk_exposure(problem, x)
        cumulative = cumulative_feature_vector(exposure)
        return np.concatenate([desc, psi, psi ** 2, cumulative[1:]])

    def task_bias_features_from_descriptor(self, descriptor, *, domain=None):
        """Low-rank signed-calibration features in canonical ``psi=(A,N)``.

        Shared exposures live on a simplex, so an intercept plus every entry
        of ``N`` is rank deficient.  Helmert contrasts retain all simplex
        information while giving the source-only calibration fit a stable,
        orthogonal coordinate system.
        """
        exposure = self.cumulative_risk_exposure_from_descriptor(
            descriptor, domain=domain)
        shared = np.asarray(exposure.N, dtype=float).reshape(-1)
        if len(shared) <= 1:
            shared_contrasts = np.empty(0, dtype=float)
        else:
            contrast = np.zeros((len(shared), len(shared) - 1), dtype=float)
            for index in range(len(shared) - 1):
                scale = np.sqrt((index + 1) * (index + 2))
                contrast[:index + 1, index] = 1.0 / scale
                contrast[index + 1, index] = -(index + 1) / scale
            shared_contrasts = shared @ contrast
        return np.concatenate([
            np.ones(1, dtype=float),
            np.asarray(exposure.A, dtype=float),
            shared_contrasts,
        ])

    def task_bias_features(self, problem, x):
        return self.task_bias_features_from_descriptor(
            self.descriptor(problem, x), domain=None)

    def task_bias_feature_names(self):
        return (
            ["bias_intercept"]
            + [f"bias_A{index}" for index in range(self.local_dim)]
            + [
                f"bias_N_contrast{index}"
                for index in range(max(self.shared_dim - 1, 0))
            ]
        )

    def _fit_task_bias_profiles(
        self,
        mean_rows,
        domains,
        mean_design_by_domain,
        mean_targets_by_domain,
        bias_features_by_domain,
        bias_weights_by_domain=None,
    ):
        """Fit source-LODO signed residual functions on canonical risk features."""
        if not mean_rows or not domains:
            self.task_bias_profile_diagnostics_ = {
                "status": "fallback_null",
                "target_data_used": False,
            }
            self.task_adaptive_bias_prior_["status"] = "fallback_null"
            return
        stack = np.vstack(mean_rows)
        residual_by_domain = {}
        residual_scale_by_domain = {}
        standardized_residual_by_domain = {}
        fitted_profiles = []
        fitted_names = []
        fitted_domains = []
        all_features = np.vstack([
            np.asarray(bias_features_by_domain[str(domain)], dtype=float)
            for domain in domains
        ])
        feature_scale = np.sqrt(np.mean(all_features ** 2, axis=0))
        feature_scale = np.maximum(feature_scale, 1e-3)
        for heldout_index, heldout_domain in enumerate(domains):
            lodo_mean = (
                np.mean(stack, axis=0)
                if len(stack) <= 1
                else np.mean(np.delete(
                    stack, heldout_index, axis=0), axis=0)
            )
            residual = (
                mean_targets_by_domain[heldout_domain]
                - mean_design_by_domain[heldout_domain] @ lodo_mean
            )
            Phi = np.asarray(
                bias_features_by_domain[heldout_domain], dtype=float)
            residual_scale = max(
                float(np.sqrt(np.mean(residual ** 2))), 1e-8)
            standardized_residual = residual / residual_scale
            scaled_phi = Phi / feature_scale[None, :]
            penalty = (
                max(float(self.ridge), 1e-2)
                * max(len(Phi), 1)
                * np.eye(Phi.shape[1], dtype=float)
            )
            try:
                scaled_beta = np.linalg.solve(
                    scaled_phi.T @ scaled_phi + penalty,
                    scaled_phi.T @ standardized_residual,
                )
            except np.linalg.LinAlgError:
                scaled_beta = np.linalg.lstsq(
                    scaled_phi.T @ scaled_phi + penalty,
                    scaled_phi.T @ standardized_residual,
                    rcond=None,
                )[0]
            beta = scaled_beta / feature_scale
            residual_by_domain[str(heldout_domain)] = np.asarray(
                residual, dtype=float)
            residual_scale_by_domain[str(heldout_domain)] = float(
                residual_scale)
            standardized_residual_by_domain[str(heldout_domain)] = np.asarray(
                standardized_residual, dtype=float)
            fitted_profiles.append(np.asarray(beta, dtype=float))
            fitted_names.append(f"source_bias:{heldout_domain}")
            fitted_domains.append(str(heldout_domain))

        feature_dim = len(fitted_profiles[0])
        profiles = np.vstack([
            np.zeros(feature_dim, dtype=float),
            np.vstack(fitted_profiles),
        ])
        names = ["null_bias_profile"] + fitted_names
        profile_domains = [None] + fitted_domains
        scores = []
        for beta, source_domain in zip(profiles, profile_domains):
            fold_scores = []
            for domain in domains:
                if source_domain is not None and str(domain) == source_domain:
                    continue
                residual = standardized_residual_by_domain[str(domain)]
                prediction = (
                    bias_features_by_domain[str(domain)] @ beta)
                fold_scores.append(float(np.mean(
                    -0.5 * (residual - prediction) ** 2
                )))
            scores.append(float(np.mean(fold_scores)) if fold_scores else 0.0)
        log_score = np.asarray(scores, dtype=float)
        log_score -= float(np.max(log_score))
        likelihood = np.exp(np.clip(log_score, -50.0, 0.0))
        likelihood /= max(float(np.sum(likelihood)), 1e-12)
        prior = 0.9 * likelihood + 0.1 / len(likelihood)
        prior /= float(np.sum(prior))
        self.task_bias_profiles_ = profiles
        self.task_bias_profile_names_ = names
        self.task_bias_profile_prior_ = prior
        self.task_bias_profile_diagnostics_ = {
            "status": "fit_source_lodo_profiles",
            "profile_names": list(names),
            "feature_names": self.task_bias_feature_names(),
            "profile_prior_weights": prior.tolist(),
            "profile_log_score": log_score.tolist(),
            "residual_scale_by_domain": copy.deepcopy(
                residual_scale_by_domain),
            "feature_scale": feature_scale.tolist(),
            "profile_output_units": "predictive_standard_deviations",
            "shared_coordinate": "helmert_simplex_contrast",
            "n_source_domains": int(len(domains)),
            "target_data_used": False,
            "target_oracle_used": False,
        }
        self._fit_adaptive_task_bias_prior(
            residual_by_domain,
            bias_features_by_domain,
            bias_weights_by_domain or {},
            feature_scale,
        )

    def _fit_adaptive_task_bias_prior(
        self,
        residual_by_domain,
        bias_features_by_domain,
        bias_weights_by_domain,
        feature_scale,
    ):
        """Fit a source-only Gaussian prior for target adaptive calibration.

        The fitted coefficients predict standardized source LODO residuals.
        Boundary weights affect only this V4 prior; the V3 discrete profile
        dictionary remains unchanged for a reproducible ablation.
        """
        domains = sorted(residual_by_domain)
        if not domains:
            self.task_adaptive_bias_prior_["status"] = "fallback_null"
            return
        feature_scale = np.maximum(
            np.asarray(feature_scale, dtype=float), 1e-3)
        scaled_profiles = []
        residual_scales = {}
        standardized_by_domain = {}
        normalized_weights_by_domain = {}
        for domain in domains:
            residual = np.asarray(
                residual_by_domain[domain], dtype=float).reshape(-1)
            Phi = np.asarray(
                bias_features_by_domain[domain], dtype=float)
            weights = np.asarray(
                bias_weights_by_domain.get(
                    domain, np.ones(len(residual), dtype=float)),
                dtype=float,
            ).reshape(-1)
            if len(weights) != len(residual):
                raise ValueError(
                    "adaptive bias weights must match source residuals")
            weights = np.clip(weights, 1e-8, 1e8)
            weights /= max(float(np.mean(weights)), 1e-12)
            scale = max(float(np.sqrt(
                np.sum(weights * residual ** 2)
                / max(float(np.sum(weights)), 1e-12)
            )), 1e-8)
            target = residual / scale
            scaled_phi = Phi / feature_scale[None, :]
            sqrt_weight = np.sqrt(weights)
            design = scaled_phi * sqrt_weight[:, None]
            response = target * sqrt_weight
            ridge = (
                max(float(self.ridge), 0.05)
                * max(float(np.sum(weights)), 1.0)
            )
            system = design.T @ design + ridge * np.eye(
                design.shape[1], dtype=float)
            try:
                scaled_beta = np.linalg.solve(
                    system, design.T @ response)
            except np.linalg.LinAlgError:
                scaled_beta = np.linalg.lstsq(
                    system, design.T @ response, rcond=None)[0]
            scaled_profiles.append(np.asarray(scaled_beta, dtype=float))
            residual_scales[domain] = float(scale)
            standardized_by_domain[domain] = target
            normalized_weights_by_domain[domain] = weights

        scaled_profiles = np.vstack(scaled_profiles)
        candidates = np.vstack([
            np.zeros(scaled_profiles.shape[1], dtype=float),
            scaled_profiles,
        ])
        candidate_names = ["null_bias_profile"] + [
            f"source_bias:{domain}" for domain in domains
        ]
        candidate_domains = [None] + domains
        scores = []
        for beta, source_domain in zip(candidates, candidate_domains):
            fold_scores = []
            for domain in domains:
                if source_domain is not None and domain == source_domain:
                    continue
                Phi = np.asarray(
                    bias_features_by_domain[domain], dtype=float)
                prediction = (Phi / feature_scale[None, :]) @ beta
                target = standardized_by_domain[domain]
                weights = normalized_weights_by_domain[domain]
                fold_scores.append(float(
                    -0.5 * np.sum(weights * (target - prediction) ** 2)
                    / max(float(np.sum(weights)), 1e-12)
                ))
            scores.append(float(np.mean(fold_scores)) if fold_scores else 0.0)
        log_score = np.asarray(scores, dtype=float)
        log_score -= float(np.max(log_score))
        likelihood = np.exp(np.clip(log_score, -50.0, 0.0))
        likelihood /= max(float(np.sum(likelihood)), 1e-12)
        mixture = 0.9 * likelihood + 0.1 / len(likelihood)
        mixture /= float(np.sum(mixture))

        raw_candidates = candidates / feature_scale[None, :]
        mean = mixture @ raw_candidates
        centered = raw_candidates - mean[None, :]
        covariance = 0.25 * np.eye(len(mean), dtype=float)
        covariance += np.einsum(
            "i,ij,ik->jk", mixture, centered, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (covariance + covariance.T))
        eigenvalues = np.clip(eigenvalues, 0.25, 4.0)
        covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(covariance)
        precision = 0.5 * (precision + precision.T)
        self.task_adaptive_bias_prior_ = {
            "status": "fit_boundary_weighted_gaussian",
            "mean": np.asarray(mean, dtype=float),
            "precision": np.asarray(precision, dtype=float),
            "feature_names": self.task_bias_feature_names(),
            "source_profile_names": candidate_names,
            "source_profile_weights": mixture.tolist(),
            "source_profile_log_score": log_score.tolist(),
            "source_profile_coefficients_scaled": candidates.tolist(),
            "feature_scale": feature_scale.tolist(),
            "residual_scale_by_domain": residual_scales,
            "prior_covariance_eigenvalue_floor": 0.25,
            "prior_covariance_eigenvalue_cap": 4.0,
            "boundary_weighted": True,
            "n_source_domains": int(len(domains)),
            "target_data_used": False,
            "target_oracle_used": False,
        }

    def _fit_hvd_beta_priors(self, records):
        by_domain = {}
        for rec in records:
            by_domain.setdefault(rec.domain, []).append(rec)
        beta_by_output = {0: [], 1: []}
        beta_domains_by_output = {0: [], 1: []}
        variance_targets_by_output_domain = {0: {}, 1: {}}
        mean_by_output = {0: [], 1: []}
        mean_domains_by_output = {0: [], 1: []}
        mean_design_by_domain = {}
        mean_targets_by_output_domain = {0: {}, 1: {}}
        bias_features_by_domain = {}
        bias_weights_by_domain = {}
        sigma_by_output = {0: [], 1: []}
        cumulative_features_by_domain = {}
        for domain, domain_records in by_domain.items():
            X_desc = np.vstack([
                np.concatenate([[1.0], self._scaled_descriptor(rec.descriptor)])
                for rec in domain_records
            ])
            X_meta = np.vstack([
                np.concatenate([[
                    1.0],
                    self._mean_prior_features_from_descriptor(rec.descriptor),
                ])
                for rec in domain_records
            ])
            mean_design_by_domain[str(domain)] = X_meta
            bias_features_by_domain[str(domain)] = np.vstack([
                self.task_bias_features_from_descriptor(
                    rec.descriptor, domain=rec.domain)
                for rec in domain_records
            ])
            F = np.vstack([
                cumulative_feature_vector(
                    self._cumulative_risk_exposure_from_record(
                        rec,
                        aligned=True,
                    )
                )
                for rec in domain_records
            ])
            cumulative_features_by_domain[str(domain)] = F
            weights = np.asarray([
                self._boundary_sample_weight(rec) for rec in domain_records
            ], dtype=float)
            weights = np.clip(weights, 1e-4, 1e4)
            bias_weights_by_domain[str(domain)] = weights.copy()
            sqrt_w = np.sqrt(weights)
            reg_mean = self.ridge * np.eye(X_desc.shape[1], dtype=float)
            reg_mean[0, 0] = 0.0
            for out_idx in (0, 1):
                y = np.asarray([float(rec.y[out_idx]) for rec in domain_records], dtype=float)
                Xw = X_desc * sqrt_w[:, None]
                yw = y * sqrt_w
                try:
                    beta_mean = np.linalg.solve(
                        Xw.T @ Xw + reg_mean,
                        Xw.T @ yw,
                    )
                except np.linalg.LinAlgError:
                    beta_mean = np.linalg.lstsq(
                        Xw.T @ Xw + reg_mean,
                        Xw.T @ yw,
                        rcond=None,
                    )[0]
                reg_meta = self.ridge * np.eye(X_meta.shape[1], dtype=float)
                reg_meta[0, 0] = 0.0
                Xmw = X_meta * sqrt_w[:, None]
                try:
                    beta_meta = np.linalg.solve(
                        Xmw.T @ Xmw + reg_meta,
                        Xmw.T @ yw,
                    )
                except np.linalg.LinAlgError:
                    beta_meta = np.linalg.lstsq(
                        Xmw.T @ Xmw + reg_meta,
                        Xmw.T @ yw,
                        rcond=None,
                    )[0]
                resid_meta = y - X_meta @ beta_meta
                mean_by_output[out_idx].append(beta_meta)
                mean_domains_by_output[out_idx].append(str(domain))
                mean_targets_by_output_domain[out_idx][str(domain)] = y
                sigma_by_output[out_idx].append(float(
                    np.sqrt(np.mean(resid_meta ** 2)) if len(resid_meta) else 0.0
                ))
                source_variance_labels = None
                if out_idx == 1 and all(
                    rec.constraint_sigma is not None for rec in domain_records
                ):
                    source_variance_labels = np.asarray([
                        float(rec.constraint_sigma) ** 2
                        for rec in domain_records
                    ], dtype=float)
                if source_variance_labels is not None:
                    resid2 = np.maximum(source_variance_labels, 1e-10)
                else:
                    resid2 = np.maximum(
                        (y - X_desc @ beta_mean) ** 2,
                        1e-10,
                    )
                if (
                    source_variance_labels is None
                    and self.hvd_noise_floor_scale > 0.0
                ):
                    sigmas = []
                    for rec in domain_records:
                        if out_idx == 1 and rec.constraint_sigma is not None:
                            sigmas.append(float(rec.constraint_sigma))
                        else:
                            sigmas.append(float(rec.sigma_level))
                    noise_floor = (
                        float(self.hvd_noise_floor_scale)
                        * np.asarray(sigmas, dtype=float)
                    ) ** 2
                    resid2 = np.maximum(resid2, noise_floor)
                resid_scale = float(np.median(resid2)) if len(resid2) else 0.0
                if resid_scale <= 1e-12:
                    resid_scale = float(np.mean(resid2) + 1e-10)
                var_weights = weights * (
                    1.0 + self.variance_weight * np.minimum(resid2 / resid_scale, 10.0)
                )
                sqrt_vw = np.sqrt(np.clip(var_weights, 1e-4, 1e4))
                Fw = F * sqrt_vw[:, None]
                rw = resid2 * sqrt_vw
                reg_var = self.ridge * np.eye(F.shape[1], dtype=float)
                try:
                    beta = np.linalg.solve(Fw.T @ Fw + reg_var, Fw.T @ rw)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(Fw.T @ Fw + reg_var, Fw.T @ rw, rcond=None)[0]
                beta = _project_psd_features(beta, self.local_dim, self.shared_dim)
                beta_by_output[out_idx].append(beta)
                beta_domains_by_output[out_idx].append(str(domain))
                variance_targets_by_output_domain[out_idx][str(domain)] = np.asarray(
                    resid2,
                    dtype=float,
                )
                self.training_diagnostics.setdefault(
                    "hvd_variance_label_source", {})[str(out_idx)] = (
                        "source_fresh_seed_variance"
                        if source_variance_labels is not None
                        else "source_residual_square_fallback"
                    )
        self._fit_task_bias_profiles(
            mean_by_output.get(1, []),
            mean_domains_by_output.get(1, []),
            mean_design_by_domain,
            mean_targets_by_output_domain.get(1, {}),
            bias_features_by_domain,
            bias_weights_by_domain,
        )
        for out_idx, rows in beta_by_output.items():
            if rows:
                stack = np.vstack(rows)
                mean_beta = np.mean(stack, axis=0)
                self.beta_prior[out_idx] = mean_beta
                self.beta_prior_components[out_idx] = stack.copy()
                self.beta_prior_component_domains[out_idx] = list(
                    beta_domains_by_output[out_idx])
                reference_predictions = np.concatenate([
                    np.maximum(F @ mean_beta, 1e-12)
                    for F in cumulative_features_by_domain.values()
                ])
                self.beta_prior_reference_mean[out_idx] = float(max(
                    float(np.mean(reference_predictions)),
                    1e-12,
                ))
                signal = float(np.mean(mean_beta ** 2))
                dispersion = float(np.mean((stack - mean_beta) ** 2))
                learned_precision = signal / max(
                    dispersion + 0.1 * signal,
                    1e-12,
                )
                self.beta_prior_precision[out_idx] = float(np.clip(
                    learned_precision,
                    0.05,
                    5.0,
                ))
                lodo_ratios = []
                domains = beta_domains_by_output[out_idx]
                for heldout_index, heldout_domain in enumerate(domains):
                    if len(stack) <= 1:
                        lodo_beta = mean_beta
                    else:
                        lodo_beta = np.mean(np.delete(
                            stack,
                            heldout_index,
                            axis=0,
                        ), axis=0)
                    F_heldout = cumulative_features_by_domain[heldout_domain]
                    target_variance = variance_targets_by_output_domain[
                        out_idx][heldout_domain]
                    shape = np.maximum(F_heldout @ lodo_beta, 1e-12)
                    shape /= max(float(np.mean(shape)), 1e-12)
                    amplitude = max(float(np.mean(target_variance)), 1e-12)
                    prediction = np.maximum(amplitude * shape, 1e-12)
                    ratio = target_variance / prediction
                    lodo_ratios.extend(
                        float(value) for value in ratio if np.isfinite(value)
                    )
                if lodo_ratios:
                    ordered = np.sort(np.asarray(lodo_ratios, dtype=float))
                    conformal_index = min(
                        len(ordered) - 1,
                        max(0, int(np.ceil(0.95 * (len(ordered) + 1))) - 1),
                    )
                    self.beta_prior_upper_scale[out_idx] = float(max(
                        1.0,
                        ordered[conformal_index],
                    ))
        for out_idx, rows in mean_by_output.items():
            if rows:
                self.mean_prior[out_idx] = np.mean(np.vstack(rows), axis=0)
                sigmas = np.asarray(sigma_by_output.get(out_idx, []), dtype=float)
                self.mean_prior_sigma[out_idx] = float(
                    np.median(sigmas) if len(sigmas) else 0.0)
        self._fit_unaligned_hvd_beta_priors(records)

    def _fit_unaligned_hvd_beta_priors(self, records):
        """Fit the HVD prior in the unaligned source coordinate system.

        The aligned and unaligned task experts use different risk coordinates,
        so sharing one coefficient vector would silently change its meaning.
        This source-only fit mirrors the aligned prior construction while
        keeping a separate coefficient family and LODO calibration constants.
        """
        by_domain = {}
        for rec in records:
            by_domain.setdefault(str(rec.domain), []).append(rec)
        beta_rows = {0: [], 1: []}
        beta_domains = {0: [], 1: []}
        mean_rows = {0: [], 1: []}
        mean_features_by_domain = {}
        mean_targets_by_output_domain = {0: {}, 1: {}}
        feature_by_domain = {}
        target_by_output_domain = {0: {}, 1: {}}
        for domain, domain_records in by_domain.items():
            descriptors = np.vstack([
                np.concatenate([[1.0], self._scaled_descriptor(rec.descriptor)])
                for rec in domain_records
            ])
            mean_features_by_domain[domain] = descriptors
            features = np.vstack([
                cumulative_feature_vector(
                    self._cumulative_risk_exposure_from_record(
                        rec,
                        aligned=False,
                    )
                )
                for rec in domain_records
            ])
            feature_by_domain[domain] = features
            weights = np.clip(np.asarray([
                self._boundary_sample_weight(rec) for rec in domain_records
            ], dtype=float), 1e-4, 1e4)
            sqrt_w = np.sqrt(weights)
            mean_reg = self.ridge * np.eye(descriptors.shape[1], dtype=float)
            mean_reg[0, 0] = 0.0
            for output_index in (0, 1):
                target = np.asarray([
                    float(rec.y[output_index]) for rec in domain_records
                ], dtype=float)
                weighted_x = descriptors * sqrt_w[:, None]
                weighted_y = target * sqrt_w
                try:
                    mean_beta = np.linalg.solve(
                        weighted_x.T @ weighted_x + mean_reg,
                        weighted_x.T @ weighted_y,
                    )
                except np.linalg.LinAlgError:
                    mean_beta = np.linalg.lstsq(
                        weighted_x.T @ weighted_x + mean_reg,
                        weighted_x.T @ weighted_y,
                        rcond=None,
                    )[0]
                mean_rows[output_index].append(mean_beta)
                mean_targets_by_output_domain[output_index][domain] = target
                variance_labels = None
                if output_index == 1 and all(
                    rec.constraint_sigma is not None for rec in domain_records
                ):
                    variance_labels = np.asarray([
                        float(rec.constraint_sigma) ** 2
                        for rec in domain_records
                    ], dtype=float)
                if variance_labels is None:
                    variance_target = np.maximum(
                        (target - descriptors @ mean_beta) ** 2,
                        1e-10,
                    )
                else:
                    variance_target = np.maximum(variance_labels, 1e-10)
                if variance_labels is None and self.hvd_noise_floor_scale > 0.0:
                    sigma = np.asarray([
                        float(rec.sigma_level) for rec in domain_records
                    ], dtype=float)
                    variance_target = np.maximum(
                        variance_target,
                        (float(self.hvd_noise_floor_scale) * sigma) ** 2,
                    )
                target_by_output_domain[output_index][domain] = variance_target
                residual_scale = max(
                    float(np.median(variance_target)),
                    float(np.mean(variance_target) + 1e-10),
                    1e-12,
                )
                variance_weights = weights * (
                    1.0
                    + self.variance_weight
                    * np.minimum(variance_target / residual_scale, 10.0)
                )
                sqrt_variance_weight = np.sqrt(np.clip(
                    variance_weights, 1e-4, 1e4))
                weighted_features = features * sqrt_variance_weight[:, None]
                weighted_target = variance_target * sqrt_variance_weight
                reg = self.ridge * np.eye(features.shape[1], dtype=float)
                try:
                    beta = np.linalg.solve(
                        weighted_features.T @ weighted_features + reg,
                        weighted_features.T @ weighted_target,
                    )
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(
                        weighted_features.T @ weighted_features + reg,
                        weighted_features.T @ weighted_target,
                        rcond=None,
                    )[0]
                beta = _project_psd_features(
                    beta, self.local_dim, self.shared_dim)
                beta_rows[output_index].append(beta)
                beta_domains[output_index].append(domain)

        sensitivity_ratios = []
        sensitivity_signed_residuals = []
        for output_index, rows in beta_rows.items():
            if not rows:
                continue
            stack = np.vstack(rows)
            mean_beta = np.mean(stack, axis=0)
            self.unaligned_beta_prior[output_index] = mean_beta
            self.unaligned_beta_prior_components[output_index] = stack.copy()
            self.unaligned_beta_prior_component_domains[output_index] = list(
                beta_domains[output_index])
            reference_predictions = np.concatenate([
                np.maximum(matrix @ mean_beta, 1e-12)
                for matrix in feature_by_domain.values()
            ])
            self.unaligned_beta_prior_reference_mean[output_index] = float(max(
                float(np.mean(reference_predictions)), 1e-12))
            signal = float(np.mean(mean_beta ** 2))
            dispersion = float(np.mean((stack - mean_beta) ** 2))
            self.unaligned_beta_prior_precision[output_index] = float(np.clip(
                signal / max(dispersion + 0.1 * signal, 1e-12),
                0.05,
                5.0,
            ))
            lodo_ratios = []
            domains = beta_domains[output_index]
            for heldout_index, heldout_domain in enumerate(domains):
                lodo_beta = (
                    mean_beta
                    if len(stack) <= 1
                    else np.mean(np.delete(stack, heldout_index, axis=0), axis=0)
                )
                heldout_features = feature_by_domain[heldout_domain]
                variance_target = target_by_output_domain[
                    output_index][heldout_domain]
                shape = np.maximum(heldout_features @ lodo_beta, 1e-12)
                shape /= max(float(np.mean(shape)), 1e-12)
                amplitude = max(float(np.mean(variance_target)), 1e-12)
                prediction = np.maximum(amplitude * shape, 1e-12)
                ratios = variance_target / prediction
                lodo_ratios.extend(
                    float(value) for value in ratios if np.isfinite(value)
                )
                if output_index == 1 and mean_rows[output_index]:
                    mean_stack = np.vstack(mean_rows[output_index])
                    lodo_mean_beta = (
                        np.mean(mean_stack, axis=0)
                        if len(mean_stack) <= 1
                        else np.mean(np.delete(
                            mean_stack, heldout_index, axis=0), axis=0)
                    )
                    heldout_target = mean_targets_by_output_domain[
                        output_index][heldout_domain]
                    heldout_mean = (
                        mean_features_by_domain[heldout_domain]
                        @ lodo_mean_beta
                    )
                    signed_residual = heldout_target - heldout_mean
                    sensitivity_signed_residuals.extend(
                        float(value)
                        for value in signed_residual
                        if np.isfinite(value)
                    )
            if lodo_ratios:
                ordered = np.sort(np.asarray(lodo_ratios, dtype=float))
                conformal_index = min(
                    len(ordered) - 1,
                    max(0, int(np.ceil(0.95 * (len(ordered) + 1))) - 1),
                )
                self.unaligned_beta_prior_upper_scale[output_index] = float(max(
                    1.0, ordered[conformal_index]))
                if output_index == 1:
                    sensitivity_ratios.extend(lodo_ratios)
        self._fit_task_sensitivity_prior(
            sensitivity_ratios,
            sensitivity_signed_residuals,
        )

    def _fit_task_sensitivity_prior(
        self,
        squared_standardized_residuals,
        signed_standardized_residuals=None,
    ):
        ratios = np.asarray(squared_standardized_residuals, dtype=float)
        ratios = ratios[np.isfinite(ratios) & (ratios >= 0.0)]
        signed_raw = np.asarray(
            [] if signed_standardized_residuals is None
            else signed_standardized_residuals,
            dtype=float,
        )
        signed_raw = signed_raw[np.isfinite(signed_raw)]
        bias_scale = (
            max(float(np.sqrt(np.mean(signed_raw ** 2))), 1e-8)
            if len(signed_raw)
            else 1.0
        )
        signed = np.clip(signed_raw / bias_scale, -6.0, 6.0)
        scale_names = ("stable", "balanced", "sensitive")
        scale_grid = np.asarray([0.5, 1.0, 2.0], dtype=float)
        penalty_grid = np.asarray([2.0, 5.0, 20.0], dtype=float)
        trust_grid = np.asarray([1.0, 0.25, 0.0], dtype=float)
        if len(ratios):
            clipped = np.clip(ratios, 0.0, 100.0)
            scale_log_score = np.asarray([
                float(np.mean(
                    -np.log(scale) - 0.5 * clipped / scale ** 2
                ))
                for scale in scale_grid
            ], dtype=float)
            scale_log_score -= float(np.max(scale_log_score))
            scale_likelihood = np.exp(np.clip(
                scale_log_score, -50.0, 0.0))
            scale_likelihood /= max(float(np.sum(scale_likelihood)), 1e-12)
            scale_prior = 0.9 * scale_likelihood + 0.1 / len(scale_grid)
            scale_prior /= float(np.sum(scale_prior))
        else:
            scale_log_score = np.zeros(len(scale_grid), dtype=float)
            scale_prior = np.ones(len(scale_grid), dtype=float) / len(scale_grid)

        functional_profiles = bool(
            self.task_bias_profile_diagnostics_.get("status")
            == "fit_source_lodo_profiles"
            and len(self.task_bias_profiles_) > 1
        )
        if functional_profiles:
            profile_names = list(self.task_bias_profile_names_)
            profile_prior = np.asarray(
                self.task_bias_profile_prior_, dtype=float)
            profile_prior /= max(float(np.sum(profile_prior)), 1e-12)
            bias_centers = np.asarray([0.0], dtype=float)
        else:
            bias_centers = (
                np.clip(np.quantile(signed, [0.2, 0.5, 0.8]), -3.0, 3.0)
                if len(signed)
                else np.asarray([0.0], dtype=float)
            )
            profile_names = [f"bias{index}" for index in range(
                len(bias_centers))]
            profile_prior = np.ones(
                len(profile_names), dtype=float) / len(profile_names)

        names = []
        scales = []
        biases = []
        bias_coefficients = []
        penalties = []
        empirical_trust = []
        for scale_index, scale_name in enumerate(scale_names):
            for profile_index, profile_name in enumerate(profile_names):
                names.append(f"{scale_name}:{profile_name}")
                scales.append(float(scale_grid[scale_index]))
                biases.append(
                    0.0
                    if functional_profiles
                    else float(bias_centers[profile_index])
                )
                if functional_profiles:
                    bias_coefficients.append(
                        self.task_bias_profiles_[profile_index].tolist())
                penalties.append(float(penalty_grid[scale_index]))
                empirical_trust.append(float(trust_grid[scale_index]))
        scales = np.asarray(scales, dtype=float)
        biases = np.asarray(biases, dtype=float)
        penalties = np.asarray(penalties, dtype=float)
        if functional_profiles:
            weights = np.outer(scale_prior, profile_prior).reshape(-1)
            weights /= float(np.sum(weights))
            log_score = np.log(np.maximum(weights, 1e-300))
            log_score -= float(np.max(log_score))
            status = "fit_functional_bias_scale"
        elif len(signed):
            log_score = np.asarray([
                float(np.mean(
                    -np.log(scale)
                    - 0.5 * ((signed - bias) / scale) ** 2
                ))
                for scale, bias in zip(scales, biases)
            ], dtype=float)
            log_score -= float(np.max(log_score))
            likelihood = np.exp(np.clip(log_score, -50.0, 0.0))
            likelihood /= max(float(np.sum(likelihood)), 1e-12)
            weights = 0.9 * likelihood + 0.1 / len(scales)
            weights /= float(np.sum(weights))
            status = "fit_signed_bias_scale"
        else:
            log_score = np.zeros(len(scales), dtype=float)
            weights = np.outer(scale_prior, profile_prior).reshape(-1)
            weights /= float(np.sum(weights))
            status = "fallback_uniform"
        self.task_sensitivity_prior_ = {
            "status": status,
            "method": (
                "source_lodo_functional_bias_scale_classes"
                if functional_profiles
                else "source_lodo_signed_bias_scale_classes"
            ),
            "class_names": list(names),
            "scales": scales.tolist(),
            "biases": biases.tolist(),
            "bias_coefficients": (
                bias_coefficients if functional_profiles else None),
            "bias_feature_names": (
                self.task_bias_feature_names()
                if functional_profiles else None),
            "decision_penalties": penalties.tolist(),
            "empirical_trust": empirical_trust,
            "prior_weights": weights.tolist(),
            "source_log_score": log_score.tolist(),
            "source_bias_centers": bias_centers.tolist(),
            "source_bias_standardization_scale": float(bias_scale),
            "functional_bias_profiles": bool(functional_profiles),
            "bias_profile_diagnostics": copy.deepcopy(
                self.task_bias_profile_diagnostics_),
            "adaptive_scale_class_names": list(scale_names),
            "adaptive_scale_scales": scale_grid.tolist(),
            "adaptive_scale_decision_penalties": penalty_grid.tolist(),
            "adaptive_scale_empirical_trust": trust_grid.tolist(),
            "adaptive_scale_prior_weights": scale_prior.tolist(),
            "adaptive_bias_prior": {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else copy.deepcopy(value)
                )
                for key, value in self.task_adaptive_bias_prior_.items()
            },
            "n_source_residuals": int(len(signed)),
            "n_source_variance_ratios": int(len(ratios)),
            "target_data_used": False,
            "target_oracle_used": False,
        }

    def _mean_prior_features_from_descriptor(self, descriptor):
        desc = self._scaled_descriptor(descriptor)
        psi = self.risk_coordinate_from_descriptor(descriptor)
        exposure = self.risk_exposure_from_descriptor(descriptor)
        cumulative = cumulative_feature_vector(exposure)
        return np.concatenate([
            desc,
            psi,
            psi ** 2,
            cumulative[1:],
        ])

    def mean_prior_features(self, problem, x):
        return self._mean_prior_features_from_descriptor(self.descriptor(problem, x))

    def source_mean_prior_predict(self, problem, x, output_index=1):
        if not self.component_enabled("mean"):
            return None
        beta = self.mean_prior.get(int(output_index))
        if beta is None:
            return None
        phi = np.concatenate([[1.0], self.mean_prior_features(problem, x)])
        if len(phi) != len(beta):
            return None
        return float(phi @ beta)

    def source_mean_prior_predict_many(self, problem, xs, output_index=1):
        if not self.component_enabled("mean"):
            return None
        beta = self.mean_prior.get(int(output_index))
        if beta is None:
            return None
        Phi = np.vstack([
            np.concatenate([[1.0], self.mean_prior_features(problem, x)])
            for x in xs
        ])
        if Phi.shape[1] != len(beta):
            return None
        return Phi @ beta

    def source_mean_prior_sigma(self, output_index=1):
        if not self.component_enabled("mean"):
            return 0.0
        return float(max(
            self.mean_prior_sigma.get(int(output_index), 0.0) or 0.0,
            1e-8,
        ))

    def _fit_anchor_distribution(self, records):
        scored = []
        for rec in records:
            psi = self.risk_coordinate_from_descriptor(rec.descriptor)
            # Source-only observed feasibility heuristic.  No held-out target
            # truth enters this score.
            margin = self._source_margin(rec)
            scaled_margin = margin / self._source_margin_scale(rec)
            violation = max(scaled_margin, 0.0)
            boundary_distance = abs(scaled_margin)
            feasible = margin <= 0.0
            score = (
                float(rec.y[0])
                + self.feasible_penalty * violation
                + self.boundary_weight * boundary_distance
                - self.feasible_bonus * float(feasible)
            )
            scored.append({
                "score": float(score),
                "objective": float(rec.y[0]),
                "margin": float(margin),
                "scaled_margin": float(scaled_margin),
                "psi": np.asarray(psi, dtype=float),
                "profile": (
                    None
                    if rec.profile is None
                    else np.asarray(rec.profile, dtype=float).reshape(-1)
                ),
                "domain": rec.domain,
                "feasible": bool(feasible),
                "anchor_type": "calibrated_score",
            })
        n_keep = max(1, min(self.anchor_count, len(scored)))
        n_elite = int(np.ceil(n_keep * np.clip(self.elite_fraction, 0.0, 1.0)))
        n_boundary = int(np.ceil(n_keep * np.clip(self.boundary_fraction, 0.0, 1.0)))
        selected = []
        seen = set()

        def add_rows(rows, limit, anchor_type):
            for row in rows:
                if len(selected) >= n_keep or limit <= 0:
                    break
                key = tuple(np.round(row["psi"], 8))
                if key in seen:
                    continue
                seen.add(key)
                item = dict(row)
                item["anchor_type"] = anchor_type
                selected.append(item)
                limit -= 1

        feasible_rows = [row for row in scored if row["feasible"]]
        feasible_rows.sort(key=lambda row: (row["objective"], abs(row["scaled_margin"])))
        add_rows(feasible_rows, n_elite, "source_feasible_elite")

        boundary_rows = sorted(
            scored,
            key=lambda row: (abs(row["scaled_margin"]), max(row["scaled_margin"], 0.0)),
        )
        add_rows(boundary_rows, n_boundary, "source_chance_boundary")

        calibrated_rows = sorted(scored, key=lambda row: row["score"])
        add_rows(calibrated_rows, n_keep, "calibrated_score")

        # Fill leftovers with far-apart psi points to keep the frozen proposal
        # from collapsing into one source-domain basin.
        while len(selected) < n_keep:
            if not selected:
                add_rows(calibrated_rows, 1, "calibrated_score")
                continue
            chosen = np.vstack([row["psi"] for row in selected])
            diverse_rows = []
            for row in scored:
                key = tuple(np.round(row["psi"], 8))
                if key in seen:
                    continue
                d = float(np.min(np.linalg.norm(chosen - row["psi"][None, :], axis=1)))
                item = dict(row)
                item["diversity_distance"] = d
                diverse_rows.append(item)
            if not diverse_rows:
                break
            diverse_rows.sort(key=lambda row: (-row["diversity_distance"], row["score"]))
            add_rows(diverse_rows, 1, "diverse_source_psi")

        self.anchor_scores = np.asarray([row["score"] for row in selected], dtype=float)
        self.anchor_psi = np.vstack([row["psi"] for row in selected])
        self.anchor_meta = [
            {
                "domain": row["domain"],
                "margin": float(row["margin"]),
                "scaled_margin": float(row["scaled_margin"]),
                "objective": float(row["objective"]),
                "feasible": bool(row["feasible"]),
                "anchor_type": row["anchor_type"],
            }
            for row in selected
        ]
        self.profile_templates = []
        profile_seen = set()
        for row in selected:
            profile = row.get("profile")
            if profile is None or len(profile) == 0:
                continue
            profile = np.clip(np.asarray(profile, dtype=float).reshape(-1), 0.0, 1.0)
            key = tuple(np.round(profile, 3))
            if key in profile_seen:
                continue
            profile_seen.add(key)
            self.profile_templates.append({
                "profile": profile,
                "domain": row["domain"],
                "anchor_type": row["anchor_type"],
                "feasible": bool(row["feasible"]),
                "score": float(row["score"]),
                "margin": float(row["margin"]),
                "objective": float(row["objective"]),
            })
        if selected:
            margins = np.asarray([row["margin"] for row in selected], dtype=float)
            self.training_diagnostics.update({
                "anchor_feasible_rate": float(np.mean(margins <= 0.0)),
                "anchor_margin_median": float(np.median(margins)),
                "anchor_margin_abs_median": float(np.median(np.abs(margins))),
                "anchor_types": {
                    anchor_type: int(sum(
                        1 for row in selected if row["anchor_type"] == anchor_type
                    ))
                    for anchor_type in sorted({row["anchor_type"] for row in selected})
                },
                "profile_template_count": int(len(self.profile_templates)),
            })

    def state_anchor_points(self, n=10, rng=None):
        if not self.component_enabled("proposal"):
            return []
        rng = rng or np.random.default_rng(self.seed)
        if len(self.anchor_psi) == 0:
            return []
        n_take = max(0, int(n))
        if (
            self.anchor_sampling_temperature > 0.0
            and len(self.anchor_scores) == len(self.anchor_psi)
        ):
            scores = np.asarray(self.anchor_scores, dtype=float)
            scale = float(np.std(scores))
            scale = max(scale, 1e-8)
            logits = -scores / (scale * self.anchor_sampling_temperature)
            logits -= float(np.max(logits))
            probs = np.exp(logits)
            probs = probs / max(float(np.sum(probs)), 1e-12)
            order = rng.choice(
                len(self.anchor_psi),
                size=min(n_take, len(self.anchor_psi)),
                replace=False,
                p=probs,
            )
        else:
            order = rng.permutation(len(self.anchor_psi))[:n_take]
        anchors = []
        for pos in order:
            psi = np.asarray(self.anchor_psi[int(pos)], dtype=float)
            anchors.append({
                "psi": psi.tolist(),
                "A": psi[: self.local_dim].tolist(),
                "N": psi[self.local_dim:].tolist(),
                "source_score": float(self.anchor_scores[int(pos)]),
                "source_meta": (
                    self.anchor_meta[int(pos)]
                    if int(pos) < len(self.anchor_meta)
                    else {}
                ),
                "coordinate": "learned_meta_psi=(A,N)",
            })
        return anchors

    def _continuous_to_tuple(self, problem, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        z = np.clip(z, 0.0, 1.0)
        try:
            return _as_tuple(problem.continuous_to_int(z))
        except (AttributeError, TypeError, ValueError):
            L = int(getattr(problem, "L", 100))
            return tuple(int(np.clip(round(v * L), 0, L)) for v in z)

    def universal_shape_candidates(self, problem, n=0, rng=None, force=False):
        """Admissible low-complexity policy shapes shared by all domains.

        These candidates use only bounds and dimension, not held-out target
        objectives, constraints, anchors, or risk coordinates.  They act like a
        weak smoothness/low-complexity prior: constants, one-control-plus-tail,
        piecewise thirds, and monotone ramps.
        """
        if not force and not self.component_enabled("proposal"):
            return []
        n_take = max(0, int(n))
        if n_take <= 0:
            return []
        d = max(1, int(getattr(problem, "d", 1)))
        rows = []

        def add(z):
            rows.append(self._continuous_to_tuple(problem, z))

        levels = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        for value in levels:
            add(np.full(d, value, dtype=float))

        constant_end = len(rows)

        head_levels = [0.25, 0.35, 0.50, 0.65, 0.15, 0.80]
        tail_levels = [0.75, 0.65, 0.85, 0.55, 0.45, 0.30]
        for head in head_levels:
            for tail in tail_levels:
                z = np.full(d, tail, dtype=float)
                z[0] = head
                add(z)

        head_tail_end = len(rows)

        third_levels = [0.25, 0.40, 0.55, 0.70]
        third_templates = [
            (a, b, c)
            for a in third_levels
            for b in third_levels
            for c in third_levels
        ]
        for a, b, c in third_templates:
            z = np.empty(d, dtype=float)
            for j in range(d):
                z[j] = (a, b, c)[min(2, int(3 * j / max(d, 1)))]
            add(z)

        thirds_end = len(rows)

        for lo, hi in [(0.20, 0.80), (0.30, 0.70), (0.40, 0.60)]:
            add(np.linspace(lo, hi, d))
            add(np.linspace(hi, lo, d))

        blocks = [
            rows[:constant_end],
            rows[constant_end:head_tail_end],
            rows[head_tail_end:thirds_end],
            rows[thirds_end:],
        ]
        balanced = []
        max_block = max((len(block) for block in blocks), default=0)
        for pos in range(max_block):
            for block in blocks:
                if pos < len(block):
                    balanced.append(block[pos])
        rows = unique_candidates(balanced)
        if len(rows) <= n_take:
            return rows
        # Keep early hand-balanced shapes and randomly thin only the excess.
        head = rows[: min(len(rows), max(min(n_take, 8), n_take // 3))]
        tail = rows[len(head):]
        rng = rng or np.random.default_rng(self.seed)
        need = max(0, n_take - len(head))
        if need > 0 and tail:
            order = rng.permutation(len(tail))[:need]
            head.extend(tail[int(i)] for i in order)
        return unique_candidates(head)[:n_take]

    def boundary_excitation_candidates(
        self,
        problem,
        n=0,
        rng=None,
        *,
        include_source_templates=True,
    ):
        """Generate a dimension-equivariant unlabeled pool for ``phi`` search.

        The pool combines a fixed universal library, source boundary-stratum
        profiles, and random low-frequency perturbations. It never calls a
        held-out objective, constraint, risk provider, state anchor, or oracle.
        Target observations enter only later when the fitted boundary posterior
        ranks this frozen-form candidate pool.
        """

        n_take = max(0, int(n))
        if n_take <= 0:
            return []
        rng = rng or np.random.default_rng(self.seed)
        d = max(1, int(getattr(problem, "d", 1)))
        # Reserve room for source-stratum replay and stochastic low-frequency
        # coverage.  The deterministic universal block is still large enough
        # to contain the full coarse head/tail grid at the paper pool size.
        universal_budget = min(
            n_take,
            max(64, int(np.ceil(0.625 * n_take))),
        )
        rows = self.universal_shape_candidates(
            problem, n=universal_budget, rng=rng, force=True)
        # This denser support library is intentionally scoped to the unlabeled
        # target candidate pool. Keeping it out of
        # ``universal_shape_candidates`` preserves the frozen source archive
        # and its proposal hash.
        dense_budget = max(0, universal_budget - len(rows))
        if dense_budget:
            levels = [float(value) for value in np.linspace(0.10, 0.90, 9)]
            dense_rows = []
            for head in levels:
                for tail in levels:
                    z = np.full(d, tail, dtype=float)
                    z[0] = head
                    dense_rows.append(self._continuous_to_tuple(problem, z))
            third_levels = [0.10, 0.30, 0.50, 0.70, 0.90]
            triples = [
                (first, second, third)
                for first in third_levels
                for second in third_levels
                for third in third_levels
            ]
            universal_rng = np.random.default_rng(20_260_718)
            triples = [
                triples[int(index)]
                for index in universal_rng.permutation(len(triples))
            ]
            for first, second, third in triples:
                z = np.empty(d, dtype=float)
                for index in range(d):
                    z[index] = (first, second, third)[min(
                        2, int(3 * index / max(d, 1)))]
                dense_rows.append(self._continuous_to_tuple(problem, z))
            rows.extend(dense_rows[:dense_budget])
        templates = (
            list(getattr(
                self.observable_mean_model,
                "boundary_profile_templates",
                [],
            ))
            if include_source_templates else []
        )
        for template in templates:
            if len(rows) >= n_take:
                break
            candidate = self._profile_to_target_candidate(
                problem, template["profile"])
            if candidate is not None:
                rows.append(candidate)
        rows = unique_candidates(rows)

        positions = (np.arange(d, dtype=float) + 0.5) / float(d)
        attempts = 0
        max_attempts = max(8 * n_take, 64)
        while len(rows) < n_take and attempts < max_attempts:
            family = attempts % 4
            if family == 0:
                # Piecewise policies span domain-generic local regimes.
                levels = rng.beta(2.0, 2.0, size=3)
                z = np.empty(d, dtype=float)
                for index in range(d):
                    z[index] = levels[min(
                        2, int(3 * index / max(d, 1)))]
            elif family == 1:
                # Head/tail policies cover the state-policy family without a
                # target-specific interpretation of either coordinate.
                z = np.full(d, float(rng.beta(2.0, 2.0)), dtype=float)
                z[0] = float(rng.beta(2.0, 2.0))
            elif family == 2 or not templates:
                # Smooth random Fourier profiles implement the shared
                # low-frequency structural prior at any raw dimension.
                z = np.full(d, float(rng.uniform(0.12, 0.88)), dtype=float)
                for frequency in range(1, 5):
                    coefficient = float(rng.normal(
                        0.0, 0.20 / float(frequency)))
                    z += coefficient * np.cos(
                        np.pi * frequency * positions)
            else:
                # Perturb a source boundary stratum in normalized profile
                # space; source labels choose the template before target time.
                template = templates[int(rng.integers(len(templates)))]
                profile = np.asarray(
                    template["profile"], dtype=float).reshape(-1)
                z = np.interp(
                    np.linspace(0.0, 1.0, d),
                    np.linspace(0.0, 1.0, len(profile)),
                    profile,
                )
                z += float(rng.normal(0.0, 0.05))
                for frequency in range(1, 4):
                    z += float(rng.normal(
                        0.0, 0.08 / frequency)) * np.cos(
                            np.pi * frequency * positions)
            rows.append(self._continuous_to_tuple(
                problem, np.clip(z, 0.0, 1.0)))
            rows = unique_candidates(rows)
            attempts += 1

        self.boundary_excitation_diagnostics = {
            "status": "materialized",
            "source_only": True,
            "target_data_used": False,
            "target_oracle_used": False,
            "pool_size": int(len(rows[:n_take])),
            "target_policy_dimension": int(d),
            "source_boundary_template_count": int(len(templates)),
            "source_boundary_templates_enabled": bool(
                include_source_templates),
            "universal_and_low_frequency_families": [
                "constant_head_tail_piecewise_ramp",
                "target_unlabeled_dense_tenth_head_tail",
                "target_unlabeled_dense_three_block",
                "continuous_piecewise_thirds",
                "continuous_head_tail",
                "random_cosine_1_to_4",
                "source_stratum_cosine_perturbation",
            ],
        }
        return rows[:n_take]

    def _profile_to_target_candidate(self, problem, profile):
        profile = np.asarray(profile, dtype=float).reshape(-1)
        d = max(1, int(getattr(problem, "d", 1)))
        if len(profile) == 0:
            return None
        if len(profile) == d:
            z = profile.copy()
        elif len(profile) == 1:
            z = np.full(d, float(profile[0]), dtype=float)
        else:
            z = np.interp(
                np.linspace(0.0, 1.0, d),
                np.linspace(0.0, 1.0, len(profile)),
                profile,
            )
        return self._continuous_to_tuple(problem, np.clip(z, 0.0, 1.0))

    def source_consensus_template_candidates(
        self,
        problem,
        n=0,
        rng=None,
        randomized=False,
    ):
        """Replay source-ranked shared profiles without target labels."""

        n_take = max(0, int(n))
        if n_take <= 0 or not self.source_consensus_templates:
            return []
        count = min(n_take, len(self.source_consensus_templates))
        if randomized and count < len(self.source_consensus_templates):
            rng = rng or np.random.default_rng(self.seed)
            scores = np.asarray([
                float(row["score"]) for row in self.source_consensus_templates
            ], dtype=float)
            scale = max(float(np.std(scores)), 0.05)
            logits = -(scores - float(np.min(scores))) / scale
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= max(float(np.sum(probabilities)), 1e-12)
            order = rng.choice(
                len(self.source_consensus_templates),
                size=count,
                replace=False,
                p=probabilities,
            ).tolist()
        else:
            order = list(range(count))
        rows = []
        for index in order:
            candidate = self._profile_to_target_candidate(
                problem,
                self.source_consensus_templates[int(index)]["profile"],
            )
            if candidate is not None:
                rows.append(candidate)
        return unique_candidates(rows)[:n_take]

    def initial_universal_candidates(self, problem, n=0, rng=None):
        """Protect a sentinel and a rank-spanning source archive design.

        The source consensus is an ordering, not a guarantee that its first
        member transfers to every held-out task.  When the target initial
        budget admits more than one consensus member, sample the complete
        frozen shortlist at approximately equal rank quantiles.  This is a
        source-only space-filling design over transfer uncertainty: it does
        not inspect target labels, but it avoids spending every protected
        target call on nearly identical top-ranked source policies.
        """

        n_take = max(0, int(n))
        if n_take <= 0 or not self.source_consensus_templates:
            return self.universal_shape_candidates(
                problem, n=n_take, rng=rng, force=True)
        library = self.universal_shape_candidates(
            problem, n=10000, rng=rng, force=True)
        rows = []
        # The first head/tail low-frequency shape is formula-free and protects
        # domains whose safe direction disagrees with source consensus.
        if len(library) > 1:
            rows.append(library[1])
        remaining = max(0, n_take - len(rows))
        if remaining > 0:
            template_count = len(self.source_consensus_templates)
            if remaining >= template_count:
                template_indices = list(range(template_count))
            elif remaining == 1:
                template_indices = [0]
            else:
                template_indices = []
                for value in np.linspace(
                    0.0, float(template_count - 1), remaining
                ):
                    index = int(np.clip(
                        np.round(value), 0, template_count - 1))
                    if index not in template_indices:
                        template_indices.append(index)
                template_indices[0] = 0
                template_indices[-1] = template_count - 1
                # Integer rounding can collide for very short shortlists.
                for index in range(template_count):
                    if len(template_indices) >= remaining:
                        break
                    if index not in template_indices:
                        template_indices.append(index)
            for index in template_indices[:remaining]:
                candidate = self._profile_to_target_candidate(
                    problem,
                    self.source_consensus_templates[int(index)]["profile"],
                )
                if candidate is not None:
                    rows.append(candidate)
        rows.extend(library)
        return unique_candidates(rows)[:n_take]

    def _dimension_equivariant_profile_coordinate(self, profile):
        """Represent a policy curve without retaining its raw dimension.

        The coordinate contains normalized ordered moments and source-learned
        cosine modes.  The same four structural switches used by the posterior
        also control this proposal coordinate, so proposal ablations no longer
        share a hidden full-prior representation.
        """
        if not self.spectral_low_frequency_prior:
            # A true no-low-frequency control keeps the complete canonical
            # profile instead of silently retaining a truncated cosine basis.
            values = self._canonical_profile(profile)
        else:
            library = self._ordered_profile_library(profile)
            frequency_count = self.ordered_exposure_max_frequency
            if (
                self.ordered_exposure_adaptive_sparsity
                and len(self.ordered_exposure_selected_frequencies) > 0
            ):
                frequencies = np.asarray(
                    self.ordered_exposure_selected_frequencies, dtype=int)
            else:
                frequencies = np.arange(1, frequency_count + 1, dtype=int)
            values = np.concatenate([
                library[:2],
                library[1 + frequencies],
            ])
            if self.ordered_exposure_frequency_penalty > 0.0:
                weights = np.concatenate([
                    np.ones(2, dtype=float),
                    1.0 / (
                        1.0
                        + self.ordered_exposure_frequency_penalty
                        * (frequencies.astype(float) - 1.0)
                    ),
                ])
                values = values * weights
        values = self._ordered_coordinate_transform(values)
        if self.ordered_exposure_basis_mode == "diagonal_quadratic":
            interactions = values ** 2
        else:
            interactions = np.asarray([
                values[i] * values[j]
                for i in range(len(values))
                for j in range(i, len(values))
            ], dtype=float)
        coordinate = np.concatenate([values, interactions])
        scale = max(float(np.linalg.norm(coordinate)), 1.0)
        return coordinate / scale

    def _dimension_equivariant_profile_candidate(self, problem, profile):
        """Synthesize one target policy from dimensionless cosine moments."""
        if self.source_design_mode == "shared_uniform":
            return self._profile_to_target_candidate(problem, profile)
        library = self._ordered_profile_library(profile)
        d = max(1, int(getattr(problem, "d", 1)))
        positions = (np.arange(d, dtype=float) + 0.5) / float(d)
        z = np.full(d, float(library[0]), dtype=float)
        for frequency in range(1, self.ordered_exposure_max_frequency + 1):
            z += float(library[1 + frequency]) * np.cos(
                np.pi * frequency * positions)
        return self._continuous_to_tuple(problem, np.clip(z, 0.0, 1.0))

    def dimension_equivariant_initial_candidates(self, problem, n=0, rng=None):
        """Build a source-only maximin atlas in normalized risk coordinates.

        Unlike raw profile interpolation, selection occurs in a coordinate
        whose size is independent of the source and target policy dimensions.
        A source-ranked seed is followed by maximin coverage, preventing all
        target calls from collapsing onto one apparently safe source shape.
        Target objective, constraint, and oracle labels are never queried.
        """
        n_take = max(0, int(n))
        if n_take <= 0:
            return []
        rng = rng or np.random.default_rng(self.seed)
        library = self.universal_shape_candidates(
            problem, n=10000, rng=rng, force=True)
        rows = (
            []
            if self.source_design_mode == "shared_uniform"
            else ([library[1]] if len(library) > 1 else list(library[:1]))
        )

        templates = []
        for rank, item in enumerate(self.source_consensus_templates):
            templates.append({
                "profile": np.asarray(item["profile"], dtype=float),
                "score": float(item.get("score", rank)),
                "origin": "source_consensus",
            })
        for rank, item in enumerate(self.profile_templates):
            templates.append({
                "profile": self._canonical_profile(item["profile"]),
                "score": float(item.get("score", rank + 1.0)),
                "origin": "source_anchor",
            })

        unique = {}
        for item in templates:
            key = tuple(np.round(self._canonical_profile(item["profile"]), 6))
            if key not in unique or item["score"] < unique[key]["score"]:
                unique[key] = item
        templates = list(unique.values())
        if templates and len(rows) < n_take:
            coordinates = np.vstack([
                self._dimension_equivariant_profile_coordinate(item["profile"])
                for item in templates
            ])
            scores = np.asarray([item["score"] for item in templates], dtype=float)
            score_scale = max(float(np.std(scores)), 1e-8)
            selected = [int(np.argmin(scores))]
            while len(selected) < min(n_take - len(rows), len(templates)):
                remaining = [
                    index for index in range(len(templates))
                    if index not in selected
                ]
                chosen = coordinates[np.asarray(selected, dtype=int)]
                distance = np.asarray([
                    float(np.min(np.linalg.norm(
                        chosen - coordinates[index][None, :], axis=1)))
                    for index in remaining
                ])
                rank_penalty = np.asarray([
                    (scores[index] - float(np.min(scores))) / score_scale
                    for index in remaining
                ])
                utility = distance / (1.0 + 0.20 * np.maximum(rank_penalty, 0.0))
                selected.append(remaining[int(np.argmax(utility))])
            for index in selected:
                rows.append(self._dimension_equivariant_profile_candidate(
                    problem, templates[index]["profile"]))
            self.dimension_equivariant_proposal_diagnostics = {
                "status": "fit",
                "source_only": True,
                "target_data_used": False,
                "target_oracle_used": False,
                "source_template_count": int(len(templates)),
                "selected_template_count": int(len(selected)),
                "coordinate_dimensions": sorted({
                    int(len(coordinate)) for coordinate in coordinates
                }),
                "source_policy_dimensions": sorted({
                    int(len(np.asarray(item["profile"]).reshape(-1)))
                    for item in templates
                }),
                "target_policy_dimension": int(getattr(problem, "d", 1)),
                "selected_origins": [templates[index]["origin"] for index in selected],
            }
        else:
            self.dimension_equivariant_proposal_diagnostics = {
                "status": "no_source_templates",
                "source_only": True,
                "target_data_used": False,
                "target_oracle_used": False,
                "target_policy_dimension": int(getattr(problem, "d", 1)),
            }
        rows.extend(library)
        return unique_candidates(rows)[:n_take]

    def risk_objective_initial_candidates(self, problem, n=0, rng=None):
        """Build a source-only safety-objective Pareto proposal atlas.

        Chance margins and objectives are ranked separately within every
        source domain before aggregation.  The proposal therefore transfers
        order information without comparing domain-specific physical scales.
        Target values are never queried.  Pareto elites preserve source-safe
        objective quality, while maximin filling retains risk-coordinate
        coverage for held-out dimensions.
        """
        n_take = max(0, int(n))
        if n_take <= 0:
            return []
        rng = rng or np.random.default_rng(self.seed)
        library = self.universal_shape_candidates(
            problem, n=10000, rng=rng, force=True)
        rows = [library[1]] if len(library) > 1 else list(library[:1])

        unique = {}
        for item in self.source_consensus_templates:
            profile = self._canonical_profile(item["profile"])
            key = tuple(np.round(profile, 6))
            candidate = {
                "profile": profile,
                "safety_score": float(item.get("score", 0.0)),
                "objective_score": float(item.get(
                    "objective_score", item.get("score", 0.0))),
                "feasible_source_count": int(item.get(
                    "feasible_source_count", 0)),
                "source_domain_count": int(item.get(
                    "source_domain_count", 0)),
            }
            old = unique.get(key)
            if old is None or (
                candidate["safety_score"], candidate["objective_score"]
            ) < (old["safety_score"], old["objective_score"]):
                unique[key] = candidate
        templates = list(unique.values())

        selected = []
        selected_roles = []
        if templates and len(rows) < n_take:
            coordinates = np.vstack([
                self._dimension_equivariant_profile_coordinate(item["profile"])
                for item in templates
            ])
            safety = np.asarray([
                item["safety_score"] for item in templates
            ], dtype=float)
            objective = np.asarray([
                item["objective_score"] for item in templates
            ], dtype=float)
            safety_rank = self._percentile_ranks(safety)
            objective_rank = self._percentile_ranks(objective)
            robust = np.asarray([
                item["source_domain_count"] > 0
                and item["feasible_source_count"] == item["source_domain_count"]
                for item in templates
            ], dtype=bool)
            eligible = np.flatnonzero(robust)
            if len(eligible) == 0:
                eligible = np.arange(len(templates), dtype=int)

            pareto = []
            for index in eligible:
                dominated = any(
                    other != index
                    and safety_rank[other] <= safety_rank[index]
                    and objective_rank[other] <= objective_rank[index]
                    and (
                        safety_rank[other] < safety_rank[index]
                        or objective_rank[other] < objective_rank[index]
                    )
                    for other in eligible
                )
                if not dominated:
                    pareto.append(int(index))
            if not pareto:
                pareto = [int(index) for index in eligible]

            limit = min(n_take - len(rows), len(templates))

            def add(index, role):
                index = int(index)
                if index in selected or len(selected) >= limit:
                    return
                selected.append(index)
                selected_roles.append(str(role))

            for weight in (1.0, 0.0, 0.5, 0.25, 0.75):
                if len(selected) >= limit:
                    break
                index = min(
                    pareto,
                    key=lambda item: (
                        weight * safety_rank[item]
                        + (1.0 - weight) * objective_rank[item],
                        safety_rank[item],
                        objective_rank[item],
                        item,
                    ),
                )
                add(index, f"pareto_weight_{weight:.2f}")

            while len(selected) < limit:
                remaining = [
                    index for index in range(len(templates))
                    if index not in selected
                ]
                if not remaining:
                    break
                if selected:
                    chosen = coordinates[np.asarray(selected, dtype=int)]
                    distance = np.asarray([
                        float(np.min(np.linalg.norm(
                            chosen - coordinates[index][None, :], axis=1)))
                        for index in remaining
                    ])
                else:
                    distance = np.ones(len(remaining), dtype=float)
                penalty = np.asarray([
                    0.20 * safety_rank[index]
                    + 0.20 * objective_rank[index]
                    + (0.75 if not robust[index] else 0.0)
                    for index in remaining
                ])
                utility = distance / (1.0 + penalty)
                chosen_index = remaining[int(np.argmax(utility))]
                add(chosen_index, "risk_coordinate_maximin")

            for index in selected:
                rows.append(self._dimension_equivariant_profile_candidate(
                    problem, templates[index]["profile"]))
            self.risk_objective_proposal_diagnostics = {
                "status": "fit",
                "source_only": True,
                "target_data_used": False,
                "target_oracle_used": False,
                "source_template_count": int(len(templates)),
                "robust_source_feasible_template_count": int(np.sum(robust)),
                "pareto_template_count": int(len(pareto)),
                "selected_template_count": int(len(selected)),
                "selected_roles": list(selected_roles),
                "selected_safety_scores": [
                    float(safety[index]) for index in selected
                ],
                "selected_objective_scores": [
                    float(objective[index]) for index in selected
                ],
                "coordinate_dimensions": sorted({
                    int(len(coordinate)) for coordinate in coordinates
                }),
                "source_policy_dimensions": sorted({
                    int(len(item["profile"])) for item in templates
                }),
                "target_policy_dimension": int(getattr(problem, "d", 1)),
                "safety_ranking_target": (
                    "source_chance_margin_percentile"),
                "objective_ranking_target": "source_objective_percentile",
                "source_design_mode": self.source_design_mode,
                "low_frequency_coordinate": bool(
                    self.spectral_low_frequency_prior),
                "generic_library_fallback_count": int(max(
                    0, n_take - len(unique_candidates(rows)))),
            }
        else:
            self.risk_objective_proposal_diagnostics = {
                "status": "no_source_templates",
                "source_only": True,
                "target_data_used": False,
                "target_oracle_used": False,
                "target_policy_dimension": int(getattr(problem, "d", 1)),
            }
        rows.extend(library)
        return unique_candidates(rows)[:n_take]

    def source_coverage_candidates(self, problem, n=0):
        """Return the deterministic source-only design protected at target time."""

        if not self.source_consensus_templates:
            return []
        return self.initial_universal_candidates(
            problem,
            n=max(0, int(n)),
            rng=np.random.default_rng(self.seed + 32452843),
        )

    def universal_expert_candidates(self, problem, n=0, rng=None):
        """Draw varied frozen templates for sequential universal proposals."""

        n_take = max(0, int(n))
        if n_take <= 0 or not self.source_consensus_templates:
            return self.universal_shape_candidates(
                problem, n=n_take, rng=rng, force=True)
        rng = rng or np.random.default_rng(self.seed)
        rows = self.source_consensus_template_candidates(
            problem,
            n=min(n_take, len(self.source_consensus_templates)),
            rng=rng,
            randomized=True,
        )
        if len(rows) < n_take:
            library = self.universal_shape_candidates(
                problem, n=10000, rng=rng, force=True)
            if library:
                order = rng.permutation(len(library))
                rows.extend(library[int(index)] for index in order)
        return unique_candidates(rows)[:n_take]

    def profile_template_candidates(self, problem, n=0, rng=None):
        """Replay source-learned normalized policy profiles on a held-out target.

        Templates are selected only from source-domain records during
        `fit_from_source_problems`.  At test time we resample each normalized
        profile to the target dimension and convert through target bounds.  This
        is a LODO meta-prior: it transfers policy shape, not target objective or
        feasibility labels.
        """
        if not self.component_enabled("proposal"):
            return []
        n_take = max(0, int(n))
        if n_take <= 0 or not self.profile_templates:
            return []
        rng = rng or np.random.default_rng(self.seed)
        d = max(1, int(getattr(problem, "d", 1)))
        order = list(range(len(self.profile_templates)))
        # Keep source-safe/low-score templates early, randomize ties to avoid a
        # single source domain dominating all held-out proposals.
        order.sort(key=lambda i: (
            0 if self.profile_templates[i].get("feasible", False) else 1,
            float(self.profile_templates[i].get("score", 0.0)),
            float(abs(self.profile_templates[i].get("margin", 0.0))),
        ))
        if len(order) > n_take:
            head = order[: max(1, min(len(order), n_take // 2))]
            tail = order[len(head):]
            need = n_take - len(head)
            if need > 0 and tail:
                pick = rng.permutation(len(tail))[:need]
                head.extend(tail[int(i)] for i in pick)
            order = head
        rows = []
        for idx in order[:n_take]:
            profile = np.asarray(
                self.profile_templates[int(idx)]["profile"],
                dtype=float,
            ).reshape(-1)
            if len(profile) == 0:
                continue
            if len(profile) == d:
                z = profile.copy()
            else:
                xp = np.linspace(0.0, 1.0, len(profile))
                xnew = np.linspace(0.0, 1.0, d)
                z = np.interp(xnew, xp, profile)
            rows.append(self._continuous_to_tuple(problem, np.clip(z, 0.0, 1.0)))
        return unique_candidates(rows)[:n_take]

    def inverse_state_anchor(self, problem, anchor, rng=None, n=1, pool_size=512):
        if not self.component_enabled("proposal"):
            return []
        rng = rng or np.random.default_rng(self.seed)
        n = max(1, int(n))
        target = None
        if isinstance(anchor, dict):
            if anchor.get("psi") is not None:
                target = np.asarray(anchor["psi"], dtype=float)
            elif anchor.get("A") is not None and anchor.get("N") is not None:
                target = np.concatenate([
                    np.asarray(anchor["A"], dtype=float),
                    np.asarray(anchor["N"], dtype=float),
                ])
        if target is None:
            return []
        rows = [problem.sample_random(rng) for _ in range(max(pool_size, 8 * n))]
        scored = []
        for x in unique_candidates(rows):
            psi = self.risk_coordinate(problem, x)
            scored.append((float(np.linalg.norm(psi - target)), _as_tuple(x)))
        scored.sort(key=lambda item: item[0])
        return [row for _, row in scored[:n]]

    def proposal_candidates(self, problem, n=32, rng=None, pool_size=1024):
        if not self.component_enabled("proposal"):
            rng = rng or np.random.default_rng(self.seed)
            n_target = max(0, int(n))
            n_alignment = min(
                len(self.alignment_profile_templates),
                int(np.ceil(0.75 * n_target)),
            )
            if (
                self.spectral_alignment_latent_proposals
                and self.alignment_latent_proposal_supported()
            ):
                rows = self.alignment_latent_candidates(
                    problem,
                    n=n_alignment,
                    rng=rng,
                    pool_size=max(
                        int(pool_size),
                        self.spectral_alignment_inverse_pool_size,
                    ),
                )
            else:
                rows = self.alignment_profile_candidates(
                    problem,
                    n=n_alignment,
                    rng=rng,
                )
            while len(rows) < n_target:
                rows.append(problem.sample_random(rng))
                rows = unique_candidates(rows)
            return rows[:n_target]
        rng = rng or np.random.default_rng(self.seed)
        n_target = max(0, int(n))
        rows = []
        n_profiles = min(
            len(self.profile_templates),
            n_target,
            max(min(n_target, 3), int(round(0.35 * n_target))),
        )
        rows.extend(self.profile_template_candidates(
            problem,
            n=n_profiles,
            rng=rng,
        ))
        n_universal = min(
            max(0, int(self.universal_shape_count)),
            max(0, n_target - len(rows)),
            max(min(max(0, n_target - len(rows)), 4), int(round(0.35 * n_target))),
        )
        rows.extend(self.universal_shape_candidates(
            problem,
            n=n_universal,
            rng=rng,
        ))
        n_anchor = max(1, n_target - len(rows))
        for anchor in self.state_anchor_points(n=n_anchor, rng=rng):
            rows.extend(self.inverse_state_anchor(
                problem,
                anchor,
                rng=rng,
                n=1,
                pool_size=max(64, int(pool_size) // max(1, n_anchor)),
            ))
        while len(rows) < n_target:
            rows.append(problem.sample_random(rng))
        return unique_candidates(rows)[:n_target]

    def _hvd_prior_family(self, coordinate_variant="aligned"):
        variant = str(coordinate_variant or "aligned").lower()
        if variant in ("aligned", "risk_aligned", "cumulative"):
            return (
                self.beta_prior,
                self.beta_prior_precision,
                self.beta_prior_reference_mean,
                self.beta_prior_upper_scale,
            )
        if variant in ("unaligned", "universal", "source"):
            return (
                self.unaligned_beta_prior,
                self.unaligned_beta_prior_precision,
                self.unaligned_beta_prior_reference_mean,
                self.unaligned_beta_prior_upper_scale,
            )
        raise ValueError(f"unknown HVD coordinate variant {coordinate_variant!r}")

    def _hvd_prior_component_family(self, coordinate_variant="aligned"):
        variant = str(coordinate_variant or "aligned").lower()
        if variant in ("aligned", "risk_aligned", "cumulative"):
            return self.beta_prior_components, self.beta_prior_component_domains
        if variant in ("unaligned", "universal", "source"):
            return (
                self.unaligned_beta_prior_components,
                self.unaligned_beta_prior_component_domains,
            )
        raise ValueError(f"unknown HVD coordinate variant {coordinate_variant!r}")

    def cumulative_hvd_prior_beta(
        self,
        output_index=1,
        feature_dim=None,
        coordinate_variant="aligned",
    ):
        if not self.component_enabled("hvd"):
            return None
        beta_store, _, reference_store, _ = self._hvd_prior_family(
            coordinate_variant)
        beta = beta_store.get(int(output_index))
        if beta is None:
            return None
        beta = np.asarray(beta, dtype=float)
        if self.component_stage == "spectral_hvd":
            beta = beta / max(
                float(reference_store.get(
                    int(output_index), 1.0)),
                1e-12,
            )
        if feature_dim is not None and len(beta) != int(feature_dim):
            return None
        return beta.copy()

    def cumulative_hvd_prior_precision(
        self, output_index=1, coordinate_variant="aligned"):
        if self.component_stage != "spectral_hvd":
            return None
        _, precision_store, _, _ = self._hvd_prior_family(coordinate_variant)
        value = precision_store.get(int(output_index))
        return None if value is None else float(value)

    def cumulative_hvd_prior_components(
        self,
        output_index=1,
        feature_dim=None,
        coordinate_variant="aligned",
    ):
        """Return source-domain PSD coefficient shapes for hierarchical transfer."""
        if not self.component_enabled("hvd"):
            return None
        component_store, domain_store = self._hvd_prior_component_family(
            coordinate_variant)
        components = component_store.get(int(output_index))
        if components is None:
            return None
        components = np.asarray(components, dtype=float)
        if components.ndim != 2 or components.shape[0] == 0:
            return None
        if feature_dim is not None and components.shape[1] != int(feature_dim):
            return None
        return {
            "coefficients": components.copy(),
            "domains": list(domain_store.get(int(output_index), [])),
        }

    def cumulative_hvd_prior_scale_mean(
        self, output_index=1, coordinate_variant="aligned"):
        if self.component_stage != "spectral_hvd":
            return None
        _, _, reference_store, _ = self._hvd_prior_family(coordinate_variant)
        value = reference_store.get(int(output_index))
        return None if value is None else float(value)

    def cumulative_hvd_prior_upper_scale(
        self, output_index=1, coordinate_variant="aligned"):
        if self.component_stage != "spectral_hvd":
            return None
        _, _, _, upper_store = self._hvd_prior_family(coordinate_variant)
        value = upper_store.get(int(output_index))
        return None if value is None else float(value)

    def cumulative_hvd_prior_min_records(self):
        return 5 if self.component_stage == "spectral_hvd" else None

    def task_sensitivity_prior(self):
        return copy.deepcopy(self.task_sensitivity_prior_)

    def source_calibrated_recommendation_slack(self):
        if not self.component_enabled("proposal"):
            return 0.0
        return float(max(
            self.training_diagnostics.get("source_recommendation_slack", 0.0) or 0.0,
            0.0,
        ))

    def observable_mean_features(self, problem, x):
        """Return the frozen source-learned constraint-mean coordinate eta."""

        if self.observable_mean_model is None:
            raise RuntimeError("observable mean coordinate is unavailable")
        return self.observable_mean_model.features(problem, x)

    def diagnostics(self):
        return {
            "status": self.fit_status,
            "component_stage": self.component_stage,
            "enabled_components": [
                name for name in (
                    "coordinate", "spectral", "hvd", "mean", "proposal"
                )
                if self.component_enabled(name)
            ],
            "source_domains": list(self.source_domains),
            "n_records": int(self.n_records),
            "observable_mean_descriptor_mode": (
                self.observable_mean_descriptor_mode),
            "observable_mean_feature_mode": self.observable_mean_feature_mode,
            "local_dim": int(self.local_dim),
            "shared_dim": int(self.shared_dim),
            "n_anchors": int(len(self.anchor_psi)),
            "n_profile_templates": int(len(self.profile_templates)),
            "n_source_consensus_templates": int(
                len(self.source_consensus_templates)),
            "source_consensus_templates": copy.deepcopy(
                self.source_consensus_diagnostics),
            "dimension_equivariant_proposal": copy.deepcopy(
                self.dimension_equivariant_proposal_diagnostics),
            "risk_objective_proposal": copy.deepcopy(
                self.risk_objective_proposal_diagnostics),
            "boundary_excitation_proposal": copy.deepcopy(
                self.boundary_excitation_diagnostics),
            "n_alignment_profile_templates": int(
                len(self.alignment_profile_templates)),
            "universal_shape_count": int(self.universal_shape_count),
            "has_beta_prior": {
                str(key): value is not None
                for key, value in self.beta_prior.items()
            },
            "beta_prior_precision": {
                str(key): float(value)
                for key, value in self.beta_prior_precision.items()
            },
            "beta_prior_reference_mean": {
                str(key): float(value)
                for key, value in self.beta_prior_reference_mean.items()
            },
            "beta_prior_upper_scale": {
                str(key): float(value)
                for key, value in self.beta_prior_upper_scale.items()
            },
            "has_unaligned_beta_prior": {
                str(key): value is not None
                for key, value in self.unaligned_beta_prior.items()
            },
            "unaligned_beta_prior_precision": {
                str(key): float(value)
                for key, value in self.unaligned_beta_prior_precision.items()
            },
            "unaligned_beta_prior_reference_mean": {
                str(key): float(value)
                for key, value in self.unaligned_beta_prior_reference_mean.items()
            },
            "unaligned_beta_prior_upper_scale": {
                str(key): float(value)
                for key, value in self.unaligned_beta_prior_upper_scale.items()
            },
            "task_sensitivity_prior": copy.deepcopy(
                self.task_sensitivity_prior_),
            "has_mean_prior": {
                str(key): value is not None
                for key, value in self.mean_prior.items()
            },
            "observable_mean_coordinate": (
                None
                if self.observable_mean_model is None
                else {
                    **self.observable_mean_model.diagnostics(),
                    "training_target": self.observable_mean_training_target,
                    "input_mode": self.observable_mean_input_mode,
                    "chance_boundary_weighted": True,
                }
            ),
            "observable_mean_role_assignment": {
                "enabled": bool(
                    self.observable_mean_role_assignment_posterior),
                "prior": self.observable_mean_role_assignment_prior,
                "prior_temperature_scale": float(
                    self.observable_mean_role_assignment_prior_temperature_scale
                ),
                "inactive_variance": float(
                    self.observable_mean_role_assignment_inactive_variance),
                "hypotheses_use_target_labels": False,
                "hypotheses_use_target_oracle": False,
            },
            "observable_variance_coordinate": (
                None
                if self.observable_variance_model is None
                else self.observable_variance_model.diagnostics()
            ),
            "spectral_basis": (
                None
                if self.spectral_basis is None
                else self.spectral_basis.diagnostics()
            ),
            "stage1_spectral_basis": (
                None
                if self.stage1_spectral_basis is None
                else self.stage1_spectral_basis.diagnostics()
            ),
            "spectral_coefficient_shrinkage": bool(
                self.spectral_coefficient_shrinkage),
            "spectral_adaptive_sparsity": bool(
                self.spectral_adaptive_sparsity),
            "spectral_low_frequency_prior": bool(
                self.spectral_low_frequency_prior),
            "spectral_frequency_adaptation": bool(
                self.spectral_frequency_adaptation),
            "spectral_frequency": dict(
                self.spectral_frequency_diagnostics),
            "spectral_risk_alignment": bool(
                self.spectral_risk_alignment),
            "risk_alignment": dict(self.risk_alignment_diagnostics),
            "alignment_episode_admission": dict(
                self.alignment_episode_diagnostics),
            "source_boundary_bracket": {
                key: copy.deepcopy(value)
                for key, value in self.source_boundary_bracket_model.items()
                if key not in {"beta", "feature_mean", "feature_scale"}
            },
            "spectral_alignment_source_episodes": int(
                self.spectral_alignment_source_episodes),
            "spectral_alignment_admission": bool(
                self.spectral_alignment_admission),
            "spectral_alignment_latent_proposals": bool(
                self.spectral_alignment_latent_proposals),
            "spectral_alignment_inverse_pool_size": int(
                self.spectral_alignment_inverse_pool_size),
            "spectral_alignment_refit_interval": int(
                self.spectral_alignment_refit_interval),
            "spectral_frequency_refit_interval": int(
                self.spectral_frequency_refit_interval),
            "spectral_additive_adaptation": bool(
                self.spectral_additive_adaptation),
            "spectral_additive": dict(
                self.spectral_additive_diagnostics),
            "spectral_additive_refit_interval": int(
                self.spectral_additive_refit_interval),
            "spectral_additive_max_saturation_fraction": float(
                self.spectral_additive_max_saturation_fraction),
            "spectral_feature_dim": int(self.spectral_feature_dim),
            "spectral_always_active_count": int(
                self.spectral_always_active_count),
            "spectral_adaptive_config": {
                "min_pip": float(self.spectral_adaptive_min_pip),
                "max_pip": float(self.spectral_adaptive_max_pip),
                "spike_ratio": float(self.spectral_adaptive_spike_ratio),
                "damping": float(self.spectral_adaptive_damping),
                "max_iter": int(self.spectral_adaptive_max_iter),
                "tolerance": float(self.spectral_adaptive_tolerance),
                "residual_floor_scale": float(
                    self.spectral_adaptive_residual_floor_scale),
                "pilot_gate_tolerance": float(
                    self.spectral_adaptive_gate_tolerance),
                "multiplicity_correction": float(
                    self.spectral_adaptive_multiplicity_correction),
                "max_effective_fraction": float(
                    self.spectral_adaptive_max_effective_fraction),
                "saturation_fraction": float(
                    self.spectral_adaptive_saturation_fraction),
            },
            "spectral_adaptive_calibration": dict(
                self.spectral_adaptive_calibration),
            "ordered_cumulative_exposure": dict(
                self.ordered_exposure_diagnostics),
            "hierarchical_boundary": copy.deepcopy(
                self.hierarchical_boundary_diagnostics),
            "ordered_coefficient_prior": {
                str(output_index): {
                    key: (
                        value.tolist()
                        if isinstance(value, np.ndarray)
                        else value
                    )
                    for key, value in prior.items()
                }
                for output_index, prior in self.ordered_coefficient_prior.items()
            },
            "spectral_coefficient_prior": {
                str(output_index): {
                    key: (
                        value.tolist()
                        if isinstance(value, np.ndarray)
                        else list(value)
                        if isinstance(value, tuple)
                        else value
                    )
                    for key, value in prior.items()
                }
                for output_index, prior in self.spectral_coefficient_prior.items()
            },
            "coordinate": dict(self.coordinate_diagnostics),
            "training": dict(self.training_diagnostics),
        }


class MetaPriorSurrogateBasis:
    """Admissible target-observation calibration basis induced by frozen psi."""

    def __init__(self, meta_prior: LearnedMetaPrior, problem):
        self.meta_prior = meta_prior
        self.problem = problem
        descriptor_dim = len(meta_prior.feature_mean)
        psi_dim = meta_prior.local_dim + meta_prior.shared_dim
        cumulative_dim = (
            1
            + meta_prior.local_dim
            + meta_prior.shared_dim * (meta_prior.shared_dim + 1) // 2
            + meta_prior.shared_dim
        )
        self.feature_dim = (
            int(meta_prior.stage1_spectral_basis.feature_dim)
            if meta_prior.component_enabled("spectral")
            else descriptor_dim + 2 * psi_dim + cumulative_dim - 1
        )

    def features(self, x):
        if self.meta_prior.component_enabled("spectral"):
            return self.meta_prior.stage1_spectral_features(self.problem, x)
        return self.meta_prior.coordinate_basis_features(self.problem, x)

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(x) for x in X])


class ObservableConstraintMeanBasis:
    """Frozen source boundary coordinate for the target constraint mean.

    The profile variant is independent of ``psi=(A,N)``.  The exposure variant
    reads the same observable state exposure but applies a separate nonlinear
    mean map and separate coefficients; cumulative HVD retains its own direct
    variance parameterization. Target observations learn only mean-head
    coefficients.
    """

    constraint_mean_coordinate = True

    def __init__(self, meta_prior: LearnedMetaPrior, problem):
        if meta_prior.observable_mean_model is None:
            raise ValueError("observable mean coordinate is unavailable")
        self.meta_prior = meta_prior
        self.problem = problem
        self._role_assignment_model = None
        self._role_assignments = ()
        self._role_assignment_prior_weights = np.empty(0, dtype=float)
        self._role_assignment_geometry_prior_weights = np.empty(
            0, dtype=float)
        self._role_assignment_feature_dim = 0
        self._role_assignment_diagnostics = {"status": "disabled"}
        self._fit_role_assignment_bank()
        self.base_feature_dim = (
            len(self._role_assignments) * self._role_assignment_feature_dim
            if self._role_assignments
            else int(meta_prior.observable_mean_model.feature_dim)
        )
        self._target_residual_raw_mean = None
        self._target_residual_raw_scale = None
        self._target_residual_projection = None
        self._target_residual_components = None
        self._target_residual_score_scale = None
        self._target_residual_points = []
        self._target_residual_diagnostics = {
            "status": "disabled",
            "requested_rank": int(
                meta_prior.observable_mean_target_residual_rank),
            "effective_rank": 0,
            "target_labels_used": False,
            "target_oracle_used": False,
        }
        self._fit_target_residual_coordinate()
        self.target_residual_rank = int(
            0 if self._target_residual_components is None
            else len(self._target_residual_components))
        self.feature_dim = self.base_feature_dim + self.target_residual_rank

    @staticmethod
    def _assignment_label(assignment):
        return "-".join(str(int(value)) for value in assignment)

    def _fit_role_assignment_bank(self):
        if not self.meta_prior.observable_mean_role_assignment_posterior:
            return
        model = self.meta_prior.observable_mean_model
        role_model = getattr(model, "role_model", model)
        aligner = self.meta_prior.observable_channel_role_aligner
        if aligner is None or not hasattr(role_model, "features_profile"):
            raise RuntimeError(
                "role-assignment posterior requires a fitted role coordinate")
        points = list(aligner.target_policy_pool(self.problem))
        if not points:
            raise RuntimeError(
                "role-assignment posterior requires an unlabeled target pool")
        exposure = get_observable_state_exposure(self.problem, points[0])
        if exposure is None:
            raise RuntimeError(
                "role-assignment posterior requires observable target exposure")
        channel_count = int(len(exposure.channel_means))
        role_count = int(aligner.n_roles)
        if channel_count <= 0 or role_count <= 0 or channel_count > role_count:
            raise RuntimeError(
                "target channel cardinality cannot be injected into source roles")
        assignments = tuple(
            tuple(int(value) for value in assignment)
            for assignment in permutations(range(role_count), channel_count)
        )
        if not assignments:
            raise RuntimeError("role-assignment posterior has no admissible atom")
        self._role_assignment_model = role_model
        self._role_assignments = assignments
        self._role_assignment_feature_dim = int(role_model.feature_dim)
        if self.meta_prior.observable_mean_role_assignment_prior in {
            "source_geometry", "source_geometry_boundary"
        }:
            prior_weights, prior_diagnostics = aligner.target_assignment_prior(
                self.problem,
                assignments,
                temperature_scale=(
                    self.meta_prior
                    .observable_mean_role_assignment_prior_temperature_scale
                ),
            )
        else:
            prior_weights = np.full(
                len(assignments), 1.0 / float(len(assignments)), dtype=float)
            prior_diagnostics = {
                "status": "fit",
                "mode": "uniform",
                "weights": prior_weights.tolist(),
                "effective_assignment_count": float(len(assignments)),
                "target_labels_used": False,
                "target_oracle_used": False,
                "permutation_equivariant": True,
            }
        self._role_assignment_prior_weights = np.asarray(
            prior_weights, dtype=float)
        self._role_assignment_geometry_prior_weights = np.asarray(
            prior_weights, dtype=float).copy()
        self._role_assignment_diagnostics = {
            "status": "fit",
            "posterior": "finite_channel_role_assignment",
            "prior": self.meta_prior.observable_mean_role_assignment_prior,
            "assignment_count": int(len(assignments)),
            "channel_count": channel_count,
            "role_count": role_count,
            "assignments": [
                self._assignment_label(value) for value in assignments
            ],
            "assignment_prior_weights": (
                self._role_assignment_prior_weights.tolist()),
            "assignment_prior_diagnostics": copy.deepcopy(
                prior_diagnostics),
            "permutation_equivariant": True,
            "active_feature_dim_per_atom": int(
                self._role_assignment_feature_dim),
            "total_stored_feature_dim": int(
                len(assignments) * self._role_assignment_feature_dim),
            "target_labels_used_to_define_assignments": False,
            "target_oracle_used_to_define_assignments": False,
        }

    def calibrate_role_assignment_boundary_posterior(
        self,
        samples,
        targets,
        observation_variances,
    ):
        """Update finite role mass from charged pilot chance margins."""

        if self.meta_prior.observable_mean_role_assignment_prior != (
            "source_geometry_boundary"
        ):
            return copy.deepcopy(self._role_assignment_diagnostics)
        if not self._role_assignments:
            raise RuntimeError(
                "boundary role posterior requires an assignment bank")
        aligner = self.meta_prior.observable_channel_role_aligner
        posterior, calibration = (
            aligner.target_boundary_assignment_posterior(
                self.problem,
                self._role_assignments,
                samples,
                targets,
                observation_variances,
                geometry_prior_weights=(
                    self._role_assignment_geometry_prior_weights),
            )
        )
        self._role_assignment_prior_weights = np.asarray(
            posterior, dtype=float)
        self._role_assignment_diagnostics.update({
            "assignment_prior_weights": (
                self._role_assignment_prior_weights.tolist()),
            "boundary_calibration": copy.deepcopy(calibration),
            "target_labels_used_to_define_assignments": bool(
                calibration.get("target_labels_used", False)),
            "target_oracle_used_to_define_assignments": False,
            "assignment_update_scope": (
                "charged_pilot_boundary_summary_then_frozen"),
        })
        return copy.deepcopy(self._role_assignment_diagnostics)

    def _target_policy_pool(self):
        requested = int(
            self.meta_prior.observable_mean_target_residual_pool_size)
        aligner = self.meta_prior.observable_channel_role_aligner
        if aligner is not None:
            points = list(aligner.target_policy_pool(self.problem))
            source = "deterministic_unlabeled_role_matching_pool"
        else:
            d = max(int(getattr(self.problem, "d", 1)), 1)
            positions = (np.arange(d, dtype=float) + 0.5) / float(d)
            profiles = [
                np.full(d, float(level), dtype=float)
                for level in np.linspace(0.05, 0.95, 12)
            ]
            profiles.extend([
                np.linspace(left, right, d)
                for left, right in (
                    (0.15, 0.85), (0.30, 0.70), (0.85, 0.15), (0.70, 0.30)
                )
            ])
            rng = np.random.default_rng(
                self.meta_prior.seed + 1_300_021 + 31 * d)
            while len(profiles) < requested:
                row = np.full(d, float(rng.uniform(0.1, 0.9)), dtype=float)
                for frequency in range(1, 5):
                    row += float(rng.normal(0.0, 0.16 / frequency)) * np.cos(
                        np.pi * frequency * positions)
                profiles.append(np.clip(row, 0.0, 1.0))
            points = [
                tuple(int(value) for value in self.problem.continuous_to_int(row))
                for row in profiles
            ]
            source = "deterministic_unlabeled_low_frequency_pool"
        points = unique_candidates(points)
        return points[:requested], source

    def _ordered_observable_descriptors(self, points):
        rows = []
        for point in points:
            exposure = get_observable_state_exposure(self.problem, point)
            if exposure is None:
                raise ValueError(
                    "target residual mean coordinate requires observable exposure")
            rows.append(canonical_observable_state_descriptor(
                exposure, mode="ordered"))
        if not rows:
            return np.empty((0, 0), dtype=float)
        values = np.vstack(rows)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                "target residual observable descriptors are non-finite")
        return values

    @staticmethod
    def _orient_components(components):
        oriented = np.asarray(components, dtype=float).copy()
        for index, row in enumerate(oriented):
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0.0:
                oriented[index] *= -1.0
        return oriented

    def _fit_target_residual_coordinate(self):
        requested_rank = int(
            self.meta_prior.observable_mean_target_residual_rank)
        if requested_rank <= 0:
            return
        points, pool_source = self._target_policy_pool()
        if len(points) < 4:
            self._target_residual_diagnostics.update({
                "status": "insufficient_unlabeled_pool",
                "pool_size": int(len(points)),
                "pool_source": pool_source,
            })
            return
        base = np.asarray(
            self.meta_prior.observable_mean_model.features_many(
                self.problem, points),
            dtype=float,
        )
        if base.shape != (len(points), self.base_feature_dim):
            raise RuntimeError("base observable mean feature dimension changed")
        base_design = np.column_stack([np.ones(len(base)), base])
        raw = self._ordered_observable_descriptors(points)
        raw_mean = np.mean(raw, axis=0)
        raw_scale = np.std(raw, axis=0)
        raw_scale = np.where(raw_scale > 1e-10, raw_scale, 1.0)
        standardized = (raw - raw_mean) / raw_scale
        rcond = float(self.meta_prior.observable_mean_target_residual_rcond)
        projection = np.linalg.pinv(base_design, rcond=rcond) @ standardized
        residual = standardized - base_design @ projection
        _, singular_values, right = np.linalg.svd(
            residual, full_matrices=False)
        tolerance = max(
            rcond * float(singular_values[0]) if len(singular_values) else 0.0,
            1e-10,
        )
        available_rank = int(np.sum(singular_values > tolerance))
        rank = min(requested_rank, available_rank)
        if rank <= 0:
            self._target_residual_diagnostics.update({
                "status": "empty_orthogonal_complement",
                "pool_size": int(len(points)),
                "pool_source": pool_source,
                "raw_descriptor_dim": int(raw.shape[1]),
                "base_design_rank": int(np.linalg.matrix_rank(base_design)),
            })
            return
        components = self._orient_components(right[:rank])
        scores = residual @ components.T
        score_scale = np.sqrt(np.mean(scores ** 2, axis=0))
        score_scale = np.maximum(score_scale, 1e-10)
        normalized = scores / score_scale
        cross = base_design.T @ normalized / float(len(base_design))
        total_energy = float(np.sum(singular_values ** 2))
        retained_energy = float(np.sum(singular_values[:rank] ** 2))
        self._target_residual_raw_mean = raw_mean
        self._target_residual_raw_scale = raw_scale
        self._target_residual_projection = projection
        self._target_residual_components = components
        self._target_residual_score_scale = score_scale
        self._target_residual_points = list(points)
        self._target_residual_diagnostics = {
            "status": "fit",
            "requested_rank": requested_rank,
            "effective_rank": int(rank),
            "pool_size": int(len(points)),
            "pool_source": pool_source,
            "raw_descriptor": "canonical_ordered_observable_state",
            "raw_descriptor_dim": int(raw.shape[1]),
            "base_feature_dim": int(self.base_feature_dim),
            "base_design_rank": int(np.linalg.matrix_rank(base_design)),
            "orthogonal_complement_rank": available_rank,
            "maximum_base_cross_moment": float(np.max(np.abs(cross))),
            "retained_residual_energy_fraction": float(
                retained_energy / max(total_energy, 1e-12)),
            "coefficient_prior_mean": "zero",
            "coefficient_prior_scale_source": (
                "source_deviation_and_coefficient_covariance"),
            "target_labels_used_to_define_coordinate": False,
            "target_oracle_used_to_define_coordinate": False,
            "target_labels_used": False,
            "target_oracle_used": False,
        }

    def _base_features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.base_feature_dim), dtype=float)
        if self._role_assignments:
            blocks = []
            role_count = int(
                self.meta_prior.observable_channel_role_aligner.n_roles)
            for assignment in self._role_assignments:
                rows = []
                for point in X:
                    exposure = get_observable_state_exposure(
                        self.problem, point)
                    if exposure is None:
                        raise RuntimeError(
                            "role-assignment feature has no observable exposure")
                    descriptor = role_aligned_observable_state_descriptor(
                        exposure,
                        assignment,
                        n_roles=role_count,
                    )
                    rows.append(
                        self._role_assignment_model.features_profile(descriptor))
                block = np.asarray(rows, dtype=float)
                if block.shape != (
                    len(X), self._role_assignment_feature_dim
                ):
                    raise RuntimeError(
                        "role-assignment feature dimension changed")
                blocks.append(block)
            values = np.hstack(blocks)
            if not np.all(np.isfinite(values)):
                raise FloatingPointError(
                    "role-assignment mean features are non-finite")
            return values
        return np.asarray(
            self.meta_prior.observable_mean_model.features_many(
                self.problem, X),
            dtype=float,
        )

    def _target_residual_features_many(self, X, base=None):
        if self.target_residual_rank <= 0:
            return np.empty((len(X), 0), dtype=float)
        base = self._base_features_many(X) if base is None else np.asarray(base)
        raw = self._ordered_observable_descriptors(X)
        standardized = (
            raw - self._target_residual_raw_mean
        ) / self._target_residual_raw_scale
        design = np.column_stack([np.ones(len(base)), base])
        residual = standardized - design @ self._target_residual_projection
        values = (
            residual @ self._target_residual_components.T
        ) / self._target_residual_score_scale
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("target residual mean features are non-finite")
        return values

    def features(self, x):
        values = self.features_many([x])[0]
        if len(values) != self.feature_dim:
            raise RuntimeError("observable mean coordinate changed dimension")
        return values

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        base = self._base_features_many(X)
        residual = self._target_residual_features_many(X, base=base)
        return np.hstack([base, residual])

    def diagnostics(self):
        aligned = self.meta_prior.observable_mean_mode == "boundary_aligned"
        input_mode = str(self.meta_prior.observable_mean_input_mode)
        model = self.meta_prior.observable_mean_model
        model_diagnostics = (
            model.diagnostics_for_problem(self.problem)
            if hasattr(model, "diagnostics_for_problem")
            else model.diagnostics()
        )
        return {
            **model_diagnostics,
            "selected_basis": (
                "source_aligned_chance_boundary_phi"
                if aligned else "source_observable_constraint_mean_eta"
            ),
            "mean_coordinate": "phi" if aligned else "eta",
            "mean_coordinate_input": input_mode,
            "observable_descriptor_mode": (
                self.meta_prior.observable_mean_descriptor_mode),
            "boundary_feature_mode": (
                self.meta_prior.observable_mean_feature_mode),
            "latent_transform": (
                self.meta_prior.observable_mean_latent_transform),
            "target_orthogonal_residual": copy.deepcopy(
                self._target_residual_diagnostics),
            "target_residual_rank": int(self.target_residual_rank),
            "target_residual_prior_scale": float(
                self.meta_prior.observable_mean_target_residual_prior_scale),
            "role_assignment_posterior": copy.deepcopy(
                self._role_assignment_diagnostics),
            "variance_coordinate": (
                "psi_v=h_v(observable_state_exposure)"
                if self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"
                else "expert_specific_psi=(A,N)"
            ),
            "shared_observable_exposure_input": bool(
                input_mode == "observable_state_exposure"
                and self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"
            ),
            "separate_mean_variance_heads": True,
            "target_labels_used_to_define_coordinate": bool(
                self._role_assignment_diagnostics.get(
                    "target_labels_used_to_define_assignments", False)),
            "target_oracle_used_to_define_coordinate": False,
        }

    def posterior_coefficient_diagnostics(self, mean, covariance):
        """Audit target-learned channel roles in the exchangeable head."""

        if (
            self.meta_prior.observable_mean_descriptor_mode
            != "exchangeable_equivariant"
        ):
            return None
        model = self.meta_prior.observable_mean_model
        block_dim = int(model.channel_block_dim)
        channel_count = int(model._target_channel_count(self.problem))
        expected = 1 + int(self.feature_dim)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        covariance = np.asarray(covariance, dtype=float)
        if len(mean) < expected or covariance.shape[0] < expected:
            raise RuntimeError(
                "exchangeable posterior coefficient dimension changed")
        mean = mean[:expected]
        covariance = covariance[:expected, :expected]

        def blocks(values):
            return np.vstack([
                values[
                    1 + channel * block_dim:
                    1 + (channel + 1) * block_dim
                ]
                for channel in range(channel_count)
            ])

        posterior_blocks = blocks(mean)
        source = self.source_parametric_prior()
        source_blocks = blocks(np.asarray(source["mean"], dtype=float))

        def maximum_distance(values):
            if len(values) < 2:
                return 0.0
            return float(max(
                np.linalg.norm(values[left] - values[right])
                for left in range(len(values))
                for right in range(left + 1, len(values))
            ))

        traces = []
        for channel in range(channel_count):
            indices = np.arange(
                1 + channel * block_dim,
                1 + (channel + 1) * block_dim,
            )
            traces.append(float(np.trace(
                covariance[np.ix_(indices, indices)])))
        prior_spread = maximum_distance(source_blocks)
        posterior_spread = maximum_distance(posterior_blocks)
        return {
            "status": "audited",
            "posterior": "target_linear_channel_coefficients",
            "channel_count": channel_count,
            "channel_block_dim": block_dim,
            "source_channel_block_means": source_blocks.tolist(),
            "posterior_channel_block_means": posterior_blocks.tolist(),
            "posterior_channel_block_covariance_traces": traces,
            "source_channel_block_maximum_distance": prior_spread,
            "posterior_channel_block_maximum_distance": posterior_spread,
            "source_prior_exchangeable": bool(prior_spread <= 1e-10),
            "target_channel_roles_differentiated": bool(
                posterior_spread > max(1e-8, 1e-6 * np.linalg.norm(mean))),
            "source_role_identity_transferred": False,
            "target_coefficients_updated_by_gpr_posterior": True,
            "target_oracle_used": False,
        }

    def _extend_source_prior(self, prior):
        prior = copy.deepcopy(prior)
        if self.target_residual_rank <= 0:
            return prior
        mean = np.asarray(prior["mean"], dtype=float).reshape(-1)
        covariance = np.asarray(prior["covariance"], dtype=float)
        expected_base = 1 + self.base_feature_dim
        if len(mean) != expected_base or covariance.shape != (
            expected_base, expected_base
        ):
            raise RuntimeError(
                "base observable source prior changed coefficient dimension")
        finite_diagonal = np.diag(covariance)
        finite_diagonal = finite_diagonal[
            np.isfinite(finite_diagonal) & (finite_diagonal > 0.0)]
        deviation = max(float(prior.get("deviation_variance", 0.0)), 1e-12)
        coefficient_reference = (
            float(np.median(finite_diagonal))
            if len(finite_diagonal) else 0.0)
        residual_variance = max(
            deviation,
            coefficient_reference,
            1e-10,
        ) * float(self.meta_prior.observable_mean_target_residual_prior_scale)
        extended_mean = np.concatenate([
            mean, np.zeros(self.target_residual_rank, dtype=float)])
        extended_covariance = np.zeros(
            (len(extended_mean), len(extended_mean)), dtype=float)
        extended_covariance[:expected_base, :expected_base] = covariance
        extended_covariance[expected_base:, expected_base:] = (
            residual_variance
            * np.eye(self.target_residual_rank, dtype=float))
        prior["mean"] = extended_mean
        prior["covariance"] = extended_covariance
        diagnostics = dict(prior.get("diagnostics", {}))
        diagnostics.update({
            "target_orthogonal_residual_rank": int(self.target_residual_rank),
            "target_orthogonal_residual_prior_mean": 0.0,
            "target_orthogonal_residual_prior_variance": float(
                residual_variance),
            "target_orthogonal_residual_prior_scale": float(
                self.meta_prior.observable_mean_target_residual_prior_scale),
            "target_orthogonal_residual_prior_reference_deviation": float(
                deviation),
            "target_orthogonal_residual_prior_reference_coefficient": float(
                coefficient_reference),
            "target_orthogonal_residual_cross_covariance": 0.0,
            "target_labels_used_to_define_residual_prior": False,
            "target_oracle_used_to_define_residual_prior": False,
        })
        prior["diagnostics"] = diagnostics
        return prior

    def _expand_role_assignment_component(self, component):
        component = copy.deepcopy(component)
        if not self._role_assignments:
            return [component]
        mean = np.asarray(component["mean"], dtype=float).reshape(-1)
        covariance = np.asarray(component["covariance"], dtype=float)
        block_dim = int(self._role_assignment_feature_dim)
        if len(mean) != 1 + block_dim or covariance.shape != (
            1 + block_dim, 1 + block_dim
        ):
            raise RuntimeError(
                "role source prior does not match one assignment block")
        inactive = float(
            self.meta_prior.observable_mean_role_assignment_inactive_variance)
        total_dim = 1 + self.base_feature_dim
        original_weight = max(
            float(component.get("prior_weight", 0.0)), 0.0)
        expanded = []
        for assignment_index, (assignment, assignment_mass) in enumerate(zip(
            self._role_assignments, self._role_assignment_prior_weights
        )):
            assignment_mass = float(assignment_mass)
            start = 1 + assignment_index * block_dim
            stop = start + block_dim
            atom_mean = np.zeros(total_dim, dtype=float)
            atom_mean[0] = float(mean[0])
            atom_mean[start:stop] = mean[1:]
            atom_covariance = inactive * np.eye(total_dim, dtype=float)
            atom_covariance[0, 0] = float(covariance[0, 0])
            atom_covariance[start:stop, start:stop] = covariance[1:, 1:]
            atom_covariance[0, start:stop] = covariance[0, 1:]
            atom_covariance[start:stop, 0] = covariance[1:, 0]
            atom = copy.deepcopy(component)
            atom["mean"] = atom_mean
            atom["covariance"] = 0.5 * (
                atom_covariance + atom_covariance.T)
            atom["prior_weight"] = float(
                original_weight * assignment_mass)
            label = self._assignment_label(assignment)
            atom["name"] = (
                f"{component.get('name', 'source')}"
                f"|role_assignment={label}")
            diagnostics = dict(atom.get("diagnostics", {}))
            diagnostics.update({
                "role_assignment_structure_posterior": True,
                "role_assignment": label,
                "role_assignment_index": int(assignment_index),
                "role_assignment_count": int(len(self._role_assignments)),
                "role_assignment_prior_mass": float(assignment_mass),
                "role_assignment_active_coefficients": [
                    0, *range(start, stop)
                ],
                "role_assignment_inactive_variance": inactive,
                "role_assignment_permutation_equivariant": True,
                "target_labels_used_to_define_role_assignment": bool(
                    self._role_assignment_diagnostics.get(
                        "target_labels_used_to_define_assignments", False)),
                "target_oracle_used_to_define_role_assignment": False,
            })
            atom["diagnostics"] = diagnostics
            expanded.append(atom)
        return expanded

    @staticmethod
    def _moment_match_components(components, diagnostics):
        weights = np.asarray([
            max(float(component.get("prior_weight", 0.0)), 0.0)
            for component in components
        ], dtype=float)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(components), dtype=float)
        weights /= float(np.sum(weights))
        means = np.vstack([
            np.asarray(component["mean"], dtype=float).reshape(-1)
            for component in components
        ])
        mean = np.sum(weights[:, None] * means, axis=0)
        covariance = np.zeros((len(mean), len(mean)), dtype=float)
        for weight, component, component_mean in zip(
            weights, components, means
        ):
            delta = component_mean - mean
            covariance += float(weight) * (
                np.asarray(component["covariance"], dtype=float)
                + np.outer(delta, delta)
            )
        covariance = 0.5 * (covariance + covariance.T)
        deviation = float(np.sum(weights * np.asarray([
            max(float(component.get("deviation_variance", 1e-12)), 1e-12)
            for component in components
        ], dtype=float)))
        result = copy.deepcopy(diagnostics)
        result.update({
            "mean": mean,
            "covariance": covariance,
            "deviation_variance": deviation,
            "prior_weight": 1.0,
        })
        result_diagnostics = dict(result.get("diagnostics", {}))
        target_labels_used = any(bool(
            dict(component.get("diagnostics", {})).get(
                "target_labels_used_to_define_role_assignment", False)
        ) for component in components)
        result_diagnostics.update({
            "role_assignment_structure_posterior": True,
            "role_assignment_component_count": int(len(components)),
            "role_assignment_moment_matched_aggregate": True,
            "target_labels_used_to_define_role_assignment": bool(
                target_labels_used),
            "target_oracle_used_to_define_role_assignment": False,
        })
        result["diagnostics"] = result_diagnostics
        return result

    def source_parametric_prior(self):
        """Return the source-only coefficient law for target conditioning."""
        if self._role_assignments:
            base = copy.deepcopy(
                self._role_assignment_model.source_parametric_prior(
                    self.problem))
            components = []
            for component in (
                self._role_assignment_model
                .source_parametric_prior_components(self.problem)
            ):
                components.extend(
                    self._expand_role_assignment_component(component))
            prior = self._moment_match_components(components, base)
        else:
            prior = self._extend_source_prior(
                self.meta_prior.observable_mean_model.source_parametric_prior(
                    self.problem))
        expected = 1 + self.feature_dim
        if len(np.asarray(prior["mean"]).reshape(-1)) != expected:
            raise RuntimeError("observable source prior changed coefficient dimension")
        return prior

    def source_parametric_prior_components(self):
        """Return source-domain coefficient laws before target reweighting."""
        if self._role_assignments:
            components = []
            for component in (
                self._role_assignment_model
                .source_parametric_prior_components(self.problem)
            ):
                components.extend(
                    self._expand_role_assignment_component(component))
        else:
            components = [
                self._extend_source_prior(component)
                for component in (
                    self.meta_prior.observable_mean_model
                    .source_parametric_prior_components(self.problem)
                )
            ]
        expected = 1 + self.feature_dim
        for component in components:
            mean = np.asarray(component["mean"], dtype=float).reshape(-1)
            covariance = np.asarray(component["covariance"], dtype=float)
            if len(mean) != expected or covariance.shape != (
                expected, expected
            ):
                raise RuntimeError(
                    "observable source component changed coefficient dimension"
                )
        return components

    def expand_target_role_assignment_components(self, components):
        """Give the target-null law the same finite assignment posterior."""

        if not self._role_assignments:
            return [copy.deepcopy(component) for component in components]
        block_dim = int(self._role_assignment_feature_dim)
        inactive = float(
            self.meta_prior.observable_mean_role_assignment_inactive_variance)
        total_dim = 1 + self.base_feature_dim
        expanded = []
        for component in components:
            name = str(component.get("name", "component"))
            if "|role_assignment=" in name:
                expanded.append(copy.deepcopy(component))
                continue
            mean = np.asarray(component["mean"], dtype=float).reshape(-1)
            covariance = np.asarray(component["covariance"], dtype=float)
            if len(mean) != total_dim or covariance.shape != (
                total_dim, total_dim
            ):
                raise RuntimeError(
                    "unstructured role-assignment component changed dimension")
            original_weight = max(
                float(component.get("prior_weight", 0.0)), 0.0)
            for assignment_index, (assignment, assignment_mass) in enumerate(
                zip(
                    self._role_assignments,
                    self._role_assignment_prior_weights,
                )
            ):
                assignment_mass = float(assignment_mass)
                start = 1 + assignment_index * block_dim
                stop = start + block_dim
                atom_mean = np.zeros(total_dim, dtype=float)
                atom_mean[0] = float(mean[0])
                atom_mean[start:stop] = mean[start:stop]
                atom_covariance = inactive * np.eye(total_dim, dtype=float)
                indices = np.asarray([0, *range(start, stop)], dtype=int)
                atom_covariance[np.ix_(indices, indices)] = covariance[
                    np.ix_(indices, indices)]
                atom = copy.deepcopy(component)
                atom["mean"] = atom_mean
                atom["covariance"] = 0.5 * (
                    atom_covariance + atom_covariance.T)
                atom["prior_weight"] = float(
                    original_weight * assignment_mass)
                label = self._assignment_label(assignment)
                atom["name"] = f"{name}|role_assignment={label}"
                diagnostics = dict(atom.get("diagnostics", {}))
                diagnostics.update({
                    "role_assignment_structure_posterior": True,
                    "role_assignment": label,
                    "role_assignment_index": int(assignment_index),
                    "role_assignment_count": int(len(self._role_assignments)),
                    "role_assignment_prior_mass": float(assignment_mass),
                    "role_assignment_active_coefficients": indices.tolist(),
                    "role_assignment_inactive_variance": inactive,
                    "role_assignment_permutation_equivariant": True,
                    "target_labels_used_to_define_role_assignment": bool(
                        self._role_assignment_diagnostics.get(
                            "target_labels_used_to_define_assignments",
                            False)),
                    "target_oracle_used_to_define_role_assignment": False,
                })
                atom["diagnostics"] = diagnostics
                expanded.append(atom)
        return expanded

    def role_assignment_oracle_expressivity_audit(self, points, targets):
        """Post-run oracle audit of the registered assignment function family."""

        if not self._role_assignments:
            return {"status": "disabled"}
        values = self._base_features_many(points)
        target = np.asarray(targets, dtype=float).reshape(-1)
        if len(values) != len(target):
            raise ValueError("role-assignment audit rows must align")

        def rank_correlation(left, right):
            left = np.asarray(left, dtype=float)
            right = np.asarray(right, dtype=float)
            if len(left) < 2:
                return None
            left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable")
            right_rank = np.argsort(
                np.argsort(right, kind="stable"), kind="stable")
            left_rank = left_rank.astype(float) - float(np.mean(left_rank))
            right_rank = right_rank.astype(float) - float(np.mean(right_rank))
            denominator = float(
                np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
            return (
                None if denominator <= 1e-12
                else float(left_rank @ right_rank / denominator)
            )

        rows = []
        block_dim = int(self._role_assignment_feature_dim)
        for index, assignment in enumerate(self._role_assignments):
            block = values[:, index * block_dim:(index + 1) * block_dim]
            design = np.column_stack([np.ones(len(block)), block])
            gram = design.T @ design + 1e-6 * np.eye(
                design.shape[1], dtype=float)
            coefficient = np.linalg.solve(gram, design.T @ target)
            prediction = design @ coefficient
            rows.append({
                "assignment": self._assignment_label(assignment),
                "median_abs_error": float(np.median(
                    np.abs(prediction - target))),
                "rank_correlation": rank_correlation(prediction, target),
            })
        best_mae = min(rows, key=lambda row: row["median_abs_error"])
        finite_rank = [
            row for row in rows if row["rank_correlation"] is not None
        ]
        best_rank = max(
            finite_rank, key=lambda row: row["rank_correlation"]
        ) if finite_rank else None
        return {
            "status": "audited",
            "assignment_count": int(len(rows)),
            "best_median_abs_error": float(best_mae["median_abs_error"]),
            "best_mae_assignment": str(best_mae["assignment"]),
            "best_rank_correlation": (
                None if best_rank is None
                else float(best_rank["rank_correlation"])),
            "best_rank_assignment": (
                None if best_rank is None else str(best_rank["assignment"])),
            "assignments": rows,
            "post_run_only": True,
            "target_oracle_used": True,
            "target_oracle_used_for_decision": False,
        }

    def expand_target_residual_rank_components(
        self,
        components,
        rank_prior,
        *,
        inactive_variance=1e-12,
    ):
        """Expand source laws over a nested residual-rank latent variable.

        All atoms share the same maximum-rank feature map.  Rank ``k`` keeps
        the first ``k`` orthogonal residual coefficients active and pins the
        remaining coefficients to zero up to numerical variance.  Target
        observations may therefore update rank mass by ordinary Gaussian
        marginal likelihood without changing coordinates or consulting an
        oracle.
        """

        values = [copy.deepcopy(component) for component in components]
        rank = int(self.target_residual_rank)
        if rank <= 0:
            return values
        probability = np.asarray(rank_prior, dtype=float).reshape(-1)
        if len(probability) != rank + 1:
            raise ValueError(
                "target residual rank prior must contain one mass for each "
                f"rank 0..{rank}")
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
            raise ValueError(
                "target residual rank prior must be finite and nonnegative")
        if float(np.sum(probability)) <= 0.0:
            raise ValueError("target residual rank prior needs positive mass")
        probability /= float(np.sum(probability))
        inactive_variance = max(float(inactive_variance), 1e-12)
        base = 1 + int(self.base_feature_dim)
        expanded = []
        for component in values:
            covariance = np.asarray(
                component["covariance"], dtype=float).copy()
            if covariance.shape != (base + rank, base + rank):
                raise RuntimeError(
                    "target residual source component dimension changed")
            residual_diagonal = np.maximum(
                np.diag(covariance)[base:], inactive_variance)
            original_weight = max(
                float(component.get("prior_weight", 0.0)), 0.0)
            for active_rank, mass in enumerate(probability):
                atom = copy.deepcopy(component)
                atom_covariance = covariance.copy()
                for index in range(rank):
                    coefficient = base + index
                    atom_covariance[coefficient, :] = 0.0
                    atom_covariance[:, coefficient] = 0.0
                    atom_covariance[coefficient, coefficient] = (
                        float(residual_diagonal[index])
                        if index < active_rank else inactive_variance)
                atom["covariance"] = atom_covariance
                atom["prior_weight"] = float(original_weight * mass)
                atom["name"] = (
                    f"{component.get('name', 'source')}"
                    f"|target_residual_rank={active_rank}")
                diagnostics = dict(atom.get("diagnostics", {}))
                diagnostics.update({
                    "target_residual_structure_posterior": True,
                    "target_residual_structure_rank": int(active_rank),
                    "target_residual_structure_max_rank": int(rank),
                    "target_residual_structure_prior_mass": float(mass),
                    "target_residual_structure_inactive_variance": float(
                        inactive_variance),
                    "target_residual_structure_nested": True,
                    "target_labels_used_to_define_structure_prior": False,
                    "target_oracle_used_to_define_structure_prior": False,
                })
                atom["diagnostics"] = diagnostics
                expanded.append(atom)
        return expanded

    def source_target_epistemic_calibration(self):
        """Calibrate role-transfer mass from source/target signature support."""

        if self.meta_prior.observable_mean_descriptor_mode not in {
            "role_aligned",
            "role_transport",
            "role_intervention_transport",
            "role_adaptive_ordered",
            "role_adaptive_set_invariant",
        }:
            return {
                "status": "not_role_aligned",
                "source_role_trust": 1.0,
                "target_labels_used": False,
                "target_oracle_used": False,
            }
        aligner = self.meta_prior.observable_channel_role_aligner
        if aligner is None:
            raise RuntimeError("role-aligned mean basis has no channel aligner")
        return aligner.target_epistemic_calibration(self.problem)

    def source_target_coordinate_selection(self):
        """Return the outcome-free role/fallback identifiability decision."""

        model = self.meta_prior.observable_mean_model
        if not hasattr(model, "selection"):
            raise RuntimeError(
                "support-adaptive coefficient transfer requires a "
                "support-adaptive observable mean coordinate")
        selection = dict(model.selection(self.problem))
        if (
            selection.get("target_labels_used")
            or selection.get("target_oracle_used")
            or selection.get("selection_uses_target_labels")
            or selection.get("selection_uses_target_oracle")
        ):
            raise RuntimeError(
                "coordinate support selection must be outcome-free")
        return selection

    def target_null_feature_geometry(self):
        """Return an unlabeled target design for function-space calibration.

        The same deterministic policy pool used to match observable channel
        roles defines the geometry.  It contains no target response, simulator
        noise realization, feasibility label, or post-run oracle quantity.
        """

        aligner = self.meta_prior.observable_channel_role_aligner
        if aligner is None:
            return {
                "status": "unavailable",
                "target_labels_used": False,
                "target_oracle_used": False,
            }
        points = aligner.target_policy_pool(self.problem)
        features = self.features_many(points)
        if len(features) == 0 or not np.all(np.isfinite(features)):
            raise RuntimeError("target null feature geometry is unavailable")
        basis = np.column_stack([np.ones(len(features)), features])
        return {
            "status": "available",
            "basis_matrix": basis,
            "pool_size": int(len(basis)),
            "basis_dim": int(basis.shape[1]),
            "pool_source": "deterministic_unlabeled_role_matching_pool",
            "target_labels_used": False,
            "target_oracle_used": False,
        }


class PilotGatedMetaPriorBasis:
    """Choose a frozen source basis with a decision-aware pilot gate.

    The gate never reads target oracle values.  Stage 1 learns only the
    constraint/risk-boundary representation: it scores leave-one-out
    chance-margin calibration, ordering near the boundary, and false-feasible
    errors.  The objective keeps the Stage-0 coordinate basis until a separate
    transferable objective module is validated.  Plain LOO NMSE and objective
    spectral scores remain in diagnostics, but do not control Stage-1 behavior.
    """

    adaptive_meta_basis = True

    def __init__(self, meta_prior: LearnedMetaPrior, problem, output_index, ridge=1.0):
        self.meta_prior = meta_prior
        self.problem = problem
        self.output_index = int(output_index)
        self.ridge = float(ridge)
        self.identity_dim = min(
            max(1, int(meta_prior.spectral_active_dim)),
            meta_prior.local_dim + meta_prior.shared_dim,
        )
        descriptor_dim = len(meta_prior.feature_mean)
        psi_dim = meta_prior.local_dim + meta_prior.shared_dim
        cumulative_dim = (
            1
            + meta_prior.local_dim
            + meta_prior.shared_dim * (meta_prior.shared_dim + 1) // 2
            + meta_prior.shared_dim
        )
        frequency_dim = max(
            (
                int(entry["basis"].feature_dim)
                for entry in meta_prior.spectral_frequency_bank
            ),
            default=0,
        )
        aligned_frequency_dim = max(
            (
                int(entry["basis"].feature_dim)
                for entry in meta_prior.risk_aligned_frequency_bank
            ),
            default=0,
        )
        coordinate_dim = descriptor_dim + 2 * psi_dim + cumulative_dim - 1
        additive_dim = (
            max(
                coordinate_dim,
                int(meta_prior.stage1_spectral_basis.feature_dim),
            )
            + int(meta_prior.spectral_additive_bank.feature_dim)
            if meta_prior.spectral_additive_bank is not None
            else 0
        )
        aligned_additive_dim = (
            max(
                coordinate_dim,
                int(meta_prior.risk_aligned_spectral_basis.feature_dim),
            ) + int(meta_prior.risk_aligned_additive_bank.feature_dim)
            if meta_prior.risk_aligned_additive_bank is not None
            and meta_prior.risk_aligned_spectral_basis is not None
            else 0
        )
        aligned_dim = (
            int(meta_prior.risk_aligned_spectral_basis.feature_dim)
            if meta_prior.risk_aligned_spectral_basis is not None
            else 0
        )
        aligned_coordinate_dim = (
            int(meta_prior.risk_subspace_alignment.feature_dim)
            if meta_prior.risk_subspace_alignment is not None
            else 0
        )
        self.feature_dim = max(
            coordinate_dim,
            self.identity_dim,
            int(meta_prior.spectral_feature_dim),
            int(meta_prior.stage1_spectral_basis.feature_dim),
            frequency_dim,
            aligned_frequency_dim,
            additive_dim,
            aligned_additive_dim,
            aligned_dim,
            aligned_coordinate_dim,
        )
        self.selected_basis = "coordinate"
        self.selected_parametric_ridge = 0.0
        self.gate_diagnostics = {
            "status": "unfit",
            "selected_basis": self.selected_basis,
            "output_index": self.output_index,
        }
        self._adaptive_sparsity_diagnostics = {"status": "not_requested"}
        self._adaptive_gate_posterior_diagnostics = {"status": "not_fit"}
        self._adaptive_loo_support_diagnostics = {"status": "not_fit"}
        self._adaptive_allowed_mask = None
        self._frequency_gate_diagnostics = {"status": "not_requested"}
        self._last_frequency_refit_n = 0
        self._risk_alignment_gate_diagnostics = {"status": "not_requested"}
        self._target_risk_alignment = None
        self._nested_alignment_fold_cache = None
        self._last_alignment_refit_n = 0
        self._locked_alignment_stage1_basis = None
        self._locked_alignment_stage1_guard = None
        self._additive_gate_diagnostics = {"status": "not_requested"}
        self._additive_base_basis = "source_spectral"
        self._additive_bank_kind = "raw"
        self._locked_additive_base_basis = None
        self.selected_additive_groups = []
        self._last_additive_refit_n = 0
        self._additive_refit_count = 0

    def _variant_features(self, x, variant):
        return self._variant_features_with_alignment(
            x, variant, self._target_risk_alignment)

    def _variant_features_with_alignment(self, x, variant, alignment):
        if variant == "coordinate":
            return self.meta_prior.coordinate_basis_features(self.problem, x)
        if variant == "source_spectral":
            return self.meta_prior.stage1_spectral_features(self.problem, x)
        if variant == "adaptive_spectral":
            return self.meta_prior.spectral_features(self.problem, x)
        if variant.startswith("frequency_band_"):
            return self.meta_prior.spectral_frequency_features(
                self.problem,
                x,
                int(variant.rsplit("_", 1)[1]),
            )
        if variant.startswith("aligned_frequency_band_"):
            return self.meta_prior.risk_aligned_frequency_features(
                self.problem,
                x,
                int(variant.rsplit("_", 1)[1]),
                adapter=alignment,
            )
        if variant == "risk_aligned_spectral":
            return self.meta_prior.risk_aligned_spectral_features(
                self.problem,
                x,
                adapter=alignment,
            )
        if variant == "risk_aligned_coordinate":
            return self.meta_prior.risk_aligned_coordinate(
                self.problem,
                x,
                adapter=alignment,
            )
        if variant == "frozen_risk_aligned_coordinate":
            return self.meta_prior.frozen_risk_aligned_coordinate(
                self.problem, x)
        if variant == "source_additive":
            base = self._variant_features(x, self._additive_base_basis)
            bank = self._selected_additive_bank()
            extra = np.zeros(bank.feature_dim, dtype=float)
            if self.selected_additive_groups:
                extra[self.selected_additive_groups] = self._additive_features(
                    x, self.selected_additive_groups)
            return np.concatenate([base, extra])
        psi = self.meta_prior.risk_coordinate(self.problem, x)
        return np.asarray(psi[: self.identity_dim], dtype=float)

    @staticmethod
    def _uses_target_alignment(variant):
        return bool(
            str(variant).startswith("risk_aligned_")
            or str(variant).startswith("aligned_frequency_band_")
        )

    def _nested_alignment_loo_predictions(
        self,
        xs,
        observations,
        target,
        variant,
        ridge=None,
    ):
        """LOO predictions with the target alignment refit inside each fold.

        Fitting the alignment once on the full pilot and only cross-validating
        the downstream ridge leaks the held-out constraint through the learned
        feature direction.  This nested version keeps the complete
        representation-learning pipeline honest.
        """
        target = np.asarray(target, dtype=float).reshape(-1)
        predictions = []
        fold_diagnostics = []
        if self._nested_alignment_fold_cache is None:
            cache = []
            for heldout in range(len(xs)):
                train_indices = [i for i in range(len(xs)) if i != heldout]
                train_observations = {
                    xs[i]: observations[xs[i]] for i in train_indices
                }
                alignment = self.meta_prior.fit_target_risk_alignment(
                    self.problem, train_observations)
                cache.append((train_indices, alignment))
            self._nested_alignment_fold_cache = cache
        for heldout, (train_indices, alignment) in enumerate(
            self._nested_alignment_fold_cache
        ):
            train_features = np.vstack([
                self._variant_features_with_alignment(
                    xs[i], variant, alignment)
                for i in train_indices
            ])
            test_features = np.asarray(
                self._variant_features_with_alignment(
                    xs[heldout], variant, alignment),
                dtype=float,
            ).reshape(1, -1)
            prediction = self._ridge_predict(
                train_features,
                target[train_indices],
                test_features,
                ridge=ridge,
            )[0]
            predictions.append(float(prediction))
            fold_diagnostics.append({
                "heldout": int(heldout),
                "alignment_accepted": bool(
                    alignment is not None and alignment.accepted),
                "alignment_status": (
                    alignment.diagnostics.get("status", "unknown")
                    if alignment is not None else "missing"
                ),
                "boundary_axis_status": (
                    alignment.diagnostics.get("boundary_axis", {}).get(
                        "status", "missing")
                    if alignment is not None else "missing"
                ),
            })
        accepted_rate = float(np.mean([
            row["alignment_accepted"] for row in fold_diagnostics
        ])) if fold_diagnostics else 0.0
        return np.asarray(predictions, dtype=float), {
            "method": "nested_alignment_loo",
            "n_folds": int(len(fold_diagnostics)),
            "adapter_cache_reused_across_variants": True,
            "alignment_accepted_rate": accepted_rate,
            "folds": fold_diagnostics,
            "target_oracle_used": False,
        }

    def _nested_aligned_additive_loo_predictions(
        self,
        xs,
        observations,
        target,
        base_variant,
        groups,
    ):
        if self._nested_alignment_fold_cache is None:
            self._nested_alignment_loo_predictions(
                xs,
                observations,
                target,
                "risk_aligned_coordinate",
            )
        bank = self.meta_prior.risk_aligned_additive_bank
        predictions = []
        for heldout, (train_indices, alignment) in enumerate(
            self._nested_alignment_fold_cache
        ):
            def features(index):
                x = xs[index]
                base = np.asarray(
                    self._variant_features_with_alignment(
                        x, base_variant, alignment),
                    dtype=float,
                ).reshape(-1)
                coordinate = (
                    self.meta_prior.risk_aligned_full_coordinate_from_descriptor(
                        self.meta_prior.descriptor(self.problem, x),
                        adapter=alignment,
                    )
                )
                extra = bank.transform_groups(coordinate, groups)
                return np.concatenate([base, extra])

            train_features = np.vstack([
                features(index) for index in train_indices
            ])
            test_features = features(heldout).reshape(1, -1)
            predictions.append(float(self._ridge_predict(
                train_features,
                np.asarray(target, dtype=float)[train_indices],
                test_features,
                ridge=self.ridge,
            )[0]))
        return np.asarray(predictions, dtype=float)

    def _selected_additive_bank(self):
        if self._additive_bank_kind == "aligned":
            return self.meta_prior.risk_aligned_additive_bank
        return self.meta_prior.spectral_additive_bank

    def _additive_features(self, x, indices):
        if self._additive_bank_kind == "aligned":
            return self.meta_prior.risk_aligned_additive_features(
                self.problem,
                x,
                indices,
                adapter=self._target_risk_alignment,
            )
        return self.meta_prior.spectral_additive_features(
            self.problem, x, indices)

    def _additive_coordinate(self, x):
        if self._additive_bank_kind == "aligned":
            return self.meta_prior.risk_aligned_coordinate(
                self.problem, x, adapter=self._target_risk_alignment)
        return self.meta_prior.risk_coordinate(self.problem, x)

    def features(self, x):
        values = np.asarray(
            self._variant_features(x, self.selected_basis), dtype=float).reshape(-1)
        out = np.zeros(self.feature_dim, dtype=float)
        out[: min(len(values), self.feature_dim)] = values[: self.feature_dim]
        return out

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(x) for x in X])

    def fit_from_observations(self, observations, output_index=None):
        output_index = self.output_index if output_index is None else int(output_index)
        self._adaptive_allowed_mask = None
        self._nested_alignment_fold_cache = None
        self.selected_parametric_ridge = 0.0
        self._frequency_gate_diagnostics = {"status": "not_requested"}
        self._risk_alignment_gate_diagnostics = {"status": "not_requested"}
        self._additive_gate_diagnostics = {"status": "not_requested"}
        self.selected_additive_groups = []
        xs = list(observations)
        if len(xs) < 5:
            self.gate_diagnostics = {
                "status": "insufficient_pilot",
                "selected_basis": self.selected_basis,
                "output_index": output_index,
                "n_observations": int(len(xs)),
            }
            return self.selected_basis
        if (
            self.meta_prior.spectral_risk_alignment
            and self.meta_prior.spectral_alignment_admission
        ):
            self._target_risk_alignment = (
                self.meta_prior.fit_target_risk_alignment(
                    self.problem, observations)
            )
            self._risk_alignment_gate_diagnostics = dict(
                self._target_risk_alignment.diagnostics)
            self._last_alignment_refit_n = int(len(xs))
        self._last_additive_refit_n = int(len(xs))
        observed = np.vstack([
            np.mean(np.asarray(observations[x], dtype=float), axis=0)
            for x in xs
        ])
        target = np.asarray(observed[:, output_index], dtype=float)
        scores = {}
        nmse_scores = {}
        component_scores = {}
        predictions = {}
        decision_predictions = {}
        nested_alignment_diagnostics = {}
        base_variants = ["coordinate", "fixed_psi", "source_spectral"]
        if (
            output_index == 1
            and self.meta_prior.spectral_risk_alignment
            and self.meta_prior.spectral_alignment_admission
        ):
            base_variants.extend([
                "frozen_risk_aligned_coordinate",
                "risk_aligned_coordinate",
                "risk_aligned_spectral",
            ])
        for variant in base_variants:
            matrix = np.vstack([self._variant_features(x, variant) for x in xs])
            if self._uses_target_alignment(variant):
                prediction, nested_diag = (
                    self._nested_alignment_loo_predictions(
                        xs,
                        observations,
                        target,
                        variant,
                    )
                )
                nested_alignment_diagnostics[variant] = nested_diag
            else:
                prediction = self._ridge_loo_predictions(matrix, target)
            nmse_scores[variant] = self._normalized_mse(target, prediction)
            decision_prediction = np.asarray(prediction, dtype=float)
            if (
                output_index == 1
                and variant == "frozen_risk_aligned_coordinate"
            ):
                decision_prediction = (
                    decision_prediction + self._risk_alignment_source_guard())
            if output_index == 1:
                components = self._constraint_decision_score(
                    target, decision_prediction)
            else:
                components = self._objective_decision_score(
                    target,
                    prediction,
                    observed[:, 1],
                )
            component_scores[variant] = components
            scores[variant] = float(components["total"])
            predictions[variant] = prediction
            decision_predictions[variant] = decision_prediction
        # Reproduce the promoted Stage-1 gate first.  The expanded adaptive
        # dictionary is a challenger; rejecting it must return exactly this
        # coordinate-or-2D-spectral decision.
        eligible = ["coordinate"]
        if output_index == 1:
            eligible.append("source_spectral")
        tie_order = {
            "coordinate": 0,
            "source_spectral": 1,
        }
        selected = min(
            eligible, key=lambda name: (scores[name], tie_order[name]))
        baseline_score = float(scores["coordinate"])
        tolerance = self.meta_prior.spectral_gate_selection_tolerance * max(
            abs(baseline_score), 1.0)
        if (
            selected != "coordinate"
            and float(scores[selected]) >= baseline_score - tolerance
        ):
            selected = "coordinate"
        if output_index == 1 and selected != "coordinate":
            baseline_components = component_scores["coordinate"]
            selected_components = component_scores[selected]
            raw_false_feasible_worse = (
                selected_components["raw_false_feasible_rate"]
                > baseline_components["raw_false_feasible_rate"] + 0.05
            )
            dangerous_limit = max(
                1.25 * baseline_components["dangerous_underprediction"],
                baseline_components["dangerous_underprediction"] + 0.05,
            )
            if (
                raw_false_feasible_worse
                or selected_components["dangerous_underprediction"] > dangerous_limit
            ):
                selected = "coordinate"
        current_stage1_selected = selected
        if output_index == 1 and self.meta_prior.spectral_risk_alignment:
            if self._locked_alignment_stage1_basis is None:
                self._locked_alignment_stage1_basis = current_stage1_selected
                self._locked_alignment_stage1_guard = float(
                    component_scores[current_stage1_selected][
                        "calibration_guard"])
            stage1_selected = str(self._locked_alignment_stage1_basis)
            selected = stage1_selected
        else:
            stage1_selected = current_stage1_selected
        if (
            output_index == 1
            and self.meta_prior.spectral_risk_alignment
            and self.meta_prior.spectral_alignment_admission
        ):
            alignment_truth_margin = self._chance_margin(target)
            alignment_boundary_support = {
                "n_feasible": int(np.sum(alignment_truth_margin <= 0.0)),
                "n_infeasible": int(np.sum(alignment_truth_margin > 0.0)),
                "min_margin": float(np.min(alignment_truth_margin)),
                "max_margin": float(np.max(alignment_truth_margin)),
            }
            source_episode_prior = self.meta_prior.alignment_episode_prior
            source_episode_prior_fit = bool(
                source_episode_prior is not None
                and source_episode_prior.diagnostics().get("status") == "fit"
            )
            alignment_variants = (
                ("frozen_risk_aligned_coordinate",)
                if source_episode_prior_fit
                else ("risk_aligned_coordinate", "risk_aligned_spectral")
            )
            challenger = min(
                alignment_variants,
                key=lambda name: (scores[name], name),
            )
            alignment_reasons = []
            if (
                self._target_risk_alignment is None
                or not self._target_risk_alignment.accepted
            ) and challenger != "frozen_risk_aligned_coordinate":
                alignment_reasons.append("target_alignment_not_accepted")
            if (
                alignment_boundary_support["n_feasible"] == 0
                or alignment_boundary_support["n_infeasible"] == 0
            ):
                alignment_reasons.append("one_sided_boundary_support")
            current_components = component_scores[stage1_selected]
            challenger_components = component_scores[challenger]
            if scores[challenger] >= scores[stage1_selected] - tolerance:
                alignment_reasons.append("insufficient_decision_gain")
            if (
                challenger_components["raw_false_feasible_rate"]
                > current_components["raw_false_feasible_rate"] + 0.05
            ):
                alignment_reasons.append("raw_false_feasible_worse")
            dangerous_limit = max(
                1.25 * current_components["dangerous_underprediction"],
                current_components["dangerous_underprediction"] + 0.05,
            )
            if (
                challenger_components["dangerous_underprediction"]
                > dangerous_limit
            ):
                alignment_reasons.append("dangerous_underprediction")
            stability = self._subsample_decision_stability(
                target,
                decision_predictions[stage1_selected],
                decision_predictions[challenger],
            )
            if (
                stability["win_rate"] < (2.0 / 3.0)
                or stability["median_gain"] <= 0.0
            ):
                alignment_reasons.append("unstable_decision_gain")
            source_episode_admission = {"status": "not_requested"}
            if source_episode_prior_fit:
                source_decision = source_episode_prior.admit(
                    alignment_truth_margin,
                    self._chance_margin(decision_predictions[stage1_selected]),
                    self._chance_margin(decision_predictions[challenger]),
                )
                source_episode_admission = dict(source_decision.diagnostics)
                if source_decision.accepted:
                    # Source episodes provide the held-out gain evidence that
                    # a tiny target LOO estimate cannot.  They may replace
                    # gain/stability objections, but never target safety
                    # objections such as one-sided support, false feasibility,
                    # or dangerous underprediction.
                    alignment_reasons = [
                        reason for reason in alignment_reasons
                        if reason not in {
                            "insufficient_decision_gain",
                            "unstable_decision_gain",
                        }
                    ]
                    source_episode_admission[
                        "overrides_target_gain_only"
                    ] = True
                else:
                    alignment_reasons.append(
                        "source_episode_admission_rejected")
            if not alignment_reasons:
                selected = challenger
            self._risk_alignment_gate_diagnostics = {
                **self._risk_alignment_gate_diagnostics,
                "challenger_variant": challenger,
                "selected_variant": selected,
                "stage1_selected_basis": stage1_selected,
                "current_stage1_challenger": current_stage1_selected,
                "stage1_basis_locked": True,
                "rejection_reasons": alignment_reasons,
                "stability": stability,
                "boundary_support": alignment_boundary_support,
                "source_episode_admission": source_episode_admission,
                "nested_loo": nested_alignment_diagnostics,
                "target_data_used": True,
                "target_oracle_used": False,
            }
        if (
            output_index == 1
            and self.meta_prior.spectral_additive_adaptation
            and not self.meta_prior.spectral_frequency_adaptation
        ):
            if self._locked_additive_base_basis is None:
                self._locked_additive_base_basis = selected
            selected = self._locked_additive_base_basis
        spectral_score_model = (
            "boundary_aligned_projector_plus_target_gate"
            if (
                str(selected).startswith("risk_aligned_")
                or selected == "frozen_risk_aligned_coordinate"
            )
            else "stage1_dense_ridge"
        )
        frequency_variants = []
        if output_index == 1 and self.meta_prior.spectral_frequency_adaptation:
            self._last_frequency_refit_n = int(len(xs))
            frequency_baseline_selected = selected
            source_adjusted = {}
            entry_by_variant = {}
            frequency_entries = list(
                self.meta_prior.spectral_frequency_bank
            ) + list(self.meta_prior.risk_aligned_frequency_bank)
            for entry in frequency_entries:
                variant = entry["variant"]
                entry_by_variant[variant] = entry
                frequency_variants.append(variant)
                if entry["is_stage1_baseline"]:
                    base_variant = entry.get(
                        "base_variant", "source_spectral")
                    components = component_scores[base_variant]
                    raw_score = float(scores[base_variant])
                    nmse = float(nmse_scores[base_variant])
                else:
                    matrix = np.vstack([
                        self._variant_features(x, variant) for x in xs
                    ])
                    if self._uses_target_alignment(variant):
                        prediction, nested_diag = (
                            self._nested_alignment_loo_predictions(
                                xs,
                                observations,
                                target,
                                variant,
                                ridge=entry["ridge"],
                            )
                        )
                        self._risk_alignment_gate_diagnostics.setdefault(
                            "nested_loo", {})[variant] = nested_diag
                    else:
                        prediction = self._ridge_loo_predictions(
                            matrix,
                            target,
                            ridge=entry["ridge"],
                        )
                    components = self._constraint_decision_score(
                        target, prediction)
                    raw_score = float(components["total"])
                    nmse = self._normalized_mse(target, prediction)
                    component_scores[variant] = components
                    scores[variant] = raw_score
                    nmse_scores[variant] = float(nmse)
                    predictions[variant] = prediction
                source_adjusted[variant] = raw_score + (
                    self.meta_prior.spectral_frequency_source_penalty
                    * -np.log(max(float(entry["source_weight"]), 1e-12))
                )
            best_variant = min(
                frequency_variants,
                key=lambda name: (source_adjusted[name], name),
            )
            best_entry = entry_by_variant[best_variant]
            baseline_entry = next(
                entry for entry in frequency_entries
                if entry["is_stage1_baseline"]
                and entry.get("base_variant", "source_spectral")
                == best_entry.get("base_variant", "source_spectral")
            )
            baseline_variant = baseline_entry["variant"]
            frequency_reasons = []
            stage1_components = component_scores[frequency_baseline_selected]
            challenger_components = (
                component_scores[best_entry.get(
                    "base_variant", "source_spectral")]
                if best_entry["is_stage1_baseline"]
                else component_scores[best_variant]
            )
            challenger_score = (
                float(scores[best_entry.get(
                    "base_variant", "source_spectral")])
                if best_entry["is_stage1_baseline"]
                else float(scores[best_variant])
            )
            frequency_tolerance = (
                self.meta_prior.spectral_gate_selection_tolerance
                * max(abs(float(scores[frequency_baseline_selected])), 1.0)
            )
            if best_entry["is_stage1_baseline"]:
                frequency_reasons.append("stage1_is_best_band")
            if challenger_score >= (
                float(scores[frequency_baseline_selected]) - frequency_tolerance
            ):
                frequency_reasons.append("insufficient_decision_gain")
            if source_adjusted[best_variant] >= (
                source_adjusted[baseline_variant] - frequency_tolerance
            ):
                frequency_reasons.append("insufficient_source_adjusted_gain")
            if (
                challenger_components["raw_false_feasible_rate"]
                > stage1_components["raw_false_feasible_rate"] + 0.05
            ):
                frequency_reasons.append("raw_false_feasible_worse")
            dangerous_limit = max(
                1.25 * stage1_components["dangerous_underprediction"],
                stage1_components["dangerous_underprediction"] + 0.05,
            )
            if (
                challenger_components["dangerous_underprediction"]
                > dangerous_limit
            ):
                frequency_reasons.append("dangerous_underprediction")
            baseline_prediction = predictions[frequency_baseline_selected]
            challenger_prediction = (
                predictions[best_entry.get(
                    "base_variant", "source_spectral")]
                if best_entry["is_stage1_baseline"]
                else predictions[best_variant]
            )
            stability = self._subsample_decision_stability(
                target,
                baseline_prediction,
                challenger_prediction,
            )
            if (
                stability["win_rate"] < (2.0 / 3.0)
                or stability["median_gain"] <= 0.0
            ):
                frequency_reasons.append("unstable_decision_gain")
            truth_margin = self._chance_margin(target)
            boundary_support = {
                "n_feasible": int(np.sum(truth_margin <= 0.0)),
                "n_infeasible": int(np.sum(truth_margin > 0.0)),
                "min_margin": float(np.min(truth_margin)),
                "max_margin": float(np.max(truth_margin)),
            }
            # A target-selected pass band is not identifiable when the pilot
            # only sees one side of the chance boundary.  The frozen source
            # basis remains available as the exact Stage-1 fallback and this
            # gate is reconsidered after more target observations arrive.
            if (
                boundary_support["n_feasible"] == 0
                or boundary_support["n_infeasible"] == 0
            ):
                frequency_reasons.append("one_sided_boundary_support")
            family_entries = [
                entry for entry in frequency_entries
                if entry.get("base_variant", "source_spectral")
                == best_entry.get("base_variant", "source_spectral")
            ]
            minimum_source_weight = 0.9 / max(len(family_entries), 1)
            if float(best_entry["source_weight"]) < minimum_source_weight:
                frequency_reasons.append("weak_source_domain_support")
            if (
                self._uses_target_alignment(best_variant)
                and (
                    self._target_risk_alignment is None
                    or not self._target_risk_alignment.accepted
                )
            ):
                frequency_reasons.append("target_alignment_not_accepted")
            if not frequency_reasons:
                selected = best_variant
                self.selected_parametric_ridge = float(best_entry["ridge"])
                spectral_score_model = "source_band_hyperprior_plus_target_gate"
            self._frequency_gate_diagnostics = {
                "status": "selected" if not frequency_reasons else "fallback_stage1",
                "selected_variant": (
                    best_variant
                    if not frequency_reasons else frequency_baseline_selected
                ),
                "challenger_variant": best_variant,
                "stage1_selected_basis": stage1_selected,
                "frequency_baseline_basis": frequency_baseline_selected,
                "rejection_reasons": frequency_reasons,
                "source_adjusted_score": {
                    name: float(value) for name, value in source_adjusted.items()
                },
                "selected_cutoff": (
                    int(best_entry["cutoff"]) if not frequency_reasons else None
                ),
                "selected_ridge": (
                    float(best_entry["ridge"]) if not frequency_reasons else 0.0
                ),
                "selected_family": best_entry.get(
                    "base_variant", "source_spectral"),
                "source_weight": float(best_entry["source_weight"]),
                "minimum_source_weight": float(minimum_source_weight),
                "stability": stability,
                "boundary_support": boundary_support,
                "target_data_used": True,
                "target_oracle_used": False,
            }
        if output_index == 1 and self.meta_prior.spectral_additive_adaptation:
            self._additive_refit_count += 1
            if self._locked_additive_base_basis is None:
                self._locked_additive_base_basis = selected
            additive_baseline = self._locked_additive_base_basis
            self._additive_bank_kind = (
                "aligned"
                if additive_baseline == "risk_aligned_spectral"
                and self.meta_prior.risk_aligned_additive_bank is not None
                else "raw"
            )
            bank = self._selected_additive_bank()
            self._additive_base_basis = additive_baseline
            base_matrix = np.vstack([
                self._variant_features(x, additive_baseline) for x in xs
            ])
            selected_groups = []
            current_components = component_scores[additive_baseline]
            current_score = float(scores[additive_baseline])
            current_adjusted = current_score
            current_prediction = (
                predictions[additive_baseline]
                if self._additive_bank_kind == "aligned"
                and additive_baseline in predictions
                else self._ridge_loo_predictions(
                    base_matrix, target, ridge=self.ridge)
            )
            selection_trace = []
            maximum = min(
                int(self.meta_prior.spectral_additive_target_max_groups),
                int(bank.feature_dim),
            )
            for step in range(maximum):
                candidates = []
                for group_index in range(bank.feature_dim):
                    if group_index in selected_groups:
                        continue
                    trial_groups = selected_groups + [group_index]
                    extra = np.vstack([
                        self._additive_features(
                            x, trial_groups)
                        for x in xs
                    ])
                    matrix = np.hstack([base_matrix, extra])
                    prediction = (
                        self._nested_aligned_additive_loo_predictions(
                            xs,
                            observations,
                            target,
                            additive_baseline,
                            trial_groups,
                        )
                        if self._additive_bank_kind == "aligned"
                        else self._ridge_loo_predictions(
                            matrix, target, ridge=self.ridge)
                    )
                    components = self._constraint_decision_score(
                        target, prediction)
                    raw_score = float(components["total"])
                    prior_cost = self.meta_prior.spectral_additive_source_penalty * sum(
                        -np.log(max(bank.source_weight(index), 1e-12))
                        for index in trial_groups
                    )
                    complexity_cost = (
                        self.meta_prior.spectral_additive_complexity_penalty
                        * len(trial_groups)
                        * np.log(max(len(xs), 2))
                        / max(len(xs), 1)
                    )
                    saturation_fraction = float(np.mean([
                        bank.support_saturation(
                            self._additive_coordinate(x),
                            group_index,
                        )
                        for x in xs
                    ]))
                    adjusted = raw_score + prior_cost + complexity_cost
                    candidates.append({
                        "group_index": int(group_index),
                        "trial_groups": list(trial_groups),
                        "prediction": prediction,
                        "components": components,
                        "raw_score": raw_score,
                        "adjusted_score": float(adjusted),
                        "prior_cost": float(prior_cost),
                        "complexity_cost": float(complexity_cost),
                        "saturation_fraction": saturation_fraction,
                    })
                if not candidates:
                    break
                challenger = min(
                    candidates,
                    key=lambda item: (
                        item["adjusted_score"], item["group_index"]),
                )
                tolerance = (
                    self.meta_prior.spectral_gate_selection_tolerance
                    * max(abs(current_score), 1.0)
                )
                rejection_reasons = []
                if challenger["raw_score"] >= current_score - tolerance:
                    rejection_reasons.append("insufficient_decision_gain")
                if challenger["adjusted_score"] >= current_adjusted - tolerance:
                    rejection_reasons.append("insufficient_penalized_gain")
                if (
                    challenger["components"]["raw_false_feasible_rate"]
                    > current_components["raw_false_feasible_rate"] + 0.05
                ):
                    rejection_reasons.append("raw_false_feasible_worse")
                dangerous_limit = max(
                    1.25 * current_components["dangerous_underprediction"],
                    current_components["dangerous_underprediction"] + 0.05,
                )
                if (
                    challenger["components"]["dangerous_underprediction"]
                    > dangerous_limit
                ):
                    rejection_reasons.append("dangerous_underprediction")
                if (
                    challenger["saturation_fraction"]
                    > self.meta_prior.spectral_additive_max_saturation_fraction
                ):
                    rejection_reasons.append("outside_source_support")
                minimum_source_weight = 0.9 / max(int(bank.feature_dim), 1)
                if (
                    bank.source_weight(challenger["group_index"])
                    < minimum_source_weight
                ):
                    rejection_reasons.append("weak_source_domain_support")
                stability = self._subsample_decision_stability(
                    target,
                    current_prediction,
                    challenger["prediction"],
                )
                if (
                    stability["win_rate"] < (2.0 / 3.0)
                    or stability["median_gain"] <= 0.0
                ):
                    rejection_reasons.append("unstable_decision_gain")
                group_index = int(challenger["group_index"])
                selection_trace.append({
                    "step": int(step),
                    "group_index": group_index,
                    "group_name": bank.group_names_[group_index],
                    "function_name": bank.function_names_[group_index],
                    "source_weight": float(bank.source_weight(group_index)),
                    "minimum_source_weight": float(minimum_source_weight),
                    "raw_score": float(challenger["raw_score"]),
                    "adjusted_score": float(challenger["adjusted_score"]),
                    "prior_cost": float(challenger["prior_cost"]),
                    "complexity_cost": float(challenger["complexity_cost"]),
                    "saturation_fraction": float(
                        challenger["saturation_fraction"]),
                    "stability": stability,
                    "rejection_reasons": list(rejection_reasons),
                })
                if rejection_reasons:
                    break
                selected_groups = list(challenger["trial_groups"])
                current_components = challenger["components"]
                current_score = float(challenger["raw_score"])
                current_adjusted = float(challenger["adjusted_score"])
                current_prediction = challenger["prediction"]
            if selected_groups:
                self.selected_additive_groups = selected_groups
                selected = "source_additive"
                self.selected_parametric_ridge = max(float(self.ridge), 0.0)
                scores[selected] = float(current_score)
                component_scores[selected] = current_components
                nmse_scores[selected] = self._normalized_mse(
                    target, current_prediction)
                spectral_score_model = (
                    "source_orthogonal_anova_prior_plus_target_forward_gate"
                )
            self._additive_gate_diagnostics = {
                "status": "selected" if selected_groups else "fallback_stage1",
                "selected_variant": (
                    "source_additive" if selected_groups else additive_baseline
                ),
                "stage1_selected_basis": stage1_selected,
                "additive_baseline_basis": additive_baseline,
                "selected_group_indices": list(selected_groups),
                "selected_group_names": [
                    bank.group_names_[index] for index in selected_groups
                ],
                "selected_function_names": [
                    bank.function_names_[index] for index in selected_groups
                ],
                "bank_kind": self._additive_bank_kind,
                "strong_heredity": bool(
                    getattr(bank, "strong_heredity", False)),
                "selection_trace": selection_trace,
                "target_data_used": True,
                "target_oracle_used": False,
            }
        adaptive_rejection_reasons = []
        if output_index == 1 and self.meta_prior.spectral_adaptive_sparsity:
            spec = self._source_adaptive_sparsity_spec()
            if spec is not None:
                adaptive_matrix = np.vstack([
                    self._variant_features(x, "adaptive_spectral") for x in xs
                ])
                adaptive_prediction = self._adaptive_loo_predictions(
                    adaptive_matrix,
                    target,
                    xs,
                    spec,
                )
                adaptive_components = self._constraint_decision_score(
                    target, adaptive_prediction)
                component_scores["adaptive_spectral"] = adaptive_components
                scores["adaptive_spectral"] = float(adaptive_components["total"])
                nmse_scores["adaptive_spectral"] = self._normalized_mse(
                    target, adaptive_prediction)
                self._adaptive_gate_posterior_diagnostics = (
                    self._fit_adaptive_posterior(
                        adaptive_matrix,
                        target,
                        xs,
                        spec,
                    ).diagnostics()
                )
                spectral_score_model = "stage1_plus_adaptive_challenger"
                stage1_components = component_scores[stage1_selected]
                stage1_score = float(scores[stage1_selected])
                adaptive_tolerance = (
                    self.meta_prior.spectral_gate_selection_tolerance
                    * max(abs(stage1_score), 1.0)
                )
                if scores["adaptive_spectral"] >= stage1_score - adaptive_tolerance:
                    adaptive_rejection_reasons.append("insufficient_decision_gain")
                if (
                    adaptive_components["raw_false_feasible_rate"]
                    > stage1_components["raw_false_feasible_rate"] + 0.05
                ):
                    adaptive_rejection_reasons.append("raw_false_feasible_worse")
                dangerous_limit = max(
                    1.25 * stage1_components["dangerous_underprediction"],
                    stage1_components["dangerous_underprediction"] + 0.05,
                )
                if adaptive_components["dangerous_underprediction"] > dangerous_limit:
                    adaptive_rejection_reasons.append("dangerous_underprediction")
                adaptive_diag = self._adaptive_gate_posterior_diagnostics
                if int(adaptive_diag.get(
                    "adaptive_active_count_0_5",
                    adaptive_diag.get("active_count_0_5", 0),
                )) < 1:
                    adaptive_rejection_reasons.append(
                        "no_identified_active_coefficient")
                posterior_pip = np.asarray(
                    adaptive_diag.get("posterior_pip", []), dtype=float)
                always_active = int(spec.get("always_active_count", 0))
                active_extra = np.where(
                    posterior_pip[always_active:] >= 0.5
                )[0]
                inclusion_frequency = np.asarray(
                    self._adaptive_loo_support_diagnostics.get(
                        "adaptive_inclusion_frequency", []),
                    dtype=float,
                )
                selected_frequency = (
                    inclusion_frequency[active_extra]
                    if len(inclusion_frequency) and len(active_extra)
                    else np.asarray([], dtype=float)
                )
                min_selected_frequency = (
                    float(np.min(selected_frequency))
                    if len(selected_frequency) else 0.0
                )
                self._adaptive_loo_support_diagnostics[
                    "full_support_min_frequency"
                ] = min_selected_frequency
                if len(active_extra) and min_selected_frequency < 0.60:
                    adaptive_rejection_reasons.append(
                        "unstable_adaptive_support")
                effective = float(adaptive_diag.get("effective_dimension", 0.0))
                maximum = max(float(
                    adaptive_diag.get("max_effective_dimension", np.inf)), 1e-12)
                if effective >= (
                    self.meta_prior.spectral_adaptive_saturation_fraction * maximum
                ):
                    adaptive_rejection_reasons.append("cardinality_saturated")
                if not adaptive_rejection_reasons:
                    admitted = np.zeros(len(posterior_pip), dtype=bool)
                    admitted[:always_active] = True
                    admitted[always_active + active_extra] = True
                    self._adaptive_allowed_mask = admitted
                    selected = "adaptive_spectral"
                else:
                    self._adaptive_allowed_mask = None
        self.selected_basis = selected
        self.gate_diagnostics = {
            "status": "fit",
            "selected_basis": self.selected_basis,
            "output_index": output_index,
            "n_observations": int(len(xs)),
            "selection_metric": "decision_aware_loo",
            "spectral_score_model": spectral_score_model,
            "stage1_selected_basis": stage1_selected,
            "adaptive_rejection_reasons": adaptive_rejection_reasons,
            "adaptive_pilot_posterior": dict(
                self._adaptive_gate_posterior_diagnostics),
            "adaptive_support_stability": dict(
                self._adaptive_loo_support_diagnostics),
            "frequency_adaptation": dict(self._frequency_gate_diagnostics),
            "risk_alignment": dict(
                self._risk_alignment_gate_diagnostics),
            "additive_adaptation": dict(self._additive_gate_diagnostics),
            "gate_scope": "constraint_boundary_only",
            "decision_score": {name: float(value) for name, value in scores.items()},
            "decision_components": component_scores,
            "loo_nmse": {name: float(value) for name, value in nmse_scores.items()},
            "selection_tolerance": float(tolerance),
            "eligible_bases": list(eligible) + (
                ["adaptive_spectral"]
                if output_index == 1 and self.meta_prior.spectral_adaptive_sparsity
                else []
            ) + (
                [
                    "frozen_risk_aligned_coordinate",
                    "risk_aligned_coordinate",
                    "risk_aligned_spectral",
                ]
                if output_index == 1
                and self.meta_prior.spectral_risk_alignment
                and self.meta_prior.spectral_alignment_admission
                else []
            ) + frequency_variants + (
                ["source_additive"]
                if output_index == 1
                and self.meta_prior.spectral_additive_adaptation
                else []
            ),
            "selected_parametric_ridge": float(
                self.selected_parametric_ridge),
            "sequential_refit_count": int(self._additive_refit_count),
            "pilot_constraint_guard": self._selected_certification_guard(
                component_scores, output_index),
        }
        return self.selected_basis

    def _selected_certification_guard(self, component_scores, output_index):
        if int(output_index) != 1:
            return 0.0
        aligned = self.selected_basis in {
            "frozen_risk_aligned_coordinate",
            "risk_aligned_coordinate",
            "risk_aligned_spectral",
        }
        if (
            not aligned
            and self._locked_alignment_stage1_basis is not None
            and self.selected_basis == self._locked_alignment_stage1_basis
            and self._locked_alignment_stage1_guard is not None
        ):
            return max(float(self._locked_alignment_stage1_guard), 0.0)
        return max(
            float(component_scores[self.selected_basis]["calibration_guard"]),
            self._risk_alignment_source_guard() if aligned else 0.0,
        )

    def diagnostics(self):
        out = dict(self.gate_diagnostics)
        out["adaptive_sparsity"] = dict(
            self._adaptive_sparsity_diagnostics)
        return out

    def should_refit_from_observations(self, observations):
        if self.output_index != 1 or len(observations) < 5:
            return False
        additive_due = bool(
            self.meta_prior.spectral_additive_adaptation and (
                len(observations) - int(self._last_additive_refit_n)
                >= int(self.meta_prior.spectral_additive_refit_interval)
            )
        )
        alignment_due = bool(
            self.meta_prior.spectral_risk_alignment
            and self.meta_prior.spectral_alignment_admission and (
                len(observations) - int(self._last_alignment_refit_n)
                >= int(self.meta_prior.spectral_alignment_refit_interval)
            )
        )
        frequency_due = bool(
            self.meta_prior.spectral_frequency_adaptation and (
                len(observations) - int(self._last_frequency_refit_n)
                >= int(self.meta_prior.spectral_frequency_refit_interval)
            )
        )
        return additive_due or alignment_due or frequency_due

    def runtime_state(self):
        return {
            "selected_basis": self.selected_basis,
            "selected_parametric_ridge": float(
                self.selected_parametric_ridge),
            "selected_additive_groups": list(self.selected_additive_groups),
            "additive_base_basis": self._additive_base_basis,
            "additive_bank_kind": self._additive_bank_kind,
            "locked_additive_base_basis": self._locked_additive_base_basis,
            "last_additive_refit_n": int(self._last_additive_refit_n),
            "last_alignment_refit_n": int(self._last_alignment_refit_n),
            "locked_alignment_stage1_basis": (
                self._locked_alignment_stage1_basis),
            "locked_alignment_stage1_guard": (
                self._locked_alignment_stage1_guard),
            "last_frequency_refit_n": int(self._last_frequency_refit_n),
            "additive_refit_count": int(self._additive_refit_count),
            "target_risk_alignment": (
                None
                if self._target_risk_alignment is None
                else {
                    "matrix": np.asarray(
                        self._target_risk_alignment.matrix, dtype=float).copy(),
                    "accepted": bool(self._target_risk_alignment.accepted),
                    "diagnostics": copy.deepcopy(
                        self._target_risk_alignment.diagnostics),
                    "boundary_axis": (
                        None
                        if self._target_risk_alignment.boundary_axis is None
                        else np.asarray(
                            self._target_risk_alignment.boundary_axis,
                            dtype=float,
                        ).copy()
                    ),
                    "expert_weights": (
                        None
                        if self._target_risk_alignment.expert_weights is None
                        else np.asarray(
                            self._target_risk_alignment.expert_weights,
                            dtype=float,
                        ).copy()
                    ),
                }
            ),
            "gate_diagnostics": copy.deepcopy(self.gate_diagnostics),
        }

    def load_runtime_state(self, state):
        self.selected_basis = str(state.get(
            "selected_basis", self.selected_basis))
        self.selected_parametric_ridge = float(state.get(
            "selected_parametric_ridge", self.selected_parametric_ridge))
        self.selected_additive_groups = [
            int(index) for index in state.get(
                "selected_additive_groups", self.selected_additive_groups)
        ]
        self._additive_base_basis = str(state.get(
            "additive_base_basis", self._additive_base_basis))
        self._additive_bank_kind = str(state.get(
            "additive_bank_kind", self._additive_bank_kind))
        self._locked_additive_base_basis = state.get(
            "locked_additive_base_basis", self._locked_additive_base_basis)
        self._last_additive_refit_n = int(state.get(
            "last_additive_refit_n", self._last_additive_refit_n))
        self._last_alignment_refit_n = int(state.get(
            "last_alignment_refit_n", self._last_alignment_refit_n))
        self._locked_alignment_stage1_basis = state.get(
            "locked_alignment_stage1_basis",
            self._locked_alignment_stage1_basis,
        )
        locked_guard = state.get(
            "locked_alignment_stage1_guard",
            self._locked_alignment_stage1_guard,
        )
        self._locked_alignment_stage1_guard = (
            None if locked_guard is None else float(locked_guard))
        self._last_frequency_refit_n = int(state.get(
            "last_frequency_refit_n", self._last_frequency_refit_n))
        self._additive_refit_count = int(state.get(
            "additive_refit_count", self._additive_refit_count))
        alignment = state.get("target_risk_alignment")
        if alignment is not None:
            self._target_risk_alignment = TargetRiskAlignment(
                matrix=np.asarray(alignment["matrix"], dtype=float),
                accepted=bool(alignment.get("accepted", False)),
                diagnostics=copy.deepcopy(alignment.get("diagnostics", {})),
                boundary_axis=(
                    None
                    if alignment.get("boundary_axis") is None
                    else np.asarray(alignment["boundary_axis"], dtype=float)
                ),
                expert_weights=(
                    None
                    if alignment.get("expert_weights") is None
                    else np.asarray(alignment["expert_weights"], dtype=float)
                ),
            )
            self._risk_alignment_gate_diagnostics = copy.deepcopy(
                self._target_risk_alignment.diagnostics)
        self.gate_diagnostics = copy.deepcopy(state.get(
            "gate_diagnostics", self.gate_diagnostics))

    def _risk_alignment_source_guard(self):
        if self.meta_prior.risk_subspace_alignment is None:
            return 0.0
        scale = max(
            abs(float(getattr(self.problem, "tau", 0.0))),
            float(getattr(self.problem, "sigma_level", 0.0)),
            1e-6,
        )
        return float(
            self.meta_prior.risk_subspace_alignment.source_residual_guard_
            * scale
        )

    def certification_guard(self):
        if self.output_index != 1:
            return 0.0
        return max(float(self.gate_diagnostics.get("pilot_constraint_guard", 0.0)), 0.0)

    def initial_parametric_coefficients(self, phi, target):
        """Fit the target-selected ridge model in the raw GPR basis."""

        matrix = np.asarray(phi, dtype=float)
        values = np.asarray(target, dtype=float).reshape(-1)
        ridge = max(float(self.selected_parametric_ridge), 0.0)
        if ridge <= 0.0:
            # The frozen basis can contain source-identifiable directions that
            # are nearly null on a ten-point held-out pilot.  A fixed condition
            # cap removes only those target-unidentifiable directions.
            rcond = 1e-3
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            threshold = (
                rcond * float(singular_values[0])
                if len(singular_values)
                else 0.0
            )
            effective_rank = int(np.sum(singular_values > threshold))
            self.gate_diagnostics.update({
                "initial_fit_solver": "truncated_svd",
                "initial_fit_rcond": float(rcond),
                "initial_fit_effective_rank": effective_rank,
                "initial_fit_matrix_rank": int(np.linalg.matrix_rank(matrix)),
                "initial_fit_max_condition": float(1.0 / rcond),
            })
            return np.linalg.lstsq(matrix, values, rcond=rcond)[0]
        features = matrix[:, 1:]
        feature_mean = np.mean(features, axis=0)
        feature_scale = np.std(features, axis=0)
        feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
        standardized = (features - feature_mean) / feature_scale
        design = np.column_stack([
            np.ones(len(standardized), dtype=float),
            standardized,
        ])
        value_mean = float(np.mean(values))
        value_scale = max(float(np.std(values)), 1e-8)
        response = (values - value_mean) / value_scale
        penalty = ridge * np.eye(design.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        self.gate_diagnostics.update({
            "initial_fit_solver": "standardized_ridge",
            "initial_fit_ridge": float(ridge),
        })
        try:
            standardized_beta = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ response,
            )
        except np.linalg.LinAlgError:
            standardized_beta = np.linalg.lstsq(
                design.T @ design + penalty,
                design.T @ response,
                rcond=None,
            )[0]
        slopes = value_scale * standardized_beta[1:] / feature_scale
        intercept = (
            value_mean
            + value_scale * standardized_beta[0]
            - float(feature_mean @ slopes)
        )
        return np.concatenate([[intercept], slopes])

    def adaptive_sparsity_spec(self, observations=None):
        """Return a source hyper-prior only when the target pilot admits it."""

        if (
            not self.meta_prior.spectral_adaptive_sparsity
            or self.output_index != 1
            or self.selected_basis != "adaptive_spectral"
        ):
            self._adaptive_sparsity_diagnostics = {
                "status": "fallback_stage1",
                "reason": (
                    "disabled"
                    if not self.meta_prior.spectral_adaptive_sparsity
                    else "non_spectral_gate"
                ),
                "selected_basis": self.selected_basis,
            }
            return None
        spec = self._source_adaptive_sparsity_spec()
        if spec is None:
            self._adaptive_sparsity_diagnostics = {
                "status": "fallback_stage1",
                "reason": "missing_source_hyperprior",
            }
            return None
        if self._adaptive_allowed_mask is None:
            self._adaptive_sparsity_diagnostics = {
                "status": "fallback_stage1",
                "reason": "missing_pilot_admitted_support",
            }
            return None
        spec["allowed_mask"] = self._adaptive_allowed_mask.tolist()
        del observations
        self._adaptive_sparsity_diagnostics = {
            **self._adaptive_sparsity_diagnostics,
            "status": "enabled",
            "method": "target_adaptive_variational_spike_slab",
            "dictionary_dim": int(spec["dictionary_dim"]),
            "source_hyperprior_only": True,
            "stage1_fallback": self.gate_diagnostics.get(
                "stage1_selected_basis"),
            "pilot_admitted_adaptive_count": int(np.sum(
                self._adaptive_allowed_mask[
                    int(spec.get("always_active_count", 0)):
                ]
            )),
        }
        return spec

    def _source_adaptive_sparsity_spec(self):
        source = self.meta_prior.spectral_coefficient_prior.get(self.output_index)
        if source is None:
            return None
        source_pip = np.asarray(source["prior_pip"], dtype=float).copy()
        always_active = int(self.meta_prior.spectral_always_active_count)
        allowed_mask = source_pip >= 0.5
        allowed_mask[:always_active] = True
        return {
            "source_pip": source_pip,
            "source_slab_scale": np.asarray(
                source["slab_scale"], dtype=float).copy(),
            "min_pip": float(self.meta_prior.spectral_adaptive_min_pip),
            "max_pip": float(self.meta_prior.spectral_adaptive_max_pip),
            "spike_ratio": float(
                self.meta_prior.spectral_adaptive_spike_ratio),
            "damping": float(self.meta_prior.spectral_adaptive_damping),
            "max_iter": int(self.meta_prior.spectral_adaptive_max_iter),
            "tolerance": float(self.meta_prior.spectral_adaptive_tolerance),
            "residual_floor_scale": float(
                self.meta_prior.spectral_adaptive_residual_floor_scale),
            "multiplicity_correction": float(
                self.meta_prior.spectral_adaptive_multiplicity_correction),
            "max_effective_fraction": float(
                self.meta_prior.spectral_adaptive_max_effective_fraction),
            "always_active_count": int(
                self.meta_prior.spectral_always_active_count),
            "allowed_mask": allowed_mask.tolist(),
            "dictionary_dim": int(len(source["prior_pip"])),
            "source_domains": list(source.get("source_domains", [])),
        }

    def _adaptive_loo_predictions(self, features, target, xs, spec):
        predictions = []
        posterior_pips = []
        for heldout in range(len(xs)):
            train = np.arange(len(xs)) != heldout
            train_y = np.asarray(target, dtype=float)[train]
            noise = max(
                float(getattr(self.problem, "sigma_level", 0.0)) ** 2,
                1e-6,
            )
            posterior = AdaptiveSpikeSlabPosterior(
                spec["source_pip"],
                spec["source_slab_scale"],
                min_pip=spec["min_pip"],
                max_pip=spec["max_pip"],
                spike_ratio=spec["spike_ratio"],
                damping=spec["damping"],
                max_iter=spec["max_iter"],
                tolerance=spec["tolerance"],
                residual_floor_scale=spec["residual_floor_scale"],
                multiplicity_correction=spec["multiplicity_correction"],
                max_effective_fraction=spec["max_effective_fraction"],
                always_active_count=spec.get("always_active_count", 0),
                allowed_mask=spec.get("allowed_mask"),
            ).fit(
                np.asarray(features, dtype=float)[train],
                train_y,
                np.full(np.sum(train), noise, dtype=float),
                [xs[index] for index in np.where(train)[0]],
                deviation_variance=max(noise, 1e-6),
            )
            predictions.append(float(
                posterior.predict_parametric_mean(
                    np.asarray(features, dtype=float)[heldout])))
            posterior_pips.append(np.asarray(
                posterior.diagnostics()["posterior_pip"], dtype=float))
        always_active = int(spec.get("always_active_count", 0))
        pip_matrix = np.vstack(posterior_pips)
        adaptive_support = pip_matrix[:, always_active:] >= 0.5
        inclusion_frequency = np.mean(adaptive_support, axis=0)
        pairwise_jaccard = []
        for first in range(len(adaptive_support)):
            for second in range(first + 1, len(adaptive_support)):
                union = np.logical_or(
                    adaptive_support[first], adaptive_support[second])
                intersection = np.logical_and(
                    adaptive_support[first], adaptive_support[second])
                pairwise_jaccard.append(
                    1.0 if not np.any(union)
                    else float(np.sum(intersection) / np.sum(union))
                )
        self._adaptive_loo_support_diagnostics = {
            "status": "fit",
            "n_folds": int(len(pip_matrix)),
            "adaptive_inclusion_frequency": inclusion_frequency.tolist(),
            "mean_pairwise_jaccard": float(
                np.mean(pairwise_jaccard)) if pairwise_jaccard else 1.0,
            "median_adaptive_support_size": float(np.median(
                np.sum(adaptive_support, axis=1))),
        }
        return np.asarray(predictions, dtype=float)

    def _fit_adaptive_posterior(self, features, target, xs, spec):
        target = np.asarray(target, dtype=float)
        noise = max(
            float(getattr(self.problem, "sigma_level", 0.0)) ** 2,
            1e-6,
        )
        return AdaptiveSpikeSlabPosterior(
            spec["source_pip"],
            spec["source_slab_scale"],
            min_pip=spec["min_pip"],
            max_pip=spec["max_pip"],
            spike_ratio=spec["spike_ratio"],
            damping=spec["damping"],
            max_iter=spec["max_iter"],
            tolerance=spec["tolerance"],
            residual_floor_scale=spec["residual_floor_scale"],
            multiplicity_correction=spec["multiplicity_correction"],
            max_effective_fraction=spec["max_effective_fraction"],
            always_active_count=spec.get("always_active_count", 0),
            allowed_mask=spec.get("allowed_mask"),
        ).fit(
            features,
            target,
            np.full(len(target), noise, dtype=float),
            xs,
            deviation_variance=max(noise, 1e-6),
        )

    def _adaptive_pilot_accepts(self, observations, spec):
        xs = list(observations)
        if len(xs) < 6:
            self._adaptive_sparsity_diagnostics = {
                "status": "fallback_stage1",
                "reason": "insufficient_adaptive_pilot",
                "n_observations": int(len(xs)),
            }
            return False
        target = np.asarray([
            float(np.mean(np.asarray(observations[x], dtype=float), axis=0)[1])
            for x in xs
        ])
        features = np.vstack([
            self.meta_prior.spectral_features(self.problem, x) for x in xs
        ])
        coordinate_features = np.vstack([
            self.meta_prior.coordinate_basis_features(self.problem, x) for x in xs
        ])
        baseline_prediction = self._ridge_loo_predictions(
            coordinate_features, target)
        baseline = self._constraint_decision_score(target, baseline_prediction)
        adaptive_prediction = self._adaptive_loo_predictions(
            features, target, xs, spec)
        adaptive = self._constraint_decision_score(target, adaptive_prediction)
        posterior_diag = self._fit_adaptive_posterior(
            features, target, xs, spec).diagnostics()
        effective = float(posterior_diag.get("effective_dimension", 0.0))
        maximum = max(float(
            posterior_diag.get("max_effective_dimension", np.inf)), 1e-12)
        nonsaturated = bool(
            effective
            < self.meta_prior.spectral_adaptive_saturation_fraction * maximum
        )
        tolerance = self.meta_prior.spectral_adaptive_gate_tolerance * max(
            abs(float(baseline["total"])), 1.0)
        accepted = bool(
            np.isfinite(adaptive["total"])
            and adaptive["total"] <= baseline["total"] + tolerance
            and adaptive["raw_false_feasible_rate"]
            <= baseline["raw_false_feasible_rate"] + 0.05
            and nonsaturated
        )
        self._adaptive_sparsity_diagnostics = {
            "status": "pilot_accepted" if accepted else "fallback_stage1",
            "reason": "adaptive_loo_gate" if accepted else "adaptive_loo_worse",
            "n_observations": int(len(xs)),
            "baseline_decision": baseline,
            "adaptive_decision": adaptive,
            "adaptive_posterior": posterior_diag,
            "nonsaturated": nonsaturated,
            "gate_tolerance": float(tolerance),
        }
        return accepted

    def record_adaptive_sparsity_diagnostics(self, diagnostics):
        pilot = dict(self._adaptive_sparsity_diagnostics)
        self._adaptive_sparsity_diagnostics = {
            "status": "fit",
            "pilot": pilot,
            "posterior": dict(diagnostics),
        }

    def apply_coefficient_prior(self, beta_mean, prior_var):
        """Apply the frozen source shrinkage prior after the pilot gate.

        The coordinate path is returned byte-for-byte unchanged.  Stage 2 is
        therefore an isolated ablation on top of the promoted Stage-1 gate.
        """

        beta = np.asarray(beta_mean, dtype=float).copy()
        base_var = max(float(prior_var), 1e-12)
        spectral_family = bool(
            self.selected_basis in {
                "source_spectral",
                "source_additive",
                "risk_aligned_spectral",
            }
            or str(self.selected_basis).startswith("frequency_band_")
            or str(self.selected_basis).startswith(
                "aligned_frequency_band_")
        )
        if self.output_index != 1 or not spectral_family:
            self.gate_diagnostics["coefficient_prior_applied"] = False
            return beta, base_var
        weights = self.meta_prior.spectral_shrinkage_weights(self.output_index)
        if weights is None or len(weights) == 0:
            self.gate_diagnostics["coefficient_prior_applied"] = False
            return beta, base_var
        n_active = min(len(weights), max(len(beta) - 1, 0))
        weights = np.asarray(weights[:n_active], dtype=float)
        prior_diag = np.full(len(beta), base_var, dtype=float)
        covariance_weight = np.maximum(
            weights ** 2,
            float(self.meta_prior.spectral_shrinkage_floor),
        )
        prior_diag[1:1 + n_active] *= covariance_weight
        source_prior = self.meta_prior.spectral_coefficient_prior.get(
            self.output_index, {})
        self.gate_diagnostics.update({
            "coefficient_prior_applied": True,
            "coefficient_prior_mode": "variance_only",
            "coefficient_prior_selected_basis": str(self.selected_basis),
            "coefficient_prior_mean_shrunk": False,
            "coefficient_prior_active_dim": int(n_active),
            "coefficient_prior_weights": weights.tolist(),
            "coefficient_prior_pip": np.asarray(
                source_prior.get("pip", []), dtype=float
            )[:n_active].tolist(),
            "coefficient_prior_variance_ratio": covariance_weight.tolist(),
        })
        return beta, prior_diag

    def _ridge_loo_score(self, features, target):
        prediction = self._ridge_loo_predictions(features, target)
        return self._normalized_mse(target, prediction)

    def _ridge_loo_predictions(self, features, target, ridge=None):
        predictions = []
        for heldout in range(len(features)):
            train = np.arange(len(features)) != heldout
            prediction = self._ridge_predict(
                features[train],
                target[train],
                features[heldout:heldout + 1],
                ridge=ridge,
            )[0]
            predictions.append(float(prediction))
        return np.asarray(predictions, dtype=float)

    @staticmethod
    def _normalized_mse(target, prediction, weights=None):
        target = np.asarray(target, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        if weights is None:
            weights = np.ones(len(target), dtype=float)
        weights = np.clip(np.asarray(weights, dtype=float), 1e-8, np.inf)
        weights = weights / float(np.sum(weights))
        center = float(np.sum(target * weights))
        scale = float(np.sum((target - center) ** 2 * weights))
        error = float(np.sum((target - prediction) ** 2 * weights))
        return error / max(scale, 1e-10)

    @staticmethod
    def _pairwise_order_loss(target, prediction, weights, scale):
        target = np.asarray(target, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        weights = np.asarray(weights, dtype=float)
        losses = []
        pair_weights = []
        scale = max(float(scale), 1e-8)
        for i in range(len(target)):
            for j in range(i + 1, len(target)):
                truth_order = np.tanh((target[i] - target[j]) / scale)
                pred_order = np.tanh((prediction[i] - prediction[j]) / scale)
                losses.append(float((truth_order - pred_order) ** 2))
                pair_weights.append(float(np.sqrt(weights[i] * weights[j])))
        if not losses:
            return 0.0
        pair_weights = np.clip(np.asarray(pair_weights), 1e-8, np.inf)
        return float(np.average(np.asarray(losses), weights=pair_weights))

    def _chance_margin(self, constraint_mean):
        alpha = float(getattr(self.problem, "alpha", 0.05))
        z_alpha = float(norm.ppf(1.0 - alpha))
        sigma = max(float(getattr(self.problem, "sigma_level", 0.04)), 1e-8)
        tau = float(getattr(self.problem, "tau", 0.0))
        return np.asarray(constraint_mean, dtype=float) + z_alpha * sigma - tau

    def _frontier_weights(self, margins):
        margins = np.asarray(margins, dtype=float)
        sigma = max(float(getattr(self.problem, "sigma_level", 0.04)), 1e-8)
        robust = 0.7413 * float(
            np.quantile(margins, 0.75) - np.quantile(margins, 0.25))
        scale = max(sigma, robust, 1e-8)
        boundary = np.exp(-0.5 * (margins / scale) ** 2)
        feasible = margins <= 0.0
        if np.any(feasible):
            frontier = boundary
        else:
            frontier = np.exp(-(margins - float(np.min(margins))) / scale)
        weight = (
            1.0
            + self.meta_prior.spectral_gate_boundary_weight
            * np.maximum(boundary, frontier)
            + feasible.astype(float)
        )
        return np.asarray(weight, dtype=float), float(scale)

    def _constraint_decision_score(self, target, prediction):
        truth_margin = self._chance_margin(target)
        pred_margin = self._chance_margin(prediction)
        weights, scale = self._frontier_weights(truth_margin)
        margin_nmse = self._normalized_mse(
            truth_margin, pred_margin, weights=weights)
        boundary_mse = float(np.average(
            ((truth_margin - pred_margin) / scale) ** 2,
            weights=weights,
        ))
        rank_loss = self._pairwise_order_loss(
            truth_margin, pred_margin, weights, scale)
        probability = 1.0 / (
            1.0 + np.exp(np.clip(pred_margin / scale, -40.0, 40.0)))
        feasible = truth_margin <= 0.0
        brier = float(np.average(
            (probability - feasible.astype(float)) ** 2,
            weights=weights,
        ))
        residual = truth_margin - pred_margin
        infeasible = ~feasible
        raw_false_feasible = (
            float(np.mean(pred_margin[infeasible] <= 0.0))
            if np.any(infeasible)
            else 0.0
        )
        calibrated = np.empty_like(pred_margin)
        quantile = float(np.clip(
            self.meta_prior.spectral_gate_calibration_quantile, 0.5, 0.99))
        for heldout in range(len(residual)):
            keep = np.arange(len(residual)) != heldout
            correction = (
                float(np.quantile(residual[keep], quantile))
                if np.any(keep)
                else 0.0
            )
            calibrated[heldout] = pred_margin[heldout] + max(correction, 0.0)
        false_feasible = (
            float(np.mean(calibrated[infeasible] <= 0.0))
            if np.any(infeasible)
            else 0.0
        )
        false_infeasible = (
            float(np.mean(calibrated[feasible] > 0.0))
            if np.any(feasible)
            else 0.0
        )
        dangerous_underprediction = float(np.average(
            (np.maximum(residual, 0.0) / scale) ** 2,
            weights=weights,
        ))
        total = (
            0.15 * margin_nmse
            + 0.20 * boundary_mse
            + 0.25 * rank_loss
            + 0.20 * brier
            + self.meta_prior.spectral_gate_dangerous_weight
            * false_feasible
            + 0.50 * self.meta_prior.spectral_gate_dangerous_weight
            * raw_false_feasible
            + 0.20 * false_infeasible
            + 0.35 * dangerous_underprediction
        )
        return {
            "total": float(total),
            "n_observed_feasible": int(np.sum(feasible)),
            "n_observed_infeasible": int(np.sum(infeasible)),
            "margin_nmse": float(margin_nmse),
            "boundary_mse": float(boundary_mse),
            "rank_loss": float(rank_loss),
            "brier": float(brier),
            "false_feasible_rate": float(false_feasible),
            "raw_false_feasible_rate": float(raw_false_feasible),
            "false_infeasible_rate": float(false_infeasible),
            "dangerous_underprediction": float(dangerous_underprediction),
            "margin_scale": float(scale),
            "calibration_guard": float(max(
                np.quantile(residual, quantile), 0.0)),
        }

    def _subsample_decision_stability(
        self,
        target,
        baseline_prediction,
        challenger_prediction,
    ):
        target = np.asarray(target, dtype=float)
        baseline_prediction = np.asarray(baseline_prediction, dtype=float)
        challenger_prediction = np.asarray(challenger_prediction, dtype=float)
        if len(target) < 6:
            gain = (
                self._constraint_decision_score(
                    target, baseline_prediction)["total"]
                - self._constraint_decision_score(
                    target, challenger_prediction)["total"]
            )
            return {
                "fold_gains": [float(gain)],
                "median_gain": float(gain),
                "win_rate": float(gain > 0.0),
            }
        gains = []
        fold_count = min(3, max(2, len(target) // 3))
        positions = np.arange(len(target))
        for fold in range(fold_count):
            keep = positions % fold_count == fold
            if np.sum(keep) < 2:
                continue
            baseline_score = self._constraint_decision_score(
                target[keep], baseline_prediction[keep])["total"]
            challenger_score = self._constraint_decision_score(
                target[keep], challenger_prediction[keep])["total"]
            gains.append(float(baseline_score - challenger_score))
        values = np.asarray(gains, dtype=float)
        return {
            "fold_gains": [float(value) for value in values],
            "median_gain": float(np.median(values)) if len(values) else 0.0,
            "win_rate": float(np.mean(values > 0.0)) if len(values) else 0.0,
        }

    def _objective_decision_score(self, target, prediction, constraint_mean):
        margins = self._chance_margin(constraint_mean)
        weights, margin_scale = self._frontier_weights(margins)
        objective_scale = max(float(np.std(target)), 1e-8)
        weighted_nmse = self._normalized_mse(target, prediction, weights=weights)
        rank_loss = self._pairwise_order_loss(
            target, prediction, weights, objective_scale)
        feasible = np.where(margins <= 0.0)[0]
        if len(feasible) == 0:
            count = min(max(2, len(target) // 3), len(target))
            relevant = np.argsort(margins)[:count]
        else:
            relevant = feasible
        chosen = int(relevant[int(np.argmin(prediction[relevant]))])
        best = float(np.min(target[relevant]))
        selection_regret = max(float(target[chosen]) - best, 0.0) / objective_scale
        total = 0.50 * weighted_nmse + 0.35 * rank_loss + 0.15 * selection_regret
        return {
            "total": float(total),
            "weighted_nmse": float(weighted_nmse),
            "rank_loss": float(rank_loss),
            "selection_regret": float(selection_regret),
            "margin_scale": float(margin_scale),
            "n_observed_feasible": int(len(feasible)),
        }

    def _ridge_predict(self, train_x, train_y, test_x, ridge=None):
        mean = np.mean(train_x, axis=0)
        scale = np.std(train_x, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        X = (train_x - mean) / scale
        X_test = (test_x - mean) / scale
        X = np.column_stack([np.ones(len(X)), X])
        X_test = np.column_stack([np.ones(len(X_test)), X_test])
        y_mean = float(np.mean(train_y))
        y_scale = max(float(np.std(train_y)), 1e-8)
        y = (np.asarray(train_y, dtype=float) - y_mean) / y_scale
        ridge = self.ridge if ridge is None else max(float(ridge), 0.0)
        penalty = ridge * np.eye(X.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X.T @ X + penalty, X.T @ y, rcond=None)[0]
        return (X_test @ beta) * y_scale + y_mean


class AdmissibleProblemAdapter:
    """Hide target-specific structural hooks while preserving simulation API."""

    def __init__(self, base_problem, variant="strict_universal"):
        self.base = base_problem
        self.problem_name = f"{base_problem.problem_name}_{variant}"
        self.d = base_problem.d
        self.L = base_problem.L
        self.alpha = base_problem.alpha
        self.tau = base_problem.tau
        self.sigma_level = base_problem.sigma_level
        self.ref_point = getattr(base_problem, "ref_point", None)
        self.variant = str(variant)

    def __getattr__(self, name):
        if name in HIDDEN_TARGET_STRUCTURAL_METHODS:
            raise AttributeError(name)
        return getattr(self.base, name)

    def admissibility_audit(self):
        if self.variant == "domain_tuned_upper_bound":
            return domain_tuned_audit().to_dict()
        return strict_universal_audit().to_dict()

    def int_bounds(self):
        return self.base.int_bounds()

    def normalize(self, x):
        return self.base.normalize(x)

    def continuous_to_int(self, x_norm):
        return self.base.continuous_to_int(x_norm)

    def sample_random(self, rng=None):
        return self.base.sample_random(rng)

    def simulate(self, x, rng=None):
        return self.base.simulate(x, rng)


class FixedTaskExpertBasis:
    """Frozen source-only basis used by one finite task expert."""

    ORDERED_VARIANTS = {"ordered_cumulative", "ordered_semiparametric"}

    def __init__(self, meta_prior, problem, variant, output_index=0):
        self.meta_prior = meta_prior
        self.problem = problem
        self.variant = str(variant)
        self.output_index = int(output_index)
        self._local_kernel = None
        self._ordered_residual_projection = None
        if self.variant in {"local_risk_kernel", "ordered_semiparametric"}:
            self._local_kernel = self._build_local_kernel()
        if self.variant == "ordered_semiparametric":
            self._ordered_residual_projection = (
                self._build_ordered_residual_projection())
        lo, _ = problem.int_bounds()
        self.feature_dim = int(len(self._raw_features(tuple(lo))))

    def _kernel_coordinate(self, x):
        if self.variant == "ordered_semiparametric":
            exposure = self.meta_prior.ordered_cumulative_risk_exposure(
                self.problem, x)
            return np.concatenate([exposure.A, exposure.N]).astype(float)
        return np.asarray(
            self.meta_prior.risk_coordinate(self.problem, x), dtype=float)

    def _build_local_kernel(self):
        rng = np.random.default_rng(self.meta_prior.seed + 15485863)
        rows = []
        rows.extend(self.meta_prior.profile_template_candidates(
            self.problem, n=32, rng=rng))
        rows.extend(self.meta_prior.universal_shape_candidates(
            self.problem, n=32, rng=rng, force=True))
        rows.extend(self.meta_prior.alignment_profile_candidates(
            self.problem, n=16, rng=rng))
        rows = unique_candidates(rows)
        if not rows:
            rows = [self.problem.sample_random(rng) for _ in range(16)]
            rows = unique_candidates(rows)
        if self.variant == "ordered_semiparametric":
            ordered_dim = int(len(
                self.meta_prior.ordered_coordinate_basis_features(
                    self.problem, rows[0])
            ))
            required_rows = ordered_dim + 1 + 6
            attempts = 0
            while len(rows) < required_rows and attempts < 512:
                rows = unique_candidates([
                    *rows,
                    self.problem.sample_random(rng),
                ])
                attempts += 1
        psi = np.vstack([self._kernel_coordinate(x) for x in rows])
        mean = np.mean(psi, axis=0)
        scale = np.std(psi, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        z = (psi - mean) / scale
        if self.variant == "ordered_semiparametric":
            ordered_dim = int(len(
                self.meta_prior.ordered_coordinate_basis_features(
                    self.problem, rows[0])
            ))
            n_centers = min(ordered_dim + 1 + 6, len(z))
        else:
            n_centers = min(6, len(z))
        center_indices = [int(np.argmin(np.sum(z ** 2, axis=1)))]
        while len(center_indices) < n_centers:
            distance = np.min(np.stack([
                np.sum((z - z[index]) ** 2, axis=1)
                for index in center_indices
            ]), axis=0)
            distance[center_indices] = -np.inf
            center_indices.append(int(np.argmax(distance)))
        centers = z[center_indices]
        if len(centers) > 1:
            pairwise = np.sqrt(np.sum(
                (centers[:, None, :] - centers[None, :, :]) ** 2,
                axis=2,
            ))
            positive = pairwise[pairwise > 1e-8]
            lengthscale = (
                float(np.median(positive)) if len(positive) else 1.0)
        else:
            lengthscale = 1.0
        return {
            "mean": mean,
            "scale": scale,
            "centers": centers,
            "lengthscale": max(lengthscale, 1e-3),
            "pool_size": int(len(rows)),
            "rows": [tuple(int(value) for value in row) for row in rows],
        }

    def _local_kernel_features(self, x):
        state = self._local_kernel
        psi = self._kernel_coordinate(x)
        z = (psi - state["mean"]) / state["scale"]
        squared = np.sum((state["centers"] - z[None, :]) ** 2, axis=1)
        return np.exp(
            -0.5 * squared / max(state["lengthscale"] ** 2, 1e-12))

    def _build_ordered_residual_projection(self):
        rows = list(self._local_kernel["rows"])
        ordered = np.vstack([
            self.meta_prior.ordered_coordinate_basis_features(self.problem, x)
            for x in rows
        ])
        kernel = np.vstack([self._local_kernel_features(x) for x in rows])
        design = np.column_stack([np.ones(len(ordered)), ordered])
        cross_dictionary = design.T @ kernel
        _left, singular_values, right_t = np.linalg.svd(
            cross_dictionary, full_matrices=True)
        leading = (
            float(singular_values[0]) if len(singular_values) else 0.0)
        tolerance = max(cross_dictionary.shape) * np.finfo(float).eps * leading
        rank = int(np.sum(singular_values > tolerance))
        nullspace = np.asarray(right_t[rank:].T, dtype=float)
        if nullspace.shape[1] < 1:
            raise RuntimeError(
                "ordered semiparametric dictionary has no orthogonal residual"
            )
        residual_dim = min(6, int(nullspace.shape[1]))
        feature_projection = np.asarray(
            nullspace[:, :residual_dim], dtype=float)
        for column in range(feature_projection.shape[1]):
            pivot = int(np.argmax(np.abs(feature_projection[:, column])))
            if feature_projection[pivot, column] < 0.0:
                feature_projection[:, column] *= -1.0
        residual = kernel @ feature_projection
        cross = design.T @ residual
        denominator = max(
            float(np.linalg.norm(design) * np.linalg.norm(residual)),
            1e-12,
        )
        gram = feature_projection.T @ feature_projection
        return {
            "feature_projection": feature_projection,
            "residualization_mode": "bounded_coefficient_nullspace",
            "ordered_dim": int(ordered.shape[1]),
            "parent_kernel_dim": int(kernel.shape[1]),
            "cross_dictionary_rank": rank,
            "nullspace_dim": int(nullspace.shape[1]),
            "residual_dim": residual_dim,
            "pool_size": int(len(rows)),
            "orthogonality_fro": float(np.linalg.norm(cross)),
            "orthogonality_relative": float(np.linalg.norm(cross) / denominator),
            "projection_orthonormal_error": float(np.linalg.norm(
                gram - np.eye(residual_dim))),
            "finite_pool_max_l2": float(np.max(np.linalg.norm(
                residual, axis=1))),
            "global_l2_bound": float(np.sqrt(kernel.shape[1])),
            "target_labels_used": False,
        }

    def _ordered_semiparametric_features(self, x):
        ordered = np.asarray(
            self.meta_prior.ordered_coordinate_basis_features(self.problem, x),
            dtype=float,
        )
        kernel = self._local_kernel_features(x)
        residual = (
            kernel
            @ self._ordered_residual_projection["feature_projection"]
        )
        return np.concatenate([ordered, residual])

    def _raw_features(self, x):
        variant = self.variant
        if variant == "universal_coordinate":
            return self.meta_prior.coordinate_basis_features(self.problem, x)
        if variant == "null_universal":
            descriptor = self.meta_prior.descriptor(self.problem, x)
            return self.meta_prior._scaled_descriptor(descriptor)
        if variant == "source_spectral":
            return self.meta_prior.stage1_spectral_features(self.problem, x)
        if variant == "risk_aligned_coordinate":
            return self.meta_prior.frozen_risk_aligned_coordinate(
                self.problem, x)
        if variant == "risk_aligned_spectral":
            return self.meta_prior.risk_aligned_spectral_features(
                self.problem, x, adapter=None)
        if variant == "local_risk_kernel":
            return self._local_kernel_features(x)
        if variant == "ordered_cumulative":
            return self.meta_prior.ordered_coordinate_basis_features(
                self.problem, x)
        if variant == "ordered_semiparametric":
            return self._ordered_semiparametric_features(x)
        if variant == "orthogonal_additive":
            base = self.meta_prior.stage1_spectral_features(self.problem, x)
            bank = self.meta_prior.spectral_additive_bank
            if bank is None:
                raise RuntimeError("orthogonal additive expert is unavailable")
            psi = self.meta_prior.risk_coordinate(self.problem, x)
            return np.concatenate([base, bank.transform(psi)])
        raise ValueError(f"unknown fixed task expert basis {variant!r}")

    def features(self, x):
        values = np.asarray(self._raw_features(x), dtype=float).reshape(-1)
        if len(values) != self.feature_dim:
            raise RuntimeError("fixed task expert basis changed dimension")
        return values

    def features_many(self, X):
        if len(X) == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        return np.vstack([self.features(x) for x in X])

    def initial_parametric_coefficients(self, phi, target):
        matrix = np.asarray(phi, dtype=float)
        values = np.asarray(target, dtype=float).reshape(-1)
        if self.variant == "local_risk_kernel":
            penalty = np.eye(matrix.shape[1], dtype=float)
            penalty[0, 0] = 0.0
            return np.linalg.solve(
                matrix.T @ matrix + penalty,
                matrix.T @ values,
            )
        return np.linalg.lstsq(matrix, values, rcond=1e-3)[0]

    def adaptive_sparsity_spec(self, observations=None):
        del observations
        if (
            self.variant not in self.ORDERED_VARIANTS
            or not self.meta_prior.ordered_exposure_adaptive_sparsity
        ):
            return None
        source = self.meta_prior.ordered_coefficient_prior.get(
            self.output_index)
        if source is None:
            return None
        source_pip = np.asarray(source["source_pip"], dtype=float)
        source_slab_scale = np.asarray(
            source["source_slab_scale"], dtype=float)
        allowed_mask = np.asarray(source["allowed_mask"], dtype=bool)
        if self.variant == "ordered_semiparametric":
            residual_dim = int(
                self._ordered_residual_projection["residual_dim"])
            source_pip = np.concatenate([
                source_pip,
                np.full(residual_dim, 0.5, dtype=float),
            ])
            source_slab_scale = np.concatenate([
                source_slab_scale,
                np.ones(residual_dim, dtype=float),
            ])
            allowed_mask = np.concatenate([
                allowed_mask,
                np.ones(residual_dim, dtype=bool),
            ])
        shared_shrinkage_groups = None
        n_local = int(source["always_active_count"])
        if self.meta_prior.ordered_exposure_basis_mode == "diagonal_quadratic":
            interaction_dim = n_local
        else:
            interaction_dim = n_local * (n_local + 1) // 2
        n_shared = max(
            len(source_pip) - n_local - interaction_dim, 0)
        if (
            self.variant == "ordered_cumulative"
            and self.meta_prior.ordered_exposure_group_ridge_learning
        ):
            group_ids = (
                [0] * n_local
                + [1] * interaction_dim
                + [2] * n_shared
            )
            initial_penalty = np.clip(
                1.0 / np.maximum(source_slab_scale ** 2, 1e-8),
                1e-4,
                1000.0,
            )
            return {
                "method": "nested_loo_group_ridge",
                "group_ids": group_ids,
                "penalty_grid": [
                    1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0,
                ],
                "initial_feature_penalty": initial_penalty.tolist(),
                "coordinate_passes": 2,
                "safety_weight": 2.0,
                "residual_floor_scale": float(
                    self.meta_prior.spectral_adaptive_residual_floor_scale),
                "dictionary_dim": int(len(source_pip)),
                "source_domains": list(source["source_domains"]),
                "selection_data": "charged_target_observations",
                "oracle_used": False,
            }
        if (
            self.variant == "ordered_cumulative"
            and self.meta_prior.ordered_exposure_group_shared_shrinkage
        ):
            shared_shrinkage_groups = (
                [-1] * n_local
                + [0] * interaction_dim
                + [1] * n_shared
            )
        return {
            "source_pip": source_pip,
            "source_slab_scale": source_slab_scale,
            "min_pip": float(self.meta_prior.spectral_adaptive_min_pip),
            "max_pip": float(self.meta_prior.spectral_adaptive_max_pip),
            "spike_ratio": float(
                self.meta_prior.spectral_adaptive_spike_ratio),
            "damping": float(self.meta_prior.spectral_adaptive_damping),
            "max_iter": int(self.meta_prior.spectral_adaptive_max_iter),
            "tolerance": float(
                self.meta_prior.spectral_adaptive_tolerance),
            "residual_floor_scale": float(
                self.meta_prior.spectral_adaptive_residual_floor_scale),
            "multiplicity_correction": float(
                self.meta_prior.spectral_adaptive_multiplicity_correction),
            "max_effective_fraction": float(
                self.meta_prior.spectral_adaptive_max_effective_fraction),
            "always_active_count": int(source["always_active_count"]),
            "allowed_mask": allowed_mask.tolist(),
            "dictionary_dim": int(len(source_pip)),
            "source_domains": list(source["source_domains"]),
            "shared_shrinkage_groups": shared_shrinkage_groups,
        }

    def diagnostics(self):
        return {
            "status": "frozen_task_expert",
            "expert": self.variant,
            "output_index": self.output_index,
            "feature_dim": self.feature_dim,
            "local_kernel_pool_size": (
                None if self._local_kernel is None
                else self._local_kernel["pool_size"]
            ),
            "local_kernel_lengthscale": (
                None if self._local_kernel is None
                else self._local_kernel["lengthscale"]
            ),
            "source_only": True,
            "target_oracle_used": False,
            "ordered_basis_mode": (
                self.meta_prior.ordered_exposure_basis_mode
                if self.variant in self.ORDERED_VARIANTS else None
            ),
            "ordered_adaptive_sparsity": bool(
                self.variant in self.ORDERED_VARIANTS
                and self.meta_prior.ordered_exposure_adaptive_sparsity
            ),
            "ordered_semiparametric": bool(
                self.variant == "ordered_semiparametric"),
            "ordered_group_shared_shrinkage": bool(
                self.variant == "ordered_cumulative"
                and self.meta_prior.ordered_exposure_group_shared_shrinkage
            ),
            "ordered_group_ridge_learning": bool(
                self.variant == "ordered_cumulative"
                and self.meta_prior.ordered_exposure_group_ridge_learning
            ),
            "ordered_residual_projection": (
                None
                if self._ordered_residual_projection is None
                else {
                    key: value
                    for key, value in self._ordered_residual_projection.items()
                    if key != "feature_projection"
                }
            ),
        }


class TaskExpertProblemView:
    """Expert-specific cumulative-risk view of an admissible LODO target."""

    ALIGNED_EXPERTS = {"risk_aligned_coordinate", "risk_aligned_spectral"}
    ORDERED_EXPERTS = {"ordered_cumulative", "ordered_semiparametric"}

    def __init__(self, problem, expert_name):
        self.problem = problem
        self.meta_prior = problem.meta_prior
        self.expert_name = str(expert_name)
        self.problem_name = f"{problem.problem_name}_{self.expert_name}"
        self._hvd_shape_reference_mean = {}

    def __getattr__(self, name):
        return getattr(self.problem, name)

    @property
    def aligned(self):
        return self.expert_name in self.ALIGNED_EXPERTS

    def risk_exposures(self, x, output_index=1):
        if self.expert_name in self.ORDERED_EXPERTS:
            return self.meta_prior.ordered_cumulative_risk_exposure(
                self.problem, x, output_index=output_index)
        if self.aligned:
            return self.meta_prior.cumulative_risk_exposure(
                self.problem, x, output_index=output_index)
        return self.meta_prior.risk_exposure(
            self.problem, x, output_index=output_index)

    def cumulative_risk_features(self, x, output_index=1):
        del output_index
        return cumulative_feature_vector(self.risk_exposures(x))

    def cumulative_risk_feature_names(self, output_index=1):
        del output_index
        lo, _ = self.int_bounds()
        return cumulative_feature_names(self.risk_exposures(tuple(lo)))

    def cumulative_risk_provider_status(self):
        return {
            "status": "available",
            "provider": "TaskExpertProblemView",
            "expert": self.expert_name,
            "coordinate": (
                "source_aligned_observable_variance_psi_v=h_v(e)"
                if self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"
                else "frozen_source_learned_ordered_cumulative"
                if self.expert_name in self.ORDERED_EXPERTS
                else "frozen_source_boundary_alignment"
                if self.aligned
                else "frozen_unaligned_source_coordinate"
            ),
            "source_domains": list(self.meta_prior.source_domains),
            "target_data_used": False,
        }

    def mean_risk_coordinate_contract(self):
        separated = self.meta_prior.observable_mean_model is not None
        aligned = bool(
            separated
            and self.meta_prior.observable_mean_mode == "boundary_aligned"
        )
        exchangeable = bool(
            aligned
            and self.meta_prior.observable_mean_descriptor_mode
            == "exchangeable_equivariant"
        )
        return {
            "status": "separated" if separated else "legacy_shared_coordinate",
            "constraint_mean_coordinate": (
                "phi_eq=exchangeable_target_linear_chance_boundary"
                if exchangeable else
                "phi=source_aligned_chance_boundary"
                if aligned
                else "eta=source_observable_constraint_mean_scores"
                if separated else "psi/source_spectral"
            ),
            "cumulative_variance_coordinate": (
                "psi_v=h_v(observable_state_exposure)"
                if self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"
                else "psi=(A,N)"
            ),
            "constraint_mean_input": (
                self.meta_prior.observable_mean_input_mode
                if separated else "legacy"),
            "constraint_mean_descriptor_mode": (
                self.meta_prior.observable_mean_descriptor_mode
                if separated else "legacy"),
            "constraint_mean_feature_mode": (
                self.meta_prior.observable_mean_feature_mode
                if separated else "legacy"),
            "constraint_mean_target_residual_rank": int(
                self.meta_prior.observable_mean_target_residual_rank
                if separated else 0),
            "constraint_mean_target_residual_definition": (
                "unlabeled_target_geometry_orthogonal_to_source_mean_span"
                if separated
                and self.meta_prior.observable_mean_target_residual_rank > 0
                else "disabled"),
            "separate_mean_variance_heads": bool(separated),
            "shared_observable_exposure_input": bool(
                separated
                and self.meta_prior.observable_mean_input_mode
                == "observable_state_exposure"
                and self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"
            ),
            "joined_object": (
                (
                    "mu_g(phi)+sqrt(beta_g)s_g(phi)"
                    if aligned else
                    "mu_g(eta)+sqrt(beta_g)s_g(eta)"
                )
                + "+z_alpha*sqrt(v_C_plus(psi))-tau"
            ),
            "coordinate_definition_uses_target_labels": False,
            "channel_role_alignment_used": bool(
                separated
                and self.meta_prior.observable_mean_descriptor_mode in {
                    "role_aligned",
                    "role_transport",
                    "role_intervention_transport",
                    "role_adaptive_ordered",
                    "role_adaptive_set_invariant",
                }
            ),
            "exchangeable_channel_role_posterior": exchangeable,
            "source_role_identity_transferred": False if exchangeable else None,
            "target_channel_roles_learned_from_charged_data": exchangeable,
            "channel_role_target_matching_uses_labels": False,
            "channel_role_target_matching_uses_oracle": False,
            "channel_role_assignment_posterior": bool(
                self.meta_prior.observable_mean_role_assignment_posterior),
            "channel_role_assignment_hypotheses_use_target_labels": False,
            "channel_role_assignment_weights_use_charged_target_labels": bool(
                self.meta_prior.observable_mean_role_assignment_posterior),
            "channel_role_assignment_weights_use_target_oracle": False,
            "source_observation_mode": self.meta_prior.source_observation_mode,
            "source_design_mode": self.meta_prior.source_design_mode,
            "eta_source_training_target": (
                self.meta_prior.observable_mean_training_target),
            "coefficient_prior_training_target": (
                "constraint_mean" if aligned
                else self.meta_prior.observable_mean_training_target
            ),
            "source_oracle_aided": bool(
                self.meta_prior.training_diagnostics.get(
                    "source_analytic_sigma_used", False)
                or self.meta_prior.training_diagnostics.get(
                    "source_analytic_teacher_used", False)
            ),
        }

    def risk_class(self, x):
        exposure = self.risk_exposures(x)
        return int(np.argmax(exposure.N)) if len(exposure.N) else 0

    def hvd_features(self, x):
        exposure = self.risk_exposures(x)
        return np.concatenate([
            np.array([1.0], dtype=float),
            exposure.A,
            exposure.A ** 2,
            exposure.N,
            exposure.N ** 2,
        ])

    def cumulative_hvd_prior_beta(self, output_index=1, feature_dim=None):
        if self.expert_name == "null_universal" or self.expert_name in self.ORDERED_EXPERTS:
            return None
        coordinate_variant = "aligned" if self.aligned else "unaligned"
        beta = self.meta_prior.cumulative_hvd_prior_beta(
            output_index=output_index,
            feature_dim=feature_dim,
            coordinate_variant=coordinate_variant,
        )
        if beta is None or self.meta_prior.component_stage != "spectral_hvd":
            return beta
        cache_key = (int(output_index), len(beta), coordinate_variant)
        if cache_key not in self._hvd_shape_reference_mean:
            lo, hi = self.int_bounds()
            lo = np.asarray(lo, dtype=int)
            hi = np.asarray(hi, dtype=int)
            expert_seed = sum(
                (index + 1) * ord(char)
                for index, char in enumerate(self.expert_name)
            )
            rng = np.random.default_rng(
                self.meta_prior.seed + 104729 + 7919 * int(output_index)
                + expert_seed)
            rows = [
                tuple(int(value) for value in rng.integers(lo, hi + 1))
                for _ in range(256)
            ]
            features = np.vstack([
                self.cumulative_risk_features(
                    x, output_index=output_index)
                for x in rows
            ])
            reference = float(np.mean(np.maximum(
                features @ beta, 1e-12)))
            self._hvd_shape_reference_mean[cache_key] = max(reference, 1e-12)
        return np.asarray(beta, dtype=float) / self._hvd_shape_reference_mean[
            cache_key]

    def cumulative_hvd_prior_precision(self, output_index=1):
        if self.expert_name == "null_universal":
            return None
        return self.meta_prior.cumulative_hvd_prior_precision(
            output_index,
            coordinate_variant=("aligned" if self.aligned else "unaligned"),
        )

    def cumulative_hvd_prior_components(self, output_index=1, feature_dim=None):
        if self.expert_name == "null_universal" or self.expert_name in self.ORDERED_EXPERTS:
            return None
        coordinate_variant = "aligned" if self.aligned else "unaligned"
        payload = self.meta_prior.cumulative_hvd_prior_components(
            output_index=output_index,
            feature_dim=feature_dim,
            coordinate_variant=coordinate_variant,
        )
        if payload is None or self.meta_prior.component_stage != "spectral_hvd":
            return payload
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        lo, hi = self.int_bounds()
        lo = np.asarray(lo, dtype=int)
        hi = np.asarray(hi, dtype=int)
        expert_seed = sum(
            (index + 1) * ord(char)
            for index, char in enumerate(self.expert_name)
        )
        rng = np.random.default_rng(
            self.meta_prior.seed + 209759 + 7919 * int(output_index)
            + expert_seed)
        rows = [
            tuple(int(value) for value in rng.integers(lo, hi + 1))
            for _ in range(256)
        ]
        features = np.vstack([
            self.cumulative_risk_features(x, output_index=output_index)
            for x in rows
        ])
        references = np.mean(np.maximum(
            features @ coefficients.T, 1e-12), axis=0)
        normalized = coefficients / np.maximum(references[:, None], 1e-12)
        return {
            "coefficients": normalized,
            "domains": list(payload.get("domains", [])),
        }

    def cumulative_hvd_prior_scale_mean(self, output_index=1):
        if self.expert_name == "null_universal":
            return None
        return self.meta_prior.cumulative_hvd_prior_scale_mean(
            output_index,
            coordinate_variant=("aligned" if self.aligned else "unaligned"),
        )

    def cumulative_hvd_prior_upper_scale(self, output_index=1):
        if self.expert_name == "null_universal":
            return None
        return self.meta_prior.cumulative_hvd_prior_upper_scale(
            output_index,
            coordinate_variant=("aligned" if self.aligned else "unaligned"),
        )

    def cumulative_hvd_prior_min_records(self):
        if self.expert_name == "null_universal":
            return None
        return self.meta_prior.cumulative_hvd_prior_min_records()

    def pilot_constraint_guard(self):
        if not self.aligned:
            return 0.0
        alignment = self.meta_prior.risk_subspace_alignment
        if alignment is None:
            return 0.0
        scale = max(
            abs(float(self.tau)),
            float(self.sigma_level),
            1e-6,
        )
        return float(max(alignment.source_residual_guard_ * scale, 0.0))


class MetaPriorProblemAdapter(AdmissibleProblemAdapter):
    """Held-out target adapter using only a frozen source-trained meta-prior."""

    def __init__(
        self,
        base_problem,
        meta_prior: LearnedMetaPrior,
        proposal_pool_size=1024,
        refinement_count=128,
    ):
        super().__init__(base_problem, variant="lodo_meta")
        self.meta_prior = meta_prior
        self.problem_name = f"{base_problem.problem_name}_lodo_meta"
        self.proposal_pool_size = int(proposal_pool_size)
        self.refinement_count = int(refinement_count)
        self.prefer_direct_gpr_basis = bool(
            self.meta_prior.observable_mean_model is not None
            or self.meta_prior.component_stage in {
                "coordinate", "spectral", "spectral_hvd"
            }
        )
        self._gpr_basis_maps = {}
        self._hvd_shape_reference_mean = {}
        self._hvd_shape_reference_pool_size = 256

    def admissibility_audit(self):
        training = self.meta_prior.training_diagnostics
        source_true_outputs = bool(
            training.get("source_analytic_teacher_used", False))
        source_true_sigma = bool(
            training.get("source_analytic_sigma_used", False))
        source_problem_hooks = bool(
            training.get("teacher_record_count", 0))
        out = lodo_meta_prior_audit(
            uses_source_true_outputs=source_true_outputs,
            uses_source_true_sigma=source_true_sigma,
            uses_source_problem_hooks=source_problem_hooks,
            source_observation_mode=training.get(
                "source_observation_mode", "unspecified"),
            source_simulator_calls=training.get(
                "source_simulator_calls", 0),
        ).to_dict()
        provider_boundary = bool(
            "provider_"
            in self.meta_prior.hierarchical_boundary_descriptor_mode
            or self.meta_prior.observable_mean_input_mode
            == "provider_exposure")
        source_oracle_aided = bool(
            source_true_outputs
            or source_true_sigma
            or source_problem_hooks)
        out.update({
            "source_design_mode": training.get(
                "source_design_mode", "random"),
            "source_universal_record_count": int(training.get(
                "source_universal_record_count", 0)),
            "tcb_boundary_descriptor_mode": (
                self.meta_prior.hierarchical_boundary_descriptor_mode),
            "tcb_target_structural_provider_used": provider_boundary,
            "observable_state_exposure_used": bool(
                self.meta_prior.observable_mean_input_mode
                == "observable_state_exposure"
                or self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"),
            "observable_state_mean_head_used": bool(
                self.meta_prior.observable_mean_input_mode
                == "observable_state_exposure"),
            "observable_state_variance_head_used": bool(
                self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"),
            "observable_state_exposure_uses_target_outcomes": False,
            "source_oracle_aided": source_oracle_aided,
            "admissible_strict_lodo": bool(
                not provider_boundary and not source_oracle_aided),
            "admissible_oracle_free_transfer": bool(
                not provider_boundary and not source_oracle_aided),
            "admissible_structure_aware": True,
        })
        if provider_boundary or source_oracle_aided:
            if provider_boundary:
                out["uses_problem_specific_formula"] = True
            out["admissible_mainline"] = False
            reasons = []
            if provider_boundary:
                reasons.append("target CumulativeRiskFeatureProvider")
            if source_oracle_aided:
                reasons.append("analytic source oracle/teacher data")
            out["notes"] = (
                "Privileged transfer track using "
                + " and ".join(reasons)
                + "; report as an upper bound, not oracle-free LODO."
            )
        out["meta_prior"] = self.meta_prior.diagnostics()
        return out

    def observable_boundary_exposure(self, x):
        """Source-frozen exposure available to the oracle-free mean head."""

        return self.meta_prior.risk_exposure(self, x, output_index=1)

    def observable_state_exposure(self, x):
        """Target-observable state/trajectory record with no outcome access."""

        exposure = get_observable_state_exposure(self.base, x)
        if exposure is None:
            raise ValueError(
                "held-out target has no observable state/trajectory exposure"
            )
        return exposure

    def provider_boundary_exposure(self, x):
        """Declared target exposure for structure-aware upper bounds only."""

        exposure = get_risk_exposure(self.base, x, output_index=1)
        if exposure is None:
            raise ValueError("held-out target has no declared risk provider")
        return exposure

    def cumulative_risk_provider_status(self):
        aligned = self.meta_prior.component_stage == "spectral_hvd"
        return {
            "status": "available",
            "provider": "LearnedMetaPrior",
            "coordinate": (
                "source_aligned_observable_variance_psi_v=h_v(e)"
                if self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"
                else "frozen_source_boundary_aligned_psi=(A,N)"
                if aligned
                else "frozen_source_trained_psi=(A,N)"
            ),
            "source_domains": list(self.meta_prior.source_domains),
            "target_data_used": False,
            "unlabeled_target_shape_reference_pool_size": (
                self._hvd_shape_reference_pool_size if aligned else 0
            ),
        }

    def mean_risk_coordinate_contract(self):
        separated = self.meta_prior.observable_mean_model is not None
        aligned = bool(
            separated
            and self.meta_prior.observable_mean_mode == "boundary_aligned"
        )
        exchangeable = bool(
            aligned
            and self.meta_prior.observable_mean_descriptor_mode
            == "exchangeable_equivariant"
        )
        return {
            "status": "separated" if separated else "legacy_shared_coordinate",
            "constraint_mean_coordinate": (
                "phi_eq=exchangeable_target_linear_chance_boundary"
                if exchangeable else
                "phi=source_aligned_chance_boundary"
                if aligned
                else "eta=source_observable_constraint_mean_scores"
                if separated else "psi/source_spectral"
            ),
            "cumulative_variance_coordinate": (
                "psi_v=h_v(observable_state_exposure)"
                if self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"
                else "psi=(A,N)"
            ),
            "constraint_mean_input": (
                self.meta_prior.observable_mean_input_mode
                if separated else "legacy"),
            "constraint_mean_descriptor_mode": (
                self.meta_prior.observable_mean_descriptor_mode
                if separated else "legacy"),
            "constraint_mean_feature_mode": (
                self.meta_prior.observable_mean_feature_mode
                if separated else "legacy"),
            "separate_mean_variance_heads": bool(separated),
            "shared_observable_exposure_input": bool(
                separated
                and self.meta_prior.observable_mean_input_mode
                == "observable_state_exposure"
                and self.meta_prior.observable_variance_input_mode
                == "observable_state_exposure"
            ),
            "joined_object": (
                (
                    "mu_g(phi)+sqrt(beta_g)s_g(phi)"
                    if aligned else
                    "mu_g(eta)+sqrt(beta_g)s_g(eta)"
                )
                + "+z_alpha*sqrt(v_C_plus(psi))-tau"
            ),
            "coordinate_definition_uses_target_labels": False,
            "channel_role_alignment_used": bool(
                separated
                and self.meta_prior.observable_mean_descriptor_mode in {
                    "role_aligned",
                    "role_transport",
                    "role_intervention_transport",
                    "role_adaptive_ordered",
                    "role_adaptive_set_invariant",
                }
            ),
            "exchangeable_channel_role_posterior": exchangeable,
            "source_role_identity_transferred": False if exchangeable else None,
            "target_channel_roles_learned_from_charged_data": exchangeable,
            "channel_role_target_matching_uses_labels": False,
            "channel_role_target_matching_uses_oracle": False,
            "channel_role_assignment_posterior": bool(
                self.meta_prior.observable_mean_role_assignment_posterior),
            "channel_role_assignment_hypotheses_use_target_labels": False,
            "channel_role_assignment_weights_use_charged_target_labels": bool(
                self.meta_prior.observable_mean_role_assignment_posterior),
            "channel_role_assignment_weights_use_target_oracle": False,
            "source_observation_mode": self.meta_prior.source_observation_mode,
            "source_design_mode": self.meta_prior.source_design_mode,
            "eta_source_training_target": (
                self.meta_prior.observable_mean_training_target),
            "coefficient_prior_training_target": (
                "constraint_mean" if aligned
                else self.meta_prior.observable_mean_training_target
            ),
            "source_oracle_aided": bool(
                self.meta_prior.training_diagnostics.get(
                    "source_analytic_sigma_used", False)
                or self.meta_prior.training_diagnostics.get(
                    "source_analytic_teacher_used", False)
            ),
        }

    def risk_exposures(self, x, output_index=1):
        return self.meta_prior.cumulative_risk_exposure(
            self, x, output_index=output_index)

    def risk_class(self, x):
        return self.meta_prior.risk_class(self, x)

    def cumulative_risk_features(self, x, output_index=1):
        return self.meta_prior.cumulative_features(self, x, output_index=output_index)

    def cumulative_risk_feature_names(self, output_index=1):
        del output_index
        return self.meta_prior.cumulative_feature_names()

    def cumulative_hvd_prior_beta(self, output_index=1, feature_dim=None):
        beta = self.meta_prior.cumulative_hvd_prior_beta(
            output_index=output_index,
            feature_dim=feature_dim,
        )
        if beta is None or self.meta_prior.component_stage != "spectral_hvd":
            return beta
        output_index = int(output_index)
        cache_key = (output_index, len(beta))
        if cache_key not in self._hvd_shape_reference_mean:
            rng = np.random.default_rng(
                self.meta_prior.seed + 104729 + 7919 * output_index)
            rows = [
                self.base.sample_random(rng)
                for _ in range(self._hvd_shape_reference_pool_size)
            ]
            features = np.vstack([
                self.meta_prior.cumulative_features(
                    self,
                    x,
                    output_index=output_index,
                )
                for x in rows
            ])
            reference = float(np.mean(np.maximum(
                features @ beta,
                1e-12,
            )))
            self._hvd_shape_reference_mean[cache_key] = max(reference, 1e-12)
        return np.asarray(beta, dtype=float) / self._hvd_shape_reference_mean[
            cache_key]

    def cumulative_hvd_prior_precision(self, output_index=1):
        return self.meta_prior.cumulative_hvd_prior_precision(
            output_index=output_index)

    def cumulative_hvd_prior_components(self, output_index=1, feature_dim=None):
        payload = self.meta_prior.cumulative_hvd_prior_components(
            output_index=output_index,
            feature_dim=feature_dim,
        )
        if payload is None or self.meta_prior.component_stage != "spectral_hvd":
            return payload
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        rng = np.random.default_rng(
            self.meta_prior.seed + 209759 + 7919 * int(output_index))
        rows = [
            self.base.sample_random(rng)
            for _ in range(self._hvd_shape_reference_pool_size)
        ]
        features = np.vstack([
            self.meta_prior.cumulative_features(
                self, x, output_index=output_index)
            for x in rows
        ])
        references = np.mean(np.maximum(
            features @ coefficients.T, 1e-12), axis=0)
        normalized = coefficients / np.maximum(references[:, None], 1e-12)
        return {
            "coefficients": normalized,
            "domains": list(payload.get("domains", [])),
        }

    def cumulative_hvd_prior_scale_mean(self, output_index=1):
        return self.meta_prior.cumulative_hvd_prior_scale_mean(
            output_index=output_index)

    def cumulative_hvd_prior_upper_scale(self, output_index=1):
        return self.meta_prior.cumulative_hvd_prior_upper_scale(
            output_index=output_index)

    def cumulative_hvd_prior_min_records(self):
        return self.meta_prior.cumulative_hvd_prior_min_records()

    def task_sensitivity_prior(self):
        return self.meta_prior.task_sensitivity_prior()

    def task_bias_features(self, x):
        return self.meta_prior.task_bias_features(self, x)

    def task_bias_feature_names(self):
        return self.meta_prior.task_bias_feature_names()

    def hierarchical_boundary_model(self):
        return self.meta_prior.hierarchical_boundary_posterior

    def hierarchical_boundary_descriptor(self, x):
        mode = self.meta_prior.hierarchical_boundary_descriptor_mode
        problem = self.base if "provider_" in mode else self
        return self.meta_prior.boundary_descriptor(problem, x, mode=mode)

    def task_boundary_bracket_candidates(
        self, n=5, rng=None, pool_size=None,
    ):
        return self.meta_prior.source_boundary_bracket_candidates(
            self,
            n=n,
            rng=rng,
            pool_size=(
                self.proposal_pool_size if pool_size is None else pool_size
            ),
        )

    def task_boundary_bracket_diagnostics(self):
        return copy.deepcopy(
            self.meta_prior.alignment_episode_diagnostics.get(
                "last_boundary_bracket",
                {"status": "not_run"},
            )
        )

    def boundary_excitation_candidates(
        self,
        n=1,
        rng=None,
        pool_size=None,
        *,
        include_source_templates=True,
    ):
        """Expose only the domain-generic/source-frozen ``phi`` pool."""

        requested = max(
            int(n),
            int(self.proposal_pool_size if pool_size is None else pool_size),
        )
        return self.meta_prior.boundary_excitation_candidates(
            self,
            n=requested,
            rng=rng,
            include_source_templates=include_source_templates,
        )

    def source_calibrated_recommendation_slack(self):
        return self.meta_prior.source_calibrated_recommendation_slack()

    def _observable_constraint_mean_basis(self):
        if self.meta_prior.observable_mean_model is None:
            return None
        if 1 not in self._gpr_basis_maps:
            self._gpr_basis_maps[1] = ObservableConstraintMeanBasis(
                self.meta_prior, self)
        return self._gpr_basis_maps[1]

    def surrogate_basis_map(self):
        # Recommendation and certification calibration model the constraint
        # boundary, so they must use the same pilot-gated representation as
        # the constraint GPR.  Falling back to an unconditional spectral map
        # here would silently bypass a coordinate gate decision.
        constraint_basis = self._gpr_basis_maps.get(1)
        if constraint_basis is not None:
            return constraint_basis
        if self.meta_prior.observable_mean_model is not None:
            return self._observable_constraint_mean_basis()
        return MetaPriorSurrogateBasis(self.meta_prior, self)

    def gpr_basis_map(self, output_index=0):
        output_index = int(output_index)
        if not self.prefer_direct_gpr_basis:
            return None
        if output_index not in self._gpr_basis_maps:
            if (
                output_index == 1
                and self.meta_prior.observable_mean_model is not None
            ):
                basis = self._observable_constraint_mean_basis()
            elif self.meta_prior.component_enabled("spectral"):
                basis = PilotGatedMetaPriorBasis(
                    self.meta_prior,
                    self,
                    output_index=output_index,
                )
            else:
                basis = MetaPriorSurrogateBasis(self.meta_prior, self)
            self._gpr_basis_maps[output_index] = basis
        return self._gpr_basis_maps[output_index]

    def meta_basis_diagnostics(self):
        return {
            str(index): (
                basis.diagnostics()
                if hasattr(basis, "diagnostics")
                else {
                    "status": "fixed",
                    "selected_basis": "coordinate",
                    "output_index": int(index),
                }
            )
            for index, basis in self._gpr_basis_maps.items()
        }

    def task_posterior_expert_specs(self, include_local_kernel=False):
        """Return only source-identifiable finite experts for this target."""
        specs = [
            {
                "name": "universal_coordinate",
                "basis": "universal_coordinate",
                "variance_mode": "factor",
                "prior_weight": 0.20,
            },
            {
                "name": "null_universal",
                "basis": "null_universal",
                "variance_mode": "pooled",
                "prior_weight": 0.10,
            },
        ]
        if self.meta_prior.stage1_spectral_basis is not None:
            specs.append({
                "name": "source_spectral",
                "basis": "source_spectral",
                "variance_mode": "factor",
                "prior_weight": 0.25,
            })
        ordered_active = (
            self.meta_prior.ordered_exposure_diagnostics.get("status") == "fit")
        latent_ordered = bool(
            ordered_active
            and self.meta_prior.ordered_exposure_latent_structure_selection)
        if self.meta_prior.risk_subspace_alignment is not None:
            if not latent_ordered:
                specs.append({
                    "name": "risk_aligned_coordinate",
                    "basis": "risk_aligned_coordinate",
                    "variance_mode": "factor",
                    "prior_weight": 0.20,
                })
            specs.append({
                "name": "risk_aligned_spectral",
                "basis": "risk_aligned_spectral",
                "variance_mode": "factor",
                "prior_weight": 0.20,
            })
        if self.meta_prior.spectral_additive_bank is not None:
            specs.append({
                "name": "orthogonal_additive",
                "basis": "orthogonal_additive",
                "variance_mode": "factor",
                "prior_weight": 0.05,
            })
        if ordered_active:
            ordered_name = (
                "ordered_semiparametric"
                if self.meta_prior.ordered_exposure_semiparametric_residual
                else "ordered_cumulative"
            )
            specs.append({
                "name": ordered_name,
                "basis": ordered_name,
                "variance_mode": "factor",
                "prior_weight": 0.15,
            })
        if include_local_kernel and not (
            ordered_active
            and self.meta_prior.ordered_exposure_replace_local_kernel
            and not latent_ordered
        ):
            specs.append({
                "name": "local_risk_kernel",
                "basis": "local_risk_kernel",
                "variance_mode": "factor",
                "prior_weight": 0.15,
            })
        total = float(sum(spec["prior_weight"] for spec in specs))
        for spec in specs:
            spec["prior_weight"] = float(spec["prior_weight"] / total)
        return specs

    def task_expert_basis_map(self, expert_name, output_index=0):
        spec = next(
            (
                item for item in self.task_posterior_expert_specs()
                if item["name"] == str(expert_name)
            ),
            None,
        )
        if spec is None and str(expert_name) == "local_risk_kernel":
            spec = {
                "name": "local_risk_kernel",
                "basis": "local_risk_kernel",
            }
        if spec is None:
            raise KeyError(f"unknown task expert {expert_name!r}")
        if spec["basis"] is None:
            return None
        if (
            int(output_index) == 1
            and self.meta_prior.observable_mean_model is not None
        ):
            return self._observable_constraint_mean_basis()
        return FixedTaskExpertBasis(
            self.meta_prior,
            self,
            spec["basis"],
            output_index=output_index,
        )

    def task_expert_problem_view(self, expert_name):
        return TaskExpertProblemView(self, expert_name)

    def task_expert_proposal_candidates(
        self,
        expert_name,
        n=1,
        rng=None,
        pool_size=1024,
    ):
        """Generate source-admissible proposals for one task expert.

        Every branch is frozen before the held-out run.  In particular, this
        method never calls the target problem's ``initial_samples``,
        ``structured_candidates``, state anchors, refinement grid, objective,
        constraint, or variance oracle.
        """
        name = str(expert_name)
        n = max(0, int(n))
        if n == 0:
            return []
        rng = rng or np.random.default_rng(self.meta_prior.seed)
        rows = []
        if name == "universal_coordinate":
            rows.extend(self.meta_prior.universal_expert_candidates(
                self,
                n=n,
                rng=rng,
            ))
        elif name == "null_universal":
            rows.extend(self.sample_random(rng) for _ in range(n))
        elif name == "source_spectral":
            rows.extend(self.meta_prior.alignment_profile_candidates(
                self,
                n=n,
                rng=rng,
            ))
        elif name in {"risk_aligned_coordinate", "risk_aligned_spectral"}:
            if self.meta_prior.alignment_latent_proposal_supported():
                rows.extend(self.meta_prior.alignment_latent_candidates(
                    self,
                    n=n,
                    rng=rng,
                    pool_size=max(int(pool_size), n),
                ))
            else:
                rows.extend(self.meta_prior.alignment_profile_candidates(
                    self,
                    n=n,
                    rng=rng,
                ))
        elif name == "orthogonal_additive":
            rows.extend(self.meta_prior.universal_shape_candidates(
                self,
                n=n,
                rng=rng,
                force=True,
            ))
        elif name == "local_risk_kernel":
            rows.extend(self.meta_prior.proposal_candidates(
                self,
                n=n,
                rng=rng,
                pool_size=max(int(pool_size), n),
            ))
        elif name in {"ordered_cumulative", "ordered_semiparametric"}:
            rows.extend(self.meta_prior.proposal_candidates(
                self,
                n=n,
                rng=rng,
                pool_size=max(int(pool_size), n),
            ))
        else:
            raise KeyError(f"unknown task proposal expert {name!r}")

        rows = unique_candidates(rows)
        attempts = 0
        while len(rows) < n and attempts < 8 * n:
            rows.append(tuple(self.sample_random(rng)))
            rows = unique_candidates(rows)
            attempts += 1
        return rows[:n]

    def task_initial_universal_candidates(self, n=1, rng=None, pool_size=1024):
        del pool_size
        return self.meta_prior.initial_universal_candidates(
            self,
            n=n,
            rng=rng,
        )

    def frozen_source_consensus_candidates(self):
        """Expose the complete frozen shortlist to every KG/recommendation pool."""

        return self.meta_prior.source_consensus_template_candidates(
            self,
            n=len(self.meta_prior.source_consensus_templates),
            randomized=False,
        )

    def frozen_source_coverage_candidates(self, n=0):
        """Expose the pre-registered target design, without target labels."""

        return self.meta_prior.source_coverage_candidates(self, n=n)

    def pilot_constraint_guard(self):
        basis = self._gpr_basis_maps.get(1)
        if basis is None or not hasattr(basis, "certification_guard"):
            return 0.0
        return max(float(basis.certification_guard()), 0.0)

    def source_mean_prior_predict_many(self, xs, output_index=1):
        return self.meta_prior.source_mean_prior_predict_many(
            self,
            xs,
            output_index=output_index,
        )

    def source_mean_prior_sigma(self, output_index=1):
        return self.meta_prior.source_mean_prior_sigma(output_index=output_index)

    def hvd_features(self, x):
        return self.meta_prior.hvd_features(self, x)

    def hvd_residual_variance_cap(self, output_index=0):
        del output_index
        return float(8.0 * max(float(self.sigma_level), 1e-8) ** 2)

    def initial_samples(self, n=5, rng=None):
        return self.meta_prior.proposal_candidates(
            self,
            n=n,
            rng=rng,
            pool_size=self.proposal_pool_size,
        )

    def state_anchor_points(self, n=10, rng=None):
        return self.meta_prior.state_anchor_points(n=n, rng=rng)

    def inverse_state_anchor(self, anchor, rng=None, n=1):
        return self.meta_prior.inverse_state_anchor(
            self,
            anchor,
            rng=rng,
            n=n,
            pool_size=self.proposal_pool_size,
        )

    def recommendation_refinement_candidates(self):
        return self.meta_prior.proposal_candidates(
            self,
            n=self.refinement_count,
            rng=np.random.default_rng(self.meta_prior.seed + 7919),
            pool_size=max(self.proposal_pool_size, 4 * self.refinement_count),
        )
