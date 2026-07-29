"""Numerical bridge for the source-to-target proposal coverage theorem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ProposalCoverageInputs:
    """Auditable finite-task quantities used by the Lean theorem."""

    source_miss: float
    domain_shift: float
    effective_dim: float
    log_library: float
    inverse_confidence_log: float
    source_samples: int
    n0: int

    def validate(self):
        values = asdict(self)
        for name, value in values.items():
            if name in {"source_samples", "n0"}:
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(self.source_miss) <= 1.0:
            raise ValueError("source_miss must lie in [0, 1]")
        if float(self.domain_shift) < 0.0:
            raise ValueError("domain_shift must be nonnegative")
        if float(self.effective_dim) < 0.0:
            raise ValueError("effective_dim must be nonnegative")
        if float(self.log_library) < 0.0:
            raise ValueError("log_library must be nonnegative")
        if float(self.inverse_confidence_log) < 0.0:
            raise ValueError("inverse_confidence_log must be nonnegative")
        if int(self.source_samples) <= 0:
            raise ValueError("source_samples must be positive")
        if int(self.n0) <= 0:
            raise ValueError("n0 must be positive")
        return self


def effective_dimension_transfer_radius(
    effective_dim,
    log_library,
    inverse_confidence_log,
    source_samples,
):
    """Match ``effectiveDimensionTransferRadius`` in Lean exactly."""

    source_samples = int(source_samples)
    if source_samples <= 0:
        raise ValueError("source_samples must be positive")
    return (
        float(effective_dim) * float(log_library)
        + float(inverse_confidence_log)
    ) / source_samples


def feasible_mass_lower_bound(inputs):
    """Return the conservative one-draw heldout feasible-mass lower bound."""

    inputs = inputs.validate()
    radius = effective_dimension_transfer_radius(
        inputs.effective_dim,
        inputs.log_library,
        inputs.inverse_confidence_log,
        inputs.source_samples,
    )
    return max(
        0.0,
        1.0 - (
            float(inputs.source_miss)
            + float(inputs.domain_shift)
            + radius
        ),
    )


def iid_hit_probability_lower_bound(feasible_mass_lower, n0):
    """Return ``1 - (1 - p_lower)^n0`` with probability-domain checks."""

    feasible_mass_lower = float(feasible_mass_lower)
    n0 = int(n0)
    if not 0.0 <= feasible_mass_lower <= 1.0:
        raise ValueError("feasible_mass_lower must lie in [0, 1]")
    if n0 <= 0:
        raise ValueError("n0 must be positive")
    return 1.0 - (1.0 - feasible_mass_lower) ** n0


def proposal_coverage_audit(inputs):
    """Serialize the optional IID randomized-proposal theorem inputs."""

    inputs = inputs.validate()
    radius = effective_dimension_transfer_radius(
        inputs.effective_dim,
        inputs.log_library,
        inputs.inverse_confidence_log,
        inputs.source_samples,
    )
    feasible_mass_lower = feasible_mass_lower_bound(inputs)
    hit_lower = iid_hit_probability_lower_bound(feasible_mass_lower, inputs.n0)
    return {
        "theory_contract_id": "source_target_proposal_coverage_v1",
        "inputs": asdict(inputs),
        "effective_dimension_transfer_radius": radius,
        "single_draw_feasible_mass_lower": feasible_mass_lower,
        "n0_at_least_one_hit_probability_lower": hit_lower,
        "formula": "1-(1-p_lower)^n0",
        "proposal_contract": "iid_randomized_draws",
        "matches_deterministic_risk_objective_atlas": False,
        "target_outcomes_used_to_fit_proposal": False,
        "requires_source_only_domain_shift_calibration": True,
    }


def deterministic_atlas_coverage_audit(
    inputs,
    *,
    atlas_size,
    unique_design_fingerprints,
):
    """Bridge the deployed deterministic finite-atlas coverage theorem."""

    inputs = inputs.validate()
    atlas_size = int(atlas_size)
    unique_design_fingerprints = int(unique_design_fingerprints)
    if atlas_size <= 0 or atlas_size > int(inputs.n0):
        raise ValueError("atlas_size must lie in [1, n0]")
    if unique_design_fingerprints != 1:
        raise ValueError(
            "deterministic atlas must have one frozen design fingerprint")
    radius = effective_dimension_transfer_radius(
        inputs.effective_dim,
        inputs.log_library,
        inputs.inverse_confidence_log,
        inputs.source_samples,
    )
    feasible_mass_lower = feasible_mass_lower_bound(inputs)
    return {
        "theory_contract_id": "source_target_finite_atlas_coverage_v1",
        "lean_theorem": (
            "SCOLHKG.Real."
            "paper_frontend_atlas_coverage_and_certificate"
        ),
        "proposal_contract": "deterministic_finite_atlas",
        "independent_draws_assumed": False,
        "atlas_size": atlas_size,
        "n0_budget": int(inputs.n0),
        "unique_design_fingerprints": unique_design_fingerprints,
        "inputs": asdict(inputs),
        "effective_dimension_transfer_radius": radius,
        "finite_atlas_feasible_mass_lower": feasible_mass_lower,
        "positive_mass_certifies_feasible_support_member": bool(
            feasible_mass_lower > 0.0),
        "target_outcomes_used_to_fit_proposal": False,
        "requires_source_only_domain_shift_calibration": True,
    }


def normalized_ranks(values):
    """Stable percentile ranks in ``[0, 1]`` with lower values preferred."""

    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("rank values must be nonempty and finite")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    if len(values) > 1:
        ranks /= float(len(values) - 1)
    return ranks


def source_only_rank_alignment_calibration(rank_vectors, *, delta=0.05):
    """Conservative source-only alignment envelope on a shared library.

    The empirical term is the largest pairwise sup-norm difference between
    source-domain rank vectors.  A simultaneous DKW radius is added for every
    source domain.  Extending this source envelope to an unseen task remains
    an explicit exchangeability/domain-envelope assumption.
    """

    vectors = [np.asarray(row, dtype=float).reshape(-1)
               for row in rank_vectors]
    if len(vectors) < 2:
        raise ValueError("rank calibration needs at least two source domains")
    sizes = {len(row) for row in vectors}
    if len(sizes) != 1 or next(iter(sizes)) <= 1:
        raise ValueError("source rank vectors need one shared finite library")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if any(
        not np.all(np.isfinite(row))
        or np.any(row < 0.0)
        or np.any(row > 1.0)
        for row in vectors
    ):
        raise ValueError("source ranks must be finite and lie in [0, 1]")
    pairwise = max(
        float(np.max(np.abs(vectors[left] - vectors[right])))
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    )
    sample_count = next(iter(sizes))
    dkw_per_domain = math.sqrt(
        math.log(2.0 * len(vectors) / float(delta))
        / (2.0 * sample_count)
    )
    bound = min(1.0, pairwise + 2.0 * dkw_per_domain)
    return {
        "contract_id": "source_only_rank_alignment_envelope_v1",
        "source_domain_count": len(vectors),
        "shared_library_size": sample_count,
        "delta": float(delta),
        "empirical_pairwise_sup_rank_error": pairwise,
        "dkw_radius_per_domain": dkw_per_domain,
        "source_only_alignment_error_bound": bound,
        "target_labels_used": False,
        "heldout_extension_assumption": (
            "source-source rank envelope upper-bounds heldout rank shift"
        ),
    }


def rank_aligned_atlas_coverage_audit(
    source_rank,
    target_rank,
    target_feasible,
    atlas_indices,
    *,
    source_only_alignment_bound,
):
    """Post-run audit of the implementation-matched rank-atlas theorem."""

    source_rank = np.asarray(source_rank, dtype=float).reshape(-1)
    target_rank = np.asarray(target_rank, dtype=float).reshape(-1)
    target_feasible = np.asarray(target_feasible, dtype=bool).reshape(-1)
    if not (
        len(source_rank) == len(target_rank) == len(target_feasible)
        and len(source_rank) > 0
    ):
        raise ValueError("rank audit arrays must have one common size")
    if any(
        not np.all(np.isfinite(row))
        or np.any(row < 0.0)
        or np.any(row > 1.0)
        for row in (source_rank, target_rank)
    ):
        raise ValueError("risk ranks must lie in [0, 1]")
    atlas = np.asarray(sorted(set(map(int, atlas_indices))), dtype=int)
    if len(atlas) == 0 or np.any(atlas < 0) or np.any(atlas >= len(source_rank)):
        raise ValueError("atlas indices are outside the finite library")
    bound = float(source_only_alignment_bound)
    if not 0.0 <= bound <= 1.0:
        raise ValueError("source-only alignment bound must lie in [0, 1]")
    observed_alignment = float(np.max(np.abs(source_rank - target_rank)))
    best_atlas_rank = float(np.min(source_rank[atlas]))
    cover_error = float(max(
        0.0,
        max(best_atlas_rank - float(value) for value in source_rank),
    ))
    if np.any(target_feasible):
        safe_ranks = target_rank[target_feasible]
        threshold = float(np.max(safe_ranks))
        interior_depth = float(threshold - np.min(safe_ranks))
    else:
        threshold = None
        interior_depth = 0.0
    required_depth = float(2.0 * bound + cover_error)
    alignment_covered = bool(observed_alignment <= bound + 1e-12)
    theorem_conditions_hold = bool(
        alignment_covered
        and threshold is not None
        and interior_depth + 1e-12 >= required_depth
    )
    observed_atlas_hit = bool(np.any(target_feasible[atlas]))
    return {
        "theory_contract_id": (
            "source_target_rank_aligned_atlas_coverage_v1"),
        "lean_theorem": (
            "SCOLHKG.Real."
            "paper_frontend_rank_aligned_atlas_and_certificate"
        ),
        "finite_library_size": len(source_rank),
        "atlas_size": len(atlas),
        "source_only_alignment_error_bound": bound,
        "post_run_observed_target_alignment_error": observed_alignment,
        "alignment_bound_covered_target": alignment_covered,
        "one_sided_source_rank_atlas_cover_error": cover_error,
        "target_safe_rank_threshold": threshold,
        "post_run_target_safe_rank_interior_depth": interior_depth,
        "required_safe_rank_depth": required_depth,
        "theorem_conditions_hold": theorem_conditions_hold,
        "observed_atlas_contains_feasible": observed_atlas_hit,
        "target_truth_used_post_run_only": True,
        "target_truth_used_for_proposal_or_selection": False,
    }


def geometric_atlas_coverage_audit(
    atlas_coordinate,
    source_support_coordinate,
    target_library_coordinate,
    target_feasible,
    *,
    atlas_target_indices,
    target_margin=None,
):
    """Finite-library audit of the geometric maximin coverage theorem."""

    atlas = np.asarray(atlas_coordinate, dtype=float)
    source = np.asarray(source_support_coordinate, dtype=float)
    target = np.asarray(target_library_coordinate, dtype=float)
    feasible = np.asarray(target_feasible, dtype=bool).reshape(-1)
    if (
        atlas.ndim != 2
        or source.ndim != 2
        or target.ndim != 2
        or atlas.shape[1] != source.shape[1]
        or atlas.shape[1] != target.shape[1]
        or len(target) != len(feasible)
        or min(len(atlas), len(source), len(target)) <= 0
        or not all(np.all(np.isfinite(row))
                   for row in (atlas, source, target))
    ):
        raise ValueError("geometric audit coordinates are inconsistent")
    atlas_target_indices = np.asarray(
        sorted(set(map(int, atlas_target_indices))), dtype=int)
    if (
        len(atlas_target_indices) != len(atlas)
        or np.any(atlas_target_indices < 0)
        or np.any(atlas_target_indices >= len(target))
    ):
        raise ValueError("atlas target indexes are invalid or nonunique")
    source_to_atlas = np.linalg.norm(
        source[:, None, :] - atlas[None, :, :], axis=2)
    cover_radius = float(np.max(np.min(source_to_atlas, axis=1)))
    infeasible_indexes = np.flatnonzero(~feasible)
    candidate_rows = []
    for center in np.flatnonzero(feasible):
        support_shift = float(np.min(np.linalg.norm(
            source - target[center][None, :], axis=1)))
        if len(infeasible_indexes):
            first_infeasible_distance = float(np.min(np.linalg.norm(
                target[infeasible_indexes] - target[center][None, :],
                axis=1,
            )))
            safe_radius = float(np.nextafter(
                first_infeasible_distance, -np.inf))
        else:
            first_infeasible_distance = None
            safe_radius = float("inf")
        slack = safe_radius - cover_radius - support_shift
        candidate_rows.append({
            "target_library_index": int(center),
            "source_support_shift": support_shift,
            "first_infeasible_distance": first_infeasible_distance,
            "finite_library_safe_radius": safe_radius,
            "coverage_slack": slack,
        })
    best = (
        None
        if not candidate_rows
        else max(candidate_rows, key=lambda row: row["coverage_slack"])
    )
    theorem_holds = bool(
        best is not None and best["coverage_slack"] >= 0.0)
    atlas_hit = bool(np.any(feasible[atlas_target_indices]))
    lipschitz = None
    if target_margin is not None:
        margins = np.asarray(target_margin, dtype=float).reshape(-1)
        if len(margins) != len(target) or not np.all(np.isfinite(margins)):
            raise ValueError("target margins do not match the audit library")
        distance = np.linalg.norm(
            target[:, None, :] - target[None, :, :], axis=2)
        margin_increase = margins[:, None] - margins[None, :]
        mask = distance > 1e-12
        empirical_L = (
            0.0
            if not np.any(mask)
            else float(max(
                0.0,
                np.max(margin_increase[mask] / distance[mask]),
            ))
        )
        lipschitz_rows = []
        for center in np.flatnonzero(feasible):
            source_distance = np.linalg.norm(
                source - target[center][None, :], axis=1)
            total_bridge = float(cover_radius + np.min(source_distance))
            safe_depth = float(max(0.0, -margins[center]))
            allowed_L = (
                float("inf")
                if total_bridge <= 0.0
                else safe_depth / total_bridge
            )
            lipschitz_rows.append({
                "target_library_index": int(center),
                "safe_depth": safe_depth,
                "cover_plus_support_shift": total_bridge,
                "max_admissible_lipschitz_constant": allowed_L,
                "empirical_library_lipschitz_slack": (
                    safe_depth - empirical_L * total_bridge
                ),
            })
        best_lipschitz = (
            None
            if not lipschitz_rows
            else max(
                lipschitz_rows,
                key=lambda row: row[
                    "empirical_library_lipschitz_slack"],
            )
        )
        lipschitz = {
            "empirical_finite_library_one_sided_lipschitz": empirical_L,
            "best_safe_center": best_lipschitz,
            "finite_library_lipschitz_condition_holds": bool(
                best_lipschitz is not None
                and best_lipschitz[
                    "empirical_library_lipschitz_slack"] >= 0.0
            ),
            "global_lipschitz_upper_bound_certified": False,
            "global_condition_status": "requires_problem_or_simulator_bound",
        }
    centered_source = source - np.mean(source, axis=0, keepdims=True)
    effective_rank = int(np.linalg.matrix_rank(
        centered_source, tol=1e-10))
    return {
        "theory_contract_id": (
            "source_target_geometric_atlas_coverage_v1"),
        "lean_theorem": (
            "SCOLHKG.Real."
            "paper_frontend_geometric_atlas_and_certificate"
        ),
        "atlas_size": len(atlas),
        "source_support_size": len(source),
        "target_audit_library_size": len(target),
        "coordinate_dimension": int(atlas.shape[1]),
        "empirical_effective_coordinate_rank": effective_rank,
        "source_support_atlas_cover_radius": cover_radius,
        "best_safe_center": best,
        "finite_library_theorem_conditions_hold": theorem_holds,
        "observed_atlas_contains_feasible": atlas_hit,
        "finite_library_safe_ball_only": True,
        "global_safe_ball_requires_separate_lipschitz_or_exhaustive_bound": True,
        "lipschitz_audit": lipschitz,
        "target_truth_used_post_run_only": True,
        "target_truth_used_for_proposal_or_selection": False,
    }
