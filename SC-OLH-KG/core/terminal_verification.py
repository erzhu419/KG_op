"""Method-independent frozen-policy terminal verification."""

from __future__ import annotations

from dataclasses import asdict
import copy

import numpy as np

from core.certification import (
    gaussian_quantile_tolerance_certificate,
    gaussian_replication_chance_certificate,
)
from core.cumulative_risk import get_risk_exposure
from core.designs import integer_design_fingerprint


TERMINAL_VERIFICATION_STREAM_TAG = 0x56455249
TERMINAL_SHORTLIST_VERIFICATION_STREAM_TAG = 0x56534C54


def _verification_method(method):
    normalized = str(
        method or "component_bonferroni"
    ).strip().lower().replace("-", "_")
    aliases = {
        "component_bonferroni": "component_bonferroni",
        "student_t_chi_square": "component_bonferroni",
        "normal_quantile_tolerance": "normal_quantile_tolerance",
        "noncentral_t_tolerance": "normal_quantile_tolerance",
    }
    if normalized not in aliases:
        raise ValueError(
            "terminal verification method must be component_bonferroni "
            "or normal_quantile_tolerance")
    normalized = aliases[normalized]
    label = (
        "gaussian_noncentral_t_tolerance"
        if normalized == "normal_quantile_tolerance"
        else "gaussian_student_t_chi_square"
    )
    return normalized, label


def verify_frozen_policy(
    problem,
    point,
    *,
    seed,
    search_evaluation_count,
    verification_budget,
    delta,
    candidate_index=0,
    method="normal_quantile_tolerance",
    mean_delta_fraction=0.5,
):
    """Verify one pre-frozen policy on an independent deterministic stream."""

    budget = int(verification_budget)
    if budget < 0:
        raise ValueError("terminal verification budget must be nonnegative")
    candidate_index = int(candidate_index)
    if candidate_index < 0:
        raise ValueError(
            "terminal verification candidate index must be nonnegative")
    method_mode, method_label = _verification_method(method)
    point = tuple(int(value) for value in point)
    stream_tag = (
        TERMINAL_VERIFICATION_STREAM_TAG
        if candidate_index == 0
        else TERMINAL_SHORTLIST_VERIFICATION_STREAM_TAG
    )
    base = {
        "enabled": bool(budget > 0),
        "status": "disabled" if budget == 0 else "pending",
        "method": method_label,
        "method_mode": method_mode,
        "policy_frozen_before_verification": True,
        "search_samples_reused": False,
        "posterior_updated_from_verification": False,
        "verification_budget": budget,
        "search_evaluation_count": int(search_evaluation_count),
        "total_evaluation_count": int(search_evaluation_count) + budget,
        "policy_fingerprint": integer_design_fingerprint([point]),
        "delta": float(delta),
        "alpha": float(problem.alpha),
        "tau": float(problem.tau),
        "mean_delta_fraction": float(mean_delta_fraction),
        "candidate_index": candidate_index,
        "target_oracle_used": False,
    }
    if budget == 0:
        return base

    noise_model = str(getattr(
        problem, "simulation_noise_model", "unknown"))
    if noise_model != "iid_gaussian":
        raise RuntimeError(
            "terminal Gaussian verification requires an iid_gaussian "
            "simulation_noise_model")

    values = []
    for replication in range(budget):
        if candidate_index == 0:
            seed_components = [
                int(seed),
                int(replication),
                TERMINAL_VERIFICATION_STREAM_TAG,
            ]
        else:
            seed_components = [
                int(seed),
                candidate_index,
                int(replication),
                TERMINAL_SHORTLIST_VERIFICATION_STREAM_TAG,
            ]
        rng = np.random.default_rng(np.random.SeedSequence(seed_components))
        values.append(np.asarray(problem.simulate(point, rng), dtype=float))
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise RuntimeError(
            "terminal verification requires objective and constraint outputs")
    if method_mode == "normal_quantile_tolerance":
        certificate = gaussian_quantile_tolerance_certificate(
            samples[:, 1],
            tau=float(problem.tau),
            alpha=float(problem.alpha),
            delta=float(delta),
        )
    else:
        certificate = gaussian_replication_chance_certificate(
            samples[:, 1],
            tau=float(problem.tau),
            alpha=float(problem.alpha),
            delta=float(delta),
            mean_delta_fraction=float(mean_delta_fraction),
        )
    return {
        **base,
        **asdict(certificate),
        "status": str(certificate.status),
        "objective_sample_mean": float(np.mean(samples[:, 0])),
        "objective_sample_std": (
            None
            if len(samples) < 2
            else float(np.std(samples[:, 0], ddof=1))
        ),
        "noise_model": noise_model,
        "verification_stream_tag": int(stream_tag),
        "verification_samples_logged": False,
        "certification_scope": (
            "fixed_terminal_policy_independent_replications"),
        "recommendation_changed": False,
        "target_oracle_used": False,
    }


def verify_frozen_shortlist(
    problem,
    frozen_shortlist,
    *,
    seed,
    search_evaluation_count,
    candidate_budgets=(80, 96),
    familywise_delta=0.05,
    method="normal_quantile_tolerance",
    mean_delta_fraction=0.5,
    shortlist_mode="posterior_primary_safe_interior",
):
    """Deploy the first independently certified policy in a frozen order."""

    records = [copy.deepcopy(record) for record in frozen_shortlist]
    points = [
        tuple(int(value) for value in record["point"])
        for record in records
    ]
    budgets = [int(value) for value in candidate_budgets]
    if not points:
        raise ValueError("terminal verification shortlist cannot be empty")
    if len(points) != len(set(points)):
        raise ValueError("terminal verification shortlist must be unique")
    if len(points) != len(budgets):
        raise ValueError(
            "terminal verification shortlist and budget lengths differ")
    if any(value < 0 for value in budgets):
        raise ValueError("terminal verification budgets must be nonnegative")
    familywise_delta = float(familywise_delta)
    if not 0.0 < familywise_delta < 1.0:
        raise ValueError("familywise delta must lie strictly between 0 and 1")
    per_candidate_delta = familywise_delta / float(len(points))

    attempts = []
    deployed = points[0]
    selected_rank = None
    if budgets[0] > 0:
        for candidate_index, (point, budget) in enumerate(
            zip(points, budgets)
        ):
            attempt = verify_frozen_policy(
                problem,
                point,
                seed=seed,
                search_evaluation_count=search_evaluation_count,
                verification_budget=budget,
                delta=per_candidate_delta,
                candidate_index=candidate_index,
                method=method,
                mean_delta_fraction=mean_delta_fraction,
            )
            attempts.append(attempt)
            if bool(attempt.get("certified", False)):
                deployed = point
                selected_rank = candidate_index + 1
                break

    actual_budget = int(sum(
        int(attempt.get("verification_budget", 0))
        for attempt in attempts
    ))
    _, method_label = _verification_method(method)
    certified = selected_rank is not None
    return deployed, {
        "enabled": bool(budgets[0] > 0),
        "status": (
            "certified"
            if certified else (
                "disabled" if budgets[0] == 0 else "abstained_uncertified"
            )
        ),
        "method": f"ordered_frozen_shortlist_{method_label}",
        "protocol": "ordered_frozen_shortlist",
        "shortlist_mode": str(shortlist_mode),
        "policy_frozen_before_verification": True,
        "shortlist_frozen_before_verification": True,
        "search_samples_reused": False,
        "posterior_updated_from_verification": False,
        "verification_budget": actual_budget,
        "verification_budget_per_candidate": budgets[0],
        "fallback_verification_budget": (
            budgets[1] if len(budgets) > 1 else budgets[0]),
        "candidate_verification_budgets": budgets,
        "max_verification_budget": int(sum(budgets)),
        "search_evaluation_count": int(search_evaluation_count),
        "total_evaluation_count": int(search_evaluation_count) + actual_budget,
        "familywise_delta": familywise_delta,
        "per_candidate_delta": per_candidate_delta,
        "mean_delta_fraction": float(mean_delta_fraction),
        "frozen_shortlist_size": len(records),
        "frozen_shortlist": records,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "certified": certified,
        "selected_shortlist_rank": selected_rank,
        "primary_policy_fingerprint": integer_design_fingerprint([points[0]]),
        "deployed_policy_fingerprint": integer_design_fingerprint([deployed]),
        "recommendation_changed": bool(deployed != points[0]),
        "target_oracle_used": False,
        "certification_scope": (
            "familywise_frozen_ordered_terminal_shortlist"),
        "verification_samples_logged": False,
    }


def cumulative_risk_coordinate(problem, point, *, require_provider=True):
    """Return the common observable cumulative-risk coordinate."""

    exposure = get_risk_exposure(problem, point, output_index=1)
    if exposure is not None:
        coordinate = np.concatenate([
            np.asarray(exposure.A, dtype=float).reshape(-1),
            np.asarray(exposure.N, dtype=float).reshape(-1),
        ])
        if coordinate.size > 0 and np.all(np.isfinite(coordinate)):
            return coordinate, "cumulative_risk_psi=(A,N)"
    if require_provider:
        raise RuntimeError(
            "terminal safe-interior selection requires a finite "
            "cumulative-risk provider coordinate")
    lo, hi = problem.int_bounds()
    lo = np.asarray(lo, dtype=float)
    scale = np.maximum(np.asarray(hi, dtype=float) - lo, 1.0)
    coordinate = (
        np.asarray(point, dtype=float).reshape(-1) - lo
    ) / scale
    return coordinate, "normalized_policy_fallback"


def select_posterior_safe_interior(
    problem,
    primary,
    initial_points,
    violation_probability,
    *,
    probability_slack=0.05,
    require_provider=True,
):
    """Select a posterior-safe, cumulative-risk-diverse frozen support."""

    slack = float(probability_slack)
    if not np.isfinite(slack) or not 0.0 <= slack <= 1.0:
        raise ValueError(
            "terminal safe-interior probability slack must lie in [0, 1]")
    unique = []
    seen = set()
    for point in initial_points:
        point = tuple(int(value) for value in point)
        if point not in seen:
            seen.add(point)
            unique.append(point)
    primary = tuple(int(value) for value in primary)
    alternatives = [point for point in unique if point != primary]
    if not alternatives:
        raise RuntimeError(
            "terminal safe-interior selection requires a distinct "
            "initial-atlas candidate")
    probability = np.asarray(
        violation_probability, dtype=float).reshape(-1)
    if len(probability) != len(unique) or not np.all(np.isfinite(probability)):
        raise RuntimeError(
            "terminal safe-interior selection requires finite posterior "
            "violation probabilities")
    minimum = float(np.min(probability))
    eligible = [
        index for index, point in enumerate(unique)
        if point != primary and float(probability[index]) <= minimum + slack
    ]
    if not eligible:
        eligible = [
            min(
                (
                    index for index, point in enumerate(unique)
                    if point != primary
                ),
                key=lambda index: (float(probability[index]), int(index)),
            )
        ]

    points_for_scale = list(unique)
    if primary not in points_for_scale:
        points_for_scale.append(primary)
    coordinates = []
    sources = []
    for point in points_for_scale:
        coordinate, source = cumulative_risk_coordinate(
            problem,
            point,
            require_provider=require_provider,
        )
        coordinates.append(np.asarray(coordinate, dtype=float))
        sources.append(source)
    if len({len(value) for value in coordinates}) != 1:
        raise RuntimeError(
            "terminal safe-interior coordinates have inconsistent dimensions")
    matrix = np.vstack(coordinates)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - center) / scale
    primary_index = points_for_scale.index(primary)
    distance = np.sqrt(np.mean(
        (
            standardized[: len(unique)]
            - standardized[primary_index][None, :]
        ) ** 2,
        axis=1,
    ))
    selected_index = max(
        eligible,
        key=lambda index: (
            float(distance[index]),
            -float(probability[index]),
            -int(index),
        ),
    )
    return {
        "point": unique[selected_index],
        "selection_contract": (
            "posterior_violation_sublevel_maximin_cumulative_risk"),
        "candidate_universe": "frozen_initial_atlas",
        "candidate_universe_size": len(unique),
        "eligible_count": len(eligible),
        "probability_slack": slack,
        "minimum_posterior_violation_probability": minimum,
        "selected_posterior_violation_probability": float(
            probability[selected_index]),
        "selected_standardized_coordinate_distance": float(
            distance[selected_index]),
        "coordinate_source": str(sources[selected_index]),
        "coordinate_dimension": int(matrix.shape[1]),
        "target_labels_used": False,
        "target_oracle_used": False,
        "verification_samples_used": False,
    }
