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


def parse_verification_candidate_budgets(value, *, default):
    """Parse one explicit ordered shortlist budget contract."""

    if value is None or str(value).strip() == "":
        budgets = tuple(int(item) for item in default)
    elif isinstance(value, str):
        budgets = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    else:
        budgets = tuple(int(item) for item in value)
    if not budgets:
        raise ValueError("terminal verification budgets cannot be empty")
    if any(item < 0 for item in budgets):
        raise ValueError(
            "terminal verification budgets must be nonnegative")
    return budgets


def select_objective_verification_challenger(
    candidates,
    objective_mean,
    probability_violation,
    *,
    maximum_violation_probability=0.5,
):
    """Select the best objective among posterior-median feasible policies.

    A violation probability at most one half is equivalent to a nonpositive
    posterior median chance margin under the Gaussian decision model. This
    admits candidates whose epistemic uncertainty prevents a paper-grade
    certificate, while the independent verifier remains solely responsible
    for deployment safety.
    """

    points = [tuple(int(value) for value in point) for point in candidates]
    objective = np.asarray(objective_mean, dtype=float).reshape(-1)
    violation = np.asarray(probability_violation, dtype=float).reshape(-1)
    if len(points) != len(objective) or len(points) != len(violation):
        raise ValueError(
            "candidate, objective, and violation arrays must have equal length")
    threshold = float(maximum_violation_probability)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "objective challenger violation threshold must lie in (0, 1]")
    eligible = (
        np.isfinite(objective)
        & np.isfinite(violation)
        & (violation <= threshold)
    )
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return {
            "status": "no_posterior_median_feasible_candidate",
            "point": None,
            "eligible_count": 0,
            "maximum_violation_probability": threshold,
            "selection_contract": (
                "posterior_median_feasible_minimum_posterior_objective"),
            "verification_required_for_deployment": True,
            "target_labels_used": False,
            "target_oracle_used": False,
            "verification_samples_used": False,
        }
    selected = int(indices[np.argmin(objective[indices])])
    return {
        "status": "selected",
        "point": points[selected],
        "eligible_count": int(len(indices)),
        "maximum_violation_probability": threshold,
        "selected_posterior_objective": float(objective[selected]),
        "selected_probability_violation": float(violation[selected]),
        "selection_contract": (
            "posterior_median_feasible_minimum_posterior_objective"),
        "verification_required_for_deployment": True,
        "target_labels_used": False,
        "target_oracle_used": False,
        "verification_samples_used": False,
    }


def build_verification_aware_shortlist(
    problem,
    primary,
    candidates,
    objective_mean,
    probability_violation,
    *,
    shortlist_size=3,
    maximum_violation_probability=0.5,
    probability_slack=0.05,
    support_selection_mode="diverse",
    require_provider=True,
    selector_posterior="method_specific_posterior",
    candidate_universe="frozen_observed_history",
):
    """Freeze the common V9 shortlist from method-specific posteriors.

    The method supplies only posterior objective means and violation
    probabilities on policies observed before verification. No target truth
    or independent verification sample enters this selector.
    """

    points = [tuple(int(value) for value in point) for point in candidates]
    objective = np.asarray(objective_mean, dtype=float).reshape(-1)
    violation = np.asarray(probability_violation, dtype=float).reshape(-1)
    if len(points) != len(objective) or len(points) != len(violation):
        raise ValueError(
            "candidate, objective, and violation arrays must have equal length")
    if len(points) != len(set(points)):
        raise ValueError(
            "verification-aware shortlist candidates must be unique")
    if not np.all(np.isfinite(objective)) or not np.all(
        np.isfinite(violation)
    ):
        raise ValueError(
            "verification-aware shortlist statistics must be finite")
    requested = int(shortlist_size)
    if requested < 1 or requested > len(points):
        raise ValueError(
            "verification-aware shortlist size must be between one and the "
            "number of candidates")
    primary = tuple(int(value) for value in primary)
    if primary not in points:
        raise ValueError(
            "verification-aware primary must belong to the frozen candidates")

    challenger = select_objective_verification_challenger(
        points,
        objective,
        violation,
        maximum_violation_probability=maximum_violation_probability,
    )
    support = select_posterior_safe_interior(
        problem,
        primary,
        points,
        violation,
        objective_mean=objective,
        selection_mode=support_selection_mode,
        probability_slack=probability_slack,
        require_provider=require_provider,
        candidate_universe=candidate_universe,
    )
    challenger_point = (
        None
        if challenger.get("point") is None
        else tuple(int(value) for value in challenger["point"])
    )
    support_point = tuple(int(value) for value in support["point"])
    ordered = []
    for point in (challenger_point, primary, support_point):
        if point is not None and point not in ordered:
            ordered.append(point)

    threshold = float(maximum_violation_probability)
    ranked_indices = sorted(
        range(len(points)),
        key=lambda index: (
            bool(float(violation[index]) > threshold),
            (
                float(objective[index])
                if float(violation[index]) <= threshold
                else float(violation[index])
            ),
            float(violation[index]),
            float(objective[index]),
            int(index),
        ),
    )
    for index in ranked_indices:
        point = points[index]
        if point not in ordered:
            ordered.append(point)
        if len(ordered) >= requested:
            break
    ordered = ordered[:requested]
    if len(ordered) != requested:
        raise RuntimeError(
            "verification-aware selector could not freeze the requested "
            "number of distinct candidates")

    index_by_point = {point: index for index, point in enumerate(points)}
    records = []
    for position, point in enumerate(ordered, start=1):
        index = index_by_point[point]
        if point == challenger_point and point == primary:
            role = "posterior_objective_primary"
        elif point == challenger_point:
            role = "posterior_objective_verification_challenger"
        elif point == primary:
            role = "posterior_feasible_primary_fallback"
        elif point == support_point:
            role = "posterior_safe_interior_diversified"
        else:
            role = "posterior_feasible_first_fallback"
        record = {
            "shortlist_position": int(position),
            "shortlist_role": role,
            "posterior_rank": int(ranked_indices.index(index) + 1),
            "point": list(point),
            "point_fingerprint": integer_design_fingerprint([point]),
            "selector_posterior": str(selector_posterior),
            "candidate_universe": str(candidate_universe),
            "selected_posterior_objective": float(objective[index]),
            "selected_posterior_violation_probability": float(
                violation[index]),
            "target_oracle_used": False,
            "verification_samples_used": False,
            "shortlist_frozen_before_verification": True,
        }
        if point == challenger_point:
            record.update({
                key: copy.deepcopy(item)
                for key, item in challenger.items()
                if key != "point"
            })
        if point == support_point:
            record.update({
                key: copy.deepcopy(item)
                for key, item in support.items()
                if key != "point"
            })
        records.append(record)
    return records, {
        "shortlist_mode": "posterior_objective_challenger_then_safe",
        "shortlist_size": int(len(records)),
        "maximum_violation_probability": threshold,
        "probability_slack": float(probability_slack),
        "support_selection_mode": str(support_selection_mode),
        "candidate_universe": str(candidate_universe),
        "candidate_count": int(len(points)),
        "selector_posterior": str(selector_posterior),
        "objective_challenger_status": str(challenger["status"]),
        "target_oracle_used": False,
        "verification_samples_used": False,
    }


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
    candidate_points,
    violation_probability,
    *,
    objective_mean=None,
    selection_mode="diverse",
    probability_slack=0.05,
    require_provider=True,
    candidate_universe="frozen_initial_atlas",
):
    """Select one frozen support from a posterior-safe probability sublevel."""

    slack = float(probability_slack)
    if not np.isfinite(slack) or not 0.0 <= slack <= 1.0:
        raise ValueError(
            "terminal safe-interior probability slack must lie in [0, 1]")
    unique = []
    seen = set()
    for point in candidate_points:
        point = tuple(int(value) for value in point)
        if point not in seen:
            seen.add(point)
            unique.append(point)
    primary = tuple(int(value) for value in primary)
    alternatives = [point for point in unique if point != primary]
    if not alternatives:
        raise RuntimeError(
            "terminal safe-interior selection requires a distinct "
            "frozen candidate")
    probability = np.asarray(
        violation_probability, dtype=float).reshape(-1)
    if len(probability) != len(unique) or not np.all(np.isfinite(probability)):
        raise RuntimeError(
            "terminal safe-interior selection requires finite posterior "
            "violation probabilities")
    mode = str(
        selection_mode or "diverse"
    ).strip().lower().replace("-", "_")
    objective_modes = {"objective_ranked", "objective_safe_ranked"}
    if mode not in {"diverse", *objective_modes}:
        raise ValueError(
            "terminal safe-interior selection mode must be "
            "'diverse', 'objective_ranked', or "
            "'objective_safe_ranked'")
    objective = None
    if objective_mean is not None:
        objective = np.asarray(objective_mean, dtype=float).reshape(-1)
        if len(objective) != len(unique) or not np.all(np.isfinite(objective)):
            raise RuntimeError(
                "objective-ranked terminal support requires finite posterior "
                "objective means")
    if mode in objective_modes and objective is None:
        raise RuntimeError(
            "objective-ranked terminal support requires objective_mean")
    minimum = float(np.min(probability))
    alternative_indices = [
        index for index, point in enumerate(unique)
        if point != primary
    ]
    alternative_minimum = float(np.min(probability[alternative_indices]))
    eligibility_reference = (
        alternative_minimum if mode == "objective_ranked" else minimum)
    eligible = [
        index for index, point in enumerate(unique)
        if (
            point != primary
            and float(probability[index]) <= eligibility_reference + slack
        )
    ]
    eligibility_status = "posterior_violation_sublevel"
    if not eligible:
        eligible = [
            min(
                alternative_indices,
                key=lambda index: (float(probability[index]), int(index)),
            )
        ]
        eligibility_status = "minimum_violation_fallback"

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
    if mode in objective_modes:
        selected_index = min(
            eligible,
            key=lambda index: (
                float(objective[index]),
                float(probability[index]),
                int(index),
            ),
        )
        selection_contract = (
            "posterior_global_violation_sublevel_minimum_posterior_objective"
            if mode == "objective_safe_ranked"
            else "posterior_violation_sublevel_minimum_posterior_objective"
        )
    else:
        selected_index = max(
            eligible,
            key=lambda index: (
                float(distance[index]),
                -float(probability[index]),
                -int(index),
            ),
        )
        selection_contract = (
            "posterior_violation_sublevel_maximin_cumulative_risk")
    return {
        "point": unique[selected_index],
        "selection_contract": selection_contract,
        "selection_mode": mode,
        "candidate_universe": str(candidate_universe),
        "candidate_universe_size": len(unique),
        "eligible_count": len(eligible),
        "probability_slack": slack,
        "minimum_posterior_violation_probability": minimum,
        "minimum_alternative_posterior_violation_probability": (
            alternative_minimum),
        "eligibility_reference": (
            "minimum_alternative"
            if mode == "objective_ranked"
            else "global_minimum"
        ),
        "eligibility_status": eligibility_status,
        "selected_posterior_violation_probability": float(
            probability[selected_index]),
        "selected_posterior_objective": (
            None
            if objective is None
            else float(objective[selected_index])
        ),
        "selected_standardized_coordinate_distance": float(
            distance[selected_index]),
        "coordinate_source": str(sources[selected_index]),
        "coordinate_dimension": int(matrix.shape[1]),
        "target_labels_used": False,
        "target_oracle_used": False,
        "verification_samples_used": False,
    }
