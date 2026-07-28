"""Method-independent frozen-policy terminal verification."""

from __future__ import annotations

from dataclasses import asdict
import copy

import numpy as np
from scipy.stats import t as student_t

from core.certification import (
    gaussian_quantile_tolerance_certificate,
    gaussian_replication_chance_certificate,
)
from core.cumulative_risk import get_risk_exposure
from core.designs import integer_design_fingerprint


TERMINAL_VERIFICATION_STREAM_TAG = 0x56455249
TERMINAL_SHORTLIST_VERIFICATION_STREAM_TAG = 0x56534C54
TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG = 0x4F424A44


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


def select_initial_empirical_objective_incumbent(
    points,
    observations,
    *,
    n0=None,
):
    """Freeze the lowest empirical-objective initial policy without truth."""

    point_rows = [tuple(int(value) for value in point) for point in points]
    observation_rows = [
        np.asarray(observation, dtype=float).reshape(-1)
        for observation in observations
    ]
    if len(point_rows) != len(observation_rows):
        raise ValueError(
            "initial incumbent points and observations must have equal length")
    limit = len(point_rows) if n0 is None else int(n0)
    if limit < 1:
        raise ValueError("initial incumbent selection requires n0 >= 1")
    point_rows = point_rows[:limit]
    observation_rows = observation_rows[:limit]
    if not point_rows:
        raise ValueError("initial incumbent selection received no observations")

    objective_by_point = {}
    first_position = {}
    for position, (point, observation) in enumerate(
        zip(point_rows, observation_rows)
    ):
        if observation.size < 1 or not np.isfinite(observation[0]):
            raise ValueError(
                "initial incumbent objective observations must be finite")
        objective_by_point.setdefault(point, []).append(float(observation[0]))
        first_position.setdefault(point, int(position))
    ranked = []
    for point, values in objective_by_point.items():
        ranked.append((
            float(np.mean(values)),
            int(first_position[point]),
            point,
            int(len(values)),
        ))
    objective_mean, position, point, count = min(ranked)
    return {
        "status": "selected",
        "point": point,
        "observed_objective_mean": float(objective_mean),
        "initial_observation_count": int(count),
        "initial_design_position": int(position + 1),
        "initial_design_size_used": int(len(point_rows)),
        "candidate_universe": "frozen_initial_design",
        "selection_contract": (
            "minimum_empirical_objective_in_frozen_initial_design"),
        "target_labels_used": True,
        "target_oracle_used": False,
        "verification_samples_used": False,
        "shortlist_frozen_before_verification": True,
    }


def freeze_objective_incumbent_shortlist(
    frozen_shortlist,
    initial_incumbent,
    *,
    shortlist_size=None,
):
    """Insert a frozen initial incumbent while preserving shortlist size."""

    records = [copy.deepcopy(record) for record in frozen_shortlist]
    if not records:
        raise ValueError("objective-guard shortlist cannot be empty")
    requested = (
        len(records) if shortlist_size is None else int(shortlist_size))
    if requested < 1 or requested > len(records):
        raise ValueError(
            "objective-guard shortlist size must preserve the frozen "
            "candidate count")
    incumbent = copy.deepcopy(initial_incumbent)
    if incumbent.get("point") is None:
        raise ValueError("initial objective incumbent must contain a point")
    incumbent_point = tuple(
        int(value) for value in incumbent["point"])

    ordered = [records[0]]
    first_point = tuple(int(value) for value in ordered[0]["point"])
    if incumbent_point != first_point:
        ordered.append({
            "shortlist_role": "frozen_initial_empirical_objective_incumbent",
            "point": list(incumbent_point),
            "point_fingerprint": integer_design_fingerprint([
                incumbent_point]),
            **{
                key: copy.deepcopy(value)
                for key, value in incumbent.items()
                if key != "point"
            },
        })
    seen = {
        tuple(int(value) for value in record["point"])
        for record in ordered
    }
    for record in records[1:]:
        point = tuple(int(value) for value in record["point"])
        if point not in seen:
            ordered.append(record)
            seen.add(point)
        if len(ordered) >= requested:
            break
    ordered = ordered[:requested]
    if len(ordered) != requested:
        raise RuntimeError(
            "objective incumbent insertion could not preserve shortlist size")

    incumbent_position = None
    for position, record in enumerate(ordered, start=1):
        point = tuple(int(value) for value in record["point"])
        record["shortlist_position"] = int(position)
        record["shortlist_frozen_before_verification"] = True
        record["target_oracle_used"] = False
        record["verification_samples_used"] = False
        record["objective_incumbent"] = bool(point == incumbent_point)
        if point == incumbent_point:
            incumbent_position = int(position)
            record.update({
                key: copy.deepcopy(value)
                for key, value in incumbent.items()
                if key != "point"
            })
    if incumbent_position is None:
        raise RuntimeError("initial objective incumbent was not frozen")
    return ordered, {
        "enabled": True,
        "objective_incumbent_position": int(incumbent_position),
        "objective_incumbent_fingerprint": integer_design_fingerprint([
            incumbent_point]),
        "shortlist_size": int(len(ordered)),
        "selection_contract": (
            "preverification_initial_empirical_incumbent_insertion"),
        "target_labels_used": True,
        "target_oracle_used": False,
        "verification_samples_used": False,
        "shortlist_frozen_before_verification": True,
    }


def verify_paired_objective_dominance(
    problem,
    challenger,
    incumbent,
    *,
    seed,
    comparison_budget,
    delta,
):
    """Test objective improvement on an independent paired-CRN stream."""

    budget = int(comparison_budget)
    if budget < 2:
        raise ValueError(
            "paired objective comparison requires at least two replications")
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("objective comparison delta must lie in (0, 1)")
    challenger = tuple(int(value) for value in challenger)
    incumbent = tuple(int(value) for value in incumbent)
    if challenger == incumbent:
        return {
            "enabled": True,
            "status": "same_policy",
            "challenger_dominates": False,
            "comparison_budget_per_policy": 0,
            "simulation_calls": 0,
            "delta": delta,
            "paired_common_random_numbers": True,
            "target_oracle_used": False,
            "comparison_samples_logged": False,
        }
    noise_model = str(getattr(
        problem, "simulation_noise_model", "unknown"))
    if noise_model != "iid_gaussian":
        raise RuntimeError(
            "paired Gaussian objective comparison requires an "
            "iid_gaussian simulation_noise_model")

    differences = []
    for replication in range(budget):
        seed_components = [
            int(seed),
            int(replication),
            TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG,
        ]
        challenger_rng = np.random.default_rng(
            np.random.SeedSequence(seed_components))
        incumbent_rng = np.random.default_rng(
            np.random.SeedSequence(seed_components))
        challenger_value = np.asarray(
            problem.simulate(challenger, challenger_rng),
            dtype=float,
        ).reshape(-1)
        incumbent_value = np.asarray(
            problem.simulate(incumbent, incumbent_rng),
            dtype=float,
        ).reshape(-1)
        if challenger_value.size < 1 or incumbent_value.size < 1:
            raise RuntimeError(
                "objective comparison requires an objective output")
        differences.append(
            float(challenger_value[0] - incumbent_value[0]))
    differences = np.asarray(differences, dtype=float)
    mean_difference = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1))
    standard_error = float(std_difference / np.sqrt(budget))
    critical_value = float(student_t.ppf(1.0 - delta, budget - 1))
    upper_bound = float(
        mean_difference + critical_value * standard_error)
    dominates = bool(upper_bound < 0.0)
    return {
        "enabled": True,
        "status": (
            "challenger_dominates"
            if dominates else "incumbent_retained"
        ),
        "challenger_dominates": dominates,
        "objective_difference_definition": (
            "challenger_minus_incumbent_for_minimization"),
        "sample_mean_difference": mean_difference,
        "sample_std_difference": std_difference,
        "standard_error": standard_error,
        "one_sided_upper_confidence_bound": upper_bound,
        "critical_value": critical_value,
        "degrees_of_freedom": int(budget - 1),
        "comparison_budget_per_policy": int(budget),
        "simulation_calls": int(2 * budget),
        "delta": delta,
        "paired_common_random_numbers": True,
        "comparison_stream_tag": int(
            TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG),
        "comparison_stream_independent_from_search_and_safety": True,
        "challenger_fingerprint": integer_design_fingerprint([
            challenger]),
        "incumbent_fingerprint": integer_design_fingerprint([
            incumbent]),
        "target_oracle_used": False,
        "comparison_samples_logged": False,
    }


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
    objective_incumbent_position=None,
    objective_comparison_budget=0,
    objective_comparison_delta=0.05,
):
    """Deploy a certified policy, optionally guarding an initial incumbent."""

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
    comparison_budget = int(objective_comparison_budget)
    if comparison_budget < 0:
        raise ValueError(
            "objective comparison budget must be nonnegative")
    incumbent_index = None
    if objective_incumbent_position is not None:
        incumbent_index = int(objective_incumbent_position) - 1
        if incumbent_index < 0 or incumbent_index >= len(points):
            raise ValueError(
                "objective incumbent position must belong to the shortlist")
    objective_guard_enabled = bool(
        incumbent_index is not None and comparison_budget > 0)
    comparison_delta = float(objective_comparison_delta)
    if objective_guard_enabled and not 0.0 < comparison_delta < 1.0:
        raise ValueError(
            "objective comparison delta must lie in (0, 1)")

    attempts = []
    attempts_by_index = {}
    deployed = points[0]
    selected_rank = None
    objective_comparison = {
        "enabled": False,
        "status": (
            "disabled"
            if incumbent_index is None or comparison_budget == 0
            else "not_required"
        ),
        "comparison_budget_per_policy": int(comparison_budget),
        "simulation_calls": 0,
        "delta": comparison_delta,
        "target_oracle_used": False,
        "comparison_samples_logged": False,
    }

    def verify_index(candidate_index):
        if candidate_index in attempts_by_index:
            return attempts_by_index[candidate_index]
        attempt = verify_frozen_policy(
            problem,
            points[candidate_index],
            seed=seed,
            search_evaluation_count=search_evaluation_count,
            verification_budget=budgets[candidate_index],
            delta=per_candidate_delta,
            candidate_index=candidate_index,
            method=method,
            mean_delta_fraction=mean_delta_fraction,
        )
        attempts.append(attempt)
        attempts_by_index[candidate_index] = attempt
        return attempt

    if budgets[0] > 0:
        if objective_guard_enabled:
            challenger_attempt = verify_index(0)
            incumbent_attempt = verify_index(incumbent_index)
            challenger_certified = bool(
                challenger_attempt.get("certified", False))
            incumbent_certified = bool(
                incumbent_attempt.get("certified", False))
            if incumbent_index == 0 and challenger_certified:
                deployed = points[0]
                selected_rank = 1
                objective_comparison["status"] = "same_policy"
            elif challenger_certified and incumbent_certified:
                objective_comparison = verify_paired_objective_dominance(
                    problem,
                    points[0],
                    points[incumbent_index],
                    seed=seed,
                    comparison_budget=comparison_budget,
                    delta=comparison_delta,
                )
                if bool(objective_comparison["challenger_dominates"]):
                    deployed = points[0]
                    selected_rank = 1
                else:
                    deployed = points[incumbent_index]
                    selected_rank = incumbent_index + 1
            elif challenger_certified:
                deployed = points[0]
                selected_rank = 1
                objective_comparison["status"] = (
                    "incumbent_not_safety_certified")
            elif incumbent_certified:
                deployed = points[incumbent_index]
                selected_rank = incumbent_index + 1
                objective_comparison["status"] = (
                    "challenger_not_safety_certified")
            else:
                objective_comparison["status"] = (
                    "neither_primary_nor_incumbent_safety_certified")
                for candidate_index in range(len(points)):
                    if candidate_index in attempts_by_index:
                        continue
                    attempt = verify_index(candidate_index)
                    if bool(attempt.get("certified", False)):
                        deployed = points[candidate_index]
                        selected_rank = candidate_index + 1
                        break
        else:
            for candidate_index in range(len(points)):
                attempt = verify_index(candidate_index)
                if bool(attempt.get("certified", False)):
                    deployed = points[candidate_index]
                    selected_rank = candidate_index + 1
                    break

    safety_budget = int(sum(
        int(attempt.get("verification_budget", 0))
        for attempt in attempts
    ))
    objective_budget = int(
        objective_comparison.get("simulation_calls", 0))
    actual_budget = int(safety_budget + objective_budget)
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
        "safety_verification_budget": safety_budget,
        "objective_comparison_budget": objective_budget,
        "objective_comparison_budget_per_policy": int(comparison_budget),
        "verification_budget_per_candidate": budgets[0],
        "fallback_verification_budget": (
            budgets[1] if len(budgets) > 1 else budgets[0]),
        "candidate_verification_budgets": budgets,
        "max_verification_budget": int(
            sum(budgets)
            + (2 * comparison_budget if objective_guard_enabled else 0)
        ),
        "search_evaluation_count": int(search_evaluation_count),
        "total_evaluation_count": int(search_evaluation_count) + actual_budget,
        "familywise_delta": familywise_delta,
        "per_candidate_delta": per_candidate_delta,
        "mean_delta_fraction": float(mean_delta_fraction),
        "frozen_shortlist_size": len(records),
        "frozen_shortlist": records,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "objective_incumbent_guard_enabled": objective_guard_enabled,
        "objective_incumbent_position": (
            None if incumbent_index is None else int(incumbent_index + 1)),
        "objective_comparison": objective_comparison,
        "certified": certified,
        "selected_shortlist_rank": selected_rank,
        "primary_policy_fingerprint": integer_design_fingerprint([points[0]]),
        "deployed_policy_fingerprint": integer_design_fingerprint([deployed]),
        "recommendation_changed": bool(deployed != points[0]),
        "target_oracle_used": False,
        "certification_scope": (
            "familywise_frozen_ordered_terminal_shortlist"
            + (
                "_with_independent_paired_objective_guard"
                if objective_guard_enabled else ""
            )
        ),
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
