"""Interchangeable decision rules over one SC-OLH posterior state.

The backends in this module do not generate candidates or fit source priors.
They only rank a shared candidate set using the same target-budget posterior.
This separation is important for attributing gains to the structural model,
the source proposal, or the online decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import norm, qmc

from core.designs import next_sobol_integer_candidate


_EPS = 1e-12


@dataclass(frozen=True)
class DecisionMoments:
    objective_mean: np.ndarray
    objective_epistemic: np.ndarray
    constraint_mean: np.ndarray
    constraint_epistemic: np.ndarray
    constraint_aleatoric: np.ndarray
    constraint_between_expert: np.ndarray
    source: str


def normal_positive_part(mean, standard_deviation):
    """Return E[max(X, 0)] for a componentwise Gaussian X."""

    mean = np.asarray(mean, dtype=float)
    sd = np.sqrt(np.maximum(
        np.asarray(standard_deviation, dtype=float) ** 2, _EPS))
    z = mean / sd
    return sd * norm.pdf(z) + mean * norm.cdf(z)


def minimization_expected_improvement(best, mean, standard_deviation):
    """Expected improvement for minimizing a Gaussian loss."""

    mean = np.asarray(mean, dtype=float)
    sd = np.sqrt(np.maximum(
        np.asarray(standard_deviation, dtype=float) ** 2, _EPS))
    improvement = float(best) - mean
    z = improvement / sd
    return np.maximum(improvement * norm.cdf(z) + sd * norm.pdf(z), 0.0)


def _variance_model_many(variance_model, output_index, candidates, problem):
    if hasattr(variance_model, "predict_certification_variance_many"):
        values = variance_model.predict_certification_variance_many(
            output_index, candidates, problem)
    elif hasattr(variance_model, "predict_variance_many"):
        values = variance_model.predict_variance_many(
            output_index, candidates, problem)
    else:
        method = getattr(
            variance_model,
            "predict_certification_variance",
            variance_model.predict_variance,
        )
        values = [method(output_index, x, problem) for x in candidates]
    return np.maximum(np.asarray(values, dtype=float), _EPS)


def _nominal_variance_model_many(
    variance_model, output_index, candidates, problem,
):
    """Observation variance used by the actual GPR rank-one update."""

    if hasattr(variance_model, "predict_variance_many"):
        values = variance_model.predict_variance_many(
            output_index, candidates, problem)
    elif hasattr(variance_model, "predict_variance"):
        values = [
            variance_model.predict_variance(output_index, x, problem)
            for x in candidates
        ]
    else:
        values = _variance_model_many(
            variance_model, output_index, candidates, problem)
    return np.maximum(np.asarray(values, dtype=float), _EPS)


def posterior_moments(
    candidates,
    obj_gpr,
    con_gpr,
    variance_model,
    problem,
    *,
    task_ensemble=None,
    aleatoric_mode="certification_upper",
    ambiguity_mode="kl_robust",
):
    """Extract common posterior moments for every online backend."""

    mode = str(
        aleatoric_mode or "certification_upper"
    ).strip().lower().replace("-", "_")
    if mode not in {"certification_upper", "posterior_central"}:
        raise ValueError(
            "aleatoric mode must be certification_upper or posterior_central"
        )
    ambiguity = str(
        ambiguity_mode or "kl_robust"
    ).strip().lower().replace("-", "_")
    if ambiguity not in {"kl_robust", "posterior_nominal"}:
        raise ValueError(
            "ambiguity mode must be kl_robust or posterior_nominal"
        )

    if task_ensemble is not None:
        objective = task_ensemble.mixture_moments_many(
            0, candidates, certification=False)
        if ambiguity == "posterior_nominal":
            constraint = task_ensemble.mixture_moments_many(
                1,
                candidates,
                certification=(mode == "certification_upper"),
            )
            constraint_mean = constraint.mean
            constraint_epistemic = constraint.epistemic
            constraint_aleatoric = constraint.aleatoric
            constraint_between = constraint.between_mean
            source = "task_posterior_nominal_cumulative"
        else:
            constraint = task_ensemble.robust_moments_many(
                1,
                candidates,
                certification=(mode == "certification_upper"),
            )
            constraint_mean = constraint.mean_upper
            constraint_epistemic = constraint.epistemic_upper
            constraint_aleatoric = constraint.aleatoric_upper
            constraint_between = constraint.nominal.between_mean
            source = "task_posterior_robust_cumulative"
        return DecisionMoments(
            objective_mean=np.asarray(objective.mean, dtype=float),
            objective_epistemic=np.maximum(
                np.asarray(objective.epistemic, dtype=float), _EPS),
            constraint_mean=np.asarray(constraint_mean, dtype=float),
            constraint_epistemic=np.maximum(
                np.asarray(constraint_epistemic, dtype=float), _EPS),
            constraint_aleatoric=np.maximum(
                np.asarray(constraint_aleatoric, dtype=float), _EPS),
            constraint_between_expert=np.maximum(
                np.asarray(constraint_between, dtype=float), 0.0),
            source=source,
        )

    return DecisionMoments(
        objective_mean=np.asarray(
            obj_gpr.posterior_mean_many(candidates), dtype=float),
        objective_epistemic=np.maximum(np.asarray(
            obj_gpr.posterior_var_many(candidates), dtype=float), _EPS),
        constraint_mean=np.asarray(
            con_gpr.posterior_mean_many(candidates), dtype=float),
        constraint_epistemic=np.maximum(np.asarray(
            con_gpr.posterior_var_many(candidates), dtype=float), _EPS),
        constraint_aleatoric=(
            _variance_model_many(variance_model, 1, candidates, problem)
            if mode == "certification_upper"
            else _nominal_variance_model_many(
                variance_model, 1, candidates, problem)
        ),
        constraint_between_expert=np.zeros(len(candidates), dtype=float),
        source="single_cumulative_hvd",
    )


def _joint_gpr_draw(model, candidates, rng):
    """Draw one coherent finite-feature posterior function realization."""

    features = np.asarray(model.augmented_feature_matrix(candidates), dtype=float)
    covariance = np.asarray(model.C, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    try:
        root = np.linalg.cholesky(
            covariance + 1e-12 * np.eye(covariance.shape[0]))
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    coefficients = np.asarray(model.a, dtype=float) + root @ rng.standard_normal(
        covariance.shape[0])
    draw = features @ coefficients

    represented_variance = np.einsum(
        "ij,jk,ik->i", features, covariance, features)
    total_variance = np.asarray(model.posterior_var_many(candidates), dtype=float)
    residual_variance = np.maximum(total_variance - represented_variance, 0.0)
    return draw + np.sqrt(residual_variance) * rng.standard_normal(len(candidates))


def _task_posterior_draw(task_ensemble, candidates, rng):
    weights = np.asarray(
        task_ensemble.structure_weights(objective=False), dtype=float)
    weights = np.maximum(weights, 0.0)
    weights /= max(float(np.sum(weights)), _EPS)
    expert_index = int(rng.choice(len(task_ensemble.states), p=weights))
    state = task_ensemble.states[expert_index]
    return (
        _joint_gpr_draw(state.gpr_models[0], candidates, rng),
        _joint_gpr_draw(state.gpr_models[1], candidates, rng),
        expert_index,
    )


def _posterior_loss_components(
    moments,
    problem,
    risk_penalty,
    violation_loss_mode="positive_part",
):
    loss_mode = str(
        violation_loss_mode or "positive_part"
    ).strip().lower().replace("-", "_")
    if loss_mode not in {"positive_part", "failure_probability"}:
        raise ValueError(
            "violation loss mode must be positive_part or failure_probability"
        )
    z_alpha = float(norm.ppf(1.0 - float(problem.alpha)))
    stochastic_margin_mean = (
        moments.constraint_mean
        + z_alpha * np.sqrt(np.maximum(moments.constraint_aleatoric, _EPS))
        - float(problem.tau)
    )
    margin_sd = np.sqrt(np.maximum(moments.constraint_epistemic, _EPS))
    expected_violation = normal_positive_part(
        stochastic_margin_mean, margin_sd)
    probability_violation = norm.cdf(stochastic_margin_mean / margin_sd)
    violation_loss = (
        probability_violation
        if loss_mode == "failure_probability"
        else expected_violation
    )
    bayes_risk = (
        moments.objective_mean + float(risk_penalty) * violation_loss)
    probability_feasible = 1.0 - probability_violation
    return {
        "stochastic_margin_mean": stochastic_margin_mean,
        "margin_sd": margin_sd,
        "expected_violation": expected_violation,
        "probability_violation": probability_violation,
        "violation_loss": violation_loss,
        "violation_loss_mode": loss_mode,
        "bayes_risk": bayes_risk,
        "probability_feasible": probability_feasible,
    }


def _observed_incumbent_loss(
    observed,
    obj_gpr,
    con_gpr,
    variance_model,
    problem,
    task_ensemble,
    risk_penalty,
    fallback,
    aleatoric_mode="certification_upper",
    violation_loss_mode="positive_part",
    ambiguity_mode="kl_robust",
):
    points = []
    for item in observed or []:
        x = item[0] if isinstance(item, (tuple, list)) and len(item) == 2 else item
        try:
            points.append(tuple(int(v) for v in x))
        except (TypeError, ValueError):
            continue
    points = list(dict.fromkeys(points))
    if not points:
        return float(np.min(fallback))
    moments = posterior_moments(
        points,
        obj_gpr,
        con_gpr,
        variance_model,
        problem,
        task_ensemble=task_ensemble,
        aleatoric_mode=aleatoric_mode,
        ambiguity_mode=ambiguity_mode,
    )
    loss = _posterior_loss_components(
        moments,
        problem,
        risk_penalty,
        violation_loss_mode=violation_loss_mode,
    )["bayes_risk"]
    return float(np.min(loss))


def _observed_point_set(observed):
    points = set()
    for item in observed or []:
        x = item[0] if isinstance(item, (tuple, list)) and len(item) == 2 else item
        try:
            points.add(tuple(int(v) for v in x))
        except (TypeError, ValueError):
            continue
    return points


def _observed_point_counts(observed):
    counts = {}
    for item in observed or []:
        x = item[0] if isinstance(item, (tuple, list)) and len(item) == 2 else item
        try:
            point = tuple(int(v) for v in x)
        except (TypeError, ValueError):
            continue
        counts[point] = counts.get(point, 0) + 1
    return counts


def _sobol_scores(problem, candidates, iteration, seed):
    bounds_lo, bounds_hi = problem.int_bounds()
    lo = np.asarray(bounds_lo, dtype=float)
    hi = np.asarray(bounds_hi, dtype=float)
    engine = qmc.Sobol(
        d=int(problem.d),
        scramble=True,
        seed=int(seed),
    )
    index = max(0, int(iteration))
    exponent = int(math.ceil(math.log2(index + 1)))
    target = engine.random_base2(exponent)[index]
    normalized = (
        np.asarray(candidates, dtype=float) - lo
    ) / np.maximum(hi - lo, 1.0)
    return -np.sum((normalized - target) ** 2, axis=1)


def _risk_coordinate_coverage_scores(problem, candidates, observed):
    """Return standardized novelty in the observable cumulative-risk space.

    The score uses only policy/state exposure features exposed by the problem
    adapter. It never evaluates target objective or constraint truth. A zero
    vector is returned when the provider is unavailable so the action-set
    contract remains deterministic on legacy problems.
    """

    if not candidates or not hasattr(problem, "cumulative_risk_features"):
        return np.zeros(len(candidates), dtype=float), "provider_unavailable"

    def features(point):
        try:
            value = problem.cumulative_risk_features(
                point, output_index=1)
        except TypeError:
            value = problem.cumulative_risk_features(point)
        row = np.asarray(value, dtype=float).reshape(-1)
        if len(row) == 0 or not np.all(np.isfinite(row)):
            raise ValueError("invalid cumulative-risk coordinate")
        return row

    try:
        candidate_features = np.vstack([
            features(point) for point in candidates
        ])
    except (AttributeError, TypeError, ValueError, FloatingPointError):
        return np.zeros(len(candidates), dtype=float), "provider_invalid"

    observed_points = sorted(_observed_point_set(observed))
    if not observed_points:
        return np.ones(len(candidates), dtype=float), "no_observed_points"
    try:
        observed_features = np.vstack([
            features(point) for point in observed_points
        ])
    except (AttributeError, TypeError, ValueError, FloatingPointError):
        return np.zeros(len(candidates), dtype=float), "observed_invalid"
    if observed_features.shape[1] != candidate_features.shape[1]:
        return np.zeros(len(candidates), dtype=float), "dimension_mismatch"

    combined = np.vstack([candidate_features, observed_features])
    center = np.median(combined, axis=0)
    scale = np.sqrt(np.mean((combined - center[None, :]) ** 2, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    candidate_z = (candidate_features - center[None, :]) / scale[None, :]
    observed_z = (observed_features - center[None, :]) / scale[None, :]
    distance = np.linalg.norm(
        candidate_z[:, None, :] - observed_z[None, :, :],
        axis=2,
    )
    return np.min(distance, axis=1), "observable_cumulative_risk"


def evaluate_or_replicate_action_set(
    candidates,
    observed,
    problem,
    *,
    iteration,
    seed,
    replication_max_per_solution=5,
    canonical_sobol_candidate=None,
    allow_replication_actions=True,
    new_action_count=1,
    new_action_policy="canonical_sobol",
    new_action_priority=None,
    baseline_new_action_count=None,
    supplemental_new_action_priorities=None,
):
    """Return a finite evaluate-or-replicate action set.

    The default preserves the historical contract: one canonical Sobol new
    point against every eligible replication.  ``canonical_plus_posterior``
    expands that discretization with the lowest-priority unobserved points;
    the caller supplies posterior Bayes risk as ``new_action_priority``.  The
    expensive fantasy refit still decides among the resulting actions.
    A supplemental policy first reproduces an explicitly recorded baseline
    subset, then appends at most one minimizer from each supplied priority.
    This makes the challenger action set a literal superset of the baseline.
    """

    candidates = [tuple(int(v) for v in x) for x in candidates]
    if not candidates:
        return {
            "active_indices": np.empty(0, dtype=int),
            "active_mask": np.empty(0, dtype=bool),
            "replicate_mask": np.empty(0, dtype=bool),
            "sobol_new_index": None,
            "canonical_sobol_index": None,
            "new_action_indices": np.empty(0, dtype=int),
            "baseline_new_action_indices": np.empty(0, dtype=int),
            "supplemental_new_action_indices": np.empty(0, dtype=int),
            "supplemental_new_action_labels": [],
            "new_action_policy": str(new_action_policy),
            "sobol_scores": np.empty(0, dtype=float),
        }
    sobol_score = _sobol_scores(problem, candidates, iteration, seed)
    observed_counts = _observed_point_counts(observed)
    observed_mask = np.asarray([
        x in observed_counts for x in candidates
    ], dtype=bool)
    replicate_mask = np.asarray([
        bool(allow_replication_actions)
        and 0 < observed_counts.get(x, 0)
        < max(1, int(replication_max_per_solution))
        for x in candidates
    ], dtype=bool)
    active_indices = list(np.flatnonzero(replicate_mask))
    new_indices = np.flatnonzero(~observed_mask)
    sobol_new_index = None
    canonical_sobol_index = None
    selected_new_indices = []
    baseline_new_indices = []
    supplemental_new_indices = []
    supplemental_new_labels = []
    if len(new_indices):
        observed_points = _observed_point_set(observed)
        canonical = (
            None
            if canonical_sobol_candidate is None
            else tuple(int(value) for value in canonical_sobol_candidate)
        )
        if canonical is None:
            try:
                canonical = next_sobol_integer_candidate(
                    problem,
                    seed,
                    observed=observed_points,
                )
            except (TypeError, RuntimeError):
                canonical = None
        canonical_indices = [
            int(index) for index in new_indices
            if canonical is not None and candidates[int(index)] == canonical
        ]
        if canonical_indices:
            canonical_sobol_index = canonical_indices[0]
            sobol_new_index = canonical_sobol_index
        else:
            sobol_new_index = int(
                new_indices[int(np.argmax(sobol_score[new_indices]))])
        selected_new_indices.append(int(sobol_new_index))
        policy = str(new_action_policy or "canonical_sobol").strip().lower()
        policy = policy.replace("-", "_")
        requested_new = max(1, int(new_action_count))
        posterior_policies = {
            "canonical_plus_posterior", "canonical_plus_posterior_risk",
            "posterior_risk",
            "canonical_plus_posterior_certificate_coverage",
            "canonical_plus_posterior_risk_certificate_coverage",
        }
        if policy in posterior_policies:
            if new_action_priority is None:
                raise ValueError(
                    "posterior-risk new action policy requires priorities")
            priority = np.asarray(new_action_priority, dtype=float).reshape(-1)
            if len(priority) != len(candidates):
                raise ValueError(
                    "new action priorities must match candidate count")
            finite_priority = np.where(
                np.isfinite(priority), priority, np.inf)
            ordered_new = sorted(
                (int(index) for index in new_indices),
                key=lambda index: (float(finite_priority[index]), index),
            )
            baseline_requested = requested_new
            supplements = list(supplemental_new_action_priorities or [])
            if supplements:
                baseline_requested = max(1, min(
                    requested_new,
                    int(
                        baseline_new_action_count
                        if baseline_new_action_count is not None
                        else requested_new - len(supplements)
                    ),
                ))
            for index in ordered_new:
                if index not in selected_new_indices:
                    selected_new_indices.append(index)
                if len(selected_new_indices) >= baseline_requested:
                    break
            baseline_new_indices = list(selected_new_indices)
            for label, values in supplements:
                if len(selected_new_indices) >= requested_new:
                    break
                priority_values = np.asarray(values, dtype=float).reshape(-1)
                if len(priority_values) != len(candidates):
                    raise ValueError(
                        "supplemental priorities must match candidate count")
                finite_values = np.where(
                    np.isfinite(priority_values), priority_values, np.inf)
                ordered_supplement = sorted(
                    (int(index) for index in new_indices),
                    key=lambda index: (float(finite_values[index]), index),
                )
                chosen = next((
                    index for index in ordered_supplement
                    if index not in selected_new_indices
                ), None)
                if chosen is not None:
                    selected_new_indices.append(int(chosen))
                    supplemental_new_indices.append(int(chosen))
                    supplemental_new_labels.append(str(label))
            for index in ordered_new:
                if len(selected_new_indices) >= requested_new:
                    break
                if index not in selected_new_indices:
                    selected_new_indices.append(index)
        elif policy not in {"canonical_sobol", "sobol"}:
            raise ValueError(f"unknown new action policy {new_action_policy!r}")
        selected_new_indices = selected_new_indices[:requested_new]
        if not baseline_new_indices:
            baseline_new_indices = list(selected_new_indices)
        active_indices = selected_new_indices + active_indices
    if not active_indices:
        active_indices = [int(np.argmax(sobol_score))]
        sobol_new_index = active_indices[0]
        selected_new_indices = [active_indices[0]]
        baseline_new_indices = [active_indices[0]]
    active_indices = np.asarray(list(dict.fromkeys(active_indices)), dtype=int)
    active_mask = np.zeros(len(candidates), dtype=bool)
    active_mask[active_indices] = True
    return {
        "active_indices": active_indices,
        "active_mask": active_mask,
        "replicate_mask": replicate_mask,
        "sobol_new_index": sobol_new_index,
        "canonical_sobol_index": canonical_sobol_index,
        "new_action_indices": np.asarray(selected_new_indices, dtype=int),
        "baseline_new_action_indices": np.asarray(
            baseline_new_indices, dtype=int),
        "supplemental_new_action_indices": np.asarray(
            supplemental_new_indices, dtype=int),
        "supplemental_new_action_labels": list(supplemental_new_labels),
        "new_action_policy": str(new_action_policy),
        "sobol_scores": sobol_score,
    }


def _hvd_information_reduction(
    candidates,
    reference_candidates,
    variance_model,
    problem,
    *,
    action_reliability,
    reference_weights,
    task_ensemble=None,
):
    """Posterior-weighted HVD information reduction without target truth."""

    def one(model, provider):
        method = getattr(model, "information_reduction_many", None)
        if method is None:
            return np.asarray(action_reliability, dtype=float)
        return np.asarray(method(
            1,
            candidates,
            reference_candidates,
            provider,
            action_reliability=action_reliability,
            reference_weights=reference_weights,
        ), dtype=float)

    if task_ensemble is None:
        return one(variance_model, problem)
    weights = np.asarray(
        task_ensemble.variance_structure_weights(), dtype=float)
    weights = np.maximum(weights, 0.0)
    weights /= max(float(np.sum(weights)), _EPS)
    gains = np.zeros(len(candidates), dtype=float)
    for weight, state in zip(weights, task_ensemble.states):
        gains += float(weight) * one(state.variance_model, state.problem)
    return np.maximum(gains, 0.0)


def _constraint_epistemic_reduction(
    candidates,
    reference_candidates,
    con_gpr,
    variance_model,
    problem,
    *,
    reference_weights,
    task_ensemble=None,
):
    """Integrated one-step reduction of constraint posterior variance.

    For a fixed finite-feature GPR, observing action ``a`` reduces the latent
    variance at reference point ``x`` by

    ``cov(g(x), g(a))^2 / (var(g(a)) + v_C(a))``.

    The calculation includes the solution-specific deviation variance on the
    diagonal. For a task ensemble it averages the within-expert reductions
    using the same target-updated structure weights as posterior prediction.
    """

    actions = [tuple(int(v) for v in x) for x in candidates]
    references = [tuple(int(v) for v in x) for x in reference_candidates]
    if not actions:
        return np.zeros(0, dtype=float)

    weights = np.maximum(
        np.asarray(reference_weights, dtype=float).reshape(-1), 0.0)
    if len(weights) != len(references):
        raise ValueError("reference_weights length mismatch")
    weight_total = float(np.sum(weights))
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        weights = np.ones(len(references), dtype=float)
        weight_total = float(len(references))
    weights /= max(weight_total, _EPS)

    def one(model, noise_model, provider):
        action_features = np.asarray(
            model.augmented_feature_matrix(actions), dtype=float)
        reference_features = np.asarray(
            model.augmented_feature_matrix(references), dtype=float)
        covariance = np.asarray(model.C, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        represented_action = np.einsum(
            "ij,jk,ik->i",
            action_features,
            covariance,
            action_features,
        )
        total_action = np.asarray(
            model.posterior_var_many(actions), dtype=float)
        local_residual = np.maximum(
            total_action - represented_action, 0.0)
        cross = reference_features @ covariance @ action_features.T
        for action_index, action in enumerate(actions):
            if local_residual[action_index] <= 0.0:
                continue
            for reference_index, reference in enumerate(references):
                if reference == action:
                    cross[reference_index, action_index] += local_residual[
                        action_index]
        observation_noise = _nominal_variance_model_many(
            noise_model, 1, actions, provider)
        denominator = np.maximum(
            total_action + observation_noise, _EPS)
        reduction = cross ** 2 / denominator[None, :]
        return np.maximum(weights @ reduction, 0.0)

    if task_ensemble is None:
        return one(con_gpr, variance_model, problem)
    structure_weights = np.asarray(
        task_ensemble.structure_weights(objective=False), dtype=float)
    structure_weights = np.maximum(structure_weights, 0.0)
    structure_weights /= max(float(np.sum(structure_weights)), _EPS)
    gains = np.zeros(len(actions), dtype=float)
    variance_weights = np.asarray(
        task_ensemble.variance_structure_weights(), dtype=float)
    variance_weights = np.maximum(variance_weights, 0.0)
    variance_weights /= max(float(np.sum(variance_weights)), _EPS)
    if task_ensemble.variance_structure_isolated:
        for mean_weight, mean_state in zip(
            structure_weights, task_ensemble.states
        ):
            for variance_weight, variance_state in zip(
                variance_weights, task_ensemble.states
            ):
                gains += float(mean_weight * variance_weight) * one(
                    mean_state.gpr_models[1],
                    variance_state.variance_model,
                    variance_state.problem,
                )
    else:
        for weight, state in zip(structure_weights, task_ensemble.states):
            gains += float(weight) * one(
                state.gpr_models[1], state.variance_model, state.problem)
    return np.maximum(gains, 0.0)


def _constraint_epistemic_margin_reduction(
    candidates,
    reference_candidates,
    con_gpr,
    variance_model,
    problem,
    *,
    reference_weights,
    beta_g,
    task_ensemble=None,
):
    """Integrated reduction of ``sqrt(beta_g) * s_g`` in response units."""

    actions = [tuple(int(v) for v in x) for x in candidates]
    references = [tuple(int(v) for v in x) for x in reference_candidates]
    weights = np.maximum(
        np.asarray(reference_weights, dtype=float).reshape(-1), 0.0)
    if len(weights) != len(references):
        raise ValueError("reference_weights length mismatch")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones(len(references), dtype=float)
        total = float(len(references))
    weights /= max(total, _EPS)
    radius_scale = np.sqrt(max(float(beta_g), 0.0))

    def one(model, noise_model, provider):
        action_features = np.asarray(
            model.augmented_feature_matrix(actions), dtype=float)
        reference_features = np.asarray(
            model.augmented_feature_matrix(references), dtype=float)
        covariance = np.asarray(model.C, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        represented_action = np.einsum(
            "ij,jk,ik->i",
            action_features,
            covariance,
            action_features,
        )
        action_variance = np.asarray(
            model.posterior_var_many(actions), dtype=float)
        reference_variance = np.maximum(np.asarray(
            model.posterior_var_many(references), dtype=float), _EPS)
        local_residual = np.maximum(
            action_variance - represented_action, 0.0)
        cross = reference_features @ covariance @ action_features.T
        for action_index, action in enumerate(actions):
            if local_residual[action_index] <= 0.0:
                continue
            for reference_index, reference in enumerate(references):
                if reference == action:
                    cross[reference_index, action_index] += local_residual[
                        action_index]
        observation_noise = _nominal_variance_model_many(
            noise_model, 1, actions, provider)
        denominator = np.maximum(
            action_variance + observation_noise, _EPS)
        reduction = np.minimum(
            np.maximum(cross ** 2 / denominator[None, :], 0.0),
            reference_variance[:, None],
        )
        updated = np.maximum(
            reference_variance[:, None] - reduction, 0.0)
        radius_reduction = radius_scale * (
            np.sqrt(reference_variance)[:, None] - np.sqrt(updated))
        return np.maximum(weights @ radius_reduction, 0.0)

    if task_ensemble is None:
        return one(con_gpr, variance_model, problem)
    structure_weights = np.asarray(
        task_ensemble.structure_weights(objective=False), dtype=float)
    structure_weights = np.maximum(structure_weights, 0.0)
    structure_weights /= max(float(np.sum(structure_weights)), _EPS)
    gains = np.zeros(len(actions), dtype=float)
    variance_weights = np.asarray(
        task_ensemble.variance_structure_weights(), dtype=float)
    variance_weights = np.maximum(variance_weights, 0.0)
    variance_weights /= max(float(np.sum(variance_weights)), _EPS)
    if task_ensemble.variance_structure_isolated:
        for mean_weight, mean_state in zip(
            structure_weights, task_ensemble.states
        ):
            for variance_weight, variance_state in zip(
                variance_weights, task_ensemble.states
            ):
                gains += float(mean_weight * variance_weight) * one(
                    mean_state.gpr_models[1],
                    variance_state.variance_model,
                    variance_state.problem,
                )
    else:
        for weight, state in zip(structure_weights, task_ensemble.states):
            gains += float(weight) * one(
                state.gpr_models[1], state.variance_model, state.problem)
    return np.maximum(gains, 0.0)


def _hvd_certification_margin_reduction(
    candidates,
    reference_candidates,
    variance_model,
    problem,
    *,
    action_reliability,
    reference_weights,
    z_alpha,
    task_ensemble=None,
):
    """HVD information mapped through ``z_alpha * sqrt(v_C_plus)``."""

    def one(model, provider):
        method = getattr(
            model, "certification_margin_information_reduction_many", None)
        if method is not None:
            return np.asarray(method(
                1,
                candidates,
                reference_candidates,
                provider,
                action_reliability=action_reliability,
                reference_weights=reference_weights,
                z_alpha=z_alpha,
            ), dtype=float)
        # Compatibility for external/lightweight variance models. The square
        # root restores response units but is not used by OrthogonalHVD.
        raw = _hvd_information_reduction(
            candidates,
            reference_candidates,
            model,
            provider,
            action_reliability=action_reliability,
            reference_weights=reference_weights,
        )
        return max(float(z_alpha), 0.0) * np.sqrt(np.maximum(raw, 0.0))

    if task_ensemble is None:
        return one(variance_model, problem)
    weights = np.asarray(
        task_ensemble.variance_structure_weights(), dtype=float)
    weights = np.maximum(weights, 0.0)
    weights /= max(float(np.sum(weights)), _EPS)
    gains = np.zeros(len(candidates), dtype=float)
    for weight, state in zip(weights, task_ensemble.states):
        gains += float(weight) * one(state.variance_model, state.problem)
    return np.maximum(gains, 0.0)


def _source_utility(problem, candidates, components, moments):
    """Frozen source-prior utility with target-updated discrepancy weighting."""

    source_obj = None
    source_con = None
    if hasattr(problem, "source_mean_prior_predict_many"):
        source_obj = problem.source_mean_prior_predict_many(
            candidates, output_index=0)
        source_con = problem.source_mean_prior_predict_many(
            candidates, output_index=1)
    if source_obj is None or source_con is None:
        return None
    source_obj = np.asarray(source_obj, dtype=float)
    source_con = np.asarray(source_con, dtype=float)
    if source_obj.shape != moments.objective_mean.shape:
        return None
    objective_disagreement = np.abs(source_obj - moments.objective_mean)
    constraint_disagreement = np.abs(source_con - moments.constraint_mean)
    total_disagreement = (
        objective_disagreement + constraint_disagreement
        + np.sqrt(np.maximum(moments.constraint_between_expert, 0.0))
    )
    scale = max(float(np.median(total_disagreement)), 1e-8)
    discrepancy_trust = np.exp(-total_disagreement / scale)
    boundary = np.exp(-0.5 * (
        components["stochastic_margin_mean"]
        / np.maximum(components["margin_sd"], 1e-8)
    ) ** 2)
    source_rank = np.argsort(np.argsort(source_obj, kind="stable"), kind="stable")
    source_rank = 1.0 - source_rank / max(len(source_rank) - 1, 1)
    return (
        discrepancy_trust * source_rank * components["probability_feasible"]
        + (1.0 - discrepancy_trust) * boundary
        * np.sqrt(np.maximum(
            moments.constraint_epistemic
            + moments.constraint_between_expert,
            0.0,
        ))
    )


def score_decision_backend(
    backend,
    candidates,
    obj_gpr,
    con_gpr,
    variance_model,
    problem,
    *,
    observed=None,
    task_ensemble=None,
    rng=None,
    iteration=0,
    seed=0,
    risk_penalty=5.0,
    decision_aleatoric_mode="certification_upper",
    violation_loss_mode="positive_part",
    decision_ambiguity_mode="kl_robust",
    source_utility_weight=1.0,
    replication_max_per_solution=5,
    certification_beta_g=1.0,
    robust_certificate_mode="separable",
    canonical_sobol_candidate=None,
    allow_replication_actions=True,
    evaluate_or_replicate_new_action_count=1,
    evaluate_or_replicate_new_action_policy="canonical_sobol",
    evaluate_or_replicate_baseline_new_action_count=None,
):
    """Rank candidates under one named backend.

    The returned ``total`` is always maximized. ``legacy`` and ``exact_kg``
    are intentionally handled by the caller because they use the historical
    additive/exact-KG implementation.
    """

    name = str(backend or "legacy").strip().lower().replace("-", "_")
    if name in {"legacy", "legacy_kg", "exact_kg", "additive"}:
        raise ValueError(f"backend {name!r} must be handled by legacy acquisition")
    if not candidates:
        return {"total": np.zeros(0, dtype=float), "backend": name}
    rng = np.random.default_rng(seed) if rng is None else rng
    moments = posterior_moments(
        candidates,
        obj_gpr,
        con_gpr,
        variance_model,
        problem,
        task_ensemble=task_ensemble,
        aleatoric_mode=decision_aleatoric_mode,
        ambiguity_mode=decision_ambiguity_mode,
    )
    components = _posterior_loss_components(
        moments,
        problem,
        risk_penalty,
        violation_loss_mode=violation_loss_mode,
    )
    certification_moments = (
        moments
        if str(decision_aleatoric_mode).strip().lower().replace(
            "-", "_") == "certification_upper"
        and str(decision_ambiguity_mode).strip().lower().replace(
            "-", "_") == "kl_robust"
        else posterior_moments(
            candidates,
            obj_gpr,
            con_gpr,
            variance_model,
            problem,
            task_ensemble=task_ensemble,
            aleatoric_mode="certification_upper",
            ambiguity_mode="kl_robust",
        )
    )
    z_alpha = float(norm.ppf(1.0 - float(problem.alpha)))
    theory_margin = (
        certification_moments.constraint_mean
        + np.sqrt(max(float(certification_beta_g), 0.0))
        * np.sqrt(np.maximum(
            certification_moments.constraint_epistemic, _EPS))
        + z_alpha * np.sqrt(np.maximum(
            certification_moments.constraint_aleatoric, _EPS))
        - float(problem.tau)
    )
    certificate_mode = str(
        robust_certificate_mode or "separable"
    ).strip().lower().replace("-", "_")
    if task_ensemble is not None and certificate_mode in {
        "joint", "joint_kl", "joint_tangent",
    }:
        theory_margin = np.asarray(
            task_ensemble.robust_chance_margin_many(
                candidates,
                beta_g=float(certification_beta_g),
                z_alpha=z_alpha,
                tau=float(problem.tau),
                certification=True,
            ).upper,
            dtype=float,
        )
    incumbent = _observed_incumbent_loss(
        observed,
        obj_gpr,
        con_gpr,
        variance_model,
        problem,
        task_ensemble,
        risk_penalty,
        components["bayes_risk"],
        aleatoric_mode=decision_aleatoric_mode,
        violation_loss_mode=violation_loss_mode,
        ambiguity_mode=decision_ambiguity_mode,
    )
    objective_sd = np.sqrt(np.maximum(moments.objective_epistemic, _EPS))
    bayes_ei = minimization_expected_improvement(
        incumbent, components["bayes_risk"], objective_sd)
    constrained_ei = minimization_expected_improvement(
        float(np.min(moments.objective_mean)),
        moments.objective_mean,
        objective_sd,
    ) * components["probability_feasible"]
    sampled_expert = None

    hvd_information = None
    hvd_reliability = None
    hvd_is_replicate = None
    hvd_sobol_new_index = None
    canonical_sobol_index = None
    constraint_epistemic_information = None
    hvd_margin_information = None
    joint_information = None
    evaluate_or_replicate_active_indices = None
    evaluate_or_replicate_baseline_indices = None
    evaluate_or_replicate_supplemental_indices = None
    evaluate_or_replicate_supplemental_labels = []
    exact_refit_required = False
    risk_coordinate_coverage = None
    risk_coordinate_coverage_source = None

    if name in {"random", "random_continuation"}:
        total = rng.random(len(candidates))
    elif name in {"sobol", "sobol_continuation"}:
        total = _sobol_scores(problem, candidates, iteration, seed)
    elif name in {"sobol_new", "sobol_new_only"}:
        observed_points = _observed_point_set(observed)
        canonical = (
            None
            if canonical_sobol_candidate is None
            else tuple(int(value) for value in canonical_sobol_candidate)
        )
        if canonical is None:
            try:
                canonical = next_sobol_integer_candidate(
                    problem,
                    seed,
                    observed=observed_points,
                )
            except (TypeError, RuntimeError):
                canonical = None
        canonical_indices = [
            index for index, candidate in enumerate(candidates)
            if canonical is not None
            and tuple(int(value) for value in candidate) == canonical
        ]
        if canonical_indices:
            canonical_sobol_index = int(canonical_indices[0])
            total = np.full(len(candidates), -1e300, dtype=float)
            total[canonical_sobol_index] = 0.0
        else:
            # Compatibility fallback for callers that rank an externally
            # supplied pool. The algorithm mainline always injects the exact
            # canonical continuation point before reaching this backend.
            total = _sobol_scores(problem, candidates, iteration, seed)
            new_mask = np.asarray([
                tuple(int(v) for v in x) not in observed_points
                for x in candidates
            ], dtype=bool)
            if np.any(new_mask):
                total = np.where(new_mask, total, -1e300)
    elif name in {
        "certificate_depth_new", "cert_depth_new", "safe_depth_new",
    }:
        # Isolate whether the learned risk coordinate can rank a genuinely
        # deep-safe new point. This is the same theory margin used by terminal
        # certification, with objective utility deliberately removed.
        total = -np.asarray(theory_margin, dtype=float)
        observed_points = _observed_point_set(observed)
        new_mask = np.asarray([
            tuple(int(v) for v in x) not in observed_points
            for x in candidates
        ], dtype=bool)
        if np.any(new_mask):
            total = np.where(new_mask, total, -1e300)
    elif name in {
        "sobol_hvd_voi", "hvd_voi_sobol",
        "sobol_joint_voi", "joint_voi_sobol",
        "sobol_exact_joint_voi", "exact_joint_voi_sobol",
    }:
        joint_voi = name in {"sobol_joint_voi", "joint_voi_sobol"}
        exact_refit_required = name in {
            "sobol_exact_joint_voi", "exact_joint_voi_sobol",
        }
        action_policy = str(
            evaluate_or_replicate_new_action_policy or ""
        ).strip().lower().replace("-", "_")
        supplemental_priorities = None
        if action_policy in {
            "canonical_plus_posterior_certificate_coverage",
            "canonical_plus_posterior_risk_certificate_coverage",
        }:
            risk_coordinate_coverage, risk_coordinate_coverage_source = (
                _risk_coordinate_coverage_scores(
                    problem, candidates, observed)
            )
            supplemental_priorities = [
                ("certificate_depth", np.asarray(theory_margin, dtype=float)),
                ("psi_coverage", -np.asarray(
                    risk_coordinate_coverage, dtype=float)),
            ]
        action_set = evaluate_or_replicate_action_set(
            candidates,
            observed,
            problem,
            iteration=iteration,
            seed=seed,
            replication_max_per_solution=replication_max_per_solution,
            canonical_sobol_candidate=canonical_sobol_candidate,
            allow_replication_actions=allow_replication_actions,
            new_action_count=evaluate_or_replicate_new_action_count,
            new_action_policy=evaluate_or_replicate_new_action_policy,
            new_action_priority=components["bayes_risk"],
            baseline_new_action_count=(
                evaluate_or_replicate_baseline_new_action_count),
            supplemental_new_action_priorities=supplemental_priorities,
        )
        active_indices = np.asarray(
            action_set["active_indices"], dtype=int)
        evaluate_or_replicate_active_indices = active_indices
        hvd_is_replicate = np.asarray(
            action_set["replicate_mask"], dtype=bool)
        hvd_sobol_new_index = action_set["sobol_new_index"]
        canonical_sobol_index = action_set["canonical_sobol_index"]
        evaluate_or_replicate_new_indices = np.asarray(
            action_set["new_action_indices"], dtype=int)
        baseline_new_indices = np.asarray(
            action_set["baseline_new_action_indices"], dtype=int)
        replication_indices = np.flatnonzero(hvd_is_replicate)
        evaluate_or_replicate_baseline_indices = np.asarray(
            list(dict.fromkeys(
                baseline_new_indices.tolist()
                + replication_indices.tolist()
            )),
            dtype=int,
        )
        evaluate_or_replicate_supplemental_indices = np.asarray(
            action_set["supplemental_new_action_indices"], dtype=int)
        evaluate_or_replicate_supplemental_labels = list(
            action_set["supplemental_new_action_labels"])
        evaluate_or_replicate_new_policy = str(
            action_set["new_action_policy"])

        if exact_refit_required:
            # The caller replaces these placeholders with fantasy-update
            # values after cloning and refitting GPR, robust HC3 and HVD.
            total = np.full(len(candidates), -1e300, dtype=float)
            total[active_indices] = 0.0
        else:
            hvd_reliability = np.ones(len(candidates), dtype=float)
            fresh_reliability = (
                moments.constraint_aleatoric
                / np.maximum(
                    moments.constraint_aleatoric
                    + moments.constraint_epistemic
                    + moments.constraint_between_expert,
                    _EPS,
                )
            )
            hvd_reliability[~hvd_is_replicate] = np.clip(
                fresh_reliability[~hvd_is_replicate], 0.0, 1.0)
            boundary_weight = 0.05 + np.exp(-0.5 * (
                components["stochastic_margin_mean"]
                / np.maximum(components["margin_sd"], 1e-8)
            ) ** 2)
            action_candidates = [candidates[index] for index in active_indices]
            action_reliability = hvd_reliability[active_indices]
            active_gain = _hvd_information_reduction(
                action_candidates,
                candidates,
                variance_model,
                problem,
                action_reliability=action_reliability,
                reference_weights=boundary_weight,
                task_ensemble=task_ensemble,
            )
            hvd_information = np.zeros(len(candidates), dtype=float)
            hvd_information[active_indices] = active_gain
            if joint_voi:
                z_alpha = float(norm.ppf(1.0 - float(problem.alpha)))
                margin_hvd_gain = _hvd_certification_margin_reduction(
                    action_candidates,
                    candidates,
                    variance_model,
                    problem,
                    action_reliability=action_reliability,
                    reference_weights=boundary_weight,
                    z_alpha=z_alpha,
                    task_ensemble=task_ensemble,
                )
                epistemic_gain = _constraint_epistemic_margin_reduction(
                    action_candidates,
                    candidates,
                    con_gpr,
                    variance_model,
                    problem,
                    reference_weights=boundary_weight,
                    beta_g=certification_beta_g,
                    task_ensemble=task_ensemble,
                )
                hvd_margin_information = np.zeros(len(candidates), dtype=float)
                constraint_epistemic_information = np.zeros(
                    len(candidates), dtype=float)
                joint_information = np.zeros(len(candidates), dtype=float)
                hvd_margin_information[active_indices] = margin_hvd_gain
                constraint_epistemic_information[active_indices] = epistemic_gain
                joint_information[active_indices] = (
                    margin_hvd_gain + epistemic_gain)
                active_gain = joint_information[active_indices]
            total = np.full(len(candidates), -1e300, dtype=float)
            total[active_indices] = active_gain
            if hvd_sobol_new_index is not None:
                tie_scale = max(float(np.max(active_gain)), 1.0)
                total[hvd_sobol_new_index] += np.finfo(float).eps * tie_scale
    elif name in {"risk_ts", "risk_aware_ts", "thompson"}:
        if task_ensemble is None:
            sampled_objective = _joint_gpr_draw(obj_gpr, candidates, rng)
            sampled_constraint = _joint_gpr_draw(con_gpr, candidates, rng)
        else:
            sampled_objective, sampled_constraint, sampled_expert = (
                _task_posterior_draw(task_ensemble, candidates, rng))
        z_alpha = float(norm.ppf(1.0 - float(problem.alpha)))
        sampled_margin = (
            sampled_constraint
            + z_alpha * np.sqrt(np.maximum(
                moments.constraint_aleatoric, _EPS))
            - float(problem.tau)
        )
        sampled_violation_loss = (
            (sampled_margin > 0.0).astype(float)
            if components["violation_loss_mode"] == "failure_probability"
            else np.maximum(sampled_margin, 0.0)
        )
        total = -(
            sampled_objective
            + float(risk_penalty) * sampled_violation_loss)
    elif name in {"bayes_risk_ei", "risk_ei"}:
        total = bayes_ei
    elif name in {"constrained_ei", "cei"}:
        total = constrained_ei
    elif name in {"transfer_utility", "source_utility", "utility"}:
        transferred = _source_utility(
            problem, candidates, components, moments)
        if transferred is None:
            transferred = np.zeros(len(candidates), dtype=float)
            utility_status = "source_mean_prior_unavailable"
        else:
            utility_status = "source_mean_prior_active"
        ei_scale = max(float(np.max(bayes_ei)), 1e-12)
        utility_scale = max(float(np.max(np.abs(transferred))), 1e-12)
        total = (
            bayes_ei / ei_scale
            + float(source_utility_weight) * transferred / utility_scale
        )
    elif name in {"n0_best", "frozen_incumbent"}:
        total = -components["bayes_risk"]
    else:
        raise ValueError(f"unknown decision backend {backend!r}")

    out = {
        "backend": name,
        "total": np.asarray(total, dtype=float),
        "posterior_source": moments.source,
        "objective_mean": moments.objective_mean,
        "objective_epistemic": moments.objective_epistemic,
        "constraint_mean": moments.constraint_mean,
        "constraint_epistemic": moments.constraint_epistemic,
        "constraint_aleatoric": moments.constraint_aleatoric,
        "constraint_between_expert": moments.constraint_between_expert,
        "stochastic_margin_mean": components["stochastic_margin_mean"],
        "expected_violation": components["expected_violation"],
        "probability_violation": components["probability_violation"],
        "violation_loss": components["violation_loss"],
        "violation_loss_mode": components["violation_loss_mode"],
        "decision_aleatoric_mode": str(decision_aleatoric_mode),
        "decision_ambiguity_mode": str(decision_ambiguity_mode),
        "probability_feasible": components["probability_feasible"],
        "theory_margin": np.asarray(theory_margin, dtype=float),
        "robust_certificate_mode": certificate_mode,
        "bayes_risk": components["bayes_risk"],
        "bayes_risk_ei": bayes_ei,
        "constrained_ei": constrained_ei,
        "incumbent_bayes_risk": float(incumbent),
        "sampled_expert": sampled_expert,
        "canonical_sobol_index": canonical_sobol_index,
        "canonical_sobol_injected": canonical_sobol_index is not None,
        "replication_actions_enabled": bool(allow_replication_actions),
        "evaluate_or_replicate_active_indices": (
            None
            if evaluate_or_replicate_active_indices is None
            else np.asarray(
                evaluate_or_replicate_active_indices, dtype=int)
        ),
        "evaluate_or_replicate_exact_refit_required": bool(
            exact_refit_required),
        "risk_coordinate_coverage": risk_coordinate_coverage,
        "risk_coordinate_coverage_source": risk_coordinate_coverage_source,
    }
    if name in {"transfer_utility", "source_utility", "utility"}:
        out["transfer_utility_status"] = utility_status
        out["transfer_utility"] = transferred
    if hvd_information is not None:
        out["hvd_information_reduction"] = hvd_information
        out["hvd_action_reliability"] = hvd_reliability
        out["hvd_action_is_replicate"] = hvd_is_replicate.astype(float)
        out["hvd_sobol_new_index"] = hvd_sobol_new_index
    elif evaluate_or_replicate_active_indices is not None:
        out["hvd_action_is_replicate"] = hvd_is_replicate.astype(float)
        out["hvd_sobol_new_index"] = hvd_sobol_new_index
    if evaluate_or_replicate_active_indices is not None:
        active = np.asarray(
            evaluate_or_replicate_active_indices, dtype=int)
        out["evaluate_or_replicate_active_count"] = int(len(active))
        out["evaluate_or_replicate_new_action_count"] = int(sum(
            not bool(hvd_is_replicate[index]) for index in active
        ))
        out["evaluate_or_replicate_replication_action_count"] = int(sum(
            bool(hvd_is_replicate[index]) for index in active
        ))
        out["evaluate_or_replicate_new_action_indices"] = (
            evaluate_or_replicate_new_indices.copy())
        out["evaluate_or_replicate_new_action_policy"] = (
            evaluate_or_replicate_new_policy)
        out["evaluate_or_replicate_baseline_indices"] = (
            None
            if evaluate_or_replicate_baseline_indices is None
            else evaluate_or_replicate_baseline_indices.copy()
        )
        out["evaluate_or_replicate_supplemental_indices"] = (
            None
            if evaluate_or_replicate_supplemental_indices is None
            else evaluate_or_replicate_supplemental_indices.copy()
        )
        out["evaluate_or_replicate_supplemental_labels"] = list(
            evaluate_or_replicate_supplemental_labels)
    if constraint_epistemic_information is not None:
        out["constraint_epistemic_information_reduction"] = (
            constraint_epistemic_information)
        out["hvd_margin_information_reduction"] = hvd_margin_information
        out["joint_information_reduction"] = joint_information
        out["joint_information_unit"] = "chance_margin_response"
        out["joint_information_contract"] = "sqrt_radius_reduction_v2"
    return out
