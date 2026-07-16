"""Interchangeable decision rules over one SC-OLH posterior state.

The backends in this module do not generate candidates or fit source priors.
They only rank a shared candidate set using the same target-budget posterior.
This separation is important for attributing gains to the structural model,
the source proposal, or the online decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, qmc


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


def posterior_moments(
    candidates,
    obj_gpr,
    con_gpr,
    variance_model,
    problem,
    *,
    task_ensemble=None,
):
    """Extract common posterior moments for every online backend."""

    if task_ensemble is not None:
        objective = task_ensemble.mixture_moments_many(
            0, candidates, certification=False)
        constraint = task_ensemble.robust_moments_many(
            1, candidates, certification=True)
        return DecisionMoments(
            objective_mean=np.asarray(objective.mean, dtype=float),
            objective_epistemic=np.maximum(
                np.asarray(objective.epistemic, dtype=float), _EPS),
            constraint_mean=np.asarray(constraint.mean_upper, dtype=float),
            constraint_epistemic=np.maximum(
                np.asarray(constraint.epistemic_upper, dtype=float), _EPS),
            constraint_aleatoric=np.maximum(
                np.asarray(constraint.aleatoric_upper, dtype=float), _EPS),
            constraint_between_expert=np.maximum(
                np.asarray(constraint.nominal.between_mean, dtype=float), 0.0),
            source="task_posterior_robust_cumulative",
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
        constraint_aleatoric=_variance_model_many(
            variance_model, 1, candidates, problem),
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


def _posterior_loss_components(moments, problem, risk_penalty):
    z_alpha = float(norm.ppf(1.0 - float(problem.alpha)))
    stochastic_margin_mean = (
        moments.constraint_mean
        + z_alpha * np.sqrt(np.maximum(moments.constraint_aleatoric, _EPS))
        - float(problem.tau)
    )
    margin_sd = np.sqrt(np.maximum(moments.constraint_epistemic, _EPS))
    expected_violation = normal_positive_part(
        stochastic_margin_mean, margin_sd)
    bayes_risk = (
        moments.objective_mean + float(risk_penalty) * expected_violation)
    probability_feasible = norm.cdf(-stochastic_margin_mean / margin_sd)
    return {
        "stochastic_margin_mean": stochastic_margin_mean,
        "margin_sd": margin_sd,
        "expected_violation": expected_violation,
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
    )
    loss = _posterior_loss_components(
        moments, problem, risk_penalty)["bayes_risk"]
    return float(np.min(loss))


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
    source_utility_weight=1.0,
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
    )
    components = _posterior_loss_components(moments, problem, risk_penalty)
    incumbent = _observed_incumbent_loss(
        observed,
        obj_gpr,
        con_gpr,
        variance_model,
        problem,
        task_ensemble,
        risk_penalty,
        components["bayes_risk"],
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

    if name in {"random", "random_continuation"}:
        total = rng.random(len(candidates))
    elif name in {"sobol", "sobol_continuation"}:
        bounds_lo, bounds_hi = problem.int_bounds()
        lo = np.asarray(bounds_lo, dtype=float)
        hi = np.asarray(bounds_hi, dtype=float)
        engine = qmc.Sobol(
            d=int(problem.d),
            scramble=True,
            seed=int(seed),
        )
        target = engine.random(max(1, int(iteration) + 1))[-1]
        normalized = (
            np.asarray(candidates, dtype=float) - lo
        ) / np.maximum(hi - lo, 1.0)
        total = -np.sum((normalized - target) ** 2, axis=1)
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
        total = -(
            sampled_objective
            + float(risk_penalty) * np.maximum(sampled_margin, 0.0)
        )
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
        "probability_feasible": components["probability_feasible"],
        "bayes_risk": components["bayes_risk"],
        "bayes_risk_ei": bayes_ei,
        "constrained_ei": constrained_ei,
        "incumbent_bayes_risk": float(incumbent),
        "sampled_expert": sampled_expert,
    }
    if name in {"transfer_utility", "source_utility", "utility"}:
        out["transfer_utility_status"] = utility_status
        out["transfer_utility"] = transferred
    return out
