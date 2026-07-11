"""Finite task-structure posterior for transferable SC-OLH-KG.

The posterior is deliberately independent of simulators and surrogate model
implementations.  Experts provide predictive moments; this module performs a
tempered generalized-Bayes update, hierarchical variance aggregation, and
KL-robust moment bounds.  Keeping this layer numerical makes the leakage
boundary explicit: it can only consume observations already charged to the
target evaluation budget.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass(frozen=True)
class MixtureMoments:
    """Hierarchical predictive moments under a finite task posterior."""

    mean: np.ndarray
    within_epistemic: np.ndarray
    between_mean: np.ndarray
    epistemic: np.ndarray
    aleatoric: np.ndarray
    total: np.ndarray


@dataclass(frozen=True)
class RobustMixtureMoments:
    """Conservative moment bounds over a KL ball around the posterior."""

    mean_upper: np.ndarray
    epistemic_upper: np.ndarray
    aleatoric_upper: np.ndarray
    total_upper: np.ndarray
    nominal: MixtureMoments
    radius: float


class FiniteTaskPosterior:
    """Tempered posterior over a finite set of transferable structures.

    The update uses a composite proper score: Gaussian log scores for all
    output channels plus an optional Bernoulli log score for the observed
    chance-boundary event.  It never consumes target truth, an optimum, or an
    uncharged simulator call.
    """

    def __init__(
        self,
        expert_names,
        prior_weights=None,
        *,
        temperature=0.5,
        temperature_decay=0.5,
        output_score_weights=None,
        boundary_score_weight=0.25,
        variance_floor=1e-10,
        minimum_weight=1e-12,
        robust_dual_grid_size=49,
        robust_dual_log_span=18.0,
        robust_dual_bisection_steps=32,
        decision_prior_protection_numerator=0.0,
        decision_prior_protection_max=0.5,
        safe_generalized=False,
        safe_boundary_score_weight=1.0,
        safe_pairwise_score_weight=1.0,
    ):
        names = tuple(str(name) for name in expert_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("task experts must be a non-empty unique sequence")
        self.expert_names = names
        self.temperature = max(float(temperature), 0.0)
        self.temperature_decay = max(float(temperature_decay), 0.0)
        self.boundary_score_weight = max(float(boundary_score_weight), 0.0)
        self.variance_floor = max(float(variance_floor), 1e-15)
        self.minimum_weight = max(float(minimum_weight), 0.0)
        self.robust_dual_grid_size = max(5, int(robust_dual_grid_size))
        self.robust_dual_log_span = max(float(robust_dual_log_span), 1.0)
        self.robust_dual_bisection_steps = max(
            0, int(robust_dual_bisection_steps))
        self.decision_prior_protection_numerator = max(
            float(decision_prior_protection_numerator), 0.0)
        self.decision_prior_protection_max = float(np.clip(
            decision_prior_protection_max, 0.0, 1.0))
        self.safe_generalized = bool(safe_generalized)
        self.safe_boundary_score_weight = max(
            float(safe_boundary_score_weight), 0.0)
        self.safe_pairwise_score_weight = max(
            float(safe_pairwise_score_weight), 0.0)

        if prior_weights is None:
            prior = np.ones(len(names), dtype=float)
        else:
            prior = np.asarray(prior_weights, dtype=float).reshape(-1)
        if len(prior) != len(names) or np.any(prior < 0.0):
            raise ValueError("prior weights must be non-negative and match experts")
        if not np.all(np.isfinite(prior)) or float(np.sum(prior)) <= 0.0:
            raise ValueError("prior weights must contain positive finite mass")
        prior = prior / float(np.sum(prior))
        self._log_prior = np.log(np.maximum(prior, self.minimum_weight or 1e-300))
        self._log_prior -= logsumexp(self._log_prior)
        self._log_weights = self._log_prior.copy()
        self._log_safe_weights = self._log_prior.copy()

        self.output_score_weights = (
            None
            if output_score_weights is None
            else np.asarray(output_score_weights, dtype=float).reshape(-1)
        )
        if self.output_score_weights is not None and (
            np.any(self.output_score_weights < 0.0)
            or not np.all(np.isfinite(self.output_score_weights))
        ):
            raise ValueError("output score weights must be finite and non-negative")
        self.n_updates = 0
        self.cumulative_log_score = np.zeros(len(names), dtype=float)
        self.safe_cumulative_log_score = np.zeros(len(names), dtype=float)
        self.last_update = {"status": "uninitialized"}

    def clone(self):
        return copy.deepcopy(self)

    def posterior_weights(self):
        return np.exp(self._log_weights).copy()

    def safe_posterior_weights(self):
        return np.exp(self._log_safe_weights).copy()

    def decision_posterior_weights(self):
        if self.safe_generalized:
            return self.safe_posterior_weights()
        return self.posterior_weights()

    def prior_weights(self):
        return np.exp(self._log_prior).copy()

    def decision_prior_mix(self):
        numerator = max(float(getattr(
            self, "decision_prior_protection_numerator", 0.0)), 0.0)
        maximum = float(np.clip(getattr(
            self, "decision_prior_protection_max", 0.5), 0.0, 1.0))
        if numerator <= 0.0 or maximum <= 0.0:
            return 0.0
        return float(min(
            maximum,
            numerator / np.sqrt(max(int(self.n_updates), 1)),
        ))

    def decision_weights(self):
        """Posterior decision mass with a vanishing frozen-prior component."""
        return self.proposal_weights(exploration=self.decision_prior_mix())

    def proposal_weights(self, exploration=0.10):
        """Mix posterior mass with the frozen source prior for exploration.

        The prior component prevents a finite-sample posterior from deleting
        an expert's proposal support before target evidence can distinguish
        it.  This is a probability mixture, not a hard admission gate.
        """
        epsilon = float(np.clip(exploration, 0.0, 1.0))
        weights = (
            (1.0 - epsilon) * self.decision_posterior_weights()
            + epsilon * self.prior_weights()
        )
        weights /= max(float(np.sum(weights)), 1e-300)
        return weights

    def proposal_allocation(
        self,
        total,
        *,
        exploration=0.10,
        minimum_per_expert=0,
    ):
        """Allocate an integer proposal budget by deterministic rounding."""
        total = max(0, int(total))
        n_experts = len(self.expert_names)
        counts = np.zeros(n_experts, dtype=int)
        if total == 0:
            return {
                name: 0 for name in self.expert_names
            }
        minimum = max(0, int(minimum_per_expert))
        if minimum > 0 and total >= minimum * n_experts:
            counts[:] = minimum
        remaining = total - int(np.sum(counts))
        weights = self.proposal_weights(exploration=exploration)
        fractional = remaining * weights
        floor = np.floor(fractional).astype(int)
        counts += floor
        leftover = total - int(np.sum(counts))
        if leftover > 0:
            # Stable index order makes equal-weight allocations reproducible.
            remainder = fractional - floor
            order = np.argsort(-remainder, kind="stable")
            counts[order[:leftover]] += 1
        return {
            name: int(count)
            for name, count in zip(self.expert_names, counts)
        }

    def _effective_temperature(self):
        if self.temperature_decay <= 0.0:
            return self.temperature
        return self.temperature / (
            max(1, self.n_updates + 1) ** self.temperature_decay
        )

    def _normalize_log_weights(self, values):
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("non-finite task-posterior log weights")
        values = values - logsumexp(values)
        if self.minimum_weight > 0.0:
            weights = np.maximum(np.exp(values), self.minimum_weight)
            weights /= float(np.sum(weights))
            values = np.log(weights)
        return values

    def update_from_predictive(
        self,
        observation,
        means,
        epistemic_vars,
        aleatoric_vars,
        *,
        tau=None,
        constraint_index=1,
        safe_pairwise_log_score=None,
        safe_pairwise_pairs=0,
        safe_pairwise_effective_weight=0.0,
    ):
        """Update weights from expert predictive moments at one evaluated point."""
        y = np.asarray(observation, dtype=float).reshape(-1)
        mu = np.asarray(means, dtype=float)
        epi = np.asarray(epistemic_vars, dtype=float)
        alea = np.asarray(aleatoric_vars, dtype=float)
        expected_shape = (len(self.expert_names), len(y))
        if mu.shape != expected_shape or epi.shape != expected_shape:
            raise ValueError("expert mean/epistemic arrays have the wrong shape")
        if alea.shape != expected_shape:
            raise ValueError("expert aleatoric array has the wrong shape")
        if not all(np.all(np.isfinite(value)) for value in (y, mu, epi, alea)):
            raise FloatingPointError("task-posterior update received non-finite moments")

        predictive_var = np.maximum(epi + alea, self.variance_floor)
        gaussian_scores = -0.5 * (
            np.log(2.0 * np.pi * predictive_var)
            + (y[None, :] - mu) ** 2 / predictive_var
        )
        if self.output_score_weights is None:
            output_weights = np.ones(len(y), dtype=float)
        else:
            if len(self.output_score_weights) != len(y):
                raise ValueError("output score weights do not match observation")
            output_weights = self.output_score_weights
        gaussian_total = gaussian_scores @ output_weights

        boundary_scores = np.zeros(len(self.expert_names), dtype=float)
        boundary_requested = (
            self.boundary_score_weight > 0.0
            or (
                self.safe_generalized
                and self.safe_boundary_score_weight > 0.0
            )
        )
        if tau is not None and boundary_requested:
            index = int(constraint_index)
            if index < 0 or index >= len(y):
                raise IndexError("constraint index is outside observation")
            sd = np.sqrt(predictive_var[:, index])
            probability = norm.cdf((float(tau) - mu[:, index]) / sd)
            probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
            event = float(y[index] <= float(tau))
            boundary_scores = (
                event * np.log(probability)
                + (1.0 - event) * np.log1p(-probability)
            )

        total_score = (
            gaussian_total
            + self.boundary_score_weight * boundary_scores
        )
        pairwise_score = (
            np.zeros(len(self.expert_names), dtype=float)
            if safe_pairwise_log_score is None
            else np.asarray(safe_pairwise_log_score, dtype=float).reshape(-1)
        )
        if len(pairwise_score) != len(self.expert_names):
            raise ValueError("safe pairwise scores must match task experts")
        if not np.all(np.isfinite(pairwise_score)):
            raise FloatingPointError("safe pairwise scores must be finite")
        index = int(constraint_index)
        if index < 0 or index >= len(y):
            raise IndexError("constraint index is outside observation")
        safe_total_score = (
            gaussian_scores[:, index]
            + self.safe_boundary_score_weight * boundary_scores
            + self.safe_pairwise_score_weight * pairwise_score
        )
        eta = self._effective_temperature()
        before = self.posterior_weights()
        safe_before = self.safe_posterior_weights()
        self._log_weights = self._normalize_log_weights(
            self._log_weights + eta * total_score
        )
        self.cumulative_log_score += total_score
        if self.safe_generalized:
            self._log_safe_weights = self._normalize_log_weights(
                self._log_safe_weights + eta * safe_total_score
            )
            self.safe_cumulative_log_score += safe_total_score
        else:
            self._log_safe_weights = self._log_weights.copy()
            self.safe_cumulative_log_score = self.cumulative_log_score.copy()
        self.n_updates += 1
        after = self.posterior_weights()
        safe_after = self.safe_posterior_weights()
        self.last_update = {
            "status": "updated",
            "temperature": float(eta),
            "gaussian_log_score": gaussian_total.tolist(),
            "boundary_log_score": boundary_scores.tolist(),
            "total_log_score": total_score.tolist(),
            "weights_before": before.tolist(),
            "weights_after": after.tolist(),
            "safe_generalized": bool(self.safe_generalized),
            "safe_constraint_log_score": gaussian_scores[:, index].tolist(),
            "safe_pairwise_log_score": pairwise_score.tolist(),
            "safe_pairwise_pairs": int(safe_pairwise_pairs),
            "safe_pairwise_effective_weight": float(
                safe_pairwise_effective_weight),
            "safe_total_log_score": safe_total_score.tolist(),
            "safe_weights_before": safe_before.tolist(),
            "safe_weights_after": safe_after.tolist(),
            "observation_source": "budgeted_target_evaluation",
            "target_oracle_used": False,
        }
        return dict(self.last_update)

    def mixture_moments(
        self,
        means,
        epistemic_vars,
        aleatoric_vars,
        *,
        weights=None,
    ):
        """Apply the law of total variance along the expert axis."""
        mu = self._expert_matrix(means, "means")
        epi = self._expert_matrix(epistemic_vars, "epistemic variances")
        alea = self._expert_matrix(aleatoric_vars, "aleatoric variances")
        if epi.shape != mu.shape or alea.shape != mu.shape:
            raise ValueError("all expert moment matrices must have equal shape")
        weights = (
            self.decision_weights()
            if weights is None
            else np.asarray(weights, dtype=float).reshape(-1)
        )
        if len(weights) != len(self.expert_names):
            raise ValueError("mixture weights must match task experts")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("mixture weights must be finite and nonnegative")
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("mixture weights must contain positive mass")
        weights = (weights / weight_sum)[:, None]
        mean = np.sum(weights * mu, axis=0)
        within_epistemic = np.sum(weights * np.maximum(epi, 0.0), axis=0)
        between_mean = np.sum(weights * (mu - mean[None, :]) ** 2, axis=0)
        epistemic = within_epistemic + between_mean
        aleatoric = np.sum(weights * np.maximum(alea, 0.0), axis=0)
        return MixtureMoments(
            mean=mean,
            within_epistemic=within_epistemic,
            between_mean=between_mean,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=epistemic + aleatoric,
        )

    def robust_mixture_moments(
        self,
        means,
        epistemic_vars,
        aleatoric_vars,
        radius,
    ):
        """Upper-bound moments for every posterior in a forward-KL ball.

        For any alternative weights ``q``, ``Var_q(mu)`` is bounded by
        ``E_q[(mu - E_p[mu])^2]``.  Robust expectation of that quantity plus
        expert epistemic variance therefore gives a valid epistemic upper
        bound without solving a non-convex variance maximization problem.
        """
        mu = self._expert_matrix(means, "means")
        epi = self._expert_matrix(epistemic_vars, "epistemic variances")
        alea = self._expert_matrix(aleatoric_vars, "aleatoric variances")
        nominal = self.mixture_moments(mu, epi, alea)
        rho = max(float(radius), 0.0)
        if rho <= 0.0:
            return RobustMixtureMoments(
                mean_upper=nominal.mean.copy(),
                epistemic_upper=nominal.epistemic.copy(),
                aleatoric_upper=nominal.aleatoric.copy(),
                total_upper=nominal.total.copy(),
                nominal=nominal,
                radius=0.0,
            )
        mean_upper = self.kl_robust_expectation(mu, rho)
        epistemic_payoff = np.maximum(epi, 0.0) + (
            mu - nominal.mean[None, :]
        ) ** 2
        epistemic_upper = self.kl_robust_expectation(epistemic_payoff, rho)
        aleatoric_upper = self.kl_robust_expectation(
            np.maximum(alea, 0.0), rho)
        return RobustMixtureMoments(
            mean_upper=np.asarray(mean_upper, dtype=float),
            epistemic_upper=np.maximum(epistemic_upper, 0.0),
            aleatoric_upper=np.maximum(aleatoric_upper, 0.0),
            total_upper=np.maximum(epistemic_upper + aleatoric_upper, 0.0),
            nominal=nominal,
            radius=rho,
        )

    def kl_robust_expectation(self, values, radius):
        """Return ``sup_q E_q[value]`` subject to ``KL(q || p) <= radius``."""
        matrix = self._expert_matrix(values, "robust expectation values")
        rho = max(float(radius), 0.0)
        if rho <= 0.0:
            result = self.decision_weights() @ matrix
        else:
            result = self._kl_robust_matrix(matrix, rho)
        if np.asarray(values).ndim == 1:
            return float(result[0])
        return result

    def _kl_robust_matrix(self, values, radius):
        """Batched finite-grid evaluation of the entropic KL dual.

        Every positive dual temperature supplies a valid upper bound. Taking
        the minimum over a deterministic finite grid therefore preserves the
        certificate while avoiding thousands of scalar optimizer calls inside
        exact KG.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise FloatingPointError("non-finite KL-robust payoff matrix")
        weights = self.decision_weights()
        maximum = np.max(values, axis=0)
        minimum = np.min(values, axis=0)
        span = maximum - minimum
        nominal = weights @ values
        result = nominal.copy()
        active = span > 1e-14
        if not np.any(active):
            return result

        active_values = values[:, active]
        active_maximum = maximum[active]
        max_mask = active_values >= active_maximum[None, :] - 1e-14
        max_mass = np.sum(weights[:, None] * max_mask, axis=0)
        saturated = radius >= -np.log(np.maximum(max_mass, 1e-300)) - 1e-12

        scale = np.maximum.reduce([
            span[active],
            np.max(np.abs(active_values), axis=0),
            np.ones(np.sum(active), dtype=float),
        ])
        grid = np.linspace(
            -self.robust_dual_log_span,
            self.robust_dual_log_span,
            self.robust_dual_grid_size,
        )
        lambdas = scale[:, None] * np.exp(grid)[None, :]
        log_weights = np.log(np.maximum(weights, 1e-300))
        logits = (
            log_weights[None, :, None]
            + active_values.T[:, :, None] / lambdas[:, None, :]
        )
        dual = lambdas * (logsumexp(logits, axis=1) + float(radius))
        active_result = np.min(dual, axis=1)
        if self.robust_dual_bisection_steps > 0:
            low = scale * np.exp(-self.robust_dual_log_span)
            high = scale * np.exp(self.robust_dual_log_span)
            for _ in range(self.robust_dual_bisection_steps):
                midpoint = np.sqrt(low * high)
                midpoint_logits = (
                    log_weights[None, :]
                    + active_values.T / midpoint[:, None]
                )
                log_normalizer = logsumexp(midpoint_logits, axis=1)
                tilted = np.exp(
                    midpoint_logits - log_normalizer[:, None])
                tilted_mean = np.sum(
                    tilted * active_values.T, axis=1)
                derivative = (
                    log_normalizer
                    + float(radius)
                    - tilted_mean / midpoint
                )
                low = np.where(derivative <= 0.0, midpoint, low)
                high = np.where(derivative > 0.0, midpoint, high)
            refined_lambda = np.sqrt(low * high)
            refined_logits = (
                log_weights[None, :]
                + active_values.T / refined_lambda[:, None]
            )
            refined_dual = refined_lambda * (
                logsumexp(refined_logits, axis=1) + float(radius))
            active_result = np.minimum(active_result, refined_dual)
        active_nominal = nominal[active]
        active_result = np.clip(
            active_result, active_nominal, active_maximum)
        active_result = np.where(
            saturated, active_maximum, active_result)
        result[active] = active_result
        return result

    def _expert_matrix(self, values, label):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or matrix.shape[0] != len(self.expert_names):
            raise ValueError(f"{label} must have experts on the first axis")
        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError(f"non-finite {label}")
        return matrix

    def entropy(self):
        weights = self.decision_posterior_weights()
        return float(-np.sum(weights * np.log(np.maximum(weights, 1e-300))))

    def kl_from_prior(self):
        weights = self.decision_posterior_weights()
        prior = self.prior_weights()
        return float(np.sum(
            weights * (
                np.log(np.maximum(weights, 1e-300))
                - np.log(np.maximum(prior, 1e-300))
            )
        ))

    def diagnostics(self):
        weights = self.posterior_weights()
        safe_weights = self.safe_posterior_weights()
        return {
            "status": "fit" if self.n_updates else "initialized",
            "expert_names": list(self.expert_names),
            "prior_weights": self.prior_weights().tolist(),
            "posterior_weights": weights.tolist(),
            "raw_posterior_weights": self.posterior_weights().tolist(),
            "safe_generalized": bool(self.safe_generalized),
            "safe_posterior_weights": safe_weights.tolist(),
            "decision_weights": self.decision_weights().tolist(),
            "decision_prior_mix": self.decision_prior_mix(),
            "decision_prior_protection_numerator": float(getattr(
                self, "decision_prior_protection_numerator", 0.0)),
            "decision_prior_protection_max": float(getattr(
                self, "decision_prior_protection_max", 0.5)),
            "posterior_by_expert": {
                name: float(weight)
                for name, weight in zip(self.expert_names, weights)
            },
            "safe_posterior_by_expert": {
                name: float(weight)
                for name, weight in zip(self.expert_names, safe_weights)
            },
            "decision_by_expert": {
                name: float(weight)
                for name, weight in zip(
                    self.expert_names, self.decision_weights())
            },
            "entropy": self.entropy(),
            "effective_experts": float(1.0 / np.sum(weights ** 2)),
            "safe_effective_experts": float(
                1.0 / np.sum(safe_weights ** 2)),
            "decision_effective_experts": float(
                1.0 / np.sum(self.decision_weights() ** 2)),
            "kl_from_prior": self.kl_from_prior(),
            "n_updates": int(self.n_updates),
            "cumulative_log_score": self.cumulative_log_score.tolist(),
            "safe_cumulative_log_score": (
                self.safe_cumulative_log_score.tolist()),
            "safe_boundary_score_weight": float(
                self.safe_boundary_score_weight),
            "safe_pairwise_score_weight": float(
                self.safe_pairwise_score_weight),
            "last_update": copy.deepcopy(self.last_update),
            "robust_dual_solver": "batched_finite_entropic_grid",
            "robust_dual_grid_size": int(self.robust_dual_grid_size),
            "robust_dual_log_span": float(self.robust_dual_log_span),
            "robust_dual_bisection_steps": int(
                self.robust_dual_bisection_steps),
            "offline_only": True,
            "target_oracle_used": False,
        }


class FiniteTaskSensitivityPosterior:
    """Posterior over task-level surrogate error sensitivity.

    A class changes the scale of the epistemic part of the prequential
    predictive distribution and carries a corresponding false-feasibility
    decision penalty.  Source tasks provide the prior; held-out observations
    update it through a proper predictive score.  The class does not alter the
    theory certificate, so learning it cannot silently relax feasibility.
    """

    def __init__(
        self,
        class_names=("stable", "balanced", "sensitive"),
        scales=(0.5, 1.0, 2.0),
        decision_penalties=(2.0, 5.0, 20.0),
        empirical_trust=(1.0, 0.25, 0.0),
        prior_weights=None,
        *,
        temperature=0.5,
        temperature_decay=0.5,
        boundary_score_weight=0.25,
        variance_floor=1e-10,
        minimum_weight=1e-12,
    ):
        names = tuple(str(value) for value in class_names)
        scales = np.asarray(scales, dtype=float).reshape(-1)
        penalties = np.asarray(decision_penalties, dtype=float).reshape(-1)
        empirical_trust = np.asarray(empirical_trust, dtype=float).reshape(-1)
        if not names or len(set(names)) != len(names):
            raise ValueError("sensitivity classes must be non-empty and unique")
        if (
            len(scales) != len(names)
            or len(penalties) != len(names)
            or len(empirical_trust) != len(names)
        ):
            raise ValueError("sensitivity class arrays must have equal length")
        if (
            np.any(scales <= 0.0)
            or np.any(penalties < 0.0)
            or not np.all(np.isfinite(scales))
            or not np.all(np.isfinite(penalties))
        ):
            raise ValueError("sensitivity scales and penalties must be finite")
        if (
            np.any(empirical_trust < 0.0)
            or np.any(empirical_trust > 1.0)
            or not np.all(np.isfinite(empirical_trust))
        ):
            raise ValueError("empirical trust must lie in [0, 1]")
        if prior_weights is None:
            prior = np.ones(len(names), dtype=float)
        else:
            prior = np.asarray(prior_weights, dtype=float).reshape(-1)
        if (
            len(prior) != len(names)
            or np.any(prior < 0.0)
            or not np.all(np.isfinite(prior))
            or float(np.sum(prior)) <= 0.0
        ):
            raise ValueError("invalid sensitivity prior weights")
        prior /= float(np.sum(prior))

        self.class_names = names
        self.scales = scales
        self.decision_penalties = penalties
        self.empirical_trust = empirical_trust
        self.temperature = max(float(temperature), 0.0)
        self.temperature_decay = max(float(temperature_decay), 0.0)
        self.boundary_score_weight = max(float(boundary_score_weight), 0.0)
        self.variance_floor = max(float(variance_floor), 1e-15)
        self.minimum_weight = max(float(minimum_weight), 0.0)
        self._log_prior = np.log(np.maximum(
            prior, self.minimum_weight or 1e-300))
        self._log_prior -= logsumexp(self._log_prior)
        self._log_weights = self._log_prior.copy()
        self.n_updates = 0
        self.cumulative_log_score = np.zeros(len(names), dtype=float)
        self.last_update = {"status": "uninitialized"}

    def clone(self):
        return copy.deepcopy(self)

    def prior_weights(self):
        return np.exp(self._log_prior).copy()

    def posterior_weights(self):
        return np.exp(self._log_weights).copy()

    def _effective_temperature(self):
        if self.temperature_decay <= 0.0:
            return self.temperature
        return self.temperature / (
            max(1, self.n_updates + 1) ** self.temperature_decay)

    def update_from_predictive(
        self,
        observation,
        mean,
        epistemic_var,
        aleatoric_var,
        *,
        tau=None,
    ):
        y = float(observation)
        mu = float(mean)
        epi = max(float(epistemic_var), 0.0)
        alea = max(float(aleatoric_var), 0.0)
        if not np.all(np.isfinite([y, mu, epi, alea])):
            raise FloatingPointError(
                "sensitivity update received non-finite predictive moments")
        predictive_var = np.maximum(
            alea + (self.scales ** 2) * epi,
            self.variance_floor,
        )
        scores = -0.5 * (
            np.log(2.0 * np.pi * predictive_var)
            + ((y - mu) ** 2) / predictive_var
        )
        boundary_scores = np.zeros(len(self.class_names), dtype=float)
        if tau is not None and self.boundary_score_weight > 0.0:
            probability = norm.cdf(
                (float(tau) - mu) / np.sqrt(predictive_var))
            probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
            event = float(y <= float(tau))
            boundary_scores = (
                event * np.log(probability)
                + (1.0 - event) * np.log1p(-probability)
            )
        total_score = scores + self.boundary_score_weight * boundary_scores
        eta = self._effective_temperature()
        before = self.posterior_weights()
        log_weights = self._log_weights + eta * total_score
        log_weights -= logsumexp(log_weights)
        if self.minimum_weight > 0.0:
            weights = np.maximum(np.exp(log_weights), self.minimum_weight)
            weights /= float(np.sum(weights))
            log_weights = np.log(weights)
        self._log_weights = log_weights
        self.cumulative_log_score += total_score
        self.n_updates += 1
        after = self.posterior_weights()
        self.last_update = {
            "status": "updated",
            "temperature": float(eta),
            "predictive_variance": predictive_var.tolist(),
            "gaussian_log_score": scores.tolist(),
            "boundary_log_score": boundary_scores.tolist(),
            "total_log_score": total_score.tolist(),
            "weights_before": before.tolist(),
            "weights_after": after.tolist(),
            "observation_source": "budgeted_target_evaluation",
            "target_oracle_used": False,
        }
        return copy.deepcopy(self.last_update)

    def expected_decision_penalty(self):
        return float(self.posterior_weights() @ self.decision_penalties)

    def expected_empirical_trust(self):
        return float(self.posterior_weights() @ self.empirical_trust)

    def posterior_violation_decision_risk(
        self,
        predicted_mean,
        residual_sigma,
        leverage,
        *,
        tau,
        aleatoric_variance=None,
    ):
        """Return the Bayes violation risk under the latent task class.

        The empirical boundary model supplies a residual scale and coefficient
        leverage.  Each latent class scales the epistemic part, while the
        posterior class probability and class-specific violation loss define
        posterior expected decision risk.  This is a recommendation loss, not
        a replacement for the conservative theory certificate.
        """
        mean = np.asarray(predicted_mean, dtype=float).reshape(-1)
        leverage = np.asarray(leverage, dtype=float).reshape(-1)
        if len(mean) != len(leverage):
            raise ValueError("predicted mean and leverage must have equal length")
        if not np.all(np.isfinite(mean)):
            raise FloatingPointError("predicted means must be finite")
        leverage = np.maximum(leverage, 0.0)
        if aleatoric_variance is None:
            aleatoric = np.zeros(len(mean), dtype=float)
        else:
            aleatoric = np.asarray(
                aleatoric_variance, dtype=float).reshape(-1)
            if len(aleatoric) != len(mean):
                raise ValueError(
                    "aleatoric variance and predicted mean must have equal length")
            if not np.all(np.isfinite(aleatoric)):
                raise FloatingPointError("aleatoric variance must be finite")
            aleatoric = np.maximum(aleatoric, 0.0)
        sigma = max(float(residual_sigma), np.sqrt(self.variance_floor))
        class_variance = (
            aleatoric[:, None]
            + sigma ** 2
            + (sigma ** 2)
            * leverage[:, None]
            * (self.scales[None, :] ** 2)
        )
        class_std = np.sqrt(np.maximum(class_variance, self.variance_floor))
        violation_probability = norm.cdf(
            (mean[:, None] - float(tau)) / class_std
        )
        weights = self.posterior_weights()
        expected_probability = violation_probability @ weights
        expected_decision_risk = violation_probability @ (
            weights * self.decision_penalties
        )
        return {
            "class_variance": class_variance,
            "class_violation_probability": violation_probability,
            "posterior_violation_probability": expected_probability,
            "posterior_expected_decision_risk": expected_decision_risk,
            "posterior_weights": weights.copy(),
            "residual_sigma": float(sigma),
            "aleatoric_variance": aleatoric,
            "affects_theory_certificate": False,
        }

    def diagnostics(self):
        weights = self.posterior_weights()
        return {
            "status": "fit",
            "class_names": list(self.class_names),
            "scales": self.scales.tolist(),
            "decision_penalties": self.decision_penalties.tolist(),
            "empirical_trust": self.empirical_trust.tolist(),
            "prior_weights": self.prior_weights().tolist(),
            "posterior_weights": weights.tolist(),
            "posterior_by_class": {
                name: float(weight)
                for name, weight in zip(self.class_names, weights)
            },
            "expected_scale": float(weights @ self.scales),
            "expected_scale_squared": float(weights @ (self.scales ** 2)),
            "expected_decision_penalty": self.expected_decision_penalty(),
            "expected_empirical_trust": self.expected_empirical_trust(),
            "entropy": float(-np.sum(weights * np.log(np.maximum(
                weights, 1e-300)))),
            "n_updates": int(self.n_updates),
            "cumulative_log_score": self.cumulative_log_score.tolist(),
            "last_update": copy.deepcopy(self.last_update),
            "affects_theory_certificate": False,
            "offline_only": True,
            "target_oracle_used": False,
        }


@dataclass
class TaskExpertState:
    """Mutable surrogate state associated with one task expert."""

    name: str
    gpr_models: list
    variance_model: object
    problem: object


class FiniteTaskModelEnsemble:
    """Expert surrogate ensemble sharing one finite task posterior."""

    def __init__(
        self,
        states,
        posterior,
        *,
        kl_radius_numerator=0.5,
        confidence_delta=0.05,
        maximum_kl_radius=4.0,
        pilot_count=0,
        sensitivity_posterior=None,
        safe_pairwise_max_history=16,
        safe_pairwise_probability_floor=1e-6,
        safe_history=None,
    ):
        self.states = list(states)
        self.posterior = posterior
        names = tuple(str(state.name) for state in self.states)
        if names != posterior.expert_names:
            raise ValueError("expert states and posterior names disagree")
        self.kl_radius_numerator = max(float(kl_radius_numerator), 0.0)
        self.confidence_delta = float(np.clip(confidence_delta, 1e-12, 1.0))
        self.maximum_kl_radius = max(float(maximum_kl_radius), 0.0)
        self.pilot_count = max(0, int(pilot_count))
        self.sensitivity_posterior = sensitivity_posterior
        self.safe_pairwise_max_history = max(
            0, int(safe_pairwise_max_history))
        self.safe_pairwise_probability_floor = float(np.clip(
            safe_pairwise_probability_floor, 1e-12, 0.49))
        self.safe_history = copy.deepcopy(list(safe_history or []))
        self.last_update = {"status": "uninitialized"}

    def clone(self, gpr_cloner=None, variance_cloner=None):
        gpr_cloner = gpr_cloner or copy.deepcopy
        variance_cloner = variance_cloner or copy.deepcopy
        states = []
        for state in self.states:
            variance_model = variance_cloner(state.variance_model)
            if hasattr(variance_model, "_last_problem"):
                variance_model._last_problem = state.problem
            states.append(TaskExpertState(
                name=state.name,
                gpr_models=[gpr_cloner(model) for model in state.gpr_models],
                variance_model=variance_model,
                problem=state.problem,
            ))
        clone = FiniteTaskModelEnsemble(
            states,
            self.posterior.clone(),
            kl_radius_numerator=self.kl_radius_numerator,
            confidence_delta=self.confidence_delta,
            maximum_kl_radius=self.maximum_kl_radius,
            pilot_count=self.pilot_count,
            sensitivity_posterior=(
                None
                if self.sensitivity_posterior is None
                else self.sensitivity_posterior.clone()
            ),
            safe_pairwise_max_history=self.safe_pairwise_max_history,
            safe_pairwise_probability_floor=(
                self.safe_pairwise_probability_floor),
            safe_history=self.safe_history,
        )
        clone.last_update = copy.deepcopy(self.last_update)
        return clone

    def effective_kl_radius(self):
        """Finite PAC-Bayes complexity radius for target-task uncertainty."""
        n = max(int(self.posterior.n_updates), 1)
        complexity = (
            self.kl_radius_numerator
            + self.posterior.kl_from_prior()
            + np.log(1.0 / self.confidence_delta)
        )
        return float(min(
            self.maximum_kl_radius,
            max(complexity / float(n), 0.0),
        ))

    def expert_moments_many(self, output_index, X, *, certification=True):
        index = int(output_index)
        means = []
        epistemic = []
        aleatoric = []
        for state in self.states:
            model = state.gpr_models[index]
            state_mean = np.asarray(model.posterior_mean_many(X), dtype=float)
            if index == 1 and hasattr(state.problem, "pilot_constraint_guard"):
                state_mean = state_mean + max(
                    float(state.problem.pilot_constraint_guard()), 0.0)
            means.append(state_mean)
            epistemic.append(model.posterior_var_many(X))
            if certification and hasattr(
                state.variance_model, "predict_certification_variance_many"
            ):
                variance = state.variance_model.predict_certification_variance_many(
                    index, X, state.problem)
            else:
                variance = state.variance_model.predict_variance_many(
                    index, X, state.problem)
            aleatoric.append(variance)
        return (
            np.asarray(means, dtype=float),
            np.asarray(epistemic, dtype=float),
            np.asarray(aleatoric, dtype=float),
        )

    def mixture_moments_many(self, output_index, X, *, certification=True):
        moments = self.expert_moments_many(
            output_index, X, certification=certification)
        weights = (
            self.posterior.posterior_weights()
            if int(output_index) == 0 and self.posterior.safe_generalized
            else None
        )
        return self.posterior.mixture_moments(*moments, weights=weights)

    def robust_moments_many(self, output_index, X, *, certification=True):
        moments = self.expert_moments_many(
            output_index, X, certification=certification)
        return self.posterior.robust_mixture_moments(
            *moments,
            radius=self.effective_kl_radius(),
        )

    def predictive_sample(self, x, z, expert_uniform):
        """Sample one shared expert identity and all output channels."""
        weights = self.posterior.decision_weights()
        cumulative = np.cumsum(weights)
        expert_index = int(np.searchsorted(
            cumulative,
            np.clip(float(expert_uniform), 0.0, 1.0 - 1e-15),
            side="right",
        ))
        expert_index = min(expert_index, len(self.states) - 1)
        state = self.states[expert_index]
        z = np.asarray(z, dtype=float).reshape(-1)
        values = []
        for output_index, standard_normal in enumerate(z):
            model = state.gpr_models[output_index]
            mu = float(model.posterior_mean(x))
            epi = float(model.posterior_var(x))
            alea = float(state.variance_model.predict_variance(
                output_index, x, state.problem))
            values.append(float(
                mu + np.sqrt(max(epi + alea, 1e-12)) * standard_normal
            ))
        return np.asarray(values, dtype=float), expert_index

    def _safe_pairwise_log_score(
        self,
        observation,
        means,
        epistemic,
        aleatoric,
        *,
        tau,
        constraint_index,
    ):
        n_experts = len(self.states)
        empty = (np.zeros(n_experts, dtype=float), {
            "pair_count": 0,
            "effective_weight": 0.0,
            "history_count": int(len(self.safe_history)),
        })
        if (
            not self.posterior.safe_generalized
            or tau is None
            or self.safe_pairwise_max_history <= 0
            or not self.safe_history
        ):
            return empty
        eligible = [
            row for row in self.safe_history
            if len(np.asarray(row.get("means", []), dtype=float)) == n_experts
            and len(np.asarray(row.get("variances", []), dtype=float)) == n_experts
        ]
        if not eligible:
            return empty
        eligible = sorted(
            enumerate(eligible),
            key=lambda item: (
                abs(float(item[1]["observation"]) - float(tau)),
                -int(item[0]),
            ),
        )[: self.safe_pairwise_max_history]
        references = [row for _, row in eligible]
        ref_y = np.asarray([
            float(row["observation"]) for row in references
        ], dtype=float)
        observed_difference = float(observation) - ref_y
        usable = np.abs(observed_difference) > 1e-12
        if not np.any(usable):
            return empty
        ref_y = ref_y[usable]
        observed_difference = observed_difference[usable]
        ref_mean = np.vstack([
            np.asarray(row["means"], dtype=float)
            for row, keep in zip(references, usable) if keep
        ]).T
        ref_variance = np.vstack([
            np.asarray(row["variances"], dtype=float)
            for row, keep in zip(references, usable) if keep
        ]).T
        index = int(constraint_index)
        current_mean = np.asarray(means, dtype=float)[:, index, None]
        current_variance = np.maximum(
            np.asarray(epistemic, dtype=float)[:, index]
            + np.asarray(aleatoric, dtype=float)[:, index],
            self.posterior.variance_floor,
        )[:, None]
        pair_variance = np.maximum(
            current_variance + ref_variance,
            self.posterior.variance_floor,
        )
        pair_sd = np.sqrt(pair_variance)
        sign = np.sign(observed_difference)[None, :]
        probability = norm.cdf(
            sign * (current_mean - ref_mean) / pair_sd)
        floor = self.safe_pairwise_probability_floor
        probability = np.clip(probability, floor, 1.0 - floor)

        common_sd = np.sqrt(np.maximum(
            np.median(pair_variance, axis=0),
            self.posterior.variance_floor,
        ))
        separation = np.tanh(
            np.abs(observed_difference) / np.maximum(common_sd, 1e-12))
        boundary_distance = np.minimum(
            abs(float(observation) - float(tau)),
            np.abs(ref_y - float(tau)),
        )
        boundary_relevance = np.exp(
            -boundary_distance / np.maximum(common_sd, 1e-12))
        pair_weight = separation * boundary_relevance
        effective_weight = float(np.sum(pair_weight))
        if effective_weight <= 1e-12:
            return empty
        score = np.sum(
            np.log(probability) * pair_weight[None, :], axis=1
        ) / effective_weight
        return np.asarray(score, dtype=float), {
            "pair_count": int(np.sum(pair_weight > 1e-12)),
            "effective_weight": effective_weight,
            "history_count": int(len(self.safe_history)),
        }

    def update(self, x, observation, existing_observations=None, *, tau=None):
        """Update task weights first, then every expert GPR and HVD."""
        y = np.asarray(observation, dtype=float).reshape(-1)
        point = [tuple(int(value) for value in np.asarray(x, dtype=int))]
        means = []
        epistemic = []
        aleatoric = []
        per_state = []
        for state in self.states:
            state_mu = np.asarray([
                state.gpr_models[i].posterior_mean(point[0])
                for i in range(len(y))
            ], dtype=float)
            state_epi = np.asarray([
                state.gpr_models[i].posterior_var(point[0])
                for i in range(len(y))
            ], dtype=float)
            state_alea = np.asarray([
                state.variance_model.predict_variance(
                    i, point[0], state.problem)
                for i in range(len(y))
            ], dtype=float)
            means.append(state_mu)
            epistemic.append(state_epi)
            aleatoric.append(state_alea)
            per_state.append((state_mu, state_epi, state_alea))

        nominal_before = self.posterior.mixture_moments(
            np.asarray(means, dtype=float),
            np.asarray(epistemic, dtype=float),
            np.asarray(aleatoric, dtype=float),
        )
        constraint_index = min(1, len(y) - 1)
        pairwise_score, pairwise_diagnostics = (
            self._safe_pairwise_log_score(
                y[constraint_index],
                np.asarray(means, dtype=float),
                np.asarray(epistemic, dtype=float),
                np.asarray(aleatoric, dtype=float),
                tau=tau,
                constraint_index=constraint_index,
            )
        )
        posterior_update = self.posterior.update_from_predictive(
            y,
            np.asarray(means, dtype=float),
            np.asarray(epistemic, dtype=float),
            np.asarray(aleatoric, dtype=float),
            tau=tau,
            constraint_index=constraint_index,
            safe_pairwise_log_score=pairwise_score,
            safe_pairwise_pairs=pairwise_diagnostics["pair_count"],
            safe_pairwise_effective_weight=(
                pairwise_diagnostics["effective_weight"]),
        )
        sensitivity_update = {"status": "disabled"}
        if self.sensitivity_posterior is not None:
            sensitivity_update = (
                self.sensitivity_posterior.update_from_predictive(
                    y[constraint_index],
                    nominal_before.mean[constraint_index],
                    nominal_before.epistemic[constraint_index],
                    nominal_before.aleatoric[constraint_index],
                    tau=tau,
                )
            )
        if self.posterior.safe_generalized:
            predictive_variance = np.maximum(
                np.asarray(epistemic, dtype=float)[:, constraint_index]
                + np.asarray(aleatoric, dtype=float)[:, constraint_index],
                self.posterior.variance_floor,
            )
            self.safe_history.append({
                "x": list(map(int, point[0])),
                "observation": float(y[constraint_index]),
                "means": np.asarray(
                    means, dtype=float)[:, constraint_index].tolist(),
                "variances": predictive_variance.tolist(),
                "observation_source": "budgeted_target_evaluation",
                "target_oracle_used": False,
            })
            if self.safe_pairwise_max_history > 0:
                self.safe_history = self.safe_history[
                    -self.safe_pairwise_max_history:
                ]
        existing = list(existing_observations or [])
        expert_updates = []
        for state, (state_mu, state_epi, state_alea) in zip(
            self.states, per_state
        ):
            for output_index in range(len(y)):
                state.gpr_models[output_index].update(
                    point[0],
                    float(y[output_index]),
                    float(state_alea[output_index]),
                )
            hvd_updates = []
            for output_index in range(len(y)):
                replicate_values = [
                    float(np.asarray(value, dtype=float)[output_index])
                    for value in existing
                ] + [float(y[output_index])]
                replicate_variance = (
                    float(np.var(replicate_values, ddof=1))
                    if len(replicate_values) >= 2
                    else None
                )
                hvd_updates.append(state.variance_model.update(
                    output_index,
                    point[0],
                    float(y[output_index]),
                    float(state_mu[output_index]),
                    state.gpr_models[output_index],
                    state.problem,
                    epistemic_var=float(state_epi[output_index]),
                    replicate_variance=replicate_variance,
                ))
            expert_updates.append({
                "expert": state.name,
                "hvd": hvd_updates,
            })
        self.last_update = {
            "status": "updated",
            "posterior": posterior_update,
            "safe_pairwise": pairwise_diagnostics,
            "sensitivity_posterior": sensitivity_update,
            "experts": expert_updates,
            "effective_kl_radius": self.effective_kl_radius(),
        }
        return copy.deepcopy(self.last_update)

    def adaptive_infeasible_penalty(self, fallback=5.0):
        if self.sensitivity_posterior is None:
            return float(fallback)
        return self.sensitivity_posterior.expected_decision_penalty()

    def diagnostics(self):
        return {
            "status": "fit",
            "posterior": self.posterior.diagnostics(),
            "sensitivity_posterior": (
                {"status": "disabled"}
                if self.sensitivity_posterior is None
                else self.sensitivity_posterior.diagnostics()
            ),
            "effective_kl_radius": self.effective_kl_radius(),
            "kl_radius_numerator": self.kl_radius_numerator,
            "confidence_delta": self.confidence_delta,
            "maximum_kl_radius": self.maximum_kl_radius,
            "pilot_count": int(self.pilot_count),
            "safe_history_count": int(len(self.safe_history)),
            "safe_pairwise_max_history": int(
                self.safe_pairwise_max_history),
            "safe_pairwise_probability_floor": float(
                self.safe_pairwise_probability_floor),
            "n_posterior_evidence": int(self.posterior.n_updates),
            "kl_radius_schedule": (
                "(source_slack+KL(Q||Pi)+log(1/delta))/n_evidence"
            ),
            "experts": [
                {
                    "name": state.name,
                    "gpr_adaptive_sparsity": [
                        model.adaptive_sparsity_diagnostics()
                        if hasattr(model, "adaptive_sparsity_diagnostics")
                        else {"status": "unavailable"}
                        for model in state.gpr_models
                    ],
                    "basis": (
                        state.gpr_models[1].basis_map.diagnostics()
                        if len(state.gpr_models) > 1
                        and getattr(state.gpr_models[1], "basis_map", None)
                        is not None
                        and hasattr(
                            state.gpr_models[1].basis_map, "diagnostics")
                        else {"status": "unavailable"}
                    ),
                    "variance": state.variance_model.diagnostics(),
                    "problem_provider": (
                        state.problem.cumulative_risk_provider_status()
                        if hasattr(state.problem, "cumulative_risk_provider_status")
                        else {"status": "missing"}
                    ),
                }
                for state in self.states
            ],
            "last_update": copy.deepcopy(self.last_update),
            "offline_only": True,
            "target_oracle_used": False,
        }
