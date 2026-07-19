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


@dataclass(frozen=True)
class RobustChanceMargin:
    """Joint KL-robust upper bound for one chance-constraint margin."""

    upper: np.ndarray
    nominal: np.ndarray
    separable_upper: np.ndarray
    tangent_epistemic_scale: np.ndarray
    tangent_aleatoric_scale: np.ndarray
    used_separable_upper: np.ndarray
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
        *,
        weights=None,
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
        nominal = self.mixture_moments(
            mu, epi, alea, weights=weights)
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
        mean_upper = self.kl_robust_expectation(
            mu, rho, weights=weights)
        epistemic_payoff = np.maximum(epi, 0.0) + (
            mu - nominal.mean[None, :]
        ) ** 2
        epistemic_upper = self.kl_robust_expectation(
            epistemic_payoff, rho, weights=weights)
        aleatoric_upper = self.kl_robust_expectation(
            np.maximum(alea, 0.0), rho, weights=weights)
        return RobustMixtureMoments(
            mean_upper=np.asarray(mean_upper, dtype=float),
            epistemic_upper=np.maximum(epistemic_upper, 0.0),
            aleatoric_upper=np.maximum(aleatoric_upper, 0.0),
            total_upper=np.maximum(epistemic_upper + aleatoric_upper, 0.0),
            nominal=nominal,
            radius=rho,
        )

    def robust_chance_margin(
        self,
        means,
        epistemic_vars,
        aleatoric_vars,
        *,
        beta_g,
        z_alpha,
        tau,
        radius,
        weights=None,
        tangent_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0),
    ):
        """Upper-bound the full margin under one shared KL-ball posterior.

        The separable bound independently maximizes mean, epistemic variance,
        and aleatoric variance and can therefore combine three incompatible
        worst-case task distributions.  Here the square-root tangent identity

        ``sqrt(v) <= v/(2s) + s/2``

        converts every positive tangent pair into one expert payoff. A single
        KL-robust expectation is then taken for that combined payoff. Taking
        the minimum over a deterministic finite tangent grid preserves the
        upper-bound guarantee.
        """

        mu = self._expert_matrix(means, "means")
        epi = np.maximum(
            self._expert_matrix(epistemic_vars, "epistemic variances"), 0.0)
        alea = np.maximum(
            self._expert_matrix(aleatoric_vars, "aleatoric variances"), 0.0)
        if epi.shape != mu.shape or alea.shape != mu.shape:
            raise ValueError("all expert moment matrices must have equal shape")
        nominal = self.mixture_moments(
            mu, epi, alea, weights=weights)
        robust = self.robust_mixture_moments(
            mu, epi, alea, radius, weights=weights)
        beta_radius = np.sqrt(max(float(beta_g), 0.0))
        aleatoric_radius = max(float(z_alpha), 0.0)
        tau = float(tau)
        nominal_margin = (
            nominal.mean
            + beta_radius * np.sqrt(np.maximum(nominal.epistemic, 0.0))
            + aleatoric_radius * np.sqrt(np.maximum(nominal.aleatoric, 0.0))
            - tau
        )
        separable_margin = (
            robust.mean_upper
            + beta_radius * np.sqrt(np.maximum(robust.epistemic_upper, 0.0))
            + aleatoric_radius * np.sqrt(np.maximum(robust.aleatoric_upper, 0.0))
            - tau
        )
        rho = max(float(radius), 0.0)
        if rho <= 0.0:
            return RobustChanceMargin(
                upper=np.asarray(nominal_margin, dtype=float),
                nominal=np.asarray(nominal_margin, dtype=float),
                separable_upper=np.asarray(separable_margin, dtype=float),
                tangent_epistemic_scale=np.sqrt(np.maximum(
                    nominal.epistemic, self.variance_floor)),
                tangent_aleatoric_scale=np.sqrt(np.maximum(
                    nominal.aleatoric, self.variance_floor)),
                used_separable_upper=np.zeros_like(
                    nominal_margin, dtype=bool),
                radius=0.0,
            )

        multipliers = np.asarray(tangent_multipliers, dtype=float).reshape(-1)
        if (
            len(multipliers) == 0
            or np.any(~np.isfinite(multipliers))
            or np.any(multipliers <= 0.0)
        ):
            raise ValueError("tangent multipliers must be finite and positive")
        epistemic_payoff = epi + (
            mu - nominal.mean[None, :]
        ) ** 2
        base_epistemic_scale = np.sqrt(np.maximum(
            nominal.epistemic, self.variance_floor))
        base_aleatoric_scale = np.sqrt(np.maximum(
            nominal.aleatoric, self.variance_floor))
        best = np.full(mu.shape[1], np.inf, dtype=float)
        best_epistemic = base_epistemic_scale.copy()
        best_aleatoric = base_aleatoric_scale.copy()
        for epistemic_multiplier in multipliers:
            epistemic_scale = np.maximum(
                base_epistemic_scale * epistemic_multiplier,
                np.sqrt(self.variance_floor),
            )
            for aleatoric_multiplier in multipliers:
                aleatoric_scale = np.maximum(
                    base_aleatoric_scale * aleatoric_multiplier,
                    np.sqrt(self.variance_floor),
                )
                payoff = mu.copy()
                constant = np.zeros(mu.shape[1], dtype=float)
                if beta_radius > 0.0:
                    payoff = payoff + (
                        beta_radius
                        * epistemic_payoff
                        / (2.0 * epistemic_scale[None, :])
                    )
                    constant = constant + 0.5 * beta_radius * epistemic_scale
                if aleatoric_radius > 0.0:
                    payoff = payoff + (
                        aleatoric_radius
                        * alea
                        / (2.0 * aleatoric_scale[None, :])
                    )
                    constant = constant + 0.5 * aleatoric_radius * aleatoric_scale
                candidate = np.asarray(
                    self.kl_robust_expectation(
                        payoff, rho, weights=weights),
                    dtype=float,
                ) + constant - tau
                improved = candidate < best
                best = np.where(improved, candidate, best)
                best_epistemic = np.where(
                    improved, epistemic_scale, best_epistemic)
                best_aleatoric = np.where(
                    improved, aleatoric_scale, best_aleatoric)
        # Every tangent candidate and the legacy separable expression is a
        # valid upper bound. Numerical roundoff must never make the challenger
        # less than the nominal center posterior.
        best = np.maximum(best, nominal_margin)
        used_separable = separable_margin < best
        best = np.minimum(best, separable_margin)
        return RobustChanceMargin(
            upper=np.asarray(best, dtype=float),
            nominal=np.asarray(nominal_margin, dtype=float),
            separable_upper=np.asarray(separable_margin, dtype=float),
            tangent_epistemic_scale=np.asarray(best_epistemic, dtype=float),
            tangent_aleatoric_scale=np.asarray(best_aleatoric, dtype=float),
            used_separable_upper=np.asarray(used_separable, dtype=bool),
            radius=rho,
        )

    def kl_robust_expectation(self, values, radius, *, weights=None):
        """Return ``sup_q E_q[value]`` subject to ``KL(q || p) <= radius``."""
        matrix = self._expert_matrix(values, "robust expectation values")
        rho = max(float(radius), 0.0)
        center = (
            self.decision_weights()
            if weights is None
            else np.asarray(weights, dtype=float).reshape(-1)
        )
        if len(center) != len(self.expert_names):
            raise ValueError("robust center weights must match task experts")
        if np.any(center < 0.0) or not np.all(np.isfinite(center)):
            raise ValueError("robust center weights must be finite and nonnegative")
        center /= max(float(np.sum(center)), 1e-300)
        if rho <= 0.0:
            result = center @ matrix
        else:
            result = self._kl_robust_matrix(matrix, rho, weights=center)
        if np.asarray(values).ndim == 1:
            return float(result[0])
        return result

    def _kl_robust_matrix(self, values, radius, *, weights=None):
        """Batched finite-grid evaluation of the entropic KL dual.

        Every positive dual temperature supplies a valid upper bound. Taking
        the minimum over a deterministic finite grid therefore preserves the
        certificate while avoiding thousands of scalar optimizer calls inside
        exact KG.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise FloatingPointError("non-finite KL-robust payoff matrix")
        weights = (
            self.decision_weights()
            if weights is None
            else np.asarray(weights, dtype=float).reshape(-1)
        )
        if len(weights) != len(self.expert_names):
            raise ValueError("robust center weights must match task experts")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("robust center weights must be finite and nonnegative")
        weights /= max(float(np.sum(weights)), 1e-300)
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


class ReplicationVarianceTaskPosterior:
    """Finite posterior over variance structures using replication only.

    The posterior deliberately ignores response means and singleton residuals.
    Each distinct replicated policy contributes one Gaussian sample-variance
    likelihood with ``replicate_count - 1`` degrees of freedom. Replacing a
    policy record and refitting from the frozen prior avoids double counting
    nested sample variances when a policy receives a third or later replicate.
    """

    def __init__(
        self,
        expert_names,
        prior_weights=None,
        *,
        temperature=1.0,
        variance_floor=1e-10,
        minimum_weight=1e-12,
    ):
        names = tuple(str(name) for name in expert_names)
        if not names or len(set(names)) != len(names):
            raise ValueError(
                "variance experts must be a non-empty unique sequence")
        prior = (
            np.ones(len(names), dtype=float)
            if prior_weights is None
            else np.asarray(prior_weights, dtype=float).reshape(-1)
        )
        if (
            len(prior) != len(names)
            or np.any(prior < 0.0)
            or not np.all(np.isfinite(prior))
            or float(np.sum(prior)) <= 0.0
        ):
            raise ValueError(
                "variance prior weights must be finite, non-negative, and "
                "match experts")
        prior /= float(np.sum(prior))
        self.expert_names = names
        self.temperature = max(float(temperature), 0.0)
        self.variance_floor = max(float(variance_floor), 1e-15)
        self.minimum_weight = max(float(minimum_weight), 0.0)
        self._log_prior = np.log(np.maximum(
            prior, self.minimum_weight or 1e-300))
        self._log_prior -= logsumexp(self._log_prior)
        self._log_weights = self._log_prior.copy()
        self.records = {}
        self.n_updates = 0
        self.last_update = {"status": "initialized"}

    def clone(self):
        return copy.deepcopy(self)

    def prior_weights(self):
        return np.exp(self._log_prior).copy()

    def posterior_weights(self):
        return np.exp(self._log_weights).copy()

    @property
    def evidence_count(self):
        return int(len(self.records))

    @property
    def effective_dof(self):
        return int(sum(
            max(int(record["replicate_count"]) - 1, 0)
            for record in self.records.values()
        ))

    def kl_from_prior(self):
        weights = self.posterior_weights()
        prior = self.prior_weights()
        return float(np.sum(weights * (
            np.log(np.maximum(weights, 1e-300))
            - np.log(np.maximum(prior, 1e-300))
        )))

    def _refit(self):
        log_weights = self._log_prior.copy()
        cumulative_score = np.zeros(len(self.expert_names), dtype=float)
        for record in self.records.values():
            dof = max(int(record["replicate_count"]) - 1, 1)
            sample_variance = max(
                float(record["sample_variance"]), self.variance_floor)
            predicted = np.maximum(
                np.asarray(record["predicted_variances"], dtype=float),
                self.variance_floor,
            )
            # Exact Gaussian sample-variance log likelihood up to terms that
            # are common to every expert.
            cumulative_score += -0.5 * float(dof) * (
                np.log(predicted) + sample_variance / predicted)
        log_weights += self.temperature * cumulative_score
        log_weights -= logsumexp(log_weights)
        if self.minimum_weight > 0.0:
            weights = np.maximum(np.exp(log_weights), self.minimum_weight)
            weights /= float(np.sum(weights))
            log_weights = np.log(weights)
        self._log_weights = log_weights
        return cumulative_score

    def update_from_replication(
        self,
        key,
        sample_variance,
        predicted_variances,
        replicate_count,
    ):
        count = int(replicate_count)
        predicted = np.asarray(predicted_variances, dtype=float).reshape(-1)
        if count < 2:
            self.last_update = {
                "status": "ignored_singleton",
                "replicate_count": count,
                "target_mean_used": False,
            }
            return copy.deepcopy(self.last_update)
        if (
            len(predicted) != len(self.expert_names)
            or not np.all(np.isfinite(predicted))
            or np.any(predicted < 0.0)
        ):
            raise ValueError(
                "predicted replication variances must be finite, "
                "non-negative, and match experts")
        sample_variance = float(sample_variance)
        if not np.isfinite(sample_variance) or sample_variance < 0.0:
            raise ValueError(
                "replication sample variance must be finite and non-negative")
        record_key = tuple(key) if isinstance(key, (tuple, list)) else key
        before = self.posterior_weights()
        self.records[record_key] = {
            "sample_variance": sample_variance,
            "predicted_variances": np.maximum(
                predicted, self.variance_floor).tolist(),
            "replicate_count": count,
        }
        cumulative_score = self._refit()
        self.n_updates += 1
        self.last_update = {
            "status": "updated_from_replication",
            "record_key": list(record_key)
            if isinstance(record_key, tuple) else str(record_key),
            "replicate_count": count,
            "effective_dof": self.effective_dof,
            "weights_before": before.tolist(),
            "weights_after": self.posterior_weights().tolist(),
            "cumulative_log_score": cumulative_score.tolist(),
            "target_mean_used": False,
            "singleton_residual_used": False,
            "target_oracle_used": False,
        }
        return copy.deepcopy(self.last_update)

    def diagnostics(self):
        weights = self.posterior_weights()
        return {
            "status": "fit" if self.evidence_count else "frozen_source_prior",
            "mode": "replication_only",
            "expert_names": list(self.expert_names),
            "prior_weights": self.prior_weights().tolist(),
            "posterior_weights": weights.tolist(),
            "posterior_by_expert": {
                name: float(weight)
                for name, weight in zip(self.expert_names, weights)
            },
            "evidence_count": self.evidence_count,
            "effective_dof": self.effective_dof,
            "n_updates": int(self.n_updates),
            "kl_from_prior": self.kl_from_prior(),
            "target_mean_used": False,
            "singleton_residual_used": False,
            "target_oracle_used": False,
            "last_update": copy.deepcopy(self.last_update),
        }


class FiniteTaskSensitivityPosterior:
    """Posterior over task-level surrogate error sensitivity.

    A class changes the signed standardized bias and epistemic scale of the
    prequential predictive distribution and carries a corresponding false-
    feasibility decision penalty. Source tasks provide the prior; held-out
    observations update it through a proper predictive score. The class does
    not relax the theory certificate.
    """

    def __init__(
        self,
        class_names=("stable", "balanced", "sensitive"),
        scales=(0.5, 1.0, 2.0),
        biases=None,
        bias_coefficients=None,
        bias_feature_names=None,
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
        biases = np.zeros(len(names), dtype=float) if biases is None else np.asarray(
            biases, dtype=float).reshape(-1)
        coefficients = (
            None
            if bias_coefficients is None
            else np.asarray(bias_coefficients, dtype=float)
        )
        penalties = np.asarray(decision_penalties, dtype=float).reshape(-1)
        empirical_trust = np.asarray(empirical_trust, dtype=float).reshape(-1)
        if not names or len(set(names)) != len(names):
            raise ValueError("sensitivity classes must be non-empty and unique")
        if (
            len(scales) != len(names)
            or len(biases) != len(names)
            or len(penalties) != len(names)
            or len(empirical_trust) != len(names)
        ):
            raise ValueError("sensitivity class arrays must have equal length")
        if (
            np.any(scales <= 0.0)
            or np.any(penalties < 0.0)
            or not np.all(np.isfinite(scales))
            or not np.all(np.isfinite(biases))
            or not np.all(np.isfinite(penalties))
        ):
            raise ValueError("sensitivity scales and penalties must be finite")
        if (
            np.any(empirical_trust < 0.0)
            or np.any(empirical_trust > 1.0)
            or not np.all(np.isfinite(empirical_trust))
        ):
            raise ValueError("empirical trust must lie in [0, 1]")
        if coefficients is not None and (
            coefficients.ndim != 2
            or coefficients.shape[0] != len(names)
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError(
                "bias coefficient rows must match sensitivity classes")
        if bias_feature_names is not None and coefficients is not None and (
            len(bias_feature_names) != coefficients.shape[1]
        ):
            raise ValueError("bias feature names do not match coefficients")
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
        self.biases = biases
        self.bias_coefficients = coefficients
        self.bias_feature_names = (
            None if bias_feature_names is None
            else tuple(str(value) for value in bias_feature_names)
        )
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

    def bias_values(self):
        """Return signed class biases, upgrading pre-bias checkpoints."""
        if not hasattr(self, "biases"):
            self.biases = np.zeros(len(self.class_names), dtype=float)
        return np.asarray(self.biases, dtype=float)

    def bias_offsets(self, reference_sd, bias_features=None):
        dimensionless = self.bias_values().copy()
        coefficients = getattr(self, "bias_coefficients", None)
        if coefficients is None:
            return dimensionless * float(reference_sd)
        if bias_features is None:
            raise ValueError("functional bias classes require bias features")
        features = np.asarray(bias_features, dtype=float).reshape(-1)
        if len(features) != coefficients.shape[1]:
            raise ValueError("functional bias feature dimension mismatch")
        dimensionless += coefficients @ features
        return dimensionless * float(reference_sd)

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
        bias_features=None,
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
        reference_sd = np.sqrt(max(alea + epi, self.variance_floor))
        predictive_mean = mu + self.bias_offsets(
            reference_sd, bias_features=bias_features)
        scores = -0.5 * (
            np.log(2.0 * np.pi * predictive_var)
            + ((y - predictive_mean) ** 2) / predictive_var
        )
        boundary_scores = np.zeros(len(self.class_names), dtype=float)
        if tau is not None and self.boundary_score_weight > 0.0:
            probability = norm.cdf(
                (float(tau) - predictive_mean) / np.sqrt(predictive_var))
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
            "predictive_mean": predictive_mean.tolist(),
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
        bias_features=None,
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
        if getattr(self, "bias_coefficients", None) is None:
            class_offset = np.broadcast_to(
                sigma * self.bias_values()[None, :],
                (len(mean), len(self.class_names)),
            )
        else:
            features = np.asarray(bias_features, dtype=float)
            if features.ndim == 1:
                features = np.broadcast_to(
                    features[None, :], (len(mean), len(features)))
            if features.ndim != 2 or features.shape[0] != len(mean):
                raise ValueError("functional bias features must align with means")
            class_offset = sigma * (
                self.bias_values()[None, :]
                + features @ self.bias_coefficients.T
            )
        class_mean = mean[:, None] + class_offset
        violation_probability = norm.cdf(
            (class_mean - float(tau)) / class_std)
        weights = self.posterior_weights()
        expected_probability = violation_probability @ weights
        expected_decision_risk = violation_probability @ (
            weights * self.decision_penalties
        )
        return {
            "class_variance": class_variance,
            "class_mean": class_mean,
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
            "biases": self.bias_values().tolist(),
            "bias_coefficients": (
                None
                if getattr(self, "bias_coefficients", None) is None
                else self.bias_coefficients.tolist()
            ),
            "bias_feature_names": (
                None
                if getattr(self, "bias_feature_names", None) is None
                else list(self.bias_feature_names)
            ),
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
            "expected_bias": float(weights @ self.bias_values()),
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


class FiniteTaskLatentPosterior:
    """Joint posterior over structural experts and task sensitivity.

    A structural expert bundles an alignment/basis choice with its GPR and HVD
    state.  The sensitivity class controls the scale and loss attached to
    epistemic constraint error.  The source prior starts factorized, but a
    proper prequential score updates the Cartesian product jointly, allowing
    target evidence to learn which structural and sensitivity combinations are
    compatible.  This posterior is initially a shadow diagnostic: it never
    changes the theory certificate or the legacy V32 decision weights.
    """

    def __init__(
        self,
        structure_posterior,
        sensitivity_posterior=None,
        *,
        calibration_mode="source_profiles",
        adaptive_bias_prior=None,
    ):
        self.structure_names = tuple(structure_posterior.expert_names)
        self.structure_prior = np.asarray(
            structure_posterior.prior_weights(), dtype=float)
        if sensitivity_posterior is None:
            self.sensitivity_names = ("fixed",)
            self.sensitivity_scales = np.ones(1, dtype=float)
            self.sensitivity_biases = np.zeros(1, dtype=float)
            self.sensitivity_bias_coefficients = None
            self.sensitivity_bias_feature_names = None
            self.sensitivity_penalties = np.full(1, 5.0, dtype=float)
            self.sensitivity_trust = np.full(1, 0.25, dtype=float)
            self.sensitivity_prior = np.ones(1, dtype=float)
        else:
            self.sensitivity_names = tuple(sensitivity_posterior.class_names)
            self.sensitivity_scales = np.asarray(
                sensitivity_posterior.scales, dtype=float).copy()
            self.sensitivity_biases = np.asarray(
                getattr(
                    sensitivity_posterior,
                    "biases",
                    np.zeros(len(self.sensitivity_names), dtype=float),
                ),
                dtype=float,
            ).copy()
            coefficients = getattr(
                sensitivity_posterior, "bias_coefficients", None)
            self.sensitivity_bias_coefficients = (
                None
                if coefficients is None
                else np.asarray(coefficients, dtype=float).copy()
            )
            self.sensitivity_bias_feature_names = copy.deepcopy(getattr(
                sensitivity_posterior, "bias_feature_names", None))
            self.sensitivity_penalties = np.asarray(
                sensitivity_posterior.decision_penalties, dtype=float).copy()
            self.sensitivity_trust = np.asarray(
                sensitivity_posterior.empirical_trust, dtype=float).copy()
            self.sensitivity_prior = np.asarray(
                sensitivity_posterior.prior_weights(), dtype=float)

        calibration_mode = str(
            calibration_mode or "source_profiles").lower()
        if calibration_mode not in ("source_profiles", "expert_ridge"):
            raise ValueError(
                "task latent calibration mode must be source_profiles or "
                "expert_ridge")
        self.calibration_mode = calibration_mode
        self.adaptive_bias_enabled = calibration_mode == "expert_ridge"
        self.adaptive_bias_feature_names = None
        self.adaptive_bias_prior_mean = None
        self.adaptive_bias_prior_precision = None
        self.adaptive_bias_precision = None
        self.adaptive_bias_information = None
        self.adaptive_bias_mean = None
        self.adaptive_bias_n_updates = np.zeros(
            len(self.structure_names), dtype=int)
        if self.adaptive_bias_enabled:
            prior_spec = dict(adaptive_bias_prior or {})
            if prior_spec.get("status") not in (
                "fit_boundary_weighted_gaussian", "test_prior"
            ):
                raise ValueError(
                    "expert_ridge calibration requires a source-fitted "
                    "adaptive bias prior")
            prior_mean = np.asarray(
                prior_spec.get("mean", []), dtype=float).reshape(-1)
            prior_precision = np.asarray(
                prior_spec.get("precision", []), dtype=float)
            if (
                not len(prior_mean)
                or prior_precision.shape != (len(prior_mean), len(prior_mean))
                or not np.all(np.isfinite(prior_mean))
                or not np.all(np.isfinite(prior_precision))
            ):
                raise ValueError("invalid adaptive bias Gaussian prior")
            prior_precision = 0.5 * (
                prior_precision + prior_precision.T)
            eigenvalues, eigenvectors = np.linalg.eigh(prior_precision)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            prior_precision = (
                eigenvectors * eigenvalues) @ eigenvectors.T
            feature_names = prior_spec.get("feature_names")
            if feature_names is not None and len(feature_names) != len(
                prior_mean
            ):
                raise ValueError(
                    "adaptive bias feature names do not match prior")
            self.adaptive_bias_feature_names = (
                None
                if feature_names is None
                else tuple(str(value) for value in feature_names)
            )
            self.adaptive_bias_prior_mean = prior_mean.copy()
            self.adaptive_bias_prior_precision = prior_precision.copy()
            self.adaptive_bias_precision = np.repeat(
                prior_precision[None, :, :],
                len(self.structure_names),
                axis=0,
            )
            prior_information = prior_precision @ prior_mean
            self.adaptive_bias_information = np.repeat(
                prior_information[None, :],
                len(self.structure_names),
                axis=0,
            )
            self.adaptive_bias_mean = np.repeat(
                prior_mean[None, :],
                len(self.structure_names),
                axis=0,
            )

        prior = np.outer(self.structure_prior, self.sensitivity_prior)
        prior /= float(np.sum(prior))
        self.minimum_weight = max(
            float(structure_posterior.minimum_weight), 0.0)
        self.variance_floor = max(
            float(structure_posterior.variance_floor), 1e-15)
        self.temperature = max(float(structure_posterior.temperature), 0.0)
        self.temperature_decay = max(
            float(structure_posterior.temperature_decay), 0.0)
        self.boundary_score_weight = max(
            float(structure_posterior.boundary_score_weight), 0.0)
        self.safe_boundary_score_weight = max(
            float(structure_posterior.safe_boundary_score_weight), 0.0)
        self.safe_pairwise_score_weight = max(
            float(structure_posterior.safe_pairwise_score_weight), 0.0)
        self.safe_generalized = bool(structure_posterior.safe_generalized)
        self.output_score_weights = (
            None
            if structure_posterior.output_score_weights is None
            else np.asarray(
                structure_posterior.output_score_weights, dtype=float).copy()
        )
        self._log_prior = np.log(np.maximum(
            prior, self.minimum_weight or 1e-300))
        self._log_prior -= logsumexp(self._log_prior)
        self._log_weights = self._log_prior.copy()
        self._log_safe_weights = self._log_prior.copy()
        self.cumulative_log_score = np.zeros_like(prior)
        self.safe_cumulative_log_score = np.zeros_like(prior)
        self.n_updates = 0
        self.last_update = {"status": "initialized"}

    def clone(self):
        return copy.deepcopy(self)

    def prior_weights(self):
        return np.exp(self._log_prior).copy()

    def posterior_weights(self, *, safe=False):
        values = self._log_safe_weights if safe else self._log_weights
        return np.exp(values).copy()

    def structure_weights(self, *, safe=False):
        return np.sum(self.posterior_weights(safe=safe), axis=1)

    def sensitivity_weights(self, *, safe=False):
        return np.sum(self.posterior_weights(safe=safe), axis=0)

    def kl_from_prior(self, *, safe=False):
        weights = self.posterior_weights(safe=safe)
        prior = self.prior_weights()
        discrete = float(np.sum(weights * (
            np.log(np.maximum(weights, 1e-300))
            - np.log(np.maximum(prior, 1e-300))
        )))
        if not self.adaptive_bias_enabled:
            return discrete
        per_structure = self.adaptive_bias_kl_by_structure()
        structure = np.sum(weights, axis=1)
        return float(discrete + structure @ per_structure)

    def adaptive_bias_kl_by_structure(self):
        if not self.adaptive_bias_enabled:
            return np.zeros(len(self.structure_names), dtype=float)
        prior_precision = self.adaptive_bias_prior_precision
        prior_mean = self.adaptive_bias_prior_mean
        sign0, logdet0 = np.linalg.slogdet(prior_precision)
        if sign0 <= 0.0:
            raise FloatingPointError(
                "adaptive bias prior precision must be positive definite")
        dimension = len(prior_mean)
        values = []
        for precision, mean in zip(
            self.adaptive_bias_precision, self.adaptive_bias_mean
        ):
            covariance = np.linalg.inv(precision)
            sign, logdet = np.linalg.slogdet(precision)
            if sign <= 0.0:
                raise FloatingPointError(
                    "adaptive bias precision must be positive definite")
            difference = mean - prior_mean
            divergence = 0.5 * (
                float(np.trace(prior_precision @ covariance))
                + float(difference @ prior_precision @ difference)
                - dimension
                + float(logdet - logdet0)
            )
            values.append(max(float(divergence), 0.0))
        return np.asarray(values, dtype=float)

    def conditional_epistemic_scale_squared(self, *, safe=True):
        """Return conservative sensitivity scale conditional on structure.

        Learned stable classes may improve Bayes decisions, but they cannot
        shrink a theory certificate.  Hence every class scale is floored at
        one before it enters constraint epistemic variance.
        """
        joint = self.posterior_weights(safe=safe)
        structure = np.sum(joint, axis=1)
        scale_squared = np.maximum(self.sensitivity_scales, 1.0) ** 2
        numerator = joint @ scale_squared
        return np.divide(
            numerator,
            structure,
            out=np.ones_like(structure),
            where=structure > 1e-300,
        )

    def _bias_feature_tensor(
        self, bias_features, n_structure, n_points, feature_dim,
    ):
        if bias_features is None:
            raise ValueError("functional task bias requires risk features")
        features = np.asarray(bias_features, dtype=float)
        if features.ndim == 1:
            features = np.broadcast_to(
                features[None, None, :],
                (n_structure, n_points, len(features)),
            )
        elif features.ndim == 2:
            if features.shape[0] == n_structure and n_points == 1:
                features = features[:, None, :]
            elif features.shape[0] == n_points:
                features = np.broadcast_to(
                    features[None, :, :],
                    (n_structure, n_points, features.shape[1]),
                )
            else:
                raise ValueError("functional bias features have ambiguous axes")
        if (
            features.ndim != 3
            or features.shape[:2] != (n_structure, n_points)
            or features.shape[2] != feature_dim
            or not np.all(np.isfinite(features))
        ):
            raise ValueError("functional bias feature dimensions do not match")
        return features

    def _adaptive_bias_moments_many(self, reference_sd, bias_features):
        reference_sd = np.asarray(reference_sd, dtype=float)
        if reference_sd.ndim != 2:
            raise ValueError("reference SD must have structure and point axes")
        zeros = np.zeros_like(reference_sd)
        if not self.adaptive_bias_enabled:
            return zeros, zeros
        feature_dim = len(self.adaptive_bias_prior_mean)
        features = self._bias_feature_tensor(
            bias_features,
            reference_sd.shape[0],
            reference_sd.shape[1],
            feature_dim,
        )
        covariance = np.asarray([
            np.linalg.inv(precision)
            for precision in self.adaptive_bias_precision
        ], dtype=float)
        dimensionless_mean = np.einsum(
            "snp,sp->sn", features, self.adaptive_bias_mean)
        dimensionless_variance = np.maximum(np.einsum(
            "snp,spq,snq->sn", features, covariance, features
        ), 0.0)
        return (
            reference_sd * dimensionless_mean,
            reference_sd ** 2 * dimensionless_variance,
        )

    def adaptive_bias_variance_many(
        self, epistemic_vars, aleatoric_vars, bias_features,
    ):
        epi = np.maximum(np.asarray(epistemic_vars, dtype=float), 0.0)
        alea = np.maximum(np.asarray(aleatoric_vars, dtype=float), 0.0)
        if epi.shape != alea.shape:
            raise ValueError("adaptive bias variance moment arrays must match")
        reference_sd = np.sqrt(np.maximum(
            epi + alea, self.variance_floor))
        return self._adaptive_bias_moments_many(
            reference_sd, bias_features)[1]

    def adaptive_bias_moments_one(
        self, structure_index, reference_sd, bias_features,
    ):
        if not self.adaptive_bias_enabled:
            return 0.0, 0.0
        index = int(structure_index)
        features = np.asarray(bias_features, dtype=float).reshape(-1)
        if len(features) != len(self.adaptive_bias_prior_mean):
            raise ValueError("adaptive bias feature dimension mismatch")
        covariance = np.linalg.inv(self.adaptive_bias_precision[index])
        mean = float(reference_sd) * float(
            features @ self.adaptive_bias_mean[index])
        variance = float(reference_sd) ** 2 * max(float(
            features @ covariance @ features), 0.0)
        return mean, variance

    def _bias_offsets_many(self, reference_sd, bias_features=None):
        reference_sd = np.asarray(reference_sd, dtype=float)
        if reference_sd.ndim != 2:
            raise ValueError("reference SD must have structure and point axes")
        dimensionless = np.broadcast_to(
            self.sensitivity_biases[None, :, None],
            (
                reference_sd.shape[0],
                len(self.sensitivity_biases),
                reference_sd.shape[1],
            ),
        ).copy()
        coefficients = self.sensitivity_bias_coefficients
        n_structure, n_points = reference_sd.shape
        if coefficients is not None:
            features = self._bias_feature_tensor(
                bias_features,
                n_structure,
                n_points,
                coefficients.shape[1],
            )
            dimensionless += np.einsum(
                "snp,cp->scn", features, coefficients)
        offsets = reference_sd[:, None, :] * dimensionless
        if self.adaptive_bias_enabled:
            adaptive_mean, _adaptive_variance = (
                self._adaptive_bias_moments_many(
                    reference_sd, bias_features))
            offsets += adaptive_mean[:, None, :]
        return offsets

    @staticmethod
    def _normal_positive_part(mean, variance):
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

    def positive_margin_decision_risk_many(
        self,
        means,
        epistemic_vars,
        aleatoric_vars,
        *,
        tau,
        z_alpha,
        bias_features=None,
    ):
        """Integrate chance-margin loss under the joint task posterior."""
        mu = np.asarray(means, dtype=float)
        epi = np.maximum(np.asarray(epistemic_vars, dtype=float), 0.0)
        alea = np.maximum(np.asarray(aleatoric_vars, dtype=float), 0.0)
        if mu.ndim != 2 or mu.shape[0] != len(self.structure_names):
            raise ValueError("constraint means must have structures first")
        if epi.shape != mu.shape or alea.shape != mu.shape:
            raise ValueError("joint task-latent moment arrays must match")
        reference_sd = np.sqrt(np.maximum(
            epi + alea, self.variance_floor))
        bias_offset = self._bias_offsets_many(
            reference_sd, bias_features=bias_features)
        _adaptive_mean, adaptive_variance = (
            self._adaptive_bias_moments_many(
                reference_sd, bias_features))
        margin_mean = (
            mu[:, None, :]
            + bias_offset
            + float(z_alpha) * np.sqrt(alea[:, None, :])
            - float(tau)
        )
        margin_variance = (
            epi[:, None, :]
            * self.sensitivity_scales[None, :, None] ** 2
            + adaptive_variance[:, None, :]
        )
        positive = self._normal_positive_part(
            margin_mean, margin_variance)
        probability = norm.cdf(
            margin_mean
            / np.sqrt(np.maximum(margin_variance, self.variance_floor))
        )
        weights = self.posterior_weights(safe=True)
        expected_positive = np.sum(
            weights[:, :, None] * positive, axis=(0, 1))
        expected_loss = np.sum(
            weights[:, :, None]
            * self.sensitivity_penalties[None, :, None]
            * positive,
            axis=(0, 1),
        )
        expected_probability = np.sum(
            weights[:, :, None] * probability, axis=(0, 1))
        centered = positive - expected_positive[None, None, :]
        disagreement = np.sqrt(np.maximum(np.sum(
            weights[:, :, None] * centered ** 2,
            axis=(0, 1),
        ), 0.0))
        return {
            "posterior_expected_positive_margin": expected_positive,
            "posterior_expected_decision_loss": expected_loss,
            "posterior_violation_probability": expected_probability,
            "model_disagreement": disagreement,
            "affects_theory_certificate": False,
        }

    def _effective_temperature(self):
        if self.temperature_decay <= 0.0:
            return self.temperature
        return self.temperature / (
            max(1, self.n_updates + 1) ** self.temperature_decay)

    def _update_adaptive_bias_posterior(
        self,
        observation,
        means,
        epistemic_vars,
        aleatoric_vars,
        *,
        tau,
        chance_z,
        bias_features,
    ):
        if not self.adaptive_bias_enabled:
            return {"status": "disabled"}
        mu = np.asarray(means, dtype=float).reshape(-1)
        epi = np.maximum(
            np.asarray(epistemic_vars, dtype=float).reshape(-1), 0.0)
        alea = np.maximum(
            np.asarray(aleatoric_vars, dtype=float).reshape(-1), 0.0)
        reference_sd = np.sqrt(np.maximum(
            epi + alea, self.variance_floor))
        feature_dim = len(self.adaptive_bias_prior_mean)
        features = self._bias_feature_tensor(
            bias_features,
            len(self.structure_names),
            1,
            feature_dim,
        )[:, 0, :]
        target = (float(observation) - mu) / reference_sd
        if tau is None or self.boundary_score_weight <= 0.0:
            boundary_relevance = np.zeros(len(mu), dtype=float)
        else:
            standardized_boundary_distance = (
                (
                    float(observation)
                    + float(chance_z) * np.sqrt(alea)
                    - float(tau)
                ) / reference_sd)
            boundary_relevance = np.exp(
                -0.5 * standardized_boundary_distance ** 2)
        update_weight = 1.0 + (
            self.boundary_score_weight * boundary_relevance)
        means_before = self.adaptive_bias_mean.copy()
        variance_before = []
        variance_after = []
        for index, (phi, value, weight) in enumerate(zip(
            features, target, update_weight
        )):
            covariance_before = np.linalg.inv(
                self.adaptive_bias_precision[index])
            variance_before.append(float(phi @ covariance_before @ phi))
            self.adaptive_bias_precision[index] += (
                float(weight) * np.outer(phi, phi))
            self.adaptive_bias_information[index] += (
                float(weight) * phi * float(value))
            try:
                self.adaptive_bias_mean[index] = np.linalg.solve(
                    self.adaptive_bias_precision[index],
                    self.adaptive_bias_information[index],
                )
            except np.linalg.LinAlgError:
                self.adaptive_bias_mean[index] = np.linalg.lstsq(
                    self.adaptive_bias_precision[index],
                    self.adaptive_bias_information[index],
                    rcond=None,
                )[0]
            covariance_after = np.linalg.inv(
                self.adaptive_bias_precision[index])
            variance_after.append(float(phi @ covariance_after @ phi))
            self.adaptive_bias_n_updates[index] += 1
        return {
            "status": "updated",
            "target_standardized_residual": target.tolist(),
            "boundary_relevance": boundary_relevance.tolist(),
            "update_weight": update_weight.tolist(),
            "coefficient_mean_before": means_before.tolist(),
            "coefficient_mean_after": self.adaptive_bias_mean.tolist(),
            "feature_variance_before": variance_before,
            "feature_variance_after": variance_after,
            "observation_source": "budgeted_target_evaluation",
            "target_oracle_used": False,
        }

    def _normalize(self, values):
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("non-finite joint task-latent weights")
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
        bias_features=None,
        chance_z=0.0,
    ):
        """Jointly score every structure/sensitivity hypothesis."""
        y = np.asarray(observation, dtype=float).reshape(-1)
        mu = np.asarray(means, dtype=float)
        epi = np.maximum(np.asarray(epistemic_vars, dtype=float), 0.0)
        alea = np.maximum(np.asarray(aleatoric_vars, dtype=float), 0.0)
        expected = (len(self.structure_names), len(y))
        if mu.shape != expected or epi.shape != expected or alea.shape != expected:
            raise ValueError("joint task-latent predictive arrays have wrong shape")
        if not all(np.all(np.isfinite(value)) for value in (y, mu, epi, alea)):
            raise FloatingPointError(
                "joint task-latent update received non-finite moments")
        index = int(constraint_index)
        if index < 0 or index >= len(y):
            raise IndexError("constraint index is outside observation")

        predictive_var = np.repeat(
            (epi + alea)[:, None, :],
            len(self.sensitivity_names),
            axis=1,
        )
        predictive_var[:, :, index] = (
            alea[:, None, index]
            + epi[:, None, index]
            * self.sensitivity_scales[None, :] ** 2
        )
        predictive_var = np.maximum(predictive_var, self.variance_floor)
        predictive_mean = np.repeat(
            mu[:, None, :], len(self.sensitivity_names), axis=1)
        reference_sd = np.sqrt(np.maximum(
            alea[:, index] + epi[:, index], self.variance_floor))
        _adaptive_mean, adaptive_variance = (
            self._adaptive_bias_moments_many(
                reference_sd[:, None], bias_features))
        predictive_var[:, :, index] += adaptive_variance[:, None, 0]
        predictive_mean[:, :, index] += (
            self._bias_offsets_many(
                reference_sd[:, None], bias_features=bias_features)[:, :, 0]
        )
        gaussian_scores = -0.5 * (
            np.log(2.0 * np.pi * predictive_var)
            + (y[None, None, :] - predictive_mean) ** 2 / predictive_var
        )
        if self.output_score_weights is None:
            output_weights = np.ones(len(y), dtype=float)
        else:
            if len(self.output_score_weights) != len(y):
                raise ValueError("output score weights do not match observation")
            output_weights = self.output_score_weights
        gaussian_total = np.sum(
            gaussian_scores * output_weights[None, None, :], axis=2)

        boundary_scores = np.zeros_like(gaussian_total)
        boundary_requested = bool(
            tau is not None and (
                self.boundary_score_weight > 0.0
                or (
                    self.safe_generalized
                    and self.safe_boundary_score_weight > 0.0
                )
            )
        )
        if boundary_requested:
            probability = norm.cdf(
                (float(tau) - predictive_mean[:, :, index])
                / np.sqrt(predictive_var[:, :, index])
            )
            probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
            event = float(y[index] <= float(tau))
            boundary_scores = (
                event * np.log(probability)
                + (1.0 - event) * np.log1p(-probability)
            )

        pairwise = (
            np.zeros(len(self.structure_names), dtype=float)
            if safe_pairwise_log_score is None
            else np.asarray(safe_pairwise_log_score, dtype=float).reshape(-1)
        )
        if len(pairwise) != len(self.structure_names):
            raise ValueError("safe pairwise scores must match structures")
        if not np.all(np.isfinite(pairwise)):
            raise FloatingPointError("safe pairwise scores must be finite")

        total_score = (
            gaussian_total
            + self.boundary_score_weight * boundary_scores
        )
        safe_total_score = (
            gaussian_scores[:, :, index]
            + self.safe_boundary_score_weight * boundary_scores
            + self.safe_pairwise_score_weight * pairwise[:, None]
        )
        eta = self._effective_temperature()
        before = self.posterior_weights()
        safe_before = self.posterior_weights(safe=True)
        self._log_weights = self._normalize(
            self._log_weights + eta * total_score)
        self.cumulative_log_score += total_score
        if self.safe_generalized:
            self._log_safe_weights = self._normalize(
                self._log_safe_weights + eta * safe_total_score)
            self.safe_cumulative_log_score += safe_total_score
        else:
            self._log_safe_weights = self._log_weights.copy()
            self.safe_cumulative_log_score = self.cumulative_log_score.copy()
        adaptive_update = self._update_adaptive_bias_posterior(
            y[index],
            mu[:, index],
            epi[:, index],
            alea[:, index],
            tau=tau,
            chance_z=chance_z,
            bias_features=bias_features,
        )
        self.n_updates += 1
        after = self.posterior_weights()
        safe_after = self.posterior_weights(safe=True)
        self.last_update = {
            "status": "updated",
            "temperature": float(eta),
            "weights_before": before.tolist(),
            "weights_after": after.tolist(),
            "safe_weights_before": safe_before.tolist(),
            "safe_weights_after": safe_after.tolist(),
            "safe_pairwise_pairs": int(safe_pairwise_pairs),
            "safe_pairwise_effective_weight": float(
                safe_pairwise_effective_weight),
            "adaptive_bias_update": adaptive_update,
            "observation_source": "budgeted_target_evaluation",
            "target_oracle_used": False,
        }
        return copy.deepcopy(self.last_update)

    def posterior_violation_decision_risk_many(
        self,
        means,
        epistemic_vars,
        aleatoric_vars,
        *,
        tau,
        bias_features=None,
    ):
        """Return posterior Bayes risk without changing certification."""
        mu = np.asarray(means, dtype=float)
        epi = np.maximum(np.asarray(epistemic_vars, dtype=float), 0.0)
        alea = np.maximum(np.asarray(aleatoric_vars, dtype=float), 0.0)
        if mu.ndim != 2 or mu.shape[0] != len(self.structure_names):
            raise ValueError("constraint means must have structures first")
        if epi.shape != mu.shape or alea.shape != mu.shape:
            raise ValueError("joint task-latent moment arrays must match")
        variance = (
            alea[:, None, :]
            + epi[:, None, :] * self.sensitivity_scales[None, :, None] ** 2
        )
        reference_sd = np.sqrt(np.maximum(
            alea + epi, self.variance_floor))
        _adaptive_mean, adaptive_variance = (
            self._adaptive_bias_moments_many(
                reference_sd, bias_features))
        variance += adaptive_variance[:, None, :]
        biased_mean = (
            mu[:, None, :]
            + self._bias_offsets_many(
                reference_sd, bias_features=bias_features)
        )
        probability = norm.cdf(
            (biased_mean - float(tau))
            / np.sqrt(np.maximum(variance, self.variance_floor))
        )
        weights = self.posterior_weights(safe=True)
        expected_probability = np.sum(
            weights[:, :, None] * probability, axis=(0, 1))
        expected_loss = np.sum(
            weights[:, :, None]
            * self.sensitivity_penalties[None, :, None]
            * probability,
            axis=(0, 1),
        )
        return {
            "posterior_violation_probability": expected_probability,
            "posterior_expected_decision_risk": expected_loss,
            "affects_theory_certificate": False,
        }

    @staticmethod
    def _entropy(weights):
        values = np.asarray(weights, dtype=float)
        return float(-np.sum(values * np.log(np.maximum(values, 1e-300))))

    def mutual_information(self, *, safe=False):
        joint = self.posterior_weights(safe=safe)
        structure = np.sum(joint, axis=1)
        sensitivity = np.sum(joint, axis=0)
        independent = structure[:, None] * sensitivity[None, :]
        return float(np.sum(joint * (
            np.log(np.maximum(joint, 1e-300))
            - np.log(np.maximum(independent, 1e-300))
        )))

    def diagnostics(
        self,
        *,
        legacy_structure_weights=None,
        legacy_sensitivity_weights=None,
    ):
        joint = self.posterior_weights()
        safe_joint = self.posterior_weights(safe=True)
        structure = np.sum(joint, axis=1)
        sensitivity = np.sum(joint, axis=0)
        safe_structure = np.sum(safe_joint, axis=1)
        safe_sensitivity = np.sum(safe_joint, axis=0)
        structure_tv = None
        sensitivity_tv = None
        legacy_sensitivity_compatible = None
        if legacy_structure_weights is not None:
            reference = np.asarray(
                legacy_structure_weights, dtype=float).reshape(-1)
            comparison = safe_structure if self.safe_generalized else structure
            structure_tv = float(0.5 * np.sum(np.abs(comparison - reference)))
        if legacy_sensitivity_weights is not None:
            reference = np.asarray(
                legacy_sensitivity_weights, dtype=float).reshape(-1)
            legacy_sensitivity_compatible = bool(
                len(reference) == len(sensitivity))
            if legacy_sensitivity_compatible:
                sensitivity_tv = float(
                    0.5 * np.sum(np.abs(sensitivity - reference)))
        expected_bias_coefficients = None
        safe_expected_bias_coefficients = None
        if self.sensitivity_bias_coefficients is not None:
            expected_bias_coefficients = (
                sensitivity @ self.sensitivity_bias_coefficients).tolist()
            safe_expected_bias_coefficients = (
                safe_sensitivity @ self.sensitivity_bias_coefficients).tolist()
        adaptive_covariance_diagonal = None
        adaptive_kl = None
        if self.adaptive_bias_enabled:
            adaptive_covariance_diagonal = [
                np.diag(np.linalg.inv(precision)).tolist()
                for precision in self.adaptive_bias_precision
            ]
            adaptive_kl = self.adaptive_bias_kl_by_structure().tolist()
        return {
            "status": "fit" if self.n_updates else "initialized",
            "inference_mode": "shadow_joint_generalized_bayes",
            "latent_state": (
                "z=(structure,theta_structure,sensitivity)"
                if self.adaptive_bias_enabled
                else "z=(structure,sensitivity)"
            ),
            "structure_bundles": "alignment+basis+GPR+HVD",
            "calibration_mode": self.calibration_mode,
            "structure_names": list(self.structure_names),
            "sensitivity_names": list(self.sensitivity_names),
            "sensitivity_biases": self.sensitivity_biases.tolist(),
            "sensitivity_bias_coefficients": (
                None
                if self.sensitivity_bias_coefficients is None
                else self.sensitivity_bias_coefficients.tolist()
            ),
            "sensitivity_bias_feature_names": (
                None
                if self.sensitivity_bias_feature_names is None
                else list(self.sensitivity_bias_feature_names)
            ),
            "joint_prior_weights": self.prior_weights().tolist(),
            "joint_posterior_weights": joint.tolist(),
            "joint_safe_weights": safe_joint.tolist(),
            "structure_marginal": structure.tolist(),
            "safe_structure_marginal": safe_structure.tolist(),
            "sensitivity_marginal": sensitivity.tolist(),
            "safe_sensitivity_marginal": safe_sensitivity.tolist(),
            "expected_signed_bias": float(
                sensitivity @ self.sensitivity_biases),
            "safe_expected_signed_bias": float(
                safe_sensitivity @ self.sensitivity_biases),
            "expected_bias_coefficients": expected_bias_coefficients,
            "safe_expected_bias_coefficients": (
                safe_expected_bias_coefficients),
            "adaptive_bias_enabled": bool(self.adaptive_bias_enabled),
            "adaptive_bias_feature_names": (
                None
                if self.adaptive_bias_feature_names is None
                else list(self.adaptive_bias_feature_names)
            ),
            "adaptive_bias_prior_mean": (
                None
                if self.adaptive_bias_prior_mean is None
                else self.adaptive_bias_prior_mean.tolist()
            ),
            "adaptive_bias_prior_precision": (
                None
                if self.adaptive_bias_prior_precision is None
                else self.adaptive_bias_prior_precision.tolist()
            ),
            "adaptive_bias_mean_by_structure": (
                None
                if self.adaptive_bias_mean is None
                else self.adaptive_bias_mean.tolist()
            ),
            "adaptive_bias_covariance_diagonal_by_structure": (
                adaptive_covariance_diagonal),
            "adaptive_bias_n_updates": self.adaptive_bias_n_updates.tolist(),
            "adaptive_bias_kl_by_structure": adaptive_kl,
            "joint_entropy": self._entropy(joint),
            "joint_safe_entropy": self._entropy(safe_joint),
            "structure_sensitivity_mutual_information": (
                self.mutual_information()),
            "safe_structure_sensitivity_mutual_information": (
                self.mutual_information(safe=True)),
            "legacy_structure_total_variation": structure_tv,
            "legacy_sensitivity_total_variation": sensitivity_tv,
            "legacy_sensitivity_state_space_compatible": (
                legacy_sensitivity_compatible),
            "n_updates": int(self.n_updates),
            "last_update": copy.deepcopy(self.last_update),
            "used_for_decision": False,
            "bias_mean_affects_theory_certificate": False,
            "bias_covariance_affects_theory_certificate": bool(
                self.adaptive_bias_enabled),
            "affects_theory_certificate": bool(
                self.adaptive_bias_enabled),
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

    supports_precomputed_expert_moments = True

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
        task_latent_posterior=None,
        task_latent_inference_mode="shadow",
        task_latent_calibration_mode="source_profiles",
        source_discrepancy_update=True,
        variance_structure_posterior_mode="shared",
        variance_structure_posterior=None,
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
        mode = str(task_latent_inference_mode or "shadow").lower()
        if mode not in ("shadow", "authoritative"):
            raise ValueError(
                "task latent inference mode must be shadow or authoritative")
        self.task_latent_inference_mode = mode
        calibration_mode = str(
            task_latent_calibration_mode or "source_profiles").lower()
        if calibration_mode not in ("source_profiles", "expert_ridge"):
            raise ValueError(
                "task latent calibration mode must be source_profiles or "
                "expert_ridge")
        self.task_latent_calibration_mode = calibration_mode
        self.source_discrepancy_update = bool(source_discrepancy_update)
        variance_mode = str(
            variance_structure_posterior_mode or "shared"
        ).strip().lower().replace("-", "_")
        if variance_mode not in {"shared", "replication_only"}:
            raise ValueError(
                "variance structure posterior mode must be shared or "
                "replication_only")
        self.variance_structure_posterior_mode = variance_mode
        self.variance_structure_posterior = (
            None
            if variance_mode == "shared"
            else (
                ReplicationVarianceTaskPosterior(
                    names,
                    posterior.prior_weights(),
                    temperature=max(float(posterior.temperature), 1e-12),
                    variance_floor=posterior.variance_floor,
                    minimum_weight=posterior.minimum_weight,
                )
                if variance_structure_posterior is None
                else variance_structure_posterior
            )
        )
        if (
            self.variance_structure_posterior is not None
            and tuple(self.variance_structure_posterior.expert_names) != names
        ):
            raise ValueError(
                "variance structure posterior and expert states disagree")
        self.task_latent_posterior = (
            FiniteTaskLatentPosterior(
                posterior,
                self._source_shadow_sensitivity(),
                calibration_mode=self.task_latent_calibration_mode,
                adaptive_bias_prior=self._source_adaptive_bias_prior(),
            )
            if task_latent_posterior is None
            else task_latent_posterior
        )
        self.last_update = {"status": "uninitialized"}

    def _source_shadow_sensitivity(self):
        """Build the source-trained sensitivity factor without decision use."""
        provider = self.states[0].problem if self.states else None
        if provider is None or not hasattr(provider, "task_sensitivity_prior"):
            return self.sensitivity_posterior
        prior = provider.task_sensitivity_prior() or {}
        if self.task_latent_calibration_mode == "expert_ridge":
            return FiniteTaskSensitivityPosterior(
                class_names=prior.get(
                    "adaptive_scale_class_names",
                    ("stable", "balanced", "sensitive"),
                ),
                scales=prior.get(
                    "adaptive_scale_scales", (0.5, 1.0, 2.0)),
                biases=None,
                bias_coefficients=None,
                bias_feature_names=None,
                decision_penalties=prior.get(
                    "adaptive_scale_decision_penalties",
                    (2.0, 5.0, 20.0),
                ),
                empirical_trust=prior.get(
                    "adaptive_scale_empirical_trust",
                    (1.0, 0.25, 0.0),
                ),
                prior_weights=prior.get(
                    "adaptive_scale_prior_weights"),
                temperature=self.posterior.temperature,
                temperature_decay=self.posterior.temperature_decay,
                boundary_score_weight=self.posterior.boundary_score_weight,
            )
        return FiniteTaskSensitivityPosterior(
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
            temperature=self.posterior.temperature,
            temperature_decay=self.posterior.temperature_decay,
            boundary_score_weight=self.posterior.boundary_score_weight,
        )

    def _source_adaptive_bias_prior(self):
        if self.task_latent_calibration_mode != "expert_ridge":
            return None
        provider = self.states[0].problem if self.states else None
        if provider is None or not hasattr(provider, "task_sensitivity_prior"):
            return None
        return copy.deepcopy(
            (provider.task_sensitivity_prior() or {}).get(
                "adaptive_bias_prior"))

    def _task_latent(self):
        """Lazily upgrade checkpoints written before the joint shadow existed."""
        if not hasattr(self, "task_latent_posterior"):
            self.task_latent_posterior = FiniteTaskLatentPosterior(
                self.posterior,
                self._source_shadow_sensitivity(),
                calibration_mode=self.task_latent_calibration_mode,
                adaptive_bias_prior=self._source_adaptive_bias_prior(),
            )
        return self.task_latent_posterior

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
            task_latent_posterior=self._task_latent().clone(),
            task_latent_inference_mode=self.task_latent_inference_mode,
            task_latent_calibration_mode=self.task_latent_calibration_mode,
            source_discrepancy_update=self.source_discrepancy_update,
            variance_structure_posterior_mode=(
                self.variance_structure_posterior_mode),
            variance_structure_posterior=(
                None
                if self.variance_structure_posterior is None
                else self.variance_structure_posterior.clone()
            ),
        )
        clone.last_update = copy.deepcopy(self.last_update)
        return clone

    def effective_kl_radius(self):
        """Finite PAC-Bayes complexity radius for target-task uncertainty."""
        n = max(int(self.posterior.n_updates), 1)
        posterior_kl = (
            self._task_latent().kl_from_prior(safe=True)
            if self.task_latent_authoritative
            else self.posterior.kl_from_prior()
        )
        complexity = (
            self.kl_radius_numerator
            + posterior_kl
            + np.log(1.0 / self.confidence_delta)
        )
        return float(min(
            self.maximum_kl_radius,
            max(complexity / float(n), 0.0),
        ))

    @property
    def task_latent_authoritative(self):
        return str(getattr(
            self, "task_latent_inference_mode", "shadow"
        )).lower() == "authoritative"

    def structure_weights(self, *, objective=False):
        if self.task_latent_authoritative:
            return self._task_latent().structure_weights(
                safe=not bool(objective))
        if objective and self.posterior.safe_generalized:
            return self.posterior.posterior_weights()
        return self.posterior.decision_weights()

    @property
    def variance_structure_isolated(self):
        return str(getattr(
            self, "variance_structure_posterior_mode", "shared"
        )).strip().lower().replace("-", "_") == "replication_only"

    def _replication_variance_posterior(self):
        """Lazily upgrade checkpoints written before the V15 head split."""
        if not self.variance_structure_isolated:
            return None
        posterior = getattr(self, "variance_structure_posterior", None)
        if posterior is None:
            posterior = ReplicationVarianceTaskPosterior(
                [state.name for state in self.states],
                self.posterior.prior_weights(),
                temperature=max(float(self.posterior.temperature), 1e-12),
                variance_floor=self.posterior.variance_floor,
                minimum_weight=self.posterior.minimum_weight,
            )
            self.variance_structure_posterior = posterior
        return posterior

    def variance_structure_weights(self):
        """Weights for HVD shapes, independent of response means in V15."""
        posterior = self._replication_variance_posterior()
        if posterior is None:
            return self.structure_weights(objective=False)
        return posterior.posterior_weights()

    def variance_effective_kl_radius(self):
        posterior = self._replication_variance_posterior()
        if posterior is None:
            return self.effective_kl_radius()
        evidence = max(int(posterior.effective_dof), 1)
        complexity = (
            self.kl_radius_numerator
            + posterior.kl_from_prior()
            + np.log(1.0 / self.confidence_delta)
        )
        return float(min(
            self.maximum_kl_radius,
            max(complexity / float(evidence), 0.0),
        ))

    def predictive_selector_weights(self):
        """Joint selector law for mean and variance structures.

        Under the isolated model a single uniform draw indexes the Cartesian
        product of the mean-task and HVD-task posteriors. This keeps exact-KG
        sampling closed under the same product posterior without requiring a
        second random-number API.
        """
        if self.task_latent_authoritative:
            mean_weights = self._task_latent().posterior_weights(
                safe=True).reshape(-1)
        else:
            mean_weights = np.asarray(
                self.posterior.decision_weights(), dtype=float).reshape(-1)
        mean_weights = np.maximum(mean_weights, 0.0)
        mean_weights /= max(float(np.sum(mean_weights)), 1e-300)
        if not self.variance_structure_isolated:
            return mean_weights
        variance_weights = np.maximum(np.asarray(
            self.variance_structure_weights(), dtype=float).reshape(-1), 0.0)
        variance_weights /= max(float(np.sum(variance_weights)), 1e-300)
        product = np.outer(mean_weights, variance_weights).reshape(-1)
        return product / max(float(np.sum(product)), 1e-300)

    def _predictive_selector_indices(self, expert_uniform):
        weights = self.predictive_selector_weights()
        flat_index = int(np.searchsorted(
            np.cumsum(weights),
            np.clip(float(expert_uniform), 0.0, 1.0 - 1e-15),
            side="right",
        ))
        flat_index = min(flat_index, len(weights) - 1)
        if self.task_latent_authoritative:
            latent_shape = self._task_latent().posterior_weights(
                safe=True).shape
            mean_state_count = int(np.prod(latent_shape))
        else:
            latent_shape = None
            mean_state_count = len(self.states)
        if self.variance_structure_isolated:
            mean_index, variance_index = np.unravel_index(
                flat_index, (mean_state_count, len(self.states)))
        else:
            mean_index = flat_index
            variance_index = None
        if latent_shape is not None:
            expert_index, sensitivity_index = np.unravel_index(
                mean_index, latent_shape)
        else:
            expert_index = int(mean_index)
            sensitivity_index = None
        if variance_index is None:
            variance_index = expert_index
        return int(expert_index), int(variance_index), sensitivity_index

    def inference_weights(self):
        if self.task_latent_authoritative:
            return self._task_latent().posterior_weights(
                safe=True).reshape(-1)
        return self.posterior.decision_posterior_weights()

    def inference_entropy(self):
        weights = self.inference_weights()
        return float(-np.sum(
            weights * np.log(np.maximum(weights, 1e-300))))

    def structure_proposal_weights(self, exploration=0.10):
        epsilon = float(np.clip(exploration, 0.0, 1.0))
        if not self.task_latent_authoritative:
            return self.posterior.proposal_weights(exploration=epsilon)
        posterior = self.structure_weights(objective=False)
        prior = self._task_latent().structure_prior
        weights = (1.0 - epsilon) * posterior + epsilon * prior
        return weights / max(float(np.sum(weights)), 1e-300)

    def structure_proposal_allocation(
        self, total, *, exploration=0.10, minimum_per_expert=0,
    ):
        total = max(0, int(total))
        counts = np.zeros(len(self.states), dtype=int)
        if total == 0:
            return {state.name: 0 for state in self.states}
        minimum = max(0, int(minimum_per_expert))
        if minimum and total >= minimum * len(self.states):
            counts[:] = minimum
        remaining = total - int(np.sum(counts))
        weights = self.structure_proposal_weights(exploration=exploration)
        fractional = remaining * weights
        floor = np.floor(fractional).astype(int)
        counts += floor
        leftover = total - int(np.sum(counts))
        if leftover:
            order = np.argsort(-(fractional - floor), kind="stable")
            counts[order[:leftover]] += 1
        return {
            state.name: int(count)
            for state, count in zip(self.states, counts)
        }

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

    def task_bias_features_many(self, X):
        latent = self._task_latent()
        if (
            latent.sensitivity_bias_coefficients is None
            and not latent.adaptive_bias_enabled
        ):
            return None
        rows = []
        for state in self.states:
            if not hasattr(state.problem, "task_bias_features"):
                raise ValueError(
                    "functional task bias requires provider risk features")
            rows.append(np.vstack([
                np.asarray(
                    state.problem.task_bias_features(x), dtype=float)
                for x in X
            ]))
        return np.asarray(rows, dtype=float)

    def expert_calibrated_constraint_moments_many(
        self, X, *, certification=True,
    ):
        """Return per-expert V4 decision moments on one shared history."""
        moments = self.expert_moments_many(
            1, X, certification=certification)
        latent = self._task_latent()
        if not latent.adaptive_bias_enabled:
            return moments
        mean, epistemic, aleatoric = moments
        reference_sd = np.sqrt(np.maximum(
            epistemic + aleatoric, latent.variance_floor))
        adaptive_mean, adaptive_variance = (
            latent._adaptive_bias_moments_many(
                reference_sd, self.task_bias_features_many(X)))
        return (
            mean + adaptive_mean,
            epistemic
            * latent.conditional_epistemic_scale_squared(
                safe=True)[:, None]
            + adaptive_variance,
            aleatoric,
        )

    def _prepare_expert_moments_many(
        self,
        output_index,
        X,
        *,
        certification=True,
        expert_moments=None,
    ):
        moments = (
            self.expert_moments_many(
                output_index, X, certification=certification)
            if expert_moments is None
            else tuple(
                np.asarray(values, dtype=float)
                for values in expert_moments
            )
        )
        if len(moments) != 3:
            raise ValueError(
                "expert moments must contain mean, epistemic, and aleatoric"
            )
        weights = None
        if self.task_latent_authoritative:
            weights = self.structure_weights(
                objective=int(output_index) == 0)
            if int(output_index) == 1:
                calibration_variance = (
                    self._task_latent().adaptive_bias_variance_many(
                        moments[1],
                        moments[2],
                        self.task_bias_features_many(X),
                    )
                )
                moments = (
                    moments[0],
                    moments[1]
                    * self._task_latent()
                    .conditional_epistemic_scale_squared(safe=True)[:, None]
                    + calibration_variance,
                    moments[2],
                )
        elif int(output_index) == 0 and self.posterior.safe_generalized:
            weights = self.posterior.posterior_weights()
        return moments, weights

    def mixture_moments_many(
        self,
        output_index,
        X,
        *,
        certification=True,
        expert_moments=None,
    ):
        moments, weights = self._prepare_expert_moments_many(
            output_index,
            X,
            certification=certification,
            expert_moments=expert_moments,
        )
        nominal = self.posterior.mixture_moments(
            *moments, weights=weights)
        if not self.variance_structure_isolated:
            return nominal
        variance_weights = np.asarray(
            self.variance_structure_weights(), dtype=float).reshape(-1)
        variance_weights /= max(float(np.sum(variance_weights)), 1e-300)
        aleatoric = variance_weights @ np.maximum(
            np.asarray(moments[2], dtype=float), 0.0)
        return MixtureMoments(
            mean=nominal.mean,
            within_epistemic=nominal.within_epistemic,
            between_mean=nominal.between_mean,
            epistemic=nominal.epistemic,
            aleatoric=np.asarray(aleatoric, dtype=float),
            total=nominal.epistemic + np.asarray(aleatoric, dtype=float),
        )

    def robust_moments_many(
        self,
        output_index,
        X,
        *,
        certification=True,
        expert_moments=None,
    ):
        moments, weights = self._prepare_expert_moments_many(
            output_index,
            X,
            certification=certification,
            expert_moments=expert_moments,
        )
        mean_robust = self.posterior.robust_mixture_moments(
            *moments,
            radius=self.effective_kl_radius(),
            weights=weights,
        )
        if not self.variance_structure_isolated:
            return mean_robust
        zeros = np.zeros_like(np.asarray(moments[2], dtype=float))
        variance_robust = self.posterior.robust_mixture_moments(
            zeros,
            zeros,
            moments[2],
            radius=self.variance_effective_kl_radius(),
            weights=self.variance_structure_weights(),
        )
        nominal = MixtureMoments(
            mean=mean_robust.nominal.mean,
            within_epistemic=mean_robust.nominal.within_epistemic,
            between_mean=mean_robust.nominal.between_mean,
            epistemic=mean_robust.nominal.epistemic,
            aleatoric=variance_robust.nominal.aleatoric,
            total=(
                mean_robust.nominal.epistemic
                + variance_robust.nominal.aleatoric
            ),
        )
        return RobustMixtureMoments(
            mean_upper=mean_robust.mean_upper,
            epistemic_upper=mean_robust.epistemic_upper,
            aleatoric_upper=variance_robust.aleatoric_upper,
            total_upper=(
                mean_robust.epistemic_upper
                + variance_robust.aleatoric_upper
            ),
            nominal=nominal,
            radius=max(
                float(mean_robust.radius),
                float(variance_robust.radius),
            ),
        )

    def robust_chance_margin_many(
        self,
        X,
        *,
        beta_g,
        z_alpha,
        tau,
        certification=True,
        tangent_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0),
        expert_moments=None,
    ):
        """Jointly robustify a chance margin over one shared task law."""

        moments, weights = self._prepare_expert_moments_many(
            1,
            X,
            certification=certification,
            expert_moments=expert_moments,
        )
        if self.variance_structure_isolated:
            nominal = self.mixture_moments_many(
                1,
                X,
                certification=certification,
                expert_moments=expert_moments,
            )
            robust = self.robust_moments_many(
                1,
                X,
                certification=certification,
                expert_moments=expert_moments,
            )
            beta_radius = np.sqrt(max(float(beta_g), 0.0))
            aleatoric_radius = max(float(z_alpha), 0.0)
            nominal_margin = (
                nominal.mean
                + beta_radius * np.sqrt(np.maximum(
                    nominal.epistemic, 0.0))
                + aleatoric_radius * np.sqrt(np.maximum(
                    nominal.aleatoric, 0.0))
                - float(tau)
            )
            separable = (
                robust.mean_upper
                + beta_radius * np.sqrt(np.maximum(
                    robust.epistemic_upper, 0.0))
                + aleatoric_radius * np.sqrt(np.maximum(
                    robust.aleatoric_upper, 0.0))
                - float(tau)
            )
            return RobustChanceMargin(
                upper=np.asarray(separable, dtype=float),
                nominal=np.asarray(nominal_margin, dtype=float),
                separable_upper=np.asarray(separable, dtype=float),
                tangent_epistemic_scale=np.sqrt(np.maximum(
                    nominal.epistemic, self.posterior.variance_floor)),
                tangent_aleatoric_scale=np.sqrt(np.maximum(
                    nominal.aleatoric, self.posterior.variance_floor)),
                used_separable_upper=np.ones_like(
                    nominal_margin, dtype=bool),
                radius=max(
                    self.effective_kl_radius(),
                    self.variance_effective_kl_radius(),
                ),
            )
        return self.posterior.robust_chance_margin(
            *moments,
            beta_g=beta_g,
            z_alpha=z_alpha,
            tau=tau,
            radius=self.effective_kl_radius(),
            weights=weights,
            tangent_multipliers=tangent_multipliers,
        )

    def predictive_sample(self, x, z, expert_uniform):
        """Sample the mean/HVD product posterior with one categorical draw."""
        sensitivity_scale = 1.0
        sensitivity_bias = 0.0
        sensitivity_bias_coefficients = None
        expert_index, variance_expert_index, sensitivity_index = (
            self._predictive_selector_indices(expert_uniform))
        if sensitivity_index is not None:
            sensitivity_scale = float(
                self._task_latent().sensitivity_scales[sensitivity_index])
            sensitivity_bias = float(
                self._task_latent().sensitivity_biases[sensitivity_index])
            coefficients = (
                self._task_latent().sensitivity_bias_coefficients)
            if coefficients is not None:
                sensitivity_bias_coefficients = np.asarray(
                    coefficients[sensitivity_index], dtype=float)
        state = self.states[expert_index]
        variance_state = self.states[variance_expert_index]
        z = np.asarray(z, dtype=float).reshape(-1)
        values = []
        for output_index, standard_normal in enumerate(z):
            model = state.gpr_models[output_index]
            mu = float(model.posterior_mean(x))
            epi = float(model.posterior_var(x))
            alea = float(variance_state.variance_model.predict_variance(
                output_index, x, variance_state.problem))
            if output_index == 1:
                reference_sd = np.sqrt(max(
                    epi + alea, self.posterior.variance_floor))
                bias_offset = sensitivity_bias * reference_sd
                adaptive_variance = 0.0
                features = None
                if sensitivity_bias_coefficients is not None:
                    features = np.asarray(
                        state.problem.task_bias_features(x), dtype=float)
                    bias_offset = reference_sd * (
                        sensitivity_bias
                        + float(sensitivity_bias_coefficients @ features)
                    )
                if self._task_latent().adaptive_bias_enabled:
                    if features is None:
                        features = np.asarray(
                            state.problem.task_bias_features(x), dtype=float)
                    adaptive_mean, adaptive_variance = (
                        self._task_latent().adaptive_bias_moments_one(
                            expert_index, reference_sd, features))
                    bias_offset += adaptive_mean
                mu += bias_offset
                epi = (
                    epi * sensitivity_scale ** 2
                    + adaptive_variance
                )
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
        task_bias_features = self.task_bias_features_many(point)
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
        if self.source_discrepancy_update:
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
                        bias_features=(
                            None
                            if task_bias_features is None
                            else task_bias_features[0, 0]
                        ),
                    )
                )
            task_latent_update = self._task_latent().update_from_predictive(
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
                bias_features=(
                    None
                    if task_bias_features is None
                    else task_bias_features[:, 0, :]
                ),
                chance_z=float(norm.ppf(1.0 - float(getattr(
                    self.states[0].problem, "alpha", 0.05
                )))),
            )
        else:
            frozen_weights = self.posterior.decision_weights().tolist()
            posterior_update = {
                "status": "frozen_source_discrepancy",
                "weights_before": frozen_weights,
                "weights_after": frozen_weights,
                "observation_source": "budgeted_target_evaluation",
                "target_oracle_used": False,
            }
            sensitivity_update = {"status": "frozen_source_discrepancy"}
            task_latent_update = {"status": "frozen_source_discrepancy"}
        if self.source_discrepancy_update and self.posterior.safe_generalized:
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
        variance_posterior = self._replication_variance_posterior()
        if variance_posterior is None:
            variance_posterior_update = {
                "status": "shared_with_mean_task_posterior",
            }
        else:
            replicate_values = [
                float(np.asarray(value, dtype=float)[constraint_index])
                for value in existing
            ] + [float(y[constraint_index])]
            replicate_variance = (
                float(np.var(replicate_values, ddof=1))
                if len(replicate_values) >= 2
                else 0.0
            )
            variance_posterior_update = (
                variance_posterior.update_from_replication(
                    (constraint_index, *point[0]),
                    replicate_variance,
                    [
                        state_alea[constraint_index]
                        for _, _, state_alea in per_state
                    ],
                    len(replicate_values),
                )
            )
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
            constraint_model = state.gpr_models[constraint_index]
            source_diagnostics = getattr(
                constraint_model,
                "source_parametric_prior_diagnostics",
                None,
            )
            if (
                source_diagnostics
                and source_diagnostics.get("adaptation_mode")
                == "sequential_target_evidence_mixture"
                and hasattr(
                    state.variance_model, "set_source_task_posterior")
            ):
                state.variance_model.set_source_task_posterior(
                    constraint_index,
                    source_diagnostics["component_names"],
                    source_diagnostics["component_posterior_weights"],
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
                    replicate_count=len(replicate_values),
                ))
            expert_updates.append({
                "expert": state.name,
                "hvd": hvd_updates,
            })
        self.last_update = {
            "status": "updated",
            "source_discrepancy_update": bool(
                self.source_discrepancy_update),
            "posterior": posterior_update,
            "safe_pairwise": pairwise_diagnostics,
            "sensitivity_posterior": sensitivity_update,
            "task_latent_posterior": task_latent_update,
            "variance_structure_posterior": variance_posterior_update,
            "experts": expert_updates,
            "effective_kl_radius": self.effective_kl_radius(),
            "variance_effective_kl_radius": (
                self.variance_effective_kl_radius()),
        }
        return copy.deepcopy(self.last_update)

    def adaptive_infeasible_penalty(self, fallback=5.0):
        if self.task_latent_authoritative:
            weights = self._task_latent().sensitivity_weights(safe=True)
            return float(
                weights @ self._task_latent().sensitivity_penalties)
        if self.sensitivity_posterior is None:
            return float(fallback)
        return self.sensitivity_posterior.expected_decision_penalty()

    def joint_terminal_risk_many(
        self, X, *, tau, alpha,
    ):
        """Posterior Bayes terminal loss under authoritative ``Q_t(z)``."""
        if not self.task_latent_authoritative:
            raise RuntimeError(
                "joint terminal risk requires authoritative task inference")
        obj_mu, _, _ = self.expert_moments_many(
            0, X, certification=False)
        con_mu, con_epi, con_alea = self.expert_moments_many(
            1, X, certification=True)
        objective_weights = self.structure_weights(objective=True)
        objective = np.asarray(objective_weights @ obj_mu, dtype=float)
        latent = self._task_latent().positive_margin_decision_risk_many(
            con_mu,
            con_epi,
            con_alea,
            tau=tau,
            z_alpha=float(norm.ppf(1.0 - float(alpha))),
            bias_features=self.task_bias_features_many(X),
        )
        expected_positive = np.asarray(
            latent["posterior_expected_positive_margin"], dtype=float)
        expected_loss = np.asarray(
            latent["posterior_expected_decision_loss"], dtype=float)
        return {
            "objective": objective,
            "expected_violation": expected_positive,
            "nominal_expected_violation": expected_positive.copy(),
            "risk": objective + expected_loss,
            "model_disagreement": np.asarray(
                latent["model_disagreement"], dtype=float),
            "violation_probability": np.asarray(
                latent["posterior_violation_probability"], dtype=float),
            "expected_decision_loss": expected_loss,
            "kl_radius": 0.0,
            "task_latent_authoritative": True,
        }

    @staticmethod
    def _structure_family(name):
        value = str(name)
        if value.startswith("risk_aligned"):
            return "risk_aligned"
        if value.startswith("ordered"):
            return "ordered"
        if value == "local_risk_kernel":
            return "local"
        if value == "null_universal":
            return "null"
        return "universal"

    def meta_coherence_diagnostics(
        self,
        X,
        *,
        tau,
        alpha,
        beta_g,
        algorithm_selected_x=None,
        robust_certificate_mode="separable",
    ):
        """Audit whether structural views support one target decision.

        This method consumes surrogate predictions only.  It reports posterior
        agreement and coordinate/HVD support on a fixed candidate set, but does
        not alter recommendation, certification, or acquisition.
        """
        candidates = list(X)
        if not candidates:
            return {
                "status": "empty_candidate_set",
                "used_for_decision": False,
                "target_oracle_used": False,
            }
        objective = self.expert_moments_many(
            0, candidates, certification=False)
        constraint = self.expert_moments_many(
            1, candidates, certification=True)
        obj_mean, _, _ = objective
        con_mean, con_epi, con_alea = constraint
        if self._task_latent().adaptive_bias_enabled:
            con_epi = (
                con_epi
                * self._task_latent()
                .conditional_epistemic_scale_squared(safe=True)[:, None]
                + self._task_latent().adaptive_bias_variance_many(
                    con_epi,
                    con_alea,
                    self.task_bias_features_many(candidates),
                )
            )
        z_alpha = float(norm.ppf(1.0 - float(alpha)))
        margins = (
            con_mean
            + np.sqrt(max(float(beta_g), 0.0))
            * np.sqrt(np.maximum(con_epi, 0.0))
            + z_alpha * np.sqrt(np.maximum(con_alea, 0.0))
            - float(tau)
        )
        weights = np.asarray(
            self.structure_weights(objective=False), dtype=float)
        weights /= max(float(np.sum(weights)), 1e-300)
        feasible_mass = weights @ (margins <= 0.0).astype(float)
        sign_agreement = float(np.mean(np.maximum(
            feasible_mass, 1.0 - feasible_mass)))

        expert_choices = []
        for expert_index in range(len(self.states)):
            feasible = np.where(margins[expert_index] <= 0.0)[0]
            if len(feasible):
                selected = int(feasible[np.argmin(
                    obj_mean[expert_index, feasible])])
            else:
                selected = int(np.lexsort((
                    obj_mean[expert_index], margins[expert_index]))[0])
            expert_choices.append(selected)

        robust_constraint = self.robust_moments_many(
            1, candidates, certification=True)
        mixture_objective = self.mixture_moments_many(
            0, candidates, certification=False).mean
        separable_margin = (
            robust_constraint.mean_upper
            + np.sqrt(max(float(beta_g), 0.0))
            * np.sqrt(np.maximum(robust_constraint.epistemic_upper, 0.0))
            + z_alpha
            * np.sqrt(np.maximum(robust_constraint.aleatoric_upper, 0.0))
            - float(tau)
        )
        joint_constraint = self.robust_chance_margin_many(
            candidates,
            beta_g=beta_g,
            z_alpha=z_alpha,
            tau=tau,
            certification=True,
        )
        robust_certificate_mode = str(
            robust_certificate_mode or "separable"
        ).lower().replace("-", "_")
        mixture_margin = (
            np.asarray(joint_constraint.upper, dtype=float)
            if robust_certificate_mode in {"joint", "joint_kl", "joint_tangent"}
            else np.asarray(separable_margin, dtype=float)
        )
        mixture_feasible = np.where(mixture_margin <= 0.0)[0]
        if len(mixture_feasible):
            selected_index = int(mixture_feasible[np.argmin(
                mixture_objective[mixture_feasible])])
        else:
            selected_index = int(np.lexsort((
                mixture_objective, mixture_margin))[0])
        algorithm_index = None
        if algorithm_selected_x is not None:
            selected_tuple = tuple(int(value) for value in algorithm_selected_x)
            candidate_tuples = [
                tuple(int(value) for value in candidate)
                for candidate in candidates
            ]
            if selected_tuple in candidate_tuples:
                algorithm_index = int(candidate_tuples.index(selected_tuple))
        audited_index = (
            selected_index if algorithm_index is None else algorithm_index)
        selected_support_mass = float(np.sum([
            weight
            for weight, choice in zip(weights, expert_choices)
            if choice == audited_index
        ]))

        weighted_margin_mean = weights @ margins
        weighted_margin_var = weights @ (
            margins - weighted_margin_mean[None, :]) ** 2
        margin_scale = np.maximum(
            np.abs(weighted_margin_mean)
            + weights @ (
                np.sqrt(np.maximum(con_epi, 0.0))
                + np.sqrt(np.maximum(con_alea, 0.0))
            ),
            1e-12,
        )
        normalized_margin_disagreement = float(np.mean(
            np.sqrt(np.maximum(weighted_margin_var, 0.0)) / margin_scale))

        family_mass = {}
        coordinate_mass = {}
        cumulative_hvd_mass = 0.0
        source_domain_sets = []
        for weight, state in zip(weights, self.states):
            family = self._structure_family(state.name)
            family_mass[family] = family_mass.get(family, 0.0) + float(weight)
            provider = (
                state.problem.cumulative_risk_provider_status()
                if hasattr(state.problem, "cumulative_risk_provider_status")
                else {"status": "missing", "coordinate": "missing"}
            )
            coordinate = str(provider.get("coordinate", "missing"))
            coordinate_mass[coordinate] = (
                coordinate_mass.get(coordinate, 0.0) + float(weight))
            source_domain_sets.append(tuple(sorted(
                str(value) for value in provider.get("source_domains", []))))
            variance = state.variance_model.diagnostics()
            active = bool(variance.get("cumulative_active", {}).get("1", False))
            provider_active = bool(
                variance.get("cumulative_provider_active", {}).get("1", False))
            if active and provider_active:
                cumulative_hvd_mass += float(weight)

        latent_risk = self._task_latent().positive_margin_decision_risk_many(
            con_mean,
            con_epi,
            con_alea,
            tau=tau,
            z_alpha=z_alpha,
            bias_features=self.task_bias_features_many(candidates),
        )
        joint_objective = np.asarray(
            self._task_latent().structure_weights(safe=False) @ obj_mean,
            dtype=float,
        )
        joint_risk = (
            joint_objective
            + np.asarray(
                latent_risk["posterior_expected_decision_loss"], dtype=float)
        )
        joint_index = int(np.lexsort((mixture_objective, joint_risk))[0])
        candidate_values = [
            np.asarray(candidate).reshape(-1).tolist()
            for candidate in candidates
        ]
        return {
            "status": "audited",
            "candidate_count": int(len(candidates)),
            "algorithm_selected_index": algorithm_index,
            "algorithm_selected_x": (
                None
                if algorithm_index is None
                else candidate_values[algorithm_index]
            ),
            "robust_reference_selected_index": selected_index,
            "robust_reference_selected_x": candidate_values[selected_index],
            "robust_certificate_mode": robust_certificate_mode,
            "selected_separable_margin": float(
                separable_margin[audited_index]),
            "selected_joint_margin": float(
                joint_constraint.upper[audited_index]),
            "selected_joint_tightening": float(
                separable_margin[audited_index]
                - joint_constraint.upper[audited_index]),
            "joint_risk_selected_index": joint_index,
            "joint_risk_selected_x": candidate_values[joint_index],
            "joint_and_robust_reference_select_same": bool(
                joint_index == selected_index),
            "selected_candidate_expert_support_mass": selected_support_mass,
            "selected_candidate_feasible_mass": float(
                feasible_mass[audited_index]),
            "mean_margin_sign_agreement": sign_agreement,
            "normalized_margin_disagreement": (
                normalized_margin_disagreement),
            "posterior_mass_by_structure_family": {
                key: float(value) for key, value in sorted(family_mass.items())
            },
            "posterior_mass_by_risk_coordinate": {
                key: float(value) for key, value in sorted(coordinate_mass.items())
            },
            "cumulative_hvd_active_mass": float(cumulative_hvd_mass),
            "source_domain_sets_consistent": bool(
                len(set(source_domain_sets)) <= 1),
            "joint_violation_probability_selected": float(
                latent_risk["posterior_violation_probability"][audited_index]),
            "joint_expected_loss_selected": float(
                latent_risk["posterior_expected_decision_loss"][audited_index]),
            "joint_bayes_risk_selected": float(joint_risk[audited_index]),
            "used_for_decision": False,
            "affects_theory_certificate": False,
            "target_oracle_used": False,
        }

    def diagnostics(self):
        sensitivity_weights = (
            None
            if self.sensitivity_posterior is None
            else self.sensitivity_posterior.posterior_weights()
        )
        latent_diagnostics = self._task_latent().diagnostics(
            legacy_structure_weights=(
                self.posterior.decision_posterior_weights()),
            legacy_sensitivity_weights=sensitivity_weights,
        )
        latent_diagnostics["decision_mode"] = str(
            self.task_latent_inference_mode)
        latent_diagnostics["used_for_decision"] = bool(
            self.task_latent_authoritative)
        latent_diagnostics["affects_theory_certificate"] = bool(
            self.task_latent_authoritative)
        return {
            "status": "fit",
            "task_latent_inference_mode": str(
                self.task_latent_inference_mode),
            "task_latent_calibration_mode": str(
                self.task_latent_calibration_mode),
            "task_latent_authoritative": bool(
                self.task_latent_authoritative),
            "source_discrepancy_update": bool(
                self.source_discrepancy_update),
            "posterior": self.posterior.diagnostics(),
            "variance_structure_posterior_mode": str(getattr(
                self, "variance_structure_posterior_mode", "shared")),
            "variance_structure_posterior": (
                {
                    "status": "shared_with_mean_task_posterior",
                    "posterior_weights": self.structure_weights(
                        objective=False).tolist(),
                    "target_mean_used": True,
                }
                if self._replication_variance_posterior() is None
                else self._replication_variance_posterior().diagnostics()
            ),
            "sensitivity_posterior": (
                {"status": "disabled"}
                if self.sensitivity_posterior is None
                else self.sensitivity_posterior.diagnostics()
            ),
            "task_latent_posterior": latent_diagnostics,
            "effective_kl_radius": self.effective_kl_radius(),
            "variance_effective_kl_radius": (
                self.variance_effective_kl_radius()),
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
