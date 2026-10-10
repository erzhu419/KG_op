"""Source price uncertainty with the public billing offset enforced exactly.

For centered quantities Z=(q-mu)/s, the coefficient law obeys
b=offset+(mu/s) beta. Orthogonal functions do not imply independent prices
or an independent intercept. Non-intercept target means remain zero.
"""

import numpy as np
from scipy.stats import norm

from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.source_function_prior import SourceFunctionPrior


def link_billing_coefficients(basis, coefficient_covariance, discrepancy, diagnostics, offset=0.):
    """Represent cost q*theta plus a known offset in centered coordinates."""
    loading = basis.mean / basis.scale
    transport = np.vstack([loading, np.eye(basis.feature_dim)])
    mean = np.r_[float(offset), np.zeros(basis.feature_dim)]
    info = {**basis.diagnostics(), **diagnostics,
        "public_mean_offset": float(offset), "intercept_loading": loading.tolist(),
        "intercept_is_independent": False, "coefficient_mean": mean.tolist(),
        "coefficient_covariance": np.asarray(coefficient_covariance).tolist()}
    return SourceFunctionPrior(basis, mean, transport @ coefficient_covariance @ transport.T,
                               discrepancy, info)


def source_price_moments(batches, noises, basis, offsets):
    """Fit unknown price coefficients from source responses after normalization."""
    coefficients = []
    for batch, noise in zip(batches, noises):
        raw_scaled = basis.transform(batch.psi) + basis.mean / basis.scale
        weights = 1. / np.maximum(noise, 1e-8)
        response = np.asarray(batch.signals).reshape(-1) - offsets
        coefficients.append(np.linalg.solve(raw_scaled.T @ (weights[:, None] * raw_scaled)
            + 1e-3 * np.eye(basis.feature_dim), raw_scaled.T @ (weights * response)))
    beta = np.asarray(coefficients)
    second_moment = beta.T @ beta / len(beta)
    floor = max(float(np.trace(second_moment) / basis.feature_dim) * .05, .01)
    return second_moment + floor * np.eye(basis.feature_dim)


def fit_billing_offset_prior(batches, noises, reference, offsets, adaptation, test):
    """Fixed, previously source-selected billing support; source-only residual CV."""
    basis = InventoryFunctionBasis(reference, InventoryFunctionBasis.billing_indices)
    scores = {0.: [], .02: []}
    for held_index, held in enumerate(batches):
        keep = [i for i in range(len(batches)) if i != held_index]
        covariance = source_price_moments([batches[i] for i in keep], [noises[i] for i in keep], basis, offsets)
        for discrepancy in scores:
            prior = link_billing_coefficients(basis, covariance, discrepancy, {})
            model = prior.target_model(held.psi)
            for index in adaptation:
                model.update([index], held.signals[index] - offsets[index], noises[held_index][index])
            candidates = np.asarray(test)[:, None]
            mean = model.posterior_mean_many(candidates) + offsets[test]
            variance = model.posterior_var_many(candidates) + noises[held_index][test]
            scores[discrepancy].append(float(np.mean(norm.logpdf(held.signals[test], mean, np.sqrt(variance)))))
    chosen = max(scores, key=lambda key: float(np.mean(scores[key])))
    covariance = source_price_moments(batches, noises, basis, offsets)
    info = {"mode": "source_linked_billing", "source_task_count": len(batches),
        "source_label_count": sum(len(b.signals) for b in batches),
        "support_role": "four billing functions fixed from the prior source-only study",
        "source_residual_scores": {str(k): v for k, v in scores.items()},
        "chosen_discrepancy_variance": chosen,
        "source_response_offsets": offsets.tolist()}
    return link_billing_coefficients(basis, covariance, chosen, info)


def pooled_linked_prior(prior):
    covariance = np.eye(prior.basis.feature_dim) * np.trace(prior.covariance[1:, 1:]) / prior.basis.feature_dim
    return link_billing_coefficients(prior.basis, covariance, prior.discrepancy_variance,
        {**prior.diagnostics, "mode": "same_source_linked_pooled",
         "coefficient_trace_matched": True}, prior.mean[0])
