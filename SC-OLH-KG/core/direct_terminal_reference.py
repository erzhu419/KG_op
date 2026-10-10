"""Historical Gaussian mean integration for the current direct certificate.

Keep the verifier's existing16-node sample-SD rule. Integrate clipped objective
quality analytically and average the uncertain constraint mean analytically
conditional on eta. Adaptive one-dimensional eta integration retains every
positive source component and the shared complete-identity mass.
"""

import copy
from functools import lru_cache

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import ndtr, roots_hermitenorm
from scipy.stats import norm, qmc

from core.guarded_deployment_kg import _sd_quadrature, _tolerance_factor
from core.legacy_joint_kg import clipped_objective_quality


def expected_quality(field, candidate, reference):
    index = 2 * candidate
    values = clipped_objective_quality(field.means[:, index],
        field.covariances[:, index, index], reference)
    return float(np.exp(field.log_weights) @ values)


def expected_certificate(field, candidate, budget, protocol):
    indices = [2 * candidate, 2 * candidate + 1]
    mean = field.means[:, indices]; covariance = field.covariances[:, indices][:, :, indices]
    g, eta = mean.T; gv, ev, cross = covariance[:, 0, 0], covariance[:, 1, 1], covariance[:, 0, 1]
    eta_sd = np.sqrt(ev)
    loading = np.divide(cross, eta_sd, out=np.zeros_like(cross), where=eta_sd > 0.)
    conditional = np.maximum(gv - loading**2, 0.)
    sd, chi_weights = _sd_quadrature(budget)
    chi_terms = _tolerance_factor(budget, protocol['chance_alpha'], protocol['familywise_delta']) * sd
    weights = np.exp(field.log_weights)

    def integrand(z):
        sigma = np.exp(.5 * (eta + eta_sd * z))
        total_sd = np.sqrt(conditional + sigma**2 / budget)
        power = ndtr(-(g[:, None] + loading[:, None] * z + sigma[:, None] * chi_terms) / total_sd[:, None]) @ chi_weights
        return float(weights @ power) * norm.pdf(z)

    value, error = quad(integrand, -12., 12., points=[0.], epsabs=1e-11, epsrel=1e-11)
    return float(value), float(error + 2 * norm.sf(12.))


def certificate_integrand(field, candidates, budgets, protocol):
    """Marginal certificate probability conditional on standardized eta."""
    candidates = np.asarray(candidates, dtype=int); budgets = np.asarray(budgets)
    g = field.means[:, 2 * candidates]; eta = field.means[:, 2 * candidates + 1]
    gv = field.covariances[:, 2 * candidates, 2 * candidates]
    ev = field.covariances[:, 2 * candidates + 1, 2 * candidates + 1]
    cross = field.covariances[:, 2 * candidates, 2 * candidates + 1]
    eta_sd = np.sqrt(ev)
    loading = np.divide(cross, eta_sd, out=np.zeros_like(cross), where=eta_sd > 0.)
    conditional = np.maximum(gv - loading**2, 0.)
    quadrature = [_sd_quadrature(int(n)) for n in budgets]
    chi_weights = np.stack([rule[1] for rule in quadrature])
    thresholds = np.stack([_tolerance_factor(int(n), protocol['chance_alpha'],
        protocol['familywise_delta']) * rule[0] for n, rule in zip(budgets, quadrature)])
    weights = np.exp(field.log_weights)

    def integrand(z):
        sigma = np.exp(.5 * (eta + eta_sd * z))
        total_sd = np.sqrt(conditional + sigma**2 / budgets)
        powers = ndtr(-(g[:, :, None] + loading[:, :, None] * z
            + sigma[:, :, None] * thresholds) / total_sd[:, :, None])
        probabilities = np.einsum('mac,ac->ma', powers, chi_weights)
        return weights @ probabilities
    return integrand


def expected_certificates(field, candidates, budgets, protocol):
    """Adaptive integration controls every action with the maximum norm."""
    probability = certificate_integrand(field, candidates, budgets, protocol)
    def integrand(z):
        return probability(z) * np.exp(-.5 * z*z) / np.sqrt(2*np.pi)

    values, error, info = quad_vec(integrand, -12., 12., points=[0.],
        epsabs=1e-11, epsrel=1e-11, norm='max', full_output=True, workers=1)
    if not info.success:
        raise RuntimeError(info.message)
    return values, float(error + 2 * norm.sf(12.))


@lru_cache(maxsize=4)
def eta_quadrature(order):
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights/np.sqrt(2*np.pi)


def fast_expected_certificates(field, candidates, budgets, protocol):
    """Historical Gaussian eta rule with unresolved actions kept adaptive.

    The16/32/64/128 rules retain all positive source components. Accept each
    action only after adjacent rules differ by at most1e-11;otherwise use the
    unchanged adaptive integral. The difference is an error diagnostic,not a
    rigorous bound. Actual-law scalar references remain the acceptance test.
    """
    candidates = np.asarray(candidates); budgets = np.asarray(budgets)
    pending = np.arange(len(candidates)); values = np.empty(len(candidates))
    previous = None; largest_difference = 0.; orders = {}
    for order in (16, 32, 64, 128):
        integrand = certificate_integrand(field, candidates[pending], budgets[pending], protocol)
        z, w = eta_quadrature(order)
        current = sum(weight * integrand(node) for node, weight in zip(z, w))
        if previous is not None:
            difference = abs(current - previous); converged = difference <= 1e-11
            if converged.any():
                largest_difference = max(largest_difference, float(difference[converged].max()))
                values[pending[converged]] = current[converged]
                orders[order] = int(converged.sum())
            pending = pending[~converged]; current = current[~converged]
        if not len(pending):
            break
        previous = current
    adaptive_error = 0.
    if len(pending):
        values[pending], adaptive_error = expected_certificates(field,
            candidates[pending], budgets[pending], protocol)
    return values, {'accepted_orders': orders, 'maximum_accepted_nested_difference': largest_difference,
        'adaptive_fallback_actions': len(pending), 'adaptive_absolute_error_bound': adaptive_error}


def direct_terminal_reference(model, action, protocol):
    assert action['kind'] == 'direct' and len(action['shortlist']) == len(action['budgets']) == 1
    candidate = action['primary']; budget = action['budgets'][0]
    rewards, errors = [], []
    for f, g in model.components:
        quality = expected_quality(f, candidate, protocol['objective_reference'])
        probability, error = expected_certificate(g, candidate, budget, protocol)
        rewards.append(quality * probability); errors.append(quality * error)
    return {'expected_reward': float(model.weights @ rewards),
        'absolute_eta_quadrature_error': float(model.weights @ errors),
        'component_rewards': rewards, 'verification_calls': budget,
        'all_source_components_retained': True, 'sample_sd_quadrature_order': 16}


def fixed_direct_projected_tower(model, action, query, power, seed, protocol):
    """Reuse the historical fixed-action head factorization with analytic utility.

    The shared posterior identity cancels against its predictive density for
    this fixed action. Average each head under its own identity predictive law,
    then multiply within that identity,as in fixed_action_factorized_reference.
    Every future field uses the actual paid Gaussian-moment update.
    """
    from core.positive_mixture_lookahead import RawPredictiveCDF
    probabilities = qmc.Sobol(2, scramble=True, seed=seed + 1).random_base2(power)
    before, after, prefix, head_values = [], [], [], []
    inverse_error = raw_error = eta_error = 0.
    for pair in model.components:
        prior, future_heads = [], []
        for head, field in enumerate(pair):
            def statistic(current):
                if head == 0:
                    return expected_quality(current, action['primary'], protocol['objective_reference']), 0.
                return expected_certificate(current, action['primary'], action['budgets'][0], protocol)
            value, error = statistic(field); prior.append(value); eta_error = max(eta_error, error)
            cdf = RawPredictiveCDF(field, query); labels = cdf.inverse(probabilities[:, head])
            reference, bound = cdf.adaptive(labels)
            inverse_error = max(inverse_error, float(np.max(abs(reference - probabilities[:, head]))) + bound)
            assert inverse_error <= 1e-8
            values = []
            for label in labels:
                future = copy.deepcopy(field); future.update_batch(query, float(label), None, 1)
                value, error = statistic(future); values.append(value)
                eta_error = max(eta_error, error); raw_error = max(raw_error, future.max_integration_relative_error)
            future_heads.append(values)
        before.append(prior[0] * prior[1])
        after.append(np.mean(future_heads[0]) * np.mean(future_heads[1]))
        count = len(probabilities) // 2
        prefix.append(np.mean(future_heads[0][:count]) * np.mean(future_heads[1][:count]))
        head_values.append(future_heads)
    prior = float(model.weights @ before); future = float(model.weights @ after)
    nested = float(model.weights @ prefix)
    return {'candidate': query, 'action': action, 'fantasy_power': power, 'fantasies_per_head_identity': len(probabilities),
        'prior_fixed_reward': prior, 'expected_future_fixed_reward': future, 'projected_fixed_action_drift': future - prior,
        'nested_prefix_expected_future_reward': nested, 'nested_expected_future_difference': abs(future - nested),
        'maximum_inverse_cdf_error': inverse_error, 'maximum_raw_integration_relative_error': raw_error,
        'maximum_absolute_eta_quadrature_error': eta_error, 'future_head_statistics_by_identity': head_values,
        'shared_identity_retained': True, 'adaptive_maximum_evaluated': False}
