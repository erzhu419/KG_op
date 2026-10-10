"""Budgeted KG terminal with analytic direct and conditional guarded utility.

The source component is shared across a head's policies. Guarded objective
qualities integrate all source components exactly; its comparison integrates
out the objective mean conditional on the paired difference. Safety integrates
out the last policy mean conditional on every eta and the other policy means.
Cross-policy covariance, paired comparison noise and the existing sample-SD
rule are unchanged. Remaining component/eta/difference integrals use Sobol.
"""

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from core.budgeted_guarded_kg import BudgetedTerminalActions, terminal_covariance_root
from core.direct_terminal_reference import expected_quality, fast_expected_certificates
from core.guarded_deployment_kg import (
    _sd_quadrature, _tolerance_factor, conditional_safety_power,
    conditional_comparison_power)
from core.legacy_joint_kg import clipped_objective_quality
from core.raw_likelihood_lookahead import risk_branch_statistics


def roots(covariance):
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return np.stack([terminal_covariance_root(c) for c in covariance])


def conditional_covariance_inverse(covariance):
    """Use ordinary inverses when the historical pseudoinverse retains all modes."""
    try:
        inverse = np.linalg.solve(covariance, np.broadcast_to(np.eye(covariance.shape[-1]), covariance.shape))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(covariance, hermitian=True, rcond=1e-12)
    # For symmetric matrices the infinity norms bound both spectral norms.
    # Only matrices safely above the old cutoff can use the ordinary inverse.
    product = np.linalg.norm(covariance, ord=np.inf, axis=(-2, -1))*np.linalg.norm(inverse, ord=np.inf, axis=(-2, -1))
    truncated = product >= 1e12
    if np.any(truncated):
        inverse[truncated] = np.linalg.pinv(covariance[truncated], hermitian=True, rcond=1e-12)
    return inverse


def condition_eta_components(field, indices, nodes, *, components=None, eta_normals=None):
    """Factor each sampled source component once instead of once per node."""
    component = (np.minimum(np.searchsorted(np.cumsum(np.exp(field.log_weights)),
        nodes[:, 0]), len(field.means) - 1) if components is None else np.asarray(components))
    selected, position = np.unique(component, return_inverse=True)
    indices = np.asarray(indices); ix = (2 * indices[:, None] + np.arange(2)).reshape(-1)
    mean = field.means[:, ix][selected]
    covariance = field.covariances[:, ix][:, :, ix][selected]
    eta_covariance = covariance[:, 1::2, 1::2]
    cross = covariance[:, ::2, 1::2]
    gain = cross @ conditional_covariance_inverse(eta_covariance)
    normals = ndtri(nodes[:, 1:1 + len(indices)]) if eta_normals is None else eta_normals
    eta = mean[position, 1::2] + np.einsum('nij,nj->ni', roots(eta_covariance)[position], normals)
    conditional_mean = mean[position, ::2] + np.einsum('nij,nj->ni', gain[position], eta - mean[position, 1::2])
    conditional_covariance = covariance[:, ::2, ::2] - gain @ cross.transpose(0, 2, 1)
    conditional_covariance = .5 * (conditional_covariance + conditional_covariance.transpose(0, 2, 1))
    return eta, conditional_mean, conditional_covariance, position


def means_given_eta(field, indices, nodes):
    """Exact Gaussian conditional;retain supported constant shared eta."""
    eta, mean, covariance, position = condition_eta_components(field, indices, nodes)
    return eta, mean, covariance[position]


def paired_objective_quality(field, indices, incumbent, nodes, protocol):
    eta, mean, covariance = means_given_eta(field, [indices[0], indices[incumbent]], nodes)
    return paired_quality_from_moments(eta, mean, covariance, ndtri(nodes[:, 3]), protocol)


def paired_quality_from_moments(eta, mean, covariance, z, protocol, weights=None):
    """Integrate the paired mean conditional on its difference and eta."""
    difference_mean = mean[:, 0] - mean[:, 1]
    variance = np.maximum(covariance[:, 0, 0] + covariance[:, 1, 1] - 2 * covariance[:, 0, 1], 0.)
    cross = np.column_stack([covariance[:, 0, 0] - covariance[:, 0, 1],
        covariance[:, 0, 1] - covariance[:, 1, 1]])
    loading = np.divide(cross, variance[:, None], out=np.zeros_like(cross), where=variance[:, None] > 0.)
    difference = difference_mean + np.sqrt(variance) * z
    quality_mean = mean + loading * (difference - difference_mean)[:, None]
    quality_variance = np.maximum(np.diagonal(covariance, axis1=1, axis2=2) - loading * cross, 0.)
    quality = clipped_objective_quality(quality_mean, quality_variance, protocol['objective_reference'])
    comparison = conditional_comparison_power(difference,
        abs(np.exp(.5 * eta[:, 0]) - np.exp(.5 * eta[:, 1])),
        protocol['comparison_budget'], protocol['comparison_delta'])
    value = comparison * quality[:, 0] + (1 - comparison) * quality[:, 1]
    return float(np.mean(value) if weights is None else weights @ value)


def guarded_risk_coordinates(field, indices, nodes, *, integrated_position=None):
    eta, mean, covariance, position = condition_eta_components(field, indices, nodes)
    return risk_coordinates_from_moments(eta, mean, covariance, position, nodes,
        integrated_position=integrated_position)


def risk_coordinates_from_moments(eta, mean, covariance, position, nodes, mean_normals=None,
        integrated_position=None):
    """Integrate a selected mean conditional on eta and all other means.

    The historical default integrates the last policy. Choosing the queried
    primary instead preserves the joint Gaussian law and its full covariance;
    only the coordinate left for analytic integration changes.
    """
    n = mean.shape[1]; integrated = n-1 if integrated_position is None else integrated_position
    other = [i for i in range(n) if i != integrated]
    first = covariance[:, other][:, :, other]
    if mean_normals is None:
        mean_normals = ndtri(nodes[:, 1 + n:2 * n])
    draws = mean[:, other] + np.einsum('nij,nj->ni', roots(first)[position], mean_normals)
    cross = covariance[:, integrated, other]
    loading = np.einsum('ni,nij->nj', cross, conditional_covariance_inverse(first))
    last_mean = mean[:, integrated] + np.einsum('ni,ni->n', loading[position], draws - mean[:, other])
    last_variance = np.maximum(covariance[:, integrated, integrated] - np.einsum('ni,ni->n', loading, cross), 0.)
    return draws, np.exp(.5 * eta), last_mean, last_variance[position]


def guarded_risk_values(coordinates, incumbent, budgets, protocol,
        integrated_position=None):
    """Pointwise values of the same correlated verifier branches."""
    draws, sigma, last_mean, last_variance = coordinates
    n = len(budgets); delta = protocol['familywise_delta'] / n
    integrated = n-1 if integrated_position is None else integrated_position
    other = [i for i in range(n) if i != integrated]; powers = [None]*n
    for column, i in enumerate(other):
        powers[i] = conditional_safety_power(draws[:, column], sigma[:, i], budgets[i], protocol['chance_alpha'], delta)
    sd, weights = _sd_quadrature(budgets[integrated])
    threshold = _tolerance_factor(budgets[integrated], protocol['chance_alpha'], delta) * sd
    total_sd = np.sqrt(last_variance + sigma[:, integrated]**2 / budgets[integrated])
    powers[integrated] = ndtr(-(last_mean[:, None] + sigma[:, integrated, None] * threshold)/total_sd[:, None]) @ weights
    branches, cost = risk_branch_statistics(np.stack(powers, axis=-1)[:, None], np.ones((len(draws), 1)),
        incumbent, budgets, protocol)
    return branches, cost


def guarded_risk_statistics(coordinates, incumbent, budgets, protocol, sample_weights=None,
        integrated_position=None):
    branches, cost = guarded_risk_values(coordinates, incumbent, budgets, protocol, integrated_position)
    weights = np.full(len(branches), 1/len(branches)) if sample_weights is None else sample_weights
    expected_cost=float(weights @ cost)
    if incumbent:
        # The historical partial-mass diagnostic still pays the two initial
        # certificates in full; its sample weights only scale later branches.
        expected_cost+=(1-float(np.sum(weights)))*(budgets[0]+budgets[incumbent])
    return weights @ branches, expected_cost


def guarded_objective_statistics(field, indices, incumbent, nodes, protocol):
    """Independent objective-head vector for the existing guarded branches."""
    quality = [expected_quality(field, int(i), protocol['objective_reference']) for i in indices]
    paired = paired_objective_quality(field, indices, incumbent, nodes, protocol) if incumbent else 0.
    return np.r_[quality, paired]


class ConditionalTerminalActions(BudgetedTerminalActions):
    """Use the corrected integrator in actual JointResponseMixture lookahead."""

    def evaluate_posterior(self, f_model, g_model, remaining_calls, power, seed, *, guarded_mean_position=None,
                           direct_certificate_cache=None):
        f, _, actions = self.evaluate_head(f_model, remaining_calls, power, seed, 0)
        g, costs, g_actions = self.evaluate_head(g_model, remaining_calls, power, seed, 1,
            guarded_mean_position=guarded_mean_position, direct_certificate_cache=direct_certificate_cache)
        assert actions == g_actions
        return np.sum(f*g, axis=1), costs, actions

    def evaluate_head(self, field, remaining_calls, power, seed, head, *, guarded_mean_position=None,
                      direct_certificate_cache=None, nodes=None):
        """Cache the native paid filter's statistics before joint-identity mixing."""
        if nodes is None:
            nodes = qmc.Sobol(6, scramble=True, seed=seed).random_base2(power)
        reference = self.protocol['objective_reference']
        quality = ({int(i): expected_quality(field, int(i), reference)
            for i in np.unique(self.shortlists)} if head == 0 else {})
        allocations = self.allocations(remaining_calls)
        direct = [(int(i[0]), b[0]) for kind, table, _, b in allocations
            if kind == 'direct' for i in table]
        # The caller may reuse this cache only for this unchanged posterior
        # and budget. Direct probabilities do not depend on Sobol power.
        certificates = {} if direct_certificate_cache is None else direct_certificate_cache
        if head == 1 and not certificates:
            probabilities, _ = fast_expected_certificates(field,
                [i for i, _ in direct], [n for _, n in direct], self.protocol)
            certificates.update(zip(direct, probabilities))
        objective = {}; safety = {}; statistics = []; costs = []; actions = []
        for kind, table, incumbents, budgets in allocations:
            for indices, incumbent in zip(table, incumbents):
                incumbent = int(incumbent); key = tuple(indices)
                vector = np.zeros(4); cost = 0.
                if kind == 'direct':
                    vector[0] = (quality[int(indices[0])] if head == 0
                        else certificates[int(indices[0]), budgets[0]])
                    if head == 1: cost = budgets[0]
                elif head == 0:
                    if (key, incumbent) not in objective:
                        paired = (paired_objective_quality(field, indices, incumbent, nodes, self.protocol)
                            if incumbent else 0.)
                        objective[key, incumbent] = np.r_[[quality[int(i)] for i in indices], paired]
                    vector = objective[key, incumbent]
                else:
                    if key not in safety:
                        safety[key] = guarded_risk_coordinates(field, indices, nodes,
                            integrated_position=guarded_mean_position)
                    vector, cost = guarded_risk_statistics(safety[key], incumbent, budgets, self.protocol,
                        integrated_position=guarded_mean_position)
                statistics.append(vector); costs.append(cost)
                actions.append({'kind': kind, 'primary': int(indices[0]), 'shortlist': indices.tolist(),
                    'incumbent_position': incumbent, 'budgets': budgets})
        return np.asarray(statistics), np.asarray(costs), actions
