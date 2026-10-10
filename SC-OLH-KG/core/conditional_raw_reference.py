"""Condition one raw episode without projecting its eta posterior.

Reuse the historical posterior-component/eta importance proposal. Given query
eta and the raw label, the other eta coordinates and all means are Gaussian.
Integrate means with the same guarded terminal functions as current KG. Earlier
paid updates remain the frozen moment fields; only this next update is exact.
"""

import numpy as np
from scipy.special import logsumexp, ndtri

from core.conditional_terminal_kg import (roots, paired_quality_from_moments,
    risk_coordinates_from_moments, guarded_risk_statistics, conditional_covariance_inverse)
from core.legacy_joint_kg import clipped_objective_quality
from core.positive_mixture_lookahead import raw_eta_proposal


def conditional_raw_coordinates(field, updated, indices, nodes, label, head_log_evidence, *, query=None,
        replicates=1, eta_label=None):
    """Condition query eta first, including queries outside deployment support."""
    query = indices[0] if query is None else query
    points = list(dict.fromkeys([query]+list(indices))); n = len(points)
    positions = [points.index(i) for i in indices]
    component = np.minimum(np.searchsorted(np.cumsum(np.exp(updated.log_weights)), nodes[:, 0]),
        len(field.means)-1)
    selected, position = np.unique(component, return_inverse=True)
    ix = (2*np.asarray(points)[:, None]+np.arange(2)).reshape(-1)
    mean = field.means[:, ix][selected]; C = field.covariances[:, ix][:, :, ix][selected]
    eta_query, logs = raw_eta_proposal(field, updated, query, component, ndtri(nodes[:, 1]),
        label, head_log_evidence, replicates=replicates, eta_label=eta_label)
    rest = np.r_[np.arange(0, 2*n, 2), np.arange(3, 2*n, 2)]
    ev = C[:, 1, 1]; cross = C[:, rest, 1]
    before_covariance = C[:, rest][:, :, rest]-cross[:, :, None]*cross[:, None, :]/ev[:, None, None]
    before_mean = mean[position][:, rest]+cross[position]/ev[position, None]*(eta_query-mean[position, 1])[:, None]
    noise = np.exp(eta_query)/replicates
    total = before_covariance[position, 0, 0]+noise
    if n == 1:
        other_eta = np.empty((len(nodes), 0))
        mean_given_eta, covariance_given_eta = before_mean, before_covariance
    else:
        # Given query eta, the observation downdates the other-eta covariance
        # by one rank. Factor/invert the prior block once per source component,
        # then transport each normal draw with the analytic rank-one square root.
        eta_covariance = before_covariance[:, n:, n:]
        root = roots(eta_covariance); inverse = conditional_covariance_inverse(eta_covariance)
        cross = before_covariance[:, n:, 0]
        whitened = np.einsum('kij,kj->ki', root.transpose(0, 2, 1),
            np.einsum('kij,kj->ki', inverse, cross))
        length = np.sum(whitened**2, axis=1)[position]
        shrink = 1/(total*(1+np.sqrt(np.maximum(1-length/total, 0.))))
        z = ndtri(nodes[:, 2:n+1]); direction = whitened[position]
        transported = z-direction*(np.sum(direction*z, axis=1)*shrink)[:, None]
        eta_mean = before_mean[:, n:]+cross[position]/total[:, None]*(label-before_mean[:, 0])[:, None]
        other_eta = eta_mean+np.einsum('nij,nj->ni', root[position], transported)
        loading = before_covariance[:, :n, n:] @ inverse
        mean_given_eta = before_mean[:, :n]+np.einsum('nij,nj->ni', loading[position], other_eta-before_mean[:, n:])
        covariance_given_eta = before_covariance[:, :n, :n]-loading @ before_covariance[:, n:, :n]
    covariance_given_eta = .5*(covariance_given_eta+covariance_given_eta.transpose(0, 2, 1))
    variance = covariance_given_eta[position, 0, 0]+noise
    gain = covariance_given_eta[position, :, 0]/variance[:, None]
    conditional_mean = mean_given_eta+gain*(label-mean_given_eta[:, 0])[:, None]
    conditional_covariance = covariance_given_eta[position]-gain[:, :, None]*covariance_given_eta[position, None, 0, :]
    conditional_covariance = .5*(conditional_covariance+conditional_covariance.transpose(0, 2, 1))
    normalizer = logsumexp(logs); weights = np.exp(logs-normalizer)
    diagnostics = {'predictive_log_evidence_error': float(normalizer-np.log(len(nodes))),
        'effective_sample_size': float(1/(weights @ weights))}
    return (np.column_stack([eta_query, other_eta])[:, positions], conditional_mean[:, positions],
        conditional_covariance[:, positions][:, :, positions], weights, diagnostics)


def raw_guarded_head_statistics(field, updated, action, nodes, label, head_log_evidence, head, protocol):
    eta, mean, covariance, weights, diagnostics = conditional_raw_coordinates(field, updated,
        action['shortlist'], nodes, label, head_log_evidence)
    if head == 0:
        quality = clipped_objective_quality(mean, np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.),
            protocol['objective_reference'])
        i = action['incumbent_position']; paired = 0.
        if i:
            pair = [0, i]
            paired = paired_quality_from_moments(eta[:, pair], mean[:, pair], covariance[:, pair][:, :, pair],
                ndtri(nodes[:, 4]), protocol, weights)
        statistics = np.r_[weights @ quality, paired]
    else:
        coordinates = risk_coordinates_from_moments(eta, mean, covariance, np.arange(len(nodes)), nodes)
        statistics = guarded_risk_statistics(coordinates, action['incumbent_position'], action['budgets'],
            protocol, weights)[0]
    return statistics, diagnostics
