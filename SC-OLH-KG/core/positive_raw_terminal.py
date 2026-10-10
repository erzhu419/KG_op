"""Historical raw adaptive-terminal comparison for positive source mixtures."""

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from core.conditional_raw_reference import conditional_raw_coordinates
from core.conditional_terminal_kg import paired_quality_from_moments, risk_coordinates_from_moments, guarded_risk_statistics
from core.guarded_deployment_kg import _sd_quadrature, _tolerance_factor
from core.legacy_joint_kg import clipped_objective_quality


def raw_terminal_component(prior_pair, updated_pair, terminal, remaining, query, labels, scores, nodes,
        *, guarded_mean_position=None, replicates=1, eta_labels=None):
    """Same guarded/direct utility; only the next paid-batch projection is removed."""
    objective = {}; safety = {}; coordinates = {}; rewards = []; costs = []; actions = []; diagnostics = []
    for kind, table, incumbents, budgets in terminal.allocations(remaining):
        for indices, incumbent in zip(table, incumbents):
            key = tuple(map(int, indices)); incumbent = int(incumbent)
            n = len(key); m = len(set((query,)+key))
            if key not in coordinates:
                coordinates[key] = [conditional_raw_coordinates(prior, updated, key, nodes, labels[head], scores[head],
                    query=query, replicates=replicates, eta_label=None if eta_labels is None else eta_labels[head])
                    for head, (prior, updated) in enumerate(zip(prior_pair, updated_pair))]
                diagnostics.extend(r[-1] for r in coordinates[key])
            (fe, fm, fc, fw, _), (ge, gm, gc, gw, _) = coordinates[key]
            if (key, incumbent) not in objective:
                quality = fw @ clipped_objective_quality(fm, np.maximum(np.diagonal(fc, axis1=1, axis2=2), 0.),
                    terminal.protocol['objective_reference'])
                paired = 0.
                if incumbent:
                    pair = [0, incumbent]
                    paired = paired_quality_from_moments(fe[:, pair], fm[:, pair], fc[:, pair][:, :, pair],
                        ndtri(nodes[:, m+1]), terminal.protocol, fw)
                objective[key, incumbent] = np.r_[quality, paired]
            if kind == 'direct':
                sd, weights = _sd_quadrature(budgets[0])
                threshold = _tolerance_factor(budgets[0], terminal.protocol['chance_alpha'], terminal.protocol['familywise_delta'])*sd
                sigma = np.exp(.5*ge[:, 0])
                total = np.sqrt(np.maximum(gc[:, 0, 0], 0.)+sigma**2/budgets[0])
                probability = gw @ (ndtr(-(gm[:, :1]+sigma[:, None]*threshold)/total[:, None]) @ weights)
                reward = objective[key, incumbent][0]*probability; cost = budgets[0]
            else:
                if key not in safety:
                    safety[key] = risk_coordinates_from_moments(ge, gm, gc, np.arange(len(nodes)), nodes,
                        mean_normals=ndtri(nodes[:, m+1:m+n]), integrated_position=guarded_mean_position)
                branches, cost = guarded_risk_statistics(safety[key], incumbent, budgets, terminal.protocol, gw,
                    integrated_position=guarded_mean_position)
                reward = objective[key, incumbent] @ branches
            rewards.append(reward); costs.append(cost)
            actions.append({'kind': kind, 'primary': key[0], 'shortlist': list(key),
                'incumbent_position': incumbent, 'budgets': budgets})
    return np.asarray(rewards), np.asarray(costs), actions, diagnostics


def raw_terminal_values(prior, updated, terminal, remaining, query, labels, scores, *, power, seed,
        utility_identities=None, guarded_mean_position=None, replicates=1, eta_labels=None):
    """Integrate selected utilities without renormalizing any source weight.

    Each omitted identity contributes between zero and its posterior weight
    to every action. Thus returned values are lower bounds, with a common
    additive upper bound equal to omitted_posterior_utility_mass. This bound
    concerns identity integration only; it does not bound Sobol quadrature.
    All source fields and their paid updates remain present.
    """
    nodes = qmc.Sobol(7, scramble=True, seed=seed).random_base2(power)
    rewards = []; costs = []; diagnostics = []
    identities = list(range(len(updated.components))) if utility_identities is None else list(utility_identities)
    for identity in identities:
        old, new = prior.components[identity], updated.components[identity]
        r, c, actions, checks = raw_terminal_component(old, new, terminal, remaining, query, labels, scores[identity], nodes,
            guarded_mean_position=guarded_mean_position,replicates=replicates,eta_labels=eta_labels)
        assert np.all(r >= -1e-12) and np.all(r <= 1.+1e-12)
        rewards.append(r); costs.append(c); diagnostics.extend(checks)
    omitted = [i for i in range(len(updated.components)) if i not in identities]
    return updated.weights[identities] @ rewards, updated.weights[identities] @ costs, actions, {
        'predictive_log_evidence_max_error': max(abs(r['predictive_log_evidence_error']) for r in diagnostics),
        'effective_sample_size_min': min(r['effective_sample_size'] for r in diagnostics),
        'positive_source_components_retained': True,
        'all_complete_source_identities_retained': len(updated.components) == len(prior.components),
        'utility_identities_integrated': identities,
        'omitted_posterior_utility_mass': float(updated.weights[omitted].sum())}
