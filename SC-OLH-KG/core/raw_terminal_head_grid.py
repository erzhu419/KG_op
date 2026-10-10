"""Reuse raw terminal head statistics; maximize inside joint predictive grids."""

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import qmc

from core.conditional_raw_reference import conditional_raw_coordinates
from core.conditional_terminal_kg import paired_quality_from_moments, risk_coordinates_from_moments, guarded_risk_statistics
from core.guarded_deployment_kg import _sd_quadrature, _tolerance_factor
from core.legacy_joint_kg import clipped_objective_quality
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments


def raw_terminal_head(prior, updated, terminal, remaining, query, label, score, nodes, head,
        guarded_mean_position=None, *, replicates=1, eta_label=None):
    """Current raw law, including queried-primary mean integration, for one head.

    Four guarded entries are three qualities/branches and the paired branch.
    Direct entries use only the first coordinate. Risk costs depend on g only.
    Full cross-policy covariance and the historical verifier remain unchanged.
    """
    coordinates = {}; objective = {}; safety = {}; statistics = []; costs = []; actions = []; checks = []
    for kind, table, incumbents, budgets in terminal.allocations(remaining):
        for indices, incumbent in zip(table, incumbents):
            key = tuple(map(int, indices)); incumbent = int(incumbent)
            n = len(key); m = len(set((query,)+key))
            if key not in coordinates:
                coordinates[key] = conditional_raw_coordinates(prior, updated, key, nodes, label, score,
                    query=query,replicates=replicates,eta_label=eta_label)
                checks.append(coordinates[key][-1])
            eta, mean, covariance, weights, _ = coordinates[key]
            vector = np.zeros(4); cost = 0.
            if head == 0:
                if (key, incumbent) not in objective:
                    quality = weights @ clipped_objective_quality(mean,
                        np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.), terminal.protocol['objective_reference'])
                    paired = 0.
                    if incumbent:
                        pair = [0, incumbent]
                        paired = paired_quality_from_moments(eta[:, pair], mean[:, pair],
                            covariance[:, pair][:, :, pair], ndtri(nodes[:, m+1]), terminal.protocol, weights)
                    objective[key, incumbent] = np.r_[quality, paired]
                if kind == 'direct':
                    vector[0] = objective[key, incumbent][0]
                else:
                    vector[:] = objective[key, incumbent]
            elif kind == 'direct':
                sd, quadrature = _sd_quadrature(budgets[0])
                threshold = _tolerance_factor(budgets[0], terminal.protocol['chance_alpha'],
                    terminal.protocol['familywise_delta'])*sd
                sigma = np.exp(.5*eta[:, 0])
                total = np.sqrt(np.maximum(covariance[:, 0, 0], 0.)+sigma**2/budgets[0])
                vector[0] = weights @ (ndtr(-(mean[:, :1]+sigma[:, None]*threshold)/total[:, None]) @ quadrature)
                cost = budgets[0]
            else:
                integrated = guarded_mean_position if guarded_mean_position is not None else (0 if key[0] == query else None)
                if key not in safety:
                    safety[key] = risk_coordinates_from_moments(eta, mean, covariance, np.arange(len(nodes)), nodes,
                        mean_normals=ndtri(nodes[:, m+1:m+n]), integrated_position=integrated)
                vector, cost = guarded_risk_statistics(safety[key], incumbent, budgets, terminal.protocol, weights,
                    integrated_position=integrated)
            statistics.append(vector); costs.append(cost)
            actions.append({'kind': kind, 'primary': key[0], 'shortlist': list(key),
                'incumbent_position': incumbent, 'budgets': budgets})
    return np.asarray(statistics), np.asarray(costs), actions, {
        'predictive_log_evidence_error': checks[0]['predictive_log_evidence_error'],
        'predictive_log_evidence_max_error': max(abs(c['predictive_log_evidence_error']) for c in checks),
        'effective_sample_size_min': min(c['effective_sample_size'] for c in checks)}


def evaluate_head_records(model, terminal, remaining, query, head, labels, positions, *, powers, seed,
        completed=None, checkpoint=None, utility_identities=None, guarded_mean_position=None, workers=1,
        replicates=1, eta_labels=None, projected_powers=()):
    """One conditioning/terminal calculation per distinct head label,not per pair."""
    records = [] if completed is None else list(completed)
    assert all(r.get('replicates',1) == replicates for r in records)
    by_position = {r['position']: i for i, r in enumerate(records)}
    pending = [int(p) for p in positions if int(p) not in by_position or
        any(str(power) not in records[by_position[int(p)]]['raw'] for power in powers) or
        any(str(power) not in records[by_position[int(p)]].get('projected', {}) for power in projected_powers)]
    if not pending:
        return records
    cache = [raw_fantasy_moments(pair[head], query, np.asarray([labels[p] for p in pending]),
        None if eta_labels is None else np.asarray([eta_labels[p] for p in pending]),replicates)
        for pair in model.components]
    # Construct each deterministic grid once on the caller thread. SciPy's
    # first Sobol direction-table initialization is not thread safe.
    nodes_by_power = {power: qmc.Sobol(7, scramble=True, seed=seed).random_base2(power) for power in powers}
    projected_nodes_by_power = {power: qmc.Sobol(6, scramble=True, seed=seed).random_base2(power)
        for power in projected_powers}
    def evaluate(item):
        sample, position = item
        tick = time.thread_time(); scores = []; fields = []
        before = records[by_position[position]] if position in by_position else None
        if before is not None:
            assert before['label'] == float(labels[position]) and before['head'] == head
            assert before.get('guarded_mean_position') == guarded_mean_position
            assert before.get('replicates',1) == replicates
            assert before.get('eta_label') == (None if eta_labels is None else float(eta_labels[position]))
        for identity, pair in enumerate(model.components):
            if utility_identities is not None and identity not in utility_identities:
                # The omitted utility still contributes its exact predictive
                # likelihood. Its full covariance update is never consumed.
                scores.append(float(logsumexp(pair[head].log_weights+cache[identity][0][sample]['log_evidence'])))
                fields.append(None)
                continue
            field = copy.deepcopy(pair[head])
            scores.append(apply_raw_moments(field, query, cache[identity][0][sample], cache[identity][1],replicates=replicates))
            assert field.paid_episodes == pair[head].paid_episodes+replicates
            fields.append(field)
        if before is not None:
            np.testing.assert_allclose(scores, before['head_log_scores'], atol=1e-10, rtol=0.)
            scores = before['head_log_scores']
        raw = {} if before is None else dict(before['raw'])
        projected = {} if before is None else dict(before.get('projected', {}))
        certificate_caches = {}
        for power in projected_powers:
            if str(power) in projected: continue
            statistics = []; costs = []
            for identity, field in enumerate(fields):
                if utility_identities is not None and identity not in utility_identities:
                    statistics.append(None); costs.append(None); continue
                s, c, actions = terminal.evaluate_head(field, remaining-replicates, power, seed, head,
                    guarded_mean_position=guarded_mean_position,
                    direct_certificate_cache=certificate_caches.setdefault(identity, {}),
                    nodes=projected_nodes_by_power[power])
                statistics.append(s); costs.append(c)
            template = next(s for s in statistics if s is not None)
            projected[str(power)] = {
                'statistics': np.asarray([np.zeros_like(template) if s is None else s for s in statistics]).tolist(),
                'costs': np.asarray([np.zeros(len(template)) if c is None else c for c in costs]).tolist()}
        for power in powers:
            if str(power) in raw:
                continue
            nodes = nodes_by_power[power]
            statistics = []; costs = []; diagnostics = []; by_identity = []
            for identity, field in enumerate(fields):
                if utility_identities is not None and identity not in utility_identities:
                    statistics.append(None); costs.append(None); by_identity.append(None)
                    continue
                s, c, actions, d = raw_terminal_head(model.components[identity][head], field,
                    terminal, remaining-replicates, query, labels[position], scores[identity], nodes, head,
                    guarded_mean_position=guarded_mean_position,replicates=replicates,
                    eta_label=None if eta_labels is None else eta_labels[position])
                statistics.append(s); costs.append(c); diagnostics.append(d); by_identity.append(d)
            template = next(s for s in statistics if s is not None)
            statistics = [np.zeros_like(template) if s is None else s for s in statistics]
            costs = [np.zeros(len(template)) if c is None else c for c in costs]
            raw[str(power)] = {'statistics': np.asarray(statistics).tolist(), 'costs': np.asarray(costs).tolist(),
                'diagnostics_by_identity': by_identity,
                'predictive_log_evidence_max_error': max(d['predictive_log_evidence_max_error'] for d in diagnostics),
                'effective_sample_size_min': min(d['effective_sample_size_min'] for d in diagnostics)}
        record = {'position': position, 'head': head, 'label': float(labels[position]),
            'replicates':replicates,'eta_label':None if eta_labels is None else float(eta_labels[position]),
            'future_remaining_calls':remaining-replicates,
            'head_log_scores': scores, 'raw': raw, 'cpu_seconds': time.thread_time()-tick,
            'utility_identities': list(range(len(fields))) if utility_identities is None else list(utility_identities)}
        if guarded_mean_position is not None:
            record['guarded_mean_position'] = guarded_mean_position
        if projected:
            record['projected'] = projected
        if before is not None:
            record['cpu_seconds'] += before['cpu_seconds']
        return record, actions
    pool = ThreadPoolExecutor(max_workers=min(workers, len(pending))) if workers > 1 else None
    try:
        evaluated = pool.map(evaluate, enumerate(pending)) if pool else map(evaluate, enumerate(pending))
        for record, actions in evaluated:
            position = record['position']
            if position in by_position:
                records[by_position[position]] = record
            else:
                records.append(record)
            if checkpoint:
                checkpoint(records, actions)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    return records


def compose_head_grid(f_statistics, g_statistics, g_costs, f_scores, g_scores, prior_weights):
    """Condition the shared identity jointly,then compose every complete action.

    Labels are integrated under the product of their predictive marginals.
    The joint/marginal density ratio restores the shared-identity dependence.
    Do not renormalize finite quadrature weights or factorize the maximum.
    """
    prior = np.asarray(prior_weights); fs = np.asarray(f_scores); gs = np.asarray(g_scores)
    log_prior = np.log(prior)
    log_joint = log_prior+fs[:, None, :]+gs[None, :, :]
    evidence = logsumexp(log_joint, axis=2)
    identity_weights = np.exp(log_joint-evidence[:, :, None])
    log_f = logsumexp(log_prior+fs, axis=1); log_g = logsumexp(log_prior+gs, axis=1)
    density_ratio = np.exp(evidence-log_f[:, None]-log_g[None, :])
    rewards = np.einsum('fgk,fkab,gkab->fga', identity_weights, f_statistics, g_statistics, optimize=True)
    costs = np.einsum('fgk,gka->fga', identity_weights, g_costs, optimize=True)
    return rewards, costs, identity_weights, density_ratio


def summarize_head_grid(f_records, g_records, f_weights, g_weights, prior_weights, *, power, baseline):
    f = np.asarray([r['raw'][str(power)]['statistics'] for r in f_records])
    g = np.asarray([r['raw'][str(power)]['statistics'] for r in g_records])
    costs = np.asarray([r['raw'][str(power)]['costs'] for r in g_records])
    values, costs, posterior, ratio = compose_head_grid(f, g, costs,
        [r['head_log_scores'] for r in f_records], [r['head_log_scores'] for r in g_records], prior_weights)
    weights = np.outer(f_weights, g_weights)*ratio
    return summarize_joint_actions(values, costs, posterior, weights, baseline, power, len(f_records)+len(g_records))


def summarize_origin_head_grid(f_records, g_records, f_weights, g_weights, prior_weights, origin, *, power, baseline):
    """An origin's contribution using the existing conditional-identity banks.

    Both label axes are drawn under the SAME origin identity. Their product
    rule already represents that origin's joint predictive law; only its
    prior mass is added. All posterior identities are still conditioned by
    both labels. Sum these origin-weighted contributions,including baselines,
    to recover the complete mixture expectation.
    """
    f = np.asarray([r['raw'][str(power)]['statistics'] for r in f_records])
    g = np.asarray([r['raw'][str(power)]['statistics'] for r in g_records])
    costs = np.asarray([r['raw'][str(power)]['costs'] for r in g_records])
    values, costs, posterior, _ = compose_head_grid(f, g, costs,
        [r['head_log_scores'] for r in f_records], [r['head_log_scores'] for r in g_records], prior_weights)
    mass = prior_weights[origin]; weights = mass*np.outer(f_weights, g_weights)
    stop, debited, fixed = baseline
    result = summarize_joint_actions(values, costs, posterior, weights,
        (mass*stop, mass*debited, fixed), power, len(f_records)+len(g_records))
    omitted = sorted(set(range(len(prior_weights)))-set(f_records[0].get('utility_identities', range(len(prior_weights)))))
    assert all(r.get('utility_identities', list(range(len(prior_weights)))) ==
        f_records[0].get('utility_identities', list(range(len(prior_weights)))) for r in f_records+g_records)
    return dict(result, origin_identity=origin, origin_prior_mass=float(mass),
        weighted_omitted_terminal_identity_mass=float(np.sum(weights*posterior[:, :, omitted].sum(axis=2))))


def summarize_origin_grid_mixture(origins, prior_weights, baseline):
    """Keep original origin masses and bound the uncomputed utility contribution.

    For each action the omitted terminal reward is in[0,posterior mass].
    Thus the error in max-minus-fixed is at most that mass,not twice it.
    Uncomputed predictive origins contribute adaptation in[0,prior mass].
    These bounds do not cover finite predictive or terminal quadrature.
    """
    computed = {r['origin_identity'] for r in origins}
    assert len(computed) == len(origins)
    rare_mass = float(sum(w for k, w in enumerate(prior_weights) if k not in computed))
    adaptation = sum(r['raw_adaptation_value'] for r in origins)
    bound = sum(r['weighted_omitted_terminal_identity_mass'] for r in origins)
    stop, debited, fixed = baseline
    lower, upper = max(0., adaptation-bound), adaptation+bound+rare_mass
    return {'raw_draw_power': origins[0]['raw_draw_power'], 'raw_adaptation_value': adaptation,
        'raw_net_reward_control_variate': adaptation+debited-stop,
        'raw_adaptation_identity_interval': [lower, upper],
        'raw_net_control_variate_identity_interval': [lower+debited-stop, upper+debited-stop],
        'verification_budget_opportunity_cost': stop-debited,
        'weighted_omitted_terminal_identity_mass': bound, 'rare_predictive_origin_mass': rare_mass,
        'identity_and_rare_origin_error_bound': bound+rare_mass,
        'quadrature_joint_probability_mass': sum(r['quadrature_joint_probability_mass'] for r in origins),
        'expected_identity_weights': np.sum([r['expected_identity_weights'] for r in origins], axis=0).tolist(),
        'joint_fantasy_count': sum(r['joint_fantasy_count'] for r in origins),
        'distinct_head_state_count': sum(r['distinct_head_state_count'] for r in origins),
        'selected_action_probabilities': np.sum([r['selected_action_probabilities'] for r in origins], axis=0).tolist(),
        'raw_fixed_action_tower_residual': sum(r['raw_fixed_action_tower_residual'] for r in origins),
        'computed_predictive_origins': sorted(computed)}


def summarize_proposal_head_grid(f_records, g_records, f_weights, g_weights, prior_weights,
        proposal_origin, predictive_origins, *, power, baseline, objective_power=None):
    """Integrate the original main-origin law using one existing proposal bank.

    The density ratio restores all requested predictive origins; posterior
    identity weights still use every source identity. No weights are fitted
    to the fixed-action expectation or normalized by their finite-grid sum.
    Omitted predictive origins retain their whole original-mass bound.
    """
    fs = np.asarray([r['head_log_scores'] for r in f_records])
    gs = np.asarray([r['head_log_scores'] for r in g_records]); prior = np.asarray(prior_weights)
    f = np.asarray([r['raw'][str(power if objective_power is None else objective_power)]['statistics'] for r in f_records])
    g = np.asarray([r['raw'][str(power)]['statistics'] for r in g_records])
    costs = np.asarray([r['raw'][str(power)]['costs'] for r in g_records])
    values, costs, posterior, _ = compose_head_grid(f, g, costs, fs, gs, prior)
    ids = np.asarray(predictive_origins)
    ratio = np.exp(logsumexp(np.log(prior[ids])+fs[:, None, ids]+gs[None, :, ids], axis=2)
        -fs[:, None, proposal_origin]-gs[None, :, proposal_origin])
    weights = np.outer(f_weights, g_weights)*ratio
    result = summarize_joint_actions(values, costs, posterior, weights, baseline, power,
        len(f_records)+len(g_records))
    if objective_power is not None:
        result['objective_raw_draw_power'] = objective_power
    utilities = f_records[0].get('utility_identities', list(range(len(prior))))
    assert all(r.get('utility_identities', list(range(len(prior)))) == utilities for r in f_records+g_records)
    omitted = sorted(set(range(len(prior)))-set(utilities))
    bound = float(np.sum(weights*posterior[:, :, omitted].sum(axis=2)))
    rare_mass = float(sum(w for k, w in enumerate(prior) if k not in predictive_origins))
    adaptation = result['raw_adaptation_value']; stop, debited, _ = baseline
    result.update(proposal_origin=proposal_origin, computed_predictive_origins=list(predictive_origins),
        weighted_omitted_terminal_identity_mass=bound, rare_predictive_origin_mass=rare_mass,
        identity_and_rare_origin_error_bound=bound+rare_mass,
        raw_net_control_variate_identity_interval=[max(0., adaptation-bound)+debited-stop,
            adaptation+bound+rare_mass+debited-stop],
        predictive_mass_residual=float(weights.sum()-prior[ids].sum()),
        predictive_density_ratio_range=[float(ratio.min()), float(ratio.max())])
    return result


def summarize_joint_actions(values, costs, posterior, weights, baseline, power, head_count):
    choices = np.lexsort((costs, -values), axis=2)[:, :, 0]
    best = np.take_along_axis(values, choices[:, :, None], axis=2)[:, :, 0]
    stop, debited, fixed = baseline; contrast = best-values[:, :, fixed]
    adaptation = float(np.sum(weights*contrast))
    return {'raw_draw_power': power, 'raw_adaptation_value': adaptation,
        'raw_net_reward_control_variate': adaptation+debited-stop,
        'verification_budget_opportunity_cost': stop-debited,
        'raw_fixed_action_tower_residual': float(np.sum(weights*values[:, :, fixed]))-debited,
        'raw_net_reward_uncorrected': float(np.sum(weights*best))-stop,
        'quadrature_joint_probability_mass': float(weights.sum()),
        'expected_identity_weights': np.einsum('fg,fgk->k', weights, posterior).tolist(),
        'joint_fantasy_count': int(weights.size), 'distinct_head_state_count': head_count,
        'selected_action_probabilities': [float(weights[choices == i].sum()) for i in range(values.shape[2])],
        'weighted_adaptive_contrasts': contrast.tolist(), 'joint_quadrature_weights': weights.tolist()}


def summarize_achievable_grid(values, costs, projected, projected_costs, posterior, weights, baseline,
        power, head_count):
    """Score the native recommendation under the raw law, including bad switches.

    The fixed-action tower control variate is valid for the raw posterior.
    Native moment-projected estimates select actions but never price their own
    achieved reward. In particular, a negative achieved contrast stays negative.
    """
    result = summarize_joint_actions(values, costs, posterior, weights, baseline, power, head_count)
    choices = np.lexsort((projected_costs, -projected), axis=2)[:, :, 0]
    achieved_values = np.take_along_axis(values, choices[:, :, None], axis=2)[:, :, 0]
    projected_values = np.take_along_axis(projected, choices[:, :, None], axis=2)[:, :, 0]
    stop, debited, fixed = baseline
    contrast = achieved_values-values[:, :, fixed]
    adaptation = float(np.sum(weights*contrast))
    result.update(raw_adaptation_of_projected_choice=adaptation,
        raw_net_reward_of_projected_choice_control_variate=adaptation+debited-stop,
        raw_net_reward_of_projected_choice_uncorrected=float(np.sum(weights*achieved_values))-stop,
        action_selection_loss=result['raw_adaptation_value']-adaptation,
        projected_value_bias_at_selected_action=float(np.sum(weights*(projected_values-achieved_values))),
        projected_action_indices=choices.tolist(),
        projected_selected_action_probabilities=[float(weights[choices == i].sum()) for i in range(values.shape[2])],
        weighted_achievable_contrasts=contrast.tolist())
    return result
