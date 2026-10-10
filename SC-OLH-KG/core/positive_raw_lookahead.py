"""Reuse historical raw-optimal/projected-choice comparisons for mixtures."""

import copy
import time

import numpy as np

from core.positive_raw_terminal import raw_terminal_values
from core.raw_likelihood_lookahead import _actions
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments


def evaluate_raw_futures(model, terminal, remaining, query, fantasies, *, raw_powers,
        projected_power, seed, completed_records=None, checkpoint=None,
        utility_identities=None, new_origin_identities=None, fantasy_coordinates=None,
        replicates=1, eta_labels=None, projected_utility_identities=None, guarded_mean_position=None):
    """Persist action values using stable origin/predictive-prefix coordinates.

    Explicit coordinates let probability strata request only their new cached
    labels. The same paid filter and full projected terminal rule are used.
    """
    if replicates < 1 or int(replicates) != replicates:
        raise ValueError('replicates must be a positive integer')
    if (replicates == 1) != (eta_labels is None):
        raise ValueError('one episode has no eta labels; batches require them')
    assert all(r.get('replicates',1) == replicates for r in (completed_records or []))
    assert all(r.get('guarded_mean_position') == guarded_mean_position for r in (completed_records or []))
    if fantasy_coordinates is None:
        count = len(fantasies)//len(model.components)
        coordinates = [(p//count, p%count) for p in range(len(fantasies))]
    else:
        coordinates = [tuple(c) for c in fantasy_coordinates]
        assert len(coordinates) == len(fantasies) and len(set(coordinates)) == len(coordinates)
        count = 1+max([c[1] for c in coordinates]+[r['identity_fantasy_position'] for r in (completed_records or [])])
    records = [] if completed_records is None else [{**r,
        'position': r['origin_identity']*count+r['identity_fantasy_position']} for r in completed_records]
    actions = _actions(terminal, remaining-replicates)
    completed = {(r['origin_identity'], r['identity_fantasy_position']) for r in records}
    pending_positions = [p for p, (identity, index) in enumerate(coordinates)
        if (identity, index) not in completed and (new_origin_identities is None or identity in new_origin_identities)]
    pending = {coordinates[p]: sample for sample, p in enumerate(pending_positions)}
    if not pending:
        return records, actions
    cache = [[raw_fantasy_moments(field, query, np.asarray(fantasies)[pending_positions, head],
        None if eta_labels is None else np.asarray(eta_labels)[pending_positions,head],replicates)
        for head, field in enumerate(pair)]
        for pair in model.components]
    for position, (coordinate, labels) in enumerate(zip(coordinates, fantasies)):
        if coordinate not in pending:
            continue
        started = time.process_time(); future = copy.deepcopy(model); scores = []
        for identity, pair in enumerate(future.components):
            scores.append([apply_raw_moments(field, query, cache[identity][head][0][pending[coordinate]],
                cache[identity][head][1],replicates=replicates)
                for head, field in enumerate(pair)])
        future._record_scores(query, scores, replicates)
        assert future.paid_episodes == model.paid_episodes+replicates
        projected, projected_cost, actions = future.terminal_values(terminal, remaining-replicates, projected_power, seed,
            utility_identities=projected_utility_identities)
        selected = int(np.lexsort((projected_cost, -projected))[0]); raw = {}
        for power in raw_powers:
            values, costs, raw_actions, diagnostics = raw_terminal_values(model, future, terminal, remaining-replicates,
                query, labels, scores, power=power, seed=seed, utility_identities=utility_identities,
                replicates=replicates,eta_labels=None if eta_labels is None else eta_labels[position],
                guarded_mean_position=guarded_mean_position)
            assert raw_actions == actions
            raw[str(power)] = {'values': values.tolist(), 'costs': costs.tolist(), **diagnostics}
        identity, index = coordinate
        records.append({'position': identity*count+index, 'origin_identity': identity, 'identity_fantasy_position': index,
            'replicates':replicates,'future_remaining_calls':remaining-replicates,
            'guarded_mean_position':guarded_mean_position,
            'omitted_projected_utility_mass':0. if projected_utility_identities is None else
                float(sum(w for i,w in enumerate(future.weights) if i not in projected_utility_identities)),
            'eta_labels':None if eta_labels is None else list(eta_labels[position]),
            'labels': list(labels), 'shared_identity_weights': future.weights.tolist(),
            'projected_values': projected.tolist(), 'projected_costs': projected_cost.tolist(),
            'projected_action_index': selected, 'raw': raw,
            'maximum_raw_integration_relative_error': max(f.max_integration_relative_error
                for pair in future.components for f in pair), 'cpu_seconds': time.process_time()-started})
        if checkpoint:
            checkpoint(records, actions)
    return records, actions


def summarize_raw_futures(records, source_weights, count, actions, baseline, raw_powers,
        *, omitted_origin_identities=None):
    """Fixed-action control variate uses the exact Bayesian tower identity.

    The next raw posterior is unprojected. Its fixed-action expectation equals
    that action's starting value at the debited budget. The starting field and
    its two earlier paid projections stay unchanged. Report the finite-fantasy
    tower residual separately;never factorize the adaptive maximum.
    """
    stop, debited, fixed = baseline; rows = []
    records = sorted(records, key=lambda r: r['position'])
    positions = [r['origin_identity']*count+r['identity_fantasy_position'] for r in records]
    assert positions == sorted(set(positions)) and all(0 <= r['identity_fantasy_position'] < count for r in records)
    missing = set(range(len(source_weights)*count))-set(positions)
    assert all(p//count in (omitted_origin_identities or []) for p in missing)
    weights = np.asarray([source_weights[r['origin_identity']]/count for r in records])
    missing_mass = float(sum(source_weights[p//count]/count for p in missing))
    projected = np.asarray([r['projected_values'] for r in records])
    choices = np.asarray([r['projected_action_index'] for r in records])
    index = np.arange(len(records)); expected_weights = weights @ np.asarray([r['shared_identity_weights'] for r in records])
    for power in raw_powers:
        values = np.asarray([r['raw'][str(power)]['values'] for r in records])
        costs = np.asarray([r['raw'][str(power)]['costs'] for r in records])
        selected = np.lexsort((costs, -values), axis=1)[:, 0]
        contrast = values[index, selected]-values[:, fixed]
        achieved = values[index, choices]-values[:, fixed]
        half = max(1, count//2); prefix = np.asarray([r['identity_fantasy_position'] < half for r in records])
        half_weights = weights[prefix]*count/half
        tail = np.asarray([r['raw'][str(power)].get('omitted_posterior_utility_mass', 0.) for r in records])
        terminal_bound = float(weights @ tail)
        half_terminal_bound = float(half_weights @ tail[prefix])
        half_missing_mass = float(sum(source_weights[p//count]/half for p in missing if p%count < half))
        adaptation = float(weights @ contrast); choice_adaptation = float(weights @ achieved)
        loss = float(weights @ (values[index, selected]-values[index, choices]))
        rows.append({'raw_draw_power': power, 'fantasies_per_identity': count, 'terminal_action_count': len(actions),
            'raw_adaptation_value': adaptation, 'raw_net_reward_control_variate': adaptation+debited-stop,
            'raw_net_reward_uncorrected': float(weights @ values[index, selected])-stop,
            'raw_fixed_action_tower_residual': float(weights @ values[:, fixed])-debited,
            'raw_adaptation_of_projected_choice': choice_adaptation,
            'raw_net_reward_of_projected_choice_control_variate': choice_adaptation+debited-stop,
            'action_selection_loss': loss,
            'projected_value_bias_at_selected_action': float(weights @ (projected[index, choices]-values[index, choices])),
            'projected_adaptation_value': float(weights @ (projected[index, choices]-projected[:, fixed])),
            'projected_raw_joint_expected_future_reward': float(weights @ projected[index, choices]),
            'primary_disagreement_probability': float(weights @ [actions[a]['primary'] != actions[b]['primary'] for a, b in zip(selected, choices)]),
            'nested_half_raw_adaptation_value': float(half_weights @ contrast[prefix]),
            'nested_raw_adaptation_difference': abs(adaptation-float(half_weights @ contrast[prefix])),
            'verification_budget_opportunity_cost': stop-debited,
            'predictive_log_evidence_max_error': max(r['raw'][str(power)]['predictive_log_evidence_max_error'] for r in records),
            'effective_sample_size_min': min(r['raw'][str(power)]['effective_sample_size_min'] for r in records),
            'maximum_raw_integration_relative_error': max(r['maximum_raw_integration_relative_error'] for r in records),
            'expected_model_weights': expected_weights.tolist(),
            'model_weight_martingale_residual': (expected_weights-source_weights).tolist(),
            'raw_selected_action_indices': selected.tolist(),
            'computed_joint_futures': len(records), 'omitted_joint_futures': len(missing),
            'omitted_predictive_origin_mass': missing_mass,
            'weighted_omitted_posterior_utility_mass': terminal_bound,
            'identity_integration_error_bound': terminal_bound+missing_mass,
            'raw_adaptation_identity_interval': [max(0., adaptation-terminal_bound), adaptation+terminal_bound+missing_mass],
            'raw_net_control_variate_identity_interval': [max(0., adaptation-terminal_bound)+debited-stop,
                adaptation+terminal_bound+missing_mass+debited-stop],
            'projected_choice_adaptation_identity_interval': [choice_adaptation-terminal_bound-missing_mass,
                choice_adaptation+terminal_bound+missing_mass],
            'action_selection_loss_identity_interval': [max(0., loss-terminal_bound), loss+terminal_bound+missing_mass],
            'nested_half_identity_integration_error_bound': half_terminal_bound+half_missing_mass})
        if missing:
            rows[-1]['covered_predictive_expected_model_weights'] = rows[-1].pop('expected_model_weights')
            rows[-1]['covered_model_weight_martingale_residual'] = rows[-1].pop('model_weight_martingale_residual')
        if len(rows) > 1:
            rows[-1]['nested_terminal_raw_adaptation_difference'] = abs(adaptation-rows[-2]['raw_adaptation_value'])
            rows[-1]['nested_terminal_projected_choice_adaptation_difference'] = abs(choice_adaptation-rows[-2]['raw_adaptation_of_projected_choice'])
    return rows
