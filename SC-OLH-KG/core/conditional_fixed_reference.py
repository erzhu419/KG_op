"""Historical fixed-action head factorization with the corrected guarded rule.

Only the fixed action factorizes. The adaptive maximum still uses joint paid
forecasts. Every conditional field uses the actual paid moment filter and
retains its positive source weights. Head checkpoints avoid repeating work.
"""

import copy

import numpy as np
from scipy.stats import qmc

from core.conditional_terminal_kg import (
    guarded_objective_statistics, guarded_risk_coordinates, guarded_risk_statistics)
from core.positive_mixture_lookahead import RawPredictiveCDF
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments


def fixed_guarded_projected_tower(model, action, query, *, terminal_power,
        fantasy_powers, seed, protocol, completed_heads=None, checkpoint=None,
        identity_fantasy_powers=None):
    assert action['kind'] == 'guarded'
    nodes = qmc.Sobol(6, scramble=True, seed=seed).random_base2(terminal_power)
    probabilities = qmc.Sobol(2, scramble=True, seed=seed+1).random_base2(max(fantasy_powers))
    heads = [] if completed_heads is None else list(completed_heads)
    for identity, pair in enumerate(model.components):
        for head, field in enumerate(pair):
            completed = next((r for r in heads if (r['identity'], r['head']) == (identity, head)), None)
            count = 2**(max(fantasy_powers) if identity_fantasy_powers is None else identity_fantasy_powers[identity])
            done = 0 if completed is None else len(completed['future_statistics'])
            if done >= count:
                continue
            def statistic(current):
                if head == 0:
                    return guarded_objective_statistics(current, action['shortlist'], action['incumbent_position'], nodes, protocol)
                coordinates = guarded_risk_coordinates(current, action['shortlist'], nodes)
                return guarded_risk_statistics(coordinates, action['incumbent_position'], action['budgets'], protocol)[0]
            before = statistic(field) if completed is None else np.asarray(completed['prior_statistics'])
            cdf = RawPredictiveCDF(field, query); labels = cdf.inverse(probabilities[done:count, head])
            reference, bound = cdf.adaptive(labels)
            inverse_error = float(np.max(abs(reference-probabilities[done:count, head])))+bound
            if completed is not None:
                inverse_error = max(inverse_error, completed['maximum_inverse_cdf_error'])
            assert inverse_error <= 1e-8
            moments, gain = raw_fantasy_moments(field, query, labels)
            values = []; error = field.max_integration_relative_error
            if completed is not None:
                error = max(error, completed['maximum_raw_integration_relative_error'])
            for result in moments:
                future = copy.deepcopy(field); apply_raw_moments(future, query, result, gain)
                values.append(statistic(future)); error = max(error, future.max_integration_relative_error)
                assert future.paid_episodes == field.paid_episodes+1
            heads = [r for r in heads if (r['identity'], r['head']) != (identity, head)]
            old_values = [] if completed is None else completed['future_statistics']
            old_labels = [] if completed is None else completed['predictive_labels']
            heads.append({'identity': identity, 'identity_name': model.names[identity], 'head': head,
                'prior_statistics': before.tolist(), 'future_statistics': old_values+np.asarray(values).tolist(),
                'predictive_labels': old_labels+labels.tolist(), 'maximum_inverse_cdf_error': inverse_error,
                'maximum_raw_integration_relative_error': error})
            if checkpoint:
                checkpoint(heads)
    prior = []; pairs = []
    for identity in range(len(model.components)):
        pair = [next(r for r in heads if (r['identity'], r['head']) == (identity, head)) for head in (0, 1)]
        prior.append(np.asarray(pair[0]['prior_statistics']) @ pair[1]['prior_statistics'])
        pairs.append([np.asarray(r['future_statistics']) for r in pair])
    before = float(model.weights @ prior); rows = []
    for power in fantasy_powers:
        count = 2**power
        rewards = [f[:count].mean(axis=0) @ g[:count].mean(axis=0) for f, g in pairs]
        after = float(model.weights @ rewards)
        row = {'candidate': query, 'action': action, 'terminal_power': terminal_power,
            'fantasy_power': power, 'fantasies_per_head_identity': count,
            'prior_fixed_reward': before, 'expected_future_fixed_reward': after,
            'projected_fixed_action_drift': after-before,
            'maximum_inverse_cdf_error': max(r['maximum_inverse_cdf_error'] for r in heads),
            'maximum_raw_integration_relative_error': max(r['maximum_raw_integration_relative_error'] for r in heads),
            'shared_identity_retained': True, 'adaptive_maximum_evaluated': False,
            'previous_paid_projection_retained': True}
        actual_counts = [min(count, len(f), len(g)) for f, g in pairs]
        row['identity_fantasies_per_head'] = actual_counts
        row['retained_identity_expectation_error_bound'] = float(sum(model.weights[i]
            for i, n in enumerate(actual_counts) if n < count))
        if rows:
            row['nested_expected_future_difference'] = abs(after-rows[-1]['expected_future_fixed_reward'])
        rows.append(row)
    return rows, heads
