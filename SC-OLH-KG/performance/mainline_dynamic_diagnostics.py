"""Post-verification diagnostics for publicly generated, shared candidates.

Only caller-supplied truth arrays enter scoring. This module neither observes
the simulator nor selects actions; every scored recommendation is frozen.
"""

import numpy as np

from core.guarded_deployment_kg import conditional_safety_power


def _mse(snapshot, truth, indices):
    if not indices:
        return None
    error = np.asarray(snapshot['mean'])[indices] - truth[indices]
    return np.mean(error**2, axis=0).tolist()


def _prediction_change(before, after, truth, indices):
    indices = sorted(indices)
    old, new = _mse(before, truth, indices), _mse(after, truth, indices)
    return {'indices': indices, 'point_count': len(indices),
            'mse_before': old, 'mse_after': new,
            'mse_change': None if not indices else (np.asarray(new)-old).tolist()}


def _conditional_reward(action, truth, feasible, protocol):
    index = action['primary']
    quality = np.clip(1-truth[index, 0]/protocol['objective_reference'], 0, 1)
    power = conditional_safety_power(truth[index, 2], np.exp(truth[index, 3]/2),
        action['budgets'][0], protocol['chance_alpha'], protocol['familywise_delta'])
    return float(feasible[index]*quality*power)


def dynamic_diagnostics(frozen, cases, truths, results, protocol):
    """Score generated-only transfer using frozen predictions and decisions.

``round_end_indices`` are exclusive global-index stops; all newly generated
policies follow the unchanged original-bank prefix. Per-round prediction
changes use all generated points still unsampled after that round, including
unreleased points. The separate second-round entry measures first-round
transfer before observing any second-round label. Neither terminal marginal
intervention is treated as an independently verified optimizer.
"""
    dynamic = frozen['dynamic_candidates']
    original, ends = dynamic['original_count'], dynamic['round_end_indices']
    generated = set(range(original, len(frozen['bank'])))
    initial_predictions, round_predictions, final_predictions = [], [], []
    unreleased_predictions, interventions = [], []
    for case in cases:
        task, arm = case['task_index'], case['arm']
        truth, feasible = truths[task]
        truth, feasible = np.asarray(truth), np.asarray(feasible)
        common = {'task_index': task, 'arm': arm}
        observed = set(frozen['plan']['initialization_indices'])
        initial_indices = sorted(generated-observed)
        initial_predictions.append({**common, 'indices': initial_indices,
            'point_count': len(initial_indices), 'mse': _mse(case['initial'], truth, initial_indices)})
        for position, trace in enumerate(case['rounds']):
            if position == 1:
                indices = set(range(ends[0], ends[1]))-observed
                unreleased_predictions.append({**common,
                    'from_phase': 'after_initialization', 'to_phase': 'before_round2_observation',
                    **_prediction_change(case['initial'], trace['before'], truth, indices)})
            observed.add(trace['decision']['query'])
            unseen_generated = generated-observed
            released_new = {i for i in unseen_generated if i < ends[position]}
            round_predictions.append({**common, 'round': trace['round'],
                'prediction_scope': 'all_generated_unsampled_including_unreleased',
                'admitted_generated_count': ends[position]-original,
                'admitted_unobserved_generated_count': len(released_new),
                'released_new_prediction_change': _prediction_change(
                    trace['before'], trace['after'], truth, released_new),
                **_prediction_change(trace['before'], trace['after'], truth, unseen_generated)})
            full = trace['terminal']['action']
            held = trace['terminal_with_new_unsampled_marginals_held_fixed']['action']
            if full['budgets'] != held['budgets']:
                raise ValueError('Generated-marginal intervention must use the same terminal budget')
            full_reward = _conditional_reward(full, truth, feasible, protocol)
            held_reward = _conditional_reward(held, truth, feasible, protocol)
            interventions.append({**common, 'round': trace['round'],
                'full_primary': full['primary'], 'held_new_unsampled_primary': held['primary'],
                'full_primary_is_generated_unobserved': full['primary'] in unseen_generated,
                'recommendation_changed': full['primary'] != held['primary'],
                'verification_budget': full['budgets'][0],
                'full_conditional_safe_reward': full_reward,
                'held_new_unsampled_conditional_safe_reward': held_reward,
                'conditional_safe_reward_difference': full_reward-held_reward})
        final_predictions.append({**common, **_prediction_change(
            case['initial'], case['rounds'][-1]['after'], truth, generated-observed)})
    recommendations, grouped = [], {}
    for row in results:
        primary = row['action']['primary']
        new, unseen = primary in generated, primary not in row['observed_indices']
        certified = bool(row['verification']['certified'])
        recommendations.append({k: row[k] for k in ('task_index', 'arm', 'policy')} | {
            'primary': primary, 'is_generated': new, 'is_unobserved': unseen,
            'is_generated_unobserved': new and unseen, 'certified': certified,
            'actual_safe_reward': float(row['scores']['actual_safe_reward']),
            'conditional_safe_reward': float(row['scores']['conditional_safe_reward']),
            'true_feasible': bool(row['scores']['true_feasible'])})
        group = grouped.setdefault((row['arm'], row['policy']), {
            'arm': row['arm'], 'policy': row['policy'], 'recommendations': 0,
            'generated_recommendations': 0, 'generated_unobserved_recommendations': 0,
            'certified_generated_recommendations': 0})
        group['recommendations'] += 1
        group['generated_recommendations'] += int(new)
        group['generated_unobserved_recommendations'] += int(new and unseen)
        group['certified_generated_recommendations'] += int(certified and new)
    ceilings = []
    for task, (truth, feasible) in truths.items():
        quality = np.clip(1-np.asarray(truth)[:, 0]/protocol['objective_reference'], 0, 1)
        for name, indices in [('original', range(original)), ('generated', sorted(generated)),
                              ('union', range(len(frozen['bank'])))]:
            valid = [i for i in indices if feasible[i]]
            best = max(valid, key=lambda i: quality[i]) if valid else None
            ceilings.append({'task_index': task, 'pool': name, 'feasible_count': len(valid),
                             'best_index': best, 'best_quality': float(quality[best]) if valid else 0.0})
    return {'head_order': ['f', 'eta_f', 'g', 'eta_g'], 'post_verification_bank_ceilings': ceilings,
        'counts': {'original_candidates': original, 'generated_candidates': len(generated),
            'generated_per_round': np.diff([original, *ends]).tolist(), 'cases': len(cases),
            'final_recommendations': len(recommendations),
            'newonly_intervention_recommendation_changes': sum(r['recommendation_changed'] for r in interventions)},
        'initial_generated_predictions': initial_predictions,
        'round_prediction_changes': round_predictions,
        'initial_to_final_prediction_changes': final_predictions,
        'round2_unreleased_prediction_changes': unreleased_predictions,
        'final_recommendations': recommendations, 'final_recommendation_counts': list(grouped.values()),
        'newonly_terminal_interventions': interventions,
        'interpretation': 'Fixed-point prediction comparisons exclude observed generated policies. '
            'Terminal marginal interventions isolate this observation at the same budget; '
            'conditional rewards are post-freeze diagnostics, not additional verified arms.'}
