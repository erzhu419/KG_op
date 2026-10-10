"""Read cached action arrays;never repeat paid filters or terminal evaluation."""

import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    saved = read('paper_artifacts/historical_positive_raw_adaptive_v1/results.json')
    bank = read('paper_artifacts/historical_positive_raw_adaptive_v1/futures.json')
    tests = read('paper_artifacts/historical_positive_raw_adaptive_v1/unit_tests.json')
    plan = read(saved['execution_plan']); powers = plan['raw_draw_powers']; count = 2**plan['fantasy_power']
    differences = []; disagreements = []
    for record in sorted(bank['records'], key=lambda r: r['position']):
        first, last = [np.asarray(record['raw'][str(power)]['values']) for power in (powers[0], powers[-1])]
        differences.append(float(np.max(abs(first-last))))
        choice = [int(np.lexsort((record['raw'][str(power)]['costs'], -np.asarray(record['raw'][str(power)]['values'])))[0])
            for power in (powers[0], powers[-1])]
        disagreements.append(choice[0] != choice[1])
    weights = np.repeat(np.asarray(bank['source_weights'])/count, count)
    selected = saved['rows'][-1]['raw_selected_action_indices']
    projected = [r['projected_action_index'] for r in sorted(bank['records'], key=lambda r: r['position'])]
    result = {'all_registered_cases_complete': saved['all_registered_cases_complete'],
        'all40_actions_evaluated': saved['all40_actions_evaluated'],
        'shared_identity_and_all_source_components_retained': saved['shared_identity_and_all_source_components_retained'],
        'adaptive_comparison_rows': saved['rows'],
        'weighted_maximum_action_resolution_change': float(weights @ differences),
        'maximum_future_action_resolution_change': max(differences),
        'weighted_terminal_resolution_action_disagreement': float(weights @ disagreements),
        'raw_vs_projected_action_disagreement_count': sum(a != b for a, b in zip(selected, projected)),
        'successful_tests': tests['tests_run'], 'all_unit_tests_successful': tests['successful'],
        'cpu_seconds_recorded': saved['cpu_seconds']+tests['cpu_seconds'],
        'cached_replay_cpu_seconds': time.process_time()-started,
        'server_jobs': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
        'previous_paid_projection_retained': True, 'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': 'The next raw posterior is evaluated without projection,but the two earlier paid updates '
            'remain projected. Four fantasies per identity and nested terminal rules provide development '
            'diagnostics,not independent benefit or rigorous numerical error bounds.',
        'next': 'Reuse the saved per-action future arrays for a same-prefix fantasy extension;resolve remaining '
            'quadrature at the measured decision margin before matched GP and independent decisions.'}
    save(ROOT/'paper_artifacts/historical_positive_raw_adaptive_v1/phase_results.json', result)
    print({k: result[k] for k in ('all_registered_cases_complete', 'all40_actions_evaluated',
        'weighted_maximum_action_resolution_change', 'weighted_terminal_resolution_action_disagreement',
        'successful_tests', 'cpu_seconds_recorded', 'cached_replay_cpu_seconds')})
    last = saved['rows'][-1]
    print({k: last[k] for k in ('raw_adaptation_value', 'raw_net_reward_control_variate',
        'raw_net_reward_of_projected_choice_control_variate', 'action_selection_loss',
        'projected_value_bias_at_selected_action', 'primary_disagreement_probability',
        'nested_raw_adaptation_difference', 'nested_terminal_raw_adaptation_difference')})
    return result


if __name__ == '__main__':
    run()
