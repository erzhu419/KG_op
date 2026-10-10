"""Read saved raw-reference results and preserve the failed32-fantasy gate."""

import time

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    first, extended = [read(f'paper_artifacts/historical_conditional_raw_guarded_v{i}/results.json') for i in (1, 2)]
    anchors = read('paper_artifacts/historical_conditional_raw_guarded_v2/anchors.json')
    tests = read('paper_artifacts/historical_conditional_raw_guarded_v1/unit_tests.json')
    joint = read('paper_artifacts/historical_conditional_guarded_kg_extension_v1/results.json')['rows'][0]
    error = 0.
    for row in extended['rows']:
        old = next((r for r in first['rows'] if (r['draw_power'], r['fantasy_power']) ==
            (row['draw_power'], row['fantasy_power'])), None)
        if old is not None:
            error = max(error, abs(row['likelihood_expected_reward']-old['likelihood_expected_reward']))
    assert error == 0.
    last = extended['rows'][-1]
    result = {'all_registered_cases_complete': first['all_registered_cases_complete'] and extended['all_registered_cases_complete'],
        'initial32_fantasy_projection_resolution_passed': first['next_episode_projection_difference_resolved'],
        'next_episode_projection_difference_resolved': extended['next_episode_projection_difference_resolved'],
        'actual_field_anchors_equivalent': anchors['all_anchors_equivalent'],
        'maximum_actual_field_head_statistic_difference': max(r['maximum_absolute_head_statistic_difference'] for r in anchors['rows']),
        'completed32_prefix_reward_difference': error,
        'latest_fixed_action_comparison': last, 'all_reference_rows': extended['rows'],
        'projection_difference_in_budget_cost_units': last['projected_minus_likelihood_expectation']/joint['verification_budget_opportunity_cost'],
        'successful_tests': tests['tests_run'], 'all_unit_tests_successful': tests['successful'],
        'reused_head_fantasy_draw_records': first['completed_head_fantasies'],
        'new_head_fantasy_draw_records': extended['completed_head_fantasies']-first['completed_head_fantasies'],
        'cpu_seconds_recorded': first['cpu_seconds']+extended['cpu_seconds']+anchors['cpu_seconds']+tests['cpu_seconds'],
        'cached_replay_cpu_seconds': time.process_time()-started,
        'server_jobs': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
        'all_source_models_and_weights_retained': extended['all_source_models_and_weights_retained'],
        'adaptive_raw_likelihood_KG_recomputed': False, 'independent_confirmation': False,
        'limitations': 'The numerical diagnostic resolves a paired fixed-action difference for the next episode '
            'only;two earlier paid updates remain projected. Nested differences are not rigorous error bounds. '
            'Fixed-action bias cannot be subtracted from adaptive KG without recomputing future decisions.',
        'next': 'Adapt historical likelihood_query_lookahead to the actual positive source mixtures using '
            'the checked conditional guarded integrator. Preserve all40 actions/shared identity updates; '
            'compare raw-optimal and projected-selected actions under the same unprojected next-episode '
            'utility before matched GP controls and independent decisions.'}
    save(ROOT/'paper_artifacts/historical_conditional_raw_guarded_v2/phase_results.json', result)
    print({k: result[k] for k in ('all_registered_cases_complete', 'next_episode_projection_difference_resolved',
        'actual_field_anchors_equivalent', 'maximum_actual_field_head_statistic_difference',
        'completed32_prefix_reward_difference', 'successful_tests', 'cpu_seconds_recorded', 'cached_replay_cpu_seconds')})
    print('paired projection difference', last['projected_minus_likelihood_expectation'],
        'draw/fantasy differences', last['nested_draw_reward_difference'], last['nested_fantasy_projection_difference'])
    return result


if __name__ == '__main__':
    run()
