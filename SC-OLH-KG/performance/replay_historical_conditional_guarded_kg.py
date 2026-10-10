"""Reconcile saved guarded KG contrasts with longer fixed-action references.

This reads cached summaries only. It never recomputes a future terminal or
overwrites the original reference/KG results. Nested differences are numerical
diagnostics, not confidence intervals or bounds on moment-projection bias.
"""

import time

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    references = [read(f'paper_artifacts/historical_conditional_guarded_reference_v{i}/results.json')
        for i in (1, 2)]
    kg = [read(f'paper_artifacts/historical_conditional_guarded_kg_{name}_v1/results.json')
        for name in ('pilot', 'extension')]
    tests = read('paper_artifacts/historical_conditional_guarded_reference_v2/unit_tests.json')
    source = read('paper_artifacts/historical_conditional_terminal_completion_v1/results.json')
    layout6 = read('paper_artifacts/historical_conditional_kg_extension_v1/results.json')
    reference_plan = read(references[-1]['execution_plan'])
    rows = []
    for saved in kg:
        plan = read(saved['execution_plan'])
        for key in ('connection_plan', 'terminal_results', 'layout', 'candidate', 'terminal_power', 'seed'):
            assert plan[key] == reference_plan[key]
        for joint in saved['rows']:
            for fixed in references[-1]['rows']:
                assert joint['candidate'] == fixed['candidate']
                stop = joint['stop_expected_reward']
                assert abs(stop-fixed['prior_fixed_reward']-joint['verification_budget_opportunity_cost']) < 1e-12
                after = fixed['expected_future_fixed_reward']
                rows.append({'layout': joint['layout'], 'candidate': joint['candidate'],
                    'joint_artifact': saved['execution_plan'], 'joint_fantasy_power': joint['joint_fantasy_power'],
                    'fixed_reference_fantasy_power': fixed['fantasy_power'],
                    'adaptation_value': joint['adaptation_value'],
                    'joint_nested_adaptation_difference': joint['nested_adaptation_difference'],
                    'fixed_action_expected_reward': after, 'projected_fixed_action_drift': after-fixed['prior_fixed_reward'],
                    'fixed_nested_expected_future_difference': fixed.get('nested_expected_future_difference'),
                    'retained_identity_expectation_error_bound': fixed['retained_identity_expectation_error_bound'],
                    'net_reward_of_paid_episode': joint['adaptation_value']+after-stop,
                    'net_reward_at_joint_half_prefix': joint['nested_prefix_adaptation_value']+after-stop,
                    'original32_reference_net_reward': joint['net_reward_of_paid_episode'],
                    'new_terminal_fantasies_for_reconciliation': 0})
    fine = [r for r in source['rows'] if r['layout'] == reference_plan['layout'] and r['power'] == 16]
    result = {'all_registered_rules_complete': all(r['all_registered_rules_complete'] for r in references),
        'all_registered_joint_cases_complete': all(r['all_registered_cases_complete'] for r in kg),
        'reference_results': references[-1]['rows'], 'joint_results': [r['rows'][0] for r in kg],
        'reconciled_rows': rows, 'original_reference_results': references[0]['rows'],
        'layout6_negative_result_retained': layout6['rows'],
        'terminal_precision': [{'remaining_calls': r['remaining_calls'],
            '16384_to65536_best_reward_difference': r['nested_best_reward_difference'],
            '16384_to65536_maximum_action_difference': r['maximum_nested_action_difference']} for r in fine],
        'all_source_models_and_weights_retained': True,
        'retained_identity_expectation_error_bound': references[-1]['retained_identity_expectation_error_bound'],
        'fixed_action_density_cancellation_used_only_before_adaptive_maximum': True,
        'all_unit_tests_successful': tests['successful'], 'successful_tests': tests['tests_run'],
        'cpu_seconds_recorded': sum(r['cpu_seconds'] for r in references+kg)+tests['cpu_seconds'],
        'cached_replay_cpu_seconds': time.process_time()-started,
        'server_jobs': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
        'decision_ranking_certified': False, 'independent_confirmation': False,
        'limitations': 'Nested fantasy/terminal differences do not bound all numerical error. '
            'Exact-likelihood and projected-filter drift remain unseparated; this one-query '
            'development result does not establish learned-prior advantage over matched GP controls.',
        'next': 'Resolve remaining fixed-reference fantasy/projection error and joint adaptive '
            'quadrature at the observed decision margin,then matched GP and independent decisions.'}
    save(ROOT/'paper_artifacts/historical_conditional_guarded_kg_extension_v1/phase_results.json', result)
    print({k: result[k] for k in ('all_registered_rules_complete', 'all_registered_joint_cases_complete',
        'successful_tests', 'all_unit_tests_successful', 'cpu_seconds_recorded', 'cached_replay_cpu_seconds')})
    if rows:
        print('last saved reconciliation', rows[-1])
    return result


if __name__ == '__main__':
    run()
