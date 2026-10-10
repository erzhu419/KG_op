"""Summarize saved conditional-terminal/KG work without numerical reruns."""

import time

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    terminal = read('paper_artifacts/historical_conditional_terminal_completion_v1/results.json')
    pilot = read('paper_artifacts/historical_conditional_kg_pilot_v1/results.json')
    tests = read('paper_artifacts/historical_conditional_terminal_v1/unit_tests.json')
    fine = [r for r in terminal['rows'] if r['power'] == 16]
    result = {'terminal_resolution_cases_completed': len(terminal['rows']),
        'all_terminal_resolution_cases_complete': terminal['all_registered_cases_complete'],
        'terminal_summary': [{'layout': r['layout'], 'remaining_calls': r['remaining_calls'],
            'best_action_index': r['best_action_index'], 'best_reward': r['best_reward'],
            '16384_to65536_best_reward_difference': r['nested_best_reward_difference'],
            '16384_to65536_maximum_action_difference': r['maximum_nested_action_difference'],
            'maximum_analytic_direct_reference_error': r['maximum_analytic_direct_reference_error'],
            'same_best_action': r['same_nested_best']} for r in fine],
        'adaptive_kg_pilot_complete': pilot['all_registered_cases_complete'],
        'adaptive_kg_pilot_rows': pilot['rows'],
        'unit_tests_passed': tests['tests_run'], 'all_unit_tests_successful': tests['successful'],
        'cpu_seconds_recorded': terminal['cpu_seconds'] + pilot['cpu_seconds'] + tests['cpu_seconds'],
        'cached_replay_cpu_seconds': time.process_time() - started,
        'server_jobs': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
        'decision_ranking_certified': False, 'independent_confirmation': False,
        'next': 'Reduce the now-measured terminal cost and resolve remaining guarded-action/fantasy '
            'and paid-filter drift at margins relevant to sampling;then matched GP and independent decisions.'}
    save(ROOT / 'paper_artifacts/historical_conditional_kg_pilot_v1/phase_results.json', result)
    print({key: result[key] for key in ('terminal_resolution_cases_completed',
        'all_terminal_resolution_cases_complete', 'adaptive_kg_pilot_complete',
        'unit_tests_passed', 'all_unit_tests_successful', 'cpu_seconds_recorded',
        'cached_replay_cpu_seconds', 'decision_ranking_certified')})
    return result


if __name__ == '__main__':
    run()
