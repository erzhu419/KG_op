"""Read saved efficiency/KG results;never regenerate forecasts or references."""

import time

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    first = read('paper_artifacts/historical_conditional_efficiency_v1/results.json')
    fast = read('paper_artifacts/historical_conditional_efficiency_v2/results.json')
    anchors = read('paper_artifacts/historical_conditional_efficiency_v2/anchors.json')
    kg = read('paper_artifacts/historical_conditional_kg_extension_v1/results.json')
    tests = [read(f'paper_artifacts/{name}/unit_tests.json') for name in (
        'historical_conditional_efficiency_v1', 'historical_conditional_efficiency_v2',
        'historical_conditional_kg_extension_v1')]
    timing = []
    for power in (12, 16):
        rows = [r for r in fast['rows'] if r['power'] == power]
        before = sum(r['reference_cpu_seconds'] for r in rows); after = sum(r['cpu_seconds'] for r in rows)
        timing.append({'terminal_power': power, 'reference_cpu_seconds': before,
            'optimized_cpu_seconds': after, 'speedup': before/after})
    result = {'all_terminal_cases_equivalent': fast['all_cases_equivalent'],
        'all_paid_anchors_equivalent': anchors['all_anchors_equivalent'],
        'maximum_terminal_reward_difference': max(r['maximum_reward_difference'] for r in fast['rows']),
        'maximum_paid_identity_certificate_difference': max(r['maximum_identity_certificate_difference'] for r in anchors['rows']),
        'timing': timing, 'first_batched_adaptive_timing_retained': first['rows'],
        'adaptive_fallbacks_in_paid_anchors': sum(d['adaptive_fallback_actions'] for r in anchors['rows']
            for d in r['certificate_diagnostics']),
        'kg_extension_complete': kg['all_registered_cases_complete'], 'kg_rows': kg['rows'],
        'successful_test_executions': sum(r['tests_run'] for r in tests),
        'all_unit_tests_successful': all(r['successful'] for r in tests),
        'cpu_seconds_recorded': first['cpu_seconds'] + anchors['cpu_seconds_this_execution']
            + kg['cpu_seconds'] + sum(r['cpu_seconds'] for r in tests),
        'cached_replay_cpu_seconds': time.process_time()-started,
        'server_jobs': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
        'decision_ranking_certified': False, 'independent_confirmation': False,
        'next': 'Prioritize layout5/query5,which originally had adaptive contrast. Reuse the '
            'historical fixed-action head factorization with the corrected guarded utility, '
            'then matched GP controls. Preserve the layout6 stopping-state negative result.'}
    save(ROOT / 'paper_artifacts/historical_conditional_kg_extension_v1/phase_results.json', result)
    print({key: result[key] for key in ('all_terminal_cases_equivalent', 'all_paid_anchors_equivalent',
        'maximum_terminal_reward_difference', 'kg_extension_complete', 'successful_test_executions',
        'all_unit_tests_successful', 'cpu_seconds_recorded', 'cached_replay_cpu_seconds')})
    return result


if __name__ == '__main__':
    run()
