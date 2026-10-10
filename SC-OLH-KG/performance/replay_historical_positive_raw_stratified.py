"""Cached replay of probability strata and retained historical action arrays."""

import time

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    saved = read('paper_artifacts/historical_positive_raw_stratified_v1/results.json')
    spec = read(saved['execution_plan'])
    bank = read('paper_artifacts/historical_positive_raw_stratified_v1/futures.json')
    old = read(spec['completed_prefix'])
    tests = read('paper_artifacts/historical_positive_raw_stratified_v1/unit_tests.json')
    complement = read('paper_artifacts/historical_stratified_complement_v1/results.json')
    by_coordinate = {(r['origin_identity'], r['identity_fantasy_position']): r for r in bank['records']}
    retained = all({k: v for k, v in by_coordinate[r['origin_identity'], r['identity_fantasy_position']].items() if k != 'position'} ==
        {k: v for k, v in r.items() if k != 'position'} for r in old['records'])
    assert retained
    result = {**saved, 'all_old_future_payloads_reused': retained,
        'successful_tests': tests['tests_run'], 'all_unit_tests_successful': tests['successful'],
        'cpu_seconds_including_tests': saved['cpu_seconds']+tests['cpu_seconds'],
        'cached_replay_cpu_seconds': time.process_time()-started,
        'complement_region_gate_passed': complement['complement_gate_passed'],
        'weighted_absolute_terminal_contrast_gate_passed': complement['weighted_absolute_terminal_contrast_gate_passed'],
        'complement_diagnostics': 'paper_artifacts/historical_stratified_complement_v1/results.json',
        'next': 'Resolve fixed guarded terminal integration,where the weighted absolute paired contrast remains '
            'above1e-4 despite a small signed mean. Reuse conditional Gaussian mean integration for the queried '
            'primary rather than the unrelated last policy;preserve joint covariance and verify the same law '
            'before matched GP and independent decisions.'}
    save(ROOT/'paper_artifacts/historical_positive_raw_stratified_v1/phase_results.json', result)
    print({k: result[k] for k in ('all_registered_rules_complete', 'new_futures', 'identity_remainder_gate_passed',
        'nested_tail_gate_passed', 'all_old_future_payloads_reused', 'successful_tests', 'cpu_seconds_including_tests')})
    for row in result['rows']:
        print({k: row[k] for k in ('tail_fantasies_per_main_origin', 'raw_draw_power', 'raw_net_reward_control_variate',
            'identity_and_rare_origin_error_bound', 'adaptation_contributions_by_region')},
            {k: row[k] for k in ('nested_tail_adaptation_difference', 'nested_terminal_raw_adaptation_difference') if k in row})
    return result


if __name__ == '__main__':
    run()
