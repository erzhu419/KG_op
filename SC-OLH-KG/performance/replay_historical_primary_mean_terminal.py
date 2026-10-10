"""Read refined action arrays; summarize numerical gates and preserved controls."""

import time

import numpy as np
from scipy.stats import qmc

from core.stratified_raw_lookahead import stratified_record_weights, summarize_stratified_futures
from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    saved = read('paper_artifacts/historical_primary_mean_terminal_v1/results.json')
    spec = read(saved['execution_plan']); assert saved['all_registered_futures_refined']
    bank = read('paper_artifacts/historical_primary_mean_terminal_v1/futures.json')
    old = read(spec['completed_futures']); tests = read('paper_artifacts/historical_primary_mean_terminal_v1/unit_tests.json')
    anchors = read('paper_artifacts/historical_primary_mean_terminal_v1/anchors.json')
    previous_diagnostics = read(spec['old_terminal_diagnostics'])
    complement = read(previous_diagnostics['execution_plan'])
    affected = [i for i, a in enumerate(bank['actions']) if a['kind'] == 'guarded' and a['primary'] == spec['candidate']]
    unaffected = [i for i in range(len(bank['actions'])) if i not in affected]
    before_by_coordinate = {(r['origin_identity'], r['identity_fantasy_position']): r for r in old['records']}
    for record in bank['records']:
        before = before_by_coordinate[record['origin_identity'], record['identity_fantasy_position']]
        for key in ('labels', 'shared_identity_weights', 'projected_values', 'projected_costs', 'projected_action_index'):
            assert record[key] == before[key]
        for power in spec['raw_draw_powers']:
            for key in ('values', 'costs'):
                np.testing.assert_array_equal(np.asarray(record['raw'][str(power)][key])[unaffected],
                    np.asarray(before['raw'][str(power)][key])[unaffected])
        if not record.get('primary_mean_refined'):
            assert record == before
    source_weights = np.asarray(bank['source_weights'])
    np.testing.assert_array_equal(source_weights, old['source_weights'])
    uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['cached_predictive_power'])
    terminal_rows = read(spec['terminal_results'])['rows']
    before, after = [next(r for r in terminal_rows if (r['layout'], r['remaining_calls'], r['power']) ==
        (spec['layout'], budget, spec['projected_draw_power'])) for budget in (360, 359)]
    baseline = before['best_reward'], after['best_reward'], after['best_action_index']
    rare_mass = float(sum(source_weights[i] for i in range(len(source_weights)) if i not in spec['main_origins']))
    complementary_rows = []
    for prefix in complement['complement_prefix_counts']:
        records, weights, regions = stratified_record_weights(bank['records'], source_weights, uniforms,
            spec['main_origins'], spec['low_constraint_probability'], max(spec['tail_counts']), prefix)
        rows = summarize_stratified_futures(records, weights, regions, bank['actions'], baseline, spec['raw_draw_powers'], rare_mass)
        for row in rows:
            row['complement_prefix_count'] = prefix
            preceding = next((r for r in reversed(complementary_rows) if r['raw_draw_power'] == row['raw_draw_power']), None)
            if preceding:
                row['nested_complement_adaptation_difference'] = abs(row['raw_adaptation_value']-preceding['raw_adaptation_value'])
            if prefix == spec['complement_prefix_count']:
                original = next(r for r in saved['rows'] if r['tail_fantasies_per_main_origin'] == max(spec['tail_counts'])
                    and r['raw_draw_power'] == row['raw_draw_power'])
                assert abs(row['raw_adaptation_value']-original['raw_adaptation_value']) < 1e-14
        complementary_rows.extend(rows)
    result = {**saved, 'all37_unaffected_action_values_and_costs_preserved': True,
        'all_frozen_projected_choices_labels_and_identity_weights_preserved': True,
        'old_rare_origin_records_preserved': True,
        'maximum_actual_source_branch_anchor_error': max(r['maximum_branch_error'] for r in anchors['rows']),
        'old_weighted_absolute_terminal_contrast_difference': previous_diagnostics['rows'][-1]['weighted_absolute_terminal_contrast_difference'],
        'refined_complementary_rows': complementary_rows,
        'refined_complementary_region_gate_passed': all(r['nested_complement_adaptation_difference'] <=
            complement['maximum_complement_adaptation_difference'] for r in complementary_rows if 'nested_complement_adaptation_difference' in r),
        'successful_tests': tests['tests_run'], 'all_unit_tests_successful': tests['successful'],
        'cpu_seconds_including_tests': saved['cpu_seconds']+tests['cpu_seconds'],
        'cached_replay_cpu_seconds': time.process_time()-started,
        'next': 'Connect matched GP controls under the same observed information,initial design,search budget '
            'and frozen verifier. The refined single-query numerical diagnostics do not establish independent '
            'source-prior superiority or a complete query ranking.'}
    save(ROOT/'paper_artifacts/historical_primary_mean_terminal_v1/phase_results.json', result)
    print({k: result[k] for k in ('all_registered_futures_refined', 'absolute_terminal_contrast_gate_passed',
        'nested_tail_gate_passed', 'refined_complementary_region_gate_passed', 'identity_remainder_gate_passed',
        'successful_tests', 'cpu_seconds_including_tests')})
    last = result['rows'][-1]
    print({k: last[k] for k in ('raw_net_reward_control_variate', 'weighted_absolute_terminal_contrast_difference',
        'nested_tail_adaptation_difference', 'action_selection_loss', 'identity_and_rare_origin_error_bound')})
    return result


if __name__ == '__main__':
    run()
