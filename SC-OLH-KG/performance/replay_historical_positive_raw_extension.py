"""Summarize cached bounded-prefix arrays without scientific recomputation."""

import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save


def run():
    started = time.process_time()
    saved = read('paper_artifacts/historical_positive_raw_adaptive_v2/results.json')
    bank = read('paper_artifacts/historical_positive_raw_adaptive_v2/futures.json')
    validation = read('paper_artifacts/historical_positive_raw_adaptive_v2/utility_validation.json')
    tests = read('paper_artifacts/historical_positive_raw_adaptive_v2/unit_tests.json')
    spec = read(saved['execution_plan']); count = 2**spec['fantasy_power']
    old = read(spec['completed_prefix']); powers = spec['raw_draw_powers']
    main_old = [r for r in old['records'] if r['origin_identity'] in spec['new_origin_identities']]
    added = [r for r in bank['records'] if r['identity_fantasy_position'] >=
        len(old['records'])//len(bank['source_weights'])]
    changes = []; disagreement_mass = 0.; identity_error = 0.; exact_count = 0; contributions = []
    fixed = next(r for r in read(spec['terminal_results'])['rows'] if
        (r['layout'], r['remaining_calls'], r['power']) == (spec['layout'], 359, spec['projected_draw_power']))['best_action_index']
    for r in bank['records']:
        raw = [r['raw'][str(p)] for p in powers]
        values = [np.asarray(a['values']) for a in raw]
        weight = bank['source_weights'][r['origin_identity']]/count
        changes.append(weight*float(np.max(abs(values[-1]-values[0]))))
        selected = int(np.lexsort((raw[-1]['costs'], -values[-1]))[0])
        disagrees = selected != r['projected_action_index']
        exact_count += disagrees
        disagreement_mass += weight*disagrees
        identity_error += weight*raw[-1].get('omitted_posterior_utility_mass', 0.)
        if r['origin_identity'] in spec['new_origin_identities']:
            contributions.append({'origin_identity': r['origin_identity'],
                'identity_fantasy_position': r['identity_fantasy_position'], 'labels': r['labels'],
                'selected_action': bank['actions'][selected],
                'raw_adaptation_contrast': float(values[-1][selected]-values[-1][fixed]),
                'weighted_adaptation_contribution': weight*float(values[-1][selected]-values[-1][fixed])})
    result = {**saved, 'successful_tests': tests['tests_run'], 'all_unit_tests_successful': tests['successful'],
        'validated_action_intervals': 40*len(validation['rows']),
        'maximum_validation_action_error': max(r['maximum_full_minus_bounded_utility'] for r in validation['rows']),
        'maximum_validation_posterior_mass_bound': max(r['posterior_utility_mass_bound'] for r in validation['rows']),
        'weighted_maximum_action_resolution_change': sum(changes),
        'computed_raw_vs_projected_action_disagreement_count': exact_count,
        'covered_predictive_action_disagreement_mass': disagreement_mass,
        'weighted_omitted_terminal_identity_mass': identity_error,
        'largest_main_origin_adaptation_contributions': sorted(contributions,
            key=lambda r: r['weighted_adaptation_contribution'], reverse=True)[:4],
        'main_origin_selected_primaries': sorted(set(r['selected_action']['primary'] for r in contributions)),
        'old_full_main_origin_mean_future_cpu_seconds': float(np.mean([r['cpu_seconds'] for r in main_old])),
        'new_bounded_main_origin_mean_future_cpu_seconds': float(np.mean([r['cpu_seconds'] for r in added])),
        'all_old_future_payloads_reused': all({k: v for k, v in next(
            r for r in bank['records'] if (r['origin_identity'], r['identity_fantasy_position']) ==
            (a['origin_identity'], a['identity_fantasy_position'])).items() if k != 'position'} ==
            {k: v for k, v in a.items() if k != 'position'} for a in old['records']),
        'cached_replay_cpu_seconds': time.process_time()-started,
        'next': 'Resolve the remaining same-prefix predictive integration before matched GP and independent '
            'decisions. Use saved joint labels/action arrays and historical tail integration; preserve failed gates.'}
    save(ROOT/'paper_artifacts/historical_positive_raw_adaptive_v2/phase_results.json', result)
    print({k: result[k] for k in ('all_registered_cases_complete', 'computed_futures',
        'identity_remainder_gate_passed', 'nested_fantasy_gate_passed', 'cpu_seconds',
        'old_full_main_origin_mean_future_cpu_seconds', 'new_bounded_main_origin_mean_future_cpu_seconds')})
    for r in saved['rows']:
        print({k: r[k] for k in ('fantasies_per_identity', 'raw_draw_power', 'raw_adaptation_value',
            'raw_net_reward_control_variate', 'action_selection_loss', 'identity_integration_error_bound',
            'nested_raw_adaptation_difference')})
    return result


if __name__ == '__main__':
    run()
