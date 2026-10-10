"""Cached checks of complementary predictive mass and terminal cancellation."""

import time

import numpy as np
from scipy.stats import qmc

from core.stratified_raw_lookahead import stratified_record_weights, summarize_stratified_futures
from performance.complete_historical_source_heads import ROOT, read, save

PLAN = 'performance/manifests/historical_stratified_complement_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); saved = read(spec['stratified_results'])
    assert saved['all_registered_rules_complete']
    parent = read(saved['execution_plan']); bank = read(spec['stratified_futures'])
    uniforms = qmc.Sobol(2, scramble=True, seed=parent['seed']+1).random_base2(parent['cached_predictive_power'])
    terminal = read(parent['terminal_results'])['rows']
    before, after = [next(r for r in terminal if (r['layout'], r['remaining_calls'], r['power']) ==
        (parent['layout'], budget, parent['projected_draw_power'])) for budget in (360, 359)]
    baseline = before['best_reward'], after['best_reward'], after['best_action_index']
    weights_source = np.asarray(bank['source_weights'])
    rare_mass = float(sum(weights_source[i] for i in range(len(weights_source)) if i not in parent['main_origins']))
    rows = []
    for prefix in spec['complement_prefix_counts']:
        records, weights, regions = stratified_record_weights(bank['records'], weights_source, uniforms,
            parent['main_origins'], parent['low_constraint_probability'], spec['fixed_tail_count'], prefix)
        current = summarize_stratified_futures(records, weights, regions, bank['actions'], baseline,
            parent['raw_draw_powers'], rare_mass)
        contrast = []
        for power in parent['raw_draw_powers']:
            values = np.asarray([r['raw'][str(power)]['values'] for r in records])
            contrast.append(values.max(axis=1)-values[:, baseline[-1]])
        absolute_change = float(weights @ abs(contrast[-1]-contrast[0]))
        fixed_change = np.asarray([r['raw'][str(parent['raw_draw_powers'][-1])]['values'][baseline[-1]]-
            r['raw'][str(parent['raw_draw_powers'][0])]['values'][baseline[-1]] for r in records])
        attribution = sorted([{'origin_identity': r['origin_identity'],
            'identity_fantasy_position': r['identity_fantasy_position'], 'labels': r['labels'],
            'weighted_absolute_contrast_change': float(w*abs(delta)),
            'weighted_fixed_action_change': float(w*fixed_delta)}
            for r, w, delta, fixed_delta in zip(records, weights, contrast[-1]-contrast[0], fixed_change)],
            key=lambda r: r['weighted_absolute_contrast_change'], reverse=True)[:6]
        for row in current:
            row['complement_prefix_count'] = prefix
            row['weighted_absolute_terminal_contrast_difference'] = absolute_change
            row['weighted_absolute_fixed_action_resolution_change'] = float(weights @ abs(fixed_change))
            row['weighted_absolute_optimal_reward_resolution_change'] = float(weights @ abs(contrast[-1]-contrast[0]+fixed_change))
            row['largest_terminal_contrast_changes'] = attribution
            row['model_weight_martingale_residual'] = (np.asarray(row['expected_model_weights'])-weights_source).tolist()
            previous = next((r for r in reversed(rows) if r['raw_draw_power'] == row['raw_draw_power']), None)
            if previous:
                row['nested_complement_adaptation_difference'] = abs(row['raw_adaptation_value']-previous['raw_adaptation_value'])
            if prefix == parent['complement_prefix_count']:
                original = next(r for r in saved['rows'] if r['tail_fantasies_per_main_origin'] == spec['fixed_tail_count']
                    and r['raw_draw_power'] == row['raw_draw_power'])
                error = abs(original['raw_adaptation_value']-row['raw_adaptation_value'])
                assert error < 1e-14
                row['completed_rule_reproduction_error'] = error
        rows.extend(current)
    result = {'execution_plan': PLAN, 'rows': rows, 'all_cached_rules_complete': True,
        'complement_gate_passed': rows[-1]['nested_complement_adaptation_difference'] <= spec['maximum_complement_adaptation_difference'],
        'weighted_absolute_terminal_contrast_gate_passed': rows[-1]['weighted_absolute_terminal_contrast_difference'] <=
            spec['maximum_weighted_absolute_terminal_contrast_difference'],
        'new_future_evaluations': 0, 'cpu_seconds': time.process_time()-started,
        'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': 'Adjacent positive rules are numerical diagnostics,not rigorous predictive error bounds or '
            'independent decisions. This is one development query;earlier paid projections remain.'}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({k: result[k] for k in ('all_cached_rules_complete', 'complement_gate_passed',
        'weighted_absolute_terminal_contrast_gate_passed', 'new_future_evaluations', 'cpu_seconds')})
    print({k: rows[-1][k] for k in ('raw_net_reward_control_variate', 'nested_complement_adaptation_difference',
        'weighted_absolute_terminal_contrast_difference')})
    return result


if __name__ == '__main__':
    run()
