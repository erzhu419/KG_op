"""Reuse historical CDF labels for a probability-weighted joint tail refinement."""

import signal
import time

import numpy as np
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_raw_lookahead import evaluate_raw_futures
from core.stratified_raw_lookahead import stratified_record_weights, summarize_stratified_futures
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_positive_raw_stratified_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; checkpoint_path = out/'futures.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_rules_complete'):
        return previous
    started = time.process_time(); rows = []; stopped = False
    parent = read(spec['completed_prefix_results']); old_plan = read(parent['execution_plan'])
    assert parent['all_registered_cases_complete'] and parent['identity_remainder_gate_passed']
    for key in ('connection_plan', 'terminal_results', 'projected_heads', 'layout', 'candidate',
            'seed', 'raw_draw_powers', 'projected_draw_power', 'utility_identities'):
        assert spec[key] == old_plan[key]
    old = read(spec['completed_prefix'])
    records = read(str(checkpoint_path.relative_to(ROOT)))['records'] if checkpoint_path.exists() else old['records']
    _, _, models = build_connection(spec['connection_plan'])
    state, model, old_terminal, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
    terminal = ConditionalTerminalActions(old_terminal.shortlists, old_terminal.incumbents, old_terminal.protocol)
    bank = read(spec['projected_heads']); bank_plan = read(bank['execution_plan'])
    assert (bank['layout'], bank['action']['primary'], bank_plan['seed']) == (spec['layout'], spec['candidate'], spec['seed'])
    np.testing.assert_array_equal(model.weights, bank['identity_weights'])
    np.testing.assert_array_equal(model.weights, old['source_weights'])
    assert max(h['maximum_inverse_cdf_error'] for h in bank['heads']) <= spec['maximum_inverse_cdf_error']
    uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['cached_predictive_power'])
    low = np.flatnonzero(uniforms[:, 1] < spec['low_constraint_probability'])
    assert len(low) == max(spec['tail_counts'])
    coordinates = [(identity, int(p)) for identity in spec['main_origins'] for p in low]
    by_head = {(h['identity'], h['head']): h for h in bank['heads']}
    fantasies = [[by_head[identity, head]['predictive_labels'][p] for head in (0, 1)] for identity, p in coordinates]
    for r in old['records']:
        np.testing.assert_array_equal(r['labels'], [by_head[r['origin_identity'], h]['predictive_labels'][r['identity_fantasy_position']]
            for h in (0, 1)])
    before, after = [next(r for r in read(spec['terminal_results'])['rows'] if
        (r['layout'], r['remaining_calls'], r['power']) == (spec['layout'], budget, spec['projected_draw_power']))
        for budget in (remaining, remaining-1)]
    baseline = before['best_reward'], after['best_reward'], after['best_action_index']
    rare_mass = float(sum(model.weights[i] for i in range(len(model.components)) if i not in spec['main_origins']))
    def checkpoint(current, actions):
        nonlocal records
        records = current
        save(checkpoint_path, {'execution_plan': plan, 'source_weights': model.weights.tolist(),
            'actions': actions, 'records': records})
        print('stratified cached futures', len(records), 'CPU', round(time.process_time()-started, 2), flush=True)
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
    try:
        records, actions = evaluate_raw_futures(model, terminal, remaining, spec['candidate'], fantasies,
            raw_powers=spec['raw_draw_powers'], projected_power=spec['projected_draw_power'], seed=spec['seed'],
            completed_records=records, checkpoint=checkpoint, utility_identities=spec['utility_identities'],
            fantasy_coordinates=coordinates)
        assert actions == old['actions'] == after['actions']
        for count in spec['tail_counts']:
            selected, weights, regions = stratified_record_weights(records, model.weights, uniforms,
                spec['main_origins'], spec['low_constraint_probability'], count, spec['complement_prefix_count'])
            current = summarize_stratified_futures(selected, weights, regions, actions, baseline,
                spec['raw_draw_powers'], rare_mass)
            for row in current:
                row['tail_fantasies_per_main_origin'] = count
                row['complement_fantasies_per_main_origin'] = spec['complement_prefix_count']-len(low[low < spec['complement_prefix_count']])
                row['model_weight_martingale_residual'] = (np.asarray(row['expected_model_weights'])-model.weights).tolist()
                preceding = next((r for r in reversed(rows) if r['raw_draw_power'] == row['raw_draw_power']), None)
                if preceding:
                    row['nested_tail_adaptation_difference'] = abs(row['raw_adaptation_value']-preceding['raw_adaptation_value'])
            rows.extend(current)
        for row in rows[:len(spec['raw_draw_powers'])]:
            original = next(r for r in parent['rows'] if r['fantasies_per_identity'] == spec['complement_prefix_count']
                and r['raw_draw_power'] == row['raw_draw_power'])
            difference = abs(row['raw_adaptation_value']-original['raw_adaptation_value'])
            assert difference <= original['omitted_predictive_origin_mass']+1e-12
            row['old16_rule_reproduction_difference'] = difference
        assert model.paid_episodes == len(state['paid_observations'])
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'rows': rows,
        'all_registered_rules_complete': len(rows) == len(spec['tail_counts'])*len(spec['raw_draw_powers']),
        'computed_futures': len(records), 'reused_complete_futures': len(old['records']),
        'new_futures': len(records)-len(old['records']), 'cpu_budget_stopped': stopped,
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started,
        'identity_remainder_gate_passed': bool(rows) and all(r['identity_and_rare_origin_error_bound'] <=
            spec['maximum_identity_integration_error_bound'] for r in rows),
        'nested_tail_gate_passed': bool(rows) and rows[-1]['nested_tail_adaptation_difference'] <=
            spec['maximum_nested_fantasy_adaptation_difference'],
        'shared_identity_and_all_source_components_retained': True, 'source_weights_renormalized': False,
        'all40_actions_evaluated': bool(rows) and rows[-1]['terminal_action_count'] == 40,
        'cost': spec['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': 'The probability partition was chosen from development diagnostics. Complementary-region '
            'quadrature stays at14 nodes. Identity intervals do not bound terminal or dominant-origin predictive '
            'quadrature. Both earlier paid projections remain;no independent source-prior advantage is established.'}
    save(path, result); return result


if __name__ == '__main__':
    r = run()
    print({k: r[k] for k in ('all_registered_rules_complete', 'new_futures', 'identity_remainder_gate_passed',
        'nested_tail_gate_passed', 'cpu_seconds', 'cpu_budget_stopped')})
