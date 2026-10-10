"""Bound low-mass utility integration; extend the frozen joint fantasy prefix."""

import copy
import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_raw_terminal import raw_terminal_values
from core.positive_raw_lookahead import evaluate_raw_futures, summarize_raw_futures
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments

PLAN = 'performance/manifests/historical_positive_raw_adaptive_v2.json'


def validate_saved_utilities(model, terminal, remaining, spec, completed, actions):
    """Saved full arrays are the reference; never recompute their eight utilities."""
    started = time.process_time(); rows = []
    selected = [r for r in completed if r['origin_identity'] in spec['validation_origins']]
    labels = np.asarray([r['labels'] for r in selected])
    cache = [[raw_fantasy_moments(field, spec['candidate'], labels[:, head]) for head, field in enumerate(pair)]
        for pair in model.components]
    for position, record in enumerate(selected):
        future = copy.deepcopy(model)
        scores = [[apply_raw_moments(field, spec['candidate'], cache[i][head][0][position], cache[i][head][1])
            for head, field in enumerate(pair)] for i, pair in enumerate(future.components)]
        future._record_scores(spec['candidate'], scores, 1)
        np.testing.assert_allclose(future.weights, record['shared_identity_weights'], atol=1e-15, rtol=1e-12)
        for power in spec['raw_draw_powers']:
            lower, _, computed_actions, check = raw_terminal_values(model, future, terminal, remaining-1,
                spec['candidate'], record['labels'], scores, power=power, seed=spec['seed'],
                utility_identities=spec['utility_identities'])
            assert computed_actions == actions
            full = np.asarray(record['raw'][str(power)]['values']); delta = full-lower
            bound = check['omitted_posterior_utility_mass']
            assert delta.min() >= -1e-12 and delta.max() <= bound+1e-12
            assert bound > 0. and len(future.components) == len(model.components)
            rows.append({'origin_identity': record['origin_identity'],
                'identity_fantasy_position': record['identity_fantasy_position'], 'raw_draw_power': power,
                'maximum_full_minus_bounded_utility': float(delta.max()),
                'posterior_utility_mass_bound': bound,
                'raw_optimal_action_agrees': int(np.argmax(full)) == int(np.argmax(lower)),
                'all40_action_intervals_hold': True})
        print('bounded reference', position+1, '/', len(selected), 'CPU', round(time.process_time()-started, 2), flush=True)
    return {'rows': rows, 'all_saved_action_intervals_hold': bool(rows),
        'full_reference_recomputed': False, 'source_weights_renormalized': False,
        'cpu_seconds': time.process_time()-started}


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; checkpoint_path = out/'futures.json'; validation_path = out/'utility_validation.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    started = time.process_time(); stopped = False; rows = []
    parent = read(spec['completed_prefix_results']); old_plan = read(parent['execution_plan'])
    assert parent['all_registered_cases_complete']
    for key in ('connection_plan', 'terminal_results', 'projected_heads', 'layout', 'candidate',
            'raw_draw_powers', 'projected_draw_power', 'seed'):
        assert spec[key] == old_plan[key]
    saved = read(spec['completed_prefix']); old_count = 2**old_plan['fantasy_power']
    count = 2**spec['fantasy_power']; assert old_count < count
    _, _, models = build_connection(spec['connection_plan'])
    state, model, old_terminal, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
    terminal = ConditionalTerminalActions(old_terminal.shortlists, old_terminal.incumbents, old_terminal.protocol)
    bank = read(spec['projected_heads']); np.testing.assert_array_equal(model.weights, bank['identity_weights'])
    np.testing.assert_array_equal(model.weights, saved['source_weights'])
    fantasies = []
    for identity in range(len(model.components)):
        pair = [next(r for r in bank['heads'] if (r['identity'], r['head']) == (identity, head)) for head in (0, 1)]
        assert all(len(h['predictive_labels']) >= count for h in pair)
        fantasies.extend(zip(pair[0]['predictive_labels'][:count], pair[1]['predictive_labels'][:count]))
    for record in saved['records']:
        np.testing.assert_array_equal(record['labels'], fantasies[record['origin_identity']*count+record['identity_fantasy_position']])
    records = read(str(checkpoint_path.relative_to(ROOT)))['records'] if checkpoint_path.exists() else saved['records']
    before, after = [next(r for r in read(spec['terminal_results'])['rows'] if
        (r['layout'], r['remaining_calls'], r['power']) == (spec['layout'], budget, spec['projected_draw_power']))
        for budget in (remaining, remaining-1)]
    baseline = before['best_reward'], after['best_reward'], after['best_action_index']
    omitted_origins = [i for i in range(len(model.components)) if i not in spec['new_origin_identities']]
    validation = read(str(validation_path.relative_to(ROOT))) if validation_path.exists() else {}
    def checkpoint(current, actions):
        nonlocal records
        records = current
        save(checkpoint_path, {'execution_plan': plan, 'source_weights': model.weights.tolist(),
            'actions': actions, 'records': records})
        print('bounded extension futures', len(records), 'CPU', round(time.process_time()-started, 2), flush=True)
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
    try:
        if not validation:
            validation = validate_saved_utilities(model, terminal, remaining, spec, saved['records'], saved['actions'])
            save(validation_path, validation)
        assert validation['all_saved_action_intervals_hold']
        records, actions = evaluate_raw_futures(model, terminal, remaining, spec['candidate'], fantasies,
            raw_powers=spec['raw_draw_powers'], projected_power=spec['projected_draw_power'], seed=spec['seed'],
            completed_records=records, checkpoint=checkpoint, utility_identities=spec['utility_identities'],
            new_origin_identities=spec['new_origin_identities'])
        assert actions == saved['actions'] == after['actions']
        for prefix_count in (old_count, count//2, count):
            prefix = [{**r, 'position': r['origin_identity']*prefix_count+r['identity_fantasy_position']}
                for r in records if r['identity_fantasy_position'] < prefix_count]
            rows.extend(summarize_raw_futures(prefix, model.weights, prefix_count, actions, baseline,
                spec['raw_draw_powers'], omitted_origin_identities=omitted_origins))
        for a, b in zip(rows[:len(spec['raw_draw_powers'])], parent['rows']):
            assert abs(a['raw_adaptation_value']-b['raw_adaptation_value']) < 1e-14
        assert model.paid_episodes == len(state['paid_observations'])
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    bound_passed = bool(rows) and all(r['identity_integration_error_bound'] <= spec['maximum_identity_integration_error_bound'] for r in rows)
    result = {'execution_plan': plan, 'rows': rows, 'computed_futures': len(records),
        'reused_complete_futures': len(saved['records']), 'new_futures': len(records)-len(saved['records']),
        'all_registered_cases_complete': len(rows) == 3*len(spec['raw_draw_powers']),
        'identity_remainder_gate_passed': bound_passed,
        'nested_fantasy_gate_passed': bool(rows) and rows[-1]['nested_raw_adaptation_difference'] <= spec['maximum_nested_fantasy_adaptation_difference'],
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'shared_identity_and_all_source_components_retained': True,
        'projected_utilities_integrate_all_identities': True, 'source_weights_renormalized': False,
        'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': 'Identity intervals bound omitted source utility/origin contributions,not terminal or fantasy '
            'quadrature. The two earlier paid projections remain. This is one development query,not source-prior superiority.',
        'cost': spec['cost']}
    save(path, result); return result


if __name__ == '__main__':
    result = run()
    print({k: result[k] for k in ('all_registered_cases_complete', 'computed_futures',
        'identity_remainder_gate_passed', 'nested_fantasy_gate_passed', 'cpu_seconds', 'cpu_budget_stopped')})
