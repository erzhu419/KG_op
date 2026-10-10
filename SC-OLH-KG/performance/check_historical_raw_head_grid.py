"""Reuse source/GP action caches to validate and time the joint head grid."""

import signal
import time

import numpy as np
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.raw_terminal_head_grid import evaluate_head_records, compose_head_grid, summarize_head_grid
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from performance.extend_historical_matched_gp import restore_saved_model

PLAN = 'performance/manifests/historical_raw_head_grid_v1.json'


def terminal_for(state):
    guarded = state['terminal_actions'][:8]
    return ConditionalTerminalActions([a['shortlist'] for a in guarded],
        [a['incumbent_position'] for a in guarded], state['terminal_protocol'])


def anchor_comparison(f, g, prior_weights, reference, power):
    values, costs, posterior, ratio = compose_head_grid(
        [f['raw'][str(power)]['statistics']], [g['raw'][str(power)]['statistics']],
        [g['raw'][str(power)]['costs']], [f['head_log_scores']], [g['head_log_scores']], prior_weights)
    old = reference['raw'][str(power)]
    mass = old.get('omitted_posterior_utility_mass', 0.)
    return {'power': power, 'maximum_reward_error': float(np.max(abs(values[0, 0]-old['values']))),
        'maximum_cost_error': float(np.max(abs(costs[0, 0]-old['costs']))),
        'maximum_shared_weight_error': float(np.max(abs(posterior[0, 0]-reference['shared_identity_weights']))),
        'cached_omitted_posterior_utility_mass': mass, 'joint_to_marginal_predictive_density_ratio': float(ratio[0, 0]),
        'reward_values': values[0, 0].tolist(), 'reference_reward_values': old['values']}


def grid_rules(heads, weights, terminal_rows, uniforms, spec):
    before, after = [next(r for r in terminal_rows if (r['remaining_calls'], r['power']) == (b, spec['raw_powers'][-1]))
        for b in (360, 359)]
    baseline = before['best_reward'], after['best_reward'], after['best_action_index']
    by_position = [{r['position']: r for r in h} for h in heads]; rows = []
    for count in spec['objective_prefixes']:
        f = [by_position[0][i] for i in range(count)]
        for prefix in spec['risk_complement_prefixes']:
            body = np.flatnonzero(uniforms[:prefix, 1] >= spec['low_constraint_probability'])
            for tail_count in spec['risk_tail_counts']:
                tail = np.flatnonzero(uniforms[:, 1] < spec['low_constraint_probability'])[:tail_count]
                positions = np.r_[tail, body]; g = [by_position[1][int(i)] for i in positions]
                gw = np.r_[np.full(tail_count, spec['low_constraint_probability']/tail_count),
                    np.full(len(body), (1.-spec['low_constraint_probability'])/len(body))]
                contrasts = []
                for power in spec['raw_powers']:
                    r = summarize_head_grid(f, g, np.full(count, 1./count), gw, weights, power=power, baseline=baseline)
                    contrasts.append(np.asarray(r.pop('weighted_adaptive_contrasts')))
                    quadrature = np.asarray(r.pop('joint_quadrature_weights'))
                    r.update(objective_prefix=count, risk_complement_prefix=prefix, risk_tail_count=tail_count)
                    if len(contrasts) == len(spec['raw_powers']):
                        r['weighted_absolute_terminal_contrast_difference'] = float(np.sum(quadrature*abs(contrasts[-1]-contrasts[0])))
                    rows.append(r)
    fine = rows[-1]
    keys = ['objective_prefix', 'risk_complement_prefix', 'risk_tail_count']
    changes = {'terminal_absolute_contrast': fine['weighted_absolute_terminal_contrast_difference']}
    for key in keys:
        levels = spec['objective_prefixes'] if key == 'objective_prefix' else spec['risk_complement_prefixes'] if key == 'risk_complement_prefix' else spec['risk_tail_counts']
        coordinates = {k: fine[k] for k in keys}; coordinates[key] = levels[-2]
        old = next(r for r in rows if all(r[k] == v for k, v in coordinates.items()) and r['raw_draw_power'] == fine['raw_draw_power'])
        changes[key] = abs(fine['raw_adaptation_value']-old['raw_adaptation_value'])
    return {'rows': rows, 'numerical_changes': changes,
        'numerical_gates_passed': max(changes.values()) <= spec['maximum_numerical_adaptation_difference']}


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    assert read(str((out/'unit_tests.json').relative_to(ROOT)))['successful']
    started = time.process_time(); stopped = False; anchors = []; summary = {}; head_counts = []
    bank = read(spec['gp_bank']); by_position = {r['identity_fantasy_position']: r for r in bank['records']}
    row = next(r for r in read(spec['gp_states'])['rows'] if (r['layout'], r['arm']) == (spec['layout'], 'source_common_task_rbf'))
    model = restore_saved_model(row)
    gp_plan = read(spec['gp_connection_plan']); states = read(read(gp_plan['connection_plan'])['development_states'])['rows']
    state = next(r for r in states if r['layout'] == spec['layout']); terminal = terminal_for(state)
    uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['cached_predictive_power'])
    positions = [np.union1d(np.arange(max(spec['objective_prefixes'])), spec['gp_anchors']),
        np.union1d(np.union1d(np.flatnonzero(uniforms[:max(spec['risk_complement_prefixes']), 1] >= spec['low_constraint_probability']),
            np.flatnonzero(uniforms[:, 1] < spec['low_constraint_probability'])[:max(spec['risk_tail_counts'])]), spec['gp_anchors'])]
    labels = [{p: r['labels'][head] for p, r in by_position.items()} for head in (0, 1)]
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        heads = []
        for head in (0, 1):
            hp = out/f'gp_head{head}.json'
            saved_path = hp if hp.exists() else ROOT/spec.get('completed_head_output', spec['output'])/f'gp_head{head}.json'
            saved = read(str(saved_path.relative_to(ROOT))) if saved_path.exists() else {}
            def checkpoint(records, actions):
                assert actions == bank['actions']
                save(hp, {'records': records, 'actions': actions, 'head': head})
            records = evaluate_head_records(model, terminal, 360, spec['candidate'], head, labels[head], positions[head],
                powers=spec['raw_powers'], seed=spec['seed'], completed=saved.get('records'), checkpoint=checkpoint)
            heads.append(records); head_counts.append(len(records))
            print('head grid cached', head, len(records), 'CPU', round(time.process_time()-started, 2), flush=True)
        indexes = [{r['position']: r for r in h} for h in heads]
        for position in spec['gp_anchors']:
            for power in spec['raw_powers']:
                anchors.append(dict(anchor_comparison(indexes[0][position], indexes[1][position], model.weights,
                    by_position[position], power), model='matched GP', position=position))
        # This anchor uses all real source identities/components,not a collapsed GP.
        source_path = out/'source_anchor_heads.json'
        cache_path = source_path if source_path.exists() else ROOT/spec.get('source_anchor_cache', str(source_path.relative_to(ROOT)))
        if cache_path.exists():
            source = read(str(cache_path.relative_to(ROOT))); source_weights = np.asarray(source['source_weights'])
            if cache_path != source_path:
                save(source_path, source)
        else:
            _, _, models = build_connection(spec['source_connection_plan'])
            signal.setitimer(signal.ITIMER_PROF, max(.001,
                spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
            source_state, source_model, _, _ = next(m for m in models if m[0]['layout'] == spec['layout'])
            source_bank = read(spec['source_bank']); target = spec['source_anchor']
            reference = next(r for r in source_bank['records'] if all(r[k] == v for k, v in target.items()))
            source_heads = [evaluate_head_records(source_model, terminal_for(source_state), 360, spec['candidate'], head,
                [reference['labels'][head]], [0], powers=spec['raw_powers'], seed=spec['seed'])[0] for head in (0, 1)]
            source = {'source_weights': source_model.weights.tolist(), 'heads': source_heads, 'reference': reference,
                'all8_complete_identities': len(source_model.components),
                'retained_positive_components': [[len(f.means), len(g.means)] for f, g in source_model.components]}
            save(source_path, source); source_weights = source_model.weights
        for power in spec['raw_powers']:
            anchors.append(dict(anchor_comparison(*source['heads'], source_weights, source['reference'], power), model='learned source mixture'))
        for a in anchors:
            tolerance = spec['maximum_anchor_reward_and_weight_error']+a['cached_omitted_posterior_utility_mass']
            a['passed'] = a['maximum_reward_error'] <= tolerance and a['maximum_shared_weight_error'] <= spec['maximum_anchor_reward_and_weight_error']
            a['cost_error_bound'] = spec['maximum_anchor_cost_error']+359*a['cached_omitted_posterior_utility_mass']
            a['passed'] &= a['maximum_cost_error'] <= a['cost_error_bound']
        save(out/'anchors.json', {'rows': anchors})
        assert all(a['passed'] for a in anchors)
        summary = grid_rules(heads, model.weights, row['terminal'], uniforms, spec)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, **summary, 'all_registered_cases_complete': bool(summary),
        'all_reference_anchors_passed': bool(anchors) and all(a['passed'] for a in anchors),
        'gp_cached_head_counts': head_counts, 'maximum_anchor_reward_error': max((a['maximum_reward_error'] for a in anchors), default=None),
        'maximum_anchor_shared_weight_error': max((a['maximum_shared_weight_error'] for a in anchors), default=None),
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started,
        'cpu_budget_stopped': stopped, 'cost': spec['cost'], 'independent_confirmation': False,
        'decision_ranking_certified': False, 'adaptive_maximum_inside_each_joint_pair': True,
        'limitations': ['One query grid pilot;complete query rankings and independent deployment comparison remain open.',
            'Grid resolutions are numerical diagnostics;two earlier paid projections are retained.']}
    save(path, result)
    print({k: result[k] for k in ('all_registered_cases_complete', 'all_reference_anchors_passed', 'cpu_seconds', 'cpu_budget_stopped')})
    if summary:
        print(summary['rows'][-1]['raw_net_reward_control_variate'], summary['numerical_changes'], summary['numerical_gates_passed'])
    return result


if __name__ == '__main__':
    run()
