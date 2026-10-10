"""Use cached origin-conditional labels for the actual eight-identity raw KG."""

import copy
import signal
import time

import numpy as np
from scipy.stats import qmc

from core.raw_terminal_head_grid import evaluate_head_records, summarize_origin_head_grid, summarize_origin_grid_mixture
from performance.check_historical_raw_head_grid import terminal_for, anchor_comparison
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_source_head_grid_v1.json'


def baseline_for(spec):
    rows = read(spec['terminal_results'])['rows']
    before, after = [next(r for r in rows if (r['layout'], r['remaining_calls'], r['power']) ==
        (spec['layout'], b, spec['raw_powers'][-1])) for b in (360, 359)]
    return before['best_reward'], after['best_reward'], after['best_action_index']


def grid_rules(heads, prior_weights, baseline, spec):
    uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['cached_predictive_power'])
    rows = []; indices = {key: {r['position']: r for r in records} for key, records in heads.items()}
    for count in spec['objective_prefixes']:
        for prefix in spec['risk_complement_prefixes']:
            body = np.flatnonzero(uniforms[:prefix, 1] >= spec['low_constraint_probability'])
            for tail_count in spec['risk_tail_counts']:
                tail = np.flatnonzero(uniforms[:, 1] < spec['low_constraint_probability'])[:tail_count]
                positions = np.r_[tail, body]
                gw = np.r_[np.full(tail_count, spec['low_constraint_probability']/tail_count),
                    np.full(len(body), (1.-spec['low_constraint_probability'])/len(body))]
                contrasts = {}; absolute_difference = 0.
                for power in spec['raw_powers']:
                    origins = []
                    for origin in spec['predictive_origins']:
                        f = [indices[origin, 0][i] for i in range(count)]
                        g = [indices[origin, 1][int(i)] for i in positions]
                        r = summarize_origin_head_grid(f, g, np.full(count, 1./count), gw,
                            prior_weights, origin, power=power, baseline=baseline)
                        contrast = np.asarray(r.pop('weighted_adaptive_contrasts'))
                        quadrature = np.asarray(r.pop('joint_quadrature_weights'))
                        if origin in contrasts:
                            absolute_difference += float(np.sum(quadrature*abs(contrast-contrasts[origin])))
                        contrasts[origin] = contrast; origins.append(r)
                    row = summarize_origin_grid_mixture(origins, prior_weights, baseline)
                    row.update(objective_prefix=count, risk_complement_prefix=prefix, risk_tail_count=tail_count,
                        origin_contributions=origins)
                    if power == spec['raw_powers'][-1]:
                        row['weighted_absolute_terminal_contrast_difference'] = absolute_difference
                    rows.append(row)
    fine = rows[-1]; keys = ('objective_prefix', 'risk_complement_prefix', 'risk_tail_count')
    changes = {'terminal_absolute_contrast': fine['weighted_absolute_terminal_contrast_difference']}
    levels = (spec['objective_prefixes'], spec['risk_complement_prefixes'], spec['risk_tail_counts'])
    for key, values in zip(keys, levels):
        coordinates = {k: fine[k] for k in keys}; coordinates[key] = values[-2]
        old = next(r for r in rows if all(r[k] == v for k, v in coordinates.items()) and
            r['raw_draw_power'] == fine['raw_draw_power'])
        changes[key] = abs(fine['raw_adaptation_value']-old['raw_adaptation_value'])
    return {'rows': rows, 'numerical_changes': changes,
        'numerical_gates_passed': max(changes.values()) <= spec['maximum_numerical_adaptation_difference'],
        'identity_remainder_gate_passed': max(r['identity_and_rare_origin_error_bound'] for r in rows) <=
            spec['maximum_identity_integration_error_bound']}


def anchors_for(heads, prior_weights, spec):
    references = read(spec['raw_reference'])['records']; rows = []
    indices = {key: {r['position']: r for r in records} for key, records in heads.items()}
    for reference in references:
        origin, position = reference['origin_identity'], reference['identity_fantasy_position']
        if origin not in spec['predictive_origins'] or position not in (0, 7, 13, 47):
            continue
        if any(position not in indices[origin, head] for head in (0, 1)):
            continue
        f, g = [indices[origin, head][position] for head in (0, 1)]
        assert [f['label'], g['label']] == reference['labels']
        for power in spec['raw_powers']:
            row = anchor_comparison(f, g, prior_weights, reference, power)
            weights = np.asarray(reference['shared_identity_weights'])
            omitted = float(sum(w for k, w in enumerate(weights) if k not in spec['utility_identities']))
            row.update(origin_identity=origin, position=position, new_omitted_posterior_utility_mass=omitted)
            row['passed'] = (row['maximum_reward_error'] <= omitted+spec['maximum_anchor_reward_and_weight_error'] and
                row['maximum_shared_weight_error'] <= spec['maximum_anchor_reward_and_weight_error'] and
                row['maximum_cost_error'] <= 359*omitted+spec['maximum_anchor_cost_error'])
            rows.append(row)
    return {'rows': rows, 'all_anchors_passed': bool(rows) and all(r['passed'] for r in rows)}


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    assert read(str((out/'unit_tests.json').relative_to(ROOT)))['successful']
    bank = read(spec['predictive_heads']); parent = read(bank['execution_plan'])
    reference_actions = read(spec['raw_reference'])['actions']
    for key in ('connection_plan', 'layout', 'candidate', 'seed'):
        assert parent[key] == spec[key]
    started = time.process_time(); heads = {}; summary = {}; anchors = {}; stopped = False
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    model = None
    try:
        connection, _, models = build_connection(spec['connection_plan'])
        # The historical factory clears its own timer; restore this stage's remainder.
        signal.setitimer(signal.ITIMER_PROF, max(.001,
            spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        assert connection['all_registered_cases_complete']
        state, model, _, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
        assert remaining == 360 and len(model.components) == 8
        assert model.weights.tolist() == bank['identity_weights']
        terminal = terminal_for(state); baseline = baseline_for(spec)
        old = read(spec['source_anchor_heads']); assert old['source_weights'] == model.weights.tolist()
        uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['cached_predictive_power'])
        positions = [np.arange(max(spec['objective_prefixes'])), np.union1d(
            np.flatnonzero(uniforms[:max(spec['risk_complement_prefixes']), 1] >= spec['low_constraint_probability']),
            np.flatnonzero(uniforms[:, 1] < spec['low_constraint_probability'])[:max(spec['risk_tail_counts'])])]
        for origin in spec['predictive_origins']:
            for head in (0, 1):
                label_bank = next(r for r in bank['heads'] if (r['identity'], r['head']) == (origin, head))
                assert label_bank['identity_name'] == model.names[origin]
                labels = label_bank['predictive_labels']; assert len(labels) == 2**spec['cached_predictive_power']
                hp = out/f'origin{origin}_head{head}.json'
                saved_path = hp if hp.exists() else ROOT/spec.get('completed_head_output', spec['output'])/hp.name
                completed = read(str(saved_path.relative_to(ROOT)))['records'] if saved_path.exists() else []
                if origin == old['reference']['origin_identity'] and not completed:
                    anchor = copy.deepcopy(old['heads'][head]); position = old['reference']['identity_fantasy_position']
                    assert anchor['label'] == labels[position]
                    anchor['position'] = position; anchor['utility_identities'] = spec['utility_identities']
                    anchor['reused_complete_source_anchor'] = True
                    for raw in anchor['raw'].values():
                        for k in range(len(model.components)):
                            if k not in spec['utility_identities']:
                                raw['statistics'][k] = np.zeros((40, 4)).tolist(); raw['costs'][k] = [0.]*40
                    completed = [anchor]
                def checkpoint(records, actions):
                    assert actions == reference_actions
                    save(hp, {'origin_identity': origin, 'head': head, 'records': records, 'actions': actions})
                records = evaluate_head_records(model, terminal, remaining, spec['candidate'], head,
                    labels, positions[head], powers=spec['raw_powers'], seed=spec['seed'],
                    completed=completed, checkpoint=checkpoint, utility_identities=spec['utility_identities'])
                heads[origin, head] = records
                if not hp.exists():
                    checkpoint(records, reference_actions)
                print('source grid cached', origin, head, len(records), 'CPU', round(time.process_time()-started, 2), flush=True)
        anchors = anchors_for(heads, model.weights, spec); save(out/'anchors.json', anchors)
        assert anchors['all_anchors_passed']
        summary = grid_rules(heads, model.weights, baseline, spec)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    cached_counts = {}
    for origin in spec['predictive_origins']:
        for head in (0, 1):
            hp = out/f'origin{origin}_head{head}.json'
            if hp.exists():
                cached_counts[f'{origin}/{head}'] = len(read(str(hp.relative_to(ROOT)))['records'])
    result = {'execution_plan': plan, **summary, 'all_registered_cases_complete': bool(summary),
        'all_reference_anchors_passed': anchors.get('all_anchors_passed', False),
        'cached_head_counts': cached_counts,
        'source_weights': None if model is None else model.weights.tolist(),
        'source_positive_component_counts': None if model is None else [[len(f.means), len(g.means)] for f, g in model.components],
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'cost': spec['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False,
        'all8_shared_posterior_identities_retained': True, 'adaptive_maximum_inside_each_joint_pair': True,
        'limitations': ['One frozen source query;full query ranking and independent deployment comparison remain open.',
            'Identity bounds exclude terminal/predictive quadrature error;earlier two paid moment projections remain.']}
    save(path, result)
    print({k: result[k] for k in ('all_registered_cases_complete', 'all_reference_anchors_passed', 'cpu_seconds', 'cpu_budget_stopped')})
    if summary:
        print(summary['rows'][-1]['raw_net_reward_control_variate'], summary['numerical_changes'], summary['numerical_gates_passed'])
    return result


if __name__ == '__main__':
    run()
