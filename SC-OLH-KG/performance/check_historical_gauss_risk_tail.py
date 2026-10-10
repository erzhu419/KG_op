"""Resolve the probability-space risk tail using the historical Legendre rule."""

import signal
import time

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import qmc

from core.positive_mixture_lookahead import RawPredictiveCDF
from core.raw_terminal_head_grid import evaluate_head_records, summarize_proposal_head_grid
from performance.check_historical_source_head_grid import baseline_for
from performance.check_historical_raw_head_grid import terminal_for
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_gauss_risk_tail_v1.json'


def summarize(heads, labels, weights, spec, *, objective_rules=None, baseline=None):
    reference = read(spec['source_reference']) if baseline is None else None
    if baseline is None:
        baseline = reference['baseline'] if 'baseline' in reference else baseline_for(read(reference['execution_plan']))
    indices = [{r['position']: r for r in h} for h in heads]
    u = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(8); rows = []
    for count in spec['objective_prefixes']:
        for prefix in spec['risk_complement_prefixes']:
            body = np.flatnonzero(u[:prefix, 1] >= spec['low_constraint_probability'])
            for order in spec['risk_tail_orders']:
                tail = labels[str(order)]
                f_rule = {'positions': range(count), 'weights': np.full(count, 1./count)} if objective_rules is None else objective_rules[str(count)]
                f = [indices[0][i] for i in f_rule['positions']]
                g = [indices[1][i] for i in tail['positions']]+[indices[1][int(i)] for i in body]
                gw = np.r_[tail['weights'], np.full(len(body), (1.-spec['low_constraint_probability'])/len(body))]
                previous = None
                for power_index, power in enumerate(spec['raw_powers']):
                    objective_power = (spec['objective_raw_powers'][power_index]
                        if 'objective_raw_powers' in spec else spec.get('objective_raw_power'))
                    r = summarize_proposal_head_grid(f, g, f_rule['weights'], gw, weights,
                        spec.get('proposal_origin', 4), spec.get('predictive_origins', [4, 5]),
                        power=power, baseline=baseline, objective_power=objective_power)
                    contrast = np.asarray(r.pop('weighted_adaptive_contrasts')); mass = np.asarray(r.pop('joint_quadrature_weights'))
                    r.update(objective_prefix=count, risk_complement_prefix=prefix, risk_tail_order=order)
                    if previous is not None:
                        r['weighted_absolute_terminal_contrast_difference'] = float(np.sum(mass*abs(contrast-previous)))
                    previous = contrast; rows.append(r)
    fine = rows[-1]; changes = {'terminal_absolute_contrast': fine['weighted_absolute_terminal_contrast_difference']}
    keys = ('objective_prefix', 'risk_complement_prefix', 'risk_tail_order')
    for key, levels in zip(keys, (spec['objective_prefixes'], spec['risk_complement_prefixes'], spec['risk_tail_orders'])):
        coordinates = {k: fine[k] for k in keys}; coordinates[key] = levels[-2]
        old = next(r for r in rows if all(r[k] == v for k, v in coordinates.items()) and r['raw_draw_power'] == fine['raw_draw_power'])
        changes[key] = abs(fine['raw_adaptation_value']-old['raw_adaptation_value'])
    difference = None if reference is None else abs(fine['raw_net_reward_control_variate']-reference['rows'][-1]['raw_net_reward_control_variate'])
    return {'rows': rows, 'numerical_changes': changes, 'reference_net_difference': difference,
        'numerical_gates_passed': max(changes.values()) <= spec['maximum_numerical_adaptation_difference'],
        'fixed_action_tower_gate_passed': abs(fine['raw_fixed_action_tower_residual'])+
            fine['identity_and_rare_origin_error_bound'] <= spec['maximum_fixed_action_tower_residual'],
        'same_model_reference_gate_passed': None if difference is None else difference <= spec['maximum_numerical_adaptation_difference']}


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; old = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if old.get('all_registered_cases_complete'):
        return old
    assert read(spec['proposal_calibration'])['all_registered_proposal_gates_passed']
    started = time.process_time(); summary = {}; stopped = False
    handler = signal.signal(signal.SIGPROF, cpu_timeout); signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        _, _, models = build_connection(spec['connection_plan'])
        signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        state, model, _, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
        labels_path = out/'labels.json'
        if labels_path.exists():
            labels = read(str(labels_path.relative_to(ROOT)))
        else:
            labels = {}; next_position = 256
            cdf = RawPredictiveCDF(model.components[4][1], spec['candidate'])
            for order in spec['risk_tail_orders']:
                x, w = roots_legendre(order); probabilities = spec['low_constraint_probability']*(x+1)/2
                values = cdf.inverse(probabilities); reference, bound = cdf.adaptive(values)
                error = float(np.max(abs(reference-probabilities)))+bound
                assert error <= spec['maximum_inverse_cdf_error']
                labels[str(order)] = {'positions': list(range(next_position, next_position+order)),
                    'probabilities': probabilities.tolist(), 'weights': (spec['low_constraint_probability']*w/2).tolist(),
                    'labels': values.tolist(), 'maximum_inverse_cdf_error': error}
                next_position += order
            save(labels_path, labels)
        lp = {p: y for rule in labels.values() for p, y in zip(rule['positions'], rule['labels'])}
        hp = out/'risk_heads.json'; completed = read(str(hp.relative_to(ROOT)))['records'] if hp.exists() else []
        def checkpoint(records, actions):
            assert actions == state['future_terminal_actions']
            save(hp, {'records': records, 'actions': actions})
        records = evaluate_head_records(model, terminal_for(state), remaining, spec['candidate'], 1, lp, sorted(lp),
            powers=spec['raw_powers'], seed=spec['seed'], completed=completed, checkpoint=checkpoint, utility_identities=[4, 5])
        heads = [read(spec['completed_objective_heads'])['records'], read(spec['completed_risk_heads'])['records']+records]
        summary = summarize(heads, labels, model.weights, spec)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, **summary, 'all_registered_cases_complete': bool(summary),
        'cpu_seconds': old.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'cost': spec['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False}
    save(path, result)
    print('Gauss tail', {k: result[k] for k in ('all_registered_cases_complete', 'cpu_seconds', 'cpu_budget_stopped')})
    if summary:
        print('Gauss tail diagnostics', summary['numerical_changes'], 'tower gate', summary['fixed_action_tower_gate_passed'],
            'net', summary['rows'][-1]['raw_net_reward_control_variate'])
    return result


if __name__ == '__main__':
    result = run()
