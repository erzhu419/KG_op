"""Refine only the axis responsible for each of the four remaining failures."""

import signal
import time

import numpy as np
from scipy.special import roots_legendre

from core.positive_mixture_lookahead import RawPredictiveCDF
from core.raw_terminal_head_grid import evaluate_head_records
from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_raw_head_grid import terminal_for
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from performance.extend_historical_matched_gp import restore_saved_model

PLAN = 'performance/manifests/historical_remaining_query_failures_v1.json'


def positive_risk_tail(model, query, proposal, rule, *, order=64, position=304, interval=None):
    """The historical positive Legendre tail, usable on each new target law."""
    low, high = (0., rule['low_constraint_probability']) if interval is None else interval
    return positive_cdf_interval(model.components[proposal][1],query,rule,order=order,
        position=position,interval=(low,high))


def positive_cdf_interval(field,query,rule,*,order,position,interval):
    """The same rule for a full field or its already saved query marginal."""
    low,high=interval
    x, w = roots_legendre(order); probabilities = low+(high-low)*(x+1)/2
    cdf = RawPredictiveCDF(field, query); labels = cdf.inverse(probabilities)
    reference, bound = cdf.adaptive(labels)
    error = float(np.max(abs(reference-probabilities)))+bound
    assert error <= rule['maximum_inverse_cdf_error']
    return {'positions': list(range(position, position+order)),
        'weights': ((high-low)*w/2).tolist(),
        'labels': labels.tolist(), 'maximum_inverse_cdf_error': error}


def run(plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); base_spec = read(parent['parent_plan'])
    original = ROOT/parent['output']; registry = read(str((original/'results.json').relative_to(ROOT)))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); path = out/'results.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    started = time.process_time(); cases = list(previous.get('cases', [])); stopped = False
    handler = signal.signal(signal.SIGPROF, cpu_timeout); signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        _, _, models = build_connection(base_spec['connection_plan'])
        signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        gps = read(base_spec['gp_states'])['rows']
        for descriptor in spec['cases']:
            name = f"{descriptor['arm']}_layout{descriptor['layout']}_query{descriptor['query']}"
            if any(c['name'] == name for c in cases):
                continue
            old = next(c for c in registry['cases'] if c['name'] == name); before = original/name
            directory = out/name; directory.mkdir(parents=True, exist_ok=True)
            state, source, _, remaining = next(r for r in models if r[0]['layout'] == descriptor['layout'])
            is_source = descriptor['arm'] == 'source_structural_mixture'
            model = source if is_source else restore_saved_model(next(r for r in gps if (r['layout'], r['arm']) == (descriptor['layout'], descriptor['arm'])))
            assert model.weights.tolist() == old['source_weights'] and old['anchors']['all_passed']
            bank = read(str((before/'label_banks.json').relative_to(ROOT))); f = read(str((before/'head0.json').relative_to(ROOT)))['records']
            hp = directory/'risk_heads.json'
            records = read(str(hp.relative_to(ROOT)))['records'] if hp.exists() else read(str((before/'head1.json').relative_to(ROOT)))['records']
            rule = dict(parent, proposal_origin=4 if is_source else 0, predictive_origins=[4, 5] if is_source else [0])
            labels = dict(zip(bank['heads'][1]['positions'], bank['heads'][1]['labels']))
            risk_rules = dict(bank['risk_rules']); stages = []
            def checkpoint(records, actions):
                assert actions == state['future_terminal_actions']; save(hp, {'records': records, 'actions': actions})
            if is_source:
                lp = directory/'risk_tail64.json'
                if lp.exists():
                    added = read(str(lp.relative_to(ROOT)))
                else:
                    added = positive_risk_tail(model, descriptor['query'], 4, parent)
                    save(lp, added)
                labels.update(zip(added['positions'], added['labels'])); risk_rules['64'] = added
                records = evaluate_head_records(model, terminal_for(state), remaining, descriptor['query'], 1,
                    labels, added['positions'], powers=spec['source_raw_powers'], seed=parent['seed'],
                    completed=records, checkpoint=checkpoint, utility_identities=[4, 5], guarded_mean_position=0)
                stages.append(summarize([f, records], risk_rules, model.weights,
                    dict(rule, raw_powers=spec['source_raw_powers'], risk_tail_orders=spec['source_risk_tail_orders']),
                    objective_rules=bank['objective_rules'], baseline=old['baseline']))
            else:
                for previous_power, power in zip(spec['gp_risk_powers'], spec['gp_risk_powers'][1:]):
                    records = evaluate_head_records(model, terminal_for(state), remaining, descriptor['query'], 1,
                        labels, bank['heads'][1]['positions'], powers=[power], seed=parent['seed'],
                        completed=records, checkpoint=checkpoint, utility_identities=[0], guarded_mean_position=0)
                    summary = summarize([f, records], risk_rules, model.weights,
                        dict(rule, raw_powers=[previous_power, power], objective_raw_power=spec['gp_objective_power']),
                        objective_rules=bank['objective_rules'], baseline=old['baseline'])
                    stages.append(summary); save(directory/f'risk_power{power}_results.json', summary)
                    print('risk-only extension', name, power, summary['numerical_changes'], 'CPU', round(time.process_time()-started, 2), flush=True)
                    if summary['numerical_gates_passed']:
                        break
            latest = stages[-1]
            gates = {'terminal_and_predictive_resolution': latest['numerical_gates_passed'],
                'identity_remainder': max(r['identity_and_rare_origin_error_bound'] for r in latest['rows']) <= base_spec['maximum_identity_integration_error_bound'],
                'predictive_mass': max(abs(r['predictive_mass_residual']) for r in latest['rows']) <= base_spec['maximum_predictive_mass_residual'],
                'density_ratio': all(base_spec['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
                    r['predictive_density_ratio_range'][1] <= base_spec['predictive_density_ratio_limits'][1] for r in latest['rows']),
                'native_anchors': old['anchors']['all_passed']}
            current = {'name': name, **descriptor, 'stages': stages, 'gates': gates,
                'adaptation_diagnostics_passed': all(gates.values()), 'source_weights': old['source_weights'],
                'baseline': old['baseline'], 'anchors': old['anchors'], 'complete': True}
            save(directory/'results.json', current); cases.append(current)
            save(path, {'execution_plan': plan, 'cases': cases, 'all_registered_cases_complete': False})
            print('axis refinement', name, 'pass', current['adaptation_diagnostics_passed'],
                'net', latest['rows'][-1]['raw_net_reward_control_variate'], 'CPU', round(time.process_time()-started, 2), flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'cases': cases, 'all_registered_cases_complete': len(cases) == len(spec['cases']),
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'cost': base_spec['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False}
    save(path, result); return result


if __name__ == '__main__':
    run()
