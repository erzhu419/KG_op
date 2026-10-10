"""Reuse raw head grids for remaining matched source/GP queries,one CPU."""

import signal
import time

import numpy as np
from scipy.stats import qmc

from core.positive_mixture_lookahead import RawPredictiveCDF
from core.raw_terminal_head_grid import evaluate_head_records, summarize_proposal_head_grid
from performance.check_historical_raw_head_grid import terminal_for
from performance.check_historical_source_head_grid import baseline_for
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from performance.extend_historical_matched_gp import restore_saved_model

PLAN = 'performance/manifests/historical_remaining_query_grids_v1.json'


def positions_for(spec):
    u = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['cached_predictive_power'])
    return u, [np.arange(max(spec['objective_prefixes'])), np.union1d(
        np.flatnonzero(u[:max(spec['risk_complement_prefixes']), 1] >= spec['low_constraint_probability']),
        np.flatnonzero(u[:, 1] < spec['low_constraint_probability'])[:max(spec['risk_tail_counts'])])]


def grid_rules(heads, weights, baseline, proposal, predictive_origins, spec):
    u, _ = positions_for(spec); rows = []
    indices = [{r['position']: r for r in records} for records in heads]
    for count in spec['objective_prefixes']:
        for prefix in spec['risk_complement_prefixes']:
            body = np.flatnonzero(u[:prefix, 1] >= spec['low_constraint_probability'])
            for tail_count in spec['risk_tail_counts']:
                tail = np.flatnonzero(u[:, 1] < spec['low_constraint_probability'])[:tail_count]
                f = [indices[0][i] for i in range(count)]; g = [indices[1][int(i)] for i in np.r_[tail, body]]
                gw = np.r_[np.full(tail_count, spec['low_constraint_probability']/tail_count),
                    np.full(len(body), (1.-spec['low_constraint_probability'])/len(body))]
                previous_contrast = None
                for power in spec['raw_powers']:
                    r = summarize_proposal_head_grid(f, g, np.full(count, 1./count), gw, weights,
                        proposal, predictive_origins, power=power, baseline=baseline)
                    contrast = np.asarray(r.pop('weighted_adaptive_contrasts')); mass = np.asarray(r.pop('joint_quadrature_weights'))
                    r.update(objective_prefix=count, risk_complement_prefix=prefix, risk_tail_count=tail_count)
                    if previous_contrast is not None:
                        r['weighted_absolute_terminal_contrast_difference'] = float(np.sum(mass*abs(contrast-previous_contrast)))
                    previous_contrast = contrast; rows.append(r)
    fine = rows[-1]; changes = {'terminal_absolute_contrast': fine['weighted_absolute_terminal_contrast_difference']}
    keys = ('objective_prefix', 'risk_complement_prefix', 'risk_tail_count')
    for key, levels in zip(keys, (spec['objective_prefixes'], spec['risk_complement_prefixes'], spec['risk_tail_counts'])):
        coordinates = {k: fine[k] for k in keys}; coordinates[key] = levels[-2]
        old = next(r for r in rows if all(r[k] == v for k, v in coordinates.items()) and r['raw_draw_power'] == fine['raw_draw_power'])
        changes[key] = abs(fine['raw_adaptation_value']-old['raw_adaptation_value'])
    return {'rows': rows, 'numerical_changes': changes,
        'numerical_gates_passed': max(changes.values()) <= spec['maximum_numerical_adaptation_difference'],
        'identity_remainder_gate_passed': max(r['identity_and_rare_origin_error_bound'] for r in rows) <= spec['maximum_identity_integration_error_bound'],
        'proposal_mass_gate_passed': max(abs(r['predictive_mass_residual']) for r in rows) <= spec['maximum_predictive_mass_residual'],
        'proposal_ratio_gate_passed': all(spec['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
            r['predictive_density_ratio_range'][1] <= spec['predictive_density_ratio_limits'][1] for r in rows)}


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    tests = read(str((out/'unit_tests.json').relative_to(ROOT))); assert tests['successful']
    assert read(spec['proposal_calibration'])['all_registered_proposal_gates_passed']
    result_path = out/'results.json'; previous = read(str(result_path.relative_to(ROOT))) if result_path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    started = time.process_time(); cases = list(previous.get('cases', [])); stopped = False
    handler = signal.signal(signal.SIGPROF, cpu_timeout); signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        connection, _, models = build_connection(spec['connection_plan']); assert connection['all_registered_cases_complete']
        signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        gp_states = read(spec['gp_states'])['rows']; u, positions = positions_for(spec)
        for case in spec['cases']:
            name = f"{case['arm']}_layout{case['layout']}_query{case['query']}"
            if any(r['name'] == name for r in cases):
                continue
            tick = time.process_time(); directory = out/name; directory.mkdir(parents=True, exist_ok=True)
            state, source, _, remaining = next(r for r in models if r[0]['layout'] == case['layout'])
            assert case['query'] in state['action_sets']['4']['active_indices']
            if case['arm'] == 'source_structural_mixture':
                model = source; proposal = spec['source_proposal_origin']; predictive = spec['source_predictive_origins']; utility = spec['source_utility_identities']
                baseline = baseline_for(dict(spec, layout=case['layout']))
            else:
                gp = next(r for r in gp_states if (r['layout'], r['arm']) == (case['layout'], case['arm']))
                assert gp['paid_observations'] == state['paid_observations']
                model = restore_saved_model(gp); proposal = 0; predictive = utility = [0]
                before, after = [next(r for r in gp['terminal'] if (r['remaining_calls'], r['power']) == (b, spec['raw_powers'][-1])) for b in (360, 359)]
                baseline = before['best_reward'], after['best_reward'], after['best_action_index']
            terminal = terminal_for(state); heads = []
            for head in (0, 1):
                hp = directory/f'head{head}.json'; lp = directory/f'labels{head}.json'
                if lp.exists():
                    bank = read(str(lp.relative_to(ROOT))); assert bank['source_weights'] == model.weights.tolist()
                else:
                    cdf = RawPredictiveCDF(model.components[proposal][head], case['query'])
                    values = cdf.inverse(u[positions[head], head]); reference, bound = cdf.adaptive(values)
                    error = float(np.max(abs(reference-u[positions[head], head])))+bound
                    assert error <= spec['maximum_inverse_cdf_error']
                    bank = {'source_weights': model.weights.tolist(), 'positions': positions[head].tolist(),
                        'labels': values.tolist(), 'maximum_inverse_cdf_error': error}
                    save(lp, bank)
                labels = dict(zip(bank['positions'], bank['labels']))
                completed = read(str(hp.relative_to(ROOT)))['records'] if hp.exists() else []
                def checkpoint(records, actions):
                    assert actions == state['future_terminal_actions']
                    save(hp, {'records': records, 'actions': actions})
                records = evaluate_head_records(model, terminal, remaining, case['query'], head, labels, positions[head],
                    powers=spec['raw_powers'], seed=spec['seed'], completed=completed,
                    checkpoint=checkpoint, utility_identities=utility)
                heads.append(records)
                print('remaining query', name, 'head', head, len(records), 'CPU', round(time.process_time()-started, 2), flush=True)
            summary = grid_rules(heads, model.weights, baseline, proposal, predictive, spec)
            row = dict(summary, name=name, **case, baseline=list(baseline), source_weights=model.weights.tolist(),
                complete=True, cpu_seconds=time.process_time()-tick)
            save(directory/'results.json', row); cases.append(row)
            save(result_path, {'execution_plan': plan, 'cases': cases, 'all_registered_cases_complete': False})
            print('query completed', name, 'net', summary['rows'][-1]['raw_net_reward_control_variate'],
                'diagnostics', summary['numerical_changes'], 'passed', summary['numerical_gates_passed'], flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'cases': cases, 'all_registered_cases_complete': len(cases) == len(spec['cases']),
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'successful_tests': tests['tests_run'], 'cost': spec['cost'], 'independent_confirmation': False,
        'decision_ranking_certified': False,
        'limitations': ['First remaining-query cohort,not a complete policy ranking or independent deployment evidence.',
            'Density and adjacent quadrature checks are numerical diagnostics;earlier paid moment projections remain.']}
    save(result_path, result); return result


if __name__ == '__main__':
    run()
