"""Screen all missing frozen queries before refining any failed terminal rule."""

import json
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import logsumexp, roots_legendre
from scipy.stats import qmc

from core.positive_mixture_lookahead import RawPredictiveCDF
from core.raw_terminal_head_grid import evaluate_head_records
from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_raw_head_grid import terminal_for
from performance.check_historical_source_head_grid import baseline_for
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from performance.extend_historical_matched_gp import restore_saved_model
from performance.refine_historical_query_objective import actual_anchor

PLAN = 'performance/manifests/historical_remaining_queries_v2.json'


def banks_for(model, proposal, identities, query, directory, spec):
    path = directory/'label_banks.json'
    if path.exists():
        bank = read(str(path.relative_to(ROOT))); assert bank['source_weights'] == model.weights.tolist()
        return bank
    u = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(8)
    heads = []; objective_rules = {}; risk_rules = {}
    for head in (0, 1):
        positions = []; probabilities = []; position = 512 if head == 0 else 256
        if head == 1:
            body = np.flatnonzero(u[:max(spec['risk_complement_prefixes']), 1] >= spec['low_constraint_probability'])
            positions.extend(body.tolist()); probabilities.extend(u[body, 1].tolist())
        for order in spec['objective_prefixes'] if head == 0 else spec['risk_tail_orders']:
            x, w = roots_legendre(order); length = 1. if head == 0 else spec['low_constraint_probability']
            rule = {'positions': list(range(position, position+order)), 'weights': (length*w/2).tolist()}
            (objective_rules if head == 0 else risk_rules)[str(order)] = rule
            positions.extend(rule['positions']); probabilities.extend((length*(x+1)/2).tolist()); position += order
        cdf = RawPredictiveCDF(model.components[proposal][head], query)
        values = cdf.inverse(probabilities); reference, bound = cdf.adaptive(values)
        error = float(np.max(abs(reference-probabilities)))+bound; assert error <= spec['maximum_inverse_cdf_error']
        densities = {str(k): (cdf if k == proposal else RawPredictiveCDF(model.components[k][head], query)).log_density(values).tolist() for k in identities}
        heads.append({'positions': positions, 'labels': values.tolist(), 'probabilities': probabilities,
            'maximum_inverse_cdf_error': error, 'predictive_log_densities': densities})
    bank = {'source_weights': model.weights.tolist(), 'heads': heads, 'objective_rules': objective_rules, 'risk_rules': risk_rules}
    save(path, bank); return bank


def density_gate(bank, proposal, identities, parent, spec):
    heads = bank['heads']; indexes = [{p: i for i, p in enumerate(h['positions'])} for h in heads]; rows = []
    u = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(8)
    for count in spec['objective_prefixes']:
        f = bank['objective_rules'][str(count)]; fi = [indexes[0][p] for p in f['positions']]
        fs = np.column_stack([np.asarray(heads[0]['predictive_log_densities'][str(k)])[fi] for k in identities])
        fp = np.asarray(heads[0]['predictive_log_densities'][str(proposal)])[fi]
        for prefix in spec['risk_complement_prefixes']:
            body = np.flatnonzero(u[:prefix, 1] >= spec['low_constraint_probability'])
            for order in spec['risk_tail_orders']:
                g = bank['risk_rules'][str(order)]; gi = [indexes[1][p] for p in g['positions']+body.tolist()]
                gs = np.column_stack([np.asarray(heads[1]['predictive_log_densities'][str(k)])[gi] for k in identities])
                gp = np.asarray(heads[1]['predictive_log_densities'][str(proposal)])[gi]
                ratio = np.exp(logsumexp(np.log(np.asarray(bank['source_weights'])[identities])+fs[:, None]+gs[None], axis=2)-fp[:, None]-gp[None])
                gw = np.r_[g['weights'], np.full(len(body), (1.-spec['low_constraint_probability'])/len(body))]
                mass = np.sum(np.outer(f['weights'], gw)*ratio)
                rows.append({'objective_prefix': count, 'risk_complement_prefix': prefix, 'risk_tail_order': order,
                    'predictive_mass_residual': float(mass-np.asarray(bank['source_weights'])[identities].sum()),
                    'predictive_density_ratio_range': [float(ratio.min()), float(ratio.max())]})
    passed = max(abs(r['predictive_mass_residual']) for r in rows) <= parent['maximum_predictive_mass_residual'] and all(
        parent['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
        r['predictive_density_ratio_range'][1] <= parent['predictive_density_ratio_limits'][1] for r in rows)
    return {'rows': rows, 'passed': passed, 'new_terminal_heads_at_precheck': 0}


def evaluate_case(model, state, remaining, baseline, bank, directory, identities, proposal, powers, spec, parent):
    heads = []
    for head in (0, 1):
        hp = directory/f'head{head}.json'; completed = read(str(hp.relative_to(ROOT)))['records'] if hp.exists() else []
        labels = bank['heads'][head]
        def checkpoint(records, actions):
            assert actions == state['future_terminal_actions']
            hp.write_text(json.dumps({'records': records, 'actions': actions}, separators=(',', ':'))+'\n')
        records = evaluate_head_records(model, terminal_for(state), remaining, spec['current_query'], head,
            dict(zip(labels['positions'], labels['labels'])), labels['positions'], powers=powers, seed=spec['seed'],
            completed=completed, checkpoint=checkpoint, utility_identities=identities, guarded_mean_position=0)
        index = {p: i for i, p in enumerate(labels['positions'])}
        assert all(max(abs(r['head_log_scores'][k]-labels['predictive_log_densities'][str(k)][index[r['position']]]) for k in identities) < 1e-8 for r in records)
        heads.append(records)
    result = summarize(heads, bank['risk_rules'], model.weights,
        dict(spec, raw_powers=powers, proposal_origin=proposal, predictive_origins=identities),
        objective_rules=bank['objective_rules'], baseline=baseline)
    gates = {'terminal_and_predictive_resolution': result['numerical_gates_passed'],
        'identity_remainder': max(r['identity_and_rare_origin_error_bound'] for r in result['rows']) <= parent['maximum_identity_integration_error_bound'],
        'predictive_mass': max(abs(r['predictive_mass_residual']) for r in result['rows']) <= parent['maximum_predictive_mass_residual'],
        'density_ratio': all(parent['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
            r['predictive_density_ratio_range'][1] <= parent['predictive_density_ratio_limits'][1] for r in result['rows'])}
    result.update(gates=gates, grid_diagnostics_passed=all(gates.values()), raw_powers=powers)
    return result


def validate_anchors(model, state, remaining, baseline, directory, identities, query, powers, spec, workers=1):
    path = directory/'anchors.json'; saved = read(str(path.relative_to(ROOT))) if path.exists() else {'rows': []}
    records = read(str((directory/'head1.json').relative_to(ROOT))); actions = records['actions']; rows = saved['rows']
    pending = [identity for identity in identities if not any(a['identity'] == identity for a in rows)]
    if workers > 1 and len(pending) > 1:
        # Initialize SciPy's shared direction tables before independent anchor threads.
        qmc.Sobol(1, scramble=False)
    def evaluate(identity):
        worst = max(records['records'], key=lambda r: float(np.max(abs(np.asarray(r['raw'][str(powers[-1])]['statistics'][identity])[:24]-
            np.asarray(r['raw'][str(powers[-2])]['statistics'][identity])[:24]))))
        drift = abs(np.asarray(worst['raw'][str(powers[-1])]['statistics'][identity])[:24]-np.asarray(worst['raw'][str(powers[-2])]['statistics'][identity])[:24])
        action_index = int(np.unravel_index(np.argmax(drift), drift.shape)[0])
        anchor_spec = {'identity': identity, 'query': query, 'guarded_allocation_position': action_index//8,
            'conditional_power': spec['anchor_conditional_power'], 'full_joint_power': spec['anchor_full_joint_power'],
            'maximum_error': spec['maximum_anchor_branch_error'], 'seed': spec['seed']}
        initial = actual_anchor(model, state, remaining, worst, actions, (baseline[0], baseline[1], action_index), anchor_spec)
        save(directory/f'anchor_identity{identity}_initial.json', initial)
        if initial['passed']:
            final = initial
        else:
            final = actual_anchor(model, state, remaining, worst, actions, (baseline[0], baseline[1], action_index),
                dict(anchor_spec, conditional_power=spec['extended_anchor_conditional_power'], full_joint_power=spec['extended_anchor_full_joint_power']))
            save(directory/f'anchor_identity{identity}_extended.json', final)
        return final
    pool = ThreadPoolExecutor(max_workers=min(workers, len(pending))) if workers > 1 and pending else None
    try:
        completed = pool.map(evaluate, pending) if pool else map(evaluate, pending)
        for final in completed:
            rows.append(final); save(path, {'rows': rows})
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    return {'rows': rows, 'all_passed': all(a['passed'] for a in rows) and len(rows) == len(identities)}


def run(plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; old = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if old.get('all_registered_cases_complete'):
        return old
    started = time.process_time(); cases = list(old.get('cases', [])); stopped = False
    handler = signal.signal(signal.SIGPROF, cpu_timeout); signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    def publish():
        result = {'execution_plan': plan, 'cases': cases, 'all_initial_cases_complete': len(cases) == len(spec['cases']),
            'all_registered_cases_complete': len(cases) == len(spec['cases']) and all(c.get('processing_complete') for c in cases),
            'passed_case_count': sum(c.get('adaptation_diagnostics_passed', False) for c in cases),
            'cpu_seconds': old.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
            'cost': parent['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False}
        save(path, result); return result
    try:
        _, _, models = build_connection(parent['connection_plan'])
        signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        gp_states = read(parent['gp_states'])['rows']; restored = {}
        def restore(case):
            state, source, _, remaining = next(r for r in models if r[0]['layout'] == case['layout'])
            assert case['query'] in state['action_sets']['4']['active_indices']
            if case['arm'] == 'source_structural_mixture':
                return state, source, remaining, baseline_for(dict(parent, layout=case['layout'])), [4, 5], 4
            if case['layout'] not in restored:
                gp = next(r for r in gp_states if (r['layout'], r['arm']) == (case['layout'], case['arm']))
                assert gp['paid_observations'] == state['paid_observations']
                before, after = [next(r for r in gp['terminal'] if (r['remaining_calls'], r['power']) == (b, parent['raw_powers'][-1])) for b in (360, 359)]
                restored[case['layout']] = restore_saved_model(gp), [before['best_reward'], after['best_reward'], after['best_action_index']]
            model, baseline = restored[case['layout']]; return state, model, remaining, baseline, [0], 0
        for case in spec['cases']:
            name = f"{case['arm']}_layout{case['layout']}_query{case['query']}"
            if any(c['name'] == name for c in cases):
                continue
            directory = out/name; directory.mkdir(parents=True, exist_ok=True)
            state, model, remaining, baseline, identities, proposal = restore(case)
            bank = banks_for(model, proposal, identities, case['query'], directory, spec)
            density = density_gate(bank, proposal, identities, parent, spec); save(directory/'density_precheck.json', density)
            current = {'name': name, **case, 'baseline': list(baseline), 'source_weights': model.weights.tolist(),
                'density_precheck_passed': density['passed'], 'stages': []}
            if density['passed']:
                summary = evaluate_case(model, state, remaining, baseline, bank, directory, identities, proposal,
                    spec['initial_raw_powers'], dict(spec, current_query=case['query']), parent)
                current['stages'].append(summary); save(directory/'initial_results.json', summary)
            else:
                current.update(processing_complete=True, adaptation_diagnostics_passed=False)
            cases.append(current); publish()
            print('initial query', name, 'density', density['passed'], 'grid', current['stages'][-1]['grid_diagnostics_passed'] if current['stages'] else False,
                'CPU', round(time.process_time()-started, 2), flush=True)
        for current in cases:
            if current.get('processing_complete'):
                continue
            directory = out/current['name']; state, model, remaining, baseline, identities, proposal = restore(current)
            summary = current['stages'][-1]
            if not summary['grid_diagnostics_passed'] and summary['raw_powers'] != spec['final_raw_powers']:
                summary = evaluate_case(model, state, remaining, baseline, read(str((directory/'label_banks.json').relative_to(ROOT))), directory,
                    identities, proposal, spec['final_raw_powers'], dict(spec, current_query=current['query']), parent)
                current['stages'].append(summary); save(directory/'refined_results.json', summary); publish()
            anchors = validate_anchors(model, state, remaining, baseline, directory, identities, current['query'], summary['raw_powers'], spec)
            current.update(processing_complete=True, anchors=anchors,
                adaptation_diagnostics_passed=summary['grid_diagnostics_passed'] and anchors['all_passed'])
            save(directory/'results.json', current); publish()
            print('final query', current['name'], 'net', summary['rows'][-1]['raw_net_reward_control_variate'],
                'pass', current['adaptation_diagnostics_passed'], 'tower', summary['fixed_action_tower_gate_passed'],
                'CPU', round(time.process_time()-started, 2), flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return publish()


if __name__ == '__main__':
    result = run(); print('remaining cohort', result['all_initial_cases_complete'], result['all_registered_cases_complete'], result['passed_case_count'])
