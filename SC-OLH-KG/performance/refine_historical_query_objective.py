"""Extend only the failed objective axis and one actual risk-branch anchor."""

import copy
import signal
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_mixture_lookahead import RawPredictiveCDF, condition_positive_raw
from core.raw_likelihood_lookahead import risk_terminal_statistics
from core.raw_terminal_head_grid import evaluate_head_records, raw_terminal_head
from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_raw_head_grid import terminal_for
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments

PLAN = 'performance/manifests/historical_query_objective_refinement_v1.json'


def actual_anchor(model, state, remaining, record, actions, baseline, spec):
    identity = spec['identity']; query = spec.get('query', 0)
    guarded = next(a for a in actions if a['kind'] == 'guarded' and a['primary'] == actions[baseline[2]]['primary'])
    terminal = ConditionalTerminalActions([guarded['shortlist']], [guarded['incumbent_position']], state['terminal_protocol'])
    prior = model.components[identity][1]; future = copy.deepcopy(prior)
    moments, gain = raw_fantasy_moments(prior, query, [record['label']])
    score = apply_raw_moments(future, query, moments[0], gain)
    assert abs(score-record['head_log_scores'][identity]) < 1e-10
    points = list(dict.fromkeys([query]+guarded['shortlist'])); ix = (2*np.asarray(points)[:, None]+np.arange(2)).ravel()
    p, f = copy.copy(prior), copy.copy(future)
    for full, marginal in ((prior, p), (future, f)):
        marginal.means = full.means[:, ix]; marginal.covariances = full.covariances[:, ix][:, :, ix]
    nodes = qmc.Sobol(1+2*len(points), scramble=True, seed=spec['seed']).random_base2(spec['full_joint_power'])
    order = [points.index(i) for i in guarded['shortlist']]
    allocation = spec.get('guarded_allocation_position', 2); budgets = terminal.allocations(remaining-1)[allocation][3]
    statistics = []; log_masses = []
    for start in range(0, len(nodes), 2**18):
        block = nodes[start:start+2**18]
        draws, weights, error = condition_positive_raw(p, f, 0, list(range(len(points))), block, record['label'], score)
        expected, _ = risk_terminal_statistics(draws.reshape(1, len(block), len(points), 2)[:, :, order], weights[None],
            guarded['incumbent_position'], budgets, terminal.protocol)
        statistics.append(expected[0]); log_masses.append(np.log(len(block))+error)
    expected = np.exp(log_masses-logsumexp(log_masses)) @ np.asarray(statistics)
    nodes = qmc.Sobol(7, scramble=True, seed=spec['seed']).random_base2(spec['conditional_power'])
    actual, _, _, _ = raw_terminal_head(prior, future, terminal, remaining-1, query, record['label'], score,
        nodes, 1, guarded_mean_position=0)
    error = float(np.max(abs(actual[allocation]-expected)))
    return {'identity': identity, 'position': record['position'], 'actual': actual[allocation].tolist(),
        'full_reference': expected.tolist(), 'error': error, 'passed': error <= spec['maximum_error'],
        'query': query, 'primary': guarded['primary'], 'shortlist': guarded['shortlist'], 'budgets': budgets,
        'guarded_allocation_position': allocation}


def run(plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); risk_spec = read(spec['risk_refinement_plan'])
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); path = out/'results.json'
    old = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if old.get('complete'):
        return old
    original = ROOT/parent['output']/spec['case']; risk = ROOT/risk_spec['output']/spec['case']
    case = read(str((original/'results.json').relative_to(ROOT)))
    started = time.process_time(); stopped = False; summary = {}
    handler = signal.signal(signal.SIGPROF, cpu_timeout); signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        _, _, models = build_connection(parent['connection_plan'])
        signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        state, model, _, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
        assert model.weights.tolist() == case['source_weights']
        lp = out/'objective_labels.json'
        if lp.exists():
            labels = read(str(lp.relative_to(ROOT)))
        else:
            cached = read(str((original/'labels0.json').relative_to(ROOT)))
            positions = np.arange(len(cached['positions']), max(spec['objective_prefixes']))
            u = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(8)
            cdf = RawPredictiveCDF(model.components[spec['proposal_origin']][0], spec['query'])
            y = cdf.inverse(u[positions, 0]); reference, bound = cdf.adaptive(y)
            error = float(np.max(abs(reference-u[positions, 0])))+bound; assert error <= spec['maximum_inverse_cdf_error']
            labels = dict(cached, positions=cached['positions']+positions.tolist(), labels=cached['labels']+y.tolist(),
                maximum_inverse_cdf_error=max(error, cached['maximum_inverse_cdf_error']))
            save(lp, labels)
        hp = out/'objective_heads.json'
        completed = read(str(hp.relative_to(ROOT)))['records'] if hp.exists() else read(str((original/'head0.json').relative_to(ROOT)))['records']
        def checkpoint(records, actions):
            assert actions == state['future_terminal_actions']; save(hp, {'records': records, 'actions': actions})
        records = evaluate_head_records(model, terminal_for(state), remaining, spec['query'], 0,
            dict(zip(labels['positions'], labels['labels'])), labels['positions'], powers=spec['raw_powers'],
            seed=spec['seed'], completed=completed, checkpoint=checkpoint, utility_identities=[4, 5])
        risk_heads = read(str((risk/'refined_risk_heads.json').relative_to(ROOT)))
        ap = out/'anchor.json'
        if ap.exists():
            anchor = read(str(ap.relative_to(ROOT)))
        else:
            record = next(r for r in risk_heads['records'] if r['position'] == spec['anchor']['position'])
            anchor = actual_anchor(model, state, remaining, record, risk_heads['actions'], case['baseline'], dict(spec['anchor'], seed=spec['seed']))
            save(ap, anchor)
        anchors = read(str((risk/'anchors.json').relative_to(ROOT)))['rows']
        anchors = [anchor if a['identity'] == anchor['identity'] else a for a in anchors]
        rule = dict(spec, source_reference=str((original/'results.json').relative_to(ROOT)))
        summary = summarize([records, risk_heads['records']], read(str((risk/'labels.json').relative_to(ROOT))), model.weights, rule)
        gates = {'actual_branch_anchors_passed': all(a['passed'] for a in anchors),
            'identity_remainder_gate_passed': max(r['identity_and_rare_origin_error_bound'] for r in summary['rows']) <= parent['maximum_identity_integration_error_bound'],
            'proposal_mass_gate_passed': max(abs(r['predictive_mass_residual']) for r in summary['rows']) <= parent['maximum_predictive_mass_residual'],
            'proposal_ratio_gate_passed': all(parent['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
                r['predictive_density_ratio_range'][1] <= parent['predictive_density_ratio_limits'][1] for r in summary['rows'])}
        summary.update(gates, anchors=anchors, all_numerical_gates_passed=summary['numerical_gates_passed'] and all(gates.values()))
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'name': spec['case'], **summary, 'complete': bool(summary),
        'cpu_seconds': old.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'cost': parent['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False}
    save(path, result)
    print('objective refinement complete', result['complete'], 'CPU', round(result['cpu_seconds'], 2), flush=True)
    if summary:
        print('net', summary['rows'][-1]['raw_net_reward_control_variate'], 'changes', summary['numerical_changes'],
            'all numerical gates', summary['all_numerical_gates_passed'], 'tower gate', summary['fixed_action_tower_gate_passed'], flush=True)
    return result


if __name__ == '__main__':
    run()
