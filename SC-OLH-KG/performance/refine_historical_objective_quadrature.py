"""Check objective density quadrature before computing any terminal utilities."""

import signal
import time

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import qmc

from core.positive_mixture_lookahead import RawPredictiveCDF
from core.raw_terminal_head_grid import evaluate_head_records
from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_raw_head_grid import terminal_for
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_query_objective_legendre_v1.json'


def density_precheck(rules, risk_heads, risk_labels, weights, parent, spec):
    risk_index = {r['position']: r for r in risk_heads}; rows = []
    u = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(8)
    for count, rule in rules.items():
        fr = np.exp(np.asarray(rule['predictive_log_density5'])-rule['predictive_log_density4'])
        for prefix in spec['risk_complement_prefixes']:
            body = np.flatnonzero(u[:prefix, 1] >= spec['low_constraint_probability'])
            for order in spec['risk_tail_orders']:
                tail = risk_labels[str(order)]; records = [risk_index[int(i)] for i in tail['positions']+body.tolist()]
                gw = np.r_[tail['weights'], np.full(len(body), (1.-spec['low_constraint_probability'])/len(body))]
                gr = np.exp([r['head_log_scores'][5]-r['head_log_scores'][4] for r in records])
                mass = weights[4]+weights[5]*np.dot(rule['weights'], fr)*np.dot(gw, gr)
                ratio = weights[4]+weights[5]*fr[:, None]*gr
                rows.append({'objective_order': int(count), 'risk_complement_prefix': prefix, 'risk_tail_order': order,
                    'predictive_mass_residual': float(mass-weights[4]-weights[5]),
                    'predictive_density_ratio_range': [float(ratio.min()), float(ratio.max())]})
    passed = max(abs(r['predictive_mass_residual']) for r in rows) <= parent['maximum_predictive_mass_residual'] and all(
        parent['predictive_density_ratio_limits'][0] <= r['predictive_density_ratio_range'][0] and
        r['predictive_density_ratio_range'][1] <= parent['predictive_density_ratio_limits'][1] for r in rows)
    return {'rows': rows, 'passed': passed, 'new_terminal_heads_at_precheck': 0}


def run(plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); risk_spec = read(spec['risk_refinement_plan'])
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); path = out/'results.json'
    old = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if old.get('complete') or old.get('density_precheck_failed'):
        return old
    original = ROOT/parent['output']/spec['case']; risk = ROOT/risk_spec['output']/spec['case']
    case = read(str((original/'results.json').relative_to(ROOT)))
    risk_heads = read(str((risk/'refined_risk_heads.json').relative_to(ROOT)))['records']
    risk_labels = read(str((risk/'labels.json').relative_to(ROOT)))
    started = time.process_time(); stopped = False; summary = {}; precheck = {}
    handler = signal.signal(signal.SIGPROF, cpu_timeout); signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        _, _, models = build_connection(parent['connection_plan'])
        signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        state, model, _, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
        assert model.weights.tolist() == case['source_weights']; lp = out/'objective_rules.json'
        if lp.exists():
            rules = read(str(lp.relative_to(ROOT)))
        else:
            rules = {}; position = 512; cdf = RawPredictiveCDF(model.components[4][0], spec['query'])
            other = RawPredictiveCDF(model.components[5][0], spec['query'])
            for order in spec['objective_prefixes']:
                x, w = roots_legendre(order); u = (x+1)/2; y = cdf.inverse(u); reference, bound = cdf.adaptive(y)
                error = float(np.max(abs(reference-u)))+bound; assert error <= spec['maximum_inverse_cdf_error']
                rules[str(order)] = {'positions': list(range(position, position+order)), 'probabilities': u.tolist(),
                    'labels': y.tolist(), 'weights': (w/2).tolist(), 'maximum_inverse_cdf_error': error,
                    'predictive_log_density4': cdf.log_density(y).tolist(), 'predictive_log_density5': other.log_density(y).tolist()}
                position += order
            save(lp, rules)
        precheck = density_precheck(rules, risk_heads, risk_labels, model.weights, parent, spec); save(out/'density_precheck.json', precheck)
        print('objective density precheck', precheck['passed'], 'max residual', max(abs(r['predictive_mass_residual']) for r in precheck['rows']), flush=True)
        if precheck['passed']:
            labels = {p: y for rule in rules.values() for p, y in zip(rule['positions'], rule['labels'])}; hp = out/'objective_heads.json'
            completed = read(str(hp.relative_to(ROOT)))['records'] if hp.exists() else []
            def checkpoint(records, actions):
                assert actions == state['future_terminal_actions']; save(hp, {'records': records, 'actions': actions})
            records = evaluate_head_records(model, terminal_for(state), remaining, spec['query'], 0, labels, sorted(labels),
                powers=spec['raw_powers'], seed=spec['seed'], completed=completed, checkpoint=checkpoint, utility_identities=[4, 5])
            densities = {p: (f, g) for rule in rules.values() for p, f, g in zip(rule['positions'], rule['predictive_log_density4'], rule['predictive_log_density5'])}
            assert all(max(abs(r['head_log_scores'][k]-densities[r['position']][k-4]) for k in (4, 5)) < 1e-8 for r in records)
            summary = summarize([records, risk_heads], risk_labels, model.weights,
                dict(spec, source_reference=str((original/'results.json').relative_to(ROOT))), objective_rules=rules)
            anchors = read(spec['anchor_results'])['anchors']; assert all(a['passed'] for a in anchors)
            gates = {'actual_branch_anchors_passed': True,
                'identity_remainder_gate_passed': max(r['identity_and_rare_origin_error_bound'] for r in summary['rows']) <= parent['maximum_identity_integration_error_bound'],
                'proposal_mass_gate_passed': max(abs(r['predictive_mass_residual']) for r in summary['rows']) <= parent['maximum_predictive_mass_residual'],
                'proposal_ratio_gate_passed': precheck['passed']}
            summary.update(gates, anchors=anchors, all_numerical_gates_passed=summary['numerical_gates_passed'] and all(gates.values()))
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'name': spec['case'], **summary, 'complete': bool(summary),
        'density_precheck_failed': bool(precheck) and not precheck['passed'],
        'cpu_seconds': old.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'cost': parent['cost'], 'independent_confirmation': False, 'decision_ranking_certified': False}
    save(path, result); print('objective quadrature complete', result['complete'], 'CPU', round(result['cpu_seconds'], 2), flush=True)
    if summary:
        print('net', summary['rows'][-1]['raw_net_reward_control_variate'], 'changes', summary['numerical_changes'],
            'all numerical gates', summary['all_numerical_gates_passed'], 'tower', summary['fixed_action_tower_gate_passed'], flush=True)
    return result


if __name__ == '__main__':
    run()
