"""Refine only failed query0 verifier branches and their risk tail."""

import copy
import signal
import time

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import qmc

from core.conditional_raw_reference import conditional_raw_coordinates
from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_mixture_lookahead import RawPredictiveCDF, condition_positive_raw
from core.raw_likelihood_lookahead import risk_terminal_statistics
from core.raw_terminal_head_grid import evaluate_head_records, raw_terminal_head
from performance.check_historical_raw_head_grid import terminal_for
from performance.check_historical_gauss_risk_tail import summarize
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from performance.extend_historical_matched_gp import restore_saved_model
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments

PLAN = 'performance/manifests/historical_query_verifier_refinement_v1.json'


def run(plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    started = time.process_time(); cases = list(previous.get('cases', [])); stopped = False
    handler = signal.signal(signal.SIGPROF, cpu_timeout); signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        _, _, models = build_connection(parent['connection_plan'])
        signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        for name in spec['case_names']:
            if any(r['name'] == name for r in cases):
                continue
            directory = out/name; directory.mkdir(parents=True, exist_ok=True)
            old_directory = ROOT/parent['output']/name; old = read(str((old_directory/'results.json').relative_to(ROOT)))
            state, source, _, remaining = next(r for r in models if r[0]['layout'] == old['layout'])
            is_source = old['arm'] == 'source_structural_mixture'
            if is_source:
                model = source; proposal = 4; predictive = utility = [4, 5]
            else:
                gp = next(r for r in read(parent['gp_states'])['rows'] if (r['layout'], r['arm']) == (old['layout'], old['arm']))
                model = restore_saved_model(gp); proposal = 0; predictive = utility = [0]
            assert model.weights.tolist() == old['source_weights']
            original = read(str((old_directory/'head1.json').relative_to(ROOT)))
            primary = original['actions'][old['baseline'][2]]['primary']
            guarded = next(a for a in original['actions'] if a['kind'] == 'guarded' and a['primary'] == primary)
            terminal = ConditionalTerminalActions([guarded['shortlist']], [guarded['incumbent_position']], state['terminal_protocol'])
            selected = [i for i, a in enumerate(original['actions']) if a['kind'] == 'guarded' and a['primary'] == primary]
            label_path = directory/'labels.json'
            if label_path.exists():
                labels = read(str(label_path.relative_to(ROOT)))
            else:
                labels = {}; position = 256; cdf = RawPredictiveCDF(model.components[proposal][1], old['query'])
                for order in spec['risk_tail_orders']:
                    x, w = roots_legendre(order); u = spec['low_constraint_probability']*(x+1)/2
                    y = cdf.inverse(u); reference, bound = cdf.adaptive(y)
                    error = float(np.max(abs(reference-u)))+bound; assert error <= parent['maximum_inverse_cdf_error']
                    labels[str(order)] = {'positions': list(range(position, position+order)), 'labels': y.tolist(),
                        'probabilities': u.tolist(), 'weights': (spec['low_constraint_probability']*w/2).tolist(),
                        'maximum_inverse_cdf_error': error}; position += order
                save(label_path, labels)
            lp = {p: y for rule in labels.values() for p, y in zip(rule['positions'], rule['labels'])}
            gp_path = directory/'gauss_heads.json'; completed = read(str(gp_path.relative_to(ROOT)))['records'] if gp_path.exists() else []
            def checkpoint(records, actions):
                assert actions == original['actions']; save(gp_path, {'records': records, 'actions': actions})
            gauss = evaluate_head_records(model, terminal_for(state), remaining, old['query'], 1, lp, sorted(lp),
                powers=spec['raw_powers'], seed=spec['seed'], completed=completed, checkpoint=checkpoint, utility_identities=utility)
            hp = directory/'refined_risk_heads.json'
            records = read(str(hp.relative_to(ROOT)))['records'] if hp.exists() else copy.deepcopy(original['records']+gauss)
            pending = [r for r in records if not r.get('primary_mean_integrated')]
            caches = {k: raw_fantasy_moments(model.components[k][1], old['query'], [r['label'] for r in pending]) for k in utility}
            anchors = []; anchor_position = max(original['records'], key=lambda r: max(
                np.max(abs(np.asarray(r['raw']['14']['statistics'][k])[selected]-np.asarray(r['raw']['12']['statistics'][k])[selected])) for k in utility))['position']
            for j, record in enumerate(pending):
                for identity in utility:
                    prior = model.components[identity][1]; future = copy.deepcopy(prior)
                    score = apply_raw_moments(future, old['query'], caches[identity][0][j], caches[identity][1])
                    assert abs(score-record['head_log_scores'][identity]) < 1e-10
                    for power in spec['raw_powers']:
                        nodes = qmc.Sobol(7, scramble=True, seed=spec['seed']).random_base2(power)
                        statistics, costs, actions, _ = raw_terminal_head(prior, future, terminal, remaining-1,
                            old['query'], record['label'], score, nodes, 1, guarded_mean_position=0)
                        assert actions[:3] == [original['actions'][i] for i in selected]
                        raw = record['raw'][str(power)]
                        for slot, index in enumerate(selected):
                            raw['statistics'][identity][index] = statistics[slot].tolist(); raw['costs'][identity][index] = float(costs[slot])
                    if record['position'] == anchor_position:
                        points = list(dict.fromkeys([old['query']]+guarded['shortlist']))
                        ix = (2*np.asarray(points)[:, None]+np.arange(2)).ravel()
                        p, f = copy.copy(prior), copy.copy(future)
                        for full, marginal in ((prior, p), (future, f)):
                            marginal.means = full.means[:, ix]; marginal.covariances = full.covariances[:, ix][:, :, ix]
                        nodes = qmc.Sobol(7, scramble=True, seed=spec['seed']).random_base2(spec['anchor_full_joint_power'])
                        draws, weights, _ = condition_positive_raw(p, f, 0, list(range(3)), nodes, record['label'], score)
                        order = [points.index(i) for i in guarded['shortlist']]
                        action = original['actions'][selected[-1]]
                        expected, _ = risk_terminal_statistics(draws.reshape(1, len(nodes), 3, 2)[:, :, order], weights[None],
                            guarded['incumbent_position'], action['budgets'], terminal.protocol)
                        nodes = qmc.Sobol(7, scramble=True, seed=spec['seed']).random_base2(spec['anchor_conditional_power'])
                        actual, _, _, _ = raw_terminal_head(prior, future, terminal, remaining-1, old['query'],
                            record['label'], score, nodes, 1, guarded_mean_position=0)
                        error = float(np.max(abs(actual[2]-expected[0])))
                        anchors.append({'identity': identity, 'position': anchor_position, 'error': error,
                            'passed': error <= spec['maximum_anchor_branch_error']})
                        save(directory/'anchors.json', {'rows': anchors})
                record['primary_mean_integrated'] = True
                save(hp, {'records': records, 'actions': original['actions']})
            if anchors:
                save(directory/'anchors.json', {'rows': anchors})
            anchors = read(str((directory/'anchors.json').relative_to(ROOT)))['rows']
            resolution_path = ROOT/'paper_artifacts/historical_query_anchor_resolution_v1/results.json'
            if resolution_path.exists():
                resolution = read(str(resolution_path.relative_to(ROOT)))
                if resolution['case'] == name and resolution['passed']:
                    anchors = [resolution if a['identity'] == resolution['identity'] else a for a in anchors]
            anchor_gate = all(a['passed'] for a in anchors)
            rule = dict(spec, source_reference=str((old_directory/'results.json').relative_to(ROOT)),
                proposal_origin=proposal, predictive_origins=predictive)
            heads = [read(str((old_directory/'head0.json').relative_to(ROOT)))['records'], records]
            summary = summarize(heads, labels, model.weights, rule)
            mass_ok = max(abs(r['predictive_mass_residual']) for r in summary['rows']) <= parent['maximum_predictive_mass_residual']
            row = dict(summary, name=name, anchors=anchors, actual_branch_anchors_passed=anchor_gate,
                proposal_mass_gate_passed=mass_ok,
                all_numerical_gates_passed=summary['numerical_gates_passed'] and mass_ok and anchor_gate)
            save(directory/'results.json', row); cases.append(row)
            save(path, {'execution_plan': plan, 'cases': cases, 'all_registered_cases_complete': False})
            print('verifier refinement', name, 'net', summary['rows'][-1]['raw_net_reward_control_variate'],
                'changes', summary['numerical_changes'], 'pass', row['all_numerical_gates_passed'], 'CPU', round(time.process_time()-started, 2), flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'cases': cases, 'all_registered_cases_complete': len(cases) == len(spec['case_names']),
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'independent_confirmation': False, 'decision_ranking_certified': False, 'cost': parent['cost']}
    save(path, result); return result


if __name__ == '__main__':
    run()
