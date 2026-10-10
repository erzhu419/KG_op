"""Restore current source priors and precheck every registered independent query."""

import copy
import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from performance.check_historical_remaining_queries import banks_for, density_gate
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import restore_learned_source_prior
from performance.connect_historical_prior_posterior import snapshot, shortlist_indices, _select_shortlist, select_initial_empirical_objective_incumbent
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import task_configs, problem_for
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture

PLAN = 'performance/manifests/historical_independent_kg_preflight_v1.json'


def check_replayed_snapshot(actual, saved, *, roundoff_units=64):
    """Keep identities and billing exact; bound cross-host floating roundoff."""
    for key in ('names', 'paid_episodes'):
        assert actual[key] == saved[key], 'Replayed posterior changed '+key
    tolerance = roundoff_units*np.finfo(float).eps
    assert len(actual['components']) == len(saved['components'])
    for current, reference in zip(actual['components'], saved['components']):
        for key in ('f_mean', 'f_covariance', 'g_mean', 'g_covariance'):
            scale = float(np.max(abs(np.asarray(reference[key]))))
            np.testing.assert_allclose(current[key], reference[key], rtol=0., atol=tolerance*scale)
    np.testing.assert_allclose(actual['weights'], saved['weights'], rtol=tolerance, atol=0.)
    assert len(actual['history']) == len(saved['history'])
    floating = {'weights_before', 'weights_after', 'mixture_log_score', 'head_log_scores', 'joint_log_scores'}
    for current, reference in zip(actual['history'], saved['history']):
        assert {k: v for k, v in current.items() if k not in floating} == {
            k: v for k, v in reference.items() if k not in floating}, 'Replayed paid record changed'
        for key in floating:
            np.testing.assert_allclose(current[key], reference[key], rtol=tolerance,
                atol=0. if key.startswith('weights') else tolerance)


def restore_gp_source_prior(row):
    state = row['prior']; assert state['paid_episodes'] == 0 and not state['history']
    assert len(state['components']) == 1
    component = state['components'][0]
    pair = tuple(JointFunctionRiskPosterior.from_moments(component[h+'_mean'], component[h+'_covariance']) for h in ('f', 'g'))
    model = JointResponseMixture(state['names'], [pair], prior_weights=state['weights'])
    assert snapshot(model) == state
    return model


def terminal_from_initialization(provider, candidates, observed, protocol, paid_indices=(0, 1)):
    paid_candidates = [candidates[i] for i in paid_indices]
    rows, _ = _select_shortlist(provider, paid_candidates, observed)
    positions = [paid_candidates.index(tuple(r['point'])) for r in rows]
    incumbent = select_initial_empirical_objective_incumbent(paid_candidates, observed, n0=2)
    positions[0] = paid_candidates.index(tuple(incumbent['point']))
    table, incumbent_positions = shortlist_indices(positions, list(paid_indices), observed,
                                                  len(candidates), range(len(candidates)))
    return ConditionalTerminalActions(table, incumbent_positions, protocol)


def actions_at(terminal, remaining):
    return [{'kind': kind, 'primary': int(indices[0]), 'shortlist': list(map(int, indices)),
        'incumbent_position': int(incumbent), 'budgets': list(budgets)}
        for kind, table, incumbents, budgets in terminal.allocations(remaining)
        for indices, incumbent in zip(table, incumbents)]


def restore_target_models(pilot, states):
    """Replay only new paid records into native priors,without new episodes."""
    gp_rows = read(pilot['gp_states'])['rows']; priors = {}
    for state in states:
        key = state['layout'], state['arm']
        if key not in priors:
            if state['arm'] == 'source_structural_mixture':
                priors[key] = restore_learned_source_prior(state['layout'], state['candidate_bank_indices'], pilot['source_connection_plan'])[0]
            else:
                row = next(r for r in gp_rows if (r['layout'], r['arm']) == key)
                assert row['candidate_bank_indices'] == state['candidate_bank_indices']
                priors[key] = restore_gp_source_prior(row)
        model = copy.deepcopy(priors[key])
        for i, values in enumerate(state['centered_observations']):
            model.update_raw(i, values)
        check_replayed_snapshot(snapshot(model), state['posterior_moment_diagnostic'])
        guarded = state['terminal_actions'][:8]
        terminal = ConditionalTerminalActions([a['shortlist'] for a in guarded],
            [a['incumbent_position'] for a in guarded], state['terminal_protocol'])
        yield state, model, terminal


def run(plan=PLAN):
    spec = read(plan); pilot = read(spec['pilot_plan']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    assert read(str((out/'unit_tests.json').relative_to(ROOT)))['successful']
    assert read(str((out/'restoration_checks.json').relative_to(ROOT)))['complete_connection_exactly_preserved']
    started = time.process_time(); numerical = read(pilot['numerical_plan'])
    source_states = read(pilot['source_states'])['rows']; gp_states = read(pilot['gp_states'])['rows']
    connection = read(pilot['source_connection_plan']); legacy = read(connection['legacy_connection_plan'])
    public = read(legacy['public_inputs']); raw_bank = public['bank']
    assert public['plan']['source_dimension'] == pilot['dimension'] == 50
    assert [s['candidate_bank_indices'][:2] for s in source_states] == [source_states[0]['candidate_bank_indices'][:2]]*2
    configs = task_configs({'source_task_count': pilot['new_task_count'], 'task_seed': pilot['task_seed'], 'family': pilot['family']})
    for i, config in enumerate(configs):
        config['task_id'] = f'learned_prior_independent_{i:02d}'
    frozen = {'pilot_plan': pilot, 'tasks': configs, 'banks': [{'layout': s['layout'],
        'candidate_bank_indices': s['candidate_bank_indices'], 'queries': s['action_sets']['4']['active_indices'],
        'policies': [raw_bank[i] for i in s['candidate_bank_indices']]} for s in source_states]}
    frozen_path = out/'frozen_inputs.json'
    if frozen_path.exists():
        assert read(str(frozen_path.relative_to(ROOT))) == frozen
    else:
        save(frozen_path, frozen)
    initial_path = out/'initializations.json'; initial = read(str(initial_path.relative_to(ROOT)))['rows'] if initial_path.exists() else []
    cases = list(previous.get('cases', [])); states = list(previous.get('states', [])); new_calls = 0; stopped = False
    priors = {}; restoration = []
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-
        previous.get('cpu_seconds', 0.)-(time.process_time()-started)))
    try:
        for state in source_states:
            layout = state['layout']; source, _, checks = restore_learned_source_prior(layout, state['candidate_bank_indices'], pilot['source_connection_plan'])
            assert snapshot(source) == state['prior']
            gp_row = next(r for r in gp_states if (r['layout'], r['arm']) == (layout, 'source_common_task_rbf'))
            assert gp_row['source_task_ids'] == [f'related_inventory_{i:02d}' for i in pilot['training_tasks']]
            priors[layout] = {'source_structural_mixture': source, 'source_common_task_rbf': restore_gp_source_prior(gp_row)}
            restoration.append({'layout': layout, 'source_prior_exactly_preserved': True, 'prior_paid_episodes': source.paid_episodes,
                'positive_components_by_identity_and_head': [[len(f.means) for f in pair] for pair in source.components],
                'positive_source_components': sum(len(f.means) for pair in source.components for f in pair),
                'maximum_source_projection_error': max(c['common_projection_max_error'] for c in checks), 'gp_zero_paid_prior_preserved': True})
        for task_index, config in enumerate(configs):
            if not any(r['task_id'] == config['task_id'] for r in initial):
                problem = problem_for(public['plan'], config)
                rng = np.random.default_rng(np.random.SeedSequence([pilot['initialization_seed'], task_index]))
                observed = np.asarray([problem.simulate(raw_bank[i], rng) for i in source_states[0]['candidate_bank_indices'][:2]])
                initial.append({'task_id': config['task_id'], 'task_index': task_index,
                    'candidate_bank_indices': source_states[0]['candidate_bank_indices'][:2],
                    'paired_observations': observed.tolist(), 'tau': float(problem.tau), 'physical_calls': 2})
                new_calls += 2; save(initial_path, {'rows': initial})
        for task_index, config in enumerate(configs):
            observed_row = next(r for r in initial if r['task_id'] == config['task_id']); observed = np.asarray(observed_row['paired_observations'])
            provider = no_outcomes(problem_for(public['plan'], config))
            for state in source_states:
                layout = state['layout']; candidates = [tuple(raw_bank[i]) for i in state['candidate_bank_indices']]
                assert provider.alpha == state['terminal_protocol']['chance_alpha']
                assert np.dot(provider.weights, provider.ref_point) == state['terminal_protocol']['objective_reference']
                terminal = terminal_from_initialization(provider, candidates, observed, state['terminal_protocol'])
                actions = actions_at(terminal, 360); future = actions_at(terminal, 359)
                assert len(actions) == len(future) == 40
                for arm in pilot['models']:
                    model = copy.deepcopy(priors[layout][arm])
                    for i, value in enumerate(observed):
                        model.update_raw(i, [value[0], value[1]-observed_row['tau']])
                    assert model.paid_episodes == 2 and len(model.history) == 2
                    name = f"{config['task_id']}_{arm}_layout{layout}"
                    if not any(r['name'] == name for r in states):
                        states.append({'name': name, 'task_id': config['task_id'], 'layout': layout, 'arm': arm,
                            'candidate_bank_indices': state['candidate_bank_indices'], 'queries': state['action_sets']['4']['active_indices'],
                            'source_weights': model.weights.tolist(), 'paid_observations': observed_row['paired_observations'],
                            'centered_observations': [[float(v[0]), float(v[1]-observed_row['tau'])] for v in observed],
                            'terminal_protocol': state['terminal_protocol'], 'terminal_actions': actions, 'future_terminal_actions': future,
                            'posterior_moment_diagnostic': snapshot(model), 'native_restoration': 'restore source priors,then apply centered_observations;do not restore positive source fields from moment diagnostics'})
                    source = arm == 'source_structural_mixture'; identities, proposal = ([4, 5], 4) if source else ([0], 0)
                    for query in state['action_sets']['4']['active_indices']:
                        case_name = f'{name}_query{query}'
                        if any(r['name'] == case_name for r in cases):
                            continue
                        directory = out/case_name; directory.mkdir(exist_ok=True)
                        bank = banks_for(model, proposal, identities, query, directory, numerical)
                        density = density_gate(bank, proposal, identities, pilot, numerical)
                        rare = float(sum(w for i, w in enumerate(model.weights) if i not in identities))
                        row = {'name': case_name, 'task_id': config['task_id'], 'layout': layout, 'arm': arm, 'query': query,
                            'density_precheck_passed': density['passed'], 'rare_predictive_origin_mass': rare,
                            'rare_origin_precheck_passed': rare <= pilot['maximum_identity_integration_error_bound'],
                            'new_terminal_head_evaluations': 0, 'density_precheck': density}
                        save(directory/'density_precheck.json', density); cases.append(row)
                        save(path, {'cases': cases, 'states': states, 'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started,
                            'all_registered_cases_complete': False})
                    print('independent precheck', config['task_id'], layout, arm, 'cases', len(cases), 'CPU', round(time.process_time()-started, 2), flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    expected = pilot['new_task_count']*2*len(pilot['models'])*6
    complete = len(cases) == expected
    passed = [r for r in cases if r['density_precheck_passed'] and r['rare_origin_precheck_passed']]
    result = {'execution_plan': plan, 'pilot_plan': spec['pilot_plan'], 'source_restoration': restoration,
        'states': states, 'cases': cases, 'all_registered_cases_complete': complete,
        'registered_query_cases': expected, 'passed_query_cases': len(passed),
        'all_density_and_rare_origin_prechecks_passed': complete and len(passed) == expected,
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'cost': {'source_fits': 0, 'new_physical_initialization_calls_this_execution': new_calls,
            'shared_physical_initialization_calls': sum(r['physical_calls'] for r in initial),
            'logical_initialization_calls_for_final48_arms': len(initial)*2*2*len(pilot['models'])*len(pilot['policies']),
            'new_search_calls': 0, 'new_verifier_calls': 0, 'new_terminal_head_evaluations': 0, 'server_jobs': 0},
        'independent_outcomes_opened': False, 'independent_confirmation': False,
        'next': 'Compute target-specific terminal baselines and all six KG values per saved model state; freeze all48 recommendations before verification.',
        'limitations': ['Predictive-density prechecks do not establish terminal precision,query rankings or deployment benefit.',
            'Six same-population d50 tasks and earlier paid moment projections remain the declared pilot scope.']}
    save(path, result)
    print('independent preflight', len(passed), '/', expected, 'CPU', round(result['cpu_seconds'], 2), 'calls', result['cost'], flush=True)
    return result


if __name__ == '__main__':
    run()
