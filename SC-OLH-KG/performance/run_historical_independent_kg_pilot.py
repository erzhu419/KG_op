"""Freeze new-target raw-KG recommendations, then independently verify them."""

import argparse
import copy
import json
import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_raw_terminal import raw_terminal_values
from performance.check_historical_independent_queries import registered_queries
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.prepare_historical_independent_kg import restore_target_models
from performance.run_legacy_one_step_closed_loop import paid_update, execute_terminal
from performance.run_legacy_related_inventory_source import problem_for
from performance.score_legacy_related_inventory_source import safe_terminal_values

PLAN = 'performance/manifests/historical_independent_kg_execution_v1.json'
STOP = 'stop_after_initialization'
KG = 'one_step_raw_kg'


def key(row):
    return row['task_id'], row['layout'], row['arm'], row['policy']


def choose_queries(states, diagnostics, baselines):
    """A missing or HOLD query blocks every search, before any simulator exists."""
    expected = registered_queries(states)
    cases = diagnostics['cases']; by_name = {c['name']: c for c in cases}
    if (len(by_name) != len(cases) or set(by_name) != {c['name'] for c in expected}
            or not diagnostics['all_model_states_ready']):
        raise ValueError('Complete all registered KG diagnostics before adaptive search')
    baseline = {c['name']: c for c in baselines['cases']}
    if (len(baseline) != len(states) or set(baseline) != {s['name'] for s in states}
            or not baselines['all_baseline_diagnostics_passed']):
        raise ValueError('Every model needs its accepted stopping baseline')
    for descriptor in expected:
        case = by_name[descriptor['name']]
        if (any(case[k] != v for k, v in descriptor.items())
                or not case['processing_complete'] or not case['adaptation_diagnostics_passed']):
            raise ValueError('Missing, mismatched or HOLD KG diagnostic')
    decisions = []
    for state in states:
        if not baseline[state['name']]['baseline_diagnostics_passed']:
            raise ValueError('Stopping baseline remains HOLD')
        rows = [by_name[f"{state['name']}_query{q}"] for q in state['queries']]
        values = [r['estimated_net_kg'] for r in rows]
        if not np.isfinite(values).all():
            raise ValueError('KG estimates must be finite')
        best = int(np.argmax(values))
        decisions.append({'state_name': state['name'], 'task_id': state['task_id'],
            'layout': state['layout'], 'arm': state['arm'],
            'query': rows[best]['query'] if values[best] > 0 else None,
            'estimated_net_kg': values[best], 'query_diagnostics': rows,
            'strict_query_ranking_certified': False})
    return decisions


def validate_recommendations(rows, states, pilot):
    expected = {(s['task_id'], s['layout'], s['arm'], p)
                for s in states for p in pilot['policies']}
    if len(rows) != len(expected) or {key(r) for r in rows} != expected:
        raise ValueError('All final recommendations must be frozen before verification')
    by_name = {s['name']: s for s in states}
    for row in rows:
        state = by_name[row['state_name']]
        if (key(row)[:3] != (state['task_id'], state['layout'], state['arm'])
                or not row['recommendation_diagnostics_passed']):
            raise ValueError('Recommendation does not match an accepted target state')
        paid = 2+int(row['query'] is not None)
        if (row['policy'] == STOP and row['query'] is not None
                or row['search_calls'] != paid):
            raise ValueError('Recommendation has inconsistent query debit')
        actions = state['future_terminal_actions'] if paid == 3 else state['terminal_actions']
        if actions[row['action_index']] != row['action']:
            raise ValueError('Recommendation changed the initialization-frozen action class')
    return True


def utility_identities(updated, arm, bound):
    identities = [4, 5] if arm == 'source_structural_mixture' else [0]
    omitted = [i for i in range(len(updated.components)) if i not in identities]
    for i in sorted(omitted, key=lambda i: -updated.weights[i]):
        if float(updated.weights[[j for j in omitted if j not in identities]].sum()) <= bound:
            break
        identities.append(i)
    return identities


def search_record(model, problem, candidates, decision, seed, stream_tag, path):
    """Resume the saved paid episode instead of simulating it a second time."""
    query = decision['query']
    if path.exists():
        record = read(str(path))
        if record['decision'] != decision:
            raise ValueError('Saved paid episode belongs to a different frozen query decision')
        updated = copy.deepcopy(model)
        evidence = updated.update_raw(query, record['centered_observation'])
        if evidence != record['evidence']:
            raise ValueError('Saved raw update does not reproduce its complete identity evidence')
    else:
        updated, observation, evidence = paid_update(model, problem, candidates, query, seed, stream_tag)
        record = {'decision': decision, 'raw_observation': observation.tolist(),
            'centered_observation': [float(observation[0]), float(observation[1]-problem.tau)],
            'evidence': evidence, 'seed': seed, 'stream_tag': stream_tag, 'physical_calls': 1}
        save(path, record)
    return updated, record


def recommendation_check(previous, current, pilot, importance):
    difference = float(np.max(abs(np.asarray(previous['values'])-current['values'])))
    checks = current['diagnostics']
    gates = {'all_action_resolution': difference <= pilot['maximum_numerical_adaptation_difference'],
        'choice_stable': previous['best_action_index'] == current['best_action_index'],
        'identity_remainder': checks['omitted_posterior_utility_mass'] <= pilot['maximum_identity_integration_error_bound'],
        'importance_evidence': checks['predictive_log_evidence_max_error'] <= importance['maximum_log_evidence_error'],
        'importance_ess': checks['effective_sample_size_min'] >= importance['minimum_reference_ESS']}
    return dict(gates=gates, maximum_action_difference=difference, passed=all(gates.values()))


def inputs(spec):
    pilot = read(spec['pilot_plan']); preflight = read(spec['preflight']); frozen = read(spec['frozen_inputs'])
    states = preflight['states']
    expected = {(t['task_id'], b['layout'], arm) for t in frozen['tasks']
                for b in frozen['banks'] for arm in pilot['models']}
    if (frozen['pilot_plan'] != pilot or len(frozen['tasks']) != pilot['new_task_count']
            or len(states) != len(expected)
            or {(s['task_id'], s['layout'], s['arm']) for s in states} != expected
            or not preflight['all_density_and_rare_origin_prechecks_passed']):
        raise ValueError('Independent task/model registration or predictive prechecks are incomplete')
    connection = read(pilot['source_connection_plan'])
    public = read(read(connection['legacy_connection_plan'])['public_inputs'])['plan']
    return pilot, frozen, states, public


def freeze(plan=PLAN):
    started = time.process_time(); spec = read(plan)
    pilot, frozen, states, public = inputs(spec)
    baselines = read(spec['baselines'])
    decisions = choose_queries(states, read(spec['query_diagnostics']), baselines)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'recommendations.json'
    if path.exists():
        result = read(str(path)); validate_recommendations(result['rows'], states, pilot)
        return result
    decision_path = out/'query_decisions.json'
    if decision_path.exists() and read(str(decision_path))['rows'] != decisions:
        raise ValueError('Query decisions have already frozen differently')
    save(decision_path, {'execution_plan': plan, 'rows': decisions})
    baseline_by_name = {c['name']: c for c in baselines['cases']}
    decision_by_name = {c['state_name']: c for c in decisions}
    progress_path = out/'recommendation_progress.json'
    progress = read(str(progress_path)) if progress_path.exists() else {'rows': [], 'cpu_seconds': 0.}
    rows = progress['rows']; previous_cpu = progress['cpu_seconds']; stopped = False
    importance = read(spec['importance_precision_plan'])['precision']
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    try:
        selected = [s for s in states if not spec.get('state_names') or s['name'] in spec['state_names']]
        pending = [s for s in selected if not all(any(key(r) == (s['task_id'], s['layout'], s['arm'], p)
                    for r in rows) for p in pilot['policies'])]
        for state, model, terminal in restore_target_models(pilot, pending):
            decision = decision_by_name[state['name']]
            task_index = next(i for i, t in enumerate(frozen['tasks']) if t['task_id'] == state['task_id'])
            layout_index = next(i for i, b in enumerate(frozen['banks']) if b['layout'] == state['layout'])
            bank = frozen['banks'][layout_index]
            candidates = [tuple(p) for p in bank['policies']]
            seed = int(np.random.SeedSequence([pilot['verification_seed'], task_index, layout_index]).generate_state(1)[0])
            for policy in pilot['policies']:
                if any(key(r) == (state['task_id'], state['layout'], state['arm'], policy) for r in rows):
                    continue
                query = decision['query'] if policy == KG else None
                row = {k: state[k] for k in ('task_id', 'layout', 'arm')}
                row.update(state_name=state['name'], task_index=task_index, policy=policy, query=query,
                    search_calls=2+int(query is not None), verification_seed=seed)
                if query is None:
                    baseline = baseline_by_name[state['name']]
                    fine = next(r for r in read(baseline['result'])['rows']
                                if r['power'] == baseline['power'] and r['remaining_calls'] == 360)
                    row.update(action=fine['best_action'], action_index=fine['best_action_index'],
                        predicted_reward=fine['best_reward'], predicted_verification_calls=fine['costs'][fine['best_action_index']],
                        recommendation_diagnostics_passed=True)
                else:
                    left = spec['maximum_freeze_cpu_seconds']-previous_cpu-(time.process_time()-started)
                    if left <= 0:
                        raise TimeoutError('Recommendation stage CPU budget reached')
                    directory = out/state['name']; directory.mkdir(exist_ok=True)
                    problem = problem_for(public, frozen['tasks'][task_index])
                    stream = task_index*4+layout_index*2+pilot['models'].index(state['arm'])
                    updated, paid = search_record(model, problem, candidates, decision,
                        pilot['search_seed'], stream, directory/'search_observation.json')
                    identities = utility_identities(updated, state['arm'], pilot['maximum_identity_integration_error_bound'])
                    raw_path = directory/'post_query_terminal.json'
                    raw = read(str(raw_path)) if raw_path.exists() else {'rows': [], 'comparisons': [], 'passed': False}
                    left = spec['maximum_freeze_cpu_seconds']-previous_cpu-(time.process_time()-started)
                    signal.setitimer(signal.ITIMER_PROF, max(.001, left))
                    try:
                        for power in spec['post_query_raw_powers']:
                            if raw['passed']:
                                break
                            if any(r['power'] == power for r in raw['rows']):
                                continue
                            values, costs, actions, checks = raw_terminal_values(model, updated, terminal, 359,
                                query, paid['centered_observation'], np.asarray(paid['evidence']['head_log_scores']),
                                power=power, seed=spec['integration_seed'], utility_identities=identities,
                                guarded_mean_position=0)
                            if actions != state['future_terminal_actions']:
                                raise ValueError('Post-query recommendation changed the forty frozen actions')
                            best = int(np.lexsort((costs, -values))[0])
                            current = {'power': power, 'values': values.tolist(), 'costs': costs.tolist(),
                                'best_action_index': best, 'diagnostics': checks}
                            if raw['rows']:
                                comparison = recommendation_check(raw['rows'][-1], current, pilot, importance)
                                raw['comparisons'].append(comparison); raw['passed'] = comparison['passed']
                            raw['rows'].append(current); save(raw_path, raw)
                    finally:
                        signal.setitimer(signal.ITIMER_PROF, 0.)
                    if not raw['passed']:
                        print('post-query recommendation HOLD', state['name'], flush=True)
                        continue
                    fine = raw['rows'][-1]; best = fine['best_action_index']
                    row.update(action=state['future_terminal_actions'][best], action_index=best,
                        predicted_reward=fine['values'][best], predicted_verification_calls=fine['costs'][best],
                        raw_search_record=str((directory/'search_observation.json').relative_to(ROOT)),
                        post_query_diagnostics=str(raw_path.relative_to(ROOT)), recommendation_diagnostics_passed=True)
                rows.append(row)
                save(progress_path, {'rows': rows, 'cpu_seconds': previous_cpu+time.process_time()-started})
                print('recommendation prepared', state['name'], policy, 'query', query, flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'rows': rows, 'cpu_seconds': previous_cpu+time.process_time()-started,
        'cpu_budget_stopped': stopped, 'all_recommendations_frozen': len(rows) == len(states)*len(pilot['policies']),
        'selected_recommendations_complete': len(rows) == len(selected)*len(pilot['policies']),
        'independent_outcomes_opened': False, 'independent_confirmation': False}
    if result['selected_recommendations_complete']:
        validate_recommendations(rows, selected, pilot)
    save(progress_path, result)
    if result['all_recommendations_frozen']:
        validate_recommendations(rows, states, pilot); save(path, result)
    return result


def paired_summary(rows, frozen, pilot, draws):
    task_ids = [t['task_id'] for t in frozen['tasks']]; layouts = [b['layout'] for b in frozen['banks']]
    indexed = {key(r): r for r in rows}
    source, gp = pilot['models']
    comparisons = [((source, KG), (gp, KG)), ((source, KG), (source, STOP)), ((gp, KG), (gp, STOP))]
    rng = np.random.default_rng(pilot['bootstrap_seed'])
    samples = rng.integers(len(task_ids), size=(draws, len(task_ids))); results = []
    for first, second in comparisons:
        metrics = {}
        for metric in ('actual_safe_reward', 'false_certificate', 'rejection', 'actual_total_calls',
                       'conditional_safe_reward', 'conditional_total_calls'):
            differences = np.array([np.mean([float(indexed[(t, l)+first]['scores'][metric])-
                float(indexed[(t, l)+second]['scores'][metric]) for l in layouts]) for t in task_ids])
            metrics[metric] = {'task_differences': differences.tolist(), 'mean_difference': float(differences.mean()),
                'task_bootstrap_95_interval': np.quantile(differences[samples].mean(axis=1), [.025, .975]).tolist()}
        results.append({'arm': list(first), 'reference': list(second), 'task_count': len(task_ids), 'metrics': metrics})
    return results


def verify(plan=PLAN):
    spec = read(plan); pilot, frozen, states, public = inputs(spec)
    out = ROOT/spec['output']; barrier = read(str(out/'recommendations.json'))
    if not barrier['all_recommendations_frozen']:
        raise ValueError('All recommendations must be frozen before verification')
    validate_recommendations(barrier['rows'], states, pilot)
    path = out/'results.json'
    if path.exists():
        return read(str(path))
    rows = copy.deepcopy(barrier['rows']); state_by_name = {s['name']: s for s in states}
    checkpoint = out/'verification_rows.json'; saved = read(str(checkpoint))['rows'] if checkpoint.exists() else []
    for row in rows:
        prior = next((r for r in saved if key(r) == key(row)), None)
        if prior is not None:
            if any(prior[k] != v for k, v in row.items()):
                raise ValueError('Saved verifier belongs to a different frozen recommendation')
            row.update(deployed=prior['deployed'], verification=prior['verification'])
            continue
        candidates = [tuple(p) for p in next(b for b in frozen['banks'] if b['layout'] == row['layout'])['policies']]
        problem = problem_for(public, frozen['tasks'][row['task_index']]); counter = [0]; simulator = problem.simulate
        def counted(point, rng=None):
            counter[0] += 1
            return simulator(point, rng)
        problem.simulate = counted
        deployed, result = execute_terminal(problem, candidates, row['action'], row['search_calls'],
            row['verification_seed'], state_by_name[row['state_name']]['terminal_protocol'], spec['total_target_call_cap'])
        if counter[0] != result['verification_budget']:
            raise RuntimeError('Physical verifier calls differ from the saved call accounting')
        row.update(deployed=deployed, verification=result)
        save(checkpoint, {'rows': [r for r in rows if 'verification' in r]})
    # All verification finishes before truth is opened; no scoring enters decisions.
    for row in rows:
        state = state_by_name[row['state_name']]
        candidates = [tuple(p) for p in next(b for b in frozen['banks'] if b['layout'] == row['layout'])['policies']]
        problem = problem_for(public, frozen['tasks'][row['task_index']])
        truth = np.array([[problem.true_objective(p), np.log(problem.true_sigma(p)[0]**2),
            problem.true_constraint_mean(p)-problem.tau, np.log(problem.true_sigma(p)[1]**2)] for p in candidates])
        feasible = np.array([problem.is_truly_feasible(p) for p in candidates])
        guarded = state['terminal_actions'][:8]
        terminal = ConditionalTerminalActions([a['shortlist'] for a in guarded],
            [a['incumbent_position'] for a in guarded], state['terminal_protocol'])
        safe, costs = safe_terminal_values(terminal, truth, feasible, spec['total_target_call_cap']-row['search_calls'])
        deployed = row['deployed']; certified = row['verification']['certified']; a = row['action_index']
        quality = float(np.clip(1-truth[deployed, 0]/terminal.protocol['objective_reference'], 0, 1))
        row['scores'] = {'actual_safe_reward': float(certified*feasible[deployed]*quality),
            'false_certificate': bool(certified and not feasible[deployed]), 'rejection': not certified,
            'actual_total_calls': row['verification']['total_evaluation_count'],
            'conditional_safe_reward': float(safe[a]), 'conditional_total_calls': float(row['search_calls']+costs[a])}
    searches = sum(read(str(p))['physical_calls'] for p in out.glob('*/search_observation.json'))
    terminal_calls = sum(r['verification']['verification_budget'] for r in rows)
    cost = {'shared_physical_initialization_calls': 2*pilot['new_task_count'], 'physical_search_calls': searches,
        'physical_terminal_calls': terminal_calls, 'physical_total_calls': 2*pilot['new_task_count']+searches+terminal_calls,
        'logical_total_calls': sum(r['scores']['actual_total_calls'] for r in rows), 'source_fits': 0}
    if cost['physical_total_calls'] > spec['maximum_physical_calls']:
        raise RuntimeError('Independent pilot exceeded registered physical budget')
    result = {'execution_plan': plan, 'rows': rows, 'cost': cost,
        'paired_comparisons': paired_summary(rows, frozen, pilot, spec['bootstrap_draws']),
        'independent_outcomes_opened': True, 'independent_deployment_evaluated': True,
        'source_advantage_conclusion': 'Pending paired-outcome analysis',
        'limitations': ['Six same-population d50 finite-bank tasks; no cross-family or free-optimization claim.',
            'Earlier paid moment projections and diagnostic rather than certified KG query rankings remain.',
            'Deployment evaluation does not by itself establish stable source-prior superiority.']}
    save(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'verify'])
    parser.add_argument('--plan', default=PLAN)
    args = parser.parse_args()
    started = time.monotonic()
    print('SCHEDULEURM_PROGRESS {"current":0,"total":1,"unit":"stage","elapsed_seconds":0}', flush=True)
    result = freeze(args.plan) if args.stage == 'freeze' else verify(args.plan)
    print(args.stage, 'recommendations', len(result['rows']), flush=True)
    if args.stage == 'freeze' and not result['selected_recommendations_complete']:
        raise SystemExit('Selected recommendations incomplete; retain saved observations and quadrature')
    print('SCHEDULEURM_PROGRESS '+json.dumps({'current':1,'total':1,'unit':'stage',
        'elapsed_seconds':time.monotonic()-started}), flush=True)
    print('DONE', flush=True)
