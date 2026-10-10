"""Historical GP controls on the current source roster and raw target contract."""

import copy
import signal
import time

import numpy as np
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_mixture_lookahead import predictive_values
from core.positive_raw_lookahead import evaluate_raw_futures
from core.positive_raw_terminal import raw_terminal_component
from core.stratified_raw_lookahead import stratified_record_weights, summarize_stratified_futures
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_prior_posterior import snapshot, problem_for
from performance.run_legacy_joint_population_matched_baselines import gp_kernel, paired_fields
from representation.joint_response_mixture import JointResponseMixture
from representation.source_known_noise_gp import fit_known_noise_gp
from representation.source_kernel_population import (
    fit_unequal_site_kernel_population, fit_joint_common_unequal_site_kernel_population,
    predict_unequal_site_kernel_population,
)

PLAN = 'performance/manifests/historical_matched_gp_connection_v1.json'


def source_tasks_for_layout(spec, layout):
    """Use precisely the existing source rows; risk coordinates need the new map."""
    selection = read(spec['selection_plan']); old = read(spec['legacy_connection_plan'])
    inputs = read(selection['source_inputs_by_layout'][layout])
    tasks = [t for t in inputs['source_tasks'] if t['name'] in selection['source_tasks']]
    assert [t['name'] for t in tasks] == selection['source_tasks']
    assert old['development_task'] not in selection['source_tasks']
    covariance = read(spec['covariance_plan'])
    field = next(r for r in read(covariance['source_fields'])['rows'] if
        (r['layout'], r['active_arm']) == (layout, covariance['proposal_arm']))
    law = next(r for r in read(spec['risk_laws'])['rows'] if
        (r['layout'], r['arm']) == (layout, field['active_arm']))
    reference = np.asarray(law['public_reference_psi'])
    risk = read(old['query_risk_inputs_by_layout'][layout])
    assert risk['training_task_ids'] == covariance['source_roster'] == [2, 3, 4, 5]
    np.testing.assert_allclose(reference, risk['public_reference_psi'], atol=1e-12, rtol=0.)
    groups = [[], []]
    for task, index in zip(tasks, covariance['source_roster']):
        sites = [s['policy'] for s in field['sites'] if s['source_task'] == index]
        assert sites == inputs['execution']['source_site_indices_by_task'][index]
        groups[0].append(dict(task, features=np.asarray(task['psi'])))
        groups[1].append(dict(task, features=reference[sites]))
    return groups, inputs, risk


def fit_source_controls(groups, optimizer):
    heads = []
    for head in range(4):
        tasks = groups[int(head >= 2)]
        anchor = fit_known_noise_gp(tasks, head, max_iterations=optimizer['max_iterations'])
        fixed = fit_unequal_site_kernel_population(tasks, head, anchor, optimizer=optimizer)
        joint = fit_joint_common_unequal_site_kernel_population(tasks, head, fixed, optimizer=optimizer)
        heads.append({'anchor': anchor, 'fixed_common': fixed, 'joint_common': joint})
        print('matched source GP head', head, 'converged',
            [r['optimizer_converged'] for r in (anchor, fixed, joint)], flush=True)
    return {'heads': heads, 'source_task_ids': [t['name'] for t in groups[0]],
        'source_label_count_per_head': sum(len(t['response']) for t in groups[0])}


def controls_from_fit(fit, mean_coordinates, risk_coordinates, sigma_level):
    moments = {'source_common_task_rbf': [], 'source_task_rbf': [], 'fixed_rbf_shared_coordinates': []}
    for head, record in enumerate(fit['heads']):
        x = mean_coordinates if head < 2 else risk_coordinates; d = record['anchor']
        moments['source_common_task_rbf'].append(predict_unequal_site_kernel_population(x, record['joint_common']))
        scale = d['normalization']['scale']; center = d['normalization']['mean']
        covariance = gp_kernel(x, d['lengthscale'], d['signal_variance'])+d['constant_variance']
        covariance += d['discrepancy_variance']*np.eye(len(x))
        moments['source_task_rbf'].append((np.full(len(x), center+scale*d['constant']), scale**2*covariance))
        fixed_mean = 0. if head < 2 else np.log(sigma_level**2)
        moments['fixed_rbf_shared_coordinates'].append((np.full(len(x), fixed_mean), gp_kernel(x, np.ones(x.shape[1]), 4.)))
    return {name: JointResponseMixture([name], [paired_fields(heads)]) for name, heads in moments.items()}


def terminal_rows(model, terminal, remaining, powers, seed):
    rows = []
    for budget in (remaining, remaining-1):
        for power in powers:
            values, costs, actions = model.terminal_values(terminal, budget, power, seed)
            best = int(np.lexsort((costs, -values))[0])
            rows.append({'remaining_calls': budget, 'power': power, 'values': values.tolist(),
                'costs': costs.tolist(), 'actions': actions, 'best_action_index': best,
                'best_action': actions[best], 'best_reward': float(values[best])})
    return rows


def refine_primary_records(model, terminal, remaining, query, records, actions, powers, seed, checkpoint):
    """Same three-action delta used for the learned prior,without extra fantasies."""
    action = next(a for a in actions if a['kind'] == 'guarded' and a['primary'] == query)
    subset = ConditionalTerminalActions([action['shortlist']], [action['incumbent_position']], terminal.protocol)
    for record in records:
        if record.get('primary_mean_refined'):
            continue
        future = copy.deepcopy(model); scores = future.update_raw(query, record['labels'])['head_log_scores']
        np.testing.assert_allclose(future.weights, record['shared_identity_weights'], atol=1e-15)
        for power in powers:
            nodes = qmc.Sobol(7, scramble=True, seed=seed).random_base2(power)
            args = (model.components[0], future.components[0], subset, remaining-1,
                query, record['labels'], scores[0], nodes)
            old, old_cost, small_actions, _ = raw_terminal_component(*args)
            new, new_cost, new_actions, _ = raw_terminal_component(*args, guarded_mean_position=0)
            assert new_actions == small_actions
            indices = [actions.index(a) for a in small_actions]
            value = np.asarray(record['raw'][str(power)]['values'])
            cost = np.asarray(record['raw'][str(power)]['costs'])
            np.testing.assert_allclose(value[indices], old, atol=1e-12, rtol=1e-10)
            value[indices[:3]] += (new-old)[:3]; cost[indices[:3]] += (new_cost-old_cost)[:3]
            record['raw'][str(power)].update(values=value.tolist(), costs=cost.tolist())
        record['primary_mean_refined'] = True
        checkpoint(records, actions)


def summarize_query(records, model, terminal_values, actions, uniforms, spec):
    before, after = [next(r for r in terminal_values if r['remaining_calls'] == budget and
        r['power'] == spec['raw_powers'][-1]) for budget in (360, 359)]
    baseline = before['best_reward'], after['best_reward'], after['best_action_index']
    rows = []
    for prefix in spec['complement_prefixes']:
        for tail_count in spec['tail_counts']:
            selected, weights, regions = stratified_record_weights(records, model.weights, uniforms, [0],
                spec['low_constraint_probability'], tail_count, prefix)
            current = summarize_stratified_futures(selected, weights, regions, actions, baseline, spec['raw_powers'], 0.)
            contrast = [np.asarray([max(r['raw'][str(p)]['values'])-r['raw'][str(p)]['values'][baseline[-1]]
                for r in selected]) for p in spec['raw_powers']]
            for row in current:
                row.update(tail_count=tail_count, complement_prefix=prefix,
                    weighted_absolute_terminal_contrast_difference=float(weights @ abs(contrast[-1]-contrast[0])),
                    verification_budget_opportunity_cost=before['best_reward']-after['best_reward'])
            rows.extend(current)
    finest = rows[-1]
    tail = next(r for r in rows if (r['complement_prefix'], r['tail_count'], r['raw_draw_power']) ==
        (spec['complement_prefixes'][-1], spec['tail_counts'][-2], spec['raw_powers'][-1]))
    bulk = next(r for r in rows if (r['complement_prefix'], r['tail_count'], r['raw_draw_power']) ==
        (spec['complement_prefixes'][-2], spec['tail_counts'][-1], spec['raw_powers'][-1]))
    changes = {'terminal_absolute_contrast': finest['weighted_absolute_terminal_contrast_difference'],
        'tail_adaptation': abs(finest['raw_adaptation_value']-tail['raw_adaptation_value']),
        'complement_adaptation': abs(finest['raw_adaptation_value']-bulk['raw_adaptation_value'])}
    return {'rows': rows, 'numerical_changes': changes, 'numerical_gates_passed':
        changes['terminal_absolute_contrast'] <= spec['maximum_terminal_contrast_difference'] and
        max(changes['tail_adaptation'], changes['complement_adaptation']) <= spec['maximum_fantasy_difference']}


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result_path = out/'results.json'
    previous = read(str(result_path.relative_to(ROOT))) if result_path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    assert read(str((out/'unit_tests.json').relative_to(ROOT)))['successful']
    started = time.process_time(); stopped = False; fits = {}; rows = []; lookahead = []
    source_fit_path = out/'source_fit.json'
    if source_fit_path.exists():
        fits = read(str(source_fit_path.relative_to(ROOT)))['by_layout']
    old = read(spec['legacy_connection_plan']); cached = read(read(spec['connection_plan'])['development_states'])['rows']
    reference = read(spec['reference_terminal'])['rows']; groups = {}
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        # All fits use tasks2-5 only and freeze before any paid target label is indexed.
        for state in cached:
            layout = state['layout']; groups[layout] = source_tasks_for_layout(spec, layout)
            if layout not in fits:
                fits[layout] = fit_source_controls(groups[layout][0], spec['optimizer'])
                save(source_fit_path, {'execution_plan': plan, 'by_layout': fits})
        all_fits_converged = all(r['optimizer_converged'] for fit in fits.values()
            for head in fit['heads'] for r in head.values())
        if not all_fits_converged:
            raise RuntimeError('Registered GP source fit incomplete; no target forecasts or retries')
        public = read(old['public_inputs'])
        jobs = []
        for state in cached:
            layout = state['layout']; _, inputs, risk = groups[layout]
            query = next(t for t in inputs['source_tasks'] if t['name'] == old['development_task'])
            mean_x = np.asarray(query['psi'])[:old['candidate_count']]
            risk_x = np.asarray(risk['heldout_public_psi'])[state['candidate_bank_indices']]
            provider = problem_for(public['plan'], public['tasks'][old['development_task_index']])
            controls = controls_from_fit(fits[layout], mean_x, risk_x, provider.sigma_level)
            guarded = state['terminal_actions'][:old['candidate_count']]
            terminal = ConditionalTerminalActions([a['shortlist'] for a in guarded],
                [a['incumbent_position'] for a in guarded], state['terminal_protocol'])
            for arm in spec['arms']:
                model = controls[arm]; prior = snapshot(model)
                for candidate, labels in enumerate(state['paid_observations']):
                    model.update_raw(candidate, labels)
                assert model.paid_episodes == old['paid_prefix'] == 2
                values = terminal_rows(model, terminal, old['total_target_call_cap']-model.paid_episodes,
                    spec['terminal_powers'], spec['seed'])
                for r in values:
                    source = next(v for v in reference if (v['layout'], v['remaining_calls'], v['power']) ==
                        (layout, r['remaining_calls'], r['power']))
                    assert r['actions'] == source['actions'] and len(r['actions']) == 40
                    r['own_model_gap_for_learned_choice'] = r['best_reward']-r['values'][source['best_action_index']]
                    r['learned_model_gap_for_this_choice'] = source['best_reward']-source['values'][r['best_action_index']]
                rows.append({'layout': layout, 'arm': arm, 'source_task_ids': fits[layout]['source_task_ids'],
                    'candidate_bank_indices': state['candidate_bank_indices'], 'paid_observations': state['paid_observations'],
                    'paid_variance_labels': 0, 'prior': prior, 'posterior': snapshot(model), 'terminal': values})
                if arm == spec['lookahead_arm']:
                    jobs.append((layout, model, terminal, values))
                print('matched GP connected', layout, arm, 'choice', values[-2]['best_action'], flush=True)
        save(out/'connected_states.json', {'execution_plan': plan, 'rows': rows,
            'all_source_fits_frozen_before_target_updates': True})
        uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['predictive_power'])
        positions = np.union1d(np.flatnonzero(uniforms[:, 1] < spec['low_constraint_probability'])[:max(spec['tail_counts'])],
            np.flatnonzero(uniforms[:max(spec['complement_prefixes']), 1] >= spec['low_constraint_probability']))
        for layout, model, terminal, values in jobs:
            query = spec['queries'][layout]; path = out/f'layout{layout}_futures.json'
            saved = read(str(path.relative_to(ROOT))) if path.exists() else {}
            records = saved.get('records', [])
            fantasies, _ = predictive_values(model, query, spec['predictive_power'], spec['seed']+1)
            def checkpoint(records, actions):
                save(path, {'layout': layout, 'query': query, 'arm': spec['lookahead_arm'],
                    'records': records, 'actions': actions, 'source_weights': model.weights.tolist()})
            records, actions = evaluate_raw_futures(model, terminal, 360, query, fantasies[positions],
                raw_powers=spec['raw_powers'], projected_power=spec['raw_powers'][-1], seed=spec['seed'],
                completed_records=records, checkpoint=checkpoint,
                fantasy_coordinates=[(0, int(p)) for p in positions])
            refine_primary_records(model, terminal, 360, query, records, actions, spec['raw_powers'], spec['seed'], checkpoint)
            summary = summarize_query(records, model, values, actions, uniforms, spec)
            lookahead.append({'layout': layout, 'candidate': query, 'arm': spec['lookahead_arm'], **summary})
            save(out/'lookahead.json', {'execution_plan': plan, 'rows': lookahead})
            print('matched GP raw KG', layout, query, summary['rows'][-1]['raw_net_reward_control_variate'],
                'gates', summary['numerical_gates_passed'], flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'all_registered_cases_complete': len(rows) == 6 and len(lookahead) == 2,
        'source_fits_converged': all(r['optimizer_converged'] for fit in fits.values() for head in fit['heads'] for r in head.values()),
        'source_fit_count': 12*len(fits), 'connected_control_count': len(rows), 'lookahead_query_count': len(lookahead),
        'lookahead_numerical_gates_passed': len(lookahead) == 2 and all(r['numerical_gates_passed'] for r in lookahead),
        'lookahead': lookahead, 'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started,
        'cpu_budget_stopped': stopped, 'cost': spec['cost'], 'independent_confirmation': False,
        'decision_ranking_certified': False, 'scope': spec['scope'],
        'limitations': ['Common/task GP uses a Gaussian source-label approximation at source-only ARD lengthscales.',
            'Fixed GP shares frozen source-derived coordinates; it is not a source-free representation control.',
            'Two paid updates retain Gaussian moment projection; only the next raw posterior is unprojected.',
            'Reciprocal valuations and KG estimates are model expectations, not deployment-performance comparisons.',
            'Only one fixed query per layout is compared; no complete query ranking or independent advantage.']}
    save(result_path, result); return result


if __name__ == '__main__':
    r = run()
    print({k: r[k] for k in ('all_registered_cases_complete', 'source_fits_converged',
        'lookahead_numerical_gates_passed', 'cpu_seconds', 'cpu_budget_stopped')})
