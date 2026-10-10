"""Reuse historical joint KG precision methods at the actual learned priors."""

import copy
import signal
import time

import numpy as np

from core.mixture_guarded_kg import mixture_raw_lookahead
from core.positive_mixture_lookahead import fixed_action_factorized_reference
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments

PLAN = 'performance/manifests/historical_learned_prior_kg_precision_v1.json'


def check_batch_filter(model, query, values):
    """One actual cached value checks the new law's full-field fantasy completion."""
    errors = []
    for pair in model.components:
        for head, field in enumerate(pair):
            sequential = copy.deepcopy(field); batched = copy.deepcopy(field)
            label = float(values[head])
            score = sequential.update_batch(query, label, None, 1)
            moments, gain = raw_fantasy_moments(field, query, [label])
            batch_score = apply_raw_moments(batched, query, moments[0], gain)
            error = max(abs(score - batch_score),
                float(np.max(abs(sequential.means - batched.means))),
                float(np.max(abs(sequential.covariances - batched.covariances))),
                float(np.max(abs(sequential.log_weights - batched.log_weights))))
            assert sequential.paid_episodes == batched.paid_episodes == field.paid_episodes + 1
            assert error < 1e-8
            errors.append(error)
    return max(errors)


def reference_passed(row, precision):
    return bool(row['predictive_log_evidence_max_error'] <= precision['maximum_log_evidence_error']
        and row['effective_sample_size_min'] >= precision['minimum_reference_ESS']
        and abs(row['likelihood_tower_drift']) <= precision['maximum_absolute_tower_drift']
        and abs(row['projected_baseline_drift']) <= precision['maximum_absolute_tower_drift']
        and row['maximum_raw_integration_relative_error'] <= precision['maximum_raw_integration_relative_error'])


def action_passed(row, reference, precision):
    return bool(reference['precision_gate_passed']
        and max(abs(v) for v in row['model_weight_martingale_residual']) <= precision['maximum_identity_martingale_residual']
        and row['maximum_raw_integration_relative_error'] <= precision['maximum_raw_integration_relative_error']
        and row['nested_adaptation_difference'] <= precision['maximum_nested_adaptation_difference'])


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    previous = (read(str(path.relative_to(ROOT))) if path.exists() else
        read(spec['resume_results']) if 'resume_results' in spec else {})
    if previous.get('all_registered_rules_complete'):
        return previous
    references, rows = previous.get('reference_rows', []), previous.get('rows', [])
    batch_checks = previous.get('batch_checks', []); stopped = []
    started = time.process_time(); precision = spec['precision']
    assert precision == read(spec['historical_precision_plan'])['precision']
    saved = read(spec['connected_states'])['rows']
    connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    baselines = {}
    for state, model, terminal, remaining in models:
        original = next(r for r in saved if r['layout'] == state['layout'])
        assert state['identities'] == original['identities']
        assert state['candidate_bank_indices'] == original['candidate_bank_indices']
        assert state['paid_observations'] == original['paid_observations']
        assert state['action_sets'] == original['action_sets']
        assert state['action_sets']['4']['active_indices'] == spec['actions'][state['layout']]
        for key in ('prior', 'posterior'):
            np.testing.assert_allclose(state[key]['weights'], original[key]['weights'], atol=1e-12, rtol=1e-12)
            for new, old in zip(state[key]['components'], original[key]['components']):
                for moment in new:
                    np.testing.assert_allclose(new[moment], old[moment], atol=1e-12, rtol=1e-12)
        values, costs, actions = model.terminal_values(terminal, remaining, spec['terminal_power'], spec['seed'])
        debited, _, future = model.terminal_values(terminal, remaining - 1, spec['terminal_power'], spec['seed'])
        baselines[state['layout']] = (values, costs, actions, debited, future)

    def record():
        latest = {}
        for row in rows:
            latest[row['layout'], row['candidate']] = row
        comparison = []
        for layout, candidates in spec['actions'].items():
            group = [latest[layout, q] for q in candidates if (layout, q) in latest]
            best = max(group, key=lambda r: r['net_reward_of_paid_episode']) if group else None
            comparison.append({'layout': layout, 'registered_action_count': len(candidates),
                'completed_action_count': len(group), 'actions_precision_passed': sum(r['precision_gate_passed'] for r in group),
                'provisional_best_candidate': best['candidate'] if best else None,
                'provisional_best_net': best['net_reward_of_paid_episode'] if best else None,
                'all_actions_precision_passed': len(group) == len(candidates) and all(r['precision_gate_passed'] for r in group)})
        complete = len(latest) == 12
        exhausted = complete and all(r['precision_gate_passed'] or
            r['joint_fantasy_power'] == spec['joint_fantasy_powers'][-1] for r in latest.values())
        result = {'execution_plan': plan, 'rows': rows, 'reference_rows': references, 'batch_checks': batch_checks,
            'comparison': comparison, 'all12_registered_actions_complete': complete,
            'all_registered_rules_complete': exhausted, 'actions_precision_passed': sum(r['precision_gate_passed'] for r in latest.values()),
            'stopped_actions': stopped, 'cpu_seconds_this_execution': time.process_time() - started,
            'previous_stopped_actions': previous.get('stopped_actions', []),
            'cpu_seconds': previous.get('cpu_seconds', 0.) + time.process_time() - started,
            'cost': spec['cost'], 'independent_confirmation': False, 'scientific_model_ranking_authorized': False,
            'scope': spec['scope']}
        save(path, result)
        return result

    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    budget_stopped = False
    try:
        for power in spec['joint_fantasy_powers']:
            for state, model, terminal, remaining in models:
                layout = state['layout']; baseline = baselines[layout]
                values, costs, actions, debited, future = baseline
                fixed = future[int(debited.argmax())]
                for query in spec['actions'][layout]:
                    known = [r for r in rows if r['layout'] == layout and r['candidate'] == query]
                    if any(r['precision_gate_passed'] or r['joint_fantasy_power'] == power for r in known):
                        continue
                    if power > spec['joint_fantasy_powers'][0] and not known:
                        continue
                    available = spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)
                    if available <= 0:
                        budget_stopped = True; break
                    signal.setitimer(signal.ITIMER_PROF, min(available, spec['maximum_single_thread_cpu_seconds_per_action']))
                    phase = 'batch_filter'; one = time.process_time()
                    try:
                        if not any(r['layout'] == layout and r['candidate'] == query for r in batch_checks):
                            error = check_batch_filter(model, query, state['paid_observations'][0])
                            batch_checks.append({'layout': layout, 'candidate': query,
                                'full_field_and_evidence_max_error': error, 'head_count': 16})
                            record()
                        phase = 'fixed_action_reference'
                        registered = [r for r in references if r['layout'] == layout and r['candidate'] == query]
                        reference = next((r for r in registered if r['precision_gate_passed']), None)
                        if reference is None:
                            for ref_power in spec['reference_fantasy_powers']:
                                reference = next((r for r in registered if r['fantasy_power'] == ref_power), None)
                                if reference is None:
                                    reference = fixed_action_factorized_reference(model, fixed, query,
                                        draw_power=spec['reference_draw_power'], terminal_power=spec['terminal_power'],
                                        fantasy_power=ref_power, seed=spec['seed'], protocol=terminal.protocol)
                                    reference.update(layout=layout, precision_gate_passed=reference_passed(reference, precision))
                                    references.append(reference); record()
                                if reference['precision_gate_passed']:
                                    break
                        np.testing.assert_allclose(reference['projected_prior_reward'], debited.max(), atol=1e-12, rtol=0.)
                        phase = 'joint_adaptive_contrast'
                        result = mixture_raw_lookahead(model, terminal, remaining, [query],
                            draw_power=spec['terminal_power'], fantasy_power=power, seed=spec['seed'],
                            baseline=baseline, fixed_action_expectations={query: reference['projected_expected_reward']})
                        row = {'layout': layout, 'stop_expected_reward': result['stop_expected_reward'], **result['lookahead'][0]}
                        prefix_error = None
                        preceding = next((r for r in known if r['joint_fantasy_power'] == power - 1), None)
                        if preceding:
                            prefix_error = abs(row['nested_prefix_adaptation_value'] - preceding['adaptation_value'])
                            assert prefix_error <= spec['maximum_nested_prefix_reproduction_error']
                        row.update(precision_gate_passed=action_passed(row, reference, precision),
                            fixed_action_reference_precision_passed=reference['precision_gate_passed'],
                            reference_fantasy_power=reference['fantasy_power'], joint_fantasy_power=power,
                            nested_prefix_reproduction_error=prefix_error, cpu_seconds=time.process_time() - one,
                            remaining_calls=remaining, future_remaining_calls=remaining - 1,
                            source_law='actual final learned source amplitudes and frozen source weights')
                        rows.append(row); record()
                        print('learned KG', layout, query, 'fantasies', 2**power, 'net', round(row['net_reward_of_paid_episode'], 6),
                            'nested', round(row['nested_adaptation_difference'], 6), 'pass', row['precision_gate_passed'], flush=True)
                    except TimeoutError:
                        stopped.append({'layout': layout, 'candidate': query, 'fantasy_power': power,
                            'phase': phase, 'reason': 'registered CPU cap'})
                        record()
                        budget_stopped = time.process_time() - started >= spec['maximum_single_thread_cpu_seconds'] - .1
                    finally:
                        signal.setitimer(signal.ITIMER_PROF, 0.)
                    if budget_stopped:
                        break
                if budget_stopped:
                    break
            if budget_stopped:
                break
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return record()


if __name__ == '__main__':
    result = run()
    print({k: result[k] for k in ('all12_registered_actions_complete', 'all_registered_rules_complete',
        'actions_precision_passed', 'cpu_seconds', 'stopped_actions')})
