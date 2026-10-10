"""Reconnect frozen actual source laws and weights to the historical KG state."""

import signal
import time

import numpy as np
from scipy.special import digamma

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save, cpu_timeout
from performance.connect_complete_historical_reuse import restore_head, check_scalar_batched
from performance.connect_historical_prior_posterior import (
    JointResponseMixture, BudgetedTerminalActions, posterior_mixture_weights,
    evaluate_or_replicate_action_set, problem_for, snapshot,
)
from performance.continue_historical_raw_source_learning import load_mixture
from performance.learn_historical_raw_source_amplitudes import covariance_builder

PLAN = 'performance/manifests/historical_learned_prior_kg_connection_v1.json'


def learned_law(original, scales):
    """Scale the common/task populations once, keeping point discrepancy fixed."""
    law = dict(original)
    law['source_covariance'] = covariance_builder(original)(scales)[0].tolist()
    law['unconditioned_common_covariance'] = (
        scales[0] * np.asarray(original['unconditioned_common_covariance'])).tolist()
    law['task_covariance'] = (scales[1] * np.asarray(original['task_covariance'])).tolist()
    return law


def restore_learned_source_prior(layout, candidate_bank_indices, plan=PLAN):
    """Restore positive source components on frozen public policy coordinates."""
    spec = read(plan)
    parent = read(spec['source_plan'])
    selection = read(parent['selection_plan']); covariance = read(parent['covariance_plan'])
    laws = {tuple(r[k] for k in (*IDENTITY, 'head')): r for r in read(parent['source_laws'])['rows']}
    learning = read(spec['source_learning']); precision = read(spec['source_precision'])
    weights = read(spec['source_structure_weights'])
    assert learning['source_learning_converged'] and precision['source_learning_converged']
    assert weights['source_only'] and weights['new_target_observations_used'] == 0
    assert weights['source_learning'] == spec['source_learning']
    assert weights['source_precision'] == spec['source_precision']
    cases = {tuple(r[k] for k in (*IDENTITY, 'head')): r for r in learning['evidence_rows']}
    records = {tuple(r[k] for k in (*IDENTITY, 'head')): r for r in precision['rows']}
    fields = read(covariance['source_fields'])['rows']; maps = read(spec['maps'])['rows']
    inputs = read(selection['source_inputs_by_layout'][layout])
    tasks = [t for t in inputs['source_tasks'] if t['name'] in selection['source_tasks']]
    assert [t['name'] for t in tasks] == selection['source_tasks']
    response = np.vstack([t['response'] for t in tasks])
    old = read(spec['legacy_connection_plan']); risk = read(old['query_risk_inputs_by_layout'][layout])
    coordinates = (np.asarray(inputs['public_mean_reference'])[candidate_bank_indices],
        np.asarray(risk['public_reference_psi'])[candidate_bank_indices])
    block = weights['by_layout'][layout]; identities = block['identities']
    source_weights = np.asarray(block['equal_prior_source_posterior_weights'])
    assert len(identities) == 8 and np.all(source_weights > 0)
    template = next(r for r in fields if (r['layout'], r['active_arm']) == (layout, covariance['proposal_arm']))
    names, components, checks = [], [], []
    for identity in identities:
        pair = []
        for head in ('f', 'g'):
            key = (*identity, head); case, record = cases[key], records[key]
            assert record['positive_mixture_precision_passed']
            law = learned_law(laws[key], case['f_common_task_scales'])
            eta_key = (*identity, 'eta_f' if head == 'f' else 'eta_g')
            eta_law = learned_law(laws[eta_key], case['eta_common_task_scales'])
            eta, masses = load_mixture(record, case)
            saved = {'positive_eta_mixture': {'eta_states_by_chain': eta.tolist(),
                'positive_component_weights': masses.reshape(-1).tolist()}}
            source = dict(template); h = int(head == 'g')
            source['batch_means'] = response[:, h].tolist()
            source['sample_variances'] = np.exp(response[:, 2 if h else 3] + digamma(1.)).tolist()
            field, projection_error = restore_head(law, eta_law, source, saved, coordinates, maps, selection)
            np.testing.assert_allclose(np.exp(field.log_weights), masses.reshape(-1), atol=1e-15, rtol=1e-12)
            checks.append({**dict(zip(IDENTITY, identity)), 'head': head,
                'response_common_task_scales': case['f_common_task_scales'],
                'risk_common_task_scales': case['eta_common_task_scales'],
                'source_component_file': record['positive_mixture_component_file'],
                'retained_source_components': len(field.means),
                'common_projection_max_error': projection_error, 'source_component_weights_retained': True})
            pair.append(field)
        names.append('/'.join(identity[1:])); components.append(tuple(pair))
    mixture = JointResponseMixture(names, components, prior_weights=source_weights)
    np.testing.assert_allclose(mixture.weights, source_weights, atol=1e-15, rtol=1e-12)
    assert mixture.paid_episodes == 0 and not mixture.history
    return mixture, identities, checks


def build_connection(plan=PLAN):
    """Restore full fields in memory for acquisition calls,without changing artifacts."""
    spec = read(plan); started = time.process_time(); old = read(spec['legacy_connection_plan'])
    states = read(spec['development_states'])['rows']
    raw = read(old['source_data']); public = read(old['public_inputs'])
    rows, checks, models = [], [], []
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)))
    stopped = False
    try:
        for cached in states:
            layout = cached['layout']
            mixture, identities, source_checks = restore_learned_source_prior(layout, cached['candidate_bank_indices'], plan)
            source_weights = np.asarray(read(spec['source_structure_weights'])['by_layout'][layout]['equal_prior_source_posterior_weights'])
            observations = np.asarray(cached['paid_observations'])
            assert len(observations) == old['paid_prefix'] == 2
            for check, field in zip(source_checks, [f for pair in mixture.components for f in pair]):
                check['scalar_batched_raw_update_max_error'] = check_scalar_batched(field, 0, observations[0, int(check['head'] == 'g')])
            checks.extend(source_checks)
            prior = snapshot(mixture)
            for candidate, value in enumerate(observations):
                before = mixture.weights.copy(); record = mixture.update_raw(candidate, value)
                _, expected = posterior_mixture_weights(before, record['joint_log_scores'])
                np.testing.assert_allclose(mixture.weights, expected, atol=1e-12, rtol=1e-12)
            assert mixture.paid_episodes == 2
            assert all(f.paid_episodes == g.paid_episodes == 2 for f, g in mixture.components)
            guarded = cached['terminal_actions'][:old['candidate_count']]
            terminal = BudgetedTerminalActions([a['shortlist'] for a in guarded],
                [a['incumbent_position'] for a in guarded], cached['terminal_protocol'])
            remaining = old['total_target_call_cap'] - mixture.paid_episodes
            values, costs, actions = mixture.terminal_values(terminal, remaining, old['terminal_draw_power'], old['seed'])
            debited, _, future_actions = mixture.terminal_values(terminal, remaining - 1, old['terminal_draw_power'], old['seed'])
            assert len(actions) == len(future_actions) == 40
            reserve = 2 * cached['terminal_protocol']['comparison_budget']
            assert max(sum(a['budgets']) + (reserve if a['kind'] == 'guarded' else 0) for a in actions) == remaining
            assert max(sum(a['budgets']) + (reserve if a['kind'] == 'guarded' else 0) for a in future_actions) == remaining - 1
            priorities = np.full(old['candidate_count'], np.inf)
            for value, action in zip(values, actions):
                priorities[action['primary']] = min(priorities[action['primary']], -float(value))
            candidates = [tuple(raw['bank'][i]) for i in cached['candidate_bank_indices']]
            problem = problem_for(public['plan'], public['tasks'][old['development_task_index']])
            observed = [(candidates[i], v) for i, v in enumerate(observations)]
            action_sets = {str(k): evaluate_or_replicate_action_set(candidates, observed, problem,
                iteration=2, seed=old['seed'], new_action_count=k,
                new_action_policy='canonical_plus_posterior', new_action_priority=priorities) for k in (1, 4)}
            assert set(action_sets['1']['active_indices']) <= set(action_sets['4']['active_indices'])
            np.testing.assert_array_equal(action_sets['1']['replicate_mask'], action_sets['4']['replicate_mask'])
            assert len(action_sets['4']['new_action_indices']) == 4
            row = {'layout': layout, 'identities': identities, 'source_prior_weights': source_weights.tolist(),
                'candidate_bank_indices': cached['candidate_bank_indices'], 'paid_observations': observations.tolist(),
                'prior': prior, 'posterior': snapshot(mixture), 'terminal_protocol': cached['terminal_protocol'],
                'remaining_calls': remaining, 'future_remaining_calls': remaining - 1,
                'terminal_values': values.tolist(), 'terminal_costs': costs.tolist(), 'terminal_actions': actions,
                'debited_terminal_values': debited.tolist(), 'future_terminal_actions': future_actions,
                'terminal_draw_power': old['terminal_draw_power'], 'terminal_seed': old['seed'],
                'action_sets': {k: {n: v.tolist() if isinstance(v, np.ndarray) else v for n, v in a.items()}
                    for k, a in action_sets.items()}, 'numerically_certified_decision_ranking': False}
            rows.append(row)
            models.append((row, mixture, terminal, remaining))
            print('learned prior connected', layout, 'heads', len(mixture.components) * 2,
                'paid', mixture.paid_episodes, 'balanced4', action_sets['4']['active_indices'], flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'all_registered_cases_complete': len(rows) == 2 and len(checks) == 32,
        'connected_heads': len(checks), 'connected_complete_identities': sum(len(r['identities']) for r in rows),
        'checks': checks, 'cpu_seconds': time.process_time() - started, 'cpu_budget_stopped': stopped,
        'cost': {'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0},
        'source_learning': spec['source_learning'], 'source_precision': spec['source_precision'],
        'source_structure_weights': spec['source_structure_weights'], 'independent_confirmation': False,
        'scope': 'Actual source law and source-weighted paid-filter/terminal/action interfaces. '
            'Terminal values use the historical coarse rule;new-law KG rankings and decision advantages remain untested.'}
    return result, {'execution_plan': plan, 'rows': rows}, models


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all_registered_cases_complete'):
        return read(str(path.relative_to(ROOT)))
    result, states, _ = build_connection(plan)
    save(out / 'connected_states.json', states)
    save(path, result)
    return result


if __name__ == '__main__':
    result = run()
    print({k: result[k] for k in ('all_registered_cases_complete', 'connected_heads',
        'connected_complete_identities', 'cpu_seconds', 'cpu_budget_stopped')})
