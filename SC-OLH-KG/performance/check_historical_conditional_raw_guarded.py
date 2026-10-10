"""Compare frozen guarded head banks with an unprojected next-episode law."""

import copy
import signal
import time

import numpy as np
from scipy.stats import qmc

from core.conditional_raw_reference import raw_guarded_head_statistics
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments

PLAN = 'performance/manifests/historical_conditional_raw_guarded_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; checkpoint_path = out/'head_fantasies.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    records = read(str(checkpoint_path.relative_to(ROOT)))['rows'] if checkpoint_path.exists() else []
    if not records and 'completed_results' in spec:
        completed = read(spec['completed_results']); assert completed['all_registered_cases_complete']
        completed_plan = read(completed['execution_plan'])
        for key in ('connection_plan', 'projected_reference', 'projected_heads', 'layout', 'candidate',
                'identity_indices', 'seed', 'maximum_retained_identity_expectation_error',
                'maximum_projection_difference_resolution_change'):
            assert spec[key] == completed_plan[key]
        records = read(spec['completed_head_fantasies'])['rows']
    started = time.process_time(); stopped = False; rows = []
    reference = read(spec['projected_reference']); assert reference['all_registered_rules_complete']
    parent = read(reference['execution_plan'])
    for key in ('connection_plan', 'layout', 'candidate', 'seed'):
        assert spec[key] == parent[key]
    bank = read(spec['projected_heads']); action = bank['action']
    assert action['kind'] == 'guarded' and action['shortlist'][0] == spec['candidate']
    connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    _, model, terminal, _ = next(r for r in models if r[0]['layout'] == spec['layout'])
    np.testing.assert_array_equal(model.weights, bank['identity_weights'])
    retained = [i for i in range(len(model.components)) if i not in spec['identity_indices']]
    retained_mass = float(sum(model.weights[i] for i in retained))
    assert retained_mass <= spec['maximum_retained_identity_expectation_error']
    nodes = {p: qmc.Sobol(6, scramble=True, seed=spec['seed']).random_base2(p) for p in spec['draw_powers']}
    precision = read(spec['historical_precision_plan'])['precision']; count = 2**max(spec['fantasy_powers'])
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
    try:
        for identity in spec['identity_indices']:
            for head, field in enumerate(model.components[identity]):
                cached = next(r for r in bank['heads'] if (r['identity'], r['head']) == (identity, head))
                labels = cached['predictive_labels'][:count]; moments, gain = raw_fantasy_moments(field, spec['candidate'], labels)
                for position, (label, result) in enumerate(zip(labels, moments)):
                    future = None
                    for power in spec['draw_powers']:
                        if any((r['identity'], r['head'], r['position'], r['draw_power']) ==
                                (identity, head, position, power) for r in records):
                            continue
                        if future is None:
                            future = copy.deepcopy(field)
                            evidence = apply_raw_moments(future, spec['candidate'], result, gain)
                        value, diagnostics = raw_guarded_head_statistics(field, future, action, nodes[power],
                            label, evidence, head, terminal.protocol)
                        records.append({'identity': identity, 'head': head, 'position': position, 'draw_power': power,
                            'predictive_label': label, 'statistics': value.tolist(), **diagnostics,
                            'maximum_raw_integration_relative_error': future.max_integration_relative_error})
                        save(checkpoint_path, {'execution_plan': plan, 'rows': records})
                print('raw guarded head', model.names[identity], head, 'complete', len(records),
                    'CPU', round(time.process_time()-started, 2), flush=True)
        for power in spec['draw_powers']:
            for fantasy_power in spec['fantasy_powers']:
                n = 2**fantasy_power; rewards = []; active = []
                for identity in range(len(model.components)):
                    heads = []
                    for head in (0, 1):
                        if identity in spec['identity_indices']:
                            values = [r for r in records if (r['identity'], r['head'], r['draw_power']) ==
                                (identity, head, power) and r['position'] < n]
                            assert len(values) == n; active.extend(values)
                            heads.append(np.mean([r['statistics'] for r in values], axis=0))
                        else:
                            cached = next(r for r in bank['heads'] if (r['identity'], r['head']) == (identity, head))
                            heads.append(np.mean(cached['future_statistics'][:n], axis=0))
                    rewards.append(heads[0] @ heads[1])
                fixed = next(r for r in reference['rows'] if r['fantasy_power'] == fantasy_power)
                raw = float(model.weights @ rewards)
                evidence_error = max(abs(r['predictive_log_evidence_error']) for r in active)
                ess = min(r['effective_sample_size'] for r in active)
                rows.append({'draw_power': power, 'fantasy_power': fantasy_power, 'action': action,
                    'likelihood_expected_reward': raw, 'prior_fixed_reward': fixed['prior_fixed_reward'],
                    'likelihood_tower_drift': raw-fixed['prior_fixed_reward'],
                    'projected_expected_reward': fixed['expected_future_fixed_reward'],
                    'projected_minus_likelihood_expectation': fixed['expected_future_fixed_reward']-raw,
                    'predictive_log_evidence_max_error': evidence_error, 'effective_sample_size_min': ess,
                    'retained_identity_expectation_error_bound': retained_mass,
                    'importance_precision_passed': bool(evidence_error <= precision['maximum_log_evidence_error']
                        and ess >= precision['minimum_weight_ESS'])})
                preceding = next((r for r in rows if r['fantasy_power'] == fantasy_power and r['draw_power'] == power-2), None)
                if preceding:
                    rows[-1]['nested_draw_reward_difference'] = abs(raw-preceding['likelihood_expected_reward'])
                preceding = next((r for r in rows if r['draw_power'] == power and r['fantasy_power'] == fantasy_power-1), None)
                if preceding:
                    rows[-1]['nested_fantasy_projection_difference'] = abs(
                        rows[-1]['projected_minus_likelihood_expectation']-preceding['projected_minus_likelihood_expectation'])
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    complete = len(rows) == len(spec['draw_powers'])*len(spec['fantasy_powers'])
    last = rows[-1] if rows else {}
    isolated = bool(complete and last['importance_precision_passed']
        and max(last['nested_draw_reward_difference'], last['nested_fantasy_projection_difference'])
            <= spec['maximum_projection_difference_resolution_change'])
    result = {'execution_plan': plan, 'rows': rows, 'completed_head_fantasies': len(records),
        'completed_prefix_artifact': spec.get('completed_results'),
        'completed_prefix_cpu_seconds': read(spec['completed_results'])['cpu_seconds'] if 'completed_results' in spec else 0.,
        'all_registered_cases_complete': complete, 'next_episode_projection_difference_resolved': isolated,
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started,
        'cpu_budget_stopped': stopped, 'retained_identity_expectation_error_bound': retained_mass,
        'all_source_models_and_weights_retained': True, 'shared_identity_retained': True,
        'adaptive_maximum_evaluated': False, 'previous_paid_projection_retained': True,
        'cost': spec['cost'], 'scope': spec['scope'], 'independent_confirmation': False}
    save(path, result); return result


if __name__ == '__main__':
    r = run()
    print({k: r[k] for k in ('all_registered_cases_complete', 'next_episode_projection_difference_resolved',
        'cpu_seconds', 'cpu_budget_stopped')})
