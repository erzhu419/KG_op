"""First-pass stopping/debited utilities for every new source/GP state."""

import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.prepare_historical_independent_kg import restore_target_models

PLAN = 'performance/manifests/historical_independent_terminal_baselines_v1.json'


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); pilot = read(spec['pilot_plan']); preflight = read(spec['preflight'])
    assert preflight['all_density_and_rare_origin_prechecks_passed'] and not preflight['independent_outcomes_opened']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); path = out/'results.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    cases = list(previous.get('cases', [])); stopped = False
    selected_states = preflight['states']
    if 'case_names' in spec:
        selected_states = [s for s in selected_states if s['name'] in spec['case_names']]
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-
        previous.get('cpu_seconds', 0.)-(time.process_time()-started)))
    try:
        pending = [r for r in selected_states if r['name'] not in {c['name'] for c in cases}]
        for state, model, terminal in restore_target_models(pilot, pending):
            directory = out/state['name']; directory.mkdir(exist_ok=True); record_path = directory/'terminal_rows.json'
            if not record_path.exists() and 'reuse_output' in spec:
                save(record_path, read(spec['reuse_output']+'/'+state['name']+'/terminal_rows.json'))
            rows = read(str(record_path.relative_to(ROOT)))['rows'] if record_path.exists() else []
            if rows:
                assert read(str(record_path.relative_to(ROOT)))['source_weights'] == model.weights.tolist()
                assert all(r['guarded_mean_position'] == spec.get('guarded_mean_position') for r in rows)
            identities = [4, 5] if state['arm'] == 'source_structural_mixture' else [0]
            omitted = float(sum(w for k, w in enumerate(model.weights) if k not in identities))
            assert omitted <= spec['maximum_identity_integration_error_bound']
            for power in spec['raw_powers']:
                for budget in spec['remaining_budgets']:
                    if any((r['power'], r['remaining_calls']) == (power, budget) for r in rows):
                        continue
                    values, costs = [], []
                    for k in identities:
                        r, c, actions = terminal.evaluate_posterior(*model.components[k], budget, power, spec['seed'],
                            guarded_mean_position=spec.get('guarded_mean_position'))
                        values.append(r); costs.append(c)
                    reward = model.weights[identities]@values; cost = model.weights[identities]@costs
                    expected = state['terminal_actions' if budget == 360 else 'future_terminal_actions']
                    assert actions == expected and np.min(reward) >= -1e-12 and np.max(reward) <= 1.+1e-12
                    best = int(np.lexsort((cost, -reward))[0])
                    rows.append({'power': power, 'remaining_calls': budget, 'guarded_mean_position': spec.get('guarded_mean_position'),
                        'values': reward.tolist(), 'costs': cost.tolist(),
                        'best_action_index': best, 'best_action': actions[best], 'best_reward': float(reward[best]),
                        'utility_identities': identities, 'omitted_terminal_identity_mass': omitted})
                    save(record_path, {'rows': rows, 'source_weights': model.weights.tolist()})
            lower, higher = spec['raw_powers']; changes = []; stable = []
            for budget in spec['remaining_budgets']:
                a, b = [next(r for r in rows if (r['power'], r['remaining_calls']) == (p, budget)) for p in (lower, higher)]
                changes.append(float(np.max(abs(np.asarray(a['values'])-b['values']))))
                stable.append(a['best_action_index'] == b['best_action_index'])
            opportunity = [next(r for r in rows if (r['power'], r['remaining_calls']) == (p, 360))['best_reward']-
                next(r for r in rows if (r['power'], r['remaining_calls']) == (p, 359))['best_reward'] for p in (lower, higher)]
            gates = {'all_action_resolution': max(changes) <= spec['maximum_terminal_action_difference'],
                'opportunity_cost_resolution': abs(opportunity[1]-opportunity[0]) <= spec['maximum_opportunity_cost_difference'],
                'stop_and_debited_choices_stable': all(stable), 'identity_remainder': omitted <= spec['maximum_identity_integration_error_bound']}
            case = {'name': state['name'], 'task_id': state['task_id'], 'layout': state['layout'], 'arm': state['arm'],
                'rows': rows, 'maximum_action_changes_by_budget': changes, 'opportunity_cost_by_power': opportunity,
                'gates': gates, 'baseline_diagnostics_passed': all(gates.values())}
            cases.append(case); save(directory/'results.json', case)
            save(path, {'cases': cases, 'all_registered_cases_complete': False,
                'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started})
            print('independent baseline', state['name'], gates, 'CPU', round(time.process_time()-started, 2), flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'cases': cases, 'all_registered_cases_complete': len(cases) == len(selected_states),
        'passed_case_count': sum(c['baseline_diagnostics_passed'] for c in cases), 'registered_case_count': len(selected_states),
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'cost': {'source_fits': 0, 'new_physical_calls': 0, 'new_search_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0},
        'independent_outcomes_opened': False, 'independent_confirmation': False,
        'next': 'Resolve failed baseline diagnostics,then integrate each new state\'s six KG queries and freeze48 recommendations before verification.'}
    save(path, result); return result


if __name__ == '__main__':
    run()
