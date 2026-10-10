"""Measure the actual corrected terminal used by joint-mixture acquisition."""

import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_conditional_terminal_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    previous = (read(str(path.relative_to(ROOT))) if path.exists() else
        read(spec['resume_results']) if 'resume_results' in spec else {})
    if previous.get('all_registered_cases_complete'):
        return previous
    started = time.process_time(); rows = previous.get('rows', []); stopped = False
    connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    old = read(spec['old_terminal_resolution'])['rows']
    analytic = read(spec['analytic_direct_reference'])['rows']
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)))
    def record():
        result = {'execution_plan': plan, 'rows': rows,
            'all_registered_cases_complete': len(rows) == 4 * len(spec['powers']),
            'cpu_seconds': previous.get('cpu_seconds', 0.) + time.process_time() - started,
            'cpu_seconds_this_execution': time.process_time() - started,
            'previous_cpu_budget_stop': previous.get('cpu_budget_stopped', False),
            'cpu_budget_stopped': stopped, 'cost': spec['cost'], 'scope': spec['scope'],
            'decision_ranking_certified': False, 'independent_confirmation': False}
        save(path, result); return result
    try:
        for power in spec['powers']:
            for state, model, parent, remaining in models:
                terminal = ConditionalTerminalActions(parent.shortlists, parent.incumbents, parent.protocol)
                for budget in (remaining, remaining - 1):
                    if any((r['layout'], r['remaining_calls'], r['power']) == (state['layout'], budget, power) for r in rows):
                        continue
                    before = time.process_time()
                    values, costs, actions = model.terminal_values(terminal, budget, power, spec['seed'])
                    original = next(r for r in old if (r['layout'], r['remaining_calls'], r['power']) == (state['layout'], budget, 14))
                    assert actions == original['actions']
                    errors = [abs(values[r['action_index']] - r['expected_reward']) for r in analytic
                        if (r['layout'], r['remaining_calls']) == (state['layout'], budget)]
                    assert len(errors) == 16 and max(errors) < 1e-11
                    best = int(np.lexsort((costs, -values))[0])
                    row = {'layout': state['layout'], 'remaining_calls': budget, 'power': power,
                        'values': values.tolist(), 'costs': costs.tolist(), 'actions': actions,
                        'best_action_index': best, 'best_action': actions[best], 'best_reward': float(values[best]),
                        'maximum_analytic_direct_reference_error': max(errors),
                        'maximum_change_from_old16384': float(np.max(abs(values - original['values']))),
                        'same_best_as_old16384': best == original['best_action_index'],
                        'cpu_seconds': time.process_time() - before}
                    preceding = next((r for r in reversed(rows) if (r['layout'], r['remaining_calls']) == (state['layout'], budget)), None)
                    if preceding:
                        delta = values - preceding['values']
                        row.update(previous_power=preceding['power'], maximum_nested_action_difference=float(np.max(abs(delta))),
                            nested_best_reward_difference=abs(row['best_reward'] - preceding['best_reward']),
                            same_nested_best=best == preceding['best_action_index'])
                    rows.append(row); record()
                    print('conditional terminal', state['layout'], budget, 2**power,
                        'best', row['best_reward'], 'action', best, 'CPU', round(row['cpu_seconds'], 2), flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return record()


if __name__ == '__main__':
    result = run()
    print({k: result[k] for k in ('all_registered_cases_complete', 'cpu_seconds', 'cpu_budget_stopped')})
