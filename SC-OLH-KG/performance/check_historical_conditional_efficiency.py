"""Measure optimized terminal equivalence at the actual cached source laws."""

import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_conditional_efficiency_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'; previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    started = time.process_time(); rows = previous.get('rows', []); stopped = False
    connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    reference = read(spec['reference'])['rows']
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)))
    def record():
        result = {'execution_plan': plan, 'rows': rows, 'all_registered_cases_complete': len(rows) == 4*len(spec['powers']),
            'all_cases_equivalent': len(rows) == 4*len(spec['powers']) and all(r['equivalent'] for r in rows),
            'cpu_seconds': previous.get('cpu_seconds', 0.) + time.process_time() - started,
            'cpu_budget_stopped': stopped, 'cost': spec['cost'], 'scope': spec['scope'],
            'decision_ranking_certified': False, 'independent_confirmation': False}
        save(path, result); return result
    try:
        for power in spec['powers']:
            for state, model, parent, remaining in models:
                terminal = ConditionalTerminalActions(parent.shortlists, parent.incumbents, parent.protocol)
                for budget in (remaining, remaining-1):
                    key = (state['layout'], budget, power)
                    if any((r['layout'], r['remaining_calls'], r['power']) == key for r in rows):
                        continue
                    old = next(r for r in reference if (r['layout'], r['remaining_calls'], r['power']) == key)
                    one = time.process_time()
                    values, costs, actions = model.terminal_values(terminal, budget, power, spec['seed'])
                    elapsed = time.process_time() - one
                    reward_error = float(np.max(abs(values - old['values'])))
                    cost_error = float(np.max(abs(costs - old['costs'])))
                    row = {'layout': state['layout'], 'remaining_calls': budget, 'power': power,
                        'maximum_reward_difference': reward_error, 'maximum_cost_difference': cost_error,
                        'same_actions': actions == old['actions'], 'same_best_action': int(values.argmax()) == old['best_action_index'],
                        'equivalent': actions == old['actions'] and reward_error <= spec['maximum_reward_difference']
                            and cost_error <= spec['maximum_cost_difference'],
                        'cpu_seconds': elapsed, 'reference_cpu_seconds': old['cpu_seconds'],
                        'speedup_over_saved_reference': old['cpu_seconds']/elapsed}
                    rows.append(row); record()
                    print('terminal efficiency', *key, 'error', reward_error, 'speedup', round(row['speedup_over_saved_reference'], 2), flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return record()


if __name__ == '__main__':
    result = run()
    print({key: result[key] for key in ('all_registered_cases_complete', 'all_cases_equivalent', 'cpu_seconds', 'cpu_budget_stopped')})
