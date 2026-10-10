"""Measure terminal quadrature resolution at the unchanged actual learned laws."""

import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_learned_terminal_resolution_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'; previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    started = time.process_time(); rows = previous.get('rows', []); stopped = False
    connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)))
    def record():
        result = {'execution_plan': plan, 'rows': rows, 'all_registered_cases_complete': len(rows) == 12,
            'cpu_seconds': time.process_time() - started, 'cpu_budget_stopped': stopped,
            'cost': spec['cost'], 'independent_confirmation': False, 'scope': spec['scope']}
        save(path, result)
        return result
    try:
        for state, model, terminal, remaining in models:
            for budget in (remaining, remaining - 1):
                for power in spec['powers']:
                    if any((r['layout'], r['remaining_calls'], r['power']) == (state['layout'], budget, power) for r in rows):
                        continue
                    values, costs, actions = model.terminal_values(terminal, budget, power, spec['seed'])
                    best = int(np.lexsort((costs, -values))[0])
                    baseline = next((r for r in rows if (r['layout'], r['remaining_calls'], r['power']) ==
                        (state['layout'], budget, spec['powers'][0])), None)
                    row = {'layout': state['layout'], 'remaining_calls': budget, 'power': power,
                        'values': values.tolist(), 'costs': costs.tolist(), 'actions': actions,
                        'best_action_index': best, 'best_action': actions[best], 'best_reward': float(values[best])}
                    if baseline:
                        assert actions == baseline['actions']
                        delta = values - baseline['values']; prior = baseline['best_action_index']
                        row.update(maximum_action_value_change_from512=float(np.max(abs(delta))),
                            fixed512_incumbent_reward=float(values[prior]),
                            fixed512_incumbent_value_change=float(delta[prior]),
                            same_best_action=best == prior)
                    rows.append(row); record()
                    print('terminal resolution', state['layout'], budget, 2**power,
                        'best', round(row['best_reward'], 7), 'action', best, flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return record()


if __name__ == '__main__':
    result = run()
    print({k: result[k] for k in ('all_registered_cases_complete', 'cpu_seconds', 'cpu_budget_stopped')})
