"""Adapt the historical analytic direct utility to all actual source components."""

import signal
import time

from core.direct_terminal_reference import direct_terminal_reference
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_learned_direct_terminal_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all64_direct_actions_complete'):
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    resolution = read(spec['terminal_resolution'])['rows']; rows = []; stopped = False
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)))
    def record():
        result = {'execution_plan': plan, 'rows': rows, 'all64_direct_actions_complete': len(rows) == 64,
            'cpu_seconds': time.process_time() - started, 'cpu_budget_stopped': stopped,
            'cost': spec['cost'], 'independent_confirmation': False, 'scope': spec['scope']}
        save(path, result); return result
    try:
        for state, model, terminal, remaining in models:
            for budget in (remaining, remaining - 1):
                coarse = next(r for r in resolution if (r['layout'], r['remaining_calls'], r['power']) == (state['layout'], budget, 9))
                fine = next(r for r in resolution if (r['layout'], r['remaining_calls'], r['power']) == (state['layout'], budget, 14))
                assert coarse['actions'] == fine['actions']
                for index, action in enumerate(fine['actions']):
                    if action['kind'] != 'direct':
                        continue
                    result = direct_terminal_reference(model, action, terminal.protocol)
                    assert result['absolute_eta_quadrature_error'] < 1e-10
                    rows.append({'layout': state['layout'], 'remaining_calls': budget, 'action_index': index,
                        'action': action, **result, 'sobol512_reward': coarse['values'][index],
                        'sobol16384_reward': fine['values'][index],
                        'analytic_minus512': result['expected_reward'] - coarse['values'][index],
                        'analytic_minus16384': result['expected_reward'] - fine['values'][index]})
                record()
                group = [r for r in rows if r['layout'] == state['layout'] and r['remaining_calls'] == budget]
                best = max(group, key=lambda r: r['expected_reward'])
                print('collapsed direct', state['layout'], budget, 'actions', len(group),
                    'best', round(best['expected_reward'], 7), 'index', best['action_index'], flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return record()


if __name__ == '__main__':
    result = run()
    print({k: result[k] for k in ('all64_direct_actions_complete', 'cpu_seconds', 'cpu_budget_stopped')})
