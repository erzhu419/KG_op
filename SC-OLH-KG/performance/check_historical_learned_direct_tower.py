"""Resolve one observed false-looking positive fixed-direct sampling signal."""

import signal
import time

import numpy as np

from core.direct_terminal_reference import fixed_direct_projected_tower
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_learned_direct_tower_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all_registered_rules_complete'):
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    state, model, terminal, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
    parent = read(spec['kg_results'])
    action = next(r['action'] for r in parent['reference_rows'] if
        (r['layout'], r['candidate']) == (spec['layout'], spec['candidate']))
    assert action['kind'] == 'direct' and action['budgets'] == [remaining - 1]
    rows = []; stopped = False; handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)))
    def record():
        result = {'execution_plan': plan, 'rows': rows, 'all_registered_rules_complete': len(rows) == len(spec['fantasy_powers']),
            'cpu_seconds': time.process_time() - started, 'cpu_budget_stopped': stopped,
            'cost': spec['cost'], 'independent_confirmation': False, 'scope': spec['scope']}
        save(path, result); return result
    try:
        for power in spec['fantasy_powers']:
            row = fixed_direct_projected_tower(model, action, spec['candidate'], power, spec['seed'], terminal.protocol)
            if rows:
                count = 2**rows[-1]['fantasy_power']
                error = float(np.max(abs(np.asarray(row['future_head_statistics_by_identity'])[:, :, :count]
                    - np.asarray(rows[-1]['future_head_statistics_by_identity']))))
                assert error <= 1e-8
                row['nested_prefix_reproduction_error'] = error
            rows.append(row); record()
            print('analytic direct tower', spec['layout'], spec['candidate'], 2**power,
                'drift', row['projected_fixed_action_drift'], 'nested', row['nested_expected_future_difference'], flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return record()


if __name__ == '__main__':
    result = run()
    print({k: result[k] for k in ('all_registered_rules_complete', 'cpu_seconds', 'cpu_budget_stopped')})
