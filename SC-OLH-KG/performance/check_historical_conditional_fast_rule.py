"""Check fast-rule timing/equivalence,including two actual paid-filter states."""

import copy
import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.direct_terminal_reference import expected_quality, expected_certificate, fast_expected_certificates
from core.positive_mixture_lookahead import predictive_values
from performance.check_historical_conditional_efficiency import run as efficiency
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_conditional_efficiency_v2.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; path = out / 'anchors.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all_registered_anchors_complete'):
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); benchmark = efficiency(plan)
    assert benchmark['all_cases_equivalent']
    connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    rows = []; stopped = False; handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)))
    def record():
        result = {'execution_plan': plan, 'rows': rows,
            'all_registered_anchors_complete': len(rows) == len(spec['fantasy_anchors']),
            'all_anchors_equivalent': len(rows) == len(spec['fantasy_anchors']) and all(r['equivalent'] for r in rows),
            'cpu_seconds_this_execution': time.process_time()-started,
            'anchor_cpu_seconds': time.process_time()-anchor_started,
            'cpu_budget_stopped': stopped, 'cost': spec['cost'], 'independent_confirmation': False}
        save(path, result); return result
    anchor_started = time.process_time()
    try:
        for anchor in spec['fantasy_anchors']:
            state, model, parent, remaining = next(r for r in models if r[0]['layout'] == anchor['layout'])
            labels, _ = predictive_values(model, anchor['candidate'], spec['anchor_fantasy_power'], spec['seed']+1)
            position = anchor['identity']*2**spec['anchor_fantasy_power'] + anchor['position']
            future = copy.deepcopy(model); future.update_raw(anchor['candidate'], labels[position])
            terminal = ConditionalTerminalActions(parent.shortlists, parent.incumbents, parent.protocol)
            one = time.process_time()
            values, _, actions = future.terminal_values(terminal, remaining-1, spec['powers'][0], spec['seed'])
            elapsed = time.process_time()-one
            selected = [i for i, a in enumerate(actions) if a['kind'] == 'direct']
            candidates = [actions[i]['primary'] for i in selected]
            budgets = [actions[i]['budgets'][0] for i in selected]
            scalar_rewards = []; probability_errors = []; diagnostics = []
            for f, g in future.components:
                fast, info = fast_expected_certificates(g, candidates, budgets, terminal.protocol)
                scalar = np.array([expected_certificate(g, i, n, terminal.protocol)[0] for i, n in zip(candidates, budgets)])
                quality = np.array([expected_quality(f, i, terminal.protocol['objective_reference']) for i in candidates])
                scalar_rewards.append(quality*scalar); probability_errors.append(float(np.max(abs(fast-scalar))))
                diagnostics.append(info)
            reference = future.weights @ scalar_rewards
            error = float(np.max(abs(values[selected]-reference)))
            row = {**anchor, 'raw_fantasy_labels': labels[position].tolist(), 'maximum_direct_reward_difference': error,
                'maximum_identity_certificate_difference': max(probability_errors), 'certificate_diagnostics': diagnostics,
                'terminal_cpu_seconds': elapsed, 'terminal_actions': len(actions),
                'future_paid_episodes': future.paid_episodes, 'source_law_unchanged': True,
                'maximum_raw_integration_relative_error': max(field.max_integration_relative_error
                    for pair in future.components for field in pair),
                'equivalent': error <= spec['maximum_reward_difference'] and max(probability_errors) <= spec['maximum_reward_difference']}
            rows.append(row); record()
            print('fast paid anchor', anchor['layout'], 'query', anchor['candidate'], 'error', error,
                'identity error', row['maximum_identity_certificate_difference'], 'CPU', elapsed, flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return record()


if __name__ == '__main__':
    result = run()
    print({key: result[key] for key in ('all_registered_anchors_complete', 'all_anchors_equivalent', 'cpu_seconds_this_execution', 'cpu_budget_stopped')})
