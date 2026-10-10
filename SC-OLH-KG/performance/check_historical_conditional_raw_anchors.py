"""Actual-field anchors against historical complete latent transport."""

import copy
import signal
import time

import numpy as np
from scipy.stats import qmc

from core.positive_mixture_lookahead import condition_positive_raw
from core.raw_likelihood_lookahead import objective_terminal_statistics, risk_terminal_statistics
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_conditional_raw_anchors_v1.json'


def run():
    spec = read(PLAN); out = ROOT/spec['output']; path = out/'anchors.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); rows = []; stopped = False
    raw = read(spec['conditional_results']); assert raw['all_registered_cases_complete']
    records = read(spec['conditional_head_fantasies'])['rows']
    action = read(spec['projected_heads'])['action']
    _, _, models = build_connection(spec['connection_plan'])
    _, model, terminal, _ = next(r for r in models if r[0]['layout'] == spec['layout'])
    nodes = qmc.Sobol(7, scramble=True, seed=spec['seed']).random_base2(spec['historical_transport_draw_power'])
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
    try:
        for identity in spec['identity_indices']:
            for head, field in enumerate(model.components[identity]):
                cached = next(r for r in records if (r['identity'], r['head'], r['position'], r['draw_power']) ==
                    (identity, head, spec['position'], spec['conditional_draw_power']))
                future = copy.deepcopy(field); label = cached['predictive_label']
                evidence = future.update_batch(spec['candidate'], label, None, 1)
                draws, weights, error = condition_positive_raw(field, future, spec['candidate'],
                    action['shortlist'], nodes, label, evidence)
                draws = draws.reshape(1, len(nodes), 3, 2)
                value = (objective_terminal_statistics(draws, weights[None], action['incumbent_position'], terminal.protocol)[0]
                    if head == 0 else risk_terminal_statistics(draws, weights[None], action['incumbent_position'],
                        action['budgets'], terminal.protocol)[0][0])
                difference = float(np.max(abs(value-cached['statistics'])))
                rows.append({'identity': identity, 'identity_name': model.names[identity], 'head': head,
                    'maximum_absolute_head_statistic_difference': difference,
                    'historical_statistics': value.tolist(), 'conditional_statistics': cached['statistics'],
                    'predictive_log_evidence_error': error, 'effective_sample_size': float(1/(weights @ weights))})
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': PLAN, 'rows': rows, 'all_anchors_complete': len(rows) == 4,
        'all_anchors_equivalent': bool(len(rows) == 4 and all(r['maximum_absolute_head_statistic_difference']
            <= spec['maximum_absolute_head_statistic_difference'] for r in rows)),
        'cpu_seconds': time.process_time()-started, 'cpu_budget_stopped': stopped}
    save(path, result); print(result); return result


if __name__ == '__main__':
    run()
