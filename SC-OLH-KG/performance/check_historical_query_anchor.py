"""Reproduce the registered higher-resolution query0 branch anchor."""

import signal
import time

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from performance.refine_historical_query_objective import actual_anchor

PLAN = 'performance/manifests/historical_query_anchor_resolution_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    parent = read('performance/manifests/historical_remaining_query_grids_v1.json')
    directory = ROOT/parent['output']/spec['case']
    original = read(str((directory/'head1.json').relative_to(ROOT)))
    case = read(str((directory/'results.json').relative_to(ROOT)))
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        _, _, models = build_connection(spec['connection_plan'])
        signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
        state, model, _, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
        assert spec['query'] == 0 and model.weights.tolist() == case['source_weights']
        record = next(r for r in original['records'] if r['position'] == spec['position'])
        result = actual_anchor(model, state, remaining, record, original['actions'], case['baseline'],
            dict(spec, maximum_error=spec['maximum_anchor_branch_error']))
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result.update(execution_plan=plan, case=spec['case'], cpu_seconds=time.process_time()-started)
    save(path, result); return result


if __name__ == '__main__':
    result = run(); print('actual branch anchor', result['case'], result['identity'], result['error'], result['passed'])
