"""Replay the completed local variance repair without resampling passed cases."""

import io
import sys
import time
import unittest

from performance.resolve_historical_source_variance import run as sample_sources, read, save, ROOT
from performance.integrate_historical_joint_source_variance import run as integrate_sources
from performance.finalize_historical_source_variance import run as finalize_sources, PLAN
from performance.connect_complete_historical_reuse import run as connect
from performance.connect_historical_raw_learning_scores import run as learning_scores


def run():
    started = time.process_time()
    for plan in ('performance/manifests/historical_source_variance_resolution_v1.json',
                 'performance/manifests/historical_source_variance_resolution_v2.json'):
        sample_sources(plan)
    integrate_sources()
    source = finalize_sources(PLAN); spec = read(PLAN); out = ROOT / spec['output']
    integration_path = out / 'integration_results.json'
    integration = read(str(integration_path.relative_to(ROOT))) if integration_path.exists() else None
    if integration is None or not integration['all12_balanced4_actions_replayed']:
        connect(PLAN); integration = read(str(integration_path.relative_to(ROOT)))
    score_path = out / 'raw_learning_scores.json'
    if not score_path.exists(): learning_scores(PLAN)
    scores = read(str(score_path.relative_to(ROOT)))
    test_path = out / 'unit_tests.json'
    if not test_path.exists() or not read(str(test_path.relative_to(ROOT)))['passed']:
        sys.path.insert(0,str(ROOT/'tests'))
        modules = ['test_source_variance_tilt', 'test_positive_source_mixture',
            'test_joint_response_mixture', 'test_terminal_constant_risk', 'test_raw_likelihood_lookahead']
        stream = io.StringIO(); one = time.process_time()
        result = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
        save(test_path, {'modules': modules, 'tests_run': result.testsRun, 'passed': result.wasSuccessful(),
            'output': stream.getvalue(), 'cpu_seconds': time.process_time()-one})
    tests = read(str(test_path.relative_to(ROOT)))
    if not tests['passed']: raise RuntimeError('Variance repair regression tests failed')
    result = {'execution_plan': PLAN, 'source_usable_cases': source['usable_cases'], 'source_case_count': len(source['rows']),
        'all_source_precision_gates_passed': source['all_source_posteriors_usable'],
        'weighted_target_KG_integration_passed': bool(integration['all32_heads_connected']
            and integration['all12_balanced4_actions_replayed'] and not integration['stopped_layouts']),
        'inverse_tilt_weighted_heads_connected': sum(c['inverse_tilt_weights_retained'] for c in integration['checks']),
        'raw_learning_head_count': len(scores['rows']), 'tests_passed': tests['passed'], 'tests_run': tests['tests_run'],
        'replay_cpu_seconds': time.process_time()-started, 'cost': spec['cost'],
        'scientific_model_ranking_authorized': False, 'no_model_adopted': True, 'independent_confirmation': False,
        'limitations': ['KG replay uses one fantasy per identity on an opened eight-policy bank.',
            'Frozen covariance/amplitude laws are not a converged raw-likelihood sparse meta-prior fit.',
            'Matched strong-control decision advantage,physical factor recovery and independent confirmation remain unestablished.']}
    save(out / 'batch_results.json', result)
    print('source variance resolution batch', result['source_usable_cases'], '/', result['source_case_count'],
        'integration', result['weighted_target_KG_integration_passed'], 'tests', tests['passed'], tests['tests_run'], flush=True)
    return result


if __name__ == '__main__': run()
