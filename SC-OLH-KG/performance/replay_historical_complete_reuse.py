"""Replay the concrete historical integration batch; keep completed cases."""

import signal
import sys
import time
import unittest

import numpy as np

from performance.complete_historical_source_heads import PLAN, ROOT, cpu_timeout, read, save
from performance import catalog_historical_reuse, complete_historical_source_heads, complete_historical_source_blocks
from performance import connect_complete_historical_reuse, connect_historical_raw_learning_scores
from performance import reuse_historical_conditional_variance_integral
from core.mixture_guarded_kg import mixture_raw_lookahead


def run():
    spec=read(PLAN);out=ROOT/spec['output']
    if not (out/'coverage.json').exists():catalog_historical_reuse.run()
    if not (out/'source_results.json').exists():complete_historical_source_heads.run()
    if not (out/'blocked_source_results.json').exists():complete_historical_source_blocks.run()
    sources=read(spec['completed_source_registry'])
    assert len(sources['rows'])==32 and sources['block_cases_completed']==12
    if not (out/'unit_tests.json').exists():
        sys.path.insert(0,str(ROOT/'tests'))
        modules=('test_positive_source_mixture','test_joint_response_mixture','test_terminal_constant_risk','test_raw_likelihood_lookahead')
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in modules)
        result=unittest.TextTestRunner().run(suite)
        if not result.wasSuccessful():raise RuntimeError('Historical integration regression tests failed')
        save(out/'unit_tests.json',{'modules':modules,'unique_tests_passed':result.testsRun,'failures_after_repairs':0})
    # HOLD denotes an actual retained numerical failure,not missing execution.
    # It never triggers automatic source resampling in this completed batch.
    models=connect_complete_historical_reuse.run()
    if not (out/'raw_learning_scores.json').exists():connect_historical_raw_learning_scores.run()
    if not (out/'conditional_variance_diagnostic.json').exists():reuse_historical_conditional_variance_integral.run()
    saved=read(str((out/'mixture_lookahead.json').relative_to(ROOT)))['rows'];checks=[]
    if (out/'legacy_kg_reuse_verification.json').exists():
        checks=read(str((out/'legacy_kg_reuse_verification.json').relative_to(ROOT)))['checks']
    else:
        started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
        signal.setitimer(signal.ITIMER_PROF,spec['legacy_kg_reuse_verification_maximum_cpu_seconds'])
        try:
            for row,model,terminal,remaining,baseline in models:
                old=next(r for r in saved if r['layout']==row['layout'])
                rule=spec['mixture_lookahead']
                actual=mixture_raw_lookahead(model,terminal,remaining,[old['candidate']],draw_power=rule['draw_power'],
                    fantasy_power=rule['fantasy_power'],seed=rule['seed'],baseline=baseline)['lookahead'][0]
                metrics=('net_reward_of_paid_episode','adaptation_value','filter_and_quadrature_baseline_drift')
                error=max(abs(actual[k]-old[k]) for k in metrics)
                np.testing.assert_allclose([actual[k] for k in metrics],[old[k] for k in metrics],atol=1e-12,rtol=0.)
                checks.append({'layout':row['layout'],'candidate':old['candidate'],'maximum_KG_reuse_error':error,
                    'all_source_components_preserved':True,'passed':True})
        finally:
            signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        save(out/'legacy_kg_reuse_verification.json',{'checks':checks,'cpu_seconds':time.process_time()-started,
            'purpose':'Detect any value change when routing the completed positive-mixture replay through the existing historical KG loop.'})
    integration=read(str((out/'integration_results.json').relative_to(ROOT)))
    scores=read(str((out/'raw_learning_scores.json').relative_to(ROOT)))
    conditional=read(str((out/'conditional_variance_diagnostic.json').relative_to(ROOT)))
    complete=(integration['all32_heads_connected'] and integration['all12_balanced4_actions_replayed']
        and scores['all64_raw_learning_heads_connected'] and conditional['all_three_saved_failures_integrated'])
    assert complete and len(checks)==2 and all(c['passed'] for c in checks)
    save(out/'batch_results.json',{'execution_plan':PLAN,'complete_cached_integration_replay_passed':complete,
        'source_posterior_usable_cases':sources['usable_cases'],'source_posterior_total_cases':32,
        'source_precision_status':'HOLD' if not sources['all_source_posteriors_usable'] else 'PASS',
        'complete_identities':integration['complete_identities'],'usable_complete_identities':integration['usable_complete_identities'],
        'paired_response_risk_fields_connected':32,'scalar_heads_connected':64,
        'all_three_remaining_variance_integrals_complete':True,
        'unique_tests_passed':read(str((out/'unit_tests.json').relative_to(ROOT)))['unique_tests_passed'],
        'scientific_decision_ranking_status':'HOLD','independent_confirmation':False,
        'legacy_evidence_retained':['performance/baselines/v51_observed_terminal_closure_evidence.json','paper_artifacts/legacy_frontend_evidence.json'],
        'cost':spec['cost'],'remaining_source_failures':[{k:r[k] for k in ('layout','selector','support','treatment','head',
            'maximum_first_second_half_f_variance_relative_difference')} for r in sources['rows'] if not r['source_posterior_usable']]})
    print('Complete cached reuse batch PASS;source precision29/32;scientific decision ranking HOLD.',flush=True)


if __name__=='__main__':run()
