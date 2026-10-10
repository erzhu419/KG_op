"""Reuse the historical fixed-action raw reference on actual positive mixtures."""

import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save
from performance.connect_complete_historical_reuse import run as restore_connected
from performance.refine_historical_raw_lookahead import cpu_timeout
from core.raw_likelihood_lookahead import fixed_action_likelihood_reference

PLAN='performance/manifests/historical_positive_likelihood_reference_v1.json'


def run(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json';previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_two_references_complete'):return previous
    rows=previous.get('rows',[]);started=time.process_time();models=restore_connected(spec['connection_plan'])
    handler=signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,
        max(spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started),1e-6));stopped=False
    try:
        for state,model,terminal,remaining,baseline in models:
            layout=state['layout']
            if any(r['layout']==layout for r in rows):continue
            values,_,actions=model.terminal_values(terminal,remaining-1,spec['terminal_action_draw_power'],spec['seed'])
            action=actions[int(values.argmax())]
            result=fixed_action_likelihood_reference(model,action,spec['queries'][layout],
                draw_power=spec['draw_power'],fantasy_power=spec['fantasy_power'],seed=spec['seed'],protocol=terminal.protocol)
            precision=spec['precision'];identity=max(abs(v) for v in result['model_weight_martingale_residual'])
            result.update(layout=layout,maximum_identity_martingale_residual=identity,
                likelihood_reference_adequate=bool(result['predictive_log_evidence_max_error']<=precision['maximum_log_evidence_error']
                    and result['effective_sample_size_min']>=precision['minimum_weight_ESS']),
                tower_precision_gate_passed=bool(abs(result['likelihood_tower_drift'])<=precision['maximum_absolute_tower_drift']
                    and identity<=precision['maximum_identity_martingale_residual']))
            rows.append(result);save(path,{'execution_plan':plan,'rows':rows,'all_two_references_complete':len(rows)==2})
            print('positive likelihood reference',layout,'likelihood drift',round(result['likelihood_tower_drift'],6),
                'projection gap',round(result['projection_minus_likelihood_baseline'],6),
                'evidence error',round(result['predictive_log_evidence_max_error'],6),
                'ESS',round(result['effective_sample_size_min'],1),'adequate',result['likelihood_reference_adequate'],flush=True)
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    result={'execution_plan':plan,'rows':rows,'all_two_references_complete':len(rows)==2,
        'all_references_adequate':len(rows)==2 and all(r['likelihood_reference_adequate'] for r in rows),
        'stopped_at_cpu_budget':stopped,'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'no_model_adopted':True,'independent_confirmation':False,'scientific_model_ranking_authorized':False,
        'limitations':['Reference is a finite self-normalized likelihood integral;ESS and evidence errors must pass before attributing projection bias.',
            'Starting fields retain their previous two paid moment projections.',
            'Fixed action only;this does not compare adaptive selection or independent decision performance.']}
    save(path,result);return result


if __name__=='__main__':run()
