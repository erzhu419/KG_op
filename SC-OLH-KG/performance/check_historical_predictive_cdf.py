"""Check inverse raw fantasies against historical adaptive CDF statistics."""

import signal
import time

import numpy as np
from scipy.stats import qmc

from core.positive_mixture_lookahead import RawPredictiveCDF
from performance.complete_historical_source_heads import ROOT,read,save
from performance.connect_complete_historical_reuse import run as restore_connected
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_predictive_cdf_v1.json'


def run(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json'
    previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_heads_complete'):return previous
    started=time.process_time();rows=previous.get('rows',[])
    models=restore_connected(spec['connection_plan'])
    handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds']);stopped=False
    nodes=qmc.Sobol(2,scramble=True,seed=spec['seed']+1).random_base2(spec['fantasy_power'])
    try:
        for state,model,*_ in models:
            layout=state['layout'];query=spec['queries'][layout]
            for identity,pair in zip(model.names,model.components):
                for h,field in enumerate(pair):
                    if any((r['layout'],r['identity'],r['head'])==(layout,identity,h) for r in rows):continue
                    phase=time.process_time();rule=RawPredictiveCDF(field,query,spec['order'])
                    values=rule.inverse(nodes[:,h]);adaptive,bound=rule.adaptive(values)
                    refined=RawPredictiveCDF(field,query,spec['refinement_order'])(values)
                    error=float(np.max(abs(adaptive-nodes[:,h])))
                    passed=bool(error<=spec['precision']['maximum_inverse_cdf_error']
                        and bound<=spec['precision']['maximum_adaptive_cdf_error_bound'])
                    row={'layout':layout,'candidate':query,'identity':identity,'head':h,
                        'retained_source_components':len(field.means),'inverse_values':values.tolist(),
                        'maximum_inverse_cdf_error':error,'adaptive_cdf_error_bound':bound,
                        'maximum_refined_cdf_error':float(np.max(abs(refined-adaptive))),
                        'maximum_order_refinement_difference':float(np.max(abs(refined-rule(values)))),
                        'maximum_eta_variance':float(rule.covariance[:,1,1].max()),
                        'precision_gate_passed':passed,'cpu_seconds':time.process_time()-phase}
                    rows.append(row);save(path,{'execution_plan':plan,'rows':rows,
                        'all_registered_heads_complete':len(rows)==32})
            current=[r for r in rows if r['layout']==layout]
            print('raw CDF',layout,'heads',len(current),'max inverse error',
                max(r['maximum_inverse_cdf_error'] for r in current),
                'passed',sum(r['precision_gate_passed'] for r in current),flush=True)
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    result={'execution_plan':plan,'rows':rows,'all_registered_heads_complete':len(rows)==32,
        'all_precision_gates_passed':len(rows)==32 and all(r['precision_gate_passed'] for r in rows),
        'stopped_at_cpu_budget':stopped,'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'scientific_model_ranking_authorized':False,'independent_confirmation':False}
    save(path,result);return result


if __name__=='__main__':run()
