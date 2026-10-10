"""Select fantasy resolution from density/identity precision before terminal KG."""

import signal
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from core.positive_mixture_lookahead import RawPredictiveCDF
from performance.complete_historical_source_heads import ROOT,read,save
from performance.connect_complete_historical_reuse import run as restore_connected
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_predictive_martingale_v1.json'


def run(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json';previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_layouts_complete'):return previous
    rows=previous.get('rows',[]);selected=previous.get('selected_fantasy_power',{})
    started=time.process_time();models=restore_connected(spec['connection_plan'])
    kg=read(spec['actual_paid_filter_reference'])['rows']
    handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds']);stopped=False
    try:
        for state,model,*_ in models:
            layout=state['layout'];query=spec['queries'][layout]
            if layout in selected:continue
            fields=[[RawPredictiveCDF(field,query,spec['order']) for field in pair] for pair in model.components]
            refined=[[RawPredictiveCDF(field,query,spec['refinement_order']) for field in pair] for pair in model.components]
            for power in spec['fantasy_powers']:
                if any(r['layout']==layout and r['fantasy_power']==power for r in rows):continue
                phase=time.process_time();nodes=qmc.Sobol(2,scramble=True,seed=spec['seed']+1).random_base2(power)
                fantasies=[];inverse_error=0.;inverse_bound=0.
                for pair in fields:
                    heads=[]
                    for h,cdf in enumerate(pair):
                        values=cdf.inverse(nodes[:,h]);reference,bound=cdf.adaptive(values)
                        inverse_error=max(inverse_error,float(np.max(abs(reference-nodes[:,h]))))
                        inverse_bound=max(inverse_bound,bound);heads.append(values)
                    fantasies.extend(zip(*heads))
                fantasies=np.asarray(fantasies);weight=np.repeat(model.weights/len(nodes),len(nodes))
                posteriors=[]
                for rules in (fields,refined):
                    scores=np.column_stack([sum(cdf.log_density(fantasies[:,h]) for h,cdf in enumerate(pair))
                        for pair in rules])+model.log_weights
                    posteriors.append(np.exp(scores-logsumexp(scores,axis=1)[:,None]))
                posterior_error=float(np.max(abs(posteriors[0]-posteriors[1])))
                residual=weight@posteriors[0]-model.weights
                actual=next(r for r in kg if r['layout']==layout)
                reproduction=(float(np.max(abs(residual-np.asarray(actual['model_weight_martingale_residual']))))
                    if power==3 else None)
                precision=spec['precision']
                adequate=bool(inverse_error+inverse_bound<=precision['maximum_inverse_cdf_error']
                    and posterior_error<=precision['maximum_posterior_weight_refinement_error']
                    and (reproduction is None or reproduction<=precision['maximum_actual_filter_reproduction_error']))
                passed=adequate and bool(np.max(abs(residual))<=precision['maximum_identity_martingale_residual'])
                row={'layout':layout,'candidate':query,'fantasy_power':power,'fantasy_count':len(fantasies),
                    'maximum_inverse_cdf_error':inverse_error,'maximum_adaptive_error_bound':inverse_bound,
                    'maximum_posterior_weight_refinement_error':posterior_error,
                    'actual_paid_filter_reproduction_error':reproduction,
                    'model_weight_martingale_residual':residual.tolist(),
                    'maximum_identity_martingale_residual':float(np.max(abs(residual))),
                    'density_quadrature_adequate':adequate,'precision_gate_passed':passed,
                    'cpu_seconds':time.process_time()-phase}
                rows.append(row)
                if passed:selected[layout]=power
                save(path,{'execution_plan':plan,'rows':rows,'selected_fantasy_power':selected,
                    'all_layouts_complete':len(selected)==2})
                print('density precision',layout,power,'residual',row['maximum_identity_martingale_residual'],
                    'posterior error',posterior_error,'adequate',adequate,'pass',passed,flush=True)
                if passed:break
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    result={'execution_plan':plan,'rows':rows,'selected_fantasy_power':selected,
        'all_layouts_complete':len(selected)==2,'stopped_at_cpu_budget':stopped,
        'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'terminal_actions_evaluated':0,'full_fantasy_moment_updates':0,
        'scientific_model_ranking_authorized':False,'independent_confirmation':False}
    save(path,result);return result


if __name__=='__main__':run()
