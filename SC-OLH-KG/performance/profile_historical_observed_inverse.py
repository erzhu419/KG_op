"""Time the reachable repeated-coordinate case against the saved n4 bank."""
import argparse
import json
import resource
import signal
import time

import numpy as np
from scipy.special import polygamma

from core.observed_batch_predictive import ObservedBatchPredictive, inverse_observed_cdf
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.refine_historical_observed_prediction import field_from_snapshot


def run(origin):
    out=ROOT/'paper_artifacts/historical_observed_batch_outer_v1'
    cached=read(str(out/f'o{origin}_r8_v32/bank.json'))
    summary=read(str(out/f'o{origin}_r8_v32/compact_summary.json'))
    spec=read('performance/manifests/historical_observed_batch_outer_v1.json')
    previous=read(spec['cached_prediction_plan']);state=read(previous['output']+'/state.json')
    field=field_from_snapshot(state['fields'][str(origin)][1])
    rule=ObservedBatchPredictive(field,0,4,summary['cdf_order'])
    points=np.asarray(cached['probabilities']);mean=float(rule.weights@rule.means[:,1])
    variance=float(rule.weights@(rule.covariance[:,1,1]+(rule.means[:,1]-mean)**2)+polygamma(1,rule.k))
    started=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,120.)
    result=dict(complete=False,origin=origin,node_count=len(points),source_components=len(field.log_weights),
        charged_to='original600reserve/cached_analysis240',maximum_cpu_seconds=120,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,full_posterior_restorations=0)
    try:
        tick=time.process_time()
        old=inverse_observed_cdf(points[:,1],rule.variance_coordinates,mean,variance)
        result['repeated_variance_cpu_seconds']=time.process_time()-tick
        tick=time.process_time();new=rule.variance_inverse(points[:,1])
        result['unique_variance_cpu_seconds']=time.process_time()-tick
        result['old_new_variance_difference']=float(np.max(abs(old-new)))
        result['cached_variance_difference']=float(np.max(abs(new-np.asarray(cached['etas']))))
        tick=time.process_time();labels,etas=rule.inverse(points)
        result['optimized_full_inverse_cpu_seconds']=time.process_time()-tick
        result['cached_full_inverse_difference']=float(max(np.max(abs(labels-np.asarray(cached['labels']))),
            np.max(abs(etas-np.asarray(cached['etas'])))))
        result['complete']=True
        result['parity_passed']=max(result['old_new_variance_difference'],result['cached_variance_difference'],
            result['cached_full_inverse_difference'])<=1e-10
    except TimeoutError:
        result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.)
        result['cpu_seconds']=time.process_time()-started
        save(out/f'inverse_profile_o{origin}.json',result)
        print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
        print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--origin',type=int,choices=[4,5],required=True)
    run(p.parse_args().origin)
