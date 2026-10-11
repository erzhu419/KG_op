"""Cheap n4 outer cubature pilot reusing the historical fixed-action snapshot."""
import argparse
import json
import resource
import signal
import time

import numpy as np
from scipy.special import roots_legendre

from core.observed_batch_predictive import ObservedBatchPredictive
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.refine_historical_observed_prediction import field_from_snapshot, conditioned_value

PLAN='performance/manifests/historical_observed_batch_outer_v1.json'


def run(origin,order,plan=PLAN,variance_order=None):
    variance_order=order if variance_order is None else variance_order
    suffix='' if variance_order==order else f'_v{variance_order}'
    spec=read(plan);out=ROOT/spec['output']/(f'o{origin}_r{order}'+suffix);out.mkdir(parents=True,exist_ok=True)
    started=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_worker_cpu_seconds'])
    result=dict(complete=False,origin=origin,order=order,variance_order=variance_order)
    try:
        assert any(j['origin']==origin and j['order']==order and j.get('variance_order',j['order'])==variance_order for j in spec['jobs'])
        previous=read(spec['cached_prediction_plan']);state=read(previous['output']+'/state.json')
        field=field_from_snapshot(state['fields'][str(origin)][1])
        x,w=roots_legendre(order);vx,vw=roots_legendre(variance_order);variance=(vx+1)/2;variance_weights=vw/2
        points=[];weights=[]
        for low,high in spec['mean_cdf_intervals']:
            means=low+(high-low)*(x+1)/2;mean_weights=(high-low)*w/2
            points.extend((m,v) for m in means for v in variance)
            weights.extend(mw*vw for mw in mean_weights for vw in variance_weights)
        points=np.asarray(points);weights=np.asarray(weights)
        assert np.all(weights>0) and abs(weights.sum()-1.)<=1e-14
        for cdf_order in previous['batch_cdf_orders']:
            cdf=ObservedBatchPredictive(field,0,4,cdf_order);labels,etas=cdf.inverse(points)
            vcdf,mcdf,bound=cdf.adaptive_coordinates(labels,etas)
            error=float(max(np.max(abs(vcdf-points[:,1])),np.max(abs(mcdf-points[:,0]))))
            if error<=previous['maximum_inverse_cdf_error'] and bound<=previous['maximum_adaptive_cdf_error_bound']:break
        cdf_passed=error<=previous['maximum_inverse_cdf_error'] and bound<=previous['maximum_adaptive_cdf_error_bound']
        result.update(cdf_order=cdf_order,maximum_inverse_cdf_error=error,adaptive_cdf_error_bound=bound,
            predictive_cdf_passed=cdf_passed,node_count=len(points),positive_weight_sum=float(weights.sum()))
        if not cdf_passed:return
        values=[];errors=[];admissions=[];orders=[]
        for i,(label,eta) in enumerate(zip(labels,etas)):
            value,error,passed,inner_order,_=conditioned_value(field,label,eta,4,1,
                state['baseline']['4']['action']['budgets'][0],state['protocol'],previous)
            values.append(value);errors.append(error);admissions.append(passed);orders.append(inner_order)
            if (i+1)%64==0:print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i+1,total=len(points),
                unit='fixed_action_outer_nodes',cpu_seconds=time.process_time()-started)),flush=True)
        save(out/'bank.json',dict(probabilities=points.tolist(),weights=weights.tolist(),labels=labels.tolist(),
            etas=etas.tolist(),values=values,inner_orders=orders))
        result.update(complete=True,mean_risk=float(weights@values),all_inner_rules_passed=all(admissions),
            inner_failed_count=sum(not p for p in admissions),maximum_evidence_error=max(errors),
            source_components=len(field.log_weights),query=state['query'],replicates=4,remaining_calls=356,
            source_prior_preserved=True,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
            full_posterior_restorations=0,independent_confirmation=False)
    except TimeoutError:result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.)
        result['cpu_seconds']=time.process_time()-started;save(out/'compact_summary.json',result)
        print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
        print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
    return result


def collect(plan=PLAN):
    spec=read(plan);previous=read(spec['cached_prediction_plan']);state=read(previous['output']+'/state.json')
    parts=[read(spec['output']+f'/o{j["origin"]}_r{j["order"]}'+
        ('' if j.get('variance_order',j['order'])==j['order'] else f'_v{j["variance_order"]}')+'/compact_summary.json') for j in spec['jobs']]
    admitted=all(p['complete'] and p['predictive_cdf_passed'] and p['all_inner_rules_passed'] for p in parts)
    rows=[]
    if admitted:
        configurations=sorted({(j['order'],j.get('variance_order',j['order'])) for j in spec['jobs']})
        for order,variance_order in configurations:
            reward=0.;origins=[]
            for origin in spec['origins']:
                quality=read(previous['output']+f'/n4/{origin}/0/results.json')
                assert quality['complete'] and quality['all_inner_rules_passed'] and quality['predictive_cdf_passed']
                f=float(np.mean(quality['values'][:2048]));g=next(p['mean_risk'] for p in parts if
                    (p['origin'],p['order'],p.get('variance_order',p['order']))==(origin,order,variance_order))
                contribution=state['prior_weights'][origin]*f*g;reward+=contribution
                origins.append(dict(origin=origin,mean_quality=f,mean_risk=g,weighted_fixed_reward=contribution))
            rows.append(dict(order=order,variance_order=variance_order,expected_fixed_reward=reward,
                tower_residual=reward-state['baseline']['4']['reward'],origins=origins))
    by={(r['order'],r['variance_order']):r for r in rows};changes={}
    if admitted:
        mean_orders=sorted({r['order'] for r in rows});variance_orders=sorted({r['variance_order'] for r in rows})
        fine_cfg=(mean_orders[-1],variance_orders[-1]);fine=by[fine_cfg]['expected_fixed_reward']
        for key,cfg in ((f'risk_mean{mean_orders[-2]}_{mean_orders[-1]}',(mean_orders[-2],variance_orders[-1])),
                (f'risk_variance{variance_orders[-2]}_{variance_orders[-1]}',(mean_orders[-1],variance_orders[-2]))):
            if cfg in by:changes[key]=abs(fine-by[cfg]['expected_fixed_reward'])
    change=abs(by[16,16]['expected_fixed_reward']-by[8,8]['expected_fixed_reward']) if rows else None
    result=dict(complete=all(p['complete'] for p in parts),all_inner_and_cdf_rules_passed=admitted,rows=rows,
        synchronous_resolution_difference=change,outer_axis_changes=changes,fixed_action_pilot_passed=bool(admitted and
            len(changes)==2 and max(changes.values())<=previous['maximum_tower_drift'] and abs(by[fine_cfg]['tower_residual'])<=previous['maximum_tower_drift']),
        cached_objective_prefix=2048,mean_cdf_intervals=spec['mean_cdf_intervals'],
        worker_cpu_seconds=sum(p['cpu_seconds'] for p in parts),estimated_cpu_seconds_including600reserves=spec['estimated_cpu_seconds_including600reserves'],
        full_posterior_restorations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        adaptive_outer_integral_admitted=False,independent_confirmation=False,
        scope='Fixed-action n4 risk-mean/variance positive cubature pilot;reuse all positive components and cached2048 objective values. Separate axis convergence and adaptive365-action KG remain open.')
    save(ROOT/spec['output']/'compact_summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','collect']);p.add_argument('--plan',default=PLAN)
    p.add_argument('--origin',type=int);p.add_argument('--order',type=int);p.add_argument('--variance-order',type=int);a=p.parse_args()
    if a.mode=='run':run(a.origin,a.order,a.plan,a.variance_order)
    else:collect(a.plan)
