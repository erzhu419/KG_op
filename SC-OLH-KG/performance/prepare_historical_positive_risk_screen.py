"""Reuse the positive outer bank and historical all-action screen at its switch."""
import argparse
import json
import resource
import signal
import time

import numpy as np
from scipy.special import logsumexp

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, compressed_write
from performance.diagnose_historical_predictive_tower import restrict_field
from performance.connect_historical_learned_prior_kg import restore_learned_source_prior
from performance.refine_historical_observed_prediction import field_from_snapshot
from representation.vectorized_joint_function_risk import local_raw_moments

PLAN='performance/manifests/historical_positive_risk_screen_v1.json'


def prepare(plan=PLAN):
    started=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    outer=read(spec['outer_plan']);pilot=read(outer['output']+'/compact_summary.json')
    assert pilot['fixed_action_pilot_passed']
    old=read(spec['historical_bound_plan']);context=compressed_read(ROOT/old['output']/'context.json.gz')
    query=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals/136.json.gz')
    assert query['points']==[context['query']] and query['prior_weights']==context['prior_weights']
    assert len(query['fields'])==8 and len(context['actions'])==365 and context['remaining_calls']==356
    # Historical action marginals save utility identities4/5 only. Restore
    # the existing two paid sites plus the query to recover all8 likelihoods.
    full=read(spec['full_bank_plan']);execution=read(full['independent_execution_plan'])
    fixed=read(outer['cached_prediction_plan'])
    state=next(s for s in read(execution['preflight'])['states'] if s['name']==fixed['state'])
    paid=next(r for r in read(full['initializations'])['rows'] if r['task_id']==state['task_id'])
    sites=list(paid['candidate_bank_indices'])+[context['query']];assert len(set(sites))==3
    source_plan=read(execution['pilot_plan'])['source_connection_plan']
    model,_,_=restore_learned_source_prior(state['layout'],sites,source_plan)
    for i,value in enumerate(paid['paired_observations']):model.update_raw(i,[value[0],value[1]-paid['tau']])
    np.testing.assert_allclose(model.weights,context['prior_weights'],rtol=1e-10,atol=0.)
    query_fields=[restrict_field(pair[1],[2]) for pair in model.components]
    for h in (4,5):
        np.testing.assert_allclose(query_fields[h].means,query['fields'][h]['means'],rtol=0.,atol=1e-10)
        np.testing.assert_allclose(query_fields[h].covariances,query['fields'][h]['covariances'],rtol=0.,atol=1e-10)
        np.testing.assert_allclose(query_fields[h].log_weights,query['fields'][h]['log_weights'],rtol=0.,atol=1e-10)
        # Use the exact historical snapshot for cached policy conditioning.
        query_fields[h]=field_from_snapshot(query['fields'][h])
    origins={};moments={};selected=[]
    for origin in outer['origins']:
        banks=[read(outer['output']+f'/o{origin}_r16_v64_s{s}/bank.json') for s in range(2)]
        points=np.concatenate([b['probabilities'] for b in banks])
        labels=np.concatenate([b['labels'] for b in banks]);etas=np.concatenate([b['etas'] for b in banks])
        weights=np.concatenate([b['weights'] for b in banks]);axes=[]
        for axis,anchor in enumerate(spec['switch_cdf']):
            values=np.unique(points[:,axis]);axes.append([values[values<anchor][-1],values[values>anchor][0]])
        rows=[]
        for mean in axes[0]:
            for variance in axes[1]:
                index=int(np.flatnonzero((points[:,0]==mean)&(points[:,1]==variance))[0])
                record=f'positive_n4_o{origin}_h1_p{index}';scores=[];saved={}
                for h,field in enumerate(query_fields):
                    local=local_raw_moments(field.means,field.covariances,float(labels[index]),float(etas[index]),4)
                    scores.append(float(logsumexp(field.log_weights+local['log_evidence'])))
                    if h in (4,5):saved[str(h)]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in local.items()}
                row=dict(record=record,position=index,label=float(labels[index]),eta_label=float(etas[index]),
                    head_log_scores=scores,cdf=points[index].tolist(),outer_weight=float(weights[index]))
                rows.append(row);moments[record]=saved;selected.append(dict(origin=origin,**row))
        origins[str(origin)]=dict(context['origins'][str(origin)],risk=rows)
    context.update(origins=origins,risk_local_moments=moments,
        risk_query_fields={str(h):query['fields'][h] for h in (4,5)})
    compressed_write(out/'context.json.gz',context)
    count=len(selected);historical=read(old['output']+'/preparation.json');jobs=[]
    for h in (4,5):
        rate=max(j['maximum_measured_nonrule_cpu_per_policy'] for j in historical['jobs'] if j['identity']==h)
        for shard,policies in enumerate(spec['policy_shards']):
            jobs.append(dict(identity=h,shard=shard,policies=policies,
                matched_cpu_per_observation_policy=rate,estimated_cpu_seconds=1.2*rate*count*len(policies)+2))
    total=600+sum(j['estimated_cpu_seconds'] for j in jobs)
    result=dict(complete=True,jobs=jobs,selected=selected,risk_observations=count,objective_heads_per_origin=16,
        action_count=365,cumulative_estimated_cpu_seconds=total,estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds'],
        source_components={str(h):len(query['fields'][h]['means']) for h in (4,5)},
        cpu_seconds=time.process_time()-started,full_posterior_restorations=0,new_physical_calls=0,
        native_replayed_sites=sites,native_paid_records_replayed=2,all_identity_likelihoods=8,
        new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        scope='Eight positive-grid coordinates bracketing the old switch. All365 actions and8 identity scores;cached local paid moments reused across policy marginals. Diagnostic only;no outer KG estimate from this subset.')
    save(out/'preparation.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default=PLAN);prepare(p.parse_args().plan)
