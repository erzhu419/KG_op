"""Close fixed-action outer integration before expanding the full KG bank."""

import argparse
import copy
import json
import signal
import time

import numpy as np
from scipy.stats import qmc

from core.observed_batch_predictive import ObservedBatchPredictive
from core.positive_mixture_lookahead import RawPredictiveCDF
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.vectorized_joint_function_risk import update_raw_observation
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_predictive_tower import restrict_field, query_direct_value, direct_value
from performance.historical_full_bank_batch_kg import restore

PLAN='performance/manifests/historical_observed_prediction_v1.json'


def prepare(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    if (out/'state.json').exists():return
    started=time.process_time();parent=read(read(spec['parent_plan'])['parent_plan'])
    model,terminal,_,_=restore(parent,spec['state']);fields={}
    for origin in spec['origins']:
        fields[str(origin)]=[]
        for original in model.components[origin]:
            field=restrict_field(original,[parent['query']])
            fields[str(origin)].append(dict(means=field.means.tolist(),covariances=field.covariances.tolist(),
                log_weights=field.log_weights.tolist(),paid_batches=field.paid_batches,paid_episodes=field.paid_episodes))
    baseline={}
    for n in spec['replicates']:
        directory=parent['output']+'/'+spec['state']+'/n'+str(n)
        row=read(directory+'/baseline/terminal_rules.json')['rows'][-1]
        action=row['best_action'];assert action['kind']=='direct' and action['primary']==parent['query']
        baseline[str(n)]=dict(action=action,reward=row['best_reward'])
    save(out/'state.json',dict(fields=fields,prior_weights=model.weights.tolist(),baseline=baseline,
        protocol=terminal.protocol,query=parent['query'],predictive_seed=parent['predictive_seed'],
        integration_seed=parent['seed'],original_output=parent['output']))
    result=dict(complete=True,cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,
        new_verifier_calls=0,source_state_restored_once=True)
    save(out/'preparation.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def field_from_snapshot(payload):
    means=np.asarray(payload['means']);covariance=np.asarray(payload['covariances'])
    count=means.shape[1]//2
    field=JointFunctionRiskPosterior.from_moments(means[0].reshape(count,2),covariance[0].reshape(count,2,count,2))
    field.means=means;field.covariances=covariance;field.log_weights=np.asarray(payload['log_weights'])
    field.paid_batches=payload['paid_batches'];field.paid_episodes=payload['paid_episodes'];return field


def conditioned_value(field,label,eta,n,head,budget,protocol,spec):
    future=copy.deepcopy(field);score=update_raw_observation(future,0,label,eta,n);previous=None
    for order in spec['inner_orders']:
        value,error=query_direct_value(field,future,label,score,head,budget,protocol,n,eta,order)
        passed=previous is not None and abs(value-previous)<=spec['maximum_inner_error'] and error<=spec['maximum_inner_error']
        if passed:break
        previous=value
    return value,error,passed,order,score


def run(n,origin,head,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];directory=out/('n'+str(n))/str(origin)/str(head)
    directory.mkdir(parents=True,exist_ok=True)
    previous=read(str(directory/'results.json')) if (directory/'results.json').exists() else {}
    count=2**spec['predictive_power']
    cached=len(previous.get('values',[])) if previous.get('complete') and previous.get('all_inner_rules_passed') else 0
    if cached>=count:return
    if cached:save(directory/('results_before'+str(count)+'.json'),previous)
    started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_job_cpu_seconds'])
    result=dict(complete=False,n=n,origin=origin,head=head,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    try:
        state=read(spec['output']+'/state.json');field=field_from_snapshot(state['fields'][str(origin)][head])
        fixed=state['baseline'][str(n)];budget=fixed['action']['budgets'][0];protocol=state['protocol']
        oldbank=read(state['original_output']+'/'+spec['state']+'/bank.json')['batches'][str(n)]
        nodes=qmc.Sobol(2 if n==1 else 4,scramble=True,
            seed=state['predictive_seed']).random_base2(spec['predictive_power'])[cached:]
        if n==1:
            probabilities=nodes[:,head];cdf=RawPredictiveCDF(field,0,spec['cdf_order']);labels=cdf.inverse(probabilities)
            if not cached:
                original=np.asarray(oldbank['labels'])[origin*8:(origin+1)*8,head]
                roundoff=float(np.max(abs(labels[:8]-original)));assert roundoff<=1e-8
                labels[:8]=original;result['old_label_roundoff']=roundoff
            reference,bound=cdf.adaptive(labels);cdf_error=float(np.max(abs(reference-probabilities)));etas=None
        else:
            probabilities=nodes[:,2*head:2*head+2]
            for order in spec['batch_cdf_orders']:
                cdf=ObservedBatchPredictive(field,0,n,order);labels,etas=cdf.inverse(probabilities)
                vref,mref,bound=cdf.adaptive_coordinates(labels,etas)
                cdf_error=max(float(np.max(abs(vref-probabilities[:,1]))),float(np.max(abs(mref-probabilities[:,0]))))
                if cdf_error<=spec['maximum_inverse_cdf_error'] and bound<=spec['maximum_adaptive_cdf_error_bound']:break
            result['cdf_order']=order
        cdf_passed=cdf_error<=spec['maximum_inverse_cdf_error'] and bound<=spec['maximum_adaptive_cdf_error_bound']
        result.update(maximum_inverse_cdf_error=max(cdf_error,previous.get('maximum_inverse_cdf_error',0.) if cached else 0.),
            adaptive_cdf_error_bound=max(bound,previous.get('adaptive_cdf_error_bound',0.) if cached else 0.),predictive_cdf_passed=cdf_passed)
        bank=read(str(directory/'bank.json')) if cached else dict(labels=[],etas=[])
        save(directory/'bank.json',dict(labels=bank['labels'][:cached]+labels.tolist(),
            etas=None if etas is None else bank['etas'][:cached]+etas.tolist()))
        if not cdf_passed:
            result['complete']=True;result['inner_evaluation_skipped']=True;return
        # Same old law and marginal state at the old observed coordinates.
        parity=previous['full_cache_parity_error'] if cached else 0.
        if not cached:
            old=read(state['original_output']+'/'+spec['state']+'/n'+str(n)+'/head/'+str(head)+'_'+str(origin)+'/head_records.json')
            old['records'].sort(key=lambda r:r['position']);fixed_index=old['actions'].index(fixed['action'])
            qnodes=qmc.Sobol(7,scramble=True,seed=state['integration_seed']).random_base2(10)
            if n==1:np.testing.assert_array_equal(labels[:8],np.asarray(oldbank['labels'])[origin*8:(origin+1)*8,head])
            for i in range(8):
                label=oldbank['labels'][origin*8+i][head]
                eta=None if n==1 else oldbank['eta'][origin*8+i][head]
                future=copy.deepcopy(field);score=update_raw_observation(future,0,label,eta,n)
                assert abs(score-old['records'][i]['head_log_scores'][origin])<=1e-10
                value,_=direct_value(field,future,0,qnodes,label,score,head,budget,protocol,n,eta)
                parity=max(parity,abs(value-old['records'][i]['raw']['10']['statistics'][origin][fixed_index][0]))
        assert parity<=1e-10
        values=previous['values'].copy() if cached else [];errors=[previous['maximum_evidence_error']] if cached else []
        admissions=[True]*cached;orders=previous['orders'].copy() if cached else []
        for i,label in enumerate(labels):
            eta=None if etas is None else etas[i]
            value,error,passed,order,_=conditioned_value(field,label,eta,n,head,budget,protocol,spec)
            values.append(value);errors.append(error);admissions.append(passed);orders.append(order)
            if (i+1)%64==0:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=cached+i+1,total=count,unit='fixed_action_labels',
                    metrics=dict(n=n,origin=origin,head=head,mean=float(np.mean(values)),cpu_seconds=time.process_time()-started))),flush=True)
        assert values[:cached]==previous.get('values',[])[:cached]
        result.update(complete=True,full_cache_parity_error=parity,values=values,orders=orders,
            reused_observations=cached,new_observations=len(labels),
            prefixes={str(p):float(np.mean(values[:p])) for p in spec['prefixes']},
            all_inner_rules_passed=all(admissions),inner_rule_failed_count=sum(not x for x in admissions),
            maximum_evidence_error=max(errors),baseline_reward=fixed['reward'],
            origin_prior_mass=state['prior_weights'][origin],observation_dimensions_per_head=1 if n==1 else 2)
    except TimeoutError:result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        result['new_cpu_seconds']=time.process_time()-started
        result['cpu_seconds']=result['new_cpu_seconds']+(previous['cpu_seconds'] if cached else 0.)
        save(directory/'results.json',result)
        print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('values','orders')}),flush=True)


def collect(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];state=read(spec['output']+'/state.json');cases=[];executions=[]
    for n in spec['replicates']:
        heads=[]
        for origin in spec['origins']:
            for head in (0,1):
                row=read(spec['output']+'/n'+str(n)+'/'+str(origin)+'/'+str(head)+'/results.json')
                heads.append({k:v for k,v in row.items() if k not in ('values','orders')})
        complete=all(h['complete'] and not h.get('inner_evaluation_skipped') for h in heads)
        admitted=complete and all(h['all_inner_rules_passed'] and h['predictive_cdf_passed'] for h in heads)
        row=dict(n=n,heads=heads,complete=complete,all_cdf_and_inner_rules_passed=admitted)
        if complete:
            by={(h['origin'],h['head']):h for h in heads};prefixes={}
            shared_prefixes=sorted(set.intersection(*(set(h['prefixes']) for h in heads)),key=int)
            for p in map(int,shared_prefixes):
                reward=sum(state['prior_weights'][o]*by[o,0]['prefixes'][str(p)]*by[o,1]['prefixes'][str(p)] for o in spec['origins'])
                prefixes[str(p)]=dict(expected_reward=reward,tower_drift=reward-state['baseline'][str(n)]['reward'])
            row.update(prefixes=prefixes,fixed_action_tower_diagnostic_passed=admitted and
                abs(prefixes[shared_prefixes[-1]]['tower_drift'])<=spec['maximum_tower_drift'])
            change=abs(prefixes[shared_prefixes[-1]]['expected_reward']-prefixes[shared_prefixes[-2]]['expected_reward'])
            row.update(last_predictive_resolution_change=change,
                fixed_action_predictive_integral_admitted=row['fixed_action_tower_diagnostic_passed'] and change<=spec['maximum_tower_drift'])
        cases.append(row);executions.extend(heads)
    preparation=read(spec['output']+'/preparation.json')
    result=dict(cases=cases,preparation=preparation,cpu_seconds=preparation['cpu_seconds']+sum(h['cpu_seconds'] for h in executions),
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        complete_adaptive_KG_bank_changed=False,omitted_origin_mass=sum(w for o,w in enumerate(state['prior_weights']) if o not in spec['origins']))
    save(out/'phase_results.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','case','collect'])
    parser.add_argument('--plan',default=PLAN);parser.add_argument('--n',type=int);parser.add_argument('--origin',type=int);parser.add_argument('--head',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.plan)
    elif args.mode=='collect':collect(args.plan)
    else:run(args.n,args.origin,args.head,args.plan)
