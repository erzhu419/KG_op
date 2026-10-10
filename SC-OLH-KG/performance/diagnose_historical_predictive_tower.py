"""Cheap fixed-action law check before expanding the complete KG action bank."""

import argparse
import copy
import json
import signal
import time

import numpy as np
from scipy.special import ndtr, logsumexp, roots_hermitenorm
from scipy.stats import qmc

from core.conditional_raw_reference import conditional_raw_coordinates
from core.guarded_deployment_kg import _sd_quadrature, _tolerance_factor
from core.legacy_joint_kg import clipped_objective_quality
from core.positive_mixture_lookahead import predictive_positive_batches, raw_eta_proposal
from representation.joint_response_mixture import JointResponseMixture
from representation.vectorized_joint_function_risk import update_raw_observation
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_batch_kg import restore
from performance.refine_historical_achievable_precision import PLAN


def restrict_field(field, points):
    result = copy.copy(field); ix = (2*np.asarray(points)[:,None]+np.arange(2)).reshape(-1)
    result.means = field.means[:,ix].copy()
    result.covariances = field.covariances[:,ix][:,:,ix].copy()
    result.log_weights = field.log_weights.copy(); result.candidate_count = len(points)
    return result


def direct_value(field, future, deploy, nodes, label, score, head, remaining, protocol, n, eta_label):
    eta,mean,covariance,weights,diagnostics = conditional_raw_coordinates(field,future,(deploy,),nodes,
        label,score,query=0,replicates=n,eta_label=eta_label)
    if head == 0:
        value = weights @ clipped_objective_quality(mean,np.maximum(covariance[:,0,0],0.)[:,None],
            protocol['objective_reference'])[:,0]
    else:
        sd,quadrature = _sd_quadrature(remaining)
        threshold = _tolerance_factor(remaining,protocol['chance_alpha'],protocol['familywise_delta'])*sd
        sigma = np.exp(.5*eta[:,0]); total = np.sqrt(np.maximum(covariance[:,0,0],0.)+sigma**2/remaining)
        value = weights @ (ndtr(-(mean+sigma[:,None]*threshold)/total[:,None]) @ quadrature)
    return float(value), abs(diagnostics['predictive_log_evidence_error'])


def query_direct_value(field,future,label,score,head,budget,protocol,n,eta_label,order):
    """All positive component masses and Gaussian eta quadrature, query=deploy."""
    z,w=roots_hermitenorm(order);component=np.repeat(np.arange(len(field.means)),order)
    eta,logs=raw_eta_proposal(field,future,0,component,np.tile(z,len(field.means)),label,score,
        replicates=n,eta_label=eta_label)
    mu,em=field.means[component,:2].T;C=field.covariances[component,:2,:2]
    cv=np.maximum(C[:,0,0]-C[:,0,1]**2/C[:,1,1],0.)
    before=mu+C[:,0,1]/C[:,1,1]*(eta-em);noise=np.exp(eta)/n
    mean=before+cv/(cv+noise)*(label-before);variance=cv*noise/(cv+noise)
    logs+=future.log_weights[component]+np.tile(np.log(w/np.sqrt(2*np.pi)),len(field.means))
    normalizer=logsumexp(logs);weights=np.exp(logs-normalizer)
    if head==0:
        value=weights@clipped_objective_quality(mean,variance,protocol['objective_reference'])
    else:
        sd,quadrature=_sd_quadrature(budget)
        threshold=_tolerance_factor(budget,protocol['chance_alpha'],protocol['familywise_delta'])*sd
        sigma=np.exp(.5*eta);total=np.sqrt(variance+sigma**2/budget)
        value=weights@(ndtr(-(mean[:,None]+sigma[:,None]*threshold)/total[:,None])@quadrature)
    return float(value),abs(float(normalizer))


def summarize_tower(heads,case):
    expected=[(0,h) for h in (0,1)] if case.startswith('gp_') else [(o,h) for o in (4,5) for h in (0,1)]
    result=dict(heads=heads,all_complete=len(heads)==len(expected) and all(h['complete'] for h in heads),
        cpu_seconds=sum(h['cpu_seconds'] for h in heads),new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        all_inner_rules_passed=len(heads)==len(expected) and all(h.get('all_inner_rules_passed',False) for h in heads),
        diagnostic_case=case,scope='Fixed direct action only; original adaptive KG bank is unchanged.')
    if not result['all_complete']:return result
    by={(h.get('origin',0),h['head']):h for h in heads};assert set(by)==set(expected)
    baseline=heads[0]['baseline_reward'];assert all(h['baseline_reward']==baseline for h in heads)
    origins=sorted({o for o,h in expected})
    masses={o:1. if case.startswith('gp_') else by[o,0]['origin_prior_mass'] for o in origins}
    prefixes={}
    for n in (8,64,512):
        reward=sum(masses[o]*by[o,0]['prefixes'][str(n)]*by[o,1]['prefixes'][str(n)] for o in origins)
        prefixes[str(n)]=dict(expected_reward=reward,tower_drift=reward-baseline)
    result.update(prefixes=prefixes,omitted_prior_mass=1-sum(masses.values()))
    result['fixed_action_tower_diagnostic_passed']=result['all_inner_rules_passed'] and abs(prefixes['512']['tower_drift'])<=1e-4
    return result


def collect(case,plan=PLAN):
    output=ROOT/read(plan)['output'];folder=output/({'gp_n4':'predictive_tower','gp_n1':'predictive_tower_n1','source_n4':'source_tower_gh'}[case])
    files=folder.glob('*/results.json' if case.startswith('gp_') else '*/*/results.json')
    heads=[{k:v for k,v in read(str(f)).items() if k not in ('values','rules','errors')} for f in sorted(files)]
    result=summarize_tower(heads,case);save(folder/'phase_results.json',result)
    print(json.dumps(result),flush=True)


def run(head, plan=PLAN, case_name='gp_n4', origin=0):
    spec=read(plan);parent=read(spec['parent_plan']);case=next(c for c in spec['cases'] if c['name']==case_name)
    folder={'gp_n4':'predictive_tower','gp_n1':'predictive_tower_n1','source_n4':'source_tower_gh'}[case_name]
    directory=ROOT/spec['output']/folder
    if case_name=='source_n4':directory=directory/str(origin)
    directory=directory/str(head);directory.mkdir(parents=True,exist_ok=True);n=case['n']
    if (directory/'results.json').exists() and read(str(directory/'results.json')).get('complete'):return
    started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,420.)
    result=dict(head=head,case=case_name,origin=origin,complete=False,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        scope='Fixed direct action within one complete identity; no adaptive maximum factorization.')
    try:
        model,terminal,_,_=restore(parent,case['state'])
        baseline_root=parent['accepted_baseline_inputs'].get(case['state'],parent['output'])
        baseline=read(baseline_root+'/'+case['state']+'/n'+str(n)+'/baseline/terminal_rules.json')['rows'][-1]
        action=baseline['best_action'];assert action['kind']=='direct'
        points=list(dict.fromkeys([parent['query'],action['primary']]))
        deploy=points.index(action['primary']);pair=[restrict_field(f,points) for f in model.components[origin]]
        result['origin_prior_mass']=float(model.weights[origin]);del model
        small=JointResponseMixture(['fixed_identity'],[pair])
        labels,etas,_=predictive_positive_batches(small,0,n,9,parent['predictive_seed'])
        bank=read(parent['output']+'/'+case['state']+'/bank.json')['batches'][str(n)]
        difference=float(np.max(abs(labels[:8]-bank['labels'][8*origin:8*(origin+1)])))
        assert difference<=1e-8;labels[:8]=bank['labels'][8*origin:8*(origin+1)];result['old_label_roundoff']=difference
        if etas is not None:np.testing.assert_array_equal(etas[:8],bank['eta'][8*origin:8*(origin+1)])
        cached=read(parent['output']+'/'+case['state']+'/n'+str(n)+'/head/'+str(head)+'_'+str(origin)+'/head_records.json')
        cached['records'].sort(key=lambda r:r['position'])
        fixed=cached['actions'].index(action);field=pair[head];values=[];rules=[];errors=[];admissions=[];parity=0.
        nodes={p:qmc.Sobol(7,scramble=True,seed=parent['seed']).random_base2(p) for p in (10,12,14,16)}
        for i in range(len(labels)):
            future=copy.deepcopy(field)
            eta_label=None if etas is None else etas[i,head]
            score=update_raw_observation(future,0,labels[i,head],eta_label,n)
            if i<8:assert abs(score-cached['records'][i]['head_log_scores'][origin])<=1e-10
            if i<8:
                reference,_=direct_value(field,future,deploy,nodes[10],labels[i,head],score,head,
                    action['budgets'][0],terminal.protocol,n,eta_label)
                parity=max(parity,abs(reference-cached['records'][i]['raw']['10']['statistics'][origin][fixed][0]))
            row=[]
            for power in ((16,32,64,128) if case_name=='source_n4' else (10,12,14,16)):
                if case_name=='source_n4':
                    assert deploy==0
                    value,error=query_direct_value(field,future,labels[i,head],score,head,action['budgets'][0],
                        terminal.protocol,n,eta_label,power)
                else:
                    value,error=direct_value(field,future,deploy,nodes[power],labels[i,head],score,head,action['budgets'][0],
                        terminal.protocol,n,eta_label)
                row.append(value)
                admitted=len(row)>1 and abs(row[-1]-row[-2])<=1e-4 and error<=1e-4
                if admitted:break
            values.append(value);rules.append(power);errors.append(error);admissions.append(admitted)
            if i+1 in (8,64,512):
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(step=i+1,total=512,unit='fixed_action_labels',
                    metrics=dict(head=head,mean=float(np.mean(values)),cpu_seconds=time.process_time()-started))),flush=True)
        assert parity<=1e-10
        result.update(complete=True,values=values,rules=rules,errors=errors,full_cache_parity_error=parity,
            prefixes={str(n):float(np.mean(values[:n])) for n in (8,64,512)},baseline_reward=baseline['best_reward'],
            maximum_evidence_error=max(errors),all_inner_rules_passed=all(admissions),
            inner_rule_failed_count=sum(not a for a in admissions))
        result['inner_rule_type']='all_component_Gauss_Hermite_order' if case_name=='source_n4' else 'Sobol_power'
    except TimeoutError:result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        result['cpu_seconds']=time.process_time()-started;save(directory/'results.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('values','rules','errors')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--head',type=int,choices=[0,1])
    parser.add_argument('--plan',default=PLAN);parser.add_argument('--case',choices=['gp_n1','gp_n4','source_n4'],default='gp_n4')
    parser.add_argument('--origin',type=int,default=0);parser.add_argument('--collect',action='store_true');args=parser.parse_args()
    if args.collect:collect(args.case,args.plan)
    elif args.head is None:parser.error('--head is required unless --collect is used')
    else:run(args.head,args.plan,args.case,args.origin)
