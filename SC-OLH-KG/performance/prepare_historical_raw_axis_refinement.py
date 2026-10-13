"""Reuse a completed grid and price one local CDF-axis refinement."""
import argparse
import copy
import json

from performance.check_historical_observed_batch_outer import outer_rule
from performance.complete_historical_source_heads import ROOT, read, save
from performance.launch_historical_component_bank import QUEUE


def prepare(mode, source_bound):
    bound=read(source_bound);source=read(bound['bank_plan'])
    assert read(bound['output']+'/compact_summary.json')['numerical_containment_passed']
    previous=read(bound['joint_reference_plans'][-1])
    completion=previous['output']+'/completion.json';closed=read(completion)
    assert closed['state']=='phase_complete' and closed['all_reference_precision_passed']
    assert closed['native_choices_all_raw_referenced']
    intervals=source['mean_cdf_intervals'];orders=source.get('risk_variance_orders',[source['risk_variance_order']]*len(intervals))
    if mode in ('midmean','uppermidmean'):
        target=[.8,.95] if mode=='midmean' else [.875,.95]
        intervals.index(target)
        midpoint=sum(target)/2
        new_intervals=[];new_orders=[]
        for interval,vorder in zip(intervals,orders,strict=True):
            pieces=[[target[0],midpoint],[midpoint,target[1]]] if interval==target else [interval]
            new_intervals.extend(pieces);new_orders.extend([vorder]*len(pieces))
        comparison=f'Split {target} at {midpoint}; all other mean and variance rules unchanged.'
    else:
        assert mode=='uppervariance'
        new_intervals=copy.deepcopy(intervals);new_orders=list(orders)
        index=intervals.index([.95,.99]);assert new_orders[index]==4
        new_orders[index]=8
        comparison='Variance order4 versus8 only on the [.95,.99] mean interval; all other coordinates unchanged.'
    points,weights=outer_rule(source['risk_mean_order'],source['risk_variance_order'],new_intervals,variance_orders=new_orders)
    old={tuple(p):w for p,w in zip(source['probability_nodes_per_origin'],source['positive_weights_per_origin'])}
    cached=[tuple(p) in old for p in points]
    assert all(old[tuple(p)]==w for p,w,reused in zip(points,weights,cached) if reused)
    origins=len(source['origins']);fresh=origins*sum(not x for x in cached)
    assert fresh==64

    rates={h:max(j['matched_cpu_per_observation_policy'] for j in source['screen_jobs'] if j['identity']==h) for h in (4,5)}
    native_rate=max(j['matched_cpu_per_row'] for j in source['native_direct_jobs'])
    resources=read(source['output']+'/worker_resources.json')
    tasks={t['id']:t for t in json.loads(QUEUE.read_text())['tasks']}
    by_shard={j['shard']:j for j in source['native_direct_jobs']}
    for resource in resources:
        pieces=tasks[resource['id']]['signature'].split('/')
        identity,kind,shard=pieces[-3:];shard=int(shard)
        cpu=resource['resource']['process_cpu_seconds']
        if kind=='screen':
            policies=source['policy_shards'][shard]
            rates[int(identity)]=max(rates[int(identity)],cpu/(source['new_risk_observations']*len(policies)))
        else:
            assert kind=='native-direct'
            native_rate=max(native_rate,cpu/by_shard[shard]['rows'])
    jobs=[]
    for h in (4,5):
        for shard,policies in enumerate(source['policy_shards']):
            jobs.append(dict(identity=h,shard=shard,policies=policies,matched_cpu_per_observation_policy=rates[h],
                priced_new_risk_observations=fresh,estimated_cpu_seconds=1.2*rates[h]*fresh*len(policies)+2.))
    native=[];all_cached=cached*origins
    rows_per_record=source['native_direct_jobs'][0]['rows']//8
    for shard,start in enumerate(range(0,len(all_cached),8)):
        reused=all(all_cached[start:start+8]);rows=len(all_cached[start:start+8])*rows_per_record
        native.append(dict(shard=shard,matched_cpu_per_row=native_rate,rows=rows,reused=reused,
            estimated_cpu_seconds=0. if reused else 1.2*native_rate*rows+2.))
    total=600.+sum(j['estimated_cpu_seconds'] for j in jobs+native)
    assert total<=3600 and all(j['estimated_cpu_seconds']<=420 for j in jobs+native)
    name=f'historical_positive_risk_{mode}_axis_v1';plan='performance/manifests/'+name+'.json';output='paper_artifacts/'+name
    (ROOT/output).mkdir(parents=True,exist_ok=True)
    spec=copy.deepcopy(source)
    spec.update(output=output,bank_plan=plan,native_reference_plan=plan,risk_record_prefix=f'positive_{mode}_v1_n4_m4_localv',
        mean_cdf_intervals=new_intervals,risk_variance_orders=new_orders,interval_choice=comparison,
        probability_nodes_per_origin=points.tolist(),positive_weights_per_origin=weights.tolist(),
        positive_weight_sum_per_origin=float(weights.sum()),risk_observations=origins*len(points),
        reused_risk_observations=origins*sum(cached),new_risk_observations=fresh,
        reused_bank_plan=source_bound,reused_mean_cdf_minimum=0.,screen_jobs=jobs,native_direct_jobs=native,
        native_direct_cost_plan=output+'/native_direct_cost_plan.json',continued_phase_cost_plan=output+'/preparation.json',
        joint_reference_plans=bound['joint_reference_plans'],pricing_reference_plans=bound['pricing_reference_plans'],
        previous_conditional_completion=completion,estimated_cpu_seconds_including600reserves=total,
        estimated_screen_cpu_seconds=sum(j['estimated_cpu_seconds'] for j in jobs),
        estimated_native_direct_cpu_seconds=sum(j['estimated_cpu_seconds'] for j in native),
        outer_axis=mode,scope='Local CDF refinement with exact cached-coordinate reuse. Original prior,8 likelihoods,512/1024 components,full covariance,F16,365 actions,356 calls and SD16 retained;confirmation unopened.')
    spec['native_direct_pricing'].update(rows=sum(j['rows'] for j in native),matched_cpu_per_row=native_rate)
    spec.pop('reference_phase_cost_plan',None)
    spec['accuracy_contract']['local_axis_comparison']=comparison
    spec['pricing']='Maximum recorded same-identity screen/native row rates,20percent margin,2setup perworker,one shared600reserve; prior phases remain charged.'
    save(ROOT/plan,spec)
    risk_name=f'historical_positive_risk_{mode}_axis_incumbents_v1';risk_plan='performance/manifests/'+risk_name+'.json'
    bound_name=f'historical_positive_risk_{mode}_axis_bound_incumbents_v1';bound_plan='performance/manifests/'+bound_name+'.json'
    risk=copy.deepcopy(previous)
    risk.update(output='paper_artifacts/'+risk_name,screen_plan=plan,incumbent_bound_plan=plan,
        previous_phase_completion=completion,frozen_cost_plan='paper_artifacts/'+risk_name+'/next_cost_plan.json',
        followup_bound_plan=bound_plan,continued_phase_cost_plan=output+'/preparation.json',refined_cases=[],
        scope='Complete raw coverage within the registered local-axis phase; preserve its original shared600 reserve.')
    risk.pop('failed_preparation_account',None);risk.pop('worker_limit_adjustment',None)
    follow=copy.deepcopy(spec)
    follow.update(output='paper_artifacts/'+bound_name,joint_reference_plans=spec['joint_reference_plans']+[risk_plan],
        pricing_reference_plans=spec['pricing_reference_plans']+[risk_plan],
        reference_phase_cost_plan=risk['output']+'/preparation.json',continued_phase_cost_plan=risk['output']+'/preparation.json')
    save(ROOT/risk_plan,risk);save(ROOT/bound_plan,follow)
    frozen=output+'/incumbent_frozen_plan.json'
    save(ROOT/frozen,dict(plans=[risk_plan],maximum_live_workers=8,priority='low',scope='Same-phase raw references after local-axis screening.'))
    result=dict(mode=mode,plan=plan,files=[plan,risk_plan,bound_plan,frozen,completion],source_bound_plan=source_bound,
        risk_observations=spec['risk_observations'],reused_risk_observations=spec['reused_risk_observations'],
        new_risk_observations=fresh,initial_workers=len(jobs)+sum(not j['reused'] for j in native),
        estimated_initial_phase_cpu_seconds=total,maximum_live_workers=8,comparison=comparison)
    save(ROOT/output/'refinement_plan.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('midmean','uppermidmean','uppervariance'),required=True)
    parser.add_argument('--source-bound',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.mode,args.source_bound)))
