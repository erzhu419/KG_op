"""Reuse the admitted bank and historical positive CDF interval rules."""

import argparse
import json
import signal
import time

import numpy as np
from scipy.stats import qmc

from core.raw_terminal_head_grid import evaluate_head_records
from core.raw_likelihood_lookahead import _actions
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_batch_kg import restore
from performance.refine_historical_achievable_precision import assemble_case, grid, prepare as prepare_precision, rule_pair, all_head_checks, original_heads
from performance.refine_historical_remaining_query_failures import positive_risk_tail,positive_cdf_interval

PLAN='performance/manifests/historical_predictive_upper_risk_v6.json'


def prepare(plan=PLAN):
    spec=read(plan);old=read(spec['reuse_precision_plan']);case=spec['cases'][0]
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    if (out/'preparation.json').exists():return
    previous=read(old['output']+'/outcome_summary.json')['cases'][0]
    assert previous['terminal_diagnostics_passed'] and previous['gates']['baseline']
    started=time.process_time()
    heads,bank,_=assemble_case(old,old['cases'][0],read(old['output']+'/registered_jobs.json')['jobs'])
    if 'risk_rule_input' in spec:
        cached=read(spec['risk_rule_input'])
        assert cached['fixed_action_only_admitted'] and cached['selected_body_orders']==spec['risk_tail_orders']
        assert cached['query']==spec['query'];np.testing.assert_array_equal(cached['prior_weights'],bank['prior_weights'])
    else:
        model,_,_,_=restore(spec,case['state']);np.testing.assert_array_equal(model.weights,bank['prior_weights'])
    rules={};jobs=[];groups=[]
    for origin in bank['origins']:
        rules[str(origin)]={};position=spec['reused_risk_count']
        for order in spec['risk_tail_orders']:
            if 'risk_rule_input' in spec:
                tail=dict(cached['body_rules'][str(origin)][str(order)],positions=list(range(position,position+order)))
            else:
                tail=positive_risk_tail(model,spec['query'],origin,spec,order=order,position=position,
                    interval=spec['risk_tail_interval'])
            rules[str(origin)][str(order)]=tail;position+=order
            jobs.extend(dict(name=case['name']+'_o'+str(origin)+'_h1_p'+str(p),case=case['name'],
                origin=origin,head=1,position=p,label=y) for p,y in zip(tail['positions'],tail['labels']))
        origin_jobs=[j for j in jobs if j['origin']==origin]
        for start in range(0,len(origin_jobs),12):
            groups.append(dict(name='group'+str(len(groups)),jobs=origin_jobs[start:start+12]))
    # The old records remain in their original directories until staging.
    (out/case['state']).mkdir(parents=True,exist_ok=True)
    save(out/case['state']/'bank.json',bank);save(out/'risk_rules.json',rules)
    result=dict(complete=True,jobs=jobs,groups=groups,cpu_seconds=time.process_time()-started,
        new_risk_states=len(jobs),reused_head_states=sum(len(h) for h in heads.values()),
        posterior_snapshot_reused=True,original_prior_weights_preserved=True,
        large_posterior_restorations=int('risk_rule_input' not in spec),
        maximum_inverse_cdf_error=max(r['maximum_inverse_cdf_error'] for rows in rules.values() for r in rows.values()),
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(out/'preparation.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('jobs','groups')}),flush=True)


def run_group(name,workers,plan=PLAN):
    spec=read(plan);case=spec['cases'][0];prep=read(spec['output']+'/preparation.json')
    group=next(g for g in prep['groups'] if g['name']==name);root=ROOT/spec['output']
    pending=[j for j in group['jobs'] if not (root/'new_heads'/j['name']/'results.json').exists()
        or not read(str(root/'new_heads'/j['name']/'results.json'))['complete']]
    if not pending:return
    started=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_case_cpu_seconds']*len(pending))
    result=dict(name=name,complete=False,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    try:
        model,terminal,_,_=restore(spec,case['state']);expected=_actions(terminal,359)
        bank=read(spec['output']+'/'+case['state']+'/bank.json');by_position={j['position']:j for j in pending}
        completed=[];finished=set()
        for job in pending:
            path=root/'new_heads'/job['name']/'head_records.json'
            if path.exists():completed.extend(read(str(path))['records'])
        def checkpoint(records,actions):
            assert actions==expected
            for row in records:
                p=row['position'];job=by_position[p];out=root/'new_heads'/job['name'];out.mkdir(parents=True,exist_ok=True)
                save(out/'head_records.json',dict(records=[row],actions=actions,prior_weights=model.weights.tolist()))
                if p in finished or any(str(power) not in row[k] for k in ('raw','projected') for power in spec['powers']):continue
                save(out/'results.json',dict(name=job['name'],complete=True,cpu_seconds=row['cpu_seconds'],
                    new_physical_calls=0,new_source_fits=0,new_verifier_calls=0))
                finished.add(p)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(finished),total=len(pending),unit='risk_head_states')),flush=True)
        evaluate_head_records(model,terminal,360,spec['query'],1,{j['position']:j['label'] for j in pending},
            list(by_position),powers=spec['powers'],projected_powers=spec['powers'],seed=spec['seed'],
            completed=completed,checkpoint=checkpoint,utility_identities=bank['utility_identities'],
            guarded_mean_position=0,workers=workers)
        result.update(complete=len(finished)==len(pending),completed_records=len(finished))
    except TimeoutError:result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        result['cpu_seconds']=time.process_time()-started;directory=root/'groups'/name;directory.mkdir(parents=True,exist_ok=True)
        save(directory/'results.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def stage(plan=PLAN):
    started=time.process_time();spec=read(plan);old=read(spec['reuse_precision_plan']);precision=read(spec['precision_plan'])
    case=spec['cases'][0];heads,bank,_=assemble_case(old,old['cases'][0],read(old['output']+'/registered_jobs.json')['jobs'])
    base=spec['accepted_baseline_inputs'][case['state']]+'/'+case['state']+'/n1/baseline'
    actions=read(base+'/terminal_rules.json')['rows'][-1]['actions']
    for origin in bank['origins']:
        for head in (0,1):
            rows=list(heads[head,origin])
            if head==1:
                for job in read(spec['output']+'/preparation.json')['jobs']:
                    if job['origin']!=origin:continue
                    path=spec['output']+'/new_heads/'+job['name']
                    assert read(path+'/results.json')['complete']
                    added=read(path+'/head_records.json');assert added['actions']==actions;rows.extend(added['records'])
            assert [r['position'] for r in rows]==list(range(case['head_counts'][head]))
            directory=ROOT/precision['head_inputs'][case['name']][head].format(origin=origin);directory.mkdir(parents=True,exist_ok=True)
            save(directory/'head_records.json',
                dict(records=rows,actions=actions,prior_weights=bank['prior_weights']))
    jobs=prepare_precision(spec['precision_plan'])
    # Start the existing policy-group refiner from completed low-order caches.
    # This replaces one large posterior load per record with shared group loads.
    inputs={}
    for job in jobs:
        key=job['head'],job['origin']
        if key not in inputs:
            inputs[key]=read(precision['head_inputs'][case['name']][job['head']].format(origin=job['origin'])+'/head_records.json')
        data=inputs[key];row=data['records'][job['ordinal']];directory=ROOT/precision['output']/'shards'/job['name']
        directory.mkdir(parents=True,exist_ok=True)
        save(directory/'head_records.json',dict(data,records=[row]))
        checks=job['initial_checks']
        save(directory/'results.json',dict(name=job['name'],complete=True,checks=checks,
            numerical_diagnostics_passed=checks['raw_passed'] and checks['projected_passed'] and checks['evidence_passed'],
            cpu_seconds=0.,low_rules_reused_for_group_refinement=True,
            new_physical_calls=0,new_source_fits=0,new_verifier_calls=0))
    result=dict(complete=True,precision_jobs=len(jobs),cpu_seconds=time.process_time()-started,
        old_inputs_preserved=True,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(ROOT/precision['output']/'staging_summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def head_rules(spec,rules,origins,objective,body_prefix,order):
    if 'retained_tail_rule_input' in spec:
        tail_rules=read(spec['retained_tail_rule_input']);result={}
        for origin in origins:
            body=rules[str(origin)][str(order)];tail=tail_rules[str(origin)][str(body_prefix)]
            result[0,origin]=dict(positions=list(range(objective)),weights=np.full(objective,1./objective).tolist())
            result[1,origin]=dict(positions=body['positions']+tail['positions'],weights=body['weights']+tail['weights'])
        return result
    quantiles=qmc.Sobol(2,scramble=True,seed=spec['predictive_seed']).random_base2(7)[:,1]
    boundary=spec['risk_tail_interval'][0];body=np.flatnonzero(quantiles[:body_prefix]<boundary).tolist()
    result={}
    for origin in origins:
        tail=rules[str(origin)][str(order)]
        result[0,origin]=dict(positions=list(range(objective)),weights=np.full(objective,1./objective).tolist())
        result[1,origin]=dict(positions=body+tail['positions'],
            weights=np.r_[np.full(len(body),boundary/len(body)),tail['weights']].tolist())
    return result


def collect(plan=PLAN):
    started=time.process_time();spec=read(plan);precision=read(spec['precision_plan']);case=precision['cases'][0]
    heads,bank,_=assemble_case(precision,case,read(precision['output']+'/registered_jobs.json')['jobs'])
    base=spec['accepted_baseline_inputs'][case['state']]+'/'+case['state']+'/n1/baseline'
    debited=read(base+'/terminal_rules.json')['rows'][-1];stop=read(spec['stop_inputs'][case['state']])['rows'][-1]
    baseline=stop['best_reward'],debited['best_reward'],debited['best_action_index']
    rules=read(spec['output']+'/risk_rules.json');f0,f1=case['objective_prefixes']
    g0,g1=spec['risk_complement_prefixes'];t0,t1=spec['risk_tail_orders'];rows=[]
    for f,g,t,raw,native in [(f0,g1,t1,1,1),(f1,g0,t1,1,1),(f1,g1,t0,1,1),
            (f1,g1,t1,0,1),(f1,g1,t1,1,0),(f1,g1,t1,1,1)]:
        row=grid(heads,np.asarray(bank['prior_weights']),bank['origins'],[f,g],baseline,raw,native,
            head_rules=head_rules(spec,rules,bank['origins'],f,g,t))
        row.update(objective_prefix=f,risk_complement_prefix=g,risk_tail_order=t);rows.append(row)
        if 'retained_tail_rule_input' in spec:row.update(risk_body_order=t,retained_risk_tail_order=g)
    fine=rows[-1];net=fine['raw_net_reward_of_projected_choice_control_variate']
    changes=[abs(net-r['raw_net_reward_of_projected_choice_control_variate']) for r in rows[:-1]]
    evidence=max(r['raw'][str(rule_pair(r,'raw')[-1])]['predictive_log_evidence_max_error'] for h in heads.values() for r in h)
    gates=dict(baseline=read(base+'/results.json')['numerical_diagnostics_passed'],
        predictive=max(changes)<=spec['maximum_numerical_adaptation_difference'],
        raw_action=fine['maximum_raw_action_resolution_difference']<=spec['maximum_action_resolution_difference'],
        native_action=fine['maximum_native_action_resolution_difference']<=spec['maximum_action_resolution_difference'],
        evidence=evidence<=1e-4,identity=fine['identity_error_bound']<=spec['maximum_identity_integration_error_bound']
            and fine['maximum_future_omitted_utility_mass']<=spec['maximum_identity_integration_error_bound'])
    tower=abs(fine['raw_fixed_action_tower_residual'])<=spec['maximum_numerical_adaptation_difference']
    current=dict(case=case['name'],replicates=1,action_count=len(debited['actions']),rows=rows,
        numerical_changes=changes,gates=gates,predictive_log_evidence_error=evidence,
        terminal_diagnostics_passed=all(v for k,v in gates.items() if k!='predictive'),
        fixed_action_predictive_tower_diagnostic_passed=tower,full_predictive_integration_admitted=all(gates.values()) and tower)
    result=dict(cases=[current],complete=True,collection_cpu_seconds=time.process_time()-started,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        algorithmic_query_selection_complete=False)
    save(ROOT/precision['output']/'outcome_summary.json',result)
    from performance.extend_historical_predictive_heads import compact_cases
    summary=dict(result,cases=compact_cases(result),scope=spec['scope'])
    save(ROOT/precision['output']/'precision_summary.json',summary)
    print('SCHEDULEURM_DONE '+json.dumps(dict(complete=True,cpu_seconds=result['collection_cpu_seconds'])),flush=True)


def finish(power,plan=PLAN):
    from performance.refine_historical_achievable_policies import merge
    started=time.process_time();spec=read(plan);precision=read(spec['precision_plan'])
    merge(power,spec['precision_plan']);collect(plan)
    summary=read(precision['output']+'/precision_summary.json')
    summary['finish_cpu_seconds']=time.process_time()-started
    save(ROOT/precision['output']/('precision_summary_at'+str(power)+'.json'),summary)
    print('SCHEDULEURM_DONE '+json.dumps(dict(complete=True,cpu_seconds=summary['finish_cpu_seconds'])),flush=True)


def validate_bulk(plan=PLAN):
    """Compare the batched checker to the saved individual416-record admission."""
    start=time.process_time();parent=read(plan);spec=read(parent['precision_plan']);case=spec['cases'][0]
    registered=read(spec['output']+'/registered_jobs.json')['requirements']
    expected={(j['origin'],j['head'],j['ordinal']):j['initial_checks'] for j in registered}
    bank=read(parent['output']+'/'+case['state']+'/bank.json');count=0;maximum=0.;check_cpu=0.
    for origin in bank['origins']:
        data=original_heads(spec,case,origin);tick=time.process_time()
        checks=all_head_checks(data,bank['prior_weights'],parent['maximum_action_resolution_difference'])
        check_cpu+=time.process_time()-tick
        for h,rows in enumerate(checks):
            for p,checked in enumerate(rows):
                count+=1;old=expected[origin,h,p]
                for key,value in checked.items():
                    if isinstance(value,float):maximum=max(maximum,abs(value-old[key]))
                    else:assert value==old[key],(origin,h,p,key)
    assert count==len(expected) and maximum<=1e-13
    result=dict(complete=True,checked_records=count,maximum_difference=maximum,
        all_flags_and_powers_match=True,batched_check_cpu_seconds=check_cpu,cpu_seconds=time.process_time()-start,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(ROOT/parent['output']/'bulk_check_validation.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def fixed_rule_rows(spec,rules,state,cached,tail_values):
    """Attribute fixed-action quadrature error within each known source identity."""
    probabilities=qmc.Sobol(2,scramble=True,seed=state['predictive_seed']).random_base2(9)[:,1]
    baseline=state['baseline']['1']['reward'];origins=sorted({o for o,h in cached})
    rows=[]
    for f,g,t in [(32,128,None),(512,512,None),(32,64,32),(32,128,16),
            (32,128,32),(512,128,32),(32,512,32),(512,512,32)]:
        reward=0.
        for origin in origins:
            objective=float(np.mean(cached[origin,0]['values'][:f]))
            risk=np.asarray(cached[origin,1]['values'][:g])
            if t is None:risk_value=float(np.mean(risk))
            else:
                body=probabilities[:g]<spec['risk_tail_interval'][0]
                risk_value=spec['risk_tail_interval'][0]*float(np.mean(risk[body]))
                risk_value+=float(np.dot(rules[str(origin)][str(t)]['weights'],tail_values[str(origin)][str(t)]))
            reward+=state['prior_weights'][origin]*objective*risk_value
        rows.append(dict(objective_prefix=f,risk_body_prefix=g,risk_tail_order=t,
            expected_reward=reward,factorized_fixed_action_tower_residual=reward-baseline))
    return rows


def diagnose_fixed(plan=PLAN):
    """Reuse historical512-point marginal diagnostics; only the tail needs new values."""
    from performance.refine_historical_observed_prediction import field_from_snapshot,conditioned_value
    started=time.process_time();spec=read(plan);root=spec['output']
    previous=read('performance/manifests/historical_observed_prediction_v1.json')
    state=read(previous['output']+'/state.json');rules=read(root+'/risk_rules.json')
    assert state['query']==spec['query'] and state['predictive_seed']==spec['predictive_seed']
    bank=read(root+'/'+spec['cases'][0]['state']+'/bank.json')
    np.testing.assert_array_equal(state['prior_weights'],bank['prior_weights'])
    base=spec['accepted_baseline_inputs'][spec['cases'][0]['state']]+'/'+spec['cases'][0]['state']+'/n1/baseline'
    baseline=read(base+'/terminal_rules.json')['rows'][-1]
    assert baseline['best_action']==state['baseline']['1']['action']
    assert abs(baseline['best_reward']-state['baseline']['1']['reward'])<=1e-12
    cached={};values={};errors=[];passed=[];evaluations=0
    for origin in bank['origins']:
        for head in (0,1):
            old=read(previous['output']+'/n1/'+str(origin)+'/'+str(head)+'/results.json')
            assert old['complete'] and old['all_inner_rules_passed'] and old['predictive_cdf_passed']
            assert len(old['values'])>=512;cached[origin,head]=old
            errors.append(old['maximum_evidence_error'])
        field=field_from_snapshot(state['fields'][str(origin)][1]);values[str(origin)]={}
        for order in spec['risk_tail_orders']:
            values[str(origin)][str(order)]=[]
            for label in rules[str(origin)][str(order)]['labels']:
                value,error,admitted,_,_=conditioned_value(field,label,None,1,1,
                    baseline['best_action']['budgets'][0],state['protocol'],previous)
                values[str(origin)][str(order)].append(value)
                errors.append(error);passed.append(admitted);evaluations+=1
    result=dict(complete=True,rows=fixed_rule_rows(spec,rules,state,cached,values),
        tail_values=values,new_fixed_action_evaluations=evaluations,reused_fixed_action_values=len(cached)*512,
        all_inner_rules_passed=all(passed),maximum_evidence_error=max(errors),
        omitted_origin_mass=sum(w for o,w in enumerate(state['prior_weights']) if o not in bank['origins']),
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False,
        scope='Known-identity fixed-action factorization for axis attribution; this does not admit the adaptive mixed-posterior KG bank.')
    save(ROOT/root/'fixed_axis_diagnostic.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='tail_values'}),flush=True)


def fixed_body_rows(state,cached,tail_rules,tail_values,body_rules,body_values,orders):
    rows=[]
    for order in orders:
        for f in (32,512):
            reward=0.
            for origin in sorted({o for o,h in cached}):
                key=str(origin);g=float(np.dot(body_rules[key][str(order)]['weights'],body_values[key][str(order)]))
                g+=float(np.dot(tail_rules[key]['32']['weights'],tail_values[key]['32']))
                reward+=state['prior_weights'][origin]*float(np.mean(cached[origin,0]['values'][:f]))*g
            rows.append(dict(objective_prefix=f,risk_body_order=order,risk_tail_order=32,
                expected_reward=reward,factorized_fixed_action_tower_residual=reward-state['baseline']['1']['reward']))
    return rows


def diagnose_body(plan=PLAN):
    from performance.refine_historical_observed_prediction import field_from_snapshot,conditioned_value
    started=time.process_time();spec=read(plan);root=spec['output']
    previous=read('performance/manifests/historical_observed_prediction_v1.json')
    state=read(previous['output']+'/state.json');tail_rules=read(root+'/risk_rules.json')
    reference=read(root+'/fixed_axis_diagnostic.json');assert reference['all_inner_rules_passed']
    cached={};rules={};values={};errors=[];passed=[];orders=[16,32,64];evaluations=0
    for origin in sorted(map(int,state['fields'])):
        cached[origin,0]=read(previous['output']+'/n1/'+str(origin)+'/0/results.json')
        field=field_from_snapshot(state['fields'][str(origin)][1]);rules[str(origin)]={};values[str(origin)]={}
        position=spec['cases'][0]['head_counts'][1]
        for order in orders:
            rule=positive_cdf_interval(field,0,spec,order=order,position=position,interval=(0.,spec['risk_tail_interval'][0]))
            rules[str(origin)][str(order)]=rule;position+=order;values[str(origin)][str(order)]=[]
            for label in rule['labels']:
                value,error,admitted,_,_=conditioned_value(field,label,None,1,1,
                    state['baseline']['1']['action']['budgets'][0],state['protocol'],previous)
                values[str(origin)][str(order)].append(value)
                errors.append(error);passed.append(admitted);evaluations+=1
    rows=fixed_body_rows(state,cached,tail_rules,reference['tail_values'],rules,values,orders)
    selected=None;tolerance=spec['maximum_numerical_adaptation_difference']
    for low,high in zip(orders,orders[1:]):
        lo=[r for r in rows if r['risk_body_order']==low];hi=[r for r in rows if r['risk_body_order']==high]
        if all(passed) and all(abs(a['expected_reward']-b['expected_reward'])<=tolerance
                and abs(b['factorized_fixed_action_tower_residual'])<=tolerance for a,b in zip(lo,hi)):
            selected=[low,high];break
    result=dict(complete=True,rows=rows,body_rules=rules,body_values=values,selected_body_orders=selected,
        query=state['query'],prior_weights=state['prior_weights'],
        fixed_action_only_admitted=selected is not None,all_inner_rules_passed=all(passed),
        maximum_evidence_error=max(errors),maximum_inverse_cdf_error=max(r['maximum_inverse_cdf_error'] for rs in rules.values() for r in rs.values()),
        new_fixed_action_evaluations=evaluations,reused_fixed_action_values=1024+reference['new_fixed_action_evaluations'],
        omitted_origin_mass=reference['omitted_origin_mass'],
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False,
        scope='Positive body-rule screening on the historical fixed-action law; full adaptive KG requires its complete action bank.')
    save(ROOT/root/'fixed_body_diagnostic.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('body_rules','body_values')}),flush=True)


def diagnose_sampling(plan=PLAN):
    """Attribute the filtered Sobol body error using only saved marginal values."""
    spec=read(plan);root=spec['output'];previous=read('performance/manifests/historical_observed_prediction_v1.json')
    state=read(previous['output']+'/state.json');body=read(root+'/fixed_body_diagnostic.json')
    u=qmc.Sobol(2,scramble=True,seed=state['predictive_seed']).random_base2(9)[:,1]
    rows=[];total=0.
    for origin in sorted(map(int,state['fields'])):
        v=np.asarray(read(previous['output']+'/n1/'+str(origin)+'/1/results.json')['values'])
        f=np.asarray(read(previous['output']+'/n1/'+str(origin)+'/0/results.json')['values'])
        rule=body['body_rules'][str(origin)]['64'];reference=float(np.dot(rule['weights'],body['body_values'][str(origin)]['64']))
        factor=float(f[:32].mean());weight=state['prior_weights'][origin]*factor;rules=[]
        for n in (64,128,512):
            mask=u[:n]<.9;estimate=.9*float(v[:n][mask].mean())
            rules.append(dict(n=n,body_count=int(mask.sum()),body_integral=estimate,
                delta_to_positive64=estimate-reference,mean_cdf_probability=float(u[:n][mask].mean()),
                exact_body_mean_cdf_probability=.45,reward_contribution_difference=weight*(estimate-reference)))
        total+=rules[1]['reward_contribution_difference']
        rows.append(dict(origin=origin,source_mass=state['prior_weights'][origin],objective_factor=factor,
            positive64_body_integral=reference,filtered_sobol=rules))
    old=next(r for r in read(root+'/fixed_axis_diagnostic.json')['rows'] if
        (r['objective_prefix'],r['risk_body_prefix'],r['risk_tail_order'])==(32,128,32))
    new=next(r for r in body['rows'] if (r['objective_prefix'],r['risk_body_order'])==(32,64))
    difference=old['factorized_fixed_action_tower_residual']-new['factorized_fixed_action_tower_residual']
    result=dict(complete=True,rows=rows,weighted_body128_difference=total,
        observed_fixed_tower_difference=difference,attribution_roundoff=abs(total-difference),query=state['query'],
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        scope='Existing frozen fixed-action cached values only; body distribution attribution, not adaptive KG admission.')
    save(ROOT/root/'body_sampling_diagnostic.json',result);print(json.dumps(result),flush=True)


def diagnose_components(plan=PLAN):
    """Check discrete component resolution on four actual body-rule observations."""
    from performance.refine_historical_observed_prediction import field_from_snapshot
    from representation.vectorized_joint_function_risk import update_raw_observation
    spec=read(plan);previous=read('performance/manifests/historical_observed_prediction_v1.json')
    state=read(previous['output']+'/state.json')
    body=read(spec.get('risk_rule_input',spec['output']+'/fixed_body_diagnostic.json'));rows=[]
    nodes={k:{p:qmc.Sobol(k,scramble=True,seed=spec['seed']).random_base2(p)[:,0]
        for p in (10,12,14)} for k in (6,7)}
    for origin in sorted(map(int,state['fields'])):
        labels=body['body_rules'][str(origin)]['32']['labels']
        for ordinal in (0,31):
            future=field_from_snapshot(state['fields'][str(origin)][1])
            update_raw_observation(future,0,labels[ordinal],None,1)
            w=np.exp(future.log_weights);cdf=np.cumsum(w);rules=[]
            for k in (6,7):
                for power in (10,12,14):
                    component=np.minimum(np.searchsorted(cdf,nodes[k][power]),len(w)-1)
                    mass=np.bincount(component,minlength=len(w))/len(component)
                    rules.append(dict(dimensions=k,power=power,represented_components=int(np.count_nonzero(mass)),
                        component_mass_total_variation=float(abs(mass-w).sum()/2)))
            rows.append(dict(origin=origin,body32_ordinal=ordinal,components=len(w),
                effective_component_count=float(1/(w@w)),numerical_component_mass_rules=rules))
    result=dict(complete=True,rows=rows,query=state['query'],new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        scope='Four cached query-marginal posterior updates; discrete component-mass resolution only. '
            'Total variation is not the full terminal action error and this does not establish its dominant contributor.')
    save(ROOT/spec['output']/'component_resolution_diagnostic.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','group','stage','collect','finish','validate-bulk','diagnose-fixed','diagnose-body','diagnose-sampling','diagnose-components'])
    parser.add_argument('--name');parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--power',type=int)
    parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    if args.mode=='prepare':prepare(args.plan)
    elif args.mode=='group':run_group(args.name,args.workers,args.plan)
    elif args.mode=='stage':stage(args.plan)
    elif args.mode=='collect':collect(args.plan)
    elif args.mode=='finish':finish(args.power,args.plan)
    elif args.mode=='validate-bulk':validate_bulk(args.plan)
    elif args.mode=='diagnose-fixed':diagnose_fixed(args.plan)
    elif args.mode=='diagnose-body':diagnose_body(args.plan)
    elif args.mode=='diagnose-sampling':diagnose_sampling(args.plan)
    else:diagnose_components(args.plan)
