"""Independently admit cached predictive heads without repeating the old bank."""

import argparse
import copy
import json
import signal
import time

import numpy as np

from core.raw_terminal_head_grid import evaluate_head_records, compose_head_grid, summarize_achievable_grid
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_batch_kg import restore

PLAN = 'performance/manifests/historical_achievable_precision_v4.json'


def original_heads(spec, case, origin):
    if 'head_inputs' in spec:
        data = [read(path.format(origin=origin)+'/head_records.json')
            for path in spec['head_inputs'][case['name']]]
        assert data[0]['actions'] == data[1]['actions']
        for d in data: d['records'].sort(key=lambda r:r['position'])
        return data
    parent = read(spec['parent_plan']); objective = read(spec['objective_plan'])
    paths = [objective['output']+'/'+str(origin) if h == 0 and case['n'] == 1
        and case['state'] == objective['state'] else parent['output']+'/'+case['state']+'/n'+str(case['n'])+
        '/head/'+str(h)+'_'+str(origin) for h in (0, 1)]
    data = [read(p+'/head_records.json') for p in paths]
    assert data[0]['actions'] == data[1]['actions']
    for d in data: d['records'].sort(key=lambda r:r['position'])
    return data


def rule_pair(row, kind):
    return sorted(map(int, row[kind]))[-2:]


def compose_records(f, g, prior, kind, level):
    rows = [f, g]
    levels = [level,level] if isinstance(level,int) else level
    statistics = [np.asarray([r[kind][str(rule_pair(r,kind)[lev])]['statistics'] for r in head])
        for head,lev in zip(rows,levels)]
    costs = np.asarray([r[kind][str(rule_pair(r,kind)[levels[1]])]['costs'] for r in g])
    return compose_head_grid(*statistics, costs, *[[r['head_log_scores'] for r in h] for h in rows], prior)


def head_checks(row, opposite, prior, head, tolerance):
    result = {}
    for kind in ('raw', 'projected'):
        low, high = rule_pair(row, kind)
        heads = ([row], opposite) if head == 0 else (opposite, [row])
        counter = [np.asarray([r[kind][str(max(map(int,r[kind])))]['statistics'] for r in opposite])]
        def values(power):
            s = np.asarray(row[kind][str(power)]['statistics'])[None]
            if head == 0:
                f, g = s, counter[0]
                costs = np.asarray([r[kind][str(max(map(int,r[kind])))]['costs'] for r in opposite])
            else:
                f, g = counter[0], s; costs = np.asarray(row[kind][str(power)]['costs'])[None]
            return compose_head_grid(f,g,costs,*[[r['head_log_scores'] for r in h] for h in heads],prior)[0]
        difference = float(np.max(abs(values(high)-values(low))))
        result[kind+'_powers'] = [low,high]
        result[kind+'_maximum_action_difference'] = difference
        result[kind+'_passed'] = difference <= tolerance
    high = rule_pair(row,'raw')[-1]
    result['predictive_log_evidence_error'] = row['raw'][str(high)]['predictive_log_evidence_max_error']
    result['evidence_passed'] = result['predictive_log_evidence_error'] <= 1e-4
    return result


def all_head_checks(data,prior,tolerance):
    """The same per-record gates, composed once per head axis and rule kind."""
    f,g=[d['records'] for d in data];checks=[[{} for r in h] for h in (f,g)]
    for kind in ('raw','projected'):
        high=compose_records(f,g,prior,kind,1)[0]
        differences=[np.max(abs(high-compose_records(f,g,prior,kind,levels)[0]),axis=axes)
            for levels,axes in [((0,1),(1,2)),((1,0),(0,2))]]
        for head,rows in enumerate((f,g)):
            for p,row in enumerate(rows):
                delta=float(differences[head][p])
                checks[head][p].update({kind+'_powers':rule_pair(row,kind),
                    kind+'_maximum_action_difference':delta,kind+'_passed':delta<=tolerance})
    for head,rows in enumerate((f,g)):
        for p,row in enumerate(rows):
            error=row['raw'][str(rule_pair(row,'raw')[-1])]['predictive_log_evidence_max_error']
            checks[head][p].update(predictive_log_evidence_error=error,evidence_passed=error<=1e-4)
    return checks


def prepare(plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); out = ROOT/spec['output']; out.mkdir(parents=True,exist_ok=True)
    jobs = []; requirements = []
    for case in spec['cases']:
        bank = read(parent['output']+'/'+case['state']+'/bank.json')
        for origin in bank['origins']:
            data = original_heads(spec,case,origin)
            checked=all_head_checks(data,bank['prior_weights'],parent['maximum_action_resolution_difference'])
            for head in (0,1):
                for ordinal,row in enumerate(data[head]['records']):
                    checks = checked[head][ordinal]
                    job = dict(case=case['name'],origin=origin,head=head,ordinal=ordinal,
                        position=row['position'],initial_checks=checks)
                    job['name'] = case['name']+'_o'+str(origin)+'_h'+str(head)+'_p'+str(ordinal)
                    requirements.append(job)
                    if not (checks['raw_passed'] and checks['projected_passed'] and checks['evidence_passed']):
                        jobs.append(job)
    save(out/'registered_jobs.json',dict(jobs=jobs,requirements=requirements,new_physical_calls=0))
    print(json.dumps(dict(jobs=len(jobs),risk_jobs=sum(j['head']==1 for j in jobs),
        objective_jobs=sum(j['head']==0 for j in jobs))),flush=True)
    return jobs


def run(name, plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan'])
    job = next(j for j in read(spec['output']+'/registered_jobs.json')['jobs'] if j['name']==name)
    case = next(c for c in spec['cases'] if c['name']==job['case'])
    out = ROOT/spec['output']/'shards'/name; out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists() and read(str(out/'results.json')).get('complete'):
        return read(str(out/'results.json'))
    started = time.process_time(); handler = signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_case_cpu_seconds'])
    result = dict(name=name,complete=False,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False)
    try:
        model,terminal,_,_ = restore(parent,case['state'])
        data = original_heads(spec,case,job['origin']); head = job['head']
        row = copy.deepcopy(data[head]['records'][job['ordinal']]); old = copy.deepcopy(row)
        path = out/'head_records.json'
        if path.exists(): row = read(str(path))['records'][0]
        labels = {row['position']:row['label']}
        eta = None if row['eta_label'] is None else {row['position']:row['eta_label']}
        opposite = data[1-head]['records']
        checks = head_checks(row,opposite,model.weights,head,parent['maximum_action_resolution_difference'])
        for index,power in enumerate(spec['refinement_powers']):
            need_raw = not (checks['raw_passed'] and checks['evidence_passed'])
            need_native = not checks['projected_passed']
            if not (need_raw or need_native): break
            if (need_raw and power <= max(map(int,row['raw']))) or (need_native and power <= max(map(int,row['projected']))):
                continue
            def checkpoint(rows,actions):
                assert actions == data[head]['actions']
                save(path,dict(records=rows,actions=actions,prior_weights=model.weights.tolist()))
            row = evaluate_head_records(model,terminal,360,parent['query'],head,labels,[row['position']],
                powers=[power] if need_raw else [],projected_powers=[power] if need_native else [],
                seed=parent['seed'],completed=[row],checkpoint=checkpoint,utility_identities=row['utility_identities'],
                guarded_mean_position=0,replicates=case['n'],eta_labels=eta)[0]
            checks = head_checks(row,opposite,model.weights,head,parent['maximum_action_resolution_difference'])
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(step=index+1,total=len(spec['refinement_powers']),
                unit='precision_levels',metrics=dict(name=name,checks=checks,cpu_seconds=time.process_time()-started))),flush=True)
        assert row['label']==old['label'] and row['eta_label']==old['eta_label']
        assert row['head_log_scores']==old['head_log_scores']
        assert row['raw']['8']==old['raw']['8'] and row['raw']['10']==old['raw']['10']
        assert row['projected']['8']==old['projected']['8'] and row['projected']['10']==old['projected']['10']
        result.update(complete=True,checks=checks,numerical_diagnostics_passed=
            checks['raw_passed'] and checks['projected_passed'] and checks['evidence_passed'],old_caches_preserved=True)
    except TimeoutError:
        result['cpu_budget_stopped'] = True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.); signal.signal(signal.SIGPROF,handler)
        result['cpu_seconds'] = time.process_time()-started; save(out/'results.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def assemble_case(spec,case,registry):
    parent = read(spec['parent_plan']); bank = read(parent['output']+'/'+case['state']+'/bank.json')
    heads = {}; executions = []
    for origin in bank['origins']:
        data = original_heads(spec,case,origin)
        for head in (0,1):
            rows = data[head]['records']
            for job in registry:
                if (job['case'],job['origin'],job['head'])!=(case['name'],origin,head): continue
                path = spec['output']+'/shards/'+job['name']
                result = read(path+'/results.json'); executions.append(result)
                if not result['complete']: raise ValueError('Predictive precision is incomplete: '+job['name'])
                replacement = read(path+'/head_records.json')
                assert replacement['actions']==data[head]['actions']
                rows[job['ordinal']] = replacement['records'][0]
            heads[head,origin] = rows
    return heads,bank,executions


def grid(heads,prior,origins,prefixes,baseline,raw_level,native_level,*,head_rules=None):
    totals = []; raw_delta = 0.; native_delta = 0.; omitted_max = 0.
    for origin in origins:
        if head_rules is None:
            f,g = [heads[h,origin][:prefixes[h]] for h in (0,1)]
            fw,gw = [np.full(n,1./n) for n in prefixes]
        else:
            selected=[]; quadrature=[]
            for h in (0,1):
                rule=head_rules[h,origin]; index={r['position']:r for r in heads[h,origin]}
                selected.append([index[p] for p in rule['positions']]);quadrature.append(np.asarray(rule['weights']))
            f,g=selected;fw,gw=quadrature
        values,costs,posterior,_ = compose_records(f,g,prior,'raw',raw_level)
        native,nc,native_posterior,_ = compose_records(f,g,prior,'projected',native_level)
        np.testing.assert_array_equal(posterior,native_posterior)
        mass = float(prior[origin])
        weights = np.full(prefixes,mass/np.prod(prefixes)) if head_rules is None else mass*np.outer(fw,gw)
        stop,debited,fixed = baseline
        row = summarize_achievable_grid(values,costs,native,nc,posterior,weights,
            (mass*stop,mass*debited,fixed),'per_record',sum(prefixes))
        utilities = f[0]['utility_identities']; assert all(r['utility_identities']==utilities for r in f+g)
        omitted = posterior[:,:,sorted(set(range(len(prior)))-set(utilities))].sum(axis=2)
        row['weighted_omitted_mass'] = float(np.sum(weights*omitted)); omitted_max=max(omitted_max,float(omitted.max()))
        totals.append(row)
        raw_delta=max(raw_delta,float(np.max(abs(values-compose_records(f,g,prior,'raw',0)[0]))))
        native_delta=max(native_delta,float(np.max(abs(native-compose_records(f,g,prior,'projected',0)[0]))))
    keys=['raw_adaptation_value','raw_net_reward_control_variate','raw_adaptation_of_projected_choice',
        'raw_net_reward_of_projected_choice_control_variate','action_selection_loss','projected_value_bias_at_selected_action',
        'raw_fixed_action_tower_residual','weighted_omitted_mass','quadrature_joint_probability_mass']
    result={k:sum(r[k] for r in totals) for k in keys}
    expected=np.sum([r['expected_identity_weights'] for r in totals],axis=0)
    rare=float(sum(w for i,w in enumerate(prior) if i not in origins))
    for k in ('raw_net_reward_control_variate','raw_net_reward_of_projected_choice_control_variate'):
        result[k]+=rare*(baseline[1]-baseline[0])
    bound=result['weighted_omitted_mass']+rare; net=result['raw_net_reward_of_projected_choice_control_variate']
    result.update(prefixes=prefixes,raw_level=raw_level,native_level=native_level,
        identity_error_bound=bound,maximum_future_omitted_utility_mass=omitted_max,
        achieved_net_identity_interval=[net-bound,net+bound],maximum_raw_action_resolution_difference=raw_delta,
        maximum_native_action_resolution_difference=native_delta,
        expected_identity_weights=expected.tolist(),covered_model_weight_martingale_residual=(expected-prior).tolist())
    return result


def collect(plan=PLAN):
    spec=read(plan);parent=read(spec['parent_plan']);registry=read(spec['output']+'/registered_jobs.json')['jobs']
    cases=[];executions=[]
    for case in spec['cases']:
        heads,bank,jobs=assemble_case(spec,case,registry);executions.extend(jobs)
        source=parent['accepted_baseline_inputs'].get(case['state'],parent['output'])
        baseline_path=source+'/'+case['state']+'/n'+str(case['n'])+'/baseline'
        admission=read(baseline_path+'/results.json');debited=read(baseline_path+'/terminal_rules.json')['rows'][-1]
        stops=read(parent['stop_inputs'][case['state']])['rows'];stop=stops[-1]
        baseline=stop['best_reward'],debited['best_reward'],debited['best_action_index']
        small,large=case['objective_prefixes'];low,high=parent['predictive_prefixes']
        rows=[grid(heads,np.asarray(bank['prior_weights']),bank['origins'],prefix,baseline,raw,native)
            for prefix,raw,native in [([small,high],1,1),([large,low],1,1),([large,high],0,1),
                ([large,high],1,0),([large,high],1,1)]]
        fine=rows[-1];net=fine['raw_net_reward_of_projected_choice_control_variate']
        changes=[abs(net-r['raw_net_reward_of_projected_choice_control_variate']) for r in rows[:-1]]
        norm=max(r['raw'][str(rule_pair(r,'raw')[-1])]['predictive_log_evidence_max_error']
            for head in heads.values() for r in head)
        gates=dict(baseline=bool(admission['numerical_diagnostics_passed'] and
            np.max(abs(np.asarray(stops[-1]['values'])-stops[-2]['values']))<=parent['maximum_action_resolution_difference']
            and stops[-1]['best_action_index']==stops[-2]['best_action_index']),
            predictive=max(changes)<=parent['maximum_numerical_adaptation_difference'],
            raw_action=fine['maximum_raw_action_resolution_difference']<=parent['maximum_action_resolution_difference'],
            native_action=fine['maximum_native_action_resolution_difference']<=parent['maximum_action_resolution_difference'],
            evidence=norm<=1e-4,identity=fine['identity_error_bound']<=parent['maximum_identity_integration_error_bound']
                and fine['maximum_future_omitted_utility_mass']<=parent['maximum_identity_integration_error_bound'])
        cases.append(dict(case=case['name'],state=case['state'],replicates=case['n'],query=parent['query'],
            action_count=len(debited['actions']),rows=rows,numerical_changes=changes,predictive_log_evidence_error=norm,
            gates=gates,numerical_diagnostics_passed=all(gates.values()),recommendation_frozen=False,
            terminal_diagnostics_passed=all(v for k,v in gates.items() if k!='predictive'),
            fixed_action_predictive_tower_diagnostic_passed=abs(fine['raw_fixed_action_tower_residual'])<=
                parent['maximum_numerical_adaptation_difference'],
            full_predictive_integration_admitted=all(gates.values()) and abs(fine['raw_fixed_action_tower_residual'])<=
                parent['maximum_numerical_adaptation_difference'],
            numerical_diagnostics_scope='Original local contrast and terminal gates on the finite predictive bank.',
            head_rule_pairs={str(h):[dict(origin=o,pairs=[{k:rule_pair(r,k) for k in ('raw','projected')}
                for r in heads[h,o]]) for o in bank['origins']] for h in (0,1)}))
    result=dict(cases=cases,executions=executions,cpu_seconds=sum(r['cpu_seconds'] for r in executions),
        all_registered_jobs_complete=all(r['complete'] for r in executions),new_physical_calls=0,
        new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,algorithmic_query_selection_complete=False)
    save(ROOT/spec['output']/'outcome_summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','case','collect'])
    parser.add_argument('--name');parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    if args.mode=='prepare':prepare(args.plan)
    elif args.mode=='collect':collect(args.plan)
    else:run(args.name,args.plan)
