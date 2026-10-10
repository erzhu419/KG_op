"""Locate decision changes in existing development caches without integration."""
import json
import argparse
import resource
import signal
import time

import numpy as np
from scipy.special import roots_legendre

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.extend_historical_upper_risk import head_rules
from performance.refine_historical_achievable_precision import assemble_case, compose_records, rule_pair

PLAN='performance/manifests/historical_predictive_positive_body_v7.json'
OUTPUT='paper_artifacts/historical_decision_regions_v1'


def region_metrics(raw,costs,native,native_costs,weights,fixed):
    raw=np.asarray(raw);native=np.asarray(native);weights=np.asarray(weights)
    optimal=np.lexsort((costs,-raw),axis=-1)[...,0]
    chosen=np.lexsort((native_costs,-native),axis=-1)[...,0]
    take=lambda values,indices:np.take_along_axis(values,indices[...,None],axis=-1)[...,0]
    gain=take(raw,optimal)-raw[...,fixed]
    achieved=take(raw,chosen)-raw[...,fixed]
    return dict(probability_mass=float(weights.sum()),raw_switch_mass=float(weights[optimal!=fixed].sum()),
        native_switch_mass=float(weights[chosen!=fixed].sum()),raw_adaptation=float(np.sum(weights*gain)),
        achieved_adaptation=float(np.sum(weights*achieved)),choice_loss=float(np.sum(weights*(gain-achieved))),
        raw_winners=sorted(map(int,np.unique(optimal))),native_winners=sorted(map(int,np.unique(chosen))),
        maximum_gain=float(gain.max()),minimum_achieved_gain=float(achieved.min()))


def observed_query_matrix(rows):
    baseline={r['name']:r for r in rows if r['policy']=='stop'};result=[]
    for arm in sorted({r['arm'] for r in rows}):
        for policy in sorted({r['policy'] for r in rows}-{'stop'}):
            selected=[r for r in rows if r['arm']==arm and r['policy']==policy]
            changed=[r for r in selected if r['action']['primary']!=baseline[r['name']]['action']['primary']]
            result.append(dict(arm=arm,policy=policy,records=len(selected),
                primary_changed=len(changed),changed_tasks=sorted({r['task_id'] for r in changed}),
                query_indices=sorted({r['candidate'] for r in selected})))
    return result


def overlay_direct(row,parts,indices):
    for level,rule in ((1,'G7'),(0,'J8')):
        value=row['raw'][str(rule_pair(row,'raw')[level])]
        for h,part in parts.items():
            replacement=part['rows'][rule]
            for j,i in enumerate(indices):
                value['statistics'][h][i]=replacement['statistics'][j]
                value['costs'][h][i]=replacement['costs'][j]


def run(mode='cached'):
    start=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,240.)
    out=ROOT/OUTPUT;out.mkdir(parents=True,exist_ok=True)
    spec=read(PLAN);precision=read(spec['precision_plan']);case=precision['cases'][0]
    heads,bank,_=assemble_case(precision,case,read(precision['output']+'/registered_jobs.json')['jobs'])
    prior=np.asarray(bank['prior_weights']);layout=read('paper_artifacts/historical_cdf_bank_v1/bank_layout.json')
    baseline_path=spec['accepted_baseline_inputs'][case['state']]+'/'+case['state']+'/n1/baseline'
    debited=read(baseline_path+'/terminal_rules.json')['rows'][-1]
    stop=read(spec['stop_inputs'][case['state']])['rows'][-1]
    assert layout['actions']==debited['actions'] and len(layout['actions'])==365
    fixed=debited['best_action_index'];rules=read(spec['output']+'/risk_rules.json')
    if mode=='direct':
        direct=read(str(out/'direct_summary.json'));assert direct['complete'] and direct['all_references_admitted']
        direct_contexts=read(str(out/'direct_contexts.json'))
    tail_rules=read(spec['retained_tail_rule_input']);selection=head_rules(spec,rules,bank['origins'],32,32,32)
    group_of={a:g['group'] for g in layout['groups'] for a in g['action_indices']}
    edges=[0.,.9,.93,.96,.99,1.];regions={str((a,b)):[] for a,b in zip(edges[:-1],edges[1:])}
    records=[];totals=[]
    for origin in bank['origins']:
        selected=[]
        for h in (0,1):
            index={r['position']:r for r in heads[h,origin]}
            selected.append([index[p] for p in selection[h,origin]['positions']])
        f,g=selected
        if mode=='direct':
            for row in g[:32]:
                name=f'source_n1_o{origin}_h1_p{row["position"]}'
                context=direct_contexts[name]
                assert row['label']==context['label'] and row['head_log_scores']==context['head_log_scores']
                assert context['prior_weights']==bank['prior_weights']
                part={h:read(str(out/'direct_parts'/name/f'{h}.json')) for h in bank['utility_identities']}
                overlay_direct(row,part,layout['groups'][136]['action_indices'])
        cdf=np.r_[.9*(roots_legendre(32)[0]+1)/2,.9+.1*(roots_legendre(32)[0]+1)/2]
        assert [r['label'] for r in g]==rules[str(origin)]['32']['labels']+tail_rules[str(origin)]['32']['labels']
        values,costs,_,_=compose_records(f,g,prior,'raw',1)
        native,nc,_,_=compose_records(f,g,prior,'projected',1)
        low=compose_records(f,g,prior,'raw',0)[0];nl=compose_records(f,g,prior,'projected',0)[0]
        weights=prior[origin]*np.outer(selection[0,origin]['weights'],selection[1,origin]['weights'])
        totals.append(region_metrics(values,costs,native,nc,weights,fixed))
        for j,row in enumerate(g):
            raw,nat=values[:,j],native[:,j];w=weights[:,j]
            metrics=region_metrics(raw,costs[:,j],nat,nc[:,j],w,fixed)
            winners=sorted(set(metrics['raw_winners']+metrics['native_winners']+[fixed]))
            active=np.asarray(winners)
            delta=abs(raw-low[:,j]);ndelta=abs(nat-nl[:,j])
            low_choice=np.lexsort((nc[:,j],-nl[:,j]),axis=-1)[:,0]
            high_choice=np.lexsort((nc[:,j],-nat),axis=-1)[:,0]
            achieved_change=abs(raw[np.arange(len(f)),high_choice]-raw[np.arange(len(f)),low_choice])
            record=dict(record=f'source_n1_o{origin}_h1_p{row["position"]}',origin=origin,
                position=row['position'],cdf=float(cdf[j]),label=row['label'],**metrics,
                winner_groups=sorted({group_of[a] for a in winners}),
                maximum_all_action_raw_difference=float(delta.max()),
                maximum_all_action_native_difference=float(ndelta.max()),
                maximum_winner_raw_difference=float(delta[:,active].max()),
                maximum_winner_native_difference=float(ndelta[:,active].max()),
                native_choice_changed_heads=int(np.count_nonzero(low_choice!=high_choice)),
                low_native_winners=sorted(map(int,np.unique(low_choice))),
                low_native_winner_groups=sorted({group_of[int(a)] for a in np.unique(low_choice)}),
                weighted_achieved_choice_change=float(np.sum(w*achieved_change)))
            records.append(record)
            for a,b in zip(edges[:-1],edges[1:]):
                if a<=cdf[j]<b:regions[str((a,b))].append(record);break
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(origin=origin,records=len(records),cpu_seconds=time.process_time()-start)),flush=True)
    additive=['probability_mass','raw_switch_mass','native_switch_mass','raw_adaptation','achieved_adaptation','choice_loss']
    regions={k:{name:sum(r[name] for r in rows) for name in additive} for k,rows in regions.items()}
    total={name:sum(r[name] for r in totals) for name in additive}
    rare=float(sum(w for i,w in enumerate(prior) if i not in bank['origins']))
    net=total['achieved_adaptation']+debited['best_reward']-stop['best_reward']
    historical=read('paper_artifacts/historical_full_bank_replicates_v1/results.json')
    result=dict(complete=True,mode=mode,query=spec['query'],action_count=365,fixed_action=fixed,
        prior_weights=prior.tolist(),computed_predictive_origins=bank['origins'],omitted_predictive_origin_mass=rare,
        regions=regions,total=total,raw_model_net_preview=net,verification_opportunity_cost=stop['best_reward']-debited['best_reward'],
        records=records,observed_query_matrix=observed_query_matrix(historical['rows']),
        actions={str(i):layout['actions'][i] for i in sorted({a for r in records for a in r['raw_winners']+r['native_winners']})},
        cpu_seconds=time.process_time()-start,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        new_integrations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False,
        scope='Opened development caches only. Original positive body/tail weights and all identity likelihoods, no weight renormalization. Same-origin objective/risk pairs follow the existing joint mixture grid. Full365-action values use historical rules whose body precision remains unresolved; selected-action differences are diagnostics, not an alternative admission gate. Historical observed-query matrix has146 direct actions and is not a full365-action KG comparison. No truth scores or confirmation block read.',
        direct_body_overlay=mode=='direct',
        direct_overlay_scope='In direct mode only group136 body risk statistics/costs are replaced: GH64 for selected values,GH128 for reference comparison. Other actions, objective heads, native values and all scores remain as cached.')
    save(out/('summary_with_direct.json' if mode=='direct' else 'summary.json'),result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('records','actions','observed_query_matrix')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cached','direct'],nargs='?',default='cached');run(p.parse_args().mode)
