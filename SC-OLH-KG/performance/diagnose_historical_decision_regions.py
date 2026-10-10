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
from performance.assemble_historical_risk_references import (
    OUTPUT as ASSEMBLY_OUTPUT, load_entries, apply_entry, reference_masks, remaining_risk_checks, entry_summary)

PLAN='performance/manifests/historical_predictive_positive_body_v7.json'
OUTPUT='paper_artifacts/historical_decision_regions_v1'


def objective_upper(statistics,identity_weights,utility_identities):
    """Verifier branches form a subprobability vector; each quality is <=1.

    The bound is conditional on the cached objective-head law. Keep the
    posterior mass of identities without utility tables as worst-case1.
    """
    qualities=np.asarray(statistics).max(axis=-1)
    selected=np.asarray(utility_identities)
    upper=np.einsum('fgk,fka->fga',np.asarray(identity_weights)[:,:,selected],qualities[:,selected],optimize=True)
    missing=np.ones(np.asarray(identity_weights).shape[-1],dtype=bool);missing[selected]=False
    return upper+np.asarray(identity_weights)[:,:,missing].sum(axis=-1)[...,None]


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


def risk_competitors(high,costs,risk_low,dominated,masks,groups,positions,tolerance=1e-4):
    """Rank uncovered competitors by their actual gap; rule differences are diagnostic."""
    high=np.asarray(high);difference=abs(high-np.asarray(risk_low))
    winners=np.lexsort((costs,-high),axis=-1)[:,0]
    gaps=high[np.arange(len(high)),winners,None]-high
    covered=np.logical_or.reduce(list(masks.values()))
    eligible=~np.asarray(dominated)&~covered[None,:]
    rows=[]
    for group in groups:
        ix=np.asarray(group['action_indices']);active=eligible[:,ix]
        if not active.any():continue
        local=np.where(active,gaps[:,ix],np.inf)
        head,action=np.unravel_index(np.argmin(local),local.shape)
        rows.append(dict(group=group['group'],action_indices=ix.tolist(),
            minimum_gap=float(local[head,action]),action_at_minimum_gap=int(ix[action]),
            objective_position=positions[head],winner_at_minimum_gap=int(winners[head]),
            risk_rule_difference_at_minimum_gap=float(difference[head,ix[action]]),
            maximum_risk_rule_difference=float(difference[:,ix][active].max()),
            eligible_action_head_pairs=int(active.sum()),
            empirical_overlap_action_head_pairs=int((active&(gaps[:,ix]<=difference[:,ix]+tolerance)).sum())))
    return sorted(rows,key=lambda r:(r['minimum_gap'],r['group']))


def overlay_direct(row,parts,indices,*,kind='raw',pair=('J8','G7')):
    for level,rule in enumerate(pair):
        value=row[kind][str(rule_pair(row,kind)[level])]
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
    if mode in ('direct','dominance','integrated'):
        direct=read(str(out/'direct_summary.json'));assert direct['complete'] and direct['all_references_admitted']
        direct_contexts=read(str(out/'direct_contexts.json'))
    if mode in ('dominance','integrated'):
        refs=read(str(out/'native_reference_summary.json'));assert refs['complete'] and refs['all_references_admitted']
        native_names={r['record'] for r in refs['records']};native_contexts=read(str(out/'contexts.json'))
    entries=load_entries(layout) if mode=='integrated' else {}
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
        applied={}
        if mode in ('direct','dominance','integrated'):
            for row in g[:32]:
                name=f'source_n1_o{origin}_h1_p{row["position"]}'
                context=direct_contexts[name]
                assert row['label']==context['label'] and row['head_log_scores']==context['head_log_scores']
                assert context['prior_weights']==bank['prior_weights']
                part={h:read(str(out/'direct_parts'/name/f'{h}.json')) for h in bank['utility_identities']}
                overlay_direct(row,part,layout['groups'][136]['action_indices'])
                if mode in ('dominance','integrated') and name in native_names:
                    context=native_contexts[name]
                    assert row['label']==context['label'] and row['head_log_scores']==context['head_log_scores'] and context['prior_weights']==bank['prior_weights']
                    part={h:read(str(out/'parts'/name/f'063_{h}_projected.json')) for h in bank['utility_identities']}
                    overlay_direct(row,part,layout['groups'][63]['action_indices'],kind='projected',pair=('F8','E8'))
        if mode=='integrated':
            for row in g:
                name=f'source_n1_o{origin}_h1_p{row["position"]}'
                for kind in ('raw','projected'):
                    applied[name,kind]=[entry for (n,k,_),entry in entries.items() if (n,k)==(name,kind)]
                    for entry in applied[name,kind]:apply_entry(row,f,prior,entry,origin)
        cdf=np.r_[.9*(roots_legendre(32)[0]+1)/2,.9+.1*(roots_legendre(32)[0]+1)/2]
        assert [r['label'] for r in g]==rules[str(origin)]['32']['labels']+tail_rules[str(origin)]['32']['labels']
        values,costs,posterior,_=compose_records(f,g,prior,'raw',1)
        native,nc,_,_=compose_records(f,g,prior,'projected',1)
        low=compose_records(f,g,prior,'raw',0)[0];nl=compose_records(f,g,prior,'projected',0)[0]
        if mode in ('dominance','integrated'):
            uppers={kind:objective_upper([r[kind][str(rule_pair(r,kind)[1])]['statistics'] for r in f],posterior,bank['utility_identities'])
                for kind in ('raw','projected')}
        if mode=='integrated':
            risk_low={kind:compose_records(f,g,prior,kind,(1,0))[0] for kind in ('raw','projected')}
            objective_low={kind:compose_records(f,g,prior,kind,(0,1))[0] for kind in ('raw','projected')}
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
            if mode in ('dominance','integrated') and j<32:
                for kind,high,difference in [('raw',raw,delta),('projected',nat,ndelta)]:
                    dominated=uppers[kind][:,j]+2e-4<high[:,fixed,None]
                    unresolved=(difference>1e-4)&~dominated
                    record[kind+'_objective_bound_screen']=dict(dominated_action_head_pairs=int(dominated.sum()),
                        imprecise_action_head_pairs=int((difference>1e-4).sum()),
                        unresolved_action_head_pairs=int(unresolved.sum()),
                        unresolved_action_indices=np.flatnonzero(unresolved.any(axis=0)).tolist(),
                        unresolved_groups=sorted({group_of[int(a)] for a in np.flatnonzero(unresolved.any(axis=0))}))
                    if mode=='integrated':
                        masks=reference_masks(layout,applied[record['record'],kind])
                        if kind=='raw':masks['direct_target'][layout['groups'][136]['action_indices']]=True
                        else:
                            if record['record'] in native_names:masks['direct_target'][layout['groups'][63]['action_indices']]=True
                            direct_ix=[a for group in layout['groups'] if group['kind']=='direct' for a in group['action_indices']]
                            levels=[row[kind][str(p)] for p in rule_pair(row,kind)]
                            assert max(np.max(abs(np.asarray(levels[0][key])[:,direct_ix]-np.asarray(levels[1][key])[:,direct_ix])) for key in ('statistics','costs'))<=1e-12
                            masks['native_analytic'][direct_ix]=True
                            masks['direct_target'][direct_ix]=False
                        masks['paired_residual'][masks['direct_target']|masks['native_analytic']]=False
                        remaining=remaining_risk_checks(abs(high-risk_low[kind][:,j]),dominated,masks)
                        remaining['unreferenced_relevant_groups']=sorted({group_of[a] for a in remaining['unreferenced_relevant_action_indices']})
                        remaining['maximum_objective_axis_difference']=float(abs(high-objective_low[kind][:,j]).max())
                        remaining['objective_axis_imprecise_action_head_pairs']=int((abs(high-objective_low[kind][:,j])>1e-4).sum())
                        record[kind+'_reference_coverage']=remaining
                        competitors=risk_competitors(high,costs[:,j] if kind=='raw' else nc[:,j],
                            risk_low[kind][:,j],dominated,masks,layout['groups'],[r['position'] for r in f])
                        record[kind+'_competitors']=dict(groups=len(competitors),
                            empirical_overlap_groups=sum(r['empirical_overlap_action_head_pairs']>0 for r in competitors),
                            empirical_overlap_action_head_pairs=sum(r['empirical_overlap_action_head_pairs'] for r in competitors),
                            closest_groups=competitors[:5])
                        record[kind+'_referenced_competitors']=risk_competitors(high,
                            costs[:,j] if kind=='raw' else nc[:,j],risk_low[kind][:,j],
                            np.zeros_like(dominated),dict(covered=np.zeros(len(layout['actions']),dtype=bool)),
                            [entry['group'] for entry in applied[record['record'],kind]],
                            [r['position'] for r in f])
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
        direct_body_overlay=mode in ('direct','dominance','integrated'),
        direct_overlay_scope='Group136 body raw risk statistics/costs use GH64 selected values and GH128 references. Objective heads and all scores remain cached. Dominance mode also replaces native group63 at its two validated conflict records with E8/F8; other values remain cached.')
    if mode=='dominance':result['objective_bound_scope']='Diagnostic conditional on cached objective heads and posterior identity weights. Verifier branch probabilities sum to at most1. Uncomputed identity utility contributes worst-case1; use a2e-4 separation margin. Two native group63 reference records are also overlaid E8/F8; other historical group references still require bank integration. No action is removed, no gate or algorithm is changed, and no rigorous quadrature error bound or full KG admission is asserted.'
    if mode=='integrated':
        result['reference_assembly']=entry_summary(entries)
        result['reference_assembly']['body_totals']={kind:{key:sum(r[kind+'_reference_coverage'][key] for r in records if r['cdf']<.9)
            for key in ('unreferenced_actions','unreferenced_imprecise_action_head_pairs','unreferenced_relevant_action_head_pairs','objective_axis_imprecise_action_head_pairs')} for kind in ('raw','projected')}
        result['scope']='Opened development state and frozen query63; original365 actions,source prior,all8 identity scores and positive body/tail weights. Actual selected reference statistics/costs replace high-rule entries only. Historical low values remain separate and their differences are not treated as new reference failures or evidence. Direct target,paired residual and native analytic coverage are reported separately; all are conditional on the matched32 cached objective heads. Unreferenced comparisons and the objective axis remain open. No full KG or independent deployment claim.'
        result['direct_overlay_scope']='All historical source-anchor,nearby full bank,selected body groups,CDF pilot and switching-tail references are integrated at their matching labels/actions; prior direct136 and native63 references are retained.'
        target=ROOT/ASSEMBLY_OUTPUT;target.mkdir(parents=True,exist_ok=True)
        save(target/'reference_sources.json',[dict(record=n,kind=k,group=gid,scope=e['scope'],sources=e['sources'],method=e['method'],action_indices=e['group']['action_indices']) for (n,k,gid),e in sorted(entries.items())])
        save(target/'summary.json',result)
        save(target/'competition_summary.json',dict(complete=True,action_count=365,fixed_action=fixed,
            records=[{k:v for k,v in r.items() if k in ('record','origin','position','cdf','label','probability_mass',
                'raw_winners','native_winners','raw_competitors','projected_competitors')} for r in records if r['cdf']<.9],
            new_integrations=0,independent_confirmation=False,
            limitations='Uncovered competitors ranked on the current development grid. The gap versus low/high rule difference is empirical, not an error bound or an admission gate. Conditional objective bounds and all365 actions are retained.'))
    else:save(out/('dominance_summary.json' if mode=='dominance' else 'summary_with_direct.json' if mode=='direct' else 'summary.json'),result)
    signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('records','actions','observed_query_matrix')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cached','direct','dominance','integrated'],nargs='?',default='cached')
    mode=p.parse_args().mode
    try:run(mode)
    finally:
        print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
