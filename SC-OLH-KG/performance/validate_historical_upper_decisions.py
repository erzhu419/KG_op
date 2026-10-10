"""Reuse the full upper bank; reference only the two switching direct groups."""
import argparse
import json
import resource
import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.validate_historical_component_bank import evaluate_group_rules, group_checks
from performance.validate_historical_component_transfer import record_bundle
from performance.validate_historical_label_control import tables
from performance.validate_historical_label_bank import valid_tables

PLAN='performance/manifests/historical_upper_decision_refs_v1.json'


def make_jobs(spec, contexts, source, out, phase, previous_estimate=0., analysis_cpu=0.):
    positions=spec['pilot_positions'] if phase=='pilot' else sorted(set(spec['positions'])-set(spec['pilot_positions']))
    jobs=[]
    for origin in spec['origins']:
        names=[f'source_n1_o{origin}_h1_p{p}' for p in positions]
        for group in spec['groups']:
            for h in contexts[names[0]]['utility_identities']:
                units=source[group,h];cost=0.
                for name in names:
                    path=out/'parts'/name/f'{group}_{h}.json'
                    rows=read(str(path))['rows'] if path.exists() else {}
                    cost+=sum(units[r]['cpu_seconds'] for r in spec['rules'] if r not in rows)+1.
                jobs.append(dict(origin=origin,group=group,identity=h,records=names,estimated_cpu_seconds=1.2*cost))
    incremental=sum(j['estimated_cpu_seconds'] for j in jobs)
    cumulative=previous_estimate+incremental+analysis_cpu+spec['analysis_reserve_cpu_seconds']
    return dict(phase=phase,jobs=jobs,estimated_integral_cpu_seconds=previous_estimate+incremental,
        incremental_estimated_cpu_seconds=incremental,measured_analysis_cpu_seconds=analysis_cpu,
        cumulative_estimated_cpu_seconds=cumulative,maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds'],
        estimated_cost_admitted=cumulative<=spec['maximum_estimated_cpu_seconds'] and
            max(j['estimated_cpu_seconds'] for j in jobs)<=spec['maximum_job_cpu_seconds'])


def prepare(phase,plan=PLAN):
    start=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    previous=0.;analysis=0.
    if phase=='full':
        pilot=read(str(out/'pilot_summary.json'));assert pilot['complete'] and pilot['all_references_admitted']
        previous=read(str(out/'pilot_plan.json'))['estimated_integral_cpu_seconds']
        analysis=read(str(out/'pilot_plan.json'))['measured_analysis_cpu_seconds']+pilot['collection_cpu_seconds']
        contexts=read(str(out/'contexts.json'))
    else:
        base=ROOT/read(spec['parent_plan'])['output'];anchor=read(str(base/'context.json'))
        precision=read(spec['precision_plan']);objective=read(spec['objective_plan'])
        registered={j['name'] for j in read(precision['output']+'/registered_jobs.json')['jobs']}
        regions={r['record']:r for r in read(spec['previous_regions'])['records']};contexts={}
        tail=read(read(precision['parent_plan'])['output']+'/risk_rules.json')
        for origin in spec['origins']:
            f=read(objective['head_inputs']['source_n1'][0].format(origin=origin)+'/head_records.json')
            assert f['actions']==anchor['actions'] and f['prior_weights']==anchor['prior_weights']
            for p in spec['positions']:
                name=f'source_n1_o{origin}_h1_p{p}';assert name in registered
                data=read(precision['output']+'/shards/'+name+'/head_records.json');row=data['records'][0]
                assert data['actions']==anchor['actions'] and data['prior_weights']==anchor['prior_weights']
                assert row['head']==1 and row['position']==p and row['replicates']==1 and row['eta_label'] is None
                rule=tail[str(origin)]['32'];i=rule['positions'].index(p);assert row['label']==rule['labels'][i]
                context=dict(name=name,origin=origin,position=p,predictive_cdf=regions[name]['cdf'],
                    predictive_weight=rule['weights'][i],prior_weights=data['prior_weights'],
                    **{k:row[k] for k in ('label','head_log_scores','utility_identities','replicates','eta_label','head')},groups={})
                for gid in spec['comparison_groups']:
                    group=anchor['groups'][gid];indices=group['action_indices'];assert group['kind']=='direct'
                    context['groups'][str(gid)]=dict(action_indices=indices,
                        opposite=[dict(position=r['position'],head_log_scores=r['head_log_scores'],
                            **{k:np.asarray(r[k][str(max(map(int,r[k])))]['statistics'])[:,indices].tolist()
                                for k in ('raw','projected')}) for r in f['records']],
                        baseline={k:dict(statistics=np.asarray(row[k][str(max(map(int,row[k])))]['statistics'])[:,indices].tolist(),
                            costs=np.asarray(row[k][str(max(map(int,row[k])))]['costs'])[:,indices].tolist()) for k in ('raw','projected')},
                        cached_high_evidence_error=row['raw'][str(max(map(int,row['raw'])))]['predictive_log_evidence_max_error'])
                contexts[name]=context
        save(out/'contexts.json',contexts)
    base=ROOT/read(spec['parent_plan'])['output']
    source={(g,h):read(str(base/'parts'/f'{g:03d}_{h}_raw.json'))['rows'] for g in spec['groups'] for h in (4,5)}
    shared=compressed_read(base/'marginals'/f'{spec["groups"][0]:03d}.json.gz')
    save(out/'preparation.json',dict(complete=True,source_components={str(h):len(shared['fields'][h]['means']) for h in shared['utility_identities']}))
    analysis+=time.process_time()-start
    work=make_jobs(spec,contexts,source,out,phase,previous,analysis)
    save(out/(phase+'_plan.json'),work);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in work.items() if k!='jobs'}),flush=True)
    return work


def run(phase,job_index,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];work=read(str(out/(phase+'_plan.json')));assert work['estimated_cost_admitted']
    job=work['jobs'][job_index];gid,h=job['group'],job['identity'];contexts=read(str(out/'contexts.json'))
    shared=compressed_read(ROOT/read(spec['parent_plan'])['output']/'marginals'/f'{gid:03d}.json.gz')
    start=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,spec['maximum_job_cpu_seconds'])
    cpu=0.;completed=0
    for name in job['records']:
        path=out/'parts'/name/f'{gid}_{h}.json';path.parent.mkdir(parents=True,exist_ok=True)
        part=read(str(path)) if path.exists() else dict(rows={})
        bundle=record_bundle(shared,contexts[name]);pending=[r for r in spec['rules'] if r not in part['rows']]
        for rule,value in evaluate_group_rules(bundle,h,'raw',pending):
            part['rows'][rule]=value;cpu+=value['cpu_seconds'];save(path,part)
        completed+=1;print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=completed,total=len(job['records']))),flush=True)
    result=dict(complete=True,job=job_index,phase=phase,records=job['records'],new_integral_cpu_seconds=cpu,
        cpu_seconds=time.process_time()-start,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    target=out/'tasks'/phase;target.mkdir(parents=True,exist_ok=True);save(target/f'{job_index}.json',result)
    signal.setitimer(signal.ITIMER_PROF,0.);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def collect(phase,plan=PLAN):
    start=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    spec=read(plan);out=ROOT/spec['output'];work=read(str(out/(phase+'_plan.json')));contexts=read(str(out/'contexts.json'))
    previous=read(str(out/'pilot_summary.json')) if phase=='full' else {}
    names=sorted({n for j in work['jobs'] for n in j['records']});base=ROOT/read(spec['parent_plan'])['output']
    shared={g:compressed_read(base/'marginals'/f'{g:03d}.json.gz') for g in spec['comparison_groups']}
    regions=read(spec['previous_regions']);old_regions={r['record']:r for r in regions['records']};checks=[];records=[]
    for name in names:
        context=contexts[name];raw={};native={};costs={};replacement={}
        for gid in spec['comparison_groups']:
            bundle=record_bundle(shared[gid],context);old=context['groups'][str(gid)]['baseline']
            raw[gid]=mixed_values(bundle,old['raw']['statistics'],old['raw']['costs'],'raw')
            native[gid]=mixed_values(bundle,old['projected']['statistics'],old['projected']['costs'],'projected')
            costs[gid]=np.asarray([a['budgets'][0] for a in bundle['actions']])
            replacement[gid]=raw[gid]
            if gid not in spec['groups']:continue
            parts=[dict(identity=h,rows=read(str(out/'parts'/name/f'{gid}_{h}.json'))['rows']) for h in bundle['utility_identities']]
            check=group_checks(bundle,parts,'raw',spec['pairs'],spec['maximum_action_difference'])
            data={r:tables(bundle,{p['identity']:p['rows'][r] for p in parts}) for r in spec['rules']}
            values=mixed_values(bundle,*data[spec['selected_rule']],'raw')
            check['cached_high_difference']=float(abs(values-raw[gid]).max())
            check['cached_high_evidence_error']=context['groups'][str(gid)]['cached_high_evidence_error']
            check['valid_unclipped_ranges']=all(valid_tables(bundle,d) for d in data.values())
            check['passed']=bool(check['passed'] and check['valid_unclipped_ranges'] and
                max(check['cached_high_difference'],check['cached_high_evidence_error'])<=spec['maximum_action_difference'])
            checks.append(dict(record=name,group=gid,checks=check))
            if check['passed']:replacement[gid]=values
        groups=spec['comparison_groups'];indices=sum([context['groups'][str(g)]['action_indices'] for g in groups],[])
        nv=np.concatenate([native[g] for g in groups],axis=1);rv=np.concatenate([raw[g] for g in groups],axis=1)
        updated=np.concatenate([replacement[g] for g in groups],axis=1);cost=np.concatenate([costs[g] for g in groups])
        chosen=np.lexsort((np.broadcast_to(cost,nv.shape),-nv),axis=-1)[:,0];selected=np.asarray(indices)[chosen]
        assert sorted(map(int,np.unique(selected)))==old_regions[name]['native_winners']
        fixed=indices.index(regions['fixed_action']);weights=context['prior_weights'][context['origin']]*context['predictive_weight']/len(chosen)
        take=lambda a:a[np.arange(len(chosen)),chosen]-a[:,fixed]
        old=float(weights*take(rv).sum());new=float(weights*take(updated).sum())
        assert abs(old-old_regions[name]['achieved_adaptation'])<=1e-12
        records.append(dict(record=name,cdf=context['predictive_cdf'],cached_achieved_adaptation=old,
            replacement_achieved_adaptation=new,weighted_change=new-old,native_winners=sorted(map(int,np.unique(selected)))))
    tasks=[read(str(out/'tasks'/phase/f'{i}.json')) for i in range(len(work['jobs']))]
    all_checks=previous.get('checks',[])+checks;all_records=previous.get('records',[])+records
    delta=sum(r['weighted_change'] for r in all_records)
    result=dict(complete=all(t['complete'] for t in tasks),phase=phase,checks=all_checks,records=all_records,
        all_references_admitted=all(r['checks']['passed'] for r in all_checks),
        direct_action_record_checks_passed=2*sum(r['checks']['passed'] for r in all_checks),
        cached_upper_resolution_reused=True,upper_resolution_scope='Historical365-action upper-tail raw/native resolution passed; additional checks here are selected direct groups only.',
        new_integral_cpu_seconds=previous.get('new_integral_cpu_seconds',0.)+sum(t['new_integral_cpu_seconds'] for t in tasks),
        successful_worker_cpu_seconds=previous.get('successful_worker_cpu_seconds',0.)+sum(t['cpu_seconds'] for t in tasks),
        maximum_task_peak_rss_mb=max([t['peak_rss_mb'] for t in tasks]+[previous.get('maximum_task_peak_rss_mb',0.)]),
        collection_cpu_seconds=time.process_time()-start,weighted_model_preview_change=delta,
        model_net_preview_with_validated_replacements=regions['raw_model_net_preview']+delta,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,full_predictive_integration_admitted=False,
        scope='Original all8 identity scores,full512/1024 positive components,covariance,365 actions and359 budget. No renormalization or clipping. This posthoc diagnostic prices existing native direct winners with validated raw replacements; it does not restrict the algorithm action class. Other body competitors and outer KG precision remain open. No independent decision claim.')
    save(out/(phase+'_summary.json'),result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('checks','records')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','collect']);p.add_argument('--phase',choices=['pilot','full'],required=True)
    p.add_argument('--job',type=int);p.add_argument('--plan',default=PLAN);a=p.parse_args()
    if a.mode=='run':run(a.phase,a.job,a.plan)
    else:{'prepare':prepare,'collect':collect}[a.mode](a.phase,a.plan)
