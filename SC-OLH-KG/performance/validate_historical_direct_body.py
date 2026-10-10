"""Use the existing zero-dimensional direct branch on all body observations."""
import argparse
import json
import resource
import signal
import time
from scipy.special import roots_legendre

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.validate_historical_component_bank import evaluate_group_rules, group_checks
from performance.validate_historical_component_transfer import prepare_contexts, record_bundle
from performance.validate_historical_label_bank import valid_tables
from performance.validate_historical_label_control import tables

OUTPUT='paper_artifacts/historical_decision_regions_v1'
PARENT='performance/manifests/historical_component_bank_v1.json'
GROUP=136
RULES=['G7','J8']


def prepare():
    start=time.process_time();out=ROOT/OUTPUT;out.mkdir(parents=True,exist_ok=True);parent=read(PARENT);base=ROOT/parent['output']
    records=[f'source_n1_o{o}_h1_p{p}' for o in (4,5) for p in range(176,224)]
    previous=0.
    if (out/'direct_summary.json').exists() and len(read(str(out/'direct_summary.json'))['records'])<len(records):
        save(out/'direct_body32_summary.json',read(str(out/'direct_summary.json')))
        save(out/'direct_body32_plan.json',read(str(out/'direct_plan.json')))
    if (out/'direct_body32_plan.json').exists():previous=read(str(out/'direct_body32_plan.json'))['estimated_cpu_seconds']
    contexts,shared,original=prepare_contexts(dict(parent_plan=PARENT,records=records,groups=[GROUP]))
    # The transfer preparer excludes its source anchor. This complete body
    # comparison includes that already validated anchor as well.
    contexts[original['name']]=dict(**{k:shared[k] for k in
        ('name','label','replicates','head_log_scores','utility_identities','prior_weights')},
        origin=5,head=1,position=207,eta_label=None,
        predictive_cdf=float(.9*(roots_legendre(32)[0][15]+1)/2),
        groups={str(GROUP):dict(opposite=shared['opposite'])})
    save(out/'direct_contexts.json',contexts)
    jobs=[];reused=0
    for origin in (4,5):
        names=[n for n in records if contexts[n]['origin']==origin]
        for identity in original['utility_identities']:
            source=read(str(base/'parts'/f'{GROUP:03d}_{identity}_raw.json'))['rows'];cost=0.
            for name in names:
                path=out/'direct_parts'/name/f'{identity}.json'
                cached=read(str(path))['rows'] if path.exists() else {}
                for directory in ('historical_cdf_bank_v1','historical_label_bank_v1'):
                    root=ROOT/'paper_artifacts'/directory
                    previous_file=root/'parts'/name/f'{GROUP:03d}_{identity}_raw.json'
                    if not previous_file.exists():continue
                    old=read(str(root/'contexts.json'))[name]
                    assert all(old[k]==contexts[name][k] for k in ('label','head_log_scores','prior_weights'))
                    cached.update({r:v for r,v in read(str(previous_file))['rows'].items() if r in RULES})
                if name==original['name']:cached.update({r:v for r,v in source.items() if r in RULES})
                reused+=len(cached)
                path.parent.mkdir(parents=True,exist_ok=True)
                save(path,dict(record=name,identity=identity,rows=cached))
                # This branch forces Sobol power0 at every named rule. J8
                # doubles G7's GH order, rather than its unused Sobol count.
                cost+=sum(source.get(r,{}).get('cpu_seconds',0.) or
                    (2. if r=='J8' else 1.)*source['G7']['cpu_seconds'] for r in RULES if r not in cached)
                cost+=1.
            jobs.append(dict(origin=origin,identity=identity,records=names,estimated_cpu_seconds=1.2*cost))
    incremental=sum(j['estimated_cpu_seconds'] for j in jobs);estimate=previous+incremental
    plan=dict(jobs=jobs,estimated_cpu_seconds=estimate,previous_estimated_cpu_seconds=previous,
        incremental_estimated_cpu_seconds=incremental,maximum_estimated_cpu_seconds=3600,
        maximum_worker_cpu_seconds=600,estimated_cost_admitted=estimate<=3600 and max(j['estimated_cpu_seconds'] for j in jobs)<=600,
        reused_rule_rows=reused,preparation_cpu_seconds=time.process_time()-start,
        group=GROUP,rules=RULES,actions_per_record=2,
        source_components={str(h):len(shared['fields'][h]['means']) for h in original['utility_identities']},
        scope='All96 body observations from16/32-point CDF rules, query/deployment63, budgets128/359. Existing all-component GH64/128 direct rule, zero-dimensional Sobol branch, all8 identity scores and32 opposite heads. Matching GH64 measured timers, GH128 twice that cost, one second per label and20 percent headroom. Four low-priority single-core workers per wave; initial64 and32 added labels share one cumulative3600 CPU second cap. No new external calls.')
    save(out/'direct_plan.json',plan);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in plan.items() if k!='jobs'}),flush=True)
    return plan


def run(identity,origin):
    out=ROOT/OUTPUT;plan=read(str(out/'direct_plan.json'));assert plan['estimated_cost_admitted']
    job=next(j for j in plan['jobs'] if (j['identity'],j['origin'])==(identity,origin))
    task=out/'direct_tasks'/f'{origin}_{identity}.json'
    if task.exists() and read(str(task)).get('records')==job['records']:return read(str(task))
    contexts=read(str(out/'direct_contexts.json'));base=ROOT/read(PARENT)['output']
    shared=compressed_read(base/'marginals'/f'{GROUP:03d}.json.gz')
    assert shared['kind']=='direct' and len(shared['points'])==1
    (out/'direct_tasks').mkdir(parents=True,exist_ok=True)
    start=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,600.)
    new_cpu=0.
    for i,name in enumerate(job['records']):
        path=out/'direct_parts'/name/f'{identity}.json';part=read(str(path))
        bundle=record_bundle(shared,contexts[name]);pending=[r for r in RULES if r not in part['rows']]
        for r,value in evaluate_group_rules(bundle,identity,'raw',pending):
            part['rows'][r]=value;new_cpu+=value['cpu_seconds'];save(path,part)
        if (i+1)%8==0:print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i+1,total=len(job['records']),origin=origin,identity=identity)),flush=True)
    result=dict(complete=True,identity=identity,origin=origin,records=job['records'],new_integral_cpu_seconds=new_cpu,
        cpu_seconds=time.process_time()-start,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    signal.setitimer(signal.ITIMER_PROF,0.);save(task,result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def collect():
    out=ROOT/OUTPUT;plan=read(str(out/'direct_plan.json'));contexts=read(str(out/'direct_contexts.json'))
    shared=compressed_read(ROOT/read(PARENT)['output']/'marginals'/f'{GROUP:03d}.json.gz');results=[]
    for name,context in contexts.items():
        bundle=record_bundle(shared,context);target={h:read(str(out/'direct_parts'/name/f'{h}.json'))['rows'] for h in bundle['utility_identities']}
        checks=group_checks(bundle,[dict(identity=h,rows=target[h]) for h in bundle['utility_identities']],'raw',[('G7','J8')],1e-4)
        checks['valid_unclipped_ranges']=all(valid_tables(bundle,tables(bundle,{h:target[h][r] for h in target})) for r in RULES)
        checks['passed']=bool(checks['passed'] and checks['valid_unclipped_ranges'])
        results.append(dict(record=name,group=GROUP,cdf=context['predictive_cdf'],action_indices=shared['action_indices'],checks=checks))
    tasks=[read(str(out/'direct_tasks'/f'{j["origin"]}_{j["identity"]}.json')) for j in plan['jobs']]
    previous=read(str(out/'direct_body32_summary.json')) if (out/'direct_body32_summary.json').exists() else {}
    result=dict(complete=all(t['complete'] for t in tasks),all_references_admitted=all(r['checks']['passed'] for r in results),
        records=results,estimated_cpu_seconds=plan['estimated_cpu_seconds'],
        new_integral_cpu_seconds=previous.get('new_integral_cpu_seconds',0.)+sum(t['new_integral_cpu_seconds'] for t in tasks),
        successful_worker_cpu_seconds=previous.get('successful_worker_cpu_seconds',0.)+sum(t['cpu_seconds'] for t in tasks),
        maximum_task_peak_rss_mb=max([t['peak_rss_mb'] for t in tasks]+[previous.get('maximum_task_peak_rss_mb',0.)]),
        maximum_action_difference=max(c['maximum_difference'] for r in results for c in r['checks']['comparisons']),
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,full_predictive_integration_admitted=False,
        scope='GH64 vs128 numerical comparison for both query-direct budgets on all96 body labels, all positive components. Timers include the initial64-label wave and added32-label wave once each. No Sobol dimension remains in this branch, so repeated Sobol rules would not be independent evidence. Other363 actions, outer KG admission and independent decision benefits remain open.')
    save(out/'direct_summary.json',result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','collect']);p.add_argument('--identity',type=int);p.add_argument('--origin',type=int);a=p.parse_args()
    if a.mode=='run':run(a.identity,a.origin)
    else:{'prepare':prepare,'collect':collect}[a.mode]()
