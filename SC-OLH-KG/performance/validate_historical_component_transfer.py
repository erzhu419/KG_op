"""Check existing risk observations using shared, frozen joint marginals."""
import argparse
import json
import resource
import signal
import time

import numpy as np
from scipy.special import roots_legendre

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.validate_historical_component_bank import evaluate_group_rules, group_checks

PLAN = 'performance/manifests/historical_component_transfer_v1.json'
STAGES = ('screen', 'adaptive', 'resolution')


def record_bundle(shared, context):
    assert context['replicates'] == 1 and context['eta_label'] is None and context['head'] == 1
    assert context['utility_identities'] == shared['utility_identities']
    assert context['prior_weights'] == shared['prior_weights']
    return dict(shared, name=context['name'], label=context['label'],
        head_log_scores=context['head_log_scores'],
        opposite=context['groups'][str(shared['group'])]['opposite'])


def part_path(out, row, identity):
    return out/'parts'/row['record']/f'{row["group"]:03d}_{identity}_{row["kind"]}.json'


def priced_plan(selected, contexts, probe, output, previous=0.):
    for row in selected:
        units = probe['groups'][str(row['group'])]['times'][row['kind']]
        cost = 0.
        for h in contexts[row['record']]['utility_identities']:
            path = part_path(output, row, h)
            cached = read(str(path))['rows'] if path.exists() else {}
            for name in row['rules']:
                if name in cached:
                    continue
                fallback = probe['resolution_units'][row['kind']+'_'+str(h)]
                cost += (.75*fallback['E9'] if name == 'L9' else
                    {'I8':1.,'J8':2.}[name]*units[str(h)]['H8'] if name in ('I8','J8') else
                    units[str(h)][name] if name in units[str(h)] else fallback[name])
            cost += 1.
        row['estimated_cpu_seconds'] = 1.2*cost
    estimate = sum(r['estimated_cpu_seconds'] for r in selected)
    return dict(selected=selected, estimated_cpu_seconds=estimate,
        cumulative_estimated_cpu_seconds=previous+estimate)


def prepare_contexts(spec):
    """Reuse the frozen joint bank with selected observations and action groups."""
    parent = read(spec['parent_plan']); base = ROOT/parent['output']
    full = read(parent['parent_plan']); precision = read(full['precision_plan'])
    assert read(str(base/'bank_tail_resolution_summary.json'))['single_record_risk_bank_precision_passed']
    original = read(str(base/'context.json')); contexts = {}; opposite = {}
    rules = read(full['output']+'/risk_rules.json')
    for name in spec['records']+[original['name']]:
        origin = int(name.split('_o')[1].split('_')[0])
        data = read(precision['output']+'/shards/'+name+'/head_records.json'); row = data['records'][0]
        assert data['actions'] == original['actions'] and data['prior_weights'] == original['prior_weights']
        if origin not in opposite:
            opposite[origin] = read(precision['head_inputs']['source_n1'][0].format(origin=origin)+'/head_records.json')
            assert opposite[origin]['actions'] == data['actions']
            assert opposite[origin]['prior_weights'] == data['prior_weights']
        context = dict(name=name, origin=origin, prior_weights=data['prior_weights'],
            **{k:row[k] for k in ('head','position','label','head_log_scores','utility_identities','replicates','eta_label')}, groups={})
        for gid in spec['groups']:
            group = original['groups'][gid]; indices = group['action_indices']
            assert len(indices) == {'guarded':3, 'direct':2}[group['kind']]
            context['groups'][str(gid)] = dict(action_indices=indices,
                opposite=[dict(position=r['position'], head_log_scores=r['head_log_scores'],
                    **{k:np.asarray(r[k][str(max(map(int,r[k])))]['statistics'])[:,indices].tolist()
                        for k in ('raw','projected')}) for r in opposite[origin]['records']],
                baseline={k:dict(statistics=np.asarray(row[k]['12']['statistics'])[:,indices].tolist(),
                    costs=np.asarray(row[k]['12']['costs'])[:,indices].tolist()) for k in ('raw','projected')})
            if group['kind'] == 'direct':
                context['groups'][str(gid)]['native_reference_difference'] = float(np.max(abs(
                    np.asarray(row['projected']['10']['statistics'])[:,indices] -
                    np.asarray(row['projected']['12']['statistics'])[:,indices])))
            shared = compressed_read(base/'marginals'/f'{gid:03d}.json.gz')
            bundle = record_bundle(shared, context)
            if name == original['name']:
                for key in ('label','head_log_scores','opposite','prior_weights','utility_identities'):
                    assert bundle[key] == shared[key]
        body = rules[str(origin)]['32']; i = body['positions'].index(row['position'])
        assert row['label'] == body['labels'][i]
        x, _ = roots_legendre(32)
        context['predictive_cdf'] = float(.9*(x[i]+1)/2)
        contexts[name] = context
    contexts.pop(original['name'])
    return contexts, shared, original


def prepare(plan=PLAN):
    spec = read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    started=time.process_time()
    contexts,shared,original=prepare_contexts(spec)
    save(out/'contexts.json', contexts)
    probe = read(str(out/'cost_probe.json'))
    selected = [dict(record=name, group=gid, kind=kind, rules=['E5','E7','F7','G7'],
        pairs=[['E5','E7'],['E7','F7'],['E7','G7']], selected_rule='E7')
        for name in spec['records'] for gid in spec['groups'] for kind in ('raw','projected')]
    initial = priced_plan(selected, contexts, probe, out)
    initial.update(maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds'],
        estimated_cost_admitted=initial['cumulative_estimated_cpu_seconds']<=spec['maximum_estimated_cpu_seconds'])
    save(out/'screen_plan.json', initial)
    result = dict(complete=True, source_components={str(h):len(shared['fields'][h]['means'])
        for h in original['utility_identities']}, records=[dict(name=n,label=c['label'],cdf=c['predictive_cdf'])
        for n,c in contexts.items()], groups=spec['groups'], formal_actions=len(original['actions']),
        selected_actions_per_record=6, reference_metadata_replay_passed=True,
        full_posterior_restorations=0, copied_joint_marginal_bytes=0,
        estimated_initial_cpu_seconds=initial['estimated_cpu_seconds'], cpu_seconds=time.process_time()-started,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False)
    save(out/'preparation.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def run(identity,kind,shard,stage,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];parent=read(spec['parent_plan'])
    (out/'tasks'/stage).mkdir(parents=True,exist_ok=True)
    work=read(str(out/(stage+'_plan.json')));assert work['estimated_cost_admitted']
    row=[r for r in work['selected'] if r['kind']==kind][shard]
    context=read(str(out/'contexts.json'))[row['record']]
    bundle=record_bundle(compressed_read(ROOT/parent['output']/'marginals'/f'{row["group"]:03d}.json.gz'),context)
    path=part_path(out,row,identity);path.parent.mkdir(parents=True,exist_ok=True)
    part=read(str(path)) if path.exists() else dict(record=row['record'],group=row['group'],kind=kind,identity=identity,rows={})
    pending=[n for n in row['rules'] if n not in part['rows']]
    result=dict(complete=False,record=row['record'],group=row['group'],identity=identity,kind=kind,shard=shard)
    tick=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_job_cpu_seconds'])
    try:
        for i,(name,value) in enumerate(evaluate_group_rules(bundle,identity,kind,pending)):
            part['rows'][name]=value;save(path,part)
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i+1,total=len(pending),unit='component_rules')),flush=True)
        result['complete']=True
    except TimeoutError:result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        result.update(cpu_seconds=time.process_time()-tick,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
        save(out/'tasks'/stage/f'{identity}_{kind}_{shard}.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def collect(stage,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];parent=read(spec['parent_plan'])
    contexts=read(str(out/'contexts.json'));work=read(str(out/(stage+'_plan.json')))
    previous={} if stage=='screen' else { (r['record'],r['group'],r['kind']):r
        for r in read(str(out/(STAGES[STAGES.index(stage)-1]+'_summary.json')))['groups']}
    tasks=[];integral_cpu=0.
    for kind in ('raw','projected'):
        for shard,row in enumerate(r for r in work['selected'] if r['kind']==kind):
            context=contexts[row['record']]
            bundle=record_bundle(compressed_read(ROOT/parent['output']/'marginals'/f'{row["group"]:03d}.json.gz'),context)
            parts=[read(str(part_path(out,row,h))) for h in context['utility_identities']]
            integral_cpu+=sum(part['rows'][name]['cpu_seconds'] for part in parts for name in row['rules'])
            checks=group_checks(bundle,parts,kind,row['pairs'],spec['maximum_action_difference'])
            baseline=context['groups'][str(row['group'])]['baseline'][kind]
            s=np.zeros_like(baseline['statistics']);c=np.zeros_like(baseline['costs'])
            for part in parts:
                r=part['rows'][row['selected_rule']];s[part['identity']]=r['statistics'];c[part['identity']]=r['costs']
            delta=float(abs(mixed_values(bundle,s,c,kind)-mixed_values(bundle,baseline['statistics'],baseline['costs'],kind)).max())
            previous[row['record'],row['group'],kind]=dict(record=row['record'],group=row['group'],kind=kind,
                checks=checks,selected_rule=row['selected_rule'],maximum_old_new_action_difference=delta)
            tasks.extend(read(str(out/'tasks'/stage/f'{h}_{kind}_{shard}.json')) for h in context['utility_identities'])
    groups=list(previous.values());failed=[r for r in groups if not r['checks']['passed']]
    complete=all(t['complete'] for t in tasks)
    next_plan=None
    if failed and stage!='resolution':
        selected=[]
        for row in failed:
            coarse_only=stage=='adaptive' and row['checks']['maximum_log_evidence_error']<=spec['maximum_action_difference'] and all(
                c['maximum_difference']<=spec['maximum_action_difference'] for c in row['checks']['comparisons'] if c['low']=='E8')
            if stage=='screen':
                rules=['E8','F8','H8'];pairs=[['E7','E8'],['E8','F8'],['E8','H8']];chosen='E8'
            elif any(c['low']=='E8' and c['high']=='H8' and c['maximum_difference']>spec['maximum_action_difference']
                    for c in row['checks']['comparisons']):
                rules=['I8','J8','K9'];pairs=[['H8','I8'],['H8','J8'],['H8','K9']];chosen='H8'
            elif coarse_only:
                rules=['E9'];pairs=[['E8','E9'],['E8','F8'],['E8','H8']];chosen='E8'
            else:
                rules=['E9','F9','L9'];pairs=[['E8','E9'],['E9','F9'],['L9','E9']];chosen='E9'
            selected.append(dict(record=row['record'],group=row['group'],kind=row['kind'],rules=rules,pairs=pairs,selected_rule=chosen))
        next_plan=priced_plan(selected,contexts,read(str(out/'cost_probe.json')),out,work['cumulative_estimated_cpu_seconds'])
        next_plan.update(maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds'],
            estimated_cost_admitted=complete and next_plan['cumulative_estimated_cpu_seconds']<=spec['maximum_estimated_cpu_seconds'])
        save(out/(STAGES[STAGES.index(stage)+1]+'_plan.json'),next_plan)
    result=dict(stage=stage,complete=complete,groups=groups,failed_groups=[{k:r[k] for k in ('record','group','kind')} for r in failed],
        selected_action_record_checks_passed={k:3*sum(r['checks']['passed'] for r in groups if r['kind']==k) for k in ('raw','projected')},
        selected_action_record_checks_total={k:3*len(spec['records'])*len(spec['groups']) for k in ('raw','projected')},
        all_selected_action_precision_passed=complete and not failed,cpu_seconds=integral_cpu,
        successful_task_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),
        cpu_measurement_scope='Saved integral timers include completed cached rules from failed starts; exclude setup and interrupted rules.',
        maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),next_plan=next_plan,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False,
        scope='Two guarded groups, all three budgets, three development observations, all identity weights/scores and fixed opposite heads. Rule differences are diagnostics; no full-bank/full-KG or independent-decision admission.')
    save(out/(stage+'_summary.json'),result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','collect'])
    parser.add_argument('--stage',choices=(*STAGES,'direct'),default='screen');parser.add_argument('--identity',type=int)
    parser.add_argument('--kind',choices=['raw','projected']);parser.add_argument('--shard',type=int)
    parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    if args.mode=='prepare':prepare(args.plan)
    elif args.mode=='run':run(args.identity,args.kind,args.shard,args.stage,args.plan)
    else:
        if args.stage=='direct':parser.error('direct stage collection uses performance.validate_historical_label_groups')
        collect(args.stage,args.plan)
