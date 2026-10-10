"""Extend per-group historical anchors to one complete development label bank."""
import argparse
import json
import resource
import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.validate_historical_component_bank import evaluate_group_rules
from performance.validate_historical_component_transfer import prepare_contexts, record_bundle, part_path
from performance.validate_historical_label_control import controlled_row, tables

PLAN = 'performance/manifests/historical_label_bank_v1.json'
STAGES = {'coarse':'E5', 'fine':'E7', 'independent':'F7', 'gh_pilot':'G5', 'gh_full':'G5'}


def anchor_part_path(out,row,identity):
    return out/'anchor_parts'/f'{row["group"]:03d}_{identity}_{row["kind"]}.json'


def anchor_tables(bundle, anchor, target, rule, selected):
    data=tables(bundle, {h:controlled_row(anchor[h][selected], target[h][rule], anchor[h][rule])
        for h in bundle['utility_identities']})
    # Direct deployment always spends its fixed budget, independently of eta
    # or label. Preserve this analytic constant through numerator subtraction.
    for i,action in enumerate(bundle['actions']):
        if action['kind']=='direct':data[1][bundle['utility_identities'],i]=action['budgets'][0]
    return data


def valid_tables(bundle, data):
    s, c = data; active = bundle['utility_identities']
    return bool(np.isfinite(s).all() and np.isfinite(c).all() and
        np.min(s[active]) >= -1e-12 and np.max(s[active]) <= 1+1e-12 and
        np.min(c[active]) >= 0 and np.max(c[active]) <= 359)


def validated_selections(spec, contexts, out):
    previous=read(spec['validated_groups']);assert previous['complete']
    known=[r for r in previous['groups'] if r['record'] in contexts]
    assert all(r['hybrid_direct_admitted'] for r in known)
    direct=read(spec['validated_direct_groups']);assert direct['complete']
    for row in direct['groups']:
        if row['record'] not in contexts:continue
        assert row['checks']['passed']
        context=contexts[row['record']];local=context['groups'][str(row['group'])]
        bundle=dict(context,actions=[{} for _ in local['action_indices']])
        data=tables(bundle,{h:read(str(part_path(out,row,h)))['rows'][row['selected_rule']]
            for h in context['utility_identities']})
        assert valid_tables(bundle,data)
        known.append(dict(record=row['record'],group=row['group'],kind=row['kind'],
            selected_method='direct_'+row['selected_rule'],hybrid_direct_admitted=True,
            selected_statistics=data[0].tolist(),selected_costs=data[1].tolist()))
    return known


def make_plan(selected, contexts, out, base, stage, spec):
    rule = STAGES[stage]; rows = []
    for old in selected:
        row = dict(old, rules=[rule]); cost = 0.
        for h in contexts[row['record']]['utility_identities']:
            cached = read(str(part_path(out,row,h)))['rows']
            if rule not in cached:
                source = read(str(base/'parts'/f'{row["group"]:03d}_{h}_{row["kind"]}.json'))['rows']
                # Two historical E7 rows were reused with zero timers; F7 has
                # identical geometry and a measured timer for these groups.
                cost += source[rule]['cpu_seconds'] or source['F7']['cpu_seconds']
        row['estimated_integral_cpu_seconds'] = cost; rows.append(row)
    workers = len(next(iter(contexts.values()))['utility_identities'])*sum(
        min(spec['shards_per_identity_kind'],sum(r['kind']==k for r in rows)) for k in ('raw','projected'))
    estimate = 1.2*(sum(r['estimated_integral_cpu_seconds'] for r in rows)+workers)
    return dict(stage=stage,selected=rows,estimated_cpu_seconds=estimate,
        maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds_per_stage'],
        estimated_cost_admitted=estimate<=spec['maximum_estimated_cpu_seconds_per_stage'],workers=workers,
        pricing_scope='Missing target integrals plus one CPU second per worker and20 percent headroom; cached anchor/target work excluded. Preparation and collection measured separately.')


def prepare_independent(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    previous=read(str(out/'fine_summary.json'));assert previous['complete']
    groups=read(str(out/'fine_groups.json'));pending={(g['record'],g['group'],g['kind'])
        for g in groups if not g['target_precision_admitted']}
    selected=[r for r in read(str(out/'fine_plan.json'))['selected']
        if (r['record'],r['group'],r['kind']) in pending]
    work=make_plan(selected,read(str(out/'contexts.json')),out,base,'independent',spec)
    save(out/'independent_plan.json',work)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in work.items() if k!='selected'}),flush=True)
    return work


def independent_reference(bundle,anchor,target,selected):
    controlled=anchor_tables(bundle,anchor,target,'F7',selected)
    valid=valid_tables(bundle,controlled)
    # Subtraction can produce a negative near-zero probability even when each
    # positive quadrature is valid. Use the already computed direct independent
    # target rule as reference; never clip the control or alter the selected bank.
    reference=controlled if valid else tables(bundle,{h:target[h]['F7'] for h in bundle['utility_identities']})
    return reference,dict(method='control_F7' if valid else 'direct_F7',
        controlled_valid_unclipped_ranges=valid,controlled_minimum_statistic=float(controlled[0].min()),
        reference_valid_unclipped_ranges=valid_tables(bundle,reference))


def independent_check(bundle,anchor,target,selected,kind,tolerance):
    variants={r:anchor_tables(bundle,anchor,target,r,selected) for r in ('E5','E7')}
    variants['F7'],reference=independent_reference(bundle,anchor,target,selected)
    values={r:mixed_values(bundle,*v,kind) for r,v in variants.items()}
    differences={r:float(abs(values[r]-values['F7']).max()) for r in ('E5','E7')}
    normalization=max(abs(rows[h][r]['diagnostics']['predictive_log_evidence_error'])
        for rows in (anchor,target) for h in bundle['utility_identities'] for r in variants)
    return dict(passed=bool(all(valid_tables(bundle,v) for v in variants.values()) and
        normalization<=tolerance and max(differences.values())<=tolerance),
        differences=differences,maximum_log_evidence_error=normalization,reference=reference)


def prepare(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    base=ROOT/read(spec['parent_plan'])['output'];started=time.process_time()
    original=read(str(base/'context.json'));expanded=dict(spec,groups=list(range(len(original['groups']))))
    contexts,shared,original=prepare_contexts(expanded);save(out/'contexts.json',contexts)
    prior=read(str(base/'bank_tail_resolution_summary.json'));selected=[];reused=0
    for name,context in contexts.items():
        for group in original['groups']:
            gid=group['group']
            for kind in ('raw','projected'):
                if kind=='projected' and group['kind']=='direct':
                    assert context['groups'][str(gid)]['native_reference_difference']<=1e-12
                    continue
                chosen=prior['groups'][gid]['selected_rules'][kind]
                assert prior['groups'][gid]['checks'][kind]['passed'] and chosen in ('G7','E8','E9')
                row=dict(record=name,group=gid,kind=kind,anchor_rule=chosen);selected.append(row)
                for h in context['utility_identities']:
                    path=part_path(out,row,h);cached={}
                    path.parent.mkdir(parents=True,exist_ok=True)
                    for directory in spec['cache_outputs']:
                        source=ROOT/directory;oldpath=part_path(source,row,h)
                        if not oldpath.exists():continue
                        previous=read(str(source/'contexts.json'))[name]
                        for key in ('label','head_log_scores','prior_weights','utility_identities'):
                            assert previous[key]==context[key]
                        assert previous['groups'][str(gid)]==context['groups'][str(gid)]
                        cached.update(read(str(oldpath))['rows'])
                    if path.exists():cached.update(read(str(path))['rows'])
                    reused+=sum(n in cached for n in STAGES.values())
                    save(path,dict(record=name,group=gid,kind=kind,identity=h,rows=cached))
    known=validated_selections(spec,contexts,out)
    save(out/'validated_groups.json',known)
    save(out/'bank_layout.json',dict(actions=original['actions'],groups=original['groups']))
    plans={s:make_plan(selected,contexts,out,base,s,spec) for s in ('coarse','fine')}
    for stage,work in plans.items():save(out/(stage+'_plan.json'),work)
    result=dict(complete=True,records=list(contexts),actions=len(original['actions']),
        source_components={str(h):len(shared['fields'][h]['means']) for h in original['utility_identities']},
        anchor_rules={k:{n:sum(r['kind']==k and r['anchor_rule']==n for r in selected)
            for n in ('G7','E8','E9')} for k in ('raw','projected')},
        reused_target_rule_rows=reused,reused_validated_groups=len(known),
        stage_estimated_cpu_seconds={s:w['estimated_cpu_seconds'] for s,w in plans.items()},
        stage_estimated_cost_admitted={s:w['estimated_cost_admitted'] for s,w in plans.items()},
        cpu_seconds=time.process_time()-started,full_posterior_restorations=0,copied_joint_marginal_bytes=0,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False)
    save(out/'preparation.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def run(identity,kind,shard,stage,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    work=read(str(out/(stage+'_plan.json')));assert work['estimated_cost_admitted']
    (out/'tasks'/stage).mkdir(parents=True,exist_ok=True)
    if stage=='fine':assert read(str(out/'coarse_summary.json'))['complete']
    if stage=='independent':assert read(str(out/'fine_summary.json'))['complete']
    if stage=='gh_pilot':assert read(str(out/'independent_summary.json'))['complete']
    if stage=='gh_full':assert read(str(out/'gh_pilot_summary.json'))['pilot_passed']
    rows=[r for r in work['selected'] if r['kind']==kind][shard::spec['shards_per_identity_kind']]
    contexts=read(str(out/'contexts.json'));started=time.process_time();new_integral=0.
    handler=signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,spec['maximum_job_cpu_seconds'])
    result=dict(stage=stage,identity=identity,kind=kind,shard=shard,complete=False,completed_groups=0)
    try:
        for row in rows:
            path=part_path(out,row,identity);part=read(str(path));rule=STAGES[stage]
            original=None
            if stage in ('gh_pilot','gh_full'):
                original=compressed_read(base/'marginals'/f'{row["group"]:03d}.json.gz')
                anchor_path=anchor_part_path(out,row,identity);anchor_path.parent.mkdir(parents=True,exist_ok=True)
                extra=read(str(anchor_path)) if anchor_path.exists() else dict(rows={})
                if rule not in extra['rows']:
                    for name,value in evaluate_group_rules(original,identity,kind,[rule]):
                        extra['rows'][name]=value;new_integral+=value['cpu_seconds'];save(anchor_path,extra)
            if rule not in part['rows']:
                bundle=record_bundle(original or compressed_read(base/'marginals'/f'{row["group"]:03d}.json.gz'),contexts[row['record']])
                for name,value in evaluate_group_rules(bundle,identity,kind,[rule]):
                    part['rows'][name]=value;new_integral+=value['cpu_seconds'];save(path,part)
            result['completed_groups']+=1
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=result['completed_groups'],total=len(rows),unit='action_groups')),flush=True)
        result['complete']=True
    except TimeoutError:result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        result.update(cpu_seconds=time.process_time()-started,new_integral_cpu_seconds=new_integral,
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        save(out/'tasks'/stage/f'{identity}_{kind}_{shard}.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def collect(stage,plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output'];started=time.process_time()
    work=read(str(out/(stage+'_plan.json')));contexts=read(str(out/'contexts.json'));layout=read(str(out/'bank_layout.json'))
    validated=validated_selections(spec,contexts,out);save(out/'validated_groups.json',validated)
    known={(r['record'],r['group'],r['kind']):r for r in validated}
    assembly=read(str(out/'fine_plan.json')) if stage=='independent' else work
    selected={(r['record'],r['group'],r['kind']):r for r in assembly['selected']};groups=[];banks={};comparisons={}
    for name,context in contexts.items():
        identities=context['utility_identities'];n=len(context['prior_weights']);a=len(layout['actions'])
        bank={k:np.zeros((n,a,4)) for k in ('raw','projected')};costs={k:np.zeros((n,a)) for k in bank}
        coarse={k:np.zeros((n,a,4)) for k in bank};coarse_costs={k:np.zeros((n,a)) for k in bank}
        baseline={k:np.zeros((n,a,4)) for k in bank};baseline_costs={k:np.zeros((n,a)) for k in bank}
        opposite={k:np.zeros((len(next(iter(context['groups'].values()))['opposite']),n,a,4)) for k in bank}
        for group in layout['groups']:
            gid=group['group'];ix=group['action_indices'];local=context['groups'][str(gid)]
            bundle=dict(context,actions=[layout['actions'][i] for i in ix],opposite=local['opposite'])
            for kind in bank:
                baseline[kind][:,ix]=local['baseline'][kind]['statistics'];baseline_costs[kind][:,ix]=local['baseline'][kind]['costs']
                opposite[kind][:,:,ix]=np.asarray([r[kind] for r in local['opposite']])
                native=kind=='projected' and group['kind']=='direct'
                key=(name,gid,kind);reference=known.get(key);delta=None;norm=0.;independent=None
                if native:
                    chosen=low=(np.asarray(local['baseline'][kind]['statistics']),np.asarray(local['baseline'][kind]['costs']))
                    method='native_reference';passed=True;admitted=True
                else:
                    row=selected[key]
                    anchor={h:read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows'] for h in identities}
                    target={h:read(str(part_path(out,row,h)))['rows'] for h in identities}
                    low=anchor_tables(bundle,anchor,target,'E5',row['anchor_rule']);chosen=low;method='control32';passed=False
                    rules=['E5'] if stage=='coarse' else ['E5','E7']
                    norm=max(abs(target[h][r]['diagnostics']['predictive_log_evidence_error']) for h in identities for r in rules)
                    if stage in ('fine','independent'):
                        high=anchor_tables(bundle,anchor,target,'E7',row['anchor_rule'])
                        delta=float(abs(mixed_values(bundle,*low,kind)-mixed_values(bundle,*high,kind)).max())
                        passed=valid_tables(bundle,low) and valid_tables(bundle,high) and norm<=spec['maximum_action_difference'] and delta<=spec['maximum_action_difference']
                        if not passed:chosen=high;method='control128_unvalidated'
                    admitted=reference is not None
                    if reference:
                        chosen=(np.asarray(reference['selected_statistics']),np.asarray(reference['selected_costs']))
                        method=reference['selected_method']+'_validated_cache'
                    elif stage=='independent':
                        independent=independent_check(bundle,anchor,target,row['anchor_rule'],kind,spec['maximum_action_difference'])
                        norm=max(norm,independent['maximum_log_evidence_error'])
                        if not independent['passed']:method='HOLD_independent_disagreement'
                groups.append(dict(record=name,group=gid,kind=kind,action_indices=ix,selected_method=method,
                    coarse_fine_maximum_difference=delta,coarse_fine_comparison_passed=bool(passed),
                    maximum_log_evidence_error=norm,valid_unclipped_ranges=valid_tables(bundle,chosen),
                    independent_check=independent,
                    independent_comparison_passed=bool(admitted or independent and independent['passed']) if stage=='independent' else None,
                    target_precision_admitted=admitted))
                bank[kind][:,ix],costs[kind][:,ix]=chosen;coarse[kind][:,ix],coarse_costs[kind][:,ix]=low
        full=dict(context,actions=layout['actions'],opposite=[dict(head_log_scores=r['head_log_scores'],
            **{k:opposite[k][i].tolist() for k in bank}) for i,r in enumerate(local['opposite'])])
        values={k:mixed_values(full,bank[k],costs[k],k) for k in bank}
        old={k:mixed_values(full,baseline[k],baseline_costs[k],k) for k in bank}
        low_values={k:mixed_values(full,coarse[k],coarse_costs[k],k) for k in bank}
        comparisons[name]={k:dict(maximum_baseline_action_difference=float(abs(values[k]-old[k]).max()),
            changed_baseline_recommendations=int(np.count_nonzero(values[k].argmax(1)!=old[k].argmax(1))),
            changed_coarse_recommendations=int(np.count_nonzero(values[k].argmax(1)!=low_values[k].argmax(1)))) for k in bank}
        banks[name]=dict(statistics={k:v.tolist() for k,v in bank.items()},costs={k:v.tolist() for k,v in costs.items()})
    tasks=[read(str(out/'tasks'/stage/f'{h}_{k}_{s}.json')) for h in identities for k in ('raw','projected')
        for s in range(spec['shards_per_identity_kind']) if s<sum(r['kind']==k for r in work['selected'])]
    previous=read(str(out/('fine_summary.json' if stage=='independent' else 'coarse_summary.json'))) if stage!='coarse' else None
    complete=all(t['complete'] for t in tasks)
    result=dict(stage=stage,complete=complete,records=list(contexts),actions=len(layout['actions']),
        reused_validated_groups=len(known),
        coarse_fine_actions_passed={k:sum(len(r['action_indices']) for r in groups if r['kind']==k and r['coarse_fine_comparison_passed']) for k in ('raw','projected')},
        target_precision_actions_admitted={k:sum(len(r['action_indices']) for r in groups if r['kind']==k and r['target_precision_admitted']) for k in ('raw','projected')},
        independent_actions_passed={k:sum(len(r['action_indices']) for r in groups if r['kind']==k and r['independent_comparison_passed']) for k in ('raw','projected')} if stage=='independent' else None,
        independent_failed_groups={k:[r['group'] for r in groups if r['kind']==k and not r['independent_comparison_passed']] for k in ('raw','projected')} if stage=='independent' else None,
        maximum_independent_difference={k:max((max(r['independent_check']['differences'].values()) if r['independent_check'] else 0.) for r in groups if r['kind']==k) for k in ('raw','projected')} if stage=='independent' else None,
        independent_reference_substitutions=[dict(record=r['record'],group=r['group'],kind=r['kind'],
            **r['independent_check']) for r in groups if r['independent_check'] and
            r['independent_check']['reference']['method']=='direct_F7'],
        coarse_fine_failed_groups={k:[r['group'] for r in groups if r['kind']==k and not r['coarse_fine_comparison_passed']] for k in ('raw','projected')} if stage!='coarse' else None,
        maximum_coarse_fine_difference={k:max((r['coarse_fine_maximum_difference'] or 0.) for r in groups if r['kind']==k) for k in ('raw','projected')},
        invalid_selected_groups=[dict(record=r['record'],group=r['group'],kind=r['kind']) for r in groups if not r['valid_unclipped_ranges']],
        recommendation_comparisons=comparisons,estimated_cpu_seconds=work['estimated_cpu_seconds'],
        new_integral_cpu_seconds=sum(t['new_integral_cpu_seconds'] for t in tasks),
        successful_task_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),
        maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),collection_cpu_seconds=time.process_time()-started,
        previous_stage_new_integral_cpu_seconds=previous['new_integral_cpu_seconds'] if previous else 0.,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False,
        scope='All original365 actions at one additional opened development label and32 fixed opposite heads. Coarse/fine comparisons are not independent/GH precision admission; only reused validated target groups and native direct references are admitted. No full KG or independent decision claim.')
    save(out/(stage+'_risk_bank.json'),banks);save(out/(stage+'_groups.json'),groups);save(out/(stage+'_summary.json'),result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','prepare-independent','run','collect'])
    parser.add_argument('--stage',choices=list(STAGES),default='coarse');parser.add_argument('--identity',type=int)
    parser.add_argument('--kind',choices=['raw','projected']);parser.add_argument('--shard',type=int)
    parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    if args.mode=='prepare':prepare(args.plan)
    elif args.mode=='prepare-independent':prepare_independent(args.plan)
    elif args.mode=='run':run(args.identity,args.kind,args.shard,args.stage,args.plan)
    else:
        if args.stage in ('gh_pilot','gh_full'):parser.error('GH collection uses performance.probe_historical_label_gh')
        collect(args.stage,args.plan)
