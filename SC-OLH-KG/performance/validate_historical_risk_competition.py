"""Validate the closest uncovered risk competitors using the existing integrator."""
import argparse
import json
import signal
import time

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.validate_historical_component_transfer import prepare_contexts, part_path, record_bundle
from performance.validate_historical_reference_upgrade import upgrade

PLAN='performance/manifests/historical_risk_competition_v1.json'
RULES=['E5','E7','E8','F8','H8','E9']
CACHES=('historical_component_transfer_v1','historical_label_control_v1',
    'historical_label_groups_v1','historical_cdf_bank_v1','historical_body_reference_pilot_v1',
    'historical_nearby_anchor_v1','historical_label_bank_v1','historical_decision_regions_v1',
    'historical_risk_competition_v1','historical_risk_competition_batch_v1')


def select_competitors(report):
    selected={}
    for kind in ('raw','projected'):
        rows=[]
        for record in report['records']:
            summary=record[kind+'_competitors'];closest=summary['closest_groups']
            # The opened report keeps five closest groups. Its actual overlaps
            # must all be present before it can drive this pilot.
            overlaps=[r for r in closest if r['empirical_overlap_action_head_pairs']]
            assert len(overlaps)==summary['empirical_overlap_groups']
            for row in closest:
                rows.append(dict(row,record=record['record'],kind=kind,cdf=record['cdf']))
        closest=min(rows,key=lambda r:(r['minimum_gap'],r['record'],r['group']))
        for row in rows:
            reasons=[]
            if row is closest:reasons.append('closest_uncovered_competitor')
            if row['empirical_overlap_action_head_pairs']:reasons.append('empirical_gap_rule_overlap')
            if reasons:selected[row['record'],kind,row['group']]=dict(row,selection_reasons=reasons)
    return list(selected.values())


def cached_rows(row,h):
    result={}
    for directory in CACHES:
        path=part_path(ROOT/'paper_artifacts'/directory,row,h)
        if not path.exists():continue
        part=read(str(path))
        for key,value in dict(record=row['record'],group=row['group'],kind=row['kind'],identity=h).items():
            assert part[key]==value
        result.update(part['rows'])
    return result


def selection_rows(spec,out):
    if 'batch_plan' in spec:
        priced=read(spec['batch_plan'])
        assert priced['budgeted_batch_estimated_cpu_seconds']<=spec['maximum_estimated_cpu_seconds']
        return [dict(row,selection_reasons=['frozen_gap_ordered_batch']) for row in priced['budgeted_batch']]
    report=read(str(out/'competition_summary.json'));assert report['complete']
    return select_competitors(report)


def prepare(plan=PLAN):
    tick=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    if (out/'preparation.json').exists():
        result=read(str(out/'preparation.json'));assert result['complete']
        print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
        return result
    selected=selection_rows(spec,out)
    contexts,shared,original=prepare_contexts(dict(spec,records=sorted({r['record'] for r in selected}),
        groups=sorted({r['group'] for r in selected})))
    base=ROOT/read(spec['parent_plan'])['output']
    source=read(str(base/'bank_tail_resolution_summary.json'))['groups'];total=0.;maximum=0.
    layout=read(str(base/'context.json'))['groups']
    probe=read('paper_artifacts/historical_component_transfer_v1/cost_probe.json')
    for row in selected:
        gid,kind=row['group'],row['kind'];assert source[gid]['checks'][kind]['passed']
        row.update(rules=RULES,anchor_rule=source[gid]['selected_rules'][kind],estimated_identity_cpu_seconds={})
        for h in contexts[row['record']]['utility_identities']:
            anchor=read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']
            assert row['anchor_rule'] in anchor and all(n in anchor for n in ('E5','E7'))
            cached=cached_rows(row,h);observed=[anchor,cached]
            for directory in CACHES:
                parts=ROOT/'paper_artifacts'/directory/'parts'
                observed.extend(read(str(p))['rows'] for p in parts.glob(f'*/{gid:03d}_{h}_{kind}.json'))
            missing=0.;units={};unit_scope={}
            for rule in RULES:
                if rule in cached:continue
                times=[d[rule]['cpu_seconds'] for d in observed if rule in d]
                unit_scope[rule]='same_group_identity_rule'
                if not times:
                    # Raw63 has no historical fine rules. Use the measured
                    # maximum for the same guarded action kind and identity.
                    unit_scope[rule]='maximum_same_action_kind_identity_rule'
                    for group in layout:
                        if group['kind']!=layout[gid]['kind']:continue
                        rows=read(str(base/'parts'/f'{group["group"]:03d}_{h}_{kind}.json'))['rows']
                        if rule in rows:times.append(rows[rule]['cpu_seconds'])
                    times.extend(g['times'][kind][str(h)][rule] for g in probe['groups'].values()
                        if rule in g['times'][kind][str(h)])
                    if rule in probe['resolution_units'][kind+'_'+str(h)]:
                        times.append(probe['resolution_units'][kind+'_'+str(h)][rule])
                units[rule]=max(times)
                missing+=units[rule]
            estimate=1.2*(missing+1);total+=estimate;maximum=max(maximum,estimate)
            row['estimated_identity_cpu_seconds'][str(h)]=estimate
            row.setdefault('matching_timer_units',{})[str(h)]=units
            row.setdefault('timer_unit_scope',{})[str(h)]=unit_scope
            path=part_path(out,row,h);path.parent.mkdir(parents=True,exist_ok=True)
            save(path,dict(record=row['record'],group=gid,kind=kind,identity=h,rows=cached))
        row['estimated_new_cpu_seconds']=sum(row['estimated_identity_cpu_seconds'].values())
    cumulative=total+sum(spec['cpu_reserves'].values())
    work=dict(selected=selected,estimated_cpu_seconds=total,cumulative_estimated_cpu_seconds=cumulative,
        maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds'],
        maximum_estimated_worker_cpu_seconds=maximum,
        estimated_cost_admitted=cumulative<=spec['maximum_estimated_cpu_seconds'] and maximum<=spec['maximum_job_cpu_seconds'])
    save(out/'contexts.json',contexts);save(out/'resolution_plan.json',work)
    result=dict(complete=True,source_components={str(h):len(shared['fields'][h]['means']) for h in original['utility_identities']},
        selected=[{k:r[k] for k in ('record','kind','group','minimum_gap','selection_reasons')} for r in selected],
        formal_actions=len(original['actions']),estimated_cpu_seconds=total,cumulative_estimated_cpu_seconds=cumulative,
        estimated_cost_admitted=work['estimated_cost_admitted'],cpu_seconds=time.process_time()-tick,
        full_posterior_restorations=0,copied_joint_marginal_bytes=0,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/'preparation.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def collect(plan=PLAN):
    tick=time.process_time();spec=read(plan);out=ROOT/spec['output'];work=read(str(out/'resolution_plan.json'))
    contexts=read(str(out/'contexts.json'));base=ROOT/read(spec['parent_plan'])['output'];groups=[];tasks=[];integral=0.
    for kind in ('raw','projected'):
        for shard,row in enumerate(r for r in work['selected'] if r['kind']==kind):
            context=contexts[row['record']];gid=row['group'];identities=context['utility_identities']
            bundle=record_bundle(compressed_read(base/'marginals'/f'{gid:03d}.json.gz'),context)
            target={h:read(str(part_path(out,row,h)))['rows'] for h in identities}
            anchor={h:read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows'] for h in identities}
            checked=upgrade(bundle,anchor,target,kind,spec['maximum_action_difference'],
                dict(selected_method='unreferenced',target_precision_admitted=False),row['anchor_rule'])
            groups.append(dict(checked,record=row['record'],group=gid,kind=kind,anchor=read(str(base/'context.json'))['name'],
                anchor_rule=row['anchor_rule'],action_indices=context['groups'][str(gid)]['action_indices'],
                minimum_cached_gap=row['minimum_gap'],selection_reasons=row['selection_reasons']))
            for h in identities:
                task=read(str(out/'tasks/resolution'/f'{h}_{kind}_{shard}.json'));tasks.append(task)
                integral+=sum(target[h][n]['cpu_seconds'] for n in row['matching_timer_units'][str(h)])
    result=dict(complete=all(t['complete'] for t in tasks),groups=groups,
        all_target_references_admitted=all(r['hybrid_direct_admitted'] for r in groups),
        new_integral_cpu_seconds=integral,successful_task_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),
        maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),
        cumulative_estimated_cpu_seconds=work['cumulative_estimated_cpu_seconds'],cpu_seconds=time.process_time()-tick,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False,
        unique_target_labels=len(contexts),selected_action_checks=sum(len(r['action_indices']) for r in groups),
        limitations='Decision-selected references at opened development labels. Finite direct/independent/GH comparisons are not rigorous error bounds or independent decision evidence. Remaining CDF groups and outer KG are open.')
    save(out/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','collect']);p.add_argument('--plan',default=PLAN);args=p.parse_args()
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    prepare(args.plan) if args.mode=='prepare' else collect(args.plan)
