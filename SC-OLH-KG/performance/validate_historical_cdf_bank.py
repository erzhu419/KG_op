"""Extend the frozen CDF grid using complete screens and selected direct references."""
import argparse
import json
import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save
from performance.diagnose_historical_component_guarded import mixed_values
from performance.validate_historical_component_bank import group_checks
from performance.validate_historical_component_transfer import part_path
from performance.validate_historical_label_bank import anchor_tables, valid_tables, independent_reference
from performance.validate_historical_label_control import tables

PLAN='performance/manifests/historical_cdf_bank_v1.json'
RULES=['E7','F7','G7','E8','F8','H8','E9']


def select_pilot(groups,layout):
    selected=[]
    for record in sorted({r['record'] for r in groups}):
        for kind,action in (('raw','guarded'),('raw','direct'),('projected','guarded')):
            rows=[r for r in groups if r['record']==record and r['kind']==kind and
                layout[r['group']]['kind']==action and not r['target_precision_admitted']]
            invalid=[r for r in rows if not r['valid_unclipped_ranges']]
            row=min(invalid,key=lambda r:r['minimum_selected_statistic']) if invalid else max(
                rows,key=lambda r:r['baseline_maximum_action_difference'])
            selected.append(dict(record=record,group=row['group'],kind=kind,
                coarse_minimum_statistic=row['minimum_selected_statistic'],
                coarse_baseline_disagreement=row['baseline_maximum_action_difference'],
                selection_reason='invalid_coarse_range' if invalid else 'largest_unvalidated_baseline_disagreement'))
    return selected


def price_rows(spec,selected,contexts):
    out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    probe=read(str(ROOT/'paper_artifacts/historical_component_transfer_v1/cost_probe.json'));total=0.;maximum_worker=0.
    for row in selected:
        row['estimated_identity_cpu_seconds']={}
        for h in contexts[row['record']]['utility_identities']:
            source=read(str(base/'parts'/f'{row["group"]:03d}_{h}_{row["kind"]}.json'))['rows']
            cached=read(str(part_path(out,row,h)))['rows'];cost=0.
            for rule in row['rules']:
                if rule in cached:continue
                units=probe['resolution_units'][row['kind']+'_'+str(h)]
                fallback=units[rule] if rule in units else max(
                    g['times'][row['kind']][str(h)][rule] for g in probe['groups'].values())
                cost+=source.get(rule,{}).get('cpu_seconds',0.) or fallback
            estimate=1.2*(cost+1);total+=estimate;maximum_worker=max(maximum_worker,estimate)
            row['estimated_identity_cpu_seconds'][str(h)]=estimate
    return total,maximum_worker


def prepare_pilot(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output']
    assert read(str(out/'coarse_summary.json'))['complete']
    contexts=read(str(out/'contexts.json'));layout=read(str(out/'bank_layout.json'))['groups']
    selected=select_pilot(read(str(out/'coarse_groups.json')),layout)
    anchors={(r['record'],r['group'],r['kind']):r['anchor_rule'] for r in read(str(out/'coarse_plan.json'))['selected']}
    for row in selected:
        row.update(anchor_rule=anchors[row['record'],row['group'],row['kind']],rules=RULES,
            pairs=[['E8','F8'],['E8','H8'],['E8','E9']],selected_rule='E8')
    total,maximum_worker=price_rows(spec,selected,contexts)
    work=dict(selected=selected,estimated_cpu_seconds=total,maximum_estimated_worker_cpu_seconds=maximum_worker,
        maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds_per_stage'],
        estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds_per_stage'] and maximum_worker<=spec['maximum_job_cpu_seconds'],
        scope='One group per record and raw guarded/direct/native guarded category, prioritizing invalid coarse ranges; otherwise largest disagreement with the unvalidated old baseline. Direct256 references checked against independent256, GH64/256 and512. All missing rules, both utility identities,20 percent headroom and one CPU second per worker included. Historical matching timers reused; absent high rules priced at measured matching maxima.')
    save(out/'pilot_plan.json',work);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in work.items() if k!='selected'}),flush=True)
    return work


def prepare_decision(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];summary=read(str(out/'summary.json'))
    assert summary['complete']
    work=read(str(out/'pilot_plan.json'));layout=read(str(out/'bank_layout.json'))['groups']
    contexts=read(str(out/'contexts.json'));decision=read(str(out/'decision_relevance.json'))
    existing={(r['record'],r['group'],r['kind']) for r in work['selected']};extra=[]
    coarse={(r['record'],r['group'],r['kind']):r for r in read(str(out/'coarse_plan.json'))['selected']}
    for name,kinds in decision['records'].items():
        for kind,row in kinds.items():
            candidates=set(map(int,row['winner_actions']))|set(map(int,row['runner_up_actions']))
            for group in layout:
                key=name,group['group'],kind
                if not candidates.intersection(group['action_indices']) or key in existing or key not in coarse:continue
                extra.append(dict(record=name,group=group['group'],kind=kind,
                    anchor_rule=coarse[key]['anchor_rule'],rules=RULES,pairs=[['E8','F8'],['E8','H8'],['E8','E9']],
                    selected_rule='E8',selection_reason='current_winner_or_runner_up_group'))
    additional,maximum=price_rows(spec,extra,contexts)
    previous=work['estimated_cpu_seconds'];work['selected'].extend(extra)
    work.update(previous_pilot_estimated_cpu_seconds=previous,additional_estimated_cpu_seconds=additional,
        estimated_cpu_seconds=previous+additional,
        maximum_estimated_worker_cpu_seconds=max(maximum,work['maximum_estimated_worker_cpu_seconds']))
    work['estimated_cost_admitted']=bool(work['estimated_cpu_seconds']<=spec['maximum_estimated_cpu_seconds_per_stage'] and
        work['maximum_estimated_worker_cpu_seconds']<=spec['maximum_job_cpu_seconds'])
    work['scope']+=' Append only current winner/runner-up groups not already selected or analytically admitted. Original worker indices preserved; previous pilot estimate plus all additional missing rules and20 percent headroom share the same3600 CPU second cap.'
    for path,data in (('initial_pilot_plan.json',read(str(out/'pilot_plan.json'))),('initial_summary.json',summary)):
        if not (out/path).exists():save(out/path,data)
    save(out/'decision_reference_plan.json',work)
    if work['estimated_cost_admitted']:save(out/'pilot_plan.json',work)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in work.items() if k!='selected'}),flush=True)
    return work


def check(bundle,anchor,target,selected,kind,tolerance):
    direct={r:tables(bundle,{h:target[h][r] for h in bundle['utility_identities']}) for r in ('E8','F8','H8','E9')}
    reference=group_checks(bundle,[dict(identity=h,rows=target[h]) for h in bundle['utility_identities']],kind,
        [('E8','F8'),('E8','H8'),('E8','E9')],tolerance)
    reference['valid_unclipped_ranges']=all(valid_tables(bundle,v) for v in direct.values())
    reference['passed']=bool(reference['passed'] and reference['valid_unclipped_ranges'])
    variants={r:anchor_tables(bundle,anchor,target,r,selected) for r in ('E5','E7','G7')}
    variants['F7'],independent=independent_reference(bundle,anchor,target,selected)
    values={r:mixed_values(bundle,*data,kind) for r,data in variants.items()}
    values.update({r:mixed_values(bundle,*data,kind) for r,data in direct.items() if r in ('E8','E9')})
    differences={candidate:{r:float(abs(v-values[candidate]).max()) for r,v in values.items()
        if r!=candidate and (candidate=='E5' or r!='E5')} for candidate in ('E5','E7')}
    norm=max(abs(rows[h][r]['diagnostics']['predictive_log_evidence_error']) for rows in (anchor,target)
        for h in bundle['utility_identities'] for r in ('E5','E7','F7','G7'))
    valid={r:valid_tables(bundle,data) for r,data in variants.items()}
    admitted={r:bool(reference['passed'] and norm<=tolerance and max(differences[r].values())<=tolerance and
        all(v for n,v in valid.items() if r=='E5' or n!='E5')) for r in ('E5','E7')}
    if admitted['E5']:method,data='control32',variants['E5']
    elif admitted['E7']:method,data='control128',variants['E7']
    elif reference['passed']:method,data='direct256',direct['E8']
    else:method,data='HOLD',None
    return dict(selected_method=method,control_admitted=admitted,comparisons=differences,
        direct_reference_checks=reference,control_valid_unclipped_ranges=valid,independent_reference=independent,
        maximum_log_evidence_error=norm),data


def collect(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    work=read(str(out/'pilot_plan.json'));contexts=read(str(out/'contexts.json'));layout=read(str(out/'bank_layout.json'))
    banks=read(str(out/'coarse_risk_bank.json'));groups=[];tasks=[];cpu=0.
    for kind in ('raw','projected'):
        for shard,row in enumerate(r for r in work['selected'] if r['kind']==kind):
            gid=row['group'];context=contexts[row['record']];ix=layout['groups'][gid]['action_indices']
            bundle=dict(context,actions=[layout['actions'][i] for i in ix],opposite=context['groups'][str(gid)]['opposite'])
            anchor={h:read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows'] for h in context['utility_identities']}
            target={h:read(str(part_path(out,row,h)))['rows'] for h in context['utility_identities']}
            checks,data=check(bundle,anchor,target,row['anchor_rule'],kind,spec['maximum_action_difference'])
            if data is not None:
                for name,array in zip(('statistics','costs'),data):
                    whole=np.asarray(banks[row['record']][name][kind]);whole[:,ix]=array;banks[row['record']][name][kind]=whole.tolist()
            groups.append(dict(record=row['record'],group=gid,kind=kind,action_indices=ix,
                selection_reason=row['selection_reason'],**checks))
            cpu+=sum(target[h][r]['cpu_seconds'] for h in context['utility_identities'] for r in row['rules'])
            tasks.extend(read(str(out/'tasks/pilot'/f'{h}_{kind}_{shard}.json')) for h in context['utility_identities'])
    complete=all(t['complete'] for t in tasks);coarse=read(str(out/'coarse_groups.json'))
    known={(r['record'],r['group'],r['kind']) for r in groups if r['selected_method']!='HOLD'}
    counts={record:{kind:sum(len(r['action_indices']) for r in coarse if r['record']==record and r['kind']==kind and
        (r['target_precision_admitted'] or (record,r['group'],kind) in known)) for kind in ('raw','projected')} for record in contexts}
    remaining={}
    for record,context in contexts.items():
        missing=0.
        for row in read(str(out/'coarse_plan.json'))['selected']:
            if row['record']!=record or (record,row['group'],row['kind']) in known:continue
            for h in context['utility_identities']:
                if 'E7' in read(str(part_path(out,row,h)))['rows']:continue
                source=read(str(base/'parts'/f'{row["group"]:03d}_{h}_{row["kind"]}.json'))['rows']
                missing+=source['E7']['cpu_seconds'] or source['F7']['cpu_seconds']
        estimate=1.2*(missing+32)
        remaining[record]=dict(estimated_cpu_seconds=estimate,estimated_cost_admitted=estimate<=spec['maximum_estimated_cpu_seconds_per_stage'],dispatched=False)
    result=dict(complete=complete,records=[dict(name=n,cdf=c['predictive_cdf'],label=c['label']) for n,c in contexts.items()],
        groups=groups,all_pilot_references_admitted=bool(complete and all(r['selected_method']!='HOLD' for r in groups)),
        target_precision_actions_admitted=counts,remaining_fine_plans=remaining,estimated_cpu_seconds=work['estimated_cpu_seconds'],
        new_integral_cpu_seconds=cpu,successful_task_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),
        maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False,full_predictive_integration_admitted=False,
        scope='Two interior development CDF observations. All365-action coarse screens are unvalidated outside the explicit pilot/direct analytic references. Selected direct256 references and actual control/direct substitutions only; no full CDF, outer KG or independent decision claim.')
    save(out/'selected_risk_bank.json',banks);save(out/'summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare-pilot','prepare-decision','collect']);parser.add_argument('--plan',default=PLAN)
    args=parser.parse_args()
    {'prepare-pilot':prepare_pilot,'prepare-decision':prepare_decision,'collect':collect}[args.mode](args.plan)
