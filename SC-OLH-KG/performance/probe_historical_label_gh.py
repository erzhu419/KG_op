"""Check GH64 on paired low-resolution residuals before pricing full coverage."""
import argparse
import json
import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save
from performance.diagnose_historical_component_guarded import mixed_values
from performance.validate_historical_component_transfer import part_path
from performance.validate_historical_label_bank import PLAN, anchor_part_path, anchor_tables, valid_tables, independent_reference


def select_pilot(groups,layout,known):
    selected={(r['record'],r['group'],r['kind']) for r in known}
    selected.update((r['record'],r['group'],r['kind']) for r in groups
        if (r.get('independent_check') or {}).get('reference',{}).get('method')=='direct_F7')
    for kind,action in (('raw','guarded'),('raw','direct'),('projected','guarded')):
        rows=[r for r in groups if r['kind']==kind and layout[r['group']]['kind']==action
            and not r['target_precision_admitted']]
        for metric in ('coarse_fine_maximum_difference','independent'):
            def difference(row):
                return max(row['independent_check']['differences'].values()) if metric=='independent' else row[metric]
            if rows:
                row=max(rows,key=difference);selected.add((row['record'],row['group'],row['kind']))
    return selected


def prepare(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    previous=read(str(out/'independent_summary.json'));assert previous['complete']
    assert not any(previous['independent_failed_groups'].values())
    contexts=read(str(out/'contexts.json'));layout=read(str(out/'bank_layout.json'))['groups']
    known=[r for r in read(spec['validated_groups'])['groups'] if r['record'] in contexts]
    chosen=select_pilot(read(str(out/'independent_groups.json')),layout,known)
    selected=[dict(r,rules=['G5']) for r in read(str(out/'fine_plan.json'))['selected']
        if (r['record'],r['group'],r['kind']) in chosen]
    total=0.
    for row in selected:
        cost=0.
        for h in contexts[row['record']]['utility_identities']:
            anchor=read(str(base/'parts'/f'{row["group"]:03d}_{h}_{row["kind"]}.json'))['rows']
            target=read(str(part_path(out,row,h)))['rows'];path=anchor_part_path(out,row,h)
            cached=read(str(path))['rows'] if path.exists() else {}
            if 'G5' not in cached:cost+=2*anchor['E5']['cpu_seconds']
            if 'G5' not in target:cost+=2*target['E5']['cpu_seconds']
        row['estimated_integral_cpu_seconds']=cost;total+=cost
    workers=2*sum(min(spec['shards_per_identity_kind'],sum(r['kind']==k for r in selected)) for k in ('raw','projected'))
    estimate=1.2*(total+workers)
    work=dict(selected=selected,estimated_cpu_seconds=estimate,workers=workers,
        maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds_per_stage'],
        estimated_cost_admitted=estimate<=spec['maximum_estimated_cpu_seconds_per_stage'],
        scope='Historical five worst pairs, all direct-independent reference substitutions, and actual worst resolution/independent pairs in raw guarded/direct and native guarded. GH64/32-point anchor and target residuals retain the original admitted anchor. Each missing integral priced at twice its matching GH32/32-point timer, plus20 percent headroom. No target128-point GH resampling.')
    save(out/'gh_pilot_plan.json',work);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in work.items() if k!='selected'}),flush=True)
    return work


def prepare_full(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    pilot=read(str(out/'gh_pilot_summary.json'));assert pilot['pilot_passed']
    contexts=read(str(out/'contexts.json'));layout=read(str(out/'bank_layout.json'))['groups']
    checked={(r['record'],r['group'],r['kind']) for r in pilot['groups'] if r['checks']['passed']}
    pending={(r['record'],r['group'],r['kind']) for r in read(str(out/'independent_groups.json'))
        if not r['target_precision_admitted'] and (r['record'],r['group'],r['kind']) not in checked}
    selected=[dict(r,rules=['G5']) for r in read(str(out/'fine_plan.json'))['selected']
        if (r['record'],r['group'],r['kind']) in pending]
    total=0.
    for row in selected:
        gid,kind=row['group'],row['kind'];cost=0.
        for h in contexts[row['record']]['utility_identities']:
            source=read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']
            target=read(str(part_path(out,row,h)))['rows'];path=anchor_part_path(out,row,h)
            cached=read(str(path))['rows'] if path.exists() else {}
            for role,data,cache in (('anchor',source,cached),('target',target,target)):
                if 'G5' not in cache:
                    cost+=pilot['measured_maximum_G5_E5_ratios'][f'{kind}_{layout[gid]["kind"]}_{h}_{role}']*data['E5']['cpu_seconds']
        row['estimated_integral_cpu_seconds']=cost;total+=cost
    workers=2*sum(min(spec['shards_per_identity_kind'],sum(r['kind']==k for r in selected)) for k in ('raw','projected'))
    estimate=1.2*(total+workers)
    work=dict(stage='gh_full',selected=selected,workers=workers,estimated_cpu_seconds=estimate,
        maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds_per_stage'],
        estimated_cost_admitted=estimate<=spec['maximum_estimated_cpu_seconds_per_stage'],
        scope='One combined stage for every missing anchor and target G5 integral. Reuse pilot and direct target references. Price uses maximum measured pilot G5/E5 ratios by identity, action kind, valuation and role, plus worker overhead and20 percent headroom.')
    save(out/'gh_full_plan.json',work);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in work.items() if k!='selected'}),flush=True)
    return work


def check(bundle,anchor,target,selected,actual,kind,tolerance):
    variants={r:anchor_tables(bundle,anchor,target,r,selected) for r in ('E5','E7','G5')}
    variants['F7'],reference=independent_reference(bundle,anchor,target,selected)
    values={r:mixed_values(bundle,*v,kind) for r,v in variants.items()}
    values['selected']=mixed_values(bundle,*actual,kind)
    differences={r:float(abs(v-values['G5']).max()) for r,v in values.items() if r!='G5'}
    norm=max(abs(rows[h][r]['diagnostics']['predictive_log_evidence_error'])
        for rows in (anchor,target) for h in bundle['utility_identities'] for r in ('E5','E7','F7','G5'))
    return dict(passed=bool(all(valid_tables(bundle,v) for v in variants.values()) and
        norm<=tolerance and max(differences.values())<=tolerance),differences=differences,
        maximum_log_evidence_error=norm,independent_reference=reference)


def collect(plan=PLAN,stage='gh_pilot'):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    work=read(str(out/(stage+'_plan.json')));contexts=read(str(out/'contexts.json'))
    layout=read(str(out/'bank_layout.json'));bank=read(str(out/'independent_risk_bank.json'))
    selected=work['selected'] if stage=='gh_pilot' else read(str(out/'gh_pilot_plan.json'))['selected']+work['selected']
    groups=[];ratios={}
    for row in selected:
        gid,kind=row['group'],row['kind'];context=contexts[row['record']];ix=layout['groups'][gid]['action_indices']
        bundle=dict(context,actions=[layout['actions'][i] for i in ix],opposite=context['groups'][str(gid)]['opposite'])
        anchor={};target={}
        for h in context['utility_identities']:
            anchor[h]=read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']
            anchor[h].update(read(str(anchor_part_path(out,row,h)))['rows'])
            target[h]=read(str(part_path(out,row,h)))['rows']
            for role,data in (('anchor',anchor[h]),('target',target[h])):
                key=f'{kind}_{layout["groups"][gid]["kind"]}_{h}_{role}'
                ratio=data['G5']['cpu_seconds']/data['E5']['cpu_seconds']
                ratios[key]=max(ratios.get(key,0.),ratio)
        actual=(np.asarray(bank[row['record']]['statistics'][kind])[:,ix],np.asarray(bank[row['record']]['costs'][kind])[:,ix])
        result=check(bundle,anchor,target,row['anchor_rule'],actual,kind,spec['maximum_action_difference'])
        groups.append(dict(record=row['record'],group=gid,kind=kind,action_indices=ix,checks=result))
    tasks=[read(str(out/'tasks'/stage/f'{h}_{k}_{s}.json')) for h in context['utility_identities']
        for k in ('raw','projected') for s in range(spec['shards_per_identity_kind'])
        if s<sum(r['kind']==k for r in work['selected'])]
    complete=all(t['complete'] for t in tasks);passed=complete and all(g['checks']['passed'] for g in groups)
    checked={(r['record'],r['group'],r['kind']) for r in groups if r['checks']['passed']}
    if stage=='gh_full':
        previous=read(str(out/'independent_groups.json'));independent=read(str(out/'independent_summary.json'))
        admitted={k:sum(len(r['action_indices']) for r in previous if r['kind']==k and
            (r['target_precision_admitted'] or (r['record'],r['group'],r['kind']) in checked)) for k in ('raw','projected')}
        failures=[dict(record=r['record'],group=r['group'],kind=r['kind']) for r in previous
            if not r['target_precision_admitted'] and (r['record'],r['group'],r['kind']) not in checked]
        result=dict(stage=stage,complete=complete,actions=len(layout['actions']),gh_comparison_actions_passed=admitted,
            paired_residual_actions_checked={k:sum(len(r['action_indices']) for r in groups if r['kind']==k) for k in admitted},
            target_precision_actions_admitted=independent['target_precision_actions_admitted'],failed_groups=failures,
            all_label_comparisons_passed=bool(passed and not failures and
                all(v==len(layout['actions']) for v in independent['independent_actions_passed'].values()) and
                all(v==len(layout['actions']) for v in independent['coarse_fine_actions_passed'].values())),
            maximum_difference={k:max(max(r['checks']['differences'].values()) for r in groups if r['kind']==k) for k in admitted},
            estimated_cpu_seconds=work['estimated_cpu_seconds'],new_integral_cpu_seconds=sum(t['new_integral_cpu_seconds'] for t in tasks),
            successful_task_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),
            new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
            full_predictive_integration_admitted=False,
            scope='One additional development label and32 fixed opposite heads. Full action coverage combines the paired GH64/32-point residual comparisons with prior direct target reference checks and native direct analytic references. Direct target precision counts unchanged; no full CDF, outer KG or independent decision claim. Worker costs exclude the separately reported pilot.')
        save(out/'gh_full_groups.json',groups);save(out/'gh_full_summary.json',result)
        print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result
    pending={(g['record'],g['group'],g['kind']) for g in read(str(out/'independent_groups.json'))
        if not g['target_precision_admitted'] and (g['record'],g['group'],g['kind']) not in checked}
    costs={k:0. for k in ('raw','projected')};roles=dict(anchor=0.,target=0.)
    for row in read(str(out/'fine_plan.json'))['selected']:
        gid,kind=row['group'],row['kind']
        if (row['record'],gid,kind) not in pending:continue
        for h in contexts[row['record']]['utility_identities']:
            source=read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']
            target=read(str(part_path(out,row,h)))['rows']
            for role,data in (('anchor',source),('target',target)):
                cost=ratios[f'{kind}_{layout["groups"][gid]["kind"]}_{h}_{role}']*data['E5']['cpu_seconds']
                costs[kind]+=cost;roles[role]+=cost
    estimate=1.2*(sum(costs.values())+32)
    next_plan=dict(estimated_cpu_seconds=estimate,estimated_missing_integral_cpu_seconds=costs,
        maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds_per_stage'],
        estimated_cost_admitted=passed and estimate<=spec['maximum_estimated_cpu_seconds_per_stage'],
        estimated_separate_stage_cpu_seconds={r:1.2*(v+32) for r,v in roles.items()},
        separate_stage_pricing_scope='Anchor G5 is prepared once for reuse by later labels; target G5 checks the current label. Separate estimates include32 worker CPU seconds each. Their sum must be reported if these stages are dispatched.',
        pilot_passed=passed,dispatched=False,pending_groups=len(pending),
        scope='Remaining groups with no direct target reference or passing pilot GH check. Cost uses the largest pilot G5/E5 timer ratio for each valuation/action kind, identity and anchor/target role, plus32 worker CPU seconds and20 percent headroom. Numerical differences remain diagnostics.')
    result=dict(complete=complete,pilot_passed=passed,groups=groups,
        actions_passed={k:sum(len(g['action_indices']) for g in groups if g['kind']==k and g['checks']['passed']) for k in costs},
        maximum_difference=max(max(g['checks']['differences'].values()) for g in groups),
        measured_maximum_G5_E5_ratios=ratios,next_plan=next_plan,estimated_cpu_seconds=work['estimated_cpu_seconds'],
        new_integral_cpu_seconds=sum(t['new_integral_cpu_seconds'] for t in tasks),
        successful_task_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False,
        scope='Selected development group-kind pairs only. Paired residual GH64/32-point checks do not constitute full-bank GH coverage or direct high-order references at the target label.')
    save(out/'gh_residual_cost_plan.json',next_plan);save(out/'gh_pilot_summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','prepare-full','collect'])
    parser.add_argument('--stage',choices=['gh_pilot','gh_full'],default='gh_pilot');args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='prepare-full':prepare_full()
    else:collect(stage=args.stage)
