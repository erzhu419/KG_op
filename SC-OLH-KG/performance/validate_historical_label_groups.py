"""Extend paired label differences to the frozen historical worst groups."""
import argparse
import json

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save
from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.validate_historical_component_bank import group_checks
from performance.validate_historical_component_transfer import prepare_contexts, part_path, record_bundle
from performance.validate_historical_label_control import controlled_row, tables

PLAN='performance/manifests/historical_label_groups_v1.json'
VARIANTS={'candidate':('E8','E7'),'coarse':('E8','E5'),
    'independent':('F8','F7'),'gh':('H8','G7')}


def check(bundle,anchor,target,kind,tolerance,direct,finer=False):
    variants={name:tables(bundle,{h:controlled_row(anchor[h][hi],target[h][lo],anchor[h][lo])
        for h in bundle['utility_identities']}) for name,(hi,lo) in VARIANTS.items()}
    values={n:mixed_values(bundle,*v,kind) for n,v in variants.items()}
    differences={n:float(abs(v-values['candidate']).max()) for n,v in values.items() if n!='candidate'}
    valid={n:bool(np.min(s[bundle['utility_identities']])>=-1e-12 and
        np.max(s[bundle['utility_identities']])<=1+1e-12 and
        np.min(c[bundle['utility_identities']])>=0 and np.max(c)<=359) for n,(s,c) in variants.items()}
    normalization=max(abs(target[h][lo]['diagnostics']['predictive_log_evidence_error'])
        for h in bundle['utility_identities'] for _,lo in VARIANTS.values())
    refs={};coarse_refs={};reference=None
    if direct or finer:
        for n in ('E8','E9') if direct else ('E8',):
            normalization=max(normalization,max(abs(target[h][n]['diagnostics']['predictive_log_evidence_error'])
                for h in bundle['utility_identities']))
            v=mixed_values(bundle,*tables(bundle,{h:target[h][n] for h in bundle['utility_identities']}),kind)
            refs[n]=float(abs(v-values['candidate']).max())
            coarse_refs[n]=float(abs(v-values['coarse']).max())
        if direct:
            reference=group_checks(bundle,[dict(identity=h,rows=target[h]) for h in bundle['utility_identities']],kind,
                [('E8','F8'),('E8','H8'),('E8','E9')],tolerance)
    # A direct finer rule supersedes the coarser E5 comparison for E7 admission.
    convergence=max(refs.values()) if direct or finer else differences['coarse']
    primary_valid=all(valid[n] for n in ('candidate','independent','gh')) and (direct or finer or valid['coarse'])
    passed=bool(primary_valid and normalization<=tolerance and convergence<=tolerance and
        differences['independent']<=tolerance and differences['gh']<=tolerance)
    coarse_differences={n:float(abs(v-values['coarse']).max()) for n,v in values.items() if n!='coarse'}
    coarse_passed=bool(all(valid.values()) and normalization<=tolerance and
        max(coarse_differences.values())<=tolerance and (not (direct or finer) or max(coarse_refs.values())<=tolerance))
    reference_passed=bool(direct and reference['passed'])
    method='control128' if passed else 'direct256' if reference_passed else 'HOLD'
    selected=variants['candidate'] if passed else tables(bundle,{h:target[h]['E8'] for h in bundle['utility_identities']}) if reference_passed else None
    return dict(comparisons=differences,direct_differences=refs,direct_reference_checks=reference,
        valid_unclipped_ranges=bool(primary_valid),maximum_log_evidence_error=normalization,
        control_rule_diagnostics_passed=passed,
        direct_control_admitted=bool(direct and passed and reference['passed']),
        finer_direct_rule_available=bool(direct or finer),selected_method=method,
        selected_statistics=selected[0].tolist() if selected is not None else None,
        selected_costs=selected[1].tolist() if selected is not None else None,
        hybrid_rule_diagnostics_passed=method!='HOLD',
        hybrid_direct_admitted=bool(reference_passed and method!='HOLD'),
        coarse_comparisons=coarse_differences,coarse_direct_differences=coarse_refs,
        coarse_rule_diagnostics_passed=coarse_passed,
        coarse_direct_control_admitted=bool(direct and coarse_passed and reference['passed']),
        direct_reference_available=direct)


def prepare(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    base=ROOT/read(spec['parent_plan'])['output'];contexts,shared,original=prepare_contexts(spec)
    historical=read(spec['selection'])['selected']
    assert sorted({r['group'] for r in historical})==spec['groups']
    prior=read(str(base/'bank_tail_resolution_summary.json'))['groups']
    costs=read(str(out/'anchor_costs.json'));selected=[];total=0.
    fallback=read('paper_artifacts/historical_component_transfer_v1/cost_probe.json')['resolution_units']
    for name in spec['records']:
        for old in historical:
            gid,kind=old['group'],old['kind'];direct=name==spec['direct_record']
            assert prior[gid]['checks'][kind]['passed'] and prior[gid]['selected_rules'][kind]=='E8'
            row=dict(record=name,group=gid,kind=kind,rules=['E5','E7','F7','G7']+
                (['E8','F8','H8','E9'] if direct else []),direct_reference=direct)
            missing=0.;spent=0.
            for h in contexts[name]['utility_identities']:
                anchor=read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']
                assert all(n in anchor for n in ('E5','E7','F7','G7','E8','F8','H8'))
                cached=read(str(part_path(out,row,h)))['rows'] if part_path(out,row,h).exists() else {}
                for n in row['rules']:
                    if n in cached:spent+=cached[n]['cpu_seconds'];continue
                    unit=costs[str(gid)+'_'+kind][str(h)]
                    missing+=unit[n] if n in unit else fallback[kind+'_'+str(h)][n]
                missing+=1.
            row['cached_cpu_seconds']=spent;row['estimated_new_cpu_seconds']=1.2*missing
            row['estimated_cpu_seconds']=spent+row['estimated_new_cpu_seconds']
            selected.append(row);total+=row['estimated_cpu_seconds']
    work=dict(selected=selected,estimated_cpu_seconds=total,cumulative_estimated_cpu_seconds=total,
        estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds'])
    save(out/'contexts.json',contexts);save(out/'resolution_plan.json',work)
    result=dict(complete=True,source_components={str(h):len(shared['fields'][h]['means']) for h in original['utility_identities']},
        records=spec['records'],group_kind_pairs=[{k:r[k] for k in ('group','kind')} for r in historical],
        combinations=len(selected),direct_reference_combinations=len(historical),
        estimated_cpu_seconds=total,estimated_cost_admitted=work['estimated_cost_admitted'],
        full_posterior_restorations=0,copied_joint_marginal_bytes=0,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False,full_predictive_integration_admitted=False)
    save(out/'preparation.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def prepare_refinement(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];initial=out/'initial_summary.json'
    if not initial.exists():save(initial,read(str(out/'summary.json')))
    previous=read(str(initial));assert previous['complete']
    contexts=read(str(out/'contexts.json'));selected=[];estimate=0.;spent=0.
    for r in previous['groups']:
        if r['control_rule_diagnostics_passed'] or r['direct_reference_available']:continue
        coarse_only=all(r['comparisons'][n]<=spec['maximum_action_difference'] for n in ('independent','gh'))
        row={k:r[k] for k in ('record','group','kind')}
        row.update(rules=['E8'] if coarse_only else ['E8','F8','H8','E9'],direct_reference=not coarse_only)
        missing=0.;done=0.
        for h in contexts[r['record']]['utility_identities']:
            cache=read(str(part_path(out,row,h)))['rows']
            reference=read(str(part_path(out,dict(row,record=spec['direct_record']),h)))['rows']
            for n in row['rules']:
                if n in cache:done+=cache[n]['cpu_seconds']
                else:missing+=reference[n]['cpu_seconds']
            missing+=1.
        row.update(cached_cpu_seconds=done,estimated_new_cpu_seconds=1.2*missing,
            estimated_cpu_seconds=done+1.2*missing)
        selected.append(row);estimate+=1.2*missing;spent+=done
    cumulative=previous['cpu_seconds']+spent+estimate
    work=dict(selected=selected,previous_measured_cpu_seconds=previous['cpu_seconds'],
        estimated_cpu_seconds=spent+estimate,cumulative_estimated_cpu_seconds=cumulative,
        estimated_cost_admitted=cumulative<=spec['maximum_estimated_cpu_seconds'])
    save(out/'adaptive_plan.json',work);print('SCHEDULEURM_DONE '+json.dumps(work),flush=True)
    return work


def collect(plan=PLAN,stage='resolution'):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    contexts=read(str(out/'contexts.json'));work=read(str(out/'resolution_plan.json'))
    extra=read(str(out/'adaptive_plan.json')) if stage=='adaptive' else None
    refinements={(r['record'],r['group'],r['kind']):r for r in extra['selected']} if extra else {}
    rows=[];tasks=[];cpu=0.
    for kind in ('raw','projected'):
        for shard,row in enumerate(r for r in work['selected'] if r['kind']==kind):
            context=contexts[row['record']];gid=row['group']
            bundle=record_bundle(compressed_read(base/'marginals'/f'{gid:03d}.json.gz'),context)
            anchor={};target={}
            for h in context['utility_identities']:
                anchor[h]=read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']
                target[h]=read(str(part_path(out,row,h)))['rows']
                cpu+=sum(target[h][n]['cpu_seconds'] for n in row['rules'])
                tasks.append(read(str(out/'tasks/resolution'/f'{h}_{kind}_{shard}.json')))
            refinement=refinements.get((row['record'],gid,kind))
            result=check(bundle,anchor,target,kind,spec['maximum_action_difference'],
                row['direct_reference'] or bool(refinement and refinement['direct_reference']),bool(refinement))
            rows.append(dict(record=row['record'],group=gid,kind=kind,**result))
    if extra:
        for kind in ('raw','projected'):
            for shard,row in enumerate(r for r in extra['selected'] if r['kind']==kind):
                for h in contexts[row['record']]['utility_identities']:
                    part=read(str(part_path(out,row,h)))['rows'];cpu+=sum(part[n]['cpu_seconds'] for n in row['rules'])
                    tasks.append(read(str(out/'tasks/adaptive'/f'{h}_{kind}_{shard}.json')))
    result=dict(complete=all(t['complete'] for t in tasks),groups=rows,cpu_seconds=cpu,
        estimated_cpu_seconds=extra['cumulative_estimated_cpu_seconds'] if extra else work['estimated_cpu_seconds'],
        comparative_actions_passed={k:3*sum(r['control_rule_diagnostics_passed'] for r in rows if r['kind']==k) for k in ('raw','projected')},
        comparative_actions_total={k:3*sum(r['kind']==k for r in rows) for k in ('raw','projected')},
        direct_control_actions_admitted={k:3*sum(r['direct_control_admitted'] for r in rows if r['kind']==k) for k in ('raw','projected')},
        direct_control_actions_tested={k:3*sum(r['direct_reference_available'] for r in rows if r['kind']==k) for k in ('raw','projected')},
        coarse_comparative_actions_passed={k:3*sum(r['coarse_rule_diagnostics_passed'] for r in rows if r['kind']==k) for k in ('raw','projected')},
        coarse_direct_actions_admitted={k:3*sum(r['coarse_direct_control_admitted'] for r in rows if r['kind']==k) for k in ('raw','projected')},
        hybrid_diagnostic_actions_passed={k:3*sum(r['hybrid_rule_diagnostics_passed'] for r in rows if r['kind']==k) for k in ('raw','projected')},
        hybrid_direct_actions_admitted={k:3*sum(r['hybrid_direct_admitted'] for r in rows if r['kind']==k) for k in ('raw','projected')},
        maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False,full_predictive_integration_admitted=False,
        scope='Frozen five historical worst group-kind pairs and three development labels. Direct 256/512 references only for explicitly marked combinations; other labels have comparative diagnostics. Failed low-order controls remain recorded, with validated direct256 fallback where available. All original budgets and identity mixture retained; no full-bank/full-KG or independent-decision admission.')
    save(out/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','prepare-refinement','collect'])
    parser.add_argument('--stage',choices=['resolution','adaptive'],default='resolution');args=parser.parse_args()
    if args.mode=='collect':collect(stage=args.stage)
    else:(prepare if args.mode=='prepare' else prepare_refinement)()
