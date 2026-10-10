"""Direct reference and measured cost for cross-observation integral reuse."""
import argparse
import json
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save
from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.validate_historical_component_bank import group_checks
from performance.validate_historical_component_transfer import part_path, record_bundle

PLAN = 'performance/manifests/historical_label_control_v1.json'
VARIANTS = {'candidate': ('E9', 'E7'), 'refined': ('E10', 'E8'),
    'independent': ('F9', 'F7'), 'gh': ('L9', 'G7')}


def controlled_row(high, target, anchor):
    """Subtract paired unnormalized numerators; normalize by target mass."""
    ma, mt, ml = [np.exp(r['diagnostics']['predictive_log_evidence_error']) for r in (high, target, anchor)]
    return {k: ((ma*np.asarray(high[k])+mt*np.asarray(target[k])-ml*np.asarray(anchor[k]))/mt).tolist()
        for k in ('statistics', 'costs')}


def tables(bundle, rows):
    identities, actions = len(bundle['prior_weights']), len(bundle['actions'])
    s, c = np.zeros((identities, actions, 4)), np.zeros((identities, actions))
    for h, row in rows.items():
        s[h], c[h] = row['statistics'], row['costs']
    return s, c


def control_check(bundle, anchor, target, tolerance):
    variants, values = {}, {}
    candidate_cpu=0.
    for name, (high, low) in VARIANTS.items():
        tick=time.process_time()
        variants[name] = tables(bundle, {h: controlled_row(anchor[h][high], target[h][low], anchor[h][low])
            for h in bundle['utility_identities']})
        values[name]=mixed_values(bundle,*variants[name],'raw')
        if name=='candidate':candidate_cpu=time.process_time()-tick
    direct = {n: mixed_values(bundle, *tables(bundle, {h:target[h][n] for h in bundle['utility_identities']}), 'raw')
        for n in ('E9', 'E10')}
    comparisons = {n: float(abs(v-values['candidate']).max()) for n, v in values.items() if n!='candidate'}
    comparisons.update({'direct_'+n:float(abs(v-values['candidate']).max()) for n,v in direct.items()})
    valid = all(np.min(s[bundle['utility_identities']])>=-1e-12 and
        np.max(s[bundle['utility_identities']])<=1+1e-12 and
        np.min(c[bundle['utility_identities']])>=0 and np.max(c)<=359 for s,c in variants.values())
    return dict(comparisons=comparisons, valid_unclipped_ranges=bool(valid),candidate_composition_cpu_seconds=candidate_cpu,
        passed=bool(valid and max(comparisons.values())<=tolerance))


def prepare(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    out.mkdir(parents=True,exist_ok=True)
    old=ROOT/read(spec['transfer_plan'])['output'];contexts=read(str(old/'contexts.json'))
    previous=read(str(old/'adaptive_summary.json'));assert previous['complete']
    selected=[dict(r) for r in previous['failed_groups']]
    selected.append(dict(record='source_n1_o5_h1_p223',group=52,kind='raw'))
    selected.sort(key=lambda r:(r['kind'],r['record'],r['group']))
    fallback=read(str(old/'cost_probe.json'))['resolution_units']
    total=0.
    for row in selected:
        coarse=row['group']==12
        row.update(rules=['E9'] if coarse else
            (['E7'] if row['kind']=='raw' else [])+['E9','F9','L9','E10'],
            pairs=[['E8','E9'],['E8','F8'],['E8','H8']] if coarse else
                [['E9','E10'],['E9','F9'],['L9','E9']], selected_rule='E8' if coarse else 'E9')
        cost=0.;spent=0.
        for h in contexts[row['record']]['utility_identities']:
            anchor=read(str(base/'parts'/f'{row["group"]:03d}_{h}_{row["kind"]}.json'))['rows']
            cached=read(str(part_path(out,row,h)))['rows'] if part_path(out,row,h).exists() else {}
            units=fallback[row['kind']+'_'+str(h)]
            for rule in row['rules']:
                if rule in cached:
                    spent+=cached[rule]['cpu_seconds'];continue
                cost+=anchor[rule]['cpu_seconds'] if rule in anchor else (
                    2.2*units['E9'] if rule=='E10' else .75*units['E9'] if rule=='L9' else units[rule])
            cost+=1.
        row['cached_cpu_seconds']=spent
        row['estimated_new_cpu_seconds']=1.2*cost
        row['estimated_cpu_seconds']=spent+row['estimated_new_cpu_seconds'];total+=row['estimated_cpu_seconds']
    work=dict(selected=selected,estimated_cpu_seconds=total,cumulative_estimated_cpu_seconds=total,
        estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds'])
    save(out/'contexts.json',contexts);save(out/'resolution_plan.json',work)
    shared=compressed_read(base/'marginals/052.json.gz')
    result=dict(complete=True,source_components={str(h):len(shared['fields'][h]['means']) for h in (4,5)},
        estimated_cpu_seconds=total,estimated_cost_admitted=work['estimated_cost_admitted'],
        previous_experiment_cpu_seconds=read(str(old/'screen_summary.json'))['cpu_seconds']+previous['cpu_seconds'],
        previous_cost_hold_preserved=True,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(out/'preparation.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def collect(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
    old=ROOT/read(spec['transfer_plan'])['output'];contexts=read(str(out/'contexts.json'))
    previous={(r['record'],r['group'],r['kind']):r for r in read(str(old/'adaptive_summary.json'))['groups']}
    work=read(str(out/'resolution_plan.json'));controls=[];tasks=[];cpu=0.;matched=[]
    tol=spec['maximum_action_difference'];replay=0.;compose_cpu=0.
    for kind in ('raw','projected'):
        for shard,row in enumerate(r for r in work['selected'] if r['kind']==kind):
            context=contexts[row['record']];gid=row['group']
            bundle=record_bundle(compressed_read(base/'marginals'/f'{gid:03d}.json.gz'),context)
            merged={};parts=[];costs={n:0. for n in ('E7','E9')}
            for h in context['utility_identities']:
                prior=read(str(part_path(old,row,h)))['rows'];fresh=read(str(part_path(out,row,h)))
                merged[h]=dict(prior,**fresh['rows']);parts.append(dict(identity=h,rows=merged[h]))
                cpu+=sum(fresh['rows'][n]['cpu_seconds'] for n in row['rules'])
                tasks.append(read(str(out/'tasks/resolution'/f'{h}_{kind}_{shard}.json')))
                if kind=='raw' and gid==52:
                    for k in ('statistics','costs'):
                        replay=max(replay,float(abs(np.asarray(prior['E7'][k])-fresh['rows']['E7'][k]).max()))
                    for n in costs:costs[n]+=fresh['rows'][n]['cpu_seconds']
            checks=group_checks(bundle,parts,kind,row['pairs'],tol)
            previous[row['record'],gid,kind]=dict(record=row['record'],group=gid,kind=kind,
                checks=checks,selected_rule=row['selected_rule'])
            if kind=='raw' and gid==52:
                anchor={h:read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows'] for h in context['utility_identities']}
                control=control_check(bundle,anchor,merged,tol);compose_cpu+=control['candidate_composition_cpu_seconds']
                controls.append(dict(record=row['record'],direct_reference_checks=checks,**control))
                matched.append(dict(record=row['record'],target_E7_cpu_seconds=costs['E7'],target_E9_cpu_seconds=costs['E9']))
    complete=all(t['complete'] for t in tasks);groups=list(previous.values())
    failed=[{k:r[k] for k in ('record','group','kind')} for r in groups if not r['checks']['passed']]
    low=sum(r['target_E7_cpu_seconds'] for r in matched);high=sum(r['target_E9_cpu_seconds'] for r in matched)
    anchor_cost=sum(read(str(base/'parts'/f'052_{h}_raw.json'))['rows']['E9']['cpu_seconds'] for h in (4,5))
    result=dict(complete=complete,groups=groups,failed_groups=failed,controls=controls,
        selected_action_record_checks_passed={k:3*sum(r['checks']['passed'] for r in groups if r['kind']==k) for k in ('raw','projected')},
        selected_action_record_checks_total=dict(raw=18,projected=18),
        all_selected_action_precision_passed=complete and not failed,
        all_control_direct_checks_passed=complete and replay<=1e-12 and
            all(r['passed'] and r['direct_reference_checks']['passed'] for r in controls),
        repeated_low_rule_maximum_difference=replay,cpu_seconds=cpu,
        estimated_cpu_seconds=work['estimated_cpu_seconds'],matched_costs=matched,
        candidate_cost=dict(target_low_cpu_seconds=low,direct_high_cpu_seconds=high,composition_cpu_seconds=compose_cpu,
            reused_anchor_candidate_cpu_seconds=anchor_cost,
            incremental_speedup=high/(low+compose_cpu),three_record_speedup_including_anchor=high/(anchor_cost+low+compose_cpu)),
        cost_scope='E7 and E9 measured in the same workers with identical fields and all components. Candidate-only cost comparison; excludes setup, anchor precision diagnostics and the direct-reference validation experiment. Shared E9 anchor initialization is reported separately.',
        maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),
        previous_cost_hold_preserved=True,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False,full_predictive_integration_admitted=False,
        scope='Three raw labels on guarded group52 versus direct 512/1024 rules, plus the five previous unresolved combinations. All three original budgets and all identity scores/weights retained. Selected actions only; no full-bank/full-KG or independent-decision admission.')
    save(out/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('groups','controls')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','collect']);args=parser.parse_args()
    (prepare if args.mode=='prepare' else collect)()
