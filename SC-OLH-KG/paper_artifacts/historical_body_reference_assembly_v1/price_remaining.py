"""Price missing opened-body references from existing matching timers only."""
import json
import argparse
import signal
import time
from collections import Counter, defaultdict

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.assemble_historical_risk_references import OUTPUT
from performance.validate_historical_component_transfer import part_path

start=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
p=argparse.ArgumentParser();p.add_argument('--plan',default='performance/manifests/historical_risk_competition_batch_v1.json');args=p.parse_args()
out=ROOT/OUTPUT;summary=read(str(out/'summary.json'));assert summary['complete']
layout=read('paper_artifacts/historical_cdf_bank_v1/bank_layout.json')
base=ROOT/'paper_artifacts/historical_component_bank_v1'
near=ROOT/'paper_artifacts/historical_label_bank_v1'
ratios=read(str(near/'gh_pilot_summary.json'))['measured_maximum_G5_E5_ratios']
coverage={(r['record'],r['kind'],r['group']) for r in read(str(out/'reference_sources.json'))}
rules=('E5','E7','F7','G5');units={};missing_anchor={};fallbacks=[]
for group in layout['groups']:
    gid=group['group']
    for kind in ('raw','projected'):
        if kind=='projected' and group['kind']=='direct':continue
        for h in (4,5):
            source=read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']
            target=read(str(part_path(near,dict(record='source_n1_o4_h1_p207',group=gid,kind=kind),h)))['rows']
            path=near/'anchor_parts'/f'{gid:03d}_{h}_{kind}.json'
            extra=read(str(path))['rows'] if path.exists() else {}
            observed=[source,target,extra]
            for directory,record in (('historical_body_reference_pilot_v1','source_n1_o4_h1_p202'),
                    ('historical_nearby_anchor_v1','source_n1_o4_h1_p197'),
                    ('historical_risk_competition_v1','source_n1_o4_h1_p196'),
                    ('historical_risk_competition_v1','source_n1_o4_h1_p199'),
                    ('historical_risk_competition_v1','source_n1_o5_h1_p193')):
                path=part_path(ROOT/'paper_artifacts'/directory,dict(record=record,group=gid,kind=kind),h)
                if path.exists():observed.append(read(str(path))['rows'])
            for directory in ('historical_risk_competition_batch_v1','historical_risk_competition_batch_v2',
                    'historical_risk_competition_batch_v3','historical_risk_competition_batch_v4'):
                observed.extend(read(str(path))['rows'] for path in
                    (ROOT/'paper_artifacts'/directory/'parts').glob(f'*/{gid:03d}_{h}_{kind}.json'))
            for rule in rules:
                times=[d[rule]['cpu_seconds'] for d in observed if rule in d and d[rule]['cpu_seconds']>0]
                if times:unit=max(times)
                elif rule=='G5':
                    unit=ratios[f'{kind}_{group["kind"]}_{h}_target']*units[gid,kind,h,'E5']
                    fallbacks.append(dict(group=gid,kind=kind,identity=h,rule=rule,method='measured_maximum_G5_E5_ratio'))
                else:raise ValueError((gid,kind,h,rule))
                units[gid,kind,h,rule]=unit
            if 'G5' not in extra:
                missing_anchor[gid,kind,h]=ratios[f'{kind}_{group["kind"]}_{h}_anchor']*source['E5']['cpu_seconds']

cache_dirs=('historical_component_transfer_v1','historical_label_control_v1',
    'historical_label_groups_v1','historical_cdf_bank_v1',
    'historical_body_reference_pilot_v1','historical_nearby_anchor_v1','historical_risk_competition_v1',
    'historical_risk_competition_batch_v1','historical_risk_competition_batch_v2','historical_risk_competition_batch_v3','historical_risk_competition_batch_v4')
records=[];counts=Counter();used_anchor=set();total=defaultdict(float)
for record in summary['records']:
    if record['cdf']>=.9:continue
    costs=defaultdict(float);missing=Counter(); relevant={k:set(record[k+'_reference_coverage']['unreferenced_relevant_groups']) for k in ('raw','projected')}
    for group in layout['groups']:
        gid=group['group']
        for kind in ('raw','projected'):
            if (record['record'],kind,gid) in coverage or kind=='raw' and gid==136 or kind=='projected' and (
                group['kind']=='direct' or gid==63 and record['position']==197):continue
            counts[kind]+=1
            for h in (4,5):
                cached={};row=dict(record=record['record'],group=gid,kind=kind)
                for directory in cache_dirs:
                    path=part_path(ROOT/'paper_artifacts'/directory,row,h)
                    if path.exists():cached.update(read(str(path))['rows'])
                for rule in rules:
                    if rule in cached:continue
                    costs[rule]+=units[gid,kind,h,rule];missing[rule]+=1
                    if gid in relevant[kind]:costs['decision_relevant_'+rule]+=units[gid,kind,h,rule]
                used_anchor.add((gid,kind,h))
    for key,value in costs.items():total[key]+=value
    records.append(dict(record=record['record'],cdf=record['cdf'],missing_rule_rows=dict(missing),
        matching_timer_cpu_seconds=dict(costs),estimated_candidate32_cpu_seconds=1.2*(costs['E5']+8),
        estimated_four_rule_comparison_cpu_seconds=1.2*(sum(costs[r] for r in rules)+8)))
anchor=sum(v for key,v in missing_anchor.items() if key in used_anchor)
estimate=1.2*(sum(total[r] for r in rules)+anchor+8)
phase=read(args.plan)
reserve=sum(phase['cpu_reserves'].values())
reserved_workers=read(phase['output']+'/resolution_plan.json')['estimated_cpu_seconds']
result=dict(complete=True,records=records,missing_group_kind_record_checks=dict(counts),
    missing_target_matching_timer_cpu_seconds=dict(total),missing_shared_anchor_G5_cpu_seconds=anchor,
    estimated_candidate32_cpu_seconds=1.2*(total['E5']+8),estimated_four_rule_comparison_cpu_seconds=estimate,
    maximum_estimated_cpu_seconds=3600,full_matrix_cost_admitted=estimate+reserved_workers+reserve<=3600,
    phase_workers_already_reserved_cpu_seconds=reserved_workers,preparation_pricing_collection_reassembly_reserve_cpu_seconds=reserve,
    phase_plan=args.plan,
    cumulative_estimated_cpu_seconds=estimate+reserved_workers+reserve,
    pricing_fallbacks=fallbacks,cpu_seconds=time.process_time()-start,new_integrations=0,dispatched=False,
    scope='Original64 fine body labels and365 actions. Exclude actual integrated references,native direct analytic entries,raw direct136 and the two native63 references. E5/E7/F7/G5 paired comparisons use maximum measured same-group,identity,rule source/p207/p202/p197 and decision-competitor times; unavailable G5 uses the existing measured maximum G5/E5 ratio. Cached target rules and shared anchors are counted once. Eight worker CPU seconds and20 percent headroom. Cost projection only: broad-label controls can fail and require additional direct references; this is not a precision or speedup claim.')
save(out/'remaining_cost_plan.json',result)
compact={k:v for k,v in summary.items() if k not in ('records','actions','observed_query_matrix')}
compact['records']=[{k:v for k,v in r.items() if k in ('record','origin','position','cdf','label','raw_winners','native_winners','raw_adaptation','achieved_adaptation','raw_reference_coverage','projected_reference_coverage')} for r in summary['records']]
for record in compact['records']:
    for kind in ('raw','projected'):
        coverage=record.get(kind+'_reference_coverage',{})
        coverage.pop('unreferenced_imprecise_action_indices',None);coverage.pop('unreferenced_relevant_action_indices',None)
compact['server_summary']=str(out/'summary.json');save(out/'compact_summary.json',compact)
signal.setitimer(signal.ITIMER_PROF,0.)
print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('records','pricing_fallbacks')}),flush=True)
