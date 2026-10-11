"""Price only competitors left by the matched marginal upper bounds."""
import json
import signal
import time

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.validate_historical_risk_competition import RULES, CACHES, cached_rows

start = time.process_time()
signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
out = ROOT/'paper_artifacts/historical_certificate_bound_body16_v3'
summary = read(str(out/'summary.json'))
assert summary['complete'] and summary['raw_marginal_numerical_gate_passed'] and summary['cached_reference_containment_passed']
selected = [r for r in summary['records'] if r['unreferenced_overlapping_action_head_pairs']]
base = ROOT/'paper_artifacts/historical_component_bank_v1'
layout = read(str(base/'context.json'))['groups']
probe = read('paper_artifacts/historical_component_transfer_v1/cost_probe.json')
from scipy.special import roots_legendre
positions = read('performance/manifests/historical_certificate_bound_body16_v3.json')['positions']
cdfs = dict(zip(positions,.9*(roots_legendre(16)[0]+1)/2))
units = {};scopes = {};records = []
for row in selected:
    gid,kind = row['group'],row['kind'];costs = {};missing = {}
    for h in (4,5):
        cached = cached_rows(row,h);missing[str(h)] = [r for r in RULES if r not in cached]
        observed = [read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows']]
        for directory in CACHES:
            observed.extend(read(str(p))['rows'] for p in
                (ROOT/'paper_artifacts'/directory/'parts').glob(f'*/{gid:03d}_{h}_{kind}.json'))
        for rule in missing[str(h)]:
            key = (gid,kind,h,rule)
            if key in units:continue
            times = [d[rule]['cpu_seconds'] for d in observed if rule in d]
            scopes[key] = 'same_group_identity_rule_maximum'
            if not times:
                scopes[key] = 'same_action_kind_identity_rule_maximum'
                for group in layout:
                    if group['kind'] != layout[gid]['kind']:continue
                    data = read(str(base/'parts'/f'{group["group"]:03d}_{h}_{kind}.json'))['rows']
                    if rule in data:times.append(data[rule]['cpu_seconds'])
                times.extend(g['times'][kind][str(h)][rule] for g in probe['groups'].values()
                    if rule in g['times'][kind][str(h)])
                if rule in probe['resolution_units'][kind+'_'+str(h)]:times.append(probe['resolution_units'][kind+'_'+str(h)][rule])
            units[key] = max(times)
        costs[str(h)] = 1.2*(sum(units[gid,kind,h,r] for r in missing[str(h)])+1.)
    records.append(dict(record=row['record'],kind=kind,group=gid,cdf=float(cdfs[int(row['record'].rsplit('_p',1)[1])]),
        minimum_gap=None,upper_overlap_action_head_pairs=row['unreferenced_overlapping_action_head_pairs'],
        maximum_upper_minus_fixed=row['maximum_upper_minus_fixed'],missing_rules=missing,
        estimated_identity_cpu_seconds=costs,estimated_new_cpu_seconds=sum(costs.values())))
phases=[];batch=[];estimate=600.;holds=[]
for row in sorted(records,key=lambda r:(-r['maximum_upper_minus_fixed'],r['record'],r['kind'],r['group'])):
    if max(row['estimated_identity_cpu_seconds'].values())>600 or row['estimated_new_cpu_seconds']+600>3600:
        holds.append(row);continue
    if estimate+row['estimated_new_cpu_seconds']>3600:
        phases.append(dict(combinations=batch,estimated_cpu_seconds=estimate));batch=[];estimate=600.
    batch.append(row);estimate+=row['estimated_new_cpu_seconds']
if batch:phases.append(dict(combinations=batch,estimated_cpu_seconds=estimate))
result = dict(complete=True,remaining_group_record_comparisons=len(records),
    counts={kind:sum(r['kind']==kind for r in records) for kind in ('raw','projected')},
    groups=sorted({r['group'] for r in records}),cdf_range=[min(r['cdf'] for r in records),max(r['cdf'] for r in records)] if records else [],
    records=records,phases=phases,holds_requiring_rule_shards=holds,
    estimated_integral_cpu_seconds=sum(r['estimated_new_cpu_seconds'] for r in records),
    estimated_all_phases_cpu_seconds=sum(p['estimated_cpu_seconds'] for p in phases),
    unit_timers=[dict(group=g,kind=k,identity=h,rule=r,cpu_seconds=v,scope=scopes[g,k,h,r]) for (g,k,h,r),v in sorted(units.items())],
    maximum_estimated_cpu_seconds_per_phase=3600,maximum_live_workers=8,dispatched=False,new_integrations=0,
    cpu_seconds=time.process_time()-start,
    scope='Only upper-bound overlaps left on the32 opened body16 labels. All six historical reference rules and original1e-4 checks are retained in this cost projection. Maximum matching measured timers,20percent headroom and setup,plus600 CPU seconds per future phase. Cost projection only; no joint references dispatched. Cost projection remains conditional on32 cached objective heads;formal KG admission remains open.')
save(out/'remaining_reference_cost_plan.json',result)
print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('records','phases','unit_timers','holds_requiring_rule_shards')}),flush=True)
