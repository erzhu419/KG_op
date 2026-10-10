"""Price the remaining closest guarded63 references, without dispatch."""
import json
import argparse
import signal
import time

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.validate_historical_risk_competition import PLAN, RULES, CACHES, cached_rows, select_competitors

tick=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
p=argparse.ArgumentParser();p.add_argument('--plan',default=PLAN);args=p.parse_args()
spec=read(args.plan);out=ROOT/spec['output'];base=ROOT/read(spec['parent_plan'])['output']
report=read('paper_artifacts/historical_body_reference_assembly_v1/competition_summary.json')
assert report['complete']
covered={(r['record'],r['kind'],r['group']) for r in
    read('paper_artifacts/historical_body_reference_assembly_v1/reference_sources.json')}
units={}
for kind in ('raw','projected'):
    for h in (4,5):
        observed=[read(str(base/'parts'/f'063_{h}_{kind}.json'))['rows']]
        for directory in CACHES:
            observed.extend(read(str(p))['rows'] for p in
                (ROOT/'paper_artifacts'/directory/'parts').glob(f'*/063_{h}_{kind}.json'))
        for rule in RULES:
            units[kind,h,rule]=max(d[rule]['cpu_seconds'] for d in observed if rule in d)
remaining=[]
for record in report['records']:
    for kind in ('raw','projected'):
        if (record['record'],kind,63) in covered or kind=='projected' and record['position']==197:continue
        groups=[r for r in record[kind+'_competitors']['closest_groups'] if r['group']==63]
        row=dict(record=record['record'],kind=kind,group=63,cdf=record['cdf'],
            minimum_gap=groups[0]['minimum_gap'] if groups else None)
        estimates={};missing={}
        for h in (4,5):
            cached=cached_rows(row,h);missing[str(h)]=[n for n in RULES if n not in cached]
            estimates[str(h)]=1.2*(sum(units[kind,h,n] for n in missing[str(h)])+1)
        remaining.append(dict(row,missing_rules=missing,estimated_identity_cpu_seconds=estimates,
            estimated_new_cpu_seconds=sum(estimates.values())))
immediate=select_competitors(report)
assert all(r['group']==63 for r in immediate)
keys={(r['record'],r['kind'],r['group']) for r in immediate}
next_rows=[r for r in remaining if (r['record'],r['kind'],r['group']) in keys]
reserve=sum(spec['cpu_reserves'].values());next_estimate=reserve+sum(r['estimated_new_cpu_seconds'] for r in next_rows)
all_estimate=reserve+sum(r['estimated_new_cpu_seconds'] for r in remaining)
batch=[];batch_estimate=reserve
for row in sorted((r for r in remaining if r['minimum_gap'] is not None),
        key=lambda r:(r['minimum_gap'],r['record'],r['kind'])):
    if batch_estimate+row['estimated_new_cpu_seconds']>spec['maximum_estimated_cpu_seconds']:break
    assert max(row['estimated_identity_cpu_seconds'].values())<=spec['maximum_job_cpu_seconds']
    batch.append(row);batch_estimate+=row['estimated_new_cpu_seconds']
other={kind:min(c['minimum_gap'] for r in report['records'] for c in
    r[kind+'_competitors']['closest_groups'] if c['group']!=63) for kind in ('raw','projected')}
result=dict(complete=True,next_closest=next_rows,next_phase_estimated_cpu_seconds=next_estimate,
    budgeted_batch=batch,budgeted_batch_estimated_cpu_seconds=batch_estimate,
    budgeted_batch_workers=2*len(batch),maximum_live_workers=8,
    next_phase_cost_admitted=next_estimate<=spec['maximum_estimated_cpu_seconds'] and
        max(v for r in next_rows for v in r['estimated_identity_cpu_seconds'].values())<=spec['maximum_job_cpu_seconds'],
    remaining_guarded63_group_kind_record_checks=len(remaining),
    all_guarded63_estimated_cpu_seconds=all_estimate,all_guarded63_cost_admitted=all_estimate<=spec['maximum_estimated_cpu_seconds'],
    closest_other_unreferenced_group_gap=other,remaining=remaining,maximum_estimated_cpu_seconds=3600,
    new_integrations=0,dispatched=False,cpu_seconds=time.process_time()-tick,
    limitations='Cost projection for a future phase only; no jobs dispatched. All six rules and both positive utility identities use maximum measured same-group/identity/rule timers plus20percent and setup. Guarded63 is the closest empirical body competitor; other groups remain unvalidated. Finite gaps are not error bounds or full maximization admission.')
save(out/'next_cost_plan.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='remaining'}),flush=True)
