"""Reuse admitted n4 references to test the historical GH16 comparison."""
import argparse
import copy
import json
import resource
import shutil
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT,read,save
from performance.diagnose_historical_component_guarded import compressed_read
from performance.check_historical_observed_batch_risk import collect as collect_risk

PLAN='performance/manifests/historical_positive_risk_uppervariance_axis_gauss_pilot_v1.json'


def prepare(plan=PLAN):
    spec=read(plan);previous=read(spec['previous_phase_completion'])
    assert previous['state']=='phase_complete' and previous['all_reference_precision_passed']
    bound=read(spec['source_bound_plan']);context=compressed_read(ROOT/read(bound['bank_plan'])['output']/'context.json.gz')
    observations={r['record']:r for v in context['origins'].values() for r in v['risk']}
    out=ROOT/spec['output'];selected=[];jobs=[]
    for i,row in enumerate(spec['registered_cases']):
        source=read(row['source_plan']);folder=ROOT/source['output'];name=row['record'];kind=row['kind']
        check=next(c for c in read(source['output']+'/compact_summary.json')['groups'] if c['record']==name and c['kind']==kind)
        assert check['candidate_passed']
        bundle=compressed_read(folder/'marginals'/(name+'.json.gz'))
        assert bundle['prior_weights']==context['prior_weights'] and bundle['query']==context['query']
        assert bundle['replicates']==4 and len(bundle['opposite'])==16 and len(bundle['head_log_scores'])==8
        np.testing.assert_allclose(bundle['head_log_scores'],observations[name]['head_log_scores'],atol=1e-10,rtol=0)
        target=out/'marginals'/(name+'.json.gz');target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(folder/'marginals'/(name+'.json.gz'),target)
        selected.append(next(s for s in read(source['output']+'/preparation.json')['selected'] if s['record']==name))
        for job in row['jobs']:
            h=job['identity'];part=copy.deepcopy(read(source['output']+'/parts/'+name+f'/{h}_{kind}.json'))
            part.update(cpu_seconds=0.,cumulative_cpu_seconds=0.,reused_from=source['output'])
            destination=out/'parts'/name/f'{h}_{kind}.json';destination.parent.mkdir(parents=True,exist_ok=True)
            save(destination,part);jobs.append(dict(job,record=name,kind=kind,group=row['group'],shard=i))
    total=600+sum(j['estimated_cpu_seconds'] for j in jobs)
    assert total<=spec['maximum_estimated_cpu_seconds']
    assert all(j['estimated_cpu_seconds']<=spec['maximum_worker_cpu_seconds'] for j in jobs)
    prep=dict(complete=True,selected=selected,jobs=jobs,source_components={'4':512,'5':1024},
        estimated_cpu_seconds_including_reserves=total,estimated_cost_admitted=True,
        source_prior_preserved=True,all_identity_likelihoods=8,maximum_live_workers=8,
        previous_phase_completion=spec['previous_phase_completion'],new_physical_calls=0,new_source_fits=0,
        new_verifier_calls=0,independent_confirmation=False)
    save(out/'preparation.json',prep)
    print('SCHEDULEURM_DONE '+json.dumps(prep),flush=True)


def collect(plan=PLAN):
    spec=read(plan);result=collect_risk(plan);out=ROOT/spec['output'];old_cpu=0.;new_cpu=0.;preserved=True;differences=[]
    for row in spec['registered_cases']:
        source=read(row['source_plan']);name=row['record'];kind=row['kind'];candidate=row['candidate']
        lower=row['gauss_reference'];higher='H8' if candidate=='E8' else 'K9'
        check=next(g for g in result['groups'] if g['record']==name and g['kind']==kind)
        differences.extend(c['maximum_difference'] for c in check['checks']['comparisons'] if c['low']==lower)
        for h in (4,5):
            before=read(source['output']+'/parts/'+name+f'/{h}_{kind}.json')['rows']
            after=read(spec['output']+'/parts/'+name+f'/{h}_{kind}.json')['rows']
            old_cpu+=before[higher]['cpu_seconds'];new_cpu+=after[lower]['cpu_seconds']
            preserved=preserved and before[candidate]['statistics']==after[candidate]['statistics'] and before[candidate]['costs']==after[candidate]['costs']
    result.update(lower_gauss_precision_passed=result['local_risk_candidate_passed'],
        maximum_lower_gauss_difference=max(differences),cached_selected_statistics_preserved=preserved,
        cached_higher_gauss_rule_cpu_seconds=old_cpu,lower_gauss_rule_cpu_seconds=new_cpu,
        lower_to_higher_gauss_cpu_ratio=new_cpu/old_cpu,
        scope='Matched n4 cached records, both utility identities and original16 objective heads. Primary rules are unchanged; only GH16 references are new. This is a numerical-cost pilot.')
    assert preserved
    save(out/'compact_summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect']);p.add_argument('--plan',default=PLAN);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.plan)
        else:collect(a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
