"""Refine the single failed independent sequence within phase35's budget."""
import copy
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from performance.complete_historical_source_heads import ROOT,read,save


def prepare():
    plan='performance/manifests/historical_positive_risk_uppervariance_axis_competition_v35.json'
    spec=read(plan);out=ROOT/spec['output'];prep=read(spec['output']+'/preparation.json')
    summary=read(spec['output']+'/compact_summary.json')
    failures=[r for r in summary['groups'] if not r['candidate_passed']]
    assert len(failures)==1
    row=failures[0];record=row['record'];kind=row['kind']
    assert row['group']==34 and kind=='raw' and row['selected_candidate']=='E9'
    assert row['marginal_numerical_containment_passed']
    assert all(c['maximum_difference']<=1e-4 for c in row['checks']['comparisons'] if c['high']!='F9')
    case=next(c for c in spec['refined_cases'] if c['record']==record and c['kind']==kind)
    assert 'independent_reference' not in case
    frozen=read(spec['frozen_cost_plan']);new=[];shard=max(j['shard'] for j in prep['jobs'])+1
    for h in (4,5):
        old=next(j for j in frozen['next_joint_jobs'] if j['record']==record and j['kind']==kind and j['identity']==h)
        unit=max(old['rules'][r]['cpu'] for r in ('E9','F9'))
        new.append(dict(record=record,kind=kind,group=row['group'],identity=h,shard=shard,candidate='E9',
            rules={'F10':dict(cpu=2.5*unit,basis='Historical1024-point refinement factor times maximum matched512-point timer')},
            estimated_cpu_seconds=1.2*2.5*unit+2.))
    original=prep['estimated_cpu_seconds_including_reserves'];total=original+sum(j['estimated_cpu_seconds'] for j in new)
    assert total<=spec['maximum_estimated_cpu_seconds']
    assert all(j['estimated_cpu_seconds']<=spec['maximum_worker_cpu_seconds'] for j in new)
    save(out/'preparation_before_refinement.json',prep)
    save(out/'compact_summary_before_refinement.json',summary)
    save(out/'manifest_before_refinement.json',spec)
    case['rules'].append('F10');case['independent_reference']='F10'
    prep=copy.deepcopy(prep);prep['jobs'].extend(new)
    prep.update(estimated_cpu_seconds_including_reserves=total,expected_cpu_seconds_including600reserves=total)
    cost=dict(original_estimated_phase_cpu_seconds=original,new_jobs=new,
        cumulative_estimated_phase_cpu_seconds=total,maximum_estimated_cpu_seconds=3600,
        shared_phase_reserve_cpu_seconds=600,new_reserve_added=False,selected_rule='E9',
        comparisons=[['E8','E9'],['E9','F10'],['E9','K9'],['F9','F10']],
        scope='Only the failed independent comparison is refined. Original selected rule and passed groups are reused; both independent agreement and sequence resolution retain1e-4.')
    save(ROOT/plan,spec);save(out/'preparation.json',prep);save(out/'refinement_cost_plan.json',cost)
    print('SCHEDULEURM_DONE '+json.dumps(dict(manifest=spec,preparation=prep,cost=cost)),flush=True)


def prepare_primary():
    plan='performance/manifests/historical_positive_risk_uppervariance_axis_competition_v35.json'
    spec=read(plan);out=ROOT/spec['output'];prep=read(spec['output']+'/preparation.json')
    summary=read(spec['output']+'/compact_summary.json');failed=[r for r in summary['groups'] if not r['candidate_passed']]
    assert len(failed)==1
    row=failed[0];assert row['marginal_numerical_containment_passed']
    assert all(c['maximum_difference']<=1e-4 for c in row['checks']['comparisons'] if c['low']!='E9' or c['high']!='F10')
    case=next(c for c in spec['refined_cases'] if c['record']==row['record'] and c['kind']==row['kind'])
    assert case['independent_reference']=='F10' and 'identity_candidates' not in case
    cost=read(spec['output']+'/refinement_cost_plan.json')
    job=copy.deepcopy(next(j for j in cost['new_jobs'] if j['identity']==4))
    job.update(shard=max(j['shard'] for j in prep['jobs'])+1,rules={'E10':job['rules']['F10']})
    total=prep['estimated_cpu_seconds_including_reserves']+job['estimated_cpu_seconds']
    assert total<=spec['maximum_estimated_cpu_seconds'] and job['estimated_cpu_seconds']<=spec['maximum_worker_cpu_seconds']
    save(out/'preparation_before_primary_refinement.json',prep)
    save(out/'compact_summary_after_independent_refinement.json',summary)
    save(out/'manifest_before_primary_refinement.json',spec)
    case['rules_by_identity']={'4':case['rules']+['E10'],'5':case['rules'].copy()}
    case['identity_candidates']={'4':'E10','5':'E9'}
    prep['jobs'].append(job);prep.update(estimated_cpu_seconds_including_reserves=total,
        expected_cpu_seconds_including600reserves=total)
    cost['new_jobs'].append(job);cost.update(cumulative_estimated_phase_cpu_seconds=total,
        selected_rules_by_identity=case['identity_candidates'],
        comparisons=[['E8','E9'],['E9','selected'],['selected','F10'],['selected','K9'],['F9','F10']],
        scope='Only identity4 primary sequence is further refined; identity5 retainsE9. IndependentF10 is the recorded numerical validation, with original likelihoods, allocations and1e-4 gates.')
    save(ROOT/plan,spec);save(out/'preparation.json',prep);save(out/'refinement_cost_plan.json',cost)
    print('SCHEDULEURM_DONE '+json.dumps(dict(manifest=spec,preparation=prep,cost=cost)),flush=True)


if __name__=='__main__':
    try:
        if sys.argv[1:]:
            assert sys.argv[1:]==['primary'];prepare_primary()
        else:prepare()
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
