"""Reuse the pair integrator for one residual query63/incumbent45 case."""
import argparse
import copy
import json
import resource
import signal
import time

import numpy as np

from core.shared_certificate_pair import certificate_pair_states
from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from performance.complete_historical_source_heads import ROOT,read,save,cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.screen_historical_observed_batch_risk import condition_risk
from performance.refine_historical_query_certificate_caps import load_caps
from performance import screen_historical_observed_batch_risk as screen


def run(plan):
    spec=read(plan);out=ROOT/spec['output'];(out/'parts').mkdir(parents=True,exist_ok=True)
    assert spec['aggregate_estimated_phase_cpu_seconds']<=3600
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,65.)
    source=read(spec['source_bound_plan']);context=compressed_read(ROOT/read(source['bank_plan'])['output']/'context.json.gz')
    r=next(r for origin in context['origins'].values() for r in origin['risk'] if r['record']==spec['record'])
    old=read(source['output']+'/summary.json');row=next(r for r in old['records'] if (r['record'],r['kind'])==(spec['record'],'raw'))
    allocations=sorted({tuple(context['actions'][a]['budgets'][:2]) for a in row['candidate_actions'] if context['actions'][a]['shortlist'][:2]==[63,45]})
    shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals/063.json.gz')
    before=field_from_snapshot(shared['fields'][4]);future=copy.deepcopy(before);score=condition_risk(future,r,4,context)
    assert len(before.means)==512 and abs(score-r['head_log_scores'][4])<=1e-10
    profiles=[]
    for name,order,power,offset in spec['rules']:
        path=out/'parts'/f'{name}.json'
        if path.exists():saved=read(str(path))
        else:
            tick=time.process_time();states,d=certificate_pair_states(before,future,[0,1],allocations,r['label'],score,context['protocol'],
                query=0,raw=True,order=order,power=power,seed=spec['seed']+offset,replicates=4,eta_label=r['eta_label'],
                integrated_position=spec.get('integrated_position',0))
            saved=dict(rule=name,states=states.tolist(),diagnostics=d,cpu_seconds=time.process_time()-tick);save(path,saved)
        profiles.append(saved);print('SCHEDULEURM_PROGRESS '+json.dumps(dict(rule=name,cpu_seconds=saved['cpu_seconds'])),flush=True)
    arrays=[np.asarray(p['states']) for p in profiles]
    difference=max(float(np.max(abs(a-b))) for a in arrays for b in arrays)
    evidence=max(abs(p['diagnostics']['predictive_log_evidence_error']) for p in profiles)
    result=dict(complete=True,record=spec['record'],identity=4,kind='raw',pair=[63,45],budgets=allocations,
        upper=(np.maximum.reduce(arrays)+1e-4).tolist(),maximum_rule_difference=difference,
        maximum_log_evidence_error=evidence,precision_passed=max(difference,evidence)<=1e-4,
        all_positive_components_retained=True,new_joint_pair_rule_evaluations=4,new_physical_calls=0,
        new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/'pair_summary.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def collect(plan):
    spec=read(plan);pair=read(spec['output']+'/pair_summary.json');result=dict(pair)
    if not pair['precision_passed']:result['state']='precision_hold'
    else:
        anchor_spec=read(spec['anchor_plan'])
        anchors={(r['kind'],r['identity'],r['record'],(45,10),tuple(b)):u for h in (4,5)
            for r in read(anchor_spec['output']+f'/summary_{h}.json')['rows'] for b,u in zip(r['budgets'],r['upper'])}
        primary={('raw',4,pair['record'],(63,45),tuple(b)):u for b,u in zip(pair['budgets'],pair['upper'])}
        caps=load_caps(anchor_spec['query_cap_plan']);screen.branch_quality_upper=coupled_branch_quality_upper
        screen.collect(spec['followup_bound_plan'],certificate_caps=caps,anchor_caps=anchors,primary_incumbent_caps=primary)
        new=read(read(spec['followup_bound_plan'])['output']+'/compact_summary.json')
        old=read(read(spec['source_bound_plan'])['output']+'/compact_summary.json');comparison={}
        for kind in ('raw','projected'):
            a=old['weighted_bank_envelope']['kinds'][kind];b=new['weighted_bank_envelope']['kinds'][kind]
            assert abs(a['best_reference_contrast']-b['best_reference_contrast'])<1e-14
            assert b['unresolved_upper_gap']<=a['unresolved_upper_gap']+1e-12
            comparison[kind]=dict(old_gap=a['unresolved_upper_gap'],new_gap=b['unresolved_upper_gap'],new_totals=new['totals'][kind])
        result.update(state='bound_refined' if new['numerical_containment_passed'] else 'containment_hold',comparison=comparison,
            numerical_containment_passed=new['numerical_containment_passed'],maximum_reference_above_new_bound=new['maximum_validated_reference_above_bound'],
            all_conditional_actions_closed=all(v['record_group_candidates']==0 for v in new['totals'].values()),original_zero_candidate_gate_unchanged=True)
    save(ROOT/spec['output']/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='upper'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','collect']);p.add_argument('--plan',required=True);a=p.parse_args()
    try:run(a.plan) if a.stage=='run' else collect(a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
