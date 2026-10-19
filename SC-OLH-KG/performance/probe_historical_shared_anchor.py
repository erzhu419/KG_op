"""Price-limited shared45/10 state diagnostic on two residual observations."""
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


def run(plan,identity):
    spec=read(plan);out=ROOT/spec['output'];(out/'parts').mkdir(parents=True,exist_ok=True)
    admission=spec['cost_admission'];assert admission['cost_admitted'] and admission['aggregate_estimated_phase_cpu_seconds']<=3600
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,admission['maximum_worker_cpu_seconds'])
    source=read(spec['source_bound_plan']);context=compressed_read(ROOT/read(source['bank_plan'])['output']/'context.json.gz')
    risk={r['record']:r for origin in context['origins'].values() for r in origin['risk']}
    shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals/063.json.gz')
    assert shared['points']==[63,45,10] and shared['prior_weights']==context['prior_weights']
    before=field_from_snapshot(shared['fields'][identity]);assert len(before.means)==spec['source_components'][str(identity)]
    summaries=[]
    for index,case in enumerate(spec['cases']):
        r=risk[case['record']];future=copy.deepcopy(before);score=condition_risk(future,r,identity,context)
        assert abs(score-r['head_log_scores'][identity])<=1e-10
        for kind in ('raw','projected'):
            profiles={}
            for name,order,power,offset in spec['rules']:
                path=out/'parts'/f'{identity}_{index}_{kind}_{name}.json'
                if path.exists():saved=read(str(path))
                else:
                    start=time.process_time()
                    values,d=certificate_pair_states(before,future,[1,2],case['anchor_budgets'],r['label'],score,context['protocol'],
                        query=0,raw=kind=='raw',order=order,power=power,seed=spec['seed']+identity*1009+index*313+offset,
                        replicates=4,eta_label=r['eta_label'])
                    assert d['all_positive_components_retained'] and d['components']==len(before.means)
                    saved=dict(record=case['record'],identity=identity,kind=kind,rule=name,
                        budgets=case['anchor_budgets'],states=values.tolist(),diagnostics=d,cpu_seconds=time.process_time()-start)
                    save(path,saved)
                profiles[name]=saved
                print('SCHEDULEURM_PROGRESS '+json.dumps({k:saved[k] for k in ('record','identity','kind','rule','cpu_seconds')}),flush=True)
            arrays=[np.asarray(p['states']) for p in profiles.values()]
            difference=max(float(np.max(abs(a-b))) for a in arrays for b in arrays)
            evidence=max(abs(p['diagnostics']['predictive_log_evidence_error']) for p in profiles.values())
            summaries.append(dict(record=case['record'],identity=identity,kind=kind,budgets=case['anchor_budgets'],
                maximum_rule_difference=difference,maximum_log_evidence_error=evidence,
                precision_passed=max(difference,evidence)<=1e-4,
                states=profiles['E9']['states'],upper=(np.maximum.reduce(arrays)+spec['numerical_padding']).tolist()))
    result=dict(complete=True,identity=identity,rows=summaries,precision_passed=all(s['precision_passed'] for s in summaries),
        maximum_rule_difference=max(s['maximum_rule_difference'] for s in summaries),
        maximum_log_evidence_error=max(s['maximum_log_evidence_error'] for s in summaries),
        all_positive_components_retained=True,new_joint_anchor_rule_evaluations=len(spec['cases'])*2*len(spec['rules']),
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    save(out/f'summary_{identity}.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def collect(plan):
    spec=read(plan);parts=[read(spec['output']+f'/summary_{h}.json') for h in (4,5)]
    result=dict(complete=True,precision_passed=all(p['precision_passed'] for p in parts),
        maximum_rule_difference=max(p['maximum_rule_difference'] for p in parts),
        maximum_log_evidence_error=max(p['maximum_log_evidence_error'] for p in parts),
        new_joint_anchor_rule_evaluations=sum(p['new_joint_anchor_rule_evaluations'] for p in parts),
        all_positive_components_retained=True,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False)
    if not result['precision_passed']:result['state']='precision_hold'
    else:
        anchors={(r['kind'],r['identity'],r['record'],(45,10),tuple(b)):upper
            for p in parts for r in p['rows'] for b,upper in zip(r['budgets'],r['upper'])}
        caps=load_caps(spec['query_cap_plan']);screen.branch_quality_upper=coupled_branch_quality_upper
        screen.collect(spec['followup_bound_plan'],certificate_caps=caps,anchor_caps=anchors)
        new=read(read(spec['followup_bound_plan'])['output']+'/compact_summary.json')
        old=read(read(spec['source_bound_plan'])['output']+'/compact_summary.json');comparison={}
        for kind in ('raw','projected'):
            a=old['weighted_bank_envelope']['kinds'][kind];b=new['weighted_bank_envelope']['kinds'][kind]
            assert abs(a['best_reference_contrast']-b['best_reference_contrast'])<1e-14
            assert b['unresolved_upper_gap']<=a['unresolved_upper_gap']+1e-12
            comparison[kind]=dict(old_gap=a['unresolved_upper_gap'],new_gap=b['unresolved_upper_gap'],
                old_totals=old['totals'][kind],new_totals=new['totals'][kind])
        result.update(state='bound_refined' if new['numerical_containment_passed'] else 'containment_hold',
            numerical_containment_passed=new['numerical_containment_passed'],
            maximum_reference_above_new_bound=new['maximum_validated_reference_above_bound'],comparison=comparison,
            all_conditional_actions_closed=all(v['record_group_candidates']==0 for v in new['totals'].values()),
            original_zero_candidate_gate_unchanged=True)
    save(ROOT/spec['output']/'summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','collect']);p.add_argument('--plan',required=True)
    p.add_argument('--identity',type=int,choices=[4,5]);a=p.parse_args()
    try:
        if a.stage=='run':run(a.plan,a.identity)
        else:collect(a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
