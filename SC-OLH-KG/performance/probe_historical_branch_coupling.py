"""Compare coupled branch bounds on a cached bank without new integrations."""
import argparse
import json
import resource
import time

from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from performance.complete_historical_source_heads import ROOT, read, save
from performance import screen_historical_observed_batch_risk as screen


def run(plan):
    spec=read(plan);source=read(spec['source_bound_plan'])
    assert spec['output']!=source['output']
    old=read(source['output']+'/compact_summary.json')
    assert old['numerical_containment_passed']
    reference=read(source['joint_reference_plans'][-1])
    assert read(reference['output']+'/compact_summary.json')['local_risk_candidate_passed']
    bank=ROOT/read(source['bank_plan'])['output']
    native=ROOT/read(source['native_reference_plan'])['output']
    assert (bank/'query_references.json').exists() and (native/'native_direct_references.json').exists()
    screen.branch_quality_upper=coupled_branch_quality_upper
    started=time.process_time();screen.collect(plan)
    new=read(spec['output']+'/compact_summary.json')
    comparison={}
    for kind in ('raw','projected'):
        before=old['weighted_bank_envelope']['kinds'][kind]
        after=new['weighted_bank_envelope']['kinds'][kind]
        assert abs(before['best_reference_contrast']-after['best_reference_contrast'])<1e-14
        assert after['unresolved_upper_gap']<=before['unresolved_upper_gap']+1e-12
        comparison[kind]=dict(old_gap=before['unresolved_upper_gap'],new_gap=after['unresolved_upper_gap'],
            reduction=1-after['unresolved_upper_gap']/before['unresolved_upper_gap'],
            old_totals=old['totals'][kind],new_totals=new['totals'][kind])
    result=dict(source_bound_plan=spec['source_bound_plan'],pilot_plan=plan,
        numerical_containment_passed=new['numerical_containment_passed'],
        maximum_reference_above_new_bound=new['maximum_validated_reference_above_bound'],
        best_reference_contrasts_preserved=True,comparison=comparison,analysis_cpu_seconds=time.process_time()-started,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,new_joint_reference_integrations=0,
        independent_confirmation=False,original_zero_candidate_gate_unchanged=True,
        source_phase_reserve=spec['analysis_reserve_admission'],
        scope='Diagnostic branch-event relaxation only; original posterior, cached references and verifier retained.')
    save(ROOT/spec['output']/'comparison_summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    try:run(args.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
