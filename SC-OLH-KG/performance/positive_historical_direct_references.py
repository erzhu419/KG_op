"""Price the best native direct alternatives under the original raw law."""
import argparse
import json
import resource
import signal
import time

import numpy as np
from scipy.special import logsumexp

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, compressed_write
from performance.validate_historical_component_bank import action_groups, group_checks, RULES

PLAN='performance/manifests/historical_positive_direct_references_v1.json'


def prepare(plan=PLAN):
    started=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    parent=read(spec['screen_plan']);screen=read(parent['output']+'/compact_summary.json')
    assert screen['numerical_containment_passed'] and screen['native_direct_reference_cpu_seconds']>0
    context=compressed_read(ROOT/parent['output']/'context.json.gz');groups=action_groups(context['actions'])
    native=read(parent['output']+'/native_direct_references.json')
    refs={(r['identity'],r['record'],r['action']):r['projected'] for r in native['rows']}
    library=read(spec['pricing_library_plan']);measured=[]
    for source in library['joint_reference_plans']:
        prior=read(source)
        for row in read(prior['output']+'/preparation.json')['selected']:
            if groups[row['group']]['kind']!='direct':continue
            for h in (4,5):
                path=ROOT/prior['output']/'parts'/row['record']/f'{h}_raw.json'
                if path.exists():measured.append(dict(group=row['group'],identity=h,rows=read(str(path))['rows']))
    jobs=[];selected=[];total=screen['cumulative_estimated_cpu_seconds']
    for origin,data in context['origins'].items():
        for observation in data['risk']:
            scores=np.log(context['prior_weights'])+np.asarray(data['objective_scores'])+observation['head_log_scores']
            posterior=np.exp(scores-logsumexp(scores,axis=-1)[:,None]);quality=np.asarray(data['objective']['projected'])
            direct=sorted({a for h,r,a in refs if r==observation['record']})
            values=np.asarray([sum(posterior[:,h]*quality[:,h,a,0]*refs[h,observation['record'],a]
                for h in (4,5)) for a in direct])
            action=direct[int(np.argmax(values.mean(axis=1)))];group=next(g for g in groups if action in g['action_indices'])
            gid=group['group'];ix=group['action_indices'];shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals'/f'{gid:03d}.json.gz')
            assert shared['prior_weights']==context['prior_weights'] and shared['points'][0]==context['query']
            points=shared['points'];record=observation['record'];bundle=dict(observation,name=record,
                query=context['query'],replicates=4,prior_weights=context['prior_weights'],utility_identities=[4,5],
                protocol=context['protocol'],seed=spec['seed'],group=gid,points=points,fields=shared['fields'],
                action_indices=ix,actions=[dict(context['actions'][a],primary=points.index(context['actions'][a]['primary']),
                    shortlist=[points.index(p) for p in context['actions'][a]['shortlist']]) for a in ix],
                opposite=[dict(position=i,head_log_scores=score,**{kind:np.asarray(data['objective'][kind])[i][:,ix].tolist()
                    for kind in ('raw','projected')}) for i,score in enumerate(data['objective_scores'])])
            compressed_write(out/'marginals'/(record+'.json.gz'),bundle)
            selected.append(dict(record=record,group=gid,origin=int(origin),position=observation['position'],
                action_indices=ix,selected_native_direct_action=action,mean_native_direct_reward=float(values.mean(axis=1).max())))
            for h in (4,5):
                old=read(f'paper_artifacts/historical_component_bank_v1/parts/{gid:03d}_{h}_raw.json')['rows'];units={}
                for rule in spec['rules']:
                    size=RULES[rule][:2]
                    times=[v['cpu_seconds'] for r in measured if r['group']==gid and r['identity']==h
                        for key,v in r['rows'].items() if key in RULES and RULES[key][:2]==size]
                    scaled=[v['cpu_seconds']*2**(size[1]-RULES[key][1]) for key,v in old.items()
                        if key in RULES and RULES[key][0]==size[0] and RULES[key][1]<=size[1]]
                    units[rule]=max(times+scaled)
                estimate=1.2*sum(units.values())+2;total+=estimate
                jobs.append(dict(record=record,identity=h,kind='raw',shard=len(selected)-1,
                    matched_rule_cpu_seconds=units,estimated_cpu_seconds=estimate))
    assert all(j['estimated_cpu_seconds']<=spec['maximum_worker_cpu_seconds'] for j in jobs)
    spec['refined_cases']=[dict(record=r['record'],kind='raw',candidate='E8') for r in selected]
    save(ROOT/plan,spec)
    result=dict(complete=True,selected=selected,jobs=jobs,source_components={'4':512,'5':1024},
        estimated_cpu_seconds_including_reserves=total,expected_cpu_seconds_including600reserves=total,
        estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds'],parent_screen_phase_estimated_cpu_seconds=screen['cumulative_estimated_cpu_seconds'],
        original_screen600reserve_retained=True,cpu_seconds=time.process_time()-started,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,full_posterior_restorations=0,independent_confirmation=False)
    save(out/'preparation.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def collect(plan=PLAN):
    spec=read(plan);prep=read(spec['output']+'/preparation.json');checks=[]
    for row in prep['selected']:
        bundle=compressed_read(ROOT/spec['output']/'marginals'/(row['record']+'.json.gz'))
        parts=[read(spec['output']+'/parts/'+row['record']+f'/{h}_raw.json') for h in (4,5)]
        assert all(p['complete'] for p in parts)
        check=group_checks(bundle,parts,'raw',[('E8','F8'),('E8','H8'),('E8','E9')],1e-4)
        error=max(max(m['raw64_128_difference'],m['maximum_log_evidence_error']) for p in parts for m in p['marginal_bounds'])
        error=max(error,max(p['maximum_branch_above_marginal_bound'] for p in parts))
        checks.append(dict(record=row['record'],kind='raw',group=row['group'],selected_candidate='E8',
            checks=check,maximum_branch_and_marginal_error=error,candidate_passed=check['passed'] and error<=1e-4))
    result=dict(complete=True,groups=checks,local_risk_candidate_passed=all(r['candidate_passed'] for r in checks),
        estimated_cpu_seconds_including_reserves=prep['estimated_cpu_seconds_including_reserves'],
        source_prior_preserved=True,action_count=365,remaining_calls=356,new_physical_calls=0,
        new_source_fits=0,new_verifier_calls=0,full_posterior_restorations=0,independent_confirmation=False,
        scope='Raw prices for the native direct alternatives at eight switch-bracketing coordinates;no adaptive outer KG estimate.')
    save(ROOT/spec['output']/'compact_summary.json',result);print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect']);p.add_argument('--plan',default=PLAN)
    a=p.parse_args();prepare(a.plan) if a.stage=='prepare' else collect(a.plan)
    print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
