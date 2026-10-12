"""Price the best native direct alternatives under the original raw law."""
import argparse
import copy
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


def prepare_incumbents(plan):
    """Price only raw references missing from the admitted native choices."""
    from performance.replay_historical_positive_risk_subset import run as replay
    from performance.screen_historical_observed_batch_risk import reference_rule_timers,price_reference_group

    started=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    previous=read(spec['previous_phase_completion']);assert previous['state']=='phase_complete'
    bound=read(spec['incumbent_bound_plan'])
    assert all(read(read(p)['output']+'/compact_summary.json')['local_risk_candidate_passed'] for p in bound['joint_reference_plans'])
    diagnostic=replay(spec['incumbent_bound_plan'],spec['output']+'/incumbent_replay.json',emit=False)
    context=compressed_read(ROOT/read(bound['bank_plan'])['output']/'context.json.gz')
    groups=action_groups(context['actions']);measured,fine=reference_rule_timers(bound)
    candidates=[dict(record=row['record'],group=gid,
        actions=[a for a in row['actions'] if a in groups[gid]['action_indices']],
        weighted_projected_choice_contrast=row['missing_raw_group_projected_contrasts'][str(gid)])
        for row in diagnostic['missing_raw_references'] for gid in row['missing_raw_groups']]
    candidates.sort(key=lambda r:(-r['weighted_projected_choice_contrast'],r['record'],r['group']))
    cases=[];jobs=[]
    phase_base=read(spec['continued_phase_cost_plan'])['estimated_cpu_seconds_including_reserves'] if 'continued_phase_cost_plan' in spec else 600.
    total=phase_base
    selected_records=set()
    for row in candidates:
        if row['record'] in selected_records:continue
        case,priced=price_reference_group(dict(record=row['record'],kind='raw',group=row['group'],
            missing_raw_actions=row['actions'],weighted_projected_choice_contrast=row['weighted_projected_choice_contrast']),groups,measured,fine)
        assert all(j['estimated_cpu_seconds']<=spec['maximum_worker_cpu_seconds'] for j in priced)
        estimate=sum(j['estimated_cpu_seconds'] for j in priced)
        if total+estimate>spec['maximum_estimated_cpu_seconds']:break
        cases.append(case);jobs.extend(priced);total+=estimate;selected_records.add(row['record'])
    frozen=dict(complete=True,source=spec['output']+'/incumbent_replay.json',next_joint_cases=cases,next_joint_jobs=jobs,
        phase_base_estimated_cpu_seconds=phase_base,expected_cpu_seconds_including600reserves=total,
        cost_admitted=bool(jobs) and total<=spec['maximum_estimated_cpu_seconds'],
        missing_native_incumbent_observations=len(diagnostic['missing_raw_references']),selected_observations=len(cases),
        remaining_missing_observations=sorted({r['record'] for r in candidates
            if (r['record'],r['group']) not in {(c['record'],c['group']) for c in cases}}),
        previous_closed_phase_completion=spec['previous_phase_completion'],
        previous_phase_estimated_cpu_seconds=previous['estimated_phase_cpu_seconds'],
        maximum_live_workers=8,priority='low',cpu_seconds=time.process_time()-started,
        scope='Raw references for previously unchecked native incumbents on the registered positive grid. The registered phase base andits one600CPU second reserve are retained;all prior completed costs remain in the campaign ledger. No failed reference is moved to a new phase.')
    save(ROOT/spec['frozen_cost_plan'],frozen)
    assert frozen['cost_admitted']
    spec['refined_cases']=[dict(record=r['record'],kind=r['kind'],candidate=r['candidate'],rules=r['rules']) for r in cases]
    save(ROOT/plan,spec);prep=prepare_frozen(plan)
    result=dict(complete=True,plan=plan,manifest=spec,frozen_cost_plan=frozen,preparation=prep)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def prepare_next(plan):
    """Start a new phase only after the previous frozen comparisons passed."""
    from performance.screen_historical_observed_batch_risk import next_plan

    current=read(plan);closed=read(current['output']+'/compact_summary.json')
    previous=current['joint_reference_plans'][-1];risk=read(previous)
    comparisons=read(risk['output']+'/compact_summary.json')
    assert closed['numerical_containment_passed'] and comparisons['local_risk_candidate_passed']
    full_grid='risk_mean_order' in current
    prefix='historical_positive_risk_full_grid' if full_grid else 'historical_positive_risk'
    stem=(ROOT/current['phase_completion']).parent.name if full_grid else (ROOT/previous).stem
    number=1 if full_grid and not stem.startswith(prefix+'_competition_v') else int(stem.rsplit('_v',1)[1])+1
    completion=read(current['phase_completion']) if 'phase_completion' in current else None
    if completion:assert completion['state']=='phase_complete' and completion['all_reference_precision_passed']
    pricing_path=f'performance/manifests/{prefix}_pricing_v{number}.json'
    pricing=copy.deepcopy(current)
    pricing.update(output=f'paper_artifacts/{prefix}_pricing_v{number}',
        closed_previous_phase_plan=previous,
        pricing='New,previously unchecked competitive groups after the prior phase passed. New600reserve;all prior phase costs remain in the campaign ledger. No failed comparison is moved to this phase.')
    pricing.pop('continued_phase_cost_plan',None)
    save(ROOT/pricing_path,pricing)
    pricing_out=ROOT/pricing['output'];pricing_out.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','compact_summary.json'):
        (pricing_out/name).write_bytes((ROOT/current['output']/name).read_bytes())
    priced=next_plan(pricing_path)
    if completion:
        priced.update(previous_closed_phase_completion=current['phase_completion'],
            previous_phase_estimated_cpu_seconds=completion['estimated_phase_cpu_seconds'])
        save(pricing_out/'next_cost_plan.json',priced)
    if not priced['cost_admitted']:
        result=dict(complete=True,prepared=False,pricing_plan=pricing_path,pricing_manifest=pricing,
            priced=priced,state='conditional_actions_closed' if not any(priced['remaining_groups'].values()) else 'cost_hold')
    else:
        risk_path=f'performance/manifests/{prefix}_competition_v{number}.json'
        bound_path=f'performance/manifests/{prefix}_screen_competition_v{number}.json'
        risk.pop('failed_preparation_account',None)
        risk.pop('worker_limit_adjustment',None)
        risk.update(output=f'paper_artifacts/{prefix}_competition_v{number}',
            frozen_cost_plan=pricing['output']+'/next_cost_plan.json',followup_bound_plan=bound_path,
            previous_closed_phase_plan=previous,
            refined_cases=[dict(record=r['record'],kind=r['kind'],candidate=r['candidate'],rules=r['rules'])
                for r in priced['next_joint_cases']],
            pricing=pricing['pricing'])
        bound=copy.deepcopy(current)
        bound.update(output=f'paper_artifacts/{prefix}_screen_competition_v{number}',
            joint_reference_plans=current['joint_reference_plans']+[risk_path],
            pricing_reference_plans=current['pricing_reference_plans']+[risk_path],
            reference_phase_cost_plan=risk['output']+'/preparation.json',
            continued_phase_cost_plan=risk['output']+'/preparation.json')
        if completion:bound['phase_completion']=risk['output']+'/completion.json'
        save(ROOT/risk_path,risk);save(ROOT/bound_path,bound);prepared=prepare_frozen(risk_path)
        result=dict(complete=True,prepared=True,pricing_plan=pricing_path,pricing_manifest=pricing,priced=priced,
            risk_plan=risk_path,risk_manifest=risk,bound_plan=bound_path,bound_manifest=bound,preparation=prepared)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def prepare_continued(plan):
    """Reuse the historical refs layout inside the current phase budget."""
    current=read(plan);screen=read(current['output']+'/compact_summary.json')
    previous=current['joint_reference_plans'][-1];risk=copy.deepcopy(read(previous))
    assert screen['numerical_containment_passed']
    assert read(risk['output']+'/compact_summary.json')['local_risk_candidate_passed']
    frozen=read(current['output']+'/next_cost_plan.json');assert frozen['cost_admitted']
    assert frozen['phase_base_estimated_cpu_seconds']==read(current['continued_phase_cost_plan'])['estimated_cpu_seconds_including_reserves']
    initial=(ROOT/current['phase_completion']).parent.name
    initial_plan='performance/manifests/'+initial+'.json'
    number=len(current['joint_reference_plans'])-current['joint_reference_plans'].index(initial_plan)
    risk_path=f'performance/manifests/{initial}_refs_v{number}.json'
    bound_name=initial.replace('_competition_','_screen_competition_')+f'_refs_v{number}'
    bound_path=f'performance/manifests/{bound_name}.json'
    risk.update(output=f'paper_artifacts/{initial}_refs_v{number}',
        frozen_cost_plan=current['output']+'/next_cost_plan.json',followup_bound_plan=bound_path,
        refined_cases=[dict(record=r['record'],kind=r['kind'],candidate=r['candidate'],rules=r['rules'])
            for r in frozen['next_joint_cases']])
    bound=copy.deepcopy(current)
    bound.update(output=f'paper_artifacts/{bound_name}',
        joint_reference_plans=current['joint_reference_plans']+[risk_path],
        pricing_reference_plans=current['pricing_reference_plans']+[risk_path],
        reference_phase_cost_plan=risk['output']+'/preparation.json',
        continued_phase_cost_plan=risk['output']+'/preparation.json')
    save(ROOT/risk_path,risk);save(ROOT/bound_path,bound);prepared=prepare_frozen(risk_path)
    result=dict(complete=True,prepared=True,pricing_plan=plan,pricing_manifest=current,priced=frozen,
        risk_plan=risk_path,risk_manifest=risk,bound_plan=bound_path,bound_manifest=bound,preparation=prepared)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


def prepare_frozen(plan):
    """Reuse registered positive-grid observations for frozen joint references."""
    from performance.refine_historical_observed_prediction import field_from_snapshot
    from performance.screen_historical_observed_batch_risk import condition_risk

    started=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,120.)
    frozen=read(spec['frozen_cost_plan']);assert frozen['cost_admitted']
    parent=read(spec['screen_plan']);context=compressed_read(ROOT/parent['output']/'context.json.gz')
    assert len(context['actions'])==365 and context['remaining_calls']==356
    groups=action_groups(context['actions']);selected=[];maximum_score_error=0.
    observations={r['record']:(int(o),data,r) for o,data in context['origins'].items() for r in data['risk']}
    for record in dict.fromkeys(r['record'] for r in frozen['next_joint_cases']):
        origin,data,row=observations[record]
        gids={r['group'] for r in frozen['next_joint_cases'] if r['record']==record};assert len(gids)==1
        gid=gids.pop();group=groups[gid];ix=group['action_indices']
        shared=compressed_read(ROOT/'paper_artifacts/historical_component_bank_v1/marginals'/f'{gid:03d}.json.gz')
        assert shared['prior_weights']==context['prior_weights'] and shared['points'][0]==context['query']
        assert len(row['head_log_scores'])==8
        points=shared['points'];bundle=dict(row,name=record,origin=origin,query=context['query'],replicates=4,
            prior_weights=context['prior_weights'],utility_identities=[4,5],protocol=context['protocol'],
            seed=spec['seed'],group=gid,kind=group['kind'],points=points,fields=shared['fields'],action_indices=ix,
            actions=[dict(context['actions'][a],primary=points.index(context['actions'][a]['primary']),
                shortlist=[points.index(p) for p in context['actions'][a]['shortlist']]) for a in ix],
            opposite=[dict(position=i,head_log_scores=score,**{kind:np.asarray(data['objective'][kind])[i][:,ix].tolist()
                for kind in ('raw','projected')}) for i,score in enumerate(data['objective_scores'])])
        assert len(bundle['opposite'])==16 and all(sum(a['budgets'])<=356 for a in bundle['actions'])
        for h in (4,5):
            field=field_from_snapshot(bundle['fields'][h]);score=condition_risk(field,row,h,context)
            maximum_score_error=max(maximum_score_error,abs(score-row['head_log_scores'][h]))
        assert maximum_score_error<=1e-10
        compressed_write(out/'marginals'/(record+'.json.gz'),bundle)
        selected.append(dict(record=record,origin=origin,position=row['position'],group=gid,action_indices=ix,
            label=row['label'],eta_label=row['eta_label'],cdf=row['cdf'],outer_weight=row['outer_weight']))
    jobs=[dict(r,shard=next(i for i,s in enumerate(selected) if s['record']==r['record'])) for r in frozen['next_joint_jobs']]
    total=frozen['phase_base_estimated_cpu_seconds']+sum(j['estimated_cpu_seconds'] for j in jobs)
    assert abs(total-frozen['expected_cpu_seconds_including600reserves'])<=1e-8
    assert all(j['estimated_cpu_seconds']<=spec['maximum_worker_cpu_seconds'] for j in jobs)
    result=dict(complete=True,selected=selected,jobs=jobs,action_count=365,remaining_calls=356,
        source_components={str(h):len(shared['fields'][h]['means']) for h in (4,5)},
        maximum_head_score_error=maximum_score_error,all_identity_likelihoods=8,
        estimated_cpu_seconds_including_reserves=total,expected_cpu_seconds_including600reserves=total,
        parent_screen_phase_estimated_cpu_seconds=frozen['phase_base_estimated_cpu_seconds'],
        original_screen600reserve_retained=True,estimated_cost_admitted=total<=spec['maximum_estimated_cpu_seconds'],
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        full_posterior_restorations=0,independent_confirmation=False)
    save(out/'preparation.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True);return result


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
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','prepare-frozen','prepare-next','prepare-continued','prepare-incumbents','collect']);p.add_argument('--plan',default=PLAN)
    a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.plan)
        elif a.stage=='prepare-frozen':prepare_frozen(a.plan)
        elif a.stage=='prepare-next':prepare_next(a.plan)
        elif a.stage=='prepare-continued':prepare_continued(a.plan)
        elif a.stage=='prepare-incumbents':prepare_incumbents(a.plan)
        else:collect(a.plan)
    finally:
        print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
