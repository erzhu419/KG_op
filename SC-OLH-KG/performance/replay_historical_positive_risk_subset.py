"""Replay admitted action references on registered positive risk grids."""
import argparse
import json
import resource
import time

import numpy as np

from core.raw_terminal_head_grid import compose_head_grid
from performance.complete_historical_source_heads import ROOT,read,save
from performance.diagnose_historical_component_guarded import compressed_read
from performance.validate_historical_component_bank import action_groups


def run(plan=None, output=None, emit=True):
    started=time.process_time()
    subset=plan is None
    if plan is None:
        status=read('paper_artifacts/historical_positive_risk_continuation_v3/status.json')
        assert status['state']=='conditional_actions_closed'
        plan=status['source_bound_plan']
    bound=read(plan);screen=read(bound['output']+'/summary.json')
    assert screen['numerical_containment_passed']
    closed=all(r['record_group_candidates']==0 for r in screen['totals'].values())
    if subset:assert closed
    base=ROOT/read(bound['bank_plan'])['output'];context=compressed_read(base/'context.json.gz')
    groups=action_groups(context['actions']);action_group={a:g['group'] for g in groups for a in g['action_indices']}
    prior=np.asarray(context['prior_weights']);coverage={(r['record'],r['kind']):r['validated_reference_actions'] for r in screen['records']}
    observations={r['record']:(int(o),i,r) for o,data in context['origins'].items() for i,r in enumerate(data['risk'])}
    tables={};costs={}
    for o,data in context['origins'].items():
        for kind in ('raw','projected'):
            tables[int(o),kind]=np.zeros((len(data['risk']),8,365,4))
            costs[int(o),kind]=np.zeros((len(data['risk']),8,365))

    def scalar(record,h,a,kind,value):
        origin,i,_=observations[record]
        tables[origin,kind][i,h,a]=[value,0.,0.,0.]
        costs[origin,kind][i,h,a]=context['actions'][a]['budgets'][0]

    for h in (4,5):
        for shard in range(len(bound['policy_shards'])):
            for row in read(str(base/'tasks'/f'{h}_{shard}.json'))['rows']:
                if 'fixed' in row:
                    for kind in ('raw','projected'):scalar(row['record'],h,355,kind,row['fixed'][kind])
    for row in read(str(base/'query_references.json'))['rows']:
        for kind in ('raw','projected'):scalar(row['record'],row['identity'],row['action'],kind,row[kind])
    for row in read(read(bound['native_reference_plan'])['output']+'/native_direct_references.json')['rows']:
        scalar(row['record'],row['identity'],row['action'],'projected',row['projected'])
    references=0
    for reference_plan in bound['joint_reference_plans']:
        spec=read(reference_plan);prep=read(spec['output']+'/preparation.json');selected={r['record']:r for r in prep['selected']}
        for check in read(spec['output']+'/compact_summary.json')['groups']:
            assert check['candidate_passed']
            name,kind=check['record'],check['kind']
            if name not in observations:continue
            origin,i,actual=observations[name]
            bundle=compressed_read(ROOT/spec['output']/'marginals'/(name+'.json.gz'))
            assert all(actual[k]==bundle[k] for k in ('label','eta_label','head_log_scores'))
            assert bundle['prior_weights']==prior.tolist() and bundle['replicates']==4
            ix=selected[name]['action_indices']
            for h in (4,5):
                part=read(spec['output']+'/parts/'+name+'/'+str(h)+'_'+kind+'.json')['rows'][check['selected_candidate']]
                tables[origin,kind][i,h,ix]=part['statistics'];costs[origin,kind][i,h,ix]=part['costs']
            references+=1
    rows=[];missing=[];incumbents=[];maximum_omitted=0.;weighted_raw_best=0.
    for o,data in context['origins'].items():
        origin=int(o);grids={}
        for kind in ('raw','projected'):
            values,c,posterior,_=compose_head_grid(np.asarray(data['objective'][kind]),tables[origin,kind],costs[origin,kind],
                data['objective_scores'],[r['head_log_scores'] for r in data['risk']],prior)
            mask=np.zeros((len(data['risk']),365),dtype=bool)
            for i,r in enumerate(data['risk']):mask[i,coverage[r['record'],kind]]=True
            grids[kind]=(np.where(mask[None],values,-np.inf),c,posterior)
        raw,rc,posterior=grids['raw'];native,nc,np_=grids['projected'];np.testing.assert_array_equal(posterior,np_)
        maximum_omitted=max(maximum_omitted,float(posterior[:,:,[0,1,2,3,6,7]].sum(axis=-1).max()))
        choices=np.lexsort((nc,-native),axis=-1)[:,:,0];raw_choices=np.lexsort((rc,-raw),axis=-1)[:,:,0]
        achieved=np.take_along_axis(raw,choices[:,:,None],axis=2)[:,:,0]
        optimum=raw.max(axis=-1)
        for i,r in enumerate(data['risk']):
            absent=sorted(set(map(int,choices[:,i]))-set(coverage[r['record'],'raw']))
            unique,counts=np.unique(choices[:,i],return_counts=True)
            weight=float(prior[origin]*r['outer_weight'])
            native_gain=float(np.mean(native[np.arange(native.shape[0]),i,choices[:,i]]-native[:,i,355]))
            best=float(np.mean(optimum[:,i]-raw[:,i,355]));weighted_raw_best+=weight*best
            incumbent=dict(record=r['record'],origin=origin,cdf=r['cdf'],mixture_weight=weight,
                native_action_counts={str(a):int(n) for a,n in zip(unique,counts)},missing_raw_actions=absent,
                missing_raw_groups=sorted({action_group[a] for a in absent}),
                mean_projected_choice_contrast=native_gain,weighted_projected_choice_contrast=weight*native_gain)
            incumbent['missing_raw_group_projected_contrasts']={}
            for gid in incumbent['missing_raw_groups']:
                ix=np.flatnonzero(np.isin(choices[:,i],groups[gid]['action_indices']))
                incumbent['missing_raw_group_projected_contrasts'][str(gid)]=float(weight*np.sum(
                    native[ix,i,choices[ix,i]]-native[ix,i,355])/native.shape[0])
            incumbents.append(incumbent)
            if absent:
                missing.append(dict(incumbent,actions=absent));continue
            gain=float(np.mean(achieved[:,i]-raw[:,i,355]))
            raw_unique,raw_counts=np.unique(raw_choices[:,i],return_counts=True)
            rows.append(dict(record=r['record'],origin=origin,cdf=r['cdf'],outer_weight=r['outer_weight'],mixture_weight=weight,
                native_action_counts={str(a):int(n) for a,n in zip(unique,counts)},raw_actions=sorted(set(map(int,raw_choices[:,i]))),
                raw_action_counts={str(a):int(n) for a,n in zip(raw_unique,raw_counts)},
                mean_fixed_reward=float(raw[:,i,355].mean()),mean_raw_reward_of_native_choice=float(achieved[:,i].mean()),
                mean_native_choice_contrast=gain,mean_raw_optimal_contrast=best,
                mean_action_selection_loss=float(np.mean(optimum[:,i]-achieved[:,i])),weighted_subset_contrast=weight*gain))
            if not closed:
                rows[-1]['mean_raw_best_admitted_contrast']=rows[-1].pop('mean_raw_optimal_contrast')
    result=dict(complete=True,source_bound_plan=plan,observations=len(observations),rows=rows,
        native_incumbents=incumbents,all_conditional_actions_closed=closed,
        weighted_raw_best_admitted_contrast=weighted_raw_best,
        weighted_projected_incumbent_contrast=sum(r['weighted_projected_choice_contrast'] for r in incumbents),
        native_choices_all_raw_referenced=not missing,missing_raw_references=missing,admitted_joint_reference_groups=references,
        retained_subset_probability_mass=sum(r['mixture_weight'] for r in rows),
        raw_covered_grid_weight=sum(r['mixture_weight'] for r in rows),
        weighted_subset_native_choice_contrast=sum(r['weighted_subset_contrast'] for r in rows) if not missing else None,
        weighted_raw_reward_of_native_incumbent_contrast=sum(r['weighted_subset_contrast'] for r in rows) if not missing else None,
        maximum_omitted_utility_identity_mass=maximum_omitted,source_components={'4':512,'5':1024},
        action_count=365,objective_heads_per_origin=16,cpu_seconds=time.process_time()-started,
        charged_to_final_phase_cached_analysis_reserve=True,new_integrations=0,new_physical_calls=0,new_source_fits=0,
        new_verifier_calls=0,full_posterior_restorations=0,full_predictive_integration_admitted=False,independent_confirmation=False,
        scope=('Eight switch-bracketing positive-grid observations with their original weights and16objective heads. Unselected outer coordinates are absent;the subset contribution is not total KG or a deployment gain.' if output is None else
            'Registered positive risk grid with16objective heads perorigin. Choices maximize admitted native references;unresolved action uppers andcontinuous outer quadrature error remain separate. Raw reward is reported only where every chosen action has an admitted raw reference.'))
    path=ROOT/(output or 'paper_artifacts/historical_positive_risk_continuation_v3/subset_replay.json')
    path.parent.mkdir(parents=True,exist_ok=True);save(path,result)
    if emit:print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');a=p.parse_args()
    run(a.plan,a.output)
    print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
