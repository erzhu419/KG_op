"""Replay finite n4 outer rules using only admitted action references."""
import argparse
import json
import resource
import signal
import time

import numpy as np
from scipy.stats import qmc

from core.raw_terminal_head_grid import compose_head_grid, summarize_achievable_grid
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.refine_historical_achievable_precision import assemble_case, rule_pair

PLAN='performance/manifests/historical_observed_batch_assembly_v1.json'


def run(plan=PLAN):
    started=time.process_time();spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,spec['maximum_cpu_seconds'])
    status=read(spec['campaign_output']+'/status.json');assert status['state']=='conditional_actions_closed'
    bound=read(status['source_bound_plan']);screen=read(bound['output']+'/summary.json')
    assert screen['numerical_containment_passed'] and all(t['record_group_candidates']==0 for t in screen['totals'].values())
    base=ROOT/read(bound['bank_plan'])['output'];context=compressed_read(base/'context.json.gz')
    pilot=read(bound['pilot_plan']);parent=read(pilot['parent_plan']);precision=read(parent['precision_plan'])
    case=next(c for c in precision['cases'] if c['name']=='source_n4')
    heads,bank,_=assemble_case(precision,case,read(precision['output']+'/registered_jobs.json')['jobs'])
    prior=np.asarray(bank['prior_weights']);assert prior.tolist()==context['prior_weights']
    source=parent['accepted_baseline_inputs'][case['state']]
    debited=read(source+'/'+case['state']+'/n4/baseline/terminal_rules.json')['rows'][-1]
    stop=read(parent['stop_inputs'][case['state']])['rows'][-1];fixed=debited['best_action_index']
    assert context['actions']==debited['actions'] and len(context['actions'])==365 and fixed==355
    coverage={(r['record'],r['kind']):r['validated_reference_actions'] for r in screen['records']}
    risk={f'source_n4_o{o}_h1_p{r["position"]}':(o,i,r) for o in bank['origins'] for i,r in enumerate(heads[1,o])}
    tables={};costs={}
    for origin in bank['origins']:
        for kind in ('raw','projected'):
            tables[origin,kind]=np.asarray([r[kind][str(rule_pair(r,kind)[-1])]['statistics'] for r in heads[1,origin]])
            costs[origin,kind]=np.asarray([r[kind][str(rule_pair(r,kind)[-1])]['costs'] for r in heads[1,origin]])

    def scalar(record,h,action,kind,value):
        origin,i,_=risk[record];tables[origin,kind][i,h,action]=[value,0.,0.,0.]
        costs[origin,kind][i,h,action]=context['actions'][action]['budgets'][0]

    for h in (4,5):
        for shard in range(len(bound['policy_shards'])):
            for row in read(str(base/'tasks'/f'{h}_{shard}.json'))['rows']:
                if 'fixed' in row:
                    for kind in ('raw','projected'):scalar(row['record'],h,fixed,kind,row['fixed'][kind])
    for row in read(str(base/'query_references.json'))['rows']:
        for kind in ('raw','projected'):scalar(row['record'],row['identity'],row['action'],kind,row[kind])
    native=read(read(bound['native_reference_plan'])['output']+'/native_direct_references.json')
    for row in native['rows']:scalar(row['record'],row['identity'],row['action'],'projected',row['projected'])
    reference_groups=0
    for reference in bound['joint_reference_plans']:
        registered=read(reference);prep=read(registered['output']+'/preparation.json')
        selected={r['record']:r for r in prep['selected']}
        for check in read(registered['output']+'/compact_summary.json')['groups']:
            if not check['candidate_passed']:continue
            name,kind=check['record'],check['kind'];origin,i,actual=risk[name]
            bundle=compressed_read(ROOT/registered['output']/'marginals'/(name+'.json.gz'))
            assert all(actual[k]==bundle[k] for k in ('label','eta_label','head_log_scores'))
            assert bundle['replicates']==4 and bundle['prior_weights']==prior.tolist()
            ix=selected[name]['action_indices']
            for h in (4,5):
                row=read(registered['output']+'/parts/'+name+'/'+str(h)+'_'+kind+'.json')['rows'][check['selected_candidate']]
                tables[origin,kind][i,h,ix]=row['statistics'];costs[origin,kind][i,h,ix]=row['costs']
            reference_groups+=1
    totals={(8,16):[],(16,8):[],(16,16):[]};missing=[];recommendations=[];attribution=[]
    fixed_plan=read('performance/manifests/historical_observed_prediction_v3.json')
    fixed_state=read(fixed_plan['output']+'/state.json')
    coordinates=qmc.Sobol(4,scramble=True,seed=parent['predictive_seed']).random_base2(4)[:,2:4]
    cached_fixed={};fixed_parity=0.
    for origin in bank['origins']:
        for head in (0,1):
            directory=fixed_plan['output']+'/n4/'+str(origin)+'/'+str(head)
            cached=read(directory+'/results.json');observations=read(directory+'/bank.json')
            assert cached['complete'] and cached['all_inner_rules_passed'] and cached['predictive_cdf_passed']
            assert len(cached['values'])==2048
            cached_fixed[origin,head]=cached
            for i,row in enumerate(heads[head,origin]):
                fixed_parity=max(fixed_parity,abs(row['label']-observations['labels'][i]),
                    abs(row['eta_label']-observations['etas'][i]))
    assert fixed_state['prior_weights']==prior.tolist() and fixed_parity<=1e-8
    for origin in bank['origins']:
        f,g=heads[0,origin],heads[1,origin];assert len(f)==len(g)==16
        masks={kind:np.zeros((16,365),dtype=bool) for kind in ('raw','projected')}
        for i,row in enumerate(g):
            name=f'source_n4_o{origin}_h1_p{row["position"]}'
            for kind in masks:masks[kind][i,coverage[name,kind]]=True
        grids={}
        for kind in masks:
            quality=np.asarray([r[kind][str(rule_pair(r,kind)[-1])]['statistics'] for r in f])
            values,c,posterior,_=compose_head_grid(quality,tables[origin,kind],costs[origin,kind],
                [r['head_log_scores'] for r in f],[r['head_log_scores'] for r in g],prior)
            grids[kind]=(np.where(masks[kind][None],values,-np.inf),c,posterior)
        raw,rc,posterior=grids['raw'];native,nc,np_=grids['projected'];np.testing.assert_array_equal(posterior,np_)
        choices=np.lexsort((nc,-native),axis=-1)[:,:,0]
        achieved=np.take_along_axis(raw,choices[:,:,None],axis=2)[:,:,0]
        for j,row in enumerate(g):
            absent=sorted(set(map(int,choices[:,j]))-set(coverage[f'source_n4_o{origin}_h1_p{row["position"]}','raw']))
            if absent:missing.append(dict(origin=origin,risk_position=row['position'],native_choices_missing_raw_references=absent))
            recommendations.append(dict(origin=origin,risk_position=row['position'],native_actions=sorted(set(map(int,choices[:,j]))),
                raw_actions=sorted(set(map(int,np.lexsort((rc,-raw),axis=-1)[:,j,0])))))
            if not absent:
                gain=float(np.mean(achieved[:,j]-raw[:,j,fixed]))
                attribution.append(dict(origin=origin,risk_position=row['position'],
                    mean_cdf=float(coordinates[j,0]),variance_cdf=float(coordinates[j,1]),
                    label=row['label'],eta_label=row['eta_label'],mean_achieved_contrast=gain,
                    weighted_achieved_contrast=float(prior[origin]*gain/16),
                    mean_fixed_reward=float(np.mean(raw[:,j,fixed])),
                    cached_origin_fixed_risk=cached_fixed[origin,1]['values'][j],
                    native_actions=sorted(set(map(int,choices[:,j])))))
        if missing:continue
        for cfg in totals:
            nf,ng=cfg;weights=np.full(cfg,prior[origin]/(nf*ng));ix=np.s_[:nf,:ng]
            row=summarize_achievable_grid(raw[ix],rc[ix],native[ix],nc[ix],posterior[ix],weights,
                (prior[origin]*stop['best_reward'],prior[origin]*debited['best_reward'],fixed),'admitted_references',nf+ng)
            omitted=sorted(set(range(len(prior)))-set(bank['utility_identities']))
            row['weighted_omitted_identity_mass']=float(np.sum(weights*posterior[ix][:,:,omitted].sum(axis=-1)))
            totals[cfg].append(row)
    keys=['raw_adaptation_value','raw_net_reward_control_variate','raw_adaptation_of_projected_choice',
        'raw_net_reward_of_projected_choice_control_variate','action_selection_loss','raw_fixed_action_tower_residual',
        'quadrature_joint_probability_mass','weighted_omitted_identity_mass']
    rare=float(sum(w for h,w in enumerate(prior) if h not in bank['origins']));rows=[]
    if not missing:
        for cfg,parts in totals.items():
            row=dict(objective_prefix=cfg[0],risk_prefix=cfg[1],**{k:sum(r[k] for r in parts) for k in keys})
            for k in ('raw_net_reward_control_variate','raw_net_reward_of_projected_choice_control_variate'):
                row[k]+=rare*(debited['best_reward']-stop['best_reward'])
            rows.append(row)
    field='raw_net_reward_of_projected_choice_control_variate';changes={axis:abs(rows[-1][field]-r[field])
        for axis,r in zip(('objective8_16','risk8_16'),rows[:-1])} if rows else {}
    passed=bool(rows and max(changes.values())<=1e-4 and abs(rows[-1]['raw_fixed_action_tower_residual'])<=1e-4)
    fixed_prefixes=[]
    for p in (8,16,64,512,1024,2048):
        reward=sum(prior[o]*np.mean(cached_fixed[o,0]['values'][:p])*np.mean(cached_fixed[o,1]['values'][:p])
            for o in bank['origins'])
        fixed_prefixes.append(dict(prefix=p,expected_reward=float(reward),tower_residual=float(reward-debited['best_reward'])))
    result=dict(complete=True,source_bound_plan=status['source_bound_plan'],rows=rows,outer_changes=changes,
        finite_outer_diagnostics_passed=passed,native_choices_all_raw_referenced=not missing,missing_achieved_references=missing,
        recommendations=recommendations,risk_outer_attribution=attribution,
        cached_fixed_action_diagnostic=dict(plan='performance/manifests/historical_observed_prediction_v3.json',
            matched_observation_maximum_difference=fixed_parity,prefixes=fixed_prefixes,
            fixed_action_only=True,adaptive_outer_integral_admitted=False),
        reference_groups=reference_groups,action_count=365,query=63,replicates=4,remaining_calls=356,
        omitted_predictive_origin_mass=rare,stop_reward=stop['best_reward'],debited_reward=debited['best_reward'],
        cpu_seconds=time.process_time()-started,estimated_cpu_seconds_including600reserves=600,
        new_integrations=0,full_posterior_restorations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        full_predictive_integration_admitted=False,independent_confirmation=False,
        scope='Finite8/16 n4 outer replay at query63 using admitted references and marginal exclusions. Native choices are priced only if raw references exist;original8 identity scores and512/1024 components retained. Continuous outer convergence,query/batch ranking andindependent deployment gain remain open.')
    save(out/'compact_summary.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('recommendations','risk_outer_attribution')}),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default=PLAN);a=p.parse_args()
    try:run(a.plan)
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
