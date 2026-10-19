import gzip,json,signal,time
from pathlib import Path
started=time.process_time()
result={'complete':False,'new_integrations':0,'scope':'Incremental finite224/F16 common-pool policy replay using paid references; no deployment/absolute-error claim.'}
def stop(*args):raise TimeoutError('Incremental policy replay reached4.5CPU budget')
signal.signal(signal.SIGPROF,stop);signal.setitimer(signal.ITIMER_PROF,4.5)
def read(path):return json.loads(Path(path).read_text())
try:
    import numpy as np
    from core.raw_terminal_head_grid import compose_head_grid,select_raw_terminal_actions
    from performance.validate_historical_component_bank import selected_reference_rule
    new_plan='performance/manifests/historical_positive_risk_uppervariance_axis_cached_pair_references_screen_v2.json'
    old_plan='performance/manifests/historical_positive_risk_uppervariance_axis_screen_competition_v60.json'
    old=read(old_plan);new=read(new_plan)
    assert old['bank_plan']==new['bank_plan'] and old['native_reference_plan']==new['native_reference_plan']
    old_refs=old['joint_reference_plans'];new_refs=new['joint_reference_plans']
    assert new_refs[:len(old_refs)]==old_refs
    targets=set();references=[]
    for p in new_refs:
        spec=read(p);prep=read(spec['output']+'/preparation.json');selected={r['record']:r for r in prep['selected']}
        for check in read(spec['output']+'/compact_summary.json')['groups']:
            if not check['candidate_passed']:continue
            if p not in old_refs:targets.add(check['record'])
            references.append((p,spec,check,selected[check['record']]))
    assert len(targets)==3
    base=read(new['bank_plan'])['output']
    with gzip.open(base+'/context.json.gz','rt') as f:context=json.load(f)
    replay_path='paper_artifacts/historical_positive_risk_uppervariance_axis_competition_v60/incumbent_replay_final.json'
    old_replay=read(replay_path)
    assert old_replay['complete'] and old_replay['source_bound_plan']==old_plan
    saved={r['record']:r for r in old_replay['raw_recommendation']['rows'] if r['record'] in targets}
    assert set(saved)==targets
    old_screen=read(old['output']+'/summary.json');new_screen=read(new['output']+'/summary.json')
    coverage=[]
    for scr in (old_screen,new_screen):coverage.append({(r['record'],r['kind']):r['validated_reference_actions'] for r in scr['records'] if r['record'] in targets})
    for before in old_screen['records']:
        if before['record'] not in targets:
            after=next(r for r in new_screen['records'] if (r['record'],r['kind'])==(before['record'],before['kind']))
            assert before['validated_reference_actions']==after['validated_reference_actions']
    tables={(name,kind):np.zeros((1,8,365,4)) for name in targets for kind in ('raw','projected')}
    costs={(name,kind):np.zeros((1,8,365)) for name in targets for kind in ('raw','projected')}
    def scalar(name,h,a,kind,v):
        tables[name,kind][0,h,a]=[v,0.,0.,0.];costs[name,kind][0,h,a]=context['actions'][a]['budgets'][0]
    shard=next(s for s,policies in enumerate(new['policy_shards']) if 63 in policies)
    fixed=set()
    for h in (4,5):
        for row in read(base+f'/tasks/{h}_{shard}.json')['rows']:
            if row['record'] in targets and 'fixed' in row:
                for kind in ('raw','projected'):scalar(row['record'],h,355,kind,row['fixed'][kind])
                fixed.add((row['record'],h))
    assert len(fixed)==6
    for row in read(base+'/query_references.json')['rows']:
        if row['record'] in targets:
            for kind in ('raw','projected'):scalar(row['record'],row['identity'],row['action'],kind,row[kind])
    for row in read(read(new['native_reference_plan'])['output']+'/native_direct_references.json')['rows']:
        if row['record'] in targets:scalar(row['record'],row['identity'],row['action'],'projected',row['projected'])
    valid={name:tables[name,'raw'].copy() for name in targets};valid_cost={name:costs[name,'raw'].copy() for name in targets}
    prior=np.asarray(context['prior_weights']);points={r['record']:(int(o),r,v) for o,v in context['origins'].items() for r in v['risk'] if r['record'] in targets}
    names=('projected','raw','raw_quality_projected_risk','projected_quality_raw_risk')
    def evaluate(name,cov):
        origin,r,data=points[name];grids={}
        for kind in ('raw','projected'):
            v,c,_,_=compose_head_grid(np.asarray(data['objective'][kind]),tables[name,kind],costs[name,kind],data['objective_scores'],[r['head_log_scores']],prior)
            mask=np.zeros(365,dtype=bool);mask[cov[name,kind]]=True
            grids[kind]=(np.where(mask[None,None],v,-np.inf),c)
        raw,rc=grids['raw'];native,nc=grids['projected'];common=np.isfinite(raw)&np.isfinite(native)
        assert np.array_equal(np.flatnonzero(common[0,0]),np.flatnonzero(common[-1,0]))
        checked,_,_,_=compose_head_grid(np.asarray(data['objective']['raw']),valid[name],valid_cost[name],data['objective_scores'],[r['head_log_scores']],prior)
        choices={'projected':np.lexsort((nc,-np.where(common,native,-np.inf)),axis=-1)[...,0],
            'raw':select_raw_terminal_actions(np.where(common,raw,-np.inf),rc)}
        for policy,fkind,gkind in (('raw_quality_projected_risk','raw','projected'),('projected_quality_raw_risk','projected','raw')):
            v,c,_,_=compose_head_grid(np.asarray(data['objective'][fkind]),tables[name,gkind],costs[name,gkind],data['objective_scores'],[r['head_log_scores']],prior)
            choices[policy]=np.lexsort((c,-np.where(common,v,-np.inf)),axis=-1)[...,0]
        rows={};heads=np.arange(16)
        for policy,chosen in choices.items():
            ix=chosen[:,0];sel=raw[heads,0,ix]-raw[:,0,355];chk=checked[heads,0,ix]-checked[:,0,355]
            rows[policy]=dict(actions=ix.tolist(),selection_raw_contrasts=sel.tolist(),validation_raw_contrasts=chk.tolist(),changed_heads=int(np.count_nonzero(ix!=choices['projected'][:,0])))
        return dict(record=name,origin=origin,mixture_weight=float(prior[origin]*r['outer_weight']),common_admitted_actions=np.flatnonzero(common[0,0]).tolist(),policies=rows)
    def insert(items):
        for p,spec,check,selected in items:
            name=check['record']
            if name not in targets:continue
            kind=check['kind'];ix=selected['action_indices']
            for h in (4,5):
                rows=read(spec['output']+f'/parts/{name}/{h}_{kind}.json')['rows'];row=rows[selected_reference_rule(check,h)]
                tables[name,kind][0,h,ix]=row['statistics'];costs[name,kind][0,h,ix]=row['costs']
                if kind=='raw':
                    hi=rows[check.get('validation_rule','E9')];valid[name][0,h,ix]=hi['statistics'];valid_cost[name][0,h,ix]=hi['costs']
    insert([r for r in references if r[0] in old_refs]);before={n:evaluate(n,coverage[0]) for n in targets};error=0.
    for n in targets:
        assert before[n]['common_admitted_actions']==saved[n]['common_admitted_actions']
        for p in names:
            assert before[n]['policies'][p]['actions']==saved[n]['policies'][p]['actions']
            for key in ('selection_raw_contrasts','validation_raw_contrasts'):
                error=max(error,float(np.max(abs(np.asarray(before[n]['policies'][p][key])-saved[n]['policies'][p][key]))))
    assert error<=1e-12
    insert([r for r in references if r[0] not in old_refs]);after={n:evaluate(n,coverage[1]) for n in targets}
    totals=json.loads(json.dumps(old_replay['raw_recommendation']['policy_totals']));deltas={}
    for p in names:
        ds=dv=dm=0.;changed=0
        for n in targets:
            b=before[n]['policies'][p];a=after[n]['policies'][p];w=after[n]['mixture_weight']
            ds+=w*(float(np.mean(a['selection_raw_contrasts']))-float(np.mean(b['selection_raw_contrasts'])))
            dv+=w*(float(np.mean(a['validation_raw_contrasts']))-float(np.mean(b['validation_raw_contrasts'])))
            dm+=w*(float(np.mean(np.asarray(a['validation_raw_contrasts']) < -1e-4))-float(np.mean(np.asarray(b['validation_raw_contrasts']) < -1e-4)))
            changed+=a['changed_heads']-b['changed_heads']
        totals[p]['selection_contrast']+=ds;totals[p]['validation_contrast']+=dv;totals[p]['changed_heads']+=changed;totals[p]['validation_negative_mass_below1e4']+=dm
        deltas[p]=dict(selection_contrast=ds,validation_contrast=dv,changed_heads=changed)
    L=new_screen['weighted_bank_envelope']['kinds']['raw']['best_reference_contrast'];G=new_screen['weighted_bank_envelope']['kinds']['raw']['unresolved_upper_gap']
    loss=new['control_variate']['sampling_opportunity_loss'];selection_gap=L-totals['raw']['selection_contrast']
    assert selection_gap>=-1e-12
    result.update(complete=True,source_replay=replay_path,current_bound_plan=new_plan,updated_records=sorted(targets),reused_records=len(old_replay['raw_recommendation']['rows'])-len(targets),
        unchanged_reference_prefix_verified=True,old_policy_actions_matched=True,maximum_old_reward_difference=error,
        policy_totals=totals,policy_deltas=deltas,changed_rows=[after[n] for n in sorted(targets)],
        raw_best_reference_contrast=L,raw_unresolved_gap=G,raw_common_pool_selection_shortfall=selection_gap,
        nominal_raw_policy_upper_gap=G+selection_gap,
        raw_net_point_preview=totals['raw']['validation_contrast']-loss,
        projected_choice_raw_net_point_preview=totals['projected']['validation_contrast']-loss,
        baseline_point_values=new['control_variate'],all_conditional_actions_closed=False,global_total_error_certificate_available=False,independent_confirmation=False)
except Exception as exc:result.update(complete=False,error_type=type(exc).__name__,error=str(exc))
finally:
    signal.setitimer(signal.ITIMER_PROF,0)
    result['reader_process_cpu_seconds']=time.process_time()-started
    print(json.dumps(result))
