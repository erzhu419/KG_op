import json,signal,time
from pathlib import Path
started=time.process_time()
result={'complete':False,'new_integrations':0,'scope':'Only arithmetic recomposition of paid fixed-action per-head values; no new terminal evaluations or predictive draws.'}
def stop(*args):raise TimeoutError('Paid-prefix reader reached0.12CPU budget')
signal.signal(signal.SIGPROF,stop);signal.setitimer(signal.ITIMER_PROF,0.12)
def read(p):return json.loads(Path(p).read_text())
try:
    spec=read('performance/manifests/historical_observed_prediction_v3.json');out=spec['output'];state=read(out+'/state.json')
    assembly=read('paper_artifacts/historical_observed_batch_assembly_v1/compact_summary.json')
    origins=spec['origins'];values={};means={}
    for origin in origins:
        means[str(origin)]={}
        for head in (0,1):
            data=read(out+f'/n4/{origin}/{head}/results.json')
            assert data['complete'] and data['all_inner_rules_passed'] and data['predictive_cdf_passed'] and len(data['values'])==2048
            values[origin,head]=data['values'];means[str(origin)][str(head)]={str(p):sum(data['values'][:p])/p for p in (16,2048)}
    rows=[]
    for f in (16,2048):
        for g in (16,2048):
            value=sum(state['prior_weights'][o]*(sum(values[o,0][:f])/f)*(sum(values[o,1][:g])/g) for o in origins)
            rows.append(dict(objective_prefix=f,risk_prefix=g,fixed_reward=value,tower_residual=value-assembly['debited_reward']))
    for p in (16,2048):
        old=next(r for r in assembly['cached_fixed_action_diagnostic']['prefixes'] if r['prefix']==p)
        new=next(r for r in rows if r['objective_prefix']==r['risk_prefix']==p)
        assert abs(old['expected_reward']-new['fixed_reward'])<=1e-12
    result.update(complete=True,source_plan='performance/manifests/historical_observed_prediction_v3.json',source_output=out,
        source_state=spec['state'],origins=origins,prior_weights=state['prior_weights'],per_head_prefix_means=means,rows=rows,
        debited_prior_point=assembly['debited_reward'],legacy_diagonal_prefixes_reproduced=True,
        saved_head_observation_match_error=assembly['cached_fixed_action_diagnostic']['matched_observation_maximum_difference'])
except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc))
finally:
    signal.setitimer(signal.ITIMER_PROF,0);result['reader_process_cpu_seconds']=time.process_time()-started;print(json.dumps(result))
