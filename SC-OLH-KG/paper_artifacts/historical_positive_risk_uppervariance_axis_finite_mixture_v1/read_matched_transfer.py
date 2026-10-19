import time
started=time.process_time()
import gzip,json,math,signal
from pathlib import Path
result={'complete':False,'new_integrations':0,'scope':'Paid fixed355 statistics on identical F16/risk224 nodes;finite-origin transfer diagnostic only,no bank replacement or absolute error certificate.'}
def stop(*args):raise TimeoutError('Matched finite-origin reader reached0.18CPU budget')
signal.signal(signal.SIGPROF,stop);signal.setitimer(signal.ITIMER_PROF,0.18)
def read(p):return json.loads(Path(p).read_text())
try:
    plan='performance/manifests/historical_positive_risk_uppervariance_axis_cached_pair_references_screen_v2.json'
    spec=read(plan);bank=read(spec['bank_plan']);base=bank['output'];compact=read(spec['output']+'/compact_summary.json')
    marker=b',"actions":';prefix=b''
    with gzip.open(base+'/context.json.gz','rb') as stream:
        while marker not in prefix:
            chunk=stream.read(131072)
            if not chunk:raise ValueError('Origin-prefix terminator absent')
            prefix+=chunk
    origin_bytes=prefix.split(marker,1)[0]
    assert origin_bytes.startswith(b'{"origins":')
    origins=json.loads(origin_bytes+b'}')['origins']
    prior=read('paper_artifacts/historical_observed_batch_assembly_v1/compact_summary.json')
    # Original full prior is also stored in the tiny admitted-prefix receipt.
    prep=read('paper_artifacts/historical_observed_prediction_v1/state.json');weights=prep['prior_weights']
    assert sorted(origins)==['4','5'] and all(len(v['objective_scores'])==16 for v in origins.values())
    shard=next(s for s,v in enumerate(spec['policy_shards']) if 63 in v)
    fixed={}
    for h in (4,5):
        task=read(base+f'/tasks/{h}_{shard}.json');assert task['complete']
        fixed[h]={r['record']:r['fixed']['raw'] for r in task['rows'] if 'fixed' in r}
    mixed=diag=flow45=flow54=omitted=0.;origin_rows=[];record_error=0.
    for origin,data in origins.items():
        o=int(origin);mix_o=diag_o=flow_o=omit_o=0.
        for risk in data['risk']:
            b4=fixed[4][risk['record']];b5=fixed[5][risk['record']];w=risk['outer_weight']/16
            for i,fs in enumerate(data['objective_scores']):
                scores=[math.log(weights[k])+fs[k]+risk['head_log_scores'][k] for k in range(8)]
                top=max(scores);terms=[math.exp(v-top) for v in scores];den=sum(terms)
                p4=terms[4]/den;p5=terms[5]/den;missing=sum(terms[k] for k in (0,1,2,3,6,7))/den
                H4=data['objective']['raw'][i][4][355][0]*b4;H5=data['objective']['raw'][i][5][355][0]*b5
                own=H4 if o==4 else H5;other=H5 if o==4 else H4;pother=p5 if o==4 else p4
                mix_o+=w*(p4*H4+p5*H5);diag_o+=w*own
                flow_o+=w*pother*(other-own);omit_o-=w*missing*own
        mixed+=weights[o]*mix_o;diag+=weights[o]*diag_o;omitted+=weights[o]*omit_o
        if o==4:flow45=weights[o]*flow_o
        else:flow54=weights[o]*flow_o
        origin_rows.append(dict(origin=o,mixed_contribution=weights[o]*mix_o,diagonal_contribution=weights[o]*diag_o,other_identity_transfer=weights[o]*flow_o,missing_utility_correction=weights[o]*omit_o))
    # The previously settled exact weighted sum comes from the same frozen screen.
    expected=0.7452258287528934
    assert abs(mixed-expected)<1e-12
    residual=mixed-diag;assert abs(residual-flow45-flow54-omitted)<1e-12
    result.update(complete=True,source_bound_plan=plan,same_node_posterior_mix=mixed,same_node_origin_diagonal=diag,
        finite_mixture_transfer_residual=residual,origin4_to5_reward_transfer=flow45,origin5_to4_reward_transfer=flow54,
        omitted_utility_diagonal_correction=omitted,origin_rows=origin_rows,summary_fixed_match_error=abs(mixed-expected),
        prior_debited_fixed_point=prior['debited_reward'],diagonal_minus_prior=diag-prior['debited_reward'],
        prior_origin_mass_not_in_nodes=1-weights[4]-weights[5],all8_likelihoods_in_posterior=True,
        density_ratio_added=False,weights_renormalized=False,context_origins_only_read=True)
except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc))
finally:
    signal.setitimer(signal.ITIMER_PROF,0);result['reader_process_cpu_seconds']=time.process_time()-started;print(json.dumps(result))
