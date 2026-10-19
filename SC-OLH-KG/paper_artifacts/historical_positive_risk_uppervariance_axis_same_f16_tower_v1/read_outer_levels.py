import json,signal,time
from pathlib import Path
started=time.process_time()
result={'complete':False,'scope':'Reuse already-paid risk grids with unchanged F16 heads. Layer differences are diagnostics,not absolute error bounds;raw prior baseline is not assumed to share finite objective draws.'}
def stop(*args):raise TimeoutError('Same-F16 outer metadata reader reached0.75CPU budget')
signal.signal(signal.SIGPROF,stop);signal.setitimer(signal.ITIMER_PROF,0.75)
def read(p):return json.loads(Path(p).read_text())
plans=[
 'performance/manifests/historical_positive_risk_full_grid_screen_competition_v19_refs_v1.json',
 'performance/manifests/historical_positive_risk_body_axis_bound_incumbents_v3.json',
 'performance/manifests/historical_positive_risk_shoulder_axis_bound_incumbents_v4.json',
 'performance/manifests/historical_positive_risk_midmean_axis_bound_incumbents_v3.json',
 'performance/manifests/historical_positive_risk_uppermidmean_axis_bound_incumbents_v4.json',
 'performance/manifests/historical_positive_risk_uppervariance_axis_cached_pair_references_screen_v2.json']
try:
    levels=[];node_values=[];baseline=None
    for plan in plans:
        spec=read(plan);summary=read(spec['output']+'/summary.json');bank=read(spec['bank_plan'])
        prep=read(bank['output']+'/preparation.json')
        if baseline is None:baseline=spec['control_variate']
        assert spec['control_variate']==baseline and summary['objective_heads_per_origin']==16 and summary['remaining_calls']==356
        selected={r['record']:r for r in prep['selected']}
        rows=[r for r in summary['records'] if r['kind']=='raw']
        assert len(rows)==summary['risk_observations'] and all(r['outer_grid_weight'] is not None for r in rows)
        vals={};origin={}
        for row in rows:
            point=selected[row['record']];o=int(point['origin']);key=(o,tuple(point['cdf']))
            vals[key]=row['mean_fixed_action_reward']
            entry=origin.setdefault(str(o),dict(weighted_fixed_reward=0.,weight=0.,records=0))
            entry['weighted_fixed_reward']+=row['outer_grid_weight']*row['mean_fixed_action_reward'];entry['weight']+=row['outer_grid_weight'];entry['records']+=1
        fixed=sum(v['weighted_fixed_reward'] for v in origin.values())
        levels.append(dict(bound_plan=plan,bank_plan=spec['bank_plan'],source_summary=spec['output']+'/summary.json',
            risk_observations=len(rows),objective_heads_per_origin=16,remaining_calls=356,
            source_numerical_containment_passed=summary['numerical_containment_passed'],
            raw_weighted_fixed_reward=fixed,raw_fixed_tower_residual=fixed-baseline['debited_fixed_reward'],
            origin_contributions=origin,represented_origin_weight=sum(v['weight'] for v in origin.values()),
            risk_grid_metadata={k:bank.get(k) for k in ('outer_axis','risk_mean_order','risk_variance_order','risk_variance_orders','mean_cdf_intervals','same_state_context_plan','historical_bound_plan')}))
        node_values.append(vals)
    comparisons=[]
    for i in range(1,len(levels)):
        low,high=levels[i-1],levels[i];common=node_values[i-1].keys()&node_values[i].keys()
        comparisons.append(dict(from_observations=low['risk_observations'],to_observations=high['risk_observations'],
            weighted_fixed_difference=high['raw_weighted_fixed_reward']-low['raw_weighted_fixed_reward'],
            retained_cdf_coordinate_count=len(common),maximum_retained_coordinate_fixed_difference=max((abs(node_values[i][k]-node_values[i-1][k]) for k in common),default=0.),
            origin_fixed_differences={o:high['origin_contributions'][o]['weighted_fixed_reward']-low['origin_contributions'][o]['weighted_fixed_reward'] for o in high['origin_contributions']}))
    result.update(complete=True,levels=levels,adjacent_comparisons=comparisons,baseline_point_values=baseline,
        same_declared_objective_heads=16,baseline_draw_identity_verified=False,new_integrations=0,full_context_arrays_read=0)
except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc))
finally:
    signal.setitimer(signal.ITIMER_PROF,0);result['reader_process_cpu_seconds']=time.process_time()-started;print(json.dumps(result))
