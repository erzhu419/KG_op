import json,signal,time
from pathlib import Path
started=time.process_time()
result={'complete':False,'scope':'Finite224/F16 fixed tower diagnostic only; not an absolute quadrature or net KG error bound.'}
def stop(*args):raise TimeoutError('Fixed tower metadata reader reached0.5CPU budget')
signal.signal(signal.SIGPROF,stop);signal.setitimer(signal.ITIMER_PROF,0.5)
try:
    def read(p):return json.loads(Path(p).read_text())
    plan='performance/manifests/historical_positive_risk_uppervariance_axis_cached_pair_references_screen_v2.json'
    spec=read(plan);summary=read(spec['output']+'/summary.json');baseline=spec['control_variate']
    values={}
    for kind in ('raw','projected'):
        rows=[r for r in summary['records'] if r['kind']==kind]
        fixed=sum(r['outer_grid_weight']*r['mean_fixed_action_reward'] for r in rows)
        values[kind]=dict(weighted_mean_fixed_reward=fixed,represented_origin_weight=sum(r['outer_grid_weight'] for r in rows))
    raw=values['raw']['weighted_mean_fixed_reward'];debit=baseline['debited_fixed_reward']
    result.update(complete=True,source_screen_plan=plan,source_screen_summary=spec['output']+'/summary.json',
        baseline_source=baseline['baseline_source'],kinds=values,prior_debited_fixed_point=debit,
        current_raw_fixed_tower_residual=raw-debit,shared_numerical_fixed_value_equal=(raw==debit),
        baseline_absolute_error_certificate_available=False,new_integrations=0)
except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc))
finally:
    signal.setitimer(signal.ITIMER_PROF,0);result['reader_process_cpu_seconds']=time.process_time()-started;print(json.dumps(result))
