"""Refine cached mixture KG queries with the historical precision rule."""

import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save
from performance.refine_historical_raw_lookahead import cpu_timeout
from performance.connect_complete_historical_reuse import run as restore_connected
from core.mixture_guarded_kg import mixture_raw_lookahead

PLAN='performance/manifests/historical_mixture_lookahead_precision_v1.json'


def run(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    result_path=out/'results.json';previous=read(str(result_path.relative_to(ROOT))) if result_path.exists() else {}
    if previous.get('all_registered_rules_complete'):return previous
    rows=previous.get('rows',[]);started=time.process_time();stopped=False
    models=restore_connected(spec['connection_plan']);precision=spec['precision']
    reference=read(spec['reference'])['rows'];coarse=[]
    handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds'])
    try:
        for state,model,terminal,remaining,baseline in models:
            layout=state['layout'];query=spec['queries'][layout]
            old=next(r for r in reference if r['layout']==layout and r['candidate']==query)
            coarse.append({'layout':layout,'query':query,**{k:old[k] for k in (
                'net_reward_of_paid_episode','adaptation_value','filter_and_quadrature_baseline_drift',
                'model_weight_martingale_residual')}})
            baselines={7:baseline}
            for power,fantasy in spec['rules']:
                if any(r['layout']==layout and r['draw_power']==power and r['fantasy_power']==fantasy for r in rows):continue
                one=time.process_time()
                if power not in baselines:
                    values,costs,actions=model.terminal_values(terminal,remaining,power,spec['seed'])
                    debit,_,future=model.terminal_values(terminal,remaining-1,power,spec['seed'])
                    baselines[power]=(values,costs,actions,debit,future)
                result=mixture_raw_lookahead(model,terminal,remaining,[query],draw_power=power,
                    fantasy_power=fantasy,seed=spec['seed'],baseline=baselines[power])
                row=result['lookahead'][0];drift=abs(row['filter_and_quadrature_baseline_drift'])
                identity=max(abs(v) for v in row['model_weight_martingale_residual'])
                passed=bool(drift<=precision['maximum_absolute_tower_drift']
                    and identity<=precision['maximum_identity_martingale_residual']
                    and row['maximum_raw_integration_relative_error']<=precision['maximum_raw_integration_relative_error'])
                rows.append({'layout':layout,'draw_power':power,'fantasy_power':fantasy,
                    'stop_expected_reward':result['stop_expected_reward'],**row,'precision_gate_passed':passed,
                    'maximum_identity_martingale_residual':identity,'cpu_seconds':time.process_time()-one})
                save(result_path,{'execution_plan':plan,'rows':rows,'coarse_reference':coarse,
                    'all_registered_rules_complete':len(rows)==len(models)*len(spec['rules']),
                    'independent_confirmation':False,'scientific_model_ranking_authorized':False,'cost':spec['cost']})
                print('mixture precision',layout,power,fantasy,'net',round(row['net_reward_of_paid_episode'],6),
                    'adapt',round(row['adaptation_value'],6),'tower',round(row['filter_and_quadrature_baseline_drift'],6),
                    'martingale',round(identity,6),'pass',passed,flush=True)
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    latest=[next((r for r in reversed(rows) if r['layout']==layout),None) for layout in spec['queries']]
    result={'execution_plan':plan,'rows':rows,'coarse_reference':coarse,
        'all_registered_rules_complete':len(rows)==len(models)*len(spec['rules']),
        'all_latest_precision_gates_passed':all(r is not None and r['precision_gate_passed'] for r in latest),
        'stopped_at_cpu_budget':stopped,'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'no_model_adopted':True,'independent_confirmation':False,'scientific_model_ranking_authorized':False,
        'limitations':['Two queries selected from opened development diagnostics;no held-out performance.',
            'Martingale residual isolates predictive fantasy integration;terminal tower drift also includes Gaussian moment projection.',
            'A PASS at two queries cannot certify a complete action ranking or a source-prior decision advantage.']}
    save(result_path,result);return result


if __name__=='__main__':run()
