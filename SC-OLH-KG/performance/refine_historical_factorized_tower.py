"""Reuse per-head terminal branches to check the fixed-action tower cheaply."""

import signal
import time

from core.positive_mixture_lookahead import fixed_action_factorized_reference
from performance.complete_historical_source_heads import ROOT,read,save
from performance.connect_complete_historical_reuse import run as restore_connected
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_factorized_tower_v1.json'


def run(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json';previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_queries_precision_passed'):return previous
    started=time.process_time();rows=previous.get('rows',[]);passed=previous.get('passed_layouts',[])
    models=restore_connected(spec['connection_plan']);kg=read(spec['projected_KG'])['rows']
    handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds']);stopped=False
    try:
        for state,model,terminal,remaining,*_ in models:
            layout=state['layout'];query=spec['queries'][layout]
            if layout in passed:continue
            values,_,actions=model.terminal_values(terminal,remaining-1,spec['terminal_power'],spec['seed'])
            action=actions[int(values.argmax())];registered=next(r for r in kg if r['layout']==layout)
            for power in spec['fantasy_powers']:
                if any(r['layout']==layout and r['fantasy_power']==power for r in rows):continue
                phase=time.process_time()
                row=fixed_action_factorized_reference(model,action,query,draw_power=spec['draw_power'],
                    terminal_power=spec['terminal_power'],fantasy_power=power,seed=spec['seed'],protocol=terminal.protocol)
                row.update(layout=layout,cpu_seconds=time.process_time()-phase)
                original=float(values.max());reproduction=abs(row['projected_prior_reward']-original)
                if reproduction>1e-12:raise RuntimeError('Per-head terminal branches changed the cached fixed action')
                precision=spec['precision']
                adequate=bool(row['predictive_log_evidence_max_error']<=precision['maximum_log_evidence_error']
                    and row['effective_sample_size_min']>=precision['minimum_weight_ESS'])
                gate=adequate and bool(abs(row['likelihood_tower_drift'])<=precision['maximum_absolute_tower_drift'])
                # E[max U]-E[U(a0)] + E[U(a0)]. The last term is now a
                # factored integral of the SAME512-draw projected estimator.
                # This records a control-variate diagnostic,not an adopted KG.
                corrected=registered['net_reward_of_paid_episode']-registered['filter_and_quadrature_baseline_drift']
                corrected+=row['projected_baseline_drift']
                row.update(reference_adequate=adequate,likelihood_tower_precision_passed=gate)
                row['projected_tower_precision_passed']=adequate and abs(row['projected_baseline_drift'])<=precision['maximum_absolute_tower_drift']
                row.update(cached_terminal_prior_reproduction_error=reproduction,
                    control_variate_net_diagnostic=corrected,
                    adaptation_value_from_cached_joint_KG=registered['adaptation_value'],
                    raw_joint_KG_baseline_drift=registered['filter_and_quadrature_baseline_drift'])
                rows.append(row)
                if gate:passed.append(layout)
                save(path,{'execution_plan':plan,'rows':rows,'passed_layouts':passed,
                    'all_queries_precision_passed':len(passed)==2})
                print('factorized tower',layout,power,'raw',row['likelihood_tower_drift'],
                    'projected',row['projected_baseline_drift'],'adequate',adequate,'pass',gate,flush=True)
                if gate:break
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    result={'execution_plan':plan,'rows':rows,'passed_layouts':passed,
        'all_queries_precision_passed':len(passed)==2,'stopped_at_cpu_budget':stopped,
        'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'scientific_model_ranking_authorized':False,'independent_confirmation':False,'no_model_adopted':True,
        'limitations':['Fixed action only;adaptive maximum still uses the opened two-dimensional jointKG nodes.',
            'Likelihood reference keeps previous two paid projections;this is not exact longitudinal inference.',
            'Control-variate net value is a development numerical diagnostic,not an independent decision advantage.']}
    save(path,result);return result


if __name__=='__main__':run()
