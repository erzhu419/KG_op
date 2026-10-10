"""Complete the existing balanced4 action comparison with a factored baseline."""

import signal
import time

import numpy as np

from core.mixture_guarded_kg import mixture_raw_lookahead
from core.positive_mixture_lookahead import fixed_action_factorized_reference
from performance.complete_historical_source_heads import ROOT,read,save
from performance.connect_complete_historical_reuse import run as restore_connected
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_control_variate_action_comparison_v1.json'


def run(plan=PLAN):
    spec=read(plan);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json';previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all12_registered_actions_complete'):return previous
    rows=previous.get('rows',[]);references=previous.get('reference_rows',[]);stopped=[]
    if not read('paper_artifacts/historical_control_variate_action_comparison_v1/batch_equivalence.json')['all32_heads_equivalent']:
        raise RuntimeError('Batched raw fantasies changed the paid filter')
    started=time.process_time();models=restore_connected(spec['connection_plan'])
    old_joint=read(spec['existing_joint_KG'])['rows'];prefix=read(spec['existing_prefix_KG'])['rows']
    old_reference=read(spec['existing_factorized_reference'])['rows'];precision=spec['precision']
    for state,model,terminal,remaining,*_ in models:
        layout=state['layout'];values,costs,actions=model.terminal_values(terminal,remaining,spec['terminal_power'],spec['seed'])
        debited,_,future=model.terminal_values(terminal,remaining-1,spec['terminal_power'],spec['seed'])
        baseline=(values,costs,actions,debited,future);action=future[int(debited.argmax())]
        for candidate in state['action_sets']['4']['active_indices']:
            candidate=int(candidate)
            if any(r['layout']==layout and r['candidate']==candidate for r in rows):continue
            one=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
            signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds_per_action'])
            try:
                known=[r for r in old_reference if r['layout']==layout and r['candidate']==candidate]
                reference=next((r for r in reversed(known) if r['likelihood_tower_precision_passed']
                    and r['projected_tower_precision_passed']),None)
                if reference is None:
                    registered=[r for r in references if r['layout']==layout and r['candidate']==candidate]
                    for power in spec['reference_fantasy_powers']:
                        cached=next((r for r in registered if r['fantasy_power']==power),None)
                        ref=(cached if cached else fixed_action_factorized_reference(model,action,candidate,
                            draw_power=spec['reference_draw_power'],terminal_power=spec['terminal_power'],
                            fantasy_power=power,seed=spec['seed'],protocol=terminal.protocol))
                        adequate=bool(ref['predictive_log_evidence_max_error']<=precision['maximum_log_evidence_error']
                            and ref['effective_sample_size_min']>=precision['minimum_reference_ESS'])
                        passed=adequate and bool(abs(ref['likelihood_tower_drift'])<=precision['maximum_absolute_tower_drift']
                            and abs(ref['projected_baseline_drift'])<=precision['maximum_absolute_tower_drift'])
                        ref.update(layout=layout,reference_adequate=adequate,reference_precision_passed=passed)
                        if not cached:references.append(ref)
                        reference=ref
                        save(path,{'execution_plan':plan,'rows':rows,'reference_rows':references,
                            'all12_registered_actions_complete':len(rows)==12})
                        if passed:break
                ref_pass=(bool(reference['reference_precision_passed']) if 'reference_precision_passed' in reference
                    else bool(reference['likelihood_tower_precision_passed'] and reference['projected_tower_precision_passed']))
                np.testing.assert_allclose(reference['projected_prior_reward'],debited.max(),atol=1e-12,rtol=0.)
                known_joint=next((r for r in old_joint if r['layout']==layout and r['candidate']==candidate),None)
                if known_joint:
                    row=dict(known_joint)
                    original=row['filter_and_quadrature_baseline_drift']
                    row['raw_joint_baseline_drift']=original
                    row['raw_joint_expected_future_reward']=row['expected_future_reward']
                    row['fixed_action_expected_reward']=reference['projected_expected_reward']
                    row['filter_and_quadrature_baseline_drift']=reference['projected_baseline_drift']
                    row['projection_tower_drift']=reference['projected_baseline_drift']
                    row['expected_future_reward']=row['adaptation_value']+reference['projected_expected_reward']
                    row['net_reward_of_paid_episode']=row['expected_future_reward']-row['stop_expected_reward']
                    previous=next(r for r in prefix if r['layout']==layout and r['candidate']==candidate)
                    row['nested_prefix_adaptation_value']=previous['adaptation_value']
                    row['nested_adaptation_difference']=abs(row['adaptation_value']-previous['adaptation_value'])
                else:
                    result=mixture_raw_lookahead(model,terminal,remaining,[candidate],draw_power=spec['terminal_power'],
                        fantasy_power=spec['joint_fantasy_power'],seed=spec['seed'],baseline=baseline,
                        fixed_action_expectations={candidate:reference['projected_expected_reward']})
                    row={'layout':layout,'stop_expected_reward':result['stop_expected_reward'],**result['lookahead'][0]}
                identity=max(abs(v) for v in row['model_weight_martingale_residual'])
                passed=ref_pass and bool(identity<=precision['maximum_identity_martingale_residual']
                    and row['maximum_raw_integration_relative_error']<=precision['maximum_raw_integration_relative_error']
                    and row['nested_adaptation_difference']<=precision['maximum_nested_adaptation_difference'])
                row.update(precision_gate_passed=passed,maximum_identity_martingale_residual=identity,
                    fixed_action_reference_precision_passed=ref_pass,reference_fantasy_power=reference['fantasy_power'],
                    joint_fantasy_power=spec['joint_fantasy_power'],cpu_seconds=time.process_time()-one,
                    reused_complete_joint_query=bool(known_joint),source_law='original passed32;learned candidates not adopted')
                rows.append(row);save(path,{'execution_plan':plan,'rows':rows,'reference_rows':references,
                    'all12_registered_actions_complete':len(rows)==12})
                print('CV action',layout,candidate,'net',round(row['net_reward_of_paid_episode'],6),
                    'contrast difference',round(row['nested_adaptation_difference'],6),'identity',round(identity,6),'pass',passed,flush=True)
            except TimeoutError:
                stopped.append({'layout':layout,'candidate':candidate,'reason':'registered per-action CPU cap'})
                break
            finally:
                signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    comparison=[]
    for state,*_ in models:
        group=[r for r in rows if r['layout']==state['layout']]
        expected=len(state['action_sets']['4']['active_indices'])
        winner=max(group,key=lambda r:r['net_reward_of_paid_episode']) if group else None
        comparison.append({'layout':state['layout'],'registered_action_count':expected,'completed_action_count':len(group),
            'all_actions_precision_passed':len(group)==expected and all(r['precision_gate_passed'] for r in group),
            'provisional_best_candidate':winner['candidate'] if winner else None,
            'provisional_best_net':winner['net_reward_of_paid_episode'] if winner else None})
    result={'execution_plan':plan,'rows':rows,'reference_rows':references,'comparison':comparison,
        'all12_registered_actions_complete':len(rows)==12,'actions_precision_passed':sum(r['precision_gate_passed'] for r in rows),
        'stopped_actions':stopped,'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'no_model_adopted':True,'independent_confirmation':False,'scientific_model_ranking_authorized':False,
        'limitations':['Complete only within the existing balanced4 plus two paid-repeat action sets on two opened eight-policy states.',
            'Nested contrast differences assess numerical stability,not a statistical confidence interval for close rankings.',
            'No strong baseline comparison or held-out source-prior advantage follows from these acquisition values.']}
    save(path,result);return result


def refine(plan='performance/manifests/historical_control_variate_action_precision_v2.json'):
    """Refine only failures;keep the fixed-action control and all parent rows."""
    spec=read(plan);parent=read(spec['parent_results']);settings=read(spec['parent_plan'])
    if not parent['all12_registered_actions_complete']:
        raise RuntimeError('Complete the original registered12 actions before refinement')
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True);path=out/'results.json'
    previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_refinements_complete'):return previous
    rows=previous.get('rows',[]);started=time.process_time();stopped=False
    precision=settings['precision'];models=restore_connected(settings['connection_plan'])
    handler=signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds'])
    try:
        for state,model,terminal,remaining,*_ in models:
            layout=state['layout']
            failed=[r for r in parent['rows'] if r['layout']==layout and not r['precision_gate_passed']]
            if not failed:continue
            values,costs,actions=model.terminal_values(terminal,remaining,settings['terminal_power'],settings['seed'])
            debit,_,future=model.terminal_values(terminal,remaining-1,settings['terminal_power'],settings['seed'])
            baseline=(values,costs,actions,debit,future)
            for old in failed:
                if not old['fixed_action_reference_precision_passed']:
                    raise RuntimeError('This refinement requires an already passed independent fixed-action reference')
                current=old;candidate=old['candidate']
                for power in spec['joint_fantasy_powers']:
                    cached=next((r for r in rows if r['layout']==layout and r['candidate']==candidate
                        and r['joint_fantasy_power']==power),None)
                    if cached:current=cached
                    else:
                        one=time.process_time()
                        result=mixture_raw_lookahead(model,terminal,remaining,[candidate],
                            draw_power=settings['terminal_power'],fantasy_power=power,seed=settings['seed'],
                            baseline=baseline,fixed_action_expectations={candidate:old['fixed_action_expected_reward']})
                        row=result['lookahead'][0]
                        prefix_error=abs(row['nested_prefix_adaptation_value']-current['adaptation_value'])
                        if prefix_error>1e-8:
                            raise RuntimeError('Refined nested fantasies changed the preceding contrast')
                        identity=max(abs(v) for v in row['model_weight_martingale_residual'])
                        passed=bool(identity<=precision['maximum_identity_martingale_residual']
                            and row['maximum_raw_integration_relative_error']<=precision['maximum_raw_integration_relative_error']
                            and row['nested_adaptation_difference']<=precision['maximum_nested_adaptation_difference'])
                        current={'layout':layout,'stop_expected_reward':result['stop_expected_reward'],**row,
                            'precision_gate_passed':passed,'joint_fantasy_power':power,
                            'fixed_action_reference_precision_passed':True,
                            'reference_fantasy_power':old['reference_fantasy_power'],
                            'maximum_identity_martingale_residual':identity,'nested_prefix_reproduction_error':prefix_error,
                            'cpu_seconds':time.process_time()-one,'source_law':old['source_law']}
                        rows.append(current);save(path,{'execution_plan':plan,'rows':rows,
                            'all_registered_refinements_complete':False})
                        print('CV refinement',layout,candidate,2**power,'net',round(row['net_reward_of_paid_episode'],6),
                            'contrast difference',round(row['nested_adaptation_difference'],6),
                            'identity',round(identity,6),'pass',passed,flush=True)
                    if current['precision_gate_passed']:break
    except TimeoutError:stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
    latest=[next((r for r in reversed(rows) if r['layout']==old['layout'] and r['candidate']==old['candidate']),old)
        for old in parent['rows']]
    complete=all(r['precision_gate_passed'] or r.get('joint_fantasy_power')==spec['joint_fantasy_powers'][-1] for r in latest)
    comparison=[]
    for state,*_ in models:
        group=[r for r in latest if r['layout']==state['layout']]
        best=max(group,key=lambda r:r['net_reward_of_paid_episode'])
        comparison.append({'layout':state['layout'],'completed_action_count':len(group),
            'all_actions_precision_passed':all(r['precision_gate_passed'] for r in group),
            'provisional_best_candidate':best['candidate'],'provisional_best_net':best['net_reward_of_paid_episode']})
    result={'execution_plan':plan,'parent_results':spec['parent_results'],'rows':rows,'latest_rows':latest,
        'comparison':comparison,'all12_registered_actions_complete':True,
        'all_registered_refinements_complete':complete,'actions_precision_passed':sum(r['precision_gate_passed'] for r in latest),
        'cpu_seconds':time.process_time()-started,'stopped_at_cpu_budget':stopped,'cost':spec['cost'],
        'no_model_adopted':True,'independent_confirmation':False,'scientific_model_ranking_authorized':False,
        'limitations':parent['limitations']}
    save(path,result);return result


if __name__=='__main__':run()
