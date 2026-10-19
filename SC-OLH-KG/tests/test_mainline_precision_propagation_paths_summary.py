"""Small saved-path substitutes preserve pairing, signed losses and source results."""
from copy import deepcopy
import json

import pytest

from performance import analyze_mainline_precision_propagation_paths as runner


def plan_fixture(count=2):
    arms = [runner.AFTER, runner.POINT, runner.FULL, runner.AFTER+'_native_terminal',
        runner.FULL+'_native_terminal', 'legacy_native_neighbors', 'same_source_gp',
        'current_structural_mixture', 'initialization_only']
    return dict(output_dir='paths', source_manifest='source.json', source_output_dir='science',
        tasks=[dict(geometry_task_seed=i+1, domain='family', geometry={'shift': i}) for i in range(count)],
        streams=[dict(name='observation_'+str(i), manifest=f'child_{i}.json', replicate=i) for i in (1, 2)],
        terminal_arms=[dict(name=arm, history_arm=arm.removesuffix('_native_terminal')) for arm in arms],
        statistical_unit='geometry mean of two streams', historical_accounts=['science/completion.json'], limits=[])


def metrics(gain=0.):
    return dict(A=1, C=1, false_certificate=0, R=.5+gain, G=gain, P=max(0., gain), L=max(0., -gain))


def comparison(contrast, prefix):
    return dict(contrast=contrast, paid_prefix=prefix, status='available', same_paid_history=prefix == 10,
        same_physical_pool=True, query_changed=prefix == 10, first_divergence=prefix == 10,
        common_state_first_divergence=prefix == 10, selected={runner.AFTER:'P1', runner.POINT:'P2'},
        focal_policies=[dict(policy_id='P1', left={'eligible':True}, right={'eligible':False},
            eligibility_transition='left_only_eligible',
            saved_left_minus_right=dict(objective_mean=.2, sampled_objective=.5))],
        common_policy_changes=[dict(policy_id='unselected')])


def task_fixture(plan, task):
    return dict(seed=task['geometry_task_seed'], domain=task['domain'], status='complete',
        streams=[dict(name=stream['name'], status='complete',
            outcomes=[dict(arm=arm['name'], protocol='old', status='complete', metrics=metrics(),
                discovery_category='initial', returned_repeat_count=0, actual_deployed_quality=.5)
                for arm in plan['terminal_arms']],
            query_comparisons=[comparison(contrast, prefix) for contrast in ('AP','FA') for prefix in (10,11,12)],
            queries=[], multi_safe_events=[], policy_terminal_paths=[], policy_paths=[])
            for stream in plan['streams']])


def outcome(task, stream, arm=runner.AFTER):
    return next(row for row in task['streams'][stream]['outcomes'] if row['arm'] == arm)


def terminal(reason, slot=3, deployed=False):
    return dict(disposition='objective_guard_retained_other_policy' if 'comparison' in reason else reason,
        terminal_reason=reason, shortlist_position=slot, safety_attempted='comparison' in reason,
        safety_passed='comparison' in reason, deployed=deployed)


def source_fixture(root, plan):
    runner.save(root/plan['source_manifest'], dict(plan, status='complete'))
    runner.save(root/'science/completion.json', {'status':'complete'})
    runner.save(root/'science/execution_status.json', dict(status='sealed', truth_access_allowed=True,
        cases=[dict(stream=stream['name'], seed=task['geometry_task_seed'], arm=arm['name'], old={'status':'complete'})
            for stream in plan['streams'] for task in plan['tasks'] for arm in plan['terminal_arms']]))
    runner.save(root/'science/mechanism/inventory.json', dict(status='sealed', inventory_complete=True))
    headline = dict(status='complete', registered_geometries=len(plan['tasks']),
        registered_terminal_cells=18*len(plan['tasks']), saved_C=0.125, saved_G=None,
        defined_G=134, source_costs=dict(physical_target_calls=609888))
    runner.save(root/'science/headline.json', headline)
    for task in plan['tasks']:
        runner.save(root/'science/cases'/f"seed_{task['geometry_task_seed']}"/'result.json', {'status':'complete'})
    return headline


def test_registration_requires_full_source_seal_and_identical_geometry(tmp_path):
    plan = plan_fixture(1)
    source_fixture(tmp_path, plan)
    assert runner.require_source(plan, root=tmp_path)['registered_terminal_outcomes'] == 18
    altered = deepcopy(plan)
    altered['tasks'][0]['geometry']['shift'] = 9
    with pytest.raises(ValueError, match='metadata changed'):
        runner.require_source(altered, root=tmp_path)
    path = tmp_path/'science/execution_status.json'
    seal = runner.read(path)
    seal['cases'].pop()
    runner.save(path, seal)
    with pytest.raises(ValueError, match='globally sealed'):
        runner.require_source(plan, root=tmp_path)


def test_signed_multi_contributions_include_point_gains_after_losses_and_geometry_pairing():
    plan = plan_fixture()
    tasks = [task_fixture(plan, task) for task in plan['tasks']]
    outcome(tasks[0],0).update(metrics=metrics(.4), discovery_category=runner.MULTI)
    outcome(tasks[0],1).update(metrics=metrics(-.1), discovery_category=runner.MULTI)
    outcome(tasks[0],1,runner.POINT).update(metrics=metrics(.2), discovery_category=runner.MULTI)
    result = runner.summarize(tasks, plan, {'source':'unchanged'})
    key = 'AP.multi_signed.net_gain'
    assert result['streams'][0]['measures'][key]['all_registered_mean'] == .2
    assert result['streams'][1]['measures'][key]['all_registered_mean'] == pytest.approx(-.15)
    assert result['geometry_mean']['measures'][key]['registered_geometries'] == 2
    assert result['geometry_mean']['measures'][key]['all_registered_mean'] == pytest.approx(.025)
    assert result['multi_safe_signed_directions'][0]['counts'] == {'positive/negative':1,'zero/zero':1}
    assert result['geometry_mean']['rows'][0]['values']['AP.multi_signed.avoided_loss_component'] == -.05


def test_repeated_multi_query_is_one_policy_for_terminal_conversion():
    plan = plan_fixture(1)
    stream = task_fixture(plan, plan['tasks'][0])['streams'][0]
    path = terminal('not_verified_after_earlier_terminal_decision')
    stream['multi_safe_events'] = [dict(arm=runner.AFTER, policy_id='P7', paid_prefix=prefix,
        query_action_kind=kind, observation_returned=True, discovery_category=runner.MULTI,
        own_terminal_path=path, discovery_gain=.02) for prefix,kind in ((10,'new'),(12,'repeat'))]
    values = runner.stream_measures(stream)
    prefix = runner.AFTER+'.multi.'
    assert values[prefix+'returned_query_events'] == 2
    assert values[prefix+'distinct_returned_policies'] == 1
    assert values[prefix+'new_queries'] == 1 and values[prefix+'repeat_queries'] == 1
    assert values[prefix+'on_shortlist'] == 1 and values[prefix+'safety_attempted'] == 0
    assert values[prefix+'terminal_reason.not_verified_after_earlier_terminal_decision'] == 1
    assert values['AP.multi_signed.net_gain'] == 0.  # A discovered point gain is not deployment contribution.


def test_better_observed_policy_keeps_missing_baseline_and_actual_comparison_coverage_distinct():
    rows = [dict(truth={'true_feasible':True,'quality':.6}, quality_gain_to_initial=None,
        terminal_path=terminal(reason)) for reason in ('not_verified_after_earlier_terminal_decision',
            'safety_certified_without_actual_objective_comparison', 'actual_objective_comparison_retained_other_policy')]
    counts = runner.terminal_counts(rows, .5)
    assert counts['known_safe_better_than_deployment'] == 3
    assert counts['improving_safe_unknown'] == 3
    assert counts['better_than_deployment.not_verified_after_earlier_terminal_decision'] == 1
    assert counts['better_than_deployment.safety_certified_without_actual_objective_comparison'] == 1
    assert counts['better_than_deployment.actual_objective_comparison_retained_other_policy'] == 1
    assert runner.terminal_counts(rows)['better_than_deployment_unknown'] == 3


def test_same_state_first_divergence_and_saved_innovation_do_not_become_temporal_updates():
    plan = plan_fixture(1)
    stream = task_fixture(plan, plan['tasks'][0])['streams'][0]
    values = runner.stream_measures(stream)
    assert values['AP.prefix10_same_history_pool'] == 1
    assert values['AP.common_state_first_divergence'] == 1
    assert values['AP.later_diverged_history_prefixes'] == 2
    first = runner.compact_initial(stream)['comparisons'][0]
    assert first['focal_policies'][0]['sampling_innovation_difference'] == pytest.approx(.3)
    assert first['temporal_initialization_update'] is None
    assert first['pre_initialization_predictions'] is None
    assert 'common_policy_changes' not in first


def test_path_failure_and_unknown_baseline_do_not_erase_source_science():
    plan = plan_fixture()
    tasks = [task_fixture(plan, task) for task in plan['tasks']]
    tasks[0]['streams'][0]['status'] = 'path_analysis_failed'
    for stream in (0,1):
        outcome(tasks[1],stream)['metrics'].update(G=None,P=None,L=None)
    source = {'registered_geometries':144, 'G_mean':None, 'defined_G':134, 'costs':{'paid':609888}}
    result = runner.summarize(tasks, plan, source)
    assert result['saved_science'] == source
    assert result['status'] == 'complete_with_analysis_failures'
    assert result['registered_terminal_outcomes'] == 36
    stat = result['geometry_mean']['measures']['AP.prefix10_query_changed']
    assert stat['all_registered_mean'] is None and stat['available_geometries'] == 1
    assert result['geometry_mean']['measures']['AP.multi_signed.net_gain']['all_registered_mean'] is None
    assert len(tasks[0]['streams'][0]['outcomes']) == 9


def test_collect_preserves_failed_analysis_cpu_and_zero_new_work_without_recharging_source(tmp_path):
    plan = plan_fixture(1)
    source = source_fixture(tmp_path, plan)
    runner.save(tmp_path/'checks.json', {'status':'passed'})
    runner.register(plan, 'plan.json', 'checks.json', root=tmp_path)
    task = runner.failed_task(plan, plan['tasks'][0], 'retained failure')
    runner.save(tmp_path/'paths/cases/seed_1/result.json', task)
    ledger = tmp_path/'paths/cases/seed_1/attempts.jsonl'
    ledger.write_text(json.dumps(dict(stage='analyze-precision-propagation-paths', status='failed',
        cpu_seconds=2., **runner.ZERO))+'\n')
    completion = runner.collect(plan, 'plan.json', root=tmp_path)
    assert completion['status'] == 'complete_with_analysis_failures'
    assert len(completion['failed_attempts']) == 1 and completion['recorded_analysis_cpu_seconds'] >= 2.
    assert all(value == 0 for value in completion['new_work'].values())
    assert runner.read(tmp_path/'paths/headline.json')['saved_science'] == source
    assert runner.read(tmp_path/'science/headline.json') == source
    assert runner.read(tmp_path/'plan.json')['status'] == completion['status']
