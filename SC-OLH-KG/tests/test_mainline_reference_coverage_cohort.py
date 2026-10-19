"""Four-terminal barriers use saved metadata, retaining every registered cell."""
from copy import deepcopy

import pytest

from performance import mainline_reference_coverage_cohort as cohort


def setup_cohort(root, seeds=range(24)):
    seeds = list(seeds)
    arms = [cohort.STOP] + [f'online_{i}' for i in range(6)]
    cells = [dict(name=arm, history_arm=arm, terminal_backend='role') for arm in arms]
    cells += [dict(name=f'online_{i}_native', history_arm=f'online_{i}',
                   terminal_backend='native') for i in range(2)]
    plan = dict(output_dir='cohort', tasks=[dict(geometry_task_seed=s) for s in seeds],
                search_arms=arms, terminal_arms=cells,
                historical_accounts=['historical_failed_attempts.json'])
    source = dict(domain='fixture', output_dir='search', target_seeds=seeds,
        task=dict(d=1000, alpha=.05), arms=[dict(name=arm) for arm in arms],
        geometry_by_seed={str(s): dict(problem_kwargs=dict(shift=s)) for s in seeds},
        decision_protocol=dict(verification_method='gaussian'))
    terminal = dict(target_seeds=seeds, arms=cells,
        verification_seed_by_task={str(s): 1000+s for s in seeds})
    cohort.save(root/'source.json',source)
    cohort.save(root/'terminal.json',terminal)
    cohort.save(root/'cohort/prepared.json',dict(replicates=[dict(replicate=1,
        families=[dict(search_manifest='source.json',terminal_manifest='terminal.json')])]))
    for seed in seeds:
        write_search(root, source, seed, cohort.STOP)
    return plan, source


def write_search(root, source, seed, arm):
    count = 10 if arm == cohort.STOP else 13
    shortlist = [dict(point=[i],shortlist_position=i+1,shortlist_role='fixture',
                      posterior_rank=i+1) for i in range(3)]
    recording = dict(status='frozen',paid_prefix=count,
        observed=[dict(point=[i],observation_indices=[i]) for i in range(10)],
        stages=[dict(phase='primary_pool',pool=[[i] for i in range(10)],
            components=dict(probability_violation=[(10-i)/10 for i in range(10)],
                            objective=list(range(10))))])
    frozen = dict(frozen_shortlist=shortlist,objective_incumbent_guard_enabled=True,
        candidate_verification_budgets=[80,128,128],max_verification_budget=352,
        objective_comparison_budget_per_policy=8,familywise_delta=.05,
        objective_comparison=dict(delta=.05/3),mean_delta_fraction=.5,
        shortlist_mode='fixture',objective_incumbent_position=2,
        terminal_recording=recording)
    search = dict(geometry=source['geometry_by_seed'][str(seed)]['problem_kwargs'],
        logical_search_calls=count,history=[dict(point=[i%10],raw=[i]) for i in range(count)],
        terminal=frozen)
    directory=root/source['output_dir']/f'seed_{seed}'/arm
    cohort.save(directory/'search.json',search)
    if arm in ('online_0','online_1'):
        cohort.save(directory/'native_terminal.json',dict(terminal=deepcopy(frozen)))


def write_online(root, source):
    failed={(r['seed'],r['arm']) for r in source.get('execution_failures',[])}
    for seed in source['target_seeds']:
        for arm in source['arms']:
            if arm['name'] != cohort.STOP and (seed,arm['name']) not in failed:
                write_search(root,source,seed,arm['name'])


def record_outcomes(root, frozen, *, failed_key=None):
    for row in frozen['rows']:
        if row['status'] != 'frozen':
            continue
        directory=root/row['execution_dir']
        charged=False
        for variant in cohort.VARIANTS:
            failed=(row['seed'],row['arm'],variant)==failed_key
            result=dict(status='failed' if failed else 'complete',
                        logical_verification_calls=3 if failed else 0,
                        cumulative_new_verification_calls=3 if failed else 0,
                        cumulative_failed_simulation_calls=int(failed))
            cohort.save(directory/variant/('failure.json' if failed else 'verification.json'),result)
            charged |= failed
        cohort.save(directory/'execution.json',dict(status='complete_with_failures' if charged else 'complete',
            physical_verification_calls=3 if charged else 0,failed_simulation_calls=int(charged)))


def test_full_24_geometry_grid_freezes_prefix_orders_before_864_inputs(tmp_path):
    plan,source=setup_cohort(tmp_path)
    first=cohort.freeze_initial_references(plan,root=tmp_path)
    assert first['frozen_references']==24
    assert first['new_target_calls']==first['new_candidate_scores']==first['posterior_restorations']==0
    order=cohort.read(tmp_path/first['rows'][0]['order_file'])
    assert order['order'][0]['point']==[9]
    assert order['selection_uses_truth'] is False
    write_online(tmp_path,source)
    cohort.require_initial_references(plan,seed=0,root=tmp_path)
    frozen=cohort.freeze_cohort_inputs(plan,root=tmp_path)
    assert len(cohort.read(tmp_path/'cohort/search_status.json')['rows'])==168
    assert frozen['registered_arm_inputs']==216
    assert frozen['registered_terminal_cells']==864
    assert all(tuple(row['variants'])==cohort.VARIANTS for row in frozen['rows'])
    row=frozen['rows'][0]
    old=cohort.read(tmp_path/row['variants']['original_old']['input_file'])
    reference=cohort.read(tmp_path/row['variants']['reference_old']['input_file'])
    assert old['frozen_shortlist'][:2]==reference['frozen_shortlist'][:2]
    assert reference['frozen_shortlist'][2]['point']==[9]
    assert cohort.freeze_initial_references(plan,root=tmp_path)==first
    assert cohort.freeze_cohort_inputs(plan,root=tmp_path)==frozen


def test_missing_stop_requires_recorded_failure_and_retains_all_dependent_cells(tmp_path):
    plan,source=setup_cohort(tmp_path)
    (tmp_path/'search/seed_0/initialization_only/search.json').unlink()
    with pytest.raises(ValueError,match='Unfinished STOP'):
        cohort.freeze_initial_references(plan,root=tmp_path)
    source['execution_failures']=[dict(seed=0,arm=arm['name'],status='initialization_dependency_failed',
        paid_initial_calls=7,failure_file='retained_initial_failure.json') for arm in source['arms']]
    cohort.save(tmp_path/'source.json',source)
    reference=cohort.freeze_initial_references(plan,root=tmp_path)
    assert reference['registered_geometries']==24 and reference['frozen_references']==23
    with pytest.raises(ValueError,match='no frozen initial'):
        cohort.require_initial_references(plan,seed=0,root=tmp_path)
    write_online(tmp_path,source)
    frozen=cohort.freeze_cohort_inputs(plan,root=tmp_path)
    missing=[r for r in frozen['rows'] if r['status']=='missing']
    assert len(missing)==9 and frozen['registered_terminal_cells']==864
    assert all(len(row['variants'])==4 for row in missing)
    assert missing[0]['variants']['original_old']['failure']['reference']['failure']['paid_initial_calls']==7


def test_single_history_and_terminal_failures_keep_nulls_costs_and_truth_barrier(tmp_path):
    plan,source=setup_cohort(tmp_path)
    cohort.freeze_initial_references(plan,root=tmp_path)
    source['execution_failures']=[dict(seed=0,arm='online_2',status='search_failed',
                                      paid_search_calls=2)]
    cohort.save(tmp_path/'source.json',source)
    write_online(tmp_path,source)
    frozen=cohort.freeze_cohort_inputs(plan,root=tmp_path)
    assert sum(row['status']=='missing' for row in frozen['rows'])==1
    with pytest.raises(ValueError,match='barrier is absent'):
        cohort.require_truth_seal(plan,root=tmp_path)
    with pytest.raises(ValueError,match='Unfinished four-terminal'):
        cohort.seal_cohort_results(plan,root=tmp_path)
    cohort.save(tmp_path/'cohort/execution_status.json',dict(status='running',truth_access_allowed=False))
    with pytest.raises(ValueError,match='barrier is incomplete'):
        cohort.require_truth_seal(plan,root=tmp_path)
    failed=(1,'online_3','reference_two_comparison')
    record_outcomes(tmp_path,frozen,failed_key=failed)
    seal=cohort.seal_cohort_results(plan,root=tmp_path)
    assert (seal['registered_terminal_cells'],seal['completed_cells'],
            seal['failed_cells'],seal['missing_cells'])==(864,859,1,4)
    assert seal['physical_verification_calls']==3 and seal['failed_simulation_calls']==1
    assert seal['historical_accounts']==plan['historical_accounts']
    assert all(row['C'] is None and row['G'] is None for row in seal['cells'])
    assert seal['truth_used'] is False
    assert cohort.require_truth_seal(plan,root=tmp_path)==seal


def test_missing_saved_stop_scores_retains_failure_without_discarding_other_geometry(tmp_path):
    plan,_=setup_cohort(tmp_path,seeds=[0,1])
    path=tmp_path/'search/seed_0/initialization_only/search.json'
    search=cohort.read(path)
    del search['terminal']['terminal_recording']['stages'][0]['components']['objective']
    cohort.save(path,search)
    result=cohort.freeze_initial_references(plan,root=tmp_path)
    assert result['registered_geometries']==2 and result['frozen_references']==1
    assert result['rows'][0]['failure']['status']=='initial_reference_admission_failed'
    cohort.require_initial_references(plan,seed=1,root=tmp_path)
    assert cohort.read(path)==search


def test_online_history_blocks_late_reference_freezing(tmp_path):
    plan,source=setup_cohort(tmp_path,seeds=[0])
    write_search(tmp_path,source,0,'online_0')
    with pytest.raises(ValueError,match='precede online'):
        cohort.freeze_initial_references(plan,root=tmp_path)
    assert not (tmp_path/'cohort/initial_references.json').exists()


def test_unfinished_history_or_native_input_cannot_be_sealed_as_failure(tmp_path):
    plan,source=setup_cohort(tmp_path,seeds=[0])
    cohort.freeze_initial_references(plan,root=tmp_path)
    with pytest.raises(ValueError,match='Unfinished search history'):
        cohort.freeze_cohort_inputs(plan,root=tmp_path)
    write_online(tmp_path,source)
    (tmp_path/'search/seed_0/online_0/native_terminal.json').unlink()
    with pytest.raises(ValueError,match='Unfinished native input'):
        cohort.freeze_cohort_inputs(plan,root=tmp_path)
    assert not (tmp_path/'cohort/frozen_inputs.json').exists()


def test_missing_one_result_blocks_truth_even_with_arm_accounting(tmp_path):
    plan,source=setup_cohort(tmp_path,seeds=[0])
    cohort.freeze_initial_references(plan,root=tmp_path)
    write_online(tmp_path,source)
    frozen=cohort.freeze_cohort_inputs(plan,root=tmp_path)
    record_outcomes(tmp_path,frozen)
    path=tmp_path/frozen['rows'][-1]['execution_dir']/cohort.VARIANTS[-1]/'verification.json'
    path.unlink()
    with pytest.raises(ValueError,match='Unfinished terminal cell'):
        cohort.seal_cohort_results(plan,root=tmp_path)
    assert not (tmp_path/'cohort/execution_status.json').exists()
