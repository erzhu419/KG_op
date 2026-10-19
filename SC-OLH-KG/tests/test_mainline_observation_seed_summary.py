"""Only synthetic JSON and a family-scoring stub; no model or truth execution."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import t

from performance import summarize_mainline_observation_seed_replication as summary
from performance.run_mainline_legacy_backend_bridge import read, save
from performance.run_mainline_terminal_equal import case_dir


def synthetic_batch(tmp_path, geometry_count=12, replicates=(1, 2), fresh=False):
    names = [summary.FULL, summary.FULL+'_native_terminal', summary.AFTER,
             summary.AFTER+'_native_terminal', 'legacy_native_neighbors', 'same_source_gp',
             summary.POINT, 'current_structural_mixture', 'initialization_only']
    arms = [dict(name=arm, history_arm=arm.removesuffix('_native_terminal'),
                 terminal_backend='stub') for arm in names]
    plan = dict(output_dir=str(tmp_path/'parent'), terminal_arms=arms, tasks=[],
                replicates=[dict(replicate=replicate) for replicate in replicates])
    prepared = dict(replicates=[])
    returned, terminal_plans = {}, []
    for index in range(geometry_count):
        old = tmp_path/'old'/str(index)
        save(old/'truth.json', [dict(point=[point], truth=dict(objective=.4, quality=.6,
             true_feasible=True)) for point in range(10)])
        task = dict(domain=f'family_{index//(geometry_count//3)}', geometry_task_seed=1000+index)
        if not fresh:
            task['initial_policy_file'] = str(old/'initialization.json')
        plan['tasks'].append(task)
    for replicate in replicates:
        rep = dict(replicate=replicate, output_dir=str(tmp_path/f'rep_{replicate}'), families=[])
        prepared['replicates'].append(rep)
        for family_index in range(3):
            domain = f'family_{family_index}'
            output = Path(rep['output_dir'])/domain
            seeds = [task['geometry_task_seed'] for task in plan['tasks'] if task['domain'] == domain]
            source = dict(output_dir=str(output/'search'), target_seeds=seeds)
            terminal = dict(output_dir=str(output/'terminal'), target_seeds=seeds, arms=arms,
                protocols={summary.PROTOCOL: [80, 128, 128]}, execution_failures=[])
            source_path, terminal_path = output/'source.json', output/'terminal.json'
            rep['families'].append(dict(domain=domain, search_manifest=str(source_path),
                                        terminal_manifest=str(terminal_path)))
            rows = []
            for seed in seeds:
                directory = Path(source['output_dir'])/f'seed_{seed}'
                save(directory/'initialization.json', dict(history=[dict(point=[point], raw=[999,999])
                                                                  for point in range(10)]))
                np.savez(directory/'neighbors.npz', points=[[100 if seed % 2 else 200]])
                for arm in arms:
                    name = arm['name']
                    if replicate == 2 and seed == 999+geometry_count and name == 'current_structural_mixture':
                        terminal['execution_failures'].append(dict(seed=seed, arm=name))
                        continue
                    certified = not (replicate == 2 and seed == 1000 and name == summary.AFTER)
                    gain = (replicate * .001*(seed-999) if name == summary.AFTER else 0.)
                    if seed == 1000 and name == summary.AFTER:
                        gain = -.05 if replicate == 1 else .39
                    baseline = .7 if fresh else .6
                    truth = dict(quality=baseline+gain, objective=1-baseline-gain, true_feasible=True)
                    rows.append(dict(seed=seed, arm=name, certified=certified,
                        certified_true_feasible=certified, false_certificate=False,
                        actual_safe_reward=truth['quality'] if certified else 0.,
                        deployed=[100 if gain else 0], deployed_truth=truth,
                        new_verification_calls=80, logical_total_calls=93,
                        certified_feasible_quality_gain=gain if certified else 0.))
                    save(directory/arm['history_arm']/'search.json',
                         dict(new_initial_calls=10 if name == 'initialization_only' else 0,
                              new_search_calls=0 if name == 'initialization_only' else 3))
                    save(case_dir(Path(terminal['output_dir']), seed, name, summary.PROTOCOL)/
                         'verification.json', dict(status='synthetic'))
            save(source_path, source)
            save(terminal_path, terminal)
            save(Path(source['output_dir'])/'completion.json', dict(cost=dict(new_target_initial_calls=40)))
            returned[str(Path(terminal['output_dir']))] = dict(status='complete', rows=rows)
            terminal_plans.append(terminal)
    return plan, prepared, returned, terminal_plans


@pytest.mark.parametrize('fresh,geometry_count,replicates', [(False, 12, (1, 2)), (True, 24, (1,))])
def test_global_barrier_precedes_all_truth_cache_reads_and_copies(tmp_path, fresh, geometry_count, replicates):
    plan, prepared, returned, terminals = synthetic_batch(tmp_path, geometry_count, replicates, fresh)
    terminal = terminals[-1]
    case_dir(Path(terminal['output_dir']), 999+geometry_count, summary.AFTER, summary.PROTOCOL).joinpath(
        'verification.json').unlink()
    reads = []

    def read_without_truth(path):
        reads.append(str(path))
        assert Path(path).name != 'truth.json'
        return read(path)

    with patch.object(summary, 'read', side_effect=read_without_truth), \
         patch.object(summary, 'copyfile', side_effect=AssertionError('truth copied before global barrier')), \
         patch.object(summary, 'summarize_family', side_effect=AssertionError('scored before barrier')), \
         patch.object(summary, 'require_all', wraps=summary.require_all) as barriers:
        with pytest.raises(RuntimeError, match='verification.json'):
            summary.score(plan, prepared)
    assert barriers.call_count == len(terminals)
    assert len(reads) == 2*len(terminals)  # Plans only; even the first complete family stays unopened.


def test_geometry_first_statistics_keep_missing_and_gate_uncertified(tmp_path):
    plan, prepared, returned, terminals = synthetic_batch(tmp_path)
    events = []
    actual_barrier = summary.require_all

    def barrier(*args):
        actual_barrier(*args)
        events.append('barrier')

    def fake_family(output, terminal):
        assert events[:6] == ['barrier']*6
        events.append('score')
        # Every family's truth cache must already be copied before collection begins.
        for rep in prepared['replicates']:
            for family in rep['families']:
                source = read(family['search_manifest'])
                for seed in source['target_seeds']:
                    assert (Path(source['output_dir'])/f'seed_{seed}'/'truth.json').is_file()
        save(output/'completion.json', dict(cost=dict(new_verification_calls=80)))
        return returned[str(output)]

    with patch.object(summary, 'require_all', side_effect=barrier), \
         patch.object(summary, 'summarize_family', side_effect=fake_family):
        result = summary.score(plan, prepared)
    assert result['registered_cells'] == 216 and result['completed_cells'] == 215
    assert len(result['rows']) == 216
    absent = next(row for row in result['rows'] if row['status'] == 'execution_failure')
    assert all(absent[key] is None for key in summary.METRICS)
    assert absent['logical_total_calls'] is None
    assert all(row['logical_total_calls'] == 93 for row in result['rows']
               if row['status'] == 'complete')
    assert all('expanded_logical_calls' not in row for row in result['rows'])
    uncertified = next(row for row in result['rows'] if row['replicate'] == 2 and
                       row['seed'] == 1000 and row['arm'] == summary.AFTER)
    assert uncertified['certification'] == uncertified['quality_increment'] == uncertified['reward'] == 0
    assert uncertified['good_new_deployment'] == 0
    assert all(uncertified[key] == 0 for key in ('new_policy_gain', 'below_best_regression',
        'dynamic_new_gain', 'shared_new_gain', 'dynamic_good_new_deployment'))
    primary = result['primary']
    values = [-.025]+[1.5*.001*(index+1) for index in range(1,12)]
    np.testing.assert_allclose([row['quality_increment'] for row in primary['geometry_mean_effects']], values)
    np.testing.assert_allclose(primary['task_equal_means']['quality_increment'], np.mean(values))
    np.testing.assert_allclose(primary['family_equal_means']['quality_increment'], np.mean(values))
    half_width = t.ppf(.975, 11)*np.std(values, ddof=1)/np.sqrt(12)
    np.testing.assert_allclose(primary['descriptive_95pct_t_interval']['quality_increment'],
                               [np.mean(values)-half_width, np.mean(values)+half_width])
    assert primary['degrees_of_freedom'] == 11
    missing_contrast = next(row for row in result['comparisons'] if row['right'] == 'current_structural_mixture')
    assert missing_contrast['geometry_mean_effects'][-1]['reward'] is None
    assert missing_contrast['task_equal_means']['reward'] is None
    assert missing_contrast['family_equal_means']['reward'] is None
    assert missing_contrast['descriptive_95pct_t_interval']['reward'] is None
    assert len(missing_contrast['geometry_mean_effects']) == 12
    assert primary['by_replicate'][0]['rows'][0]['quality_increment'] == pytest.approx(-.05)
    for terminal in terminals:
        assert read(Path(terminal['output_dir'])/'results.json')['scope'] == summary.SCOPE


def test_fresh_single_stream_owns_truth_and_decomposes_signed_quality(tmp_path):
    plan, prepared, returned, terminals = synthetic_batch(tmp_path, 24, (1,), fresh=True)
    plan.update(scope='Frozen unseen-geometry confirmation.',
        statistical_unit='24 fresh geometries in three known families, one stream each.',
        interval_scope='Fixed registered unseen-geometry cohort, not within-geometry noise replication.')
    events = []
    actual_barrier = summary.require_all

    def barrier(*args):
        actual_barrier(*args)
        events.append('barrier')

    def fake_family(output, terminal):
        assert events[:3] == ['barrier']*3
        events.append('score')
        family = next(row for row in prepared['replicates'][0]['families']
                      if read(row['terminal_manifest'])['output_dir'] == str(output))
        source = read(family['search_manifest'])
        for seed in source['target_seeds']:
            path = Path(source['output_dir'])/f'seed_{seed}'/'truth.json'
            assert not path.exists(), 'New geometry must not inherit an old geometry truth cache'
            save(path, [dict(point=[point], truth=dict(objective=.3, quality=.7,
                  true_feasible=True)) for point in range(10)])
        save(output/'completion.json', dict(cost=dict(new_verification_calls=80)))
        return returned[str(output)]

    with patch.object(summary, 'require_all', side_effect=barrier), \
         patch.object(summary, 'copyfile', side_effect=AssertionError('fresh geometry copied old truth')), \
         patch.object(summary, 'summarize_family', side_effect=fake_family):
        result = summary.score(plan, prepared)
    assert result['registered_cells'] == result['completed_cells'] == 216
    assert result['scope'] == plan['scope'] and result['statistical_unit'] == plan['statistical_unit']
    assert 'both' not in result['weighting']
    primary = result['primary']
    assert primary['degrees_of_freedom'] == 23 and len(primary['geometry_mean_effects']) == 24
    assert primary['interval_scope'] == plan['interval_scope']
    values = [-.05]+[.001*(index+1) for index in range(1, 24)]
    np.testing.assert_allclose([row['quality_increment'] for row in primary['geometry_mean_effects']], values)
    for method in ('task_equal_means', 'family_equal_means'):
        assert primary[method]['quality_increment'] == pytest.approx(np.mean(values))
    half_width = t.ppf(.975, 23)*np.std(values, ddof=1)/np.sqrt(24)
    np.testing.assert_allclose(primary['descriptive_95pct_t_interval']['quality_increment'],
                              [np.mean(values)-half_width, np.mean(values)+half_width])
    for row in result['rows']:
        assert row['best_feasible_initial_truth']['quality'] == .7
        assert row['quality_increment'] == pytest.approx(row['new_policy_gain']-row['below_best_regression'])
        assert row['new_policy_gain'] == pytest.approx(row['dynamic_new_gain']+row['shared_new_gain'])
    after = {row['seed']: row for row in result['rows'] if row['arm'] == summary.AFTER}
    assert after[1000]['new_policy_gain'] == 0 and after[1000]['below_best_regression'] == pytest.approx(.05)
    assert after[1001]['shared_new_gain'] == pytest.approx(.002)
    assert after[1001]['dynamic_good_new_deployment'] == 0
    assert after[1002]['dynamic_new_gain'] == pytest.approx(.003)
    assert after[1002]['dynamic_good_new_deployment'] == 1


def test_no_feasible_initial_keeps_observed_outcomes_without_fabricating_quality(tmp_path):
    plan, prepared, returned, _ = synthetic_batch(tmp_path, 24, (1,), fresh=True)

    def fake_family(output, terminal):
        family = next(row for row in prepared['replicates'][0]['families']
                      if read(row['terminal_manifest'])['output_dir'] == str(output))
        source = read(family['search_manifest'])
        for seed in source['target_seeds']:
            save(Path(source['output_dir'])/f'seed_{seed}'/'truth.json', [
                dict(point=[point], truth=dict(objective=.3, quality=.7,
                     true_feasible=seed != 1000)) for point in range(10)])
        result = returned[str(output)]
        for row in result['rows']:
            if row['seed'] == 1000 and row['arm'] == summary.POINT:
                row.update(certified=False, certified_true_feasible=False, actual_safe_reward=0.)
        save(output/'completion.json', dict(cost=dict(new_verification_calls=80)))
        return result

    with patch.object(summary, 'summarize_family', side_effect=fake_family):
        result = summary.score(plan, prepared)
    assert result['registered_cells'] == result['completed_cells'] == 216
    undefined = [row for row in result['rows'] if row['seed'] == 1000]
    assert len(undefined) == 9
    for row in undefined:
        assert row['status'] == 'complete'
        assert row['quality_reference_status'] == 'no_true_feasible_initial'
        assert row['best_feasible_initial_truth'] is None
        assert all(row[key] is None for key in ('best_initial_reference', 'quality_increment',
            'new_policy_gain', 'below_best_regression', 'dynamic_new_gain', 'shared_new_gain',
            'dynamic_good_new_deployment', 'good_new_deployment'))
        assert all(row[key] is not None for key in ('certification', 'reward', 'actual_certificate',
            'false_certificate', 'physical_verification_calls', 'logical_total_calls'))
    point = next(row for row in undefined if row['arm'] == summary.POINT)
    assert point['certification'] == point['reward'] == 0
    assert result['primary']['task_equal_means']['certification'] == pytest.approx(1/24)
    for comparison in result['comparisons']:
        assert comparison['task_equal_means']['quality_increment'] is None
        assert comparison['descriptive_95pct_t_interval']['quality_increment'] is None
    full_after = next(row for row in result['comparisons'] if row['left'] == summary.FULL)
    paired = full_after['jointly_certified_pairs_conditional'][0]
    assert paired['completed_pairs'] == 24
    assert paired['quality_reference_missing_seeds'] == [1000]
    assert len(paired['quality_differences']) == 23
