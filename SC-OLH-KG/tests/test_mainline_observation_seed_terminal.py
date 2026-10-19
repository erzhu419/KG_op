"""Missing baselines remain distinct from zero gain and failed execution."""
import numpy as np

from performance import diagnose_mainline_observation_seed_terminal as diagnostic
from performance.run_mainline_legacy_backend_bridge import read, save


def test_funnel_keeps_no_feasible_initial_and_failed_cell(tmp_path):
    search, terminal = tmp_path/'search', tmp_path/'terminal'
    arm = diagnostic.AFTER
    definition = dict(name=arm, history_arm=arm, terminal_backend='role')
    terminal_plan = dict(source_output_dir=str(search), output_dir=str(terminal),
        target_seeds=[1, 2, 3], arms=[definition,
            dict(name=diagnostic.FULL, history_arm=diagnostic.FULL, terminal_backend='role')],
        protocols={diagnostic.PROTOCOL: [80, 128, 128]},
        execution_failures=[dict(seed=3, arm=arm)])
    save(tmp_path/'terminal_plan.json', terminal_plan)
    family_rows, scored = [], []
    for seed in (1, 2):
        directory = search/f'seed_{seed}'
        initial = [dict(point=[point], raw=[.3, -.1]) for point in range(10)]
        cache = [dict(point=[point], truth=dict(objective=.3, quality=.7,
                 true_feasible=seed == 2)) for point in range(10)]
        cache.append(dict(point=[10], truth=dict(objective=.2, quality=.8, true_feasible=seed == 1)))
        save(directory/'truth.json', cache)
        np.savez(directory/'neighbors.npz', points=[[0], [10]])
        save(directory/arm/'search.json', dict(history=initial+[dict(point=[10], raw=[.2, -.1])],
            terminal=dict(objective_challenger=dict(eligible_count=1), frozen_shortlist=[])))
        deployed = [10] if seed == 1 else [0]
        location = terminal/f'seed_{seed}'/arm/diagnostic.PROTOCOL
        save(location/'input.json', dict(primary=deployed, frozen_shortlist=[dict(
            point=deployed, shortlist_position=1, shortlist_role='primary')]))
        save(location/'verification.json', dict(deployed=deployed, new_verification_calls=80,
            verification=dict(certified=True, status='certified', verification_budget=80,
                objective_comparison={}, attempts=[dict(candidate_index=0, certified=True,
                    upper_margin=-.1, verification_budget=80, sample_mean=-.2, sample_std=.01)])))
        family_rows.append(dict(seed=seed, arm=arm, deployed=deployed, certified=True,
            actual_safe_reward=.8 if seed == 1 else .7,
            certified_feasible_quality_gain=.1 if seed == 1 else 0.,
            certified_initial_quality_component=.7))
        scored.append(dict(replicate=1, seed=seed, arm=arm, status='complete',
            good_new_deployment=None if seed == 1 else 0,
            best_feasible_initial_truth=None if seed == 1 else cache[0]['truth'],
            quality_increment=None if seed == 1 else 0.))
    # A failed arm does not remove the other completed arm of the same geometry.
    for seed in (1, 2, 3):
        sample = min(seed, 2)
        directory = search/f'seed_{seed}'
        save(directory/'truth.json', read(search/f'seed_{sample}'/'truth.json'))
        np.savez(directory/'neighbors.npz', points=[[0], [10]])
        save(directory/diagnostic.FULL/'search.json', read(search/f'seed_{sample}'/arm/'search.json'))
        for filename in ('input.json', 'verification.json'):
            save(terminal/f'seed_{seed}'/diagnostic.FULL/diagnostic.PROTOCOL/filename,
                 read(terminal/f'seed_{sample}'/arm/diagnostic.PROTOCOL/filename))
        family_rows.append(dict(family_rows[sample-1], seed=seed, arm=diagnostic.FULL))
        scored.append(dict(scored[sample-1], seed=seed, arm=diagnostic.FULL))
    scored.append(dict(replicate=1, seed=3, arm=arm, status='execution_failure'))
    save(terminal/'results.json', dict(rows=family_rows))
    save(tmp_path/'source/results.json', dict(status='complete_with_execution_failures',
         rows=scored, completed_cells=5, registered_cells=6))
    save(tmp_path/'prepared.json', dict(replicates=[dict(replicate=1, output_dir=str(tmp_path/'rep'),
        families=[dict(domain='family', terminal_manifest=str(tmp_path/'terminal_plan.json'))])]))
    result = diagnostic.diagnose(dict(prepared_file=str(tmp_path/'prepared.json'),
        source_output_dir=str(tmp_path/'source'), target_seeds=[1, 2, 3], replicates=[1],
        source_execution_failures=terminal_plan['execution_failures']))
    assert result['registered_terminal_cells'] == 6 and result['terminal_cells'] == 5
    assert result['failed_terminal_cells'] == 1 and len(result['missing_cells']) == 1
    undefined = next(case for case in result['cases'] if case['seed'] == 1)
    assert undefined['certified'] and undefined['actual_safe_reward'] == .8
    assert undefined['good_new_policies'] is None and undefined['shortlist'][0]['good_new'] is None
    assert all(value is None for value in undefined['funnel'].values())
    assert len(result['undefined_baseline_cells']) == 2
    summary = result['arms'][0]
    assert summary['tasks'] == 2 and summary['registered_tasks'] == 3
    assert summary['known_baseline_tasks'] == summary['baseline_undefined_tasks'] == summary['missing_terminal_tasks'] == 1
    assert all(value is None for value in summary['funnel'].values())
    assert all(value == 0 for value in summary['known_baseline_funnel'].values())
    assert summary['by_family'][0]['registered_tasks'] == 3
