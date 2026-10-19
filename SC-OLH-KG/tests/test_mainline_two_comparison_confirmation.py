"""Fresh paired-terminal wiring; temporary metadata only, with no model or draws."""
from copy import deepcopy
import json

import numpy as np
import pytest

from performance import run_mainline_two_comparison_confirmation as runner
from performance import mainline_modified_terminal as modified
from performance import register_mainline_two_comparison_development as registration

PROJECT = runner.ROOT
SEARCH_ARMS = ('legacy_native_neighbors', 'same_source_gp', 'current_structural_mixture',
    'legacy_role_full', 'legacy_role_after_init_point_only', 'legacy_role_point_only',
    'initialization_only')


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    plan = runner.read(runner.MANIFEST)
    plan.pop('execution_failures', None)
    parent = runner.read(PROJECT / plan['source_manifest'])
    for family in parent['families']:
        for key in ('search_manifest', 'terminal_manifest'):
            family[key] = str(PROJECT / family[key])
    runner.save(tmp_path/'parent.json', parent)
    plan['source_manifest'] = str(tmp_path/'parent.json')
    for module in (runner, runner.replication, runner.fresh, runner.verifier,
                   runner.candidate, modified, registration):
        monkeypatch.setattr(module, 'ROOT', tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('Wiring checks cannot run models, workers, or random draws')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)
    monkeypatch.setattr(runner.replication.subprocess, 'run', forbidden)
    monkeypatch.setattr(runner.fresh, 'load_pickle', forbidden)
    monkeypatch.setattr(runner.fresh, 'build_fresh_backend', forbidden)
    return plan, tmp_path


def test_fresh24_prepare_preserves_sources_frontend_and_search_cap(cohort):
    plan, root = cohort
    frozen = deepcopy(plan)
    prepared = runner.prepare(plan)
    assert plan == frozen and len(prepared['replicates']) == 1
    assert len(plan['tasks']) == 24 and plan['geometry_seed'] == 2026104000
    parent = runner.read(plan['source_manifest'])
    retained = ('task', 'anchor', 'frozen_assets', 'neighbor_count', 'neighbor_radius',
        'neighbor_safe_margin_scale', 'neighbor_rng', 'decision_protocol', 'frontends',
        'role_prior', 'point_only', 'after_init_point_only')
    seen = []
    for family in prepared['replicates'][0]['families']:
        template = next(f for f in parent['families'] if f['domain'] == family['domain'])
        source = runner.read(root/family['search_manifest'])
        original = runner.read(template['search_manifest'])
        terminal = runner.read(root/family['terminal_manifest'])
        assert {k:source[k] for k in retained} == {k:original[k] for k in retained}
        assert source['task']['d'] == 1000 and source['anchor']['search_calls'] == 13
        assert len(source['target_seeds']) == 8 and terminal['arms'] == plan['terminal_arms']
        assert terminal['protocols'] == {runner.PROTOCOL:[80,128,128]}
        assert 'fixed_initialization_file_by_task' not in source
        assert 'fixed_neighbors_file_by_task' not in source
        for task in [t for t in plan['tasks'] if t['domain'] == family['domain']]:
            seed = task['geometry_task_seed']
            assert source['geometry_by_seed'][str(seed)] == task['geometry']
            assert source['observation_seed_by_task'][str(seed)] == seed+100
            assert terminal['verification_seed_by_task'][str(seed)] == seed+200
            seen.append(seed)
    assert sorted(seen) == list(range(2026104001,2026104025))
    assert not list(root.rglob('search.json')) and not list(root.rglob('truth.json'))
    assert runner.prepare(plan) == prepared


def write_failure(root, job, *, returned=1, with_prefix=True):
    source = runner.read(root/job['search_manifest'])
    directory = root/source['output_dir']/f"seed_{job['seed']}"/job['arm']
    directory.mkdir(parents=True, exist_ok=True)
    if with_prefix:
        runner.save(directory/'failed_search_raw_1.json', [dict(point=[i],raw=[i,-1.]) for i in range(returned)])
        (directory/'failed_search_engine.pkl').write_bytes(b'synthetic-not-a-model')
    field = 'new_target_initial_calls' if job['arm'] == runner.STOP else 'new_target_search_calls'
    (directory/'attempts.jsonl').write_text(json.dumps({field:returned,'error':'synthetic failure',
        'status':'failed','cpu_seconds':0.,'wall_seconds':0.})+'\n')
    return directory


@pytest.mark.parametrize('arm', SEARCH_ARMS)
def test_every_search_arm_failure_retains_its_terminal_dependencies(cohort, arm):
    plan, root = cohort
    prepared = runner.prepare(plan)
    jobs = runner.replication.search_jobs(prepared, arm == runner.STOP)
    job = next(j for j in jobs if j['arm'] == arm)
    directory = write_failure(root, job)
    record = runner.retain_search_failure(job)
    assert record['logical_search_calls'] == (1 if arm == runner.STOP else 11)
    assert record['failed_raw_file'] == str((directory/'failed_search_raw_1.json').relative_to(root))
    assert (root/record['failed_state_file']).read_bytes() == b'synthetic-not-a-model'
    source = runner.read(root/job['search_manifest'])
    terminal = runner.read(root/job['terminal_manifest'])
    affected = set(SEARCH_ARMS) if arm == runner.STOP else {arm}
    assert {r['arm'] for r in source['execution_failures']} == affected
    expected = {r['name'] for r in plan['terminal_arms'] if r['history_arm'] in affected}
    assert {r['arm'] for r in terminal['execution_failures']} == expected
    assert {r['seed'] for r in terminal['execution_failures']} == {job['seed']}
    runner.retain_search_failure(job)
    assert len(runner.read(root/job['terminal_manifest'])['execution_failures']) == len(expected)
    assert not any(j['seed'] == job['seed'] and j['arm'] in affected
                   for initial in (True,False) for j in runner.replication.search_jobs(prepared, initial))


def test_pre_observation_failure_does_not_invent_raw_or_state(cohort):
    plan, root = cohort
    prepared = runner.prepare(plan)
    job = next(j for j in runner.replication.search_jobs(prepared, False)
               if j['arm'] == 'legacy_role_point_only')
    write_failure(root, job, returned=0, with_prefix=False)
    record = runner.retain_search_failure(job)
    assert record['new_search_calls'] == 0 and record['logical_search_calls'] == 10
    assert record['failed_raw_file'] is record['failed_state_file'] is None
    assert not list(root.rglob('failed_search_raw_*.json'))


def test_search_barriers_keep_failed_geometry_and_other_arms_without_truth(cohort, monkeypatch):
    plan, root = cohort
    events = []
    stopped = plan['tasks'][0]['geometry_task_seed']
    point_failed = plan['tasks'][1]['geometry_task_seed']

    def stage(output, name, jobs, workers):
        events.append((name,len(jobs)))
        assert not list(root.rglob('truth.json'))
        if name == 'fixed-neighbors':
            assert len(list(root.rglob('search.json'))) == 23
        if name == 'online-search':
            assert len(list(root.rglob('neighbors.npz'))) == 23
        failed = []
        for job in jobs:
            command = job['command']
            assert command[3] == runner.MODULE
            if name == 'fixed-neighbors':
                source = runner.read(root/command[command.index('--child-manifest')+1])
                seed = int(command[command.index('--seed')+1])
                directory = root/source['output_dir']/f'seed_{seed}'
                runner.save(directory/'neighbors.json', {})
                (directory/'neighbors.npz').write_bytes(b'synthetic-not-numpy')
                continue
            source = runner.read(root/job['search_manifest'])
            if (job['seed'],job['arm']) in ((stopped,runner.STOP),(point_failed,'legacy_role_point_only')):
                write_failure(root, job)
                failed.append(dict(job,returncode=1))
            else:
                runner.save(root/source['output_dir']/f"seed_{job['seed']}"/job['arm']/'search.json', {})
        return failed

    monkeypatch.setattr(runner.replication, 'run_stage', stage)
    seal = runner.run_search(plan, 3, root/'manifest.json')
    assert events == [('initialization',24),('fixed-neighbors',23),('online-search',138)]
    assert seal['status'] == 'sealed' and seal['registered_histories'] == 168
    assert seal['completed_histories'] == 160 and len(seal['execution_failures']) == 2
    assert sum(r['status']=='failed' for r in seal['rows']) == 8
    assert not seal['truth_used'] and not list(root.rglob('verification.json'))
    assert not list(root.rglob('truth.json'))


def saved_histories(plan, root):
    """Small saved records include a repeated observation and an original-list reorder."""
    frozen = dict(frozen_shortlist=[dict(point=[i], posterior_rank=rank, shortlist_position=i+1,
        shortlist_role=('primary','challenger','backup')[i]) for i, rank in enumerate((1,3,2))],
        objective_incumbent_guard_enabled=True, candidate_verification_budgets=[80,128,128],
        max_verification_budget=352, objective_comparison_budget_per_policy=8, familywise_delta=.05,
        objective_comparison=dict(delta=.05/3), mean_delta_fraction=.5,
        shortlist_mode='posterior_objective_challenger_then_safe', objective_incumbent_position=1)
    for _, family in runner.families(plan):
        source = runner.read(root/family['search_manifest'])
        terminal = runner.read(root/family['terminal_manifest'])
        runner.save(root/terminal['inherited_replay_check_file'], dict(status='passed'))
        for seed in source['target_seeds']:
            for arm in SEARCH_ARMS:
                n = 10 if arm == runner.STOP else 13
                history = [dict(point=[i%12],raw=[float(i),-1.]) for i in range(n)]
                directory = root/source['output_dir']/f'seed_{seed}'/arm
                runner.save(directory/'search.json', dict(seed=seed,arm=arm,history=history,
                    geometry=source['geometry_by_seed'][str(seed)]['problem_kwargs'],
                    logical_search_calls=n, terminal=frozen, truth_used=False))
                if arm in runner.ROLES:
                    runner.save(directory/'native_terminal.json', dict(terminal=frozen,truth_used=False))
    rows = runner.search_barrier(plan)
    runner.save(root/plan['output_dir']/'search_status.json', dict(status='sealed',rows=rows,
        registered_histories=168,completed_histories=168,execution_failures=[],truth_used=False))


def test_paired_input_freeze_shares_history_and_list_and_blocks_partial_freeze(cohort, monkeypatch):
    plan, root = cohort
    saved_histories(plan, root)
    native = sorted(root.rglob('native_terminal.json'))[-1]
    saved = native.read_bytes()
    native.unlink()
    calls = []
    monkeypatch.setattr(runner.verifier, 'verify', lambda *args, **kw: calls.append(args))
    with pytest.raises(FileNotFoundError):
        runner.freeze_inputs(plan)
    assert not (root/plan['output_dir']/'frozen_inputs.json').exists()
    with pytest.raises(FileNotFoundError):
        runner.verify_old(plan, {})
    assert not calls
    native.write_bytes(saved)
    frozen = runner.freeze_inputs(plan)
    assert frozen['registered_pairs'] == frozen['frozen_pairs'] == 216
    for row in frozen['rows']:
        old, new = runner.read(root/row['old_input']), runner.read(root/row['new_input'])
        assert new == runner.candidate_input(old)
        assert new['source_case'] == old['source_case'] == row['source_case']
        assert new['history_arm'] == old['history_arm'] == row['history_arm']
        assert new['frozen_shortlist'] == old['frozen_shortlist']
        assert [r['point'] for r in old['frozen_shortlist']] == [[2],[0],[1]]
        history = runner.read(root/row['source_case']/'search.json')['history']
        assert len(history) == (10 if row['history_arm'] == runner.STOP else 13)
        if len(history) == 13:
            assert history[0]['point'] == history[-1]['point']
            assert history[0]['raw'] != history[-1]['raw']
    assert not calls and not list(root.rglob('truth.json'))


def test_old_terminal_partial_failure_blocks_new_without_fabricating_raw(cohort, monkeypatch):
    plan, root = cohort
    saved_histories(plan, root)
    # One real-supported POINT search failure remains a registered missing pair.
    replica, family = runner.families(plan)[0]
    source = runner.read(root/family['search_manifest'])
    job = dict(replicate=replica['replicate'], seed=source['target_seeds'][0],
        arm='legacy_role_point_only', **family)
    directory = root/source['output_dir']/f"seed_{job['seed']}"/job['arm']
    (directory/'search.json').unlink()
    write_failure(root, job)
    runner.retain_search_failure(job)
    frozen = runner.freeze_inputs(plan)
    assert frozen['frozen_pairs'] == 215 and frozen['missing_pairs'] == 1
    failed_key = source['target_seeds'][1], 'legacy_role_full'
    by_key = {(r['seed'],r['arm']):r for r in frozen['rows']}
    old_calls, new_calls, events = [], [], []

    def old_verify(output, child, seed, arm, protocol):
        old_calls.append((seed,arm))
        row = by_key[seed,arm]
        directory = (root/row['old_input']).parent
        item = runner.read(root/row['old_input'])
        points = [r['point'] for r in item['frozen_shortlist']]
        if (seed,arm) == failed_key:
            runner.save(directory/'failed_verification_raw_1.json',
                        [dict(point=points[0],raw=[0.,-1.])]*17)
            (directory/'attempts.jsonl').write_text(json.dumps(dict(status='failed',new_verification_calls=17,
                error='synthetic old terminal failure',cpu_seconds=0.,wall_seconds=0.))+'\n')
            raise RuntimeError('synthetic old terminal failure')
        raw = ([dict(point=points[0],raw=[0.,-1.])]*80 +
               [dict(point=points[1],raw=[0.,-1.])]*128 +
               [dict(point=points[i%2],raw=[0.,-1.]) for i in range(16)])
        runner.save(directory/'paid_verification_raw.json',raw)
        runner.save(directory/'verification.json',dict(verification_seed=item['verification_seed'],
            new_verification_calls=224,verification=dict(verification_budget=224,
                objective_comparison_budget=16,attempts=[dict(candidate_index=i,
                    verification_budget=n,certified=True) for i,n in enumerate((80,128))])))

    def new_verify(child, row):
        key = row['seed'],row['arm']
        assert key != failed_key and key != (job['seed'],job['arm'])
        assert len(old_calls) == 215
        new_calls.append(key)
        result = dict(status='complete',seed=row['seed'],arm=row['arm'])
        runner.save(runner.candidate.case_dir(child,row)/'verification.json',result)
        return result

    def stage(output, name, jobs, workers):
        events.append((name,len(jobs)))
        assert runner.read(output/'execution_status.json')['truth_access_allowed'] is False
        failures = []
        for task in jobs:
            command = task['command']
            key = int(command[command.index('--seed')+1]),command[command.index('--arm')+1]
            try:
                (runner.verify_old if name=='old-terminals' else runner.verify_new)(plan,by_key[key])
            except RuntimeError:
                assert name == 'old-terminals' and key == failed_key
                failures.append(dict(task,returncode=1))
        return failures

    monkeypatch.setattr(runner.verifier,'verify',old_verify)
    monkeypatch.setattr(runner.candidate,'verify_case',new_verify)
    monkeypatch.setattr(runner.replication,'run_stage',stage)
    seal = runner.run_terminal(plan,3,root/'manifest.json')
    assert events == [('old-terminals',215),('new-terminals',214)]
    assert seal['registered_pairs'] == 216 and seal['registered_terminal_cells'] == 432
    assert seal['completed_old'] == seal['completed_new'] == 214
    assert seal['truth_access_allowed'] and not seal['truth_used']
    pair = next(r for r in seal['cases'] if (r['seed'],r['arm']) == failed_key)
    assert pair['old']['status'] == 'failed' and pair['new']['status'] == 'missing'
    assert pair['new']['reason'] == 'old_verification_failed'
    failure = runner.read(root/pair['old']['failure_file'])
    assert failure['new_verification_calls'] == len(runner.read(root/failure['failed_raw_file'])) == 17
    assert not (root/pair['old']['failure_file']).with_name('paid_verification_raw.json').exists()
    with pytest.raises(RuntimeError,match='not retried'):
        runner.verify_old(plan,by_key[failed_key])
    assert len(old_calls) == 215 and len(new_calls) == 214
    inventory = runner.read(root/plan['output_dir']/'block_inventory.json')
    assert inventory['registered_cells'] == 216 and inventory['available_cells'] == 214
    assert all(r['quality_baseline_defined'] is None for r in inventory['rows'])
    assert not list(root.rglob('truth.json'))
