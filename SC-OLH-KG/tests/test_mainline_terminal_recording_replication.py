"""Changed two-stream barriers and deterministic cache reuse, without science."""
from copy import deepcopy

import pytest

from performance import run_mainline_terminal_recording_replication as run


@pytest.fixture
def plans(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run.audit, 'ROOT', tmp_path)
    task = dict(domain='fixture', geometry_task_seed=17, optimizer_seed=17,
        geometry=dict(problem_kwargs=dict(shift=[0., 0., 0.])))
    children = [dict(output_dir=f'stream_{i}', tasks=[deepcopy(task)]) for i in (1, 2)]
    plan = dict(output_dir='outer', reference_manifest='reference.json', tasks=[task],
        streams=[dict(name=f'observation_{i}', manifest=f'child_{i}.json') for i in (1, 2)])
    for row, child in zip(plan['streams'], children):
        run.save(tmp_path/row['manifest'], child)
    return plan, children, tmp_path


def inventory(plan):
    run.save(run.inventory_path(plan), dict(status='sealed', registered_streams=2,
        registered_terminal_cells=864, registered_task_streams=48))


def test_missing_second_seal_blocks_first_inventory(plans, monkeypatch):
    plan, children, _ = plans
    calls = []
    def seal(child):
        calls.append(child['output_dir'])
        if child == children[1]:
            raise RuntimeError('second stream not sealed')
        return dict(registered_pairs=216, registered_terminal_cells=432,
                    completed_old=216, completed_new=216), []
    monkeypatch.setattr(run.audit.terminal, 'load_sealed', seal)
    monkeypatch.setattr(run.audit, 'freeze', lambda *args: pytest.fail('First pool opened too early'))
    with pytest.raises(RuntimeError, match='second stream not sealed'):
        run.freeze(plan)
    assert calls == ['stream_1', 'stream_2']
    assert not run.inventory_path(plan).exists()


def test_joint_seal_precedes_both_inventories_and_completed_freeze_is_reused(plans, monkeypatch):
    plan, _, _ = plans
    calls = []
    def seal(child):
        calls.append(('seal', child['output_dir']))
        return dict(registered_pairs=216, registered_terminal_cells=432,
                    completed_old=215, completed_new=215), []
    def freeze(child):
        calls.append(('inventory', child['output_dir']))
        return dict(distinct_policy_task_units=10, maximum_analytic_function_calls=30)
    monkeypatch.setattr(run.audit.terminal, 'load_sealed', seal)
    monkeypatch.setattr(run.audit, 'freeze', freeze)
    result = run.freeze(plan)
    assert calls == [('seal', 'stream_1'), ('seal', 'stream_2'),
                     ('inventory', 'stream_1'), ('inventory', 'stream_2')]
    assert result['registered_terminal_cells'] == 864 and not result['truth_used']
    assert result['maximum_analytic_function_calls'] == 60
    assert run.freeze(plan) == result and len(calls) == 4


def test_task_cannot_open_any_cache_before_joint_inventory(plans, monkeypatch):
    plan, _, _ = plans
    monkeypatch.setattr(run, 'reuse_task_truth', lambda *args: pytest.fail('Premature truth reuse'))
    with pytest.raises(RuntimeError, match='Both streams'):
        run.task_score(plan, 17)


def test_second_stream_runs_after_retained_first_failure_without_retry(plans, monkeypatch):
    plan, children, root = plans
    inventory(plan)
    run.save(root/'reference.json', dict(output_dir='reference'))
    calls = []
    def reuse(outer, child, seed, previous):
        calls.append(('reuse', child['output_dir'], [p['output_dir'] for p in previous]))
    def score(child, seed):
        calls.append(('score', child['output_dir']))
        if child == children[0]:
            raise ValueError('retained fixture scoring failure')
        run.save(run.audit.directory(child)/f'seed_{seed}/result.json', dict(status='complete'))
    monkeypatch.setattr(run, 'reuse_task_truth', reuse)
    monkeypatch.setattr(run.audit, 'task_score', score)
    result = run.task_score(plan, 17)
    assert [row['status'] for row in result['outcomes']] == ['failed', 'complete']
    assert calls == [('reuse', 'stream_1', ['reference']), ('score', 'stream_1'),
                     ('reuse', 'stream_2', ['reference', 'stream_1']), ('score', 'stream_2')]
    assert run.task_score(plan, 17) == result and len(calls) == 4
    assert 'retained fixture scoring failure' in run.read(
        run.audit.directory(children[0])/'seed_17/failure.json')['reason']


def test_truth_reuse_preserves_new_observations_and_merges_missing_fields_only(plans, monkeypatch):
    plan, children, root = plans
    inventory(plan)
    child = children[1]
    previous = deepcopy(children[0])
    paths = {p['output_dir']: root/p['output_dir']/'seed_17' for p in (child, previous)}
    monkeypatch.setattr(run, 'task_directory', lambda p,t: paths[p['output_dir']])
    for p, raw in ((child, 9.), (previous, 3.)):
        run.save(paths[p['output_dir']]/'initialization.json',
                 dict(history=[dict(point=[1, 0], raw=[raw, -1.])]))
    current = paths[child['output_dir']]
    old = paths[previous['output_dir']]
    run.audit.cache_save(current/'truth.json', {(1, 0): dict(objective=.2)})
    run.audit.cache_save(old/'truth.json', {(1, 0): dict(objective=.2, true_feasible=True),
                                         (2, 0): dict(objective=.1, true_feasible=False)})
    before = (current/'initialization.json').read_text()
    result = run.reuse_task_truth(plan, child, 17, [previous])
    assert result['new_truth_function_calls'] == 0 and result['cached_policies'] == 2
    assert run.audit.cache_read(current/'truth.json')[(1, 0)] == dict(objective=.2, true_feasible=True)
    assert (current/'initialization.json').read_text() == before
    changed = deepcopy(previous)
    changed['tasks'][0]['geometry']['problem_kwargs']['shift'][0] = .1
    with pytest.raises(AssertionError):
        run.reuse_task_truth(plan, child, 17, [changed])


def test_collect_requires_results_or_retained_failures_from_both_streams(plans, monkeypatch):
    plan, children, _ = plans
    inventory(plan)
    run.save(run.audit.directory(children[0])/'seed_17/result.json', dict(status='complete'))
    monkeypatch.setattr(run.audit, 'collect', lambda *args: pytest.fail('First stream collected before second finished'))
    with pytest.raises(AssertionError, match='Unfinished two-stream'):
        run.collect(plan)
