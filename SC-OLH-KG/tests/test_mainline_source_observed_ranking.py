"""Source ranking contracts with synthetic rows and stubbed model operations."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.transfer_archive import FrozenTransferArchive, TransferTaskArchive
from performance.run_mainline_source_policy_holdout import training_archive


def synthetic_archive():
    points = np.arange(64, dtype=int)[:, None] * np.ones((1, 2), dtype=int)
    tasks = []
    for name, offset in [('InventorySupplyChain', 0), ('QueueResourceControl', 1000)]:
        raws = tuple(np.array([[offset+i, 100+i], [offset+i+2, 102+i],
                               [offset+i+4, 104+i]], dtype=float) for i in range(64))
        tasks.append(TransferTaskArchive(name=name, X=points/100., X_integer=points,
            Y_mean=np.array([x.mean(axis=0) for x in raws]), Y_replicates=raws,
            replicate_variance=np.full((64, 2), 4.),
            mean_observation_variance=np.full((64, 2), 4/3),
            constraint_sigma=np.arange(64, dtype=float), tau=.3, alpha=.05,
            origins=tuple(f'policy-{i}' for i in range(64))))
    return FrozenTransferArchive(tuple(tasks), 80, 'synthetic', 'existing')


def test_frozen_whole_policy_folds_exclude_every_held_response_and_metadata(tmp_path, monkeypatch):
    from performance import run_mainline_source_observed_ranking as runner
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    archive = synthetic_archive()
    runner.save(tmp_path/'partition.json', dict(output_dir='old'))
    plan = dict(partition_manifest='partition.json')
    splits = []
    for fold, held in enumerate(np.roll(np.arange(64), 7).reshape(4, 16)):
        saved = dict(training=np.setdiff1d(np.arange(64), held).tolist(), held=held.tolist())
        runner.save(tmp_path/'old'/f'fold_{fold}'/'split.json', saved)
        split = runner.frozen_split(plan, archive, fold)
        assert split == saved
        splits.append(split)
    assert sorted(np.concatenate([s['held'] for s in splits])) == list(range(64))
    for split in splits:
        train = training_archive(archive, split['training'])
        assert train.simulator_calls == 288
        for original, subset in zip(archive.tasks, train.tasks):
            assert not set(map(tuple, subset.X_integer)) & set(map(tuple, original.X_integer[split['held']]))
            for field in ('X', 'X_integer', 'Y_mean', 'replicate_variance',
                          'mean_observation_variance', 'constraint_sigma'):
                np.testing.assert_array_equal(getattr(subset, field), getattr(original, field)[split['training']])
                assert not np.shares_memory(getattr(subset, field), getattr(original, field))
            assert subset.origins == tuple(original.origins[i] for i in split['training'])
            for j, i in enumerate(split['training']):
                np.testing.assert_array_equal(subset.Y_replicates[j], original.Y_replicates[i])
                assert not np.shares_memory(subset.Y_replicates[j], original.Y_replicates[i])


def test_score_labels_use_unused_two_repeats_and_center_constraint_once():
    from performance import summarize_mainline_source_observed_ranking as scoring
    task = synthetic_archive().tasks[0]
    held = np.arange(31, 15, -1)
    data = dict(Y_replicates=np.array(task.Y_replicates), tau=task.tau)
    original = scoring.scoring_labels(data, held)
    data['Y_replicates'][held[:13], 0] = 1e9
    poisoned = scoring.scoring_labels(data, held)
    for key in original:
        np.testing.assert_array_equal(original[key], poisoned[key])
    np.testing.assert_array_equal(poisoned['replicate_counts'], [2]*13+[3]*3)
    np.testing.assert_allclose(poisoned['mean'][:13, 0], held[:13]+3)
    np.testing.assert_allclose(poisoned['mean'][13:, 0], held[13:]+2)
    np.testing.assert_allclose(poisoned['mean'][:13, 1], held[:13]+103-task.tau)
    np.testing.assert_allclose(poisoned['variance'][:13], 2.)
    np.testing.assert_allclose(poisoned['variance'][13:], 4.)
    snapshot = dict(mean=np.column_stack([poisoned['mean'][:, 0], np.zeros(16),
        poisoned['mean'][:, 1], np.zeros(16)]), variance=np.ones((16, 4)),
        noise=poisoned['variance'])
    observed = scoring.prediction_metrics(snapshot, poisoned, np.arange(13))
    unseen = scoring.prediction_metrics(snapshot, poisoned, np.arange(13, 16))
    assert observed['f_mse'] == observed['g_mse'] == 0
    assert observed['f_noise_mse'] == observed['g_noise_mse'] == 0
    assert observed['f_mean_prediction_variance_with_label_noise'] == 2.
    assert unseen['f_mean_prediction_variance_with_label_noise'] == pytest.approx(1+4/3)


def test_missing_case_keeps_registered_denominator_and_null_unconditional_mean():
    from performance import summarize_mainline_source_observed_ranking as scoring
    summary = scoring.aggregate([
        dict(status='complete', metrics={'f_mse': 2.}),
        dict(status='forecast_failed', metrics={})], ['f_mse'])
    assert summary['registered_cases'] == 2 and summary['completed_cases'] == 1
    assert summary['metrics']['f_mse'] == dict(registered_cases=2, available_cases=1,
        mean=None, conditional_available_mean=2.)


@pytest.mark.parametrize('heldout', [
    'InventorySupplyChain', 'QueueResourceControl', 'FactorShockStatePolicyRZDT1'])
def test_native_fit_uses_each_archives_heldout_family(monkeypatch, heldout):
    from performance import mainline_role_source_holdout as helper
    families = ['InventorySupplyChain', 'QueueResourceControl', 'FactorShockStatePolicyRZDT1']
    archive = synthetic_archive()
    source_names = [name for name in families if name != heldout]
    archive = replace(archive, tasks=tuple(replace(task, name=name)
        for task, name in zip(archive.tasks, source_names)))
    training = training_archive(archive, list(range(48)))
    materialized = []

    def task(domain, **kwargs):
        materialized.append(domain)
        return dict(args={'domains': ','.join(families)}, heldout=domain, seed=80)

    def fit(args, actual_heldout, seed, *, teacher):
        assert actual_heldout == heldout and seed == 80 and not teacher
        assert [name for name in args['domains'].split(',') if name != actual_heldout] == source_names
        assert args['source_records_per_domain'] == 48
        return SimpleNamespace(n_records=96)

    monkeypatch.setattr(helper, 'materialize_v69_task', task)
    monkeypatch.setattr(helper.benchmark, 'train_meta_prior', fit)
    assert helper.fit_native(training, heldout_family=heldout).n_records == 96
    assert materialized == [heldout]


ARMS = ['legacy_native', 'legacy_role_full',
        'legacy_role_after_init_point_only', 'legacy_role_point_only']


def test_score_cannot_read_labels_until_all_twelve_fold_records_exist(tmp_path, monkeypatch):
    from performance import summarize_mainline_source_observed_ranking as scoring
    monkeypatch.setattr(scoring, 'ROOT', tmp_path)
    plan = dict(output_dir='out', folds=4, arms=ARMS, statistical_unit='overlapping',
        source_archives=[dict(heldout_target_family=name, path=f'sources/{name}.json',
                             source_families=['first', 'second']) for name in 'ABC'])
    accessed = []

    def read(path):
        path = Path(path).relative_to(tmp_path)
        accessed.append(str(path))
        assert path.parts[0] != 'sources', 'Scoring labels were opened before the global barrier'
        name, fold = path.parts[1], int(path.parts[2].split('_')[-1])
        if (name, fold) == ('C', 3):
            raise FileNotFoundError('final forecast not frozen yet')
        return dict(archive_heldout_family=name, fold=fold, status='fold_failed',
                    cases=[], failure='retained synthetic failure')

    monkeypatch.setattr(scoring, 'read', read)
    with pytest.raises(FileNotFoundError, match='not frozen'):
        scoring.score(plan)
    assert len(accessed) == 12


def test_partial_fold_and_failed_arm_are_retained_with_null_metrics(tmp_path, monkeypatch):
    from performance import summarize_mainline_source_observed_ranking as scoring
    monkeypatch.setattr(scoring, 'ROOT', tmp_path)
    tasks = [dict(name=t.name, Y_replicates=t.Y_replicates, tau=t.tau, alpha=t.alpha)
             for t in synthetic_archive().tasks]
    scoring.save(tmp_path/'archive.json', dict(tasks=tasks))
    plan = dict(output_dir='out', folds=1, arms=ARMS, statistical_unit='overlapping',
        source_archives=[dict(heldout_target_family='FactorShockStatePolicyRZDT1',
                             path='archive.json', source_families=[t['name'] for t in tasks])])
    labels = scoring.scoring_labels(tasks[0], list(range(16)))
    snapshot = dict(status='complete', mean=np.column_stack([labels['mean'][:, 0],
        np.zeros(16), labels['mean'][:, 1], np.zeros(16)]), variance=np.ones((16, 4)),
        noise=labels['variance'], probability_violation=np.full(16, .25))
    predictions = {arm: dict(status='complete', snapshots={'10': snapshot, '13': snapshot},
        terminal13=dict(selected_index=0, eligible_indices=list(range(13))), failures=[])
        for arm in ARMS}
    predictions[ARMS[0]] = dict(status='failed', snapshots={}, terminal13=None,
                               failures=['synthetic native failure'])
    predictions[ARMS[3]] = dict(status='partial_failure', snapshots={
        '10': snapshot, '13': dict(status='failed', error='synthetic later failure')},
        terminal13=None, failures=['synthetic later failure'])
    frozen = dict(archive_heldout_family='FactorShockStatePolicyRZDT1', fold=0,
        status='fold_failed', failure='second family failed', cases=[dict(
            family=tasks[0]['name'], held_policy_ids=list(range(16)), predictions=predictions)])
    result = scoring.score(plan, {('FactorShockStatePolicyRZDT1', 0): frozen})
    assert result['registered_source_cases'] == 2 and result['registered_arm_cases'] == 8
    assert result['completed_arm_cases'] == 2 and len(result['failures']) == 6
    for row in result['rows']:
        if row['status'] == 'failed':
            assert set(row['metrics']) == set(scoring.METRICS)
            assert all(value is None for value in row['metrics'].values())
    after = next(a for a in result['arms'] if a['arm'] == ARMS[2])
    metric = after['metrics']['prefix13_observed13_f_mse']
    assert metric == dict(registered_cases=2, available_cases=1, mean=None,
                          conditional_available_mean=0.)
    partial = next(row for row in result['rows'] if row['status'] == 'partial_failure')
    assert partial['metrics']['prefix10_observed13_f_mse'] == 0.
    assert partial['metrics']['prefix13_observed13_f_mse'] is None
    assert partial['metrics']['unobserved3_f_mse_change_13_minus_10'] is None
    assert partial['terminal'] is None


def test_forecast_native_initial10_then3_and_actual_after_operator(monkeypatch):
    from performance import mainline_source_observed_ranking as adapter
    original = synthetic_archive().tasks[0]
    held = list(range(15, -1, -1))
    events = []

    class FirstRepeatOnly:
        def __init__(self, index):
            self.index = index

        def __getitem__(self, repeat):
            assert self.index in held[:13] and repeat == 0, 'Scoring repeats leaked into forecast'
            return original.Y_replicates[self.index][0]

    task = replace(original, Y_replicates=tuple(FirstRepeatOnly(i) for i in range(64)))
    problem = SimpleNamespace(d=2, tau=task.tau, alpha=task.alpha)
    engine = SimpleNamespace(problem=problem, replayed_initial_calls=0, history=[])

    def materialize(domain, **kwargs):
        events.append(('materialize', domain))
        return dict(heldout=domain, args=dict(d=1000, initial_design_points=[[999]]))

    def build(domain, initial_rows, source_prior):
        config = adapter.native.materialize_v69_task(domain)
        assert config['heldout'] == task.name and config['args']['d'] == 2
        assert 'initial_design_points' not in config['args']
        assert len(initial_rows) == 13 and source_prior == 'prior'

        def initialize(points):
            assert points == [tuple(task.X_integer[i]) for i in held[:10]]
            engine.history.extend(initial_rows[:10]); engine.replayed_initial_calls += 10
            events.append(('native_initial', 10))

        engine._fit_initial_belief = initialize
        return engine

    def update(actual, row):
        assert actual is engine
        engine.history.append(row); engine.replayed_initial_calls += 1
        events.append(('native_update', len(engine.history)))

    def snapshot(*args):
        return dict(status='complete', mean=np.zeros((16, 4)), variance=np.ones((16, 4)),
                    noise=np.ones((16, 2)), probability_violation=np.full(16, .25))

    def operator(name):
        def compute(bundle, projection, bank, rows, **kwargs):
            assert len(bank) == 16 and len(rows) in (10, 13)
            assert rows == engine.history[:len(rows)]
            events.append((name, len(rows)))
            return snapshot()
        return compute

    monkeypatch.setattr(adapter.native, 'materialize_v69_task', materialize)
    monkeypatch.setattr(adapter.native, 'build_native', build)
    monkeypatch.setattr(adapter, 'update_native', update)
    monkeypatch.setattr(adapter, 'native_snapshot', snapshot)
    monkeypatch.setattr(adapter, 'build_source_engine', lambda *args, **kw: SimpleNamespace(problem=problem))
    monkeypatch.setattr(adapter, 'role_snapshot', lambda projection, bank, state: state)
    monkeypatch.setattr(adapter.role, 'role_moment_state', operator('full'))
    monkeypatch.setattr(adapter.after_init, 'after_init_point_only_moment_state', operator('after10'))
    monkeypatch.setattr(adapter.point_only, 'point_only_moment_state', operator('point'))
    cost = {}
    result = adapter.forecast('bundle', 'prior', task, dict(fold=2, held=held), cost, 'QueueResourceControl')
    assert events == [('materialize', 'QueueResourceControl'), ('native_initial', 10),
        ('native_update', 11), ('native_update', 12), ('native_update', 13),
        ('full', 10), ('full', 13), ('after10', 10), ('after10', 13), ('point', 10), ('point', 13)]
    assert cost == dict(native_replayed_raw_rows=13, paired_raw_adaptation_exposures=13,
                        role_posterior_evaluations=6)
    assert all(record['status'] == 'complete' for record in result['predictions'].values())
    assert result['held_policy_ids'] == held


def test_native_feedback_uses_preupdate_moments_and_updates_hvd_and_task_posterior():
    from performance import mainline_source_observed_ranking as adapter
    events, observations = [], {}
    raw, point = np.array([1.1, 2.2]), (4, 7)
    models = []
    for head in range(2):
        models.append(SimpleNamespace(
            posterior_mean=lambda x, i=head: 10.+i,
            posterior_var=lambda x, i=head: 20.+i,
            update=lambda x, y, noise, i=head: events.append(('gpr', i, y, noise))))

    def store(actual):
        assert actual == point
        observations[point] = [raw]
        engine.history.append((point, raw))
        events.append(('raw',))
        return raw

    def hvd(head, x, y, mean, model, problem, **kwargs):
        assert mean == 10+head and model is models[head]
        assert kwargs == dict(epistemic_var=20.+head, replicate_count=1, replicate_variance=None)
        events.append(('hvd', head))

    def task_update(x, y, **kwargs):
        assert kwargs == dict(existing_observations=[], tau=.3)
        np.testing.assert_array_equal(y, raw)
        events.append(('task',))

    engine = SimpleNamespace(gpr=models, problem=SimpleNamespace(tau=.3), observations=observations,
        history=[None]*10, config=SimpleNamespace(source_constraint_mean_adaptation_mode='sequential_evidence_mixture'),
        _refresh_sequential_basis=lambda: events.append(('refresh',)), _simulate_and_store=store,
        _configure_hvd_source_task_posterior=lambda *args: events.append(('configure',)),
        variance_model=SimpleNamespace(predict_variance=lambda i, x, p: 30.+i, update=hvd),
        task_ensemble=SimpleNamespace(update=task_update),
        _update_posterior_dominance_incumbent=lambda **kwargs: events.append(('incumbent', kwargs['stage'])))
    adapter.update_native(engine, dict(point=list(point), raw=raw.tolist()))
    assert events == [('refresh',), ('raw',), ('gpr', 0, 1.1, 30.), ('gpr', 1, 2.2, 31.),
                      ('configure',), ('hvd', 0), ('hvd', 1), ('task',), ('incumbent', 11)]

