"""Failed-query provenance with deterministic substitutes, without real models."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from performance import run_mainline_role_fresh as runner


@pytest.fixture
def fake_search(tmp_path, monkeypatch):
    seed, arm = 17, 'legacy_role_after_init_point_only'
    geometry = dict(task_geometry_shift=[0., 0., 0.], task_geometry_radius_scale=1.)
    plan = dict(domain='fixture', target_seeds=[seed],
        geometry_by_seed={str(seed):dict(problem_kwargs=geometry)})
    initial = [dict(point=[i, 0], raw=[float(i), -1.]) for i in range(10)]
    points = [(10, 1), (11, 1), (12, 1)]
    task_dir = tmp_path/f'seed_{seed}'
    runner.save(task_dir/runner.STOP/'search.json', dict(history=initial))
    runner.save(task_dir/'neighbors.json', {})
    np.savez_compressed(task_dir/'neighbors.npz', points=np.asarray(points))
    engines, saved_engines = [], Mock()

    class FakeEngine:
        def __init__(self, initial_rows):
            self.history = [(tuple(row['point']), row['raw']) for row in initial_rows]
            self.iteration_log, self.frozen_terminal = [], {}
            self.config = SimpleNamespace(truth_pool_diagnostics=False, observation_seed=19)
            self.problem = SimpleNamespace(d=1000,
                base=SimpleNamespace(base=SimpleNamespace(**geometry)), simulate=self.observe)
            self.fail_on = None
            self.attempted = 0

        def observe(self, point, rng=None):
            self.attempted += 1
            if self.attempted == self.fail_on:
                raise RuntimeError('synthetic observation failure')
            return np.array([float(self.attempted), -1.])

        def run_search(self):
            for index, point in enumerate(points):
                self.audit['stages'].append(dict(paid_prefix=len(self.history),
                    selected_id=index, selected_index=index, action_ids=[0, 1, 2],
                    score=dict(sampled_objective=[1., 2., 3.], sampled_margin=[-.1, -.2, -.3])))
                raw = self.problem.simulate(point)
                self.history.append((point, raw))
                self.iteration_log.append(dict(x_selected=list(point)))

    def build(plan, seed, arm, initial_rows):
        engine = FakeEngine(initial_rows)
        engine.fail_on = build.fail_on
        engines.append(engine)
        return engine

    build.fail_on = None

    @contextmanager
    def capture(engine, supplied_points):
        engine.audit = dict(points=[tuple(p) for p in supplied_points],
            shared_ids=[0, 1, 2], stages=[])
        yield engine.audit

    def forbidden(*args, **kwargs):
        raise AssertionError('Fixture cannot restore models or draw random numbers')

    monkeypatch.setattr(runner, 'build_fresh_backend', build)
    monkeypatch.setattr(runner, 'shared_neighbor_candidates', capture)
    monkeypatch.setattr(runner, 'save_engine', saved_engines)
    monkeypatch.setattr(runner, 'load_pickle', forbidden)
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)
    return SimpleNamespace(output=tmp_path, plan=plan, seed=seed, arm=arm,
        directory=task_dir/arm, initial=initial, points=points, engines=engines,
        build=build, saved_engines=saved_engines)


def test_success_keeps_existing_query_files_and_observation_semantics(fake_search):
    case = fake_search
    runner.search(case.output, case.plan, case.seed, case.arm)
    result = runner.read(case.directory/'search.json')
    audit = runner.read(case.directory/'query_audit.json')
    assert set(audit) == {'shared_ids', 'stages'}
    assert [stage['paid_prefix'] for stage in audit['stages']] == [10, 11, 12]
    assert result['history'][:10] == case.initial
    assert result['logical_search_calls'] == 13
    assert (result['new_initial_calls'], result['new_search_calls']) == (0, 3)
    assert len(runner.read(case.directory/'paid_search_raw.json')) == 3
    with np.load(case.directory/'query_policies.npz') as bank:
        assert bank['points'].tolist() == [list(p) for p in case.points]
        assert bank['points'].dtype == np.dtype('int16')
    assert not list(case.directory.glob('failed_*'))
    case.saved_engines.assert_called_once_with(case.directory/'search_engine.pkl', case.engines[0])


def test_failure_keeps_pre_observation_stage_and_matching_returned_raws(fake_search):
    case = fake_search
    case.build.fail_on = 2
    with pytest.raises(RuntimeError, match='synthetic observation failure'):
        runner.search(case.output, case.plan, case.seed, case.arm)
    audit_path, = case.directory.glob('failed_query_audit_*.json')
    audit = runner.read(audit_path)
    assert audit['status'] == 'partial'
    assert audit['paths_relative_to'] == 'audit_directory'
    assert audit['returned_raw_count'] == 1
    assert [stage['paid_prefix'] for stage in audit['stages']] == [10, 11]
    assert audit['stages'] == deepcopy(case.engines[0].audit['stages'])
    raw_path = audit_path.parent/audit['paid_raw_file']
    bank_path = audit_path.parent/audit['query_policies_file']
    assert raw_path.stem.removeprefix('failed_search_raw_') == audit_path.stem.removeprefix('failed_query_audit_')
    assert runner.read(raw_path) == [dict(point=list(case.points[0]), raw=[1., -1.])]
    with np.load(bank_path) as bank:
        assert bank['points'].tolist() == [list(p) for p in case.points]
    assert len(case.engines[0].history) == 11 and case.engines[0].attempted == 2
    assert case.engines[0].audit['points'] == case.points
    assert not (case.directory/'query_audit.json').exists()
    assert not (case.directory/'paid_search_raw.json').exists()
    assert not (case.directory/'search.json').exists()
    case.saved_engines.assert_called_once_with(case.directory/'failed_search_engine.pkl', case.engines[0])
