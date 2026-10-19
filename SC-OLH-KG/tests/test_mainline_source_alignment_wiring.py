"""Exercise saved-engine routing and frozen terminal inputs with substitutes only."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from performance import mainline_role_backend as backend
from performance import run_mainline_role_fresh as fresh
from performance.run_mainline_same_policy import transform
from performance.run_mainline_terminal_confirmation import input_record
from test_mainline_terminal_recording import engine_fixture, construct


NEW_ARMS = (
    'source_permuted_role_after_init_point_only',
    'source_permuted_role_point_only',
)


@pytest.mark.parametrize('arm', NEW_ARMS)
def test_new_terminal_inputs_keep_old_shortlist_and_352_cap(arm):
    """New names must reach the existing observed-only input and old transform."""
    engine = engine_fixture()
    engine.backend_arm = arm
    frozen = construct(engine)['terminal']
    geometry = dict(task_geometry_shift=[.01, -.02, .03], task_geometry_radius_scale=1.01)
    parent = dict(domain='InventorySupplyChain', task=dict(d=1000),
        geometry_by_seed={'91': dict(problem_kwargs=geometry)},
        decision_protocol=dict(verification_method='normal_quantile_tolerance'))
    search = dict(geometry=geometry, logical_search_calls=13,
        history=[dict(point=list(p), raw=y.tolist()) for p, y in engine.history])
    row = input_record(search, frozen, parent,
        dict(name=arm, history_arm=arm, terminal_backend='role'), 91, 501,
        'synthetic/search', engine.problem.tau)
    old = transform(row)
    assert old['candidate_budgets'] == [80, 128, 128]
    assert sum(old['candidate_budgets']) + 2 * old['objective_comparison_budget'] == 352
    assert old['objective_comparison_delta'] == .05 / 3
    assert old['verification_seed'] == 501 and old['arm'] == arm
    assert all(tuple(p['point']) in engine.observations for p in old['frozen_shortlist'])
    assert old['search_evaluation_count'] == 13
    assert engine.config.terminal_verification_budget == 80


@pytest.mark.parametrize('arm,expected', [
    (NEW_ARMS[0], 'permuted'), (NEW_ARMS[1], 'permuted'),
    ('legacy_role_after_init_point_only', 'original'),
    ('legacy_role_point_only', 'original'),
    ('same_source_gp', 'original'),
])
def test_saved_verification_routes_bundle_and_reuses_completed_prefix(tmp_path, monkeypatch, arm, expected):
    """save_engine removes the bundle: verification must reload the right source once."""
    monkeypatch.setattr(fresh, 'ROOT', tmp_path)
    monkeypatch.setattr(backend, 'ROOT', tmp_path)
    plan = dict(domain='InventorySupplyChain', target_seeds=[91], arms=[dict(name=arm)],
        frozen_assets=dict(source_models_file='original.pkl'),
        source_permuted_assets=dict(source_models_file='permuted.pkl', native_source_prior_file='risk.pkl'))
    front = object()
    context = object()
    engine = SimpleNamespace(backend_bundle=None, source_prior=front,
        role_projection_context=context, history=[None] * 13,
        role_restoration_trace=[], defer_terminal=True)
    selected = [[1], [2], [3]]
    search = dict(primary=selected[0], terminal=dict(frozen_shortlist=[dict(point=p) for p in selected]))
    directory = fresh.case_dir(tmp_path, 91, arm)
    fresh.save(directory / 'search.json', search)
    loads = []

    def fake_load(path):
        loads.append(path.name)
        return path.stem

    monkeypatch.setattr(backend, 'load_pickle', fake_load)
    monkeypatch.setattr(fresh, 'load_pickle', lambda path: engine)
    monkeypatch.setattr(fresh, 'save_engine', lambda *args: None)

    @contextmanager
    def no_simulations(engine, cost, field, observations):
        cost[field] = 0
        yield

    monkeypatch.setattr(fresh, 'count_simulations', no_simulations)

    def terminal(primary):
        assert engine.backend_bundle == expected
        assert engine.source_prior is front and engine.role_projection_context is context
        assert engine.defer_terminal is False
        return primary, dict(frozen_shortlist=search['terminal']['frozen_shortlist'], verification_budget=0)

    engine._terminal_verification_protocol = terminal
    fresh.verify(tmp_path, plan, 91, arm)
    assert loads == [expected + '.pkl']
    saved = fresh.read(directory / 'verification.json')
    assert saved['new_verification_calls'] == 0 and saved['deployed'] == selected[0]
    fresh.verify(tmp_path, plan, 91, arm)
    assert loads == [expected + '.pkl']  # A completed call is not repeated.

