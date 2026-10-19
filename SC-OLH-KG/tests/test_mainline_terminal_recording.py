"""Exercise real terminal construction with deterministic score substitutes.

The three engine classes cover OLD/STOP, GP/structural and FULL/AFTER/POINT.
The native terminal bypass is shared by FULL-native and AFTER-native. No engine
constructor, simulator, model fit, posterior restore or verification is run.
"""
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from algorithms.single_olhkg import SingleOLHKGConfig
from performance import mainline_terminal_recording as recording
from performance import run_mainline_role_fresh as search_runner
from performance import run_mainline_terminal_confirmation as terminal_runner
from performance.diagnose_mainline_terminal_backend import freeze
from performance.mainline_legacy_backend import LegacyPosteriorAlgorithm
from performance.mainline_legacy_native import LegacyNativeAlgorithm, SearchFrozen
from performance.mainline_role_backend import RolePosteriorAlgorithm
from performance.prepare_mainline_two_comparison_terminal import candidate_input
from performance.run_mainline_same_policy import transform
from performance.run_mainline_terminal_confirmation import input_record
from test_mainline_failed_query_audit import fake_search


def point(index):
    return (index,) + (0,) * 999


def forbidden(*args, **kwargs):
    raise AssertionError('Terminal recording must not simulate, fit or restore')


def engine_fixture(cls=RolePosteriorAlgorithm, *, stop=False, same_policy=False,
                   fail_score_at=None):
    """Supply saved history and scores; retain real dispatch and terminal logic."""
    engine = object.__new__(cls)
    engine.config = SingleOLHKGConfig(
        N=10 if stop else 13, n0=10, decision_backend='constrained_ts',
        decision_recommend_observed_only=True, decision_terminal_rule='feasible_first',
        decision_terminal_maximum_violation_probability=.5,
        terminal_verification_policy='ordered_frozen_shortlist',
        terminal_verification_shortlist_size=3,
        terminal_verification_shortlist_mode='posterior_objective_challenger_then_safe',
        terminal_verification_budget=80, terminal_verification_fallback_budget=128,
        terminal_verification_method='normal_quantile_tolerance',
        terminal_verification_delta=.05, terminal_objective_incumbent_guard=True,
        terminal_objective_comparison_budget=8, terminal_objective_comparison_delta=.05/3,
        terminal_objective_challenger_max_violation_probability=.5,
        terminal_safe_interior_candidate_scope='observed',
        terminal_safe_interior_selection_mode='diverse',
        terminal_safe_interior_probability_slack=.05,
        terminal_safe_interior_require_provider=True, truth_pool_diagnostics=False)
    engine.backend_arm = ('initialization_only' if stop else {
        LegacyNativeAlgorithm: 'legacy_native_v69',
        LegacyPosteriorAlgorithm: 'same_source_gp',
        RolePosteriorAlgorithm: 'legacy_role_after_init_point_only',
    }[cls])
    engine._decision_context = 0
    engine.defer_terminal = True
    engine.frozen_terminal = None
    engine.rng = np.random.default_rng(87)
    engine.history = [(point(i), np.asarray([float(i + 1), .1])) for i in range(10)]
    if not stop:
        engine.history += [(point(10), np.asarray([2., .1])),
                           (point(11), np.asarray([3., .1])),
                           (point(11), np.asarray([4., .1]))]
    engine.observations = {}
    for x, raw in engine.history:
        engine.observations.setdefault(x, []).append(raw)
    engine._last_terminal_pool = [point(99), point(11), point(10), point(8), point(0)]
    engine.problem = SimpleNamespace(d=1000, tau=0., alpha=.1,
        risk_exposures=lambda x, **kwargs: dict(A=[x[0], x[0] ** 2], N=[1.]),
        simulate=forbidden, true_objective=forbidden, true_chance_margin=forbidden)
    engine.gpr = [SimpleNamespace(fit=forbidden), SimpleNamespace(fit=forbidden)]
    engine.variance_model = SimpleNamespace(fit=forbidden)
    engine.task_ensemble = None
    engine._posterior_views = forbidden
    engine._true_chance_margin = forbidden
    engine._evaluate_recommendation = forbidden
    engine.calls = dict(score=[], predict=[])

    def score(self, gpr, variance, pool, **kwargs):
        self.calls['score'].append(dict(context=self._decision_context,
            points=[tuple(x) for x in pool]))
        if len(self.calls['score']) == fail_score_at:
            raise RuntimeError('deterministic terminal score failure')
        ids = np.asarray([x[0] for x in pool])
        objective = 1. + ids / 100.
        objective[ids == 9] = -100.  # Lowest objective is ineligible at the .5 gate.
        role = bool(self._decision_context)
        objective[ids == (10 if role else 11)] = -3.
        objective[ids == (11 if role else 10)] = -2.
        if same_policy:
            objective[ids == 0] = -4.
        probability = np.full(len(pool), .2)
        probability[ids == 3] = .01  # Unique safe-sublevel fallback.
        probability[ids == 8] = .5  # Inclusive boundary remains in the pool.
        probability[ids == 9] = .75
        zeros = np.zeros(len(pool))
        return dict(objective=objective, objective_variance=np.full(len(pool), .03),
            risk=objective + probability, expected_violation=probability,
            probability_violation=probability, violation_loss=probability,
            nominal_expected_violation=probability, robust_expected_violation=probability,
            nominal_probability_violation=probability, robust_probability_violation=probability,
            model_disagreement=zeros, violation_loss_mode='positive_part',
            decision_aleatoric_mode='plugin', decision_ambiguity_mode='none', kl_radius=0.,
            constraint_posterior_source='role_fixture' if role else 'native_fixture')

    def predict(self, *, pool):
        self.calls['predict'].append([tuple(x) for x in pool])
        return tuple(pool[0]), {}

    engine._terminal_bayes_risk_components = MethodType(score, engine)
    engine._solve_posterior_recommendation = MethodType(predict, engine)
    return engine


def construct(engine, *, native=False):
    before = deepcopy(engine.rng.bit_generator.state)
    global_before = np.random.get_state()
    history = [(x, raw.copy()) for x, raw in engine.history]
    if native:
        result = freeze(engine, native=True)
    else:
        primary, details = engine._decision_backend_terminal_recommendation(engine._last_terminal_pool)
        with pytest.raises(SearchFrozen):
            engine._terminal_verification_protocol(primary)
        result = dict(**engine.frozen_terminal, details=details)
    assert engine.rng.bit_generator.state == before
    global_after = np.random.get_state()
    assert global_after[0] == global_before[0]
    np.testing.assert_array_equal(global_after[1], global_before[1])
    assert global_after[2:] == global_before[2:]
    assert engine.config.terminal_verification_budget == 80
    assert engine._decision_context == 0
    for (actual_x, actual_y), (expected_x, expected_y) in zip(engine.history, history):
        assert actual_x == expected_x
        np.testing.assert_array_equal(actual_y, expected_y)
    return result


def without_record(result):
    result = deepcopy(result)
    result['terminal'].pop('terminal_recording', None)
    return result


@pytest.mark.parametrize('cls', [LegacyNativeAlgorithm, LegacyPosteriorAlgorithm, RolePosteriorAlgorithm])
def test_real_terminal_path_is_unchanged_and_records_full_observed_pool(monkeypatch, cls):
    """Detect changed decisions, extra predictions, lost physical identity or insertion stages."""
    engine = engine_fixture(cls)
    actual = construct(engine)
    with monkeypatch.context() as patch:
        patch.setattr(recording, 'capture', lambda *args, **kwargs: None)
        plain = engine_fixture(cls)
        expected = construct(plain)
    assert without_record(actual) == without_record(expected)
    assert engine.calls == plain.calls
    assert len(engine.calls['score']) == 3 and len(engine.calls['predict']) == 1
    assert expected['terminal']['terminal_recording'] is None
    terminal = actual['terminal']
    assert terminal['candidate_verification_budgets'] == [80, 128, 128]
    assert terminal['max_verification_budget'] == 352
    assert terminal['verification_budget'] == 0 and terminal['attempts'] == []
    trace = terminal['terminal_recording']
    assert trace['status'] == 'frozen' and trace['paid_prefix'] == 13
    assert trace['posterior_context'] == (0 if cls is LegacyNativeAlgorithm else 1)
    assert len(trace['observed']) == 12
    assert trace['observed'][11] == dict(point=list(point(11)), observation_indices=[11, 12], replicate_count=2)
    assert trace['history_observed_indices'] == list(range(12)) + [11]
    stages = {row['phase']: row for row in trace['stages']}
    assert list(stages) == ['start', 'primary_pool', 'primary_order', 'objective_gate',
        'objective_choice', 'safe_pool', 'safe_fallback', 'pre_incumbent',
        'post_incumbent', 'frozen_shortlist']
    for phase in ('primary_pool', 'objective_gate', 'safe_pool'):
        row = stages[phase]
        assert {tuple(x) for x in row['pool']} == set(engine.observations)
        assert len(row['pool']) == 12 and all(len(x) == 1000 for x in row['pool'])
        assert set(row['available_components']) == set(row['components'])
        assert len(row['components']['objective']) == len(row['pool'])
    order = stages['primary_order']
    assert order['ranked_points'] == [stages['primary_pool']['pool'][i] for i in order['order']]
    assert order['selected'] == actual['primary'] == stages['objective_choice']['selected']['point']
    assert order['maximum_violation_probability'] == stages['objective_gate']['maximum_violation_probability'] == .5
    assert stages['safe_pool']['probability_slack'] == .05
    assert stages['safe_fallback']['selected']['point'] == list(point(3))
    before = stages['pre_incumbent']['shortlist']
    after = stages['post_incumbent']['shortlist']
    assert list(point(0)) not in [row['point'] for row in before]
    assert after[1]['point'] == list(point(0)) and after[1]['objective_incumbent']
    assert after[1]['shortlist_role'] == 'frozen_initial_empirical_objective_incumbent'
    assert after[2]['point'] == before[1]['point'] == list(point(3))
    assert before[2]['point'] not in [row['point'] for row in after]
    assert after == stages['frozen_shortlist']['shortlist'] == terminal['frozen_shortlist']


def test_stop_keeps_its_ten_paid_observations_and_no_adaptive_points(monkeypatch):
    """Detect accidental attribution of the full-search prefix to the STOP control."""
    engine = engine_fixture(LegacyNativeAlgorithm, stop=True)
    result = construct(engine)
    with monkeypatch.context() as patch:
        patch.setattr(recording, 'capture', lambda *args, **kwargs: None)
        plain = engine_fixture(LegacyNativeAlgorithm, stop=True)
        assert without_record(result) == without_record(construct(plain))
    assert engine.calls == plain.calls
    trace = result['terminal']['terminal_recording']
    assert trace['paid_prefix'] == 10 and trace['backend_arm'] == 'initialization_only'
    assert len(trace['observed']) == 10
    assert all(row['point'][0] < 10 for row in trace['observed'])


def test_native_bypass_resets_role_record_and_preserves_native_decision(monkeypatch):
    """Detect native scores being attached to the preceding role-terminal record."""
    engine = engine_fixture()
    role = construct(engine)
    role_trace = deepcopy(role['terminal']['terminal_recording'])
    before_calls = deepcopy(engine.calls)
    native = construct(engine, native=True)
    with monkeypatch.context() as patch:
        patch.setattr(recording, 'capture', lambda *args, **kwargs: None)
        plain = engine_fixture()
        expected = construct(plain, native=True)
    assert without_record(native) == without_record(expected)
    assert len(engine.calls['score']) - len(before_calls['score']) == len(plain.calls['score']) == 4
    assert len(engine.calls['predict']) - len(before_calls['predict']) == len(plain.calls['predict']) == 1
    assert engine.calls['score'][len(before_calls['score']):] == plain.calls['score']
    assert native['primary'] == list(point(11)) and role['primary'] == list(point(10))
    trace = native['terminal']['terminal_recording']
    assert trace['posterior_context'] == 0 and role_trace['posterior_context'] == 1
    assert role['terminal']['terminal_recording'] == role_trace
    for stage in trace['stages']:
        if 'components' in stage:
            assert stage['components']['constraint_posterior_source'] == 'native_fixture'
    assert [s['phase'] for s in trace['stages']].count('start') == 1


def test_score_failure_retains_partial_record_without_fabricated_frozen_list():
    """Detect losing already captured ranking when a later terminal score fails."""
    engine = engine_fixture(fail_score_at=2)
    primary, _ = engine._decision_backend_terminal_recommendation(engine._last_terminal_pool)
    before_rng = deepcopy(engine.rng.bit_generator.state)
    with pytest.raises(RuntimeError, match='deterministic terminal score failure'):
        engine._terminal_verification_protocol(primary)
    trace = recording.snapshot(engine)
    assert trace['status'] == 'partial' and trace['paid_prefix'] == 13
    assert [s['phase'] for s in trace['stages']] == ['start', 'primary_pool', 'primary_order']
    assert trace['stages'][-1]['selected'] == list(primary)
    assert engine.frozen_terminal is None and engine._decision_context == 0
    assert engine.config.terminal_verification_budget == 80
    assert engine.rng.bit_generator.state == before_rng
    assert len(engine.calls['score']) == 2 and len(engine.calls['predict']) == 1


@pytest.mark.parametrize('same_policy', [False, True])
def test_record_survives_real_terminal_input_transform_and_allocation(same_policy):
    """Detect trace loss/aliasing or unrecorded role changes in the existing adapters."""
    engine = engine_fixture(same_policy=same_policy)
    frozen = construct(engine)['terminal']
    search = dict(geometry={'fixture': 1}, logical_search_calls=13,
        history=[dict(point=list(x), raw=y.tolist()) for x, y in engine.history])
    parent = dict(domain='fixture', task=dict(d=1000),
        geometry_by_seed={'1': {'problem_kwargs': search['geometry']}},
        decision_protocol=dict(verification_method='normal_quantile_tolerance'))
    cell = dict(name='legacy_role_after_init_point_only', history_arm='legacy_role_after_init_point_only', terminal_backend='role')
    original = input_record(search, frozen, parent, cell, 1, 2, 'fixture/search', 0.)
    assert original['terminal_recording'] == frozen['terminal_recording']
    original_before = deepcopy(original)
    converted = transform(original)
    assert original == original_before
    assert converted['input_changed'] is same_policy
    assert converted['objective_incumbent_position'] == 2
    trace = converted['terminal_recording']
    assert trace['stages'][:-1] == original['terminal_recording']['stages']
    assert trace['stages'][-1]['phase'] == 'same_policy_transform'
    assert trace['stages'][-1]['before'] == original['frozen_shortlist']
    assert trace['stages'][-1]['after'] == converted['frozen_shortlist']
    assert trace['stages'][-1]['changed'] is same_policy
    if same_policy:
        assert converted['primary'] != original['primary']
        assert [r['shortlist_role'] for r in converted['frozen_shortlist']] == [
            'distinct_posterior_challenger', 'initial_objective_incumbent', 'original_remaining']
        assert all('source_shortlist_role' in r for r in converted['frozen_shortlist'])
    candidate = candidate_input(converted)
    assert candidate['terminal_recording'] == converted['terminal_recording']
    assert candidate['frozen_shortlist'] == converted['frozen_shortlist']
    assert candidate['candidate_budgets'] == [80, 120, 120]
    assert candidate['objective_comparison_delta'] == .05/6
    assert candidate['maximum_objective_comparisons'] == 2
    candidate['terminal_recording']['stages'][-1]['after'][0]['point'][0] = 999
    assert converted['terminal_recording']['stages'][-1]['after'][0]['point'][0] != 999
    original['terminal_recording']['observed'][0]['point'][0] = 999
    assert frozen['terminal_recording']['observed'][0]['point'][0] == 0


def test_search_exception_saves_partial_record_with_matching_paid_raw(fake_search, monkeypatch):
    """Detect a terminal failure losing the charged history or mixing attempt IDs."""
    case = fake_search
    original_build = case.build

    def build(*args, **kwargs):
        engine = original_build(*args, **kwargs)
        paid_search = engine.run_search

        def fail_at_terminal():
            paid_search()
            recording.capture(engine, 'start')
            raise RuntimeError('synthetic terminal failure')

        engine.run_search = fail_at_terminal
        return engine

    monkeypatch.setattr(search_runner, 'build_fresh_backend', build)
    with pytest.raises(RuntimeError, match='synthetic terminal failure'):
        search_runner.search(case.output, case.plan, case.seed, case.arm)
    path, = case.directory.glob('failed_terminal_recording_*.json')
    saved = search_runner.read(path)
    raw_path = path.parent/saved['paid_raw_file']
    assert raw_path.stem.removeprefix('failed_search_raw_') == path.stem.removeprefix('failed_terminal_recording_')
    assert saved['returned_raw_count'] == len(search_runner.read(raw_path)) == 3
    assert saved['terminal_recording'] == recording.snapshot(case.engines[0])
    assert saved['terminal_recording']['status'] == 'partial'
    assert saved['terminal_recording']['paid_prefix'] == 13 and not saved['truth_used']
    assert not (case.directory/'search.json').exists()


def test_native_exception_saves_the_actual_partial_record(tmp_path, monkeypatch):
    """Detect native-freeze exception handling discarding the new native record."""
    engine = engine_fixture(fail_score_at=2)
    seed, arm = 17, 'legacy_role_after_init_point_only'
    plan = dict(output_dir='cohort', domain='fixture')
    directory = tmp_path/'cohort'/f'seed_{seed}'/arm
    terminal_runner.save(directory/'search.json', dict(
        history=[dict(point=list(x), raw=y.tolist()) for x, y in engine.history]))
    monkeypatch.setattr(terminal_runner, 'ROOT', tmp_path)
    monkeypatch.setattr(terminal_runner, 'load_pickle', lambda path: engine)
    with pytest.raises(RuntimeError, match='deterministic terminal score failure'):
        terminal_runner.native_terminal(plan, seed, arm)
    path, = directory.glob('failed_native_terminal_recording_*.json')
    saved = terminal_runner.read(path)
    assert saved['terminal_recording'] == recording.snapshot(engine)
    assert saved['terminal_recording']['status'] == 'partial'
    assert saved['terminal_recording']['posterior_context'] == 0
    assert saved['terminal_recording']['stages'][-1]['phase'] == 'primary_order'
    assert not saved['truth_used'] and not (directory/'native_terminal.json').exists()
