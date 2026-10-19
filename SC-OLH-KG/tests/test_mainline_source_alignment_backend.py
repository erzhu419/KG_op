"""Source routing and original decisions with synthetic objects only.

No source pickle, simulator, fitted source law or posterior restoration is
used. Actual coefficient projection, constrained TS and terminal construction
are exercised around supplied substitute moments.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from acquisition.decision_backends import score_decision_backend
from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig
from core.gpr import ParametricGPR
from performance import mainline_legacy_source_models as source_models
from performance import mainline_role_backend as backend
from performance import mainline_source_alignment_backend as alignment
from performance import run_mainline_legacy_backend_bridge as bridge
from performance.mainline_legacy_backend import LegacyPosteriorAlgorithm, posterior_views
from performance.mainline_legacy_native import LegacyNativeAlgorithm
from performance.mainline_role_prior import role_inputs, role_prior
from test_mainline_role_prior import LinearBasis, PublicEncoder, source_pair
from test_mainline_terminal_recording import construct, engine_fixture


AFTER = 'source_permuted_role_after_init_point_only'
POINT = 'source_permuted_role_point_only'
ORIGINAL_SEARCH = (
    'legacy_native_neighbors', 'same_source_gp', 'current_structural_mixture',
    'legacy_role_full', 'legacy_role_after_init_point_only',
    'legacy_role_point_only', 'initialization_only',
)


def forbidden(*args, **kwargs):
    raise AssertionError('No simulation, source fit or posterior restoration in wiring checks')


def point(i):
    return (i,) + (0,) * 998 + (i + 1,)


class SyntheticBasis:
    feature_dim = 2

    def __init__(self, prior, problem):
        self.meta_prior, self.problem = prior, problem

    def source_parametric_prior(self):
        return deepcopy(self.meta_prior.law)

    def features_many(self, rows):
        return np.asarray(rows)[:, [0, -1]] / 100 * self.meta_prior.scale


def synthetic_prior(offset):
    return SimpleNamespace(scale=1 + offset, law=dict(
        mean=np.array([.9 + offset, .2, -.1]),
        covariance=np.diag([.08 + offset / 10, .04, .03]),
        deviation_variance=.12 + offset / 10,
        diagnostics=dict(source_record_count=12)))


def plan():
    return dict(domain='fixture', geometry_by_seed={
        '17': dict(problem_kwargs=dict(task_geometry_shift=[.01, -.02, .03],
                                       task_geometry_radius_scale=1.02))},
        observation_seed_by_task={'17': 71},
        frozen_assets=dict(native_source_prior_file='real_native.pkl',
                           source_models_file='real_bundle.pkl'),
        source_permuted_assets=dict(native_source_prior_file='permuted_native.pkl',
                                   source_models_file='permuted_bundle.pkl'))


def synthetic_state(bank):
    ids = np.asarray([x[0] for x in bank], dtype=float)
    mean = np.column_stack([1 - ids / 100, np.full(len(bank), -4.),
                            -.25 + ids / 200, np.full(len(bank), -5.)])
    return dict(mean=mean, cov=(np.eye(4 * len(bank)) * .02).reshape(
        len(bank), 4, len(bank), 4), noise=np.full((len(bank), 2), .001))


def fake_native(domain, *, seed, N, source_prior, initial_rows,
                algorithm_class, problem_kwargs):
    engine = object.__new__(algorithm_class)
    engine.config = SingleOLHKGConfig(seed=seed, N=N, n0=10,
        source_constraint_mean_coefficient_prior=True,
        source_constraint_mean_hyperlaw_mode='single_gaussian_draw',
        source_constraint_mean_deviation_mode='latent_shared')
    engine.problem = SimpleNamespace(d=1000, L=100, tau=.37, alpha=.05,
        meta_prior=source_prior, geometry=deepcopy(problem_kwargs),
        simulate=forbidden, true_objective=forbidden, true_chance_margin=forbidden)
    engine.source_prior = source_prior
    engine.gpr = [None, ParametricGPR(1000, basis_map=SyntheticBasis(source_prior, engine.problem))]
    engine.history = [(point(i), np.array([.2, .4])) for i in range(10)]
    engine.initial_rows = initial_rows
    engine.observations = {x: [y] for x, y in engine.history}
    engine.rng = np.random.default_rng(seed)
    engine.rec_rng = np.random.default_rng(seed + 1)
    engine._moment_cache, engine._local_noise_cache = {}, {}
    engine.backend_trace, engine._decision_context = [], 0
    return engine


@pytest.fixture
def wiring(monkeypatch):
    originals = dict(real_native=synthetic_prior(0), permuted_native=synthetic_prior(.5),
                     real_bundle=object(), permuted_bundle=object())
    reads = []

    def load(path):
        reads.append(path.name)
        return originals[path.stem]

    monkeypatch.setattr(backend, 'load_pickle', load)
    monkeypatch.setattr(backend, 'build_native', fake_native)
    monkeypatch.setattr(alignment, 'ObservableConstraintMeanBasis', SyntheticBasis)
    for name in ('restore_model', 'restore_moment_state', 'fit_source_models'):
        monkeypatch.setattr(source_models, name, forbidden)
    return originals, reads


@pytest.mark.parametrize('arm', [AFTER, POINT])
def test_permuted_build_keeps_real_frontend_and_actual_geometry(wiring, arm):
    assets, reads = wiring
    initial = [{'point': list(point(0)), 'raw': [.2, .4]}]
    control = backend.build_fresh_backend(plan(), 17, arm, initial)
    original = backend.build_fresh_backend(plan(), 17, 'legacy_role_after_init_point_only', initial)
    context = control.role_projection_context
    assert control.source_prior is control.problem.meta_prior is assets['real_native']
    assert control.gpr[1].basis_map.meta_prior is assets['real_native']
    assert context.source_prior is context.gpr[1].basis_map.meta_prior is assets['permuted_native']
    assert control.backend_bundle is assets['permuted_bundle']
    assert context.gpr[1] is not control.gpr[1]
    assert context.problem is control.problem and context.gpr[1].d == 1000
    assert context.problem.geometry == plan()['geometry_by_seed']['17']['problem_kwargs']
    assert context.problem.tau == .37
    assert context.config is not control.config and vars(context.config) == vars(control.config)
    assert context.config.observation_seed == 71 and control.initial_rows is initial
    assert not hasattr(context, 'history') and not hasattr(context, 'rng')
    assert reads[:3] == ['real_native.pkl', 'permuted_native.pkl', 'permuted_bundle.pkl']
    assert control.rng.bit_generator.state == original.rng.bit_generator.state
    assert control.rec_rng.bit_generator.state == original.rec_rng.bit_generator.state
    for namespace in ('task_expert', 'observed_neighbor'):
        np.testing.assert_array_equal(control._proposal_rng(10, namespace).normal(size=4),
                                      original._proposal_rng(10, namespace).normal(size=4))


def test_role_inputs_replace_all_native_risk_terms_and_center_tau_once(wiring, monkeypatch):
    from performance import mainline_role_prior as role
    assets, _ = wiring
    engine = backend.build_fresh_backend(plan(), 17, AFTER, [])
    global_before = np.random.get_state()
    identity_context = alignment.build_role_projection_context(engine, assets['real_native'])
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0] and global_before[2:] == global_after[2:]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    bank = [point(20), point(40)]
    monkeypatch.setattr(role, 'legacy_exposure_descriptors',
                        lambda problem, rows, dimension: np.zeros((len(rows), 10)))
    bundle = dict(mean_center=np.zeros(10), mean_scale=np.ones(10),
        risk_center=np.zeros(10), risk_scale=np.ones(10), encoder=PublicEncoder(),
        maps={('spectral', 'head_specific', head): LinearBasis() for head in (0, 2, 3)},
        identities=['spectral/head_specific/observed_task'],
        components=[(source_pair(), source_pair())], log_weights=np.zeros(1))
    original, original_design = role_inputs(bundle, engine, bank)
    same, same_design = role_inputs(bundle, identity_context, bank)
    permuted, permuted_design = role_inputs(bundle, engine.role_projection_context, bank)
    real_law, same_law, permuted_law = [x['components'][0][1]['law']
                                      for x in (original, same, permuted)]
    for key in ('mean', 'common', 'task', 'white'):
        np.testing.assert_array_equal(real_law[key], same_law[key])
    np.testing.assert_array_equal(original_design[0][1][0], same_design[0][1][0])
    source_law = assets['permuted_native'].law
    expected_mean = source_law['mean'].copy()
    expected_mean[0] -= .37
    np.testing.assert_array_equal(permuted_law['mean'], expected_mean)
    expected_covariance = source_law['covariance'] + np.eye(3) * (
        source_law['deviation_variance'] * (1 - 1 / 12) / 3)
    np.testing.assert_allclose(permuted_law['common'], expected_covariance)
    assert permuted_law['white'] == pytest.approx(source_law['deviation_variance'] / 12)
    assert not np.array_equal(real_law['mean'], permuted_law['mean'])
    assert not np.array_equal(original_design[0][1][0], permuted_design[0][1][0])
    assert permuted['components'][0][0] is bundle['components'][0][0]
    for field in ('eta_mean', 'eta_covariance', 'risk', 'log_weights'):
        assert permuted['components'][0][1][field] is bundle['components'][0][1][field]
    assert permuted['encoder'] is bundle['encoder']
    assert permuted['log_weights'] is bundle['log_weights']
    # Projection may not modify the source law or the proposal GPR.
    assert assets['permuted_native'].law['mean'][0] == 1.4
    np.testing.assert_array_equal(role_prior(engine)['mean'], assets['real_native'].law['mean'])


@pytest.mark.parametrize('arm', sorted(backend.ALL_ROLE_ARMS))
def test_explicit_operator_context_and_per_engine_cache(wiring, monkeypatch, arm):
    assets, _ = wiring
    calls = []

    def moment(operator):
        def calculate(bundle, context, bank, history, **kwargs):
            calls.append((operator, bundle, context, kwargs.get('cache')))
            return synthetic_state(bank)
        return calculate

    monkeypatch.setattr(backend, 'role_moment_state', moment('full'))
    monkeypatch.setattr(backend, 'after_init_point_only_moment_state', moment('after_init_point_only'))
    monkeypatch.setattr(backend, 'point_only_moment_state', moment('point_only'))
    engines = [backend.build_fresh_backend(plan(), 17, arm, []) for _ in range(2)]
    for engine in engines:
        first = engine._posterior_views([point(11)])
        assert engine._posterior_views([point(11)]) is first
    assert len(calls) == 2
    for index, engine in enumerate(engines):
        operator, bundle, context, cache = calls[index]
        assert operator == backend.ROLE_OPERATORS[arm]
        assert context is getattr(engine, 'role_projection_context', engine)
        expected = assets['permuted_bundle' if arm in alignment.SOURCE_PERMUTED_ROLE_ARMS else 'real_bundle']
        assert bundle is expected
        if operator != 'full':
            assert cache is engine._point_only_cache
    assert engines[0]._moment_cache is not engines[1]._moment_cache
    if arm != 'legacy_role_full':
        assert calls[0][3] is not calls[1][3]
    engines[0].history.append((point(11), np.array([.2, .4])))
    engines[0]._posterior_views([point(11)])
    assert len(calls) == 3 and calls[-1][3] is calls[0][3]


@pytest.mark.parametrize('arm', [AFTER, POINT, 'legacy_role_after_init_point_only'])
def test_checkpoint_bundle_reload_preserves_context_and_paid_prefix(wiring, monkeypatch, arm):
    assets, reads = wiring
    engine = backend.build_fresh_backend(plan(), 17, arm, [])
    saved = []
    monkeypatch.setattr(bridge, 'save_pickle', lambda path, item: saved.append(deepcopy(item)))
    bridge.save_engine('fixture-not-written.pkl', engine)
    recovered = saved[0]
    assert recovered.backend_bundle is None and recovered._moment_cache == {}
    context = getattr(recovered, 'role_projection_context', None)
    before = deepcopy(recovered.rng.bit_generator.state)
    backend.reload_backend_bundle(recovered, plan(), arm)
    expected = 'permuted_bundle' if arm in alignment.SOURCE_PERMUTED_ROLE_ARMS else 'real_bundle'
    assert recovered.backend_bundle is assets[expected] and reads[-1] == expected + '.pkl'
    assert getattr(recovered, 'role_projection_context', None) is context
    assert recovered.rng.bit_generator.state == before and len(recovered.history) == 10
    if context is not None:
        assert context.problem is recovered.problem
        assert context.source_prior is context.gpr[1].basis_map.meta_prior


def old_build(plan, seed, arm, initial_rows):
    """Pre-Stage89 construction, retained here as the finite regression oracle."""
    prior = backend.load_pickle(backend.ROOT / plan['frozen_assets']['native_source_prior_file'])
    cls = (backend.RolePosteriorAlgorithm if arm in backend.ROLE_ARMS else
           LegacyPosteriorAlgorithm if arm in backend.MODEL_ARMS else LegacyNativeAlgorithm)
    engine = fake_native(plan['domain'], seed=seed, N=10 if arm == backend.STOP else 13,
        source_prior=prior, initial_rows=initial_rows, algorithm_class=cls,
        problem_kwargs=plan['geometry_by_seed'][str(seed)]['problem_kwargs'])
    engine.config.observation_seed = plan['observation_seed_by_task'][str(seed)]
    engine.backend_arm = 'legacy_native_v69' if arm == 'legacy_native_neighbors' else arm
    if arm in backend.MODEL_ARMS:
        engine.backend_bundle = backend.load_pickle(backend.ROOT / plan['frozen_assets']['source_models_file'])
    return engine


@pytest.mark.parametrize('arm', ORIGINAL_SEARCH)
def test_original_search_classes_real_ts_choices_and_random_streams_unchanged(wiring, monkeypatch, arm):
    calls = []

    def moment(bundle, context, bank, history, **kwargs):
        calls.append(context)
        return synthetic_state(bank)

    for name in ('role_moment_state', 'after_init_point_only_moment_state', 'point_only_moment_state'):
        monkeypatch.setattr(backend, name, moment)
    monkeypatch.setattr(source_models, 'restore_moment_state',
        lambda bundle, arm, problem, bank, history: synthetic_state(bank))
    expected, actual = old_build(plan(), 17, arm, []), backend.build_fresh_backend(plan(), 17, arm, [])
    assert type(actual) is type(expected) and vars(actual.config) == vars(expected.config)
    assert not hasattr(actual, 'role_projection_context')
    candidates = [point(i) for i in range(13)]
    # Use the actual constrained-TS scorer and draws, with finite substitute
    # moments. The reference bypasses the new dispatch/context route entirely.
    views, variance = posterior_views(candidates, synthetic_state(candidates), actual.problem.tau)
    reference = score_decision_backend('constrained_ts', candidates, *views, variance,
        expected.problem, rng=expected.rng, observed=expected.observations)
    if arm in backend.MODEL_ARMS:
        observed = actual._score_backend('constrained_ts', candidates, *views, variance,
            actual.problem, rng=actual.rng, observed=actual.observations)
    else:
        observed = score_decision_backend('constrained_ts', candidates, *views, variance,
            actual.problem, rng=actual.rng, observed=actual.observations)
    np.testing.assert_array_equal(observed['total'], reference['total'])
    assert np.argmax(observed['total']) == np.argmax(reference['total'])
    assert actual.rng.bit_generator.state == expected.rng.bit_generator.state
    assert actual.rec_rng.bit_generator.state == expected.rec_rng.bit_generator.state
    assert all(context is actual for context in calls)
    np.testing.assert_array_equal(actual._proposal_rng(10, 'task_expert').normal(size=4),
                                  expected._proposal_rng(10, 'task_expert').normal(size=4))


@pytest.mark.parametrize('arm,native', [
    ('legacy_role_full', False), ('legacy_role_full', True),
    ('legacy_role_after_init_point_only', False), ('legacy_role_after_init_point_only', True),
    ('legacy_native_neighbors', False), ('same_source_gp', False),
    ('legacy_role_point_only', False), ('current_structural_mixture', False),
    ('initialization_only', False),
])
def test_all_nine_original_terminal_controls_keep_selection_and_352_cap(arm, native):
    cls = (backend.RolePosteriorAlgorithm if arm in backend.ROLE_ARMS else
           LegacyPosteriorAlgorithm if arm in backend.MODEL_ARMS else LegacyNativeAlgorithm)
    engine = engine_fixture(cls, stop=arm == backend.STOP)
    engine.backend_arm = 'legacy_native_v69' if arm == 'legacy_native_neighbors' else arm
    result = construct(engine, native=native)
    terminal = result['terminal']
    assert terminal['candidate_verification_budgets'] == [80, 128, 128]
    assert terminal['max_verification_budget'] == 352
    assert terminal['verification_budget'] == 0 and terminal['attempts'] == []
    expected_primary = 0 if arm == backend.STOP else (
        11 if native or cls is LegacyNativeAlgorithm else 10)
    assert result['primary'][0] == expected_primary
    shortlist = [row['point'][0] for row in terminal['frozen_shortlist']]
    if arm == backend.STOP:
        assert shortlist[0] == 0 and 3 in shortlist and len(set(shortlist)) == 3
    else:
        assert shortlist == [expected_primary, 0, 3]
