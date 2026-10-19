from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from performance.mainline_legacy_backend import (
    LegacyPosteriorAlgorithm, build_backend, restore_point_only_noise,
)
from performance.mainline_legacy_native import LegacyNativeAlgorithm
from performance.mainline_legacy_source_models import _condition_pair, restore_model
from performance.mainline_point_only import PointOnlyResponseMixture
import tests.test_mainline_legacy_source_models as fixtures


def test_single_policy_projection_matches_source_zero_product_with_repeats():
    fixture = fixtures.LegacySourceModelTests()
    fixture.setUp()
    record, _ = _condition_pair(fixture.f, fixture.eta, fixture.labels,
                                fixture.variances, fixture.states, -1.37)
    names = ['spectral/joint/observed_task', 'spectral/joint/deconvolved_task']
    bundle = dict(mean_center=np.zeros(10), mean_scale=np.ones(10),
        risk_center=np.zeros(10), risk_scale=np.ones(10), encoder=fixtures.PublicEncoder(),
        maps={('spectral', 'joint', -1): fixtures.LinearBasis()}, identities=names,
        components=[(record, record), (dict(record, mean=record['mean']+.2), record)],
        log_weights=np.log([.3, .7]))
    problem = fixtures.no_outcomes(fixtures.build_scalarized_problem(
        'InventorySupplyChain', 6, 100, .04, .05, [.5, .5]))
    problem.tau = .37
    bank = [tuple([value]*6) for value in (20, 40, 60)]
    history = [{'point': bank[0], 'raw': [.27, .37+.31]},
               {'point': bank[1], 'raw': [.49, .37+.18]},
               {'point': bank[0], 'raw': [.33, .37+.32]}]
    product = PointOnlyResponseMixture.from_model(restore_model(
        bundle, 'source_structural_mixture', problem, bank))
    for row in history:
        product.update_raw(bank.index(row['point']), [row['raw'][0], row['raw'][1]-problem.tau])
    expected = product.moment_state()['noise']
    actual = restore_point_only_noise(bundle, problem, bank[::-1], history)
    np.testing.assert_allclose(actual, expected[::-1], rtol=2e-11, atol=2e-11)
    # A foreign raw changes no noise marginal, including an unseen policy.
    foreign = {'point': bank[1], 'raw': [1.4, problem.tau+1.2]}
    changed = restore_point_only_noise(bundle, problem, bank, history+[foreign])
    np.testing.assert_allclose(changed[[0, 2]], expected[[0, 2]], rtol=0, atol=2e-11)
    assert abs(changed[1, 1]-expected[1, 1]) > 1e-5


def bare_engine(arm, bank):
    engine = object.__new__(LegacyPosteriorAlgorithm)
    engine.backend_arm, engine.backend_bundle = arm, object()
    engine.history = [(bank[0], [.3, .1]), (bank[1], [.4, -.2])]
    engine.observations = {point: [raw] for point, raw in engine.history}
    engine.problem = SimpleNamespace(tau=2., alpha=.05)
    engine._moment_cache, engine._local_noise_cache = {}, {}
    engine._decision_context, engine.backend_trace = 0, []
    return engine


def test_local_noise_changes_terminal_only_and_refreshes_after_new_raw():
    bank = [(0,), (1,), (2,)]
    state = {'mean': np.array([[.3, 0., -.4, 0.], [.5, 0., -.3, 0.], [.4, 0., -.2, 0.]]),
             'cov': np.eye(12).reshape(3, 4, 3, 4)*.02,
             'noise': np.array([[.01, .3], [.02, .4], [.03, .5]])}
    full = bare_engine('current_structural_mixture', bank)
    local = bare_engine('current_structural_local_noise', bank)
    local_noise = np.array([[7., .001], [8., .002], [9., .003]])
    def capture(self, models, variance, pool, **kwargs):
        return models, variance, kwargs['task_ensemble']
    with patch('performance.mainline_legacy_source_models.restore_moment_state', return_value=state) as restored, \
         patch('performance.mainline_legacy_backend.restore_point_only_noise', return_value=local_noise) as own, \
         patch.object(LegacyNativeAlgorithm, '_terminal_bayes_risk_components', capture):
        scores = [engine._score_backend('constrained_ts', bank, None, None, None, engine.problem,
            rng=np.random.default_rng(19), decision_aleatoric_mode='posterior_central',
            decision_ambiguity_mode='posterior_nominal')['total'] for engine in (full, local)]
        np.testing.assert_array_equal(*scores)
        assert not own.called
        assert all(call.args[1] == 'source_structural_mixture' for call in restored.call_args_list)
        assert local._terminal_bayes_risk_components(['native'], 'native_noise', bank,
            task_ensemble='native_experts') == (['native'], 'native_noise', 'native_experts')
        with local._posterior_decision():
            views, noise, experts = local._terminal_bayes_risk_components([], None, bank)
            assert experts is None
            np.testing.assert_array_equal(views[0].posterior_mean_many(bank), state['mean'][:, 0])
            np.testing.assert_array_equal(views[1].C, state['cov'][:, 2, :, 2])
            np.testing.assert_array_equal(noise.predict_variance_many(0, bank, local.problem), state['noise'][:, 0])
            np.testing.assert_array_equal(noise.predict_variance_many(1, bank, local.problem), local_noise[:, 1])
            local._terminal_bayes_risk_components([], None, bank)
            assert own.call_count == 1
            local.history.append((bank[0], [.5, -.3]))
            local._terminal_bayes_risk_components([], None, bank)
            assert own.call_count == 2
            assert len(own.call_args.args[3]) == 3
        # Terminal substitution never mutates the full acquisition cache.
        np.testing.assert_array_equal(local._posterior_views(bank)[1].noise, state['noise'])


def test_explicit_shared_initialization_accepts_new_seed_for_all_arms():
    rows = [{'point': [1], 'raw': [.2, -.1]}]
    for arm in ('legacy_native_v69', 'initialization_only', 'same_source_gp',
                'current_structural_mixture', 'current_structural_local_noise'):
        with patch('performance.mainline_legacy_backend.build_native', return_value=SimpleNamespace()) as native, \
             patch('performance.mainline_legacy_backend.cached_initial_rows', side_effect=AssertionError('old seed')):
            engine = build_backend('InventorySupplyChain', arm, bundle={}, source_prior={},
                                   seed=2026101631, initial_rows=rows)
            assert engine.backend_arm == arm
            assert native.call_args.kwargs['initial_rows'] is rows
            assert native.call_args.kwargs['seed'] == 2026101631
            assert native.call_args.kwargs['N'] == (10 if arm == 'initialization_only' else 13)
