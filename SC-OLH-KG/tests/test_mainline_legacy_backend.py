from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from acquisition.decision_backends import _joint_gpr_draw, score_decision_backend
from performance.mainline_legacy_backend import (
    LegacyPosteriorAlgorithm, posterior_views,
)
from performance.mainline_legacy_native import LegacyNativeAlgorithm


def test_coherent_covariance_and_constraint_scale_reach_native_ts():
    bank = [(0,), (1,)]
    state = {'mean': np.array([[.3, 0., -.4, 0.], [.5, 0., -.3, 0.]]),
             'cov': np.zeros((2, 4, 2, 4)), 'noise': np.full((2, 2), .001)}
    state['cov'][:, 0, :, 0] = [[.02, .01], [.01, .03]]
    state['cov'][:, 2, :, 2] = [[.01, .008], [.008, .02]]
    views, noise = posterior_views(bank, state, tau=2.)
    reversed_bank = bank[::-1]
    features = views[0].augmented_feature_matrix(reversed_bank)
    np.testing.assert_allclose(features @ views[0].C @ features.T, [[.03, .01], [.01, .02]])
    np.testing.assert_allclose(views[1].posterior_mean_many(bank), [1.6, 1.7])
    problem = SimpleNamespace(tau=2., alpha=.05)
    result = score_decision_backend('constrained_ts', bank, *views, noise, problem,
        rng=np.random.default_rng(19), decision_aleatoric_mode='posterior_central',
        decision_ambiguity_mode='posterior_nominal')
    rng = np.random.default_rng(19)
    objective = _joint_gpr_draw(views[0], bank, rng)
    constraint = _joint_gpr_draw(views[1], bank, rng)
    assert np.all(constraint + 1.6448536269514722*np.sqrt(.001) <= 2.)
    np.testing.assert_allclose(result['total'], -objective)


def test_posterior_replacement_is_scoped_to_final_decision():
    engine = object.__new__(LegacyPosteriorAlgorithm)
    engine._decision_context = 0
    engine._posterior_views = lambda pool: (['new_obj', 'new_con'], 'new_noise')
    calls = []
    def record(self, models, variance, pool, **kwargs):
        calls.append((models, variance, kwargs['task_ensemble']))
        return {}
    with patch.object(LegacyNativeAlgorithm, '_terminal_bayes_risk_components', record):
        engine._terminal_bayes_risk_components(['old'], 'old_noise', [1], task_ensemble='old_experts')
        with engine._posterior_decision():
            engine._terminal_bayes_risk_components(['old'], 'old_noise', [1], task_ensemble='old_experts')
        engine._terminal_bayes_risk_components(['old'], 'old_noise', [1], task_ensemble='old_experts')
    assert calls == [(['old'], 'old_noise', 'old_experts'),
                     (['new_obj', 'new_con'], 'new_noise', None),
                     (['old'], 'old_noise', 'old_experts')]
