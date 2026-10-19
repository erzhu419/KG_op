import unittest

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from performance.mainline_legacy_source_models import _condition_pair, _eta_rule, _stream_head
from performance.mainline_source_task_conditioning import condition_existing_source_task
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior


def make_law(mean, design, common, task, white):
    return dict(mean=mean, common=common, task=task, white=white,
        source_design=design, source_mean=design@mean,
        source_covariance=design@common@design.T
            + block_diag(*[d@task@d.T for d in np.split(design, 2)])
            + white*np.eye(len(design)))


class ExistingSourceTaskConditioningTests(unittest.TestCase):
    def test_same_family_task_cross_and_other_family_common_only(self):
        design = np.array([[1., -.2], [1., .4], [1., -.5], [1., .7]])
        law = make_law(np.array([.3, .1]), design, np.diag([.08, .03]),
                       np.diag([.12, .06]), .01)
        risk = make_law(np.array([-4., .1]), design, np.diag([.05, .02]),
                        np.diag([.08, .03]), .02)
        labels = np.array([.51, .62, .12, .20]); variances = np.full(4, .02)
        settings = {'seed': 1622, 'quadrature_power': 2}
        nodes, logz = _eta_rule(risk, variances, settings, settings['seed'])
        original, _ = _condition_pair(law, risk, labels, variances, nodes, logz)
        conditioned, checks = condition_existing_source_task(
            original, labels, variances, settings, slice(0, 2))
        self.assertLess(checks['maximum_reconstruction_error'], 1e-12)
        self.assertTrue(np.array_equal(conditioned['law']['task'], np.zeros((2, 2))))
        self.assertTrue(np.array_equal(conditioned['risk']['task'], np.zeros((2, 2))))
        self.assertGreater(np.max(np.abs(original['law']['task'])), 0.)
        query = np.array([[1., -.1], [1., .6]])
        # Build the physical query/source covariance independently. Only the
        # first source family shares the query's task random effect.
        cross = np.column_stack([query@(law['common']+law['task'])@design[:2].T,
                                 query@law['common']@design[2:].T])
        prior = query@(law['common']+law['task'])@query.T + law['white']*np.eye(2)
        likelihood = VarianceReferencePosterior(law['source_mean'], law['source_covariance'],
                                                labels, variances)
        for node in (0, len(nodes)-1):
            _, factor, alpha = likelihood.evaluate(nodes[node])
            np.testing.assert_allclose(query@conditioned['mean'][node],
                query@law['mean']+cross@alpha, atol=1e-12)
            np.testing.assert_allclose(query@conditioned['covariance'][node]@query.T
                + law['white']*np.eye(2), prior-cross@cho_solve(factor, cross.T), atol=1e-12)
        eta_cross = np.column_stack([query@(risk['common']+risk['task'])@design[:2].T,
                                     query@risk['common']@design[2:].T])
        eta_gain = cho_solve(cho_factor(risk['source_covariance'], lower=True), eta_cross.T).T
        np.testing.assert_allclose(conditioned['eta_mean']@query.T,
            query@risk['mean']+(nodes-risk['source_mean'])@eta_gain.T, atol=1e-12)
        np.testing.assert_allclose(query@conditioned['eta_covariance']@query.T,
            query@(risk['common']+risk['task'])@query.T-eta_gain@eta_cross.T, atol=1e-12)
        self.assertGreater(np.max(np.abs(conditioned['mean']-original['mean'])), .1)
        # The existing stream projection must not add a second task deviation.
        _, mean, covariance, _, _ = _stream_head(conditioned, query, query, [], 0, 2)
        weights = np.exp(conditioned['log_weights'])
        node_means = conditioned['mean']@query.T
        expected_mean = weights@node_means
        residual = node_means-expected_mean
        expected_covariance = query@np.einsum('k,kij->ij', weights,
            conditioned['covariance'])@query.T + law['white']*np.eye(2)
        expected_covariance += (residual.T*weights)@residual
        np.testing.assert_allclose(mean[:, 0], expected_mean, atol=1e-12)
        np.testing.assert_allclose(covariance[:, 0, :, 0], expected_covariance, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
