import unittest

import numpy as np

from performance.diagnose_mainline_objective_transfer import decompose_mean_shift, identity_predictions


class ObjectiveTransferTests(unittest.TestCase):
    def test_interaction_is_required_for_exact_mean_shift(self):
        w0, w1 = np.array([.8, .2]), np.array([.3, .7])
        m0 = np.array([[0., 2.], [4., -1.]])
        m1 = np.array([[1., 1.], [2., 2.]])
        terms = decompose_mean_shift(w0, m0, w1, m1)
        np.testing.assert_allclose(sum(terms[k] for k in ('within_identity', 'identity_reweight', 'identity_interaction')), w1@m1-w0@m0)
        self.assertGreater(np.max(abs(terms['identity_interaction'])), .1)
        self.assertLess(terms['reconstruction_max_absolute_error'], 1e-14)

    def test_head_only_and_cumulative_weights_follow_same_saved_path(self):
        weights = np.array([.5, .5]); means = np.array([[0., 1.], [3., 2.]])
        first = np.array([[2., -1.], [0., 1.]])
        second = np.array([[0., 1.], [1., 0.]])
        rules, predictions = identity_predictions(weights, first, means)
        self.assertGreater(rules['f_only'][0], .5)
        self.assertLess(rules['g_only'][0], .5)
        np.testing.assert_allclose(rules['joint'], weights)
        cumulative, _ = identity_predictions(weights, first+second, means)
        sequential, _ = identity_predictions(rules['f_only'], second, means)
        np.testing.assert_allclose(cumulative['f_only'], sequential['f_only'])
        np.testing.assert_allclose(predictions['fixed'], weights@means)


if __name__ == '__main__':
    unittest.main()
