import copy
import unittest

import numpy as np

from performance.check_historical_learned_prior_kg_precision import check_batch_filter
from representation.joint_response_mixture import JointResponseMixture
from test_positive_source_mixture import field


class LearnedPriorKGPrecisionTests(unittest.TestCase):
    def test_full_field_batch_completion_matches_paid_filter_without_mutation(self):
        model = JointResponseMixture(['first', 'second'], [
            (field([.1, .7], [.3, .7]), field([-.5, .15], [.8, .2])),
            (field([.8], [1.]), field([-.55, -.2], [.6, .4]))], prior_weights=[.8, .2])
        for pair in model.components:
            for current in pair:
                current.covariances[:, 0::2, 0::2] = .04 * np.eye(3)
                current.covariances[:, 1::2, 1::2] = .2 * np.eye(3)
                current.covariances[:, 2, 3] = current.covariances[:, 3, 2] = .025
        before = copy.deepcopy(model)
        error = check_batch_filter(model, 1, [.35, -.45])
        self.assertLess(error, 1e-8)
        np.testing.assert_array_equal(model.weights, before.weights)
        self.assertEqual(model.paid_episodes, 0)
        self.assertEqual(model.history, [])
        for new_pair, old_pair in zip(model.components, before.components):
            for new, old in zip(new_pair, old_pair):
                np.testing.assert_array_equal(new.means, old.means)
                np.testing.assert_array_equal(new.covariances, old.covariances)
                np.testing.assert_array_equal(new.log_weights, old.log_weights)


if __name__ == '__main__':
    unittest.main()
