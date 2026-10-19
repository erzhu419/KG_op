"""Candidate release must constrain both KG and its terminal opportunity set."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance.mainline_moment_policy import available_state, query_action, terminal_action


class MainlineCandidateReleaseTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {'objective_reference': 1., 'chance_alpha': .05,
                         'familywise_delta': .05}
        mean = np.array([[.9, -6., -.4, -6.], [.6, -6., -.4, -6.],
                         [.05, -6., -.4, -6.]])
        covariance = np.diag([.0001, .01, .0001, .01,
                              .04, .01, .0001, .01,
                              .36, .01, .0001, .01])
        covariance[0, 4] = covariance[4, 0] = .0005
        covariance[4, 8] = covariance[8, 4] = .01
        covariance[1, 6] = covariance[6, 1] = .0001
        self.state = {'mean': mean, 'cov': covariance.reshape(3, 4, 3, 4),
                      'noise': np.full((3, 2), .001)}

    def test_strong_future_policy_is_available_only_after_release(self):
        first = available_state(self.state, 2)
        # Retain both response cross-covariance and the cross-head entry;
        # global policy 1 must remain index 1 for the caller's paid update.
        self.assertEqual(first['cov'].shape, (2, 4, 2, 4))
        self.assertEqual(first['cov'][0, 0, 1, 0], .0005)
        self.assertEqual(first['cov'][0, 1, 1, 2], .0001)
        first_query = query_action(first, 100, self.protocol)
        self.assertEqual(first_query['query'], 1)
        self.assertEqual(len(first_query['scores73']), 2)
        self.assertGreater(first_query['adaptation_value'], .005)
        self.assertEqual(terminal_action(first, 100, self.protocol)['action']['primary'], 1)
        final = available_state(self.state, 3)
        final_query = query_action(final, 100, self.protocol)
        self.assertEqual(final_query['query'], 2)
        self.assertGreater(final_query['adaptation_value'], .05)
        self.assertEqual(terminal_action(final, 100, self.protocol)['action']['primary'], 2)

    def test_unreleased_marginals_cannot_change_first_round_kg(self):
        before = query_action(available_state(self.state, 2), 100, self.protocol)
        changed = {key: value.copy() for key, value in self.state.items()}
        changed['mean'][2] = [-100., 3., -10., 2.]
        changed['cov'][2, :, 2, :] *= 10
        changed['noise'][2] *= 100
        after = query_action(available_state(changed, 2), 100, self.protocol)
        self.assertEqual(after['query'], before['query'])
        np.testing.assert_array_equal(after['scores73'], before['scores73'])


if __name__ == '__main__':
    unittest.main()
