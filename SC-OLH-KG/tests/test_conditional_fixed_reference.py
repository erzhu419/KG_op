import copy
import unittest

import numpy as np

from core.conditional_fixed_reference import fixed_guarded_projected_tower
from core.conditional_terminal_kg import ConditionalTerminalActions
from representation.joint_response_mixture import JointResponseMixture
from test_positive_source_mixture import field, terminal


class ConditionalFixedReferenceTests(unittest.TestCase):
    def test_retained_identity_error_bound_preserves_all_weights(self):
        model = JointResponseMixture(['A', 'B'], [
            (field([.1, .7], [.3, .7]), field([-.5, .15], [.8, .2])),
            (field([.8], [1.]), field([-.55], [1.]))], prior_weights=[1-1e-7, 1e-7])
        parent = terminal(); current = ConditionalTerminalActions(parent.shortlists, parent.incumbents, parent.protocol)
        _, _, actions = model.terminal_values(current, 360, 8, 202610077)
        kwargs = {'terminal_power': 8, 'fantasy_powers': [1, 2], 'seed': 202610077, 'protocol': current.protocol}
        full, heads = fixed_guarded_projected_tower(model, actions[2], 2, **kwargs)
        prefix = [{**h, 'future_statistics': h['future_statistics'][:2],
            'predictive_labels': h['predictive_labels'][:2]} for h in heads]
        mixed, updated = fixed_guarded_projected_tower(model, actions[2], 2, completed_heads=prefix,
            identity_fantasy_powers={0: 2, 1: 1}, **kwargs)
        self.assertEqual(mixed[-1]['identity_fantasies_per_head'], [4, 2])
        bound = mixed[-1]['retained_identity_expectation_error_bound']
        self.assertAlmostEqual(bound, 1e-7, places=15)
        self.assertLessEqual(abs(mixed[-1]['expected_future_fixed_reward']-full[-1]['expected_future_fixed_reward']), bound)
        self.assertEqual(len(updated), 4)
        self.assertEqual(mixed[0], full[0])
        for head in (0, 1):
            actual = next(r for r in updated if (r['identity'], r['head']) == (0, head))
            expected = next(r for r in heads if (r['identity'], r['head']) == (0, head))
            self.assertEqual(actual['future_statistics'][:2], expected['future_statistics'][:2])
            self.assertEqual(actual['predictive_labels'][:2], expected['predictive_labels'][:2])
            np.testing.assert_allclose(actual['future_statistics'], expected['future_statistics'], atol=1e-11, rtol=0.)
            np.testing.assert_allclose(actual['predictive_labels'], expected['predictive_labels'], atol=1e-11, rtol=0.)
        np.testing.assert_allclose(model.weights, [1-1e-7, 1e-7], atol=1e-15)

    def test_shared_identity_reference_matches_terminal_and_checkpoint_replay(self):
        model = JointResponseMixture(['A', 'B'], [
            (field([.1, .7], [.3, .7]), field([-.5, .15], [.8, .2])),
            (field([.8], [1.]), field([-.55], [1.]))], prior_weights=[.2, .8])
        before = copy.deepcopy(model)
        parent = terminal(); current = ConditionalTerminalActions(parent.shortlists, parent.incumbents, parent.protocol)
        values, _, actions = model.terminal_values(current, 360, 12, 202610077)
        checkpoints = []
        rows, heads = fixed_guarded_projected_tower(model, actions[2], 2, terminal_power=12,
            fantasy_powers=[1, 2], seed=202610077, protocol=current.protocol,
            checkpoint=lambda h: checkpoints.append(len(h)))
        self.assertAlmostEqual(rows[-1]['prior_fixed_reward'], values[2], places=12)
        f = np.array([h['prior_statistics'] for h in heads if h['head'] == 0])
        g = np.array([h['prior_statistics'] for h in heads if h['head'] == 1])
        self.assertGreater(abs((model.weights @ f) @ (model.weights @ g)-values[2]), .001)
        self.assertEqual(checkpoints, [1, 2, 3, 4])
        checkpoints.clear()
        replay, _ = fixed_guarded_projected_tower(model, actions[2], 2, terminal_power=12,
            fantasy_powers=[1, 2], seed=202610077, protocol=current.protocol,
            completed_heads=heads[:1], checkpoint=lambda h: checkpoints.append(len(h)))
        self.assertEqual(replay, rows); self.assertEqual(checkpoints, [2, 3, 4])
        self.assertFalse(rows[-1]['adaptive_maximum_evaluated'])
        self.assertEqual(model.paid_episodes, before.paid_episodes)
        self.assertEqual(model.history, before.history)
        for old_pair, new_pair in zip(before.components, model.components):
            for old, new in zip(old_pair, new_pair):
                np.testing.assert_array_equal(new.means, old.means)
                np.testing.assert_array_equal(new.covariances, old.covariances)
                np.testing.assert_array_equal(new.log_weights, old.log_weights)


if __name__ == '__main__':
    unittest.main()
