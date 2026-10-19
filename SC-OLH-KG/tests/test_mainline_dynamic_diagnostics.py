import json
import unittest

import numpy as np

from performance.mainline_dynamic_diagnostics import dynamic_diagnostics


class DynamicDiagnosticsTests(unittest.TestCase):
    def fixture(self):
        truth = np.column_stack([np.linspace(.2, .7, 6), np.full(6, -5.),
                                 np.full(6, -.3), np.full(6, -5.)])
        initial = truth+1
        after1 = truth+.5
        after1[[0, 2]] = truth[[0, 2]]+100
        after2 = truth+.25
        after2[[0, 2, 4]] = truth[[0, 2, 4]]+100
        def snap(mean):
            return {'mean': mean.tolist()}
        def terminal(index, budget):
            return {'action': {'primary': index, 'budgets': [budget]}}
        rounds = [
            {'round': 1, 'decision': {'query': 2}, 'before': snap(initial), 'after': snap(after1),
             'terminal': terminal(3, 9), 'terminal_with_new_unsampled_marginals_held_fixed': terminal(0, 9)},
            {'round': 2, 'decision': {'query': 4}, 'before': snap(after1), 'after': snap(after2),
             'terminal': terminal(4, 8), 'terminal_with_new_unsampled_marginals_held_fixed': terminal(1, 8)}]
        frozen = {'bank': [[i] for i in range(6)], 'plan': {'initialization_indices': [0, 1]},
                  'dynamic_candidates': {'original_count': 2, 'round_end_indices': [4, 6], 'origins': [{}]*6}}
        cases = [{'task_index': 0, 'arm': 'source', 'initial': snap(initial), 'rounds': rounds, 'rows': []}]
        rows = [{'task_index': 0, 'arm': 'source', 'policy': policy, 'action': {'primary': primary},
                 'observed_indices': observed, 'verification': {'certified': True},
                 'scores': {'actual_safe_reward': float(1-truth[primary, 0]/1.5),
                            'conditional_safe_reward': float(.95*(1-truth[primary, 0]/1.5)),
                            'true_feasible': True}}
                for policy, primary, observed in [('stop', 5, [0, 1]), ('online', 4, [0, 1, 2, 4])]]
        protocol = {'objective_reference': 1.5, 'chance_alpha': .05, 'familywise_delta': .05}
        return frozen, cases, {0: (truth, np.ones(6, dtype=bool))}, rows, protocol

    def test_new_point_masks_and_pre_release_prediction_are_distinct(self):
        result = dynamic_diagnostics(*self.fixture())
        first, second = result['round_prediction_changes']
        self.assertEqual(first['indices'], [3, 4, 5])
        self.assertEqual(second['indices'], [3, 5])
        self.assertEqual(first['prediction_scope'], 'all_generated_unsampled_including_unreleased')
        self.assertEqual(first['released_new_prediction_change']['indices'], [3])
        self.assertEqual(second['released_new_prediction_change']['indices'], [3, 5])
        np.testing.assert_allclose(first['released_new_prediction_change']['mse_change'], -.75)
        np.testing.assert_allclose(first['mse_before'], 1.)
        np.testing.assert_allclose(first['mse_after'], .25)
        np.testing.assert_allclose(second['mse_change'], -.1875)
        final = result['initial_to_final_prediction_changes'][0]
        self.assertEqual(final['indices'], [3, 5])
        np.testing.assert_allclose(final['mse_change'], -.9375)
        unreleased = result['round2_unreleased_prediction_changes'][0]
        self.assertEqual(unreleased['indices'], [4, 5])
        np.testing.assert_allclose(unreleased['mse_change'], -.75)

    def test_frozen_recommendations_and_intervention_rewards_use_global_indices(self):
        result = dynamic_diagnostics(*self.fixture())
        stop, online = result['final_recommendations']
        self.assertTrue(stop['is_generated_unobserved'])
        self.assertTrue(online['is_generated'])
        self.assertFalse(online['is_unobserved'])
        self.assertTrue(online['true_feasible'])
        self.assertAlmostEqual(online['actual_safe_reward'], .6)
        self.assertAlmostEqual(online['conditional_safe_reward'], .57)
        first, second = result['newonly_terminal_interventions']
        self.assertTrue(first['full_primary_is_generated_unobserved'])
        self.assertFalse(second['full_primary_is_generated_unobserved'])
        self.assertLess(first['conditional_safe_reward_difference'], 0.)
        self.assertEqual(result['counts']['generated_per_round'], [2, 2])
        self.assertEqual(result['counts']['final_recommendations'], 2)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
