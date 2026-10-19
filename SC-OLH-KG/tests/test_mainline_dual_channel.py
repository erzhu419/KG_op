from copy import deepcopy
import unittest

import numpy as np

from performance.mainline_dual_channel import DualChannelResponseMixture, FrozenQueryDualChannelResponseMixture
from performance.mainline_moment_policy import moment_state, query_action, terminal_action
from performance.mainline_point_only import PointOnlyResponseMixture
from tests.test_mainline_point_only import initialized_mixture, point_arrays


PROTOCOL = {'objective_reference': 1., 'chance_alpha': .05, 'familywise_delta': .05}


class DualChannelTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        for key in ('mean', 'cov', 'noise'):
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_query_and_terminal_channels_match_independent_native_models(self):
        initialized = initialized_mixture()
        reference_full = deepcopy(initialized)
        reference_point = PointOnlyResponseMixture.from_model(initialized)
        dual = DualChannelResponseMixture.from_model(initialized)
        initial_paid = dual.paid_episodes
        for step, (query, raw) in enumerate([(0, [.82, .06]), (1, [.73, -.11]), (0, [.59, -.08])]):
            self.assert_state_equal(dual.query_state(), moment_state(reference_full))
            full_choice = query_action(moment_state(reference_full), 360-step, PROTOCOL)
            dual_choice = query_action(dual.query_state(), 360-step, PROTOCOL)
            self.assertEqual(dual_choice, full_choice)
            before = dual.moment_state()
            before_weights = dual.weights
            query_before = dual.query_state()
            local_before = [point_arrays(local, 0) for local in dual.terminal_model.local_models]
            expected_full = reference_full.update_raw(query, raw)
            expected_point = reference_point.update_raw(query, raw)
            record = dual.update_raw(query, raw)
            self.assertEqual(record['selection_evidence'], expected_full)
            self.assertEqual(record['terminal_evidence'], expected_point)
            self.assertEqual(record['shared_raw_values'], raw)
            self.assertEqual(dual.paid_episodes, initial_paid+step+1)
            self.assertEqual(dual.terminal_model.paid_episodes, dual.selection_model.paid_episodes)
            self.assertEqual(record['replicates'], 1)
            self.assert_state_equal(dual.query_state(), moment_state(reference_full))
            self.assert_state_equal(dual.moment_state(), reference_point.moment_state())
            np.testing.assert_array_equal(dual.selection_weights, reference_full.weights)
            np.testing.assert_array_equal(dual.weights, reference_point.weights)
            for actual_pair, expected_pair in zip(dual.selection_model.components, reference_full.components):
                for actual, expected in zip(actual_pair, expected_pair):
                    for key in ('means', 'covariances', 'log_weights'):
                        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
            terminal = dual.moment_state()
            for other in set(range(3))-{query}:
                np.testing.assert_array_equal(terminal['mean'][other], before['mean'][other])
                np.testing.assert_array_equal(terminal['cov'][other], before['cov'][other])
                np.testing.assert_array_equal(terminal['noise'][other], before['noise'][other])
                np.testing.assert_array_equal(dual.weights[other], before_weights[other])
                for actual, expected in zip(point_arrays(dual.terminal_model.local_models[other], 0), local_before[other]):
                    np.testing.assert_array_equal(actual, expected)
            self.assertEqual(terminal_action(terminal, 359-step, PROTOCOL),
                             terminal_action(reference_point.moment_state(), 359-step, PROTOCOL))
            if step == 0:
                other = np.asarray([1, 2])
                self.assertGreater(np.max(abs(dual.query_state()['mean'][other]-query_before['mean'][other])), 1e-6)
        self.assertEqual(len(dual.history), len(initialized.history))
        self.assertEqual([r['candidate'] for r in dual.history], [2, 1, 0, 1, 0])

    def test_initialized_marginals_and_both_snapshot_channels_are_independent(self):
        initialized = initialized_mixture()
        dual = DualChannelResponseMixture.from_model(initialized)
        self.assertIs(dual.selection_model, initialized)
        query = dual.query_state(); terminal = dual.moment_state()
        query_saved = {key: value.copy() for key, value in query.items()}
        terminal_saved = {key: value.copy() for key, value in terminal.items()}
        np.testing.assert_allclose(query['mean'], terminal['mean'], atol=1e-14)
        np.testing.assert_allclose(query['noise'], terminal['noise'], atol=1e-14)
        for q in range(3):
            np.testing.assert_allclose(query['cov'][q, :, q, :], terminal['cov'][q, :, q, :], atol=1e-14)
        dual.update_raw(0, [.82, .06])
        self.assert_state_equal(query, query_saved)
        self.assert_state_equal(terminal, terminal_saved)
        query['mean'][0, 0] = 100.
        terminal['noise'][0, 0] = 100.
        self.assertNotEqual(dual.query_state()['mean'][0, 0], 100.)
        self.assertNotEqual(dual.moment_state()['noise'][0, 0], 100.)


class FrozenQueryChannelTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        for key in ('mean', 'cov', 'noise'):
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_two_raw_updates_keep_query_frozen_and_charge_terminal_once(self):
        initialized = initialized_mixture()
        expected_selection = moment_state(initialized)
        expected_weights = initialized.weights.copy()
        reference = PointOnlyResponseMixture.from_model(initialized)
        frozen = FrozenQueryDualChannelResponseMixture.from_model(initialized)
        initial_paid = frozen.paid_episodes
        for step, (query, raw) in enumerate([(0, [.82, .06]), (1, [.73, -.11])]):
            remaining = 362 - frozen.paid_episodes
            self.assertEqual(remaining, 360 - step)
            self.assertEqual(query_action(frozen.query_state(), remaining, PROTOCOL),
                             query_action(expected_selection, remaining, PROTOCOL))
            before, weights = frozen.moment_state(), frozen.weights
            reference.update_raw(query, raw)
            record = frozen.update_raw(query, raw)
            self.assertEqual(frozen.paid_episodes, initial_paid + step + 1)
            self.assertFalse(record['selection_updated'])
            self.assertEqual(record['shared_raw_values'], raw)
            self.assert_state_equal(frozen.query_state(), expected_selection)
            np.testing.assert_array_equal(frozen.selection_weights, expected_weights)
            after = frozen.moment_state()
            self.assert_state_equal(after, reference.moment_state())
            self.assertGreater(np.max(abs(after['mean'][query] - before['mean'][query])), 1e-6)
            for other in set(range(3)) - {query}:
                for key in ('mean', 'cov', 'noise'):
                    np.testing.assert_array_equal(after[key][other], before[key][other])
                np.testing.assert_array_equal(frozen.weights[other], weights[other])
        self.assertEqual(362 - frozen.paid_episodes, 358)
        self.assertEqual([row['candidate'] for row in frozen.history], [2, 1, 0, 1])
        self.assertEqual(initialized.paid_episodes, initial_paid)

    def test_frozen_snapshots_are_independent_and_runner_routes_source_arm(self):
        from performance.run_mainline_two_round73 import FROZEN_QUERY, base_model
        frozen = FrozenQueryDualChannelResponseMixture.from_model(initialized_mixture())
        query, terminal, weights = frozen.query_state(), frozen.moment_state(), frozen.selection_weights
        query_saved = {key: value.copy() for key, value in query.items()}
        terminal_saved = {key: value.copy() for key, value in terminal.items()}
        frozen.update_raw(0, [.82, .06])
        self.assert_state_equal(query, query_saved)
        self.assert_state_equal(terminal, terminal_saved)
        query['mean'][0, 0] = 100.
        query['cov'][0, 0, 1, 0] = 100.
        query['noise'][0, 0] = 100.
        weights[0] = 100.
        terminal['mean'][0, 0] = 100.
        self.assert_state_equal(frozen.query_state(), query_saved)
        self.assertNotEqual(frozen.selection_weights[0], 100.)
        self.assertNotEqual(frozen.moment_state()['mean'][0, 0], 100.)
        self.assertEqual(base_model(FROZEN_QUERY), 'source_structural_mixture')


if __name__ == '__main__':
    unittest.main()
