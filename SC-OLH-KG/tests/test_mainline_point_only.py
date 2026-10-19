from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from core.legacy_joint_kg import legacy_joint_utility
from performance.mainline_moment_policy import moment_state, query_action
from performance.mainline_point_only import PointOnlyResponseMixture
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


def initialized_mixture():
    rng = np.random.default_rng(20261015)
    pairs = []
    for identity in range(2):
        fields = []
        for head in range(2):
            means, covariances = [], []
            for component in range(3):
                mean = np.empty((3, 2))
                mean[:, 0] = np.array([.5, .7, .9])-.6*head+.12*identity+.04*component
                mean[:, 1] = np.array([-3., -2.7, -2.4])-.5*head+.1*identity+.08*component
                loading = .07*rng.normal(size=(6, 4))
                covariance = loading @ loading.T+np.diag([.03, .16]*3)
                means.append(mean.reshape(-1)); covariances.append(covariance)
            field = JointFunctionRiskPosterior.from_moments(
                means[0].reshape(3, 2), covariances[0].reshape(3, 2, 3, 2))
            field.means, field.covariances = np.asarray(means), np.asarray(covariances)
            field.log_weights = np.log([.2, .3, .5])
            field.component_pairs = [(0, 0)]*3
            fields.append(field)
        pairs.append(tuple(fields))
    model = JointResponseMixture(['source_a', 'source_b'], pairs, [.4, .6])
    model.update_raw(2, [.95, .16])
    model.update_raw(1, [.65, .02])
    return model


def point_arrays(model, candidate):
    arrays = [model.weights.copy()]
    index = slice(2*candidate, 2*candidate+2)
    for pair in model.components:
        for field in pair:
            arrays.extend([field.means[:, index].copy(), field.covariances[:, index, index].copy(),
                           field.log_weights.copy()])
    return arrays


class PointOnlyTests(unittest.TestCase):
    def assert_local_matches(self, point_only, candidate, native, native_candidate):
        for actual, expected in zip(point_arrays(point_only.local_models[candidate], 0),
                                    point_arrays(native, native_candidate)):
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def test_native_marginals_updates_and_independent_histories(self):
        original = initialized_mixture()
        expected = moment_state(original)
        isolated = PointOnlyResponseMixture.from_model(original)
        state = isolated.moment_state()
        snapshot = {key: value.copy() for key, value in state.items()}
        np.testing.assert_allclose(state['mean'], expected['mean'], atol=1e-14)
        np.testing.assert_allclose(state['noise'], expected['noise'], atol=1e-14)
        for q in range(3):
            np.testing.assert_allclose(state['cov'][q, :, q, :], expected['cov'][q, :, q, :], atol=1e-14)
            self.assert_local_matches(isolated, q, original, q)
            for j in range(3):
                if j != q:
                    np.testing.assert_array_equal(state['cov'][q, :, j, :], np.zeros((4, 4)))
            # Both source identity dependence and native response/eta covariance survive.
            self.assertGreater(abs(state['cov'][q, 0, q, 2]), 1e-8)
            self.assertGreater(abs(state['cov'][q, 0, q, 1]), 1e-8)
            self.assertGreater(abs(state['cov'][q, 2, q, 3]), 1e-8)
            for local_pair, full_pair in zip(isolated.local_models[q].components, original.components):
                for local, full in zip(local_pair, full_pair):
                    self.assertFalse(np.shares_memory(local.means, full.means))
                    self.assertFalse(np.shares_memory(local.covariances, full.covariances))
                    self.assertFalse(np.shares_memory(local.log_weights, full.log_weights))

        reference_zero, reference_one = deepcopy(original), deepcopy(original)
        before_other = [point_arrays(isolated.local_models[q], 0) for q in (1, 2)]
        expected_record = reference_zero.update_raw(0, [.82, .06])
        actual_record = isolated.update_raw(0, [.82, .06])
        self.assertAlmostEqual(actual_record['mixture_log_score'], expected_record['mixture_log_score'], places=12)
        self.assert_local_matches(isolated, 0, reference_zero, 0)
        for q, saved in zip((1, 2), before_other):
            for actual, old in zip(point_arrays(isolated.local_models[q], 0), saved):
                np.testing.assert_array_equal(actual, old)

        first_updated = point_arrays(isolated.local_models[0], 0)
        reference_one.update_raw(1, [.73, -.11])
        isolated.update_raw(1, [.73, -.11])
        self.assert_local_matches(isolated, 1, reference_one, 1)
        for actual, old in zip(point_arrays(isolated.local_models[0], 0), first_updated):
            np.testing.assert_array_equal(actual, old)
        for actual, old in zip(point_arrays(isolated.local_models[2], 0), before_other[1]):
            np.testing.assert_array_equal(actual, old)

        second_updated = point_arrays(isolated.local_models[1], 0)
        reference_zero.update_raw(0, [.59, -.08])
        isolated.update_raw(0, [.59, -.08])
        self.assert_local_matches(isolated, 0, reference_zero, 0)
        for actual, old in zip(point_arrays(isolated.local_models[1], 0), second_updated):
            np.testing.assert_array_equal(actual, old)
        self.assertEqual(isolated.paid_episodes, original.paid_episodes+3)
        self.assertEqual([r['candidate'] for r in isolated.history], [2, 1, 0, 1, 0])
        self.assertEqual([len(p.history) for p in isolated.local_models], [2, 1, 0])
        # Earlier returned snapshots are detached from all subsequent updates.
        for key in state:
            np.testing.assert_array_equal(state[key], snapshot[key])
        self.assertGreater(np.max(np.abs(isolated.weights[0]-isolated.weights[2])), 1e-5)

    def test_query_fantasies_change_only_the_queried_policy(self):
        point_only = PointOnlyResponseMixture.from_model(initialized_mixture())
        state = point_only.moment_state()
        calls = []

        def record(f, fv, risk, covariance, protocol):
            calls.append(tuple(np.asarray(x).copy() for x in (f, fv, risk, covariance)))
            return legacy_joint_utility(f, fv, risk, covariance, protocol)

        protocol = {'objective_reference': 1., 'chance_alpha': .05, 'familywise_delta': .05}
        with patch('core.legacy_joint_kg.legacy_joint_utility', side_effect=record):
            chosen = query_action(state, 358, protocol)
        self.assertEqual(len(chosen['scores73']), 3)
        self.assertTrue(np.isfinite(chosen['scores73']).all())
        self.assertEqual(len(calls), 1+3*9)
        baseline = calls[0]
        for query in range(3):
            other = np.asarray([q for q in range(3) if q != query])
            for fantasy in calls[1+9*query:1+9*(query+1)]:
                for actual, old in zip(fantasy, baseline):
                    np.testing.assert_array_equal(actual[other], old[other])


if __name__ == '__main__':
    unittest.main()
