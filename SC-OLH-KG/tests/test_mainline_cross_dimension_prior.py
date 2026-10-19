"""Check frozen representation replay and genuine new-coordinate prediction."""

import unittest

import numpy as np

from performance.connect_historical_learned_prior_kg import PLAN, read, restore_learned_source_prior
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_cross_dimension_prior import query_coordinates, restore_cross_dimension_prior
from performance.run_legacy_related_inventory_source import problem_for


class CrossDimensionPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reuse = read('performance/manifests/historical_full_bank_development_v1.json')
        cls.pilot = read(read(cls.reuse['independent_execution_plan'])['pilot_plan'])
        old = read(read(PLAN)['legacy_connection_plan'])
        public = read(old['public_inputs'])
        cls.problem = no_outcomes(problem_for(public['plan'], public['tasks'][2]))
        cls.bank = [tuple(p) for p in public['bank']]
        cls.coordinates = query_coordinates('5', cls.problem, cls.bank)

    def test_reconstruction_matches_all_frozen_source_coordinates(self):
        diagnostics = self.coordinates['diagnostics']
        self.assertEqual(diagnostics['candidate_count'], 73)
        self.assertEqual(diagnostics['frozen_encoder_training_rows'], 32)
        self.assertLess(diagnostics['old_mean_reference_max_absolute_error'], 1e-12)
        self.assertLess(diagnostics['old_risk_reference_max_absolute_error'], 1e-8)
        self.assertEqual(diagnostics['new_simulator_calls'], 0)

    def test_explicit_coordinates_reproduce_cache_and_change_both_priors(self):
        indices = [45, 10, 63]
        coordinates = {key: np.asarray(self.coordinates[key])[indices].tolist() for key in ('mean', 'risk')}
        legacy, _, _ = restore_learned_source_prior('5', indices)
        expected_mean, expected_covariance = legacy.predict_joint()
        del legacy
        changed = {key: np.asarray(value).copy() for key, value in coordinates.items()}
        changed['mean'][0, 0] += .1
        changed['risk'][0, 0] += .1
        for arm in ('source_structural_mixture', 'source_common_task_rbf'):
            with self.subTest(arm=arm):
                model, diagnostics = restore_cross_dimension_prior(self.reuse, self.pilot, '5', arm, coordinates)
                mean, covariance = model.predict_joint()
                if arm == 'source_structural_mixture':
                    np.testing.assert_allclose(mean, expected_mean, atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(covariance, expected_covariance, atol=1e-12, rtol=1e-12)
                self.assertEqual(model.paid_episodes, 0)
                self.assertEqual(diagnostics['source_law_optimizations'], 0)
                self.assertEqual(diagnostics['gp_optimizations'], 0)
                del model
                shifted, _ = restore_cross_dimension_prior(self.reuse, self.pilot, '5', arm, changed)
                new_mean, new_covariance = shifted.predict_joint()
                self.assertGreater(max(np.max(abs(new_mean-mean)), np.max(abs(new_covariance-covariance))), 1e-6)
                self.assertFalse(shifted.history)


if __name__ == '__main__':
    unittest.main()
