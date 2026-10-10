import copy
import unittest

import numpy as np

from performance.connect_historical_prior_posterior import snapshot
from performance.prepare_historical_independent_kg import restore_gp_source_prior, check_replayed_snapshot
from performance.check_historical_independent_queries import cohort_status
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


class IndependentPriorTests(unittest.TestCase):
    def setUp(self):
        mean = np.array([[.1, -4.], [.2, -3.9]])
        covariance = np.eye(4).reshape(2, 2, 2, 2)*.03
        fields = [JointFunctionRiskPosterior.from_moments(mean, covariance) for _ in range(2)]
        self.model = JointResponseMixture(['source GP'], [tuple(fields)])
        self.prior = snapshot(self.model)
        self.model.update_raw(0, [.15, .03]); self.model.update_raw(1, [.22, -.01])
        self.row = {'prior': self.prior, 'posterior': snapshot(self.model)}

    def test_new_target_starts_before_development_observations(self):
        restored = restore_gp_source_prior(self.row)
        self.assertEqual(snapshot(restored), self.prior)
        self.assertEqual(restored.paid_episodes, 0)
        self.assertNotEqual(snapshot(restored), self.row['posterior'])
        before = copy.deepcopy(self.row)
        restored.update_raw(1, [.12, -.05])
        self.assertEqual(restored.paid_episodes, 1)
        self.assertEqual(len(restored.history), 1)
        self.assertEqual(self.row, before)

    def test_machine_roundoff_keeps_evidence_fields_and_billing_consistent(self):
        saved = snapshot(self.model); actual = copy.deepcopy(saved)
        actual['history'][1]['mixture_log_score'] = np.nextafter(actual['history'][1]['mixture_log_score'], np.inf)
        actual['history'][0]['head_log_scores'][0][0] = np.nextafter(actual['history'][0]['head_log_scores'][0][0], np.inf)
        actual['components'][0]['g_covariance'][0][0][0][0] = np.nextafter(actual['components'][0]['g_covariance'][0][0][0][0], np.inf)
        check_replayed_snapshot(actual, saved)
        for key in ('head_log_scores', 'components', 'paid_episodes', 'weights'):
            bad = copy.deepcopy(actual)
            if key == 'head_log_scores':
                bad['history'][0][key][0][0] += 1e-10
            elif key == 'components':
                bad[key][0]['f_mean'][0][0] += 1e-10
            elif key == 'paid_episodes':
                bad[key] += 1
            else:
                bad[key][0] += 1e-10
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_replayed_snapshot(bad, saved)

    def test_development_values_do_not_change_the_new_target_prior(self):
        first = restore_gp_source_prior(self.row)
        self.model.update_raw(0, [.31, .11]); self.row['posterior'] = snapshot(self.model)
        second = restore_gp_source_prior(self.row)
        for model in (first, second):
            model.update_raw(0, [.06, -.03]); model.update_raw(1, [.17, .02])
        self.assertEqual(snapshot(first), snapshot(second))


class IndependentCoverageTests(unittest.TestCase):
    def test_partial_or_hold_query_blocks_a_model_policy(self):
        states = [{'name': 'new_task_source', 'task_id': 'new_task', 'layout': '5',
            'arm': 'source_structural_mixture', 'queries': [5, 4]}]
        cases = [dict(name='new_task_source_query5', state_name='new_task_source',
            processing_complete=True, adaptation_diagnostics_passed=True)]
        self.assertFalse(cohort_status(states, cases)['all_model_states_ready'])
        cases.append(dict(name='new_task_source_query4', state_name='new_task_source',
            processing_complete=True, adaptation_diagnostics_passed=False))
        status = cohort_status(states, cases)
        self.assertTrue(status['all_registered_cases_complete'])
        self.assertFalse(status['all_model_states_ready'])
        cases[-1]['adaptation_diagnostics_passed'] = True
        self.assertTrue(cohort_status(states, cases)['all_model_states_ready'])

    def test_development_query_cannot_replace_a_new_task_query(self):
        states = [{'name': 'independent', 'task_id': 'new_task', 'layout': '6',
            'arm': 'source_common_task_rbf', 'queries': [6]}]
        cases = [dict(name='development_query6', state_name='development',
            processing_complete=True, adaptation_diagnostics_passed=True)]
        with self.assertRaises(AssertionError):
            cohort_status(states, cases)


if __name__ == '__main__':
    unittest.main()
