import unittest

from performance.confirm_historical_joint_source_stationarity import joint_confirmation
from performance.confirm_historical_previous_source_laws import round_transition


class JointSourceStationarityTests(unittest.TestCase):
    def historical_transition(self):
        before = {'f_common_task_scales': [.2, .4], 'eta_common_task_scales': [.3, .6]}
        after = {'f_common_task_scales': [.2, .4], 'eta_common_task_scales': [.31, .61],
            'source_Mstep_pair_passed': True, 'evidence_ratio_precision_passed': True}
        response = {'old_common_task_covariance_scales': [.2, .4],
            'common_task_covariance_scales': [.2, .4], 'source_update_accepted': True}
        risk = {'old_common_task_covariance_scales': [.3, .6],
            'common_task_covariance_scales': [.31, .61], 'source_update_accepted': True,
            'source_Mstep_gate_passed': True, 'full_and_split_Q_improvements': [.01, .005, .015]}
        return before, after, response, risk

    def test_historical_predecessor_requires_actual_acceptance_and_split_gates(self):
        before, after, response, risk = self.historical_transition()
        self.assertTrue(round_transition(before, after, response, risk, [.2, .4], [.31, .61]))
        risk['source_update_accepted'] = False
        self.assertFalse(round_transition(before, after, response, risk, [.2, .4], [.31, .61]))
        risk['source_update_accepted'] = True
        risk['full_and_split_Q_improvements'][1] = -.001
        self.assertFalse(round_transition(before, after, response, risk, [.2, .4], [.31, .61]))
        risk['full_and_split_Q_improvements'][1] = .005
        after['evidence_ratio_precision_passed'] = False
        self.assertFalse(round_transition(before, after, response, risk, [.2, .4], [.31, .61]))

    def test_historical_risk_reweighting_rejects_response_law_change(self):
        before, after, response, risk = self.historical_transition()
        before['f_common_task_scales'] = [.19, .4]
        with self.assertRaises(AssertionError):
            round_transition(before, after, response, risk, [.2, .4], [.31, .61])

    def test_same_law_and_unaccepted_proposal_do_not_establish_consecutive_convergence(self):
        risk = {'score_precision_passed': True, 'stationarity_diagnostic_passed': True}
        response = {'single_iteration_stationarity_passed': True}
        self.assertTrue(joint_confirmation([.2, .4], [.199, .399], risk, risk, response, response, True, True))
        self.assertFalse(joint_confirmation([.2, .4], [.2, .4], risk, risk, response, response, True, True))
        self.assertFalse(joint_confirmation([.2, .4], [.199, .399], risk, risk, response, response, True, False))

    def test_risk_only_convergence_and_uncertified_response_are_insufficient(self):
        risk = {'score_precision_passed': True, 'stationarity_diagnostic_passed': True}
        response = {'single_iteration_stationarity_passed': True}
        failed_response = {'single_iteration_stationarity_passed': False}
        self.assertFalse(joint_confirmation([.2, .4], [.199, .399], risk, risk,
            failed_response, response, True, True))
        self.assertFalse(joint_confirmation([.2, .4], [.199, .399], risk, risk, response, response, False, True))
        unprecise = {'score_precision_passed': False, 'stationarity_diagnostic_passed': True}
        self.assertFalse(joint_confirmation([.2, .4], [.199, .399], risk, unprecise,
            response, response, True, True))


if __name__ == '__main__':
    unittest.main()
