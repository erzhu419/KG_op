import unittest
import json

import numpy as np

from performance.diagnose_historical_outer_bridge import competition_envelope, outer_diagnostics


class OuterBridgeTests(unittest.TestCase):
    def test_flat_net_does_not_hide_failed_tower_and_result_is_serializable(self):
        rows = [dict(raw_net_reward_of_projected_choice_control_variate=np.float64(.1),
            raw_fixed_action_tower_residual=np.float64(.001)) for _ in range(4)]
        result = outer_diagnostics(rows,1e-4)
        self.assertFalse(result['finite_outer_diagnostics_passed'])
        rows[-1]['raw_fixed_action_tower_residual'] = np.float64(1e-6)
        self.assertTrue(json.loads(json.dumps(outer_diagnostics(rows,1e-4)))['finite_outer_diagnostics_passed'])
        rows[1]['raw_net_reward_of_projected_choice_control_variate'] = .101
        self.assertFalse(outer_diagnostics(rows,1e-4)['finite_outer_diagnostics_passed'])

    def test_unreferenced_values_cannot_hide_a_bound_overlap(self):
        values = [[.6,.1,.2],[.7,.1,.2]]
        upper = [[.6,.8,.5],[.7,.75,.6]]
        result = competition_envelope(values,upper,[True,False,False],[.02,.03],
            [dict(group=0,action_indices=[0]),dict(group=1,action_indices=[1,2])])
        self.assertAlmostEqual(result['potential_uplift'],.02*.2+.03*.05)
        self.assertAlmostEqual(result['overlap_probability_mass'],.05)
        self.assertEqual(result['overlapping_action_head_pairs'],2)
        self.assertEqual([r['group'] for r in result['groups']],[1])
        # Existing finite-rule agreement is irrelevant to this upper envelope.
        self.assertGreater(result['potential_uplift'],0.)

    def test_covered_actions_are_fixed_and_probability_mass_is_preserved(self):
        result = competition_envelope([[.7,.8,.1]],[[1.,1.,.8001]],
            [True,True,False],[.04],[dict(group=2,action_indices=[2])])
        np.testing.assert_allclose(result['known_best'],[.8])
        self.assertAlmostEqual(result['potential_uplift'],.04*.0001)
        self.assertAlmostEqual(result['overlap_probability_mass'],.04)
        self.assertEqual(result['overlapping_action_head_pairs'],1)


if __name__ == '__main__':
    unittest.main()
