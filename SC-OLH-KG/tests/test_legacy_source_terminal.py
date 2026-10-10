import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.diagnose_legacy_source_terminal import action_attributions,fixed_action_value,freeze_roster
from test_guarded_deployment_kg import PROTOCOL
from test_joint_response_mixture import mixture


class SourceTerminalTests(unittest.TestCase):
    def test_same_action_attribution_matches_complete_identity_terminal_value(self):
        model=mixture(); model.update_raw(1,[.55,-.45]); before=model.predict_joint()
        terminal=BudgetedTerminalActions([[0,1,2],[1,0,2],[2,0,1]],[1,0,1],PROTOCOL)
        reward,cost,actions=model.terminal_values(terminal,352,7,80701)
        truth=np.tile([.4,-4.,-.5,-4.],(3,1))
        for index in (1,len(actions)-1):
            result=action_attributions(model,actions[index],PROTOCOL,truth,7,80701)
            fixed=fixed_action_value(model,actions[index],PROTOCOL,7,80701)
            self.assertEqual(fixed,result['posterior'])
            self.assertAlmostEqual(result['posterior']['reward'],reward[index],places=12)
            self.assertAlmostEqual(result['posterior']['verification_calls'],cost[index],places=12)
        for a,b in zip(before,model.predict_joint()): np.testing.assert_array_equal(a,b)
        self.assertEqual(model.paid_episodes,1)

    def test_scale_substitution_does_not_use_the_scoring_constraint_mean(self):
        model=mixture(((.2,-.5),(.6,-.5)),mean_variance=1e-8)
        action={'shortlist':[1],'incumbent_position':0,'budgets':[128]}
        a=np.tile([.4,-4.,-.5,-4.],(3,1)); b=a.copy(); b[:,2]=.5
        good=action_attributions(model,action,PROTOCOL,a,7,80702)
        bad=action_attributions(model,action,PROTOCOL,b,7,80702)
        self.assertEqual(good['eta_g_truth'],bad['eta_g_truth'])
        self.assertGreater(good['g_mean_truth']['reward'],.5)
        self.assertLess(bad['g_mean_truth']['reward'],1e-8)

    def test_roster_and_retained_policy_use_only_paid_model_predictions(self):
        class Model:
            paid_episodes=4
            def terminal_values(self,terminal,remaining,power,seed):
                self.remaining=remaining
                _,_,actions=terminal.evaluate(np.tile([.4,-4.,-.5,-4.],(1,8,1)),remaining)
                reward=np.array([.8 if a['primary']==7 else .7 if a['primary']==6 else .1 for a in actions])
                return reward,np.array([sum(a['budgets']) for a in actions]),actions
        original=BudgetedTerminalActions([[i,0,1] if i not in (0,1) else [i,1-i,2] for i in range(8)],
            [0,1]+[1]*6,PROTOCOL)
        model=Model(); terminal,retained,roster=freeze_roster(model,original,[0,1,2,3],362,9,80701)
        self.assertEqual(model.remaining,358)
        self.assertEqual(roster,[0,1,2,3,7,6]); self.assertEqual(retained,7)
        self.assertEqual(set(terminal.shortlists[:,0]),set(roster))


if __name__=='__main__':
    unittest.main()
