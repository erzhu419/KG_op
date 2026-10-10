import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.run_legacy_batch_closed_loop import paid_batch_update,select_terminal
from test_guarded_deployment_kg import PROTOCOL
from test_joint_response_mixture import mixture
from test_legacy_one_step_closed_loop import PaidProblem


class BatchClosedLoopTests(unittest.TestCase):
    def test_fresh_batch_centers_constraint_and_charges_three_new_episodes(self):
        model=mixture(); model.update_raw(1,[.55,-.45])
        before=model.predict_joint(); problem=PaidProblem(); problem.tau=.15
        candidates=[(0,),(1,),(2,)]
        means=[]; variances=[]
        for pair in model.components:
            prediction=[field.predict(full_cov=True) for field in pair]
            means.append([m[2,0] for m,C in prediction])
            variances.append([C[2,0,2,0]+np.exp(-4.)/3 for m,C in prediction])
        updated,values,evidence=paid_batch_update(model,problem,candidates,2,3,80651,1398031183)
        centered_mean=values.mean(axis=0)-[0,problem.tau]
        logw=model.log_weights+norm.logpdf(centered_mean,means,np.sqrt(variances)).sum(axis=1)
        np.testing.assert_allclose(updated.weights,np.exp(logw-logsumexp(logw)),atol=2e-6)
        self.assertEqual(problem.calls,3); self.assertEqual(updated.paid_episodes,4)
        self.assertEqual(evidence['replicates'],3); self.assertEqual(evidence['candidate'],2)
        self.assertEqual(model.paid_episodes,1); self.assertEqual(len(model.history),1)
        for a,b in zip(before,model.predict_joint()): np.testing.assert_array_equal(a,b)

    def test_new_point_and_first_repeat_share_fresh_noise_coordinates(self):
        problem=PaidProblem(); model=mixture(); candidates=[(0,),(1,),(2,)]
        _,single,_=paid_batch_update(model,problem,candidates,1,1,80652,1398031183)
        _,batch,_=paid_batch_update(model,problem,candidates,2,3,80652,1398031183)
        np.testing.assert_allclose(single[0]-[.7,-.5],batch[0]-[.6,-.5],atol=1e-14)
        self.assertEqual(problem.calls,4)
        self.assertFalse(np.array_equal(batch[0],batch[1]))

    def test_paid_count_is_debited_before_every_terminal_allocation(self):
        model=mixture(); model.update_raw(1,[.55,-.45]); calls=[]
        updated,_,_=paid_batch_update(model,PaidProblem(),[(0,),(1,),(2,)],1,3,80653,1398031183)
        class Terminal(BudgetedTerminalActions):
            def allocations(self,remaining):
                calls.append(remaining)
                return super().allocations(remaining)
        terminal=Terminal([[0,1,2],[1,0,2],[2,0,1]],[1,0,1],PROTOCOL)
        spec={'total_target_call_cap':71,'terminal_draw_power':5,'integration_seed':80621}
        choice=select_terminal(updated,terminal,updated.paid_episodes,spec)
        self.assertTrue(calls); self.assertEqual(set(calls),{67})
        action=choice['action']; reserve=2*PROTOCOL['comparison_budget'] if action['kind']=='guarded' else 0
        self.assertLessEqual(updated.paid_episodes+sum(action['budgets'])+reserve,71)


if __name__=='__main__':
    unittest.main()
