import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from performance.run_legacy_one_step_closed_loop import execute_terminal,paid_update
from core.budgeted_guarded_kg import BudgetedTerminalActions
from test_guarded_deployment_kg import PROTOCOL
from test_joint_response_mixture import mixture


class PaidProblem:
    alpha,tau,simulation_noise_model=.05,0.,'iid_gaussian'
    def __init__(self):
        self.calls=0
    def simulate(self,point,rng):
        self.calls+=1
        return np.array([[.2,-.5],[.7,-.5],[.6,-.5]][point[0]])+.02*rng.normal(size=2)
    def true_objective(self,point):
        raise AssertionError('truth cannot enter update or verifier')
    true_constraint_mean=true_objective
    true_sigma=true_objective
    is_truly_feasible=true_objective


class OneStepTests(unittest.TestCase):
    def test_retention_keeps_incumbents_and_three_policy_budget(self):
        original=np.array([[0,1,2],[1,2,0],[2,1,0]])
        terminal=BudgetedTerminalActions(original.copy(),[1,0,1],PROTOCOL)
        terminal.retain_recommendation(3)
        np.testing.assert_array_equal(terminal.shortlists[:3],original)
        np.testing.assert_array_equal(terminal.shortlists[3:],[[0,1,3],[1,2,3],[2,1,3]])
        np.testing.assert_array_equal(terminal.incumbents,[1,0,1,1,0,1])
        allocations=terminal.allocations(357)
        for kind,table,inc,budgets in allocations:
            if kind=='guarded':
                self.assertEqual(table.shape[1],3)
                self.assertEqual(sum(budgets)+2*PROTOCOL['comparison_budget'],357)
            else:
                np.testing.assert_array_equal(table[:,0],[0,1,2])
        existing=BudgetedTerminalActions(original.copy(),[1,0,1],PROTOCOL)
        existing.retain_recommendation(0)
        np.testing.assert_array_equal(existing.shortlists,original)

    def test_two_heads_charge_one_episode_and_preserve_initial_model(self):
        model=mixture(); problem=PaidProblem(); candidates=[(0,),(1,),(2,)]
        updated,value,evidence=paid_update(model,problem,candidates,1,80621,1398031183)
        self.assertEqual(problem.calls,1)
        self.assertEqual(updated.paid_episodes,1)
        self.assertEqual(model.paid_episodes,0); self.assertEqual(model.history,[])
        self.assertEqual(evidence['candidate'],1)
        np.testing.assert_array_equal(value,problem.simulate((1,),np.random.default_rng(np.random.SeedSequence([80621,1398031183]))))

    def test_verifier_executes_guard_comparison_and_direct_debits(self):
        candidates=[(0,),(1,),(2,)]; problem=PaidProblem()
        guarded={'kind':'guarded','primary':0,'shortlist':[0,1,2],'incumbent_position':1,'budgets':[20,20,10]}
        index,result=execute_terminal(problem,candidates,guarded,5,81621,PROTOCOL,71)
        self.assertTrue(result['certified']); self.assertEqual(index,0)
        self.assertEqual(result['objective_comparison_budget'],16)
        self.assertEqual(problem.calls,56); self.assertEqual(result['total_evaluation_count'],61)
        direct={'kind':'direct','primary':0,'shortlist':[0],'incumbent_position':0,'budgets':[31]}
        before=problem.calls
        _,result=execute_terminal(problem,candidates,direct,4,81621,PROTOCOL,35)
        self.assertEqual(problem.calls-before,31); self.assertEqual(result['total_evaluation_count'],35)
        self.assertEqual(result['objective_comparison_budget'],0)
        before=problem.calls
        with self.assertRaises(ValueError): execute_terminal(problem,candidates,direct,5,81621,PROTOCOL,35)
        self.assertEqual(problem.calls,before)


if __name__=='__main__':
    unittest.main()
