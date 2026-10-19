import unittest

import numpy as np
from scipy.optimize import linprog

from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from core.marginal_certificate_bound import branch_quality_upper
from core.raw_likelihood_lookahead import risk_branch_statistics


class CoupledBranchBoundTest(unittest.TestCase):
    def test_primary_and_incumbent_mass_cannot_be_spent_twice(self):
        q=np.ones(4);caps=np.array([.2,.2,0.])
        self.assertAlmostEqual(branch_quality_upper(q,caps,1),.6)
        self.assertAlmostEqual(coupled_branch_quality_upper(q,caps,1),.4)

    def test_vectorized_dual_matches_independent_primal_lp_for_both_incumbents(self):
        rng=np.random.default_rng(9013)
        q=rng.uniform(size=(100,4));caps=rng.uniform(size=(100,3))
        for incumbent in (1,2):
            upper=coupled_branch_quality_upper(q,caps,incumbent)
            A=np.zeros((4,4));A[0,[0,3]]=1;A[1,[incumbent,3]]=1
            A[2,3-incumbent]=1;A[3,:]=1
            actual=[]
            for quality,marginals in zip(q,caps):
                result=linprog(-quality,A_ub=A,b_ub=[marginals[0],marginals[incumbent],marginals[3-incumbent],1.],
                    bounds=(0.,None),method='highs')
                self.assertTrue(result.success)
                actual.append(-result.fun)
            np.testing.assert_allclose(upper,actual,atol=1e-12,rtol=0.)
            self.assertTrue(np.all(upper<=branch_quality_upper(q,caps,incumbent)+1e-13))

    def test_contains_arbitrarily_correlated_joint_verifier_events(self):
        rng=np.random.default_rng(3107)
        events=np.array([[a,b,c] for a in (0.,1.) for b in (0.,1.) for c in (0.,1.)])
        for incumbent in (1,2):
            for _ in range(40):
                masses=rng.dirichlet(np.ones(8));q=rng.uniform(size=4)
                branches,_=risk_branch_statistics(events[None],masses[None],incumbent,[81,130,132],dict(comparison_budget=8))
                marginals=masses@events
                caps=marginals+rng.uniform(size=3)*(1-marginals)
                upper=coupled_branch_quality_upper(q,caps,incumbent)
                self.assertGreaterEqual(upper,float(q@branches[0])-1e-14)

    def test_saturated_caps_and_primary_incumbent_preserve_existing_cases(self):
        q=np.array([.9,.4,.3,.7])
        self.assertAlmostEqual(coupled_branch_quality_upper(q,np.ones(3),1),.9)
        self.assertEqual(coupled_branch_quality_upper(q,[.2,.3,.4],0),branch_quality_upper(q,[.2,.3,.4],0))


if __name__=='__main__':
    unittest.main()
