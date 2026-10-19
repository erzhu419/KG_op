import copy
import itertools
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog
from scipy.stats import norm

from core.shared_certificate_pair import (anchor_branch_quality_upper,certificate_pair_states,
    guard_statistics_from_pair_states,primary_incumbent_quality_upper)
from core.guarded_deployment_kg import conditional_safety_power,_sd_quadrature,_tolerance_factor
from core.raw_likelihood_lookahead import risk_branch_statistics
from representation.joint_function_risk import JointFunctionRiskPosterior


class SharedCertificatePairTest(unittest.TestCase):
    def test_pair_states_reconstruct_correlated_guard_branches_and_cost(self):
        events=np.array(list(itertools.product((0,1),repeat=3)),dtype=float)
        p,i,b=events.T;budgets=[80,129,131];protocol=dict(comparison_budget=8)
        rng=np.random.default_rng(20261015)
        for _ in range(30):
            law=rng.dirichlet(np.ones(8))
            pair=np.array([law[2*p+i==k].sum() for k in range(4)])
            actual,cost=risk_branch_statistics(events[None],law[None],1,budgets,protocol)
            reconstructed,reconstructed_cost=guard_statistics_from_pair_states(pair,budgets,protocol)
            np.testing.assert_allclose(reconstructed[[0,1,3]],actual[0,[0,1,3]],atol=1e-14,rtol=0.)
            self.assertEqual(reconstructed[2],0.)
            self.assertGreaterEqual(actual[0,2],0.)
            self.assertLessEqual(actual[0,2],b@law+1e-14)
            np.testing.assert_allclose(reconstructed_cost,cost[0],atol=1e-12,rtol=0.)

    def test_pair_states_reconstruction_broadcasts_profiles_and_budgets(self):
        events=np.array(list(itertools.product((0,1),repeat=3)),dtype=float)
        p,i,_=events.T;protocol=dict(comparison_budget=8)
        budgets=np.array([[80,129,131],[129,80,131],[129,129,82]])
        laws=np.random.default_rng(20261016).dirichlet(np.ones(8),size=2)
        pairs=np.array([[law[2*p+i==k].sum() for k in range(4)] for law in laws])
        statistics,costs=guard_statistics_from_pair_states(pairs[:,None],budgets,protocol)
        self.assertEqual(statistics.shape,(2,3,4));self.assertEqual(costs.shape,(2,3))
        for profile,law in enumerate(laws):
            for allocation,budget in enumerate(budgets):
                actual,cost=risk_branch_statistics(events[None],law[None],1,budget,protocol)
                np.testing.assert_allclose(statistics[profile,allocation,[0,1,3]],actual[0,[0,1,3]],
                    atol=1e-14,rtol=0.)
                self.assertEqual(statistics[profile,allocation,2],0.)
                np.testing.assert_allclose(costs[profile,allocation],cost[0],atol=1e-12,rtol=0.)
        shared_statistics,shared_costs=guard_statistics_from_pair_states(pairs[0],budgets,protocol)
        np.testing.assert_array_equal(shared_statistics,statistics[0])
        np.testing.assert_array_equal(shared_costs,costs[0])

    def test_primary_incumbent_states_contain_every_correlated_three_event_law(self):
        states=np.array(list(itertools.product((0,1),repeat=3)));p,i,b=states.T
        rng=np.random.default_rng(719)
        for _ in range(150):
            law=rng.dirichlet(np.ones(8));q=rng.uniform(size=4)
            pair=np.array([law[2*p+i==k].sum() for k in range(4)])
            actual=law@(q[0]*p*(1-i)+q[1]*(1-p)*i+q[3]*p*i+q[2]*(1-p)*(1-i)*b)
            bound=primary_incumbent_quality_upper(q,pair+1e-4,b@law+1e-4)
            self.assertGreaterEqual(bound,actual-1e-13)

    def test_anchor_bound_matches_eight_state_primal_and_contains_correlated_laws(self):
        states=np.array(list(itertools.product((0,1),repeat=3)));p,i,b=states.T
        Aeq=np.array([(2*i+b==k).astype(float) for k in range(4)])
        rng=np.random.default_rng(20261014)
        for _ in range(150):
            law=rng.dirichlet(np.ones(8));pi=Aeq@law;cap=min(1.,p@law+.1);q=rng.uniform(size=4)
            rewards=q[0]*p*(1-i)+q[1]*(1-p)*i+q[3]*p*i+q[2]*(1-p)*(1-i)*b
            lp=linprog(-rewards,A_ub=p[None],b_ub=[cap],A_eq=Aeq,b_eq=pi,bounds=(0,None),method='highs')
            self.assertTrue(lp.success)
            bound=anchor_branch_quality_upper(q,cap,pi)
            self.assertAlmostEqual(bound,-lp.fun,places=11)
            self.assertGreaterEqual(anchor_branch_quality_upper(q,cap,pi+1e-4),law@rewards-1e-13)

    def test_pair_states_use_original_guard_delta_and_correlated_gaussian_mean_integration(self):
        means=np.array([[-1.5,0.],[-1.6,0.],[-1.55,0.]])
        covariance=np.diag([.01,.1,.002,1e-12,.001,1e-12]);covariance[2,4]=covariance[4,2]=.0008
        field=JointFunctionRiskPosterior.from_moments(means,covariance.reshape(3,2,3,2))
        protocol=dict(chance_alpha=.1,familywise_delta=.05)
        actual,d=certificate_pair_states(field,copy.deepcopy(field),[1,2],[[80,131]],-1.5,0.,protocol,
            query=0,raw=False,order=16,power=10,seed=20261014)
        sd,w=_sd_quadrature(80);factor=_tolerance_factor(80,.1,.05/3)
        def integrand(z):
            gb=-1.55+np.sqrt(.001)*z;gi=-1.6+.0008/np.sqrt(.001)*z;gv=.002-.0008**2/.001
            pi=norm.cdf(-(gi+factor*sd)/np.sqrt(gv+1/80)) @ w
            pb=conditional_safety_power(gb,1.,131,.1,.05/3)
            return norm.pdf(z)*np.array([(1-pi)*(1-pb),(1-pi)*pb,pi*(1-pb),pi*pb])
        expected=np.array([quad(lambda z:integrand(z)[j],-10,10,epsabs=1e-11)[0] for j in range(4)])
        np.testing.assert_allclose(actual[0],expected,atol=5e-5,rtol=0.)
        alternate,_=certificate_pair_states(field,copy.deepcopy(field),[1,2],[[80,131]],-1.5,0.,protocol,
            query=0,raw=False,order=16,power=10,seed=20261014,integrated_position=1)
        np.testing.assert_allclose(alternate[0],expected,atol=5e-5,rtol=0.)
        self.assertAlmostEqual(actual.sum(),1.,places=12)
        self.assertTrue(d['all_positive_components_retained']);self.assertEqual(d['verifier_sd_order'],16)


if __name__=='__main__':unittest.main()
