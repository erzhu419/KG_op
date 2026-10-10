import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.guarded_deployment_kg import deployment_values
from core.raw_likelihood_lookahead import (condition_raw,factorized_terminal_values,
    fixed_action_likelihood_reference,likelihood_query_lookahead,predictive_raw_values,
    project_raw,query_first_draws)
from representation.joint_function_risk import local_joint_moments
from test_guarded_deployment_kg import PROTOCOL
from test_joint_response_mixture import mixture


class RawLikelihoodTests(unittest.TestCase):
    def test_factorization_matches_independent_cross_product_with_guarded_branches(self):
        f=np.array([[[.3,-5],[.5,-4],[.7,-5]],[[.65,-3],[.45,-5],[.7,-4]]])
        g=np.array([[[-.2,-4],[-.3,-5],[-.2,-5]],[[.02,-5],[-.1,-4],[-.3,-5]]])
        fw,gw=np.array([.3,.7]),np.array([.6,.4])
        for inc,budgets in [(1,[80,128,128]),(0,[80,128,128]),(0,[128])]:
            m=len(budgets)
            draws=np.array([np.column_stack([f[i,:m],g[j,:m]]) for i,j in itertools.product(range(2),repeat=2)])
            weights=np.array([fw[i]*gw[j] for i,j in itertools.product(range(2),repeat=2)])
            r,c=factorized_terminal_values(f[None,:,:m],g[None,:,:m],fw[None],gw[None],inc,budgets,PROTOCOL)
            expected=np.array([deployment_values(d[None],np.arange(m)[None],np.array([inc]),
                dict(PROTOCOL,candidate_budgets=budgets)) for d in draws]).reshape(4,2)
            np.testing.assert_allclose([r[0],c[0]],weights@expected,atol=1e-12)

    def test_raw_transport_matches_known_noise_gaussian_conditional(self):
        mean=np.array([.3,-4.,.8,-5.])
        C=np.array([[.2,0.,.1,0.],[0.,1e-8,0.,0.],[.1,0.,.3,0.],[0.,0.,0.,.02]])
        z=ndtri(qmc.Sobol(4,scramble=True,seed=80601).random_base2(14))
        draws=mean+z@np.linalg.cholesky(C).T
        y=.6; noise=np.exp(-4.)
        local=local_joint_moments(mean[:2],C[:2,:2],y,None,1)
        conditioned,w,_=condition_raw(mean,C,draws,np.array([y]),[local])
        m=(w[0,:,None]*conditioned[0]).sum(axis=0)
        V=(conditioned[0]-m).T@(w[0,:,None]*(conditioned[0]-m))
        expected_mean=mean+C[:,0]/(C[0,0]+noise)*(y-mean[0])
        expected_cov=C-np.outer(C[:,0],C[0,:])/(C[0,0]+noise)
        np.testing.assert_allclose(m,expected_mean,atol=1e-4)
        np.testing.assert_allclose(V,expected_cov,atol=1e-4)

    def test_non_gaussian_eta_reference_matches_adaptive_local_integral(self):
        mean=np.array([.2,-4.]); C=np.array([[.12,.06],[.06,1.3]])
        z=ndtri(qmc.Sobol(2,scramble=True,seed=80602).random_base2(15))
        means=mean[None]; cov=C.reshape(1,2,1,2)
        m,V,draws,_=query_first_draws(means,cov,0,[0],z)
        exact=local_joint_moments(mean,C,.8,None,1)
        conditioned,w,score=condition_raw(m,V,draws,np.array([.8]),[exact])
        weighted=(w[0,:,None]*conditioned[0]).sum(axis=0)
        covariance=(conditioned[0]-weighted).T@(w[0,:,None]*(conditioned[0]-weighted))
        np.testing.assert_allclose(weighted,exact['mean'],atol=.0005)
        np.testing.assert_allclose(covariance,exact['covariance'],atol=.0005)
        self.assertAlmostEqual(score[0],exact['log_evidence'],delta=.0005)
        projected=project_raw(m,V,draws,[exact])[0]
        np.testing.assert_allclose(projected.mean(axis=0),exact['mean'],atol=.0005)
        np.testing.assert_allclose(np.cov(projected,rowvar=False),exact['covariance'],atol=.0005)

    def test_posterior_eta_proposal_resolves_observed_extreme_fantasy(self):
        # Actual component/query moments and a saved fantasy from stage13.
        mean=np.array([.31060177692847596,-9.728724135248596])
        C=np.array([[.06996215963457467,1.8066891655622146e-5],
                    [1.8066891655622146e-5,1.8804123752594053]])
        y=3.3036545929706134
        z=ndtri(qmc.Sobol(2,scramble=True,seed=80622).random_base2(10))
        m,V,draws,_=query_first_draws(mean[None],C.reshape(1,2,1,2),0,[0],z)
        exact=local_joint_moments(mean,C,y,None,1)
        conditioned,w,score=condition_raw(m,V,draws,np.array([y]),[exact])
        self.assertAlmostEqual(score[0],exact['log_evidence'],delta=.01)
        self.assertGreater(float(1/(w[0]**2).sum()),500.)
        np.testing.assert_allclose(w[0]@conditioned[0],exact['mean'],atol=.01)

    def test_eta_proposal_keeps_other_policy_conditional_moments(self):
        mean=np.array([.2,-4.,.5,-3.])
        local=np.array([[.12,.06],[.06,1.3]])
        load=np.array([[.4,.05],[-.3,.2]])
        C=np.block([[local,local@load.T],[load@local,load@local@load.T+np.diag([.04,.2])]])
        z=ndtri(qmc.Sobol(4,scramble=True,seed=80624).random_base2(14))
        m,V,draws,_=query_first_draws(mean.reshape(2,2),C.reshape(2,2,2,2),0,[0,1],z)
        exact=local_joint_moments(mean[:2],local,.8,None,1)
        conditioned,w,_=condition_raw(m,V,draws,np.array([.8]),[exact])
        posterior_mean=w[0]@conditioned[0]
        centered=conditioned[0]-posterior_mean
        posterior_covariance=centered.T@(w[0,:,None]*centered)
        gain=np.linalg.solve(local,C[:2]).T
        np.testing.assert_allclose(posterior_mean,mean+gain@(exact['mean']-mean[:2]),atol=.001)
        np.testing.assert_allclose(posterior_covariance,C+gain@(exact['covariance']-local)@gain.T,atol=.002)

    def test_complete_reference_debits_once_and_preserves_paid_state(self):
        model=mixture(); before=model.predict_joint()
        budgets=[]
        class Terminal(BudgetedTerminalActions):
            def allocations(self,remaining):
                budgets.append(remaining)
                return super().allocations(remaining)
        terminal=Terminal(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        result=likelihood_query_lookahead(model,terminal,352,1,draw_power=3,fantasy_power=1,seed=80603)
        self.assertEqual(budgets,[352,351])
        self.assertEqual(result['fantasy_count'],4)
        self.assertEqual(result['future_remaining_calls'],351)
        self.assertEqual(model.paid_episodes,0); self.assertEqual(model.history,[])
        np.testing.assert_array_equal(model.predict_joint()[0],before[0])
        np.testing.assert_array_equal(model.predict_joint()[1],before[1])
        self.assertAlmostEqual(result['net_reward_of_paid_episode'],result['adaptation_value']+
            result['likelihood_tower_drift']-result['verification_budget_opportunity_cost'])

    def test_four_dimensional_predictive_rule_matches_analytic_raw_variances(self):
        fm=np.array([[.2,-2.]]); gm=np.array([[-.3,-3.]])
        fc=np.array([[.1,.03],[.03,.9]]).reshape(1,2,1,2)
        gc=np.array([[.08,-.04],[-.04,1.]]).reshape(1,2,1,2)
        y=predictive_raw_values([((fm,fc),(gm,gc))],0,15,80604)
        np.testing.assert_allclose(y.mean(axis=0),[.2,-.3],atol=.001)
        np.testing.assert_allclose(y.var(axis=0),[.1+np.exp(-2+.9/2),.08+np.exp(-3+1/2)],atol=.002)

    def test_fixed_baseline_retains_shared_identity_and_known_noise_equivalence(self):
        model=mixture(((.1,.2),(1.2,-.5)),mean_variance=1e-8)
        action={'kind':'direct','primary':1,'shortlist':[1],'incumbent_position':0,'budgets':[128]}
        result=fixed_action_likelihood_reference(model,action,1,draw_power=7,fantasy_power=2,
            seed=80605,protocol=PROTOCOL)
        self.assertAlmostEqual(result['prior_expected_reward'],.5*(1-1.2/1.5),delta=.0001)
        self.assertAlmostEqual(result['projection_minus_likelihood_baseline'],0.,delta=.0001)
        self.assertEqual(model.paid_episodes,0)


if __name__=='__main__':
    unittest.main()
