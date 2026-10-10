import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import digamma,ndtri,polygamma
from scipy.stats import qmc

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.raw_likelihood_lookahead import (condition_raw,likelihood_query_lookahead,
    predictive_batch_statistics,preposterior_information,query_first_draws,_posterior_weights)
from representation.joint_function_risk import local_joint_moments
from test_guarded_deployment_kg import PROTOCOL
from test_joint_response_mixture import mixture


class ReplicatedMixtureTests(unittest.TestCase):
    def test_one_observation_batch_is_the_existing_raw_update(self):
        a=mixture(); b=copy.deepcopy(a)
        a.update_raw(1,[.55,-.45]); b.update_replicates(1,[[.55,-.45]])
        np.testing.assert_array_equal(a.weights,b.weights)
        for x,y in zip(a.predict_joint(),b.predict_joint()): np.testing.assert_array_equal(x,y)
        self.assertEqual(b.paid_episodes,1)

    def test_known_noise_batch_matches_sequential_bayes_and_counts_only_new_calls(self):
        a=mixture(); a.update_raw(1,[.55,-.45]); b=copy.deepcopy(a)
        values=np.array([[.5,-.42],[.6,-.38],[.53,-.49]])
        a.update_replicates(1,values)
        for value in values: b.update_raw(1,value)
        np.testing.assert_allclose(a.weights,b.weights,atol=2e-6)
        for x,y in zip(a.predict_joint(),b.predict_joint()): np.testing.assert_allclose(x,y,atol=2e-6)
        self.assertEqual(a.paid_episodes,4)
        self.assertTrue(all(f.paid_episodes==g.paid_episodes==4 for f,g in a.components))
        self.assertEqual(a.history[-1]['replicates'],3)

    def test_batch_predictive_statistics_have_correct_mean_variance_and_eta_noise(self):
        fm=np.array([[.2,-2.]]); gm=np.array([[-.3,-3.]])
        fc=np.array([[.1,.03],[.03,.9]]).reshape(1,2,1,2)
        gc=np.array([[.08,-.04],[-.04,1.]]).reshape(1,2,1,2)
        moments=[((fm,fc),(gm,gc))]
        for count in (2,3):
            values,eta=predictive_batch_statistics(moments,0,count,15,80641)
            np.testing.assert_allclose(values.mean(axis=0),[.2,-.3],atol=.001)
            np.testing.assert_allclose(values.var(axis=0),[.1+np.exp(-2+.9/2)/count,.08+np.exp(-3+.5)/count],atol=.002)
            np.testing.assert_allclose(eta.mean(axis=0),[-2,-3],atol=.003)
            np.testing.assert_allclose(eta.var(axis=0),np.array([.9,1.])+polygamma(1,(count-1)/2),atol=.02)

    def test_batch_transport_retains_exact_mean_eta_covariance_and_evidence(self):
        mean=np.array([.2,-4.]); C=np.array([[.12,.06],[.06,1.3]])
        values=np.array([.4,.52,.31]); count=3; k=(count-1)/2
        eta=float(np.log(values.var(ddof=1))-digamma(k)+np.log(k))
        exact=local_joint_moments(mean,C,values.mean(),eta,count)
        z=ndtri(qmc.Sobol(2,scramble=True,seed=80642).random_base2(14))
        m,V,draws,_=query_first_draws(mean[None],C.reshape(1,2,1,2),0,[0],z)
        conditioned,w,score=condition_raw(m,V,draws,np.array([values.mean()]),[exact],
            replicates=count,eta_labels=np.array([eta]))
        posterior_mean=w[0]@conditioned[0]; centered=conditioned[0]-posterior_mean
        np.testing.assert_allclose(posterior_mean,exact['mean'],atol=.001)
        np.testing.assert_allclose(centered.T@(w[0,:,None]*centered),exact['covariance'],atol=.002)
        self.assertAlmostEqual(score[0],exact['log_evidence'],delta=.001)

    def test_full_field_information_matches_paid_filter_after_identity_mixing(self):
        model=mixture()
        for f,g in model.components:
            for field in (f,g): field.covariances[:,1::2,1::2]+=.8
        model.update_raw(1,[.55,-.45]); future=copy.deepcopy(model)
        values=np.array([[.5,-.42],[.6,-.38],[.53,-.49]])
        future.update_replicates(1,values)
        moments=[(f.predict(full_cov=True),g.predict(full_cov=True)) for f,g in model.components]
        means=np.tile(values.mean(axis=0),(2,1)); eta=np.tile(np.log(values.var(axis=0,ddof=1))+np.euler_gamma,(2,1))
        results,_,postw=_posterior_weights(moments,1,means,model.log_weights,replicates=3,eta_labels=eta)
        information,_=preposterior_information(model,moments,1,results,postw,model.weights,1,[0,1,2])
        _,before=model.predict_joint(); _,after=future.predict_joint()
        for i in range(3):
            np.testing.assert_allclose(information[str(i)],1-np.diag(after[i,:,i,:])/np.diag(before[i,:,i,:]),atol=1e-10)

    def test_batch_lookahead_debits_all_episodes_and_keeps_paid_state(self):
        model=mixture(); before=model.predict_joint(); debits=[]
        class Terminal(BudgetedTerminalActions):
            def allocations(self,remaining):
                debits.append(remaining)
                return super().allocations(remaining)
        terminal=Terminal([[0,1,2],[1,0,2],[2,0,1]],[1,0,1],PROTOCOL)
        result=likelihood_query_lookahead(model,terminal,352,1,draw_power=4,fantasy_power=1,seed=80643,replicates=3)
        self.assertEqual(debits,[352,349]); self.assertEqual(result['future_remaining_calls'],349)
        self.assertEqual(model.paid_episodes,0); self.assertEqual(model.history,[])
        for x,y in zip(before,model.predict_joint()): np.testing.assert_array_equal(x,y)
        self.assertAlmostEqual(result['net_reward_of_batch'],result['adaptation_value']+
            result['likelihood_tower_drift']-result['verification_budget_opportunity_cost'])
        self.assertAlmostEqual(result['net_reward_of_batch'],3*result['net_reward_of_paid_episode'])
        self.assertGreaterEqual(result['approximation_choice_loss'],0.)
        self.assertAlmostEqual(result['projection_minus_likelihood_best_value'],
            result['projected_value_bias_at_selected_action']-result['approximation_choice_loss'])
        self.assertAlmostEqual(result['likelihood_net_reward_of_projected_choice'],
            result['likelihood_adaptation_of_projected_choice']+result['likelihood_tower_drift']-
            result['verification_budget_opportunity_cost'])


if __name__=='__main__':
    unittest.main()
