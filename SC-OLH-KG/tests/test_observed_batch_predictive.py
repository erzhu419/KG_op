import copy
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.special import polygamma
from scipy.stats import qmc

from core.observed_batch_predictive import ObservedBatchPredictive, predictive_observed_batches
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.vectorized_joint_function_risk import update_raw_observation


class ObservedBatchPredictiveTests(unittest.TestCase):
    def field(self):
        field=JointFunctionRiskPosterior.from_moments(np.array([[-.1,-3.]]),
            np.array([[.08,.04],[.04,.4]]).reshape(1,2,1,2))
        field.means=np.array([[-.1,-3.],[.2,-2.4]])
        field.covariances=np.array([[[.08,.04],[.04,.4]],[[.12,-.02],[-.02,.6]]])
        field.log_weights=np.log([.35,.65]);return field

    def test_inverse_matches_independent_adaptive_joint_cdf(self):
        field=self.field();rule=ObservedBatchPredictive(field,0,4,128)
        probabilities=qmc.Sobol(2,scramble=True,seed=713).random_base2(5)
        values,eta=rule.inverse(probabilities)
        vcdf,conditional,error=rule.adaptive_coordinates(values,eta)
        np.testing.assert_allclose(vcdf,probabilities[:,1],atol=1e-8,rtol=0.)
        np.testing.assert_allclose(conditional,probabilities[:,0],atol=1e-8,rtol=0.)
        self.assertLess(error,1e-8);self.assertEqual(field.paid_episodes,0)

    def test_predictive_joint_density_matches_the_paid_batch_likelihood(self):
        field=self.field();rule=ObservedBatchPredictive(field,0,4,128)
        values=np.array([-.2,.1,.45]);eta=np.array([-3.5,-2.5,-1.5])
        logs=rule.log_density(values,eta)
        for i in range(3):
            score=update_raw_observation(copy.deepcopy(field),0,values[i],eta[i],4)
            self.assertAlmostEqual(logs[i],score,places=10)

    def test_tensor_rule_keeps_repeated_variance_coordinates_in_order(self):
        field=self.field();rule=ObservedBatchPredictive(field,0,4,128)
        probabilities=np.array([(m,v) for m in (.03,.4,.95) for v in (.8,.15,.55,.15)])
        values,eta=rule.inverse(probabilities)
        vcdf,mcdf,bound=rule.adaptive_coordinates(values,eta)
        np.testing.assert_allclose(vcdf,probabilities[:,1],atol=1e-8,rtol=0.)
        np.testing.assert_allclose(mcdf,probabilities[:,0],atol=1e-8,rtol=0.)
        self.assertLess(bound,1e-8)
        for y,t,score in zip(values,eta,rule.log_density(values,eta)):
            paid=update_raw_observation(copy.deepcopy(field),0,y,t,4)
            self.assertAlmostEqual(paid,score,places=10)

    def test_observed_coordinates_retain_batch_mean_variance_dependence(self):
        field=self.field();rule=ObservedBatchPredictive(field,0,4,128)
        probabilities=qmc.Sobol(2,scramble=True,seed=80647).random_base2(10)
        values,eta=rule.inverse(probabilities);w=np.exp(field.log_weights)
        mean=w@field.means;covariance=w@field.covariances[:,0,1]+w@((field.means[:,0]-mean[0])*(field.means[:,1]-mean[1]))
        variance=w@(field.covariances[:,1,1]+(field.means[:,1]-mean[1])**2)+polygamma(1,1.5)
        np.testing.assert_allclose([values.mean(),eta.mean()],mean,atol=.004,rtol=0.)
        self.assertAlmostEqual(float(np.mean((values-mean[0])*(eta-mean[1]))),covariance,delta=.006)
        self.assertAlmostEqual(float(np.mean((eta-mean[1])**2)),variance,delta=.015)

    def test_full_bank_entry_keeps_complete_identity_and_head_coordinates(self):
        pairs=[[self.field(),self.field()] for _ in range(2)]
        model=SimpleNamespace(components=pairs,weights=np.array([.7,.3]))
        values,eta,weights=predictive_observed_batches(model,0,4,4,713)
        self.assertEqual(values.shape,(32,2));self.assertEqual(eta.shape,(32,2))
        np.testing.assert_allclose([weights[:16].sum(),weights[16:].sum()],model.weights,atol=1e-14,rtol=0.)
        nodes=qmc.Sobol(4,scramble=True,seed=713).random_base2(4)
        for origin,pair in enumerate(pairs):
            for head,field in enumerate(pair):
                rule=ObservedBatchPredictive(field,0,4,128);part=slice(16*origin,16*(origin+1))
                v,m,bound=rule.adaptive_coordinates(values[part,head],eta[part,head])
                np.testing.assert_allclose(v,nodes[:,2*head+1],atol=1e-8,rtol=0.)
                np.testing.assert_allclose(m,nodes[:,2*head],atol=1e-8,rtol=0.)
                self.assertEqual(field.paid_episodes,0)


if __name__=='__main__':unittest.main()
