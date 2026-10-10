import copy
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, qmc
from scipy.special import ndtri

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.mixture_guarded_kg import mixture_raw_lookahead
from core.raw_likelihood_lookahead import likelihood_query_lookahead, fixed_action_likelihood_reference
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


def field(means,weights):
    means=np.asarray(means);n=3
    m=np.zeros((len(means),n,2));m[:,:,0]=means[:,None];m[:,:,1]=-4.
    covariance=np.tile(np.eye(2*n)*1e-8,(len(means),1,1))
    result=JointFunctionRiskPosterior.from_moments(m[0],covariance[0].reshape(n,2,n,2))
    result.means=m.reshape(len(means),2*n);result.covariances=covariance
    result.log_weights=np.log(weights);result.component_pairs=[(0,0)]*len(means)
    return result


def terminal():
    return BudgetedTerminalActions([[0,1,2]],[1],{'objective_reference':1.5,'familywise_delta':.05,
        'comparison_budget':8,'comparison_delta':.05,'chance_alpha':.05})


class PositiveSourceMixtureTests(unittest.TestCase):
    def test_scalar_risk_variance_matches_full_joint_before_and_after_paid_update(self):
        from core.mixture_guarded_kg import query_risk_variance
        model=JointResponseMixture(['A','B'],[(field([.1,.7],[.3,.7]),field([-.5,.15],[.8,.2])),
            (field([.8],[1.]),field([-.55,-.2],[.6,.4]))])
        model.log_weights=np.log([.2,.8])
        for h,(_,g) in enumerate(model.components):
            g.means[:,3]=[-2-h,-4-h];g.covariances[:,3,3]=.4
        for values in (None,[.3,-.15]):
            if values is not None:model.update_raw(1,values)
            _,covariance=model.predict_joint()
            self.assertAlmostEqual(query_risk_variance(model,1),covariance[1,3,1,3],places=12)

    def test_control_variate_preserves_joint_adaptation_and_changes_stop_decision(self):
        model=JointResponseMixture(['source'],[(field([.1,.8],[.5,.5]),field([-.5],[1.]))])
        r=mixture_raw_lookahead(model,terminal(),360,[0],draw_power=3,fantasy_power=1,
            seed=202610077,fixed_action_expectations={0:.2})
        row=r['lookahead'][0]
        self.assertAlmostEqual(row['fixed_action_expected_reward'],.2,places=12)
        self.assertAlmostEqual(row['adaptation_value'],row['raw_joint_expected_future_reward']
            -row['raw_joint_baseline_drift']-r['stop_expected_reward']+row['verification_budget_opportunity_cost'],places=12)
        self.assertAlmostEqual(row['net_reward_of_paid_episode'],row['adaptation_value']+.2-r['stop_expected_reward'],places=12)
        self.assertEqual(r['choice'],'stop')
        self.assertEqual(model.paid_episodes,0)

    def test_factorized_future_reference_preserves_shared_identity_and_tower(self):
        from core.positive_mixture_lookahead import fixed_action_factorized_reference
        from core.guarded_deployment_kg import conditional_safety_power
        model=JointResponseMixture(['A','B'],[(field([.1,.7],[.3,.7]),field([-.5,.15],[.8,.2])),
            (field([.8],[1.]),field([-.55],[1.]))])
        action={'kind':'direct','primary':1,'shortlist':[1],'incumbent_position':0,'budgets':[360]}
        protocol=terminal().protocol
        reference=fixed_action_factorized_reference(model,action,0,draw_power=10,terminal_power=9,
            fantasy_power=5,seed=202610077,protocol=protocol)
        power=conditional_safety_power(np.array([-.5,.15,-.55]),np.full(3,np.exp(-2.)),360,.05,.05)
        expected=.5*(.3*(1-.1/1.5)+.7*(1-.7/1.5))*(.8*power[0]+.2*power[1])+.5*(1-.8/1.5)*power[2]
        self.assertAlmostEqual(reference['likelihood_prior_reward'],expected,delta=.002)
        self.assertLess(abs(reference['likelihood_tower_drift']),.003)
        self.assertLess(reference['predictive_log_evidence_max_error'],.002)
        self.assertTrue(reference['shared_identity_retained'])
        self.assertFalse(reference['adaptive_maximum_factorized'])
        self.assertEqual(model.paid_episodes,0)

    def test_two_dimensional_predictive_rule_retains_positive_mixture_cdf(self):
        from core.positive_mixture_lookahead import predictive_values
        model=JointResponseMixture(['source'],[(field([.1,.7],[.3,.7]),field([-.5],[1.]))])
        values,weights=predictive_values(model,0,5,202610077)
        nodes=qmc.Sobol(2,scramble=True,seed=202610077).random_base2(5)
        sd=np.sqrt(1e-8+np.exp(-4.))
        actual=np.column_stack([.3*norm.cdf(values[:,0],.1,sd)+.7*norm.cdf(values[:,0],.7,sd),
            norm.cdf(values[:,1],-.5,sd)])
        np.testing.assert_allclose(actual,nodes,atol=2e-9,rtol=0.)
        np.testing.assert_allclose(weights,np.full(32,1/32),atol=1e-12,rtol=0.)

    def test_conditional_raw_cdf_retains_mean_risk_correlation(self):
        from scipy.integrate import quad
        from core.positive_mixture_lookahead import RawPredictiveCDF
        f=field([.2],[1.]);f.covariances[0,:2,:2]=[[.12,.06],[.06,1.3]]
        rule=RawPredictiveCDF(f,0)
        values=np.linspace(-1.,1.,40).reshape(5,8)
        expected=(norm.cdf((values.reshape(-1,1)-rule.locations)/rule.scales)
            @rule.integration_weights).reshape(values.shape)
        np.testing.assert_allclose(rule(values),expected,atol=1e-15,rtol=0.)
        for y in (-.2,.2,.7):
            expected=quad(lambda z:norm.pdf(z)*norm.cdf((y-.2-.06/np.sqrt(1.3)*z)
                /np.sqrt(.12-.06**2/1.3+np.exp(-4+np.sqrt(1.3)*z))),-12.,12.,epsabs=1e-11)[0]
            self.assertAlmostEqual(float(rule(y)),expected,delta=1e-9)
            density=quad(lambda z:norm.pdf(z)*norm.pdf(y,.2+.06/np.sqrt(1.3)*z,
                np.sqrt(.12-.06**2/1.3+np.exp(-4+np.sqrt(1.3)*z))),-12.,12.,epsabs=1e-11)[0]
            self.assertAlmostEqual(float(rule.log_density([y])[0]),np.log(density),delta=1e-9)

    def test_inverse_retains_historical_labels_and_adaptive_probability_accuracy(self):
        from core.positive_mixture_lookahead import RawPredictiveCDF
        f=field([-.2,.5,1.1],[.15,.6,.25])
        f.covariances[:,:2,:2]=[[.12,.06],[.06,1.3]]
        rule=RawPredictiveCDF(f,0)
        probabilities=np.r_[np.linspace(.00005,.125,16),qmc.Sobol(1,scramble=True,seed=202610078).random_base2(5).ravel()]
        mean=float(rule.weights@rule.means[:,0])
        variance=float(rule.weights@(rule.covariance[:,0,0]+(rule.means[:,0]-mean)**2
            +np.exp(rule.means[:,1]+rule.covariance[:,1,1]/2)))
        radius=np.sqrt(variance)
        while rule(mean-radius)>probabilities.min() or rule(mean+radius)<probabilities.max():radius*=2
        left=np.full(len(probabilities),mean-radius);right=np.full(len(probabilities),mean+radius)
        for _ in range(44):
            middle=(left+right)/2;below=rule(middle)<probabilities
            left=np.where(below,middle,left);right=np.where(below,right,middle)
        values=rule.inverse(probabilities.reshape(6,8))
        np.testing.assert_allclose(values.ravel(),(left+right)/2,atol=2e-12,rtol=0.)
        native,bound=rule.adaptive(values.ravel())
        self.assertLess(float(np.max(abs(native-probabilities)))+bound,1e-8)

    def test_conditional_reference_preserves_exactly_shared_risk(self):
        from core.positive_mixture_lookahead import condition_positive_raw
        f=field([.1,.7],[.3,.7]);f.covariances[:,1::2,1::2]=.2*np.ones((3,3))
        after=copy.deepcopy(f);score=after.update_batch(0,.2,None,1)
        nodes=qmc.Sobol(5,scramble=True,seed=202610077).random_base2(8)
        draws,_,_=condition_positive_raw(f,after,0,[0,1],nodes,.2,score)
        np.testing.assert_allclose(draws[:,1],draws[:,3],atol=1e-7,rtol=0.)

    def test_component_transport_matches_historical_conditional_raw_reference(self):
        from core.positive_mixture_lookahead import condition_positive_raw
        from core.raw_likelihood_lookahead import condition_raw, query_first_draws
        f=field([.2],[1.]);f.covariances[0,:2,:2]=[[.12,.06],[.06,1.3]]
        after=copy.deepcopy(f);score=after.update_batch(0,.8,None,1)
        nodes=qmc.Sobol(5,scramble=True,seed=202610077).random_base2(10)
        actual=condition_positive_raw(f,after,0,[0,1],nodes,.8,score)
        mean=f.means[0].reshape(3,2);cov=f.covariances[0].reshape(3,2,3,2)
        m,C,draws,_=query_first_draws(mean,cov,0,[0,1],ndtri(nodes[:,1:]))
        local={'mean':after.means[0,:2],'covariance':after.covariances[0,:2,:2]}
        expected=condition_raw(m,C,draws,np.array([.8]),[local])
        np.testing.assert_allclose(actual[0],expected[0][0],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(actual[1],expected[1][0],atol=1e-12,rtol=1e-12)
        self.assertAlmostEqual(actual[2],expected[2][0]-score,places=12)

    def test_fixed_action_reference_retains_components_and_known_noise_equivalence(self):
        model=JointResponseMixture(['source'],[(field([.1,.7],[.3,.7]),field([-.5],[1.]))])
        action={'kind':'direct','primary':1,'shortlist':[1],'incumbent_position':0,'budgets':[360]}
        result=fixed_action_likelihood_reference(model,action,0,draw_power=11,fantasy_power=2,
            seed=202610077,protocol=terminal().protocol)
        self.assertTrue(result['positive_source_components_retained'])
        self.assertAlmostEqual(result['prior_expected_reward'],.3*(1-.1/1.5)+.7*(1-.7/1.5),delta=.001)
        self.assertLess(result['predictive_log_evidence_max_error'],.002)
        self.assertAlmostEqual(result['projection_minus_likelihood_baseline'],0.,delta=.001)
        self.assertEqual(model.paid_episodes,0)

    def test_actual_mixed_source_fantasy_keeps_frozen_raw_integral_precision(self):
        from representation.vectorized_joint_function_risk import local_raw_moments
        from representation.joint_function_risk import local_joint_moments
        data=json.loads((Path(__file__).parent/'fixtures'/'historical_mixture_raw_fantasy.json').read_text())
        means,cov=np.array(data['means']),np.array(data['covariances']);label=data['label']
        result=local_raw_moments(means,cov,label)
        self.assertLessEqual(result['integration_relative_error'],1e-8)
        indices={0,len(means)-1,int(np.argmax(abs(means[:,0]-label)))}
        for i in indices:
            scalar=local_joint_moments(means[i],cov[i],label,None,1)
            for key in ('mean','covariance','log_evidence'):
                np.testing.assert_allclose(result[key][i],scalar[key],atol=1e-8,rtol=1e-8)

    def test_mixture_retains_supported_exactly_shared_constant_risk(self):
        f,g=field([.4,.8],[.5,.5]),field([-.5],[1.])
        for current in (f,g):
            current.covariances[:,1::2,1::2]=.2*np.ones((3,3))
        values,_,_=terminal().evaluate_posterior(f,g,360,10,202610077)
        self.assertTrue(np.isfinite(values).all())
        # The mean/eta block is exactly singular for this already-supported
        # constant-risk field; retain its covariance rather than add noise.
        from core.positive_mixture_lookahead import field_draws
        from scipy.stats import qmc
        draws=field_draws(f,[0,1,2],qmc.Sobol(7,scramble=True,seed=202610077).random_base2(10))[0]
        # The inherited PSD eigensolver has sqrt(machine-epsilon) draw
        # roundoff on its nominally zero eigenvalues.
        np.testing.assert_allclose(draws[:,0,1],draws[:,1,1],atol=1e-7,rtol=0.)

    def test_terminal_integrates_clipped_reward_before_mixing(self):
        f,g=field([.1,2.2],[.5,.5]),field([-.5],[1.])
        reward,_,actions=terminal().evaluate_posterior(f,g,360,10,202610077)
        direct=next(i for i,a in enumerate(actions) if a['kind']=='direct' and a['budgets']==[360])
        self.assertAlmostEqual(reward[direct],.5*(1-.1/1.5),delta=1e-4)

    def test_batched_paid_weights_equal_known_noise_component_evidence(self):
        f=field([.1,.7],[.3,.7]);means=f.means[:,0].copy();before=f.log_weights.copy()
        expected=before+norm.logpdf(.2,means,np.sqrt(1e-8+np.exp(-4.)))
        expected-=logsumexp(expected)
        f.update_batch(0,.2,None,1)
        np.testing.assert_allclose(f.log_weights,expected,atol=2e-7,rtol=0.)
        self.assertEqual(f.paid_episodes,1)

    def test_historical_kg_uses_actual_mixture_fantasies_without_paid_mutation(self):
        model=JointResponseMixture(['source'],[(field([.1,.8],[.5,.5]),field([-.5],[1.]))])
        before=copy.deepcopy(model)
        replay=mixture_raw_lookahead(model,terminal(),360,[0],draw_power=3,fantasy_power=1,seed=202610077)
        self.assertTrue(replay['lookahead'][0]['positive_source_mixtures_retained'])
        self.assertTrue(replay['lookahead'][0]['actual_paid_filter_used_for_every_fantasy'])
        self.assertEqual(model.paid_episodes,0)
        np.testing.assert_array_equal(model.components[0][0].means,before.components[0][0].means)
        with self.assertRaisesRegex(ValueError,'Positive source mixtures'):
            likelihood_query_lookahead(model,terminal(),360,0,draw_power=3,fantasy_power=0)


if __name__=='__main__':unittest.main()
