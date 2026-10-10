import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import digamma,logsumexp
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.mixture_guarded_kg import mixture_raw_lookahead
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.source_function_prior import FixedFunctionBasis,SourceFunctionPrior
from test_guarded_deployment_kg import PROTOCOL


def mixture(fg=((.25,-.3),(.75,-.7)),mean_variance=.04,prior_weights=None):
    psi=np.array([[-1.],[0.],[1.]])
    basis=FixedFunctionBasis(psi)
    def prior(mean,variance):
        return SourceFunctionPrior(basis,np.r_[mean,np.zeros(basis.feature_dim)],
            np.diag(np.r_[variance,np.full(basis.feature_dim,1e-8)]),1e-8,{})
    def field(mean):
        return JointFunctionRiskPosterior([prior(mean,mean_variance)],[prior(-4.,1e-8)],psi)
    return JointResponseMixture(["first","second"],[(field(f),field(g)) for f,g in fg],prior_weights)


class JointResponseMixtureTests(unittest.TestCase):
    def test_forecast_history_is_frozen_before_later_paid_updates(self):
        from performance.check_legacy_public_mean_risk_uncertainty import forecast
        model=mixture()
        row=forecast({'task_id':'snapshot','task_index':0,'kind':'source'},'test','test',model)
        model.update_raw(1,[.2,-.4])
        self.assertEqual(row['paid_history'],[])
        self.assertEqual(row['paid_predictive_log_score'],0.)
        self.assertEqual(len(model.history),1)

    def test_batch_joint_density_retains_identity_and_matches_known_noise(self):
        model=mixture(((0.,1.),(1.,0.)),mean_variance=.001,prior_weights=[.3,.7])
        v=np.array([[field.predict()[1][1,0] for field in pair] for pair in model.components])
        component=norm.logpdf([0.,0.],np.array([[0.,1.],[1.,0.]]),np.sqrt(v+np.exp(-4.)))
        expected=logsumexp(model.log_weights+component.sum(axis=1))
        result=model.batch_log_scores(1,[0.,0.],[None,None],1)
        self.assertAlmostEqual(result['joint'],expected,delta=2e-5)
        self.assertLess(result['joint'],sum(result['heads'])-10.)
        self.assertEqual(model.paid_episodes,0)
        self.assertEqual(model.history,[])

    def test_fresh_replicated_batch_score_equals_prequential_evidence(self):
        model=mixture(); model.update_raw(1,[.1,-.4])
        before=model.predict_joint()
        values=np.array([[.2,-.1],[.1,-.5],[.3,-.4]])
        k=(len(values)-1)/2
        eta=np.log(values.var(axis=0,ddof=1))-digamma(k)+np.log(k)
        score=model.batch_log_scores(1,values.mean(axis=0),eta,len(values))
        np.testing.assert_array_equal(model.predict_joint()[0],before[0])
        np.testing.assert_array_equal(model.predict_joint()[1],before[1])
        record=model.update_replicates(1,values)
        self.assertAlmostEqual(score['joint'],record['mixture_log_score'],places=10)

    def test_prequential_joint_weights_match_known_noise_gaussian_filter(self):
        model=mixture()
        self.check_prequential_joint_weights(model,[.5,.5])

    def test_source_weights_survive_initialization_and_paid_gaussian_updates(self):
        model=mixture(prior_weights=[8.,2.])
        self.assertEqual(model.paid_episodes,0)
        self.assertEqual(model.history,[])
        np.testing.assert_allclose(model.weights,[.8,.2],atol=1e-15)
        self.check_prequential_joint_weights(model,[.8,.2])

    def check_prequential_joint_weights(self,model,prior_weights):
        means=np.array([[.25,-.3],[.75,-.7]])
        v=np.array([[m.predict(full_cov=True)[1][1,0,1,0] for m in pair] for pair in model.components])
        logw=np.log(prior_weights); sigma2=np.exp(-4.)
        for values in ([.55,-.45],[.5,-.25]):
            y=np.asarray(values)
            scores=norm.logpdf(y,means,np.sqrt(v+sigma2)).sum(axis=1)
            expected_score=float(logsumexp(logw+scores))
            logw+=scores; logw-=logsumexp(logw)
            means+=v/(v+sigma2)*(y-means); v*=sigma2/(v+sigma2)
            record=model.update_raw(1,y)
            np.testing.assert_allclose(model.weights,np.exp(logw),atol=2e-6)
            self.assertAlmostEqual(record["mixture_log_score"],expected_score,delta=2e-6)
        self.assertEqual(model.paid_episodes,2)
        self.assertTrue(all(f.paid_episodes==g.paid_episodes==2 for f,g in model.components))

    def test_shared_identity_preserves_cross_head_covariance_and_terminal_reward(self):
        model=mixture(((.1,.2),(1.2,-.5)),mean_variance=1e-8)
        mean,C=model.predict_joint()
        self.assertAlmostEqual(C[1,0,1,2],-.1925,places=7)
        terminal=BudgetedTerminalActions(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        values,_,actions=model.terminal_values(terminal,352,8,105230)
        i=next(i for i,a in enumerate(actions) if a["kind"]=="direct" and a["primary"]==1 and a["budgets"]==[128])
        # The good-objective model is unsafe; only the .2-quality model deploys.
        self.assertAlmostEqual(values[i],.5*(1-1.2/1.5),delta=.0001)

    def test_terminal_mixture_uses_updated_identity_weights(self):
        model=mixture(((.1,.2),(1.2,-.5)),mean_variance=1e-8)
        model.update_raw(1,[1.18,-.49])
        self.assertGreater(model.weights[1],.999)
        terminal=BudgetedTerminalActions(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        r,c,a=model.terminal_components(terminal,352,7,105231)
        mr,mc,ma=model.terminal_values(terminal,352,7,105231)
        direct=next(i for i,x in enumerate(a) if x["kind"]=="direct" and x["primary"]==1 and x["budgets"]==[128])
        self.assertGreater(mr[direct],.19)
        self.assertEqual(a,ma)

    def test_stratified_fantasies_update_real_filters_without_paid_mutation(self):
        model=mixture()
        before=model.predict_joint()
        class Recorder(BudgetedTerminalActions):
            def evaluate_posterior(self,f,g,remaining,power,seed):
                budgets.append(remaining)
                return super().evaluate_posterior(f,g,remaining,power,seed)
        budgets=[]
        terminal=Recorder(np.array([[0,1,2],[1,0,2],[2,0,1]]),np.array([1,0,1]),PROTOCOL)
        result=mixture_raw_lookahead(model,terminal,352,[1],draw_power=3,fantasy_power=1,seed=105232)
        self.assertEqual(budgets[:2],[352,352])
        self.assertTrue(all(x==351 for x in budgets[2:]))
        self.assertEqual(result["fantasy_count"],4)
        self.assertEqual(model.paid_episodes,0)
        self.assertEqual(model.history,[])
        np.testing.assert_array_equal(model.predict_joint()[0],before[0])
        np.testing.assert_array_equal(model.predict_joint()[1],before[1])


if __name__=="__main__":
    unittest.main()
