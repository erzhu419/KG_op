import unittest

import numpy as np

from performance.integrate_historical_changed_source import ChangedSourcePosterior,integrate_change,variance_gradient
from performance.learn_historical_raw_source_amplitudes import normal_logs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.resolve_historical_changed_candidate_tails import PrecisionTiltedPosterior
from performance.finalize_historical_changed_candidate import integrate_site_variance
from performance.prepare_legacy_f_variance_reference import variance_reference
from performance.resolve_historical_source_variance import VarianceTiltedPosterior


class ChangedSourceConditioningTests(unittest.TestCase):
    def test_fused_site_integral_matches_full_bridge_integral(self):
        mean=np.array([.2,-.4]);K=np.array([[.5,.15],[.15,.3]])
        risk_mean=np.array([-2.,-1.]);old_risk=np.array([[.8,.2],[.2,.6]])
        new_risk=np.array([[.4,.1],[.1,.5]]);labels=[.6,-.1];s2=[.01,.04]
        old=VarianceTiltedPosterior(mean,K,labels,s2,[0],mean,[.03,.02])
        relative=ChangedSourcePosterior(mean,.7*K,labels,s2,risk_mean,old_risk,new_risk)
        target=VarianceReferencePosterior(mean,.7*K,labels,s2)
        tilted=PrecisionTiltedPosterior(old,relative,old.log_tilt,[0,1],mean,[.03,.02],.8)
        rm,rc=variance_reference(risk_mean,new_risk,s2)
        states=np.array([[-3.,-2.],[-1.,-.2],[-2.,-.4]])
        full=integrate_change(tilted,target,states,rm,rc,[1.,0.],lambda p,e:tilted.log_tilt_at(p,e))
        fused=integrate_site_variance(tilted,target,states,rm,rc,0)
        np.testing.assert_allclose(fused['displacements'],full['displacements'][:,0],atol=1e-10)
        np.testing.assert_allclose(fused['second_moments'],full['covariances'][:,0,0]
            +full['displacements'][:,0]**2,atol=1e-10)
        np.testing.assert_allclose(fused['inverse_weights'],np.exp(full['log_change_weights']),atol=1e-10)

    def test_variance_bridge_restores_new_density_and_bounds_evidence_weights(self):
        mean=np.array([.2,-.4]);K=np.array([[.5,.15],[.15,.3]])
        old_risk=np.array([[.8,.2],[.2,.6]]);new_risk=np.array([[.4,.1],[.1,.5]])
        old=VarianceReferencePosterior(mean,K,[.6,-.1],[.01,.04])
        relative=ChangedSourcePosterior(mean,.7*K,[.6,-.1],[.01,.04],[-2.,-1.],old_risk,new_risk)
        base=VarianceReferencePosterior(mean,.7*K,[.6,-.1],[.01,.04])
        tilted=PrecisionTiltedPosterior(old,relative,lambda e:0.,[0,1],mean,[.03,.02],.8)
        for point in (np.array([-3.,-2.]),np.array([.4,1.]),np.array([-5.,.3])):
            logh=tilted.log_tilt_at(point)
            self.assertAlmostEqual(tilted.evaluate(point)[0]-logh,base.evaluate(point)[0],places=12)
            self.assertLessEqual(np.exp(-logh),1.)
            self.assertLessEqual(np.exp(tilted.reverse_ratio(point)-logh-.8),1.)

    def test_tilted_source_change_keeps_normalizer_and_within_eta_variance(self):
        class Old:
            def evaluate(self,p):return np.log1p(p[0]**2),p,None

        class New:
            mean=np.zeros(1)

            def evaluate(self,p):return .5*np.log(2.)-(p[0]-.4)**2+.5*p[0]**2,p,None

            def moments(self,evaluation):return evaluation[1],np.array([[.2]])

        value=integrate_change(Old(),New(),np.array([[-2.],[1.],[3.]]),np.zeros(1),np.eye(1),[1.],
            lambda point,evaluation:evaluation[0])
        np.testing.assert_allclose(value['eta'],.4,atol=1e-10)
        np.testing.assert_allclose(value['eta_covariances'],.5,atol=1e-10)
        np.testing.assert_allclose(value['displacements'],.4,atol=1e-10)
        np.testing.assert_allclose(value['covariances'],.7,atol=1e-10)
        np.testing.assert_allclose(value['log_change_weights'],np.log(.5),atol=1e-10)
        np.testing.assert_allclose(value['old_inverse_tilt_weights'],.5,atol=1e-10)

    def test_new_risk_prior_ratio_cancels_the_unchanged_reference(self):
        mean=np.array([.2,-.4]);K=np.array([[.5,.15],[.15,.3]])
        old_risk=np.array([[.8,.2],[.2,.6]]);new_risk=np.array([[.4,.1],[.1,.5]])
        risk_mean=np.array([-2.,-1.]);labels=[.6,-.1];s2=[.01,.04]
        old=VarianceReferencePosterior(mean,K,labels,s2)
        new=ChangedSourcePosterior(mean,K,labels,s2,risk_mean,old_risk,new_risk)
        for point in (risk_mean,np.array([-3.,.4]),np.array([1.,-2.])):
            actual=new.evaluate(point)[0]-old.evaluate(point)[0]
            expected=normal_logs(point[None,:],risk_mean,new_risk)[0]-normal_logs(point[None,:],risk_mean,old_risk)[0]
            self.assertAlmostEqual(actual,expected,places=12)

    def test_correlated_gaussian_change_along_a_variance_gradient(self):
        mean=np.array([-.2,.3]);cov=np.array([[1.,.4],[.4,1.5]])
        target_mean=np.array([.5,-.2]);target_cov=np.array([[.6,.1],[.1,.8]])
        gradient=np.array([.5,-.7]);states=np.array([[-1.,.4],[.3,1.2],[.7,-.2]])
        class Old:
            def evaluate(self,p):return 0.,p,None

        class New:
            mean=np.zeros(2)

            def evaluate(self,p):
                ratio=normal_logs(p[None,:],target_mean,target_cov)[0]-normal_logs(p[None,:],mean,cov)[0]
                return ratio,p,None

            def moments(self,evaluation):return evaluation[1],.2*np.eye(2)

        sd=np.sqrt(gradient@cov@gradient);d=cov@gradient/sd
        origins=states-((states-mean)@gradient/sd)[:,None]*d
        precision=np.linalg.inv(target_cov);variance=1/(d@precision@d)
        centers=-(origins-target_mean)@precision@d*variance
        result=integrate_change(Old(),New(),states,mean,cov,gradient,lambda p,e:0.)
        np.testing.assert_allclose(result['eta'],origins+centers[:,None]*d,atol=1e-10)
        within=np.outer(d,d)*variance
        np.testing.assert_allclose(result['eta_covariances'],np.tile(within,(3,1,1)),atol=1e-10)
        residual=np.array([d[1],-d[0]])
        expected=normal_logs((states@residual)[:,None],np.array([target_mean@residual]),
            np.array([[residual@target_cov@residual]]))-normal_logs((states@residual)[:,None],
            np.array([mean@residual]),np.array([[residual@cov@residual]]))
        np.testing.assert_allclose(result['log_change_weights'],expected,atol=1e-10)

    def test_centered_variance_gradient_matches_finite_difference(self):
        model=VarianceReferencePosterior([.2,-.4],[[.5,.15],[.15,.3]],[.6,-.1],[.01,.04])
        states=np.array([[-3.,-2.],[-1.,-.2],[-2.,-.4]])
        weights=np.array([.2,.5,.3]);center=.1;site=1
        actual=variance_gradient(model,states,weights,site,center)
        def functional(points):
            moments=[model.moments(model.evaluate(p)) for p in points]
            return sum(w*(c[site,site]+(d[site]-center)**2) for w,(d,c) in zip(weights,moments))
        step=1e-5
        expected=[(functional(states+step*d)-functional(states-step*d))/(2*step) for d in np.eye(2)]
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-7)


if __name__=='__main__':unittest.main()
