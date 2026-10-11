import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import norm

from core.marginal_certificate_bound import gaussian_feasibility_upper, condition_pair, certificate_upper, branch_quality_upper
from core.guarded_deployment_kg import conditional_safety_power
from core.raw_likelihood_lookahead import risk_branch_statistics


class MarginalCertificateBoundTests(unittest.TestCase):
    def test_correlated_gaussian_bound_contains_adaptive_feasibility_integral(self):
        z = ndtri(.9)
        for cross in (-.12,.12):
            mean = np.array([-.4,-2.]);covariance = np.array([[.16,cross],[cross,.5]])
            sd = np.sqrt(covariance[1,1]);gv = covariance[0,0]-cross**2/covariance[1,1]
            actual = quad(lambda t:norm.pdf(t)*ndtr(-(mean[0]+cross/sd*t+z*np.exp((mean[1]+sd*t)/2))/np.sqrt(gv)),
                -10,10,epsabs=1e-11)[0]
            upper = gaussian_feasibility_upper(mean,covariance,z)
            self.assertGreaterEqual(upper,actual-1e-10)
            self.assertLess(upper,1.)

    def test_raw_conditioning_matches_full_joint_gaussian_and_keeps_query_policy(self):
        mean = np.array([[.1,-2.,-.2,-1.]])
        root = np.array([[.4,0,0,0],[.1,.5,0,0],[.2,-.1,.3,0],[.05,.2,-.1,.4]])
        covariance = (root@root.T)[None];eta = np.array([-1.7]);label = -.3
        m,c = condition_pair(mean,covariance,0,1,eta,label)
        observed = [1,0];target = [2,3]
        noise = np.diag([0.,np.exp(eta[0])]);C = covariance[0]
        gain = C[np.ix_(target,observed)]@np.linalg.inv(C[np.ix_(observed,observed)]+noise)
        np.testing.assert_allclose(m[0],mean[0,target]+gain@(np.array([eta[0],label])-mean[0,observed]),atol=1e-14)
        np.testing.assert_allclose(c[0],C[np.ix_(target,target)]-gain@C[np.ix_(observed,target)],atol=1e-14)
        m,c = condition_pair(mean,covariance,0,0,eta,label)
        self.assertAlmostEqual(m[0,1],eta[0]);self.assertAlmostEqual(c[0,1,1],0.)
        self.assertTrue(np.isfinite(gaussian_feasibility_upper(m,c,ndtri(.9))).all())

    def test_boundary_uses_implemented_verifier_power(self):
        protocol = dict(chance_alpha=.1,familywise_delta=.05)
        for budget,size in [(81,3),(130,3),(359,1)]:
            upper = certificate_upper(0.,budget,size,protocol)
            g = -ndtri(.9)+np.array([0.,.1,1.])
            actual = conditional_safety_power(g,np.ones(3),budget,.1,.05/size)
            self.assertTrue(np.all(actual<=upper+1e-14))

    def test_branch_bound_contains_correlated_verifier_and_tightens_quality_only(self):
        rng = np.random.default_rng(19);powers = rng.uniform(size=(200,1,3))
        branches,_ = risk_branch_statistics(powers,np.ones((200,1)),1,[81,130,132],dict(comparison_budget=8))
        marginals = powers[:,0].mean(axis=0);quality = np.array([.9,.4,.3,.7])
        upper = branch_quality_upper(quality,marginals,1)
        self.assertGreaterEqual(upper,quality@branches.mean(axis=0)-1e-14)
        self.assertLessEqual(upper,quality.max()+1e-14)
        self.assertLess(branch_quality_upper(quality,[.01,.2,.1],1),.9)


if __name__ == '__main__':
    unittest.main()
