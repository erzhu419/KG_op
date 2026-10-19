import unittest
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import norm

from core.guarded_deployment_kg import conditional_safety_power, _sd_quadrature, _tolerance_factor
from core.marginal_certificate_bound import gaussian_feasibility_upper, multilevel_certificate_upper,projected_ratio_cdf_upper


class MultilevelCertificateTest(unittest.TestCase):
    protocol=dict(chance_alpha=.1,familywise_delta=.05)
    thresholds=np.sort(np.r_[[-4.,-3.5,-3.,-2.75,-2.5,-2.25,-2.,-1.9,-1.8,-1.7,-1.6,-1.5,-1.,-.5],-ndtri(.9)])

    def test_step_envelope_contains_sd16_power_on_both_sides_of_each_threshold(self):
        t=np.r_[np.linspace(-8.,2.,2001),self.thresholds,self.thresholds-1e-10,self.thresholds+1e-10]
        indicators=(t[:,None]<=self.thresholds).astype(float)
        for budget,size in [(81,3),(130,3),(132,3),(356,1)]:
            upper=multilevel_certificate_upper(indicators,self.thresholds,budget,size,self.protocol)
            actual=conditional_safety_power(t,np.ones_like(t),budget,.1,.05/size)
            self.assertTrue(np.all(upper>=actual-1e-14))
            self.assertTrue(np.all(upper<=1.+1e-14))

    def test_upper_contains_exact_sd16_expectation_with_correlated_mean_and_log_variance(self):
        for mean,covariance in [(np.array([-1.8,0.]),np.array([[.12,.025],[.025,.04]])),
                (np.array([-.4,-1.2]),np.array([[.14,-.06],[-.06,.3]]))]:
            caps=np.asarray([gaussian_feasibility_upper(mean,covariance,-t) for t in self.thresholds])
            ev=covariance[1,1];loading=covariance[0,1]/np.sqrt(ev)
            gv=covariance[0,0]-covariance[0,1]**2/ev
            for budget in (81,130):
                sd,w=_sd_quadrature(budget,16);factor=_tolerance_factor(budget,.1,.05/3)
                def integrand(z):
                    sigma=np.exp((mean[1]+np.sqrt(ev)*z)/2)
                    g=mean[0]+loading*z
                    power=ndtr(np.sqrt(budget)*(-g/sigma-factor*sd)/np.sqrt(1+budget*gv/sigma**2)) @ w
                    return norm.pdf(z)*power
                actual=quad(integrand,-10,10,epsabs=1e-11)[0]
                upper=multilevel_certificate_upper(caps,self.thresholds,budget,3,self.protocol)
                self.assertGreaterEqual(upper,actual-1e-10)
                self.assertLess(upper,1.)

    def test_projected_event_bounds_keep_all_component_weights(self):
        field=SimpleNamespace(means=np.array([[-2.,0.],[-1.,0.]]),
            covariances=np.array([[[.1,0.],[0.,0.]],[[.4,0.],[0.,0.]]]),log_weights=np.log([.3,.7]))
        upper=projected_ratio_cdf_upper(field,0,self.thresholds)
        exact=.3*ndtr((self.thresholds+2.)/np.sqrt(.1))+.7*ndtr((self.thresholds+1.)/np.sqrt(.4))
        np.testing.assert_allclose(upper,exact,atol=1e-14)


if __name__=='__main__':
    unittest.main()
