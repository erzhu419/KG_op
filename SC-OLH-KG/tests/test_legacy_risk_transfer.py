import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2,norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from performance.diagnose_legacy_risk_transfer import sample_variance_pit


class VarianceCalibrationTests(unittest.TestCase):
    def test_known_variance_pit_uses_the_replicate_degrees_of_freedom(self):
        for replicates in (2,3,5):
            percentiles=np.array([.1,.5,.9])
            sample=np.exp(-3.)*chi2.ppf(percentiles,replicates-1)/(replicates-1)
            np.testing.assert_allclose(sample_variance_pit(-3.,0.,sample,replicates),percentiles,atol=1e-12)

    def test_latent_variance_is_integrated_before_calibration(self):
        mean,variance,sample=-4.,.7,.02
        expected=quad(lambda eta:chi2.cdf(2*sample*np.exp(-eta),2)*norm.pdf(eta,mean,np.sqrt(variance)),
            mean-10*np.sqrt(variance),mean+10*np.sqrt(variance),epsabs=1e-10)[0]
        self.assertAlmostEqual(float(sample_variance_pit(mean,variance,sample,3)),expected,delta=1e-7)


if __name__=='__main__':
    unittest.main()
