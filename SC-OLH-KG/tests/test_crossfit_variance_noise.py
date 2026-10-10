import copy
import unittest
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

from performance.check_legacy_crossfit_variance_noise import task_noise
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.variance_only_moments import variance_only_moments


class CrossfitVarianceNoiseTests(unittest.TestCase):
    def test_variance_posterior_matches_independent_hermite_integration_and_units(self):
        mean, variance, s2, n = -7.3, .65, .0008, 3
        nodes, weights = np.polynomial.hermite.hermgauss(120)
        eta = mean+np.sqrt(2*variance)*nodes; k = (n-1)/2
        logs = np.log(weights/np.sqrt(np.pi))+k*(np.log(k)+np.log(s2)-eta)-k*np.exp(np.log(s2)-eta)-gammaln(k)
        weights = np.exp(logs-logsumexp(logs)); expected = weights@eta
        result = variance_only_moments(mean, variance, s2, n)
        np.testing.assert_allclose([result['eta_mean'], result['eta_variance'], result['expected_episode_variance']],
            [expected, weights@((eta-expected)**2), weights@np.exp(eta)], rtol=1e-9, atol=1e-10)
        scale = 100.
        transformed = variance_only_moments(mean+np.log(scale), variance, s2*scale, n)
        np.testing.assert_allclose([transformed['eta_mean']-np.log(scale), transformed['eta_variance'],
            transformed['expected_episode_variance']/scale],
            [result['eta_mean'], result['eta_variance'], result['expected_episode_variance']], rtol=1e-9, atol=1e-10)

    def test_current_task_means_and_holdout_labels_do_not_fit_its_noise_prior(self):
        folder = Path('paper_artifacts/legacy_joint_population_function_coverage_source_v1/layout3')
        full = read(folder/'inputs.json'); source = full['execution']
        inp, risk = read(folder/'task0/inputs.json'), read(folder/'task0/risk_source_fit.json')
        data = read(source['source_data']); public = read(Path(source['source_data']).with_name('frozen_inputs.json'))
        descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], t)),
            [tuple(p) for p in public['bank']], 50) for i, t in enumerate(public['tasks'])}
        baseline = task_noise(data, public, source, 0, full, inp, risk, descriptors, 3)
        changed = copy.deepcopy(data)
        for values in changed['tasks'][0]['Y_replicates']:
            for pair in values:
                pair[0] += 1000.; pair[1] -= 2000.
        for i in source['heldout_tasks']:
            for j, values in enumerate(changed['tasks'][i]['Y_replicates']):
                for k, pair in enumerate(values):
                    pair[:] = [10000.+j+k, -20000.-2*j+k]
        replay = task_noise(changed, public, source, 0, full, inp, risk, descriptors, 3)
        np.testing.assert_array_equal(baseline[0], replay[0]); self.assertEqual(baseline[1], replay[1])


if __name__ == '__main__':
    unittest.main()
