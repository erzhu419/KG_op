import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from representation.joint_function_risk import JointFunctionRiskPosterior, local_joint_moments
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


def prior(basis, intercept):
    return SourceFunctionPrior(basis, np.r_[intercept, np.zeros(basis.feature_dim)],
        np.diag(np.r_[.04, np.full(basis.feature_dim, .002)]), .01, {})


class PublicControlMeanTests(unittest.TestCase):
    def test_public_coordinates_match_actual_control_groups_at_both_supported_dimensions(self):
        rng = np.random.default_rng(80901)
        for dimension in (50, 1000):
            problem = no_outcomes(build_scalarized_problem('InventorySupplyChain', dimension, 100, .02, .05, [.5, .5]))
            points = [tuple(rng.integers(10, 80, dimension)) for _ in range(3)]
            actual = public_control_mean_coordinates(problem, points)
            expected = [[*[g.mean() for g in np.array_split(np.asarray(p)/100, 3)], np.std(np.asarray(p)/100)] for p in points]
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0.)
            self.assertEqual(actual.shape, (3, 4))

    def test_moment_interface_matches_native_raw_filter_when_head_coordinates_are_shared(self):
        psi = np.array([[-1.], [0.], [1.]])
        basis = FixedFunctionBasis(psi)
        priors = [prior(basis, v) for v in (.3, -.05, -3., -4.)]
        actual = paired_fields([source_prior_moments(p, psi) for p in priors])
        expected = (JointFunctionRiskPosterior([priors[0]], [priors[3]], psi),
                    JointFunctionRiskPosterior([priors[1]], [priors[2]], psi))
        for j, values in ((0, [.35, -.10]), (1, [.28, .02])):
            for head, (a, b) in enumerate(zip(actual, expected)):
                self.assertAlmostEqual(a.update_batch(j, values[head], None, 1), b.update_batch(j, values[head], None, 1), places=12)
        for a, b in zip(actual, expected):
            for am, bm in zip(a.predict(full_cov=True), b.predict(full_cov=True)):
                np.testing.assert_allclose(am, bm, atol=2e-14, rtol=1e-12)

    def test_four_dimensional_mean_and_ten_dimensional_eta_keep_head_order_and_condition_all_sites(self):
        rng = np.random.default_rng(80902)
        mean_psi, risk_psi = rng.normal(size=(7, 4)), rng.normal(size=(7, 10))
        mean_basis, risk_basis = FixedFunctionBasis(mean_psi), FixedFunctionBasis(risk_psi)
        priors = [prior(mean_basis, .3), prior(mean_basis, -.05), prior(risk_basis, -3.), prior(risk_basis, -4.)]
        moments = [source_prior_moments(p, x) for p, x in zip(priors, [mean_psi, mean_psi, risk_psi, risk_psi])]
        fields = paired_fields(moments)
        for field, (a, b), value in zip(fields, [(0, 3), (1, 2)], [.35, -.10]):
            before_mean, before_cov = field.predict(full_cov=True)
            np.testing.assert_array_equal(before_mean, np.column_stack([moments[a][0], moments[b][0]]))
            np.testing.assert_array_equal(before_cov[:, 0, :, 0], moments[a][1])
            np.testing.assert_array_equal(before_cov[:, 1, :, 1], moments[b][1])
            np.testing.assert_array_equal(before_cov[:, 0, :, 1], np.zeros((7, 7)))
            mean = before_mean.reshape(-1); covariance = before_cov.reshape(14, 14)
            local = local_joint_moments(mean[:2], covariance[:2, :2], value, None, 1)
            gain = np.linalg.solve(covariance[:2, :2], covariance[:2]).T
            expected_mean = mean+gain @ (local['mean']-mean[:2])
            expected_cov = covariance+gain @ (local['covariance']-covariance[:2, :2]) @ gain.T
            field.update_batch(0, value, None, 1)
            after_mean, after_cov = field.predict(full_cov=True)
            np.testing.assert_allclose(after_mean.reshape(-1), expected_mean, atol=1e-13)
            np.testing.assert_allclose(after_cov.reshape(14, 14), expected_cov, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
