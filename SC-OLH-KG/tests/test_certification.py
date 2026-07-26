import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm, t


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition.olhkg import OLHKGAcquisition  # noqa: E402
from core.certification import (  # noqa: E402
    CertificationResult,
    conservative_chance_margin,
    decompose_chance_margin,
    gaussian_replication_chance_certificate,
)


class DummyConstraintGPR:
    def posterior_mean_many(self, candidates):
        return np.full(len(candidates), 0.1, dtype=float)

    def posterior_var_many(self, candidates):
        return np.array([0.25, 0.0][:len(candidates)], dtype=float)


class DummyVarianceModel:
    def predict_certification_variance_many(self, i, candidates, problem):
        del i, problem
        return np.full(len(candidates), 0.04, dtype=float)


class DummyProblem:
    tau = 1.0
    alpha = 0.05


class CertificationTests(unittest.TestCase):
    def test_theory_margin_uses_epistemic_and_cumulative_variance(self):
        cert = conservative_chance_margin(
            [0.1],
            [0.25],
            [0.04],
            tau=1.0,
            alpha=0.05,
            beta_g=4.0,
            mode="theory",
        )
        expected = 0.1 + 2.0 * 0.5 + norm.ppf(0.95) * 0.2 - 1.0
        self.assertAlmostEqual(float(cert.margin[0]), expected)
        self.assertEqual(cert.mode, "theory")
        self.assertEqual(cert.beta_g, 4.0)

    def test_legacy_margin_zeroes_epistemic_term(self):
        theory = conservative_chance_margin(
            [0.1], [0.25], [0.04], tau=1.0, alpha=0.05, beta_g=4.0)
        legacy = conservative_chance_margin(
            [0.1], [0.25], [0.04], tau=1.0, alpha=0.05,
            beta_g=4.0, mode="legacy")
        self.assertGreater(float(theory.margin[0]), float(legacy.margin[0]))
        self.assertEqual(legacy.beta_g, 0.0)

    def test_guard_decomposition_reconstructs_joint_robust_margin(self):
        nominal = conservative_chance_margin(
            [0.1, -0.4],
            [0.25, 0.01],
            [0.04, 0.01],
            tau=1.0,
            alpha=0.05,
            beta_g=4.0,
        )
        joint = CertificationResult(
            margin=np.asarray(nominal.margin) + np.array([0.3, -0.05]),
            mu=nominal.mu,
            epistemic_var=nominal.epistemic_var,
            aleatoric_var=nominal.aleatoric_var,
            beta_g=nominal.beta_g,
            z_alpha=nominal.z_alpha,
            tau=nominal.tau,
            mode=nominal.mode,
        )
        decomposition = decompose_chance_margin(joint)
        np.testing.assert_allclose(
            decomposition.reconstructed_margin, joint.margin)
        np.testing.assert_allclose(
            decomposition.joint_epistemic_guard, [0.3, 0.0])
        np.testing.assert_allclose(
            decomposition.favorable_coupling, [0.0, -0.05])
        self.assertEqual(decomposition.dominant_mode[0], "epistemic")
        self.assertEqual(decomposition.dominant_mode[1], "interior")

    def test_gaussian_replication_certificate_matches_t_chi_square_bound(self):
        values = np.array([-0.20, -0.18, -0.19, -0.21, -0.17])
        result = gaussian_replication_chance_certificate(
            values,
            tau=0.0,
            alpha=0.05,
            delta=0.10,
        )
        n = len(values)
        sample_std = float(np.std(values, ddof=1))
        expected_t = float(t.ppf(0.95, n - 1))
        expected_chi2 = float(chi2.ppf(0.05, n - 1))
        expected_mean_upper = float(
            np.mean(values) + expected_t * sample_std / np.sqrt(n))
        expected_sigma_upper = float(
            sample_std * np.sqrt((n - 1) / expected_chi2))
        expected_margin = (
            expected_mean_upper
            + norm.ppf(0.95) * expected_sigma_upper
        )
        self.assertAlmostEqual(result.mean_upper, expected_mean_upper)
        self.assertAlmostEqual(result.sigma_upper, expected_sigma_upper)
        self.assertAlmostEqual(result.upper_margin, expected_margin)
        self.assertEqual(result.sample_count, n)
        self.assertEqual(result.status, "certified")
        self.assertTrue(result.certified)

    def test_gaussian_replication_certificate_requires_two_new_samples(self):
        result = gaussian_replication_chance_certificate(
            [-0.5],
            tau=0.0,
            alpha=0.05,
            delta=0.05,
        )
        self.assertEqual(result.status, "insufficient_replications")
        self.assertFalse(result.certified)
        self.assertTrue(np.isinf(result.upper_margin))

    def test_gaussian_replication_certificate_requires_nonnegative_z(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            gaussian_replication_chance_certificate(
                [-1.0, -0.9],
                tau=0.0,
                alpha=0.75,
            )

    def test_acquisition_feasibility_details_expose_theory_bound(self):
        candidates = [(0,), (1,)]
        acquisition = OLHKGAcquisition(
            beta_g=4.0,
            certification_mode="theory",
        )
        _, details = acquisition.feasibility_scores(
            candidates,
            DummyConstraintGPR(),
            DummyVarianceModel(),
            DummyProblem(),
        )
        first_expected = 0.1 + 2.0 * 0.5 + norm.ppf(0.95) * 0.2 - 1.0
        second_expected = 0.1 + norm.ppf(0.95) * 0.2 - 1.0
        self.assertAlmostEqual(details[0]["chance_margin"], first_expected)
        self.assertAlmostEqual(details[1]["chance_margin"], second_expected)
        self.assertEqual(details[0]["certification_mode"], "theory")
        self.assertEqual(details[0]["variance_g"], 0.04)
        self.assertEqual(details[0]["epistemic_variance_g"], 0.25)

    def test_constraint_epistemic_score_targets_uncertain_boundary(self):
        acquisition = OLHKGAcquisition(
            constraint_epistemic_margin_softening=3.0,
        )
        details = [
            {
                "chance_margin": 0.6,
                "variance_g": 0.01,
                "total_certification_variance_g": 0.04,
                "epistemic_variance_g": 0.25,
            },
            {
                "chance_margin": 0.1,
                "variance_g": 0.01,
                "total_certification_variance_g": 0.04,
                "epistemic_variance_g": 0.01,
            },
        ]
        scores = acquisition.constraint_epistemic_scores(details)
        self.assertEqual(scores.shape, (2,))
        self.assertGreater(float(scores[0]), float(scores[1]))


if __name__ == "__main__":
    unittest.main()
