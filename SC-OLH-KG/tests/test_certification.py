import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition.olhkg import OLHKGAcquisition  # noqa: E402
from core.certification import conservative_chance_margin  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
