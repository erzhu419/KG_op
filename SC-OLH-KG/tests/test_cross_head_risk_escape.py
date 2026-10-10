"""I/C/R endpoints, old-model limits and parameter-integration controls."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.cross_head_risk_escape import (assemble_profile, fit_source_prior,
    macro_prior, mixture_scores, posterior_macro_masses)
from performance.cross_head_structure_mixture import predictive_scores
from performance.source_calibrated_adaptation import fit_source_calibration, prior_adjusted_log_score


class RiskEscapeTests(unittest.TestCase):
    def setUp(self):
        self.heads = [{"step": 9 + i, "candidate": i, "head_log_evidence": e,
                       "eta_component_log_density": s} for i, (e, s) in enumerate([
            ([[2., -1.], [.3, .5], [-.8, .1]], [-.4, -1.3]),
            ([[3., -2.], [2., -1.], [.2, -.7]], [-1., -.4]),
            ([[4., -2.], [3., -1.], [.5, -.6]], [-.9, -.3])])]
        self.risks = [{"step": h["step"], "candidate": h["candidate"],
            "source": float(predictive_scores([h], 0.)[0]), "adapted": -1.4 + .1 * i,
            "null_posterior": p} for i, (h, p) in enumerate(zip(self.heads, (.2, .6, .8)))]
        self.profile = assemble_profile(self.heads, self.risks, "risk_ard_mixture", "null_posterior")

    def test_pure_components_reproduce_saved_densities_and_null_evidence(self):
        e, s = self.profile["log_evidence"], self.profile["log_predictive"]
        for k in range(3):
            np.testing.assert_allclose(mixture_scores(e, s, np.eye(3)[k]), s[:, k], atol=1e-13)
        posterior = posterior_macro_masses(e, [.5, 0., .5])
        np.testing.assert_allclose(posterior[:, 0], [f["null_posterior"] for f in self.risks], atol=1e-13)

    def test_no_escape_matches_a14_and_no_coordination_matches_a12(self):
        e, s = self.profile["log_evidence"], self.profile["log_predictive"]
        for alpha in (0., .4, 1.):
            np.testing.assert_allclose(mixture_scores(e, s, macro_prior(alpha, 0.)),
                                       predictive_scores(self.heads, alpha), atol=1e-13)
        for q in (0., .3, 1.):
            expected = [prior_adjusted_log_score(f["source"], f["adapted"], f["null_posterior"], q) for f in self.risks]
            np.testing.assert_allclose(mixture_scores(e, s, macro_prior(0., q)), expected, atol=1e-13)

    def test_generic_prior_preserves_flat_alpha_and_a12_q_calibration(self):
        risks = [{**r, "source": h["eta_component_log_density"][1]} for h, r in zip(self.heads, self.risks)]
        p = assemble_profile(self.heads, risks, "fixed_source", "null_posterior")
        fit = fit_source_prior(p["log_evidence"], p["log_predictive"], 32)
        old = fit_source_calibration(risks, 32)
        self.assertAlmostEqual(fit["alpha_mean"], .5, places=12)
        self.assertAlmostEqual(fit["alpha_sd"], np.sqrt(1. / 12.), places=12)
        self.assertAlmostEqual(fit["q_mean"], old["q_mean"], places=12)
        self.assertAlmostEqual(fit["alpha_q_covariance"], 0., places=12)

    def test_integrated_prior_masses_preserve_parameter_correlation(self):
        alpha, q, probability = np.array([.1, .9]), np.array([.2, .8]), np.array([.25, .75])
        priors = macro_prior(alpha, q)
        expected_prior = probability @ priors
        naive = macro_prior(float(probability @ alpha), float(probability @ q))
        self.assertGreater(abs(expected_prior[1] - naive[1]), .08)
        e, s = self.profile["log_evidence"], self.profile["log_predictive"]
        logs = np.log(probability)[None, :, None] + np.log(priors)[None, :, :] + e[:, None, :]
        expected = logsumexp(logs + s[:, None, :], axis=(1, 2)) - logsumexp(logs, axis=(1, 2))
        np.testing.assert_allclose(mixture_scores(e, s, expected_prior), expected, atol=1e-13)

    def test_future_eta_score_changes_do_not_change_current_model_weights(self):
        e, s = self.profile["log_evidence"], self.profile["log_predictive"]
        fit = fit_source_prior(e, s, 16)
        weights = posterior_macro_masses(e, fit["macro_prior_mass"])
        changed_scores = s.copy()
        changed_scores[:, 2] += 10.
        changed_prediction = mixture_scores(e, changed_scores, fit["macro_prior_mass"])
        np.testing.assert_array_equal(posterior_macro_masses(e, fit["macro_prior_mass"]), weights)
        self.assertFalse(np.allclose(changed_prediction, mixture_scores(e, s, fit["macro_prior_mass"])))


if __name__ == "__main__":
    unittest.main()
