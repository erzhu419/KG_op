"""Joint-mixture controls, chronological isolation and source-only calibration."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.cross_head_structure_mixture import (CHOICES, CrossHeadStructureMixture,
    calibrate_source_archives, fit_source_components, forecast_sequence, joint_log_weights,
    predictive_scores, structural_weights)
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.transfer_mixture import TransferMixtureGPR


class CrossHeadStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.psi = np.random.default_rng(21).uniform(-2., 2., (24, 2))
        basis = FixedFunctionBasis(cls.psi)
        small = FixedFunctionBasis(cls.psi, [0, 2])
        cls.full = [SourceFunctionPrior(basis, np.zeros(basis.feature_dim + 1),
            np.eye(basis.feature_dim + 1) * (.1 + .03 * h), .02, {}) for h in range(3)]
        cls.structural = [SourceFunctionPrior(small, np.zeros(3), np.eye(3) * .2, .03, {}) for _ in range(3)]

    def model(self):
        return CrossHeadStructureMixture(self.structural, self.full, self.psi, self.psi)

    def test_zero_alpha_factorizes_and_each_component_updates_once(self):
        joint = self.model()
        old = [TransferMixtureGPR([self.structural[h], self.full[h]], self.psi, self.psi) for h in range(3)]
        rng = np.random.default_rng(9)
        for index in (0, 2, 0, 3, 4, 5, 7, 8):
            response, noise = rng.normal(size=3), np.array([.01, .04, .2])
            joint.update(np.array([index]), response, noise)
            for h in range(3):
                old[h].update(np.array([index]), response[h], noise[h])
                for a, b in zip(joint.models[h], old[h].models):
                    np.testing.assert_allclose(a.a, b.a, atol=1e-13)
                    np.testing.assert_allclose(a.C, b.C, atol=1e-13)
        expected = [m.diagnostics()["structural_weight"] for m in old]
        np.testing.assert_allclose(structural_weights(joint.log_evidence, 0.), expected, atol=1e-13)
        mean, var, _ = joint.marginal_moments(np.array([0]))
        np.testing.assert_allclose(mean, [m.posterior_mean_many([[0]])[0] for m in old], atol=1e-13)
        np.testing.assert_allclose(var, [m.posterior_var_many([[0]])[0] for m in old], atol=1e-13)
        self.assertEqual(joint.observations, 8)

    def test_collapsed_score_matches_direct_eight_configuration_enumeration(self):
        history = [[2., -1.], [.3, .5], [-.8, .1]]
        densities = np.array([-.4, -1.3])
        forecast = {"head_log_evidence": history, "eta_component_log_density": densities.tolist()}
        for alpha in (0., .3, 1.):
            expected = logsumexp(joint_log_weights(history, alpha) + densities[CHOICES[:, 2]])
            self.assertAlmostEqual(predictive_scores([forecast], alpha)[0], expected, places=12)

    def test_independent_risk_configuration_survives_strong_mean_head_evidence(self):
        evidence = np.array([[20., -20.], [20., -20.], [-20., 20.]])
        logs = joint_log_weights(evidence, .8)
        discordant = np.all(CHOICES == [0, 0, 1], axis=1)
        self.assertGreater(np.exp(logs[discordant])[0], .999)
        self.assertLess(structural_weights(evidence, .8)[2], .001)

    def test_current_and_future_labels_cannot_change_issued_forecast(self):
        rng = np.random.default_rng(11)
        labels, noise = rng.normal(size=(12, 3)), np.tile([.01, .04, .2], (12, 1))
        before = forecast_sequence(self.model(), range(12), lambda t, i: (labels[t], noise[t]), .2)[0]
        changed, changed_noise = labels.copy(), noise.copy()
        changed[8:] += 5.
        changed_noise[8:, :2] *= 100.
        after = forecast_sequence(self.model(), range(12), lambda t, i: (changed[t], changed_noise[t]), .2)[0]
        for key in ("head_log_evidence", "eta_component_mean", "eta_component_variance"):
            self.assertEqual(before[key], after[key])
        self.assertNotEqual(before["independent_log_score"], after["independent_log_score"])

    def test_held_function_is_excluded_from_all_three_component_fits(self):
        rng = np.random.default_rng(42)
        tasks = []
        for task in range(4):
            psi = rng.uniform(-2., 2., (24, 2))
            labels = np.column_stack([psi[:, 0] * (1. + .1 * task), psi[:, 1] ** 2,
                                      -.3 + .2 * psi[:, 0]])
            tasks.append({"psi": psi, "response": labels, "noise": np.tile([.01, .04, .2], (24, 1))})
        full, structural, info = fit_source_components(tasks, 0, self.psi, active_dim=2)
        changed = deepcopy(tasks)
        changed[0]["response"] += 100.
        changed[0]["noise"] *= 50.
        other_full, other_structural, other_info = fit_source_components(changed, 0, self.psi, active_dim=2)
        self.assertEqual(info, other_info)
        self.assertEqual(info["training_tasks"], [1, 2, 3])
        self.assertEqual(info["source_label_count_per_head"], 72)
        for a, b in zip(full + structural, other_full + other_structural):
            np.testing.assert_array_equal(a.mean, b.mean)
            np.testing.assert_array_equal(a.covariance, b.covariance)
            self.assertEqual(a.basis.names, b.basis.names)

    def test_calibration_uses_source_forecasts_and_preserves_flat_uncertainty(self):
        forecast = {"head_log_evidence": [[0., 0.]] * 3, "eta_component_log_density": [-.3, -.3]}
        source = [{"seed": 48, "archive_group": "shared", "forecasts": [forecast]}]
        fits = calibrate_source_archives(source)
        self.assertAlmostEqual(fits[0]["alpha_mean"], .5, places=12)
        self.assertAlmostEqual(fits[0]["alpha_sd"], np.sqrt(1. / 12.), places=12)
        # Target scores are absent from the calibration interface.
        changed_target = {"eta_component_log_density": [-100., 100.]}
        predictive_scores([{**forecast, **changed_target}], fits[0]["alpha_mean"])
        self.assertEqual(calibrate_source_archives(source), fits)


if __name__ == "__main__":
    unittest.main()
