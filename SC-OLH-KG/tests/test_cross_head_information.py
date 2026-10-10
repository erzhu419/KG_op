"""A13 checks: mixture uncertainty, coefficient boundaries and head scaling."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.cross_head_information import coefficient_energy, posterior_signals, score_support
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.transfer_mixture import TransferMixtureGPR


class CrossHeadInformationTests(unittest.TestCase):
    def test_mixture_energy_retains_between_component_uncertainty(self):
        psi = np.random.default_rng(8).uniform(-2., 2., (24, 2))
        basis = FixedFunctionBasis(psi)
        width = basis.feature_dim + 1
        priors = [SourceFunctionPrior(basis, np.zeros(width), np.eye(width) * .2, .02, {}) for _ in range(2)]
        mixture = TransferMixtureGPR(priors, psi, psi)
        mixture.log_weights = np.log([.3, .7])
        mixture.models[0].a[1:3] = [3., -1.]
        mixture.models[1].a[1:3] = [-3., 1.]
        mixture.refresh()
        expected = sum(w * (m.a[1:] ** 2 + np.diag(m.C)[1:])
                       for w, m in zip([.3, .7], mixture.models))
        np.testing.assert_allclose(coefficient_energy(mixture, basis.feature_dim), expected / expected.sum())
        naive = mixture.a[1:] ** 2 + np.diag(mixture.models[0].C)[1:]
        self.assertGreater(np.max(np.abs(expected / expected.sum() - naive / naive.sum())), .05)

    def test_intercept_and_appended_discrepancy_do_not_enter_energy(self):
        model = SimpleNamespace(a=np.array([0., 1., 2., 3., 0.]), C=np.diag([1., 2., 3., 4., 1.]))
        expected = coefficient_energy(model, 3)
        model.a[[0, 4]] = [1e6, -1e6]
        model.C[0, 0] = model.C[4, 4] = 1e12
        np.testing.assert_array_equal(coefficient_energy(model, 3), expected)

    def test_joint_signal_normalizes_each_head_before_combining(self):
        heads = [SimpleNamespace(a=np.array([0., 3., 1.]), C=np.zeros((3, 3))),
                 SimpleNamespace(a=np.array([0., 1., 3.]), C=np.zeros((3, 3))),
                 SimpleNamespace(a=np.array([0., 2., 1.]), C=np.zeros((3, 3)))]
        before = posterior_signals(heads, 2)
        heads[0].a *= 100.
        heads[1].a *= .01
        after = posterior_signals(heads, 2)
        np.testing.assert_allclose(before["fg"], [.5, .5])
        np.testing.assert_allclose(after["fg"], before["fg"])

    def test_scoring_truth_changes_scores_without_changing_fixed_signals(self):
        models = [SimpleNamespace(a=np.array([0., 3., 1.]), C=np.eye(3)) for _ in range(3)]
        signals = posterior_signals(models, 2)
        saved = deepcopy(signals)
        score_a, score_b = score_support(signals, [0]), score_support(signals, [1])
        self.assertNotEqual(score_a["support_mass"]["fg"], score_b["support_mass"]["fg"])
        self.assertEqual(signals, saved)
        self.assertEqual(posterior_signals(models, 2), saved)


if __name__ == "__main__":
    unittest.main()
