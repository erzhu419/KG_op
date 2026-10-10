import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_meta_prior_mechanism import PROTOCOL, dictionary, kg_scores, observe
from representation.source_function_prior import SourceFunctionPrior
from representation.transferable_spectral import SourceDomainBatch


class SourceFunctionPriorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(771)
        self.psi = rng.uniform(-2, 2, (32, 8))
        self.source = [rng.uniform(-2, 2, (40, 8)) for _ in range(3)]
        self.protocol = json.loads(PROTOCOL.read_text())

    def prior(self, coordinate):
        batches = [SourceDomainBatch(str(i), X, (i + 1) * dictionary(X)[:, coordinate])
                   for i, X in enumerate(self.source)]
        return SourceFunctionPrior.fit(batches, [np.full(40, 0.01)] * 3,
                                       self.psi, "learned", active_dim=2)

    def test_source_labels_change_functions_posterior_and_query_scores(self):
        first, second = self.prior(0), self.prior(5)
        self.assertNotEqual(first.diagnostics["selected_names"],
                            second.diagnostics["selected_names"])
        candidates = np.arange(len(self.psi))[:, None]
        models = []
        for prior in (first, second):
            heads = [prior.target_model(self.psi) for _ in range(3)]
            for i in range(8):
                # Entire paid target design and labels are identical.
                for head, model in enumerate(heads):
                    model.update([i], [float(dictionary(self.psi)[i, 0]), -0.6, -2.5][head], 0.03)
            models.append(heads)
        delta = np.max(np.abs(models[0][0].posterior_mean_many(candidates)
                              - models[1][0].posterior_mean_many(candidates)))
        self.assertGreater(delta, 0.01)
        scores = [kg_scores(heads, candidates, self.protocol) for heads in models]
        self.assertGreater(float(np.max(np.abs(scores[0] - scores[1]))), 1e-4)
        np.testing.assert_array_equal(first.mean[1:], 0.0)
        np.testing.assert_array_equal(second.mean[1:], 0.0)

    def test_integer_candidate_adapter_and_unvisited_discrepancy_update(self):
        prior = self.prior(0)
        model = prior.target_model(self.psi)
        candidates = np.arange(len(self.psi))[:, None]
        A = model.augmented_feature_matrix(candidates)
        covariance = A @ model.C @ A.T + model.lambda_i * np.eye(len(A))
        original = model.posterior_mean_many(candidates)
        noise = 0.03
        expected = original + covariance[:, 9] / (covariance[9, 9] + noise) * 0.7
        model.update([9], original[9] + 0.7, noise)
        np.testing.assert_allclose(model.posterior_mean_many(candidates), expected, atol=1e-10)
        self.assertEqual(model.sol_to_idx, {(9,): 0})

    def test_acquisition_has_no_simulator_or_target_truth_input(self):
        prior = self.prior(0)
        heads = [prior.target_model(self.psi) for _ in range(3)]
        for i in range(8):
            for head, model in enumerate(heads):
                model.update([i], [0.1 * i, -0.7, -2.5][head], 0.1)
        candidates = np.arange(len(self.psi))[:, None]
        before = [copy.deepcopy(model.__dict__) for model in heads]
        score = kg_scores(heads, candidates, self.protocol)
        self.assertTrue(np.isfinite(score).all())
        for model, state in zip(heads, before):
            np.testing.assert_array_equal(model.a, state["a"])
            np.testing.assert_array_equal(model.C, state["C"])

    def test_replicated_log_variance_label_bias_correction(self):
        truth = np.array([0.0, -0.3, -1.7])
        labels = [observe(truth, i + 7100, 0, 0, 12)[0] for i in range(1200)]
        np.testing.assert_allclose(np.mean(labels, axis=0), truth, atol=0.04)

    def test_frozen_budget_includes_online_search_and_charged_validation(self):
        self.assertEqual(self.protocol["target_points"] - self.protocol["initial_points"], 16)
        self.assertGreater(self.protocol["verification_replicates"], 0)
        self.assertEqual(len(self.protocol["seeds"]) * len(self.protocol["regimes"]), 32)


if __name__ == "__main__":
    unittest.main()
