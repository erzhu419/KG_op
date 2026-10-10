import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.transferable_spectral import SourceDomainBatch
from representation.conditional_function_prior import conditional_selection, enable_target_adaptation
from performance.run_meta_prior_mechanism_v2 import prepare_sources, PROTOCOL


class ConditionalPriorTests(unittest.TestCase):
    def test_joint_selection_recovers_quadratic_interaction_without_sine_duplicate(self):
        rng = np.random.default_rng(901)
        public = rng.uniform(-2, 2, (96, 8))
        basis = FixedFunctionBasis(public)
        indices = [0, 10, 32]
        batches = []
        for task in range(8):
            X = rng.uniform(-2, 2, (64, 8))
            beta = rng.choice([-1, 1], 3) * rng.uniform(0.8, 1.4, 3)
            y = basis.transform(X)[:, indices] @ beta + rng.normal(0, 0.03, 64)
            batches.append(SourceDomainBatch(str(task), X, y))
        chosen, _ = conditional_selection(batches, [np.full(64, 0.03 ** 2)] * 8, basis)
        self.assertTrue(set(indices).issubset(chosen))
        self.assertNotIn(16, chosen)  # correlated sine of coordinate zero

    def test_paid_target_can_update_a_source_disfavored_function(self):
        rng = np.random.default_rng(902)
        psi = rng.uniform(-2, 2, (64, 8))
        basis = FixedFunctionBasis(psi)
        prior = SourceFunctionPrior(basis, np.zeros(69), np.eye(69), 0.005,
                    {"source_pip": [0.9] + [0.05] * 67, "source_slab_scale": [1.0] * 68})
        model = prior.target_model(psi)
        y = 3.0 * basis.transform(psi)[:, 6]
        records = [([i], y[i], 0.01) for i in range(48)]
        enable_target_adaptation(model, prior, records)
        diagnostic = model.adaptive_sparsity_diagnostics()
        self.assertGreater(diagnostic["posterior_pip"][6], 0.5)
        self.assertGreater(diagnostic["escaped_source_spike_count"], 0)
        model.update([48], y[48], 0.01)
        self.assertEqual(model.adaptive_sparsity_diagnostics()["n_observations"], 49)

    def test_separate_heads_control_changes_sources_and_targets_together(self):
        protocol = json.loads(PROTOCOL.read_text())
        tasks, supports = prepare_sources(8, "separate_head_structure", protocol)
        shared, _ = prepare_sources(8, "shared_structure", protocol)
        self.assertNotEqual(supports[0], supports[2])
        np.testing.assert_array_equal(tasks[0]["response"][:, 0], shared[0]["response"][:, 0])
        self.assertFalse(np.array_equal(tasks[0]["response"][:, 2], shared[0]["response"][:, 2]))


if __name__ == "__main__":
    unittest.main()
