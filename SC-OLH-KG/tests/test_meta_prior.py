import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    AdmissibleProblemAdapter,
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)


class MetaPriorTests(unittest.TestCase):
    def _problem(self, name, d=8):
        return ScalarizedProblem(make_problem(name, d=d, L=100, sigma=0.04))

    def test_strict_adapter_hides_target_specific_structural_hooks(self):
        base = self._problem("FactorShockStatePolicyRZDT1")
        self.assertTrue(hasattr(base, "risk_exposures"))
        self.assertTrue(hasattr(base, "initial_samples"))
        wrapped = AdmissibleProblemAdapter(base)
        self.assertFalse(hasattr(wrapped, "risk_exposures"))
        self.assertFalse(hasattr(wrapped, "hvd_features"))
        self.assertFalse(hasattr(wrapped, "initial_samples"))
        self.assertFalse(hasattr(wrapped, "recommendation_refinement_candidates"))
        self.assertTrue(wrapped.admissibility_audit()["admissible_mainline"])

    def test_lodo_meta_prior_exposes_frozen_learned_risk_coordinate(self):
        rng = np.random.default_rng(7)
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem("FactorShockStatePolicyRZDT1")),
            ("InventorySupplyChain", self._problem("InventorySupplyChain")),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            anchor_count=5,
            seed=7,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=8,
            rng=rng,
        )
        target = MetaPriorProblemAdapter(self._problem("QueueResourceControl"), prior)
        diag = prior.diagnostics()
        self.assertIn("training", diag)
        self.assertIn("source_feasible_rate", diag["training"])
        self.assertIn("anchor_types", diag["training"])
        x = target.sample_random(np.random.default_rng(8))
        exposure = target.risk_exposures(x)
        self.assertEqual(exposure.A.shape, (3,))
        self.assertEqual(exposure.N.shape, (2,))
        self.assertAlmostEqual(float(np.sum(exposure.N)), 1.0, places=7)
        self.assertEqual(
            len(target.cumulative_risk_features(x)),
            1 + 3 + 2 * (2 + 1) // 2 + 2,
        )
        self.assertIsNotNone(target.cumulative_hvd_prior_beta(1))
        self.assertTrue(target.admissibility_audit()["admissible_mainline"])
        self.assertGreater(len(target.initial_samples(n=3, rng=np.random.default_rng(9))), 0)


if __name__ == "__main__":
    unittest.main()
