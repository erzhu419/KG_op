import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from encoders.policy_state_encoder import StateCoupledFeatureMap  # noqa: E402
from problems.rzdt import HighDimStatePolicyRZDT1, StatePolicyRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.manifold import (  # noqa: E402
    KernelManifoldEncoder,
    ManifoldRiskDecomposer,
    PCAManifoldEncoder,
)
from representation.ssl_encoder import (  # noqa: E402
    ContrastivePolicyEncoder,
    MaskedTrajectoryEncoder,
    NextRiskEncoder,
    SmallTransformerEncoder,
)


def tiny_records():
    records = []
    for pid, queue in [("p0", 1.0), ("p0_rep", 1.2), ("p1", 8.0)]:
        for t in range(4):
            records.append({
                "policy_id": pid,
                "time": str(t),
                "state": f"s{t % 2}",
                "action": f"a{t % 3}",
                "occupancy": "1",
                "queue": str(queue + 0.1 * t),
                "wait": str(0.5 * queue),
                "flow": str(10.0 - queue),
                "demand_shock": str(0.05 * t),
            })
    return records


def tiny_records_with_x():
    xs = {
        "p0": (20, 70, 70, 70, 70),
        "p1": (50, 90, 90, 90, 90),
    }
    records = []
    for pid, x in xs.items():
        queue = 1.0 if pid == "p0" else 8.0
        for t in range(3):
            records.append({
                "policy_id": pid,
                "time": str(t),
                "state": f"s{t}",
                "action": f"a{t}",
                "occupancy": "1",
                "queue": str(queue),
                "wait": str(0.5 * queue),
                "flow": str(10.0 - queue),
                "demand_shock": "0.1",
                "x": " ".join(map(str, x)),
            })
    return records


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.problem = ScalarizedProblem(StatePolicyRZDT1(d=5, L=100, sigma=0.04))

    def test_pca_manifold_features_are_stable_and_finite(self):
        encoder = PCAManifoldEncoder(
            self.problem,
            latent_dim=4,
            fit_pool_size=32,
            rng=np.random.default_rng(0),
        )
        x = (25, 70, 70, 70, 70)
        f1 = encoder.features(x)
        f2 = encoder.features(x)
        self.assertEqual(f1.shape, (4,))
        self.assertTrue(np.all(np.isfinite(f1)))
        self.assertTrue(np.allclose(f1, f2))
        self.assertEqual(encoder.diagnostics()["encoder"], "pca_manifold")

    def test_kernel_manifold_small_sample_falls_back_to_pca(self):
        encoder = KernelManifoldEncoder(
            self.problem,
            latent_dim=4,
            fit_pool_size=2,
            rng=np.random.default_rng(1),
            auto_fit=False,
        ).fit([(0, 70, 70, 70, 70), (50, 70, 70, 70, 70)])
        diag = encoder.diagnostics()
        self.assertEqual(diag["status"], "fit_pca_fallback")
        self.assertEqual(encoder.features((25, 70, 70, 70, 70)).shape, (4,))

    def test_manifold_inverse_candidates_return_raw_policies(self):
        encoder = PCAManifoldEncoder(
            self.problem,
            latent_dim=4,
            fit_pool_size=32,
            rng=np.random.default_rng(2),
        )
        candidates = encoder.inverse_candidates(
            n_anchors=5,
            inverse_pool_size=40,
            inverse_neighbors=2,
            rng=np.random.default_rng(3),
        )
        self.assertGreaterEqual(len(candidates), 5)
        self.assertTrue(all(len(x) == self.problem.d for x in candidates))

    def test_manifold_candidates_use_problem_state_anchors_when_available(self):
        problem = ScalarizedProblem(HighDimStatePolicyRZDT1(d=256, L=100, sigma=0.04))
        encoder = PCAManifoldEncoder(
            problem,
            latent_dim=4,
            fit_pool_size=16,
            rng=np.random.default_rng(7),
        )
        candidates = encoder.state_space_candidates(
            n_anchors=6,
            inverse_pool_size=20,
            inverse_neighbors=2,
            rng=np.random.default_rng(8),
        )
        self.assertGreaterEqual(len(candidates), 6)
        self.assertTrue(all(len(x) == problem.d for x in candidates))
        self.assertEqual(encoder.diagnostics()["last_inverse_mode"], "problem_state_anchor")
        spreads = [problem.base.policy_state(x)[2] for x in candidates]
        self.assertLessEqual(max(spreads), 1e-12)

    def test_explicit_manifold_basis_overrides_problem_default_basis(self):
        problem = ScalarizedProblem(HighDimStatePolicyRZDT1(d=128, L=100, sigma=0.04))
        alg = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                use_state_coupling=True,
                use_state_basis=True,
                state_basis_mode="manifold",
                encoder_kind="pca_manifold",
                encoder_latent_dim=4,
                encoder_fit_pool_size=16,
                seed=9,
            ),
        )
        basis_map = alg.gpr[0].basis_map
        self.assertIsInstance(basis_map, StateCoupledFeatureMap)
        self.assertEqual(basis_map.feature_dim, 4)

    def test_raw_manifold_basis_can_compress_raw_features(self):
        problem = ScalarizedProblem(HighDimStatePolicyRZDT1(d=128, L=100, sigma=0.04))
        encoder = PCAManifoldEncoder(
            problem,
            latent_dim=4,
            fit_pool_size=16,
            rng=np.random.default_rng(11),
        )
        basis = StateCoupledFeatureMap(
            problem,
            encoder,
            mode="raw+manifold",
            raw_basis_dim=16,
            raw_projection_seed=7,
        )
        x = tuple([25] * problem.d)
        f1 = basis.features(x)
        f2 = basis.features(x)
        self.assertEqual(basis.feature_dim, 20)
        self.assertEqual(f1.shape, (20,))
        self.assertTrue(np.all(np.isfinite(f1)))
        np.testing.assert_allclose(f1, f2)

    def test_masked_ssl_record_diagnostics_are_finite(self):
        encoder = MaskedTrajectoryEncoder(problem=None, latent_dim=4).fit(tiny_records())
        feat = encoder.features("p0")
        diag = encoder.diagnostics()
        self.assertEqual(feat.shape, (4,))
        self.assertEqual(diag["encoder"], "ssl_masked")
        self.assertGreaterEqual(diag["masked_baseline_mse"], 0.0)
        self.assertGreaterEqual(diag["masked_reconstruction_mse"], 0.0)

    def test_masked_ssl_records_with_x_project_raw_policy(self):
        encoder = MaskedTrajectoryEncoder(
            self.problem,
            latent_dim=4,
            records_or_policy_pool=tiny_records_with_x(),
        )
        feat = encoder.features((20, 70, 70, 70, 70))
        self.assertEqual(feat.shape, (4,))
        self.assertTrue(np.all(np.isfinite(feat)))
        self.assertEqual(encoder.diagnostics()["n_policies_with_x"], 2)

    def test_contrastive_encoder_separates_risk_regimes(self):
        encoder = ContrastivePolicyEncoder(problem=None, latent_dim=4).fit(tiny_records())
        diag = encoder.diagnostics()
        self.assertEqual(diag["encoder"], "ssl_contrastive")
        self.assertGreaterEqual(diag["n_classes"], 2)
        self.assertGreaterEqual(diag["contrastive_separation"], 0.0)

    def test_next_risk_and_transformer_fallback_features(self):
        next_encoder = NextRiskEncoder(
            self.problem,
            latent_dim=4,
            fit_pool_size=16,
            rng=np.random.default_rng(4),
        )
        transformer = SmallTransformerEncoder(
            self.problem,
            latent_dim=4,
            fit_pool_size=16,
            rng=np.random.default_rng(5),
        )
        x = (25, 70, 70, 70, 70)
        self.assertEqual(next_encoder.features(x).shape, (4,))
        self.assertEqual(transformer.features(x).shape, (4,))
        self.assertIn("torch_status", transformer.diagnostics())

    def test_manifold_risk_decomposition_sums_to_total(self):
        encoder = PCAManifoldEncoder(
            self.problem,
            latent_dim=4,
            fit_pool_size=32,
            rng=np.random.default_rng(6),
        )
        blocks = ManifoldRiskDecomposer(encoder).decompose(
            (25, 70, 70, 70, 70),
            total_variance=0.37,
        )
        self.assertAlmostEqual(
            blocks["total"],
            blocks["tangent"] + blocks["normal"] + blocks["shared"] + blocks["residual"],
        )
        self.assertTrue(all(blocks[key] >= 0.0 for key in (
            "tangent",
            "normal",
            "shared",
            "residual",
        )))


if __name__ == "__main__":
    unittest.main()
