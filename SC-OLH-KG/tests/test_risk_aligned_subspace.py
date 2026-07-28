import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.risk_aligned_subspace import (  # noqa: E402
    BoundaryAlignedRiskSubspaces,
)
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)
from representation.additive_groups import (  # noqa: E402
    TransferableAdditiveGroupBank,
)
from representation.transferable_spectral import SourceDomainBatch  # noqa: E402


def _rotation(seed, dim):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


class BoundaryAlignedRiskSubspaceTests(unittest.TestCase):
    @staticmethod
    def _problem(name, d=8):
        return ScalarizedProblem(make_problem(name, d=d, L=100, sigma=0.04))

    def _batches(self, n=240, dim=4):
        batches = []
        for domain_index in range(4):
            rng = np.random.default_rng(100 + domain_index)
            margin = rng.uniform(-1.8, 1.8, size=n)
            latent = np.column_stack([
                margin,
                0.20 * margin ** 2 + 0.08 * rng.normal(size=n),
                0.08 * rng.normal(size=n),
                0.08 * rng.normal(size=n),
            ])[:, :dim]
            rotation = _rotation(900 + domain_index, dim)
            psi = latent @ rotation.T
            objective = 0.3 * latent[:, 1] - 0.2 * margin
            signals = np.column_stack([
                objective,
                margin,
                np.tanh(margin),
                objective + 3.0 * np.maximum(margin, 0.0),
            ])
            batches.append(SourceDomainBatch(
                domain=f"source{domain_index}",
                psi=psi,
                signals=signals,
                sample_weight=1.0 + np.exp(-0.5 * margin ** 2),
            ))
        return batches

    def test_source_procrustes_reduces_conditional_prototype_discrepancy(self):
        model = BoundaryAlignedRiskSubspaces(
            active_dim=4,
            subspace_dim=2,
            domain_penalty=0.75,
        ).fit(self._batches())
        diag = model.diagnostics()
        self.assertEqual(diag["status"], "fit")
        self.assertLess(
            diag["prototype_discrepancy_after"],
            diag["prototype_discrepancy_before"],
        )
        self.assertGreater(diag["prototype_alignment_gain"], 0.25)
        self.assertLess(diag["projector_idempotence_error"], 1e-8)
        self.assertEqual(model.feature_dim, 10)
        self.assertEqual(len(diag["expert_domains"]), 4)
        self.assertAlmostEqual(sum(diag["expert_prior_weights"]), 1.0)
        self.assertGreaterEqual(diag["source_residual_guard"], 0.0)

    def test_features_depend_on_projectors_not_subspace_axis_rotation(self):
        model = BoundaryAlignedRiskSubspaces(
            active_dim=4,
            subspace_dim=2,
        ).fit(self._batches())
        rng = np.random.default_rng(88)
        x = rng.normal(size=4)
        baseline = model.transform(x)
        original_projectors = [value.copy() for value in model.projectors_]
        original_axes = [value.copy() for value in model.boundary_axes_]
        for index, basis in enumerate(model.subspace_bases_):
            q = _rotation(300 + index, basis.shape[1])
            rotated = basis @ q
            model.projectors_[index] = rotated @ rotated.T
        transformed = model.transform(x)
        np.testing.assert_allclose(transformed, baseline, atol=1e-10)
        for actual, expected in zip(model.projectors_, original_projectors):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
        for actual, expected in zip(model.boundary_axes_, original_axes):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_target_pilot_adapter_is_orthogonal_and_improves_alignment(self):
        model = BoundaryAlignedRiskSubspaces(
            active_dim=4,
            subspace_dim=2,
            target_adapter_ridge=0.25,
            target_min_gain=0.01,
        ).fit(self._batches())
        rng = np.random.default_rng(991)
        margin = np.linspace(-1.7, 1.7, 48)
        latent = np.column_stack([
            margin,
            0.20 * margin ** 2 + 0.02 * rng.normal(size=len(margin)),
            0.02 * rng.normal(size=len(margin)),
            0.02 * rng.normal(size=len(margin)),
        ])
        target_rotation = _rotation(444, 4)
        psi = latent @ target_rotation.T
        adapter = model.fit_target_adapter(psi, margin)
        diag = adapter.diagnostics
        np.testing.assert_allclose(
            adapter.matrix.T @ adapter.matrix,
            np.eye(4),
            atol=1e-8,
        )
        self.assertLessEqual(diag["aligned_loss"], diag["identity_loss"])
        self.assertGreater(diag["alignment_gain"], 0.0)
        self.assertEqual(diag["target_oracle_used"], False)
        aligned = model.transform(psi, adapter=adapter)
        self.assertEqual(aligned.shape, (len(psi), model.feature_dim))
        self.assertTrue(np.all(np.isfinite(aligned)))

    def test_source_batch_transform_is_frozen_and_finite(self):
        batches = self._batches(n=64)
        model = BoundaryAlignedRiskSubspaces(active_dim=3).fit(batches)
        fingerprint = model.fingerprint()
        transformed = model.transform_batches(batches)
        self.assertEqual(len(transformed), len(batches))
        for source, target in zip(batches, transformed):
            self.assertEqual(target.domain, source.domain)
            self.assertEqual(target.psi.shape, (len(source.psi), model.feature_dim))
            self.assertTrue(np.all(np.isfinite(target.psi)))
        self.assertEqual(fingerprint, model.fingerprint())

    def test_aligned_interactions_obey_strong_heredity(self):
        batches = self._batches(n=128)
        alignment = BoundaryAlignedRiskSubspaces(active_dim=4).fit(batches)
        aligned_batches = alignment.transform_batches(batches)
        bank = TransferableAdditiveGroupBank(
            max_groups=8,
            max_interactions=2,
            strong_heredity=True,
        ).fit(aligned_batches)
        names = set(bank.diagnostics()["group_names"])
        for name in names:
            if not name.startswith("interaction:"):
                continue
            _, left, right = name.split(":", 2)
            self.assertIn(f"main:{left}", names)
            self.assertIn(f"main:{right}", names)
        self.assertTrue(bank.diagnostics()["strong_heredity"])

    def test_meta_prior_gate_uses_only_target_pilot_for_alignment(self):
        source_names = [
            "FactorShockStatePolicyRZDT1",
            "InventorySupplyChain",
            "StatePolicyRZDT1",
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=4,
            spectral_risk_alignment=True,
            spectral_alignment_active_dim=4,
            spectral_alignment_subspace_dim=2,
            seed=515,
        ).fit_from_source_problems(
            [(name, self._problem(name)) for name in source_names],
            n_records_per_domain=16,
            rng=np.random.default_rng(516),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl"), prior)
        rng = np.random.default_rng(517)
        observations = {}
        for _ in range(10):
            x = target.sample_random(rng)
            observations.setdefault(x, []).append(target.simulate(x, rng))
        basis = target.gpr_basis_map(output_index=1)
        selected = basis.fit_from_observations(observations, output_index=1)
        self.assertIn(selected, {
            "coordinate", "source_spectral", "risk_aligned_coordinate",
            "risk_aligned_spectral",
        })
        diag = basis.diagnostics()
        self.assertIn("risk_alignment", diag)
        self.assertFalse(diag["risk_alignment"].get("target_oracle_used", True))
        correction = diag["risk_alignment"].get("risk_correction", {})
        self.assertFalse(correction.get("target_variance_oracle_used", True))
        support = diag["risk_alignment"]["boundary_support"]
        if support["n_feasible"] == 0 or support["n_infeasible"] == 0:
            self.assertIn(
                "one_sided_boundary_support",
                diag["risk_alignment"]["rejection_reasons"],
            )
            self.assertNotIn(selected, {
                "risk_aligned_coordinate", "risk_aligned_spectral",
            })
        nested = diag["risk_alignment"]["nested_loo"]
        self.assertEqual(set(nested), {
            "risk_aligned_coordinate", "risk_aligned_spectral",
        })
        for value in nested.values():
            self.assertEqual(value["method"], "nested_alignment_loo")
            self.assertEqual(value["n_folds"], len(observations))
            self.assertTrue(value["adapter_cache_reused_across_variants"])
            self.assertTrue(all(
                row["heldout"] < len(observations) for row in value["folds"]
            ))
        self.assertEqual(
            prior.diagnostics()["risk_alignment"]["status"], "fit")
        state = basis.runtime_state()
        clone = target.gpr_basis_map(output_index=0)
        clone.load_runtime_state(state)
        np.testing.assert_allclose(
            clone.runtime_state()["target_risk_alignment"]["matrix"],
            state["target_risk_alignment"]["matrix"],
        )

    def test_aligned_frequency_bank_is_an_exact_fallback_family(self):
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=4,
            spectral_risk_alignment=True,
            spectral_frequency_adaptation=True,
            spectral_frequency_cutoffs=(3, 5),
            spectral_low_frequency_components=5,
            spectral_frequency_ridges=(0.01, 1.0),
            seed=711,
        ).fit_from_source_problems(
            [
                (name, self._problem(name))
                for name in (
                    "FactorShockStatePolicyRZDT1",
                    "InventorySupplyChain",
                    "StatePolicyRZDT1",
                )
            ],
            n_records_per_domain=12,
            rng=np.random.default_rng(712),
        )
        entries = prior.risk_aligned_frequency_bank
        self.assertGreater(len(entries), 0)
        baselines = [entry for entry in entries if entry["is_stage1_baseline"]]
        self.assertEqual(len(baselines), 1)
        self.assertIs(
            baselines[0]["basis"], prior.risk_aligned_spectral_basis)
        self.assertTrue(all(
            entry["base_variant"] == "risk_aligned_spectral"
            for entry in entries
        ))


if __name__ == "__main__":
    unittest.main()
