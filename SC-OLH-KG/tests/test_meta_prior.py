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
    PilotGatedMetaPriorBasis,
)
from representation.transferable_spectral import (  # noqa: E402
    SourceDomainBatch,
    TransferableSpectralBasis,
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
            universal_shape_count=8,
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
        self.assertGreater(diag["n_profile_templates"], 0)
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
        universal = prior.universal_shape_candidates(
            target,
            n=6,
            rng=np.random.default_rng(10),
        )
        self.assertGreaterEqual(len(universal), 6)
        self.assertTrue(all(len(x) == target.d for x in universal))
        profiles = prior.profile_template_candidates(
            target,
            n=4,
            rng=np.random.default_rng(12),
        )
        self.assertGreater(len(profiles), 0)
        self.assertTrue(all(len(x) == target.d for x in profiles))

    def test_teacher_distillation_uses_source_hooks_without_exposing_target_hooks(self):
        rng = np.random.default_rng(11)
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem("FactorShockStatePolicyRZDT1")),
            ("InventorySupplyChain", self._problem("InventorySupplyChain")),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            anchor_count=8,
            teacher_records_per_domain=6,
            teacher_weight=4.0,
            teacher_pool_size=64,
            anchor_sampling_temperature=0.5,
            universal_shape_count=12,
            seed=11,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=6,
            rng=rng,
        )
        diag = prior.diagnostics()
        training = diag["training"]
        self.assertEqual(training["teacher_record_count"], 12)
        self.assertEqual(
            training["record_origins"]["source_domain_tuned_teacher"],
            12,
        )
        target_base = self._problem("QueueResourceControl")
        self.assertGreater(len(target_base.recommendation_refinement_candidates()), 0)
        target = MetaPriorProblemAdapter(target_base, prior)
        self.assertTrue(target.admissibility_audit()["admissible_mainline"])
        self.assertFalse(hasattr(AdmissibleProblemAdapter(target_base), "risk_exposures"))
        candidates = target.recommendation_refinement_candidates()
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), target.refinement_count)

    def test_source_invariant_spectral_basis_is_orthogonal_and_frozen(self):
        batches = []
        for domain_idx in range(3):
            rng = np.random.default_rng(100 + domain_idx)
            psi = rng.normal(size=(96, 5))
            shared = np.sin(np.pi * np.tanh(0.5 * psi[:, 0]))
            nuisance = 0.08 * rng.normal(size=96)
            signals = np.column_stack([
                (1.0 + 0.1 * domain_idx) * shared + nuisance,
                -0.8 * shared + 0.15 * np.tanh(0.5 * psi[:, 1]) + nuisance,
            ])
            batches.append(SourceDomainBatch(
                domain=f"source{domain_idx}",
                psi=psi,
                signals=signals,
            ))
        basis = TransferableSpectralBasis(
            active_dim=4,
            low_frequency_components=10,
            n_neighbors=12,
        ).fit(batches)
        diag = basis.diagnostics()
        self.assertEqual(diag["status"], "fit")
        self.assertLess(diag["max_offdiag_gram"], 1e-5)
        self.assertLess(diag["max_diag_error"], 1e-5)
        self.assertTrue(any("psi0" in name for name in diag["selected_names"]))
        fingerprint = basis.fingerprint()
        features = basis.transform(np.zeros(5, dtype=float))
        self.assertEqual(features.shape, (4,))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertEqual(fingerprint, basis.fingerprint())

    def test_spectral_stage_does_not_enable_later_meta_components(self):
        rng = np.random.default_rng(31)
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem("FactorShockStatePolicyRZDT1")),
            ("InventorySupplyChain", self._problem("InventorySupplyChain")),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=4,
            seed=31,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=12,
            rng=rng,
        )
        target = MetaPriorProblemAdapter(self._problem("QueueResourceControl"), prior)
        diag = prior.diagnostics()
        self.assertEqual(diag["enabled_components"], ["coordinate", "spectral"])
        self.assertIsNotNone(diag["spectral_basis"])
        self.assertIsNone(target.cumulative_hvd_prior_beta(1))
        self.assertIsNone(target.source_mean_prior_predict_many(
            [target.sample_random(np.random.default_rng(32))],
            output_index=1,
        ))
        self.assertEqual(target.state_anchor_points(n=4), [])
        x = target.sample_random(np.random.default_rng(33))
        frozen = prior.spectral_basis.fingerprint()
        features = target.surrogate_basis_map().features(x)
        self.assertEqual(features.shape, (4,))
        self.assertEqual(frozen, prior.spectral_basis.fingerprint())

        basis = target.gpr_basis_map(output_index=1)
        self.assertIs(target.surrogate_basis_map(), basis)
        rng_gate = np.random.default_rng(34)
        observations = {}
        for _ in range(10):
            x_gate = target.sample_random(rng_gate)
            spectral = prior.spectral_features(target, x_gate)
            observations[x_gate] = [np.array([0.0, float(spectral[0])])]
        selected = basis.fit_from_observations(observations, output_index=1)
        self.assertIn(selected, {"coordinate", "fixed_psi", "source_spectral"})
        self.assertEqual(basis.features(next(iter(observations))).shape, (basis.feature_dim,))
        self.assertEqual(basis.diagnostics()["status"], "fit")

    def test_decision_aware_gate_selects_boundary_predictive_spectral_basis(self):
        rng = np.random.default_rng(41)
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_gate_selection_tolerance=0.0,
            seed=41,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=12,
            rng=rng,
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        xs = []
        for index in range(10):
            row = [50] * target.d
            row[0] = 5 + 10 * index
            row[1] = int((37 * index + 11) % 100)
            xs.append(tuple(row))

        prior.spectral_features = lambda problem, x: np.asarray([
            float(x[0]) / 100.0,
            (float(x[0]) / 100.0) ** 2,
        ])
        prior.risk_coordinate = lambda problem, x: np.asarray([
            float(x[1]) / 100.0,
            (float(x[1]) / 100.0) ** 2,
            0.0,
            0.0,
        ])
        prior.coordinate_basis_features = lambda problem, x: np.asarray([
            float(x[1]) / 100.0,
            np.sin(float(x[1])),
        ])
        z_sigma = 1.6448536269514722 * float(target.sigma_level)
        observations = {}
        for x in xs:
            latent = float(x[0]) / 100.0
            constraint_mean = 0.55 - latent - z_sigma
            observations[x] = [np.asarray([latent ** 2, constraint_mean])]

        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1, ridge=0.1)
        selected = gate.fit_from_observations(observations, output_index=1)
        diag = gate.diagnostics()
        self.assertEqual(selected, "source_spectral")
        self.assertEqual(diag["selection_metric"], "decision_aware_loo")
        self.assertLess(
            diag["decision_score"]["source_spectral"],
            diag["decision_score"]["fixed_psi"],
        )
        self.assertLess(
            diag["decision_score"]["source_spectral"],
            diag["decision_score"]["coordinate"],
        )
        self.assertIn(
            "dangerous_underprediction",
            diag["decision_components"]["source_spectral"],
        )

        objective_gate = PilotGatedMetaPriorBasis(
            prior, target, output_index=0, ridge=0.1)
        objective_selected = objective_gate.fit_from_observations(
            observations, output_index=0)
        self.assertEqual(objective_selected, "coordinate")
        self.assertEqual(
            objective_gate.diagnostics()["eligible_bases"], ["coordinate"])
        self.assertEqual(
            objective_gate.diagnostics()["gate_scope"],
            "constraint_boundary_only",
        )

    def test_decision_aware_gate_falls_back_to_coordinate_baseline(self):
        rng = np.random.default_rng(42)
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_gate_selection_tolerance=0.0,
            seed=42,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=12,
            rng=rng,
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        xs = []
        for index in range(10):
            row = [50] * target.d
            row[0] = 5 + 10 * index
            row[1] = int((37 * index + 11) % 100)
            xs.append(tuple(row))

        prior.coordinate_basis_features = lambda problem, x: np.asarray([
            float(x[0]) / 100.0,
            (float(x[0]) / 100.0) ** 2,
        ])
        prior.spectral_features = lambda problem, x: np.asarray([
            float(x[1]) / 100.0,
            np.sin(float(x[1])),
        ])
        prior.risk_coordinate = lambda problem, x: np.asarray([
            float(x[1]) / 100.0,
            (float(x[1]) / 100.0) ** 2,
            0.0,
            0.0,
        ])
        z_sigma = 1.6448536269514722 * float(target.sigma_level)
        observations = {}
        for x in xs:
            latent = float(x[0]) / 100.0
            observations[x] = [np.asarray([latent ** 2, 0.55 - latent - z_sigma])]

        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1, ridge=0.1)
        selected = gate.fit_from_observations(observations, output_index=1)
        diag = gate.diagnostics()
        self.assertEqual(selected, "coordinate")
        self.assertEqual(diag["eligible_bases"], ["coordinate", "source_spectral"])
        self.assertLessEqual(
            diag["decision_score"]["coordinate"],
            diag["decision_score"]["source_spectral"],
        )


if __name__ == "__main__":
    unittest.main()
