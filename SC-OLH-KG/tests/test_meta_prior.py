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
from performance.benchmark_lodo_meta_prior import meta_source_seed  # noqa: E402
from variance.orthogonal_hvd import OrthogonalHVD  # noqa: E402


class MetaPriorTests(unittest.TestCase):
    def test_source_prior_seed_is_frozen_across_target_seeds(self):
        config = {"meta_source_seed_mode": "frozen"}
        self.assertEqual(meta_source_seed(config, 0), 0)
        self.assertEqual(meta_source_seed(config, 19), 0)
        config["meta_source_seed_mode"] = "per_target"
        self.assertEqual(meta_source_seed(config, 19), 19)

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

        ordered = TransferableSpectralBasis(
            active_dim=4,
            low_frequency_components=10,
            n_neighbors=12,
            orthogonalization="ordered_cholesky",
        ).fit(batches)
        ordered_diag = ordered.diagnostics()
        self.assertEqual(
            ordered_diag["orthogonalization"], "ordered_cholesky")
        self.assertLess(ordered_diag["max_offdiag_gram"], 1e-5)
        self.assertLess(ordered_diag["max_diag_error"], 1e-5)
        np.testing.assert_allclose(
            np.tril(ordered.whitening_, k=-1), 0.0, atol=1e-12)

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
        self.assertEqual(
            features.shape, (prior.spectral_basis.feature_dim,))
        self.assertLessEqual(prior.spectral_basis.feature_dim, 4)
        self.assertEqual(frozen, prior.spectral_basis.fingerprint())

    def test_spectral_hvd_transfers_frozen_aligned_variance_prior(self):
        rng = np.random.default_rng(311)
        sources = [
            (
                "FactorShockStatePolicyRZDT1",
                self._problem("FactorShockStatePolicyRZDT1"),
            ),
            ("InventorySupplyChain", self._problem("InventorySupplyChain")),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=3,
            component_stage="spectral_hvd",
            spectral_active_dim=4,
            spectral_risk_alignment=True,
            seed=311,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=rng,
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl"), prior)
        diagnostics = prior.diagnostics()
        self.assertEqual(
            diagnostics["enabled_components"],
            ["coordinate", "spectral", "hvd"],
        )
        beta = target.cumulative_hvd_prior_beta(1)
        self.assertIsNotNone(beta)
        duplicate_target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl"), prior)
        np.testing.assert_allclose(
            beta,
            duplicate_target.cumulative_hvd_prior_beta(1),
            atol=0.0,
            rtol=0.0,
        )
        provider_status = target.cumulative_risk_provider_status()
        self.assertFalse(provider_status["target_data_used"])
        self.assertEqual(
            provider_status["unlabeled_target_shape_reference_pool_size"],
            256,
        )
        self.assertGreater(
            target.cumulative_hvd_prior_precision(output_index=1), 0.0)
        self.assertGreaterEqual(
            target.cumulative_hvd_prior_upper_scale(output_index=1), 1.0)
        x = target.sample_random(np.random.default_rng(312))
        exposure = target.risk_exposures(x)
        self.assertTrue(np.all(exposure.A >= 0.0))
        self.assertTrue(np.all(exposure.N >= 0.0))
        self.assertAlmostEqual(float(np.sum(exposure.N)), 1.0, places=7)
        self.assertEqual(exposure.meta["target_data_used"], False)
        self.assertEqual(
            len(target.cumulative_risk_features(x)),
            len(beta),
        )

        samples = [
            target.sample_random(np.random.default_rng(320 + index))
            for index in range(8)
        ]
        residuals = np.linspace(0.05, 0.12, len(samples))
        hvd = OrthogonalHVD(mode="factor", n_outputs=2)
        hvd.fit_from_residuals(
            samples,
            residuals,
            output_index=1,
            problem=target,
        )
        hvd_diagnostics = hvd.diagnostics()
        self.assertTrue(hvd_diagnostics["cumulative_prior_used"]["1"])
        self.assertEqual(
            hvd_diagnostics["cumulative_activation_records"]["1"], 5)
        self.assertTrue(hvd_diagnostics["cumulative_active"]["1"])
        self.assertGreater(
            hvd_diagnostics["cumulative_prior_scale"]["1"], 0.0)
        self.assertTrue(
            hvd_diagnostics["cumulative_prior_replication_only"]["1"])
        hvd.update(
            1,
            samples[0],
            y=0.0,
            mu=0.0,
            problem=target,
            replicate_variance=0.002,
        )
        replicated_diagnostics = hvd.diagnostics()
        self.assertEqual(
            replicated_diagnostics["cumulative_prior_scale_source"]["1"],
            "within_solution_replication",
        )
        self.assertEqual(
            replicated_diagnostics["cumulative_prior_target_weight"]["1"], 1)

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

    def test_adaptive_challenger_rejection_restores_exact_stage1_basis(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=8,
            spectral_adaptive_sparsity=True,
            spectral_gate_selection_tolerance=0.0,
            seed=45,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(45),
        )
        self.assertGreater(prior.stage1_spectral_basis.feature_dim, 0)
        self.assertLessEqual(prior.stage1_spectral_basis.feature_dim, 6)
        self.assertEqual(prior.spectral_basis.feature_dim, 6)
        self.assertEqual(prior.spectral_feature_dim, 8)
        self.assertEqual(prior.spectral_always_active_count, 2)

        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        xs = []
        for index in range(10):
            row = [50] * target.d
            row[0] = 5 + 10 * index
            row[1] = int((37 * index + 11) % 100)
            xs.append(tuple(row))

        prior.stage1_spectral_features = lambda problem, x: np.asarray([
            float(x[0]) / 100.0,
            (float(x[0]) / 100.0) ** 2,
        ])
        prior.spectral_features = lambda problem, x: np.asarray([
            float(x[1]) / 100.0,
            np.sin(float(x[1])),
            np.cos(float(x[1])),
            float(x[1] % 7) / 7.0,
            float(x[1] % 5) / 5.0,
            float(x[1] % 3) / 3.0,
            float(x[1] > 50),
            1.0,
        ])
        prior.coordinate_basis_features = lambda problem, x: np.asarray([
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
            observations[x] = [np.asarray([
                latent ** 2,
                0.55 - latent - z_sigma,
            ])]

        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1, ridge=0.1)
        gate._adaptive_loo_predictions = lambda features, values, keys, spec: (
            np.full(len(values), 10.0, dtype=float)
        )
        selected = gate.fit_from_observations(observations, output_index=1)
        diag = gate.diagnostics()

        self.assertEqual(diag["stage1_selected_basis"], "source_spectral")
        self.assertEqual(selected, "source_spectral")
        self.assertTrue(diag["adaptive_rejection_reasons"])
        self.assertIsNone(gate.adaptive_sparsity_spec(observations))
        x = xs[0]
        expected = prior.stage1_spectral_features(target, x)
        actual = gate.features(x)
        np.testing.assert_allclose(actual[:len(expected)], expected)
        np.testing.assert_allclose(actual[len(expected):], 0.0)
        self.assertFalse(np.allclose(
            actual[:len(expected)],
            prior.spectral_features(target, x)[:len(expected)],
        ))

    def test_source_coefficient_shrinkage_is_spectral_constraint_only(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_coefficient_shrinkage=True,
            spectral_shrinkage_strength=1.0,
            spectral_shrinkage_floor=0.05,
            seed=43,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(43),
        )
        diag = prior.diagnostics()
        constraint_prior = diag["spectral_coefficient_prior"]["1"]
        weights = np.asarray(constraint_prior["weight"], dtype=float)
        self.assertEqual(weights.shape, (2,))
        self.assertTrue(np.all(weights >= 0.05))
        self.assertTrue(np.all(weights <= 1.0))
        self.assertAlmostEqual(float(np.max(weights)), 1.0)
        self.assertEqual(constraint_prior["normalization"], "max_one")

        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1)
        beta = np.ones(gate.feature_dim + 1, dtype=float)
        unchanged_beta, unchanged_var = gate.apply_coefficient_prior(beta, 2.0)
        np.testing.assert_allclose(unchanged_beta, beta)
        self.assertEqual(unchanged_var, 2.0)

        gate.selected_basis = "source_spectral"
        shrunk_beta, prior_diag = gate.apply_coefficient_prior(beta, 2.0)
        np.testing.assert_allclose(shrunk_beta, beta)
        self.assertEqual(prior_diag.shape, beta.shape)
        self.assertTrue(np.all(prior_diag[1:3] <= 2.0))
        self.assertTrue(gate.diagnostics()["coefficient_prior_applied"])
        self.assertEqual(
            gate.diagnostics()["coefficient_prior_mode"], "variance_only")

    def test_frequency_bank_is_source_only_and_contains_exact_stage1(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=6,
            spectral_low_frequency_components=8,
            spectral_frequency_adaptation=True,
            spectral_frequency_cutoffs=(3, 8),
            spectral_frequency_ridges=(0.1, 1.0),
            seed=46,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(46),
        )

        self.assertGreater(prior.stage1_spectral_basis.feature_dim, 0)
        self.assertLessEqual(prior.stage1_spectral_basis.feature_dim, 6)
        self.assertEqual(len(prior.spectral_frequency_bank), 4)
        baselines = [
            entry for entry in prior.spectral_frequency_bank
            if entry["is_stage1_baseline"]
        ]
        self.assertEqual(len(baselines), 1)
        self.assertIs(baselines[0]["basis"], prior.stage1_spectral_basis)
        self.assertEqual(baselines[0]["cutoff"], 8)
        self.assertEqual(baselines[0]["ridge"], 1.0)
        diag = prior.diagnostics()["spectral_frequency"]
        self.assertEqual(diag["status"], "fit")
        self.assertFalse(diag["target_data_used"])
        self.assertAlmostEqual(sum(
            entry["source_weight"] for entry in prior.spectral_frequency_bank
        ), 1.0)

    def test_frequency_gate_learns_target_band_and_ridge_without_oracle(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_low_frequency_components=8,
            spectral_frequency_adaptation=True,
            spectral_frequency_cutoffs=(3, 8),
            spectral_frequency_ridges=(0.1, 1.0),
            spectral_frequency_source_penalty=0.0,
            spectral_gate_selection_tolerance=0.0,
            seed=47,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(47),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        challenger = next(
            entry for entry in prior.spectral_frequency_bank
            if not entry["is_stage1_baseline"]
        )
        challenger["source_weight"] = 1.0
        xs = []
        observations = {}
        for index in range(10):
            row = [50] * target.d
            row[0] = 5 + 10 * index
            x = tuple(row)
            xs.append(x)
            value = -0.25 + 0.05 * index
            observations[x] = [np.asarray([value ** 2, value])]

        prior.coordinate_basis_features = lambda problem, x: np.asarray([
            0.0, float(x[0]) / 100.0,
        ])
        prior.stage1_spectral_features = lambda problem, x: np.asarray([
            1.0, float(x[0]) / 100.0,
        ])
        prior.risk_coordinate = lambda problem, x: np.asarray([
            2.0, float(x[0]) / 100.0, 0.0, 0.0,
        ])
        prior.spectral_frequency_features = lambda problem, x, index: np.asarray([
            7.0 if int(index) == int(challenger["index"]) else 9.0,
            float(x[0]) / 100.0,
        ])

        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1, ridge=1.0)

        def fake_loo(features, values, ridge=None):
            marker = float(features[0, 0])
            if np.isclose(marker, 7.0):
                return values.copy()
            if np.isclose(marker, 1.0):
                return values + 0.15
            return values + 0.40

        gate._ridge_loo_predictions = fake_loo
        selected = gate.fit_from_observations(observations, output_index=1)
        diag = gate.diagnostics()

        self.assertEqual(selected, challenger["variant"])
        self.assertEqual(
            gate.selected_parametric_ridge, float(challenger["ridge"]))
        self.assertEqual(diag["frequency_adaptation"]["status"], "selected")
        self.assertTrue(diag["frequency_adaptation"]["target_data_used"])
        self.assertFalse(diag["frequency_adaptation"]["target_oracle_used"])
        self.assertGreater(
            diag["frequency_adaptation"]["stability"]["win_rate"], 0.5)
        self.assertGreater(
            diag["frequency_adaptation"]["boundary_support"]["n_feasible"],
            0,
        )

    def test_frequency_gate_requires_two_sided_boundary_support_and_refits(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_low_frequency_components=8,
            spectral_frequency_adaptation=True,
            spectral_frequency_cutoffs=(3, 8),
            spectral_frequency_ridges=(0.1, 1.0),
            spectral_frequency_source_penalty=0.0,
            spectral_frequency_refit_interval=5,
            spectral_gate_selection_tolerance=0.0,
            seed=470,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(470),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        challenger = next(
            entry for entry in prior.spectral_frequency_bank
            if not entry["is_stage1_baseline"]
        )
        challenger["source_weight"] = 1.0
        observations = {}
        for index in range(10):
            row = [50] * target.d
            row[0] = 5 + 9 * index
            x = tuple(row)
            value = 0.25 + 0.02 * index
            observations[x] = [np.asarray([value ** 2, value])]

        prior.coordinate_basis_features = lambda problem, x: np.asarray([
            0.0, float(x[0]) / 100.0,
        ])
        prior.stage1_spectral_features = lambda problem, x: np.asarray([
            1.0, float(x[0]) / 100.0,
        ])
        prior.risk_coordinate = lambda problem, x: np.asarray([
            2.0, float(x[0]) / 100.0, 0.0, 0.0,
        ])
        prior.spectral_frequency_features = lambda problem, x, index: np.asarray([
            7.0 if int(index) == int(challenger["index"]) else 9.0,
            float(x[0]) / 100.0,
        ])
        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1, ridge=1.0)

        def fake_loo(features, values, ridge=None):
            marker = float(features[0, 0])
            if np.isclose(marker, 7.0):
                return values.copy()
            if np.isclose(marker, 1.0):
                return values + 0.15
            return values + 0.40

        gate._ridge_loo_predictions = fake_loo
        selected = gate.fit_from_observations(observations, output_index=1)
        frequency = gate.diagnostics()["frequency_adaptation"]
        self.assertNotEqual(selected, challenger["variant"])
        self.assertIn(
            "one_sided_boundary_support", frequency["rejection_reasons"])
        self.assertEqual(frequency["boundary_support"]["n_feasible"], 0)
        self.assertFalse(gate.should_refit_from_observations(observations))

        for index in range(10, 15):
            row = [50] * target.d
            row[0] = 60 + 7 * (index - 10)
            x = tuple(row)
            value = -0.20 + 0.01 * (index - 10)
            observations[x] = [np.asarray([value ** 2, value])]
        self.assertTrue(gate.should_refit_from_observations(observations))
        state = gate.runtime_state()
        restored = PilotGatedMetaPriorBasis(prior, target, output_index=1)
        restored.load_runtime_state(state)
        self.assertTrue(restored.should_refit_from_observations(observations))

    def test_frequency_gate_fallback_and_ridge_initialization(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_low_frequency_components=8,
            spectral_frequency_adaptation=True,
            spectral_frequency_cutoffs=(3, 8),
            spectral_frequency_ridges=(0.1, 1.0),
            spectral_frequency_source_penalty=0.0,
            spectral_gate_selection_tolerance=0.0,
            seed=48,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(48),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        observations = {}
        for index in range(10):
            row = [50] * target.d
            row[0] = 5 + 10 * index
            x = tuple(row)
            value = -0.25 + 0.05 * index
            observations[x] = [np.asarray([value ** 2, value])]
        prior.coordinate_basis_features = lambda problem, x: np.asarray([
            0.0, float(x[0]) / 100.0,
        ])
        prior.stage1_spectral_features = lambda problem, x: np.asarray([
            1.0, float(x[0]) / 100.0,
        ])
        prior.risk_coordinate = lambda problem, x: np.asarray([
            2.0, float(x[0]) / 100.0, 0.0, 0.0,
        ])
        prior.spectral_frequency_features = lambda problem, x, index: np.asarray([
            9.0, float(x[0]) / 100.0,
        ])
        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1, ridge=1.0)

        def fake_loo(features, values, ridge=None):
            marker = float(features[0, 0])
            if np.isclose(marker, 1.0):
                return values.copy()
            if np.isclose(marker, 0.0):
                return values + 0.30
            return values + 0.60

        gate._ridge_loo_predictions = fake_loo
        selected = gate.fit_from_observations(observations, output_index=1)
        self.assertEqual(selected, "source_spectral")
        self.assertEqual(gate.selected_parametric_ridge, 0.0)
        expected = prior.stage1_spectral_features(target, next(iter(observations)))
        actual = gate.features(next(iter(observations)))
        np.testing.assert_allclose(actual[:len(expected)], expected)
        np.testing.assert_allclose(actual[len(expected):], 0.0)

        matrix = np.column_stack([
            np.ones(8),
            np.linspace(-1.0, 1.0, 8),
            np.linspace(-1.0, 1.0, 8),
        ])
        values = np.linspace(-2.0, 2.0, 8)
        least_squares = np.linalg.lstsq(matrix, values, rcond=None)[0]
        gate.selected_parametric_ridge = 100.0
        regularized = gate.initial_parametric_coefficients(matrix, values)
        self.assertTrue(np.all(np.isfinite(regularized)))
        self.assertLess(
            float(np.linalg.norm(regularized[1:])),
            float(np.linalg.norm(least_squares[1:])),
        )

        near_null = np.column_stack([
            np.ones(10),
            np.linspace(-1.0, 1.0, 10),
            np.linspace(-1.0, 1.0, 10) + 1e-9 * np.arange(10),
        ])
        near_null_values = np.linspace(-0.5, 0.5, 10)
        gate.selected_parametric_ridge = 0.0
        stable = gate.initial_parametric_coefficients(
            near_null, near_null_values)
        self.assertTrue(np.all(np.isfinite(stable)))
        self.assertLess(float(np.max(np.abs(stable))), 10.0)
        self.assertEqual(
            gate.gate_diagnostics["initial_fit_solver"],
            "truncated_svd",
        )
        self.assertEqual(gate.gate_diagnostics["initial_fit_rcond"], 1e-3)

    def test_additive_group_bank_is_source_only_orthogonal_anova(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_additive_adaptation=True,
            spectral_additive_max_groups=8,
            seed=49,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(49),
        )
        diag = prior.diagnostics()["spectral_additive"]
        self.assertEqual(diag["status"], "fit")
        self.assertFalse(diag["target_data_used"])
        self.assertLessEqual(diag["feature_dim"], 8)
        self.assertTrue(all(
            name.startswith(("main:", "interaction:"))
            for name in diag["group_names"]
        ))
        self.assertLessEqual(diag["max_offdiag_gram"], 1e-8)
        self.assertLessEqual(diag["max_diag_error"], 1e-8)
        self.assertLessEqual(diag["max_stage1_cross_gram"], 1e-8)
        self.assertAlmostEqual(sum(diag["source_weights"]), 1.0)
        self.assertTrue(diag["strong_heredity"])

    def test_additive_gate_learns_groups_from_target_pilot_only(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_additive_adaptation=True,
            spectral_additive_max_groups=4,
            spectral_additive_target_max_groups=2,
            spectral_additive_source_penalty=0.0,
            spectral_additive_complexity_penalty=0.0,
            spectral_gate_selection_tolerance=0.0,
            seed=50,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(50),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        observations = {}
        for index in range(10):
            row = [50] * target.d
            row[0] = 5 + 10 * index
            x = tuple(row)
            value = -0.25 + 0.05 * index
            observations[x] = [np.asarray([value ** 2, value])]
        prior.coordinate_basis_features = lambda problem, x: np.asarray([
            0.0, float(x[0]) / 100.0,
        ])
        prior.stage1_spectral_features = lambda problem, x: np.asarray([
            1.0, float(x[0]) / 100.0,
        ])
        prior.risk_coordinate = lambda problem, x: np.asarray([
            2.0, float(x[0]) / 100.0, 0.0, 0.0,
        ])
        prior.spectral_additive_features = lambda problem, x, indices: np.asarray([
            7.0 if int(index) == 0 else 9.0 for index in indices
        ])
        prior.spectral_additive_bank.support_saturation = (
            lambda psi, index: False)

        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1, ridge=1.0)

        def fake_loo(features, values, ridge=None):
            if features.shape[1] > 2 and np.isclose(features[0, -1], 7.0):
                return values.copy()
            if np.isclose(features[0, 0], 1.0):
                return values + 0.15
            return values + 0.40

        gate._ridge_loo_predictions = fake_loo
        selected = gate.fit_from_observations(observations, output_index=1)
        diag = gate.diagnostics()["additive_adaptation"]
        self.assertEqual(selected, "source_additive")
        self.assertEqual(gate.selected_additive_groups, [0])
        self.assertEqual(diag["status"], "selected")
        self.assertEqual(diag["selected_group_indices"], [0])
        self.assertTrue(diag["target_data_used"])
        self.assertFalse(diag["target_oracle_used"])
        self.assertEqual(gate.selected_parametric_ridge, 1.0)

        prior.spectral_additive_bank.source_weight = lambda index: 0.0
        guarded = PilotGatedMetaPriorBasis(
            prior, target, output_index=1, ridge=1.0)
        guarded._ridge_loo_predictions = fake_loo
        guarded_selected = guarded.fit_from_observations(
            observations, output_index=1)
        guarded_diag = guarded.diagnostics()["additive_adaptation"]
        self.assertNotEqual(guarded_selected, "source_additive")
        self.assertIn(
            "weak_source_domain_support",
            guarded_diag["selection_trace"][0]["rejection_reasons"],
        )

    def test_adaptation_stages_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "separate stages"):
            LearnedMetaPrior(
                spectral_frequency_adaptation=True,
                spectral_additive_adaptation=True,
            )

    def test_frequency_and_additive_flags_preserve_stage1_fallback(self):
        sources = [
            ("InventorySupplyChain", self._problem("InventorySupplyChain")),
            ("QueueResourceControl", self._problem("QueueResourceControl")),
        ]

        def fit(**kwargs):
            return LearnedMetaPrior(
                local_dim=2,
                shared_dim=2,
                component_stage="spectral",
                spectral_active_dim=4,
                seed=510,
                **kwargs,
            ).fit_from_source_problems(
                sources,
                n_records_per_domain=16,
                rng=np.random.default_rng(511),
            )

        control = fit()
        frequency = fit(spectral_frequency_adaptation=True)
        additive = fit(spectral_additive_adaptation=True)
        self.assertEqual(
            frequency.stage1_spectral_basis.fingerprint(),
            control.stage1_spectral_basis.fingerprint(),
        )
        self.assertEqual(
            additive.stage1_spectral_basis.fingerprint(),
            control.stage1_spectral_basis.fingerprint(),
        )
        self.assertEqual(
            frequency.stage1_spectral_basis.feature_dim,
            control.stage1_spectral_basis.feature_dim,
        )
        self.assertEqual(
            additive.stage1_spectral_basis.feature_dim,
            control.stage1_spectral_basis.feature_dim,
        )

    def test_additive_gate_refit_schedule_and_runtime_state_roundtrip(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=2,
            spectral_additive_adaptation=True,
            spectral_additive_refit_interval=5,
            seed=51,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(51),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        rng = np.random.default_rng(52)
        observations = {}
        while len(observations) < 10:
            x = target.sample_random(rng)
            observations[x] = [target.simulate(x, rng)]
        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1)
        gate.fit_from_observations(observations, output_index=1)
        self.assertFalse(gate.should_refit_from_observations(observations))
        while len(observations) < 15:
            x = target.sample_random(rng)
            observations[x] = [target.simulate(x, rng)]
        self.assertTrue(gate.should_refit_from_observations(observations))
        gate.fit_from_observations(observations, output_index=1)
        self.assertFalse(gate.should_refit_from_observations(observations))

        state = gate.runtime_state()
        restored = PilotGatedMetaPriorBasis(prior, target, output_index=1)
        restored.load_runtime_state(state)
        self.assertEqual(restored.runtime_state(), state)
        x = next(iter(observations))
        np.testing.assert_allclose(restored.features(x), gate.features(x))

    def test_adaptive_coefficient_prior_transfers_hyperparameters_not_weights(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=6,
            spectral_adaptive_sparsity=True,
            seed=44,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(44),
        )
        source = prior.diagnostics()["spectral_coefficient_prior"]["1"]
        calibration = prior.diagnostics()["spectral_adaptive_calibration"]
        self.assertEqual(calibration["status"], "fit")
        self.assertFalse(calibration["target_data_used"])
        self.assertIn(
            str(calibration["selected_spike_ratio"]),
            calibration["candidate_scores"],
        )
        prior_pip = np.asarray(source["prior_pip"], dtype=float)
        slab_scale = np.asarray(source["slab_scale"], dtype=float)
        self.assertEqual(
            prior_pip.shape, (prior.spectral_feature_dim,))
        self.assertEqual(
            slab_scale.shape, (prior.spectral_feature_dim,))
        self.assertLessEqual(prior.spectral_feature_dim, 6)
        self.assertTrue(np.all((prior_pip >= 0.05) & (prior_pip <= 0.95)))
        self.assertTrue(np.all(slab_scale > 0.0))

        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        objective = PilotGatedMetaPriorBasis(prior, target, output_index=0)
        objective.selected_basis = "source_spectral"
        self.assertIsNone(objective.adaptive_sparsity_spec())

        constraint = PilotGatedMetaPriorBasis(prior, target, output_index=1)
        constraint.selected_basis = "adaptive_spectral"
        constraint._adaptive_allowed_mask = np.ones(
            prior.spectral_feature_dim, dtype=bool)
        spec = constraint.adaptive_sparsity_spec()
        self.assertIsNotNone(spec)
        np.testing.assert_allclose(spec["source_pip"], prior_pip)
        self.assertTrue(
            constraint.diagnostics()["adaptive_sparsity"][
                "source_hyperprior_only"])

    def test_disabled_alignment_admission_is_exact_stage1_gate(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=4,
            spectral_risk_alignment=True,
            spectral_alignment_source_episodes=2,
            spectral_alignment_admission=False,
            teacher_records_per_domain=8,
            teacher_pool_size=128,
            seed=518,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
                ("StatePolicyRZDT1", self._problem("StatePolicyRZDT1")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(519),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        rng = np.random.default_rng(520)
        observations = {}
        while len(observations) < 10:
            x = target.sample_random(rng)
            observations[x] = [target.simulate(x, rng)]
        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1)
        selected = gate.fit_from_observations(observations, output_index=1)
        diagnostics = gate.diagnostics()
        self.assertIn(selected, {"coordinate", "source_spectral"})
        self.assertEqual(diagnostics["risk_alignment"]["status"], "not_requested")
        self.assertNotIn(
            "frozen_risk_aligned_coordinate", diagnostics["eligible_bases"])
        self.assertNotIn("risk_aligned_coordinate", diagnostics["eligible_bases"])
        self.assertFalse(gate.should_refit_from_observations(observations))

    def test_rejected_sequential_alignment_keeps_frozen_stage1_guard(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral",
            spectral_active_dim=4,
            spectral_risk_alignment=True,
            spectral_alignment_target_min_gain=100.0,
            spectral_alignment_refit_interval=5,
            seed=521,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
                ("StatePolicyRZDT1", self._problem("StatePolicyRZDT1")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(522),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        rng = np.random.default_rng(523)
        observations = {}
        while len(observations) < 10:
            x = target.sample_random(rng)
            observations[x] = [target.simulate(x, rng)]
        gate = PilotGatedMetaPriorBasis(prior, target, output_index=1)
        first = gate.fit_from_observations(observations, output_index=1)
        first_guard = gate.certification_guard()
        locked = gate.runtime_state()["locked_alignment_stage1_basis"]
        self.assertEqual(first, locked)

        while len(observations) < 15:
            x = target.sample_random(rng)
            observations[x] = [target.simulate(x, rng)]
        second = gate.fit_from_observations(observations, output_index=1)
        diagnostics = gate.diagnostics()
        self.assertEqual(second, locked)
        self.assertEqual(diagnostics["stage1_selected_basis"], locked)
        self.assertAlmostEqual(gate.certification_guard(), first_guard)
        self.assertTrue(
            diagnostics["risk_alignment"]["stage1_basis_locked"])


if __name__ == "__main__":
    unittest.main()
