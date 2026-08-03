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
    ObservableConstraintMeanBasis,
    PilotGatedMetaPriorBasis,
)
from representation.transferable_spectral import (  # noqa: E402
    SourceDomainBatch,
    TransferableSpectralBasis,
)
from performance.benchmark_lodo_meta_prior import meta_source_seed  # noqa: E402
from variance.orthogonal_hvd import OrthogonalHVD  # noqa: E402
from core.cumulative_risk import (  # noqa: E402
    RiskExposure,
    canonical_risk_descriptor,
)
from core.gpr import ParametricGPR  # noqa: E402


class MetaPriorTests(unittest.TestCase):
    def test_canonical_risk_descriptor_is_permutation_invariant(self):
        first = canonical_risk_descriptor(RiskExposure(
            [0.2, 0.8, 0.4], [0.1, 0.7]))
        second = canonical_risk_descriptor(RiskExposure(
            [0.4, 0.2, 0.8], [0.7, 0.1]))
        self.assertEqual(first.shape, (42,))
        np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-12)

    def test_hierarchical_boundary_descriptor_modes_are_auditable(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1")),
            ("InventorySupplyChain", self._problem("InventorySupplyChain")),
        ]
        prior = LearnedMetaPrior(seed=229).fit_from_source_problems(
            sources,
            n_records_per_domain=8,
            rng=np.random.default_rng(229),
            hierarchical_boundary_config={
                "descriptor_mode": "provider_risk",
                "coordinate": "boundary_latent",
                "geometry": "linear_monotone",
                "rank": 2,
            },
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl"), prior)
        x = target.sample_random(np.random.default_rng(230))
        descriptor = target.hierarchical_boundary_descriptor(x)
        self.assertEqual(descriptor.shape, (42,))
        self.assertEqual(
            prior.hierarchical_boundary_diagnostics["descriptor_mode"],
            "provider_risk",
        )
        audit = target.admissibility_audit()
        self.assertTrue(audit["tcb_target_structural_provider_used"])
        self.assertFalse(audit["admissible_strict_lodo"])
        self.assertTrue(audit["admissible_structure_aware"])
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
            source_observation_mode="replicated",
            source_observation_replicates=2,
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

        high_dim_target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=1000), prior)
        atlas_small = prior.dimension_equivariant_initial_candidates(
            target,
            n=6,
            rng=np.random.default_rng(41),
        )
        atlas_large = prior.dimension_equivariant_initial_candidates(
            high_dim_target,
            n=6,
            rng=np.random.default_rng(41),
        )
        self.assertEqual(len(atlas_small), 6)
        self.assertEqual(len(atlas_large), 6)
        self.assertTrue(all(len(x) == target.d for x in atlas_small))
        self.assertTrue(all(len(x) == 1000 for x in atlas_large))
        self.assertEqual(len(set(atlas_large)), 6)
        self.assertEqual(
            prior.dimension_equivariant_proposal_diagnostics[
                "target_policy_dimension"],
            1000,
        )

    def test_ordered_cumulative_expert_uses_source_learned_low_rank_coordinate(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            anchor_count=5,
            ordered_cumulative_exposure=True,
            ordered_exposure_max_frequency=6,
            ordered_exposure_active_dim=2,
            seed=29,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=10,
            rng=np.random.default_rng(29),
        )
        ordered_diag = prior.diagnostics()["ordered_cumulative_exposure"]
        self.assertEqual(ordered_diag["status"], "fit")
        self.assertEqual(len(ordered_diag["selected_frequencies"]), 2)
        self.assertFalse(ordered_diag["target_data_used"])

        target = MetaPriorProblemAdapter(
            self._problem("InventorySupplyChain", d=12), prior)
        x = target.sample_random(np.random.default_rng(30))
        view = target.task_expert_problem_view("ordered_cumulative")
        exposure = view.risk_exposures(x)
        self.assertEqual(exposure.A.shape, (4,))
        self.assertEqual(exposure.N.shape, (2,))
        self.assertTrue(np.all(np.isfinite(exposure.A)))
        self.assertAlmostEqual(float(np.sum(exposure.N)), 1.0, places=7)
        self.assertEqual(
            len(view.cumulative_risk_features(x)),
            1 + 4 + 2 * (2 + 1) // 2 + 2,
        )
        basis = target.task_expert_basis_map("ordered_cumulative")
        self.assertEqual(basis.feature_dim, 4 + 4 * (4 + 1) // 2 + 2)
        self.assertIsNone(view.cumulative_hvd_prior_beta(1))
        self.assertIn(
            "ordered_cumulative",
            [spec["name"] for spec in target.task_posterior_expert_specs()],
        )

    def test_ordered_sparse_expert_replaces_local_without_growing_ensemble(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            anchor_count=5,
            ordered_cumulative_exposure=True,
            ordered_exposure_max_frequency=6,
            ordered_exposure_active_dim=2,
            ordered_exposure_basis_mode="diagonal_quadratic",
            ordered_exposure_adaptive_sparsity=True,
            ordered_exposure_replace_local_kernel=True,
            seed=31,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=10,
            rng=np.random.default_rng(31),
        )
        target = MetaPriorProblemAdapter(
            self._problem("InventorySupplyChain", d=12), prior)
        specs = target.task_posterior_expert_specs(include_local_kernel=True)
        names = [spec["name"] for spec in specs]
        self.assertIn("ordered_cumulative", names)
        self.assertNotIn("local_risk_kernel", names)

        basis = target.task_expert_basis_map(
            "ordered_cumulative", output_index=1)
        self.assertEqual(basis.feature_dim, 4 + 4 + 2)
        sparse = basis.adaptive_sparsity_spec()
        self.assertEqual(sparse["dictionary_dim"], basis.feature_dim)
        self.assertEqual(sparse["always_active_count"], 4)
        self.assertEqual(len(sparse["source_pip"]), basis.feature_dim)
        ordered_diag = prior.diagnostics()["ordered_cumulative_exposure"]
        self.assertEqual(ordered_diag["basis_mode"], "diagonal_quadratic")
        self.assertTrue(ordered_diag["adaptive_sparsity"])

    def test_ordered_orthogonality_ablation_preserves_the_function_span(self):
        source_names = (
            "FactorShockStatePolicyRZDT1",
            "QueueResourceControl",
        )

        def fit(orthogonal):
            sources = [
                (name, self._problem(name, d=12)) for name in source_names
            ]
            return LearnedMetaPrior(
                local_dim=3,
                shared_dim=2,
                ordered_cumulative_exposure=True,
                ordered_exposure_max_frequency=6,
                ordered_exposure_active_dim=2,
                ordered_exposure_orthogonal_coordinates=orthogonal,
                seed=37,
            ).fit_from_source_problems(
                sources,
                n_records_per_domain=10,
                rng=np.random.default_rng(37),
            )

        orthogonal = fit(True)
        correlated = fit(False)
        np.testing.assert_array_equal(
            orthogonal.ordered_exposure_selected_frequencies,
            correlated.ordered_exposure_selected_frequencies,
        )
        problem = self._problem("InventorySupplyChain", d=12)
        x = problem.sample_random(np.random.default_rng(38))
        orthogonal_values = orthogonal.ordered_local_exposure(problem, x)
        correlated_values = correlated.ordered_local_exposure(problem, x)
        indices = np.arange(len(orthogonal_values))
        distance = indices[:, None] - indices[None, :]
        mixing = np.where(distance >= 0, 0.75 ** distance, 0.0)
        self.assertEqual(np.linalg.matrix_rank(mixing), len(orthogonal_values))
        np.testing.assert_allclose(
            correlated_values, mixing @ orthogonal_values, atol=1e-12)
        self.assertFalse(correlated.diagnostics()[
            "ordered_cumulative_exposure"]["orthogonal_coordinates"])

    def test_ordered_semiparametric_residual_is_orthogonal_single_expert(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            anchor_count=5,
            ordered_cumulative_exposure=True,
            ordered_exposure_max_frequency=6,
            ordered_exposure_active_dim=2,
            ordered_exposure_basis_mode="diagonal_quadratic",
            ordered_exposure_adaptive_sparsity=True,
            ordered_exposure_replace_local_kernel=True,
            ordered_exposure_semiparametric_residual=True,
            seed=41,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=10,
            rng=np.random.default_rng(41),
        )
        target = MetaPriorProblemAdapter(
            self._problem("InventorySupplyChain", d=12), prior)
        specs = target.task_posterior_expert_specs(include_local_kernel=True)
        names = [spec["name"] for spec in specs]
        self.assertIn("ordered_semiparametric", names)
        self.assertNotIn("ordered_cumulative", names)
        self.assertNotIn("local_risk_kernel", names)

        basis = target.task_expert_basis_map(
            "ordered_semiparametric", output_index=1)
        self.assertEqual(basis.feature_dim, 4 + 4 + 2 + 6)
        features = basis.features_many([
            target.sample_random(np.random.default_rng(seed))
            for seed in range(4)
        ])
        self.assertEqual(features.shape, (4, basis.feature_dim))
        self.assertTrue(np.all(np.isfinite(features)))
        diagnostics = basis.diagnostics()
        projection = diagnostics["ordered_residual_projection"]
        self.assertFalse(projection["target_labels_used"])
        self.assertEqual(
            projection["residualization_mode"],
            "bounded_coefficient_nullspace",
        )
        self.assertEqual(projection["residual_dim"], 6)
        self.assertGreaterEqual(projection["nullspace_dim"], 6)
        self.assertLess(projection["orthogonality_relative"], 1e-10)
        self.assertLess(projection["projection_orthonormal_error"], 1e-10)

        residual = features[:, -projection["residual_dim"]:]
        self.assertTrue(np.all(
            np.linalg.norm(residual, axis=1)
            <= projection["global_l2_bound"] + 1e-10
        ))
        broad_features = basis.features_many([
            tuple([value] * target.d)
            for value in range(0, 101, 5)
        ])
        broad_residual = broad_features[:, -projection["residual_dim"]:]
        self.assertTrue(np.all(
            np.linalg.norm(broad_residual, axis=1)
            <= projection["global_l2_bound"] + 1e-10
        ))

        sparse = basis.adaptive_sparsity_spec()
        self.assertEqual(sparse["dictionary_dim"], basis.feature_dim)
        self.assertEqual(sparse["always_active_count"], 4)
        np.testing.assert_allclose(sparse["source_pip"][-6:], 0.5)
        provider = target.task_expert_problem_view(
            "ordered_semiparametric")
        self.assertEqual(
            provider.cumulative_risk_provider_status()["coordinate"],
            "frozen_source_learned_ordered_cumulative",
        )

    def test_ordered_local_structure_is_a_six_expert_latent_choice(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="spectral_hvd",
            spectral_risk_alignment=True,
            ordered_cumulative_exposure=True,
            ordered_exposure_active_dim=2,
            ordered_exposure_basis_mode="diagonal_quadratic",
            ordered_exposure_adaptive_sparsity=True,
            ordered_exposure_replace_local_kernel=False,
            ordered_exposure_latent_structure_selection=True,
            seed=42,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=10,
            rng=np.random.default_rng(42),
        )
        target = MetaPriorProblemAdapter(
            self._problem("InventorySupplyChain", d=12), prior)
        specs = target.task_posterior_expert_specs(include_local_kernel=True)
        names = [spec["name"] for spec in specs]
        self.assertEqual(len(names), 6)
        self.assertIn("ordered_cumulative", names)
        self.assertIn("local_risk_kernel", names)
        self.assertIn("risk_aligned_spectral", names)
        self.assertNotIn("risk_aligned_coordinate", names)
        self.assertNotIn("ordered_semiparametric", names)
        self.assertAlmostEqual(sum(
            spec["prior_weight"] for spec in specs), 1.0)
        self.assertTrue(
            prior.diagnostics()["ordered_cumulative_exposure"][
                "latent_structure_selection"])

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LearnedMetaPrior(
                ordered_exposure_semiparametric_residual=True,
                ordered_exposure_latent_structure_selection=True,
            )

    def test_ordered_group_shrinkage_learns_group_strength_not_direction(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="spectral_hvd",
            spectral_risk_alignment=True,
            ordered_cumulative_exposure=True,
            ordered_exposure_active_dim=2,
            ordered_exposure_basis_mode="diagonal_quadratic",
            ordered_exposure_adaptive_sparsity=True,
            ordered_exposure_latent_structure_selection=True,
            ordered_exposure_group_shared_shrinkage=True,
            seed=43,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=10,
            rng=np.random.default_rng(43),
        )
        target = MetaPriorProblemAdapter(
            self._problem("InventorySupplyChain", d=12), prior)
        basis = target.task_expert_basis_map(
            "ordered_cumulative", output_index=1)
        spec = basis.adaptive_sparsity_spec()
        self.assertEqual(
            spec["shared_shrinkage_groups"],
            [-1] * 4 + [0] * 4 + [1] * 2,
        )
        self.assertTrue(
            basis.diagnostics()["ordered_group_shared_shrinkage"])
        self.assertTrue(
            prior.diagnostics()["ordered_cumulative_exposure"][
                "group_shared_shrinkage"])

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LearnedMetaPrior(
                ordered_exposure_semiparametric_residual=True,
                ordered_exposure_group_shared_shrinkage=True,
            )

    def test_ordered_group_ridge_selects_complexity_from_target_data(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="spectral_hvd",
            spectral_risk_alignment=True,
            ordered_cumulative_exposure=True,
            ordered_exposure_active_dim=2,
            ordered_exposure_basis_mode="diagonal_quadratic",
            ordered_exposure_adaptive_sparsity=True,
            ordered_exposure_latent_structure_selection=True,
            ordered_exposure_group_ridge_learning=True,
            seed=44,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=10,
            rng=np.random.default_rng(44),
        )
        target = MetaPriorProblemAdapter(
            self._problem("InventorySupplyChain", d=12), prior)
        specs = target.task_posterior_expert_specs(include_local_kernel=True)
        self.assertEqual(len(specs), 6)
        basis = target.task_expert_basis_map(
            "ordered_cumulative", output_index=1)
        spec = basis.adaptive_sparsity_spec()
        self.assertEqual(spec["method"], "nested_loo_group_ridge")
        self.assertEqual(
            spec["group_ids"], [0] * 4 + [1] * 4 + [2] * 2)
        self.assertNotIn("max_effective_fraction", spec)
        self.assertFalse(spec["oracle_used"])
        self.assertTrue(
            basis.diagnostics()["ordered_group_ridge_learning"])

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LearnedMetaPrior(
                ordered_exposure_adaptive_sparsity=True,
                ordered_exposure_group_shared_shrinkage=True,
                ordered_exposure_group_ridge_learning=True,
            )

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
        audit = target.admissibility_audit()
        self.assertFalse(audit["admissible_mainline"])
        self.assertTrue(audit["source_oracle_aided"])
        self.assertTrue(audit["uses_source_true_outputs"])
        self.assertTrue(audit["uses_source_true_sigma"])
        self.assertFalse(hasattr(AdmissibleProblemAdapter(target_base), "risk_exposures"))
        candidates = target.recommendation_refinement_candidates()
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), target.refinement_count)

    def test_observable_mean_coordinate_is_oracle_free_and_separate_from_hvd(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1")),
            ("InventorySupplyChain", self._problem("InventorySupplyChain")),
        ]

        def forbidden(*_args, **_kwargs):
            raise AssertionError("source oracle was called")

        for _name, problem in sources:
            problem.true_sigma = forbidden
            problem.true_outputs = forbidden
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="consensus",
            source_observation_mode="replicated",
            source_observation_replicates=3,
            source_design_mode="universal_mixture",
            source_universal_fraction=0.75,
            teacher_records_per_domain=0,
            seed=517,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=8,
            rng=np.random.default_rng(517),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl"), prior)
        audit = target.admissibility_audit()
        self.assertTrue(audit["admissible_mainline"])
        self.assertTrue(audit["admissible_oracle_free_transfer"])
        self.assertFalse(audit["source_oracle_aided"])
        self.assertFalse(audit["uses_source_true_outputs"])
        self.assertFalse(audit["uses_source_true_sigma"])
        self.assertEqual(audit["source_observation_mode"], "replicated")
        self.assertEqual(audit["source_design_mode"], "universal_mixture")
        self.assertEqual(audit["source_universal_record_count"], 12)
        self.assertEqual(audit["source_simulator_calls"], 48)
        contract = target.mean_risk_coordinate_contract()
        self.assertEqual(contract["status"], "separated")
        self.assertFalse(contract["source_oracle_aided"])
        self.assertEqual(contract["source_design_mode"], "universal_mixture")
        self.assertEqual(
            contract["eta_source_training_target"], "constraint_mean")
        self.assertIn("mu_g(eta)", contract["joined_object"])

        constraint_basis = target.gpr_basis_map(output_index=1)
        objective_basis = target.gpr_basis_map(output_index=0)
        expert_basis = target.task_expert_basis_map(
            "universal_coordinate", output_index=1)
        self.assertIsInstance(
            constraint_basis, ObservableConstraintMeanBasis)
        self.assertIsInstance(expert_basis, ObservableConstraintMeanBasis)
        self.assertNotIsInstance(
            objective_basis, ObservableConstraintMeanBasis)
        x = target.sample_random(np.random.default_rng(518))
        self.assertTrue(np.all(np.isfinite(constraint_basis.features(x))))
        self.assertEqual(constraint_basis.feature_dim, 2)
        exposure = target.task_expert_problem_view(
            "universal_coordinate").risk_exposures(x)
        self.assertEqual(exposure.A.shape, (3,))
        self.assertEqual(exposure.N.shape, (2,))

    def test_boundary_aligned_phi_and_excitation_pool_are_oracle_free(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]

        def forbidden(*_args, **_kwargs):
            raise AssertionError("oracle was called")

        for _name, problem in sources:
            problem.true_sigma = forbidden
            problem.true_outputs = forbidden
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_latent_dim=2,
            source_observation_mode="replicated",
            source_observation_replicates=3,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            seed=519,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(519),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=100), prior)
        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        pool = target.boundary_excitation_candidates(
            n=64,
            rng=np.random.default_rng(520),
            pool_size=64,
        )
        self.assertEqual(len(pool), 64)
        self.assertEqual(len(set(pool)), 64)
        self.assertTrue(all(len(point) == 100 for point in pool))
        contract = target.mean_risk_coordinate_contract()
        self.assertEqual(
            contract["constraint_mean_coordinate"],
            "phi=source_aligned_chance_boundary",
        )
        self.assertEqual(
            contract["coefficient_prior_training_target"],
            "constraint_mean",
        )
        basis = target.gpr_basis_map(output_index=1)
        self.assertEqual(basis.feature_dim, 2)
        self.assertEqual(
            basis.diagnostics()["output_mode"], "boundary_aligned")
        diagnostics = prior.boundary_excitation_diagnostics
        self.assertTrue(diagnostics["source_only"])
        self.assertFalse(diagnostics["target_data_used"])
        self.assertFalse(diagnostics["target_oracle_used"])

    def test_boundary_aligned_mode_requires_chance_margin_supervision(self):
        with self.assertRaisesRegex(ValueError, "chance-margin strata"):
            LearnedMetaPrior(
                observable_mean_coordinate=True,
                observable_mean_mode="boundary_aligned",
                observable_mean_training_target="constraint_mean",
            )

    def test_source_learned_exposure_mean_head_is_oracle_free(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="source_learned_exposure",
            observable_mean_latent_dim=4,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=521,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(521),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target structural provider was called")

        target.base.risk_exposures = forbidden
        target.base.cumulative_risk_features = forbidden
        x = target.sample_random(np.random.default_rng(522))
        basis = target.gpr_basis_map(output_index=1)
        features = basis.features(x)
        self.assertEqual(features.shape, (4,))
        self.assertTrue(np.all(np.isfinite(features)))
        diagnostics = basis.diagnostics()
        self.assertEqual(
            diagnostics["mean_coordinate_input"],
            "source_learned_exposure",
        )
        self.assertTrue(diagnostics["separate_mean_variance_heads"])
        contract = target.mean_risk_coordinate_contract()
        self.assertEqual(
            contract["constraint_mean_input"],
            "source_learned_exposure",
        )
        self.assertTrue(contract["separate_mean_variance_heads"])
        audit = target.admissibility_audit()
        self.assertTrue(audit["admissible_strict_lodo"])
        self.assertFalse(audit["tcb_target_structural_provider_used"])

    def test_observable_state_mean_head_is_oracle_free(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_latent_dim=4,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=527,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(527),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or risk provider was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        x = target.sample_random(np.random.default_rng(528))
        basis = target.gpr_basis_map(output_index=1)
        features = basis.features(x)
        self.assertEqual(features.shape, (4,))
        self.assertTrue(np.all(np.isfinite(features)))
        diagnostics = basis.diagnostics()
        self.assertEqual(
            diagnostics["mean_coordinate_input"],
            "observable_state_exposure",
        )
        self.assertTrue(diagnostics["separate_mean_variance_heads"])
        self.assertTrue(
            diagnostics["alignment"]["observable_state_exposure"])
        audit = target.admissibility_audit()
        self.assertTrue(audit["admissible_strict_lodo"])
        self.assertTrue(audit["observable_state_exposure_used"])
        self.assertFalse(audit["tcb_target_structural_provider_used"])

    def test_invariant_quadratic_mean_head_contract_is_frozen_and_auditable(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="set_invariant",
            observable_mean_feature_mode="diagonal_quadratic",
            observable_mean_latent_dim=3,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=529,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(529),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=100), prior)
        basis = target.gpr_basis_map(output_index=1)
        x = target.sample_random(np.random.default_rng(530))
        self.assertEqual(basis.features(x).shape, (6,))
        diagnostics = basis.diagnostics()
        self.assertEqual(
            diagnostics["observable_descriptor_mode"], "set_invariant")
        self.assertEqual(
            diagnostics["boundary_feature_mode"], "diagonal_quadratic")
        contract = target.mean_risk_coordinate_contract()
        self.assertEqual(
            contract["constraint_mean_descriptor_mode"], "set_invariant")
        self.assertEqual(
            contract["constraint_mean_feature_mode"],
            "diagonal_quadratic",
        )
        self.assertFalse(contract["coordinate_definition_uses_target_labels"])

    def test_role_aligned_linear_mean_head_matches_target_without_labels(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_aligned",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=531,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(531),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or oracle was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        basis = target.gpr_basis_map(output_index=1)
        x = target.sample_random(np.random.default_rng(532))
        self.assertEqual(basis.features(x).shape, (3,))
        diagnostics = basis.diagnostics()
        self.assertEqual(
            diagnostics["observable_descriptor_mode"], "role_aligned")
        role_diagnostics = diagnostics["alignment"][
            "channel_role_alignment"]
        self.assertFalse(role_diagnostics["target_labels_used"])
        self.assertFalse(role_diagnostics["target_oracle_used"])
        self.assertTrue(role_diagnostics["target_matches"])
        contract = target.mean_risk_coordinate_contract()
        self.assertEqual(
            contract["constraint_mean_descriptor_mode"], "role_aligned")
        self.assertFalse(contract["coordinate_definition_uses_target_labels"])

    def test_role_transport_mean_head_supports_unseen_cardinality(self):
        sources = [
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_transport",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=537,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(537),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or oracle was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        basis = target.gpr_basis_map(output_index=1)
        x = target.sample_random(np.random.default_rng(538))
        features = basis.features(x)
        diagnostics = basis.diagnostics()
        alignment = diagnostics["alignment"]["channel_role_alignment"]
        target_match = next(iter(alignment["target_matches"].values()))
        self.assertEqual(features.shape, (3,))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertEqual(
            diagnostics["observable_descriptor_mode"], "role_transport")
        self.assertTrue(alignment["partial_transport"])
        self.assertEqual(target_match["channel_count"], 2)
        self.assertEqual(len(target_match["transport_weights"]), 2)
        calibration = basis.source_target_epistemic_calibration()
        self.assertGreaterEqual(
            calibration["epistemic_covariance_scale"], 1.0)
        self.assertFalse(calibration["target_labels_used"])
        self.assertFalse(calibration["target_oracle_used"])

    def test_role_assignment_posterior_is_finite_low_rank_and_oracle_free(self):
        sources = [
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            observable_mean_role_assignment_posterior=True,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=543,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(543),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or oracle was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        basis = target.gpr_basis_map(output_index=1)
        x = target.sample_random(np.random.default_rng(544))
        diagnostics = basis.diagnostics()["role_assignment_posterior"]
        self.assertEqual(diagnostics["assignment_count"], 6)
        self.assertEqual(diagnostics["active_feature_dim_per_atom"], 3)
        self.assertEqual(basis.features(x).shape, (18,))
        self.assertTrue(diagnostics["permutation_equivariant"])
        self.assertFalse(
            diagnostics["target_labels_used_to_define_assignments"])
        self.assertFalse(
            diagnostics["target_oracle_used_to_define_assignments"])

        components = basis.source_parametric_prior_components()
        self.assertEqual(len(components), 12)
        self.assertTrue(all(
            len(component["mean"]) == 19
            and component["covariance"].shape == (19, 19)
            and len(component["diagnostics"][
                "role_assignment_active_coefficients"]) == 4
            for component in components
        ))
        aggregate = basis.source_parametric_prior()
        self.assertEqual(aggregate["mean"].shape, (19,))
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(aggregate["covariance"]))),
            -1e-9,
        )
        null = dict(aggregate)
        null["name"] = "target:null"
        null["prior_weight"] = 0.5
        null_components = basis.expand_target_role_assignment_components([null])
        self.assertEqual(len(null_components), 6)
        self.assertAlmostEqual(
            sum(component["prior_weight"] for component in null_components),
            0.5,
        )
        contract = target.mean_risk_coordinate_contract()
        self.assertTrue(contract["channel_role_assignment_posterior"])
        self.assertTrue(
            contract[
                "channel_role_assignment_weights_use_charged_target_labels"])
        self.assertFalse(
            contract["channel_role_assignment_weights_use_target_oracle"])

    def test_source_geometry_role_prior_is_shared_by_source_and_null_orbits(self):
        sources = [
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            observable_mean_role_assignment_posterior=True,
            observable_mean_role_assignment_prior="source_geometry",
            observable_mean_role_assignment_prior_temperature_scale=0.5,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=545,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(545),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or oracle was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        basis = target.gpr_basis_map(output_index=1)
        role = basis.diagnostics()["role_assignment_posterior"]
        assignments = list(role["assignments"])
        expected = np.asarray(role["assignment_prior_weights"], dtype=float)
        prior_diagnostics = role["assignment_prior_diagnostics"]
        self.assertAlmostEqual(float(np.sum(expected)), 1.0, places=12)
        self.assertGreater(float(np.max(expected)), float(np.min(expected)))
        self.assertTrue(
            prior_diagnostics["maximum_prior_matches_hard_assignment"])
        self.assertFalse(prior_diagnostics["target_labels_used"])
        self.assertFalse(prior_diagnostics["target_oracle_used"])

        source_mass = {assignment: 0.0 for assignment in assignments}
        for component in basis.source_parametric_prior_components():
            label = component["diagnostics"]["role_assignment"]
            source_mass[label] += float(component["prior_weight"])
        source = np.asarray([
            source_mass[assignment] for assignment in assignments],
            dtype=float,
        )
        source /= float(np.sum(source))
        np.testing.assert_allclose(source, expected, atol=1e-12, rtol=1e-12)

        aggregate = basis.source_parametric_prior()
        null = dict(aggregate)
        null["name"] = "target:null"
        null["prior_weight"] = 1.0
        expanded = basis.expand_target_role_assignment_components([null])
        null_mass = {
            component["diagnostics"]["role_assignment"]: float(
                component["prior_weight"])
            for component in expanded
        }
        np.testing.assert_allclose(
            [null_mass[assignment] for assignment in assignments],
            expected,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_intervention_transport_uses_source_only_response_roles(self):
        sources = [
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=18)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=18)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_intervention_transport",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=539,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(539),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or oracle was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        basis = target.gpr_basis_map(output_index=1)
        x = target.sample_random(np.random.default_rng(540))
        self.assertEqual(basis.features(x).shape, (3,))
        diagnostics = basis.diagnostics()
        alignment = diagnostics["alignment"]["channel_role_alignment"]
        target_match = next(iter(alignment["target_matches"].values()))
        self.assertEqual(
            diagnostics["observable_descriptor_mode"],
            "role_intervention_transport",
        )
        self.assertEqual(alignment["signature_mode"], "intervention_response")
        self.assertTrue(alignment["barycentric_transport"])
        self.assertEqual(
            alignment["source_signature_pool"],
            "deterministic_unlabeled_intervention_pool",
        )
        self.assertEqual(
            target_match["transport_geometry"], "barycentric_response")
        self.assertFalse(target_match["target_labels_used"])
        self.assertFalse(target_match["target_oracle_used"])

    def test_target_orthogonal_residual_is_outcome_free_and_updates_online(self):
        sources = [
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=18)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=18)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_tanh",
            observable_mean_target_residual_rank=2,
            observable_mean_target_residual_prior_scale=0.5,
            observable_mean_target_residual_pool_size=64,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=541,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(541),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or oracle was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        basis = target.gpr_basis_map(output_index=1)
        diagnostics = basis.diagnostics()["target_orthogonal_residual"]
        self.assertEqual(diagnostics["status"], "fit")
        self.assertEqual(diagnostics["effective_rank"], 2)
        self.assertLess(diagnostics["maximum_base_cross_moment"], 1e-8)
        self.assertFalse(diagnostics["target_labels_used"])
        self.assertFalse(diagnostics["target_oracle_used"])

        pool = basis._target_residual_points
        features = basis.features_many(pool)
        base = np.column_stack([
            np.ones(len(pool)), features[:, :basis.base_feature_dim]])
        residual = features[:, basis.base_feature_dim:]
        np.testing.assert_allclose(
            base.T @ residual / len(pool),
            0.0,
            atol=1e-8,
            rtol=0.0,
        )
        source_prior = basis.source_parametric_prior()
        self.assertEqual(source_prior["mean"].shape, (1 + basis.feature_dim,))
        self.assertEqual(
            source_prior["covariance"].shape,
            (1 + basis.feature_dim, 1 + basis.feature_dim),
        )
        np.testing.assert_allclose(source_prior["mean"][-2:], 0.0)
        np.testing.assert_allclose(
            source_prior["covariance"][:-2, -2:], 0.0)

        rank_components = basis.expand_target_residual_rank_components(
            basis.source_parametric_prior_components(),
            [0.7, 0.2, 0.1],
        )
        self.assertEqual(
            len(rank_components),
            3 * len(basis.source_parametric_prior_components()),
        )
        self.assertAlmostEqual(
            sum(component["prior_weight"] for component in rank_components),
            sum(component["prior_weight"] for component in
                basis.source_parametric_prior_components()),
            places=12,
        )
        by_rank = {
            component["diagnostics"]["target_residual_structure_rank"]:
            component
            for component in rank_components[:3]
        }
        rank0 = np.diag(by_rank[0]["covariance"])[-2:]
        rank1 = np.diag(by_rank[1]["covariance"])[-2:]
        rank2 = np.diag(by_rank[2]["covariance"])[-2:]
        np.testing.assert_allclose(rank0, 1e-12)
        self.assertGreater(rank1[0], 1e-12)
        self.assertAlmostEqual(rank1[1], 1e-12)
        self.assertTrue(np.all(rank2 > 1e-12))
        self.assertTrue(all(
            not component["diagnostics"][
                "target_oracle_used_to_define_structure_prior"]
            for component in rank_components
        ))

        model = ParametricGPR(d=target.d, basis_map=basis)
        model.set_parametric_prior(
            source_prior["mean"],
            source_prior["deviation_variance"],
            source_prior["covariance"],
        )
        candidate_index = int(np.argmax(np.linalg.norm(residual, axis=1)))
        candidate = pool[candidate_index]
        residual_slice = slice(
            1 + basis.base_feature_dim,
            1 + basis.base_feature_dim + basis.target_residual_rank,
        )
        before = float(np.trace(model.C[residual_slice, residual_slice]))
        model.update(candidate, 0.25, 0.01)
        after = float(np.trace(model.C[residual_slice, residual_slice]))
        self.assertLess(after, before)

    def test_role_adaptive_mean_head_falls_back_on_unseen_cardinality(self):
        sources = [
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_set_invariant",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=533,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(533),
        )
        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or oracle was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        basis = target.gpr_basis_map(output_index=1)
        x = target.sample_random(np.random.default_rng(534))
        self.assertEqual(basis.features(x).shape, (3,))
        diagnostics = basis.diagnostics()
        selection = diagnostics["role_coordinate_selection"]
        self.assertFalse(selection["channel_cardinality_supported"])
        self.assertEqual(selection["selected_coordinate"], "set_invariant")
        self.assertFalse(selection["target_labels_used"])
        self.assertFalse(selection["target_oracle_used"])
        self.assertEqual(
            basis.source_target_coordinate_selection(), selection)
        components = basis.source_parametric_prior_components()
        self.assertTrue(components)
        self.assertTrue(all(
            component["diagnostics"]["role_coordinate_selection"]
            ["selected_coordinate"] == "set_invariant"
            for component in components
        ))

    def test_role_adaptive_support_residual_is_oracle_free_and_linear(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("QueueResourceControl", self._problem(
                "QueueResourceControl", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="observable_state_exposure",
            observable_mean_descriptor_mode="role_adaptive_ordered",
            observable_mean_feature_mode="linear",
            observable_mean_latent_dim=3,
            observable_mean_latent_transform="source_support_residual",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=535,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(535),
        )
        target = MetaPriorProblemAdapter(
            self._problem("InventorySupplyChain", d=100), prior)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("target outcome or oracle was called")

        target.base.true_objective = forbidden
        target.base.true_constraint_mean = forbidden
        target.base.true_sigma = forbidden
        target.base.risk_exposures = forbidden
        basis = target.gpr_basis_map(output_index=1)
        x = target.sample_random(np.random.default_rng(536))
        features = basis.features(x)
        diagnostics = basis.diagnostics()["selected_coordinate_diagnostics"]
        transform = diagnostics["latent_transform_diagnostics"]

        self.assertEqual(features.shape, (4,))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertGreaterEqual(features[-1], 0.0)
        self.assertLessEqual(features[-1], 1.0)
        self.assertEqual(
            diagnostics["latent_transform"], "source_support_residual")
        self.assertTrue(transform["residual_channel"])
        self.assertFalse(transform["selection_uses_target_data"])
        self.assertFalse(transform["selection_uses_target_oracle"])
        source_prior = basis.source_parametric_prior()
        self.assertEqual(source_prior["mean"].shape, (5,))
        self.assertEqual(source_prior["covariance"].shape, (5, 5))

    def test_provider_exposure_mean_head_is_upper_bound_only(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="coordinate",
            observable_mean_coordinate=True,
            observable_mean_mode="boundary_aligned",
            observable_mean_training_target="chance_margin",
            observable_mean_input_mode="provider_exposure",
            observable_mean_latent_dim=2,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            teacher_records_per_domain=0,
            seed=523,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(523),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=100), prior)
        basis = target.gpr_basis_map(output_index=1)
        x = target.sample_random(np.random.default_rng(524))
        self.assertTrue(np.all(np.isfinite(basis.features(x))))
        audit = target.admissibility_audit()
        self.assertFalse(audit["admissible_strict_lodo"])
        self.assertFalse(audit["admissible_mainline"])
        self.assertTrue(audit["tcb_target_structural_provider_used"])
        self.assertTrue(audit["uses_problem_specific_formula"])

    def test_template_free_boundary_pool_is_shared_across_mean_heads(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]

        def fit(input_mode):
            return LearnedMetaPrior(
                local_dim=3,
                shared_dim=2,
                component_stage="coordinate",
                observable_mean_coordinate=True,
                observable_mean_mode="boundary_aligned",
                observable_mean_training_target="chance_margin",
                observable_mean_input_mode=input_mode,
                observable_mean_latent_dim=2,
                source_observation_mode="replicated",
                source_observation_replicates=2,
                source_design_mode="universal_mixture",
                source_universal_fraction=1.0,
                teacher_records_per_domain=0,
                seed=525,
            ).fit_from_source_problems(
                sources,
                n_records_per_domain=16,
                rng=np.random.default_rng(525),
            )

        profile = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=100),
            fit("policy_profile"),
        )
        exposure = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=100),
            fit("source_learned_exposure"),
        )
        kwargs = {
            "n": 64,
            "pool_size": 64,
            "include_source_templates": False,
        }
        left = profile.boundary_excitation_candidates(
            rng=np.random.default_rng(526), **kwargs)
        right = exposure.boundary_excitation_candidates(
            rng=np.random.default_rng(526), **kwargs)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)
        self.assertFalse(profile.meta_prior.boundary_excitation_diagnostics[
            "source_boundary_templates_enabled"])

    def test_source_consensus_templates_are_oracle_free_ranked_and_varied(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]

        def forbidden(*_args, **_kwargs):
            raise AssertionError("analytic source oracle was called")

        for _name, problem in sources:
            problem.true_sigma = forbidden
            problem.true_outputs = forbidden
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            component_stage="spectral_hvd",
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="universal_mixture",
            source_universal_fraction=1.0,
            source_consensus_template_count=8,
            teacher_records_per_domain=0,
            seed=519,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=16,
            rng=np.random.default_rng(519),
        )
        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=12), prior)
        diagnostics = prior.diagnostics()["source_consensus_templates"]
        self.assertEqual(diagnostics["status"], "fit")
        self.assertEqual(diagnostics["n_source_domains"], 2)
        self.assertGreaterEqual(diagnostics["n_selected_templates"], 2)
        self.assertEqual(
            diagnostics["objective_ranking_target"],
            "observed_source_objective_percentile",
        )
        self.assertTrue(all(
            "objective_score" in item for item in diagnostics["selected"]))
        self.assertFalse(diagnostics["target_data_used"])
        self.assertFalse(diagnostics["target_oracle_used"])
        self.assertFalse(target.admissibility_audit()["source_oracle_aided"])

        library = prior.universal_shape_candidates(
            target, n=10000, rng=np.random.default_rng(1), force=True)
        initial = target.task_initial_universal_candidates(
            n=2, rng=np.random.default_rng(2))
        consensus = prior.source_consensus_template_candidates(target, n=1)
        self.assertEqual(initial[0], library[1])
        self.assertEqual(initial[1], consensus[0])
        frozen = target.frozen_source_consensus_candidates()
        self.assertEqual(
            len(frozen), diagnostics["n_selected_templates"])
        self.assertEqual(frozen[0], consensus[0])

        coverage = target.task_initial_universal_candidates(
            n=6, rng=np.random.default_rng(3))
        protected = target.frozen_source_coverage_candidates(n=6)
        self.assertEqual(coverage, protected)
        self.assertEqual(coverage[0], library[1])
        self.assertEqual(coverage[1], frozen[0])
        self.assertEqual(coverage[-1], frozen[-1])
        self.assertEqual(len(set(coverage)), len(coverage))

        high_dim_target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=200), prior)
        risk_objective = prior.risk_objective_initial_candidates(
            high_dim_target, n=6, rng=np.random.default_rng(521))
        proposal_diag = prior.diagnostics()["risk_objective_proposal"]
        self.assertEqual(len(risk_objective), 6)
        self.assertEqual(len(set(risk_objective)), 6)
        self.assertTrue(all(len(point) == 200 for point in risk_objective))
        self.assertEqual(proposal_diag["status"], "fit")
        self.assertEqual(proposal_diag["target_policy_dimension"], 200)
        self.assertGreater(proposal_diag["selected_template_count"], 0)
        self.assertGreaterEqual(
            proposal_diag["robust_source_feasible_template_count"], 0)
        self.assertFalse(proposal_diag["target_data_used"])
        self.assertFalse(proposal_diag["target_oracle_used"])
        universal = prior.universal_shape_candidates(
            high_dim_target, n=10000,
            rng=np.random.default_rng(521), force=True)
        protected_risk_objective = prior.risk_objective_initial_candidates(
            high_dim_target,
            n=6,
            rng=np.random.default_rng(521),
            protect_lower_envelope_sentinel=True,
        )
        protected_diag = prior.diagnostics()["risk_objective_proposal"]
        self.assertEqual(protected_risk_objective[0], universal[0])
        self.assertEqual(protected_risk_objective[1], universal[1])
        self.assertTrue(
            protected_diag["universal_lower_envelope_sentinel"])
        self.assertEqual(
            protected_diag["universal_sentinel_indices"], [0, 1])
        source_only = prior.risk_objective_template_initial_candidates(
            high_dim_target, n=5, rng=np.random.default_rng(521))
        self.assertEqual(len(source_only), 5)
        self.assertNotIn(universal[1], source_only)

        rng = np.random.default_rng(520)
        sequential = [
            target.task_expert_proposal_candidates(
                "universal_coordinate", n=1, rng=rng)[0]
            for _ in range(12)
        ]
        self.assertGreater(len(set(sequential)), 1)

    def test_shared_uniform_source_control_is_paired_and_not_low_frequency(self):
        sources = [
            ("FactorShockStatePolicyRZDT1", self._problem(
                "FactorShockStatePolicyRZDT1", d=12)),
            ("InventorySupplyChain", self._problem(
                "InventorySupplyChain", d=12)),
        ]
        prior = LearnedMetaPrior(
            local_dim=3,
            shared_dim=2,
            spectral_low_frequency_prior=False,
            source_observation_mode="replicated",
            source_observation_replicates=2,
            source_design_mode="shared_uniform",
            source_consensus_template_count=6,
            teacher_records_per_domain=0,
            seed=522,
        ).fit_from_source_problems(
            sources,
            n_records_per_domain=12,
            rng=np.random.default_rng(522),
        )
        records = prior.source_records_
        first = [rec for rec in records if rec.domain == sources[0][0]]
        second = [rec for rec in records if rec.domain == sources[1][0]]
        self.assertEqual(len(first), 12)
        self.assertTrue(all(
            rec.origin == "universal_shared_uniform" for rec in records))
        np.testing.assert_allclose(
            np.vstack([rec.profile for rec in first]),
            np.vstack([rec.profile for rec in second]),
        )
        consensus = prior.diagnostics()["source_consensus_templates"]
        self.assertEqual(consensus["status"], "fit")
        self.assertEqual(consensus["consensus_origin"], "universal_shared_uniform")
        training = prior.diagnostics()["training"]
        self.assertEqual(training["source_universal_record_count"], 24)
        self.assertEqual(training["source_simulator_calls"], 48)

        target = MetaPriorProblemAdapter(
            self._problem("QueueResourceControl", d=50), prior)
        points = prior.risk_objective_initial_candidates(
            target, n=5, rng=np.random.default_rng(523))
        proposal = prior.diagnostics()["risk_objective_proposal"]
        self.assertEqual(len(points), 5)
        self.assertFalse(proposal["low_frequency_coordinate"])
        self.assertEqual(proposal["source_design_mode"], "shared_uniform")
        self.assertEqual(proposal["generic_library_fallback_count"], 0)

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

        nonorthogonal = TransferableSpectralBasis(
            active_dim=4,
            low_frequency_components=10,
            n_neighbors=12,
            orthogonalization="none",
        ).fit(batches)
        nonorthogonal_diag = nonorthogonal.diagnostics()
        self.assertEqual(nonorthogonal_diag["orthogonalization"], "none")
        self.assertEqual(
            nonorthogonal_diag["selected_names"], diag["selected_names"])
        np.testing.assert_allclose(
            nonorthogonal.whitening_, np.eye(4), atol=1e-12)
        self.assertGreater(nonorthogonal_diag["max_offdiag_gram"], 1e-8)

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
        components = target.cumulative_hvd_prior_components(
            output_index=1, feature_dim=len(beta))
        self.assertIsNotNone(components)
        self.assertEqual(
            components["domains"],
            ["FactorShockStatePolicyRZDT1", "InventorySupplyChain"],
        )
        self.assertEqual(components["coefficients"].shape, (2, len(beta)))
        duplicate_components = duplicate_target.cumulative_hvd_prior_components(
            output_index=1, feature_dim=len(beta))
        np.testing.assert_allclose(
            components["coefficients"],
            duplicate_components["coefficients"],
            atol=0.0,
            rtol=0.0,
        )
        sensitivity = target.task_sensitivity_prior()
        self.assertEqual(sensitivity["status"], "fit_functional_bias_scale")
        self.assertAlmostEqual(
            sum(sensitivity["prior_weights"]), 1.0, places=12)
        self.assertEqual(
            len(sensitivity["biases"]), len(sensitivity["scales"]))
        self.assertGreater(len(sensitivity["biases"]), 3)
        self.assertTrue(sensitivity["functional_bias_profiles"])
        self.assertEqual(
            len(sensitivity["bias_coefficients"]),
            len(sensitivity["scales"]),
        )
        self.assertFalse(sensitivity["target_data_used"])
        x = target.sample_random(np.random.default_rng(312))
        bias_features = target.task_bias_features(x)
        self.assertEqual(
            len(bias_features),
            1 + prior.local_dim + max(prior.shared_dim - 1, 0),
        )
        self.assertTrue(np.all(np.isfinite(bias_features)))
        self.assertEqual(
            sensitivity["bias_profile_diagnostics"]["shared_coordinate"],
            "helmert_simplex_contrast",
        )
        self.assertEqual(
            sensitivity["bias_profile_diagnostics"]["profile_output_units"],
            "predictive_standard_deviations",
        )
        adaptive = sensitivity["adaptive_bias_prior"]
        self.assertEqual(
            adaptive["status"], "fit_boundary_weighted_gaussian")
        self.assertTrue(adaptive["boundary_weighted"])
        self.assertFalse(adaptive["target_data_used"])
        adaptive_mean = np.asarray(adaptive["mean"], dtype=float)
        adaptive_precision = np.asarray(
            adaptive["precision"], dtype=float)
        self.assertEqual(adaptive_mean.shape, bias_features.shape)
        self.assertEqual(
            adaptive_precision.shape,
            (len(bias_features), len(bias_features)),
        )
        self.assertTrue(np.all(
            np.linalg.eigvalsh(adaptive_precision) > 0.0))
        covariance_eigenvalues = np.linalg.eigvalsh(
            np.linalg.inv(adaptive_precision))
        self.assertTrue(np.all(covariance_eigenvalues >= 0.25 - 1e-10))
        self.assertTrue(np.all(covariance_eigenvalues <= 4.0 + 1e-10))
        self.assertEqual(
            len(sensitivity["adaptive_scale_class_names"]), 3)
        self.assertAlmostEqual(
            sum(sensitivity["adaptive_scale_prior_weights"]),
            1.0,
            places=12,
        )
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

        for expert_name in ("universal_coordinate", "source_spectral"):
            expert_problem = target.task_expert_problem_view(expert_name)
            expert_beta = expert_problem.cumulative_hvd_prior_beta(1)
            self.assertIsNotNone(expert_beta)
            self.assertEqual(
                len(expert_beta),
                len(expert_problem.cumulative_risk_features(samples[0])),
            )
            expert_hvd = OrthogonalHVD(mode="factor", n_outputs=2)
            expert_hvd.fit_from_residuals(
                samples,
                residuals,
                output_index=1,
                problem=expert_problem,
            )
            expert_diagnostics = expert_hvd.diagnostics()
            self.assertTrue(
                expert_diagnostics["cumulative_prior_used"]["1"])
            self.assertTrue(
                expert_diagnostics["cumulative_provider_active"]["1"])
            self.assertTrue(
                expert_diagnostics["cumulative_active"]["1"])
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

    def test_frequency_then_additive_composes_but_adaptive_sparsity_is_separate(self):
        prior = LearnedMetaPrior(
            spectral_frequency_adaptation=True,
            spectral_additive_adaptation=True,
        )
        self.assertTrue(prior.spectral_frequency_adaptation)
        self.assertTrue(prior.spectral_additive_adaptation)
        with self.assertRaisesRegex(ValueError, "spike-and-slab"):
            LearnedMetaPrior(
                spectral_frequency_adaptation=True,
                spectral_adaptive_sparsity=True,
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

    def test_source_boundary_bracket_is_lodo_fit_and_target_label_free(self):
        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral_hvd",
            spectral_active_dim=4,
            spectral_risk_alignment=True,
            spectral_alignment_source_episodes=2,
            teacher_records_per_domain=8,
            teacher_pool_size=128,
            seed=611,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", self._problem("InventorySupplyChain")),
                ("QueueResourceControl", self._problem("QueueResourceControl")),
                ("StatePolicyRZDT1", self._problem("StatePolicyRZDT1")),
            ],
            n_records_per_domain=24,
            rng=np.random.default_rng(612),
        )
        model = prior.diagnostics()["source_boundary_bracket"]
        self.assertEqual(model["status"], "fit")
        self.assertEqual(
            model["method"], "source_domain_lodo_boundary_ordering")
        self.assertFalse(model["target_data_used"])
        self.assertFalse(model["target_oracle_used"])
        self.assertGreaterEqual(model["n_domains"], 2)

        target = MetaPriorProblemAdapter(
            self._problem("FactorShockStatePolicyRZDT1"), prior)
        candidates = target.task_boundary_bracket_candidates(
            n=5,
            rng=np.random.default_rng(613),
            pool_size=96,
        )
        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(set(candidates)), 5)
        diagnostics = target.task_boundary_bracket_diagnostics()
        self.assertEqual(diagnostics["status"], "ok")
        self.assertFalse(diagnostics["target_labels_used"])
        self.assertFalse(diagnostics["target_oracle_used"])
        self.assertLess(diagnostics["score_min"], diagnostics["score_max"])


if __name__ == "__main__":
    unittest.main()
