import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from representation.transferable_boundary import (  # noqa: E402
    BoundaryFamilyMixturePosterior,
    BoundaryFamilySemiparametricPosterior,
    BoundaryFamilySynthesisPosterior,
    HierarchicalSignedDistancePosterior,
    TransferableChanceBoundaryPosterior,
)
from performance.benchmark_boundary_coordinate_screen import (  # noqa: E402
    _source_boundary_pilot_indices,
)


SUBMIT_SCRIPT = REPO / "scripts/submit_scolhkg_boundary_screen_scheduler.py"
SPEC = importlib.util.spec_from_file_location("boundary_screen_submit", SUBMIT_SCRIPT)
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


class TransferableBoundaryTests(unittest.TestCase):
    @staticmethod
    def source_data():
        grid = np.linspace(-1.5, 1.5, 41)
        first = np.column_stack([grid, 0.25 * np.sin(grid)])
        second = np.column_stack([grid, -0.20 * np.sin(grid)])
        descriptors = np.vstack([first, second])
        margins = np.concatenate([grid, grid + 0.05 * np.sin(2.0 * grid)])
        domains = np.asarray(["a"] * len(grid) + ["b"] * len(grid), dtype=object)
        return descriptors, margins, domains

    def test_all_coordinate_geometry_pairs_are_finite(self):
        descriptors, margins, domains = self.source_data()
        for coordinate in TransferableChanceBoundaryPosterior.COORDINATES:
            for geometry in TransferableChanceBoundaryPosterior.GEOMETRIES:
                with self.subTest(coordinate=coordinate, geometry=geometry):
                    model = TransferableChanceBoundaryPosterior(
                        coordinate=coordinate,
                        geometry=geometry,
                        adaptation="frozen",
                        rank=2,
                    ).fit(descriptors, margins, domains)
                    prediction = model.predict(descriptors[:8])
                    upper = model.predict_upper(descriptors[:8])
                    self.assertTrue(np.all(np.isfinite(prediction)))
                    self.assertTrue(np.all(upper >= prediction))
                    self.assertFalse(model.diagnostics()["target_oracle_used"])

    def test_orthogonal_pilot_alignment_recovers_rotated_boundary(self):
        descriptors, margins, domains = self.source_data()
        rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        target = descriptors[:41] @ rotation
        pilot_index = np.asarray([0, 5, 10, 15, 20, 25, 30, 35, 40])
        eval_index = np.asarray([2, 7, 12, 17, 22, 27, 32, 37])
        frozen = TransferableChanceBoundaryPosterior(
            coordinate="learned_psi",
            geometry="linear_monotone",
            adaptation="frozen",
            rank=2,
            adaptation_ridge=0.0,
        ).fit(descriptors, margins, domains)
        aligned = TransferableChanceBoundaryPosterior(
            coordinate="learned_psi",
            geometry="linear_monotone",
            adaptation="orthogonal_shift",
            rank=2,
            adaptation_ridge=0.0,
        ).fit(descriptors, margins, domains)
        adapter = aligned.fit_target_adapter(
            target[pilot_index], margins[pilot_index])
        frozen_error = float(np.mean((
            frozen.predict(target[eval_index]) - margins[eval_index]
        ) ** 2))
        aligned_error = float(np.mean((
            aligned.predict(target[eval_index], adapter=adapter)
            - margins[eval_index]
        ) ** 2))
        self.assertLess(aligned_error, frozen_error)
        self.assertEqual(adapter.mode, "orthogonal_shift")
        self.assertEqual(
            adapter.diagnostics["effective_label_adaptation_dimension"], 4)
        self.assertEqual(adapter.calibration_coordinate_count, 1)
        self.assertGreater(adapter.residual_scale, 0.0)
        self.assertGreater(adapter.residual_degrees_of_freedom, 0.0)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(adapter.output_covariance))),
            -1e-12,
        )
        self.assertTrue(np.all(
            aligned.predict_upper(
                target[eval_index], adapter=adapter)
            >= aligned.predict(target[eval_index], adapter=adapter)
        ))
        self.assertFalse(adapter.diagnostics["target_oracle_used"])

    def test_low_rank_quadratic_is_psd(self):
        descriptors, margins, domains = self.source_data()
        model = TransferableChanceBoundaryPosterior(
            coordinate="boundary_latent",
            geometry="low_rank_psd",
            adaptation="frozen",
            rank=2,
        ).fit(descriptors, margins ** 2 - 0.5, domains)
        eigenvalues = np.linalg.eigvalsh(model.geometry_state_["quadratic"])
        self.assertGreaterEqual(float(np.min(eigenvalues)), -1e-10)

    def test_domain_intercept_shift_does_not_become_residual_noise(self):
        descriptors, margins, domains = self.source_data()
        model = TransferableChanceBoundaryPosterior(
            coordinate="learned_psi",
            geometry="linear_monotone",
            adaptation="shift_scale",
            rank=2,
            adaptation_ridge=5.0,
            calibration_prior_df=1.0,
            target_residual_rank=1,
        ).fit(descriptors, margins, domains)
        pilot = descriptors[:20]
        shifted_margin = margins[:20] + 8.0
        adapter = model.fit_target_adapter(pilot, shifted_margin)
        prediction = model.predict(pilot, adapter=adapter)
        self.assertLess(float(np.sqrt(np.mean(
            (prediction - shifted_margin) ** 2))), 1.0)
        self.assertLess(adapter.residual_scale, 2.0)
        self.assertGreater(adapter.output_offset, 5.0)

    def test_hierarchical_signed_distance_recovers_positive_target_scale(self):
        grid = np.linspace(-1.5, 1.5, 61)
        base = np.column_stack([grid, np.sin(grid)])
        descriptors = np.vstack([base, base, base])
        domains = np.asarray(
            ["a"] * len(grid) + ["b"] * len(grid) + ["c"] * len(grid),
            dtype=object,
        )
        margins = np.concatenate([
            -0.5 + 0.8 * grid,
            0.4 + 1.3 * grid,
            1.1 + 2.0 * grid,
        ])
        model = HierarchicalSignedDistancePosterior(
            coordinate="learned_psi",
            geometry="linear_monotone",
            rank=2,
            effect_ridge=0.1,
        ).fit(
            descriptors,
            margins,
            domains,
            margin_variance=np.full(len(margins), 0.01),
            replicate_count=np.full(len(margins), 4),
        )
        pilot_index = np.asarray([0, 10, 20, 30, 40, 50, 60])
        target_margin = 2.5 + 1.7 * grid
        adapter = model.fit_target_adapter(
            base[pilot_index],
            target_margin[pilot_index],
            pilot_variance=np.full(len(pilot_index), 0.01),
            replicate_count=np.full(len(pilot_index), 4),
        )
        prediction = model.predict(base, adapter=adapter)
        upper = model.predict_upper(base, adapter=adapter)
        self.assertGreater(adapter.output_scale, 0.0)
        self.assertLess(float(np.sqrt(np.mean(
            (prediction - target_margin) ** 2))), 0.25)
        self.assertTrue(np.all(upper >= prediction))
        self.assertEqual(adapter.effective_dimension, 2)
        self.assertTrue(adapter.diagnostics["positive_output_scale"])
        self.assertTrue(
            model.diagnostics()["replicate_aware_likelihood"])

    def test_hierarchical_effect_covariance_shrinks_with_replicates(self):
        descriptors, margins, domains = self.source_data()
        model = HierarchicalSignedDistancePosterior(
            coordinate="learned_psi",
            geometry="linear_monotone",
            rank=2,
        ).fit(
            descriptors,
            margins,
            domains,
            margin_variance=np.full(len(margins), 0.04),
        )
        pilot = descriptors[:12]
        target = margins[:12] + 0.3
        one = model.fit_target_adapter(
            pilot,
            target,
            pilot_variance=np.full(len(pilot), 0.04),
            replicate_count=np.ones(len(pilot)),
        )
        many = model.fit_target_adapter(
            pilot,
            target,
            pilot_variance=np.full(len(pilot), 0.04),
            replicate_count=np.full(len(pilot), 16),
        )
        self.assertLess(
            float(np.trace(many.output_covariance)),
            float(np.trace(one.output_covariance)),
        )

    def test_hierarchical_source_rotation_recovers_common_boundary_shape(self):
        rng = np.random.default_rng(231)
        descriptors = []
        margins = []
        domains = []
        for name, angle, location, scale in (
            ("a", 0.0, -0.2, 0.8),
            ("b", 0.8, 0.3, 1.2),
            ("c", -0.6, 0.8, 1.6),
        ):
            canonical = rng.normal(size=(120, 2))
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.asarray([
                [cosine, -sine], [sine, cosine]])
            descriptors.append(canonical @ rotation.T)
            margins.append(
                location
                + scale * (
                    canonical[:, 0] + 0.25 * canonical[:, 1] ** 2))
            domains.extend([name] * len(canonical))
        descriptors = np.vstack(descriptors)
        margins = np.concatenate(margins)
        domains = np.asarray(domains, dtype=object)
        common = dict(
            coordinate="learned_psi",
            geometry="low_rank_psd",
            rank=2,
            hierarchy_iterations=8,
            rotation_ridge=0.1,
        )
        unaligned = HierarchicalSignedDistancePosterior(
            **common, rotation_mode="none").fit(
                descriptors, margins, domains)
        aligned = HierarchicalSignedDistancePosterior(
            **common, rotation_mode="planar").fit(
                descriptors, margins, domains)
        self.assertLess(
            aligned.diagnostics()["source_boundary_rmse"],
            0.1 * unaligned.diagnostics()["source_boundary_rmse"],
        )
        self.assertTrue(
            aligned.diagnostics()["source_rotation_alignment"])

    def test_one_orthogonal_target_residual_repairs_shifted_boundary(self):
        rng = np.random.default_rng(232)
        descriptors = []
        margins = []
        domains = []
        for name, location, scale, gamma in (
            ("a", -0.2, 0.8, 0.4),
            ("b", 0.3, 1.2, -0.3),
            ("c", 0.8, 1.6, 0.2),
        ):
            coordinate = rng.normal(size=(100, 2))
            descriptors.append(coordinate)
            margins.append(
                location + scale * coordinate[:, 0]
                + gamma * coordinate[:, 1])
            domains.extend([name] * len(coordinate))
        descriptors = np.vstack(descriptors)
        margins = np.concatenate(margins)
        domains = np.asarray(domains, dtype=object)
        target = rng.normal(size=(80, 2))
        target_margin = 0.1 + 1.1 * target[:, 0] + 0.5 * target[:, 1]
        pilot = np.arange(12)
        common = dict(
            coordinate="learned_psi",
            geometry="linear_monotone",
            rank=2,
            hierarchy_iterations=5,
        )
        rigid = HierarchicalSignedDistancePosterior(
            **common, target_residual_rank=0).fit(
                descriptors, margins, domains)
        residual = HierarchicalSignedDistancePosterior(
            **common,
            target_residual_rank=1,
            residual_ridge=0.1,
        ).fit(descriptors, margins, domains)
        rigid_adapter = rigid.fit_target_adapter(
            target[pilot], target_margin[pilot])
        residual_adapter = residual.fit_target_adapter(
            target[pilot], target_margin[pilot])
        rigid_rmse = float(np.sqrt(np.mean((
            rigid.predict(target, rigid_adapter) - target_margin) ** 2)))
        residual_rmse = float(np.sqrt(np.mean((
            residual.predict(target, residual_adapter) - target_margin) ** 2)))
        self.assertLess(residual_rmse, 0.1 * rigid_rmse)
        self.assertEqual(residual_adapter.effective_dimension, 3)
        self.assertEqual(
            residual_adapter.diagnostics["orthogonal_residual_rank"], 1)

    @staticmethod
    def boundary_family_data():
        grid = np.linspace(-1.5, 1.5, 61)
        descriptor = np.column_stack([grid, 0.2 * grid ** 2])
        positive_shape = grid + 0.8 * grid ** 2
        negative_shape = grid - 0.8 * grid ** 2
        descriptors = np.vstack([descriptor, descriptor, descriptor])
        margins = np.concatenate([
            -0.1 + positive_shape,
            0.2 + 1.1 * positive_shape,
            0.1 + negative_shape,
        ])
        domains = np.asarray(
            ["positive_a"] * len(grid)
            + ["positive_b"] * len(grid)
            + ["negative"] * len(grid),
            dtype=object,
        )
        return descriptor, descriptors, margins, domains

    def test_boundary_family_posterior_concentrates_from_pilot_evidence(self):
        target, descriptors, margins, domains = self.boundary_family_data()
        model = BoundaryFamilyMixturePosterior(
            base_model_kwargs={
                "coordinate": "learned_psi",
                "geometry": "low_rank_psd",
                "rank": 2,
                "hierarchy_iterations": 4,
                "effect_ridge": 0.1,
                "upper_alpha": 0.05,
            },
            family_delta=0.20,
            evidence_temperature=1.0,
        ).fit(
            descriptors,
            margins,
            domains,
            margin_variance=np.full(len(margins), 0.01),
            replicate_count=np.full(len(margins), 4),
        )
        pilot = np.arange(0, len(target), 4)
        target_margin = 0.05 + 1.05 * (
            target[:, 0] + 4.0 * target[:, 1])
        adapter = model.fit_target_adapter(
            target[pilot],
            target_margin[pilot],
            pilot_variance=np.full(len(pilot), 0.01),
            replicate_count=np.full(len(pilot), 4),
        )
        weights = np.asarray(adapter.posterior_weights)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertTrue(np.all(weights > 0.0))
        preferred = model.family_labels_.index("leave_out:negative")
        self.assertEqual(int(np.argmax(weights)), preferred)
        self.assertGreater(weights[preferred], model.source_prior_weights_[preferred])
        self.assertLess(
            adapter.diagnostics["effective_family_count"], len(weights))
        self.assertFalse(adapter.diagnostics["target_label_used"])
        self.assertFalse(adapter.diagnostics["target_oracle_used"])

    def test_boundary_family_certificate_envelopes_credible_families(self):
        target, descriptors, margins, domains = self.boundary_family_data()
        model = BoundaryFamilyMixturePosterior(
            base_model_kwargs={
                "coordinate": "learned_psi",
                "geometry": "low_rank_psd",
                "rank": 2,
                "hierarchy_iterations": 3,
                "upper_alpha": 0.05,
            },
            family_delta=0.20,
            evidence_temperature=0.5,
            family_guard_scale=0.25,
        ).fit(descriptors, margins, domains)
        pilot = np.arange(0, len(target), 5)
        adapter = model.fit_target_adapter(
            target[pilot],
            0.1 + target[pilot, 0] + 4.0 * target[pilot, 1],
        )
        components = model.predict_components(target[1::6], adapter=adapter)
        included = components["family_upper"][components["credible_indices"]]
        self.assertTrue(np.all(
            components["upper"] + 1e-12 >= np.max(included, axis=0)))
        self.assertTrue(np.all(
            components["family_selection_guard"] >= 0.0))
        self.assertGreaterEqual(
            components["credible_mass"], 1.0 - model.family_delta - 1e-12)

    def test_atomic_boundary_families_select_compatible_source_shapes(self):
        target, descriptors, margins, domains = self.boundary_family_data()
        model = BoundaryFamilyMixturePosterior(
            base_model_kwargs={
                "coordinate": "learned_psi",
                "geometry": "low_rank_psd",
                "rank": 2,
                "hierarchy_iterations": 4,
                "effect_ridge": 0.1,
                "upper_alpha": 0.01,
            },
            family_strategy="source_domain_atoms",
            family_delta=0.025,
            evidence_temperature=1.0,
        ).fit(
            descriptors,
            margins,
            domains,
            margin_variance=np.full(len(margins), 0.01),
            replicate_count=np.full(len(margins), 4),
        )
        pilot = np.arange(0, len(target), 4)
        target_margin = 0.05 + 1.05 * (
            target[:, 0] + 4.0 * target[:, 1])
        adapter = model.fit_target_adapter(
            target[pilot],
            target_margin[pilot],
            pilot_variance=np.full(len(pilot), 0.01),
            replicate_count=np.full(len(pilot), 4),
        )
        labels = model.family_labels_
        incompatible = labels.index("source_atom:negative")
        compatible = [
            labels.index("source_atom:positive_a"),
            labels.index("source_atom:positive_b"),
        ]
        self.assertLess(adapter.posterior_weights[incompatible], 1e-6)
        self.assertGreater(
            float(np.sum(adapter.posterior_weights[compatible])), 0.99)
        self.assertNotIn(
            "source_atom:negative",
            adapter.diagnostics["credible_family_labels"],
        )
        components = model.predict_components(target, adapter=adapter)
        self.assertGreater(int(np.sum(components["upper"] <= 0.0)), 0)
        self.assertTrue(all(
            family.diagnostics()["single_source_domain_family"]
            for family in model.families_
        ))

    def test_single_family_mixture_is_exact_pooled_fallback(self):
        descriptors, margins, domains = self.source_data()
        model = BoundaryFamilyMixturePosterior(
            base_model_kwargs={
                "coordinate": "learned_psi",
                "geometry": "linear_monotone",
                "rank": 1,
                "hierarchy_iterations": 3,
            },
            family_guard_scale=1.0,
        ).fit(descriptors, margins, domains)
        self.assertEqual(model.diagnostics()["family_count"], 1)
        pilot = np.arange(0, 20, 2)
        adapter = model.fit_target_adapter(
            descriptors[pilot], margins[pilot] + 0.2)
        family = model.families_[0]
        family_adapter = adapter.family_adapters[0]
        evaluation = descriptors[20:30]
        np.testing.assert_allclose(
            model.predict(evaluation, adapter=adapter),
            family.predict(evaluation, adapter=family_adapter),
        )
        np.testing.assert_allclose(
            model.predict_upper(evaluation, adapter=adapter),
            family.predict_upper(evaluation, adapter=family_adapter),
        )

    def test_boundary_family_synthesis_learns_nonnegative_combination(self):
        target, descriptors, margins, domains = self.boundary_family_data()
        model = BoundaryFamilySynthesisPosterior(
            base_model_kwargs={
                "coordinate": "learned_psi",
                "geometry": "low_rank_psd",
                "rank": 2,
                "hierarchy_iterations": 4,
                "effect_ridge": 0.1,
                "upper_alpha": 0.05,
            },
            coefficient_ridge=0.1,
            coefficient_prior_strength=0.25,
        ).fit(
            descriptors,
            margins,
            domains,
            margin_variance=np.full(len(margins), 0.01),
            replicate_count=np.full(len(margins), 4),
        )
        positive = target[:, 0] + 4.0 * target[:, 1]
        negative = target[:, 0] - 4.0 * target[:, 1]
        target_margin = 0.05 + 0.55 * positive + 0.45 * negative
        pilot = np.arange(0, len(target), 3)
        prior_error = float(np.mean((
            model.predict(target) - target_margin) ** 2))
        adapter = model.fit_target_adapter(
            target[pilot],
            target_margin[pilot],
            pilot_variance=np.full(len(pilot), 0.01),
            replicate_count=np.full(len(pilot), 4),
        )
        prediction = model.predict(target, adapter=adapter)
        upper = model.predict_upper(target, adapter=adapter)
        adapted_error = float(np.mean((prediction - target_margin) ** 2))
        self.assertLess(adapted_error, prior_error)
        self.assertTrue(np.all(adapter.coefficients >= -1e-12))
        self.assertTrue(np.all(upper >= prediction))
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(adapter.output_covariance))),
            -1e-10,
        )
        self.assertEqual(
            adapter.effective_dimension,
            model.diagnostics()["family_count"] + 1,
        )
        self.assertTrue(adapter.diagnostics["source_dictionary_frozen"])
        self.assertFalse(adapter.diagnostics["target_oracle_used"])

    def test_semiparametric_boundary_residual_is_frozen_and_orthogonal(self):
        target, descriptors, margins, domains = self.boundary_family_data()
        common = {
            "base_model_kwargs": {
                "coordinate": "learned_psi",
                "geometry": "low_rank_psd",
                "rank": 2,
                "hierarchy_iterations": 4,
                "effect_ridge": 0.1,
                "upper_alpha": 0.05,
            },
            "coefficient_ridge": 0.1,
            "coefficient_prior_strength": 0.25,
        }
        synthesis = BoundaryFamilySynthesisPosterior(**common).fit(
            descriptors, margins, domains)
        semiparametric = BoundaryFamilySemiparametricPosterior(
            **common,
            residual_feature_count=2,
            residual_ridge=1.0,
        ).fit(descriptors, margins, domains)
        positive = target[:, 0] + 4.0 * target[:, 1]
        negative = target[:, 0] - 4.0 * target[:, 1]
        target_margin = (
            0.05 + 0.55 * positive + 0.45 * negative
            + 0.25 * np.sin(2.5 * target[:, 0])
        )
        pilot = np.arange(0, len(target), 2)
        synthesis_adapter = synthesis.fit_target_adapter(
            target[pilot], target_margin[pilot])
        semiparametric_adapter = semiparametric.fit_target_adapter(
            target[pilot], target_margin[pilot])
        synthesis_error = float(np.mean((
            synthesis.predict(target, synthesis_adapter) - target_margin
        ) ** 2))
        semiparametric_error = float(np.mean((
            semiparametric.predict(target, semiparametric_adapter)
            - target_margin
        ) ** 2))
        self.assertLess(semiparametric_error, synthesis_error)
        self.assertLess(
            semiparametric.diagnostics()["orthogonality_relative"], 1e-10)
        self.assertEqual(
            len(semiparametric_adapter.residual_coefficients), 2)
        self.assertTrue(semiparametric_adapter.diagnostics[
            "orthogonal_residual_dictionary_frozen"])
        self.assertTrue(np.all(
            semiparametric.predict_upper(
                target, semiparametric_adapter)
            >= semiparametric.predict(target, semiparametric_adapter)
        ))

    def test_source_boundary_pilot_selection_never_reads_target_outcomes(self):
        descriptors, margins, domains = self.source_data()
        model = TransferableChanceBoundaryPosterior(
            coordinate="hybrid_explicit_latent",
            geometry="rbf",
            adaptation="shift_scale",
            rank=2,
        ).fit(descriptors, margins, domains)
        first, first_diag = _source_boundary_pilot_indices(
            model, descriptors, 10)
        second, second_diag = _source_boundary_pilot_indices(
            model, descriptors, 10)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(
            first_diag["target_outcomes_used_for_selection"])
        self.assertFalse(second_diag["target_oracle_used"])
        self.assertGreaterEqual(first_diag["design_rank"], 3)

        robust, robust_diag = _source_boundary_pilot_indices(
            model, descriptors, 10, robust=True)
        source_score = model.predict(descriptors)
        self.assertTrue(robust_diag["robust_support_clipping"])
        self.assertGreaterEqual(
            float(np.min(source_score[robust])),
            robust_diag["source_support_lower"] - 1e-12,
        )
        self.assertLessEqual(
            float(np.max(source_score[robust])),
            robust_diag["source_support_upper"] + 1e-12,
        )

    def test_scheduler_builds_exactly_96_source_only_tasks(self):
        args = argparse.Namespace(
            deploy=Path("/deploy"),
            python="/env/bin/python",
            run_id="tcb_test",
            nodes=",".join(SUBMIT.CPU_NODES),
            cpu=12,
            ram_mb=4096,
            records_per_domain=96,
            pilot_size=10,
            evaluation_size=48,
            pool_multiplier=4,
            jobs=5,
            data_seed=33001,
            allow_duplicate=False,
        )
        specs = SUBMIT.build_specs(args)
        self.assertEqual(len(specs), 96)
        self.assertEqual(len({row["signature"] for row in specs}), 96)
        self.assertEqual(
            {row["require_node"] for row in specs}, set(SUBMIT.CPU_NODES))
        self.assertTrue(all(
            "SCOLHKG_OFFLINE=1" in row["cmd"]
            and "checkpoints" in row["stage_excludes"]
            and row["allow_no_resume"]
            for row in specs
        ))
        self.assertTrue(all(row["cpu"] == 12 for row in specs))
        self.assertTrue(all(
            "--jobs 5" in row["cmd"]
            and "OPENBLAS_NUM_THREADS=2" in row["cmd"]
            for row in specs
        ))

    def test_scheduler_selects_stable_diagnostic_subset(self):
        args = argparse.Namespace(
            deploy=Path("/deploy"),
            python="/env/bin/python",
            run_id="tcb_test",
            nodes=",".join(SUBMIT.CPU_NODES),
            cpu=12,
            ram_mb=4096,
            records_per_domain=96,
            pilot_size=10,
            evaluation_size=48,
            pool_multiplier=4,
            jobs=5,
            data_seed=33001,
            allow_duplicate=False,
            task_indices="0",
        )
        specs = SUBMIT.build_specs(args)
        self.assertEqual(len(specs), 1)
        self.assertIn(
            "explicit_stable__linear_monotone__frozen__r2",
            specs[0]["signature"],
        )


if __name__ == "__main__":
    unittest.main()
