import copy
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpr import ParametricGPR  # noqa: E402
from representation.adaptive_sparsity import (  # noqa: E402
    AdaptiveGroupRidgePosterior,
    AdaptiveSpikeSlabPosterior,
)


class _IdentityBasis:
    feature_dim = 6

    @staticmethod
    def features(x):
        return np.asarray(x, dtype=float)

    @staticmethod
    def features_many(X):
        return np.asarray(X, dtype=float)

    def record_adaptive_sparsity_diagnostics(self, diagnostics):
        self.last_adaptive_diagnostics = dict(diagnostics)


class _PaddedBasis(_IdentityBasis):
    feature_dim = 10

    @staticmethod
    def features(x):
        return np.concatenate([np.asarray(x, dtype=float), np.zeros(4)])

    @staticmethod
    def features_many(X):
        rows = np.asarray(X, dtype=float)
        return np.hstack([rows, np.zeros((len(rows), 4))])


class AdaptiveSparsityTests(unittest.TestCase):
    def _data(self):
        rng = np.random.default_rng(9102)
        X = rng.normal(size=(48, 6))
        y = 2.5 * X[:, 0] + 0.15 * rng.normal(size=len(X))
        return X, y

    def _spec(self):
        return {
            "source_pip": np.full(6, 0.15),
            "source_slab_scale": np.ones(6),
            "min_pip": 0.05,
            "max_pip": 0.95,
            "spike_ratio": 0.08,
            "damping": 0.5,
            "max_iter": 50,
            "tolerance": 1e-6,
            "residual_floor_scale": 0.05,
        }

    def test_target_evidence_can_escape_a_wrong_source_spike(self):
        X, y = self._data()
        posterior = AdaptiveSpikeSlabPosterior(
            self._spec()["source_pip"],
            self._spec()["source_slab_scale"],
            spike_ratio=0.08,
            residual_floor_scale=0.05,
        ).fit(
            X,
            y,
            np.full(len(X), 0.15 ** 2),
            [(i,) for i in range(len(X))],
            deviation_variance=0.005,
        )
        pip = np.asarray(posterior.diagnostics()["posterior_pip"])
        self.assertGreater(pip[0], 0.90)
        self.assertLess(float(np.max(pip[1:])), 0.10)
        self.assertEqual(
            posterior.diagnostics()["escaped_source_spike_count"], 1)
        self.assertLessEqual(
            posterior.diagnostics()["effective_dimension"],
            posterior.diagnostics()["max_effective_dimension"] + 1e-8,
        )

    def test_mask_uncertainty_is_nonnegative_and_reproducible(self):
        X, y = self._data()
        kwargs = dict(
            features=X,
            response=y,
            noise_variance=np.full(len(X), 0.15 ** 2),
            sample_keys=[(i,) for i in range(len(X))],
            deviation_variance=0.005,
        )
        first = AdaptiveSpikeSlabPosterior(
            np.full(6, 0.15), np.ones(6), spike_ratio=0.08
        ).fit(**kwargs)
        second = AdaptiveSpikeSlabPosterior(
            np.full(6, 0.15), np.ones(6), spike_ratio=0.08
        ).fit(**kwargs)
        floor = first.mask_uncertainty(X)
        self.assertTrue(np.all(floor >= 0.0))
        np.testing.assert_allclose(
            first.result_.mean, second.result_.mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            first.result_.covariance,
            second.result_.covariance,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_shared_group_shrinkage_is_rotation_equivariant(self):
        rng = np.random.default_rng(91021)
        latent = rng.normal(size=(80, 2))
        X = np.column_stack([
            0.2 * latent[:, 0],
            3.0 * latent[:, 1],
            np.zeros(len(latent)),
            np.zeros(len(latent)),
        ])
        direction = np.asarray([1.5, -0.75, 0.0, 0.0], dtype=float)
        y = X @ direction + 0.08 * rng.normal(size=len(X))
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        common = dict(
            response=y,
            noise_variance=np.full(len(X), 0.08 ** 2),
            sample_keys=[(index,) for index in range(len(X))],
            deviation_variance=1e-6,
        )
        kwargs = dict(
            source_pip=np.asarray([0.15, 0.25, 0.35, 0.45]),
            source_slab_scale=np.asarray([0.5, 0.75, 1.0, 1.25]),
            spike_ratio=0.05,
            shared_shrinkage_groups=[0, 0, 0, 0],
        )
        original = AdaptiveSpikeSlabPosterior(**kwargs).fit(
            features=X, **common)
        rotated = AdaptiveSpikeSlabPosterior(**kwargs).fit(
            features=X @ rotation, **common)

        probe_latent = rng.normal(size=(21, 2))
        probes = np.column_stack([
            0.2 * probe_latent[:, 0],
            3.0 * probe_latent[:, 1],
            np.zeros(len(probe_latent)),
            np.zeros(len(probe_latent)),
        ])
        np.testing.assert_allclose(
            original.predict_parametric_mean(probes),
            rotated.predict_parametric_mean(probes @ rotation),
            rtol=1e-7,
            atol=1e-8,
        )
        original_pip = np.asarray(
            original.diagnostics()["posterior_pip"], dtype=float)
        rotated_pip = np.asarray(
            rotated.diagnostics()["posterior_pip"], dtype=float)
        np.testing.assert_allclose(original_pip, original_pip[0], atol=1e-12)
        np.testing.assert_allclose(rotated_pip, rotated_pip[0], atol=1e-12)
        np.testing.assert_allclose(original_pip, rotated_pip, atol=1e-10)
        group = original.diagnostics()["shared_shrinkage_groups"]["0"]
        self.assertTrue(original.diagnostics()["shared_shrinkage_active"])
        self.assertEqual(group["indices"], [0, 1, 2, 3])
        self.assertAlmostEqual(
            group["effective_dimension"], 4.0 * original_pip[0])

    def test_negative_group_ids_preserve_default_sparse_posterior(self):
        X, y = self._data()
        common = dict(
            features=X,
            response=y,
            noise_variance=np.full(len(X), 0.15 ** 2),
            sample_keys=[(index,) for index in range(len(X))],
            deviation_variance=0.005,
        )
        baseline = AdaptiveSpikeSlabPosterior(
            np.full(6, 0.15), np.ones(6), spike_ratio=0.08
        ).fit(**common)
        explicit = AdaptiveSpikeSlabPosterior(
            np.full(6, 0.15),
            np.ones(6),
            spike_ratio=0.08,
            shared_shrinkage_groups=[-1] * 6,
        ).fit(**common)
        np.testing.assert_allclose(
            baseline.result_.mean, explicit.result_.mean, atol=1e-12)
        np.testing.assert_allclose(
            baseline.result_.covariance,
            explicit.result_.covariance,
            atol=1e-12,
        )
        self.assertFalse(
            explicit.diagnostics()["shared_shrinkage_active"])

    def test_nested_loo_group_ridge_learns_signal_group_complexity(self):
        rng = np.random.default_rng(91022)
        X = rng.normal(size=(24, 6))
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1]
        y += 0.08 * rng.normal(size=len(X))
        posterior = AdaptiveGroupRidgePosterior(
            [0, 0, 0, 1, 1, 1],
            penalty_grid=(0.01, 1.0, 100.0),
            coordinate_passes=2,
        ).fit(
            X,
            y,
            np.full(len(X), 0.08 ** 2),
            [(index,) for index in range(len(X))],
            deviation_variance=1e-6,
        )
        diag = posterior.diagnostics()
        groups = diag["groups"]
        self.assertEqual(diag["method"], "nested_loo_group_ridge")
        self.assertFalse(diag["oracle_used"])
        self.assertLess(
            groups["0"]["selected_penalty"],
            groups["1"]["selected_penalty"],
        )
        self.assertTrue(diag["complexity_selection_valid"])
        self.assertLessEqual(diag["effective_dimension"], 6.0 + 1e-8)
        self.assertTrue(np.all(posterior.mask_uncertainty(X) >= 0.0))

    def test_nested_loo_group_ridge_is_rotation_equivariant(self):
        rng = np.random.default_rng(91023)
        latent = rng.normal(size=(28, 2))
        X = np.column_stack([
            0.2 * latent[:, 0],
            3.0 * latent[:, 1],
            np.zeros(len(latent)),
            np.zeros(len(latent)),
        ])
        y = 1.2 * X[:, 0] - 0.7 * X[:, 1]
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        common = dict(
            response=y,
            noise_variance=np.full(len(X), 0.05 ** 2),
            sample_keys=[(index,) for index in range(len(X))],
            deviation_variance=1e-6,
        )
        kwargs = dict(
            group_ids=[0, 0, 0, 0],
            penalty_grid=(0.01, 1.0, 100.0),
        )
        original = AdaptiveGroupRidgePosterior(**kwargs).fit(
            features=X, **common)
        rotated = AdaptiveGroupRidgePosterior(**kwargs).fit(
            features=X @ rotation, **common)
        probe_latent = rng.normal(size=(17, 2))
        probes = np.column_stack([
            0.2 * probe_latent[:, 0],
            3.0 * probe_latent[:, 1],
            np.zeros(len(probe_latent)),
            np.zeros(len(probe_latent)),
        ])
        np.testing.assert_allclose(
            original.predict_parametric_mean(probes),
            rotated.predict_parametric_mean(probes @ rotation),
            rtol=1e-7,
            atol=1e-8,
        )
        self.assertEqual(
            original.diagnostics()["groups"]["0"]["selected_penalty"],
            rotated.diagnostics()["groups"]["0"]["selected_penalty"],
        )

    def test_adaptive_dictionary_strictly_nests_stage1_prefix(self):
        rng = np.random.default_rng(9103)
        X = rng.normal(size=(48, 6))
        y = 1.2 * X[:, 0] + 2.5 * X[:, 3] + 0.1 * rng.normal(size=len(X))
        posterior = AdaptiveSpikeSlabPosterior(
            np.full(6, 0.15),
            np.ones(6),
            spike_ratio=0.08,
            always_active_count=2,
        ).fit(
            X,
            y,
            np.full(len(X), 0.1 ** 2),
            [(i,) for i in range(len(X))],
            deviation_variance=0.005,
        )
        diag = posterior.diagnostics()
        pip = np.asarray(diag["posterior_pip"])
        np.testing.assert_allclose(pip[:2], 1.0, atol=0.0)
        self.assertEqual(diag["always_active_count"], 2)
        self.assertGreaterEqual(diag["adaptive_active_count_0_5"], 1)
        self.assertGreaterEqual(diag["active_count_0_5"], 3)

    def test_effective_fraction_caps_total_including_fixed_prefix(self):
        rng = np.random.default_rng(91031)
        X = rng.normal(size=(20, 20))
        y = X[:, 0] + X[:, 7] + 0.1 * rng.normal(size=len(X))
        posterior = AdaptiveSpikeSlabPosterior(
            np.full(20, 0.5),
            np.ones(20),
            min_pip=0.05,
            spike_ratio=0.05,
            always_active_count=4,
            max_effective_fraction=0.35,
        ).fit(
            X,
            y,
            np.full(len(X), 0.1 ** 2),
            [(i,) for i in range(len(X))],
            deviation_variance=0.005,
        )
        diag = posterior.diagnostics()
        self.assertAlmostEqual(diag["max_effective_dimension"], 7.0)
        self.assertLessEqual(diag["effective_dimension"], 7.0 + 1e-8)

    def test_group_bayes_factor_cannot_escape_total_dimension_cap(self):
        rng = np.random.default_rng(91032)
        X = rng.normal(size=(20, 11))
        y = 20.0 * np.sum(X[:, -3:], axis=1)
        posterior = AdaptiveSpikeSlabPosterior(
            np.full(11, 0.5),
            np.ones(11),
            min_pip=0.05,
            max_pip=0.95,
            spike_ratio=0.05,
            always_active_count=4,
            max_effective_fraction=0.35,
            shared_shrinkage_groups=[-1] * 4 + [0] * 4 + [1] * 3,
        ).fit(
            X,
            y,
            np.full(len(X), 0.01 ** 2),
            [(index,) for index in range(len(X))],
            deviation_variance=1e-6,
        )
        diag = posterior.diagnostics()
        groups = diag["shared_shrinkage_groups"]
        self.assertGreater(groups["1"]["log_bayes_factor"], 50.0)
        self.assertLessEqual(
            diag["effective_dimension"],
            diag["max_effective_dimension"] + 1e-10,
        )
        self.assertTrue(diag["effective_dimension_budget_respected"])
        self.assertGreaterEqual(
            diag["effective_dimension_budget_slack"], -1e-10)
        pip = np.asarray(diag["posterior_pip"], dtype=float)
        np.testing.assert_allclose(pip[4:8], pip[4], atol=1e-12)
        np.testing.assert_allclose(pip[8:11], pip[8], atol=1e-12)

    def test_pilot_admitted_support_cannot_expand_during_refit(self):
        rng = np.random.default_rng(9104)
        X = rng.normal(size=(48, 6))
        y = 4.0 * X[:, 3] + 0.05 * rng.normal(size=len(X))
        allowed = np.asarray([True, True, False, False, True, False])
        posterior = AdaptiveSpikeSlabPosterior(
            np.full(6, 0.5),
            np.ones(6),
            min_pip=0.05,
            spike_ratio=0.05,
            always_active_count=2,
            allowed_mask=allowed,
        ).fit(
            X,
            y,
            np.full(len(X), 0.05 ** 2),
            [(i,) for i in range(len(X))],
            deviation_variance=0.0025,
        )
        diag = posterior.diagnostics()
        pip = np.asarray(diag["posterior_pip"])
        self.assertAlmostEqual(float(pip[3]), 0.05, places=12)
        self.assertEqual(diag["allowed_adaptive_count"], 1)
        self.assertEqual(diag["frozen_out_adaptive_count"], 3)

    def test_gpr_refits_from_history_and_fantasy_clone_is_isolated(self):
        X, y = self._data()
        rows = [tuple(np.round(10.0 * row).astype(int)) for row in X[:17]]
        model = ParametricGPR(6, basis_map=_IdentityBasis())
        model.enable_adaptive_sparsity(
            self._spec(),
            rows[:16],
            y[:16],
            np.full(16, 0.15 ** 2),
            deviation_variance=0.005,
        )
        self.assertTrue(model.adaptive_sparsity_enabled())
        before_mean = model.a.copy()
        before_cov = model.C.copy()
        model._refit_adaptive_sparsity()
        np.testing.assert_allclose(model.a, before_mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(model.C, before_cov, rtol=1e-12, atol=1e-12)

        clone = copy.deepcopy(model)
        clone.update(rows[16], y[16], 0.15 ** 2)
        self.assertEqual(len(model._adaptive_records), 16)
        self.assertEqual(len(clone._adaptive_records), 17)
        self.assertFalse(np.allclose(clone.a[: model.p], model.a[: model.p]))

        x = rows[0]
        feature = model.augmented_feature(x)
        conditional = float(feature @ model.C @ feature)
        total = model.posterior_var(x)
        self.assertGreaterEqual(total, conditional)
        self.assertGreaterEqual(model.adaptive_model_uncertainty(x), 0.0)

    def test_gpr_passes_shared_groups_to_fantasy_clone(self):
        X, y = self._data()
        rows = [tuple(np.round(10.0 * row).astype(int)) for row in X[:17]]
        spec = self._spec()
        spec["shared_shrinkage_groups"] = [0, 0, 0, 1, 1, 1]
        model = ParametricGPR(6, basis_map=_IdentityBasis())
        model.enable_adaptive_sparsity(
            spec,
            rows[:16],
            y[:16],
            np.full(16, 0.15 ** 2),
            deviation_variance=0.005,
        )
        self.assertTrue(
            model.adaptive_sparsity_diagnostics()[
                "shared_shrinkage_active"])
        clone = copy.deepcopy(model)
        clone.update(rows[16], y[16], 0.15 ** 2)
        groups = clone.adaptive_sparsity_diagnostics()[
            "shared_shrinkage_groups"]
        self.assertEqual(groups["0"]["indices"], [0, 1, 2])
        self.assertEqual(groups["1"]["indices"], [3, 4, 5])
        self.assertGreater(
            groups["0"]["posterior_pip"],
            groups["1"]["posterior_pip"] + 0.5,
        )

    def test_gpr_group_ridge_reselects_complexity_in_fantasy_clone(self):
        X, y = self._data()
        rows = [tuple(np.round(10.0 * row).astype(int)) for row in X[:17]]
        spec = {
            "method": "nested_loo_group_ridge",
            "group_ids": [0, 0, 0, 1, 1, 1],
            "penalty_grid": [0.01, 1.0, 100.0],
            "initial_feature_penalty": [1.0] * 6,
            "coordinate_passes": 2,
            "safety_weight": 2.0,
            "residual_floor_scale": 0.05,
            "dictionary_dim": 6,
        }
        model = ParametricGPR(6, basis_map=_IdentityBasis())
        model.enable_adaptive_sparsity(
            spec,
            rows[:16],
            y[:16],
            np.full(16, 0.15 ** 2),
            deviation_variance=0.005,
        )
        before = model.adaptive_sparsity_diagnostics()
        self.assertEqual(before["method"], "nested_loo_group_ridge")
        self.assertTrue(before["complexity_selection_valid"])
        clone = copy.deepcopy(model)
        clone.update(rows[16], y[16], 0.15 ** 2)
        after = clone.adaptive_sparsity_diagnostics()
        self.assertEqual(after["n_observations"], 17)
        self.assertGreater(after["models_tested"], 1)
        self.assertEqual(model.adaptive_sparsity_diagnostics(), before)

    def test_gpr_group_ridge_survives_checkpoint_round_trip(self):
        X, y = self._data()
        rows = [tuple(np.round(10.0 * row).astype(int)) for row in X[:17]]
        spec = {
            "method": "nested_loo_group_ridge",
            "group_ids": [0, 0, 0, 1, 1, 1],
            "penalty_grid": [0.01, 1.0, 100.0],
            "initial_feature_penalty": [1.0] * 6,
            "coordinate_passes": 2,
            "safety_weight": 2.0,
            "residual_floor_scale": 0.05,
            "dictionary_dim": 6,
        }
        model = ParametricGPR(6, basis_map=_IdentityBasis())
        model.enable_adaptive_sparsity(
            spec,
            rows[:16],
            y[:16],
            np.full(16, 0.15 ** 2),
            deviation_variance=0.005,
        )
        restored = pickle.loads(pickle.dumps(
            model, protocol=pickle.HIGHEST_PROTOCOL))
        np.testing.assert_allclose(restored.a, model.a, atol=0.0)
        np.testing.assert_allclose(restored.C, model.C, atol=0.0)
        self.assertEqual(
            restored.adaptive_sparsity_diagnostics(),
            model.adaptive_sparsity_diagnostics(),
        )

        restored.update(rows[16], y[16], 0.15 ** 2)
        after = restored.adaptive_sparsity_diagnostics()
        self.assertEqual(after["n_observations"], 17)
        self.assertTrue(after["complexity_selection_valid"])
        self.assertEqual(len(model._adaptive_records), 16)

    def test_gpr_excludes_fixed_padding_from_sparse_dictionary(self):
        X, y = self._data()
        rows = [tuple(np.round(10.0 * row).astype(int)) for row in X[:16]]
        spec = self._spec()
        spec["dictionary_dim"] = 6
        model = ParametricGPR(6, basis_map=_PaddedBasis())
        model.enable_adaptive_sparsity(
            spec,
            rows,
            y[:16],
            np.full(16, 0.15 ** 2),
            deviation_variance=0.005,
        )
        self.assertEqual(
            model.adaptive_sparsity_diagnostics()["dictionary_dim"], 6)
        np.testing.assert_allclose(model.a[7:model.p], 0.0, atol=1e-12)
        self.assertEqual(len(model.a), model.p + len(model.sampled_set))


if __name__ == "__main__":
    unittest.main()
