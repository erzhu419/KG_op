import copy
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpr import ParametricGPR  # noqa: E402
from representation.adaptive_sparsity import (  # noqa: E402
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
