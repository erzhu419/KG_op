import unittest

import numpy as np

from representation.meta_prior import LearnedMetaPrior
from representation.source_boundary_episodes import SourceBoundaryEpisodePrior


class _BoundsOnlyTarget:
    d = 6
    L = 100

    @staticmethod
    def sample_random(rng):
        return tuple(int(value) for value in rng.integers(0, 101, size=6))


class SourceBoundaryEpisodePriorTests(unittest.TestCase):
    def test_disjoint_source_episodes_admit_supported_alignment(self):
        rng = np.random.default_rng(91)
        domains = []
        margins = []
        baseline = []
        aligned = []
        for domain, shift in (("source_a", -0.05), ("source_b", 0.05)):
            value = np.linspace(-1.2, 1.2, 80) + shift
            domains.extend([domain] * len(value))
            margins.extend(value)
            baseline.extend(rng.normal(0.0, 1.0, size=(len(value), 2)))
            aligned.extend(np.column_stack([
                value + rng.normal(0.0, 0.01, len(value)),
                value ** 2,
            ]))
        prior = SourceBoundaryEpisodePrior(
            pilot_size=10,
            evaluation_size=24,
            episodes_per_domain=6,
            ridge=0.1,
            support_multiplier=3.0,
            seed=92,
        ).fit(domains, margins, baseline, aligned)
        diagnostics = prior.diagnostics()
        self.assertEqual(diagnostics["status"], "fit")
        self.assertTrue(diagnostics["all_splits_disjoint"])
        self.assertEqual(diagnostics["n_domains"], 2)
        self.assertGreater(diagnostics["evaluation_win_rate"], 0.60)
        self.assertTrue(all(
            set(row["pilot_indices"]).isdisjoint(row["evaluation_indices"])
            for row in prior.episode_rows_
        ))

        target_margin = np.concatenate([
            np.asarray([-0.10]), np.linspace(0.05, 1.0, 11)
        ])
        baseline_loo = np.zeros_like(target_margin)
        aligned_loo = target_margin + 0.01 * np.sin(np.arange(12))
        decision = prior.admit(target_margin, baseline_loo, aligned_loo)
        self.assertTrue(decision.accepted, decision.diagnostics)
        self.assertFalse(decision.diagnostics["target_oracle_used"])
        self.assertGreater(
            decision.diagnostics["source_gain_lower_quartile"], 0.0)

    def test_target_pilot_harm_is_rejected(self):
        value = np.tile(np.linspace(-1.0, 1.0, 48), 2)
        domains = np.asarray(["a"] * 48 + ["b"] * 48)
        baseline = np.column_stack([value, value ** 2])
        aligned = baseline.copy()
        prior = SourceBoundaryEpisodePrior(
            pilot_size=8,
            evaluation_size=16,
            episodes_per_domain=3,
            seed=93,
        ).fit(domains, value, baseline, aligned)
        target = np.linspace(-0.8, 0.8, 10)
        decision = prior.admit(target, target, -target)
        self.assertFalse(decision.accepted)
        self.assertIn(
            "target_pilot_false_feasible_risk",
            decision.diagnostics["rejection_reasons"],
        )

    def test_one_sided_source_history_cannot_fit_prior(self):
        margins = np.linspace(0.1, 2.0, 40)
        features = np.column_stack([margins, margins ** 2])
        prior = SourceBoundaryEpisodePrior().fit(
            ["only_source"] * len(margins),
            margins,
            features,
            features,
        )
        self.assertEqual(
            prior.diagnostics()["status"],
            "insufficient_source_boundary_episodes",
        )

    def test_admission_switch_does_not_change_source_only_proposals(self):
        prior = LearnedMetaPrior(
            component_stage="spectral",
            spectral_risk_alignment=True,
            spectral_alignment_source_episodes=6,
            spectral_alignment_admission=False,
            seed=94,
        )
        prior.alignment_profile_templates = [
            {
                "profile": np.asarray(profile, dtype=float),
                "domain": domain,
                "feasible": feasible,
                "scaled_margin": margin,
                "origin": "source_boundary",
            }
            for profile, domain, feasible, margin in (
                ([0.1, 0.2, 0.3], "source_a", True, -0.05),
                ([0.8, 0.7, 0.6], "source_a", False, 0.08),
                ([0.2, 0.4, 0.6], "source_b", True, -0.07),
                ([0.9, 0.5, 0.1], "source_b", False, 0.06),
            )
        ]
        target = _BoundsOnlyTarget()
        control = prior.proposal_candidates(
            target, n=8, rng=np.random.default_rng(95))
        prior.spectral_alignment_admission = True
        challenger = prior.proposal_candidates(
            target, n=8, rng=np.random.default_rng(95))
        self.assertEqual(control, challenger)
        self.assertEqual(len(control), 8)
        self.assertTrue(all(len(row) == target.d for row in control))

    def test_latent_inverse_uses_source_coordinates_without_target_labels(self):
        prior = LearnedMetaPrior(
            component_stage="spectral",
            spectral_alignment_latent_proposals=True,
            spectral_alignment_inverse_pool_size=64,
            seed=96,
        )
        prior.risk_subspace_alignment = object()
        prior.alignment_profile_templates = [
            {
                "profile": np.asarray([0.1, 0.2, 0.3]),
                "domain": "source_a",
                "feasible": True,
                "scaled_margin": -0.05,
                "origin": "source_boundary",
                "aligned_coordinate": np.asarray([0.10]),
            },
            {
                "profile": np.asarray([0.8, 0.7, 0.6]),
                "domain": "source_b",
                "feasible": False,
                "scaled_margin": 0.08,
                "origin": "source_boundary",
                "aligned_coordinate": np.asarray([0.80]),
            },
        ]
        prior.frozen_risk_aligned_coordinate = (
            lambda problem, x: np.asarray([float(x[0]) / problem.L]))
        target = _BoundsOnlyTarget()
        first = prior.alignment_latent_candidates(
            target, n=6, rng=np.random.default_rng(97), pool_size=64)
        second = prior.alignment_latent_candidates(
            target, n=6, rng=np.random.default_rng(97), pool_size=64)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertLessEqual(
            min(abs(float(row[0]) / target.L - 0.10) for row in first),
            0.01,
        )
        diagnostics = prior.alignment_episode_diagnostics[
            "last_latent_inverse"]
        self.assertFalse(diagnostics["target_labels_used"])
        self.assertFalse(diagnostics["target_oracle_used"])

    def test_latent_proposal_gate_uses_source_episode_win_rate_only(self):
        prior = LearnedMetaPrior(seed=98)
        source = SourceBoundaryEpisodePrior(min_global_win_rate=2.0 / 3.0)
        source.diagnostics_ = {
            "status": "fit",
            "evaluation_win_rate": 0.60,
            "target_data_used": False,
            "target_oracle_used": False,
        }
        prior.alignment_episode_prior = source
        self.assertFalse(prior.alignment_latent_proposal_supported())
        source.diagnostics_["evaluation_win_rate"] = 0.75
        self.assertTrue(prior.alignment_latent_proposal_supported())
        diagnostics = prior.alignment_episode_diagnostics[
            "latent_proposal_source_gate"]
        self.assertFalse(diagnostics["target_data_used"])
        self.assertFalse(diagnostics["target_oracle_used"])


if __name__ == "__main__":
    unittest.main()
