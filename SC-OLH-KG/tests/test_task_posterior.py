import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.task_posterior import (  # noqa: E402
    FiniteTaskModelEnsemble,
    FiniteTaskPosterior,
    TaskExpertState,
)
from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)


class FiniteTaskPosteriorTests(unittest.TestCase):
    def test_predictive_evidence_moves_mass_to_supported_expert(self):
        posterior = FiniteTaskPosterior(
            ["supported", "misspecified", "null"],
            [0.4, 0.4, 0.2],
            temperature=1.0,
            temperature_decay=0.0,
            output_score_weights=[0.25, 1.0],
        )
        before = posterior.posterior_weights()
        update = posterior.update_from_predictive(
            [0.05, -0.02],
            means=np.array([
                [0.0, 0.0],
                [1.5, 1.0],
                [0.2, 0.3],
            ]),
            epistemic_vars=np.full((3, 2), 0.02),
            aleatoric_vars=np.full((3, 2), 0.01),
            tau=0.0,
        )
        after = posterior.posterior_weights()
        self.assertGreater(after[0], before[0])
        self.assertLess(after[1], before[1])
        self.assertAlmostEqual(float(np.sum(after)), 1.0, places=12)
        self.assertFalse(update["target_oracle_used"])

    def test_hierarchical_variance_is_within_plus_between(self):
        posterior = FiniteTaskPosterior(
            ["left", "right"], [0.25, 0.75])
        moments = posterior.mixture_moments(
            means=np.array([[0.0, 2.0], [2.0, 4.0]]),
            epistemic_vars=np.array([[1.0, 0.5], [3.0, 1.5]]),
            aleatoric_vars=np.array([[0.2, 0.4], [0.6, 0.8]]),
        )
        np.testing.assert_allclose(
            moments.epistemic,
            moments.within_epistemic + moments.between_mean,
        )
        np.testing.assert_allclose(
            moments.total,
            moments.epistemic + moments.aleatoric,
        )
        np.testing.assert_allclose(moments.between_mean, [0.75, 0.75])

    def test_kl_robust_moments_cannot_be_more_optimistic(self):
        posterior = FiniteTaskPosterior(
            ["safe", "unsafe", "null"], [0.6, 0.2, 0.2])
        means = np.array([[-0.3], [0.5], [0.1]])
        epistemic = np.array([[0.02], [0.08], [0.04]])
        aleatoric = np.array([[0.01], [0.12], [0.05]])
        nominal = posterior.robust_mixture_moments(
            means, epistemic, aleatoric, radius=0.0)
        robust = posterior.robust_mixture_moments(
            means, epistemic, aleatoric, radius=0.5)
        self.assertGreaterEqual(robust.mean_upper[0], nominal.mean_upper[0])
        self.assertGreaterEqual(
            robust.epistemic_upper[0], nominal.epistemic_upper[0])
        self.assertGreaterEqual(
            robust.aleatoric_upper[0], nominal.aleatoric_upper[0])

    def test_batched_entropic_dual_dominates_sampled_kl_ball(self):
        posterior = FiniteTaskPosterior(
            ["a", "b", "c"], [0.5, 0.3, 0.2],
            robust_dual_grid_size=65,
        )
        values = np.array([
            [-0.5, 0.2, 1.0],
            [0.4, -0.1, 0.7],
            [1.2, 0.8, -0.2],
        ])
        radius = 0.35
        upper = posterior.kl_robust_expectation(values, radius)
        prior = posterior.posterior_weights()
        rng = np.random.default_rng(77)
        samples = rng.dirichlet(np.ones(3), size=20000)
        kl = np.sum(samples * (
            np.log(np.maximum(samples, 1e-300))
            - np.log(prior[None, :])
        ), axis=1)
        admissible = samples[kl <= radius]
        sampled_values = admissible @ values
        self.assertTrue(np.all(
            np.max(sampled_values, axis=0) <= upper + 1e-10))

    def test_pac_bayes_radius_matches_proved_schedule(self):
        posterior = FiniteTaskPosterior(["a", "b"], [0.5, 0.5])
        ensemble = FiniteTaskModelEnsemble(
            [
                TaskExpertState("a", [], None, None),
                TaskExpertState("b", [], None, None),
            ],
            posterior,
            kl_radius_numerator=0.5,
            confidence_delta=0.05,
            maximum_kl_radius=10.0,
        )
        self.assertAlmostEqual(
            ensemble.effective_kl_radius(),
            0.5 + np.log(20.0),
            places=12,
        )

    def test_proposal_mixture_preserves_prior_support_and_budget(self):
        posterior = FiniteTaskPosterior(
            ["dominant", "rare", "null"], [0.7, 0.2, 0.1],
            temperature=2.0,
            temperature_decay=0.0,
        )
        for _ in range(4):
            posterior.update_from_predictive(
                [0.0, 0.0],
                means=np.array([
                    [0.0, 0.0],
                    [2.0, 2.0],
                    [1.0, 1.0],
                ]),
                epistemic_vars=np.full((3, 2), 0.01),
                aleatoric_vars=np.full((3, 2), 0.01),
                tau=0.0,
            )
        epsilon = 0.2
        proposal = posterior.proposal_weights(exploration=epsilon)
        np.testing.assert_array_less(
            epsilon * posterior.prior_weights() - 1e-15,
            proposal,
        )
        allocation = posterior.proposal_allocation(
            11,
            exploration=epsilon,
            minimum_per_expert=1,
        )
        self.assertEqual(sum(allocation.values()), 11)
        self.assertTrue(all(count >= 1 for count in allocation.values()))

    def test_clone_update_does_not_mutate_original(self):
        posterior = FiniteTaskPosterior(["a", "b"], [0.5, 0.5])
        clone = posterior.clone()
        clone.update_from_predictive(
            [0.0, 0.0],
            means=np.array([[0.0, 0.0], [1.0, 1.0]]),
            epistemic_vars=np.full((2, 2), 0.01),
            aleatoric_vars=np.full((2, 2), 0.01),
            tau=0.0,
        )
        np.testing.assert_allclose(
            posterior.posterior_weights(), [0.5, 0.5])
        self.assertFalse(np.allclose(
            clone.posterior_weights(), posterior.posterior_weights()))

    def test_exact_kg_updates_finite_task_posterior_end_to_end(self):
        def problem(name):
            return ScalarizedProblem(make_problem(
                name, d=5, L=30, sigma=0.04))

        prior = LearnedMetaPrior(
            local_dim=2,
            shared_dim=2,
            component_stage="spectral_hvd",
            spectral_active_dim=3,
            spectral_risk_alignment=True,
            seed=901,
        ).fit_from_source_problems(
            [
                ("InventorySupplyChain", problem("InventorySupplyChain")),
                ("QueueResourceControl", problem("QueueResourceControl")),
            ],
            n_records_per_domain=8,
            rng=np.random.default_rng(902),
        )
        target = MetaPriorProblemAdapter(
            problem("FactorShockStatePolicyRZDT1"), prior)
        for spec in target.task_posterior_expert_specs():
            proposals = target.task_expert_proposal_candidates(
                spec["name"],
                n=3,
                rng=np.random.default_rng(910),
                pool_size=32,
            )
            self.assertEqual(len(proposals), 3)
        config_values = dict(
            N=5,
            n0=4,
            K1=2,
            K2=0,
            posterior_pool_size=8,
            posterior_keep=2,
            axis_candidate_count=0,
            state_candidate_count=0,
            eval_pool_size=8,
            evaluate_interval=0,
            use_problem_initial_samples=True,
            use_boundary_initial_samples=False,
            use_recommendation_refinement=False,
            recommendation_axis_oracle=False,
            recommendation_axis_candidate_count=0,
            recommendation_calibration=False,
            certification_calibration=False,
            acquisition_mode="exact_mc",
            exact_kg_mc_samples=1,
            exact_kg_jobs=2,
            exact_kg_parallel_backend="process_fork",
            task_posterior_mode="finite",
            task_posterior_candidate_count=5,
            task_posterior_recommendation_count=5,
            task_posterior_proposal_pool_size=32,
            seed=903,
        )
        with tempfile.TemporaryDirectory() as tmp:
            config_values.update({
                "checkpoint_dir": tmp,
                "checkpoint_resume": True,
                "checkpoint_interval": 1,
            })
            algorithm = SingleOLHKGAlgorithm(
                target, SingleOLHKGConfig(**config_values))
            result = algorithm.run()
            diagnostics = result["task_posterior"]
            self.assertIsNotNone(diagnostics)
            self.assertEqual(
                result["task_initial_design"]["status"], "generated")
            self.assertFalse(
                result["task_initial_design"]["target_oracle_used"])
            self.assertEqual(diagnostics["posterior"]["n_updates"], 2)
            self.assertEqual(diagnostics["pilot_count"], 3)
            self.assertAlmostEqual(
                sum(diagnostics["posterior"]["posterior_weights"]),
                1.0,
                places=10,
            )
            self.assertIn(
                "exact_kg_task_entropy_gain_selected",
                algorithm.iteration_log[0],
            )
            self.assertEqual(
                algorithm.iteration_log[0]["task_expert_proposals"]["status"],
                "generated",
            )
            self.assertFalse(diagnostics["target_oracle_used"])
            first_weights = np.asarray(
                diagnostics["posterior"]["posterior_weights"], dtype=float)

            resumed_values = dict(config_values)
            resumed_values["N"] = 6
            resumed = SingleOLHKGAlgorithm(
                MetaPriorProblemAdapter(
                    problem("FactorShockStatePolicyRZDT1"), prior),
                SingleOLHKGConfig(**resumed_values),
            )
            resumed_result = resumed.run()
            resumed_diag = resumed_result["task_posterior"]["posterior"]
            self.assertEqual(resumed_diag["n_updates"], 3)
            self.assertEqual(len(resumed.history), 6)
            self.assertEqual(len(resumed.iteration_log), 2)
            checkpoint_before = np.asarray(
                resumed.iteration_log[1]["task_posterior_before"][
                    "posterior_weights"],
                dtype=float,
            )
            np.testing.assert_allclose(
                checkpoint_before, first_weights, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
