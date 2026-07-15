import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import baselines.botorch_adapters as adapters  # noqa: E402
from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
    canonical_failure_tolerance,
    canonical_scbo_bounds,
    canonical_ts_candidate_count,
    canonical_turbo_bounds,
    is_botorch_available,
)
from problems.rzdt import StatePolicyRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


@unittest.skipUnless(is_botorch_available(), "BoTorch is not installed")
class BoTorchAdapterTests(unittest.TestCase):
    def _problem(self):
        return ScalarizedProblem(StatePolicyRZDT1(d=5, L=100, sigma=0.04))

    def test_turbo_and_scbo_run_real_botorch_path(self):
        initial_designs = []
        for method in ("botorch_turbo", "botorch_scbo"):
            config = BoTorchBaselineConfig(
                N=6,
                n0=4,
                seed=7,
                method=method,
                raw_samples=8,
                num_restarts=2,
                maxiter=10,
                batch_candidates=16,
                ts_candidates=32,
            )
            result = BoTorchBaseline(self._problem(), config).run()
            self.assertEqual(result["backend"], "botorch")
            self.assertEqual(result["method"], method)
            self.assertEqual(result["n_simulations"], 6)
            self.assertIn("x_recommended", result)
            self.assertEqual(result["initial_design"], "sobol")
            self.assertTrue(result["botorch_strict_failures"])
            self.assertFalse(result["botorch_timeout_fallback"])
            self.assertEqual(result["botorch_candidate_failures"], 0)
            self.assertEqual(len(result["history"]), 6)
            self.assertEqual(result["posterior_certificate_kind"],
                             "gp_latent_ucb_plus_nominal_aleatoric_shift")
            self.assertAlmostEqual(
                result["posterior_chance_margin"],
                result["posterior_chance_margin_mean"]
                + np.sqrt(result["posterior_beta_g"])
                * result["posterior_chance_margin_std"],
            )
            expected_fidelity = (
                "canonical_turbo1_ts" if method == "botorch_turbo"
                else "canonical_scbo_constrained_ts")
            self.assertEqual(result["algorithm_fidelity"], expected_fidelity)
            initial_designs.append([
                row["x"] for row in result["history"][: config.n0]])
        self.assertEqual(initial_designs[0], initial_designs[1])

    def test_saasbo_runs_with_tiny_nuts_budget(self):
        config = BoTorchBaselineConfig(
            N=5,
            n0=4,
            seed=9,
            method="botorch_saasbo",
            raw_samples=8,
            num_restarts=2,
            maxiter=10,
            saas_warmup_steps=4,
            saas_num_samples=4,
            saas_thinning=1,
            saas_max_tree_depth=2,
            saas_mc_samples=16,
        )
        result = BoTorchBaseline(self._problem(), config).run()
        self.assertEqual(result["backend"], "botorch")
        self.assertEqual(result["method"], "botorch_saasbo")
        self.assertEqual(result["n_simulations"], 5)
        self.assertTrue(result["saas_constrained"])
        self.assertFalse(result["saas_nuts_schedule"]["formal_budget"])
        self.assertEqual(
            result["algorithm_fidelity"],
            "saas_fully_bayesian_nuts_constrained_qlogei",
        )

    def test_canonical_tutorial_constants_and_trust_regions(self):
        self.assertEqual(canonical_failure_tolerance(3), 4)
        self.assertEqual(canonical_failure_tolerance(50), 50)
        self.assertEqual(canonical_ts_candidate_count(5), 2000)
        self.assertEqual(canonical_ts_candidate_count(50), 5000)
        center = torch.tensor([0.25, 0.5, 0.75], dtype=torch.double)
        lengthscales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.double)
        lower, upper = canonical_turbo_bounds(center, 0.8, lengthscales)
        weights = lengthscales / lengthscales.mean()
        weights = weights / torch.prod(weights.pow(1.0 / 3.0))
        self.assertTrue(torch.allclose(
            lower, torch.clamp(center - 0.4 * weights, 0.0, 1.0)))
        self.assertTrue(torch.allclose(
            upper, torch.clamp(center + 0.4 * weights, 0.0, 1.0)))
        scbo_lower, scbo_upper = canonical_scbo_bounds(center, 0.8)
        self.assertTrue(torch.allclose(
            scbo_lower, torch.clamp(center - 0.4, 0.0, 1.0)))
        self.assertTrue(torch.allclose(
            scbo_upper, torch.clamp(center + 0.4, 0.0, 1.0)))

    def test_sobol_initial_design_matches_tutorial_reference(self):
        config = BoTorchBaselineConfig(
            N=5,
            n0=5,
            seed=31,
            method="botorch_turbo",
        )
        baseline = BoTorchBaseline(self._problem(), config)
        actual = baseline._initial_samples()
        reference = torch.quasirandom.SobolEngine(
            dimension=self._problem().d,
            scramble=True,
            seed=config.seed,
        ).draw(config.n0).to(dtype=torch.double)
        expected = [
            tuple(self._problem().continuous_to_int(row.numpy()))
            for row in reference
        ]
        self.assertEqual(actual, expected)

    def test_sparse_perturbations_match_tutorial_contract(self):
        problem = ScalarizedProblem(
            StatePolicyRZDT1(d=50, L=100, sigma=0.04))
        baseline = BoTorchBaseline(problem, BoTorchBaselineConfig(
            N=5,
            n0=4,
            seed=37,
            method="botorch_turbo",
        ))
        center = torch.full((problem.d,), 0.5, dtype=torch.double)
        lower = torch.full((problem.d,), 0.2, dtype=torch.double)
        upper = torch.full((problem.d,), 0.8, dtype=torch.double)
        candidates = baseline._ts_candidate_pool(
            center, lower, upper, n_candidates=512, seed=41)
        changed = candidates != center.expand_as(candidates)
        self.assertTrue(torch.all(changed.sum(dim=1) >= 1))
        self.assertTrue(torch.all(candidates >= lower))
        self.assertTrue(torch.all(candidates <= upper))
        self.assertLess(float(changed.double().mean()), 0.5)

    def test_trust_region_state_transitions_match_tutorial_reference(self):
        state = adapters._TrustRegionState(
            dim=5,
            constrained=False,
            length=0.8,
            length_min=0.5 ** 7,
            length_max=1.6,
            success_tolerance=2,
            failure_tolerance=2,
        )
        state.update_turbo(1.0)
        state.update_turbo(1.1)
        self.assertEqual(state.length, 1.6)
        self.assertEqual(state.success_counter, 0)
        state.update_turbo(1.0)
        state.update_turbo(1.0)
        self.assertEqual(state.length, 0.8)
        self.assertEqual(state.failure_counter, 0)

        constrained = adapters._TrustRegionState(
            dim=5,
            constrained=True,
            length=0.8,
            length_min=0.5 ** 7,
            length_max=1.6,
            success_tolerance=10,
            failure_tolerance=5,
        )
        constrained.update_scbo(objective=-2.0, constraint=0.4)
        constrained.update_scbo(objective=-3.0, constraint=0.2)
        self.assertAlmostEqual(constrained.best_constraint, 0.2)
        constrained.update_scbo(objective=-4.0, constraint=-0.1)
        self.assertAlmostEqual(constrained.best_constraint, -0.1)
        self.assertAlmostEqual(constrained.best_value, -4.0)

    def test_formal_saas_defaults_and_full_restart_design(self):
        config = BoTorchBaselineConfig(method="botorch_turbo", N=20, n0=5)
        self.assertGreaterEqual(config.saas_warmup_steps, 256)
        self.assertGreaterEqual(config.saas_num_samples, 128)
        self.assertEqual(config.saas_thinning, 16)
        baseline = BoTorchBaseline(self._problem(), config)
        baseline._tr.restart_triggered = True
        rows = baseline._restart_if_needed(remaining_budget=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(baseline._restart_design_sizes, [3])

    def test_sampler_classes_are_the_canonical_botorch_ones(self):
        self.assertEqual(
            adapters.MaxPosteriorSampling.__module__,
            "botorch.generation.sampling",
        )
        self.assertEqual(
            adapters.ConstrainedMaxPosteriorSampling.__module__,
            "botorch.generation.sampling",
        )

    def test_checkpoint_can_extend_a_completed_canonical_run(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "canonical.pkl")
            first = BoTorchBaseline(self._problem(), BoTorchBaselineConfig(
                N=5,
                n0=4,
                seed=17,
                method="botorch_turbo",
                ts_candidates=32,
                maxiter=10,
                checkpoint_path=checkpoint,
                checkpoint_interval=1,
            )).run()
            second = BoTorchBaseline(self._problem(), BoTorchBaselineConfig(
                N=6,
                n0=4,
                seed=17,
                method="botorch_turbo",
                ts_candidates=32,
                maxiter=10,
                checkpoint_path=checkpoint,
                checkpoint_resume=True,
                checkpoint_interval=1,
            )).run()
            self.assertTrue(second["checkpoint_resumed"])
            self.assertEqual(second["n_simulations"], 6)
            self.assertEqual(first["history"], second["history"][:5])


if __name__ == "__main__":
    unittest.main()
