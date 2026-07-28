import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import re
from unittest.mock import patch

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
from problems.rzdt import (  # noqa: E402
    FactorShockStatePolicyRZDT1,
    StatePolicyRZDT1,
)
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

    def test_botorch_freezes_own_posterior_terminal_shortlist_before_truth(self):
        problem = ScalarizedProblem(
            FactorShockStatePolicyRZDT1(d=5, L=100, sigma=0.04))
        config = BoTorchBaselineConfig(
            N=5,
            n0=4,
            seed=13,
            method="botorch_scbo",
            raw_samples=8,
            num_restarts=2,
            maxiter=10,
            ts_candidates=32,
        )
        result = BoTorchBaseline(problem, config).run(
            freeze_terminal_shortlist=True,
            terminal_probability_slack=0.05,
            terminal_require_provider=True,
        )
        shortlist = result["frozen_terminal_shortlist"]
        self.assertTrue(
            result["terminal_shortlist_frozen_before_truth_metrics"])
        self.assertEqual(len(shortlist), 2)
        self.assertEqual(shortlist[0]["point"], result["x_recommended"])
        self.assertEqual(
            shortlist[0]["selector_posterior"],
            "botorch_latent_chance_margin_posterior",
        )
        self.assertEqual(
            shortlist[1]["coordinate_source"],
            "cumulative_risk_psi=(A,N)",
        )
        self.assertFalse(shortlist[1]["target_labels_used"])
        self.assertFalse(shortlist[1]["target_oracle_used"])
        self.assertFalse(shortlist[1]["verification_samples_used"])

    def test_botorch_v9_freezes_three_policy_objective_challenger(self):
        problem = ScalarizedProblem(
            FactorShockStatePolicyRZDT1(d=5, L=100, sigma=0.04))
        config = BoTorchBaselineConfig(
            N=5,
            n0=4,
            seed=17,
            method="botorch_scbo",
            raw_samples=8,
            num_restarts=2,
            maxiter=10,
            ts_candidates=32,
        )
        result = BoTorchBaseline(problem, config).run(
            freeze_terminal_shortlist=True,
            terminal_probability_slack=0.05,
            terminal_require_provider=True,
            terminal_shortlist_mode=(
                "posterior_objective_challenger_then_safe"),
            terminal_shortlist_size=3,
            terminal_maximum_violation_probability=0.5,
        )
        shortlist = result["frozen_terminal_shortlist"]
        self.assertEqual(len(shortlist), 3)
        self.assertEqual(
            len({tuple(row["point"]) for row in shortlist}), 3)
        self.assertTrue(all(
            row["target_oracle_used"] is False for row in shortlist))
        self.assertTrue(all(
            row["verification_samples_used"] is False for row in shortlist))

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

    def test_deterministic_contract_is_recorded_on_cpu(self):
        previous = torch.are_deterministic_algorithms_enabled()
        config = BoTorchBaselineConfig(
            N=4,
            n0=4,
            seed=31,
            method="botorch_scbo",
            torch_deterministic=True,
        )
        try:
            result = BoTorchBaseline(self._problem(), config).run()
        finally:
            torch.use_deterministic_algorithms(previous)
        contract = result["torch_determinism"]
        self.assertTrue(contract["enabled"])
        self.assertEqual(
            contract["algorithm_mode"],
            "torch_deterministic_algorithms",
        )
        self.assertEqual(
            contract["stage_seed_schedule"],
            "per_iteration_stage_seed_v1",
        )

    def test_saas_duplicate_rounding_uses_discrete_acquisition_fallback(self):
        baseline = BoTorchBaseline(
            self._problem(),
            BoTorchBaselineConfig(
                N=5,
                n0=4,
                seed=23,
                method="botorch_saasbo",
                raw_samples=8,
                num_restarts=2,
                maxiter=10,
            ),
        )
        duplicate = (0, 0, 0, 0, 0)
        fallback = (1, 1, 1, 1, 1)
        baseline.history = [(duplicate, np.zeros(2))]
        continuous = torch.zeros(
            1, self._problem().d, dtype=torch.double)
        with (
            patch.object(
                adapters,
                "optimize_acqf",
                return_value=(continuous, torch.zeros(1)),
            ),
            patch.object(
                baseline,
                "_discrete_saas_acquisition_fallback",
                return_value=fallback,
            ) as discrete,
        ):
            selected = baseline._optimize_saas_acquisition(
                lambda value: value.sum(dim=(-1, -2)))

        self.assertEqual(selected, fallback)
        discrete.assert_called_once()
        self.assertEqual(
            baseline._saas_discrete_candidate_fallback_count, 1)

    def test_saas_doubling_refit_conditions_on_each_new_observation(self):
        config = BoTorchBaselineConfig(
            N=5,
            n0=4,
            seed=19,
            method="botorch_saasbo",
            raw_samples=8,
            num_restarts=2,
            maxiter=10,
            saas_warmup_steps=4,
            saas_num_samples=4,
            saas_thinning=1,
            saas_max_tree_depth=2,
            saas_mc_samples=16,
            saas_refit_schedule="doubling",
        )

        result = BoTorchBaseline(self._problem(), config).run()

        self.assertEqual(result["saas_refit_schedule"], "doubling")
        self.assertEqual(result["saas_full_refit_count"], 1)
        self.assertEqual(result["saas_condition_count"], 1)
        self.assertEqual(result["saas_condition_failures"], 0)
        self.assertTrue(
            result["saas_nuts_schedule"][
                "posterior_conditions_on_every_observation"
            ]
        )
        self.assertEqual(
            result["algorithm_fidelity"],
            "saas_fully_bayesian_periodic_hyperposterior_constrained_qlogei",
        )

    def test_saas_conditioning_does_not_retain_autograd_graphs(self):
        baseline = BoTorchBaseline(
            self._problem(),
            BoTorchBaselineConfig(
                N=5,
                n0=4,
                method="botorch_saasbo",
                saas_refit_schedule="doubling",
            ),
        )

        class RecordingModel:
            def __init__(self):
                self.calls = []

            def condition_on_observations(self, *, X, Y):
                self.calls.append((
                    torch.is_grad_enabled(),
                    bool(X.requires_grad),
                    bool(Y.requires_grad),
                ))
                return self

            def eval(self):
                return self

        objective_model = RecordingModel()
        constraint_model = RecordingModel()
        baseline.history = [(("x0",), np.zeros(3)), (("x1",), np.ones(3))]
        baseline._saas_cached_models = (objective_model, constraint_model)
        baseline._saas_cached_history_size = 1
        tensors = (
            torch.randn(2, 5, requires_grad=True),
            torch.randn(2, 1, requires_grad=True),
            torch.randn(2, 1, requires_grad=True),
        )

        with patch.object(
            baseline, "_training_tensors", return_value=tensors
        ):
            baseline._condition_saas_models_to_history()

        self.assertEqual(objective_model.calls, [(False, False, False)])
        self.assertEqual(constraint_model.calls, [(False, False, False)])
        self.assertEqual(baseline._saas_condition_count, 1)

    def test_saas_doubling_refit_thresholds_are_geometric(self):
        baseline = BoTorchBaseline(
            self._problem(),
            BoTorchBaselineConfig(
                N=40,
                n0=10,
                method="botorch_saasbo",
                saas_refit_schedule="doubling",
            ),
        )
        baseline._saas_cached_models = (object(), object())
        baseline._saas_last_refit_history_size = 10

        self.assertFalse(baseline._saas_should_refit(19))
        self.assertTrue(baseline._saas_should_refit(20))

    def test_saas_doubling_refit_can_stop_at_audited_history_cap(self):
        baseline = BoTorchBaseline(
            self._problem(),
            BoTorchBaselineConfig(
                N=397,
                n0=10,
                method="botorch_saasbo",
                saas_refit_schedule="doubling",
                saas_refit_max_history=80,
            ),
        )
        baseline._saas_cached_models = (object(), object())
        baseline._saas_last_refit_history_size = 40

        self.assertFalse(baseline._saas_should_refit(79))
        self.assertTrue(baseline._saas_should_refit(80))
        baseline._saas_last_refit_history_size = 80
        self.assertFalse(baseline._saas_should_refit(160))
        self.assertFalse(baseline._saas_should_refit(397))

    def test_saas_resume_rebuilds_capped_refit_then_conditions_suffix(self):
        baseline = BoTorchBaseline(
            self._problem(),
            BoTorchBaselineConfig(
                N=397,
                n0=10,
                method="botorch_saasbo",
                saas_refit_schedule="doubling",
                saas_refit_max_history=80,
            ),
        )

        class RecordingModel:
            def __init__(self):
                self.condition_sizes = []
                self.prediction_strategy = None
                self.posterior_sizes = []

            def posterior(self, X):
                self.posterior_sizes.append(len(X))
                self.prediction_strategy = object()

            def condition_on_observations(self, *, X, Y):
                self.condition_sizes.append((len(X), len(Y)))
                return self

            def eval(self):
                return self

        models = (RecordingModel(), RecordingModel())
        baseline.history = [
            ((index,), np.zeros(3)) for index in range(159)
        ]
        tensors = (
            torch.randn(159, 5),
            torch.randn(159, 1),
            torch.randn(159, 1),
        )

        with patch.object(
            baseline, "_training_tensors", return_value=tensors
        ), patch.object(
            baseline, "_fit_saas_models", return_value=models
        ) as fit_models:
            baseline._saas_models_for_history()

        fit_args = fit_models.call_args.args
        self.assertEqual([len(tensor) for tensor in fit_args], [80, 80, 80])
        self.assertEqual(models[0].condition_sizes, [(79, 79)])
        self.assertEqual(models[1].condition_sizes, [(79, 79)])
        self.assertEqual(models[0].posterior_sizes, [1])
        self.assertEqual(models[1].posterior_sizes, [1])
        self.assertEqual(baseline._saas_last_refit_history_size, 80)
        self.assertEqual(baseline._saas_cached_history_size, 159)
        self.assertEqual(baseline._saas_resume_rebuild_history_size, 80)
        self.assertEqual(baseline._saas_condition_count, 79)

    def test_saas_progress_eta_models_growing_fit_cost(self):
        config = BoTorchBaselineConfig(
            N=404,
            n0=10,
            seed=9,
            method="botorch_saasbo",
            progress_logging=True,
        )
        baseline = BoTorchBaseline(self._problem(), config)
        exponent = 1.7
        scale = 5.0
        baseline._progress_start_unit = 20
        baseline._progress_timing = [
            (current, scale * current ** exponent)
            for current in range(40, 240, 5)
        ]
        baseline.history = [None] * 240
        elapsed = scale * 240 ** exponent
        stream = io.StringIO()
        with patch("baselines.botorch_adapters.time.time", return_value=elapsed):
            with redirect_stdout(stream):
                baseline._emit_progress(0.0)

        line = stream.getvalue()
        match = re.search(r"ETA ([0-9.]+)s", line)
        self.assertIsNotNone(match)
        projected_eta = float(match.group(1))
        naive_eta = elapsed / 240.0 * (404 - 240)
        self.assertIn("eta_model=growing_iter_cost", line)
        self.assertGreater(projected_eta, 1.5 * naive_eta)

    def test_resumed_capped_saas_eta_excludes_one_off_rebuild(self):
        config = BoTorchBaselineConfig(
            N=397,
            n0=10,
            seed=9,
            method="botorch_saasbo",
            progress_logging=True,
            saas_refit_schedule="doubling",
            saas_refit_max_history=80,
        )
        baseline = BoTorchBaseline(self._problem(), config)
        baseline._progress_start_unit = 159
        baseline._progress_timing = []
        baseline.history = [None] * 160
        stream = io.StringIO()
        with patch(
            "baselines.botorch_adapters.time.time",
            side_effect=[446.0, 453.0],
        ):
            with redirect_stdout(stream):
                baseline._emit_progress(0.0)
                baseline.history.append(None)
                baseline._emit_progress(0.0)

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("Iter 161/397", lines[0])
        self.assertIn("ETA 1652.0s", lines[0])
        self.assertIn("eta_model=rolling_condition_cost", lines[0])

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

    def test_checkpoint_replays_torch_pyro_stage_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "canonical.pkl")
            config = BoTorchBaselineConfig(
                N=5,
                n0=4,
                seed=23,
                method="botorch_saasbo",
                checkpoint_path=checkpoint,
                checkpoint_interval=1,
            )
            first = BoTorchBaseline(self._problem(), config)
            for x in first._initial_samples():
                first._simulate(x)
                first._save_checkpoint()
            with first._deterministic_torch_stage("saas_nuts:objective") as seed_a:
                draw_a = torch.rand(8)

            with Path(checkpoint).open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(
                payload["stochastic_schedule"]["kind"],
                "per_iteration_stage_seed_v1",
            )

            torch.manual_seed(999999)
            resumed_config = BoTorchBaselineConfig(**{
                **vars(config),
                "checkpoint_resume": True,
            })
            resumed = BoTorchBaseline(self._problem(), resumed_config)
            with resumed._deterministic_torch_stage("saas_nuts:objective") as seed_b:
                draw_b = torch.rand(8)

            self.assertEqual(seed_a, seed_b)
            self.assertTrue(torch.equal(draw_a, draw_b))
            self.assertEqual(
                [x for x, _ in first.history],
                [x for x, _ in resumed.history],
            )
            np.testing.assert_allclose(
                np.asarray([y for _, y in first.history]),
                np.asarray([y for _, y in resumed.history]),
            )

    def test_parallel_saas_models_match_serial_seeded_posteriors(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            generator = torch.Generator().manual_seed(101)
            train_X = torch.rand(
                6, self._problem().d, dtype=torch.double,
                generator=generator,
            )
            train_obj = torch.rand(
                6, 1, dtype=torch.double, generator=generator)
            train_con = torch.rand(
                6, 1, dtype=torch.double, generator=generator)
            common = dict(
                N=6,
                n0=4,
                seed=29,
                method="botorch_saasbo",
                saas_warmup_steps=2,
                saas_num_samples=2,
                saas_thinning=1,
                saas_max_tree_depth=2,
                saas_parallel_threads_per_model=1,
            )
            serial = BoTorchBaseline(
                self._problem(),
                BoTorchBaselineConfig(
                    **common,
                    saas_parallel_models=False,
                ),
            )
            parallel = BoTorchBaseline(
                self._problem(),
                BoTorchBaselineConfig(
                    **common,
                    saas_parallel_models=True,
                    saas_parallel_min_total_steps=0,
                    saas_parallel_fallback=False,
                ),
            )
            serial_models = serial._fit_saas_models(
                train_X, train_obj, train_con)
            parallel_models = parallel._fit_saas_models(
                train_X, train_obj, train_con)
            probe = train_X[:2]
            for serial_model, parallel_model in zip(
                serial_models, parallel_models
            ):
                serial_posterior = serial_model.posterior(probe)
                parallel_posterior = parallel_model.posterior(probe)
                self.assertTrue(torch.equal(
                    serial_posterior.mean,
                    parallel_posterior.mean,
                ))
                self.assertTrue(torch.equal(
                    serial_posterior.variance,
                    parallel_posterior.variance,
                ))
            self.assertEqual(parallel._saas_parallel_fit_count, 1)
            self.assertEqual(parallel._saas_parallel_failures, 0)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
