"""Native V69 restoration checks; source data and target init are replayed."""
import unittest
from unittest.mock import patch

import numpy as np

from baselines.transfer_archive import frozen_archive_from_meta_prior
from performance import benchmark_lodo_meta_prior as benchmark
from performance.mainline_legacy_native import (
    build_native, cached_initial_rows, materialize_v69_task,
)
from problems.single_objective import ScalarizedProblem


class LegacyNativeTests(unittest.TestCase):
    def test_original_configuration_is_materialized(self):
        args = materialize_v69_task("FactorShockStatePolicyRZDT1")["args"]
        self.assertEqual((args["N"], args["n0"]), (13, 10))
        self.assertEqual(args["decision_backend"], "constrained_ts")
        self.assertEqual(args["decision_terminal_rule"], "feasible_first")
        self.assertEqual(args["target_shared_shock_scale"], 1.0)
        self.assertEqual(args["structural_prior_profile"], "none")
        self.assertEqual(args["task_posterior_mode"], "finite")
        self.assertEqual(args["terminal_verification_budget"], 80)
        self.assertEqual(args["terminal_verification_fallback_budget"], 128)
        self.assertEqual(args["terminal_objective_comparison_budget"], 8)

    def test_fresh_stream_reuses_fixed_design_and_reaches_native_config(self):
        original = materialize_v69_task("FactorShockStatePolicyRZDT1", seed=80)
        fresh = materialize_v69_task("FactorShockStatePolicyRZDT1", seed=181)
        self.assertEqual(original["args"], fresh["args"])
        self.assertEqual((original["seed"], fresh["seed"]), (80, 181))
        self.assertEqual(fresh["initial_design_seed"], 80)

        class ConfigCaptured(Exception):
            pass

        captured = []

        def capture_config(problem, config):
            captured.append(config)
            raise ConfigCaptured()

        # Stop at the real config constructor: no source fit or target run.
        with patch.object(benchmark, "build_target_problem", return_value=(object(), {})), \
                patch.object(benchmark, "_source_informed_archive_contract"), \
                patch.object(benchmark, "SingleOLHKGAlgorithm", side_effect=capture_config):
            with self.assertRaises(ConfigCaptured):
                benchmark.run_one(fresh)
        self.assertEqual(captured[0].seed, 181)
        self.assertEqual(captured[0].initial_design_points, tuple(
            tuple(point) for point in original["args"]["initial_design_points"]))

    def test_full_source_init_and_shortlist_without_simulation_or_truth(self):
        rows = cached_initial_rows("InventorySupplyChain")
        with patch.object(ScalarizedProblem, "simulate", side_effect=AssertionError("new simulation")):
            engine = build_native("InventorySupplyChain", N=10, initial_rows=rows)
            with patch.object(engine, "_evaluate_recommendation", side_effect=AssertionError("truth audit")), patch.object(engine, "_true_best_feasible_cached", side_effect=AssertionError("truth search")):
                returned = engine.run_search()
        self.assertIs(returned, engine)
        self.assertEqual(len(engine.history), 10)
        self.assertEqual(engine.replayed_initial_calls, 10)
        self.assertEqual(engine.actual_search_calls, 0)
        np.testing.assert_array_equal([v for _, v in engine.history], [r["raw"] for r in rows])
        restored = frozen_archive_from_meta_prior(engine.source_prior, source_seed=0)
        self.assertEqual(restored.fingerprint, engine.source_prior.frozen_archive.fingerprint)
        self.assertIsNotNone(engine.task_ensemble)
        self.assertGreater(len(engine.task_ensemble.posterior.expert_names), 1)
        terminal = engine.frozen_terminal["terminal"]
        self.assertEqual(len(terminal["frozen_shortlist"]), 3)
        self.assertEqual(terminal["candidate_verification_budgets"], [80, 128, 128])
        self.assertEqual(terminal["max_verification_budget"], 352)
        self.assertEqual(terminal["attempts"], [])
        self.assertEqual(engine.config.terminal_verification_budget, 80)


if __name__ == "__main__":
    unittest.main()
