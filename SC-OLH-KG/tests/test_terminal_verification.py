import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from problems.rzdt import RZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class TerminalVerificationTests(unittest.TestCase):
    @staticmethod
    def _algorithm(seed=17, budget=8):
        problem = ScalarizedProblem(
            RZDT1(d=5, L=20, sigma=0.01, heteroscedastic=True))
        config = SingleOLHKGConfig(
            N=2,
            n0=2,
            K1=2,
            K2=0,
            seed=seed,
            terminal_verification_budget=budget,
            terminal_verification_delta=0.05,
        )
        return SingleOLHKGAlgorithm(problem, config)

    def test_verification_stream_is_reproducible_and_does_not_update_search(self):
        point = (0, 0, 0, 0, 0)
        first = self._algorithm()._terminal_fixed_policy_verification(point)
        second_algorithm = self._algorithm()
        second = second_algorithm._terminal_fixed_policy_verification(point)

        self.assertEqual(first, second)
        self.assertEqual(first["verification_budget"], 8)
        self.assertEqual(first["sample_count"], 8)
        self.assertEqual(first["total_evaluation_count"], 8)
        self.assertTrue(first["policy_frozen_before_verification"])
        self.assertFalse(first["search_samples_reused"])
        self.assertFalse(first["posterior_updated_from_verification"])
        self.assertFalse(first["recommendation_changed"])
        self.assertFalse(first["target_oracle_used"])
        self.assertEqual(second_algorithm.history, [])
        self.assertEqual(second_algorithm.observations, {})

    def test_disabled_verification_performs_no_simulations(self):
        result = self._algorithm(
            budget=0)._terminal_fixed_policy_verification((0,) * 5)
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["verification_budget"], 0)
        self.assertEqual(result["total_evaluation_count"], 0)

    def test_negative_verification_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self._algorithm(
                budget=-1)._terminal_fixed_policy_verification((0,) * 5)

    def test_full_run_freezes_search_and_counts_verification_separately(self):
        control = self._algorithm(seed=23, budget=0).run()
        verified = self._algorithm(seed=23, budget=4).run()
        self.assertEqual(
            control["target_design_fingerprint"],
            verified["target_design_fingerprint"],
        )
        self.assertEqual(
            control["online_action_sequence_fingerprint"],
            verified["online_action_sequence_fingerprint"],
        )
        self.assertEqual(control["x_recommended"], verified["x_recommended"])
        self.assertEqual(verified["n_search_simulations"], 2)
        self.assertEqual(verified["n_verification_simulations"], 4)
        self.assertEqual(verified["n_simulations"], 6)
        self.assertFalse(
            verified["terminal_verification"]["recommendation_changed"])


if __name__ == "__main__":
    unittest.main()
