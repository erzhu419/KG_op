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

    def test_ordered_shortlist_spends_familywise_delta_and_uses_fallback(self):
        algorithm = self._algorithm(seed=31, budget=8)
        algorithm.config.terminal_verification_policy = (
            "ordered_frozen_shortlist")
        algorithm.config.terminal_verification_shortlist_size = 2
        primary = (0, 0, 0, 0, 0)
        fallback = (1, 1, 1, 1, 1)
        algorithm._last_terminal_bayes_ranked_points = [
            primary, fallback]
        calls = []

        def fake_verification(
            point,
            *,
            verification_delta=None,
            candidate_index=0,
        ):
            calls.append((
                tuple(point), verification_delta, candidate_index))
            return {
                "enabled": True,
                "status": (
                    "certified" if candidate_index == 1
                    else "not_certified"),
                "method": "gaussian_student_t_chi_square",
                "policy_frozen_before_verification": True,
                "search_samples_reused": False,
                "posterior_updated_from_verification": False,
                "verification_budget": 8,
                "sample_count": 8,
                "certified": candidate_index == 1,
                "candidate_index": candidate_index,
                "delta": verification_delta,
                "target_oracle_used": False,
            }

        algorithm._terminal_fixed_policy_verification = fake_verification
        deployed, result = algorithm._terminal_verification_protocol(
            primary)

        self.assertEqual(deployed, fallback)
        self.assertEqual(
            calls,
            [(primary, 0.025, 0), (fallback, 0.025, 1)],
        )
        self.assertTrue(result["certified"])
        self.assertTrue(result["recommendation_changed"])
        self.assertEqual(result["selected_shortlist_rank"], 2)
        self.assertEqual(result["verification_budget"], 16)
        self.assertEqual(result["max_verification_budget"], 16)
        self.assertEqual(result["familywise_delta"], 0.05)
        self.assertEqual(result["per_candidate_delta"], 0.025)

    def test_ordered_shortlist_stops_after_primary_certificate(self):
        algorithm = self._algorithm(seed=37, budget=8)
        algorithm.config.terminal_verification_policy = (
            "ordered_frozen_shortlist")
        algorithm.config.terminal_verification_shortlist_size = 2
        primary = (0, 0, 0, 0, 0)
        fallback = (1, 1, 1, 1, 1)
        algorithm._last_terminal_bayes_ranked_points = [
            primary, fallback]
        calls = []

        def fake_verification(
            point,
            *,
            verification_delta=None,
            candidate_index=0,
        ):
            calls.append(tuple(point))
            return {
                "enabled": True,
                "status": "certified",
                "method": "gaussian_student_t_chi_square",
                "policy_frozen_before_verification": True,
                "search_samples_reused": False,
                "posterior_updated_from_verification": False,
                "verification_budget": 8,
                "sample_count": 8,
                "certified": True,
                "candidate_index": candidate_index,
                "delta": verification_delta,
                "target_oracle_used": False,
            }

        algorithm._terminal_fixed_policy_verification = fake_verification
        deployed, result = algorithm._terminal_verification_protocol(
            primary)

        self.assertEqual(deployed, primary)
        self.assertEqual(calls, [primary])
        self.assertEqual(result["selected_shortlist_rank"], 1)
        self.assertEqual(result["verification_budget"], 8)
        self.assertFalse(result["recommendation_changed"])

    def test_ordered_shortlist_requires_frozen_ranked_candidates(self):
        algorithm = self._algorithm(seed=41, budget=8)
        algorithm.config.terminal_verification_policy = (
            "ordered_frozen_shortlist")
        algorithm.config.terminal_verification_shortlist_size = 2
        with self.assertRaisesRegex(
            RuntimeError, "posterior-ranked candidates"
        ):
            algorithm._terminal_verification_protocol((0,) * 5)


if __name__ == "__main__":
    unittest.main()
