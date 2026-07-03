import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders.policy_state_encoder import SyntheticPolicyStateEncoder  # noqa: E402
from problems.rzdt import RegimeRZDT1  # noqa: E402


class StateEncoderTests(unittest.TestCase):
    def test_coupling_scores_reward_coverage_gaps(self):
        problem = RegimeRZDT1(d=3, L=100, sigma=0.05)
        encoder = SyntheticPolicyStateEncoder(problem)
        candidates = [(0, 0, 0), (50, 0, 0), (100, 0, 0)]
        observed = [(0, 0, 0), (100, 0, 0)]
        scores = encoder.propagation_scores(candidates, observed)
        self.assertLess(scores[0], 1e-12)
        self.assertLess(scores[2], 1e-12)
        self.assertGreater(scores[1], 0.99)

    def test_coupling_scores_are_zero_when_all_candidates_equally_covered(self):
        problem = RegimeRZDT1(d=3, L=100, sigma=0.05)
        encoder = SyntheticPolicyStateEncoder(problem)
        candidates = [(0, 0, 0), (100, 0, 0)]
        observed = [(0, 0, 0), (100, 0, 0)]
        scores = encoder.propagation_scores(candidates, observed)
        self.assertTrue((scores == 0.0).all())


if __name__ == "__main__":
    unittest.main()
