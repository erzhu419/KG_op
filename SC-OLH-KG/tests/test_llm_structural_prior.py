import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.rzdt import RZDT2  # noqa: E402
from representation.llm_structural_prior import (  # noqa: E402
    LLMStructuralPriorAdvisor,
)


class FakeLLMAdvisor(LLMStructuralPriorAdvisor):
    def _call_llm(self, messages):
        joined = "\n".join(str(msg.get("content", "")) for msg in messages)
        self.assert_no_problem_name(joined)
        return """
        {
          "abstain": false,
          "confidence": 0.8,
          "candidate_region_priors": [
            {
              "name": "low_mean_boundary",
              "descriptor_center": {
                "mean": 0.2, "std": 0.1, "min": 0.0, "max": 0.5,
                "q10": 0.0, "q25": 0.1, "q50": 0.2,
                "q75": 0.3, "q90": 0.4,
                "low_fraction": 0.8, "high_fraction": 0.0,
                "center_norm": 0.3, "diff_mean_abs": 0.05
              },
              "radius": 0.35,
              "weight": 0.9,
              "rationale": "generic low-exposure region"
            }
          ],
          "acquisition_weights": {
            "kg_objective": 0.25,
            "kg_feasibility": 0.35,
            "kg_variance": 0.25,
            "kg_coupling": 0.15
          }
        }
        """

    @staticmethod
    def assert_no_problem_name(text):
        if "RZDT" in text:
            raise AssertionError("prompt leaked benchmark name")


class LLMStructuralPriorTests(unittest.TestCase):
    def test_offline_guard_blocks_request_before_api_lookup(self):
        advisor = LLMStructuralPriorAdvisor(
            base_url="https://example.invalid",
            model="never-called",
        )
        with patch.dict("os.environ", {"SCOLHKG_OFFLINE": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "network access disabled"):
                advisor._call_llm([])

    def test_fake_llm_prior_generates_bounded_candidates(self):
        problem = RZDT2(d=5, L=100, sigma=0.04)
        rng = np.random.default_rng(4)
        observations = {}
        for _ in range(8):
            x = tuple(problem.sample_random(rng))
            observations[x] = [np.asarray(problem.simulate(x, rng), dtype=float)]
        advisor = FakeLLMAdvisor(
            base_url="https://example.invalid",
            model="fake",
            min_obs=4,
            gate_floor=0.1,
        )
        regions, info = advisor.propose(
            problem,
            observations,
            iteration=3,
            budget_remaining=20,
        )
        self.assertEqual(info["status"], "ok")
        self.assertGreater(info["gate"], 0.0)
        rows = advisor.inverse_candidates(
            problem,
            regions,
            n=6,
            rng=np.random.default_rng(5),
            pool_size=80,
            gate=info["gate"],
        )
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows), len(set(rows)))
        lo, hi = problem.int_bounds()
        for row in rows:
            self.assertTrue(np.all(np.asarray(row) >= lo))
            self.assertTrue(np.all(np.asarray(row) <= hi))


if __name__ == "__main__":
    unittest.main()
