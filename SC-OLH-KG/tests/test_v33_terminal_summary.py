import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.summarize_v33_terminal_matrix import (  # noqa: E402
    DOMAINS,
    VARIANTS,
    summarize,
)


POLICIES = {
    "v32": ("legacy", "legacy"),
    "posterior_only": ("legacy", "off"),
    "commit_before_switch": ("commit_before_switch", "certified_only"),
    "terminal_kg_1step": ("terminal_kg_1step", "certified_only"),
    "terminal_kg_depth3": ("terminal_kg_depth3", "certified_only"),
}


def row(variant, domain, seed):
    policy, override = POLICIES[variant]
    true_feasible = not (
        domain != "FactorShockStatePolicyRZDT1" and seed >= 5)
    completed = 1 if variant == "terminal_kg_1step" else 0
    terminal_evaluations = 3 if variant.startswith("terminal_kg") else 0
    return {
        "_variant": variant,
        "heldout": domain,
        "seed": seed,
        "finalist_replication_policy": policy,
        "finalist_empirical_override": override,
        "true_feasible": true_feasible,
        "posterior_feasible": true_feasible,
        "false_feasible": False,
        "feasible_simple_regret": 0.1 if true_feasible else None,
        "simple_regret": 0.1,
        "constraint_violation": 0.0 if true_feasible else 0.01,
        "true_chance_margin": -0.1 if true_feasible else 0.01,
        "algorithm_time_sec": 12.0,
        "wall_time_sec": 13.0,
        "replicated_finalist_used": (
            variant == "terminal_kg_1step" and seed == 0),
        "replicated_finalist_empirical_certificate": True,
        "finalist_replication": {
            "policy": policy,
            "empirical_override_policy": override,
            "completed_target_count": completed,
            "terminal_kg_evaluations": terminal_evaluations,
            "target_oracle_used": False,
            "terminal_kg_rows": [
                {
                    "terminal_kg_arm_count": 4,
                    "terminal_kg_selected_gain": 0.02,
                }
                for _ in range(terminal_evaluations)
            ],
        },
    }


class V33TerminalSummaryTests(unittest.TestCase):
    def test_preregistered_gate_passes_complete_safe_matrix(self):
        rows = [
            row(variant, domain, seed)
            for variant in VARIANTS
            for domain in DOMAINS
            for seed in range(7)
        ]
        result = summarize(rows)
        self.assertEqual(len(result["summaries"]), 15)
        self.assertTrue(result["primary_gate"]["passed"])
        self.assertEqual(
            result["primary_gate"]["informative_completion_challenger"],
            21,
        )

    def test_uncertified_override_rejects_primary_gate(self):
        rows = [
            row(variant, domain, seed)
            for variant in VARIANTS
            for domain in DOMAINS
            for seed in range(7)
        ]
        target = next(
            item for item in rows
            if item["_variant"] == "terminal_kg_1step"
            and item["heldout"] == DOMAINS[0]
            and item["seed"] == 0
        )
        target["replicated_finalist_empirical_certificate"] = False
        result = summarize(rows)
        self.assertFalse(result["primary_gate"]["passed"])
        self.assertFalse(
            result["primary_gate"]["no_uncertified_override_passed"])


if __name__ == "__main__":
    unittest.main()
