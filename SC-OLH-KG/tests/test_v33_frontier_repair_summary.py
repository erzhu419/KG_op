import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.summarize_v33_frontier_repair import (  # noqa: E402
    DOMAINS,
    MANDATORY_FRONTIER_LABELS,
    VARIANT_CONFIGS,
    summarize,
)


class V33FrontierRepairSummaryTests(unittest.TestCase):
    @staticmethod
    def rows():
        feasible_counts = {
            "v32": (7, 5, 3),
            "v33_legacy_4": (7, 2, 5),
            "v33_coherent_coverage_4": (7, 5, 5),
            "v33_coherent_coverage_8": (7, 5, 5),
        }
        rows = []
        for config, variant in VARIANT_CONFIGS.items():
            policy, override, frontier, arms, contract = config
            coherent = contract == "certified_lexicographic"
            for domain_index, domain in enumerate(DOMAINS):
                for seed in range(7):
                    feasible = seed < feasible_counts[variant][domain_index]
                    labels = (
                        list(MANDATORY_FRONTIER_LABELS)
                        + ["expert_safety_nomination:a"] * max(arms - 4, 0)
                        if coherent else ["minimum_bayes_risk"]
                    )
                    rows.append({
                        "_variant": variant,
                        "heldout": domain,
                        "seed": seed,
                        "finalist_replication_policy": policy,
                        "finalist_empirical_override": override,
                        "finalist_frontier_policy": frontier,
                        "finalist_terminal_max_arms": arms,
                        "decision_contract_mode": contract,
                        "true_feasible": feasible,
                        "posterior_feasible": False,
                        "false_feasible": False,
                        "feasible_simple_regret": 0.01 if feasible else None,
                        "constraint_violation": 0.0 if feasible else 0.02,
                        "true_chance_margin": -0.01 if feasible else 0.02,
                        "algorithm_time_sec": 600.0 if not coherent else 720.0,
                        "wall_time_sec": 605.0 if not coherent else 725.0,
                        "replicated_finalist_used": False,
                        "finalist_replication": {
                            "labels": labels,
                            "frontier_policy": frontier,
                            "coherent_three_layer_contract": coherent,
                            "target_oracle_used": False,
                            "terminal_kg_rows": (
                                [{
                                    "terminal_kg_value_mode": (
                                        "certified_lexicographic"),
                                }] * 3 if coherent else []
                            ),
                        },
                    })
        return rows

    def test_preregistered_repair_gate_accepts_coherent_nonworse_matrix(self):
        result = summarize(self.rows())
        self.assertTrue(result["primary_gate"]["passed"])
        challenger = [
            row for row in result["summaries"]
            if row["variant"] == "v33_coherent_coverage_8"
        ]
        self.assertTrue(all(
            row["mandatory_frontier_coverage_count"] == 7
            and row["terminal_lexicographic_count"] == 21
            and row["coherent_contract_audit_pass"]
            for row in challenger
        ))

    def test_missing_reserved_axis_fails_contract_gate(self):
        rows = self.rows()
        target = next(
            row for row in rows
            if row["_variant"] == "v33_coherent_coverage_8"
            and row["heldout"] == DOMAINS[0]
            and row["seed"] == 0
        )
        target["finalist_replication"]["labels"].remove(
            "minimum_certificate_margin")
        result = summarize(copy.deepcopy(rows))
        self.assertFalse(result["primary_gate"]["passed"])
        self.assertFalse(result["primary_gate"][
            "coherent_contract_passed"])


if __name__ == "__main__":
    unittest.main()
