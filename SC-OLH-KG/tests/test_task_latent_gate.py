import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.compare_task_latent_gate import compare  # noqa: E402


def row(
    domain, seed, *, feasible, regret, violation, authoritative=False,
    calibration_mode="source_profiles",
):
    return {
        "heldout": domain,
        "seed": seed,
        "true_feasible": feasible,
        "false_feasible": False,
        "feasible_simple_regret": regret if feasible else None,
        "constraint_violation": violation,
        "task_latent_inference_mode": (
            "authoritative" if authoritative else "shadow"),
        "task_latent_calibration_mode": calibration_mode,
        "task_posterior": {
            "task_latent_authoritative": authoritative,
        },
    }


class TaskLatentGateTests(unittest.TestCase):
    def test_gate_requires_nonworse_metrics_in_every_domain(self):
        domains = ["Factor", "Inventory", "Queue"]
        baseline = [
            row(domain, seed, feasible=seed == 0, regret=0.1,
                violation=0.1)
            for domain in domains for seed in range(2)
        ]
        challenger = [
            row(domain, seed, feasible=True, regret=0.05,
                violation=0.0, authoritative=True)
            for domain in domains for seed in range(2)
        ]
        result = compare(baseline, challenger, domains=domains)
        self.assertTrue(result["passed"])
        self.assertEqual(result["n_pairs"], 6)

        challenger[-1]["constraint_violation"] = 0.5
        failed = compare(baseline, challenger, domains=domains)
        self.assertFalse(failed["passed"])
        queue = next(
            value for value in failed["summaries"]
            if value["heldout"] == "Queue")
        self.assertFalse(queue["checks"]["mean_violation_nonincreasing"])

        wrong_mode = compare(
            baseline,
            challenger,
            domains=domains,
            required_calibration_mode="expert_ridge",
        )
        self.assertFalse(wrong_mode["passed"])
        challenger[-1]["constraint_violation"] = 0.0
        for value in challenger:
            value["task_latent_calibration_mode"] = "expert_ridge"
        right_mode = compare(
            baseline,
            challenger,
            domains=domains,
            required_calibration_mode="expert_ridge",
        )
        self.assertTrue(right_mode["passed"])

    def test_gate_rejects_unpaired_rows(self):
        baseline = [row(
            "Queue", 0, feasible=True, regret=0.1, violation=0.0)]
        with self.assertRaises(ValueError):
            compare(baseline, [], domains=["Queue"])


if __name__ == "__main__":
    unittest.main()
