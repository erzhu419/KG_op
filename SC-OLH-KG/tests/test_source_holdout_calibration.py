"""Verify held-function isolation, time order and archive-level inference."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.source_holdout_calibration import fit_source_fold, forecast_held_source, summarize_source_holdout
from performance.target_risk_residual import predictive_group_score


class SourceHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inputs = ROOT / "paper_artifacts/meta_prior_calibration_replay_a_20261003/execution_plan.json"
        cls.protocol = json.loads(inputs.read_text())["base_protocol"]
        cls.tasks, _ = prepare_sources(48, "shared_structure", cls.protocol)
        cls.reference = np.random.default_rng(51048).uniform(-2., 2.,
            (cls.protocol["reference_bank_size"], cls.protocol["psi_dim"]))
        plan = ROOT / "paper_artifacts/meta_prior_risk_family_reweighting_a9_20261003/execution_plan.json"
        cls.families = json.loads(plan.read_text())["public_families"]
        cls.full, cls.structural, cls.info = fit_source_fold(cls.tasks, 0, cls.reference)

    def test_entire_held_function_is_excluded_from_source_prior_fit(self):
        changed = copy.deepcopy(self.tasks)
        changed[0]["response"][:, 2] += 3.
        changed[0]["noise"][:, 2] *= 2.
        full, structural, info = fit_source_fold(changed, 0, self.reference)
        self.assertEqual(info["training_tasks"], list(range(1, 8)))
        self.assertEqual(info["source_label_count"], 7 * self.protocol["source_points"])
        for before, after in ((self.full, full), (self.structural, structural)):
            np.testing.assert_array_equal(before.mean, after.mean)
            np.testing.assert_array_equal(before.covariance, after.covariance)
            self.assertEqual(before.discrepancy_variance, after.discrepancy_variance)

    def test_future_held_labels_do_not_change_issued_evidence(self):
        priors = [self.structural, self.full]
        task = self.tasks[0]
        a = forecast_held_source(priors, task, self.families, 32, [9])[0]
        changed = copy.deepcopy(task)
        changed["response"][8:, 2] += 3.
        b = forecast_held_source(priors, changed, self.families, 32, [9])[0]
        for key in ("null_posterior", "family_posterior", "mean_rho_posterior", "training_log_evidence"):
            self.assertEqual(a[key], b[key])
        self.assertNotEqual(a["source"], b["source"])

    def test_source_control_matches_original_predictor_for_both_priors(self):
        task = self.tasks[0]
        for priors in ([self.full], [self.structural, self.full]):
            f = forecast_held_source(priors, task, self.families, 32, [9])[0]
            expected = predictive_group_score(priors, task["psi"], np.arange(8), task["response"][:8, 2],
                task["noise"][:8, 2], 0., 8, task["response"][8:9, 2], task["noise"][8:9, 2])
            self.assertAlmostEqual(f["source"], expected, places=11)

    def test_summary_averages_tasks_within_seed_before_bootstrap(self):
        spec = {"cohort": {"seeds": list(range(16)), "source_archive_groups": ["shared", "separate_heads"],
            "priors": ["fixed_source", "risk_ard_mixture"], "held_tasks_per_archive": 8}, "paired_bootstrap_seed": 20261020}
        rows = [{"seed": s, "archive_group": group, "held_task": task, "arm": arm,
                 "source_mean_log_score": .1 if arm == "risk_ard_mixture" else 0.,
                 "adapted_mean_log_score": (.1 if arm == "risk_ard_mixture" else 0.) + s / 10. + task,
                 "a9_mean_log_score": (.1 if arm == "risk_ard_mixture" else 0.) + s / 100.}
                for s in range(16) for group in spec["cohort"]["source_archive_groups"]
                for arm in spec["cohort"]["priors"] for task in range(8)]
        result = summarize_source_holdout(rows, spec)
        for g in result["groups"]:
            self.assertEqual(g["seed_archives"], 16)
            self.assertEqual(g["held_tasks"], 128)
            expected = np.arange(16) / (10. if g["contrast"] == "adapted_minus_source" else 100.)
            expected += 3.5 if g["contrast"] == "adapted_minus_source" else 0.
            np.testing.assert_allclose(g["seed_gains"], expected, atol=1e-12)
        for c in result["source_contrasts"]:
            self.assertAlmostEqual(c["mean_gain"], .1, places=12)


if __name__ == "__main__":
    unittest.main()
