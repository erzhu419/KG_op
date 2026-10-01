import unittest

from performance.assess_public_battery_source_archive import assess


class PopulationSupportTests(unittest.TestCase):
    def rows(self):
        return [dict(period="2024-04-15", window_start_minute=minute, profile_id=profile,
                     dispatch_rule=rule, window_success=profile == "constant_pair_safe" or minute == 0,
                     cost_GBP=(10 + minute + offset) if profile == "constant_pair_safe" or minute == 0 else None)
                for profile, offset in (("constant_pair_safe", 0), ("functional_pair_risky", 5))
                for minute in (0, 60, 120) for rule in ("conservative_admission", "full_request_reference")]

    def assess(self, rows):
        return assess(rows, ["constant_pair_safe", "functional_pair_risky"], [], ["2024-04-15"],
                      [0, 60, 120], ["conservative_admission", "full_request_reference"], 3)

    def test_failures_keep_null_cost_and_do_not_prune_the_library(self):
        result = self.assess(self.rows())
        self.assertEqual(result["reliable_joint_candidates"], ["constant_pair_safe"])
        self.assertFalse(result["source_library_pruned"])
        self.assertFalse(result["original_objective_atlas_interface_ready"])
        risky = [g for g in result["groups"] if g["profile_id"] == "functional_pair_risky"]
        self.assertTrue(all(g["full_population_mean_cost_GBP"] is None for g in risky))
        self.assertEqual(result["dispatch_rule_comparison"]["maximum_successful_cost_difference_GBP"], 0)

    def test_only_common_success_windows_enter_a_comparison(self):
        result = assess(self.rows(), ["constant_pair_safe", "functional_pair_risky"], [], ["2024-04-15"],
                        [0, 60, 120], ["conservative_admission", "full_request_reference"], 1)
        shared = result["common_cost_comparisons"][0]
        self.assertEqual(shared["common_successful_windows"], 1)
        self.assertEqual(shared["mean_cost_GBP_by_candidate"], {"constant_pair_safe": 10, "functional_pair_risky": 15})

    def test_partial_or_duplicate_population_is_rejected(self):
        for rows in (self.rows()[:-1], self.rows() + self.rows()[:1]):
            with self.assertRaisesRegex(ValueError, "complete unique"):
                self.assess(rows)

    def test_missing_or_fabricated_cost_is_rejected(self):
        rows = self.rows()
        rows[0]["cost_GBP"] = None
        with self.assertRaisesRegex(ValueError, "cost"):
            self.assess(rows)
        rows = self.rows()
        next(r for r in rows if not r["window_success"])["cost_GBP"] = 0
        with self.assertRaisesRegex(ValueError, "cost"):
            self.assess(rows)
