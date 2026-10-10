import unittest
import numpy as np

from performance.source_head_association import allocation_alignment, selection_stability, source_archives


class SourceHeadAssociationTests(unittest.TestCase):
    def test_family_control_removes_family_only_overlap(self):
        result = allocation_alignment([4., 4., 0., 0.], [0], {"a": [0, 1], "b": [2, 3]})
        self.assertEqual(result["excess_over_uniform"], .25)
        self.assertEqual(result["excess_over_family_matched"], 0.)

    def test_specific_function_overlap_survives_family_control(self):
        result = allocation_alignment([8., 0., 0., 0.], [0], {"a": [0, 1], "b": [2, 3]})
        self.assertEqual(result["risk_variance_mass"], 1.)
        self.assertEqual(result["excess_over_family_matched"], .5)

    def test_reuse_deduplicates_without_accessing_target_values(self):
        class MetadataOnlyRow(dict):
            def __getitem__(self, key):
                if key not in ("seed", "regime", "arm"):
                    raise AssertionError(f"target field accessed: {key}")
                return super().__getitem__(key)
        info = [{"selected_indices": [0]}, {"selected_indices": [1]}, {"source_function_variance": [1.] * 68}]
        records = [{"row": MetadataOnlyRow(seed=48, regime=r, arm="risk_ard_mixture"), "source_info": info}
                   for r in ("shared_structure", "high_frequency_risk")]
        records.append({"row": MetadataOnlyRow(seed=48, regime="shared_structure", arm="fixed_source"), "source_info": [{}, {}, {}]})
        archives, reuse = source_archives(records)
        self.assertEqual(list(archives), [(48, "shared")])
        self.assertEqual(reuse[48, "shared"], 2)

    def test_holdout_stability_describes_partial_selection(self):
        result = selection_stability([0, 1], [[0, 1], [0, 2]])
        self.assertAlmostEqual(result["mean_full_vs_holdout_jaccard"], 2. / 3.)
        self.assertAlmostEqual(result["mean_holdout_pair_jaccard"], 1. / 3.)
        self.assertEqual(result["mean_full_selection_retention"], .75)
        np.testing.assert_array_equal(result["source_selection_frequency"][:3], [1., .5, .5])


if __name__ == "__main__":
    unittest.main()
