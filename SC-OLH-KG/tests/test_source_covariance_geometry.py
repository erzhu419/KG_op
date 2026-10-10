import unittest

from performance.source_covariance_geometry import direction_geometry


class SourceCovarianceGeometryTests(unittest.TestCase):
    def test_high_overlap_can_still_have_large_equalization_change(self):
        result = direction_geometry([.8, .1, .1, 0.], [0, 1])
        self.assertAlmostEqual(result["selected_covariance_mass"], .9)
        self.assertAlmostEqual(result["outside_covariance_mass"], .1)
        self.assertAlmostEqual(result["flat_reallocation_total_variation"], .4)
        self.assertAlmostEqual(result["extra_within_selection_redistribution"], .3)
        self.assertLess(result["decomposition_error"], 1e-14)

    def test_already_equal_selected_variances_only_move_outside_mass(self):
        result = direction_geometry([.3, .3, .2, .2], [0, 1])
        self.assertAlmostEqual(result["flat_reallocation_total_variation"], .4)
        self.assertEqual(result["extra_within_selection_redistribution"], 0.)


if __name__ == "__main__":
    unittest.main()
