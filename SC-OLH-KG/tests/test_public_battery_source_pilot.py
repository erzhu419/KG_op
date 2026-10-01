import unittest

import numpy as np

from core.profile_atlas import profile_cosine_coordinate
from performance.inspect_public_battery_source_pilot import ordered_coordinates, pilot_work


class SourceInterfaceTests(unittest.TestCase):
    def test_channel_swap_swaps_coordinates_without_a_join_boundary(self):
        nodes=(np.arange(1000)+.5)/1000
        pair=np.array([np.full(1000,.35),np.full(1000,.75)])
        coords=ordered_coordinates(nodes,np.array([pair,pair[::-1]]))
        width=len(profile_cosine_coordinate(pair[0],nodes=nodes))
        np.testing.assert_allclose(coords[0,:width],coords[1,width:])
        np.testing.assert_allclose(coords[0,width:],coords[1,:width])
        self.assertAlmostEqual(coords[0,0],.35)
        self.assertAlmostEqual(coords[0,width],.75)
        np.testing.assert_allclose(coords[0,1:9],0,atol=1e-14)

    def test_all_frozen_pilot_cells_are_distinct_and_keep_both_rules(self):
        protocol=dict(source_periods=[{"start":"2023-07-15T00:00:00Z"},{"start":"2023-10-15T00:00:00Z"}],
                      pilot=dict(window_start_minutes=[0,4320],profiles=["c0","c4","f10","f37"],
                                 dispatch_rules=["conservative_admission","full_request_reference"]))
        rows=pilot_work(protocol)
        self.assertEqual(len(rows),32)
        self.assertEqual(len({tuple(r.values()) for r in rows}),32)
        self.assertTrue(all(r["period"].startswith("2023") for r in rows))
