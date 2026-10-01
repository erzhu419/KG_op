"""Target visibility and money accounting tests; no physics/controller calls."""
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from performance.inspect_public_battery_announced_library import (
    ROOT, PROTOCOL, boundary_records, decode_library, forecast_stress, instant, read,
    settlement_cash, target_fractions, terminal_adjustment,
)
from problems.public_battery_dispatch import positive_integral
from problems.public_battery_pending_availability import pending_pulse_power


class AnnouncedLibraryTests(unittest.TestCase):
    def test_separate_reproduction_uses_saved_boundary_without_public_requests(self):
        protocol = read(PROTOCOL)
        data = read(ROOT / protocol["cached_data_protocol"])
        with TemporaryDirectory() as folder, patch("requests.Session.get", side_effect=AssertionError("repeat public fetch")):
            records, coverage, logs = boundary_records(protocol, data, Path(folder))
        self.assertEqual(logs, [])
        self.assertEqual(sum(len(rows) for rows in records.values()), 12)
        self.assertTrue(all(r["reused_preflight_boundary_slots"] == {"forecast": 3, "price": 3} for r in coverage))

    def test_original_library_constants_and_pair_order(self):
        roster, nodes, channels, metadata = decode_library(read(PROTOCOL))
        self.assertEqual(channels.shape, (154, 2, 1000))
        self.assertEqual(roster[100]["profile_id"], "functional_pair_0010_0037")
        self.assertTrue(metadata["constant_continuity_pass"])
        np.testing.assert_array_equal(target_fractions(nodes, channels[37], [.0, .2, 1.]),
                                      [[.35, .35, .35], [.75, .75, .75]])

    def test_ordered_channels_interpolate_separately(self):
        np.testing.assert_allclose(target_fractions([0., 1.], [[.1, .9], [.8, .2]], [.25, .75]),
                                   [[.3, .7], [.65, .35]])

    def test_actual_UTC_publication_cutoff_and_ramp(self):
        delivery = instant("2024-07-25T01:30:00Z")
        decision = delivery - timedelta(minutes=60)
        fit = {"forecast_q05_MW": 100., "forecast_q95_MW": 200., "absolute_ramp_q95_MW": 40.}
        rows = {delivery - timedelta(minutes=30): {"forecast_MW": 110., "forecast_publish_time_utc": decision.isoformat()},
                delivery: {"forecast_MW": 130., "forecast_publish_time_utc": decision.isoformat()}}
        self.assertEqual(forecast_stress(rows, delivery, decision, fit), (.5, 0.))
        rows[delivery - timedelta(minutes=30)]["forecast_publish_time_utc"] = (decision + timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "publication"):
            forecast_stress(rows, delivery, decision, fit)

    def test_sign_crossing_gross_energy(self):
        np.testing.assert_allclose(positive_integral([10., -20.], [-20., 40.], 1 / 60), [1 / 36, 2 / 9])
        np.testing.assert_allclose(positive_integral([-10., 20.], [20., -40.], 1 / 60), [1 / 9, 1 / 18])

    def test_exact_1_30_1_pulse_and_resumed_PN(self):
        first, last = pending_pulse_power([[10., -10.], [40., -40.]], [-20., 20.])
        np.testing.assert_allclose(positive_integral(first, last, 1 / 60).sum(axis=0), [681 / 36, 61 / 6])
        np.testing.assert_allclose(positive_integral(-first, -last, 1 / 60).sum(axis=0), [61 / 6, 681 / 36])
        baseline_first, baseline_last = pending_pulse_power([[10., -10.], [40., -40.]])
        np.testing.assert_allclose(positive_integral(baseline_first, baseline_last, 1 / 60).sum(axis=0), [25., 0.])

    def test_prices_follow_actual_execution_clock(self):
        start = instant("2024-04-25T00:00:00Z")
        rows = {start: {"price_GBP_per_MWh": 10.},
                start + timedelta(minutes=30): {"price_GBP_per_MWh": -20.}}
        first, last = pending_pulse_power([[10., -20.], [-30., 40.]])
        cash, imported, exported = settlement_cash(first, last, start, rows)
        np.testing.assert_allclose(imported, [[0., 10.], [15., 0.]])
        np.testing.assert_allclose(exported, [[5., 0.], [0., 20.]])
        self.assertEqual(cash, 150.)

    def test_terminal_value_uses_discharge_efficiency(self):
        self.assertEqual(terminal_adjustment([49., 49.], [20., 30.], 100., .92), (92., 4416.))
        self.assertEqual(terminal_adjustment([49., 49.], [49., 49.], -100., .92), (-92., -0.))


if __name__ == "__main__":
    unittest.main()
