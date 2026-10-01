"""Pulse-boundary continuity and independent triangle/hold energy accounting."""
import numpy as np
import pytest

from performance.inspect_public_battery_announced_pair import pair_power
from problems.public_battery_visibility import trajectory

CONTEXT = {"first_hour_locked_PN_MW": [[0, 0], [0, 0]]}
ROSTER = {"first_tail_PN_MW": {"export_export": [-49, 0]},
          "second_hour_locked_PN_MW": [[0, 0], [0, 0]],
          "absolute_peak_MW": {"export": [49, 49], "import": [-49, -49]}}
ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}


def test_first_return_uses_intervention_and_second_ramp_resumes_its_own_zero_PN():
    first, last = pair_power(CONTEXT, "export_export", ROSTER)
    assert first.shape == last.shape == (120, 2)
    assert first[0].tolist() == [0, 0]
    assert last[0].tolist() == [49, 49]
    assert first[31].tolist() == [49, 49]
    assert last[31].tolist() == [-49, 0]
    assert first[32].tolist() == last[59].tolist() == [-49, 0]
    assert first[60].tolist() == [0, 0]
    assert last[60].tolist() == [49, 49]
    assert first[91].tolist() == [49, 49]
    assert last[91].tolist() == first[92].tolist() == last[119].tolist() == [0, 0]


def test_full_pair_inventory_matches_hand_integrated_crossing_ramp_and_tail():
    first, last = pair_power(CONTEXT, "export_export", ROSTER)
    initial = np.array([34.3, 73.5])
    metrics = trajectory(first, last, initial, 120, ASSET)
    # Unit 1: first return crosses zero halfway; only its 28-minute tail imports.
    export_1 = 49 * (30 / 60 + 1 / 120 + 1 / 240 + 31 / 60)
    import_1 = 49 * (1 / 240 + 28 / 60)
    final_1 = initial[0] + .92 * import_1 - export_1 / .92
    final_2 = initial[1] - 2 * 49 * 31 / 60 / .92
    assert metrics["final_SOC_MWh"] == pytest.approx([final_1, final_2])
    assert metrics["power_bound_violation_MW"] == 0
