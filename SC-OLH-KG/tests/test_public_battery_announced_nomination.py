"""Sign-conditioned readiness, future isolation and independent exact physics."""
from copy import deepcopy

import numpy as np
import pytest

from problems.public_battery_announced_nomination import nominate_announced_pair, tail_stock_projection
from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_visibility import trajectory

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
COUNTS = {"test_nomination_calls": 0, "test_scalar_stock_projection_evaluations": 0,
          "test_trajectory_evaluations": 0}


def announcements(second="export"):
    return [{"release_minute": 960, "delivery_minute": 1080, "direction": "export", "peak_MW": [49, 49]},
            {"release_minute": 1020, "delivery_minute": 1140, "direction": second,
             "peak_MW": [49, 49] if second == "export" else [-49, -49]}]


def call(rows, soc=(34.3, 73.5)):
    command, info = nominate_announced_pair(soc, [0, 0], [0, 0], [.35, .75], 1050, rows, ASSET)
    COUNTS["test_nomination_calls"] += 1
    COUNTS["test_scalar_stock_projection_evaluations"] += info["scalar_stock_projection_evaluations"]
    return command, info


def test_second_announced_sign_changes_readiness_and_reserves_shared_power_first():
    export, a = call(announcements("export"))
    imported, b = call(announcements("import"))
    assert a["minimum_required_gross_power_MW"] > 0
    assert b["minimum_required_gross_power_MW"] == 0
    assert not np.allclose(export, imported)
    for command, info in ((export, a), (imported, b)):
        assert np.abs(command).sum() <= 98 + 1e-8
        stock = np.array(info["projected_second_call_SOC_MWh"])
        band = np.array(info["second_stock_band_MWh"])
        assert np.all(stock >= band[:, 0] - 1e-8) and np.all(stock <= band[:, 1] + 1e-8)
        assert info["scalar_stock_projection_evaluations"] <= 300


def test_unreleased_second_call_cannot_create_a_pair_command():
    rows = announcements()
    rows[1]["release_minute"] = 1110
    changed = deepcopy(rows)
    changed[1].update(direction="import", peak_MW=[-49, -49])
    for pending in (rows, changed):
        command, info = call(pending)
        assert command is None and not info["known_pair_complete"]
        assert info["scalar_stock_projection_evaluations"] == 0


def test_projection_matches_independent_physics_with_nonzero_pending_PN_and_return_crossings():
    initial, current, following, peak, command = map(np.array,
        ([50, 50], [10, -10], [-15, 20], [49, 49], [-35, 28]))
    projected = tail_stock_projection(initial, current, following, peak, command, ASSET)
    COUNTS["test_scalar_stock_projection_evaluations"] += 2
    prefix = np.repeat(current[None], 30, axis=0)
    pulse = pending_pulse_power(np.vstack((following, command)), peak)
    first, last = (np.concatenate((prefix, endpoint)) for endpoint in pulse)
    result = trajectory(first, last, initial, 90, ASSET)
    COUNTS["test_trajectory_evaluations"] += 1
    assert projected == pytest.approx(result["final_SOC_MWh"], abs=1e-8, rel=0)


@pytest.mark.parametrize("soc,reason", [([1, 1], "empty_stock_band"), ([28, 28], "shared_power_infeasible")])
def test_infeasible_stock_or_shared_power_is_retained_without_a_replacement_command(soc, reason):
    command, info = call(announcements(), soc)
    assert command is None and not info["nomination_feasible"]
    assert info["nomination_reason"] == reason
