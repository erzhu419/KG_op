"""Ramp area, loss direction, future isolation and optimistic-bound necessity."""
import numpy as np
import pytest

from problems.public_battery_full_service_bound import full_service_bound, pulse_area_minutes
from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_visibility import trajectory

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}


def bound(directions, horizon=120):
    marks = [{"minute": 60 * i, "direction": d} for i, d in enumerate(directions)]
    return full_service_bound(marks, [49., 49.], 98., .92, .92, horizon, 1e-8, 1e-8)


def test_ramp_area_includes_both_triangles_and_excludes_future_pulses():
    assert pulse_area_minutes([-60, 0, 1, 31, 31.5, 32, 60]) == pytest.approx([0, 0, .5, 30.5, 30.875, 31, 31])


@pytest.mark.parametrize("direction,sign", [("export", 1), ("import", -1)])
@pytest.mark.parametrize("pn,initial", [([0., 0.], [49., 49.]), ([24., -24.], [70., 28.])])
def test_bound_is_optimistic_against_exact_independent_unit_conversion(direction, sign, pn, initial):
    first, last = pending_pulse_power([pn, pn], np.array(pn) + sign * 49)
    actual = trajectory(first, last, np.array(initial), 60, ASSET)
    marks = [{"minute": 0, "direction": direction}]
    _, prefix = full_service_bound(marks, initial, 98., .92, .92, 60, 1e-8, 1e-8)
    total_actual = sum(actual["final_SOC_MWh"])
    assert total_actual <= prefix["total_SOC_upper_bound_MWh"][-1]
    if pn == [0., 0.]:
        ideal = 98 - 98 * 31 / 60 / .92 if sign == 1 else 98 + .92 * 98 * 31 / 60
        assert total_actual == pytest.approx(ideal)
    else:
        assert total_actual < prefix["total_SOC_upper_bound_MWh"][-1] - .1


def test_two_exports_have_a_hand_computed_first_violating_prefix():
    summary, prefix = bound(["export", "export"])
    # At minute 84: 31+23.5 peak-minutes; at 85: 31+24.5.
    assert summary["first_inspected_violating_minute"] == 85
    assert prefix["total_SOC_upper_bound_MWh"][84] > 0
    assert prefix["total_SOC_upper_bound_MWh"][85] < 0
    assert summary["first_violation"]["export_calls_started"] == 2
    assert summary["first_violation"]["export_calls_completed"] == 1


def test_a_future_call_cannot_contribute_to_a_current_energy_prefix():
    _, prefix = bound([None, "export"])
    assert np.all(prefix["ideal_net_export_MWh"][:61] == 0)
    assert prefix["ideal_net_export_MWh"][61] == pytest.approx(98 / 120)


def test_nonviolating_relaxation_does_not_certify_feasibility():
    summary, _ = bound([None, None])
    assert not summary["conditional_full_service_infeasible"]
    assert summary["first_violation"] is None


def test_retained_power_tolerance_is_included_without_relaxing_stock_threshold():
    summary, prefix = bound([None])
    assert prefix["power_tolerance_correction_MWh"][60] == pytest.approx(1e-8 / .92)
    assert summary["physical_total_SOC_lower_bound_MWh"] == -2e-8
