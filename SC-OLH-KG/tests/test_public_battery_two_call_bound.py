"""Check ramp separation, continuous turning extrema and optimistic tolerances."""
import numpy as np
import pytest

from problems.public_battery_pending_availability import pending_pulse_power
from problems.public_battery_two_call_bound import first_call_separation, two_call_bound
from problems.public_battery_visibility import trajectory

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}


@pytest.fixture(autouse=True)
def count_paths(monkeypatch):
    import problems.public_battery_two_call_bound as module
    original = trajectory
    def counted(*args, **kwargs):
        count_paths.calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "trajectory", counted)
    monkeypatch.setitem(globals(), "trajectory", counted)


count_paths.calls = 0


def test_zero_reference_separation_has_exact_triangle_energy_and_tail_cancels():
    stocks = {}
    for direction, sign in (("export", 1), ("import", -1)):
        path = pending_pulse_power([[0, 0], [0, 0]], sign * np.array([49, 49]))
        stocks[direction] = [trajectory(*path, np.array([49, 49]), n, ASSET)["final_SOC_MWh"]
                             for n in (32, 60)]
    difference = np.array(stocks["import"]) - np.array(stocks["export"])
    expected = 49 * 31 / 60 * (.92 + 1 / .92)
    assert difference == pytest.approx(np.full((2, 2), expected))
    assert difference.min() >= first_call_separation(49, .92, .92)


def test_nonzero_locked_reference_keeps_branch_gap_above_slope_bound():
    final = []
    for sign in (1, -1):
        path = pending_pulse_power([[24, -24], [-20, 20]], sign * np.array([49, 49]))
        final.append(trajectory(*path, np.array([49, 49]), 60, ASSET)["final_SOC_MWh"])
    assert np.min(np.array(final[1]) - final[0]) >= first_call_separation(49, .92, .92)


def test_optimistic_band_includes_return_ramp_turning_extrema():
    result = two_call_bound(ASSET, 49, stock_tolerance=0, power_tolerance=0)
    a, q, eta, hours = 49, 98, .92, 1 / 60
    # Interior return-ramp zero crossings determine these extrema.
    lower = a / eta / 2 + (2 * a*a / eta - eta * q*q) / (2 * (q+a)) * hours
    max_import = eta * a / 2 + (2 * eta * a*a - q*q / eta) / (2 * (q+a)) * hours
    assert result["second_stock_band_MWh"] == pytest.approx([lower, 98 - max_import])
    assert -result["response_kernels"]["export"]["minimum_delta_MWh"] > -result["response_kernels"]["export"]["final_delta_MWh"]
    assert result["response_kernels"]["import"]["maximum_delta_MWh"] > result["response_kernels"]["import"]["final_delta_MWh"]
    assert result["uniform_four_sign_guarantee_impossible"]


def test_existing_power_and_stock_tolerances_enlarge_the_necessary_band():
    exact = two_call_bound(ASSET, 49, stock_tolerance=0, power_tolerance=0)
    relaxed = two_call_bound(ASSET, 49)
    lower, upper = relaxed["second_stock_band_without_stock_tolerance_MWh"]
    assert relaxed["optimistic_reference_limit_MW"] == 98 + 1e-8
    assert relaxed["second_stock_band_MWh"] == [lower - 1e-8, upper + 1e-8]
    assert relaxed["second_stock_band_width_MWh"] > exact["second_stock_band_width_MWh"]
    assert relaxed["separation_excess_MWh"] > 1


def test_nonpositive_gap_is_inconclusive_rather_than_a_feasibility_certificate():
    result = two_call_bound({**ASSET, "unit_energy_MWh": [196., 196.]}, 49)
    assert result["separation_excess_MWh"] < 0
    assert not result["uniform_four_sign_guarantee_impossible"]


def test_asymmetric_efficiencies_preserve_the_general_slope_floor():
    asset = {**ASSET, "charge_efficiency": .8, "discharge_efficiency": .95}
    final = []
    for sign in (1, -1):
        path = pending_pulse_power([[0, 0], [0, 0]], sign * np.array([49, 49]))
        final.append(trajectory(*path, np.array([49, 49]), 60, asset)["final_SOC_MWh"])
    assert np.min(np.array(final[1]) - final[0]) >= first_call_separation(49, .8, .95)
