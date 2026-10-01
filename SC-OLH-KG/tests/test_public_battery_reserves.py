from datetime import timedelta

import numpy as np

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_reserves import (inventory_reserve_planning_terms,
                                            power_reserve_planning_terms, reserve_planning_terms)
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_units import simulate_units

NOW = instant("2024-04-15T00:00:00Z")
ASSET = {"power_MW": 98., "energy_MWh": 196., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}


def instruction(first, last, left, right, received, number=1):
    return Segment(NOW + timedelta(minutes=left), NOW + timedelta(minutes=right),
                   NOW + timedelta(minutes=received), number, first, last)


def terms_for(units, minutes=120):
    plans = receipt_plans(units, NOW, minutes)
    history = site_workload(units, NOW - timedelta(minutes=90), NOW + timedelta(minutes=minutes))
    return reserve_planning_terms(plans, history, ASSET)


def test_reversal_reserves_keep_both_gross_energies_and_exclude_current_minute():
    units = [[instruction(-60, 60, -90, 0, -90), instruction(90, 90, 0, 30, 0, 2)], []]
    terms = terms_for(units)
    # Two 45-minute triangles, each 22.5 MWh. The new minute-zero BOA is
    # observable for projection, but is not yet executed reserve history.
    assert np.allclose(terms["target_lower_MWh"][0], [22.5 / .92, 0])
    assert np.allclose(terms["target_upper_MWh"][0], [98 - .92 * 22.5, 98])
    assert np.allclose(terms["observed_power_bound_MW"][0], [98, 38])
    assert not terms["reserve_conflict"][0]


def test_observed_opposite_unit_and_received_future_headroom_both_apply():
    units = [[instruction(20, 20, -30, 0, -30)],
             [instruction(40, 40, -30, 0, -30), instruction(80, 80, 60, 90, 0, 2)]]
    terms = terms_for(units)
    assert np.allclose(terms["observed_power_bound_MW"][0], [58, 78])
    assert np.allclose(terms["future_power_bound_MW"][0], [18, 78])


def test_unreceived_future_revision_cannot_change_earlier_reserves_or_nominations():
    runs = []
    for future in (20, -40):
        units = [[instruction(30, 30, -30, 0, -30),
                  instruction(future, future, 60, 90, 45, 2)], []]
        terms = terms_for(units)
        workload = site_workload(units, NOW, NOW + timedelta(minutes=120))
        result = simulate_units(workload, np.ones(4), np.full((1, 4), .5), [0], 120, ASSET, terms)
        runs.append((terms, result))
    for field in ("target_lower_MWh", "target_upper_MWh", "future_power_bound_MW"):
        assert np.array_equal(runs[0][0][field][:2], runs[1][0][field][:2])
    assert runs[0][1]["nomination_trace"][:2] == runs[1][1]["nomination_trace"][:2]


def test_history_changes_preparation_without_changing_initial_stock_or_lead():
    units = [[], [instruction(50, 50, -30, 0, -30), instruction(50, 50, 60, 90, 60, 2)]]
    workload = site_workload(units, NOW, NOW + timedelta(minutes=90))
    result = simulate_units(workload, np.ones(3), np.zeros((1, 3)), [0], 90, ASSET, terms_for(units, 90))
    trace = result["nomination_trace"][0]
    assert trace["decision_minute"] == 0 and trace["delivery_minute"] == 60
    assert trace["first_policy_window_SOC_MWh"] == [49, 49]
    assert np.isclose(trace["first_policy_window_nomination_MW"][0], 48)
    assert result["success"][0, 0]
    assert np.allclose(result["unit_final_SOC_MWh"][0, 0], [49 - 24 / .92, 49 - 25 / .92])
    assert np.max(np.abs(result["energy_balance_error_MWh"])) < 1e-10


def test_inventory_only_replenishes_history_without_restricting_recent_power():
    units = [[], [instruction(50, 50, -30, 0, -30)]]
    plans = receipt_plans(units, NOW, 90)
    history = site_workload(units, NOW - timedelta(minutes=90), NOW + timedelta(minutes=90))
    terms = inventory_reserve_planning_terms(plans, history, ASSET)
    assert np.allclose(terms["future_power_bound_MW"][0], [98, 98])
    workload = site_workload(units, NOW, NOW + timedelta(minutes=90))
    result = simulate_units(workload, np.ones(3), np.full((1, 3), .5), [0], 90, ASSET, terms)
    # 25 MWh historical export moves the midpoint up by 25/(2*.92) MWh;
    # restoring this over the delivery half hour requires -25/.92**2 MW.
    assert np.allclose(result["nomination_trace"][0]["first_policy_window_nomination_MW"], [0, -25 / .92**2])
    assert np.allclose(result["unit_final_SOC_MWh"][0, 0], [49, 49 + 25 / (2 * .92)])


def test_power_only_keeps_original_stock_target_and_observed_headroom():
    units = [[], [instruction(50, 50, -30, 0, -30)]]
    plans = receipt_plans(units, NOW, 90)
    history = site_workload(units, NOW - timedelta(minutes=90), NOW + timedelta(minutes=90))
    terms = power_reserve_planning_terms(plans, history, ASSET)
    assert np.allclose(terms["target_lower_MWh"][0], [0, 0])
    assert np.allclose(terms["target_upper_MWh"][0], [98, 98])
    assert np.allclose(terms["future_power_bound_MW"][0], [48, 98])
    workload = site_workload(units, NOW, NOW + timedelta(minutes=90))
    result = simulate_units(workload, np.ones(3), np.full((1, 3), .5), [0], 90, ASSET, terms)
    assert result["nomination_trace"][0]["first_policy_window_nomination_MW"] == [0, 0]
    assert result["success"][0, 0] and np.allclose(result["unit_final_SOC_MWh"][0, 0], [49, 49])
