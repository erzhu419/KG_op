from datetime import timedelta

import numpy as np

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_site import (inventory_piece, nominate_site, planning_terms,
                                         receipt_plans, simulate_site, site_workload)

ASSET = {"power_MW": 98., "energy_MWh": 196., "charge_efficiency": .92,
         "discharge_efficiency": .92}
NOW = instant("2024-04-15T00:00:00Z")


def instruction(power, received_minutes=0, number=1):
    return Segment(NOW + timedelta(minutes=60), NOW + timedelta(minutes=90),
                   NOW + timedelta(minutes=received_minutes), number, power, power)


def nomination(units):
    terms = planning_terms(receipt_plans(units, NOW, 120), ASSET)
    return nominate_site(np.array([[98.]]), np.zeros((1, 1)), np.zeros((1, 1)),
                         np.array([[98.]]), np.array([0]), terms, ASSET)[0, 0]


def test_future_receipts_cannot_change_current_nomination():
    assert nomination([[instruction(20, 30)], []]) == nomination([[instruction(-40, 30)], []]) == 0


def test_received_future_points_change_nomination_and_reach_target():
    units = [[instruction(20)], []]
    command = nomination(units)
    assert np.isclose(command, -40 / .92 ** 2)
    workload = site_workload(units, NOW, NOW + timedelta(minutes=120))
    terms = planning_terms(receipt_plans(units, NOW, 120), ASSET)
    result = simulate_site(workload, np.ones(4), np.full((1, 4), .5), [0], 90, ASSET, terms)
    assert result["success"][0, 0]
    assert np.isclose(result["final_SOC_MWh"][0, 0], 98)


def test_latest_received_revision_wins_per_unit():
    units = [[instruction(20), instruction(30, 0, 2)], [instruction(-10)]]
    plans = receipt_plans(units, NOW, 120)
    assert np.array_equal(plans["first_MW"][0, 60], [30, -10])
    actual = site_workload(units, NOW, NOW + timedelta(minutes=120))
    assert np.array_equal(actual["first_MW"][60], [30, -10])


def test_opposed_unit_flows_retain_gross_losses():
    delta, _, _, outgoing, incoming, peak = inventory_piece(np.array([[20., -20.]]),
        np.array([[20., -20.]]), np.array([.5]), ASSET)
    assert outgoing[0] == incoming[0] == 10
    assert np.isclose(delta[0], .92 * 10 - 10 / .92)
    assert delta[0] < 0 and peak[0] == 40


def test_gross_limit_stops_window_and_does_not_assign_full_cost():
    workload = {"active": np.ones((1, 2), bool), "first_MW": np.array([[60., -50.]]),
                "last_MW": np.array([[60., -50.]])}
    terms = planning_terms(receipt_plans([[], []], NOW, 30), ASSET)
    result = simulate_site(workload, np.array([10.]), np.full((1, 1), .5), [0], 1, ASSET, terms)
    assert not result["success"][0, 0] and result["power_failure"][0, 0]
    assert result["first_failure_minute"][0, 0] == 0
    assert np.isnan(result["cost_GBP"][0, 0])


def test_interior_soc_extremum_detected_even_when_minute_end_is_safe():
    first, last = np.array([[-20., 0.]]), np.array([[-20., 40.]])
    delta, _, high, _, _, _ = inventory_piece(first, last, np.array([1 / 60]), ASSET)
    assert delta[0] < 0 and high[0] > .01
    workload = {"active": np.ones((1, 2), bool), "first_MW": first, "last_MW": last}
    terms = planning_terms(receipt_plans([[], []], NOW, 30), ASSET)
    result = simulate_site(workload, np.array([10.]), np.full((1, 1), .5), [0], 1, ASSET, terms,
                           initial_fraction=(196 - .01) / 196)
    assert result["inventory_failure"][0, 0] and not result["success"][0, 0]


def test_pooled_success_can_hide_impossible_independent_unit_inventory():
    stop = NOW + timedelta(minutes=600)
    units = [[Segment(NOW, stop, NOW, 1, 20, 20)],
             [Segment(NOW, stop, NOW, 1, -20, -20)]]
    workload = site_workload(units, NOW, stop)
    terms = planning_terms(receipt_plans(units, NOW, 600), ASSET)
    result = simulate_site(workload, np.ones(20), np.full((1, 20), .5), [0], 600, ASSET, terms)
    assert result["success"][0, 0] and not result["equal_partition_success"][0, 0]
    minimum = result["unit_minimum_SOC_change_MWh"][0, 0]
    maximum = result["unit_maximum_SOC_change_MWh"][0, 0]
    assert np.isclose(minimum[0], -200 / .92) and np.isclose(maximum[1], 200 * .92)
