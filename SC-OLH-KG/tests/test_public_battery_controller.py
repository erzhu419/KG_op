"""Physical and causal failures relevant to the delayed-nomination pilot."""
from datetime import timedelta

import numpy as np
import pytest

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_dispatch import minute_workload, simulate

ASSET = {"power_MW": 49, "energy_MWh": 98, "charge_efficiency": .92, "discharge_efficiency": .92}
START = instant("2024-04-15T00:00:00Z")


def run(segments=(), targets=(0, .5, 1), prices=None, minutes=120):
    workload = minute_workload(segments, START, START + timedelta(minutes=minutes), 49)
    signal = np.array([np.full(minutes // 30, t) for t in targets])
    prices = np.full(minutes // 30, 100.0) if prices is None else prices
    return simulate(workload, prices, signal, [0], minutes, ASSET)


def test_common_initial_inventory_is_not_free_revenue():
    result = run()
    # Sell the common inventory or leave it idle: identical value at a fixed price.
    assert result["final_SOC_MWh"][0, 0] == pytest.approx(0)
    assert result["metered_cash_GBP"][0, 0] < 0
    assert result["cost_GBP"][0, 0] == pytest.approx(0, abs=1e-9)
    assert result["cost_GBP"][1, 0] == 0
    # Filling consumes electricity and incurs the independently computed round-trip loss.
    assert result["cost_GBP"][2, 0] == pytest.approx(49 * 100 * (1 - .92 ** 2))
    assert np.max(np.abs(result["energy_balance_error_MWh"])) < 1e-10


def test_future_prices_and_future_unreceived_workload_do_not_change_nominations():
    original = run(targets=(.5,), minutes=240)
    prices = np.array([100, 100, 200, -20, 300, 200, -40, 50])
    valued = run(targets=(1,), minutes=240)
    repriced = run(targets=(1,), prices=prices, minutes=240)
    assert valued["nomination_trace"] == repriced["nomination_trace"]
    assert not np.allclose(valued["cost_GBP"], repriced["cost_GBP"])
    future = Segment(START + timedelta(minutes=90), START + timedelta(minutes=120),
                     START + timedelta(minutes=90), 1, 49, 49)
    changed = run((future,), targets=(.5,), minutes=240)
    assert [r for r in original["nomination_trace"] if r["decision_minute"] <= 90] == [
        r for r in changed["nomination_trace"] if r["decision_minute"] <= 90]
    assert original["nomination_trace"] != changed["nomination_trace"]
    assert all(r["delivery_minute"] - r["decision_minute"] == 60 for r in changed["nomination_trace"])


def test_power_clipping_and_opposed_flows_keep_their_physical_energy():
    ramp = Segment(START, START + timedelta(minutes=1), START, 1, 50, -50)
    result = run((ramp,), targets=(.5,), minutes=60)
    requested_each = 50 / 4 / 60
    # A 50-to-zero half-minute ramp clips a triangle of height 1 MW above 49.
    clipped_each = 1 / 2 * (1 / 100) / 60
    assert result["gross_export_MWh"][0, 0] == pytest.approx(requested_each - clipped_each)
    assert result["gross_import_MWh"][0, 0] == pytest.approx(requested_each - clipped_each)
    assert result["BOA_undelivered_MWh"][0, 0] == pytest.approx(2 * clipped_each)
    assert result["minimum_SOC_MWh"][0, 0] == pytest.approx(49 - (requested_each - clipped_each) / .92)
    assert np.max(np.abs(result["energy_balance_error_MWh"])) < 1e-10


def test_dispatch_can_leave_a_previously_nominated_baseline_undeliverable():
    call = Segment(START + timedelta(minutes=20), START + timedelta(minutes=50),
                   START + timedelta(minutes=20), 1, 49, 49)
    result = run((call,), targets=(0,), minutes=120)
    assert result["BOA_undelivered_MWh"][0, 0] == 0
    assert result["baseline_undelivered_MWh"][0, 0] > 0
    assert result["minimum_SOC_MWh"][0, 0] >= -1e-10
    assert result["maximum_SOC_MWh"][0, 0] <= 98 + 1e-10
    assert np.max(np.abs(result["energy_balance_error_MWh"])) < 1e-10
