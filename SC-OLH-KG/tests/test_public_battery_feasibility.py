"""Analytic cases distinguish physics, power mismatch and a poor causal rule."""
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_dispatch import minute_workload, simulate
from problems.public_battery_feasibility import physical_oracle

ROOT = Path(__file__).resolve().parents[1]
START = instant("2024-04-15T00:00:00Z")


def oracle(segments, *, minutes=120, initial=49):
    protocol = json.loads((ROOT / "performance/manifests/public_battery_physical_oracle_v1_20260930.json").read_text())
    protocol["initial_SOC_MWh"] = initial
    workload = minute_workload(segments, START, START + timedelta(minutes=minutes), 49)
    result, powers = physical_oracle(workload, 0, minutes, protocol)
    return result, powers, workload, protocol


def test_idle_workload_has_a_physical_replayed_witness():
    result, powers, _, _ = oracle([])
    assert result["status"] == "replayed_feasible"
    np.testing.assert_array_equal(powers[:2], [0, 0])
    assert np.max(np.abs(powers)) <= 49 + 1e-9
    assert result["undelivered_MWh"] <= 1e-8
    assert abs(result["energy_balance_error_MWh"]) < 1e-8


def test_uninterrupted_dispatch_cannot_replenish_from_baseline():
    call = Segment(START, START + timedelta(minutes=120), START, 1, 49, 49)
    result, powers, _, _ = oracle([call])
    assert result["status"] == "energy_or_commitment_infeasible"
    assert result["LP_solved"]
    assert powers is None


def test_recorded_power_level_above_reference_has_an_unavoidable_shortfall():
    call = Segment(START, START + timedelta(minutes=1), START, 1, 50, 50)
    result, powers, _, _ = oracle([call])
    assert result["status"] == "power_infeasible"
    assert result["minimum_power_shortfall_MWh"] == pytest.approx(1 / 60)
    assert not result["LP_solved"]
    assert powers is None


def test_within_minute_zero_crossing_can_fail_despite_a_feasible_net_offset():
    ramp = Segment(START, START + timedelta(minutes=1), START, 1, 49, -49)
    net_offset = .92 * (49 / 4 / 60) - (49 / 4 / 60) / .92
    assert .1 + net_offset > 0
    result, _, _, _ = oracle([ramp], initial=.1)
    assert result["status"] == "energy_or_commitment_infeasible"


def test_feasible_physics_does_not_imply_a_given_causal_controller_succeeds():
    call = Segment(START + timedelta(minutes=20), START + timedelta(minutes=50),
                   START + timedelta(minutes=20), 1, 49, 49)
    result, _, workload, protocol = oracle([call])
    assert result["status"] == "replayed_feasible"
    causal = simulate(workload, np.full(4, 100), np.zeros((1, 4)), [0], 120, protocol["asset"])
    assert causal["baseline_undelivered_MWh"][0, 0] > 1e-8
