from datetime import timedelta
import json
from pathlib import Path

import numpy as np

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_unit_feasibility import replay_schedule, unit_physical_oracle
from problems.public_battery_units import simulate_units, unit_planning_terms

ROOT = Path(__file__).resolve().parents[1]
START = instant("2024-04-15T00:00:00Z")


def call(first, last, left, right):
    return Segment(START + timedelta(minutes=left), START + timedelta(minutes=right),
                   START + timedelta(minutes=left), left + 1, first, last)


def oracle(units, minutes=120, initial_fraction=.5):
    protocol = json.loads((ROOT / "performance/manifests/public_battery_unit_physical_oracle_v1_20260930.json").read_text())
    protocol["controls"]["initial_soc_fraction"] = initial_fraction
    workload = site_workload(units, START, START + timedelta(minutes=minutes))
    result, powers = unit_physical_oracle(workload, 0, minutes, protocol)
    return result, powers, workload, protocol


def test_idle_witness_preserves_bootstrap_power_and_independent_energy_accounts():
    result, powers, _, _ = oracle([[], []])
    assert result["status"] == "replayed_feasible"
    assert np.all(powers[:2] == 0)
    assert np.max(np.sum(np.abs(powers), axis=1)) <= 98 + 1e-8
    assert np.max(np.abs(result["energy_balance_error_MWh"])) < 1e-8


def test_no_inventory_transfer_between_units_even_when_total_stock_is_sufficient():
    result, powers, _, _ = oracle([[call(20, 20, 0, 180)], [call(-20, -20, 0, 180)]], minutes=180)
    assert result["status"] == "energy_or_commitment_infeasible" and powers is None


def test_shared_nomination_limit_can_prevent_two_individually_feasible_replenishments():
    # Only 60--90 minutes can replenish before the contiguous export block.
    # Each unit needs 67.16 MW; the simultaneous demand exceeds the 98 MW site.
    demand = [call(49, 49, left, left + 30) for left in (90, 120, 150)]
    single, _, _, _ = oracle([demand, []], minutes=180)
    joint, powers, _, _ = oracle([demand, demand], minutes=180)
    assert single["status"] == "replayed_feasible"
    assert joint["status"] == "energy_or_commitment_infeasible" and powers is None


def test_both_forced_units_above_site_power_fail_before_solving():
    result, powers, _, _ = oracle([[call(50, 50, 0, 1)], [call(50, 50, 0, 1)]])
    assert result["status"] == "forced_power_infeasible" and not result["LP_solved"]
    assert powers is None


def test_replay_detects_boa_baseline_collision_despite_legal_nomination_sum():
    _, _, workload, protocol = oracle([[call(49, 49, 60, 90)], []])
    powers = np.zeros((4, 2))
    powers[2, 1] = -60
    replay = replay_schedule(workload, 0, powers, protocol)
    assert replay["maximum_nomination_power_violation_MW"] == 0
    assert replay["maximum_gross_power_violation_MW"] == 11


def test_minute_interior_unit_violation_cannot_be_hidden_by_net_energy():
    result, _, workload, protocol = oracle([[call(-50, 50, 0, 1)], []], initial_fraction=(98 - .1) / 98)
    assert result["status"] == "energy_or_commitment_infeasible"
    replay = replay_schedule(workload, 0, np.zeros((4, 2)), protocol)
    assert replay["final_unit_SOC_MWh"][0] < 98
    assert replay["maximum_inventory_bound_violation_MWh"] > .09


def test_perfect_information_feasibility_does_not_imply_causal_policy_success():
    units = [[call(49, 49, 20, 50)], []]
    result, _, workload, protocol = oracle(units)
    assert result["status"] == "replayed_feasible"
    terms = unit_planning_terms(receipt_plans(units, START, 120), protocol["asset"])
    causal = simulate_units(workload, np.ones(4), np.zeros((1, 4)), [0], 120, protocol["asset"], terms)
    assert not causal["success"][0, 0]


def test_suffix_preserves_distinct_actual_stocks_and_nonzero_pending_powers():
    _, _, workload, protocol = oracle([[], []], minutes=90, initial_fraction=[20 / 98, 80 / 98])
    pending = np.array([[10., -10.], [-10., 10.]])
    protocol["controls"]["initial_nomination_MW"] = pending.tolist()
    result, powers = unit_physical_oracle(workload, 0, 90, protocol)
    assert result["status"] == "replayed_feasible"
    assert np.allclose(powers[:2], pending)
    assert result["first_hour_maximum_nomination_MW"] == 10
    assert result["maximum_initial_nomination_deviation_MW"] < 1e-8
    prefix = replay_schedule(workload, 0, powers[:2], protocol)
    loss = 5 / .92 - 5 * .92
    assert np.allclose(prefix["final_unit_SOC_MWh"], [20 - loss, 80 - loss])
    changed = powers.copy()
    changed[0, 0] += 1
    assert np.isclose(replay_schedule(workload, 0, changed, protocol)["maximum_initial_nomination_deviation_MW"], 1)


def test_suffix_cannot_replace_depleted_stock_with_half_full_reset():
    units = [[call(49, 49, 0, 30)], []]
    actual, powers, _, _ = oracle(units, minutes=90, initial_fraction=[10 / 98, .5])
    reset, _, _, _ = oracle(units, minutes=90)
    assert actual["status"] == "energy_or_commitment_infeasible" and powers is None
    assert reset["status"] == "replayed_feasible"


def test_suffix_cannot_cancel_locked_nomination_to_avoid_boa_collision():
    _, _, workload, protocol = oracle([[], [call(49, 49, 0, 30)]], minutes=90)
    protocol["controls"]["initial_nomination_MW"] = [[-60, 0], [0, 0]]
    result, powers = unit_physical_oracle(workload, 0, 90, protocol)
    assert result["status"] == "energy_or_commitment_infeasible" and powers is None


def test_action_suffix_locks_third_slot_instead_of_repairing_candidate_nomination():
    units = [[call(20, 20, 90, 120)], []]
    free, _, workload, protocol = oracle(units, initial_fraction=[10 / 98, .5])
    assert free["status"] == "replayed_feasible"
    protocol["controls"]["initial_nomination_minutes"] = 90
    protocol["controls"]["initial_nomination_MW"] = [[0, 0], [0, 0], [0, 0]]
    fixed_bad, powers = unit_physical_oracle(workload, 0, 120, protocol)
    assert fixed_bad["status"] == "energy_or_commitment_infeasible" and powers is None
    protocol["controls"]["initial_nomination_MW"][2] = [-20, 0]
    fixed_good, powers = unit_physical_oracle(workload, 0, 120, protocol)
    assert fixed_good["status"] == "replayed_feasible"
    assert np.allclose(powers[:3], protocol["controls"]["initial_nomination_MW"])
    assert fixed_good["maximum_initial_nomination_deviation_MW"] < 1e-8


def test_segment_suffix_restricts_candidate_and_replay_detects_escape():
    units = [[call(20, 20, 90, 120)], []]
    _, _, workload, protocol = oracle(units, initial_fraction=[10 / 98, .5])
    segment = [[0, 0], [-20, 0]]
    result, powers = unit_physical_oracle(workload, 0, 120, protocol, segment)
    assert result["status"] == "replayed_feasible"
    assert -20 - 1e-8 <= powers[2, 0] <= 1e-8
    assert result["maximum_nomination_segment_deviation_MW"] < 1e-8
    bad, missing = unit_physical_oracle(workload, 0, 120, protocol, [[0, 0], [-1, 0]])
    assert bad["status"] == "energy_or_commitment_infeasible" and missing is None
    changed = powers.copy()
    changed[2, 0] = -30
    assert np.isclose(replay_schedule(workload, 0, changed, protocol, segment)["maximum_nomination_segment_deviation_MW"], 10)


def test_saturated_constant_action_segment_remains_fixed():
    _, _, workload, protocol = oracle([[call(20, 20, 90, 120)], []], initial_fraction=[10 / 98, .5])
    result, powers = unit_physical_oracle(workload, 0, 120, protocol, [[-20, 0], [-20, 0]])
    assert result["status"] == "replayed_feasible"
    assert np.allclose(powers[2], [-20, 0])
    assert result["maximum_nomination_segment_deviation_MW"] < 1e-8
