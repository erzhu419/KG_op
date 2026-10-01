"""Released announcements, admitted pulse suffixes and bounded safety fallback."""
from copy import deepcopy

import numpy as np
import pytest

from problems.public_battery_announced_nomination import nominate_announced
from problems.public_battery_causal_service import accepted_plan, simulate_service
from problems.public_battery_dispatch import positive_integral
from problems.public_battery_pending_availability import pending_pulse_power

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
PEAKS = {"export": [49., 49.], "import": [-49., -49.]}
ZERO = np.zeros(2)


def rows(first="export", second="export", minute=150):
    return [{"release_minute": delivery - 120, "delivery_minute": delivery,
             "direction": direction, "peak_MW": None if direction is None else PEAKS[direction]}
            for direction, delivery in zip((first, second), (minute + 30, minute + 90))]


def nominate(announcements, soc=(34.3, 73.5), minute=150, current=ZERO,
             following=ZERO, accepted=None, prefix=None, horizon=360):
    return nominate_announced(soc, current, following, [.35, .75],
                              accepted_plan(None, minute) if accepted is None else accepted,
                              minute, announcements, horizon, ASSET, execution_prefix=prefix)


def test_changes_to_unreleased_announcements_leave_policy_and_safety_unchanged():
    known = rows()
    future = rows("import", "import", minute=270)
    changed = deepcopy(future)
    changed[0].update(direction="export", peak_MW=PEAKS["export"])
    a, ai = nominate(known + future)
    b, bi = nominate(known + changed)
    np.testing.assert_array_equal(a, b)
    assert ai == bi and ai["reference_safety_path_evaluations"] == 3


def test_missing_active_or_inactive_announcement_is_not_assumed_empty():
    with pytest.raises(ValueError, match="missing released in-horizon announcement"):
        nominate(rows(None, None)[:1])
    command, info = nominate(rows(None, None)[:1], horizon=240)
    assert not info["known_reference_failure"] and info["second_direction"] is None
    assert np.abs(command).sum() <= 98


def test_own_admitted_suffix_replaces_PN_in_pending_stock_without_mutation():
    origin, tail = np.array([49., 49.]), np.array([-35., -35.])
    first, last = pending_pulse_power([ZERO, tail], PEAKS["export"])
    delta = (.92 * positive_integral(-first, -last, 1 / 60)
             - positive_integral(first, last, 1 / 60) / .92)
    soc = origin + np.cumsum(delta, axis=0)[29]
    pulse = {"start_minute": 120, "first_MW": first[:32], "last_MW": last[:32]}
    plan = accepted_plan(pulse, 150)
    before = {key: value.copy() for key, value in plan.items()}
    prefix = {"initial_SOC_MWh": origin, "first_MW": first[:30], "last_MW": last[:30]}
    _, info = nominate(rows(None, None), soc=soc, current=tail, accepted=plan, prefix=prefix)
    assert info["pending_inventory_change_MWh"] == pytest.approx(delta[30:].sum(axis=0), abs=1e-12)
    assert info["pending_inventory_change_MWh"] != pytest.approx([.92 * 35 / 2] * 2)
    assert info["reference_execution_prefix_minutes"] == 30
    for key in before:
        np.testing.assert_array_equal(plan[key], before[key])


def test_observed_boundary_keeps_the_original_hour_accumulation():
    origin = np.array([21.407847318429766, 60.60784731842977])
    locked = [[-28.026418872978777, -28.026418872978777], [-77.76873438027549, -20.23126561972451]]
    first, last = pending_pulse_power(locked, [38.96143282324511] * 2)
    pulse = {"start_minute": 120, "first_MW": first[:32], "last_MW": last[:32]}
    _, info = nominate(rows(None, None), soc=[.8236155536778185, 40.02361555367782],
                       current=locked[1], accepted=accepted_plan(pulse, 150),
                       prefix={"initial_SOC_MWh": origin, "first_MW": first[:30], "last_MW": last[:30]})
    assert not info["known_reference_failure"]
    assert max(info["zero_reference_physics"]["unit_bound_violation_MWh"]) <= 1e-8


@pytest.mark.parametrize("soc,reason", [([1, 1], "empty_stock_band"), ([28, 28], "shared_power_infeasible")])
def test_band_failure_keeps_safe_zero_and_records_planning_failure(soc, reason):
    command, info = nominate(rows(), soc=soc)
    np.testing.assert_array_equal(command, ZERO)
    assert info["planning_failure"] and not info["known_reference_failure"]
    assert info["nomination_reason"] == reason


def test_unsafe_zero_reference_remains_a_known_commitment_failure():
    command, info = nominate(rows(None, None), soc=[1, 49], current=[20, 0], following=[20, 0])
    np.testing.assert_array_equal(command, ZERO)
    assert info["known_reference_failure"]
    assert info["nomination_reason"] == "retained_unsafe_zero_reference"


def test_unsafe_preference_is_interpolated_within_the_frozen_budget():
    command, info = nominate(rows(), soc=[49, 90])
    assert info["nomination_reason"] == "safe_interpolation"
    assert 0 < info["reference_safety_scale"] < 1
    assert info["reference_safety_path_evaluations"] == 51
    assert info["scalar_stock_projection_evaluations"] <= 300
    assert max(info["nominated_reference_physics"]["unit_bound_violation_MWh"]) <= 1e-8
    assert np.abs(command).sum() <= 98


@pytest.mark.parametrize("rule", ["conservative_admission", "full_request_reference"])
def test_whole_hour_zero_slot_precedes_admission_and_inactive_slots_are_not_requests(rule):
    marks = [{"minute": minute, "direction": direction}
             for minute, direction in zip(range(0, 240, 60), (None, None, "export", "export"))]
    ledger = rows(minute=90)
    summary, hours, decisions = simulate_service([49, 49], [ZERO, ZERO], [.35, .75], marks,
                                               rule, 240, [49, 49], ASSET,
                                               absolute_request_peaks_MW=PEAKS, announcement_ledger=ledger)
    assert summary["physical_complete"] and summary["fulfilled_requests"] == 2
    assert summary["planned_requests"] == 2 and summary["inactive_hourly_clocks"] == 2
    assert [h["request_fulfilled"] for h in hours[:2]] == [None, None]
    assert hours[2]["locked_PN_MW"][0] == [0, 0]
    tail = next(d for d in decisions if d["minute"] == 90)
    assert hours[2]["locked_PN_MW"][1] == tail["nomination_MW"]
    assert all(d["nomination_MW"] == [0, 0] for d in decisions if d["minute"] % 60 == 0)


@pytest.mark.parametrize("rule", ["conservative_admission", "full_request_reference"])
def test_failed_requests_keep_the_denominator_and_interrupted_prefix(rule):
    marks = [{"minute": 0, "direction": "export"}, {"minute": 60, "direction": "export"}]
    summary, hours, _ = simulate_service([1, 97], [ZERO, ZERO], [.35, .75], marks,
                                        rule, 120, [49, 49], ASSET,
                                        absolute_request_peaks_MW=PEAKS, announcement_ledger=[])
    assert summary["planned_requests"] == 2 and not summary["window_success"]
    assert summary["fulfilled_requests"] < 2
    if rule == "full_request_reference":
        assert summary["completed_minutes"] == 1 and summary["unobserved_requests"] == 1
        assert hours[0]["delivery_prefix_minutes"] == 1 and not hours[0]["delivery_accounting_complete"]
    else:
        assert summary["physical_complete"] and summary["unfulfilled_observed_requests"] > 0
        assert sum(summary["unmet_increase_MWh"]) > 0
