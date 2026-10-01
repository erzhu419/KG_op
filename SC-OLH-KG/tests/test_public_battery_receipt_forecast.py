"""Receipt-prefix causality, inventory allocation and genuine reference safety."""
from datetime import timedelta

import numpy as np
import pytest

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_causal_service import accepted_plan, receipt_marks, simulate_service
from problems.public_battery_receipt_forecast import nominate_receipt_forecast, prioritize_inventory
from problems.public_battery_pending_availability import reference_pulse_power
from problems.public_battery_visibility import trajectory

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
PEAKS = {"export": [49., 49.], "import": [-49., -49.]}
ZERO = np.zeros(2)


def cue(minute, direction):
    return {"minute": minute, "direction": direction, "source": None if direction is None
            else {"received_minute": minute - 1, "unit": 1, "acceptance_number": 1}}


def nomination(soc=(34.3, 73.5), target=(34.3, 73.5), current=ZERO, following=ZERO,
               minute=330, direction="export", accepted=None):
    return nominate_receipt_forecast(np.array(soc), np.array(current), np.array(following), np.array(target),
                                     accepted_plan(None, minute) if accepted is None else accepted,
                                     minute, cue(minute, direction), PEAKS, ASSET)


def test_empty_cue_keeps_the_ordinary_command_and_does_not_run_a_forecast():
    command, info = nomination(direction=None)
    assert command == pytest.approx([0., 0.])
    assert not info["forecast_used"] and info["reference_safety_path_evaluations"] == 0


def test_whole_hour_cue_keeps_received_pulse_control():
    command, info = nomination(minute=360)
    assert command == pytest.approx(info["ordinary_nomination_MW"])
    assert not info["forecast_used"] and info["analytic_nomination_calls"] == 1


def test_forecast_never_mutates_the_actual_accepted_plan():
    plan = accepted_plan(None, 330)
    before = {key: value.copy() for key, value in plan.items()}
    command, info = nomination(accepted=plan)
    for key in before:
        np.testing.assert_array_equal(plan[key], before[key])
    assert info["forecast_used"] and info["forecast_call_minute"] == 360
    assert command[0] < 0 and np.abs(command).sum() <= 98. + 1e-8
    assert max(info["nominated_reference_physics"]["unit_bound_violation_MWh"]) <= 1e-8


@pytest.mark.parametrize("direction,projected,preferred", [
    ("export", [1., 70.], [-49., -49.]), ("import", [110., 30.], [49., 49.])])
def test_inventory_priority_keeps_the_critical_unit_in_the_planning_band(direction, projected, preferred):
    lower, upper = np.zeros(2), np.full(2, 98.)
    if direction == "export":
        lower[:] = 49 * 31 / 60 / .92
    else:
        upper[:] -= 49 * 31 / 60 * .92
    command, info = prioritize_inventory(np.array(preferred), np.array(projected), np.full(2, .5),
                                          np.full(2, 98.), lower, upper, ASSET)
    assert info["forecast_band_feasible"]
    assert abs(command[0]) > 49 and abs(command[1]) < 49
    assert np.abs(command).sum() <= 98. + 1e-8
    stock = np.array(info["forecast_stock_after_candidate_MWh"])
    assert np.all(stock >= lower - 1e-8) and np.all(stock <= upper + 1e-8)


def test_infeasible_forecast_band_is_reported_without_becoming_a_service_promise():
    preferred = np.array([-49., -49.])
    command, info = prioritize_inventory(preferred, np.array([-5., -5.]), np.full(2, .5),
                                          np.full(2, 98.), np.full(2, 49 * 31 / 60 / .92),
                                          np.full(2, 98.), ASSET)
    assert not info["forecast_band_feasible"] and info["forecast_required_gross_power_MW"] > 98
    np.testing.assert_array_equal(command, preferred)


def test_forecast_cannot_erase_an_unsafe_ordinary_reference():
    command, info = nomination(soc=(1., 49.), current=[20., 0.], following=[20., 0.])
    assert info["known_reference_failure"]
    assert info["nomination_reason"] == "retained_unsafe_ordinary_reference"
    np.testing.assert_array_equal(command, info["ordinary_nomination_MW"])


def test_unsafe_forecast_command_is_interpolated_against_the_actual_reference():
    command, info = nomination(soc=(49., 90.), target=(49., 90.))
    assert info["nomination_reason"] == "safe_interpolation"
    assert 0 < info["reference_safety_scale"] < 1
    assert info["reference_safety_path_evaluations"] == 50
    power = np.repeat(np.vstack((ZERO, ZERO, command)), 30, axis=0)
    metrics = trajectory(power, power, np.array([49., 90.]), 90, ASSET)
    assert max(metrics["unit_bound_violation_MWh"]) <= 1e-8
    assert metrics["power_bound_violation_MW"] <= 1e-8


def run(directions, cues):
    marks = [{"minute": 60 * i, "direction": d} for i, d in enumerate(directions)]
    return simulate_service([49., 49.], [[0., 0.], [0., 0.]], [.5, .5], marks,
                            "conservative_admission", 60 * len(marks), [49., 49.], ASSET,
                            absolute_request_peaks_MW=PEAKS, nomination_cues=cues)


def test_predicting_a_call_does_not_execute_it_or_create_a_delivery_observation():
    cues = [cue(0, None), cue(30, "export"), cue(60, None), cue(90, None)]
    summary, hours, decisions = run([None, None, None], cues)
    assert summary["physical_complete"] and summary["fulfilled_requests"] == 0
    assert decisions[1]["forecast_used"] and decisions[1]["delivery_minute"] == 90
    assert hours[1]["initial_SOC_MWh"] == [49., 49.]
    assert all(h["request_fulfilled"] is None for h in hours)


def test_future_marks_and_cues_do_not_affect_current_declarations_or_nominations():
    first_cues = [cue(0, None), cue(30, "export"), cue(60, "export"), cue(90, "export")]
    second_cues = [cue(0, None), cue(30, "export"), cue(60, "import"), cue(90, "import")]
    _, first_hours, first_decisions = run([None, "export", None], first_cues)
    _, second_hours, second_decisions = run([None, "import", None], second_cues)
    assert first_hours[0] == second_hours[0]
    assert first_decisions[:2] == second_decisions[:2]


def test_receipt_after_nomination_cannot_supply_the_current_cue():
    start = instant("2024-04-15T00:00:00Z")
    at = lambda minute: start + timedelta(minutes=minute)
    known = Segment(at(314), at(315), at(301), 6770, 0., 2.)
    future = Segment(at(352), at(359), at(349), 6771, 0., 49.)
    changed_future = Segment(at(352), at(359), at(349), 6771, 0., -49.)
    first = receipt_marks([[known, future], []], start, [330])
    second = receipt_marks([[known, changed_future], []], start, [330])
    assert first == second and first[0]["direction"] == "export"
    assert first[0]["source"]["received_minute"] == 301


def test_observed_half_hour_reference_retains_the_executed_hour_origin():
    initial = np.array([21.407847318429766, 60.60784731842977])
    locked = [[-28.026418872978777, -28.026418872978777], [-77.76873438027549, -20.23126561972451]]
    baseline = np.repeat(locked, 30, axis=0)
    first, last = reference_pulse_power(baseline, baseline, [38.96143282324511] * 2)
    pulse = {"start_minute": 3660, "first_MW": first[:32], "last_MW": last[:32]}
    _, info = nominate_receipt_forecast(
        np.array([0.8236155536778185, 40.02361555367782]), np.array(locked[1]),
        np.array([-1.1175387557768315, -55.81636666271207]), np.array([34.3, 73.5]),
        accepted_plan(pulse, 3690), 3690, cue(3690, "export"), PEAKS, ASSET,
        execution_prefix={"initial_SOC_MWh": initial, "first_MW": first[:30], "last_MW": last[:30]})
    assert not info["known_reference_failure"]
    assert max(info["ordinary_reference_physics"]["unit_bound_violation_MWh"]) <= 1e-8
    assert info["reference_execution_prefix_minutes"] == 30
