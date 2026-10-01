"""Causal event order, locked PN, inactive obligations and stopped-prefix accounting."""
from datetime import timedelta

import numpy as np
import pytest

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_causal_service import receipt_marks, simulate_service

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
ZERO_PN = [[0., 0.], [0., 0.]]


def run(directions, rule="conservative_admission", initial=(49., 49.), targets=(.5, .5)):
    marks = [{"minute": 60 * i, "direction": direction} for i, direction in enumerate(directions)]
    return simulate_service(initial, ZERO_PN, targets, marks, rule, 60 * len(marks), [49., 49.], ASSET)


def test_marks_use_the_received_hour_and_keep_the_frozen_priority():
    start = instant("2024-04-15T00:00:00Z")
    at = lambda minute: start + timedelta(minutes=minute)
    units = [[Segment(at(0), at(32), at(-60), 1, 0., 20.),
              Segment(at(0), at(32), at(-1), 2, 0., 30.),
              Segment(at(10), at(42), at(10), 3, 0., 40.)],
             [Segment(at(0), at(32), at(-1), 2, 0., -30.),
              Segment(at(60), at(92), at(60), 4, 0., 0.)]]
    marks = receipt_marks(units, start, [0, 60, 120])
    assert [m["direction"] for m in marks] == ["import", "export", None]
    assert marks[0]["source"]["unit"] == 2
    assert marks[1]["source"]["received_minute"] == 10


def test_nomination_uses_the_now_received_pulse_but_keeps_the_current_hour_locked():
    summary, hours, nominations = run(["export", None, None])
    assert summary["physical_complete"]
    assert hours[0]["available_capacity_MW"] == [[49., 49.], [49., 49.]]
    # 31 equivalent peak-minutes; this energy is already committed before PN.
    required_import = (49 * 31 / 60 / .92) / (.5 * .92)
    assert nominations[0]["delivery_minute"] == 60
    assert required_import > 49.  # The site bound caps joint replenishment at 98 MW.
    assert nominations[0]["nomination_MW"] == pytest.approx([-49.] * 2)
    assert hours[1]["initial_SOC_MWh"] == pytest.approx([49 - 49 * 31 / 60 / .92] * 2)
    assert hours[0]["locked_PN_MW"] == ZERO_PN
    assert nominations[1]["locked_PN_MW"][0] == [0., 0.]
    assert nominations[1]["locked_PN_MW"][1] == pytest.approx(nominations[0]["nomination_MW"])
    assert max(abs(v) for v in summary["energy_balance_error_MWh"]) < 1e-10


def test_future_marks_leave_current_declarations_and_nominations_unchanged():
    _, first_hours, first_decisions = run(["export", "export", None])
    _, second_hours, second_decisions = run(["export", "import", None])
    assert first_hours[0] == second_hours[0]
    assert first_decisions[:2] == second_decisions[:2]


def test_inactive_hours_do_not_hide_a_fixed_service_capacity_shortfall():
    summary, hours, _ = run([None], initial=(1., 97.))
    assert summary["physical_complete"] and not summary["window_success"]
    assert summary["activation_hours"] == 0 and summary["capacity_shortfall_hours"] == 1
    assert hours[0]["service_failure_reasons"] == ["firm_availability_shortfall"]
    assert summary["unmet_increase_MWh"] == [0., 0.]
    assert summary["unmet_decrease_MWh"] == [0., 0.]


def test_admission_preserves_service_failure_and_signed_delivery_shortfall():
    summary, hours, _ = run(["export"], initial=(1., 97.))
    assert summary["physical_complete"] and not summary["window_success"]
    assert summary["request_shortfall_hours"] == 1
    assert np.sum(summary["unmet_increase_MWh"]) > 40
    assert summary["unmet_decrease_MWh"] == [0., 0.]
    assert hours[0]["delivery_accounting_complete"]
    assert summary["minimum_accepted_SOC_MWh"][0] >= -1.1e-8


def test_full_request_physical_failure_does_not_become_a_completed_service_window():
    summary, hours, _ = run(["export", None], rule="full_request_reference", initial=(1., 97.))
    assert not summary["physical_complete"] and not summary["window_success"]
    assert summary["first_physical_failure"]["minute"] == 1
    assert summary["completed_minutes"] == 1
    assert summary["unassessed_service_hours"] == 1
    assert not summary["delivery_accounting_complete"]
    assert not hours[0]["delivery_accounting_complete"] and not hours[0]["service_success"]
    assert "physical_failure" in hours[0]["service_failure_reasons"]
    assert summary["final_or_last_safe_SOC_MWh"] == pytest.approx([1 - 49 / 120 / .92, 97 - 49 / 120 / .92])


def test_observed_import_boundary_uses_the_same_accumulation_as_the_declaration():
    # First pilot's minute-1680 state, which previously false-stopped at 1711.
    initial = [49.795311713169205, 88.99531171316954]
    locked = [[28.511373552231436, 28.511373552231696],
              [-1.0812606848523264e-13, 7.844391802791507e-14]]
    marks = [{"minute": 0, "direction": "import"}]
    summary, hours, _ = simulate_service(initial, locked, [.35, .75], marks,
                                         "conservative_admission", 60, [49., 49.], ASSET)
    assert summary["physical_complete"] and not summary["window_success"]
    assert summary["maximum_accepted_SOC_MWh"][1] <= 98. + 1e-8
    assert hours[0]["delivery_accounting_complete"]
