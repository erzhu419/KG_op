"""Absolute-power semantics, supported joint envelopes and honest delivery counts."""
import numpy as np
import pytest

from problems.public_battery_absolute_dispatch import declare_absolute
from problems.public_battery_causal_service import simulate_service
from problems.public_battery_pending_availability import evaluate_pending, reference_pulse_power
from problems.public_battery_visibility import trajectory

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
PEAKS = {"export": [49., 49.], "import": [-49., -49.]}
ZERO_PN = [[0., 0.], [0., 0.]]


def run(directions, rule="conservative_admission", initial=(49., 49.), locked=ZERO_PN):
    marks = [{"minute": 60 * i, "direction": d} for i, d in enumerate(directions)]
    return simulate_service(initial, locked, [.35, .75], marks, rule, len(marks) * 60,
                            [49., 49.], ASSET, absolute_request_peaks_MW=PEAKS)


@pytest.mark.parametrize("first_unit_PN", [27.048, 9.016])
@pytest.mark.parametrize("direction", ["export", "import"])
def test_saved_nonzero_PN_states_receive_absolute_peak_and_resume_reference(first_unit_PN, direction):
    locked = [[first_unit_PN, -53.26086956521739], [0., 0.]]
    declaration = declare_absolute([49., 49.], locked, PEAKS[direction], ASSET)
    result = evaluate_pending(declaration, PEAKS[direction], [49., 49.], ASSET)
    assert result["service_success"]
    assert declaration["locked_baseline_MW"] == pytest.approx(np.array(locked))
    assert declaration["anchor_MW"] == pytest.approx([0., 0.])
    assert result["admitted_peak_MW"] == pytest.approx(PEAKS[direction])
    first, last = result["power_first_MW"], result["power_last_MW"]
    assert first[0] == pytest.approx(locked[0])
    assert last[0] == pytest.approx(PEAKS[direction])
    assert first[1:31] == pytest.approx(np.tile(PEAKS[direction], (30, 1)))
    assert last[31] == pytest.approx(locked[1])
    assert first[32:] == pytest.approx(np.zeros((28, 2)))
    energy = .92 * result["gross_import_MWh"] - result["gross_export_MWh"] / .92
    assert np.array(result["physics"]["final_SOC_MWh"]) - 49 == pytest.approx(energy)
    assert result["total_unmet_energy_MWh"] == 0.


def test_no_activation_does_not_create_a_delivered_request_or_reserve_failure():
    summary, hours, _ = run([None], initial=(1., 97.))
    assert summary["physical_complete"] and summary["window_success"]
    assert summary["limited_declaration_hours"] == 1
    assert summary["planned_requests"] == summary["fulfilled_requests"] == 0
    assert hours[0]["request_fulfilled"] is None
    assert not hours[0]["request_observed"] and not hours[0]["service_failure_reasons"]


def test_partial_request_stays_failed_and_joint_amplitude_preserves_both_stocks():
    summary, hours, _ = run(["export"], initial=(1., 97.))
    assert summary["physical_complete"] and not summary["window_success"]
    assert summary["fulfilled_requests"] == 0 and summary["unfulfilled_observed_requests"] == 1
    assert summary["unmet_increase_MWh"][0] > 24
    assert summary["unmet_decrease_MWh"] == [0., 0.]
    assert hours[0]["admitted_peak_MW"][0] == hours[0]["admitted_peak_MW"][1]
    declaration = declare_absolute([1., 97.], ZERO_PN, PEAKS["export"], ASSET)
    for scale in (0., .25, .5, .75, 1.):
        first, last = reference_pulse_power(declaration["reference_first_MW"],
                                           declaration["reference_last_MW"],
                                           scale * declaration["guaranteed_peak_MW"])
        metrics = trajectory(first, last, [1., 97.], 60, ASSET)
        assert max(metrics["unit_bound_violation_MWh"]) <= 1e-8
        assert metrics["power_bound_violation_MW"] <= 1e-8


def test_unsafe_zero_revision_retains_reference_and_records_rejection():
    locked = [[-20., 0.], [20., 0.]]
    summary, hours, _ = run(["import"], initial=(3., 49.), locked=locked)
    assert summary["physical_complete"] and not summary["window_success"]
    assert hours[0]["declaration_reasons"]["import"] == "no_sustainable_neutral_pulse"
    assert hours[0]["admitted_peak_MW"] is None
    assert not hours[0]["request_fulfilled"] and summary["unmet_decrease_MWh"][0] > 0


def test_new_pulse_cannot_erase_known_unsafe_reference():
    summary, hours, _ = run(["import"], rule="full_request_reference", initial=(1., 49.),
                            locked=[[20., 0.], [20., 0.]])
    assert summary["physical_complete"] and not summary["window_success"]
    assert hours[0]["known_commitment_failure"] and not hours[0]["request_fulfilled"]
    assert summary["known_commitment_failure_hours"] == 1


def test_future_marks_do_not_enter_declarations_or_delayed_PN():
    _, first_hours, first_decisions = run(["export", "export", None])
    _, second_hours, second_decisions = run(["export", "import", None])
    assert first_hours[0] == second_hours[0]
    assert first_decisions[:2] == second_decisions[:2]


def test_interrupted_raw_delivery_keeps_all_planned_requests_and_censors_energy():
    summary, hours, _ = run(["export", None, "import"], rule="full_request_reference", initial=(1., 97.))
    assert summary["completed_minutes"] == 1 and not summary["physical_complete"]
    assert summary["planned_requests"] == 2 and summary["observed_requests"] == 1
    assert summary["unfulfilled_observed_requests"] == summary["unobserved_requests"] == 1
    assert summary["fulfilled_requests"] == 0 and summary["unfulfilled_or_unassessed_requests"] == 2
    assert not hours[0]["request_fulfilled"] and not hours[0]["delivery_accounting_complete"]
    assert summary["unmet_increase_MWh"] == [0., 0.]  # Raw execution tracked only the safe prefix.
