"""Received commitments replace PN, survive pulse expiry and exclude late receipts."""
import numpy as np
import pytest

from performance.inspect_public_battery_received_availability import fixture_reference
from problems.public_battery_pending_availability import declare_pending, evaluate_pending

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
PN = [[-13., -13.], [-13., -13.]]
SPEC = {"locked_baseline_MW": PN,
        "known_BOA": {"unit": 1, "received_minute": -1, "start_minute": 0,
                      "stop_minute": 60, "power_MW": 20}}


def test_received_boa_replaces_pn_without_another_instruction():
    reference = fixture_reference(SPEC, {})
    declaration = declare_pending([49., 49.], PN, [49., 49.], "export", ASSET,
                                  reference_power=reference)
    result = evaluate_pending(declaration, None, [49., 49.], ASSET)
    assert result["physical_feasible"] and result["service_success"]
    assert not result["request_observed"]
    assert result["gross_export_MWh"] == pytest.approx([20., 0.])
    assert result["gross_import_MWh"] == pytest.approx([0., 13.])
    # Headroom is relative to PN, not another 49 MW on top of the current 20.
    assert declaration["guaranteed_peak_MW"] == pytest.approx([36., 36.])


def test_absolute_zero_revision_restores_the_received_boa_at_expiry():
    declaration = declare_pending([49., 49.], PN, [49., 49.], "export", ASSET,
                                  reference_power=fixture_reference(SPEC, {}))
    result = evaluate_pending(declaration, [0., 0.], [49., 49.], ASSET)
    assert result["request_observed"] and result["service_success"]
    assert result["admitted_peak_MW"] == pytest.approx([0., 0.])
    assert result["power_first_MW"][0] == pytest.approx([20., -13.])
    assert result["power_last_MW"][31] == pytest.approx([20., -13.])
    assert np.all(result["power_first_MW"][32:] == [20., -13.])
    assert result["gross_export_MWh"] == pytest.approx([20 * 29 / 60, 0.])
    assert result["gross_import_MWh"] == pytest.approx([0., 13 * 29 / 60])


def test_future_receipt_cannot_change_the_current_reference_or_capacity():
    future = {"future_BOA": {"unit": 1, "received_minute": 10, "start_minute": 10,
                              "stop_minute": 60, "power_MW": 40}}
    plain, extra = fixture_reference(SPEC, {}), fixture_reference(SPEC, future)
    np.testing.assert_array_equal(plain, extra)
    first = declare_pending([49., 49.], PN, [49., 49.], "export", ASSET, reference_power=plain)
    second = declare_pending([49., 49.], PN, [49., 49.], "export", ASSET, reference_power=extra)
    assert first["baseline_physics"] == second["baseline_physics"]
    np.testing.assert_array_equal(first["available_capacity_MW"], second["available_capacity_MW"])


def test_unsafe_received_commitment_is_not_erased_by_a_rescuing_request():
    declaration = declare_pending([1., 97.], PN, [49., 49.], "export", ASSET,
                                  reference_power=fixture_reference(SPEC, {}))
    result = evaluate_pending(declaration, [0., 0.], [49., 49.], ASSET)
    assert declaration["known_commitment_failure"] and result["known_commitment_failure"]
    assert not result["physical_feasible"] and not result["service_success"]
    assert result["admitted_peak_MW"] is None and result["total_unmet_energy_MWh"] > 0


def test_already_received_tail_limits_capacity_and_sets_the_return_endpoint():
    first = np.tile([20., -13.], (60, 1))
    first[32:, 0] = 40.
    declaration = declare_pending([38., 49.], PN, [49., 49.], "export", ASSET,
                                  reference_power=(first, first))
    peak = (38 * .92 * 60 - .5 * 20 - (.5 + 28) * 40) / 31
    assert declaration["guaranteed_peak_MW"] == pytest.approx([peak, peak])
    result = evaluate_pending(declaration, [36., 36.], [49., 49.], ASSET)
    assert result["physical_feasible"] and not result["service_success"]
    assert result["power_last_MW"][31] == pytest.approx([40., -13.])
    assert result["physics"]["final_SOC_MWh"][0] == pytest.approx(0., abs=1.1e-8)


def test_reference_can_be_safe_even_when_pn_alone_and_neutral_revision_are_unsafe():
    reference = np.tile([-13., 20.], (60, 1))
    declaration = declare_pending([1., 97.], PN, [49., 49.], "export", ASSET,
                                  reference_power=(reference, reference))
    assert not declaration["known_commitment_failure"]
    assert declaration["reason"] == "no_sustainable_neutral_pulse"
    absent = evaluate_pending(declaration, None, [49., 49.], ASSET)
    zero = evaluate_pending(declaration, [0., 0.], [49., 49.], ASSET)
    assert absent["physical_feasible"] and zero["physical_feasible"]
    assert not zero["service_success"] and zero["admitted_peak_MW"] is None
    # A rejected zero request has both signed errors; they must not cancel.
    assert zero["unmet_increase_MWh"] == pytest.approx([13 * 31 / 60, 0.])
    assert zero["unmet_decrease_MWh"] == pytest.approx([0., 20 * 31 / 60])
