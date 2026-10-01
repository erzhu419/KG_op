"""Availability must protect physics without concealing unmet service."""
import numpy as np
import pytest

from problems.public_battery_availability import declare_pulse, evaluate_pulse

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}


@pytest.mark.parametrize("sign", [1, -1])
def test_half_full_fixture_delivers_exact_ramp_and_hold_energy(sign):
    request = np.array([49., 49.]) * sign
    declaration = declare_pulse([49., 49.], request, ASSET)
    result = evaluate_pulse(declaration, request, request, ASSET)
    # Independently, 30 full minutes plus two half-minute triangles.
    grid_energy = 49 * (30 + .5 + .5) / 60
    final = 49 - grid_energy / .92 if sign == 1 else 49 + grid_energy * .92
    assert result["physical_feasible"] and result["service_success"]
    assert result["physics"]["final_SOC_MWh"] == pytest.approx([final, final])
    assert result["total_unmet_energy_MWh"] == pytest.approx(0.)


@pytest.mark.parametrize("sign,limited_unit", [(1, 0), (-1, 1)])
def test_unequal_stock_partial_dispatch_is_still_service_failure(sign, limited_unit):
    request = np.array([49., 49.]) * sign
    declaration = declare_pulse([1., 97.], request, ASSET)
    result = evaluate_pulse(declaration, request, request, ASSET)
    available_energy = .92 if sign == 1 else 1 / .92
    assert result["physical_feasible"] and not result["service_success"]
    assert result["total_unmet_energy_MWh"] == pytest.approx(49 * 31 / 60 - available_energy)
    assert result["physics"]["final_SOC_MWh"][limited_unit] == pytest.approx(0. if sign == 1 else 98.)
    assert result["availability_shortfall_MW"][limited_unit] > 0


def test_opposed_requests_use_gross_site_power_and_retain_both_deficits():
    declaration = declare_pulse([49., 49.], [60., -60.], ASSET)
    result = evaluate_pulse(declaration, [60., -60.], [49., -49.], ASSET)
    assert declaration["guaranteed_MW"] == pytest.approx([49., -49.])
    assert result["physical_feasible"] and not result["service_success"]
    assert result["unmet_export_MWh"] == pytest.approx([11 * 31 / 60, 0.])
    assert result["unmet_import_MWh"] == pytest.approx([0., 11 * 31 / 60])
    assert result["total_unmet_energy_MWh"] == pytest.approx(22 * 31 / 60)


def test_zero_declaration_cannot_hide_service_obligation_when_no_request_arrives():
    empty = declare_pulse([49., 49.], [0., 0.], ASSET)
    result = evaluate_pulse(empty, [0., 0.], [49., 49.], ASSET)
    assert result["physical_feasible"] and not result["service_success"]
    assert result["total_unmet_energy_MWh"] == 0
    assert result["availability_shortfall_MW"] == pytest.approx([49., 49.])
    full = declare_pulse([49., 49.], [49., 49.], ASSET)
    idle = evaluate_pulse(full, [0., 0.], [49., 49.], ASSET)
    assert idle["service_success"]
    assert idle["physics"]["final_SOC_MWh"] == [49., 49.]


def test_unbacked_declaration_fails_availability_even_without_activation():
    declaration = declare_pulse([1., 97.], [49., 49.], ASSET)
    result = evaluate_pulse(declaration, [0., 0.], [49., 49.], ASSET)
    assert result["physical_feasible"] and not result["service_success"]
    assert result["availability_shortfall_MW"][0] == pytest.approx(49 - .92 * 60 / 31)
    assert result["total_unmet_energy_MWh"] == 0
