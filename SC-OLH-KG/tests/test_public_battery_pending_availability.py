"""Nonzero PN, empty instructions and the full locked tail affect obligations."""
import numpy as np
import pytest

from problems.public_battery_pending_availability import declare_pending, evaluate_pending

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
STEADY = [[-13., -13.], [-13., -13.]]
CHANGING = [[-13., -13.], [13., 13.]]


def test_nonzero_pn_offer_is_headroom_but_instruction_is_absolute():
    declaration = declare_pending([49., 49.], STEADY, [49., 49.], "export", ASSET)
    result = evaluate_pending(declaration, [36., 36.], [49., 49.], ASSET)
    assert declaration["guaranteed_peak_MW"] == pytest.approx([36., 36.])
    assert result["service_success"] and result["physical_feasible"]
    imported = 2 * 13**2 / (2 * 49 * 60) + 28 * 13 / 60
    exported = 2 * 36**2 / (2 * 49 * 60) + 30 * 36 / 60
    assert result["gross_import_MWh"] == pytest.approx([imported, imported])
    assert result["gross_export_MWh"] == pytest.approx([exported, exported])
    assert result["physics"]["final_SOC_MWh"] == pytest.approx([49 + .92 * imported - exported / .92] * 2)


def test_no_instruction_and_absolute_zero_have_different_energy():
    declaration = declare_pending([49., 49.], STEADY, [49., 49.], "export", ASSET)
    absent = evaluate_pending(declaration, None, [49., 49.], ASSET)
    zero = evaluate_pending(declaration, [0., 0.], [49., 49.], ASSET)
    assert not absent["request_observed"] and zero["request_observed"]
    assert absent["service_success"] and zero["service_success"]
    assert absent["gross_import_MWh"] == pytest.approx([13., 13.])
    assert zero["gross_import_MWh"] == pytest.approx([13 * 29 / 60] * 2)
    assert zero["admitted_peak_MW"] == pytest.approx([0., 0.])


def test_import_reservation_accounts_for_baseline_and_shared_gross_limit():
    declaration = declare_pending([49., 49.], STEADY, [49., 49.], "import", ASSET)
    result = evaluate_pending(declaration, [-62., -62.], [49., 49.], ASSET)
    assert result["physical_feasible"] and not result["service_success"]
    assert declaration["guaranteed_peak_MW"] == pytest.approx([-49., -49.])
    assert declaration["available_capacity_MW"] == pytest.approx([36., 36.])
    assert result["unmet_decrease_MWh"] == pytest.approx([13 * 31 / 60] * 2)


@pytest.mark.parametrize("direction,peak", [("export", 62.), ("import", -62.)])
def test_changed_locked_pn_remains_in_headroom_and_dispatch_tail(direction, peak):
    declaration = declare_pending([49., 49.], CHANGING, [49., 49.], direction, ASSET)
    result = evaluate_pending(declaration, [peak, peak], [49., 49.], ASSET)
    assert result["physical_feasible"] and not result["service_success"]
    assert declaration["available_capacity_MW"] == pytest.approx([36., 36.])
    assert result["power_last_MW"][31].tolist() == [13., 13.]
    assert np.all(result["power_first_MW"][32:] == 13.)


def test_capacity_covers_the_locked_tail_not_only_the_boas_32_minutes():
    baseline = [[13., 13.], [13., 13.]]
    declaration = declare_pending([20., 20.], baseline, [49., 49.], "export", ASSET)
    # 31 peak-equivalent minutes plus 29 baseline-equivalent minutes.
    peak = (20 * .92 * 60 - 29 * 13) / 31
    assert declaration["guaranteed_peak_MW"] == pytest.approx([peak, peak])
    result = evaluate_pending(declaration, [62., 62.], [49., 49.], ASSET)
    assert result["physical_feasible"] and not result["service_success"]
    assert result["physics"]["final_SOC_MWh"] == pytest.approx([0., 0.], abs=1.1e-8)


def test_stock_bound_includes_the_return_ramps_internal_zero_crossing():
    declaration = declare_pending([75., 75.], CHANGING, [49., 49.], "import", ASSET)
    result = evaluate_pending(declaration, [-62., -62.], [49., 49.], ASSET)
    assert result["physical_feasible"]
    assert result["physics"]["maximum_SOC_MWh"] == pytest.approx([98., 98.], abs=1.1e-8)
    # Peak charge occurs strictly inside the last ramp, rather than at a minute node.
    assert np.max(result["SOC_trace_MWh"]) < 98 - 1e-4


def test_an_infeasible_locked_plan_cannot_depend_on_an_unseen_rescue():
    declaration = declare_pending([1., 97.], STEADY, [49., 49.], "export", ASSET)
    absent = evaluate_pending(declaration, None, [49., 49.], ASSET)
    assert declaration["known_commitment_failure"]
    assert not absent["physical_feasible"] and not absent["service_success"]
    assert absent["physics"]["final_SOC_MWh"] == pytest.approx([12.96, 108.96])
    rescue = evaluate_pending(declaration, [-5., 36.], [49., 49.], ASSET)
    assert rescue["known_commitment_failure"] and not rescue["service_success"]
    assert rescue["admitted_peak_MW"] is None
    assert rescue["total_unmet_energy_MWh"] > 0
