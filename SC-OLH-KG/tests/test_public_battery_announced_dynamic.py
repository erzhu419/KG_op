"""Dynamic target routing and cost censoring without real policy/physics calls."""
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest

from performance.inspect_public_battery_announced_dynamic import economics, instant, schedule
from problems.public_battery_causal_service import simulate_service

ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
COUNTS = {"mocked_service_executions": 0, "mocked_nomination_calls": 0,
          "mocked_declaration_calls": 0, "mocked_minute_physics_calls": 0}


def mocked_service(targets):
    seen = []

    def nomination(soc, current, following, fractions, plan, minute, *args, **kwargs):
        seen.append((minute, np.asarray(fractions).tolist()))
        COUNTS["mocked_nomination_calls"] += 1
        return np.zeros(2), {"known_reference_failure": False, "planning_failure": False,
                            "nomination_reason": "test_routing", "reference_safety_path_evaluations": 0,
                            "scalar_stock_projection_evaluations": 0}

    def declaration(*args, **kwargs):
        COUNTS["mocked_declaration_calls"] += 1
        return {"envelope_path_evaluations": 0, "available_capacity_MW": [49., 49.],
                "known_commitment_failure": False, "reason": "test_routing"}

    def physics(first, last, soc, *args):
        COUNTS["mocked_minute_physics_calls"] += 1
        return {"final_SOC_MWh": soc, "minimum_SOC_MWh": soc, "maximum_SOC_MWh": soc,
                "unit_bound_violation_MWh": [0., 0.], "power_bound_violation_MW": 0.}

    marks = [{"minute": i, "direction": None} for i in range(0, 240, 60)]
    with patch("problems.public_battery_causal_service.nominate_announced", side_effect=nomination), \
         patch("problems.public_battery_causal_service.declare_absolute", side_effect=declaration), \
         patch("problems.public_battery_causal_service.trajectory", side_effect=physics):
        result = simulate_service([49, 49], [[0, 0], [0, 0]], targets, marks,
                                 "conservative_admission", 240, [49, 49], ASSET,
                                 absolute_request_peaks_MW={"export": [49, 49], "import": [-49, -49]},
                                 announcement_ledger=[])
    COUNTS["mocked_service_executions"] += 1
    return result, seen


def test_dynamic_rows_reach_correct_units_without_terminal_rows():
    targets = np.array([[.1 * i, .9 - .1 * i] for i in range(6)])
    (_, _, nominations), seen = mocked_service(targets)
    assert [minute for minute, _ in seen] == [0, 30, 60, 90, 120, 150]
    assert [value for _, value in seen] == targets.tolist()
    assert [r["target_fractions"] for r in nominations] == targets.tolist()
    assert nominations[-1]["delivery_minute"] == 210


def test_static_vector_and_constant_schedule_have_equal_routing():
    a, seen_a = mocked_service([.35, .75])
    b, seen_b = mocked_service(np.tile([.35, .75], (6, 1)))
    assert a == b and seen_a == seen_b


def test_target_row_count_is_the_number_of_eligible_nominations():
    with pytest.raises(ValueError, match="eligible nomination"):
        simulate_service([49, 49], [[0, 0], [0, 0]], np.zeros((8, 2)), [],
                         "conservative_admission", 240, [49, 49], ASSET)


def test_forecast_schedule_uses_actual_delivery_clock_and_separate_channels():
    start = instant("2024-10-25T00:00:00Z")
    rows = {start + timedelta(minutes=30 * i):
            {"forecast_MW": 100 + 10 * i, "forecast_publish_time_utc": (start - timedelta(hours=2)).isoformat()}
            for i in range(1, 8)}
    fit = {"forecast_q05_MW": 100., "forecast_q95_MW": 200., "absolute_ramp_q95_MW": 100.}
    targets, metadata = schedule(rows, start, 240, [0., 1.], [[0., 1.], [1., 0.]], fit)
    expected = np.arange(2, 8) / 10
    np.testing.assert_allclose(targets, np.column_stack((expected, 1 - expected)))
    assert metadata["eligible_decisions"] == 6
    assert metadata["last_eligible_delivery_utc"] == (start + timedelta(minutes=210)).isoformat()
    rows[start + timedelta(minutes=30)]["forecast_publish_time_utc"] = (start + timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError, match="publication"):
        schedule(rows, start, 240, [0., 1.], [[0., 1.], [1., 0.]], fit)


@pytest.mark.parametrize("success,complete", [(False, True), (False, False), (True, False)])
def test_failed_or_partial_windows_have_no_full_cost(success, complete):
    with patch("performance.inspect_public_battery_announced_dynamic.pending_pulse_power", side_effect=AssertionError("censored path")):
        value = economics({"window_success": success, "physical_complete": complete}, [], None, {}, ASSET)
    assert value["cost_GBP"] is None


def test_successful_cost_uses_actual_prices_and_inventory_adjustment():
    start = instant("2024-04-25T01:00:00Z")
    final = [49. - 5. / .92, 49. + .92 * 10.]
    result = {"window_success": True, "physical_complete": True, "planned_minutes": 60, "completed_minutes": 60,
              "gross_import_MWh": [0., 10.], "gross_export_MWh": [5., 0.], "final_or_last_safe_SOC_MWh": final}
    hours = [{"minute": 0, "initial_SOC_MWh": [49, 49], "delivery_prefix_minutes": 60,
              "delivery_accounting_complete": True, "locked_PN_MW": [[10., 0.], [0., -20.]], "admitted_peak_MW": None}]
    prices = {start: {"price_GBP_per_MWh": 10.}, start + timedelta(minutes=30): {"price_GBP_per_MWh": -20.}}
    value = economics(result, hours, start, prices, ASSET)
    assert value["metered_cash_GBP"] == pytest.approx(-250.)
    assert value["inventory_adjustment_GBP"] == pytest.approx(69.28)
    assert value["cost_GBP"] == pytest.approx(-180.72)
    assert value["last_execution_settlement_utc"] == (start + timedelta(minutes=30)).isoformat()


def test_incomplete_success_record_blocks_economic_accounting():
    with pytest.raises(ValueError, match="incomplete"):
        economics({"window_success": True, "physical_complete": True, "planned_minutes": 60, "completed_minutes": 59},
                  [], None, {}, ASSET)
