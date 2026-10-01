from datetime import timedelta
import json

import numpy as np

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_units import forced_run_bounds, nominate_units, simulate_units, unit_planning_terms

ASSET = {"power_MW": 98., "energy_MWh": 196., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}
NOW = instant("2024-04-15T00:00:00Z")


def instruction(first, last, left=60, right=90, received=0):
    return Segment(NOW + timedelta(minutes=left), NOW + timedelta(minutes=right),
                   NOW + timedelta(minutes=received), 1, first, last)


def nomination(units, target=49.):
    terms = unit_planning_terms(receipt_plans(units, NOW, 120), ASSET)
    return nominate_units(np.full((1, 1, 2), 49.), np.zeros((1, 1, 2)),
                          np.zeros((1, 1, 2)), np.full((1, 1, 2), target),
                          np.array([0]), terms, ASSET)[0, 0]


def run(units, minutes, target=.5, prices=None):
    workload = site_workload(units, NOW, NOW + timedelta(minutes=minutes))
    terms = unit_planning_terms(receipt_plans(units, NOW, minutes), ASSET)
    prices = np.ones(minutes // 30) if prices is None else prices
    return simulate_units(workload, prices, np.full((1, minutes // 30), target),
                           [0], minutes, ASSET, terms)


def test_unreceived_future_instruction_does_not_change_nomination():
    a = nomination([[instruction(20, 20, received=30)], []])
    b = nomination([[instruction(-40, -40, received=30)], []])
    assert np.array_equal(a, b) and np.array_equal(a, [0, 0])


def test_received_pending_instruction_replenishes_its_own_unit():
    units = [[instruction(20, 20, left=0, right=30)], []]
    command = nomination(units)
    assert np.isclose(command[0], -20 / .92 ** 2) and command[1] == 0
    result = run(units, 90)
    assert result["success"][0, 0]
    assert np.allclose(result["unit_final_SOC_MWh"][0, 0], [49, 49])
    assert np.max(np.abs(result["energy_balance_error_MWh"])) < 1e-10


def test_known_boa_headroom_and_joint_nomination_power_limit():
    command = nomination([[instruction(50, 50)], []], target=98)
    assert np.allclose(command, [0, -48])
    no_instruction = nomination([[], []], target=98)
    assert np.allclose(no_instruction, [-49, -49])


def test_other_unit_surplus_cannot_supply_an_empty_unit():
    units = [[instruction(20, 20, left=0, right=600)],
             [instruction(-20, -20, left=0, right=600)]]
    result = run(units, 600)
    assert not result["success"][0, 0]
    assert result["unit_inventory_failure"][0, 0, 0]
    assert result["BOA_inventory_failure"][0, 0]
    assert not result["power_failure"][0, 0]
    assert np.isnan(result["cost_GBP"][0, 0])


def test_unit_ramp_reversal_checks_inventory_before_safe_minute_end():
    units = [[instruction(-50, 50, left=0, right=1)], []]
    asset = ASSET
    workload = site_workload(units, NOW, NOW + timedelta(minutes=30))
    terms = unit_planning_terms(receipt_plans(units, NOW, 30), asset)
    result = simulate_units(workload, np.ones(1), np.full((1, 1), .5), [0], 1,
                            asset, terms, initial_fraction=(98 - .1) / 98)
    # A whole-minute net calculation ends below capacity, but charging first
    # adds .192 MWh and violates the unit bound before the ramp crosses zero.
    assert result["unit_inventory_failure"][0, 0, 0] and not result["success"][0, 0]


def test_nomination_lead_and_accounting_prices_do_not_change_control():
    a = run([[], []], 120, target=.8)
    b = run([[], []], 120, target=.8, prices=np.array([1000., -1000., 1., 1.]))
    assert a["success"][0, 0] and b["success"][0, 0]
    assert a["nomination_trace"] == b["nomination_trace"]
    assert np.isclose(a["metered_cash_GBP"][0, 0], b["metered_cash_GBP"][0, 0])
    assert np.allclose(a["unit_final_SOC_MWh"][0, 0], [78.4, 78.4])
    initial = np.array(ASSET["unit_energy_MWh"]) * .5
    balance = (initial + .92 * a["unit_gross_import_MWh"][0, 0]
               - a["unit_gross_export_MWh"][0, 0] / .92)
    assert np.allclose(balance, a["unit_final_SOC_MWh"][0, 0])


def test_forced_run_bound_preserves_zero_instructions_and_resets_only_in_idle_gaps():
    workload = {"active": np.array([[True, False], [True, False], [True, False],
                                    [False, False], [True, False]]),
                "first_MW": np.array([[30., 0], [0, 0], [30., 0], [0, 0], [30., 0]]),
                "last_MW": np.array([[30., 0], [0, 0], [30., 0], [0, 0], [30., 0]])}
    bound = forced_run_bounds(workload, 0, 5, ASSET)[0]
    assert np.isclose(bound["required_capacity_MWh"], 1 / .92)
    assert (bound["from_minute"], bound["to_minute"]) == (0, 3)


def test_previous_nomination_snapshot_retains_actual_stock_and_queue_order():
    units = [[instruction(49, 49, left=80, right=180, received=80)], []]
    result = run(units, 180, target=.8)
    context = result["first_failure_contexts"][0]
    previous = context["previous_nomination"]
    assert context["nomination_decision_minute"] == 90
    assert previous["nomination_decision_minute"] == 60
    assert previous["decision_SOC_MWh"] == [49, 49]
    assert 49 < context["decision_SOC_MWh"][0] < 78.4
    assert previous["pending_MW"][1] == context["pending_MW"][0]
    assert previous["nomination_MW"] == context["pending_MW"][1]
    json.dumps(context)


def test_equal_unit_channels_preserve_shared_engine_outputs():
    units = [[instruction(49, 49, left=20, right=50, received=20)], []]
    workload = site_workload(units, NOW, NOW + timedelta(minutes=180))
    terms = unit_planning_terms(receipt_plans(units, NOW, 180), ASSET)
    shared = np.array([np.full(6, .2), np.full(6, .8)])
    a = simulate_units(workload, np.arange(6), shared, [0, 60], 120, ASSET, terms)
    b = simulate_units(workload, np.arange(6), np.repeat(shared[..., None], 2, axis=-1), [0, 60], 120, ASSET, terms)
    for field in ("success", "first_failure_minute", "cost_GBP", "unit_final_SOC_MWh",
                  "energy_balance_error_MWh", "power_failure", "unit_inventory_failure"):
        assert np.array_equal(a[field], b[field], equal_nan=True)
    assert a["nomination_trace"] == b["nomination_trace"]
    assert a["first_failure_contexts"] == b["first_failure_contexts"]


def test_independent_channels_keep_unit_assignment_batch_axes_and_nomination_lead():
    workload = site_workload([[], []], NOW, NOW + timedelta(minutes=180))
    terms = unit_planning_terms(receipt_plans([[], []], NOW, 180), ASSET)
    targets = np.broadcast_to(np.array([[.25, .75], [.75, .25]])[:, None, :], (2, 6, 2))
    result = simulate_units(workload, np.ones(6), targets, [0, 60], 120, ASSET, terms)
    assert np.all(result["success"])
    assert np.allclose(result["unit_final_SOC_MWh"], np.array([[24.5, 73.5], [73.5, 24.5]])[:, None, :])
    trace = result["nomination_trace"]
    assert trace[0]["first_policy_window_SOC_MWh"] == [49, 49]
    assert trace[0]["first_policy_window_target_SOC_MWh"] == [24.5, 73.5]
    assert all(r["delivery_minute"] - r["decision_minute"] == 60 for r in trace)
    assert max(np.abs(result["energy_balance_error_MWh"]).ravel()) < 1e-10


def test_unreceived_future_boa_does_not_change_independent_channel_decisions():
    targets = np.broadcast_to([.3, .8], (1, 8, 2))

    def evaluate(power):
        units = [[instruction(power, power, left=120, right=150, received=90)], []]
        workload = site_workload(units, NOW, NOW + timedelta(minutes=240))
        terms = unit_planning_terms(receipt_plans(units, NOW, 240), ASSET)
        return simulate_units(workload, np.ones(8), targets, [0], 240, ASSET, terms)["nomination_trace"]

    a, b = evaluate(20), evaluate(-20)
    assert [r for r in a if r["decision_minute"] < 90] == [r for r in b if r["decision_minute"] < 90]
    assert next(r for r in a if r["decision_minute"] == 90) != next(r for r in b if r["decision_minute"] == 90)
