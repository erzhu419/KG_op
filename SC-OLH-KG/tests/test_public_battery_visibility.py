from datetime import timedelta
import json

import numpy as np

from performance.inspect_public_battery_dispatch import Segment, instant
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_units import simulate_units, unit_planning_terms
from problems.public_battery_visibility import classify_failure, receipt_metadata

START = instant("2024-04-15T00:00:00Z")
ASSET = {"power_MW": 98., "energy_MWh": 196., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}


def segment(a, b, left, right, received):
    return Segment(START + timedelta(minutes=left), START + timedelta(minutes=right),
                   START + timedelta(minutes=received), left + 1, a, b)


def trace(units, target=.5, initial=.5):
    stop = START + timedelta(minutes=120)
    workload = site_workload(units, START, stop)
    plans = receipt_plans(units, START, 120)
    result = simulate_units(workload, np.ones(4), np.full((1, 4), target), [0], 120,
                            ASSET, unit_planning_terms(plans, ASSET), initial_fraction=initial)
    context = result["first_failure_contexts"][0]
    return context, classify_failure(context, 0, plans, workload,
                                     receipt_metadata(units, START, stop), ASSET)


def test_later_receipt_changes_safe_submitted_path_and_snapshot_stays_at_decision():
    context, report = trace([[segment(49, 49, 20, 50, 20)], []], target=0)
    assert context["nomination_decision_minute"] == 0
    assert context["decision_SOC_MWh"] == [49, 49]
    assert context["pending_MW"] == [[0, 0], [0, 0]]
    assert np.allclose(context["nomination_MW"], [49, 49])
    assert report["classification"] == "later_receipt_path_change"
    assert report["later_changed_receipts"] == [{"unit": 1, "acceptance_number": 21, "received_minute": 20}]
    assert max(report["known_prefix"]["unit_bound_violation_MWh"]) == 0
    json.dumps({"context": context, "report": report})


def test_known_interior_violation_is_checked_at_actual_failing_subinterval():
    context, report = trace([[segment(-50, 50, 60, 61, 0)], []], initial=(98 - .1) / 98)
    assert context["failure_minute"] == 60
    assert np.isclose(context["failure_piece_stop_fraction"], .5)
    assert report["classification"] == "known_new_delivery_violation"
    assert report["predicted_prefix_minutes"] == 60.5
    assert report["known_prefix"]["unit_bound_violation_MWh"][0] > .09
    assert report["later_changed_receipts"] == []


def test_known_pending_violation_remains_distinct_from_new_delivery_plan():
    # A later import rescues the pending execution, then a new export overrides
    # the recovery nomination. The earlier pending forecast was already unsafe.
    units = [[segment(-20, -20, 121, 150, 121), segment(20, 20, 180, 210, 180)], []]
    stop = START + timedelta(minutes=240)
    workload = site_workload(units, START, stop)
    plans = receipt_plans(units, START, 240)
    context = {"nomination_decision_minute": 120, "failure_minute": 187,
               "failure_piece_stop_fraction": 1., "decision_SOC_MWh": [49, 49],
               "pending_MW": [[98, 0], [98, 0]], "nomination_MW": [-98, 0]}
    report = classify_failure(context, 0, plans, workload, receipt_metadata(units, START, stop), ASSET)
    assert report["classification"] == "known_pending_hour_violation"
    assert report["known_pending_hour"]["unit_bound_violation_MWh"][0] > 57
    assert report["actual_prefix"]["unit_bound_violation_MWh"][0] > 0
    assert len(report["later_changed_receipts"]) == 2
