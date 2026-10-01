from datetime import timedelta

import numpy as np

from performance.inspect_public_battery_dispatch import Segment, instant
from performance.inspect_public_battery_unit_targets_visibility import first_failure_piece_matches
from problems.public_battery_site import receipt_plans, site_workload
from problems.public_battery_units import simulate_units, unit_planning_terms

START = instant("2024-04-15T00:00:00Z")
ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.], "charge_efficiency": .92, "discharge_efficiency": .92}


def retained_context(first, last, left, right, received, target=.5, initial=.5):
    s = Segment(START + timedelta(minutes=left), START + timedelta(minutes=right),
                START + timedelta(minutes=received), 1, first, last)
    units = [[s], []]
    workload = site_workload(units, START, START + timedelta(minutes=120))
    terms = unit_planning_terms(receipt_plans(units, START, 120), ASSET)
    result = simulate_units(workload, np.ones(4), np.full((1, 4), target), [0], 120,
                            ASSET, terms, initial_fraction=initial)
    return result["first_failure_contexts"][0], workload


def test_first_piece_check_rejects_a_stale_failure_minute():
    c, workload = retained_context(49, 49, 20, 50, 20, target=0)
    actual = first_failure_piece_matches(c, 0, workload, ASSET, 1e-8)
    assert actual["safe_before_failing_piece"] and actual["piece_stop_is_reconstruction_boundary"]
    assert actual["inventory_failure"] == [True, False]
    changed = {**c, "failure_minute": c["failure_minute"] + 1}
    assert not first_failure_piece_matches(changed, 0, workload, ASSET, 1e-8)["safe_before_failing_piece"]


def test_first_piece_check_preserves_interior_zero_crossing_failure():
    c, workload = retained_context(-50, 50, 60, 61, 0, initial=(98 - .1) / 98)
    assert c["failure_piece_stop_fraction"] == .5
    actual = first_failure_piece_matches(c, 0, workload, ASSET, 1e-8)
    assert actual["safe_before_failing_piece"] and actual["piece_stop_is_reconstruction_boundary"]
    changed = {**c, "failure_piece_stop_fraction": 1.}
    assert not first_failure_piece_matches(changed, 0, workload, ASSET, 1e-8)["safe_before_failing_piece"]
