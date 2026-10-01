"""Causal release, binding requests and inactive/warmup service denominators."""
from copy import deepcopy

import pytest

from problems.public_battery_announcements import (
    announce_requests, delivery_clock_roster, released_announcements,
)

CONTRACT = {"announcement_lead_minutes": 120,
            "absolute_peak_MW": {"export": [49, 49], "import": [-49, -49]},
            "total_execution_horizon_minutes": 300, "announcement_clocks": {"step_minutes": 60}}
MARKS = [{"minute": 0, "direction": None, "source": None},
         {"minute": 60, "direction": "export", "source": {"received_minute": 35., "last_MW": 3.}},
         {"minute": 120, "direction": "import", "source": {"received_minute": 95., "last_MW": -2.}}]


def test_translation_keeps_source_direction_and_fixed_peak_without_mutating_inputs():
    original = deepcopy(MARKS)
    ledger = announce_requests(MARKS, CONTRACT)
    assert MARKS == original
    assert [r["release_minute"] for r in ledger] == [0, 60, 120]
    assert [r["delivery_minute"] for r in ledger] == [120, 180, 240]
    assert [r["source"] for r in ledger] == [m["source"] for m in MARKS]
    assert [r["peak_MW"] for r in ledger] == [None, [49, 49], [-49, -49]]


def test_future_changes_do_not_affect_visible_prefix_and_release_boundary_is_inclusive():
    ledger = announce_requests(MARKS, CONTRACT)
    changed = deepcopy(ledger)
    changed[1].update(direction="import", peak_MW=[-49, -49])
    changed[2]["source"]["last_MW"] = -25.
    assert released_announcements(ledger, 30) == released_announcements(changed, 30) == [ledger[0]]
    assert released_announcements(ledger, 60) == ledger[:2]


def test_controller_planning_edits_do_not_rewrite_a_binding_announcement():
    ledger = announce_requests(MARKS, CONTRACT)
    original = deepcopy(ledger)
    visible = released_announcements(ledger, 60)
    visible[1]["peak_MW"][0] = 0
    visible[1]["source"]["last_MW"] = -49.
    assert ledger == original
    assert released_announcements(ledger, 60) == original[:2]


def test_source_received_after_release_is_rejected_even_if_it_precedes_delivery():
    future = deepcopy(MARKS)
    future[1]["source"]["received_minute"] = 65.
    with pytest.raises(ValueError, match="not received by announcement release"):
        announce_requests(future, CONTRACT)


def test_warmup_and_inactive_slots_are_separate_from_active_service_obligations():
    delivery = delivery_clock_roster(announce_requests(MARKS, CONTRACT), CONTRACT)
    assert [r["delivery_minute"] for r in delivery] == [0, 60, 120, 180, 240]
    assert [r["bootstrap"] for r in delivery] == [True, True, False, False, False]
    assert all(r["release_minute"] is None and r["source"] is None for r in delivery[:2])
    assert delivery[2]["release_minute"] == 0 and delivery[2]["direction"] is None
    assert sum(r["direction"] is not None for r in delivery) == 2
