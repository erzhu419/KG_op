"""Binding future service requests generated from already received public signs."""
from copy import deepcopy


def announce_requests(marks, contract):
    """Publish each original mark now, for its fixed later delivery clock."""
    ledger = []
    for mark in marks:
        release, direction = mark["minute"], mark["direction"]
        if direction is not None and mark["source"]["received_minute"] > release:
            raise ValueError("selected public source is not received by announcement release")
        ledger.append({"release_minute": release,
                       "delivery_minute": release + contract["announcement_lead_minutes"],
                       "direction": direction,
                       "peak_MW": None if direction is None else deepcopy(contract["absolute_peak_MW"][direction]),
                       "source": deepcopy(mark["source"])})
    return ledger


def released_announcements(ledger, decision_minute):
    """Controller input owns its copy; planning cannot change binding requests."""
    return deepcopy([row for row in ledger if row["release_minute"] <= decision_minute])


def delivery_clock_roster(ledger, contract):
    """Full dispatcher schedule, with two explicit startup slots in this task."""
    by_delivery = {row["delivery_minute"]: row for row in ledger}
    roster = []
    for minute in range(0, contract["total_execution_horizon_minutes"],
                        contract["announcement_clocks"]["step_minutes"]):
        bootstrap = minute < contract["announcement_lead_minutes"]
        row = {"release_minute": None, "delivery_minute": minute, "direction": None,
               "peak_MW": None, "source": None} if bootstrap else deepcopy(by_delivery[minute])
        roster.append({**row, "bootstrap": bootstrap})
    return roster
