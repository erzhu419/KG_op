"""Real BOA revisions must not duplicate physical power or reveal future receipts."""
import json
from pathlib import Path

import pytest

from performance.inspect_public_battery_dispatch import decode, energy_pieces, instant, known_segment, reconstruct, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_artifacts/public_battery_application_spec_20260930"


def recorded_segments():
    rows = json.loads((DATA / "pillswood_acceptances_20240414_15.json").read_text())["payload"]["data"]
    return decode(rows, "E_PILLB-1")


def test_recorded_revision_is_available_only_after_receipt():
    segments = recorded_segments()
    target = instant("2024-04-15T23:50:30Z")
    # These are three real, successive revisions of the same future minute.
    for observed, number, power in [("23:30", 6788, 17.5), ("23:36", 6790, 19), ("23:44", 6791, 36)]:
        at = instant(f"2024-04-15T{observed}:00Z")
        selected = known_segment(segments, target, at)
        prefix = [s for s in segments if s.received <= at]
        assert selected == known_segment(prefix, target, at)
        assert selected.number == number
        assert selected.power(target) == pytest.approx(power)


def test_recorded_overlap_does_not_invent_a_hardware_violation():
    segments = recorded_segments()
    ledger = reconstruct(segments, instant("2024-04-15T00:00:00Z"), instant("2024-04-16T00:00:00Z"))
    summary = summarize(ledger, {"power_MW": 49, "energy_MWh": 98,
                                 "charge_efficiency": .92, "discharge_efficiency": .92})
    assert summary["power_within_reference_limit"]
    assert summary["peak_requested_MW"] == 49
    assert summary["naive_sum_peak_MW"] > 49
    assert summary["overlapping_BOA_hours"] > 0
    gaps = [r for r in ledger if not r["active_BOA"]]
    assert gaps
    assert all(r["from_MW"] is None and r["to_MW"] is None for r in gaps)


def test_opposed_energy_and_mid_segment_inventory_extrema_are_preserved():
    assert energy_pieces(10, -10, 1) == [(2.5, 0), (0, 2.5)]
    # Net zero energy still needs 2.5 MWh of initial inventory during the ramp.
    row = {"from_utc": "2024-04-15T00:00:00Z", "to_utc": "2024-04-15T01:00:00Z",
           "active_BOA": True, "from_MW": 10, "to_MW": -10, "overlapping_segments": 1,
           "naive_sum_from_MW": 10, "naive_sum_to_MW": -10, "export_MWh": 2.5, "import_MWh": 2.5}
    summary = summarize([row], {"power_MW": 10, "energy_MWh": 10,
                               "charge_efficiency": 1, "discharge_efficiency": 1})
    assert summary["idle_gap_envelope"]["initial_inventory_lower_MWh"] == 2.5
    assert summary["idle_gap_envelope"]["terminal_inventory_change_MWh"] == 0
