"""Check scientific denominators and the separation of early and late cues."""
import pytest

from performance.inspect_public_battery_information_coverage import audit


def mark(minute, direction, received=None):
    return {"minute": minute, "direction": direction,
            "source": None if direction is None else {"received_minute": received}}


def test_bootstrap_pair_is_retained_but_excluded_from_deadline_denominators():
    marks = [mark(0, "export", -5), mark(60, "import", 35), mark(120, "export", 95)]
    cues = [mark(30, "export", 25), mark(90, "import", 75)]
    summary, rows = audit(marks, cues)
    assert summary["total"]["pairs"] == 2
    assert summary["total"]["bootstrap_deadlines"] == 1
    assert summary["total"]["auditable_deadlines"] == 1
    assert rows[0]["deadline_cue"] is None
    assert rows[0]["final_first_source_already_received"] is None
    assert rows[1]["deadline_matches_first"] is False
    assert rows[1]["deadline_matches_second"] is True
    assert rows[1]["final_first_source_already_received"] is False


def test_absent_early_cue_is_not_replaced_by_a_correct_late_forecast():
    marks = [mark(960, "import", 935), mark(1020, "import", 1005)]
    summary, rows = audit(marks, [mark(930, None), mark(990, "import", 935)])
    assert summary["total"]["deadline_cue_absent"] == 1
    assert summary["total"]["deadline_cue_matches_second"] == 0
    assert summary["total"]["late_cue_matches_second"] == 1
    assert rows[0]["tail_PN_delivery_minute"] == 990
    assert rows[0]["late_PN_delivery_minute"] == 1050
    assert rows[0]["final_first_source_already_received"] is False


def test_a_missing_cached_clock_is_not_counted_as_an_absent_signal():
    with pytest.raises(KeyError, match="930"):
        audit([mark(960, "import", 935), mark(1020, "import", 1005)],
              [mark(990, "import", 935)])
