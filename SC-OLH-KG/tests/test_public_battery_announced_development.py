"""Window timestamps, full final-window support and exact retained pilot reuse."""
from copy import deepcopy

import pytest

from performance.inspect_public_battery_announced_development import (
    read, sample_marks, validate_pilot_workload, window_workload,
)

COUNTS = {"test_workload_constructions": 0, "test_cached_sample_mark_constructions": 0,
          "test_pilot_workload_comparisons": 0}
CONTRACT = {"announcement_lead_minutes": 120,
            "absolute_peak_MW": {"export": [49, 49], "import": [-49, -49]},
            "total_execution_horizon_minutes": 300,
            "announcement_clocks": {"stop_exclusive_minute": 180, "step_minutes": 60}}


def workload(marks, start, contract=CONTRACT):
    COUNTS["test_workload_constructions"] += 1
    return window_workload(marks, start, contract)


def test_all_source_times_translate_without_discarding_received_prefix_or_mutating_sample():
    marks = [{"minute": 0, "direction": None, "source": None},
             {"minute": 60, "direction": "export", "source": {"received_minute": 35., "start_minute": 40.,
              "stop_minute": 72., "unit": 2, "acceptance_number": 8, "last_MW": 3.}},
             {"minute": 120, "direction": None, "source": None},
             {"minute": 180, "direction": "import", "source": {"received_minute": 179., "start_minute": 180.,
              "stop_minute": 220., "unit": 1, "acceptance_number": 9, "last_MW": -2.}}]
    original = deepcopy(marks)
    announcements, deliveries, metadata = workload(marks, 60)
    assert marks == original
    assert [row["release_minute"] for row in announcements] == [0, 60, 120]
    assert [row["delivery_minute"] for row in announcements] == [120, 180, 240]
    assert announcements[0]["source"] == {**original[1]["source"], "received_minute": -25.,
                                         "start_minute": -20., "stop_minute": 12.}
    assert announcements[-1]["source"]["received_minute"] == 119.
    assert metadata["planned_requests"] == 2
    assert [row["direction"] for row in deliveries] == [None, None, "export", None, "import"]


def test_last_registered_window_uses_only_source_clocks_inside_ten_days():
    protocol = read("performance/manifests/public_battery_announced_development_screen_v1_20261001.json")
    contract = read(protocol["task_protocol"])["contract"]
    marks = [{"minute": minute, "direction": "export", "source": {"received_minute": minute - 1,
             "start_minute": minute, "stop_minute": minute + 32}}
             for minute in range(0, 14400, 60)]
    announcements, deliveries, metadata = workload(marks, protocol["roster"]["source_window_start_minutes"][-1], contract)
    assert len(announcements) == 168 and len(deliveries) == 170
    assert metadata["source_window_stop_exclusive_minute"] == 14400
    assert announcements[-1]["release_minute"] == 10020 and deliveries[-1]["delivery_minute"] == 10140
    assert metadata["planned_requests"] == 168


def test_april_start_zero_reproduces_the_entire_saved_pilot_workload():
    protocol = read("performance/manifests/public_battery_announced_development_screen_v1_20261001.json")
    base = read(protocol["cached_data_protocol"])
    pilot = read(protocol["basis_protocol"])
    contract = read(protocol["task_protocol"])["contract"]
    COUNTS["test_cached_sample_mark_constructions"] += 1
    marks = sample_marks(base, base["development"][0])
    announcements, deliveries, metadata = workload(marks, 0, contract)
    COUNTS["test_pilot_workload_comparisons"] += 1
    validate_pilot_workload(announcements, deliveries, pilot)
    assert metadata["planned_requests"] == 95 and metadata["direction_counts"] == {"export": 50, "import": 45}


def test_changed_workload_cannot_reuse_pilot_outcomes():
    protocol = read("performance/manifests/public_battery_announced_development_screen_v1_20261001.json")
    pilot = read(protocol["basis_protocol"])
    announcements = read(pilot["task_outputs"] + "/announcements.json")
    deliveries = read(pilot["task_outputs"] + "/delivery_clocks.json")
    changed = deepcopy(announcements)
    changed[0]["direction"] = "export" if changed[0]["direction"] != "export" else "import"
    COUNTS["test_pilot_workload_comparisons"] += 1
    with pytest.raises(ValueError, match="reuse is blocked"):
        validate_pilot_workload(changed, deliveries, pilot)


def test_truncated_source_clock_population_blocks_execution():
    with pytest.raises(ValueError, match="every frozen announcement clock"):
        workload([{"minute": 0, "direction": None, "source": None}], 0)
