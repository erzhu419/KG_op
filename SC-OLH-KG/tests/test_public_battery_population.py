"""Population stopping, request censoring and saved-cell accounting."""
from unittest.mock import patch

import numpy as np

from problems.public_battery_causal_service import simulate_service

COUNTS = {"mocked_service_executions": 0, "mocked_nomination_calls": 0,
          "mocked_declaration_calls": 0, "mocked_minute_physics_calls": 0}
ASSET = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
         "charge_efficiency": .92, "discharge_efficiency": .92}


def execute(stop=True, known_failure_minute=None, planning_failure_minute=None):
    def declaration(*args, **kwargs):
        COUNTS["mocked_declaration_calls"] += 1
        return {"envelope_path_evaluations": 0, "available_capacity_MW": [0., 0.],
                "known_commitment_failure": False, "anchor_MW": np.zeros(2),
                "guaranteed_peak_MW": np.zeros(2), "reason": "test"}

    def nomination(soc, current, following, fractions, plan, minute, *args, **kwargs):
        COUNTS["mocked_nomination_calls"] += 1
        return np.zeros(2), {"known_reference_failure": minute == known_failure_minute,
                            "planning_failure": minute == planning_failure_minute, "nomination_reason": "test",
                            "reference_safety_path_evaluations": 0, "scalar_stock_projection_evaluations": 0}

    def physics(first, last, soc, *args):
        COUNTS["mocked_minute_physics_calls"] += 1
        return {"final_SOC_MWh": soc, "minimum_SOC_MWh": soc, "maximum_SOC_MWh": soc,
                "unit_bound_violation_MWh": [0., 0.], "power_bound_violation_MW": 0.}

    marks = [{"minute": minute, "direction": direction} for minute, direction in
             zip(range(0, 240, 60), [None, None, "export", "import"])]
    with patch("problems.public_battery_causal_service.declare_absolute", side_effect=declaration), \
         patch("problems.public_battery_causal_service.nominate_announced", side_effect=nomination), \
         patch("problems.public_battery_causal_service.trajectory", side_effect=physics):
        result = simulate_service([49, 49], [[0, 0], [0, 0]], [.35, .75], marks,
                                 "conservative_admission", 240, [49, 49], ASSET,
                                 absolute_request_peaks_MW={"export": [49, 49], "import": [-49, -49]},
                                 announcement_ledger=[], stop_on_service_failure=stop)
    COUNTS["mocked_service_executions"] += 1
    return result


def test_stop_after_observed_rejection_preserves_denominator_and_actual_prefix():
    result, hours, nominations = execute()
    assert result["completed_minutes"] == 120 and result["minute_physics_evaluations"] == 120
    assert result["planned_requests"] == 2 and result["observed_requests"] == 1
    assert result["unobserved_requests"] == 1 and result["unfulfilled_or_unassessed_requests"] == 2
    assert result["first_dispatch_failure"] == {"minute": 120, "reasons": ["request_shortfall"]}
    assert result["stopped_on_service_failure"] and not result["physical_complete"]
    assert result["physical_outcome_scope"] == "executed_prefix"
    assert hours[-1]["delivery_prefix_minutes"] == 0
    assert hours[-1]["unmet_increase_MWh"] == [0., 0.]  # no delivery is fabricated
    assert [n["minute"] for n in nominations] == [0, 30, 60, 90]


def test_nomination_failure_stops_before_next_physical_minute():
    result, hours, nominations = execute(known_failure_minute=30)
    assert result["completed_minutes"] == 30 and result["minute_physics_evaluations"] == 30
    assert result["first_dispatch_failure"] == {"minute": 30, "reasons": ["known_commitment_failure_at_nomination"]}
    assert result["unobserved_requests"] == 2 and hours[-1]["delivery_prefix_minutes"] == 30
    assert nominations[-1]["minute"] == 30


def test_existing_full_execution_still_runs_after_service_failure():
    result, hours, _ = execute(stop=False)
    assert result["completed_minutes"] == 240 and result["physical_complete"]
    assert not result["window_success"] and result["planned_requests"] == 2
    assert result["unobserved_requests"] == 0 and hours[-1]["delivery_prefix_minutes"] == 60
    assert "stopped_on_service_failure" not in result


def test_safe_zero_planning_failure_does_not_stop_service_classification():
    result, _, nominations = execute(planning_failure_minute=30)
    assert nominations[1]["planning_failure"] and not nominations[1]["known_reference_failure"]
    assert result["completed_minutes"] == 120
    assert result["first_dispatch_failure"] == {"minute": 120, "reasons": ["request_shortfall"]}
