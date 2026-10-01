"""The observed July boundary uses different arithmetic at an execution hour."""
import numpy as np

from problems.public_battery_visibility import trajectory


def test_hour_break_matches_actual_hour_origins_at_observed_tolerance_boundary():
    asset = {"power_MW": 98., "unit_energy_MWh": [98., 98.],
             "charge_efficiency": .92, "discharge_efficiency": .92}
    initial = np.array([19.599999999999852, 39.199999999999946])
    first = np.repeat([[0., 0.], [25.244800000000005, -38.34782608695678], [0., 0.],
                       [10.819200018399764, 25.065526671950458]], 30, axis=0)
    original = trajectory(first, first, initial, 120, asset)
    a = trajectory(first[:60], first[:60], initial, 60, asset)
    b = trajectory(first[60:], first[60:], np.array(a["final_SOC_MWh"]), 60, asset)
    aligned = trajectory(first, first, initial, 120, asset, hour_break_minutes=[60])
    assert original["unit_bound_violation_MWh"][0] == 9.999997274690031e-09
    assert original["unit_bound_violation_MWh"][0] <= 1e-8
    assert aligned["unit_bound_violation_MWh"][0] > 1e-8
    assert aligned["final_SOC_MWh"] == b["final_SOC_MWh"]
    assert aligned["minimum_SOC_MWh"] == np.minimum(a["minimum_SOC_MWh"], b["minimum_SOC_MWh"]).tolist()
    assert aligned["maximum_SOC_MWh"] == np.maximum(a["maximum_SOC_MWh"], b["maximum_SOC_MWh"]).tolist()
