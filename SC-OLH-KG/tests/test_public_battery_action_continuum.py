import numpy as np

from problems.public_battery_action_continuum import (
    inverse_unit_targets, nomination_for_fraction, scalar_action_segments,
    segment_fraction_at_theta, theta_at_segment_fraction,
)


ASSET = {"unit_energy_MWh": [98., 98.], "power_MW": 98.,
         "charge_efficiency": .92, "discharge_efficiency": .92}
STATE = {"decision_SOC_MWh": [70., 20.], "pending_MW": [[0., 0.], [0., 0.]]}


def check_family(terms):
    segments = scalar_action_segments(STATE, 0, terms, ASSET)
    assert segments[0]["theta_left"] == 0 and segments[-1]["theta_right"] == 1
    assert len(segments) <= 14
    for segment in segments:
        left, right = np.asarray(segment["nomination_endpoints_MW"])
        for position in (.25, .5, .75):
            theta = segment["theta_left"] + position * (segment["theta_right"] - segment["theta_left"])
            fraction = segment_fraction_at_theta(theta, segment)
            actual = nomination_for_fraction(theta, STATE, 0, terms, ASSET)
            assert np.allclose(actual, left + fraction * (right - left), atol=1e-10, rtol=0)
            assert abs(theta_at_segment_fraction(fraction, segment) - theta) < 1e-12
    return segments


def test_analytic_family_covers_sign_clipping_and_projective_power_scaling():
    terms = {"forced": np.zeros((1, 3, 2)), "idle_hours": np.full((1, 3, 2), .5),
             "future_power_bound_MW": np.array([[98., 98.]])}
    segments = check_family(terms)
    assert any(s["shared_power_normalized"] for s in segments)
    breakpoints = [s["theta_left"] for s in segments]
    assert any(np.isclose(t, 70 / 98) for t in breakpoints)
    assert any(np.isclose(t, 20 / 98) for t in breakpoints)
    normalized = next(s for s in segments if s["shared_power_normalized"] and not np.allclose(*s["nomination_endpoints_MW"]))
    theta = (normalized["theta_left"] + normalized["theta_right"]) / 2
    assert not np.isclose(segment_fraction_at_theta(theta, normalized), .5)


def test_unit_with_no_idle_time_has_zero_nomination_across_the_family():
    terms = {"forced": np.zeros((1, 3, 2)), "idle_hours": np.array([[[.5, .5], [.5, .5], [0., .5]]]),
             "future_power_bound_MW": np.array([[98., 60.]])}
    segments = check_family(terms)
    assert all(np.asarray(s["nomination_endpoints_MW"])[:, 0].tolist() == [0., 0.] for s in segments)


def test_independent_inverse_targets_reproduce_mixed_action_at_shared_power_limit():
    terms = {"forced": np.zeros((1, 3, 2)), "idle_hours": np.full((1, 3, 2), .5),
             "future_power_bound_MW": np.array([[98., 98.]])}
    desired = np.array([22., -76.])
    raw, fractions = inverse_unit_targets(desired, STATE, 0, terms, ASSET)
    assert np.all(raw == fractions) and fractions[0] != fractions[1]
    assert np.allclose(nomination_for_fraction(fractions, STATE, 0, terms, ASSET), desired, atol=1e-10, rtol=0)


def test_inverse_domain_clipping_preserves_representable_power_plateau():
    terms = {"forced": np.array([[[25., 0.], [25., 0.], [0., 0.]]]),
             "idle_hours": np.full((1, 3, 2), .5), "future_power_bound_MW": np.array([[10., 98.]])}
    raw, fractions = inverse_unit_targets([10., 0.], STATE, 0, terms, ASSET)
    assert raw[0] > 1 and fractions[0] == 1
    assert np.allclose(nomination_for_fraction(fractions, STATE, 0, terms, ASSET), [10., 0.], atol=1e-10, rtol=0)
