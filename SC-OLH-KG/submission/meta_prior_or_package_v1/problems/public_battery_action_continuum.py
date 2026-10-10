"""Shared-target action segments and constructive independent unit targets."""
import numpy as np

from problems.public_battery_dispatch import inventory_rate
from problems.public_battery_units import nominate_units


def nomination_for_fraction(fraction, snapshot, decision, terms, asset):
    soc = np.array(snapshot["decision_SOC_MWh"])[None, None, :]
    current, following = np.array(snapshot["pending_MW"])[:, None, None, :]
    return nominate_units(soc, current, following,
                          fraction * np.asarray(asset["unit_energy_MWh"]),
                          np.array([decision]), terms, asset)[0, 0]


def inverse_unit_targets(command, snapshot, decision, terms, asset):
    current, following = np.asarray(snapshot["pending_MW"])
    idle = terms["idle_hours"][decision]
    projected = (np.asarray(snapshot["decision_SOC_MWh"]) + terms["forced"][decision].sum(axis=0)
                 + inventory_rate(current, asset) * idle[0] + inventory_rate(following, asset) * idle[1])
    target = projected + inventory_rate(np.asarray(command), asset) * idle[2]
    raw = target / np.asarray(asset["unit_energy_MWh"])
    return raw, np.clip(raw, 0, 1)


def scalar_action_segments(snapshot, decision, terms, asset):
    """Cover the complete scalar family, without a target grid.

    Between clipping/sign breakpoints the raw pair is affine. Above the site
    limit it is divided by one positive affine total; this projective map also
    covers the full line segment between the endpoint nominations.
    """
    capacity = np.asarray(asset["unit_energy_MWh"])
    idle = terms["idle_hours"][decision]
    current, following = np.asarray(snapshot["pending_MW"])
    projected = (np.asarray(snapshot["decision_SOC_MWh"]) + terms["forced"][decision].sum(axis=0)
                 + inventory_rate(current, asset) * idle[0] + inventory_rate(following, asset) * idle[1])
    bound = terms["future_power_bound_MW"][decision]
    points = {0., 1.}
    for unit in (0, 1):
        if idle[2, unit] > 0:
            for stock in (projected[unit],
                          projected[unit] - bound[unit] * idle[2, unit] / asset["discharge_efficiency"],
                          projected[unit] + bound[unit] * idle[2, unit] * asset["charge_efficiency"]):
                fraction = float(stock / capacity[unit])
                if 0 < fraction < 1:
                    points.add(fraction)

    def raw_total(fraction):
        difference = projected - fraction * capacity
        power = np.where(difference >= 0, difference * asset["discharge_efficiency"],
                         difference / asset["charge_efficiency"])
        power = np.divide(power, idle[2], out=np.zeros_like(power), where=idle[2] > 0)
        return float(np.sum(np.abs(np.clip(power, -bound, bound))))

    initial = sorted(points)
    for left, right in zip(initial[:-1], initial[1:]):
        first, last = raw_total(left), raw_total(right)
        if (first - asset["power_MW"]) * (last - asset["power_MW"]) < 0:
            points.add(left + (right - left) * (asset["power_MW"] - first) / (last - first))
    points = sorted(points)
    return [{"theta_left": left, "theta_right": right,
             "nomination_endpoints_MW": [nomination_for_fraction(t, snapshot, decision, terms, asset).tolist() for t in (left, right)],
             "raw_absolute_totals_MW": [raw_total(left), raw_total(right)],
             "shared_power_normalized": raw_total((left + right) / 2) > asset["power_MW"]}
            for left, right in zip(points[:-1], points[1:])]


def segment_fraction_at_theta(theta, segment):
    relative = (theta - segment["theta_left"]) / (segment["theta_right"] - segment["theta_left"])
    if segment["shared_power_normalized"]:
        first, last = segment["raw_absolute_totals_MW"]
        return relative * last / ((1 - relative) * first + relative * last)
    return relative


def theta_at_segment_fraction(fraction, segment):
    relative = fraction
    if segment["shared_power_normalized"]:
        first, last = segment["raw_absolute_totals_MW"]
        relative = fraction * first / ((1 - fraction) * last + fraction * first)
    return segment["theta_left"] + relative * (segment["theta_right"] - segment["theta_left"])
