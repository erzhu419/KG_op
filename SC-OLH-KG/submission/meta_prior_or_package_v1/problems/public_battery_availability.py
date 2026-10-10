"""Zero-baseline, one-shot availability contract for a fixed 32-minute pulse.

Each unit offers one signed direction before seeing the request. The independent
service obligation is fixed by the caller. Proportional site rationing is a
simulation rule; neither it nor these fixtures identifies counterfactual BM
dispatch. Nonzero baselines and pending commitments are outside this component.
"""
import numpy as np

from problems.public_battery_dispatch import positive_integral
from problems.public_battery_visibility import trajectory

PULSE_NODES = np.r_[0., np.ones(31), 0.]
PULSE_ENERGY_HOURS = 31 / 60  # thirty-minute hold plus two half-triangle ramps


def pulse_power(level_MW):
    """Whole-minute endpoints: ramp 0-1, hold 1-31, return 31-32."""
    nodes = PULSE_NODES[:, None] * np.asarray(level_MW, float)
    return nodes[:-1], nodes[1:]


def declare_pulse(initial_SOC_MWh, offered_MW, asset):
    """Fix jointly sustainable signed availability from observable own stock."""
    initial = np.asarray(initial_SOC_MWh, float)
    offered = np.asarray(offered_MW, float)
    capacity = np.asarray(asset["unit_energy_MWh"])
    export = np.minimum(initial * asset["discharge_efficiency"] / PULSE_ENERGY_HOURS,
                        asset["power_MW"])
    imported = np.minimum((capacity - initial) / (asset["charge_efficiency"] * PULSE_ENERGY_HOURS),
                          asset["power_MW"])
    bound = np.where(offered >= 0, export, imported)
    guaranteed = np.sign(offered) * np.minimum(np.abs(offered), bound)
    gross = np.sum(np.abs(guaranteed))
    if gross > asset["power_MW"]:
        guaranteed *= asset["power_MW"] / gross
    return {"initial_SOC_MWh": initial.copy(), "offered_MW": offered.copy(),
            "guaranteed_MW": guaranteed, "maximum_export_MW": export,
            "maximum_import_MW": imported}


def evaluate_pulse(declaration, proposal_MW, obligation_MW, asset, tolerance=1e-8):
    """Admit within fixed availability and retain both kinds of service deficit."""
    proposal = np.asarray(proposal_MW, float)
    obligation = np.asarray(obligation_MW, float)
    guaranteed = declaration["guaranteed_MW"]
    same_direction = proposal * guaranteed >= 0
    admitted = np.where(same_direction,
                        np.sign(proposal) * np.minimum(np.abs(proposal), np.abs(guaranteed)), 0.)
    available_for_obligation = np.maximum(np.sign(obligation) * guaranteed, 0.)
    availability_shortfall = np.maximum(np.abs(obligation) - available_for_obligation, 0.)
    requested_first, requested_last = pulse_power(proposal)
    first, last = pulse_power(admitted)
    export = positive_integral(first, last, 1 / 60)
    imported = positive_integral(-first, -last, 1 / 60)
    requested_export = positive_integral(requested_first, requested_last, 1 / 60)
    requested_import = positive_integral(-requested_first, -requested_last, 1 / 60)
    unmet_export = np.maximum(np.sum(requested_export - export, axis=0), 0.)
    unmet_import = np.maximum(np.sum(requested_import - imported, axis=0), 0.)
    initial = declaration["initial_SOC_MWh"]
    physics = trajectory(first, last, initial, 32, asset)
    physical_feasible = (max(physics["unit_bound_violation_MWh"]) <= tolerance
                         and physics["power_bound_violation_MW"] <= tolerance)
    service_success = (physical_feasible and np.max(availability_shortfall) <= tolerance
                       and max(np.max(unmet_export), np.max(unmet_import)) <= tolerance)
    delta = asset["charge_efficiency"] * imported - export / asset["discharge_efficiency"]
    stocks = np.vstack((initial, initial + np.cumsum(delta, axis=0)))
    return {"obligation_MW": obligation, "proposal_MW": proposal, "admitted_MW": admitted,
            "availability_shortfall_MW": availability_shortfall,
            "request_shortfall_MW": np.abs(proposal - admitted),
            "gross_export_MWh": np.sum(export, axis=0), "gross_import_MWh": np.sum(imported, axis=0),
            "unmet_export_MWh": unmet_export, "unmet_import_MWh": unmet_import,
            "total_unmet_energy_MWh": float(np.sum(unmet_export + unmet_import)),
            "physical_feasible": bool(physical_feasible), "service_success": bool(service_success),
            "physics": physics, "SOC_trace_MWh": stocks,
            "power_first_MW": first, "power_last_MW": last}
