"""Source-fitted missing public mean directions for the native risk field.

The six-column spectral g map lacks the public linear mean subspace.
Residualizing mean4 against the frozen map avoids adding a second copy of its
represented source directions. A Gaussian source coefficient adds both a mean
correction and a cross-policy covariance before any target data.
"""

import numpy as np

from performance.connect_complete_historical_reuse import design_for
from performance.connect_historical_learned_prior_kg import read


def public_mean_residual(coordinates, calibration):
    mean, risk = (np.asarray(coordinates[key]) for key in ('mean', 'risk'))
    law = dict(layout=calibration['layout'], selector='spectral',
               support='head_specific', head='g')
    design = design_for(law, (mean, risk), read(calibration['maps'])['rows'],
                        read(calibration['selection_plan']))
    return mean - design @ np.asarray(calibration['projection_coefficients'])


def add_risk_directions(model, residual, coefficient_mean, coefficient_covariance, prefix):
    """Add independent Gaussian source coefficients to specified native heads."""
    assert model.paid_episodes == 0 and not model.history
    residual = np.asarray(residual)
    coefficient_mean = np.asarray(coefficient_mean)
    coefficient_covariance = np.asarray(coefficient_covariance)
    shift = residual @ coefficient_mean
    covariance = residual @ coefficient_covariance @ residual.T
    affected = []
    for name, (_, risk) in zip(model.names, model.components):
        if name.startswith(prefix):
            risk.means[:, ::2] += shift
            risk.covariances[:, ::2, ::2] += covariance
            affected.append(name)
    assert affected, 'The correction must match the omitted-direction source identities'
    return {'affected_identities': affected, 'coefficient_mean': coefficient_mean.tolist(),
            'coefficient_covariance': coefficient_covariance.tolist(),
            'residual': residual.tolist(), 'mean_shift': shift.tolist(),
            'paid_episodes': model.paid_episodes,
            'native_mean_and_cross_policy_covariance_updated': True}


def apply_public_mean_correction(model, coordinates, calibration):
    residual = public_mean_residual(coordinates, calibration)
    return add_risk_directions(model, residual, calibration['beta_mean'],
                               calibration['beta_covariance'],
                               calibration['affected_identity_prefix'])
