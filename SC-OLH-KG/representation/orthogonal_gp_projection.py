"""Project a learned Gaussian field into a declared finite function space."""

import numpy as np


def project_gaussian_field(design, mean, covariance, white_discrepancy):
    """Uniform public-policy OLS projection; retain the original white term."""
    design, mean, covariance = np.asarray(design), np.asarray(mean), np.asarray(covariance)
    count = len(design); identity = np.eye(count)
    operator, _, rank, _ = np.linalg.lstsq(design, identity, rcond=None)
    smooth = covariance-float(white_discrepancy)*identity
    coefficient_mean = operator@mean
    coefficient_covariance = operator@smooth@operator.T
    coefficient_covariance = .5*(coefficient_covariance+coefficient_covariance.T)
    projected_mean = design@coefficient_mean
    projected_covariance = design@coefficient_covariance@design.T+float(white_discrepancy)*identity
    complement = identity-design@operator
    discarded = complement@smooth@complement.T
    off_diagonal = coefficient_covariance-np.diag(np.diag(coefficient_covariance))
    return {'mean': projected_mean, 'covariance': projected_covariance,
        'coefficient_mean': coefficient_mean, 'coefficient_covariance': coefficient_covariance,
        'diagnostics': {'design_columns': int(design.shape[1]), 'design_rank': int(rank),
            'mean_projection_rms': float(np.sqrt(np.mean((projected_mean-mean)**2))),
            'smooth_covariance_trace': float(np.trace(smooth)),
            'discarded_smooth_covariance_trace_fraction': float(np.trace(discarded)/np.trace(smooth)),
            'coefficient_off_diagonal_squared_mass_fraction': float(np.sum(off_diagonal**2)/np.sum(coefficient_covariance**2)),
            'white_discrepancy_retained': float(white_discrepancy)}}
