"""Batched implementation of the existing paid Gaussian moment filter."""

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, logsumexp

from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.physical_risk_prior import log_sample_variance_density


def local_raw_moments(means, covariances, label, eta_label=None, replicates=1):
    if replicates < 1 or int(replicates) != replicates:
        raise ValueError('replicates must be a positive integer')
    if (replicates == 1) != (eta_label is None):
        raise ValueError('one episode has no eta label; batches require one')
    gm, em = np.asarray(means).T; covariances = np.asarray(covariances)
    gv, ev, cross = covariances[:, 0, 0], covariances[:, 1, 1], covariances[:, 0, 1]
    sd = np.sqrt(ev); conditional = np.maximum(gv-cross**2/ev, 0.); loading = cross/sd
    sample_variance = None
    if replicates > 1:
        k = (replicates-1)/2
        sample_variance = np.exp(np.asarray(eta_label)+digamma(k)-np.log(k))

    def at(t):
        eta = em+sd*t; observation = np.exp(eta)/replicates; prior_mean = gm+loading*t
        total = conditional+observation; residual = label-prior_mean
        density = -.5*(np.log(2*np.pi)+t*t+np.log(2*np.pi*total)+residual**2/total)
        if sample_variance is not None:
            density += log_sample_variance_density(eta, sample_variance, replicates)
        posterior_mean = prior_mean+conditional/total*residual
        posterior_variance = conditional*observation/total
        return density, posterior_mean, posterior_variance

    grid = np.linspace(-12., 12., 9)
    log_grid = np.stack([at(t)[0] for t in grid]); shifts = log_grid.max(axis=0)
    points = np.unique(np.r_[0., grid[np.argmax(log_grid, axis=0)]])
    # The actual mixed-source fantasies can put a component's peak between
    # coarse grid points. Center each integrand near its own peak so the
    # shared absolute error is meaningful for every component normalizer.
    center = grid[np.argmax(log_grid, axis=0)]
    left, right = np.maximum(center-3., -12.), np.minimum(center+3., 12.)
    golden = (np.sqrt(5.)-1.)/2
    for _ in range(32):
        low = right-golden*(right-left); high = left+golden*(right-left)
        keep_left = at(low)[0] >= at(high)[0]
        left, right = np.where(keep_left, left, low), np.where(keep_left, high, right)
    shifts = np.maximum(shifts, at((left+right)/2)[0])

    def integrand(t):
        density, posterior_mean, posterior_variance = at(t)
        dg, de = posterior_mean-gm, sd*t
        return np.exp(density-shifts)[:, None]*np.column_stack(
            [np.ones(len(gm)), dg, posterior_variance+dg**2, de, de**2, dg*de])

    integral, error, info = quad_vec(integrand, -12., 12., points=points, norm='max', full_output=True,
        epsabs=1e-12, epsrel=1e-12, limit=128, cache_size=16*1024**2, workers=1)
    if not info.success or not np.all(integral[:, 0] > 0):
        raise RuntimeError(f'Batched raw moment integration failed: {info.message}')
    statistics = integral/integral[:, :1]; displacement = statistics[:, [1, 3]]
    covariance = np.empty((len(gm), 2, 2))
    covariance[:, 0, 0] = statistics[:, 2]; covariance[:, 1, 1] = statistics[:, 4]
    covariance[:, 0, 1] = covariance[:, 1, 0] = statistics[:, 5]
    covariance -= displacement[:, :, None]*displacement[:, None, :]
    relative_error = float(error/integral[:, 0].min())
    if relative_error > 1e-8:
        raise RuntimeError(f'Batched raw moment error exceeds frozen tolerance: {relative_error}')
    return {'mean': np.asarray(means)+displacement, 'covariance': covariance,
        'log_evidence': shifts+np.log(integral[:, 0]), 'integration_relative_error': relative_error,
        'integrand_evaluations': int(info.neval), 'largest_integrand_array_bytes': int(integral.nbytes)}


def raw_fantasy_moments(field,candidate,labels,eta_labels=None,replicates=1):
    """Reuse the paid batch integrand for four observed statistics at a time."""
    ix=[2*candidate,2*candidate+1]
    means=field.means[:,ix];covariance=field.covariances[:,ix][:,:,ix]
    gain=np.linalg.solve(covariance,field.covariances[:,ix,:]).transpose(0,2,1)
    count=len(means);results=[]
    for start in range(0,len(labels),4):
        block=np.asarray(labels[start:start+4]);n=len(block)
        eta = None if eta_labels is None else np.repeat(np.asarray(eta_labels[start:start+4]),count)
        batch=local_raw_moments(np.tile(means,(n,1)),np.tile(covariance,(n,1,1)),
            np.repeat(block,count),eta,replicates)
        for j in range(n):
            part=slice(j*count,(j+1)*count)
            results.append({k:batch[k][part] for k in ('mean','covariance','log_evidence')})
            results[-1]['integration_relative_error']=batch['integration_relative_error']
    return results,gain


def apply_raw_moments(field,candidate,result,gain=None,*,replicates=1):
    """Complete the original field from already integrated local moments."""
    indices = [2*candidate, 2*candidate+1]
    local_mean = field.means[:, indices].copy()
    local_covariance = field.covariances[:, indices][:, :, indices].copy()
    if gain is None:gain = np.linalg.solve(local_covariance, field.covariances[:, indices, :]).transpose(0, 2, 1)
    field.means += np.einsum('kij,kj->ki', gain, result['mean']-local_mean)
    correction = gain@(result['covariance']-local_covariance)@gain.transpose(0, 2, 1)
    field.covariances += correction
    field.covariances = .5*(field.covariances+field.covariances.transpose(0, 2, 1))
    field.log_weights += result['log_evidence']; evidence = float(logsumexp(field.log_weights))
    field.log_weights -= evidence; field.paid_batches += 1; field.paid_episodes += replicates
    field.max_integration_relative_error = max(field.max_integration_relative_error, result['integration_relative_error'])
    return evidence


def update_raw_observation(field, candidate, label, eta_label=None, replicates=1):
    indices=[2*candidate,2*candidate+1]
    result=local_raw_moments(field.means[:,indices],field.covariances[:,indices][:,:,indices],
                             label,eta_label,replicates)
    return apply_raw_moments(field,candidate,result,replicates=replicates)


def joint_variance_field(basis, psi, source, variances, weights, eta):
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    gram = np.asarray(source['joint_variance_source_gram']); factor = cho_factor(gram, lower=True)
    inverse = cho_solve(factor, np.eye(len(gram))); inverse = .5*(inverse+inverse.T)
    rhs = np.asarray(source['joint_variance_source_rhs']); update = design@np.asarray(source['common_coefficient_update_columns'])
    beta00 = np.asarray(source['coefficient_mean_at_zero_level_and_selected_shape_variances'])
    h00 = np.asarray(source['common_covariance_at_zero_level_and_selected_shape_variances'])
    residual = np.asarray(source['residual_diagonal_without_selected_shape_excess_at_native_level']).copy(); residual[0] = 0.
    unit = np.asarray(source['task_group_unit_coefficient_diagonal'])
    base = design@h00@design.T+(design*residual)@design.T+source['native_law']['discrepancy_variance']*np.eye(len(psi))
    shape_covariance = (design*unit)@design.T; level_covariance = np.outer(design[:, 0], design[:, 0])
    count, sites = len(variances), len(psi); split = source['level_column_count']
    means = np.empty((count, sites, 2)); covariances = np.zeros((count, sites, 2, sites, 2))
    means[:, :, 1] = eta[0]; covariances[:, :, 1, :, 1] = eta[1]
    for i, (level, shape) in enumerate(variances):
        diagonal = np.r_[np.full(split, level), np.full(len(gram)-split, shape)]
        posterior = cho_solve(cho_factor(inverse+np.diag(diagonal), lower=True), np.eye(len(gram)))
        adjustment = inverse-inverse@posterior@inverse; adjustment = .5*(adjustment+adjustment.T)
        means[i, :, 0] = design@beta00-update@adjustment@rhs
        covariances[i, :, 0, :, 0] = base+update@adjustment@update.T+level*level_covariance+shape*shape_covariance
    field = JointFunctionRiskPosterior.from_moments(means[0], covariances[0])
    field.means, field.covariances = means.reshape(count, 2*sites), covariances.reshape(count, 2*sites, 2*sites)
    field.log_weights = np.log(weights/np.sum(weights)); field.component_pairs = [(0, 0)]*count
    return field
