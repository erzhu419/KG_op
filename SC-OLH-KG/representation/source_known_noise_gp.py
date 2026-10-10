"""Source RBF task prior with observed label error, no function dictionary."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def fit_known_noise_gp(tasks, head, max_iterations=512):
    responses = np.vstack([t['response'] for t in tasks])[:, head]
    center, scale = float(responses.mean()), float(responses.std())
    records = [(np.asarray(t['features']), (np.asarray(t['response'])[:, head]-center)/scale,
                np.asarray(t['noise'])[:, head]/scale**2) for t in tasks]
    differences = [(x[:, None]-x[None, :])**2 for x, _, _ in records]
    dimension = records[0][0].shape[1]; count = len(responses)-1

    def likelihood(parameters, details=False):
        lengths = np.exp(parameters[:dimension]); signal, discrepancy = parameters[-2:]
        solved, info, rhs, logdet = [], 0., 0., 0.
        for difference, (_, y, noise) in zip(differences, records):
            distances = difference/lengths**2
            kernel = np.exp(-.5*distances.sum(axis=2))
            K = signal*kernel+np.diag(noise+discrepancy)
            factor = cho_factor(K, lower=True)
            inverse = cho_solve(factor, np.eye(len(y))); one = inverse @ np.ones(len(y))
            info += one.sum(); rhs += one @ y; logdet += 2*np.log(np.diag(factor[0])).sum()
            solved.append((y, kernel, distances, inverse, one))
        mean = rhs/info; value = .5*(logdet+np.log(info)+count*np.log(2*np.pi))
        gradient = np.zeros(dimension+2)
        for y, kernel, distances, inverse, one in solved:
            error = y-mean; alpha = inverse @ error
            value += .5*error @ alpha
            sensitivity = .5*(inverse-np.outer(alpha, alpha)-np.outer(one, one)/info)
            gradient[:dimension] += np.sum((sensitivity*signal*kernel)[:, :, None]*distances, axis=(0, 1))
            gradient[-2] += np.sum(sensitivity*kernel); gradient[-1] += np.trace(sensitivity)
        if details:
            return mean, 1/info
        return value/count, gradient/count

    initial = np.r_[np.zeros(dimension), 1., .01]
    fitted = minimize(likelihood, initial, method='L-BFGS-B', jac=True,
        bounds=[(np.log(.05), np.log(20.))]*dimension+[(0., None), (0., None)],
        options={'maxiter': max_iterations, 'ftol': 1e-9, 'gtol': 1e-5})
    mean, variance = likelihood(fitted.x, details=True)
    return {'normalization': {'mean': center, 'scale': scale},
        'constant': mean, 'constant_variance': variance,
        'lengthscale': np.exp(fitted.x[:dimension]).tolist(),
        'signal_variance': float(fitted.x[-2]), 'discrepancy_variance': float(fitted.x[-1]),
        'optimizer_converged': bool(fitted.success), 'optimizer_iterations': int(fitted.nit),
        'optimizer_message': str(fitted.message), 'source_restricted_nll_per_label': float(fitted.fun),
        'initial_restricted_nll_per_label': float(likelihood(initial)[0]),
        'source_label_count': len(responses)}
