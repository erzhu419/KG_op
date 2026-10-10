"""Four-field Gaussian moment filtering with paired observation evidence.

The latent order is (f, eta_f, g, eta_g). Mean variables integrate analytically;
only the two log variances require quadrature. Projection after each paid batch
has the same status as the existing two-field Gaussian moment filter.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, logsumexp, roots_hermitenorm

from representation.joint_function_risk import local_joint_moments
from representation.physical_risk_prior import log_sample_variance_density

INTEGRATION_ORDERS = (24, 48, 96, 192, 384)
INTEGRATION_TOLERANCE = 1e-7


@lru_cache(maxsize=5)
def normal_grid(order):
    nodes, weights = roots_hermitenorm(order)
    points = np.stack(np.meshgrid(nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 2)
    logs = np.log(weights)-.5*np.log(2*np.pi)
    return points, (logs[:, None]+logs[None, :]).reshape(-1)


def paired_local_moments(mean, covariance, labels, eta_labels, replicates):
    mean, covariance, labels = np.asarray(mean), np.asarray(covariance), np.asarray(labels)
    if replicates < 1 or int(replicates) != replicates:
        raise ValueError('replicates must be a positive integer')
    if any((replicates == 1) != (e is None) for e in eta_labels):
        raise ValueError('single episodes have no variance labels; batches require both')
    if not np.any(covariance[:2, 2:]):
        heads = [local_joint_moments(mean[2*h:2*h+2], covariance[2*h:2*h+2, 2*h:2*h+2],
            labels[h], eta_labels[h], replicates) for h in range(2)]
        result_covariance = np.zeros((4, 4))
        for h, result in enumerate(heads):
            result_covariance[2*h:2*h+2, 2*h:2*h+2] = result['covariance']
        return {'mean': np.concatenate([r['mean'] for r in heads]), 'covariance': result_covariance,
            'log_evidence': sum(r['log_evidence'] for r in heads),
            'integration_error': max(r['integration_relative_error'] for r in heads), 'integration_order': 0}

    mu, eta = np.array([0, 2]), np.array([1, 3])
    eta_covariance = covariance[np.ix_(eta, eta)]; eta_factor = np.linalg.cholesky(eta_covariance)
    loading = np.linalg.solve(eta_covariance, covariance[np.ix_(eta, mu)]).T
    conditional = covariance[np.ix_(mu, mu)]-loading@covariance[np.ix_(eta, mu)]
    sample_variance = None
    if replicates > 1:
        k = (replicates-1)/2
        sample_variance = np.exp(np.asarray(eta_labels)+digamma(k)-np.log(k))
    prior_logdet = 2*np.log(np.diag(eta_factor)).sum()

    def values(points):
        points = np.atleast_2d(points); shift = points-mean[eta]
        whitened = np.linalg.solve(eta_factor, shift.T).T
        prior_mu = mean[mu]+shift@loading.T; error = labels-prior_mu
        noise = np.exp(points)/replicates
        a, b, c = conditional[0, 0]+noise[:, 0], conditional[0, 1], conditional[1, 1]+noise[:, 1]
        determinant = a*c-b*b
        alpha = np.column_stack([(c*error[:, 0]-b*error[:, 1])/determinant,
            (a*error[:, 1]-b*error[:, 0])/determinant])
        logs = -np.log(2*np.pi)-.5*(prior_logdet+(whitened**2).sum(axis=1))
        logs += -np.log(2*np.pi)-.5*(np.log(determinant)+(error*alpha).sum(axis=1))
        gradient = alpha@loading-.5*noise*np.column_stack([c/determinant, a/determinant])+ .5*noise*alpha**2
        if sample_variance is not None:
            logs += log_sample_variance_density(points, sample_variance, replicates).sum(axis=1)
            gradient += k*(sample_variance*np.exp(-points)-1)
        gradient -= np.linalg.solve(eta_covariance, shift.T).T
        posterior_mu = prior_mu+alpha@conditional.T
        inverse = np.empty((len(points), 2, 2))
        inverse[:, 0, 0], inverse[:, 1, 1] = c/determinant, a/determinant
        inverse[:, 0, 1] = inverse[:, 1, 0] = -b/determinant
        posterior_mu_covariance = conditional-np.einsum('ij,njk,kl->nil', conditional, inverse, conditional)
        return logs, gradient, posterior_mu, posterior_mu_covariance

    def objective(z):
        logs, gradient, _, _ = values(mean[eta]+eta_factor@z)
        return -logs[0], -(gradient[0]@eta_factor)

    result = minimize(objective, np.zeros(2), method='BFGS', jac=True, options={'gtol': 1e-7})
    mode = mean[eta]+eta_factor@result.x
    step = 1e-3; hessian = np.column_stack([(objective(result.x+step*np.eye(2)[j])[1]
        -objective(result.x-step*np.eye(2)[j])[1])/(2*step) for j in range(2)])
    hessian = .5*(hessian+hessian.T)
    proposal = 2*eta_factor@np.linalg.solve(hessian, eta_factor.T)
    proposal_factor = np.linalg.cholesky(proposal)
    proposal_logdet = 2*np.log(np.diag(proposal_factor)).sum()
    standard_deviation = np.sqrt(np.diag(covariance)); previous = None
    for order in INTEGRATION_ORDERS:
        z, log_weights = normal_grid(order); points = mode+z@proposal_factor.T
        logs, _, posterior_mu, posterior_mu_covariance = values(points)
        proposal_logs = -np.log(2*np.pi)-.5*(proposal_logdet+(z**2).sum(axis=1))
        log_mass = logs-proposal_logs+log_weights; evidence = float(logsumexp(log_mass))
        weights = np.exp(log_mass-evidence)
        displacement = np.empty((len(points), 4))
        displacement[:, mu] = posterior_mu-mean[mu]; displacement[:, eta] = points-mean[eta]
        change = weights@displacement; centered = displacement-change
        projected = (centered.T*weights)@centered
        projected[np.ix_(mu, mu)] += np.einsum('n,nij->ij', weights, posterior_mu_covariance)
        statistics = np.r_[evidence, change/standard_deviation,
            (projected/np.outer(standard_deviation, standard_deviation)).reshape(-1)]
        if previous is not None:
            error = float(np.max(abs(statistics-previous)))
            if error < INTEGRATION_TOLERANCE:
                return {'mean': mean+change, 'covariance': projected, 'log_evidence': evidence,
                    'integration_error': error, 'integration_order': order}
        previous = statistics
    raise RuntimeError(f'paired quadrature did not converge: normalized moment/log-evidence difference {error:.3g}')


class PairedFunctionRiskMixture:
    """One four-field Gaussian per complete source risk identity."""

    def __init__(self, names, means, covariances):
        self.names = list(names); self.candidate_count = len(means[0])
        self.means = np.asarray(means).reshape(len(names), -1).copy()
        self.covariances = np.asarray(covariances).reshape(len(names), 4*self.candidate_count, 4*self.candidate_count).copy()
        self.log_weights = np.full(len(names), -np.log(len(names)))
        self.paid_episodes = 0; self.history = []

    @property
    def weights(self):
        return np.exp(self.log_weights)

    def update_raw(self, candidate, values):
        return self._update(candidate, values, [None, None], 1)

    def update_replicates(self, candidate, values):
        values = np.asarray(values); count = len(values); eta = [None, None]
        if count > 1:
            k = (count-1)/2
            eta = np.log(np.var(values, axis=0, ddof=1))-digamma(k)+np.log(k)
        return self._update(candidate, values.mean(axis=0), eta, count)

    def _update(self, candidate, labels, eta, replicates):
        indices = 4*candidate+np.arange(4); before = self.log_weights.copy(); evidences, errors, orders = [], [], []
        for i in range(len(self.names)):
            mean, covariance = self.means[i], self.covariances[i]
            local = covariance[np.ix_(indices, indices)]
            result = paired_local_moments(mean[indices], local, labels, eta, replicates)
            gain = np.linalg.solve(local, covariance[indices, :]).T
            self.means[i] = mean+gain@(result['mean']-mean[indices])
            projected = covariance+gain@(result['covariance']-local)@gain.T
            self.covariances[i] = .5*(projected+projected.T)
            evidences.append(result['log_evidence']); errors.append(result['integration_error']); orders.append(result['integration_order'])
        evidence = float(logsumexp(before+evidences)); self.log_weights = before+evidences-evidence
        self.paid_episodes += replicates
        record = {'candidate': int(candidate), 'replicates': int(replicates), 'joint_log_scores': evidences,
            'mixture_log_score': evidence, 'weights_before': np.exp(before).tolist(), 'weights_after': self.weights.tolist(),
            'max_integration_error': max(errors), 'integration_orders': orders}
        self.history.append(record)
        return record

    def predict_joint(self):
        mean = self.weights@self.means; centered = self.means-mean
        covariance = np.einsum('m,mij->ij', self.weights, self.covariances)+(centered.T*self.weights)@centered
        n = self.candidate_count
        return mean.reshape(n, 4), covariance.reshape(n, 4, n, 4)

    def expected_episode_variance(self):
        n = self.candidate_count; indices = (4*np.arange(n)[:, None]+np.array([1, 3])).reshape(-1)
        variances = np.diagonal(self.covariances, axis1=1, axis2=2)[:, indices]
        return (self.weights@np.exp(self.means[:, indices]+.5*variances)).reshape(n, 2)

    def batch_log_scores(self, candidate, means, eta, replicates):
        indices = 4*candidate+np.arange(4); joint, heads, errors, orders = [], [], [], []
        for mean, covariance in zip(self.means, self.covariances):
            local_mean, local_covariance = mean[indices], covariance[np.ix_(indices, indices)]
            result = paired_local_moments(local_mean, local_covariance, means, eta, replicates)
            joint.append(result['log_evidence']); errors.append(result['integration_error']); orders.append(result['integration_order'])
            heads.append([local_joint_moments(local_mean[2*h:2*h+2], local_covariance[2*h:2*h+2, 2*h:2*h+2],
                means[h], eta[h], replicates)['log_evidence'] for h in range(2)])
        return {'joint': float(logsumexp(self.log_weights+joint)),
            'heads': logsumexp(self.log_weights[:, None]+heads, axis=0).tolist(),
            'max_integration_error': max(errors), 'integration_orders': orders}
