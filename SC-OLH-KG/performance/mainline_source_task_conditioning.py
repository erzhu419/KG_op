"""Project the frozen source fit back onto an existing source task.

This is for held-out physical policies of that task. Its training raws already
entered the source likelihood and must not be replayed as new target data.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from performance.mainline_legacy_source_models import _eta_rule
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior


def _coefficient_cross(law, family_slice):
    design = law['source_design']
    cross = law['common'] @ design.T
    cross[:, family_slice] += law['task'] @ design[family_slice].T
    return cross


def condition_existing_source_task(record, labels, sample_variances, settings, family_slice):
    """Return an f/eta_f record for the same family's unobserved policies.

    The coefficient is common + this source task's deviation. Condition it
    jointly on each source eta node, using the same raw source likelihood as
    the original fit. Setting query task covariance to zero prevents adding
    an independent, new-task deviation after conditioning this existing task.
    """
    law, risk = record['law'], record['risk']
    states, _ = _eta_rule(risk, sample_variances, settings, settings['seed'])
    model = VarianceReferencePosterior(law['source_mean'], law['source_covariance'],
                                      labels, sample_variances)
    common_cross = law['common'] @ law['source_design'].T
    task_cross = _coefficient_cross(law, family_slice)
    eta_common_cross = risk['common'] @ risk['source_design'].T
    eta_task_cross = _coefficient_cross(risk, family_slice)
    eta_factor = cho_factor(risk['source_covariance'], lower=True)
    eta_common_gain = cho_solve(eta_factor, eta_common_cross.T).T
    eta_task_gain = cho_solve(eta_factor, eta_task_cross.T).T
    eta_residual = states - risk['source_mean']
    eta_common_mean = risk['mean'] + eta_residual @ eta_common_gain.T
    eta_common_cov = risk['common'] - eta_common_gain @ eta_common_cross.T
    eta_task_mean = risk['mean'] + eta_residual @ eta_task_gain.T
    eta_task_cov = risk['common'] + risk['task'] - eta_task_gain @ eta_task_cross.T
    means, covariances, logs = [], [], []
    checks = {'common_mean': 0., 'common_covariance': 0.,
              'common_eta_mean': float(np.max(np.abs(eta_common_mean-record['eta_mean']))),
              'common_eta_covariance': float(np.max(np.abs(eta_common_cov-record['eta_covariance'])))}
    for node, state in enumerate(states):
        log_density, factor, alpha = model.evaluate(state)
        common_mean = law['mean'] + common_cross @ alpha
        common_cov = law['common'] - common_cross @ cho_solve(factor, common_cross.T)
        checks['common_mean'] = max(checks['common_mean'],
            float(np.max(np.abs(common_mean-record['mean'][node]))))
        checks['common_covariance'] = max(checks['common_covariance'],
            float(np.max(np.abs(common_cov-record['covariance'][node]))))
        means.append(law['mean'] + task_cross @ alpha)
        covariance = law['common'] + law['task'] - task_cross @ cho_solve(factor, task_cross.T)
        covariances.append(.5*(covariance+covariance.T))
        logs.append(log_density)
    logs = np.asarray(logs)
    checks['normalized_node_log_weights'] = float(np.max(np.abs(
        logs-logsumexp(logs)-record['log_weights'])))
    maximum = max(checks.values())
    if maximum > 1e-9:
        raise ValueError(f'Source node/likelihood reconstruction mismatch: {checks}')
    conditioned = dict(record, law=dict(law, task=np.zeros_like(law['task'])),
        risk=dict(risk, task=np.zeros_like(risk['task'])), mean=np.asarray(means),
        covariance=np.asarray(covariances), eta_mean=eta_task_mean,
        eta_covariance=.5*(eta_task_cov+eta_task_cov.T))
    return conditioned, {'maximum_reconstruction_error': maximum, 'errors': checks,
                         'source_nodes': len(states), 'training_observations_added_again': False}
