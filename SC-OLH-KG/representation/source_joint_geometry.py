"""Joint common/task covariance learning inside the original sparse basis."""

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from representation.orthogonal_function_prior import spectral_weights
from representation.source_common_geometry import CommonGeometryLikelihood


class JointCommonTaskGeometryLikelihood:
    def __init__(self, source, native, common_geometry, task_template):
        self.source, self.native = source, native
        self.common = CommonGeometryLikelihood(source, native, common_geometry)
        self.common_count = len(self.common.initial)
        self.task_groups = np.flatnonzero((np.asarray(native['source_group_variances']) > 0)
            | (np.asarray(native['common_group_variances']) > 0))
        basis = source.basis
        self.columns = np.r_[0, 1+np.flatnonzero(np.isin(basis.group_index, self.task_groups))]
        saved = np.asarray(task_template['template_coefficient_covariance'])
        principal = saved[np.ix_(self.columns, self.columns)]; diagonal = np.diag(principal)
        assert np.all(diagonal > 0)
        self.task_correlation = principal/np.sqrt(np.outer(diagonal, diagonal))
        self.task_budgets = np.asarray(task_template['template_group_budgets'])
        self.template_level = task_template['template_level_variance']
        self.initial = np.r_[self.common.initial,
            np.asarray(native['source_group_variances'])[self.task_groups]/source.signal,
            native['spectral_exponent'], native['residual_coefficient_diagonal'][0]/source.signal, 0.]
        self.bounds = self.common.bounds+[(0., None)]*len(self.task_groups)+[(0., 4.), (0., None), (0., 1.)]
        self.names = [f'common_group_sd:{basis.group_names[g]}' for g in self.common.groups]+[
            'common_spectral_exponent', 'common_correlation_strength']+[
            f'task_group_variance_over_signal:{basis.group_names[g]}' for g in self.task_groups]+[
            'task_spectral_exponent', 'independent_task_intercept_variance_over_signal', 'task_template_strength']
        self.design = np.vstack([b[0] for b in source.blocks])
        self.error = np.concatenate([b[1] for b in source.blocks])
        self.task_slices = []; offset = 0
        for design, _, _ in source.blocks:
            self.task_slices.append(slice(offset, offset+len(design))); offset += len(design)
        np.testing.assert_allclose(self.task_covariance(self.initial[self.common_count:])[2], saved, atol=1e-14, rtol=1e-12)

    def task_covariance(self, parameters):
        basis, signal = self.source.basis, self.source.signal
        gamma, variance_parameter, q = parameters[-3:]
        weights, derivative = spectral_weights(basis, gamma)
        groups = np.zeros(len(basis.group_names)); groups[self.task_groups] = signal*parameters[:-3]
        excess = np.r_[0., groups[basis.group_index]*weights]
        marginal = np.r_[self.template_level, self.task_budgets[basis.group_index]*weights]
        template = np.zeros((len(excess), len(excess)))
        template[np.ix_(self.columns, self.columns)] = self.task_correlation*np.sqrt(np.outer(marginal[self.columns], marginal[self.columns]))
        log_derivative = np.r_[0., derivative/weights]
        template_derivative = .5*template*(log_derivative[:, None]+log_derivative[None, :])
        base = np.diag(np.r_[0., np.full(basis.feature_dim, self.source.tail)])+(1-q)*np.diag(excess)+q*template
        covariance = base.copy(); covariance[0, 0] += signal*variance_parameter
        directions = [(1-q)*np.diag(np.r_[0., signal*weights*(basis.group_index == g)]) for g in self.task_groups]
        directions.append((1-q)*np.diag(np.r_[0., groups[basis.group_index]*derivative])+q*template_derivative)
        intercept = np.zeros_like(base); intercept[0, 0] = signal
        directions.extend([intercept, template-np.diag(excess)])
        return covariance, base, template, groups, directions

    def source_covariance(self, common, task):
        return self.design@common@self.design.T+block_diag(*[
            d@task@d.T+np.diag(noise+self.native['discrepancy_variance']) for d, _, noise in self.source.blocks])

    def evaluate(self, parameters):
        parameters = np.asarray(parameters)
        common, common_groups, common_directions = self.common.covariance(parameters[:self.common_count])
        task, task_base, template, task_groups, task_directions = self.task_covariance(parameters[self.common_count:])
        covariance = self.source_covariance(common, task)
        factor = cho_factor(covariance, lower=True); alpha = cho_solve(factor, self.error)
        inverse = cho_solve(factor, np.eye(len(self.error)))
        value = .5*(len(self.error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+self.error@alpha)/len(self.error)
        score = .5*(inverse-np.outer(alpha, alpha))/len(self.error)
        common_score = self.design.T@score@self.design
        active_score = common_score[np.ix_(self.common.active+1, self.common.active+1)]
        task_score = sum(d.T@score[s, s]@d for (d, _, _), s in zip(self.source.blocks, self.task_slices))
        gradient = np.r_[[np.sum(active_score*direction) for direction in common_directions],
            [np.sum(task_score*direction) for direction in task_directions]]
        beta = self.source.prior.mean+common@self.design.T@alpha
        h = common-common@self.design.T@inverse@self.design@common; h = .5*(h+h.T)
        state = {'source_negative_log_likelihood_per_label': float(value), 'coefficient_mean': beta,
            'population_mean_covariance': h, 'common_prior_coefficient_covariance': common,
            'residual_coefficient_covariance': task, 'task_covariance_at_zero_independent_intercept': task_base,
            'task_template_at_fitted_spectrum': template, 'common_group_variances': common_groups,
            'source_group_variances': task_groups, 'common_spectral_exponent': float(parameters[self.common_count-2]),
            'common_correlation_strength': float(parameters[self.common_count-1]),
            'spectral_exponent': float(parameters[-3]), 'independent_task_intercept_variance': float(self.source.signal*parameters[-2]),
            'task_template_strength': float(parameters[-1])}
        return float(value), gradient, state
