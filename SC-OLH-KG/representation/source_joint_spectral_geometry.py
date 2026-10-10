"""Joint spectral learning with fixed source-derived correlation templates."""

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from representation.orthogonal_function_prior import spectral_weights


class JointSpectralGeometryLikelihood:
    """Four source parameters; positive tails have separate source budgets."""

    def __init__(self, basis, tasks, common, task, columns, center, white, tail_fraction):
        self.basis, self.templates = basis, (np.asarray(common), np.asarray(task))
        self.columns, self.center, self.white = np.asarray(columns), center, white
        self.designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(t['psi'])]) for t in tasks]
        self.design = np.vstack(self.designs)
        self.error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks]) - center
        self.noise = np.concatenate([np.asarray(t['noise'])[:, 0] for t in tasks]) + white
        self.budgets = [np.bincount(basis.group_index, weights=np.diag(c)[1:], minlength=len(basis.group_names)) for c in self.templates]
        self.tails = [tail_fraction * np.diag(c)[1:].sum() / basis.feature_dim for c in self.templates]
        assert all(tau > 0 for tau in self.tails)
        self.label_count = len(self.error)

    def covariance(self, component, root_scale, exponent):
        template = self.templates[component]; diagonal = np.diag(template)
        weights, derivative = spectral_weights(self.basis, exponent)
        marginal = self.budgets[component][self.basis.group_index] * weights
        ratio = np.zeros_like(diagonal); ratio[0] = 1.
        shape = self.columns[1:]
        ratio[shape] = np.sqrt(marginal[shape - 1] / diagonal[shape])
        scaling = ratio.copy(); scaling[1:] *= root_scale
        dscale = ratio.copy(); dscale[0] = 0.
        dexponent = np.r_[0., .5 * scaling[1:] * derivative / weights]
        covariance = template * np.outer(scaling, scaling)
        root_gradient = template * (np.outer(dscale, scaling) + np.outer(scaling, dscale))
        exponent_gradient = template * (np.outer(dexponent, scaling) + np.outer(scaling, dexponent))
        covariance += np.diag(np.r_[0., np.full(self.basis.feature_dim, self.tails[component])])
        return covariance, root_gradient, exponent_gradient

    def evaluate(self, parameters):
        common, ca, cp = self.covariance(0, parameters[0], parameters[1])
        task, ra, rp = self.covariance(1, parameters[2], parameters[3])
        ds = self.design
        system = ds @ common @ ds.T + block_diag(*[d @ task @ d.T for d in self.designs]) + np.diag(self.noise)
        factor = cho_factor(system, lower=True)
        solved = cho_solve(factor, np.column_stack([self.error, np.eye(self.label_count)]))
        alpha, inverse = solved[:, 0], solved[:, 1:]
        contrast = .5 * (inverse - np.outer(alpha, alpha)) / self.label_count
        directions = (ds @ ca @ ds.T, ds @ cp @ ds.T,
            block_diag(*[d @ ra @ d.T for d in self.designs]), block_diag(*[d @ rp @ d.T for d in self.designs]))
        gradient = np.asarray([np.sum(contrast * direction) for direction in directions])
        value = .5 * (self.label_count * np.log(2 * np.pi) + 2 * np.log(np.diag(factor[0])).sum() + self.error @ alpha) / self.label_count
        beta = np.r_[self.center, np.zeros(self.basis.feature_dim)] + common @ ds.T @ alpha
        h = common - common @ ds.T @ cho_solve(factor, ds @ common); h = .5 * (h + h.T)
        return float(value), gradient, {'common_prior_coefficient_covariance': common, 'task_coefficient_covariance': task,
            'source_posterior_coefficient_mean': beta, 'source_posterior_common_covariance': h}
