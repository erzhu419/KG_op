"""Joint Gaussian source likelihood and Jeffreys measure for two variances."""

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular


class TaskLevelShapeReference:
    def __init__(self, system):
        self.system = system
        levels = block_diag(*[np.ones((len(t['psi']), 1)) for t in system['tasks']])
        self.level_count = levels.shape[1]
        self.u = np.column_stack([levels, system['u']])
        self.native_level = float(system['native']['residual_coefficient_diagonal'][0])
        self.native_shape = system['variance']; self.signal = system['native']['source_signal_scale']
        self.c00 = system['c0']-self.native_level*levels@levels.T
        factor = cho_factor(self.c00, lower=True); inverse_u = cho_solve(factor, self.u)
        alpha = cho_solve(factor, system['error'])
        self.gram = self.u.T@inverse_u; self.gram = .5*(self.gram+self.gram.T)
        self.rhs = self.u.T@alpha
        self.update = system['common']@system['design'].T@inverse_u
        self.beta00 = system['m0']+system['common']@system['design'].T@alpha
        self.h00 = system['common']-system['common']@system['design'].T@cho_solve(factor, system['design'])@system['common']
        self.h00 = .5*(self.h00+self.h00.T)
        self.nll00 = float(.5*(len(alpha)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+system['error']@alpha))
        whitened = solve_triangular(np.tril(factor[0]), self.u, lower=True)
        singular = np.linalg.svd(whitened, compute_uv=False)
        self.rank = int(np.sum(singular > max(whitened.shape)*np.finfo(float).eps*singular[0]))
        assert self.rank == self.u.shape[1], 'Joint variance design is not full column rank.'
        assert self.level_count > 4 and system['rank'] > 4
        gram_factor = cho_factor(self.gram, lower=True)
        self.gram_inverse = cho_solve(gram_factor, np.eye(len(self.rhs)))
        self.gram_inverse = .5*(self.gram_inverse+self.gram_inverse.T)
        self.gram_log_determinant = float(2*np.log(np.diag(gram_factor[0])).sum())

    def evaluate(self, level, shape, *, moments=True):
        assert level >= 0 and shape >= 0
        variances = np.r_[np.full(self.level_count, level), np.full(len(self.rhs)-self.level_count, shape)]
        factor = cho_factor(self.gram_inverse+np.diag(variances), lower=True)
        b = cho_solve(factor, np.eye(len(self.rhs))); b = .5*(b+b.T)
        adjustment = self.gram_inverse-self.gram_inverse@b@self.gram_inverse
        adjustment = .5*(adjustment+adjustment.T)
        nll = self.nll00+.5*(self.gram_log_determinant+2*np.log(np.diag(factor[0])).sum()-self.rhs@adjustment@self.rhs)
        split = self.level_count; vv, aa, va = b[:split, :split], b[split:, split:], b[:split, split:]
        fisher = .5*np.array([[np.sum(vv**2), np.sum(va**2)], [np.sum(va**2), np.sum(aa**2)]])
        determinant = float(np.linalg.det(fisher)); assert determinant > 0
        correlation = float(fisher[0, 1]/np.sqrt(fisher[0, 0]*fisher[1, 1]))
        value = {'total_source_nll': float(nll), 'fisher': fisher, 'fisher_correlation': correlation,
            'joint_jeffreys_log_measure': float(.5*np.log(determinant)),
            'joint_to_product_conditional_jeffreys_factor': float(np.sqrt(1-correlation**2))}
        if moments:
            value['beta'] = self.beta00-self.update@adjustment@self.rhs
            h = self.h00+self.update@adjustment@self.update.T
            value['h'] = .5*(h+h.T)
        return value

    def dense_verification(self, level, shape):
        value = self.evaluate(level, shape); s = self.system
        uv, ug = self.u[:, :self.level_count], self.u[:, self.level_count:]
        factor = cho_factor(self.c00+level*uv@uv.T+shape*ug@ug.T, lower=True)
        alpha = cho_solve(factor, s['error'])
        nll = .5*(len(alpha)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+s['error']@alpha)
        beta = s['m0']+s['common']@s['design'].T@alpha
        h = s['common']-s['common']@s['design'].T@cho_solve(factor, s['design'])@s['common']
        b = self.u.T@cho_solve(factor, self.u); split = self.level_count
        fisher = .5*np.array([[np.sum(b[:split, :split]**2), np.sum(b[:split, split:]**2)],
            [np.sum(b[:split, split:]**2), np.sum(b[split:, split:]**2)]])
        for key, expected in [('total_source_nll', nll), ('beta', beta), ('h', h), ('fisher', fisher)]:
            np.testing.assert_allclose(value[key], expected, atol=1e-8, rtol=1e-8)
        return {'interior_total_nll_error': float(abs(value['total_source_nll']-nll)),
            'interior_fisher_max_relative_error': float(np.max(np.abs(value['fisher']/fisher-1.))),
            'dense_joint_likelihood_fisher_beta_h_verified': True}
