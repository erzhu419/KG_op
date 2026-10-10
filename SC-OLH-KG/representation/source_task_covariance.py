"""Source common-function posterior with one learned task covariance direction."""

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from representation.source_joint_population import GaussianPopulationLikelihood


class SourceTaskDirectionLikelihood:
    def __init__(self, prior, batches, noises, common_diagonal, residual_diagonal, discrepancy, direction):
        source = GaussianPopulationLikelihood(prior, batches, noises)
        self.prior, self.blocks, self.label_count = prior, source.blocks, source.label_count
        self.origin = np.asarray(prior.mean)
        self.common_diagonal = np.asarray(common_diagonal)
        self.residual_diagonal = np.asarray(residual_diagonal)
        self.direction = np.asarray(direction); self.discrepancy = discrepancy
        self.precision0 = np.diag(1/self.common_diagonal)

    @classmethod
    def from_paired_heads(cls, priors, batches, noises, common, residual, discrepancies, direction):
        """Join f/g observations within each source task before integrating it."""
        heads = [GaussianPopulationLikelihood(p, b, n) for p, b, n in zip(priors, batches, noises)]
        result = cls.__new__(cls)
        result.origin = np.concatenate([p.mean for p in priors])
        result.blocks = [(block_diag(*[h.blocks[i][0] for h in heads]),
            np.concatenate([h.blocks[i][1] for h in heads]),
            np.concatenate([h.blocks[i][2]+d for h, d in zip(heads, discrepancies)]))
            for i in range(len(heads[0].blocks))]
        result.label_count = sum(h.label_count for h in heads)
        result.common_diagonal, result.residual_diagonal = np.concatenate(common), np.concatenate(residual)
        result.direction, result.discrepancy = np.asarray(direction), 0.
        result.precision0 = np.diag(1/result.common_diagonal)
        return result

    def evaluate(self, scale):
        residual_covariance = np.diag(self.residual_diagonal)+float(scale)*np.outer(self.direction, self.direction)
        precision, rhs = self.precision0.copy(), np.zeros(len(self.origin))
        logdet = np.log(self.common_diagonal).sum(); solved = []
        for x, error, noise in self.blocks:
            variance = x@residual_covariance@x.T+np.diag(noise+self.discrepancy)
            factor = cho_factor(variance, lower=True)
            solved_x = cho_solve(factor, x); solved_error = cho_solve(factor, error)
            direction = x@self.direction; solved_direction = cho_solve(factor, direction)
            precision += x.T@solved_x; rhs += x.T@solved_error
            logdet += 2*np.log(np.diag(factor[0])).sum()
            solved.append((x, error, factor, direction, solved_direction))
        factor = cho_factor(precision, lower=True)
        displacement = cho_solve(factor, rhs); covariance = cho_solve(factor, np.eye(len(rhs)))
        logdet += 2*np.log(np.diag(factor[0])).sum()
        quadratic = float(displacement@self.precision0@displacement); gradient = 0.
        for x, error, factor, direction, solved_direction in solved:
            residual = error-x@displacement; alpha = cho_solve(factor, residual)
            quadratic += residual@alpha; projected = x.T@solved_direction
            gradient += .5*(direction@solved_direction-projected@covariance@projected-(direction@alpha)**2)
        value = .5*(self.label_count*np.log(2*np.pi)+logdet+quadratic)/self.label_count
        return float(value), np.array([gradient/self.label_count]), {
            'mean': self.origin+displacement, 'mean_covariance': covariance,
            'residual_covariance': residual_covariance, 'discrepancy': self.discrepancy}
