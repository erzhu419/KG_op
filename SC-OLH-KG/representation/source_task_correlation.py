"""Learn cross-function task correlation while fixing both task marginals."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class SourceFixedMarginalCorrelationLikelihood:
    def __init__(self, paired_source):
        self.origin = paired_source.origin
        self.blocks = paired_source.blocks
        self.label_count = paired_source.label_count
        self.common_diagonal = paired_source.common_diagonal
        self.residual_diagonal = paired_source.residual_diagonal
        self.precision0 = paired_source.precision0
        self.width = len(self.origin)//2
        self.normalized_direction = np.zeros(len(self.origin))
        for start in (0, self.width):
            section = slice(start, start+self.width)
            variance = self.residual_diagonal[section]
            direction = paired_source.direction[section]
            support = variance > 0
            if np.any(direction[~support] != 0):
                raise ValueError('correlation direction leaves the native covariance support')
            energy = np.sum(direction[support]**2/variance[support])
            if energy == 0:
                raise ValueError('both heads need a nonzero supported correlation direction')
            self.normalized_direction[section] = direction/np.sqrt(energy)
        cross = np.outer(self.normalized_direction[:self.width], self.normalized_direction[self.width:])
        self.derivative = np.zeros((len(self.origin), len(self.origin)))
        self.derivative[:self.width, self.width:] = cross
        self.derivative[self.width:, :self.width] = cross.T
        native = np.diag(self.residual_diagonal)
        self.task_matrices = [(x, error,
            x@native@x.T+np.diag(noise+paired_source.discrepancy), x@self.derivative@x.T)
            for x, error, noise in self.blocks]

    def evaluate(self, correlation):
        residual_covariance = np.diag(self.residual_diagonal)+float(correlation)*self.derivative
        precision, rhs = self.precision0.copy(), np.zeros(len(self.origin))
        logdet = np.log(self.common_diagonal).sum(); solved = []
        for x, error, native, derivative in self.task_matrices:
            factor = cho_factor(native+float(correlation)*derivative, lower=True)
            solved_x = cho_solve(factor, x)
            precision += x.T@solved_x; rhs += x.T@cho_solve(factor, error)
            logdet += 2*np.log(np.diag(factor[0])).sum()
            solved.append((x, error, factor, derivative, solved_x))
        factor = cho_factor(precision, lower=True)
        displacement = cho_solve(factor, rhs)
        covariance = cho_solve(factor, np.eye(len(rhs)))
        logdet += 2*np.log(np.diag(factor[0])).sum()
        quadratic = float(displacement@self.precision0@displacement); gradient = 0.
        for x, error, factor, derivative, solved_x in solved:
            alpha = cho_solve(factor, error-x@displacement)
            quadratic += (error-x@displacement)@alpha
            gradient += .5*(np.trace(cho_solve(factor, derivative))
                -np.sum((solved_x@covariance@solved_x.T)*derivative.T)-alpha@derivative@alpha)
        value = .5*(self.label_count*np.log(2*np.pi)+logdet+quadratic)/self.label_count
        return float(value), np.array([gradient/self.label_count]), {
            'mean': self.origin+displacement, 'mean_covariance': covariance,
            'residual_covariance': residual_covariance}
