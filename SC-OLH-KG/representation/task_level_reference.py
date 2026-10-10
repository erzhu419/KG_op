"""Conditional reference posterior for one physical task-intercept variance."""

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq


class TaskLevelReferencePosterior:
    def __init__(self, eigenvalues, rhs, source_signal, native_variance, quadrature, slice_upper):
        self.eigenvalues, self.rhs = np.asarray(eigenvalues), np.asarray(rhs)
        self.signal, self.native = source_signal, native_variance
        self.tolerance = quadrature
        self.native_nll_difference = self.nll_difference(native_variance)
        self.points = sorted(set(v/(source_signal+v) for v in (native_variance, slice_upper) if v > 0))
        q_native = native_variance/(source_signal+native_variance)
        self.log_scale = self.log_q_weight(q_native)
        self.scalar_evaluations = self.vector_evaluations = 0
        self.maximum_scalar_error = 0.
        self.normalization, self.normalization_error = self.integrate_scalar(self.q_weight, 0., 1., normalization=True)

    def nll_difference(self, variance):
        x = variance*self.eigenvalues
        return float(.5*np.sum(np.log1p(x)-variance/(1+x)*self.rhs**2))

    def information(self, variance):
        return float(.5*np.sum((self.eigenvalues/(1+variance*self.eigenvalues))**2))

    def log_q_weight(self, q):
        variance = self.signal*q/(1-q)
        return (-self.nll_difference(variance)+self.native_nll_difference+.5*np.log(self.information(variance))
            +np.log(self.signal)-2*np.log1p(-q))

    def q_weight(self, q):
        if q >= 1.:
            return 0.
        return float(np.exp(self.log_q_weight(q)-self.log_scale))

    def integrate_scalar(self, function, low, high, *, normalization=False):
        value, error, info, *failure = quad(function, low, high,
            points=[p for p in self.points if low < p < high], full_output=True,
            epsabs=0. if normalization else self.tolerance['absolute_tolerance_for_scaled_integrals'],
            epsrel=self.tolerance['relative_tolerance'], limit=self.tolerance['maximum_subintervals'])
        self.scalar_evaluations += info['neval']; self.maximum_scalar_error = max(self.maximum_scalar_error, error)
        if failure:
            raise RuntimeError(f'Conditional reference scalar integration failed: {failure[0]}')
        absolute = 0. if normalization else self.tolerance['absolute_tolerance_for_scaled_integrals']
        assert error <= max(absolute, self.tolerance['relative_tolerance']*abs(value))
        return float(value), float(error)

    def cdf(self, q):
        if q <= 0.:
            return 0.
        if q >= 1.:
            return 1.
        return self.integrate_scalar(lambda x: self.q_weight(x)/self.normalization, 0., q)[0]

    def moments(self, probabilities):
        size = len(self.eigenvalues)
        def integrand(q):
            self.vector_evaluations += 1
            variance_over_signal = q/(1-q)
            x = self.signal*variance_over_signal*self.eigenvalues; shrinkage = x/(1+x)
            values = np.r_[1., variance_over_signal, variance_over_signal**2, shrinkage, np.outer(shrinkage, shrinkage).ravel()]
            return self.q_weight(q)/self.normalization*values
        values, error, info = quad_vec(integrand, 0., 1., points=self.points, norm='max', full_output=True,
            epsabs=self.tolerance['absolute_tolerance_for_scaled_integrals'], epsrel=self.tolerance['relative_tolerance'],
            limit=self.tolerance['maximum_subintervals'], workers=1)
        if not info.success:
            raise RuntimeError(f'Conditional reference vector integration failed: {info.message}')
        assert error <= max(self.tolerance['absolute_tolerance_for_scaled_integrals'],
            self.tolerance['relative_tolerance']*np.max(np.abs(values)))
        np.testing.assert_allclose(values[0], 1., atol=2*self.tolerance['relative_tolerance'], rtol=0.)
        mean, second = self.signal*values[1], self.signal**2*values[2]
        assert second >= mean**2
        quantiles = {}
        for p in probabilities:
            q = brentq(lambda q: self.cdf(q)-p, 0., 1., xtol=1e-12)
            quantiles[str(p)] = float(self.signal*q/(1-q))
        expected_shrinkage = values[3:3+size]
        shrinkage_covariance = values[3+size:].reshape(size, size)-np.outer(expected_shrinkage, expected_shrinkage)
        shrinkage_covariance = .5*(shrinkage_covariance+shrinkage_covariance.T)
        return {'mean': float(mean), 'standard_deviation': float(np.sqrt(second-mean**2)), 'quantiles': quantiles,
            'expected_shrinkage': expected_shrinkage, 'shrinkage_covariance': shrinkage_covariance,
            'normalization_integral': self.normalization, 'normalization_absolute_error': self.normalization_error,
            'normalization_relative_error': self.normalization_error/self.normalization,
            'normalized_vector_integral_error': float(error), 'normalized_mass_error': float(abs(values[0]-1.)),
            'normalized_vector_evaluations': self.vector_evaluations}
