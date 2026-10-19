import unittest

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from performance.diagnose_mainline_joint_source import delete_fold, gaussian_log_density


class SourceDeletionTest(unittest.TestCase):
    def test_precision_deletion_and_mean_likelihood_ratio_match_direct_conditioning(self):
        rng = np.random.default_rng(19)
        L = rng.normal(size=(9, 5))
        K = L @ L.T+.3*np.eye(9)
        F, y = rng.normal(size=(9, 3)), rng.normal(size=9)
        eta = rng.normal(size=(8, 9))
        noise, s2 = np.exp(eta)/3, np.exp(rng.normal(size=9))
        Ps = np.asarray([np.linalg.inv(K+np.diag(n)) for n in noise])
        alpha, pf = Ps @ y, Ps @ F
        state = {'noise': noise, 'eta': eta, 'F': F, 'precision': Ps, 'alpha': alpha,
            'pf': pf, 'information': np.asarray([F.T @ p for p in pf]),
            'score': alpha @ F, 'log_mass': np.full(8, -np.log(8))}
        test = np.asarray([1, 4, 7]); train = np.setdiff1d(np.arange(9), test)
        fold = delete_fold(state, y, s2, test)
        delta = np.asarray([.2, -.1, .3])
        for j, n in enumerate(noise):
            C = K+np.diag(n)
            fac = cho_factor(C[np.ix_(train, train)], lower=True)
            cross = K[np.ix_(test, train)]
            mean = cross @ cho_solve(fac, y[train])
            cov = K[np.ix_(test, test)]-cross @ cho_solve(fac, cross.T)
            np.testing.assert_allclose(fold['means'][j], mean, atol=1e-12)
            np.testing.assert_allclose(fold['latent_variances'][j], np.diag(cov), atol=1e-12)
            newmean = F[test] @ delta+cross @ cho_solve(fac, y[train]-F[train] @ delta)
            np.testing.assert_allclose(fold['means'][j]+fold['mean_loading'][j] @ delta, newmean, atol=1e-12)
            direct = gaussian_log_density(y[train]-F[train] @ delta, C[np.ix_(train, train)])
            direct -= gaussian_log_density(y[train], C[np.ix_(train, train)])
            fast = fold['score'][j] @ delta-.5*delta @ fold['information'][j] @ delta
            self.assertAlmostEqual(direct, fast, places=11)


if __name__ == '__main__':
    unittest.main()
