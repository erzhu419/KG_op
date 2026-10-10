import copy
import itertools
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import chi2, norm, t
from scipy.special import roots_genlaguerre
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.verification_kg import (predictive_pass_probability, verification_kg_scores,
                                risk_only_kg_scores, verification_quadrature)
from core.verification_kg import posterior_covariances, verification_utility
from representation.adaptive_sparsity import AdaptiveSpikeSlabPosterior
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.source_risk_ard import fit_source_risk_ard
from representation.transfer_mixture import TransferMixtureGPR
from representation.transferable_spectral import SourceDomainBatch

PROTOCOL = json.loads((ROOT / "performance/meta_prior_mechanism_v2_protocol.json").read_text())


class VerificationMetaPriorTests(unittest.TestCase):
    def test_normalized_chi_quadrature_preserves_old_nodes_and_large_budgets(self):
        old_nodes, old_weights = roots_genlaguerre(12, 255 / 2 - 1)
        _, _, terms, weights = verification_quadrature(256, .05)
        constant = t.ppf(.975, 255) / np.sqrt(256) + norm.ppf(.95) * np.sqrt(255 / chi2.ppf(.025, 255))
        np.testing.assert_allclose(terms, np.sqrt(2 * old_nodes / 255) * constant, rtol=1e-13)
        np.testing.assert_allclose(weights, old_weights / old_weights.sum(), atol=1e-13)
        for reps in (736, 928, 1120):
            _, _, terms, weights = verification_quadrature(reps, .05)
            self.assertTrue(np.isfinite(terms).all() and np.isfinite(weights).all())
            self.assertAlmostEqual(float(weights.sum()), 1., places=13)

    def test_predictive_certificate_power_matches_independent_simulation(self):
        rng = np.random.default_rng(9771)
        cases = [(0., -.55, -2., .02, .1), (0., -.25, -2.5, .03, .2),
                 (0., -.12, -3., 0., 0.)]
        for reps, (_, g, eta, vg, veta) in [(r, c) for r in (256, 928, 1120) for c in cases]:
            df = reps - 1
            mean, variance = np.array([0., g, eta]), np.array([0., vg, veta])
            protocol = {**PROTOCOL, "verification_replicates": reps}
            predicted = float(predictive_pass_probability(mean, variance, protocol))
            sigma = np.exp(.5 * (eta + np.sqrt(veta) * rng.normal(size=100000)))
            actual_mean = g + np.sqrt(vg) * rng.normal(size=len(sigma))
            sample_mean = actual_mean + sigma / np.sqrt(reps) * rng.normal(size=len(sigma))
            sample_sd = sigma * np.sqrt(rng.chisquare(df, len(sigma)) / df)
            upper = sample_mean + sample_sd * (t.ppf(.975, df) / np.sqrt(reps)
                    + norm.ppf(.95) * np.sqrt(df / chi2.ppf(.025, df)))
            self.assertAlmostEqual(predicted, float(np.mean(upper <= 0)), delta=.006)

    def test_original_sparse_capacity_erases_source_at_initial_budget(self):
        rng = np.random.default_rng(9772)
        X = rng.normal(size=(8, 68))
        y = X[:, :3].sum(axis=1)
        pip = np.r_[np.full(3, .95), np.full(65, .05)]
        args = (X, y, np.full(8, .001), [(i,) for i in range(8)])
        old = AdaptiveSpikeSlabPosterior(pip, np.ones(68)).fit(
            *args, deviation_variance=.005).diagnostics()
        np.testing.assert_allclose(old["posterior_pip"], .05, atol=1e-12)
        new = AdaptiveSpikeSlabPosterior(pip, np.ones(68), max_effective_fraction=1.,
                    multiplicity_correction=0.).fit(*args, deviation_variance=.005).diagnostics()
        self.assertGreater(new["max_effective_dimension"], sum(pip))
        self.assertGreater(max(new["posterior_pip"]), .5)

    def test_mixture_keeps_between_model_uncertainty_and_updates_each_once(self):
        rng = np.random.default_rng(9773)
        psi = rng.uniform(-2, 2, (16, 8))
        basis = FixedFunctionBasis(psi)
        priors = [SourceFunctionPrior(basis, np.r_[intercept, np.zeros(68)],
                                     np.eye(69) * .1, .02, {}) for intercept in (-1., 1.)]
        mixture = TransferMixtureGPR(priors, psi)
        np.testing.assert_allclose(mixture.C[0, 0], 1.1)
        mixture.update([0], 1., .01)
        self.assertGreater(mixture.diagnostics()["full_dictionary_weight"], .5)
        self.assertTrue(all(len(m.sampled_set) == 1 for m in mixture.models))
        self.assertEqual(mixture.diagnostics()["paid_observations"], 1)

    def test_source_covariance_learning_detects_signed_shared_functions(self):
        rng = np.random.default_rng(9774)
        public = rng.uniform(-2, 2, (96, 8))
        basis = FixedFunctionBasis(public)
        support = [5, 11, 34]
        batches = []
        for task in range(8):
            psi = rng.uniform(-2, 2, (64, 8))
            y = basis.transform(psi)[:, support] @ rng.choice([-1., 1.], 3)
            y += rng.normal(0, .1, 64)
            batches.append(SourceDomainBatch(str(task), psi, y))
        prior = fit_source_risk_ard(batches, [np.full(64, .01)] * 8, public)
        self.assertTrue(set(support).issubset(prior.diagnostics["source_selected_indices"]))
        np.testing.assert_array_equal(prior.mean[1:], 0.)

    def test_joint_kg_changes_no_live_posterior(self):
        rng = np.random.default_rng(9775)
        psi = rng.uniform(-2, 2, (8, 8))
        basis = FixedFunctionBasis(psi, [0, 8, 32])
        prior = SourceFunctionPrior(basis, np.zeros(4), np.eye(4), .02, {})
        models = [prior.target_model(psi) for _ in range(3)]
        for i in range(4):
            for head, model in enumerate(models):
                model.update([i], [0., -.6, -2.5][head], .05)
        states = [copy.deepcopy(m.__dict__) for m in models]
        scores = verification_kg_scores(models, np.arange(8)[:, None], PROTOCOL)
        self.assertTrue(np.isfinite(scores).all())
        for model, state in zip(models, states):
            np.testing.assert_array_equal(model.a, state["a"])
            np.testing.assert_array_equal(model.C, state["C"])

    def test_action_blocks_match_dense_fantasies_across_a_block_boundary(self):
        rng = np.random.default_rng(9778)
        psi = rng.uniform(-2, 2, (35, 8))
        basis = FixedFunctionBasis(psi, [0, 8, 32])
        models = [SourceFunctionPrior(basis, np.r_[value, np.zeros(3)],
                  np.eye(4) * .1, .02, {}).target_model(psi) for value in (4.6, -.6, -2.5)]
        candidates = np.arange(len(psi))[:, None]
        mean = np.column_stack([m.posterior_mean_many(candidates) for m in models])
        variance = np.column_stack([m.posterior_var_many(candidates) for m in models])
        from scipy.special import polygamma
        noise = np.column_stack([np.full(len(psi), .12**2 / 12),
            np.exp(mean[:, 2] + .5*variance[:, 2]) / 12,
            np.full(len(psi), polygamma(1, 5.5))])
        slope = posterior_covariances(models, candidates) / np.sqrt(variance + noise)[None]
        nodes, weights = hermgauss(3)
        triples = list(itertools.product(range(3), repeat=3))
        normals = np.array([[np.sqrt(2)*nodes[j] for j in triple] for triple in triples])
        joint_weights = np.array([np.prod([weights[j]/np.sqrt(np.pi) for j in triple]) for triple in triples])
        future_mean = mean[:, None, None] + slope[:, :, None]*normals[None, None]
        future_variance = np.maximum(variance[:, None] - slope**2, 0)
        # Objective fantasies cross the quality clipping boundary. The maximum
        # must stay inside the joint expectation after factorizing P and Q.
        self.assertGreater(float(future_mean[..., 0].max()), 5.)
        for reps in (256, 928):
            protocol = {**PROTOCOL, "verification_replicates": reps}
            future, _ = verification_utility(future_mean, future_variance[:, :, None], protocol)
            current, _ = verification_utility(mean, variance, protocol)
            dense = np.max(future, axis=0) @ joint_weights - np.max(current)
            np.testing.assert_allclose(verification_kg_scores(models, candidates, protocol), dense, atol=1e-12, rtol=1e-10)

    def test_risk_only_kg_matches_joint_kg_when_means_are_known(self):
        rng = np.random.default_rng(9776)
        psi = rng.uniform(-2, 2, (8, 8))
        basis = FixedFunctionBasis(psi, [0, 8, 32])
        known = [SourceFunctionPrior(basis, np.r_[value, np.zeros(3)],
                                    np.zeros((4, 4)), 0, {}).target_model(psi)
                 for value in (.2, -.6)]
        risk = SourceFunctionPrior(basis, np.r_[-2.5, np.zeros(3)], np.eye(4), .02, {}).target_model(psi)
        for i in range(4):
            risk.update([i], -2.5 + .1 * i, .2)
        candidates = np.arange(8)[:, None]
        mean = np.column_stack([m.posterior_mean_many(candidates) for m in known + [risk]])
        variance = np.column_stack([m.posterior_var_many(candidates) for m in known + [risk]])
        old = verification_kg_scores(known + [risk], candidates, PROTOCOL)
        actual = risk_only_kg_scores(risk, candidates, mean, variance, PROTOCOL)
        np.testing.assert_allclose(actual, old, rtol=1e-10, atol=1e-12)

    def test_expanding_candidates_keeps_the_prior_coordinate_system(self):
        rng = np.random.default_rng(9777)
        psi = rng.uniform(-2, 2, (96, 8))
        reference = psi[:24]
        priors = [SourceFunctionPrior(FixedFunctionBasis(reference, support),
                  np.r_[-2.5, np.zeros(len(support))], np.eye(len(support)+1)*.1, .02, {})
                  for support in ([0, 8, 32], list(range(68)))]
        small = TransferMixtureGPR(priors, reference)
        large = TransferMixtureGPR(priors, psi, reference)
        for i in range(8):
            small.update([i], -2.5 + .1*i, .2)
            large.update([i], -2.5 + .1*i, .2)
        prefix = np.arange(24)[:, None]
        np.testing.assert_allclose(small.posterior_mean_many(prefix), large.posterior_mean_many(prefix))
        np.testing.assert_allclose(small.posterior_var_many(prefix), large.posterior_var_many(prefix))


if __name__ == "__main__":
    unittest.main()
