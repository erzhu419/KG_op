import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import nct, norm, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.guarded_deployment_kg import (
    conditional_comparison_power, conditional_safety_power, deployment_values,
    guarded_raw_kg, posterior_draws, shortlist_outcomes, _completion)
from core.terminal_verification import verify_frozen_shortlist
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


PROTOCOL = {"objective_reference": 1.5, "candidate_budgets": [80, 128, 128],
            "familywise_delta": .05, "chance_alpha": .05,
            "comparison_budget": 8, "comparison_delta": .05}


class GaussianTerminalProblem:
    simulation_noise_model = "iid_gaussian"
    alpha, tau = .05, 0.
    means = np.array([[.60, -.21], [.62, -.225], [.70, -.25]])
    sigmas = np.array([[.05, .10], [.10, .10], [.05, .10]])

    def simulate(self, point, rng):
        i = point[0]
        return self.means[i] + self.sigmas[i] * rng.normal(size=2)


def models():
    psi = np.array([[-1.], [0.], [1.]])
    basis = FixedFunctionBasis(psi)
    def prior(intercept):
        return SourceFunctionPrior(basis, np.r_[intercept, np.zeros(basis.feature_dim)],
            np.diag(np.r_[.02, np.full(basis.feature_dim, .0008)]), .002, {})
    return (JointFunctionRiskPosterior([prior(.5)], [prior(-5.)], psi),
            JointFunctionRiskPosterior([prior(-.3)], [prior(-4.)], psi))


class GuardedDeploymentTests(unittest.TestCase):
    def test_conditional_powers_match_noncentral_t_laws(self):
        g = np.array([-.16, -.21, -.25, -.3, .1])
        sigma = np.full(5, .1)
        for n in [80, 128]:
            factor = nct.ppf(1 - .05 / 3, n-1, norm.ppf(.95) * np.sqrt(n))
            expected = nct.sf(factor, n-1, -g * np.sqrt(n) / sigma)
            np.testing.assert_allclose(conditional_safety_power(g, sigma, n, .05, .05/3), expected, atol=2e-5)
        difference = np.array([-.08, -.02, 0., .04])
        expected = nct.sf(t.ppf(.95, 7), 7, -difference * np.sqrt(8) / .05)
        np.testing.assert_allclose(conditional_comparison_power(difference, np.full(4, .05), 8, .05), expected, atol=2e-5)
        np.testing.assert_array_equal(conditional_comparison_power(difference, np.zeros(4), 8, .05), [1,1,0,0])

    def test_full_branch_reward_and_cost_match_actual_verifier(self):
        problem = GaussianTerminalProblem()
        draw = np.column_stack([problem.means[:,0], np.log(problem.sigmas[:,0]**2),
                                problem.means[:,1], np.log(problem.sigmas[:,1]**2)])[None]
        predicted, expected_cost = deployment_values(draw, np.array([[0,1,2]]), np.array([1]), PROTOCOL)
        shortlist = [{"point": [i]} for i in range(3)]
        rewards, costs = [], []
        for seed in range(600):
            deployed, result = verify_frozen_shortlist(problem, shortlist, seed=105200+seed,
                search_evaluation_count=10, candidate_budgets=PROTOCOL["candidate_budgets"],
                familywise_delta=.05, objective_incumbent_position=2,
                objective_comparison_budget=8, objective_comparison_delta=.05)
            rewards.append(float(result["certified"]) * (1-problem.means[deployed[0],0]/1.5))
            costs.append(result["verification_budget"])
        self.assertAlmostEqual(float(predicted[0]), float(np.mean(rewards)), delta=.035)
        self.assertAlmostEqual(float(expected_cost[0]), float(np.mean(costs)), delta=10.)

    def test_same_primary_incumbent_uses_sequential_fallback_cost(self):
        q, p = np.array([.8,.7,.6]), np.array([.2,.4,.9])
        reward, cost = shortlist_outcomes(q,p,0.,[80,128,128],0,8)
        self.assertAlmostEqual(float(reward), .2*.8+.8*.4*.7+.8*.6*.9*.6)
        self.assertAlmostEqual(float(cost), 80+.8*128+.8*.6*128)

    def test_fantasy_completion_matches_paid_single_episode_update(self):
        f, g = models()
        draws = posterior_draws(f,g,15,105211)[...,2:]
        before, covariance = g.predict(full_cov=True)
        transported = _completion(g,draws,1,-1.1)
        g.update_batch(1,-1.1,None,1)
        after, C = g.predict(full_cov=True)
        np.testing.assert_allclose(transported.mean(axis=0),after,atol=.001)
        np.testing.assert_allclose(np.cov(transported.reshape(len(draws),-1),rowvar=False),C.reshape(6,6),atol=.0003)
        self.assertGreater(abs(after[1,1]-before[1,1]),.001)

    def test_single_terminal_action_has_zero_kg_without_updating_paid_state(self):
        f,g=models()
        before=g.predict(full_cov=True)
        result=guarded_raw_kg(f,g,np.array([[0,1,2]]),np.array([1]),PROTOCOL,
                              draw_power=4,fantasy_power=2,seed=105212)
        np.testing.assert_array_equal(result["kg"],[0.])
        self.assertEqual(g.paid_episodes,0)
        np.testing.assert_array_equal(g.predict(full_cov=True)[0],before[0])


if __name__ == "__main__":
    unittest.main()
