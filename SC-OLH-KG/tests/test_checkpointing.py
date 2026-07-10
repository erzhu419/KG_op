import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from core.gpr import ParametricGPR  # noqa: E402
from problems.rzdt import RZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class _SwitchingBasis:
    feature_dim = 1

    def __init__(self, n0):
        self.n0 = int(n0)
        self.selected_basis = "left"

    def features(self, x):
        value = float(np.asarray(x, dtype=float)[0]) / 20.0
        if self.selected_basis == "right":
            value = 1.0 - value
        return np.asarray([value], dtype=float)

    def features_many(self, rows):
        return np.vstack([self.features(row) for row in rows])

    def fit_from_observations(self, observations, output_index=None):
        if len(observations) > self.n0:
            self.selected_basis = "right"
        return self.selected_basis

    def should_refit_from_observations(self, observations):
        return len(observations) > self.n0 and self.selected_basis != "right"

    def initial_parametric_coefficients(self, phi, target):
        return np.linalg.lstsq(phi, target, rcond=None)[0]

    @staticmethod
    def adaptive_sparsity_spec(observations):
        return None

    @staticmethod
    def apply_coefficient_prior(beta, prior_var):
        return beta, prior_var

    def runtime_state(self):
        return {
            "selected_basis": self.selected_basis,
            "selected_parametric_ridge": 0.0,
            "selected_additive_groups": [],
            "additive_base_basis": "",
            "additive_bank_kind": "",
        }

    def load_runtime_state(self, state):
        self.selected_basis = str(state["selected_basis"])


class CheckpointingTests(unittest.TestCase):
    def test_repeated_observation_matches_rank_one_variance_reduction(self):
        model = ParametricGPR(d=2, lambda_i=0.2, prior_var=3.0)
        x = (4, 7)
        noise = 0.35
        variance_before = model.posterior_var(x)
        mean_before = model.posterior_mean(x)
        model.update(x, mean_before, noise)
        variance_after = model.posterior_var(x)
        expected_reduction = variance_before ** 2 / (variance_before + noise)
        self.assertAlmostEqual(
            variance_before - variance_after,
            expected_reduction,
            places=10,
        )
        self.assertGreaterEqual(variance_after, 0.0)

    def test_replication_candidates_are_posterior_ranked_observed_points(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=6,
                n0=4,
                K1=4,
                K2=0,
                replication_candidate_count=2,
                replication_max_per_solution=2,
                seed=20,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        candidates = algorithm._replication_candidates()
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(row in algorithm.observations for row in candidates))
        blocked = candidates[0]
        algorithm.observations[blocked].append(
            algorithm.observations[blocked][0].copy())
        self.assertNotIn(blocked, algorithm._replication_candidates())

    def test_feature_switch_rebuilds_gpr_by_replaying_history(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=6,
            n0=4,
            K1=4,
            K2=0,
            eval_pool_size=8,
            seed=21,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        basis = _SwitchingBasis(config.n0)
        algorithm.gpr[1] = ParametricGPR(
            problem.d,
            config.lambda_i,
            config.prior_var,
            normalize_func=problem.normalize,
            basis_map=basis,
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        self.assertEqual(basis.selected_basis, "left")

        while True:
            x = problem.sample_random(algorithm.rng)
            if x not in algorithm.observations:
                break
        mu_before = [model.posterior_mean(x) for model in algorithm.gpr]
        epistemic_before = [model.posterior_var(x) for model in algorithm.gpr]
        sigma2_before = [
            algorithm.variance_model.predict_variance(i, x, problem)
            for i in range(2)
        ]
        y = algorithm._simulate_and_store(x)
        for i, model in enumerate(algorithm.gpr):
            model.update(x, y[i], sigma2_before[i])
            algorithm.variance_model.update(
                i,
                x,
                y[i],
                mu_before[i],
                model,
                problem,
                epistemic_var=epistemic_before[i],
            )
        algorithm.iteration_log.append({
            "x_selected": list(x),
            "Y_observed": [float(value) for value in y],
            "sigma2_before": [float(value) for value in sigma2_before],
        })
        old_coefficients = algorithm.gpr[1].a.copy()

        events = algorithm._refresh_sequential_basis()
        event = next(row for row in events if row["output_index"] == 1)
        self.assertEqual(event["before_basis"], "left")
        self.assertEqual(event["after_basis"], "right")
        self.assertTrue(event["changed"])
        self.assertTrue(event["gpr_rebuilt"])
        self.assertEqual(event["replayed_updates"], 1)
        self.assertEqual(event["rebuild_initial_records"], config.n0)
        self.assertFalse(np.array_equal(old_coefficients, algorithm.gpr[1].a))
        self.assertIn(tuple(x), algorithm.gpr[1].sol_to_idx)
        self.assertTrue(np.all(np.isfinite(
            algorithm.gpr[1].posterior_mean_many(samples + [x]))))

    def test_resume_extends_previous_true_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "kg_ckpt"
            base = dict(
                n0=4,
                K1=4,
                K2=0,
                acquisition_mode="exact_mc",
                exact_kg_mc_samples=1,
                exact_kg_jobs=1,
                eval_pool_size=10,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_resume=True,
                checkpoint_interval=1,
                checkpoint_keep_last=2,
                seed=23,
            )
            problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
            first = SingleOLHKGAlgorithm(
                problem,
                SingleOLHKGConfig(N=5, **base),
            )
            first_result = first.run()
            self.assertEqual(first_result["n_simulations"], 5)
            self.assertTrue((checkpoint_dir / "checkpoint_latest.pkl").exists())
            self.assertLessEqual(
                len(list(checkpoint_dir.glob("checkpoint_stage_*.pkl"))),
                2,
            )

            resumed_problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
            resumed = SingleOLHKGAlgorithm(
                resumed_problem,
                SingleOLHKGConfig(N=7, **base),
            )
            resumed_result = resumed.run()
            self.assertEqual(resumed_result["n_simulations"], 7)
            self.assertEqual(len(resumed.history), 7)
            self.assertEqual(len(resumed.iteration_log), 3)
            self.assertEqual(
                [x for x, _ in resumed.history[:5]],
                [x for x, _ in first.history],
            )


if __name__ == "__main__":
    unittest.main()
