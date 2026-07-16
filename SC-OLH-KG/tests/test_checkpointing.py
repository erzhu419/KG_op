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

    def test_adaptive_replication_and_dominance_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "dominance_ckpt"
            base = dict(
                n0=4,
                K1=3,
                K2=0,
                decision_backend="exact_kg",
                acquisition_mode="exact_mc",
                exact_kg_mc_samples=1,
                exact_kg_terminal_mode="bayes_risk_dominance",
                adaptive_replication_voi=True,
                replication_candidate_count=2,
                posterior_dominance_enabled=True,
                posterior_dominance_delta=0.10,
                finalist_replication_budget=0,
                certification_recheck_top_k=0,
                eval_pool_size=8,
                evaluate_interval=0,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_resume=True,
                checkpoint_interval=1,
                seed=228,
            )
            first = SingleOLHKGAlgorithm(
                ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
                SingleOLHKGConfig(N=5, **base),
            )
            first_result = first.run()
            first_history = first_result["posterior_dominance"]["history"]
            self.assertEqual(len(first_history), 2)
            self.assertIsNotNone(
                first_result["posterior_dominance"]["incumbent"])

            resumed = SingleOLHKGAlgorithm(
                ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
                SingleOLHKGConfig(N=6, **base),
            )
            resumed_result = resumed.run()
            resumed_history = resumed_result[
                "posterior_dominance"]["history"]
            self.assertEqual(resumed_result["n_simulations"], 6)
            self.assertEqual(len(resumed_history), 3)
            self.assertEqual(resumed_history[:2], first_history)
            self.assertTrue(
                resumed_result["adaptive_replication_voi"]["unified_exact_voi"])
            self.assertFalse(
                resumed_result["posterior_dominance"]["target_oracle_used"])

    def test_certification_recheck_consumes_budget_and_resumes_once_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "recheck_ckpt"
            base = dict(
                n0=4,
                K1=4,
                K2=0,
                acquisition_mode="additive",
                exact_kg_mc_samples=0,
                eval_pool_size=8,
                evaluate_interval=0,
                certification_recheck_top_k=1,
                certification_recheck_min_replicates=3,
                certification_recheck_soft_margin_scale=1e6,
                observed_incumbent_use_replicate_variance=True,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_resume=True,
                checkpoint_interval=1,
                seed=212,
            )
            first = SingleOLHKGAlgorithm(
                ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
                SingleOLHKGConfig(N=6, **base),
            )
            first_result = first.run()
            self.assertEqual(first_result["n_simulations"], 6)
            self.assertEqual(
                [row["selection_policy"] for row in first.iteration_log],
                ["certification_recheck", "certification_recheck"],
            )
            targets = first_result["task_initial_design"][
                "certification_recheck"]["targets"]
            self.assertEqual(len(targets), 1)
            target = tuple(targets[0])
            self.assertEqual(len(first.observations[target]), 3)
            self.assertEqual(
                first_result["candidate_source_counts"][
                    "certification_recheck"],
                2,
            )

            resumed = SingleOLHKGAlgorithm(
                ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
                SingleOLHKGConfig(N=7, **base),
            )
            resumed_result = resumed.run()
            self.assertEqual(resumed_result["n_simulations"], 7)
            self.assertEqual(len(resumed.observations[target]), 3)
            self.assertEqual(
                resumed.iteration_log[-1]["selection_policy"],
                "acquisition",
            )

    def test_finalist_replication_is_in_budget_frozen_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "finalist_ckpt"
            base = dict(
                n0=4,
                K1=4,
                K2=0,
                acquisition_mode="exact_mc",
                exact_kg_mc_samples=1,
                eval_pool_size=8,
                evaluate_interval=0,
                finalist_replication_budget=2,
                finalist_replication_count=1,
                finalist_replication_min_replicates=100,
                finalist_replication_delta=0.05,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_resume=True,
                checkpoint_interval=1,
                seed=214,
            )
            first = SingleOLHKGAlgorithm(
                ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
                SingleOLHKGConfig(N=7, **base),
            )
            first_result = first.run()
            self.assertEqual(first_result["n_simulations"], 7)
            self.assertEqual(
                [row["selection_policy"] for row in first.iteration_log[-2:]],
                ["finalist_replication", "finalist_replication"],
            )
            self.assertTrue(all(
                row["exact_kg_skipped_reason"]
                == "forced_finalist_replication"
                for row in first.iteration_log[-2:]
            ))
            summary = first_result["finalist_replication"]
            self.assertTrue(summary["initialized"])
            self.assertEqual(summary["frozen_stage"], 5)
            self.assertEqual(summary["forced_evaluations"], 2)
            self.assertFalse(summary["target_oracle_used"])
            frozen_target = tuple(summary["targets"][0])
            first_count = len(first.observations[frozen_target])

            resumed = SingleOLHKGAlgorithm(
                ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
                SingleOLHKGConfig(N=8, **base),
            )
            resumed_result = resumed.run()
            self.assertEqual(resumed_result["n_simulations"], 8)
            self.assertEqual(
                tuple(resumed_result["finalist_replication"]["targets"][0]),
                frozen_target,
            )
            self.assertEqual(
                len(resumed.observations[frozen_target]), first_count + 1)
            self.assertEqual(
                resumed.iteration_log[-1]["selection_policy"],
                "finalist_replication",
            )

    def test_replicated_finalist_fallback_is_safety_first_not_certificate(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                n0=4,
                K1=4,
                K2=0,
                recommendation_slack_initial=1e6,
                finalist_replication_budget=2,
                finalist_replication_count=2,
                finalist_replication_min_replicates=2,
                finalist_replication_delta=0.5,
                finalist_replication_variance_prior_df=0.0,
                seed=215,
            ),
        )
        samples = algorithm._initial_samples()
        algorithm._fit_initial_belief(samples)
        unsafe = (0, 0, 0)
        safe = (20, 20, 20)
        algorithm._finalist_replication_initialized = True
        algorithm._finalist_replication_targets = [unsafe, safe]
        algorithm._finalist_replication_labels = [
            "minimum_bayes_risk",
            "minimum_nominal_expected_violation",
        ]
        algorithm.observations[unsafe] = [
            np.asarray([0.1, 0.2]),
            np.asarray([0.1, 0.2]),
        ]
        algorithm.observations[safe] = [
            np.asarray([0.5, -1.0]),
            np.asarray([0.5, -1.0]),
        ]
        selected, details = algorithm._solve_posterior_recommendation(
            pool=[unsafe, safe])
        self.assertEqual(selected, safe)
        self.assertTrue(details["replicated_finalist_used"])
        self.assertTrue(
            details["replicated_finalist_empirical_certificate"])
        self.assertFalse(details["posterior_feasible"])
        self.assertFalse(
            details["replicated_finalist_target_oracle_used"])
        self.assertLessEqual(
            details["replicated_finalist_selected"]["upper_chance_margin"],
            0.0,
        )

    def test_two_stage_contract_keeps_fallback_uncertified(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=20,
                n0=4,
                acquisition_mode="exact_mc",
                finalist_replication_budget=3,
                finalist_replication_fixed_universe=True,
                seed=219,
            ),
        )
        frozen = [(0, 0, 0), (1, 1, 1)]
        algorithm._finalist_replication_pool = list(frozen)
        algorithm._finalist_replication_targets = list(frozen)
        summary = {
            "initialized": True,
            "frozen_stage": 17,
            "fixed_universe": True,
            "forced_evaluations": 3,
            "target_oracle_used": False,
            "mathematically_closed": False,
        }
        recommendation = {
            "posterior_feasible": False,
            "replicated_finalist_used": True,
            "replicated_finalist_empirical_certificate": False,
            "replicated_finalist_reason": "minimum_replicated_upper_margin",
        }

        contract = algorithm._two_stage_decision_contract_summary(
            recommendation, summary)

        self.assertEqual(
            contract["terminal_status"],
            "uncertified_least_risk_fallback",
        )
        self.assertFalse(contract["fallback_claims_certification"])
        self.assertTrue(contract["implementation_contract_closed"])
        self.assertFalse(contract["global_exact_kg_claim"])
        self.assertEqual(contract["initial_design_budget"], 4)
        self.assertEqual(contract["adaptive_search_budget"], 13)
        self.assertTrue(contract["verification_budget_fully_charged"])
        self.assertTrue(
            contract["adaptive_search_acquisition_configured_as_exact_kg"])

    def test_expert_stratified_nomination_survives_mixture_mass_collapse(self):
        class FakePosterior:
            expert_names = ("dominant", "deleted_but_safe")

        class FakeEnsemble:
            posterior = FakePosterior()

            @staticmethod
            def expert_moments_many(output_index, X, certification=True):
                del output_index, certification
                self.assertEqual(len(X), 3)
                means = np.asarray([
                    [0.10, 0.20, 0.30],
                    [0.30, 0.20, -0.50],
                ])
                epistemic = np.full_like(means, 1e-6)
                aleatoric = np.full_like(means, 1e-6)
                return means, epistemic, aleatoric

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=10,
                n0=4,
                finalist_replication_budget=2,
                finalist_replication_count=2,
                finalist_replication_expert_stratified=True,
                seed=216,
            ),
        )
        algorithm.task_ensemble = FakeEnsemble()
        algorithm._terminal_bayes_risk_components = lambda *args, **kwargs: {
            "risk": np.asarray([2.0, 0.0, 1.0]),
            "nominal_expected_violation": np.asarray([0.2, 0.1, 0.3]),
            "expected_violation": np.asarray([0.3, 0.2, 0.4]),
            "model_disagreement": np.asarray([0.1, 0.2, 0.3]),
        }
        pool = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        info = algorithm._initialize_finalist_replication_targets(8, pool)
        self.assertEqual(
            algorithm._finalist_replication_targets,
            [pool[1], pool[2]],
        )
        self.assertEqual(
            algorithm._finalist_replication_labels,
            ["minimum_bayes_risk",
             "expert_safety_nomination:deleted_but_safe"],
        )
        self.assertTrue(info["expert_stratified"])
        self.assertEqual(
            info["frozen_metrics"][1]["source"], "expert_stratified")
        self.assertFalse(info["target_oracle_used"])

    def test_adaptive_expert_race_refreshes_and_ignores_incomplete_archive(self):
        class FakePosterior:
            expert_names = ("dominant", "switching")

        class FakeEnsemble:
            posterior = FakePosterior()
            mode = 0

            def expert_moments_many(self, output_index, X, certification=True):
                del output_index, certification
                assert len(X) == 3
                if self.mode == 0:
                    means = np.asarray([
                        [0.10, 0.20, 0.30],
                        [0.30, 0.20, -0.50],
                    ])
                else:
                    means = np.asarray([
                        [-0.60, 0.20, 0.30],
                        [0.30, 0.20, 0.10],
                    ])
                epistemic = np.full_like(means, 1e-6)
                aleatoric = np.full_like(means, 1e-6)
                return means, epistemic, aleatoric

            @staticmethod
            def robust_moments_many(output_index, X, certification=True):
                del output_index, certification

                class Moments:
                    aleatoric_upper = np.full(len(X), 1e-4)

                return Moments()

        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=10,
                n0=4,
                finalist_replication_budget=2,
                finalist_replication_count=2,
                finalist_replication_min_replicates=2,
                finalist_replication_expert_stratified=True,
                finalist_replication_adaptive_race=True,
                finalist_replication_fixed_universe=True,
                seed=217,
            ),
        )
        ensemble = FakeEnsemble()
        algorithm.task_ensemble = ensemble
        algorithm._terminal_bayes_risk_components = lambda *args, **kwargs: {
            "risk": np.asarray([2.0, 0.0, 1.0]),
            "nominal_expected_violation": np.asarray([0.2, 0.1, 0.3]),
            "expected_violation": np.asarray([0.3, 0.2, 0.4]),
            "model_disagreement": np.asarray([0.1, 0.2, 0.3]),
        }
        pool = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]

        first, first_info = algorithm._finalist_replication_candidate(8, pool)
        self.assertEqual(first, pool[2])
        self.assertTrue(first_info["adaptive_race"])
        algorithm.observations[pool[2]] = [np.asarray([0.5, 0.0])]

        ensemble.mode = 1
        refreshed, refreshed_info = algorithm._finalist_replication_candidate(
            9, [(9, 9, 9)])
        self.assertEqual(refreshed, pool[0])
        self.assertEqual(
            refreshed_info["status"],
            "forced_adaptive_finalist_replication",
        )
        self.assertEqual(
            algorithm._finalist_replication_targets,
            [pool[1], pool[2], pool[0]],
        )
        self.assertEqual(len(
            algorithm._finalist_replication_refresh_history), 2)
        self.assertEqual(algorithm._finalist_replication_pool, pool)

        algorithm.observations[pool[0]] = [
            np.asarray([0.4, -10.0]),
            np.asarray([0.4, -10.0]),
        ]
        algorithm.observations[pool[1]] = [np.asarray([0.1, 0.0])]
        selected, details = (
            algorithm._replicated_finalist_recommendation_index(pool))
        self.assertEqual(selected, 0)
        self.assertTrue(details["replicated_finalist_used"])
        self.assertTrue(details["replicated_finalist_adaptive_race"])
        self.assertEqual(
            len(details["replicated_finalist_incomplete_rows"]), 2)
        self.assertEqual(
            details["replicated_finalist_selected"][
                "familywise_multiplicity"],
            4,
        )

    def test_adaptive_finalist_race_checkpoint_state_round_trips(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        config = SingleOLHKGConfig(
            N=8,
            n0=4,
            finalist_replication_budget=2,
            finalist_replication_adaptive_race=True,
            seed=218,
        )
        algorithm = SingleOLHKGAlgorithm(problem, config)
        target = (1, 2, 3)
        algorithm._finalist_replication_initialized = True
        algorithm._finalist_replication_targets = [target]
        algorithm._finalist_replication_labels = ["adaptive"]
        algorithm._finalist_replication_frozen_stage = 6
        algorithm._finalist_replication_active_target = target
        algorithm._finalist_replication_active_label = "adaptive"
        algorithm._finalist_replication_refresh_history = [{
            "stage": 6,
            "target": list(target),
            "target_oracle_used": False,
        }]
        algorithm._finalist_replication_pool = [target, (4, 5, 6)]
        payload = algorithm._runtime_checkpoint_payload(7, "test")

        restored = SingleOLHKGAlgorithm(problem, config)
        restored._load_checkpoint_payload(payload)
        self.assertEqual(restored._finalist_replication_targets, [target])
        self.assertEqual(
            restored._finalist_replication_active_target, target)
        self.assertEqual(
            restored._finalist_replication_active_label, "adaptive")
        self.assertEqual(
            restored._finalist_replication_refresh_history,
            algorithm._finalist_replication_refresh_history,
        )
        self.assertEqual(
            restored._finalist_replication_pool,
            algorithm._finalist_replication_pool,
        )

    def test_certified_only_finalist_never_uses_uncertified_subset(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        unsafe = (0, 0, 0)
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                finalist_replication_budget=2,
                finalist_replication_min_replicates=2,
                finalist_replication_delta=0.5,
                finalist_replication_variance_prior_df=0.0,
                finalist_empirical_override="certified_only",
                seed=219,
            ),
        )
        algorithm._finalist_replication_initialized = True
        algorithm._finalist_replication_targets = [unsafe]
        algorithm._finalist_replication_labels = ["minimum_bayes_risk"]
        algorithm.observations[unsafe] = [
            np.asarray([0.1, 1.0]),
            np.asarray([0.1, 1.0]),
        ]
        selected, details = (
            algorithm._replicated_finalist_recommendation_index([unsafe]))
        self.assertIsNone(selected)
        self.assertFalse(details["replicated_finalist_used"])
        self.assertEqual(
            details["replicated_finalist_reason"],
            "no_empirically_certified_finalist",
        )
        self.assertEqual(
            details["replicated_finalist_override_policy"],
            "certified_only",
        )

    def test_posterior_only_disables_empirical_finalist_override(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(finalist_empirical_override="off", seed=220),
        )
        selected, details = (
            algorithm._replicated_finalist_recommendation_index(
                [(0, 0, 0)]))
        self.assertIsNone(selected)
        self.assertEqual(
            details["replicated_finalist_reason"],
            "empirical_override_disabled",
        )

    def test_commit_before_switch_waits_for_active_arm(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        active = (1, 2, 3)
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=10,
                n0=4,
                finalist_replication_budget=3,
                finalist_replication_min_replicates=2,
                finalist_replication_adaptive_race=True,
                finalist_replication_policy="commit_before_switch",
                seed=221,
            ),
        )
        algorithm._finalist_replication_initialized = True
        algorithm._finalist_replication_active_target = active
        algorithm._finalist_replication_active_label = "active"
        algorithm._finalist_replication_targets = [active]
        algorithm._finalist_replication_labels = ["active"]
        algorithm.observations[active] = [np.asarray([0.0, -0.1])]
        calls = []
        algorithm._refresh_finalist_replication_targets = (
            lambda stage, pool: calls.append((stage, pool)) or {
                "status": "refreshed",
                "target_oracle_used": False,
            }
        )
        info = algorithm._initialize_finalist_replication_targets(
            7, [active])
        self.assertEqual(info["status"], "active_target_commit_incomplete")
        self.assertEqual(calls, [])
        algorithm.observations[active].append(np.asarray([0.0, -0.1]))
        info = algorithm._initialize_finalist_replication_targets(
            8, [active])
        self.assertEqual(info["status"], "refreshed")
        self.assertEqual(len(calls), 1)

    def test_observed_safety_frontier_completes_charged_challenger(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        safe = (1, 2, 3)
        bayes = (4, 5, 6)
        other = (7, 8, 9)
        pool = [safe, bayes, other]
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=10,
                n0=4,
                finalist_replication_budget=3,
                finalist_replication_count=2,
                finalist_replication_min_replicates=2,
                finalist_replication_adaptive_race=True,
                finalist_replication_fixed_universe=True,
                finalist_replication_policy="commit_before_switch",
                finalist_frontier_policy="observed_safety_reserved",
                seed=222,
            ),
        )
        algorithm.observations[safe] = [np.asarray([0.4, -1.0])]
        algorithm.observations[bayes] = [np.asarray([0.1, 0.2])]
        algorithm.observations[other] = [np.asarray([0.2, 0.1])]
        algorithm._terminal_bayes_risk_components = lambda *args, **kwargs: {
            "risk": np.asarray([2.0, 0.0, 1.0]),
            "nominal_expected_violation": np.asarray([0.2, 0.1, 0.3]),
            "expected_violation": np.asarray([0.3, 0.2, 0.4]),
            "model_disagreement": np.asarray([0.1, 0.2, 0.3]),
        }
        algorithm.variance_model.predict_certification_variance = (
            lambda *args, **kwargs: 1e-4)

        selected, info = algorithm._finalist_replication_candidate(7, pool)
        self.assertEqual(selected, safe)
        self.assertEqual(
            algorithm._finalist_replication_targets, [bayes, safe])
        self.assertEqual(
            algorithm._finalist_replication_active_label,
            "observed_safety_rank_1",
        )
        self.assertEqual(info["status"], "forced_adaptive_finalist_replication")
        algorithm.observations[safe].append(np.asarray([0.4, -1.0]))
        selected, _ = algorithm._finalist_replication_candidate(8, pool)
        self.assertEqual(selected, bayes)

        index, details = algorithm._replicated_finalist_recommendation_index(pool)
        self.assertEqual(index, 0)
        self.assertTrue(details["replicated_finalist_used"])
        self.assertFalse(details["replicated_finalist_target_oracle_used"])

    def test_top_two_observed_safety_arms_commit_before_expert_refresh(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        bayes_and_safest = (1, 2, 3)
        second_safest = (4, 5, 6)
        other = (7, 8, 9)
        pool = [bayes_and_safest, second_safest, other]
        problem.frozen_source_coverage_candidates = lambda n=0: list(pool)
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=10,
                n0=4,
                task_posterior_mandatory_universal_count=3,
                finalist_replication_budget=3,
                finalist_replication_count=2,
                finalist_observed_safety_count=2,
                finalist_replication_min_replicates=2,
                finalist_replication_adaptive_race=True,
                finalist_replication_fixed_universe=True,
                finalist_replication_policy="commit_before_switch",
                finalist_frontier_policy="observed_safety_reserved",
                seed=224,
            ),
        )
        algorithm.observations[bayes_and_safest] = [
            np.asarray([0.2, -1.0])]
        algorithm.observations[second_safest] = [
            np.asarray([0.3, -0.5])]
        algorithm.observations[other] = [np.asarray([0.1, 0.2])]
        algorithm._terminal_bayes_risk_components = lambda *args, **kwargs: {
            "risk": np.asarray([0.0, 1.0, 2.0]),
            "nominal_expected_violation": np.asarray([0.1, 0.2, 0.3]),
            "expected_violation": np.asarray([0.2, 0.3, 0.4]),
            "model_disagreement": np.asarray([0.1, 0.2, 0.3]),
        }
        refresh_calls = []

        def refresh(stage, candidates):
            refresh_calls.append((stage, candidates))
            return {"status": "refreshed", "target_oracle_used": False}

        algorithm._refresh_finalist_replication_targets = refresh

        selected, _ = algorithm._finalist_replication_candidate(7, pool)
        self.assertEqual(selected, bayes_and_safest)
        self.assertEqual(
            algorithm._finalist_replication_labels[0],
            "minimum_bayes_risk+observed_safety_rank_1",
        )
        self.assertEqual(refresh_calls, [])
        algorithm.observations[bayes_and_safest].append(
            np.asarray([0.2, -1.0]))

        selected, info = algorithm._finalist_replication_candidate(8, pool)
        self.assertEqual(selected, second_safest)
        self.assertEqual(info["status"], "forced_adaptive_finalist_replication")
        self.assertEqual(info["label"], "observed_safety_rank_2")
        self.assertEqual(refresh_calls, [])
        algorithm.observations[second_safest].append(
            np.asarray([0.3, -0.5]))

        algorithm._finalist_replication_candidate(9, pool)
        self.assertEqual(len(refresh_calls), 1)

    def test_observed_safety_prefers_frozen_source_coverage_scope(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        protected = (1, 2, 3)
        noisy_random = (4, 5, 6)
        problem.frozen_source_coverage_candidates = (
            lambda n=0: [protected] if n > 0 else [])
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=10,
                n0=4,
                task_posterior_mandatory_universal_count=2,
                seed=223,
            ),
        )
        algorithm.observations[protected] = [np.asarray([0.4, -0.2])]
        algorithm.observations[noisy_random] = [np.asarray([0.2, -1.0])]
        challenger = algorithm._observed_safety_challenger()
        self.assertEqual(challenger["x"], protected)
        self.assertEqual(
            challenger["selection_scope"], "frozen_source_coverage")
        self.assertEqual(challenger["protected_candidate_count"], 1)

    def test_terminal_depth3_policy_uses_remaining_horizon(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        arms = [(1, 2, 3), (4, 5, 6)]
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                N=10,
                n0=4,
                finalist_replication_budget=3,
                finalist_replication_policy="terminal_kg_depth3",
                finalist_terminal_max_arms=2,
                seed=222,
            ),
        )
        algorithm._finalist_replication_initialized = True
        algorithm._finalist_replication_targets = list(arms)
        algorithm._finalist_replication_labels = ["a", "b"]
        calls = []

        def fake_terminal(choices, pool, *, depth, stage):
            calls.append((list(choices), list(pool), depth, stage))
            return choices[1], {
                "terminal_kg_selected_gain": 0.5,
                "terminal_kg_depth": depth,
            }

        algorithm._terminal_replication_kg_candidate = fake_terminal
        selected, info = algorithm._finalist_replication_candidate(7, arms)
        self.assertEqual(selected, arms[1])
        self.assertEqual(info["terminal_kg_depth"], 3)
        selected, info = algorithm._finalist_replication_candidate(9, arms)
        self.assertEqual(selected, arms[1])
        self.assertEqual(info["terminal_kg_depth"], 1)
        self.assertEqual([call[2] for call in calls], [3, 1])

    def test_observed_incumbent_uses_shrunk_replicate_variance(self):
        problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
        algorithm = SingleOLHKGAlgorithm(
            problem,
            SingleOLHKGConfig(
                certification_recheck_min_replicates=3,
                certification_recheck_variance_prior_df=2.0,
                observed_incumbent_use_replicate_variance=True,
                observed_incumbent_margin_scale=1e6,
                seed=213,
            ),
        )
        x = (1, 2, 3)
        constraints = np.asarray([-0.20, -0.10, -0.15], dtype=float)
        algorithm.observations = {
            x: [np.asarray([1.0, value], dtype=float) for value in constraints]
        }
        incumbent = algorithm._observed_nominal_incumbent()
        sample_var = float(np.var(constraints, ddof=1))
        expected_sigma = np.sqrt(
            (2.0 * sample_var + 2.0 * problem.sigma_level ** 2) / 4.0)
        self.assertEqual(
            incumbent["empirical_sigma_source"], "replicate_shrinkage")
        self.assertEqual(incumbent["replicate_count"], 3)
        self.assertAlmostEqual(
            incumbent["empirical_sigma"], expected_sigma, places=12)

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
            terminal_pool = list(first._last_terminal_pool)
            self.assertTrue(terminal_pool)
            self.assertTrue((checkpoint_dir / "checkpoint_latest.pkl").exists())
            self.assertLessEqual(
                len(list(checkpoint_dir.glob("checkpoint_stage_*.pkl"))),
                2,
            )

            checkpoint_probe = SingleOLHKGAlgorithm(
                ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03)),
                SingleOLHKGConfig(N=5, **base),
            )
            self.assertEqual(
                checkpoint_probe._try_resume_from_checkpoint(), 5)
            self.assertEqual(
                checkpoint_probe._last_terminal_pool, terminal_pool)

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
