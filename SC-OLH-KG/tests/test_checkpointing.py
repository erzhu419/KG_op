import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig  # noqa: E402
from problems.rzdt import RZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402


class CheckpointingTests(unittest.TestCase):
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
