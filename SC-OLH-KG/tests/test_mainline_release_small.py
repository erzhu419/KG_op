"""Deterministic release wiring checks; no source restoration or simulation."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


ENTRYPOINT = Path(__file__).resolve().parents[1] / "reproducibility/mainline_release/run_small.py"
SPEC = importlib.util.spec_from_file_location("mainline_release_small", ENTRYPOINT)
small = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(small)


class ReleaseSmallTests(unittest.TestCase):
    def test_build_engine_uses_no_unavailable_source_receipt_fields(self):
        # The retained runtime bundle lacks source_quadrature_nodes; neither
        # that fit-summary field nor a historical call ledger is needed here.
        bundle = dict(identities=["spectral/head_specific/observed_task"], diagnostics={})
        prior = SimpleNamespace(diagnostics=lambda: {})
        config = SimpleNamespace(decision_backend="constrained_ts", llm_prior_enabled=False,
            initial_design_points=[(i,) for i in range(10)])
        benchmark = ModuleType("performance.benchmark_lodo_meta_prior")
        benchmark._source_informed_archive_contract = lambda *_: None
        benchmark.build_scalarized_problem = lambda *_args, **_kwargs: object()
        benchmark.parse_weights = lambda _: (.5, .5)
        benchmark.build_target_problem = lambda *_: None
        benchmark.SingleOLHKGAlgorithm = lambda *_: None
        benchmark.run_one = lambda _: benchmark.SingleOLHKGAlgorithm(None, config)
        performance = ModuleType("performance")
        performance.benchmark_lodo_meta_prior = benchmark
        backend = ModuleType("performance.mainline_role_backend")
        backend.RolePosteriorAlgorithm = lambda *_args, **_kwargs: SimpleNamespace()
        representation = ModuleType("representation")
        meta = ModuleType("representation.meta_prior")
        meta.MetaPriorProblemAdapter = lambda target, *_args, **_kwargs: target
        modules = {module.__name__: module for module in (
            performance, benchmark, backend, representation, meta)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = dict(heldout="InventorySupplyChain", seed=2026120001,
                args=dict(d=1000, N=13, n0=10, L=100, sigma=.04, alpha=.05,
                          weights=".5,.5", meta_proposal_pool_size=1024, meta_refinement_count=128))
            (root / "task.json").write_text(json.dumps(task))
            (root / "initial.json").write_text(json.dumps([
                dict(point=[i], raw=[.1, -.2]) for i in range(10)]))
            for name in ("native.pkl", "source.pkl"):
                (root / name).write_bytes(b"fixture; pickle.load is mocked")
            case = dict(domain="InventorySupplyChain", geometry_task_seed=2026120001,
                observation_seed=2026130001, problem_kwargs={}, resolved_task_file="task.json",
                initial_rows_file="initial.json", native_source_prior_file="native.pkl",
                source_models_file="source.pkl")
            receipt = dict(source_model_restorations=0)
            with patch.dict("sys.modules", modules), patch.object(
                    small.pickle, "load", side_effect=[prior, bundle]):
                engine = small.build_engine(case, root, receipt)
            self.assertIs(engine.backend_bundle, bundle)
            self.assertEqual(receipt["source_model_restorations"], 2)
            self.assertEqual(receipt["source_identities"], bundle["identities"])
            self.assertNotIn("source_quadrature_nodes", receipt)
            self.assertNotIn("source_archive_calls_reused", receipt)

    def exercise(self, root, *, fail=False):
        case = root / "case.json"
        case.write_text(json.dumps(dict(schema="mainline-small-v1", kind="release_acceptance_only",
            domain="registered-fixture", geometry_task_seed=1, observation_seed=2,
            verification_seed=3, source_commit="frozen", shared_neighbors_file="neighbors.json",
            problem_kwargs={})))
        (root / "neighbors.json").write_text(json.dumps(dict(points=[[20]], metadata={"count": 1})))
        output = root / "attempt"
        initial = [((i,), [float(i), 0.]) for i in range(10)]

        def original_simulate(_point, _rng=None):
            if fail:
                raise RuntimeError("retained simulator failure")
            return [0.5, -0.5]

        engine = SimpleNamespace(history=list(initial), replayed_initial_calls=10,
            problem=SimpleNamespace(tau=0., simulate=original_simulate),
            _true_chance_margin=lambda *_: None, _evaluate_recommendation=lambda *_: None)

        def run_search():
            for prefix in (10, 11, 12):
                point = (prefix,)
                raw = engine.problem.simulate(point)
                engine.history.append((point, raw))
                engine.audit["points"].append(point)
                engine.audit["stages"].append(dict(paid_prefix=prefix, selected_id=prefix-10,
                    selected_index=0, native_candidate_count=1, added_candidate_count=1,
                    scored_ids=[0], action_sources=["registered_route"],
                    score=dict(total=[.1], sampled_feasibility_mode="feasible_objective",
                               sampled_feasible_count=1)))
            engine.frozen_terminal = dict(primary=[10], terminal={"frozen": True})

        engine.run_search = run_search

        @contextmanager
        def shared_neighbors(_engine, _points):
            engine.audit = {"points": [], "stages": []}
            yield engine.audit

        def verify(problem, _item):
            # This is the barrier whose regression would leak terminal samples.
            self.assertTrue((output / "frozen_terminal_input.json").is_file())
            self.assertEqual(len(engine.history), 13)
            problem.simulate((10,))
            problem.simulate((11,))
            return (10,), dict(verification_budget=2, max_verification_budget=352, certified=True)

        neighbors_module = ModuleType("performance.mainline_shared_neighbors")
        neighbors_module.shared_neighbor_candidates = shared_neighbors
        verifier_module = ModuleType("performance.run_mainline_terminal_equal")
        verifier_module.run_verifier = verify
        with patch.object(small, "build_engine", return_value=engine), \
                patch.object(small, "terminal_input", return_value={"frozen": True}), \
                patch.object(small, "version", return_value="fixture"), \
                patch.dict("sys.modules", {neighbors_module.__name__: neighbors_module,
                                            verifier_module.__name__: verifier_module}):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "retained simulator failure"):
                    small.run(case, output)
            else:
                small.run(case, output)
        return case, output

    def test_freeze_precedes_independent_verifier_and_actual_calls_close(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output = self.exercise(Path(directory))
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["reused_initial_calls"], 10)
            self.assertEqual(receipt["new_target_search_calls"], 3)
            self.assertEqual(receipt["new_verification_calls"], 2)
            self.assertEqual(receipt["logical_total_calls"], 15)
            self.assertEqual(receipt["new_physical_target_calls"], 5)
            self.assertTrue(receipt["independent_terminal_history_unchanged"])

    def test_failed_call_and_fee_retained_and_same_attempt_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            case, output = self.exercise(Path(directory), fail=True)
            original = (output / "receipt.json").read_text()
            receipt = json.loads(original)
            raw = json.loads((output / "raw_calls.jsonl").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["new_target_search_calls"], 1)
            self.assertEqual(receipt["new_verification_calls"], 0)
            self.assertEqual(receipt["automatic_retries"], 0)
            self.assertGreaterEqual(receipt["cpu_seconds"], 0.)
            self.assertEqual(raw["status"], "failed")
            with self.assertRaises(FileExistsError):
                small.run(case, output)
            self.assertEqual((output / "receipt.json").read_text(), original)


if __name__ == "__main__":
    unittest.main()
