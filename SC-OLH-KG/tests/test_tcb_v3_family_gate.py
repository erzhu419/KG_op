import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from performance.benchmark_tcb_v3_family_gate import (  # noqa: E402
    build_parser,
    run_gate,
)
from performance.summarize_tcb_v3_family_gate import (  # noqa: E402
    _family_audit,
)


SUBMIT_SCRIPT = REPO / "scripts/submit_scolhkg_tcb_v3_family_gate_scheduler.py"
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "tcb_v3_family_submit", SUBMIT_SCRIPT)
SUBMIT = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(SUBMIT)

V32_SUBMIT_SCRIPT = (
    REPO / "scripts/submit_scolhkg_tcb_v32_family_calibration_gate_scheduler.py")
V32_SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "tcb_v32_family_submit", V32_SUBMIT_SCRIPT)
V32_SUBMIT = importlib.util.module_from_spec(V32_SUBMIT_SPEC)
V32_SUBMIT_SPEC.loader.exec_module(V32_SUBMIT)


class TCBV3FamilyGateTests(unittest.TestCase):
    def _tiny_args(self, output):
        args = build_parser().parse_args([
            "--domains",
            (
                "RZDT1,StatePolicyRZDT1,FactorShockStatePolicyRZDT1,"
                "InventorySupplyChain,QueueResourceControl"
            ),
            "--heldout", "InventorySupplyChain",
            "--target-seed", "3",
            "--source-records", "8",
            "--source-replicates", "2",
            "--pilot-pool", "12",
            "--evaluation-pool", "14",
            "--n0", "4",
            "--target-replicates", "2",
            "--pilot-policies", "random,source_boundary",
            "--descriptor-mode", "raw",
            "--coordinate", "boundary_latent",
            "--geometry", "linear_monotone",
            "--rank", "1",
            "--hierarchy-iterations", "2",
            "--family-delta", "0.025",
            "--evidence-temperature", "0.5",
            "--family-strategy", "source_domain_atoms",
            "--out", str(output),
        ])
        args.domains = (
            "RZDT1",
            "StatePolicyRZDT1",
            "FactorShockStatePolicyRZDT1",
            "InventorySupplyChain",
            "QueueResourceControl",
        )
        args.pilot_policies = ("random", "source_boundary")
        return args

    def test_gate_builds_frozen_library_and_uses_only_target_pilots(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._tiny_args(Path(directory) / "result.json")
            result = run_gate(args, "InventorySupplyChain", 3)
        self.assertEqual(
            result["model"]["model_version"],
            "tcb_v3_boundary_family_mixture",
        )
        self.assertEqual(result["model"]["family_count"], 4)
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(result["leakage_contract"][
            "outer_target_excluded_from_source_library"])
        for row in result["rows"]:
            diagnostic = row["adapter_diagnostics"]
            self.assertFalse(row["target_oracle_used_for_fit"])
            self.assertFalse(
                row["target_oracle_used_for_hyperparameter_selection"])
            self.assertFalse(diagnostic["target_label_used"])
            self.assertFalse(diagnostic["target_oracle_used"])
            self.assertTrue(diagnostic["family_parameters_frozen"])
            self.assertEqual(
                diagnostic["evidence_protocol"],
                "leave_one_pilot_out_generalized_bayes",
            )
            self.assertEqual(
                len(diagnostic["family_target_adapters"]),
                result["model"]["family_count"],
            )
            self.assertTrue(all(
                item["output_scale"] > 0.0
                and item["residual_scale"] > 0.0
                and item["parameter_covariance_trace"] >= 0.0
                for item in diagnostic["family_target_adapters"]
            ))
            self.assertGreaterEqual(
                diagnostic["credible_family_mass"], 0.975 - 1e-12)
            self.assertEqual(row["target_evaluations_used"], 8)
            self.assertEqual(row["source_inner_lodo"]["fold_count"], 4)
        self.assertTrue(_family_audit(result["rows"])["passed"])

    def test_scheduler_builds_six_by_five_by_three_offline_tasks(self):
        args = SUBMIT.build_parser().parse_args([
            "--run-id", "tcb_v3_test",
            "--deploy", "/deploy",
            "--python", "/env/bin/python",
            "--n-seeds", "3",
        ])
        specs, grid, domains = SUBMIT.build_specs(args)
        self.assertEqual(len(grid), 6)
        self.assertTrue(all(
            row["family_delta"] <= 0.04 for row in grid
        ))
        self.assertEqual(
            {row["family_strategy"] for row in grid},
            {"source_domain_atoms", "pooled_plus_source_domain_atoms"},
        )
        self.assertEqual(domains, 5)
        self.assertEqual(len(specs), 90)
        self.assertEqual(len({row["signature"] for row in specs}), 90)
        self.assertEqual(
            {row["require_node"] for row in specs}, set(SUBMIT.shared.CPU_NODES))
        self.assertTrue(all(row["cpu"] == 12 for row in specs))
        self.assertTrue(all(
            "SCOLHKG_OFFLINE=1" in row["cmd"]
            and "benchmark_tcb_v3_family_gate.py" in row["cmd"]
            and "--family-delta" in row["cmd"]
            and "--family-strategy" in row["cmd"]
            and set(row["allowed_nodes"]) == set(SUBMIT.shared.CPU_NODES)
            and "checkpoints" in row["stage_excludes"]
            for row in specs
        ))

    def test_scheduler_prepares_fresh_lightweight_deploy(self):
        with tempfile.TemporaryDirectory() as directory:
            target = SUBMIT.prepare_local_deploy(Path(directory))
            self.assertTrue((
                target / "performance/benchmark_tcb_v3_family_gate.py"
            ).is_file())
            self.assertTrue((
                target / "representation/transferable_boundary.py"
            ).is_file())
            self.assertFalse((target / "profiles").exists())
            self.assertFalse((target / "results").exists())
            self.assertFalse((target / "checkpoints").exists())

    def test_v32_scheduler_adds_orthogonal_family_calibration(self):
        args = V32_SUBMIT.build_parser().parse_args([
            "--run-id", "tcb_v32_test",
            "--deploy", "/deploy",
            "--python", "/env/bin/python",
            "--n-seeds", "3",
        ])
        specs, grid, domains = V32_SUBMIT.build_specs(args)
        self.assertEqual(domains, 5)
        self.assertEqual(len(grid), 6)
        self.assertEqual(len(specs), 90)
        self.assertEqual(grid[0]["target_residual_rank"], 0)
        self.assertTrue(all(
            row["target_residual_rank"] > 0 for row in grid[1:]
        ))
        self.assertTrue(any(
            row["target_residual_rank"] == 2 for row in grid
        ))
        self.assertTrue(all(
            row["family_delta"] + 0.01 <= 0.05 for row in grid
        ))
        self.assertEqual(
            {row["require_node"] for row in specs},
            set(V32_SUBMIT.v3.shared.CPU_NODES),
        )
        self.assertTrue(all(
            row["cpu"] == 12
            and "--target-residual-rank" in row["cmd"]
            and "--residual-ridge" in row["cmd"]
            and "SCOLHKG_OFFLINE=1" in row["cmd"]
            for row in specs
        ))


if __name__ == "__main__":
    unittest.main()
