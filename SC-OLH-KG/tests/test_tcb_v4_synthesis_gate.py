import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from performance.benchmark_tcb_v4_synthesis_gate import (  # noqa: E402
    build_parser,
    run_gate,
)
from performance.summarize_tcb_v4_synthesis_gate import (  # noqa: E402
    _synthesis_audit,
)


SUBMIT_SCRIPT = REPO / "scripts/submit_scolhkg_tcb_v4_synthesis_gate_scheduler.py"
SPEC = importlib.util.spec_from_file_location("tcb_v4_submit", SUBMIT_SCRIPT)
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


class TCBV4SynthesisGateTests(unittest.TestCase):
    def _tiny_args(self, output):
        args = build_parser().parse_args([
            "--domains",
            (
                "RZDT1,StatePolicyRZDT1,FactorShockStatePolicyRZDT1,"
                "InventorySupplyChain,QueueResourceControl"
            ),
            "--heldout", "InventorySupplyChain",
            "--target-seed", "4",
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
            "--coefficient-ridge", "0.1",
            "--coefficient-prior-strength", "0.5",
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

    def test_gate_uses_frozen_nonnegative_source_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._tiny_args(Path(directory) / "result.json")
            result = run_gate(args, "InventorySupplyChain", 4)
        self.assertEqual(
            result["model"]["model_version"],
            "tcb_v4_boundary_family_synthesis",
        )
        self.assertEqual(result["model"]["family_count"], 4)
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(result["leakage_contract"][
            "outer_target_excluded_from_source_dictionary"])
        for row in result["rows"]:
            diagnostic = row["adapter_diagnostics"]
            self.assertTrue(diagnostic["source_dictionary_frozen"])
            self.assertTrue(diagnostic["nonnegative_family_coefficients"])
            self.assertFalse(diagnostic["target_label_used"])
            self.assertFalse(diagnostic["target_oracle_used"])
            self.assertEqual(
                len(diagnostic["coefficients"]), row["family_count"])
            self.assertEqual(
                row["adapter_effective_dimension"], row["family_count"] + 1)
        self.assertTrue(_synthesis_audit(result["rows"])["passed"])

    def test_scheduler_builds_complete_offline_matrix(self):
        args = SUBMIT.build_parser().parse_args([
            "--run-id", "tcb_v4_test",
            "--deploy", "/deploy",
            "--python", "/env/bin/python",
            "--n-seeds", "3",
        ])
        specs, grid, domains = SUBMIT.build_specs(args)
        self.assertEqual(domains, 5)
        self.assertEqual(len(grid), 6)
        self.assertEqual(len(specs), 90)
        self.assertEqual(len({row["signature"] for row in specs}), 90)
        self.assertEqual(
            {row["require_node"] for row in specs},
            set(SUBMIT.v3.shared.CPU_NODES),
        )
        self.assertTrue(all(
            row["cpu"] == 12
            and "SCOLHKG_OFFLINE=1" in row["cmd"]
            and "benchmark_tcb_v4_synthesis_gate.py" in row["cmd"]
            and "--coefficient-ridge" in row["cmd"]
            and "--coefficient-prior-strength" in row["cmd"]
            for row in specs
        ))


if __name__ == "__main__":
    unittest.main()
