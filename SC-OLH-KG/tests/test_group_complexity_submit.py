import argparse
import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_group_complexity_audit.py"
SPEC = importlib.util.spec_from_file_location("complexity_submit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GroupComplexitySubmitTests(unittest.TestCase):
    def test_audit_stays_on_checkpoint_node_and_excludes_checkpoint_sync(self):
        args = argparse.Namespace(
            deploy=Path("/deploy"),
            python="/env/bin/python",
            manifest=Path("/deploy/SC-OLH-KG/performance/manifest.json"),
            source_run_id="v26b",
            run_id="v27_audit",
            pool_size=256,
            cpu=2,
            ram_mb=4096,
            allow_duplicate=False,
        )
        tasks = [{
            "id": "t1",
            "status": "done",
            "node": "node004",
            "signature": (
                "KG_op/scolhkg_manifest/v26b/InventorySupplyChain/seed3"
            ),
        }]
        specs = MODULE.build_specs(args, tasks)
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec["require_node"], "node004")
        self.assertEqual(spec["allowed_nodes"], ["node004"])
        self.assertIn("--checkpoint", spec["cmd"])
        self.assertIn("/checkpoints/v26b/InventorySupplyChain/seed3/", spec["cmd"])
        self.assertIn("checkpoints", spec["stage_excludes"])
        self.assertNotIn("ckpt_dir", spec)
        self.assertIn("SCOLHKG_OFFLINE=1", spec["cmd"])


if __name__ == "__main__":
    unittest.main()
